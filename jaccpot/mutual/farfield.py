"""Mutual (dual) far field on the Dehnen real spherical-harmonic basis.

The production downward sweep is one-directional: a node's local expansion
accumulates the fields of the sources it accepted, and the reciprocal effect is
picked up separately when the traversal reaches the other node. Here each
well-separated pair ``(A, B)`` appears **once** in the frozen list and a single
pass adds B's multipole field to A's local expansion *and* A's to B's -- Dehnen's
mutual M2L.

Why this conserves momentum exactly
-----------------------------------
Both forces are gradients of the *same* truncated mutual interaction energy
``W_AB = <M_A, T(R) M_B>``: the total force on A is ``-dW/dx_A`` and on B is
``-dW/dx_B = +dW/dx_A``. Because a single M2L evaluation supplies both sides from
one set of translation coefficients, ``F_A + F_B`` cancels *algebraically*, not
just to the truncation error -- measured at ~4e-16 relative, flat in ``theta``
and in the expansion order (``tests/integration/test_mutual_fmm.py``). The
remaining cascades preserve it because they are exact re-expansions: M2M loses
nothing mapping child degrees ``<= n`` into a parent degree ``n``, and L2L
re-centres a degree-``p`` polynomial into another degree-``p`` polynomial.

That is the whole reason to prefer this over a COM correction
(``a_i -= sum_j m_j a_j / sum_j m_j``): the correction also zeroes the momentum
sum, but by smearing a uniform nonlocal offset over every particle instead of
delivering each pair's back-reaction to the partner that actually caused it. It
does not survive a per-level decomposition, so it cannot support a block step.

Every stage is pure JAX over host-frozen index arrays, with static (Python) loop
bounds over tree levels, so the whole sweep transposes cleanly under
``jax.grad``.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.operators.dtypes import squared_radius_floor

# The two rotate helpers are package-private but imported deliberately: this
# module needs the *same* world<->z rotations the production rot-scale M2L uses,
# with only the z-translation core swapped (see `_m2l_batch`). Re-deriving them
# from the public rotation-block builders would duplicate logic that must not
# drift from the kernel it is checked against.
from jaccpot.operators.m2l_real_rot_scale import (
    _rotate_local_from_z_single,
    _rotate_multipole_to_z_single,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_with_grad,
    l2l_real,
    m2m_real,
    p2m_real_direct,
    sh_size,
    translate_along_z_m2l_real,
)

__all__ = [
    "MutualTreeArrays",
    "mutual_upward_sweep",
    "mutual_far_field_forces",
]

_M2L_BATCH_BUDGET = 1 << 16


class MutualTreeArrays(NamedTuple):
    """Device-resident frozen topology consumed by the far-field sweep."""

    num_nodes: int
    order: int
    leaf_nodes: Array
    leaf_particles: Array
    leaf_particle_valid: Array
    level_nodes: Tuple[Array, ...]
    level_parents: Tuple[Array, ...]
    far_source: Array
    far_target: Array
    far_valid: Array


def _safe_translate(
    coeffs: Array,
    deltas: Array,
    translate: object,
    *,
    order: int,
) -> Array:
    """Apply a rotation-based translation, guarding the zero-displacement case.

    The rotate-scale operators normalise ``delta`` to build the alignment
    rotation, so a zero displacement is a ``0/0`` -- NaN in the forward for some
    orders and, worse, a NaN cotangent even where the forward survives. A zero
    displacement is a genuine configuration (a parent whose centre of mass
    coincides with its only massive child), and the correct answer there is the
    identity, so substitute a dummy axis and select the untranslated coefficients
    back. Both branches are evaluated, which is what keeps the reverse finite.
    """
    d2 = jnp.sum(deltas * deltas, axis=-1)
    # Anything at or below the operators' own squared-radius floor is degenerate
    # for them, not just an exact zero, so test against the same floor they use.
    live = d2 > jnp.asarray(squared_radius_floor(deltas.dtype), deltas.dtype)
    axis = jnp.zeros_like(deltas).at[:, 0].set(jnp.ones_like(d2))
    safe = jnp.where(live[:, None], deltas, axis)
    out = jax.vmap(lambda c, d: translate(c, d, order=int(order)))(coeffs, safe)
    return jnp.where(live[:, None], out, coeffs)


def _nondegenerate_offsets(offsets: Array) -> Array:
    """Nudge L2P evaluation offsets off the expansion centre.

    ``evaluate_local_real`` builds the azimuth from ``x / rho`` with ``rho``
    clamped to :func:`~jaccpot.operators.dtypes.squared_radius_floor`. At an
    evaluation point sitting *exactly* on the expansion centre that clamp keeps
    the potential finite but collapses the azimuth, and the returned gradient
    loses its x and y components entirely -- only ``d/dz`` survives. A leaf
    holding a single particle hits this every time, because the particle *is* the
    leaf's centre of mass.

    The limit as the offset goes to zero is direction-independent (it is just the
    degree-1 term), so substituting a displacement just above the floor recovers
    the correct gradient to full precision while being physically nil: ~1e-27 in
    float64, ~1e-16 in float32, against expansion scales of order unity.
    """
    dtype = offsets.dtype
    floor = jnp.asarray(squared_radius_floor(dtype), dtype)
    nudge = jnp.sqrt(floor) * jnp.asarray(1024.0, dtype)
    r2 = jnp.sum(offsets * offsets, axis=-1)
    degenerate = r2 <= floor
    axis = jnp.zeros_like(offsets).at[..., 0].set(nudge)
    return jnp.where(degenerate[..., None], axis, offsets)


def _m2l_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    use_pallas: bool,
    interpret: bool,
) -> Array:
    """Batched rotate + z-translate + rotate-back real M2L, differentiable either way.

    This mirrors
    :func:`~jaccpot.operators.m2l_real_rot_scale.m2l_rot_scale_real_batch` but
    routes the z-translation core through
    :func:`~jaccpot.pallas.m2l_core_z_real.m2l_core_z_real_pallas_cvjp` rather
    than the bare Pallas kernel.

    That distinction is load-bearing, not cosmetic. ``pallas_call`` has no
    JVP/transpose rule, and the production helper calls the *bare* kernel, so
    ``use_pallas=True`` yields a forward that cannot be differentiated -- on CPU
    the Pallas path is simply unsupported and silently falls back, so the failure
    only appears on an actual GPU. Jaccpot already ships the missing reverse rule
    as a ``custom_vjp`` (Pallas forward, autodiff of the pure-JAX recurrence twin);
    using it is what keeps ``backend="pallas"`` differentiable.

    ``interpret`` runs the Pallas kernel in interpret mode, which works on CPU and
    is how the parity test exercises the kernel logic without a GPU.
    """
    from jaccpot.pallas.m2l_core_z_real import (
        m2l_core_z_real_pallas_cvjp,
        pallas_m2l_real_supported,
    )

    p = int(order)
    # NaN-safe radius: `norm` has a 0/0 reverse gradient at delta == 0. The
    # double-`where` keeps the cotangent finite there; the forward is unchanged.
    r2 = jnp.sum(deltas * deltas, axis=1)
    positive = r2 > 0
    radii = jnp.where(positive, jnp.sqrt(jnp.where(positive, r2, 1.0)), 0.0)

    rotated = jax.vmap(lambda m, d: _rotate_multipole_to_z_single(m, d, order=p))(
        multipoles, deltas
    )

    floored = jnp.maximum(radii, jnp.asarray(1.0e-30, dtype=radii.dtype))
    if use_pallas and (interpret or pallas_m2l_real_supported()):
        locals_z = m2l_core_z_real_pallas_cvjp(
            rotated, floored, p, bool(interpret), "triton"
        )
    else:
        locals_z = jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=p))(
            rotated, floored
        )

    return jax.vmap(lambda l, d: _rotate_local_from_z_single(l, d, order=p))(
        locals_z, deltas
    )


def mutual_upward_sweep(
    positions: Array,
    masses: Array,
    tree: MutualTreeArrays,
) -> Tuple[Array, Array, Array]:
    """Return ``(node_mass, node_center, multipoles)`` for the live inputs.

    Centres are centres of mass, recomputed from the live positions/masses on
    every call -- only the *discrete* structure is frozen. Leaves are reduced
    from their padded particle blocks and internal nodes are then accumulated
    level by level from the deepest upward, so every node's children are complete
    before it is read.
    """
    p = int(tree.order)
    dtype = positions.dtype
    num_nodes = int(tree.num_nodes)

    particles = tree.leaf_particles
    valid = tree.leaf_particle_valid
    x = positions[particles]
    m = jnp.where(valid, masses[particles], jnp.zeros_like(masses[particles]))

    node_mass = (
        jnp.zeros((num_nodes,), dtype=dtype).at[tree.leaf_nodes].add(jnp.sum(m, axis=1))
    )
    node_moment = (
        jnp.zeros((num_nodes, 3), dtype=dtype)
        .at[tree.leaf_nodes]
        .add(jnp.sum(m[..., None] * x, axis=1))
    )

    # Deepest level first: `level_nodes[d]` sits at depth d+1, its parents at d.
    for nodes, parents in zip(reversed(tree.level_nodes), reversed(tree.level_parents)):
        node_mass = node_mass.at[parents].add(node_mass[nodes])
        node_moment = node_moment.at[parents].add(node_moment[nodes])

    massive = node_mass > 0
    safe_mass = jnp.where(massive, node_mass, jnp.ones_like(node_mass))
    centers = jnp.where(massive[:, None], node_moment / safe_mass[:, None], 0.0)

    # P2M into the leaves, then M2M upward on the same level schedule.
    leaf_centers = centers[tree.leaf_nodes]
    deltas = x - leaf_centers[:, None, :]
    per_particle = jax.vmap(
        jax.vmap(lambda d, mm: p2m_real_direct(d, mm, order=p), in_axes=(0, 0)),
        in_axes=(0, 0),
    )(deltas, m)
    leaf_multipoles = jnp.sum(jnp.where(valid[..., None], per_particle, 0.0), axis=1)
    multipoles = (
        jnp.zeros((num_nodes, sh_size(p)), dtype=dtype)
        .at[tree.leaf_nodes]
        .add(leaf_multipoles.astype(dtype))
    )
    for nodes, parents in zip(reversed(tree.level_nodes), reversed(tree.level_parents)):
        # M2M convention: delta = child centre - parent centre.
        translated = _safe_translate(
            multipoles[nodes], centers[nodes] - centers[parents], m2m_real, order=p
        )
        multipoles = multipoles.at[parents].add(translated.astype(dtype))

    return node_mass, centers, multipoles


def _dual_m2l(
    centers: Array,
    multipoles: Array,
    tree: MutualTreeArrays,
    far_weights: Optional[Array],
    *,
    use_pallas: bool,
    interpret: bool = False,
) -> Array:
    """Accumulate both directions of every well-separated pair into node locals.

    ``far_source``/``far_target`` already carry each canonical pair twice, once
    per direction, built from a single canonical list. Batching the two
    directions together is not just convenient: it guarantees the same kernel,
    the same order and the same rounding mode apply to both halves of a pair.
    """
    p = int(tree.order)
    dtype = centers.dtype
    num_nodes = int(tree.num_nodes)
    locals_ = jnp.zeros((num_nodes, sh_size(p)), dtype=dtype)

    n_pairs = int(tree.far_source.shape[0])
    if n_pairs == 0:
        return locals_

    chunk = min(max(1, _M2L_BATCH_BUDGET), n_pairs)
    steps = (n_pairs + chunk - 1) // chunk
    pad = steps * chunk - n_pairs
    source = tree.far_source
    target = tree.far_target
    valid = tree.far_valid
    weights = (
        jnp.ones((n_pairs,), dtype=dtype)
        if far_weights is None
        else jnp.asarray(far_weights, dtype=dtype)
    )
    if pad:
        source = jnp.concatenate([source, jnp.zeros((pad,), dtype=source.dtype)])
        target = jnp.concatenate([target, jnp.zeros((pad,), dtype=target.dtype)])
        valid = jnp.concatenate([valid, jnp.zeros((pad,), dtype=bool)])
        weights = jnp.concatenate([weights, jnp.zeros((pad,), dtype=dtype)])

    def body(acc: Array, idx: Array) -> tuple[Array, None]:
        start = idx * chunk
        src = lax.dynamic_slice_in_dim(source, start, chunk)
        tgt = lax.dynamic_slice_in_dim(target, start, chunk)
        live = lax.dynamic_slice_in_dim(valid, start, chunk)
        w = lax.dynamic_slice_in_dim(weights, start, chunk)
        # M2L convention: delta = target centre - source centre.
        delta = centers[tgt] - centers[src]
        contrib = _m2l_batch(
            multipoles[src],
            delta,
            order=p,
            use_pallas=use_pallas,
            interpret=interpret,
        )
        contrib = contrib * jnp.where(live, w, 0.0)[:, None].astype(contrib.dtype)
        return acc.at[tgt].add(contrib.astype(acc.dtype)), None

    locals_, _ = lax.scan(body, locals_, jnp.arange(steps))
    return locals_


def _push_locals_down(locals_: Array, centers: Array, tree: MutualTreeArrays) -> Array:
    """L2L cascade, shallowest level first so parents are complete when read."""
    p = int(tree.order)
    for nodes, parents in zip(tree.level_nodes, tree.level_parents):
        # L2L convention: delta = parent centre - child centre.
        translated = _safe_translate(
            locals_[parents], centers[parents] - centers[nodes], l2l_real, order=p
        )
        locals_ = locals_.at[nodes].add(translated.astype(locals_.dtype))
    return locals_


def _l2p_forces(
    positions: Array,
    masses: Array,
    centers: Array,
    locals_: Array,
    tree: MutualTreeArrays,
) -> Array:
    """Evaluate leaf local expansions at their particles and return forces."""
    p = int(tree.order)
    particles = tree.leaf_particles
    valid = tree.leaf_particle_valid
    x = positions[particles]
    m = jnp.where(valid, masses[particles], jnp.zeros_like(masses[particles]))
    leaf_locals = locals_[tree.leaf_nodes]
    # L2P convention: delta = expansion centre - evaluation point, and the
    # returned gradient is d(phi)/d(delta); the acceleration is its negation.
    offsets = _nondegenerate_offsets(centers[tree.leaf_nodes][:, None, :] - x)
    grads = jax.vmap(
        lambda coeffs, offs: jax.vmap(
            lambda o: evaluate_local_real_with_grad(coeffs, o, order=p)[0]
        )(offs),
        in_axes=(0, 0),
    )(leaf_locals, offsets)
    contrib = jnp.where(valid[..., None], -grads * m[..., None], 0.0)
    return jnp.zeros_like(positions).at[particles].add(contrib.astype(positions.dtype))


def mutual_far_field_forces(
    positions: Array,
    masses: Array,
    tree: MutualTreeArrays,
    *,
    G: float = 1.0,
    far_weights: Optional[Array] = None,
    use_pallas: bool = False,
    interpret: bool = False,
    return_multipoles: bool = False,
) -> Array | Tuple[Array, Array, Array]:
    """Return the mutual far-field **force** on every particle.

    ``far_weights`` scales each directed pair entry. Because the two directions of
    a pair carry the *same* weight, the scaling commutes with the ``F_A + F_B``
    cancellation and momentum stays exact for any weighting -- which is what lets
    one traversal serve a whole block-step boundary.
    """
    node_mass, centers, multipoles = mutual_upward_sweep(positions, masses, tree)
    locals_ = _dual_m2l(
        centers,
        multipoles,
        tree,
        far_weights,
        use_pallas=use_pallas,
        interpret=interpret,
    )
    locals_ = _push_locals_down(locals_, centers, tree)
    forces = _l2p_forces(positions, masses, centers, locals_, tree) * jnp.asarray(
        G, dtype=positions.dtype
    )
    if return_multipoles:
        return forces, node_mass, centers
    return forces

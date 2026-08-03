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

import os
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
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
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

# Peak bytes one chunk's *padded* rotation-block tensors may reach on the fused
# Pallas lane. The fused kernel takes the world<->z rotations as explicit
# ``(pairs, Bp, mdp, mdp)`` operands, so a chunk costs far more memory per pair
# than the three-stage sandwich (which only ever materialises ``(pairs, C)``
# coefficient vectors). At order 4 in float64 that is ~16 KB per pair per block
# against ~0.2 KB, so reusing the 65536-pair count would ask for gigabytes.
_M2L_FUSED_BLOCK_BUDGET_BYTES = 128 << 20


def _m2l_lane(use_pallas: bool, interpret: bool) -> str:
    """Which M2L implementation to dispatch: ``"jax"``, ``"fused"`` or ``"zcore"``.

    Selected by ``JACCPOT_MUTUAL_M2L``; ``"auto"`` (the default) resolves to:

    * ``"fused"`` under ``interpret`` -- interpret mode exists to execute the
      real kernel logic on CPU, so the parity and differentiability tests must
      keep reaching it. Routing interpret to pure JAX would make them vacuous;
    * ``"jax"`` on real hardware, **because the Pallas M2L is measurably slower
      here**. On an A100 at N=10^4 the whole far field costs 17.1 ms in pure JAX
      against 20.0 ms fused and 20.3 ms z-core -- a ~0.85x regression that both
      Pallas shapes share.

    That the two shapes are indistinguishable is the informative part. Fusing the
    rotations away changes nothing, so the rotation ``vmap``s were never what
    dominated the far field. What the fused kernel *does* change is its operand
    traffic: it takes the world<->z rotations as explicit ``(pairs, Bp, mdp,
    mdp)`` arrays, 32 KB per pair at order 4 in float64 against the sandwich's
    0.39 KB of coefficient vectors -- an 82x amplification. The sandwich builds
    the same blocks inside a fused XLA kernel and never spills them to HBM.

    The production real-M2L path does not contradict this: it faces the same
    trade and simply has different pair statistics. Here every directed pair has
    its own ``delta``, so no rotation block is ever reused.

    Both Pallas lanes stay wired, differentiable and covered by the interpret
    tests; set ``JACCPOT_MUTUAL_M2L=fused`` (or ``zcore``) to force one on
    hardware and reproduce the A/B.
    """
    if not use_pallas:
        return "jax"
    choice = os.environ.get("JACCPOT_MUTUAL_M2L", "auto").strip().lower()
    if choice not in {"auto", "jax", "fused", "zcore"}:
        raise ValueError(
            "JACCPOT_MUTUAL_M2L must be one of 'auto', 'jax', 'fused', 'zcore'; "
            f"got {choice!r}"
        )
    if choice == "auto":
        choice = "fused" if interpret else "jax"
    if choice == "jax":
        return "jax"
    if interpret:
        return choice
    # The fused gate is `pallas_m2l_real_fused_supported` (Ampere sm_80+), not the
    # z-core's `pallas_m2l_real_supported` (any gpu/tpu): the latter would route a
    # pre-Ampere GPU into a Triton lowering that fails at runtime.
    from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported

    return choice if pallas_m2l_real_fused_supported() else "jax"


def _fused_m2l_chunk(order: int, itemsize: int) -> int:
    """Pairs per scan step that keep the fused lane's block tensors in budget."""
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_tables

    tables = m2l_real_fused_tables(int(order))
    per_pair = max(1, int(tables["Bp"]) * int(tables["mdp"]) ** 2 * int(itemsize))
    return max(1, _M2L_FUSED_BLOCK_BUDGET_BYTES // per_pair)


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

    Three lanes, all computing the same operator as
    :func:`~jaccpot.operators.m2l_real_rot_scale.m2l_rot_scale_real_batch`:

    * **fused Pallas** (``use_pallas`` and Ampere+, or ``interpret``) -- one
      launch per chunk does rotate -> z-translate -> rotate-back on chip via
      :func:`~jaccpot.pallas.m2l_real_fused.m2l_real_fused_pallas_cvjp`. The
      rotations are handed over as explicit per-degree blocks, built by the *same*
      ``real_rotation_*`` ops the single-pair helpers use, so the two lanes are
      the same arithmetic in a different order.
    * **z-core Pallas** (``JACCPOT_MUTUAL_FUSED_M2L=0``) -- the original
      three-stage sandwich, kept as the A/B reference for the fused lane.
    * **pure JAX** -- the fallback and the correctness/AD oracle.

    Both Pallas lanes go through a ``custom_vjp`` wrapper rather than the bare
    kernel, and that distinction is load-bearing, not cosmetic. ``pallas_call``
    has no JVP/transpose rule, and jaccpot's own ``m2l_core_z_real`` helper calls
    the *bare* kernel, so routing through it would yield a forward that cannot be
    differentiated -- and on CPU the Pallas path is simply unsupported and
    silently falls back, so the failure would only appear on an actual GPU.

    ``interpret`` runs the Pallas kernel in interpret mode, which works on CPU and
    is how the parity tests exercise the kernel logic without a GPU.
    """
    p = int(order)
    # NaN-safe radius: `norm` has a 0/0 reverse gradient at delta == 0. The
    # double-`where` keeps the cotangent finite there; the forward is unchanged.
    r2 = jnp.sum(deltas * deltas, axis=1)
    positive = r2 > 0
    radii = jnp.where(positive, jnp.sqrt(jnp.where(positive, r2, 1.0)), 0.0)
    floored = jnp.maximum(radii, jnp.asarray(1.0e-30, dtype=radii.dtype))

    lane = _m2l_lane(use_pallas, interpret)

    if lane == "fused":
        from jaccpot.pallas.m2l_real_fused import m2l_real_fused_pallas_cvjp

        dtype = multipoles.dtype
        blocks_to_z = real_rotation_blocks_to_z_multipole_batch(
            deltas, order=p, dtype=dtype
        )
        blocks_from_z = real_rotation_blocks_from_z_local_batch(
            deltas, order=p, dtype=dtype
        )
        return m2l_real_fused_pallas_cvjp(
            multipoles,
            blocks_to_z,
            blocks_from_z,
            floored,
            p,
            bool(interpret),
            "triton",
        )

    rotated = jax.vmap(lambda m, d: _rotate_multipole_to_z_single(m, d, order=p))(
        multipoles, deltas
    )

    if lane == "zcore":
        from jaccpot.pallas.m2l_core_z_real import m2l_core_z_real_pallas_cvjp

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

    budget = _M2L_BATCH_BUDGET
    if _m2l_lane(use_pallas, interpret) == "fused":
        budget = min(budget, _fused_m2l_chunk(p, dtype.itemsize))
    chunk = min(max(1, budget), n_pairs)
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
        # Padding slots carry src == tgt == 0, so their delta is exactly zero and
        # `_m2l_batch` floors the radius to 1e-30. The z-core then evaluates
        # r**-(p+1), which at that floor is 5.6e151 at order 4 and 2.0e213 at
        # order 6: finite in float64, but far past **inf in float32**. The
        # trailing `* where(live, w, 0)` below then turns inf * 0 into NaN rather
        # than dropping the slot, poisoning its target node -- which the L2L
        # cascade proceeds to broadcast across the tree.
        #
        # Substitute a unit axis *before* the reciprocal, the same double-`where`
        # discipline the kernels use; the zero weight still drops the slot.
        #
        # Two things kept this hidden: it needs float32 (float64 only overflows
        # from order 10, where 30*(p+1) passes 308), and it needs the directed
        # pair list not to divide the chunk -- which takes > 65536 directed far
        # pairs, i.e. roughly N > 10^4.
        safe_axis = jnp.zeros_like(delta).at[:, 2].set(jnp.ones_like(delta[:, 2]))
        delta = jnp.where(live[:, None], delta, safe_axis)
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

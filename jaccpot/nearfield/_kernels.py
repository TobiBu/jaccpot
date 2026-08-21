"""The near-field force arithmetic: self blocks, pair blocks, chunked pairs.

The innermost layer -- softened Newtonian acceleration (and optional potential)
between padded leaf blocks. Nothing here knows about the tree, the traversal or
the schedules; it takes blocks of positions and masses and returns blocks of
accelerations.

ACCUMULATION ORDER IS LOAD-BEARING. NUMERICS_AND_JAX §1: do not change the order
or associativity of the reductions in P2P. The per-block ``sum`` over the source
axis and the masking that precedes it are what keep the padded slots contributing
exactly zero rather than ``0 * inf``; see ``bench/audit_nearfield_padding.py``.

``_bucketed_chunk_pair_accels_remat`` is the ``jax.checkpoint`` wrapper the
bucketed reverse pass uses to trade recompute for residual memory, which is why
it is a module-level binding rather than an inline decorator: the un-remat'd
primal is still called directly on the forward-only path.

Split out of ``near_field.py`` (Tier 1.5, A.9 seam 2); every function body is
unchanged.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array

from jaccpot.runtime.grad_options import analytic_p2p_vjp_enabled

from .grad import _pair_accel_cvjp

__all__: list[str] = []


def _self_contributions(
    leaf_positions: Array,
    leaf_masses: Array,
    mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Compute intra-leaf particle-particle contributions.

    Self interaction only -- each leaf against itself, with the diagonal removed.
    The cross-leaf half is :func:`_pair_contributions`; summing the two is what
    makes the near field. This term is deliberately **not** covered by
    ``_pair_accel_cvjp`` (see the comment at the remat site), so it differentiates
    through the ordinary autodiff of the block below.

    Parameters
    ----------
    leaf_positions : Array
        ``[num_leaves, W, 3]`` positions, leaf-major and slot-padded to the leaf
        capacity ``W``. Padded slots must be excluded by ``mask``; their
        coordinate values are otherwise unconstrained and never read.
    leaf_masses : Array
        ``[num_leaves, W]`` masses under the same padding.
    mask : Array
        ``[num_leaves, W]`` boolean occupancy. Load-bearing twice over: it gates
        the pair block, and it zeroes the output rows for padded targets. A mask
        that admits a padded slot contributes a spurious body at whatever
        coordinates the padding holds.
    softening_sq : Union[float, Array]
        Plummer softening **squared**, added to every squared separation. A
        scalar applies to all leaves.
    G : Array
        Gravitational constant. An array, not a float, so it can be traced.
    compute_potential : bool
        Whether to return potentials. Static under ``jit`` -- it selects the
        return arity, so it cannot be a traced value.

    Returns
    -------
    Array
        ``[num_leaves, W, 3]`` accelerations, zero on padded slots.
    Optional[Array]
        ``[num_leaves, W]`` potentials when ``compute_potential`` is true, else
        ``None``. Turning it off skips the potential reduction but not the pair
        work, and the scan still carries a zero-filled ``[num_leaves, W]`` stack
        that is then dropped -- so it saves flops, not that allocation.
    """
    dtype = leaf_positions.dtype
    leaf_size = leaf_positions.shape[1]
    identity = jnp.eye(leaf_size, dtype=bool)

    def compute_single(args: tuple[Array, Array, Array]) -> tuple[Array, Array]:
        positions_leaf, masses_leaf, mask_leaf = args
        diff = positions_leaf[:, None, :] - positions_leaf[None, :, :]
        dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq

        pair_mask = mask_leaf[:, None] & mask_leaf[None, :] & (~identity)
        safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(pair_mask, inv_r, 0.0)
        inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)

        weighted = inv_dist3[:, :, None] * masses_leaf[None, :, None]
        accel_leaf = -G * jnp.sum(weighted * diff, axis=1)
        accel_leaf = jnp.where(mask_leaf[:, None], accel_leaf, 0.0)

        if compute_potential:
            pot_leaf = -G * jnp.sum(inv_r * masses_leaf[None, :], axis=1)
            pot_leaf = jnp.where(mask_leaf, pot_leaf, 0.0)
        else:
            pot_leaf = jnp.zeros((leaf_size,), dtype=dtype)

        return accel_leaf, pot_leaf

    # Rematerialize the per-leaf block. ``compute_single`` builds (W, W, 3) and
    # (W, W) intermediates, and the scan retains them for EVERY leaf, so the
    # residual is O(leaves * W^2) -- N*W in practice, i.e. ~1.4 GB at N=200000 and
    # ~7.5 GB at N=1048576 on the canonical leaf-256 config. Its inputs are only
    # the (W, 3) / (W,) slices of one leaf, so remat trades one extra intra-leaf
    # pass for a W-fold reduction.
    #
    # Note this term is NOT covered by ``_pair_accel_cvjp``: that rule handles
    # cross-leaf pair blocks, while intra-leaf self interaction is computed here.
    _compute_single_remat = jax.checkpoint(compute_single)

    def scan_step(
        carry: Any, args: tuple[Array, Array, Array]
    ) -> tuple[Any, tuple[Array, Array]]:
        accel_leaf, pot_leaf = _compute_single_remat(args)
        return carry, (accel_leaf, pot_leaf)

    _, (accels, potentials) = lax.scan(
        scan_step,
        None,
        (leaf_positions, leaf_masses, mask),
    )

    if compute_potential:
        return accels, potentials
    return accels, None


def _pair_contributions(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Compute one target-leaf vs source-leaf contribution block.

    The scalar reference form: one target leaf against one source leaf, softened
    Newtonian. ``_pair_contributions_batched`` is the vectorised twin used in
    production and must agree with this to reassociation.

    **Accumulation order is load-bearing** (NUMERICS_AND_JAX §1): the sum over the
    source axis, and the masking that precedes it, are what the goldens were taken
    with.

    Parameters
    ----------
    target_positions : Array
        Padded target-leaf positions ``[W, 3]``.
    target_mask : Array
        Padded target-leaf validity ``[W]``; ``False`` slots produce zero output.
    source_positions : Array
        Padded source-leaf positions ``[W, 3]``.
    source_masses : Array
        Padded source-leaf masses ``[W]``.
    source_mask : Array
        Padded source-leaf validity, same shape as ``source_masses``; masked
        sources are zeroed **before** the sum, so they contribute exactly ``0``
        rather than ``0 * inf``.
    softening_sq : Union[float, Array]
        Squared Plummer softening length, added to every squared separation.
    G : Array
        Gravitational constant.
    compute_potential : bool
        Also return per-target potentials. Static under ``jit``.

    Returns
    -------
    Tuple[Array, Optional[Array]]
        ``(accelerations, potentials)``; the second element is ``None`` unless
        ``compute_potential``.
    """
    dtype = target_positions.dtype

    source_pos = source_positions
    source_mass = source_masses
    source_active = source_mask
    mass_effective = jnp.where(source_active, source_mass, 0.0)

    soft = softening_sq

    def when_valid(pos: Array) -> tuple[Array, Array]:
        diff = pos - source_pos
        dist_sq = jnp.sum(diff * diff, axis=1) + soft
        mask_src = source_active

        safe_dist_sq = jnp.where(mask_src, dist_sq, jnp.ones_like(dist_sq))
        inv_r = lax.rsqrt(safe_dist_sq)
        inv_r = jnp.where(mask_src, inv_r, 0.0)
        inv_dist3 = jnp.where(mask_src, inv_r * inv_r * inv_r, 0.0)

        weighted = inv_dist3[:, None] * mass_effective[:, None]
        accel = -G * jnp.sum(weighted * diff, axis=0)

        if compute_potential:
            pot = -G * jnp.sum(inv_r * mass_effective)
        else:
            pot = jnp.zeros((), dtype=dtype)

        return accel, pot

    def scan_step(
        carry: Any, data: tuple[Array, Array]
    ) -> tuple[Any, tuple[Array, Array]]:
        pos, valid = data

        accel, pot = lax.cond(
            valid,
            when_valid,
            lambda _: (
                jnp.zeros((3,), dtype=dtype),
                jnp.zeros((), dtype=dtype),
            ),
            pos,
        )
        return carry, (accel, pot)

    _, (accels, potentials) = lax.scan(
        scan_step,
        None,
        (target_positions, target_mask),
    )

    if compute_potential:
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


@partial(jax.jit, static_argnames=("compute_potential",))
def _pair_contributions_batched(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Vectorized pair contributions for a batch of target/source leaf pairs.

    The production accel path, and the one the differentiable near field runs.
    Numerically equivalent to :func:`_pair_contributions` over a batch of leaf
    pairs.

    WHY THERE IS A BRANCH AT THE TOP. When ``analytic_p2p_vjp_enabled()`` and no
    potential is requested, this routes through ``_pair_accel_cvjp`` -- the analytic
    symmetric-tidal-tensor ``custom_vjp``. The **forward is byte-identical** either
    way; the point is the reverse, which that rule computes in O(N) memory instead
    of retaining a residual per scan iteration. Masks are handed over as 0/1 floats
    and ``softening_sq``/``G`` as arrays specifically so the reverse rule closes
    over no tracer.

    Parameters
    ----------
    target_positions : Array
        Padded target-leaf positions ``[num_pairs, W, 3]``.
    target_mask : Array
        Padded target-leaf validity ``[num_pairs, W]``; ``False`` slots produce zero output.
    source_positions : Array
        Padded source-leaf positions ``[num_pairs, W, 3]``.
    source_masses : Array
        Padded source-leaf masses ``[num_pairs, W]``.
    source_mask : Array
        Padded source-leaf validity, same shape as ``source_masses``; masked
        sources are zeroed **before** the sum, so they contribute exactly ``0``
        rather than ``0 * inf``.
    softening_sq : Union[float, Array]
        Squared Plummer softening length, added to every squared separation.
    G : Array
        Gravitational constant.
    compute_potential : bool
        Also return per-target potentials. Static under ``jit``.

    Returns
    -------
    Tuple[Array, Optional[Array]]
        ``(accelerations, potentials)``; the second element is ``None`` unless
        ``compute_potential``.
    """
    if analytic_p2p_vjp_enabled() and not compute_potential:
        # Accel-only path (the differentiable path): route through the analytic
        # symmetric-tidal-tensor custom_vjp. Forward is byte-identical; masks are
        # passed as 0/1 floats and softening/G as arrays so the reverse needs no
        # closure over tracers.
        dtype = target_positions.dtype
        accels = _pair_accel_cvjp(
            target_positions,
            source_positions,
            source_masses,
            target_mask.astype(dtype),
            source_mask.astype(dtype),
            jnp.asarray(softening_sq, dtype=dtype),
            jnp.asarray(G, dtype=dtype),
        )
        return accels, None
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = lax.rsqrt(safe_dist_sq)
    inv_r = jnp.where(pair_mask, inv_r, 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)

    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    if compute_potential:
        potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


@partial(jax.jit, static_argnames=("compute_potential",))
def _pair_contributions_batched_componentwise(
    target_positions: Array,
    target_mask: Array,
    source_positions: Array,
    source_masses: Array,
    source_mask: Array,
    *,
    softening_sq: Union[float, Array],
    G: Array,
    compute_potential: bool,
) -> Tuple[Array, Optional[Array]]:
    """Vectorized pair contributions with explicit Cartesian components.

    Same result as :func:`_pair_contributions_batched`, written with ``dx``/``dy``/
    ``dz`` split out instead of a vector difference. That is not cosmetic: keeping
    the three components separate lets XLA fuse the three reductions without
    materialising a ``[pairs, W, W, 3]`` difference tensor, which is the shape that
    dominates the large-N target-block kernels' peak. It has no ``custom_vjp``
    branch, so it is the form used where the analytic reverse does not apply.

    Parameters
    ----------
    target_positions : Array
        Padded target-leaf positions ``[num_pairs, W, 3]``.
    target_mask : Array
        Padded target-leaf validity ``[num_pairs, W]``; ``False`` slots produce zero output.
    source_positions : Array
        Padded source-leaf positions ``[num_pairs, W, 3]``.
    source_masses : Array
        Padded source-leaf masses ``[num_pairs, W]``.
    source_mask : Array
        Padded source-leaf validity, same shape as ``source_masses``; masked
        sources are zeroed **before** the sum, so they contribute exactly ``0``
        rather than ``0 * inf``.
    softening_sq : Union[float, Array]
        Squared Plummer softening length, added to every squared separation.
    G : Array
        Gravitational constant.
    compute_potential : bool
        Also return per-target potentials. Static under ``jit``.

    Returns
    -------
    Tuple[Array, Optional[Array]]
        ``(accelerations, potentials)``; the second element is ``None`` unless
        ``compute_potential``.
    """
    dx = target_positions[:, :, None, 0] - source_positions[:, None, :, 0]
    dy = target_positions[:, :, None, 1] - source_positions[:, None, :, 1]
    dz = target_positions[:, :, None, 2] - source_positions[:, None, :, 2]
    dist_sq = dx * dx + dy * dy + dz * dz + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]

    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    weighted = inv_r * inv_r * inv_r * source_masses[:, None, :]
    accel_x = -G * jnp.sum(weighted * dx, axis=2)
    accel_y = -G * jnp.sum(weighted * dy, axis=2)
    accel_z = -G * jnp.sum(weighted * dz, axis=2)
    accels = jnp.stack((accel_x, accel_y, accel_z), axis=-1)
    accels = jnp.where(target_mask[..., None], accels, 0.0)

    if compute_potential:
        potentials = -G * jnp.sum(inv_r * source_masses[:, None, :], axis=2)
        potentials = jnp.where(target_mask, potentials, 0.0)
        return accels, potentials

    return accels, None


def _bucketed_chunk_pair_accels(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    target_leaf_local: Array,
    source_leaf_local: Array,
    valid_edge: Array,
    softening_sq: Array,
    G: Array,
) -> Tuple[Array, Array]:
    """Gather one edge chunk's leaf tensors and evaluate its near-field pair block.

    Deliberately takes the **leaf-major buffers plus leaf-id index vectors** rather
    than pre-gathered positions, so callers can wrap it in ``jax.checkpoint``:
    ``lax.scan``'s partial-eval hoists the scan-invariant leaf buffers out and
    counts them once, leaving only two integer leaf-id vectors and a mask stacked
    per chunk.

    That is the single largest reverse-pass term at galaxy scale. The gather sits
    *outside* :func:`_pair_accel_cvjp`, which explicitly saves its own inputs, so
    rematerializing only the gather would achieve nothing -- the consumer would
    save the gather's outputs anyway. Rematerializing the composite (gather **and**
    pair evaluation) is what collapses it: measured **77.8 B per (edge x
    max_leaf_size)** by ``bench/audit_reverse_residuals.py``, i.e. 8.7 GB at
    N=200000 and 124 GB at N=1048576 on the canonical leaf-256 config, versus
    ~14 B per *edge* once rematerialized.

    Parameters
    ----------
    leaf_positions : Array
        Padded per-leaf positions ``[num_leaves, W, 3]``, scan-invariant so
        partial-eval hoists it out.
    leaf_masses : Array
        Padded per-leaf masses ``[num_leaves, W]``.
    leaf_mask : Array
        Padded per-leaf validity ``[num_leaves, W]``.
    target_leaf_local : Array
        Target leaf id per edge in this chunk, ``[chunk]``. Integer, no gradient.
    source_leaf_local : Array
        Source leaf id per edge in this chunk, ``[chunk]``. Integer, no gradient.
    valid_edge : Array
        Per-edge validity ``[chunk]``; padded edges contribute exactly zero.
    softening_sq : Array
        Squared Plummer softening length. An array, not a float, so the analytic
        reverse rule closes over no tracer.
    G : Array
        Gravitational constant, an array for the same reason.

    Returns
    -------
    Tuple[Array, Array]
        ``(pair_accelerations, target_mask)``. The caller applies the scatter,
        which is linear and therefore needs only indices and the mask in reverse.
    """
    target_positions = leaf_positions[target_leaf_local]
    target_mask = leaf_mask[target_leaf_local] & valid_edge[:, None]
    source_positions = leaf_positions[source_leaf_local]
    source_masses = leaf_masses[source_leaf_local]
    source_mask = leaf_mask[source_leaf_local] & valid_edge[:, None]
    pair_acc, _ = _pair_contributions_batched(
        target_positions,
        target_mask,
        source_positions,
        source_masses,
        source_mask,
        softening_sq=softening_sq,
        G=G,
        compute_potential=False,
    )
    return pair_acc, target_mask


_bucketed_chunk_pair_accels_remat = jax.checkpoint(_bucketed_chunk_pair_accels)

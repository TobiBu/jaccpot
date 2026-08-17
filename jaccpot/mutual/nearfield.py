"""Mutual (symmetric) near-field P2P.

Jaccpot's production near field is a **gather**: every target leaf sums over its
neighbour leaves, so the pair ``(A, B)`` is evaluated once while computing A's
force and again -- independently -- while computing B's. The two evaluations use
separately-rounded intermediates, so the impulses they apply are not exact
negatives and ``sum_i m_i a_i`` only vanishes to the force accuracy.

Here each unordered leaf pair is evaluated **once** and the resulting block is
applied twice: ``+F`` to the target leaf's particles and ``-F`` to the source
leaf's. The antisymmetry is then exact at the bit level, because

* IEEE subtraction guarantees ``fl(x_j - x_i) == -fl(x_i - x_j)``, and
* the scalar prefactor ``c_ij = G m_i m_j / r_ij^3`` is built from symmetric
  operands, so ``c_ij`` and ``c_ji`` are the same float,

and the two summations of ``c_ij * dr_ij`` differ only in the order they are
reduced. Accumulating **forces** and dividing by mass once at the end keeps
``sum_i m_i a_i == sum_i F_i``, which is the quantity that cancels.

Level weighting
---------------
``level_weights`` turns one traversal into any linear combination of the
block-step levels. Each particle pair is multiplied by
``level_weights[max(rung_i, rung_j)]`` -- the *exact* per-particle level of the
pair, matching nornax's definition. Since the weight is one symmetric scalar per
pair, it multiplies both ``+F`` and ``-F`` identically and cannot break the
cancellation. Passing a one-hot vector selects a single level; passing the
boundary's kick weights evaluates a whole sub-step boundary in a single pass.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

__all__ = [
    "mutual_near_field_forces",
    "resolve_near_chunk_size",
    "mutual_nearfield_pallas_active",
]

# Peak bytes the (chunk, S, S, 3) pair tensor is allowed to reach. The reverse
# pass materialises tensors of this shape, so it is the knob that bounds
# gradient memory as well as forward memory.
_PAIR_TENSOR_BUDGET_BYTES = 64 << 20

# The Pallas lane never materialises the S x S tile in HBM -- it lives in
# registers and only the (chunk, S, 4) reductions are written -- so its chunk is
# bounded by a much smaller per-pair footprint and can be far larger. Fewer,
# bigger grids beat many small launches.
_PALLAS_OUTPUT_BUDGET_BYTES = 256 << 20


def resolve_near_chunk_size(
    max_leaf_size: int, itemsize: int, requested: Optional[int] = None
) -> int:
    """Pick how many leaf pairs to process per scan step.
    A memory/throughput knob only: chunking changes how the pair tensors are
    blocked, never the result.

    Parameters
    ----------
    max_leaf_size : int
        Slots per leaf block; the pair tensors go as its square.
    itemsize : int
        Bytes per element of the working dtype.
    requested : Optional[int]
        Explicit override; ``None`` derives the chunk from the memory budget.

    Returns
    -------
    int
        Leaf pairs per scan step, at least ``1``.
    """
    if requested is not None:
        return max(1, int(requested))
    per_pair = max(1, int(max_leaf_size) ** 2 * 3 * int(itemsize))
    return max(1, _PAIR_TENSOR_BUDGET_BYTES // per_pair)


def _resolve_pallas_chunk_size(
    max_leaf_size: int, itemsize: int, requested: Optional[int] = None
) -> int:
    """Pick the scan chunk for the Pallas lane, bounded by its output tensors.
    Bounded separately from the pure-JAX lane because the Pallas kernel's output
    blocks, not its pair tensors, are what dominate its footprint.

    Parameters
    ----------
    max_leaf_size : int
        Slots per leaf block; the pair tensors go as its square.
    itemsize : int
        Bytes per element of the working dtype.
    requested : Optional[int]
        Explicit override; ``None`` derives the chunk from the memory budget.

    Returns
    -------
    int
        Leaf pairs per scan step, at least ``1``.
    """
    if requested is not None:
        return max(1, int(requested))
    # Two (chunk, S, 4) outputs, forward; the analytic reverse writes the same
    # shape again for positions plus (chunk, S) for masses.
    per_pair = max(1, 2 * int(max_leaf_size) * 4 * int(itemsize))
    return max(1, _PALLAS_OUTPUT_BUDGET_BYTES // per_pair)


def mutual_nearfield_pallas_active(use_pallas: bool, interpret: bool) -> bool:
    """Whether the near field should dispatch the mutual Pallas kernel.
    Parameters
    ----------
    use_pallas : bool
        Whether the caller asked for the Pallas lane.
    interpret : bool
        Whether the kernel would run in interpret mode. Interpret mode counts as
        active, so the parity tests reach the real kernel logic on CPU rather
        than silently falling back to pure JAX.

    Returns
    -------
    bool
        ``True`` when the Pallas kernel should be dispatched.
    """
    if not use_pallas:
        return False
    if interpret:
        return True
    from jaccpot.pallas.nearfield_mutual import pallas_nearfield_mutual_supported

    return bool(pallas_nearfield_mutual_supported())


def _pair_weights(
    rung_a: Optional[Array],
    rung_b: Optional[Array],
    level_weights: Optional[Array],
) -> Optional[Array]:
    """Return ``level_weights[max(rung_i, rung_j)]`` for a particle-pair block.
    A pair belongs to the level of its FINER endpoint, which is why the reduction
    is a max. This is the exact per-particle predicate; the far field's
    cell-level approximation lives in :mod:`jaccpot.mutual.force`.

    Parameters
    ----------
    rung_a : Optional[Array]
        Per-slot rung of the ``a`` block, or ``None``.
    rung_b : Optional[Array]
        Per-slot rung of the ``b`` block, or ``None``.
    level_weights : Optional[Array]
        Weight per interaction level, or ``None``.

    Returns
    -------
    Optional[Array]
        Per-pair weights, or ``None`` when any input is ``None`` -- the
        unweighted full-force case.
    """
    if level_weights is None or rung_a is None or rung_b is None:
        return None
    pair_level = jnp.maximum(rung_a[:, :, None], rung_b[:, None, :])
    # `jnp.take` defaults to mode="fill", which returns NaN for an out-of-range
    # index on a float array -- a rung above k_max would silently poison the whole
    # force. Clamp instead; callers that can see concrete rungs reject the
    # out-of-range case up front (BlockStepFMM._validate_rung).
    pair_level = jnp.clip(pair_level, 0, int(level_weights.shape[0]) - 1)
    return jnp.take(level_weights, pair_level, axis=0, mode="clip")


def _block_forces(
    x_a: Array,
    m_a: Array,
    x_b: Array,
    m_b: Array,
    valid: Array,
    weights: Optional[Array],
    *,
    softening: float,
    G: float,
) -> Array:
    """Return ``F[c, i, j] = G m_i m_j dr_ij / r_ij^3`` for a batch of blocks.

    ``dr`` is ``x_b - x_a``, so ``F`` is the force on the ``a`` particle. The
    caller obtains the ``b`` side by negating and reducing over the other axis --
    never by recomputing, which is the whole point.

    Parameters
    ----------
    x_a : Array
        ``(chunk, S, 3)`` positions of the ``a`` blocks.
    m_a : Array
        ``(chunk, S)`` masses of the ``a`` blocks; padding slots carry zero.
    x_b : Array
        ``(chunk, S, 3)`` positions of the ``b`` blocks.
    m_b : Array
        ``(chunk, S)`` masses of the ``b`` blocks.
    valid : Array
        ``(chunk, S, S)`` pair validity mask.
    weights : Optional[Array]
        ``(chunk, S, S)`` level weights from :func:`_pair_weights`, or ``None``
        for unit weights.
    softening : float
        Plummer softening length.
    G : float
        Gravitational constant.

    Returns
    -------
    Array
        ``(chunk, S, S, 3)`` per-pair force on the ``a`` particle. Masked pairs
        are exactly zero.
    """
    dr = x_b[:, None, :, :] - x_a[:, :, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + jnp.asarray(softening, dtype=dr.dtype) ** 2
    # Double-`where`: the guarded r2 is substituted *before* the reciprocal so the
    # reverse pass never evaluates d(r^-3)/d(r2) at an invalid pair. A single
    # trailing `where` would still produce a NaN cotangent from the masked branch.
    safe_r2 = jnp.where(valid, r2, jnp.ones_like(r2))
    inv_r3 = jnp.where(valid, safe_r2 ** (-1.5), jnp.zeros_like(r2))
    scale = jnp.asarray(G, dtype=dr.dtype) * m_a[:, :, None] * m_b[:, None, :] * inv_r3
    if weights is not None:
        scale = scale * weights.astype(scale.dtype)
    return scale[..., None] * dr


def _gather_leaf_block(
    positions: Array,
    masses: Array,
    rung: Optional[Array],
    particles: Array,
    valid: Array,
) -> tuple[Array, Array, Optional[Array]]:
    """Gather a padded leaf block; padding slots carry exactly zero mass.
    Zero mass rather than a mask is what makes the padding inert in the force sum
    without a separate branch.

    Parameters
    ----------
    positions : Array
        All particle positions, in Morton-sorted order.
    masses : Array
        All particle masses, same order.
    rung : Optional[Array]
        All particle rungs, same order, or ``None``.
    particles : Array
        ``(chunk, S)`` particle indices of the blocks to gather.
    valid : Array
        ``(chunk, S)`` occupancy mask over those slots.

    Returns
    -------
    tuple[Array, Array, Optional[Array]]
        ``(positions, masses, rungs)`` for the block, with ``rungs`` ``None``
        exactly when ``rung`` was.
    """
    x = positions[particles]
    m = jnp.where(valid, masses[particles], jnp.zeros_like(masses[particles]))
    r = None if rung is None else rung[particles]
    return x, m, r


def mutual_near_field_forces(
    positions: Array,
    masses: Array,
    *,
    leaf_particles: Array,
    leaf_particle_valid: Array,
    near_a: Array,
    near_b: Array,
    near_valid: Array,
    self_leaves: Array,
    softening: float,
    G: float = 1.0,
    rung: Optional[Array] = None,
    level_weights: Optional[Array] = None,
    chunk_size: Optional[int] = None,
    use_pallas: bool = False,
    interpret: bool = False,
) -> Array:
    """Return the mutual near-field **force** on every particle.

    Parameters
    ----------
    positions : Array
        ``(N, 3)`` particle positions in Morton-sorted order. Differentiable.
    masses : Array
        ``(N,)`` particle masses in Morton-sorted order. Differentiable.
    leaf_particles : Array
        ``(L, S)`` padded particle indices per leaf.
    leaf_particle_valid : Array
        ``(L, S)`` mask marking which slots of ``leaf_particles`` are real.
    near_a : Array
        Canonical cross-leaf pair's first leaf, as a **leaf-list index**.
    near_b : Array
        Canonical cross-leaf pair's second leaf, as a **leaf-list index**.
    near_valid : Array
        Mask over the pair list; the lists are padded to a whole number of chunks
        and this marks the padding.
    self_leaves : Array
        ``(L,)`` leaf-list indices whose intra-leaf interactions are evaluated.
    softening : float
        Plummer softening length.
    G : float
        Gravitational constant.
    rung : Optional[Array]
        Per-particle block-step rung, or ``None`` for an unweighted evaluation.
    level_weights : Optional[Array]
        ``(k_max + 1,)`` per-level weights; see the module docstring.
    chunk_size : Optional[int]
        Leaf pairs per scan step; ``None`` derives it from the pair-tensor budget.
    use_pallas : bool
        Route the leaf-pair blocks through
        :mod:`jaccpot.pallas.nearfield_mutual` (Pallas forward + hand-written
        analytic reverse) instead of the pure-JAX tensors.
    interpret : bool
        Run the Pallas kernel under CPU interpret semantics, which is what keeps
        the parity tests non-vacuous off-GPU.

    Returns
    -------
    Array
        ``(N, 3)`` forces (**not** accelerations), summing to zero to
        floating-point round-off.
    """
    positions = jnp.asarray(positions)
    masses = jnp.asarray(masses)
    dtype = positions.dtype
    forces = jnp.zeros_like(positions)

    leaf_particles = jnp.asarray(leaf_particles)
    leaf_particle_valid = jnp.asarray(leaf_particle_valid)
    max_leaf_size = int(leaf_particles.shape[1])
    pallas = mutual_nearfield_pallas_active(use_pallas, interpret)
    if pallas:
        chunk = _resolve_pallas_chunk_size(max_leaf_size, dtype.itemsize, chunk_size)
    else:
        chunk = resolve_near_chunk_size(max_leaf_size, dtype.itemsize, chunk_size)

    # The kernel takes masks and rungs as FLOAT arrays: they cross a `custom_vjp`
    # boundary, and a bool/int argument there has tangent type `float0`, which a
    # `bwd` cannot hand back a `zeros_like` for. Same `_f` convention the radix
    # fast lane's custom_vjp uses.
    num_levels = (
        0 if level_weights is None or rung is None else int(level_weights.shape[0])
    )
    lw_arr = (
        jnp.ones((1,), dtype=dtype)
        if num_levels <= 0
        else jnp.asarray(level_weights, dtype=dtype)
    )
    # Match the pure-JAX lane's `mode="clip"` take: a rung above k_max is clamped
    # onto the last level rather than silently dropped.
    rung_f = (
        None
        if rung is None
        else jnp.clip(jnp.asarray(rung), 0, max(num_levels - 1, 0)).astype(dtype)
    )
    soft_sq = jnp.asarray(softening, dtype=dtype) ** 2
    g_arr = jnp.asarray(G, dtype=dtype)

    def _pallas_block(
        la: Array,
        lb: Array,
        va: Array,
        vb: Array,
        *,
        exclude_diagonal: bool,
        emit_b: bool,
    ) -> tuple[Array, Array, Array, Array]:
        """Gather two leaf blocks, run the kernel, return ``(F_a, F_b)``.
        Parameters
        ----------
        la : Array
            Leaf index of the ``a`` side of each pair in the chunk.
        lb : Array
            Leaf index of the ``b`` side.
        va : Array
            Occupancy mask of the ``a`` blocks.
        vb : Array
            Occupancy mask of the ``b`` blocks.
        exclude_diagonal : bool
            Drop ``i == j``; set on the self-pair path, where ``la is lb``.
        emit_b : bool
            Produce the ``b``-side force. Cleared on the self-pair path, where
            both sides are the same block.

        Returns
        -------
        tuple[Array, Array, Array, Array]
            ``(particles_a, particles_b, F_a, F_b)`` -- the two index arrays the
            caller scatters into, alongside the two force blocks.
        """
        from jaccpot.pallas.nearfield_mutual import mutual_leafpair_block_cvjp

        pa, pb = leaf_particles[la], leaf_particles[lb]
        xa = positions[pa]
        xb = positions[pb]
        ma = jnp.where(va, masses[pa], jnp.zeros_like(masses[pa]))
        mb = jnp.where(vb, masses[pb], jnp.zeros_like(masses[pb]))
        ra = jnp.zeros(ma.shape, dtype=dtype) if rung_f is None else rung_f[pa]
        rb = jnp.zeros(mb.shape, dtype=dtype) if rung_f is None else rung_f[pb]
        f_a, f_b = mutual_leafpair_block_cvjp(
            xa,
            ma,
            va.astype(dtype),
            xb,
            mb,
            vb.astype(dtype),
            ra,
            rb,
            lw_arr,
            soft_sq,
            g_arr,
            num_levels,
            bool(exclude_diagonal),
            bool(emit_b),
            bool(interpret),
        )
        return pa, pb, f_a, f_b

    # ---- intra-leaf (self) blocks -------------------------------------------
    # The full L x L block is evaluated (minus the diagonal) rather than an upper
    # triangle: the full block is already exactly antisymmetric, so its row sums
    # cancel by construction, and it avoids a triangular gather that would cost
    # more than the flops it saves.
    self_leaves = jnp.asarray(self_leaves)
    n_self = int(self_leaves.shape[0])
    if n_self:
        # Clamp to the actual count: the chunk is a memory *budget*, and a budget
        # larger than the work would otherwise pad the scan out to a full chunk
        # of dead slots -- which the kernels still evaluate.
        self_chunk = min(chunk, n_self)
        self_steps = (n_self + self_chunk - 1) // self_chunk
        self_pad = self_steps * self_chunk - n_self
        padded_self = jnp.concatenate(
            [self_leaves, jnp.zeros((self_pad,), dtype=self_leaves.dtype)]
        )
        # Padding slots repeat leaf 0, so they are silenced through the *particle*
        # validity mask. Dropping them by leaf id instead would need a duplicate
        # id, and that leaf would then be accumulated twice.
        self_live = jnp.concatenate(
            [jnp.ones((n_self,), dtype=bool), jnp.zeros((self_pad,), dtype=bool)]
        )

        def self_body(acc: Array, idx: Array) -> tuple[Array, None]:
            start = idx * self_chunk
            leaves = lax.dynamic_slice_in_dim(padded_self, start, self_chunk)
            live = lax.dynamic_slice_in_dim(self_live, start, self_chunk)
            valid_slot = leaf_particle_valid[leaves] & live[:, None]
            if pallas:
                # Intra-leaf: the full block minus its diagonal is already exactly
                # antisymmetric, so only the `a` side is emitted -- applying both
                # sides to the same particles would double count.
                particles, _, contrib, _ = _pallas_block(
                    leaves,
                    leaves,
                    valid_slot,
                    valid_slot,
                    exclude_diagonal=True,
                    emit_b=False,
                )
                contrib = jnp.where(valid_slot[..., None], contrib, 0.0)
                return acc.at[particles].add(contrib.astype(acc.dtype)), None
            particles = leaf_particles[leaves]
            x, m, r = _gather_leaf_block(positions, masses, rung, particles, valid_slot)
            pair_valid = valid_slot[:, :, None] & valid_slot[:, None, :]
            pair_valid = pair_valid & ~jnp.eye(max_leaf_size, dtype=bool)[None, :, :]
            weights = _pair_weights(r, r, level_weights)
            block = _block_forces(
                x, m, x, m, pair_valid, weights, softening=softening, G=G
            )
            contrib = jnp.where(valid_slot[..., None], jnp.sum(block, axis=2), 0.0)
            return acc.at[particles].add(contrib.astype(acc.dtype)), None

        forces, _ = lax.scan(self_body, forces, jnp.arange(self_steps))

    # ---- cross-leaf pairs ----------------------------------------------------
    near_a = jnp.asarray(near_a)
    near_b = jnp.asarray(near_b)
    near_valid = jnp.asarray(near_valid)
    n_pairs = int(near_a.shape[0])
    if n_pairs == 0:
        return forces
    pair_chunk = min(chunk, n_pairs)
    steps = (n_pairs + pair_chunk - 1) // pair_chunk
    pad = steps * pair_chunk - n_pairs
    if pad:
        near_a = jnp.concatenate([near_a, jnp.zeros((pad,), dtype=near_a.dtype)])
        near_b = jnp.concatenate([near_b, jnp.zeros((pad,), dtype=near_b.dtype)])
        near_valid = jnp.concatenate([near_valid, jnp.zeros((pad,), dtype=bool)])

    def pair_body(acc: Array, idx: Array) -> tuple[Array, None]:
        start = idx * pair_chunk
        la = lax.dynamic_slice_in_dim(near_a, start, pair_chunk)
        lb = lax.dynamic_slice_in_dim(near_b, start, pair_chunk)
        live = lax.dynamic_slice_in_dim(near_valid, start, pair_chunk)
        va = leaf_particle_valid[la] & live[:, None]
        vb = leaf_particle_valid[lb] & live[:, None]
        if pallas:
            pa, pb, contrib_a, contrib_b = _pallas_block(
                la, lb, va, vb, exclude_diagonal=False, emit_b=True
            )
            contrib_a = jnp.where(va[..., None], contrib_a, 0.0)
            contrib_b = jnp.where(vb[..., None], contrib_b, 0.0)
            acc = acc.at[pa].add(contrib_a.astype(acc.dtype))
            acc = acc.at[pb].add(contrib_b.astype(acc.dtype))
            return acc, None
        pa, pb = leaf_particles[la], leaf_particles[lb]
        xa, ma, ra = _gather_leaf_block(positions, masses, rung, pa, va)
        xb, mb, rb = _gather_leaf_block(positions, masses, rung, pb, vb)
        pair_valid = va[:, :, None] & vb[:, None, :]
        weights = _pair_weights(ra, rb, level_weights)
        block = _block_forces(
            xa, ma, xb, mb, pair_valid, weights, softening=softening, G=G
        )
        # One evaluation, two applications: the `b` side is the *negation* of the
        # same tensor, never an independent recomputation.
        contrib_a = jnp.where(va[..., None], jnp.sum(block, axis=2), 0.0)
        contrib_b = jnp.where(vb[..., None], -jnp.sum(block, axis=1), 0.0)
        acc = acc.at[pa].add(contrib_a.astype(acc.dtype))
        acc = acc.at[pb].add(contrib_b.astype(acc.dtype))
        return acc, None

    forces, _ = lax.scan(pair_body, forces, jnp.arange(steps))
    return forces

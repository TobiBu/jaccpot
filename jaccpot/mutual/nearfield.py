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

__all__ = ["mutual_near_field_forces", "resolve_near_chunk_size"]

# Peak bytes the (chunk, S, S, 3) pair tensor is allowed to reach. The reverse
# pass materialises tensors of this shape, so it is the knob that bounds
# gradient memory as well as forward memory.
_PAIR_TENSOR_BUDGET_BYTES = 64 << 20


def resolve_near_chunk_size(
    max_leaf_size: int, itemsize: int, requested: Optional[int] = None
) -> int:
    """Pick how many leaf pairs to process per scan step."""
    if requested is not None:
        return max(1, int(requested))
    per_pair = max(1, int(max_leaf_size) ** 2 * 3 * int(itemsize))
    return max(1, _PAIR_TENSOR_BUDGET_BYTES // per_pair)


def _pair_weights(
    rung_a: Optional[Array],
    rung_b: Optional[Array],
    level_weights: Optional[Array],
) -> Optional[Array]:
    """Return ``level_weights[max(rung_i, rung_j)]`` for a particle-pair block."""
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
    """Gather a padded leaf block; padding slots carry exactly zero mass."""
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
) -> Array:
    """Return the mutual near-field **force** on every particle.

    Parameters
    ----------
    positions, masses :
        ``(N, 3)`` / ``(N,)`` in Morton-sorted order. Differentiable inputs.
    leaf_particles, leaf_particle_valid :
        ``(L, S)`` padded particle indices per leaf and their validity mask.
    near_a, near_b, near_valid :
        Canonical cross-leaf pairs as **leaf-list indices**, padded to a whole
        number of chunks; ``near_valid`` masks the padding.
    self_leaves :
        ``(L,)`` leaf-list indices whose intra-leaf interactions are evaluated.
    rung, level_weights :
        Optional block-step level weighting; see the module docstring.

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
    chunk = resolve_near_chunk_size(max_leaf_size, dtype.itemsize, chunk_size)

    # ---- intra-leaf (self) blocks -------------------------------------------
    # The full L x L block is evaluated (minus the diagonal) rather than an upper
    # triangle: the full block is already exactly antisymmetric, so its row sums
    # cancel by construction, and it avoids a triangular gather that would cost
    # more than the flops it saves.
    self_leaves = jnp.asarray(self_leaves)
    n_self = int(self_leaves.shape[0])
    if n_self:
        self_steps = (n_self + chunk - 1) // chunk
        self_pad = self_steps * chunk - n_self
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
            start = idx * chunk
            leaves = lax.dynamic_slice_in_dim(padded_self, start, chunk)
            live = lax.dynamic_slice_in_dim(self_live, start, chunk)
            particles = leaf_particles[leaves]
            valid_slot = leaf_particle_valid[leaves] & live[:, None]
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
    steps = (n_pairs + chunk - 1) // chunk
    pad = steps * chunk - n_pairs
    if pad:
        near_a = jnp.concatenate([near_a, jnp.zeros((pad,), dtype=near_a.dtype)])
        near_b = jnp.concatenate([near_b, jnp.zeros((pad,), dtype=near_b.dtype)])
        near_valid = jnp.concatenate([near_valid, jnp.zeros((pad,), dtype=bool)])

    def pair_body(acc: Array, idx: Array) -> tuple[Array, None]:
        start = idx * chunk
        la = lax.dynamic_slice_in_dim(near_a, start, chunk)
        lb = lax.dynamic_slice_in_dim(near_b, start, chunk)
        live = lax.dynamic_slice_in_dim(near_valid, start, chunk)
        pa, pb = leaf_particles[la], leaf_particles[lb]
        va = leaf_particle_valid[la] & live[:, None]
        vb = leaf_particle_valid[lb] & live[:, None]
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

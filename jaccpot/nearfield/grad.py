"""Reverse-mode rules for the near field -- the term that dominates the gradient.

Profiling put the near field at ~83% of the forward and ~91% of the reverse of
``differentiable_accelerations``, so essentially all of the gradient path's cost
and all of its memory risk live here. This module holds the two analytic reverse
rules and their shared machinery, extracted from ``near_field.py`` (still ~4200
lines of forward kernels) so the differentiable surface can be read on its own.

Two rules, for two near-field layouts:

* :func:`_pair_accel_cvjp` -- the **bucketed edge-list** lane. The reverse is the
  symmetric tidal tensor ``J = -G sum_s m_s (I/r^3 - 3 r r^T / r^5)`` contracted
  with the output cotangent (``J^T c = J c``) in one extra pair pass. Its pair
  intermediates are **rematerialized** rather than stored: storing them cost
  ``40 * near_field_edges * max_leaf_size**2`` bytes, measured at a single
  **52 GiB** allocation at N=65536 against 1.13 GB for the whole forward.
* :func:`_leafpair_accel_analytic_vjp` -- the **leaf-major prepacked** lane, in
  O(N) memory. This is the one that makes galaxy scale reachable: the bucketed
  reverse OOMs at N=200000 (30 GB peak) where this completes in 6.8 GB.

The recurring theme, and the reason both are hand-written rather than
``jax.vjp`` of a pure-JAX twin: **a ``bwd`` rule is never itself differentiated**,
so everything it computes is a transient bounded by the tile size instead of a
residual retained per scan iteration. Taking the reverse from ``jax.vjp(twin)``
is correct but linearizes the twin, reinstating its O(edges * W) scan residuals
as a backward-pass peak (~67 GB at N=1048576 on the canonical leaf-256 config).
Those twins remain the correctness oracles in
``tests/unit/test_custom_vjp_parity.py``; they are not production reverses.

**Nothing here imports from ``near_field``**, which is what keeps the two files
acyclic. The split follows that constraint rather than a tidier one: the bucketed
rule is self-contained, so its whole ``custom_vjp`` lives here, while the
leaf-major rule's forward *is* a Pallas kernel in ``near_field``, so only its
mathematics (:func:`_leafpair_accel_analytic_vjp`) moved and the wrapper stayed
behind next to the kernel it wraps.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Tuple
from jax import lax
from jaxtyping import Array
from yggdrax.dtypes import INDEX_DTYPE

from jaccpot._env import env_float, env_int

__all__ = [
    "build_leafpair_reverse_tiers",
    "clear_leafpair_reverse_tier_cache",
]

_FLOAT32_EXACT_INT_LIMIT = 2**24


def _check_float_id_range(num_particles: int, dtype: Any, *, what: str) -> None:
    """Refuse float id encoding when the ids would not be exactly representable.

    ``custom_vjp`` rules must not close over tracers, so index/mask arrays have to
    be passed as ordinary arguments -- and integer args would yield ``float0``
    cotangents rather than plain zeros. The differentiable lanes therefore encode
    ids as floats (reconstructed with ``round().astype``), which is exact only
    below the mantissa bound.
    """
    if (
        jnp.dtype(dtype) == jnp.float32
        and int(num_particles) > _FLOAT32_EXACT_INT_LIMIT
    ):
        raise ValueError(
            f"{what}: float32 represents integers exactly only up to "
            f"{_FLOAT32_EXACT_INT_LIMIT} but N={int(num_particles)}. The "
            "differentiable near-field lane encodes particle ids as floats (a "
            "custom_vjp rule may not close over tracers, and integer args would "
            "yield float0 cotangents), so ids beyond that bound would be silently "
            "rounded. Use float64, or split the evaluation."
        )


def _grad_rev_tier_max() -> int:
    """Maximum occupancy tiers for the analytic reverse (``JACCPOT_GRAD_REV_TIERS``)."""
    return env_int("JACCPOT_GRAD_REV_TIERS", 4)


def _grad_rev_tier_min_gain() -> float:
    """Minimum predicted slot-visit reduction before tiering pays for itself.

    See :func:`build_leafpair_reverse_tiers` for the measurements behind 3.0.
    """
    return env_float("JACCPOT_GRAD_REV_TIER_MIN_GAIN", 3.0)


def build_leafpair_reverse_tiers(
    source_valid: Any,
    *,
    slot_tile: int,
    max_tiers: Optional[int] = None,
    min_gain: Optional[float] = None,
) -> Optional[Tuple[Tuple[Tuple[int, ...], int], ...]]:
    """Partition leaves into static occupancy tiers for the analytic reverse.

    The prepacked payload is a rectangle whose width is the GLOBAL maximum
    neighbour count. That maximum comes from a single pathological leaf (measured:
    exactly ``leaves - 1``, i.e. it neighbours every other leaf, at every N and
    every geometry tried), while the median leaf uses a fraction of it -- fill is
    45% at N=200000 and 14.5% at N=1000000. Every leaf therefore pays the worst
    leaf's width.

    This splits the leaves into a few groups by how many slots they actually use,
    so each group's reverse pass reads only its own width. Two properties matter:

    * widths are **static** (computed here, on the host, from the frozen topology),
      so each pass compiles to a right-sized ``lax.scan``; and
    * leaves keep their **Morton order within a tier** -- only the visiting order
      across tiers changes, and source ids are never renumbered. A global
      occupancy sort was tried instead and ran ~7x slower precisely because it
      broke that locality.

    Returns ``None`` when tiering cannot pay (one tier, or no usable split), which
    keeps the single-pass path byte-identical. Leaf ids come back as plain int
    tuples so the result is hashable -- it rides through the ``custom_vjp`` in
    ``nondiff_argnums``, which JAX requires to be hashable and comparable.
    """
    valid = np.asarray(jax.device_get(source_valid))
    num_leaves = int(valid.shape[0])
    if num_leaves == 0:
        return None
    slots = int(np.prod(valid.shape[1:]))
    counts = valid.reshape(num_leaves, slots).sum(axis=1)
    tile = max(1, int(slot_tile))
    # Round each leaf's occupancy up to a whole tile: that is the width its pass
    # must read.
    widths = (np.maximum(counts, 1) + tile - 1) // tile * tile
    limit = int(_grad_rev_tier_max() if max_tiers is None else max_tiers)
    if limit <= 1:
        return None

    unique = np.unique(widths)
    if unique.size <= 1:
        return None  # uniform occupancy: the rectangle is already right-sized
    # Geometric tier edges: cheap, and matches how occupancy is distributed (a long
    # tail of near-empty leaves under one very wide outlier).
    edges = np.unique(
        np.round(
            np.geomspace(max(unique[0], tile), unique[-1], num=min(limit, unique.size))
        ).astype(np.int64)
    )
    edges = (edges + tile - 1) // tile * tile

    tiers: list[tuple[tuple[int, ...], int]] = []
    lower = 0
    for edge in edges:
        members = np.nonzero((widths > lower) & (widths <= edge))[0]
        if members.size:
            # np.nonzero returns ascending indices => Morton order preserved.
            tiers.append((tuple(int(i) for i in members), int(min(edge, slots))))
        lower = int(edge)
    if not tiers or len(tiers) == 1:
        return None
    covered = sum(len(t[0]) for t in tiers)
    if covered != num_leaves:  # defensive: never silently drop a leaf
        raise ValueError(
            f"reverse tiering covered {covered} of {num_leaves} leaves; "
            "every leaf must appear in exactly one tier"
        )
    # Accept only when the slot-visit saving is big enough to pay for the extra
    # indirection. Tiering makes the target-side gather go through an index array
    # instead of a consecutive range, and splits one scan into several smaller
    # ones; both cost throughput, and at small leaf counts the per-pass fixed cost
    # is amortised over very little work. MEASURED, near-field reverse only,
    # contention-corrected against the untouched far-field row:
    #
    #   N=200000  (782 leaves): predicted 1.64x fewer slot visits -> 4.3x SLOWER
    #   N=1000000 (3907 leaves): predicted 4.33x fewer slot visits -> 1.91x FASTER
    #
    # So the break-even predicted reduction lies between those; 3.0 is the
    # conservative pick. Calibrated on exactly two points on one A100 -- re-measure
    # before trusting it on other hardware or a very different occupancy profile.
    tiered_slots = sum(len(t[0]) * int(t[1]) for t in tiers)
    if tiered_slots <= 0:
        return None
    reduction = float(num_leaves * slots) / float(tiered_slots)
    if reduction < float(_grad_rev_tier_min_gain() if min_gain is None else min_gain):
        return None
    return tuple(tiers)


_TIER_CACHE_MAX = 8
_tier_cache: "OrderedDict[tuple[Any, ...], tuple[Any, Any]]" = OrderedDict()


def clear_leafpair_reverse_tier_cache() -> None:
    """Drop every memoized reverse-tier plan (tests / memory pressure)."""
    _tier_cache.clear()


def _leafpair_reverse_tiers_cached(
    cache_key_mask: Any,
    source_valid: Any,
    *,
    slot_tile: int,
    max_tiers: Optional[int] = None,
    min_gain: Optional[float] = None,
) -> Optional[Tuple[Tuple[Tuple[int, ...], int], ...]]:
    """Memoized :func:`build_leafpair_reverse_tiers`, keyed on frozen topology.

    Tiering depends only on the payload's validity mask, which is frozen
    topology -- but the builder has to pull that mask to the host to histogram
    it. Uncached, that is a device-to-host copy of the whole padded mask on
    **every forward call** (32M elements at N=1000000), for a value that cannot
    change between steps of an optimisation loop.

    Keyed by the *identity* of the payload's own mask array, the same cheap-and-
    sound key :func:`~jaccpot.runtime._nearfield_fastlane.leaf_major_nearfield_payload_cached`
    uses: the entry keeps a strong reference to that array, so a live entry pins
    its key and no freed object can have its id reused by a different array.

    ``cache_key_mask`` is the raw payload attribute (stable across calls, since
    the payload itself is memoized); ``source_valid`` is what actually gets
    histogrammed. They are the same data -- passing both lets the reduction read
    the host-side original when there is one, instead of a fresh device copy.
    """
    resolved_max_tiers = _grad_rev_tier_max() if max_tiers is None else int(max_tiers)
    resolved_min_gain = (
        _grad_rev_tier_min_gain() if min_gain is None else float(min_gain)
    )
    key = (id(cache_key_mask), int(slot_tile), resolved_max_tiers, resolved_min_gain)
    hit = _tier_cache.get(key)
    if hit is not None:
        _tier_cache.move_to_end(key)
        return hit[1]

    source = cache_key_mask if isinstance(cache_key_mask, np.ndarray) else source_valid
    try:
        tiers = build_leafpair_reverse_tiers(
            source,
            slot_tile=slot_tile,
            max_tiers=resolved_max_tiers,
            min_gain=resolved_min_gain,
        )
    except jax.errors.JAXTypeError:
        # The mask reached us as a tracer, so no static slot width can be read.
        # The single full-width pass is always correct; do NOT cache this, since
        # the key would be a tracer's id.
        return None
    except Exception as exc:  # pragma: no cover - defensive
        # Tiering is a pure optimisation, so a failure here must not break an
        # otherwise-working gradient -- but it must not be silent either: falling
        # back costs ~1.9x on the reverse at N=1000000.
        warnings.warn(
            f"reverse-pass occupancy tiering failed ({exc!r}); falling back to a "
            "single full-width pass. The gradient is unaffected, but the reverse "
            "will be slower at large N.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    _tier_cache[key] = (cache_key_mask, tiers)
    while len(_tier_cache) > _TIER_CACHE_MAX:
        _tier_cache.popitem(last=False)
    return tiers


def _leafpair_accel_analytic_vjp(
    leaf_positions: Array,
    leaf_masses: Array,
    leaf_mask: Array,
    leaf_particle_idx: Array,
    source_leaf_ids: Array,
    source_valid: Array,
    cotangent: Array,
    *,
    softening_sq: Array,
    G: Array,
    leaf_batch: int,
    slot_tile: int,
    skip_empty_tiles: bool = True,
    occupancy_sort: bool = False,
    tiers: Optional[Tuple[Tuple[Any, int], ...]] = None,
) -> Tuple[Array, Array]:
    """Analytic reverse of the leaf-pair near field in **O(N) memory**.

    Returns ``(leaf_positions_bar, leaf_masses_bar)`` for a particle-order output
    ``cotangent``.

    Why hand-written rather than ``jax.vjp`` of a pure-JAX twin: a ``bwd`` rule is
    never itself differentiated, so everything it computes is a *transient* bounded
    by the tile size instead of a *residual* retained per scan iteration. Taking
    the reverse from ``jax.vjp(twin)`` is correct but linearizes the twin, which
    reinstates the twin's own O(edges * W) scan residuals as a peak during the
    backward -- ~67 GB at N=1048576 on the canonical leaf-256 config. This form
    keeps only O(N) accumulators, which is what makes 1M gradients reachable.

    The contraction is the same symmetric near-field tidal tensor already used by
    :func:`_pair_accel_cvjp_bwd`,
    ``J = -G sum_s m_s (I/r^3 - 3 r r^T / r^5)`` with ``J^T c = J c``, just applied
    per leaf pair and accumulated leaf-major so no particle-order scatter is
    needed (cotangents are w.r.t. the leaf-major tensors; the caller's gather
    transposes them back to particle order).

    Newton's third law supplies the source-side term: the contribution a target
    leaf receives from a source leaf gives equal and opposite position sensitivity
    to that source leaf, scattered by source leaf id.
    """
    dtype = leaf_positions.dtype
    num_leaves = int(leaf_positions.shape[0])
    width = int(leaf_positions.shape[1])
    num_slots = int(source_leaf_ids.shape[1]) * int(source_leaf_ids.shape[2])
    if num_leaves == 0 or width == 0 or num_slots == 0:
        return jnp.zeros_like(leaf_positions), jnp.zeros_like(leaf_masses)

    # Normalise the scalars to arrays. A caller whose ``G``/``softening_sq`` are
    # Python constants inside a jitted region hands them over as JAX host-side
    # literals (``TypedNdArray``), which reach this rule through the custom_vjp
    # residual and do not implement ``__neg__`` -- ``-G`` below then raises. The
    # single-GPU caller happens to pass concrete device arrays, so only a jitted
    # caller (e.g. the distributed shard_map body) trips it.
    G = jnp.asarray(G, dtype)
    softening_sq = jnp.asarray(softening_sq, dtype)

    slot_ids = jnp.reshape(source_leaf_ids, (num_leaves, num_slots))
    slot_valid = jnp.reshape(source_valid, (num_leaves, num_slots))

    # Cotangent, gathered leaf-major and masked exactly as the forward masks its
    # output. O(N).
    cot_leaf = jnp.where(
        leaf_mask[..., None], cotangent[leaf_particle_idx], jnp.zeros((), dtype)
    )

    # DO NOT occupancy-SORT here (permuting the leaf arrays themselves). Tried and
    # measured **~6.8x SLOWER** at N=200000 (near-only reverse 1632 -> 20987 ms,
    # after dividing out a 1.9x GPU-contention factor read off the untouched
    # far-field row) and a wash at 1M. Leaves arrive in Morton order, which keeps
    # the per-tile source gather ``leaf_positions[safe_src]`` and the
    # ``.at[safe_src].add`` scatter spatially coherent; a global permutation
    # destroys that and the extra memory traffic dwarfs the arithmetic saved.
    #
    # ``tiers`` is the salvaged version of that idea. It changes only WHICH TARGET
    # leaves each pass visits and how many slots that pass reads -- source ids are
    # never renumbered and ``leaf_positions`` is never permuted, so source-side
    # locality is untouched. Each tier is a (leaf-index array, slot width) pair with
    # STATIC width, built on the host from the occupancy histogram, and leaves stay
    # in Morton order within a tier. A tier of near-empty leaves then reads only its
    # own narrow slot window instead of the global maximum, which is what actually
    # removes the padding: measured fill is 45% at N=200000 and 14.5% at N=1000000,
    # i.e. most of the rectangle is padding set by a single pathological leaf.
    del occupancy_sort

    batch = max(1, min(int(leaf_batch), num_leaves))
    tile = max(1, min(int(slot_tile), num_slots))
    leaf_offsets = jnp.arange(batch, dtype=INDEX_DTYPE)
    slot_offsets = jnp.arange(tile, dtype=INDEX_DTYPE)

    def _pass(carry, tier_leaves, tier_slots):
        """One occupancy tier: ``tier_leaves`` targets against ``tier_slots`` slots."""
        tier_count = int(tier_leaves.shape[0])
        leaf_starts = jnp.arange(0, tier_count, batch, dtype=INDEX_DTYPE)
        slot_starts = jnp.arange(0, tier_slots, tile, dtype=INDEX_DTYPE)

        def leaf_body(
            carry: Tuple[Array, Array], leaf_start: Array
        ) -> Tuple[Tuple[Array, Array], None]:
            pos_bar, mass_bar = carry
            pos_in_tier = leaf_start + leaf_offsets
            tgt_in_range = pos_in_tier < tier_count
            # Target leaf ids stay GLOBAL: only the visiting order changes.
            safe_tgt = tier_leaves[jnp.where(tgt_in_range, pos_in_tier, 0)]

            tgt_pos = leaf_positions[safe_tgt]  # (B, W, 3)
            tgt_mask = leaf_mask[safe_tgt] & tgt_in_range[:, None]
            cot_t = jnp.where(
                tgt_mask[..., None], cot_leaf[safe_tgt], jnp.zeros((), dtype)
            )

            def slot_body(
                inner: Tuple[Array, Array, Array], slot_start: Array
            ) -> Tuple[Tuple[Array, Array, Array], None]:
                pos_acc, mass_acc, tgt_acc = inner
                sl = slot_start + slot_offsets
                sl_in_range = sl < tier_slots
                safe_sl = jnp.where(sl_in_range, sl, 0)

                src_leaf = slot_ids[safe_tgt][:, safe_sl]  # (B, T)
                valid_slot = slot_valid[safe_tgt][:, safe_sl] & sl_in_range[None, :]
                valid_slot = valid_slot & tgt_in_range[:, None]
                safe_src = jnp.where(valid_slot, src_leaf, 0)

                def _apply(acc_in):
                    pos_in, mass_in, tgt_in = acc_in
                    src_pos = leaf_positions[safe_src]  # (B, T, W, 3)
                    src_mass = leaf_masses[safe_src]  # (B, T, W)
                    src_mask = leaf_mask[safe_src] & valid_slot[..., None]

                    # (B, T, Wt, Ws, 3)
                    diff = tgt_pos[:, None, :, None, :] - src_pos[:, :, None, :, :]
                    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
                    pair_mask = tgt_mask[:, None, :, None] & src_mask[:, :, None, :]
                    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
                    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
                    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
                    inv_dist5 = jnp.where(pair_mask, inv_dist3 * inv_r * inv_r, 0.0)

                    cot_b = cot_t[:, None, :, None, :]  # (B, 1, Wt, 1, 3)
                    cd = jnp.sum(cot_b * diff, axis=-1)  # (B, T, Wt, Ws)
                    m = src_mass[:, :, None, :, None]  # (B, T, 1, Ws, 1)
                    pair = m * (
                        inv_dist3[..., None] * cot_b
                        - 3.0 * inv_dist5[..., None] * cd[..., None] * diff
                    )

                    # Target side: sum over sources (slots and their particles).
                    tgt_contrib = -G * jnp.sum(pair, axis=(1, 3))  # (B, Wt, 3)
                    # Source side (third law): sum over targets, scattered by source leaf.
                    src_contrib = G * jnp.sum(pair, axis=2)  # (B, T, Ws, 3)
                    src_mass_contrib = -G * jnp.sum(
                        inv_dist3 * cd, axis=2
                    )  # (B, T, Ws)

                    src_contrib = jnp.where(
                        valid_slot[..., None, None], src_contrib, 0.0
                    )
                    src_mass_contrib = jnp.where(
                        valid_slot[..., None], src_mass_contrib, 0.0
                    )

                    return (
                        pos_in.at[safe_src].add(src_contrib),
                        mass_in.at[safe_src].add(src_mass_contrib),
                        tgt_in + tgt_contrib,
                    )

                # Skip whole tiles that carry no valid source slot. Every term above is
                # masked by ``valid_slot``, so an all-invalid tile contributes exactly
                # zero -- this is a pure work saving, not an approximation. It matters
                # because the prepacked payload is padded to the GLOBAL maximum neighbour
                # count (one leaf typically neighbours every other), so the fill is
                # 45% at N=200000 and 14.5% at N=1000000 on the canonical galaxy config:
                # most tiles are pure padding. The forward lane has had this skip all
                # along (``_accumulate_target_block_tile_sequence``); its absence here is
                # why the near-field reverse ran ~18x its own forward.
                if skip_empty_tiles:
                    return (
                        lax.cond(jnp.any(valid_slot), _apply, lambda acc: acc, inner),
                        None,
                    )
                return _apply(inner), None

            (pos_bar, mass_bar, tgt_total), _ = lax.scan(
                slot_body,
                (pos_bar, mass_bar, jnp.zeros_like(tgt_pos)),
                slot_starts,
            )
            tgt_total = jnp.where(tgt_in_range[:, None, None], tgt_total, 0.0)
            pos_bar = pos_bar.at[safe_tgt].add(tgt_total)
            return (pos_bar, mass_bar), None

        return lax.scan(leaf_body, carry, leaf_starts)[0]

    carry = (jnp.zeros_like(leaf_positions), jnp.zeros_like(leaf_masses))
    if tiers is None:
        carry = _pass(carry, jnp.arange(num_leaves, dtype=INDEX_DTYPE), num_slots)
    else:
        # Python loop: tier count and each tier's slot width are STATIC, so every
        # pass compiles to its own right-sized scan.
        for tier_leaves, tier_slots in tiers:
            if len(tier_leaves) == 0 or int(tier_slots) == 0:
                continue
            carry = _pass(
                carry,
                jnp.asarray(tier_leaves, dtype=INDEX_DTYPE),
                int(tier_slots),
            )
    positions_bar, masses_bar = carry
    return positions_bar, masses_bar


def _pair_accel_masked_accels(
    target_positions: Array,
    source_positions: Array,
    source_masses: Array,
    target_mask: Array,
    source_mask: Array,
    softening_sq: Union[float, Array],
    G: Array,
) -> Array:
    """Accel-only batched pair contributions (matches the accel output of
    :func:`_pair_contributions_batched`)."""
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]
    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
    weighted = inv_dist3 * source_masses[:, None, :]
    accels = -G * jnp.sum(weighted[..., None] * diff, axis=2)
    return jnp.where(target_mask[..., None], accels, 0.0)


def _pair_accel_pair_terms(
    target_positions: Array,
    source_positions: Array,
    target_mask: Array,
    source_mask: Array,
    softening_sq: Union[float, Array],
) -> Tuple[Array, Array, Array]:
    """``(diff, inv_dist3, inv_dist5)`` exactly as :func:`_pair_accel_masked_accels`
    forms them.

    Factored out so the analytic reverse rule can REMATERIALIZE these
    ``(B, Wt, Ws)``-shaped pair intermediates instead of carrying them in the
    ``custom_vjp`` residual, without the forward and reverse expressions drifting
    apart. See :func:`_pair_accel_cvjp_fwd` for why that matters.
    """
    diff = target_positions[:, :, None, :] - source_positions[:, None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1) + softening_sq
    pair_mask = target_mask[:, :, None] & source_mask[:, None, :]
    safe_dist_sq = jnp.where(pair_mask, dist_sq, jnp.ones_like(dist_sq))
    inv_r = jnp.where(pair_mask, lax.rsqrt(safe_dist_sq), 0.0)
    inv_dist3 = jnp.where(pair_mask, inv_r * inv_r * inv_r, 0.0)
    inv_dist5 = jnp.where(pair_mask, inv_dist3 * inv_r * inv_r, 0.0)
    return diff, inv_dist3, inv_dist5


@jax.custom_vjp
def _pair_accel_cvjp(
    target_positions: Array,
    source_positions: Array,
    source_masses: Array,
    target_mask_f: Array,
    source_mask_f: Array,
    softening_sq: Array,
    G: Array,
) -> Array:
    """Accel-only batched near-field pair kernel with an analytic reverse rule.

    All arguments are float arrays (masks as 0/1 floats), so the reverse returns
    ordinary zero cotangents for the non-differentiated inputs (masks, softening,
    G) -- no closure over tracers (which ``custom_vjp`` forbids). The forward is
    byte-identical to the accel output of :func:`_pair_contributions_batched`; the
    reverse is the analytic symmetric near-field tidal tensor
    ``J = -G Σ_s m_s (I/r³ − 3 r rᵀ/r⁵)`` contracted with the output cotangent
    (``Jᵀc = Jc``) in one extra pair pass -- exactly the reverse rule a future
    fused-Pallas near-field ``custom_vjp`` reuses. Verified bit-for-bit against
    autodiff in ``tests/unit/test_custom_vjp_parity.py``.
    """
    target_mask = target_mask_f > 0.5
    source_mask = source_mask_f > 0.5
    return _pair_accel_masked_accels(
        target_positions,
        source_positions,
        source_masses,
        target_mask,
        source_mask,
        softening_sq,
        G,
    )


def _pair_accel_cvjp_fwd(
    target_positions,
    source_positions,
    source_masses,
    target_mask_f,
    source_mask_f,
    softening_sq,
    G,
):
    # The residual carries only the O(B*W) INPUTS; the O(B*Wt*Ws) pair
    # intermediates are rematerialized in the reverse pass. Storing
    # (diff, inv_dist3, inv_dist5) instead cost 5 doubles per particle PAIR, and
    # because the bucketed near-field drives this kernel from a ``lax.scan`` over
    # edge chunks, reverse mode retained EVERY chunk's residual -- i.e.
    # ``40 * near_field_edges * max_leaf_size**2`` bytes in total, independent of
    # ``edge_chunk_size``. Measured: a single 52.14 GiB ``diff`` allocation at
    # N=65536 / leaf 64 on a 40 GB A100, versus 1.13 GB for the whole forward.
    # Recomputing costs one extra pair pass inside a backward that already
    # materializes a (B, Wt, Ws, 3) array, and keeps the reverse residual
    # O(edges * max_leaf_size) so galaxy-scale N fits.
    accels = _pair_accel_masked_accels(
        target_positions,
        source_positions,
        source_masses,
        target_mask_f > 0.5,
        source_mask_f > 0.5,
        softening_sq,
        G,
    )
    residual = (
        target_positions,
        source_positions,
        source_masses,
        target_mask_f,
        source_mask_f,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )
    return accels, residual


def _pair_accel_cvjp_bwd(residual, cotangent):
    (
        target_positions,
        source_positions,
        source_masses,
        target_mask_f,
        source_mask_f,
        softening_sq,
        G,
    ) = residual
    # Rematerialize the pair intermediates the forward deliberately did not save
    # (see _pair_accel_cvjp_fwd). Same expressions, so the analytic reverse below
    # is unchanged bit-for-bit.
    diff, inv_dist3, inv_dist5 = _pair_accel_pair_terms(
        target_positions,
        source_positions,
        target_mask_f > 0.5,
        source_mask_f > 0.5,
        softening_sq,
    )
    # Forward masks accels by target_mask, so mask the incoming cotangent alike.
    cot = jnp.where(target_mask_f[..., None] > 0.5, cotangent, 0.0)  # (B, Wt, 3)
    cd = jnp.sum(cot[:, :, None, :] * diff, axis=-1)  # (B, Wt, Ws) = c_t · r_ts
    m = source_masses[:, None, :, None]  # (B, 1, Ws, 1)
    # per-pair P_{t,s,l} = m_s [ inv_dist3 c_{t,l} - 3 inv_dist5 (c_t·r) r_l ]
    pair = m * (
        inv_dist3[..., None] * cot[:, :, None, :]
        - 3.0 * inv_dist5[..., None] * cd[..., None] * diff
    )  # (B, Wt, Ws, 3)
    target_positions_bar = -G * jnp.sum(pair, axis=2)  # sum over sources
    source_positions_bar = G * jnp.sum(pair, axis=1)  # sum over targets (3rd law)
    source_masses_bar = -G * jnp.sum(inv_dist3 * cd, axis=1)  # sum over targets
    # Zero cotangents for the non-differentiated inputs (masks, softening, G).
    return (
        target_positions_bar,
        source_positions_bar,
        source_masses_bar,
        jnp.zeros_like(target_mask_f),
        jnp.zeros_like(source_mask_f),
        jnp.zeros_like(softening_sq),
        jnp.zeros_like(G),
    )


_pair_accel_cvjp.defvjp(_pair_accel_cvjp_fwd, _pair_accel_cvjp_bwd)

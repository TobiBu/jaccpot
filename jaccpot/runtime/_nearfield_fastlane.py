"""Leaf-major near-field fast lane for the radix ``FMMPreparedState``.

The radix prepared state carries the near field as an **edge list**: a CSR
neighbour buffer plus the flat ``(target_leaf, source_leaf)`` vectors the
bucketed kernel scans over. The fast lane
(:func:`jaccpot.nearfield._fast_lane.compute_leaf_p2p_accelerations_radix_fast_lane`)
instead wants a **leaf-major** payload -- for every target leaf, a padded block
of source-leaf ids -- which is the layout the large-N pipeline bakes at
``prepare_state`` time and the radix pipeline does not.

This module builds that payload from the CSR view the radix state already has,
so the differentiable path can route the near field through the fast lane's
``custom_vjp`` (Pallas forward + analytic O(N) leaf-pair reverse) instead of the
bucketed gather/scatter, which profiling showed dominates both the forward
(~83%) and the reverse (~91%) of ``differentiable_accelerations``.

**The transpose runs on the HOST, in NumPy.** Two reasons, one of them
load-bearing:

* it is pure frozen topology, so the result belongs in the jaxpr as a constant,
  not as index arithmetic restaged on every call; and
* under an outer ``jax.jit`` every ``jnp`` operation is staged out even when its
  inputs are concrete constants, so a device-side ``jnp.max(counts)`` (needed to
  size the padded block) is a tracer and cannot become a static shape. Doing the
  reduction in NumPy on the concrete prepared-state arrays sidesteps that
  entirely -- the same trick ``differentiable_accelerations`` already uses for
  the inverse permutation.

The payload depends only on the frozen topology, so it is memoized per
neighbour-list identity: an optimisation loop differentiating the same prepared
state rebuilds it once, not once per step.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import jax
import numpy as np

from ._large_n_types import RadixFastNearfieldPayload
from .dtypes import INDEX_DTYPE
from .fmm_constants import _env_int

# Leaf-major payload knobs. The defaults mirror the large-N pipeline
# (``nearfield_target_leaf_batch_size=32``, ``fallback_block_tile_size=8``);
# ``BLOCK_SIZE`` is the inner lane width of the ``(leaf, block, lane)`` layout.
_FASTLANE_BLOCK_SIZE = "JACCPOT_NEARFIELD_FASTLANE_BLOCK_SIZE"
_FASTLANE_LEAF_BATCH = "JACCPOT_NEARFIELD_FASTLANE_LEAF_BATCH"
_FASTLANE_BLOCK_TILE = "JACCPOT_NEARFIELD_FASTLANE_BLOCK_TILE"

_PAYLOAD_CACHE_MAX = 8
_payload_cache: (
    "OrderedDict[tuple[Any, ...], tuple[Any, RadixFastNearfieldPayload]]"
) = OrderedDict()


class NearfieldTopologyNotConcrete(RuntimeError):
    """Raised when the frozen near-field topology reaches the builder as tracers."""


def clear_nearfield_fastlane_payload_cache() -> None:
    """Drop every memoized leaf-major payload (tests / memory pressure).

    Returns
    -------
    None
        Clears the module-level cache in place.
    """
    _payload_cache.clear()


def _host(array: Any, what: str) -> np.ndarray:
    """Pull a frozen-topology array to the host, or say precisely why we cannot.

    Parameters
    ----------
    array : Any
        Frozen-topology array to materialise on the host.
    what : str
        Name of the array, used to make the failure message actionable.

    Returns
    -------
    np.ndarray
        The host copy.

    Raises
    ------
    NearfieldTopologyNotConcrete
        If ``array`` is a tracer. This lane needs host constants -- see the
        module docstring -- so a traced input cannot be worked around here and
        the caller has to prepare outside the trace.
    """
    try:
        return np.asarray(jax.device_get(array))
    except Exception as exc:  # tracer, or anything else non-concrete
        raise NearfieldTopologyNotConcrete(
            f"the leaf-major near-field payload needs a concrete {what}, but it "
            "arrived as a traced value. The payload is frozen topology and must "
            "be derived from a prepared state built OUTSIDE the trace; a "
            "near-field view constructed inside jax.jit cannot be used."
        ) from exc


def _fastlane_tiles() -> tuple[int, int, int]:
    """Read the fast-lane tile sizes from the environment.

    Returns
    -------
    tuple[int, int, int]
        ``(batch_tile_t, batch_tile_s, fallback_block_tile_size)``.
    """
    return (
        max(1, _env_int(_FASTLANE_BLOCK_SIZE, 8)),
        max(1, _env_int(_FASTLANE_LEAF_BATCH, 32)),
        max(1, _env_int(_FASTLANE_BLOCK_TILE, 8)),
    )


def resolve_leaf_particle_groups(
    *,
    node_ranges: np.ndarray,
    leaf_nodes: np.ndarray,
    num_particles: int,
    max_leaf_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``(leaf_particle_indices, leaf_particle_mask)`` from node ranges.

    The radix ``nearfield_interop`` view carries explicit particle groups only
    for some configurations; otherwise leaf membership is implicit in the
    contiguous Morton ranges. This reproduces
    :func:`~jaccpot.nearfield.near_field._prepare_leaf_data`'s index/mask half so
    both lanes address identical particles.

    Parameters
    ----------
    node_ranges : np.ndarray
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : np.ndarray
        Node index of each leaf.
    num_particles : int
        Total particle count.
    max_leaf_size : int
        Slot capacity per leaf.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(leaf_particle_indices, leaf_particle_mask)``, both
        ``(num_leaves, max_leaf_size)``.
    """
    leaf_ranges = node_ranges[leaf_nodes]
    counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
    idx = np.arange(int(max_leaf_size), dtype=np.int64)
    particle_idx = leaf_ranges[:, 0][:, None] + idx[None, :]
    valid = idx[None, :] < counts[:, None]
    safe_idx = np.clip(particle_idx, 0, max(int(num_particles) - 1, 0))
    return safe_idx, valid


def build_leaf_major_nearfield_payload(
    *,
    node_ranges: Any,
    leaf_nodes: Any,
    offsets: Any,
    neighbors: Any,
    num_particles: int,
    max_leaf_size: int,
    leaf_particle_indices: Optional[Any] = None,
    leaf_particle_mask: Optional[Any] = None,
) -> RadixFastNearfieldPayload:
    """Transpose the CSR neighbour edge list into the leaf-major fast-lane layout.

    ``offsets``/``neighbors`` are the near-field CSR view (node ids, possibly
    with ``-1`` padding inside each leaf's slice when the neighbour list is not
    compacted -- the traced/``shard_map`` branch keeps a full
    ``[num_leaves * max_neighbors]`` buffer). Padding is masked out here exactly
    the way :func:`~jaccpot.nearfield.near_field.prepare_leaf_neighbor_pairs`
    masks it for the bucketed lane, so both lanes see the *same* edge set and
    therefore compute the same force.

    ``source_particle_ids``/``source_particle_mask`` are returned empty on
    purpose: that is what selects the prepacked source-leaf-id layout (the lane
    with the analytic O(N) reverse) over the materialized per-particle "pairs"
    layout, whose reverse is not wired.

    Parameters
    ----------
    node_ranges : Any
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : Any
        Node index of each leaf.
    offsets : Any
        Per-leaf offsets into ``neighbors``.
    neighbors : Any
        Flat neighbour-node list, possibly ``-1``-padded inside each leaf slice.
    num_particles : int
        Total particle count.
    max_leaf_size : int
        Slot capacity per leaf.
    leaf_particle_indices : Optional[Any]
        Explicit per-leaf membership when the interop view carries it, else
        ``None`` to derive it from ``node_ranges``.
    leaf_particle_mask : Optional[Any]
        Occupancy mask matching ``leaf_particle_indices``.

    Returns
    -------
    RadixFastNearfieldPayload
        Host-resident (NumPy, not JAX) payload. Host constants re-enter each
        trace cleanly, whereas arrays minted inside one trace would leak into
        the next as that trace's tracers.

    Raises
    ------
    ValueError
        If the CSR ``offsets`` describe a different number of leaves than the
        leaf-particle groups do. The two must share one leaf ordering, and a
        mismatch means the caller mixed views from different topologies.
    """
    block, batch, tile = _fastlane_tiles()

    node_ranges_np = _host(node_ranges, "node_ranges").astype(np.int64, copy=False)
    leaf_nodes_np = _host(leaf_nodes, "leaf_nodes").astype(np.int64, copy=False)
    offsets_np = _host(offsets, "neighbour offsets").astype(np.int64, copy=False)
    neighbors_np = _host(neighbors, "neighbour node ids").astype(np.int64, copy=False)

    if leaf_particle_indices is None or leaf_particle_mask is None:
        target_ids_np, target_mask_np = resolve_leaf_particle_groups(
            node_ranges=node_ranges_np,
            leaf_nodes=leaf_nodes_np,
            num_particles=int(num_particles),
            max_leaf_size=int(max_leaf_size),
        )
    else:
        target_ids_np = _host(leaf_particle_indices, "leaf_particle_indices").astype(
            np.int64, copy=False
        )
        target_mask_np = _host(leaf_particle_mask, "leaf_particle_mask").astype(bool)

    num_leaves = int(target_ids_np.shape[0])
    total_nodes = int(node_ranges_np.shape[0])
    index_dtype = np.dtype(INDEX_DTYPE)

    counts_np = offsets_np[1:] - offsets_np[:-1] if offsets_np.size > 1 else None
    if counts_np is not None and int(counts_np.shape[0]) != num_leaves:
        # The CSR segmentation and the leaf-particle groups must index the same
        # leaves in the same order, or the payload would attribute one leaf's
        # neighbours to another -- a wrong force, not a crash.
        raise ValueError(
            "near-field CSR offsets describe "
            f"{int(counts_np.shape[0])} leaves but the leaf-particle groups have "
            f"{num_leaves}; they must be the same leaf ordering."
        )
    max_neighbors = (
        int(counts_np.max()) if (num_leaves > 0 and counts_np is not None) else 0
    )

    # NumPy, not jnp, all the way out. The payload is memoized and reused across
    # traces; jnp arrays minted inside the first trace would be that trace's
    # tracers and leak into the next one (UnexpectedTracerError). Host constants
    # re-enter each trace cleanly -- the lane's own ``jnp.asarray`` converts them.
    def _payload(
        source_ids: np.ndarray, source_valid: np.ndarray
    ) -> RadixFastNearfieldPayload:
        return RadixFastNearfieldPayload(
            target_leaf_ids=np.arange(num_leaves, dtype=index_dtype),
            target_particle_ids=target_ids_np.astype(index_dtype, copy=False),
            target_particle_mask=target_mask_np,
            source_leaf_ids=source_ids.astype(index_dtype, copy=False),
            source_leaf_valid_mask=source_valid,
            source_particle_ids=np.zeros((0, 0, 0), dtype=index_dtype),
            source_particle_mask=np.zeros((0, 0, 0), dtype=bool),
            batch_tile_t=batch,
            batch_tile_s=block,
            fallback_block_tile_size=tile,
        )

    if num_leaves == 0 or max_neighbors == 0:
        # No cross-leaf near field (single-leaf tree, or every neighbour slot
        # padded). The lane still needs a rank-3 source array so the prepacked
        # branch is selected; an all-invalid block yields a zero pair term and
        # leaves only the intra-leaf self interaction.
        return _payload(
            np.zeros((num_leaves, 1, block), dtype=index_dtype),
            np.zeros((num_leaves, 1, block), dtype=bool),
        )

    num_blocks = (max_neighbors + block - 1) // block
    slots = num_blocks * block

    # node id -> leaf slot; -1 for interior nodes and for the padding sentinel.
    leaf_lookup = np.full((total_nodes,), -1, dtype=np.int64)
    leaf_lookup[leaf_nodes_np] = np.arange(num_leaves, dtype=np.int64)

    slot_offsets = np.arange(slots, dtype=np.int64)
    in_range = slot_offsets[None, :] < counts_np[:, None]
    edge_idx = np.where(in_range, offsets_np[:-1, None] + slot_offsets[None, :], 0)
    neighbor_nodes = neighbors_np[edge_idx]
    source_leaf_ids = leaf_lookup[np.clip(neighbor_nodes, 0, total_nodes - 1)]
    valid = in_range & (neighbor_nodes >= 0) & (source_leaf_ids >= 0)
    source_leaf_ids = np.where(valid, source_leaf_ids, 0)

    return _payload(
        source_leaf_ids.reshape((num_leaves, num_blocks, block)),
        valid.reshape((num_leaves, num_blocks, block)),
    )


def leaf_major_nearfield_payload_cached(
    *,
    node_ranges: Any,
    leaf_nodes: Any,
    offsets: Any,
    neighbors: Any,
    num_particles: int,
    max_leaf_size: int,
    leaf_particle_indices: Optional[Any] = None,
    leaf_particle_mask: Optional[Any] = None,
) -> RadixFastNearfieldPayload:
    """Memoized :func:`build_leaf_major_nearfield_payload`.

    Keyed by the *identity* of the topology arrays -- they come off a frozen
    prepared state, so identity is the cheapest sound key. The cache entry keeps
    a strong reference to those arrays, which is what makes ``id()`` safe here:
    a live entry pins its arrays, so no freed object can have its id reused by a
    different array that would then hit a stale entry.

    Parameters
    ----------
    node_ranges : Any
        Per-node ``[start, stop)`` span into the sorted particle order.
    leaf_nodes : Any
        Node index of each leaf.
    offsets : Any
        Per-leaf offsets into ``neighbors``.
    neighbors : Any
        Flat neighbour-node list, possibly ``-1``-padded inside each leaf slice.
    num_particles : int
        Total particle count.
    max_leaf_size : int
        Slot capacity per leaf.
    leaf_particle_indices : Optional[Any]
        Explicit per-leaf membership when the interop view carries it, else
        ``None`` to derive it from ``node_ranges``.
    leaf_particle_mask : Optional[Any]
        Occupancy mask matching ``leaf_particle_indices``.

    Returns
    -------
    RadixFastNearfieldPayload
        The memoized payload; identical object across calls with the same
        topology arrays.
    """
    key_arrays = (
        node_ranges,
        leaf_nodes,
        offsets,
        neighbors,
        leaf_particle_indices,
        leaf_particle_mask,
    )
    key = (
        tuple(id(arr) for arr in key_arrays),
        int(num_particles),
        int(max_leaf_size),
        _fastlane_tiles(),
    )
    hit = _payload_cache.get(key)
    if hit is not None:
        _payload_cache.move_to_end(key)
        return hit[1]

    payload = build_leaf_major_nearfield_payload(
        node_ranges=node_ranges,
        leaf_nodes=leaf_nodes,
        offsets=offsets,
        neighbors=neighbors,
        num_particles=int(num_particles),
        max_leaf_size=int(max_leaf_size),
        leaf_particle_indices=leaf_particle_indices,
        leaf_particle_mask=leaf_particle_mask,
    )
    _payload_cache[key] = (key_arrays, payload)
    while len(_payload_cache) > _PAYLOAD_CACHE_MAX:
        _payload_cache.popitem(last=False)
    return payload


def nearfield_topology_arrays(
    tree: Any,
    neighbor_list: Any,
    nearfield_interop: Optional[Any],
) -> dict[str, Any]:
    """Resolve the raw CSR topology the payload builder needs.

    Prefers the prepared state's interop view when it has one; otherwise reads
    the tree/neighbour-list arrays *directly* rather than going through
    :func:`~jaccpot.runtime.kernels.core._build_nearfield_interop_data`. That
    builder does device-side index work, and anything it produces inside an
    outer ``jax.jit`` is a tracer -- unusable here (see the module docstring).

    Parameters
    ----------
    tree : Any
        Prepared tree.
    neighbor_list : Any
        Near-field neighbour list.
    nearfield_interop : Optional[Any]
        Prepared interop view; preferred when present precisely because it was
        built outside any trace.

    Returns
    -------
    dict[str, Any]
        The CSR arrays :func:`build_leaf_major_nearfield_payload` consumes,
        keyed by its parameter names.
    """
    if nearfield_interop is not None:
        return {
            "node_ranges": nearfield_interop.node_ranges,
            "leaf_nodes": nearfield_interop.leaf_nodes,
            "offsets": nearfield_interop.offsets,
            "neighbors": nearfield_interop.neighbors,
            "leaf_particle_indices": nearfield_interop.leaf_particle_indices,
            "leaf_particle_mask": nearfield_interop.leaf_particle_mask,
        }
    return {
        "node_ranges": tree.node_ranges,
        "leaf_nodes": neighbor_list.leaf_indices,
        "offsets": neighbor_list.offsets,
        "neighbors": neighbor_list.neighbors,
        "leaf_particle_indices": None,
        "leaf_particle_mask": None,
    }


__all__ = [
    "NearfieldTopologyNotConcrete",
    "build_leaf_major_nearfield_payload",
    "clear_nearfield_fastlane_payload_cache",
    "leaf_major_nearfield_payload_cached",
    "nearfield_topology_arrays",
    "resolve_leaf_particle_groups",
]

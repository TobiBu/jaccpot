"""Switch detection for the reconstruction's rebuilt topologies.

This module is an **adapter**, not a second implementation. Identity, diffing
and switch counting all come from jaccpot's core facility
(:mod:`jaccpot.mutual.identity`, D-009): the component digests are produced by
its :func:`~jaccpot.mutual.identity.digest_arrays`, the comparable object
returned is its own :class:`~jaccpot.mutual.identity.TopologyFingerprint`, and
the run-level counting is its
:class:`~jaccpot.mutual.identity.TopologySwitchCounter`. What the adapter adds
is the two things the core facility cannot know about:

1. **The representation.** The differentiable path runs on a radix
   :class:`~jaccpot.runtime.fmm_state.FMMPreparedState`, not a
   :class:`~jaccpot.mutual.topology.MutualTopology`, so the six-component
   fingerprint has to be re-expressed over the state's own arrays.
2. **Intensive rates.** See below -- this is the substantive part.

Why an "any change?" counter is not the measurement
---------------------------------------------------
The core counter answers *did the discrete structure move at all between these
two rebuilds?* That question is **extensive**: its answer is driven to "yes" by
particle count alone, so over 3e7 parameters it reads 100% and carries no
information. D-016 recorded this the expensive way -- the Yggdrax scale-up's
headline switch rate saturated at 1.000 across five decades of N and was
reported as "uninformative" before the metric, rather than the physics, was
identified as the cause. Intensive per-particle rates over the same range
resolve a clean monotone curve.

So every rate this module reports as its own is **intensive** -- a fraction of
particles, or a fraction of interacting pairs:

``slot_churn``
    Fraction of particles whose slot in the Morton ordering moved.
``mean_rank_shift_fraction``
    Mean ``|delta slot| / P``. Falls as ``P`` grows if the churn is *local*.
``leaf_churn``
    Fraction of particles whose leaf changed **content** -- that is, the set of
    particles it shares a leaf with is not the same set. Leaf *ids* are not
    comparable across rebuilds, so this is computed from content keys.
``near_source_count_churn``
    Fraction of particles for which the number of sources reaching them by
    direct summation changed.
``near_pair_churn`` / ``far_pair_churn``
    Jaccard distance between the content-canonicalised near leaf-pair and far
    node-pair sets. **These two are the interaction-list rates** -- the ones
    D-016 says section 7 must measure for itself, because every Yggdrax rate,
    its ``leaf_change_fraction`` included, is permutation-derived and is not an
    interaction-list change.

The extensive answer is still reported, under ``extensive``, because it is the
core facility's contract and because the contrast between it and the intensive
rates is itself the point. It is labelled so no figure can mistake it for a
rate that means something at scale.

What counts as topology, and what does not
-------------------------------------------
Measured before this was written, on the ``accurate`` preset at N=240: a
position perturbation of 1e-9 changes nothing; 1e-6 changes **only the Morton
codes**; 1e-3 changes the permutation, the leaf partition and the near-field
CSR. Morton codes are quantised coordinates -- they can change without any
particle changing leaf, without any pair changing status, and without the
permutation moving. Counting them would inflate the switch rate with
quantisation noise that has no effect on the gradient. **Morton codes are
therefore excluded from identity.** The permutation, node membership, node
ranges, the far list and the near CSR are what the fixed-topology contract
freezes, and they are what is fingerprinted.

Content keys
------------
Comparing *which particles sit together* across two rebuilds needs a key that
does not depend on how the rebuild happened to label its leaves. The key used
here is an additive multiset hash: each particle id is mixed to 64 bits, and a
node's key is the sum of its members' mixed ids modulo ``2**64``. Summation
makes the key order-independent (which is what a *set* key must be) and
additive over contiguous spans, so one prefix sum over the sorted order gives
the key of **every** node -- leaves and internal alike -- in O(P). This is a
diagnostic hash, not a security parameter: nothing here defends against a
chosen-input collision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jaccpot.mutual.identity import (
    TopologyDiff,
    TopologyFingerprint,
    TopologySwitchCounter,
    digest_arrays,
)

__all__ = [
    "COMPONENTS",
    "ChurnRates",
    "RadixStructure",
    "SwitchLog",
    "churn_between",
    "fingerprint_prepared_state",
    "radix_structure",
]

#: Component names, from cheapest-to-change to most consequential. Mirrors the
#: core facility's own ordering so a diff reads the same either side.
COMPONENTS: Tuple[str, ...] = (
    "permutation",
    "tree_shape",
    "leaf_partition",
    "node_ranges",
    "far_pairs",
    "near_pairs",
)

#: Constants for the splitmix64 finaliser. Two independent streams are mixed
#: into one key so an additive multiset hash cannot be defeated by a single
#: stream's structure.
_MIX_A = np.uint64(0x9E3779B97F4A7C15)
_MIX_B = np.uint64(0xBF58476D1CE4E5B9)
_MIX_C = np.uint64(0x94D049BB133111EB)
_STREAM_SALT = np.uint64(0xD6E8FEB86659FD93)


def _host(array: Any) -> np.ndarray:
    """Bring an array to the host as NumPy, whatever it started as.

    Parameters
    ----------
    array : Any
        A NumPy array, a JAX device array, or anything ``np.asarray`` accepts.

    Returns
    -------
    np.ndarray
        Host-side array. The core identity facility deliberately refuses
        non-NumPy input -- it is documented as host-side -- so an adapter over
        a JAX-carried state has to land its arrays here first.
    """
    return np.asarray(array)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Mix an integer array to well-distributed 64-bit values.

    Parameters
    ----------
    values : np.ndarray
        Integer array, any width.

    Returns
    -------
    np.ndarray
        ``uint64`` array of the same shape. The splitmix64 finaliser; all
        arithmetic wraps modulo ``2**64``, which is exact and deterministic.
    """
    with np.errstate(over="ignore"):
        x = values.astype(np.uint64, copy=True)
        x = x + _MIX_A
        x = (x ^ (x >> np.uint64(30))) * _MIX_B
        x = (x ^ (x >> np.uint64(27))) * _MIX_C
        return x ^ (x >> np.uint64(31))


def _combine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Mix two 64-bit streams into one key.

    Parameters
    ----------
    left : np.ndarray
        First ``uint64`` stream.
    right : np.ndarray
        Second ``uint64`` stream.

    Returns
    -------
    np.ndarray
        ``uint64`` combined key.
    """
    with np.errstate(over="ignore"):
        return _splitmix64(left ^ _splitmix64(right + _STREAM_SALT))


def _expansion_order(state: Any) -> int:
    """Recover the expansion order from a prepared state's multipole width.

    Parameters
    ----------
    state : Any
        An ``FMMPreparedState``.

    Returns
    -------
    int
        ``p`` such that the multipole array is ``(num_nodes, (p + 1) ** 2)``,
        or ``-1`` if the width is unavailable or not a perfect square. The
        order is a configuration constant and cannot change between rebuilds,
        so it enters the fingerprint only to stop two runs at different orders
        comparing equal.
    """
    try:
        width = int(_host(state.upward.multipoles.multipoles).shape[-1])
    except Exception:  # pragma: no cover - backend without packed multipoles
        return -1
    root = int(round(width**0.5))
    return root - 1 if root * root == width else -1


def fingerprint_prepared_state(state: Any) -> TopologyFingerprint:
    """Fingerprint an ``FMMPreparedState`` as a core ``TopologyFingerprint``.

    Parameters
    ----------
    state : Any
        Output of :meth:`jaccpot.FastMultipoleMethod.prepare_state` on the
        radix backend.

    Returns
    -------
    TopologyFingerprint
        The **core** facility's fingerprint type, so
        :func:`~jaccpot.mutual.identity.diff_topologies` and
        :class:`~jaccpot.mutual.identity.TopologySwitchCounter` apply
        unchanged. Morton codes are deliberately not included; see the module
        docstring.
    """
    tree = state.tree
    topo = tree.topology
    near = state.neighbor_list
    far = state.interactions
    empty = np.zeros((0,), dtype=np.int64)
    return TopologyFingerprint(
        num_particles=int(tree.num_particles),
        num_nodes=int(tree.num_nodes),
        num_internal=int(topo.num_internal_nodes),
        max_leaf_size=int(state.max_leaf_size),
        order=_expansion_order(state),
        theta=float(state.theta),
        permutation=digest_arrays(
            _host(tree.inverse_permutation), _host(topo.particle_indices)
        ),
        tree_shape=digest_arrays(
            _host(topo.parent),
            _host(topo.left_child),
            _host(topo.right_child),
            _host(topo.left_is_leaf),
            _host(topo.right_is_leaf),
            _host(topo.leaf_depths),
        ),
        leaf_partition=digest_arrays(
            _host(near.leaf_indices),
            _host(near.particle_order_leaf_indices),
            _host(near.particle_order_to_native_leaf),
        ),
        node_ranges=digest_arrays(_host(topo.node_ranges)),
        far_pairs=digest_arrays(
            empty if far is None else _host(far.sources),
            empty if far is None else _host(far.targets),
            empty if far is None else _host(far.offsets),
        ),
        near_pairs=digest_arrays(
            _host(near.offsets), _host(near.neighbors), _host(near.counts)
        ),
    )


@dataclass(frozen=True)
class RadixStructure:
    """The per-particle and per-pair structure one rebuild produced.

    Everything here is host-side NumPy and rebuild-invariant, so two of these
    from different rebuilds can be compared directly. Retained arrays are O(P)
    and O(pairs); at P = 1e7 that is a few hundred MB of host memory per
    retained structure, and :class:`SwitchLog` keeps at most one previous one.

    Attributes
    ----------
    num_particles : int
        ``P`` -- sources **and** tracers, since the tree holds both. The
        tracers are a fixed subset, so at the paper's ``M << N`` they dilute
        these rates negligibly; the results JSON records both counts.
    slot_of_particle : np.ndarray
        ``(P,)`` int64. Position of each particle in the Morton ordering.
    leaf_key_of_particle : np.ndarray
        ``(P,)`` uint64 content key of the leaf each particle landed in.
    near_source_count : np.ndarray
        ``(P,)`` int64 number of source particles that reach each particle by
        direct summation.
    near_pair_keys : np.ndarray
        ``(n_near,)`` uint64, sorted and unique: the content-canonicalised near
        leaf-pair set.
    far_pair_keys : np.ndarray
        ``(n_far,)`` uint64, sorted and unique: the content-canonicalised far
        node-pair set. Empty when the far list is empty, which is what a small
        ``N`` at a tight ``theta`` gives -- every pair is near.
    """

    num_particles: int
    slot_of_particle: np.ndarray
    leaf_key_of_particle: np.ndarray
    near_source_count: np.ndarray
    near_pair_keys: np.ndarray
    far_pair_keys: np.ndarray


def _node_content_keys(
    particle_indices: np.ndarray, node_ranges: np.ndarray
) -> np.ndarray:
    """Content key of every node, from one prefix sum over the sorted order.

    Parameters
    ----------
    particle_indices : np.ndarray
        ``(P,)`` sorted-slot to original-particle-id map.
    node_ranges : np.ndarray
        ``(num_nodes, 2)`` inclusive ``[start, end]`` spans in sorted order.

    Returns
    -------
    np.ndarray
        ``(num_nodes,)`` uint64 keys. A node's key is the sum of its members'
        mixed ids modulo ``2**64``; summation is additive over contiguous
        spans, so a single prefix sum answers for every node at once, and it is
        order-independent, so it keys the *set* of members rather than their
        arrangement.
    """
    with np.errstate(over="ignore"):
        # uint64 first: NumPy refuses to xor int64 with a uint64 scalar under
        # the 'safe' casting rule, and the ids are non-negative by construction.
        ids = particle_indices.astype(np.uint64)
        stream_a = _splitmix64(ids)
        stream_b = _splitmix64(ids ^ _STREAM_SALT)
        zero = np.zeros(1, dtype=np.uint64)
        cum_a = np.concatenate([zero, np.cumsum(stream_a, dtype=np.uint64)])
        cum_b = np.concatenate([zero, np.cumsum(stream_b, dtype=np.uint64)])
        start = node_ranges[:, 0].astype(np.int64)
        stop = np.maximum(node_ranges[:, 1].astype(np.int64) + 1, start)
        return _combine(cum_a[stop] - cum_a[start], cum_b[stop] - cum_b[start])


def _canonical_pair_keys(left_keys: np.ndarray, right_keys: np.ndarray) -> np.ndarray:
    """Canonicalise a list of unordered node pairs into sorted unique keys.

    Parameters
    ----------
    left_keys : np.ndarray
        ``(n,)`` uint64 content keys of one side of each pair.
    right_keys : np.ndarray
        ``(n,)`` uint64 content keys of the other side.

    Returns
    -------
    np.ndarray
        ``(m,)`` sorted unique uint64 pair keys, ``m <= n``. Order within a
        pair is removed before hashing, so ``(a, b)`` and ``(b, a)`` collapse
        -- the near CSR lists both directions.
    """
    if left_keys.size == 0:
        return np.zeros((0,), dtype=np.uint64)
    low = np.minimum(left_keys, right_keys)
    high = np.maximum(left_keys, right_keys)
    return np.unique(_combine(low, high))


def _spans_to_indices(starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Expand half-open integer spans into the concatenated index array.

    Parameters
    ----------
    starts : np.ndarray
        ``(k,)`` span starts.
    stops : np.ndarray
        ``(k,)`` span stops, exclusive.

    Returns
    -------
    np.ndarray
        ``(sum(stops - starts),)`` int64 indices, spans concatenated in order.
        Vectorised rather than a Python loop over spans, because at N = 1e7
        there are ~1e6 of them.
    """
    lengths = np.maximum(stops - starts, 0).astype(np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros((0,), dtype=np.int64)
    # Offset ramp: a global arange minus each span's own start offset gives the
    # per-span 0..len-1 ramp in one pass.
    span_of_index = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    span_starts_cum = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]]
    )
    within = np.arange(total, dtype=np.int64) - span_starts_cum[span_of_index]
    return starts.astype(np.int64)[span_of_index] + within


def radix_structure(state: Any) -> RadixStructure:
    """Extract the rebuild-invariant structure the intensive rates compare.

    Parameters
    ----------
    state : Any
        An ``FMMPreparedState`` from the radix backend.

    Returns
    -------
    RadixStructure
        Host-side per-particle and per-pair structure. O(P) work and O(P)
        memory, with no Python loop over particles or pairs.
    """
    tree = state.tree
    topo = tree.topology
    near = state.neighbor_list
    far = state.interactions

    particle_indices = _host(topo.particle_indices).astype(np.int64)
    num_particles = int(particle_indices.size)
    node_ranges = _host(topo.node_ranges).astype(np.int64)

    # slot_of_particle inverts the ordering: particle_indices maps slot ->
    # original id, and the rate wanted here is per particle.
    slot_of_particle = np.empty(num_particles, dtype=np.int64)
    slot_of_particle[particle_indices] = np.arange(num_particles, dtype=np.int64)

    node_keys = _node_content_keys(particle_indices, node_ranges)

    # The near-field CSR lists the leaf node ids explicitly; that is the
    # authority on which nodes are leaves, rather than an id-range convention.
    leaf_nodes = _host(near.leaf_indices).astype(np.int64)
    leaf_starts = node_ranges[leaf_nodes, 0]
    leaf_stops = node_ranges[leaf_nodes, 1] + 1
    leaf_sizes = np.maximum(leaf_stops - leaf_starts, 0)

    # Map each particle to its leaf by scattering leaf ranks over the sorted
    # slots each leaf spans, then reading through the permutation. The spans
    # are contiguous and disjoint, so a repeat/arange scatter is exact.
    leaf_of_slot = np.full(num_particles, -1, dtype=np.int64)
    if leaf_nodes.size:
        leaf_rank = np.repeat(np.arange(leaf_nodes.size, dtype=np.int64), leaf_sizes)
        slot_positions = _spans_to_indices(leaf_starts, leaf_stops)
        leaf_of_slot[slot_positions] = leaf_rank
    leaf_of_particle = leaf_of_slot[slot_of_particle]

    leaf_keys = node_keys[leaf_nodes]
    # A particle outside every listed leaf (padding) gets key 0, which compares
    # equal to itself across rebuilds and so contributes no spurious churn.
    leaf_key_of_particle = np.where(
        leaf_of_particle >= 0,
        leaf_keys[np.maximum(leaf_of_particle, 0)] if leaf_nodes.size else np.uint64(0),
        np.uint64(0),
    ).astype(np.uint64)

    # Direct-summation reach: every source in a neighbouring leaf, plus the
    # particle's own leaf minus itself.
    offsets = _host(near.offsets).astype(np.int64)
    neighbors = _host(near.neighbors).astype(np.int64)
    counts = _host(near.counts).astype(np.int64)
    size_of_node = np.zeros(int(node_ranges.shape[0]), dtype=np.int64)
    size_of_node[leaf_nodes] = leaf_sizes
    if neighbors.size:
        per_leaf_neighbour_total = np.add.reduceat(
            size_of_node[neighbors], offsets[:-1]
        )
        per_leaf_neighbour_total = np.where(counts > 0, per_leaf_neighbour_total, 0)
    else:
        per_leaf_neighbour_total = np.zeros(leaf_nodes.size, dtype=np.int64)
    per_leaf_reach = per_leaf_neighbour_total + leaf_sizes - 1
    near_source_count = np.where(
        leaf_of_particle >= 0,
        per_leaf_reach[np.maximum(leaf_of_particle, 0)] if leaf_nodes.size else 0,
        0,
    ).astype(np.int64)

    # Canonicalised pair sets. The near CSR is over leaves; the far list is
    # over nodes at any level, which the prefix-sum keys already cover.
    near_left = np.repeat(leaf_nodes, counts) if leaf_nodes.size else leaf_nodes
    near_pair_keys = _canonical_pair_keys(node_keys[near_left], node_keys[neighbors])
    if far is None:
        far_pair_keys = np.zeros((0,), dtype=np.uint64)
    else:
        far_pair_keys = _canonical_pair_keys(
            node_keys[_host(far.sources).astype(np.int64)],
            node_keys[_host(far.targets).astype(np.int64)],
        )

    return RadixStructure(
        num_particles=num_particles,
        slot_of_particle=slot_of_particle,
        leaf_key_of_particle=leaf_key_of_particle,
        near_source_count=near_source_count,
        near_pair_keys=near_pair_keys,
        far_pair_keys=far_pair_keys,
    )


@dataclass(frozen=True)
class ChurnRates:
    """Intensive rates between two consecutive rebuilds.

    Every field is a fraction in ``[0, 1]``, so it is comparable across ``N``.
    That is the whole point: the extensive "did anything change?" answer is
    driven to 1 by ``N`` alone (D-016).

    Attributes
    ----------
    slot_churn : float
        Fraction of particles whose Morton slot moved.
    mean_rank_shift_fraction : float
        Mean ``|delta slot|`` divided by ``P``. A pervasive but *local* churn
        shows a high ``slot_churn`` and a falling value here.
    leaf_churn : float
        Fraction of particles whose leaf changed content.
    near_source_count_churn : float
        Fraction of particles whose direct-summation source count changed.
    near_pair_churn : float
        Jaccard distance between the near leaf-pair sets: ``|symmetric
        difference| / |union|``.
    far_pair_churn : Optional[float]
        The same for the far node-pair set, or ``None`` when both rebuilds had
        an empty far list -- which is what a small ``N`` at a tight ``theta``
        gives, and is not the same statement as "no far pair changed".
    """

    slot_churn: float
    mean_rank_shift_fraction: float
    leaf_churn: float
    near_source_count_churn: float
    near_pair_churn: float
    far_pair_churn: Optional[float]

    def as_record(self: "ChurnRates") -> Dict[str, Any]:
        """Return a JSON-safe dict of the rates.

        Returns
        -------
        Dict[str, Any]
            Field name to value.
        """
        return {
            "slot_churn": self.slot_churn,
            "mean_rank_shift_fraction": self.mean_rank_shift_fraction,
            "leaf_churn": self.leaf_churn,
            "near_source_count_churn": self.near_source_count_churn,
            "near_pair_churn": self.near_pair_churn,
            "far_pair_churn": self.far_pair_churn,
        }


def _jaccard_distance(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    """Symmetric-difference fraction of two sorted unique key sets.

    Parameters
    ----------
    left : np.ndarray
        Sorted unique uint64 keys.
    right : np.ndarray
        Sorted unique uint64 keys.

    Returns
    -------
    Optional[float]
        ``|symmetric difference| / |union|``, or ``None`` when both sets are
        empty -- there is no rate to report, which is a different statement
        from a rate of zero.
    """
    if left.size == 0 and right.size == 0:
        return None
    union = int(np.union1d(left, right).size)
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    return 0.0 if union == 0 else (union - intersection) / union


def churn_between(previous: RadixStructure, current: RadixStructure) -> ChurnRates:
    """Compute the intensive rates between two rebuilds.

    Parameters
    ----------
    previous : RadixStructure
        Structure from the earlier rebuild.
    current : RadixStructure
        Structure from the later rebuild.

    Returns
    -------
    ChurnRates
        The intensive rates.

    Raises
    ------
    ValueError
        If the two structures hold different particle counts, or none at all.
        Comparing them would silently produce a meaningless rate.
    """
    if previous.num_particles != current.num_particles:
        raise ValueError(
            "cannot compare structures with different particle counts "
            f"({previous.num_particles} vs {current.num_particles}); a churn "
            "rate across a changing N is not a rate"
        )
    p = float(current.num_particles)
    if p == 0.0:
        raise ValueError("cannot compute churn over an empty particle set")
    shift = np.abs(
        current.slot_of_particle.astype(np.float64)
        - previous.slot_of_particle.astype(np.float64)
    )
    near_churn = _jaccard_distance(previous.near_pair_keys, current.near_pair_keys)
    return ChurnRates(
        slot_churn=float(
            np.mean(previous.slot_of_particle != current.slot_of_particle)
        ),
        mean_rank_shift_fraction=float(np.mean(shift) / p),
        leaf_churn=float(
            np.mean(previous.leaf_key_of_particle != current.leaf_key_of_particle)
        ),
        near_source_count_churn=float(
            np.mean(previous.near_source_count != current.near_source_count)
        ),
        near_pair_churn=0.0 if near_churn is None else near_churn,
        far_pair_churn=_jaccard_distance(previous.far_pair_keys, current.far_pair_keys),
    )


class SwitchLog:
    """Per-iteration switch log: the core counter plus intensive rates.

    Feed it the prepared state from every rebuild. Extensive counting is
    delegated to :class:`~jaccpot.mutual.identity.TopologySwitchCounter`; the
    intensive rates this class adds are what fig19 plots, for the reason the
    module docstring gives.

    Parameters
    ----------
    intensive : bool
        Whether to compute the intensive rates at all. Turning them off leaves
        the extensive counting and saves the O(P) host work and memory.
    intensive_every : int
        Compute the intensive rates every ``k``-th observation. The default of
        1 measures every rebuild; a larger value samples the path, trading
        resolution for host work at large ``N``. Recorded in the summary, so a
        figure cannot mistake a sampled rate for a complete one.

    Raises
    ------
    ValueError
        If ``intensive_every`` is below 1.
    """

    def __init__(
        self: "SwitchLog", *, intensive: bool = True, intensive_every: int = 1
    ) -> None:
        if int(intensive_every) < 1:
            raise ValueError(f"intensive_every must be >= 1, got {intensive_every!r}")
        self._counter = TopologySwitchCounter()
        self._intensive = bool(intensive)
        self._intensive_every = int(intensive_every)
        self._previous_structure: Optional[RadixStructure] = None
        self._previous_index: Optional[int] = None
        self._events: List[Dict[str, Any]] = []
        self._churn: List[Dict[str, Any]] = []

    def observe(
        self: "SwitchLog", state: Any, *, iteration: Optional[int] = None
    ) -> Tuple[str, ...]:
        """Record a rebuilt state and return which components changed.

        Parameters
        ----------
        state : Any
            Prepared state from this iteration's rebuild.
        iteration : Optional[int]
            Iteration index to record; defaults to the observation count.

        Returns
        -------
        Tuple[str, ...]
            Differing component names against the previous observation; empty
            for no switch, and empty for the first observation.
        """
        index = self._counter.observations if iteration is None else int(iteration)
        observation = self._counter.observations
        diff: Optional[TopologyDiff] = self._counter.observe(
            fingerprint_prepared_state(state)
        )
        changed: Tuple[str, ...] = () if diff is None else diff.changed
        if changed:
            self._events.append(
                {
                    "iteration": index,
                    "changed": list(changed),
                    "interaction_lists_changed": (
                        diff is not None and diff.interaction_lists_changed
                    ),
                }
            )
        if self._intensive and observation % self._intensive_every == 0:
            structure = radix_structure(state)
            if self._previous_structure is not None:
                record = churn_between(self._previous_structure, structure).as_record()
                record["iteration"] = index
                record["previous_iteration"] = self._previous_index
                self._churn.append(record)
            self._previous_structure = structure
            self._previous_index = index
        return changed

    @property
    def observations(self: "SwitchLog") -> int:
        """Return the number of states observed.

        Returns
        -------
        int
            Count including the first.
        """
        return self._counter.observations

    @property
    def switches(self: "SwitchLog") -> int:
        """Return the extensive switch count.

        Returns
        -------
        int
            Comparisons in which anything at all changed. Saturates at
            ``comparisons`` once ``N`` is large; see the module docstring.
        """
        return self._counter.switches

    @property
    def interaction_switches(self: "SwitchLog") -> int:
        """Return the comparisons in which a pair list changed at all.

        Returns
        -------
        int
            Always ``<= switches``. Still extensive -- the intensive answer to
            the same question is ``near_pair_churn``.
        """
        return self._counter.interaction_switches

    @property
    def switch_rate(self: "SwitchLog") -> float:
        """Return the extensive switch rate.

        Returns
        -------
        float
            ``switches / comparisons``. Reported for continuity with the core
            facility; **do not plot this against N** -- it is precisely the
            metric D-016 identified as an artefact of an extensive measure.
        """
        return self._counter.switch_rate

    def churn(self: "SwitchLog") -> List[Dict[str, Any]]:
        """Return the per-comparison intensive rates.

        Returns
        -------
        List[Dict[str, Any]]
            One record per measured comparison, each carrying the six rates
            plus the iteration indices it spans.
        """
        return list(self._churn)

    def mean_churn(self: "SwitchLog") -> Dict[str, Optional[float]]:
        """Return the path-averaged intensive rates.

        Returns
        -------
        Dict[str, Optional[float]]
            Mean of each rate over the measured comparisons; ``None`` for a
            rate that was never available (an empty far list throughout, or
            nothing measured at all).
        """
        names = (
            "slot_churn",
            "mean_rank_shift_fraction",
            "leaf_churn",
            "near_source_count_churn",
            "near_pair_churn",
            "far_pair_churn",
        )
        out: Dict[str, Optional[float]] = {}
        for name in names:
            values = [r[name] for r in self._churn if r[name] is not None]
            out[name] = float(np.mean(values)) if values else None
        return out

    def summary(self: "SwitchLog") -> Dict[str, Any]:
        """Return a JSON-safe summary for a results file.

        Returns
        -------
        Dict[str, Any]
            The core counter's extensive summary under ``extensive``, the
            intensive rates under ``intensive``, and the event list. No arrays.
        """
        extensive = dict(self._counter.summary())
        extensive["events"] = list(self._events)
        return {
            "switch_metric": "radix_topology_excluding_morton_codes",
            "content_key": "additive_multiset_splitmix64",
            # Named blocks so a figure cannot read the extensive count as a
            # rate that survives scaling. D-016 is why this distinction is in
            # the artifact and not only in a docstring.
            "extensive": extensive,
            "intensive": {
                "measured": self._intensive,
                "every": self._intensive_every,
                "comparisons": len(self._churn),
                "mean": self.mean_churn(),
                "per_comparison": self.churn(),
            },
        }

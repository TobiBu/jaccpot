"""Identity of a frozen mutual topology: did the discrete structure change?

:class:`~jaccpot.mutual.topology.MutualTopology` is the discrete structure that
:meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations` freezes and
severs from the gradient. Gradients are exact *at fixed topology*, so any claim
about optimisation surviving per-iteration rebuilds rests on knowing how often
that structure actually changed along the path -- not on assuming it did or did
not.

Before this module there was no way to ask. Four facilities came close and none
answered it:

* :func:`jaccpot.runtime._interaction_cache.pair_policy_cache_identity` hashes
  the *acceptance criterion* so a cached interaction list is not served to a
  request that would accept different pairs. It answers "may I reuse this?",
  which is a question about inputs, not about the structure that came out.
* :meth:`~jaccpot.mutual.topology.MutualTopology.summary` returns size counters
  and says outright that they are safe to compare across runs. They are -- but
  two genuinely different topologies with the same pair counts compare equal,
  and equal counts is exactly the coincidence a switch diagnostic must not
  mistake for no switch.
* ``odisseo.jaccpot_coupling._prepared_state_shape_signature`` compares the
  dtype/shape multiset of a prepared state. Necessary for deciding whether a
  compiled program can be reused, blind to which pairs are in the list.
* ``odisseo.differentiable.topology_drift`` reports particle displacement in
  units of mean leaf extent. A *proxy*, and a good one for deciding whether a
  frozen window is still trustworthy -- but a displacement of half a cell may
  change no pair at all, or many.

So this module compares the structure itself, and reports **which part** of it
moved. That distinction is the point: a permutation that changed while every
interaction pair stayed identical is a different event from the near list
gaining pairs, and a diagnostic that collapses both to "switched" cannot tell
you which regime an optimisation is in.

Cost
----
:func:`fingerprint_topology` hashes every array, so it is ``O(total bytes)`` --
linear in ``N`` plus the two pair lists. Measured on this project's CPU box,
uniform cube, ``theta=0.5``, order 4, leaf 16, against a **warm** rebuild (the
first build is compilation-dominated and would flatter the comparison):

======  ==========  ===========  ==========  ========
     N  warm build  fingerprint   identical  fp/build
======  ==========  ===========  ==========  ========
  1024      1060ms       0.13ms      0.03ms     0.01%
  4096      1272ms       0.64ms      0.05ms     0.05%
 16384      2515ms       4.36ms      0.17ms     0.17%
======  ==========  ===========  ==========  ========

So one fingerprint per rebuild is free in practice -- two parts in a thousand
of the build it describes, at the largest size measured. It is not free in the
asymptotic sense, and a caller fingerprinting inside a tight loop over an
unchanged topology should hold the result instead.

:func:`topologies_identical` short-circuits on the first difference and
allocates nothing, which is cheaper again by an order of magnitude. Prefer it
when both topologies are in hand; :func:`fingerprint_topology` is for when one
must be *retained* to compare against something built later.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from jaccpot.mutual.topology import MutualTopology

__all__ = [
    "TopologyDiff",
    "TopologyFingerprint",
    "TopologySwitchCounter",
    "diff_topologies",
    "fingerprint_topology",
    "topologies_identical",
]

#: Digest length in bytes. 16 gives a 32-character hex digest and a collision
#: probability far below anything an optimisation path can reach: at 1e6
#: distinct topologies the birthday bound is ~1e-27. Not a security parameter --
#: nothing here defends against a chosen-input attack.
_DIGEST_BYTES = 16

#: The scalar fields that change compiled kernel shapes or the acceptance
#: criterion. Ordered for stable reporting.
_SCALAR_FIELDS: Tuple[str, ...] = (
    "num_particles",
    "num_nodes",
    "num_internal",
    "max_leaf_size",
    "order",
    "theta",
)

#: The array groups that make up the structure, each digested separately so a
#: diff can name what moved. Ordered from "cheapest to change" to "most
#: consequential" for reporting purposes only.
_COMPONENT_FIELDS: Tuple[str, ...] = (
    "permutation",
    "tree_shape",
    "leaf_partition",
    "node_ranges",
    "far_pairs",
    "near_pairs",
)


def _feed_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    """Mix one array's dtype, shape and bytes into ``hasher``.

    Parameters
    ----------
    hasher : hashlib._Hash
        Hasher to update in place.
    array : np.ndarray
        Array to digest. Made C-contiguous first, so a transposed view and its
        copy digest identically.

    Raises
    ------
    TypeError
        If ``array`` is not a NumPy array. A topology is documented as
        host-side NumPy; a JAX tracer or device array here means the caller is
        inside a trace, where this facility cannot answer.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(
            "topology identity is a host-side facility and needs NumPy arrays; "
            f"got {type(array).__name__}. If this is a tracer, you are inside a "
            "jax transformation -- fingerprint the topology before tracing."
        )
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(repr(contiguous.shape).encode("utf-8"))
    hasher.update(contiguous.tobytes())


def _digest(*arrays: Union[np.ndarray, Sequence[np.ndarray]]) -> str:
    """Digest one or more arrays, or tuples of arrays, into a hex string.

    Parameters
    ----------
    *arrays : Union[np.ndarray, Sequence[np.ndarray]]
        Arrays to digest, in order. A sequence is digested as its length
        followed by each element, so ``(a, b)`` and ``(a, b, c)`` cannot
        collide by concatenation.

    Returns
    -------
    str
        Hex digest of length ``2 * _DIGEST_BYTES``.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for item in arrays:
        if isinstance(item, np.ndarray):
            _feed_array(hasher, item)
        else:
            sequence = tuple(item)
            hasher.update(f"seq:{len(sequence)}".encode("utf-8"))
            for element in sequence:
                _feed_array(hasher, element)
    return hasher.hexdigest()


@dataclass(frozen=True)
class TopologyFingerprint:
    """A small, comparable stand-in for one :class:`MutualTopology`.

    Holds no arrays, so a long optimisation path can retain one per iteration
    without pinning the topologies themselves. Equality is exact: two
    fingerprints compare equal if and only if every scalar and every component
    digest matches.

    Attributes
    ----------
    num_particles : int
        Particle count the topology was built for.
    num_nodes : int
        Total tree nodes.
    num_internal : int
        Node id at which leaves start.
    max_leaf_size : int
        Padded leaf block width ``S``. Part of identity because it sets kernel
        shapes: a change here forces recompilation even if every pair is the
        same.
    order : int
        Multipole expansion order.
    theta : float
        Multipole acceptance parameter.
    permutation : str
        Digest of the Morton forward and inverse permutations.
    tree_shape : str
        Digest of the child arrays and the per-level node and parent groupings.
    leaf_partition : str
        Digest of the leaf ids, padded particle blocks and validity mask -- that
        is, which particle sits in which leaf.
    node_ranges : str
        Digest of the per-node inclusive particle spans.
    far_pairs : str
        Digest of the canonical well-separated node pair list.
    near_pairs : str
        Digest of the canonical near leaf pair list.
    """

    num_particles: int
    num_nodes: int
    num_internal: int
    max_leaf_size: int
    order: int
    theta: float

    permutation: str
    tree_shape: str
    leaf_partition: str
    node_ranges: str
    far_pairs: str
    near_pairs: str

    @property
    def digest(self: "TopologyFingerprint") -> str:
        """Return a single digest over every field.

        Returns
        -------
        str
            Hex digest of length ``2 * _DIGEST_BYTES``, suitable as a dict key
            or a label in a results JSON.
        """
        hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
        for name in _SCALAR_FIELDS:
            value = getattr(self, name)
            # float.hex() is exact; repr() of a float is not guaranteed to be
            # across interpreters, and theta must not compare equal by rounding.
            token = value.hex() if isinstance(value, float) else repr(value)
            hasher.update(f"{name}={token}".encode("utf-8"))
        for name in _COMPONENT_FIELDS:
            hasher.update(f"{name}={getattr(self, name)}".encode("utf-8"))
        return hasher.hexdigest()


def fingerprint_topology(topology: MutualTopology) -> TopologyFingerprint:
    """Digest a topology into a comparable fingerprint.

    Parameters
    ----------
    topology : MutualTopology
        The frozen topology to digest. Its arrays must be host-side NumPy,
        which is what :func:`~jaccpot.mutual.topology.build_mutual_topology`
        produces.

    Returns
    -------
    TopologyFingerprint
        Fingerprint holding scalars and six component digests, no arrays.

    Notes
    -----
    Propagates :class:`TypeError` from :func:`_feed_array` if any array leaf is
    not a NumPy array, which is what a JAX tracer or device array here means.

    Linear in the total byte size of the topology's arrays -- 0.13ms at
    N=1024 rising to 4.36ms at N=16384, against warm rebuilds of 1060ms and
    2515ms respectively. The module docstring carries the full table. No
    timing is asserted in the test suite: a wall-clock gate on a shared runner
    would be flaky, and the margin here is three orders of magnitude, so a
    regression would have to be enormous to matter.
    """
    return TopologyFingerprint(
        num_particles=int(topology.num_particles),
        num_nodes=int(topology.num_nodes),
        num_internal=int(topology.num_internal),
        max_leaf_size=int(topology.max_leaf_size),
        order=int(topology.order),
        theta=float(topology.theta),
        permutation=_digest(topology.forward_permutation, topology.inverse_permutation),
        tree_shape=_digest(
            topology.left_child,
            topology.right_child,
            topology.level_nodes,
            topology.parent_of_level_nodes,
        ),
        leaf_partition=_digest(
            topology.leaf_nodes,
            topology.leaf_particles,
            topology.leaf_particle_valid,
        ),
        node_ranges=_digest(topology.node_particle_ranges),
        far_pairs=_digest(topology.far_a, topology.far_b),
        near_pairs=_digest(topology.near_a, topology.near_b),
    )


def _as_fingerprint(
    value: Union[MutualTopology, TopologyFingerprint],
) -> TopologyFingerprint:
    """Coerce a topology or an existing fingerprint to a fingerprint.

    Parameters
    ----------
    value : Union[MutualTopology, TopologyFingerprint]
        Either form is accepted so callers holding retained fingerprints do not
        pay to re-digest.

    Returns
    -------
    TopologyFingerprint
        ``value`` itself if it is already a fingerprint, else its digest.
    """
    if isinstance(value, TopologyFingerprint):
        return value
    return fingerprint_topology(value)


@dataclass(frozen=True)
class TopologyDiff:
    """What changed between two topologies.

    Attributes
    ----------
    identical : bool
        True when nothing changed.
    changed : Tuple[str, ...]
        Names of the differing fields, scalars first then components, each in
        the module's declared order. Empty when ``identical``.
    left : TopologyFingerprint
        Fingerprint of the first argument.
    right : TopologyFingerprint
        Fingerprint of the second.
    """

    identical: bool
    changed: Tuple[str, ...]
    left: TopologyFingerprint
    right: TopologyFingerprint

    @property
    def interaction_lists_changed(self: "TopologyDiff") -> bool:
        """Whether either pair list moved.

        Returns
        -------
        bool
            True if ``far_pairs`` or ``near_pairs`` differ. This is the
            narrower question a switch diagnostic usually wants: the leaf
            partition and the permutation can move while the set of interacting
            pairs is unchanged.
        """
        return "far_pairs" in self.changed or "near_pairs" in self.changed

    def describe(self: "TopologyDiff") -> str:
        """Return a one-line human-readable summary.

        Returns
        -------
        str
            ``"identical"``, or the changed field names with old and new values
            for scalars.
        """
        if self.identical:
            return "identical"
        parts = []
        for name in self.changed:
            if name in _SCALAR_FIELDS:
                parts.append(
                    f"{name} {getattr(self.left, name)!r}->"
                    f"{getattr(self.right, name)!r}"
                )
            else:
                parts.append(name)
        return "changed: " + ", ".join(parts)


def diff_topologies(
    left: Union[MutualTopology, TopologyFingerprint],
    right: Union[MutualTopology, TopologyFingerprint],
) -> TopologyDiff:
    """Compare two topologies and report which parts differ.

    Parameters
    ----------
    left : Union[MutualTopology, TopologyFingerprint]
        First topology, or a fingerprint retained from an earlier build.
    right : Union[MutualTopology, TopologyFingerprint]
        Second topology, or its fingerprint.

    Returns
    -------
    TopologyDiff
        The comparison, naming every differing field.

    Notes
    -----
    Propagates :class:`TypeError` from :func:`fingerprint_topology` if a
    topology argument holds non-NumPy arrays. Passing fingerprints instead of
    topologies cannot raise.
    """
    left_fp = _as_fingerprint(left)
    right_fp = _as_fingerprint(right)
    changed = tuple(
        name
        for name in (*_SCALAR_FIELDS, *_COMPONENT_FIELDS)
        if getattr(left_fp, name) != getattr(right_fp, name)
    )
    return TopologyDiff(
        identical=not changed,
        changed=changed,
        left=left_fp,
        right=right_fp,
    )


def topologies_identical(left: MutualTopology, right: MutualTopology) -> bool:
    """Return whether two topologies are the same discrete structure.

    Short-circuits on the first difference and hashes nothing, so this is the
    cheapest form when both topologies are in hand. Use
    :func:`fingerprint_topology` instead when a topology must be *retained* for
    later comparison, and :func:`diff_topologies` when the answer needs to say
    what moved.

    Parameters
    ----------
    left : MutualTopology
        First topology.
    right : MutualTopology
        Second topology.

    Returns
    -------
    bool
        True when every scalar and every array matches exactly.
    """
    for name in _SCALAR_FIELDS:
        if getattr(left, name) != getattr(right, name):
            return False
    array_fields = (
        "forward_permutation",
        "inverse_permutation",
        "left_child",
        "right_child",
        "leaf_nodes",
        "leaf_particles",
        "leaf_particle_valid",
        "node_particle_ranges",
        "far_a",
        "far_b",
        "near_a",
        "near_b",
    )
    for name in array_fields:
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return False
    for name in ("level_nodes", "parent_of_level_nodes"):
        left_levels = getattr(left, name)
        right_levels = getattr(right, name)
        if len(left_levels) != len(right_levels):
            return False
        for left_level, right_level in zip(left_levels, right_levels):
            if not np.array_equal(left_level, right_level):
                return False
    return True


class TopologySwitchCounter:
    """Count topology switches along a sequence of rebuilds.

    The diagnostic Paper I's static-payoff section needs: feed it the topology
    from each iteration and it reports how often the discrete structure moved,
    and which part moved. Retains one fingerprint per observation and no
    arrays.

    Examples
    --------
    >>> counter = TopologySwitchCounter()                    # doctest: +SKIP
    >>> for step in range(num_steps):                        # doctest: +SKIP
    ...     topology = rebuild(positions)                    # doctest: +SKIP
    ...     counter.observe(topology)                        # doctest: +SKIP
    >>> counter.switch_rate                                  # doctest: +SKIP
    """

    def __init__(self: "TopologySwitchCounter") -> None:
        self._previous: Optional[TopologyFingerprint] = None
        self._observations = 0
        self._switches = 0
        self._interaction_switches = 0
        self._changed_components: "Counter[str]" = Counter()
        self._history: list[TopologyFingerprint] = []

    def observe(
        self: "TopologySwitchCounter",
        topology: Union[MutualTopology, TopologyFingerprint],
    ) -> Optional[TopologyDiff]:
        """Record one topology and report how it differs from the previous one.

        Parameters
        ----------
        topology : Union[MutualTopology, TopologyFingerprint]
            The topology built at this iteration.

        Returns
        -------
        Optional[TopologyDiff]
            ``None`` for the first observation, which has nothing to compare
            against and is not counted as a switch. Otherwise the diff against
            the immediately preceding observation.

        Notes
        -----
        Propagates :class:`TypeError` from :func:`fingerprint_topology` if
        ``topology`` holds non-NumPy arrays.
        """
        fingerprint = _as_fingerprint(topology)
        self._history.append(fingerprint)
        if self._previous is None:
            self._previous = fingerprint
            self._observations = 1
            return None
        diff = diff_topologies(self._previous, fingerprint)
        self._observations += 1
        if not diff.identical:
            self._switches += 1
            self._changed_components.update(diff.changed)
            if diff.interaction_lists_changed:
                self._interaction_switches += 1
        self._previous = fingerprint
        return diff

    @property
    def observations(self: "TopologySwitchCounter") -> int:
        """Number of topologies observed.

        Returns
        -------
        int
            Count including the first, which cannot be a switch.
        """
        return self._observations

    @property
    def comparisons(self: "TopologySwitchCounter") -> int:
        """Number of consecutive pairs compared.

        Returns
        -------
        int
            ``max(observations - 1, 0)`` -- the denominator for a switch rate.
        """
        return max(self._observations - 1, 0)

    @property
    def switches(self: "TopologySwitchCounter") -> int:
        """Number of comparisons in which anything changed.

        Returns
        -------
        int
            Switch count.
        """
        return self._switches

    @property
    def interaction_switches(self: "TopologySwitchCounter") -> int:
        """Number of comparisons in which a pair list changed.

        Returns
        -------
        int
            Always ``<= switches``. The gap is the number of rebuilds that
            reordered particles or re-cut leaves without changing which pairs
            interact -- worth reporting separately, because the two have
            different consequences for a frozen-topology gradient.
        """
        return self._interaction_switches

    @property
    def switch_rate(self: "TopologySwitchCounter") -> float:
        """Fraction of comparisons that switched.

        Returns
        -------
        float
            ``switches / comparisons``, or ``0.0`` when nothing has been
            compared yet. Reported as a rate rather than a count so paths of
            different length are comparable.
        """
        if self.comparisons == 0:
            return 0.0
        return self._switches / self.comparisons

    @property
    def unique_topologies(self: "TopologySwitchCounter") -> int:
        """Number of distinct topologies seen, in any order.

        Returns
        -------
        int
            Distinct combined digests. Smaller than ``switches + 1`` when the
            path revisits a structure it had before, which a consecutive-pair
            switch count cannot show.
        """
        return len({fingerprint.digest for fingerprint in self._history})

    def changed_components(
        self: "TopologySwitchCounter",
    ) -> dict[str, int]:
        """Return how often each field changed.

        Returns
        -------
        dict[str, int]
            Field name to count, ordered most frequent first. Names are the
            module's scalar and component field names.
        """
        return dict(self._changed_components.most_common())

    def summary(self: "TopologySwitchCounter") -> dict[str, Any]:
        """Return a JSON-safe summary, for a results file or a figure caption.

        Returns
        -------
        dict[str, Any]
            Counts, the switch rate, the distinct-topology count and the
            per-field change counts. No arrays and no fingerprints, so it is
            safe to serialise directly.
        """
        return {
            "observations": self.observations,
            "comparisons": self.comparisons,
            "switches": self.switches,
            "interaction_switches": self.interaction_switches,
            "switch_rate": self.switch_rate,
            "unique_topologies": self.unique_topologies,
            "changed_components": self.changed_components(),
        }

"""Switch detection for the reconstruction's rebuilt topologies.

Built **on top of** jaccpot's core identity facility
(:mod:`jaccpot.mutual.identity`, D-009) -- this module does not reimplement
interaction-list hashing. What it adds is the adapter the reconstruction
needs: the differentiable path runs on a radix
:class:`~jaccpot.runtime.fmm_state.FMMPreparedState`, not a
:class:`~jaccpot.mutual.topology.MutualTopology`, so the six-component
fingerprint is re-expressed over the state's own arrays, and the per-iteration
log records which component moved.

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
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "COMPONENTS",
    "RadixTopologyFingerprint",
    "SwitchLog",
    "diff_fingerprints",
    "fingerprint_prepared_state",
]

_DIGEST_BYTES = 16

#: Component names, from cheapest-to-change to most consequential.
COMPONENTS: Tuple[str, ...] = (
    "permutation",
    "tree_shape",
    "leaf_partition",
    "node_ranges",
    "far_pairs",
    "near_pairs",
)


def _digest(*arrays: Any) -> str:
    """Digest arrays -- JAX or NumPy -- into a hex string.

    Parameters
    ----------
    *arrays : Any
        Arrays to digest in order. JAX device arrays are brought to host
        first; ``None`` entries contribute a fixed token so an absent optional
        list still fingerprints deterministically.

    Returns
    -------
    str
        Hex digest of length ``2 * _DIGEST_BYTES``.
    """
    hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    for item in arrays:
        if item is None:
            hasher.update(b"none")
            continue
        a = np.ascontiguousarray(np.asarray(item))
        hasher.update(str(a.dtype).encode("utf-8"))
        hasher.update(repr(a.shape).encode("utf-8"))
        hasher.update(a.tobytes())
    return hasher.hexdigest()


@dataclass(frozen=True)
class RadixTopologyFingerprint:
    """Comparable stand-in for one prepared state's frozen structure.

    Attributes
    ----------
    num_particles : int
        Combined source + tracer count.
    num_leaves : int
        Leaf count.
    leaf_size : int
        Configured leaf size.
    theta : float
        Acceptance parameter.
    permutation : str
        Digest of the inverse permutation and sorted particle indices.
    tree_shape : str
        Digest of parent, children and leaf flags.
    leaf_partition : str
        Digest of the leaf-index arrays of the near-field CSR.
    node_ranges : str
        Digest of the per-node particle spans.
    far_pairs : str
        Digest of the M2L interaction list.
    near_pairs : str
        Digest of the near-field neighbour CSR.
    """

    num_particles: int
    num_leaves: int
    leaf_size: int
    theta: float
    permutation: str
    tree_shape: str
    leaf_partition: str
    node_ranges: str
    far_pairs: str
    near_pairs: str

    @property
    def digest(self: "RadixTopologyFingerprint") -> str:
        """Return a single digest over every field.

        Returns
        -------
        str
            Hex digest; ``theta`` enters via ``float.hex()`` so it compares
            exactly.
        """
        hasher = hashlib.blake2b(digest_size=_DIGEST_BYTES)
        for name in ("num_particles", "num_leaves", "leaf_size"):
            hasher.update(f"{name}={getattr(self, name)!r}".encode("utf-8"))
        hasher.update(f"theta={float(self.theta).hex()}".encode("utf-8"))
        for name in COMPONENTS:
            hasher.update(f"{name}={getattr(self, name)}".encode("utf-8"))
        return hasher.hexdigest()


def fingerprint_prepared_state(state: Any) -> RadixTopologyFingerprint:
    """Fingerprint an ``FMMPreparedState`` from the differentiable path.

    Parameters
    ----------
    state : Any
        Output of :meth:`jaccpot.FastMultipoleMethod.prepare_state` on the
        radix backend.

    Returns
    -------
    RadixTopologyFingerprint
        Fingerprint over the frozen structure. Morton codes are deliberately
        not included; see the module docstring.
    """
    tree = state.tree
    topo = tree.topology
    near = state.neighbor_list
    far = state.interactions
    return RadixTopologyFingerprint(
        num_particles=int(tree.num_particles),
        num_leaves=int(tree.num_leaves),
        leaf_size=int(getattr(topo, "leaf_size", state.max_leaf_size)),
        theta=float(state.theta),
        permutation=_digest(tree.inverse_permutation, topo.particle_indices),
        tree_shape=_digest(
            topo.parent,
            topo.left_child,
            topo.right_child,
            topo.left_is_leaf,
            topo.right_is_leaf,
            topo.leaf_depths,
        ),
        leaf_partition=_digest(
            near.leaf_indices,
            near.particle_order_leaf_indices,
            near.particle_order_to_native_leaf,
        ),
        node_ranges=_digest(topo.node_ranges),
        far_pairs=_digest(
            None if far is None else far.sources,
            None if far is None else far.targets,
            None if far is None else far.offsets,
        ),
        near_pairs=_digest(near.offsets, near.neighbors, near.counts),
    )


def diff_fingerprints(
    left: RadixTopologyFingerprint, right: RadixTopologyFingerprint
) -> Tuple[str, ...]:
    """Return the names of the fields that differ.

    Parameters
    ----------
    left : RadixTopologyFingerprint
        First fingerprint.
    right : RadixTopologyFingerprint
        Second fingerprint.

    Returns
    -------
    Tuple[str, ...]
        Differing field names, scalars first then components in
        :data:`COMPONENTS` order. Empty when identical.
    """
    scalars = ("num_particles", "num_leaves", "leaf_size", "theta")
    return tuple(
        name
        for name in (*scalars, *COMPONENTS)
        if getattr(left, name) != getattr(right, name)
    )


class SwitchLog:
    """Per-iteration topology-switch log for a reconstruction run.

    Feed it the prepared state from every rebuild. It records, per iteration,
    whether anything moved and which components did, and summarises the run
    into the switch statistics fig19 needs. Retains fingerprints only.
    """

    def __init__(self: "SwitchLog") -> None:
        self._previous: Optional[RadixTopologyFingerprint] = None
        self._history: List[RadixTopologyFingerprint] = []
        self._events: List[Dict[str, Any]] = []
        self._changed: "Counter[str]" = Counter()

    def observe(
        self: "SwitchLog", state: Any, *, iteration: Optional[int] = None
    ) -> Tuple[str, ...]:
        """Record a rebuilt state and return what changed since the last one.

        Parameters
        ----------
        state : Any
            Prepared state from this iteration's rebuild.
        iteration : Optional[int]
            Iteration index to record; defaults to the observation count.

        Returns
        -------
        Tuple[str, ...]
            Differing field names against the previous observation; empty for
            no switch, and empty for the first observation.
        """
        fp = fingerprint_prepared_state(state)
        index = len(self._history) if iteration is None else int(iteration)
        self._history.append(fp)
        if self._previous is None:
            self._previous = fp
            return ()
        changed = diff_fingerprints(self._previous, fp)
        if changed:
            self._changed.update(changed)
            self._events.append(
                {
                    "iteration": index,
                    "changed": list(changed),
                    "interaction_lists_changed": (
                        "far_pairs" in changed or "near_pairs" in changed
                    ),
                }
            )
        self._previous = fp
        return changed

    @property
    def observations(self: "SwitchLog") -> int:
        """Return the number of states observed.

        Returns
        -------
        int
            Count including the first.
        """
        return len(self._history)

    @property
    def switches(self: "SwitchLog") -> int:
        """Return the number of consecutive comparisons that changed.

        Returns
        -------
        int
            Switch count.
        """
        return len(self._events)

    @property
    def interaction_switches(self: "SwitchLog") -> int:
        """Return the switches in which a pair list changed.

        Returns
        -------
        int
            Always ``<= switches``; the gap is reorderings and re-cuts that
            left the interacting pairs unchanged.
        """
        return sum(1 for e in self._events if e["interaction_lists_changed"])

    @property
    def switch_rate(self: "SwitchLog") -> float:
        """Return switches per comparison.

        Returns
        -------
        float
            ``switches / (observations - 1)``, or ``0.0`` before any
            comparison.
        """
        comparisons = max(self.observations - 1, 0)
        return 0.0 if comparisons == 0 else self.switches / comparisons

    def summary(self: "SwitchLog") -> Dict[str, Any]:
        """Return a JSON-safe summary for a results file.

        Returns
        -------
        Dict[str, Any]
            Counts, rates, unique topologies, per-component change counts and
            the event list. No arrays.
        """
        return {
            "switch_metric": "radix_topology_excluding_morton_codes",
            "observations": self.observations,
            "comparisons": max(self.observations - 1, 0),
            "switches": self.switches,
            "interaction_switches": self.interaction_switches,
            "switch_rate": self.switch_rate,
            "unique_topologies": len({fp.digest for fp in self._history}),
            "changed_components": dict(self._changed.most_common()),
            "events": list(self._events),
        }

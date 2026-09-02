"""Momentum-conserving (mutual) FMM on the Dehnen real spherical-harmonic basis.

This subpackage implements the Dehnen (2014) *mutual* restructure of the FMM so
that total linear momentum is conserved to floating-point round-off rather than
to the multipole truncation error. It is the force engine behind
:class:`jaccpot.nornax_adapter.BlockStepFMM`, which exposes it to an
individual-timestep block-step KDK leapfrog.

Why a separate evaluation path
------------------------------
Jaccpot's production runtime is *target-centric*: the near field is a gather
(each target sums over its neighbours) and the far field is a one-directional
downward sweep. Each pair is therefore evaluated twice, independently, and the
two impulses differ in the last bits -- so ``sum_i m_i a_i`` is only zero to the
FMM's force accuracy (~1e-3 .. 1e-5), not to round-off. Momentum conservation is
the *defining* correctness property of a block-step integrator, so the mutual
path computes every interaction **once** and applies ``+f``/``-f`` to both
endpoints.

The three mechanisms
--------------------
``topology``
    A symmetric dual-tree traversal (host-side, frozen) that visits each
    well-separated node pair and each near leaf pair **exactly once**, in a
    canonical ``a < b`` order. Symmetry of the near/far partition is a
    precondition for mutual accumulation and is what the canonical list buys.
``nearfield``
    A leaf-pair P2P block kernel: one ``dr``/``1/r^3`` evaluation per pair,
    ``+F`` scattered to the target leaf and ``-F`` to the source leaf. Because
    IEEE guarantees ``fl(x_j - x_i) == -fl(x_i - x_j)`` and the scalar prefactor
    is symmetric to the last bit, the two impulses are exact bit-negatives.
``farfield``
    A **dual M2L**: for a well-separated pair ``(A, B)`` the same evaluation adds
    B's field to A's local expansion and A's to B's. Both forces are gradients of
    the *same* truncated mutual interaction energy, so ``F_A + F_B`` cancels
    algebraically -- measured at ~4e-16 relative, independent of ``theta`` and of
    the expansion order (see ``tests/integration/test_mutual_fmm.py``).

Forces, not accelerations, are accumulated; the mass division happens once at the
end. That keeps ``sum_i m_i a_i == sum_i F_i``, which is the quantity that
cancels structurally.

Rung awareness
--------------
``force`` adds the block-step vocabulary: every interaction is assigned a level
and can be gated on it, so a single traversal can serve a whole sub-step
boundary. See :mod:`jaccpot.mutual.force` for the exact/approximate split
semantics.

Everything here is pure JAX at fixed topology, in the style of
:meth:`jaccpot.FastMultipoleMethod.differentiable_accelerations`: the discrete
structure is a compile-time constant and the numeric pipeline is re-evaluated on
live inputs, so ``jax.grad`` over the force is exact.
"""

from __future__ import annotations

from jaccpot.mutual.farfield import mutual_far_field_forces
from jaccpot.mutual.force import (
    MutualFMMState,
    MutualForceResult,
    active_level_floor,
    boundary_level_weights,
    boundary_weight_table,
    build_mutual_state,
    is_sync_boundary,
    level_weights_from_floor,
    mutual_accelerations,
    mutual_level_accelerations,
    mutual_weighted_accelerations,
    n_sub,
)
from jaccpot.mutual.identity import (
    TopologyDiff,
    TopologyFingerprint,
    TopologySwitchCounter,
    diff_topologies,
    fingerprint_topology,
    topologies_identical,
)
from jaccpot.mutual.nearfield import mutual_near_field_forces
from jaccpot.mutual.topology import (
    MutualTopology,
    build_mutual_topology,
    build_mutual_topology_from_tree,
)

__all__ = [
    "MutualTopology",
    "build_mutual_topology",
    "build_mutual_topology_from_tree",
    "TopologyDiff",
    "TopologyFingerprint",
    "TopologySwitchCounter",
    "diff_topologies",
    "fingerprint_topology",
    "topologies_identical",
    "MutualFMMState",
    "build_mutual_state",
    "mutual_near_field_forces",
    "mutual_far_field_forces",
    "mutual_accelerations",
    "mutual_level_accelerations",
    "mutual_weighted_accelerations",
    "boundary_level_weights",
    "boundary_weight_table",
    "level_weights_from_floor",
    "active_level_floor",
    "is_sync_boundary",
    "n_sub",
    "MutualForceResult",
]

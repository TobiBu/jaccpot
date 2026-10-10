"""Multi-GPU distributed FMM force evaluation.

Promotes the per-device ``jax.shard_map`` assembly that was first validated in a
hand-rolled complex-basis test (4-GPU far field, within ~0.24-1% of a direct N-body
sum; removed with the complex basis in the 2026-10 cleanup, C) into a reusable,
benchmarkable API.

The heavy lifting (SFC domain decomposition, per-device local tree, real
upward sweep, locally-essential-tree halo import, self + cross M2L, L2L cascade,
L2P, and the combined near-field P2P over ``[local ; halo]``) is unchanged from
the validated test; only the tunables have been promoted to
:class:`DistributedFMMConfig` and the host-side scaffolding (Morton pre-split,
padding, global-id tracking, reassembly) factored into helpers.

The tree/LET building blocks live in :mod:`yggdrax.distributed`; the force
operators are jaccpot's real spherical-harmonic path, the single-GPU fast lane's.
"""

from __future__ import annotations

from .fmm import (
    DistributedFMMConfig,
    DistributedFMMResult,
    distributed_fmm_accelerations,
    make_force_evaluator,
    partition_for_devices,
    scatter_to_input_order,
)

__all__ = [
    "DistributedFMMConfig",
    "DistributedFMMResult",
    "distributed_fmm_accelerations",
    "make_force_evaluator",
    "partition_for_devices",
    "scatter_to_input_order",
]

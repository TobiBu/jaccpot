"""Multi-GPU distributed FMM force evaluation.

Promotes the per-device ``jax.shard_map`` assembly that was validated in
``tests/test_distributed_solidfmm_far_shardmap.py`` (4-GPU solidfmm far-field,
within ~0.24-1% of a direct N-body sum) into a reusable, benchmarkable API.

The heavy lifting (SFC domain decomposition, per-device local tree, solidfmm
upward sweep, locally-essential-tree halo import, self + cross M2L, L2L cascade,
L2P, and the combined near-field P2P over ``[local ; halo]``) is unchanged from
the validated test; only the tunables have been promoted to
:class:`DistributedFMMConfig` and the host-side scaffolding (Morton pre-split,
padding, global-id tracking, reassembly) factored into helpers.

The tree/LET building blocks live in :mod:`yggdrax.distributed`; the force
operators are jaccpot's solidfmm complex path.
"""

from __future__ import annotations

from .fmm import (
    DistributedFMMConfig,
    DistributedFMMResult,
    distributed_fmm_accelerations,
    make_force_evaluator,
    partition_for_devices,
)

__all__ = [
    "DistributedFMMConfig",
    "DistributedFMMResult",
    "distributed_fmm_accelerations",
    "make_force_evaluator",
    "partition_for_devices",
]

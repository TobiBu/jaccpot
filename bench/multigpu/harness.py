"""Shared multi-GPU benchmark harness for strong_scaling.py, weak_scaling.py,
comm_overhead.py, and load_balance.py.

This is the harness referenced but not yet present in
docs/phase5_multigpu_pallas_foldin_plan.md (item 5d: "multi-GPU perf/scaling
... via the benchmark_multigpu/ harness"). It wraps
jaccpot.distributed.fmm's shard_map driver with per-stage timers so strong
scaling, weak scaling, the comm/compute split, and load balance all come out
of the same runs.

Before relying on this for real numbers, confirm against
docs/phase5_multigpu_pallas_foldin_plan.md's STATUS block which basis/MAC
the distributed path is currently running (solidfmm+bh as of the last
update) -- see PROJECT_PLAN.md Phase 0.
"""

from __future__ import annotations

import dataclasses

import numpy as np

STAGE_NAMES = (
    "local_tree_build",
    "self_m2l_near",
    "all_gather_coarse",
    "coarse_m2m",
    "cross_walk",
    "halo_import",
    "remote_m2l",
    "p2p_combined",
)

COMM_STAGES = ("all_gather_coarse", "cross_walk", "halo_import")


@dataclasses.dataclass
class MultiGPURunResult:
    n_particles: int
    n_gpus: int
    wall_clock_total: float
    stage_times: dict[str, float]
    per_gpu_interaction_counts: np.ndarray  # for load-balance


def run_once(
    n_particles: int,
    n_gpus: int,
    distribution: str = "uniform_cube",
    seed: int = 0,
) -> MultiGPURunResult:
    """Run the distributed FMM once and collect timings.

    TODO: call jaccpot.distributed.fmm's shard_map driver directly (see
    jaccpot/distributed/fmm.py's docstring for the per-device pipeline
    stages), instrumenting each of STAGE_NAMES with a
    jax.block_until_ready()-guarded timer. per_gpu_interaction_counts should
    come from the M2L/P2P list sizes per device (needed for load_balance.py
    on a clustered distribution).
    """
    raise NotImplementedError(
        "Instrument jaccpot/distributed/fmm.py's shard_map pipeline stage by "
        "stage. This is the main new code this paper plan requires -- see "
        "PROJECT_PLAN.md Phase 2."
    )


def strong_scaling(
    n_particles: int, gpu_counts: list[int], **kw
) -> list[MultiGPURunResult]:
    return [run_once(n_particles, g, **kw) for g in gpu_counts]


def weak_scaling(
    n_per_gpu: int, gpu_counts: list[int], **kw
) -> list[MultiGPURunResult]:
    return [run_once(n_per_gpu * g, g, **kw) for g in gpu_counts]

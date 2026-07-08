"""Profile GPU utilization of the strict fused static-radix step.

Runs a device-resident ``strict_run_v2`` multi-step scan (the pattern odisseo
uses) and samples SM utilization with ``nvidia-smi dmon`` during execution.

Observed on an RTX 2080 Ti (sm_75, 200k, p=4 theta=0.6 leaf=256):
  - during execution the GPU sits at ~97-98% SM occupancy -- the fused lane is
    NOT idle-bound; it keeps the GPU busy (one-time ~29s compile per num_steps
    profile shows as a leading 0% region).
  - nsys shows the near-field executes as ~2500 tiny (~1-2us) fused kernels per
    step vs a handful of efficient GEMMs for the far-field: the near-field is
    fragmented + low arithmetic intensity (memory-bound, ~7% of FLOP peak).

The near-field is the bottleneck. A pure-JAX dense-block P2P does not beat the
current ~0.24-0.27s near-field (XLA materializes the WxW distance matrix to
HBM). The real lever is a SRAM-tiling Pallas P2P kernel, which requires
compute capability >= 8.0 (Ampere+); it cannot run on sm_75.

Usage: python bench/profile_fused_gpu_util.py [physical_gpu_index]
(set CUDA_VISIBLE_DEVICES to the same GPU).
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import time


def _set_env() -> None:
    for k, v in dict(
        JACCPOT_STATIC_STRICT_GPU_MODE="on",
        JACCPOT_STATIC_STRICT_FUSED_MODE="on",
        JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET="200000",
        JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH="0",
        JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY="1",
        JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK="1",
        JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS="1",
        JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP="131072",
        JACCPOT_LARGE_N_COMPILED_STATE_MODE="on",
        JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED="1",
        JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF="64",
        JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP="2097152",
    ).items():
        os.environ.setdefault(k, v)


def main() -> None:
    phys_gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
    n = int(os.environ.get("PROFILE_N", "200000"))
    steps = int(os.environ.get("PROFILE_STEPS", "300"))
    _set_env()
    import jax
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod

    pos = jax.random.uniform(
        jax.random.PRNGKey(0), (n, 3), minval=-1, maxval=1, dtype=jnp.float32
    )
    vel = 0.01 * jax.random.normal(jax.random.PRNGKey(2), (n, 3), dtype=jnp.float32)
    mass = jax.random.uniform(
        jax.random.PRNGKey(1), (n,), minval=0.1, maxval=1.0, dtype=jnp.float32
    )
    state0 = jnp.stack([pos, vel], axis=1)
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        complex_rotation="solidfmm",
        theta=0.6,
        nearfield_mode="bucketed",
        nearfield_edge_chunk_size=64,
        grouped_interactions=False,
        working_dtype=jnp.float32,
        tree_build_mode="static_radix",
        fixed_order=4,
    )

    def run(nsteps):
        out, _, _ = solver.strict_run_v2(
            state=state0,
            masses=mass,
            dt=2.0e-4,
            num_steps=nsteps,
            refresh_every=1,
            leaf_size=256,
            max_order=4,
            theta=0.6,
            return_history=False,
        )
        return out

    print("warming (compile)...", flush=True)
    run(2).block_until_ready()

    dmon = subprocess.Popen(
        ["nvidia-smi", "dmon", "-i", phys_gpu, "-s", "u", "-d", "1", "-c", "120"],
        stdout=subprocess.PIPE,
        text=True,
    )
    t0 = time.perf_counter()
    run(steps).block_until_ready()
    wall = time.perf_counter() - t0
    dmon.terminate()
    sm = []
    for line in dmon.stdout:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                sm.append(int(parts[1]))
            except ValueError:
                pass
    print(
        f"strict_run_v2 {steps} steps @{n}: wall={wall:.2f}s  per-step={wall/steps*1000:.1f} ms",
        flush=True,
    )
    if sm:
        busy = [x for x in sm if x > 5]
        print(
            f"GPU SM%%: mean={statistics.mean(sm):.0f} (excl compile idle: mean={statistics.mean(busy) if busy else 0:.0f}) "
            f"min={min(sm)} max={max(sm)} n={len(sm)}",
            flush=True,
        )
    diag = solver.get_runtime_diagnostics()
    print(
        f"fused_active={diag.get('strict_fused_mode_active')} "
        f"fallback={diag.get('strict_fused_fallback_count')} "
        f"compile={diag.get('strict_fused_compile_count')}",
        flush=True,
    )


if __name__ == "__main__":
    main()

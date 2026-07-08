"""Per-stage breakdown of the strict large-N refresh hot path.

Runs the non-fused strict prepare/refresh+evaluate loop with jaccpot's per-stage
refresh timers enabled (JACCPOT_REFRESH_TIMING_ENABLE=1) and dumps the
_refresh_timing_*_seconds breakdown from get_runtime_diagnostics(). This answers
"where does per-step time actually go" (tree / upward geometry|mass|p2m|m2m /
dual / downward / nearfield / eval) on the current hardware, so we optimize the
real bottleneck rather than an assumed one.

    PROFILE_N=200000 PROFILE_STEPS=20 CUDA_VISIBLE_DEVICES=<free> \
        micromamba run -n odisseo python bench/profile_refresh_stage_breakdown.py
"""

from __future__ import annotations

import json
import os
import time


def _set_env() -> None:
    env = dict(
        JACCPOT_REFRESH_TIMING_ENABLE="1",
        JACCPOT_PROFILE_UPWARD_STAGES="1",
        JACCPOT_STATIC_STRICT_GPU_MODE="on",
        JACCPOT_STATIC_STRICT_FUSED_MODE="off",  # non-fused: per-stage timers work
        JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH="0",
        JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF="64",
        JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP="2097152",
    )
    for k, v in env.items():
        os.environ.setdefault(k, v)


def main() -> None:
    n = int(os.environ.get("PROFILE_N", "200000"))
    steps = int(os.environ.get("PROFILE_STEPS", "20"))
    warmup = int(os.environ.get("PROFILE_WARMUP", "3"))
    _set_env()

    import jax
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod

    cc = getattr(jax.devices()[0], "compute_capability", "n/a")
    pos = jax.random.uniform(
        jax.random.PRNGKey(0), (n, 3), minval=-1, maxval=1, dtype=jnp.float32
    )
    vel = 0.01 * jax.random.normal(jax.random.PRNGKey(2), (n, 3), dtype=jnp.float32)
    mass = jax.random.uniform(
        jax.random.PRNGKey(1), (n,), minval=0.1, maxval=1.0, dtype=jnp.float32
    )
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

    prepared = None
    p = pos

    def step(prepared, p):
        prepared, acc = solver.strict_prepare_refresh_and_evaluate(
            prepared, p, mass, leaf_size=256, max_order=4, theta=0.6
        )
        jax.block_until_ready(acc)
        return prepared, acc

    # Warmup (compile + first full prepare).
    for _ in range(max(1, warmup)):
        prepared, _ = step(prepared, p)
        p = p + 1e-4 * vel

    # Reset the timing counters after warmup, then measure `steps` refreshes.
    # The strict one-call helper does not auto-toggle the timing flag (the
    # odisseo coupling sets it explicitly), so force it on around the loop.
    impl = solver._impl
    for attr in list(vars(impl)):
        if attr.startswith("_refresh_timing_") and attr.endswith("_seconds"):
            setattr(impl, attr, 0.0)
    impl._refresh_timing_enabled = True

    t0 = time.perf_counter()
    for _ in range(steps):
        impl._refresh_timing_active = True
        prepared, _ = step(prepared, p)
        impl._refresh_timing_active = False
        p = p + 1e-4 * vel
    wall = time.perf_counter() - t0

    stages = {
        attr[len("_refresh_timing_") : -len("_seconds")]: float(getattr(impl, attr))
        for attr in vars(impl)
        if attr.startswith("_refresh_timing_") and attr.endswith("_seconds")
    }
    per_step = {k: v / steps for k, v in stages.items() if v}
    ranked = sorted(per_step.items(), key=lambda kv: -kv[1])

    print(f"compute_capability={cc} n={n} steps={steps} fused=off")
    print(f"wall={wall:.3f}s  per-step={wall/steps*1000:.1f} ms")
    print("per-step stage seconds (nonzero, ranked):")
    for k, v in ranked:
        print(f"  {v*1000:8.2f} ms  {k}")
    out = os.environ.get("PROFILE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(
                {
                    "cc": str(cc),
                    "n": n,
                    "steps": steps,
                    "wall_seconds": wall,
                    "per_step_ms": wall / steps * 1000,
                    "per_step_stage_seconds": per_step,
                },
                f,
                indent=2,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

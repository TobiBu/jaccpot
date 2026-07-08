"""Attribute the fused strict_run_v2 per-step cost to stages by ablation.

Host-side per-stage timers are unreliable for the fused device-resident lane
(constant-folding vs eager-dispatch artifacts). Instead this runs the *real*
fused jitted scan repeatedly under the built-in cumulative diag modes and
attributes cost by difference in measured per-step wall time:

    integrator_only <= tree_only <= upward_only <= downward_only <= full

    tree     = tree_only      - integrator_only
    upward   = upward_only    - tree_only
    downward = downward_only  - upward_only
    eval+nf  = full           - downward_only

Each mode compiles its own fused runner (~30s) so this takes a few minutes.

    PROFILE_N=200000 PROFILE_STEPS=40 CUDA_VISIBLE_DEVICES=<free> \
        micromamba run -n odisseo python bench/profile_fused_stage_ablation.py
"""

from __future__ import annotations

import json
import os
import time

FUSED_ENV = dict(
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
)

MODES = ["integrator_only", "tree_only", "upward_only", "downward_only", "full"]


def main() -> None:
    n = int(os.environ.get("PROFILE_N", "200000"))
    steps = int(os.environ.get("PROFILE_STEPS", "40"))
    for k, v in FUSED_ENV.items():
        os.environ.setdefault(k, v)

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
    state0 = jnp.stack([pos, vel], axis=1)

    def measure(mode: str) -> float:
        os.environ["JACCPOT_STRICT_REFRESH_DIAG_MODE"] = mode
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

        run(2).block_until_ready()  # warmup / compile
        t0 = time.perf_counter()
        run(steps).block_until_ready()
        per_step_ms = (time.perf_counter() - t0) / steps * 1000
        fused = bool(solver.get_runtime_diagnostics().get("strict_fused_mode_active"))
        return per_step_ms, fused

    results = {}
    for mode in MODES:
        try:
            ms, fused = measure(mode)
            results[mode] = {"per_step_ms": ms, "fused_active": fused}
            print(f"{mode:16s} {ms:8.1f} ms/step  fused={fused}", flush=True)
        except Exception as exc:
            results[mode] = {"error": str(exc)[:200]}
            print(f"{mode:16s} ERROR: {str(exc)[:160]}", flush=True)

    def d(a, b):
        if (
            a in results
            and b in results
            and "per_step_ms" in results[a]
            and "per_step_ms" in results[b]
        ):
            return results[a]["per_step_ms"] - results[b]["per_step_ms"]
        return None

    attrib = {
        "integrator+overhead": results.get("integrator_only", {}).get("per_step_ms"),
        "tree": d("tree_only", "integrator_only"),
        "upward": d("upward_only", "tree_only"),
        "downward": d("downward_only", "upward_only"),
        "eval+nearfield": d("full", "downward_only"),
        "full_total": results.get("full", {}).get("per_step_ms"),
    }
    print("\n=== attributed per-step cost (ms) ===")
    for k, v in attrib.items():
        if v is not None:
            print(f"  {k:22s} {v:8.1f}")
    out = os.environ.get("PROFILE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(
                {
                    "cc": str(cc),
                    "n": n,
                    "steps": steps,
                    "modes": results,
                    "attributed_ms": attrib,
                },
                f,
                indent=2,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

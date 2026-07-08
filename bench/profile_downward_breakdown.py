"""Split the fused downward pass (515ms/step, the real bottleneck) into its
plan-build vs M2L/L2L-compute parts, measured in the real jitted scan.

The downward stage mixes topology-fixed work (dual-traversal artifact build:
leaf-neighbor lists + far-pair candidate enumeration) with genuinely
position/mass-dependent FLOPs (M2L, L2L). Only the former is cacheable across
same-topology refreshes, so we must know its share before optimizing (the M2M
lesson: do not optimize a small slice).

Uses cumulative (diag_mode, detail_diag_mode) combos and attributes by
difference in measured per-step wall time:

    upward_only/full                    -> A  (no downward)
    downward_only/downward_artifacts    -> B  (A + plan build)
    downward_only/m2l_only              -> C  (B + M2L)
    downward_only/full                  -> D  (full downward)

    plan_build = B - A ; m2l = C - B ; l2l+rest = D - C

    PROFILE_N=200000 PROFILE_STEPS=40 CUDA_VISIBLE_DEVICES=<free> \
        micromamba run -n odisseo python bench/profile_downward_breakdown.py
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

# (label, diag_mode, detail_diag_mode)
COMBOS = [
    ("A_upward", "upward_only", "full"),
    ("B_plan", "downward_only", "downward_artifacts_only"),
    ("C_m2l", "downward_only", "m2l_only"),
    ("D_full_down", "downward_only", "full"),
]


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

    def measure(diag, detail):
        os.environ["JACCPOT_STRICT_REFRESH_DIAG_MODE"] = diag
        os.environ["JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE"] = detail
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

        run(2).block_until_ready()
        t0 = time.perf_counter()
        run(steps).block_until_ready()
        return (time.perf_counter() - t0) / steps * 1000

    res = {}
    for label, diag, detail in COMBOS:
        try:
            ms = measure(diag, detail)
            res[label] = ms
            print(
                f"{label:14s} diag={diag:14s} detail={detail:24s} {ms:8.1f} ms/step",
                flush=True,
            )
        except Exception as exc:
            res[label] = None
            print(f"{label:14s} ERROR: {str(exc)[:160]}", flush=True)

    def d(a, b):
        return (
            (res[a] - res[b])
            if res.get(a) is not None and res.get(b) is not None
            else None
        )

    attrib = {
        "plan_build (cacheable?)": d("B_plan", "A_upward"),
        "m2l_compute": d("C_m2l", "B_plan"),
        "l2l+rest": d("D_full_down", "C_m2l"),
        "downward_total": d("D_full_down", "A_upward"),
    }
    print("\n=== downward sub-attribution (ms/step) ===")
    for k, v in attrib.items():
        if v is not None:
            print(f"  {k:26s} {v:8.1f}")
    out = os.environ.get("PROFILE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(
                {
                    "cc": str(cc),
                    "n": n,
                    "steps": steps,
                    "combos": res,
                    "attributed_ms": attrib,
                },
                f,
                indent=2,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

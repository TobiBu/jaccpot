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

Set ``PROFILE_REPEATS=3`` (default 1) to re-time each already-compiled mode and
report the within-mode spread. Do that before reading any attributed difference
as a code effect: an attributed stage is a difference of two measured modes, so
it is only meaningful when it exceeds the noise in those modes, and this workload
is noisy enough that it often does not. Measured on an A100, ``eval+nearfield``
had a 35% within-side spread -- four times the largest cross-tree difference
being attributed to a refactor at the time. With repeats > 1 the script names the
stages that are not readable at the measured noise level.

    PROFILE_N=200000 PROFILE_STEPS=40 PROFILE_REPEATS=3 CUDA_VISIBLE_DEVICES=<free> \
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
    repeats = max(1, int(os.environ.get("PROFILE_REPEATS", "1")))
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

    def measure(mode: str) -> tuple[list[float], bool]:
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
        # Repeats re-time the already-compiled runner, so they cost steps, not a
        # recompile. They exist because a single timing cannot be told apart from
        # the machine: the same worktree measured 15% apart in two windows hours
        # apart on the CPU box, and on an A100 the within-side spread of
        # 'eval+nearfield' reached 35% -- four times the largest cross-tree
        # difference anyone was trying to attribute to a code change.
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            run(steps).block_until_ready()
            samples.append((time.perf_counter() - t0) / steps * 1000)
        fused = bool(solver.get_runtime_diagnostics().get("strict_fused_mode_active"))
        return samples, fused

    def median(values):
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    def spread_pct(values):
        """Peak-to-peak as a percentage of the median; 0.0 for a single sample."""
        if len(values) < 2:
            return 0.0
        centre = median(values)
        return (max(values) - min(values)) / abs(centre) * 100.0 if centre else 0.0

    results = {}
    for mode in MODES:
        try:
            samples, fused = measure(mode)
            ms = median(samples)
            spread = spread_pct(samples)
            results[mode] = {
                "per_step_ms": ms,
                "samples_ms": samples,
                "spread_pct": spread,
                "fused_active": fused,
            }
            suffix = (
                f"  spread={spread:5.1f}% over {len(samples)}" if repeats > 1 else ""
            )
            print(
                f"{mode:16s} {ms:8.1f} ms/step  fused={fused}{suffix}",
                flush=True,
            )
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

    # Say when the attribution cannot support the numbers it just printed. An
    # attributed stage is a difference of two measured modes, so it is only
    # readable if it is larger than the noise in those modes.
    worst_noise_ms = max(
        (
            r["spread_pct"] / 100.0 * r["per_step_ms"]
            for r in results.values()
            if "per_step_ms" in r and "spread_pct" in r
        ),
        default=0.0,
    )
    if repeats < 2:
        print(
            "\n  NOTE: single timing per mode. Set PROFILE_REPEATS=3 before reading "
            "any difference below as an effect -- one sample cannot be told apart "
            "from machine drift."
        )
    else:
        print(f"\n  worst within-mode noise: +/-{worst_noise_ms:.1f} ms/step")
        unreadable = [
            k
            for k, v in attrib.items()
            if v is not None and k != "full_total" and abs(v) < worst_noise_ms
        ]
        if unreadable:
            print(
                "  NOT READABLE at this noise level (|attributed| < noise): "
                + ", ".join(unreadable)
            )
    if attrib.get("tree") is not None and attrib["tree"] < 0:
        print(
            "  NOTE: 'tree' is negative, so the modes are not the monotone ladder "
            "this attribution assumes; treat the split as indicative only."
        )
    out = os.environ.get("PROFILE_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(
                {
                    "cc": str(cc),
                    "n": n,
                    "steps": steps,
                    "repeats": repeats,
                    "worst_within_mode_noise_ms": worst_noise_ms,
                    "modes": results,
                    "attributed_ms": attrib,
                },
                f,
                indent=2,
            )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

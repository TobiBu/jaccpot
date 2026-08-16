"""Split the fused downward pass into its plan-build vs M2L/L2L-compute parts.

The downward stage mixes topology-fixed work (dual-traversal artifact build:
leaf-neighbor lists + far-pair candidate enumeration) with genuinely
position/mass-dependent FLOPs (M2L, L2L). Only the former is cacheable across
same-topology refreshes, so we must know its share before optimizing (the M2M
lesson: do not optimize a small slice).

    PROFILE_N=200000 PROFILE_STEPS=40 CUDA_VISIBLE_DEVICES=<free> \
        python bench/profile_downward_breakdown.py

WHY THIS NO LONGER ATTRIBUTES BY DIFFERENCE. It used to run four
``(diag_mode, detail_diag_mode)`` combos and subtract adjacent per-step wall times:

    upward_only/full -> A, downward_only/downward_artifacts_only -> B,
    downward_only/m2l_only -> C, downward_only/full -> D
    plan_build = B - A ; m2l = C - B ; l2l+rest = D - C

That is unsound, because **the detail modes are not a monotone ladder.** They are
per-sub-stage probes, each returning early with its own keep-alive dependencies, so a
mode that does strictly less work can cost more, and two modes can each contain
something the other does not. Measured on an A100 at N=200000, 20 steps:

    tree_only/full             79.9 ms/step
    upward_only/full   (A)    260.1
    downward/artifacts (B)    145.1      <- between tree_only and A, so B is missing
    upward_only/p2m_only      303.3      <- strictly less work than full, yet +17%

``plan_build = B - A`` was therefore reported as **-85.7 ms/step** -- a negative cost,
printed without complaint, which is how it went unnoticed. Only ``D - A`` survives, both
sides being ``detail=full``.

AND THERE IS NO WORKING SUBSTITUTE ON THIS LANE. The runtime does instrument these
stages directly -- ``refresh_dual_artifact_build_seconds`` is the plan build,
``refresh_dual_m2l_compute_seconds`` and ``refresh_dual_l2l_compute_seconds`` the compute
halves -- but those counters are written by ``refresh_prepared_state``, the host-side
eager refresh. The fused device-resident lane this script profiles runs one jitted scan
and never calls it, so ``refresh_timing_calls`` comes back **0** and every counter stays
0.0. ``profile_fused_stage_ablation.py`` says the same thing from the other direction:
host-side per-stage timers are unreliable for the fused lane, which is why attribution by
difference exists at all. This script still reads the counters and says plainly when they
did not record, rather than reporting zeros as measurements.

SO WHAT CAN BE TRUSTED HERE. ``downward_total = D - A`` (both ``detail=full``), and the
``diag_mode``-only ladder in ``profile_fused_stage_ablation.py``, which *is* nested by
construction. **The downward split this script was written for -- plan build versus M2L
versus L2L on the fused lane -- is not currently obtainable by any available method.**
Two routes would fix it, neither attempted here: give the detail-mode early returns
matching keep-alive dependencies so they become a real ladder, or instrument the stages
inside the fused scan so the counters record there.

The guard below rejects a *negative* difference, which is what went unnoticed. Note what
it cannot do: a positive difference is not thereby valid. ``m2l_compute = C - B`` comes
out positive and is still contaminated, because B is missing upward work that C contains
(B sits below ``upward_only`` above).
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
    # What makes the per-stage timers record at all; without it every counter stays
    # 0.0 and ``refresh_timing_calls`` is 0.
    JACCPOT_REFRESH_TIMING_ENABLE="1",
)

# (label, diag_mode, detail_diag_mode) -- the optional cross-check only.
COMBOS = [
    ("A_upward", "upward_only", "full"),
    ("B_plan", "downward_only", "downward_artifacts_only"),
    ("C_m2l", "downward_only", "m2l_only"),
    ("D_full_down", "downward_only", "full"),
]

# Larger combo, smaller combo. Each difference is a cost only if the pair is nested.
ABLATION_PAIRS = {
    "plan_build (cacheable?)": ("B_plan", "A_upward"),
    "m2l_compute": ("C_m2l", "B_plan"),
    "l2l+rest": ("D_full_down", "C_m2l"),
    "downward_total": ("D_full_down", "A_upward"),
}


def _system(n):
    """The uniform-cube state the previous records in docs/ were taken on."""
    import jax
    import jax.numpy as jnp

    pos = jax.random.uniform(
        jax.random.PRNGKey(0), (n, 3), minval=-1, maxval=1, dtype=jnp.float32
    )
    vel = 0.01 * jax.random.normal(jax.random.PRNGKey(2), (n, 3), dtype=jnp.float32)
    mass = jax.random.uniform(
        jax.random.PRNGKey(1), (n,), minval=0.1, maxval=1.0, dtype=jnp.float32
    )
    return jnp.stack([pos, vel], axis=1), mass


def _solver():
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(
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


def _run(solver, state0, mass, nsteps):
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


def _measure(state0, mass, steps, diag, detail):
    """Per-step wall time (ms) for one (diag, detail) combo."""
    os.environ["JACCPOT_STRICT_REFRESH_DIAG_MODE"] = diag
    os.environ["JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE"] = detail
    solver = _solver()
    _run(solver, state0, mass, 2).block_until_ready()
    t0 = time.perf_counter()
    _run(solver, state0, mass, steps).block_until_ready()
    return (time.perf_counter() - t0) / steps * 1000, solver


#: The downward split this script exists to report, plus the surrounding stages for
#: context. ``dual_downward`` is the parent of the three below it, so it is labelled as
#: an aggregate and must not be added to them.
_REPORTED = (
    "refresh_total_seconds",
    "refresh_tree_build_seconds",
    "refresh_upward_compute_seconds",
    "refresh_dual_downward_seconds",
    "refresh_dual_artifact_build_seconds",
    "refresh_dual_downward_compute_seconds",
    "refresh_dual_m2l_compute_seconds",
    "refresh_dual_l2l_compute_seconds",
    "refresh_nearfield_seconds",
    "refresh_evaluate_seconds",
)


def _print_stage_timers(diagnostics: dict) -> dict[str, float]:
    """Report the curated counters per refresh call, marking aggregates.

    Expected to report nothing on the fused device-resident lane: these counters are
    written by the host-side ``refresh_prepared_state``, which the fused jitted scan does
    not call. Saying so is the point -- the alternative is printing a column of 0.0 that
    reads like a measurement.

    Parameters
    ----------
    diagnostics : dict
        ``FastMultipoleMethod.get_runtime_diagnostics()``.

    Returns
    -------
    dict[str, float]
        The reported counters, in seconds per refresh call. Empty when the counters
        did not record.
    """
    from jaccpot.runtime.fmm_stage_timing import aggregate_counter_names

    aggregates = aggregate_counter_names()
    calls = int(diagnostics.get("refresh_timing_calls", 0) or 0)
    if calls <= 0:
        print(
            "  refresh_timing_calls == 0: the counters did not record, so there is\n"
            "  nothing to report here. Expected on the fused device-resident lane --\n"
            "  they are written by the host-side refresh_prepared_state, which the\n"
            "  fused jitted scan never calls. Not a configuration error."
        )
        return {}
    per_call = {}
    for name in _REPORTED:
        if name not in diagnostics:
            continue
        seconds = float(diagnostics[name]) / calls
        per_call[name] = seconds
        role = "aggregate" if name in aggregates else "leaf"
        label = name.removeprefix("refresh_").removesuffix("_seconds")
        if seconds == 0.0:
            print(
                f"  {label:34s} {'-':>9s}      ({role}; 0.0 -- took no time, or "
                "not instrumented here)"
            )
        else:
            print(f"  {label:34s} {seconds * 1e3:9.1f} ms  ({role})")
    return per_call


def main() -> None:
    n = int(os.environ.get("PROFILE_N", "200000"))
    steps = int(os.environ.get("PROFILE_STEPS", "40"))
    ablation = os.environ.get("PROFILE_ABLATION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for k, v in FUSED_ENV.items():
        os.environ.setdefault(k, v)

    import jax

    cc = getattr(jax.devices()[0], "compute_capability", "n/a")
    state0, mass = _system(n)

    # The authoritative measurement: one full run, per-stage timers read directly.
    per_step_ms, solver = _measure(state0, mass, steps, "full", "full")
    diagnostics = solver.get_runtime_diagnostics()
    print(f"=== N={n} steps={steps} cc={cc}  full run: {per_step_ms:.1f} ms/step ===")
    print("\n=== refresh stage timers, per refresh call (host-side counters) ===")
    per_call = _print_stage_timers(diagnostics)

    out = {
        "cc": str(cc),
        "n": n,
        "steps": steps,
        "full_run_ms_per_step": per_step_ms,
        "refresh_timing_calls": int(diagnostics.get("refresh_timing_calls", 0) or 0),
        "stage_seconds_per_call": per_call,
    }

    if ablation:
        res = {}
        for label, diag, detail in COMBOS:
            try:
                ms, _ = _measure(state0, mass, steps, diag, detail)
                res[label] = ms
                print(
                    f"{label:14s} diag={diag:14s} detail={detail:24s} {ms:8.1f} ms/step",
                    flush=True,
                )
            except Exception as exc:
                res[label] = None
                print(f"{label:14s} ERROR: {str(exc)[:160]}", flush=True)

        attrib, invalid = {}, []
        for label, (bigger, smaller) in ABLATION_PAIRS.items():
            if res.get(bigger) is None or res.get(smaller) is None:
                continue
            value = res[bigger] - res[smaller]
            attrib[label] = value
            if value < 0.0:
                invalid.append(label)
        print("\n=== ablation cross-check (ms/step) ===")
        for label, value in attrib.items():
            bigger, smaller = ABLATION_PAIRS[label]
            if label in invalid:
                print(
                    f"  {label:26s} {value:8.1f}   <-- INVALID: {bigger} "
                    f"({res[bigger]:.1f}) is cheaper than {smaller} "
                    f"({res[smaller]:.1f}), so the combos are not nested and this is "
                    "not a stage cost"
                )
            else:
                print(f"  {label:26s} {value:8.1f}")
        if invalid:
            print(
                f"\n!! {len(invalid)} of {len(attrib)} ablation differences are "
                "negative and therefore invalid -- expected on this lane, see the "
                "module docstring.\n   Of the rest, only downward_total is trustworthy: "
                "a positive difference is not automatically a nested one."
            )
        out["ablation_ms"] = res
        out["ablation_attributed_ms"] = attrib
        out["ablation_invalid"] = invalid

    path = os.environ.get("PROFILE_OUT")
    if path:
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

# FMM fused-lane per-step profiling (2026-07-08)

Reliable attribution of the strict fused `strict_run_v2` per-step cost, measured
on an **RTX 2080 Ti (sm_75), 200k particles, leaf=256, order=4, theta=0.6,
float32**. Purpose: stop future work from re-chasing the wrong bottleneck (as an
earlier plan did, based on a stale non-fused profile).

## Headline

The fused production lane is **compute-bound, not idle-bound**: ~936 ms/step at
**88% mean SM utilization** (`bench/profile_fused_gpu_util.py`). The often-quoted
"8.5% util / launch-latency-bound / upward pass dominates" figure came from a
non-fused / eager measurement and **does not describe the fused lane**.

## Per-step attribution (ablation, `bench/profile_fused_stage_ablation.py`)

Measured inside the real jitted scan via cumulative diag modes
(`JACCPOT_STRICT_REFRESH_DIAG_MODE`), so it is immune to host-timer artifacts:

| Stage | ms/step | share |
|---|---|---|
| downward (dual traversal + M2L/L2L) | 515 | 44% |
| eval + near-field | 412 | 35% |
| upward (geometry + mass + P2M + M2M) | 145 | 12% |
| tree build | 66 | 6% |
| integrator + overhead | 24 | 2% |
| **full** | **~1164** | |

Downward sub-split (`bench/profile_downward_breakdown.py`, detail diag modes):

| Downward sub-stage | ms/step | cacheable? |
|---|---|---|
| plan-build (dual-tree walk / far-pair + neighbor lists) | 195 | **No** — position-dependent |
| M2L compute | 224 | No — genuine FLOPs |
| L2L + rest | 108 | No — genuine FLOPs |

## Why there is no cheap *exact* structural win here

- **M2M / upward is only ~12%**, and M2M itself is ~5 ms (~0.5%). The committed
  fix (`perf(upward): skip padded empty Morton levels`) is correct and
  bit-identical but end-to-end negligible in this config; it used a non-production
  rotation in its microbench, which overstated it.
- **Host-side per-stage timers are unreliable for the fused lane.** The same
  `compute_tree_geometry` measures 0.8 ms jitted-with-folded-topology vs 452 ms in
  the non-fused eager path — a 500× artifact. Only in-scan ablation is trustworthy.
- **The 195 ms plan-build is position-dependent**: it is a dual-tree walk applying
  the MAC test to `TreeGeometry` (node centers/radii), which change every step as
  particles move. Caching the near/far partition across steps produces stale
  interaction lists — an approximation. This is exactly why `_interaction_cache`
  is deliberately position-invalidated for static-radix.

## Real levers (each with a cost)

1. Faster near-field (35%): the Pallas kernel needs **sm_80 (Ampere)**; unavailable
   on sm_75. Largest single win where hardware allows.
2. A Pallas M2L kernel (complex-harmonic): large new effort.
3. Algorithmic accuracy tradeoffs: higher `theta` (fewer far pairs) / lower order.
4. Approximate MAC-partition caching / multi-step refresh cadence: trades accuracy.

## Tools (added this round)

- `bench/bench_upward_sweep.py` — isolated upward-sweep microbench + `static_num_levels` A/B.
- `bench/profile_fused_stage_ablation.py` — in-scan per-stage attribution (the reliable one).
- `bench/profile_downward_breakdown.py` — downward plan-build vs M2L/L2L split.
- `bench/profile_refresh_stage_breakdown.py` — non-fused per-stage host timers (kept for reference; note the eager-overhead caveat above).

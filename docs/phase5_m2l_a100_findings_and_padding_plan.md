# Phase-5 fused M2L: A100 findings + wiring + padding plan (2026-07-13)

Branch: `feat/phase5-pallas-m2l-prototype` (now up to date with main + the
`perf/octree-rsh-kernels` real-basis grouped/cached M2L). Context for whoever
picks up the fused-M2L work next.

## A100 profile — why the M2L is the target
200k, order 4, θ0.8, leaf 256, static-radix fused fast lane (`benchmark_a100/a100_findings.md`):

| stage | pure JAX | Pallas ON |
|---|---|---|
| tree+upward | 4.1 ms (2%) | 4.1 ms (3%) |
| far-field M2L+L2L | 56.6 ms (21%) | 56.7 ms (**47%**) |
| near-field P2P+L2P | 210.6 ms (78%) | 59.2 ms (49%) |
| **total** | 271 ms | **120 ms** |

Pallas near-field = 2.26× overall (210→59 ms). The M2L is untouched by Pallas and
is now the co-bottleneck (47%). **It is launch-bound, not compute-bound**: pure-JAX
M2L is only 1.23× faster on the A100 than the 2080 Ti despite ~5× the fp32 FLOPs, so
the A100's compute sits idle waiting on the per-pair tiny-kernel launches. The win is
collapsing launches (a fused kernel), exactly as the near-field Pallas kernel did.

## Why NOT the grouped/cached real M2L here
The octree grouped/cached real kernel (`m2l_rot_scale_real_batch_cached_blocks`)
relies on **grid-quantized displacements** (few interaction classes) from fixed-cell
box centers. The radix `static_radix` tree has adaptive particle-chunk leaves
("leaves are not fixed spatial cells", `yggdrax/tree.py`) and the fast lane uses
center-of-mass centers → `jnp.unique(centers[tgt]−centers[src])` ≈ one class per pair
→ grouping degenerates. Measured: per-pair real rotation M2L = 52.6 ms vs complex
56.7 ms (only 7% — rotation alone doesn't beat launch-bound). Grouping is fundamentally
an octree optimization; it does not transfer to the radix lane. (A separate follow-up
experiment: build the radix tree with `use_morton_geometry=True` for grid centers and
retry grouping — unproven, and changes the center convention, so needs accuracy revalidation.)

## What is already wired (commit 66a5bb0)
`jaccpot/runtime/_fmm_impl.py`:
- `_fused_complex_m2l_pallas_active()` — gate on env `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`
  AND `pallas_m2l_complex_fused_supported()` (sm_80+), evaluated at trace time.
- `_apply_cached_complex_m2l(src_mult, deltas, blocks_to_z, blocks_from_z, order)` —
  builds `r=|deltas|` and calls `m2l_complex_fused_pallas` when active, else the
  reference `_m2l_complex_batch_cached_kernel`.
- Wired into BOTH `_accumulate_solidfmm_m2l_fullbatch` and the chunked-scan; the
  branch condition is `rotation == "cached" OR _fused_complex_m2l_pallas_active()`, so
  it also triggers on the `solidfmm` rotation the production large_n lane uses (it
  builds the rotation blocks itself regardless of nominal rotation mode).
- Flag-off is byte-for-behavior identical (verified A100 force err 0.2815%, no regression).

## Blocker: the Phase-5 v0 kernel does not lower on Triton
`m2l_complex_fused_pallas` was only ever tested via `interpret=True` (CPU). On a real
A100 it fails: *"Pallas Triton lowering requires ... size is a power of 2. Encountered
an array of shape (5, 9)."* The per-pair shapes are non-power-of-2:
- rotations (Wigner-D, block-diagonal by degree ℓ): `(2ℓ+1)×(2ℓ+1)`, max `9` (ℓ=4) → forces 9→16.
- z-translation core (block-diagonal by order m): `(p+1−|m|)×(p+1−|m|)`, max `p+1=5` → only needs 5→8.
- packed vector `C=(p+1)²=25` → 32.

## Fix options (pick per profile, not blindly)
1. **Uniform pow2 pad** (5→8, 9→16, 25→32). Crudest, most FLOP-waste, but ~free on the
   launch-bound A100 (extra FLOPs run in idle compute). Fastest path to a number that
   confirms the launch-collapse win. Would hurt the compute-bound 2080 Ti (not the target).
2. **m-major restructure** (the kernel's intended "v1"): organize the z-core by m-column
   so its padded dim is 5→**8** not 9→16, and reuse each pair's rotation across m-columns.
   Saves compute and is the cleaner kernel — matters only if fusion turns it compute-bound.
3. **Hand-rolled FMA blocks**: express the small `(2ℓ+1)` block-matmuls as explicit
   multiply-accumulate loops instead of `tl.dot`, sidestepping the pow2 rule with no
   padding. Most manual, least waste.

## Recommended sequencing
1. Do **option 1 (pad-to-measure)** first. Validate against the existing
   `interpret=True` parity test + the pure-JAX twin `m2l_complex_fused_jax`, then
   benchmark on the A100 (target M2L 56.7 → ~20 ms, total 120 → ~80 ms, ~1.5×).
2. Re-profile: if the fused M2L is now compute-bound, implement **option 2 (m-major)**
   to recover the padding waste. If still launch-bound, padding waste is irrelevant.
3. Only after complex M2L works: consider the real-basis fused kernel and/or the
   Morton-geometry grouping experiment.

## Repro / benchmark
- `benchmark_a100/run_a100_profile.sh` (pure-JAX vs Pallas 3-stage decomposition).
- `benchmark_a100/run_a100_real_m2l.sh` (real-basis rotation M2L cost).
- Enable the fused M2L: `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1 ODISSEO_FMM_USE_PALLAS=1`
  (+ the `env_fused.sh` fast-lane knobs). Currently errors on GPU until the kernel is fixed.

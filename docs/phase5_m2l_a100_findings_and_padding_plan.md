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

---

## UPDATE 2026-07-13 (kernel now lowers on Triton; wiring blocked on convention)

Executed option-1 pad-to-measure. Two Triton-GPU blockers on the v0 kernel were
fixed (commit eefcd3d, `jaccpot/pallas/m2l_complex_fused.py`):
1. **power-of-2 shapes** — pad tables/buffers to Cp/Bp/mdp/Kp; padded lanes inert.
2. **no `gather` on Triton GPU** — reformulate pack/unpack/z-core gathers as
   constant-matrix elementwise-multiply+reduce (one-hot `Ppack`/`Uunpack` +
   dense z-core `Z = Zsign*Zfact*r**-Zexp`). Verified viable by a Triton probe.

Validation: interpret parity 9/9; **real-Triton A100 parity vs
`m2l_complex_reference_batch_cached_blocks` = relerr ~3e-7** (orders 2-4). The
Phase-5 kernel is now GPU-functional.

### BLOCKER 1 — convention mismatch (why the fast-lane wiring gives 67% error)
The fused kernel matches the **cached-blocks** convention. The production large_n
fast lane computes M2L via `_m2l_complex_batch_kernel(rotation="solidfmm")`, which
differs from the cached reference by **relerr ~1.8** — and it is NOT a delta-sign,
conjugate, or to_z/from_z basis-swap difference (all tried, all ~1.1). It is a
deeper normalization/scaling convention. So enabling
`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` on the fast lane yields ~68% force error.
=> Need either a solidfmm-convention fused kernel, or a solidfmm<->cached coeff
conversion, or switch the whole lane (upward+M2L+downward) to the cached convention
consistently (the large_n gate currently requires complex_rotation="solidfmm").

### BLOCKER 2 — no speedup per-chunk
Even ignoring correctness, the fused M2L measured 60.7 ms vs 56.7 ms (SLOWER). The
chunked scan calls it per-chunk (~32 chunks at chunk_size 4096) and the gather-free
reformulation adds dense selection-matrix FLOPs; the launch-collapse benefit needs
the M2L batched over ALL far pairs in one call (raise chunk_size to cover the pair
buffer, or a dedicated unchunked fused path), not 32 per-chunk launches.

### Next
1. Reconcile the convention: study `_m2l_complex_batch_kernel` solidfmm rotation +
   z-translation normalization vs the cached reference; build solidfmm-convention
   blocks/z-core for the fused kernel (or a coeff transform). Re-validate force parity.
2. Batch the fused M2L unchunked (single launch over the compact far-pair buffer) to
   actually collapse launches; re-measure (target M2L 56.7 -> ~20 ms).
Flag stays OFF until both are done (kernel fix is committed + validated standalone).

---

## UPDATE 2026-07-14 (Path B done — real z-core Pallas: correct but no speedup)

Fixed + wired the real-valued M2L Pallas kernel (commit bfaaad2):
- `m2l_core_z_real_pallas` now takes `backend` (default `"triton"`); it was
  defaulting to Mosaic-GPU which rejects the small per-(pair,coeff) tiles
  ("bytes=100 not divisible by warpgroup 128"). On Triton it lowers + runs on the
  A100 (relerr ~1e-6 vs pure-JAX, orders 2-4).
- Wired into the fast-lane real dispatch via `_apply_real_m2l` (gated by
  `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` + `pallas_m2l_real_supported`), swapping
  only the z-core kernel inside the correctly-masked real scan. Convention-self-
  consistent: A100 force err **0.2831% (correct)** — no 68% mismatch.

**Result: correct but NO speedup.** real M2L = 56.3 ms with the Pallas z-core vs
**52.6 ms** pure-JAX (slight regression); total 119.5 vs 115.7 ms. => the z-core is
NOT the launch-bound cost. The M2L launch-bound cost lives in the per-pair JAX
**rotations** (rotate to/from z, block-diagonal Wigner) + the **per-chunk scan** (32
chunks @ chunk_size 4096). Fusing only the z-core leaves both untouched.

### Implication for Path A (now the clear target)
A speedup needs the **full** rotate -> z-translate -> rotate-back fused into ONE
kernel (collapsing the rotation launches), run **UNCHUNKED** (single launch over the
whole compact far-pair buffer, not 32 chunks), and in the **solidfmm convention**
(what `_m2l_complex_batch_kernel(rotation="solidfmm")` produces — the phase5 kernel's
cached-blocks reference is NOT it; relerr ~1.8). Steps:
1. Derive solidfmm-convention rotation blocks + z-core for the fused complex kernel
   (or a solidfmm<->cached coeff transform), re-validate force parity in the lane.
2. Call it unchunked over the far-pair buffer (raise chunk_size to the buffer size or
   a dedicated single-launch path).
Both Path A/B flags stay OFF until Path A delivers a validated speedup.

---

## UPDATE 2026-07-14 (DECISIVE: the M2L kernel is NOT the far-field bottleneck)

Built a fully-fused REAL M2L Pallas kernel (`jaccpot/pallas/m2l_real_fused.py`,
commit 3d4cb72): full rotate->z-translate->rotate-back in one Triton kernel,
gather-free + pow2-padded, real (Dehnen) convention. Validated A100 relerr ~1e-15
vs `m2l_rot_scale_real_batch`. Wired flag-gated into `_apply_real_m2l`.

**It does NOT speed up the A100 far-field — and neither can any M2L kernel.**
Measured far-field (M2L+L2L) at 200k/order4/theta0.8, all ~53-58 ms:
| M2L variant | far-field ms |
|---|---|
| complex solidfmm (production) | 56.7 |
| real rot-scale (pure JAX)     | 52.6 |
| real z-core Pallas            | 56.3 |
| real fully-fused Pallas       | 55.8 (chunked) / 58.2 (unchunked) |

Localization: standalone (JIT'd) the ENTIRE real M2L for 64,698 far pairs (build
blocks + rotate + z + rotate) = **9.25 ms** (block-build 9.12). The fast-lane
far-field bucket is ~56 ms => the M2L kernel is only ~16%; **~84% is plumbing**:
per-pair source-multipole gather (`multip_packed[src]`), scatter-add to locals
(`segment_sum`), the **L2L** downsweep, and the **2x padding** (131072 far-pair
buffer vs ~64,698 active). Chunk size (32 chunks vs 1) makes no difference.

### => Re-scoped levers for the A100 far-field (NOT the M2L kernel)
1. **Padding**: process ~64,698 active pairs, not the 131072 buffer (fixed-shape
   compaction / right-size the compact-far-pair cap). ~2x of the far-field volume.
2. **Gather/scatter**: the per-pair source-multipole gather + segment_sum scatter
   dominate; grouping/deduping source nodes or a fused gather-M2L-scatter kernel.
3. **L2L** downsweep cost (separate from M2L) — decompose M2L-only vs L2L first.
The complex + real fused M2L kernels are kept as validated, reusable infra (they'd
matter once the plumbing shrinks, or at higher order / other HW), but the flags
stay OFF: they don't move the A100 number. The big A100 win remains the near-field
Pallas kernel (already in: 210->59 ms).

---

## UPDATE 2026-07-14 (far-field root cause: interaction-list CONSTRUCTION, not M2L)

Decomposed the ~56 ms A100 far-field bucket with clean per-step GPU medians +
production-size micro-benchmarks:
- **M2L accumulate** (`_accumulate_real_m2l_chunked_scan`, gather+kernel+scatter+scan,
  standalone at 1563 nodes / 64,698 pairs / cap 131072): **14.1 ms** (chunk 4096); the
  raw M2L kernel over the gathered pairs is 10.4 ms.
- **gather** `mult[src]` = 0.16 ms; **segment_sum scatter** -> nodes = 0.35 ms (negligible).
- **padding**: tightening the far-pair cap 131072 -> 81920 saved 0.8 ms (the chunked
  scan already skips padded chunks via `lax.cond`). Dead end.
- **L2L**: ~2.5 ms incremental (full_downward - m2l_only).
- **=> by elimination, ~40 ms/step is the dual-tree FAR-PAIR CONSTRUCTION**
  (`_build_dual_tree_artifacts(need_compact_far_pairs=True)` ->
  `build_compact_far_pairs_and_leaf_neighbor_lists`, the MAC traversal that emits the
  ~64,698 far pairs), rebuilt every step (refresh_every locked at 1). ~70% of the
  far-field, ~1/3 of the whole 120 ms step. (Internal `runtime_refresh_*` host timers
  under-report it -- they measure async dispatch, not GPU completion.)

### Consequence
NO M2L kernel work (Path A complex-fused, Path B real-fused/z-core) can help the A100
far-field -- they optimize the 14 ms accumulate, not the 40 ms interaction-list build.
Padding + gather/scatter are also non-levers (measured). **The far-field lever is the
far-pair interaction-list construction** (`_interaction_cache` / dual-tree traversal):
make it cheaper per step, or reuse it across steps (blocked: refresh_every=1). The
fused M2L kernels remain validated infra (flags OFF). The big A100 win is still the
near-field Pallas (210->59 ms, in). Near-field (59) + M2L-accumulate (14) + L2L (~3) +
far-pair-build (~40) + tree/upward (~4) ~ 120 ms checks out.

# FMM step pipeline-fusion — future workstream plan

**Status:** scoped, not started. Drafted 2026-07-10 after the kernel-count
optimization investigation on `feat/fmm-cut-kernel-count`.

## Problem statement (measured)

The device-only fused lane runs the whole integration as one device-resident
`jax.lax.scan` (`_fmm_impl.py:4217`), ~**150 ms/step** at N=200k / A100 / θ=0.6.
A `jax.profiler` perfetto trace shows the step is **~2,100 GPU kernels** but only
**~31 ms of actual kernel compute** — i.e. **~120 ms/step is on-device
inter-kernel overhead** (launch/scheduling/dependency stalls across a long serial
chain), not host dispatch and not compute.

### What is already ruled out (do NOT re-chase)

| Lever tried | Result |
|---|---|
| M2L chunk count 160 → 5 GEMMs | **flat** (<2%) — M2L overlaps, off critical path |
| near-field `num_warps` / `num_stages` | **hurts** (8 warps = 0.46×) |
| near-field full-W vectorized tile | **3–30× worse** (register spills) |
| CUDA graph / `--xla_gpu_enable_command_buffer` | +9% at most; `WHILE` capture won't compile |

**Key lesson: kernel count ≠ wall time.** Cutting 155 M2L kernels did nothing
because they overlap and are off the critical path. Any rewrite MUST target the
*critical-path* kernels, not kernel count in aggregate.

### Trace signature of the ~2,100 kernels/step
- ~184×/step each: `loop_add_fusion`, `loop_multiply_fusion`,
  `loop_transpose_fusion`, `input_reduce_fusion`, `dynamic_update_slice_fusion`
  — the fingerprint of **per-ℓ (order 0..4) / per-node Python-loop / vmap** ops
  emitting many tiny kernels.
- ~160 `cutlass_80_tensorop_c1688gemm` — M2L (32 chunks × 5 ℓ), off critical path.
- near-field `leafpair`, Morton `sort`, `reorder/gather`.

## Goal

Cut the ~120 ms on-device overhead by collapsing the ~2,100 serial small kernels
into a few large ones **on the critical path**. Realistic target: 150 → ~80–100
ms/step (1.5–2×). Bonsai's ~10 ms needs Phase 3 (a monolithic kernel; likely not
worth it against the 10× already banked at 0.118 s).

## Phases

### Phase 0 — Critical-path attribution (BLOCKING, ~2–3 d, no risk)
Kernel count ≠ wall, so first identify which kernels are actually on the critical
path. Method: per-stage ablation like the M2L A/B — artificially collapse or
no-op each stage's kernels (tree/sort, upward P2M/M2M, M2L, L2L/L2P, near-field,
integrate) and measure the wall delta; and/or extract the longest dependency
chain from the perfetto device timeline. **Deliverable:** stages ranked by
critical-path wall contribution. **Gate:** only stages whose collapse moves wall
proceed to Phase 1+. This prevents another M2L-style dead end.

### Phase 1 — Batch per-ℓ / per-node loops (highest confidence, ~1–2 wk, med risk)
The ~184×/step `loop_*`/`reduce` fusions are per-ℓ or per-node loops. Replace
vmap-of-Python-loop / per-ℓ matmuls with a single batched `einsum`/`dot` over all
ℓ and nodes at once. Targets: `operators/m2l_real_rot_scale.py:34-38`,
`operators/real_harmonics.py::l2l_real`, and the upward/downward per-ℓ assembly.
Collapses hundreds of tiny kernels into a few large ones. **Only** where Phase 0
flags critical-path. Parity ≤1e-5 vs current; run `test_real_sh_roundtrip`,
`test_real_basis_runtime`, energy-drift regression.

### Phase 2 — Fuse adjacent elementwise stages (~1–2 wk, med risk)
Merge elementwise pre/post-scale, masking, and `dynamic_update_slice` ops into
their neighboring compute stages (P2M/M2M/L2L/L2P) so XLA emits fewer boundary
kernels — mostly jax-level restructuring to avoid intermediate materialization so
XLA's fusion pass captures them. Depends on Phase 0 ranking.

### Phase 3 — Monolithic fused step kernel (STRETCH, months, high risk)
Write the per-step traversal + M2L + P2P + update as one/few Pallas/CUDA kernels,
Bonsai-style. The only path to ~10 ms, but a near-reimplementation of the FMM
core. **Not recommended** unless a much faster step becomes a hard requirement;
the 10× (0.118 s) already covers the practical need.

## Recommendation

Gate hard on **Phase 0**. It is cheap and decisive: it either reveals 2–3
critical-path stages (making Phase 1/2 a targeted 1.5–2× worth doing) or shows the
overhead is irreducibly spread across many stages (making the rewrite low-ROI —
stop and keep the banked 10×). Given four negative results this round, do not
start Phase 1 without a Phase 0 that names concrete critical-path stages.

## Anchors
- Scan step body: `jaccpot/runtime/_fmm_impl.py:4123-4211`; scan `:4217`.
- Per-ℓ loops: `operators/m2l_real_rot_scale.py`, `operators/real_harmonics.py`.
- Repro harness: `bench/nearfield_leafpair_microbench.py`; fast-lane trace tool
  in scratchpad `fastlane_trace.py`; per-stage A/B pattern from the M2L /
  pallas / command-buffer experiments this round.

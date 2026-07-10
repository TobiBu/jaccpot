# Phase 5 — Pallas GPU-kernel speedups for the jaccpot FMM

Status: planning + first prototype (2026-07-10). Author handoff doc so the perf work
can resume on Ampere+ hardware (see "Hardware" — it cannot be benchmarked on the dev box).

## Hardware constraint (read first)

The dev server has **RTX 2080 Ti (Turing, sm_75)**. In JAX 0.9 **no Pallas GPU backend
runs on sm_75**:
- Mosaic GPU (default) emits `nvvm.cp.async.bulk.wait_group` (Hopper/Ampere TMA) → "not
  supported on sm_75 / Pass pipeline failed".
- `pl.pallas_call(..., backend="triton")` → `FAILED_PRECONDITION: Triton support is only
  enabled for Ampere GPUs (compute capability 8.0) and up, but got compute cap 7.5`.

Consequences:
- **Develop + correctness-check locally with `interpret=True`** (runs kernel semantics on
  CPU/XLA; also runs in CI). This is how the prototype is validated here.
- **All Pallas GPU execution + timing must happen on Ampere+ (A100/H100).** Do not expect
  Pallas GPU numbers on the dev box.
- The existing sm_80 gate in `jaccpot/pallas/nearfield_fused_leaf.py:65`
  (`pallas_nearfield_fused_supported`, requires `compute_capability >= 8.0`) already
  encodes this; the new M2L kernel should gate the same way.

## Where time goes (measured; 200k, leaf 256, order 4, θ 0.6, fp32, RTX 2080 Ti)

Fused fast-lane is compute-bound (~88% SM), ~936–1164 ms/step. Source:
`docs/fmm_fused_perstep_profiling_2026-07-08.md`.

| Stage | ms/step | share |
|---|---|---|
| downward: dual-walk **195** + **M2L compute 224** + L2L 108 | 515 | 44% |
| eval + near-field (L2P + P2P) | 412 | 35% |
| upward (geometry + mass + P2M + M2M) | 145 | 12% |
| tree build | 66 | 6% |

**M2L compute (224 ms) is the single largest genuine-FLOP block and is still plain `vmap`**
(`operators/complex_ops.py:m2l_complex_reference_batch`). It is the top target.

Caveat: these are sm_75 numbers where the near-field Pallas kernel cannot run, and the
"A100: 1224→119 ms/step" figure (`_fmm_impl.py:2391`) is a *fusion* win, not Pallas, with no
published per-stage breakdown. **First A100 task: a fresh in-scan ablation** (extend
`bench/profile_fused_stage_ablation.py`) to confirm M2L is still the top block on Ampere.

## The "fast-lane" is XLA fusion, not Pallas

`JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY=1` (default) → `_prepare_state_dual_and_downward_strict_streamed_fast`
(`_fmm_impl.py:8097`). ~10× win is from device-resident execution (compact COO far-pairs,
no host round-trips, fixed-shape static-radix), all plain JAX/XLA in one scan. Pallas is
orthogonal: the only optionally-Pallas stage reachable from the fast lane is P2P near-field
(`use_pallas`, default False). So Pallas kernels compose with the fast lane; they don't
replace it.

## Prioritized kernel targets

### 1. Complex-basis M2L kernel — highest impact (targets the 224 ms)

Structure of the M2L (`m2l_complex_reference` / `..._batch_cached_blocks`):
1. **rotate to z**: block-diagonal-by-ℓ complex matmul — `_apply_complex_rotation_blocks_padded_batch`
   is `einsum("nbij,nbj->nbi", blocks_to_z, pack_by_ell(M))`. Blocks: `[N, p+1, 2p+1, 2p+1]`.
2. **z-core**: `translate_along_z_m2l_complex` — for each output (n,m), `out = Σ_{k=|m|}^{p-n}
   (-1)^m · fact[n+k] / r^(n+k+1) · M_rot[k, m]`. Real coefficients × complex coeffs;
   diagonal in m (couples only across ℓ within the same m).
3. **rotate back**: block-diagonal complex matmul with `blocks_from_z`.

Why XLA is slow here: `vmap` materialises `M_rot` and `local_z` in HBM per pair and issues
many tiny per-pair GEMMs. **Kernel design**: fuse rotate→z-core→rotate per pair so the
intermediates stay in registers/shared memory; consume the same precomputed rotation blocks.
Pallas has no complex dtype → carry real/imag as two real arrays (complex matmul =
`(Ar·xr − Ai·xi) + i(Ar·xi + Ai·xr)`; z-core coefficients are real so they apply to real and
imag independently).

Optimization ladder (v0 → A100 tuning):
- **v0 (this prototype)**: grid over pairs; per program does block-diag rotate → z-core →
  rotate on real/imag; validated via `interpret` against `m2l_complex_reference_batch_cached_blocks`.
  Directly swappable for the `vmap` at `_fmm_impl.py:11651/11659`.
- **v1 (A100)**: exploit block-diagonal-by-ℓ (don't densify) and stream pairs.
- **v2 (A100)**: **class-major** — one program-block per displacement class
  (`GroupedInteractionBuffers`, `yggdrax/grouped_interactions.py`): load that class's rotation
  blocks into shared memory ONCE, stream all its pairs, scatter-accumulate into target locals,
  and **fuse the gather** (`multip[src]`, `centers[tgt]-centers[src]`) currently done outside
  the kernel (`_fmm_impl.py:11635`). This is where the rotation-block reuse pays off.

### 2. Promote + tune the existing near-field P2P kernel — lower effort (35% block)

`jaccpot/pallas/nearfield_fused_leaf.py` is already register-tiled and streams sources
(kernels at `:136` pairs, `:409` leafpair). Work: (a) flip it to default on Ampere instead of
`use_pallas=False`; (b) shared-memory target streaming + leaf-size template specialization;
(c) fold the L2P eval into the same kernel (both live in the 412 ms eval+near block); (d) tune
`target_subtile`/`num_warps`/`num_stages` on A100.

### 3. L2P kernel — medium; fuse into #2 or standalone.

### Deferred / low priority
- Upward P2M/M2M (12%, M2M ~0.5%), L2L — small.
- Dual-walk plan-build (195 ms) — tree traversal, poor Pallas fit; already device-resident.
- Radix sort vs `jnp.lexsort` (`yggdrax/morton.py:95`) — tree build is only 6%.

## Dev + validation workflow

1. Each kernel ships with a **pure-jnp reference** (repo convention:
   `nearfield_fused_leaf_jax`, and for M2L the reference is
   `m2l_complex_reference_batch[_cached_blocks]`).
2. **`interpret=True` unit tests** assert kernel == reference to tight tol (fp32 ~1e-4, fp64
   ~1e-10). These run on the 2080s and in CI (CPU).
3. **A100**: real GPU execution; extend `bench/profile_fused_stage_ablation.py` (in-scan
   attribution) + `bench/guard_large_n_radix_fast_lane.py`; report M2L-compute ms XLA-vs-Pallas
   across the 65k→524k N-ladder (`examples/benchmark_gpu_n_ladder_production.py`); tune knobs.
   All bench scripts already integrate `autocvd` for free-GPU selection.

## Prototype delivered in this pass

- `jaccpot/pallas/m2l_complex_fused.py` — v0 fused complex-M2L Pallas kernel
  (`m2l_complex_fused_pallas`) + jnp reference + sm_80 support gate.
- `tests/pallas/test_m2l_complex_fused.py` — `interpret=True` parity test vs
  `m2l_complex_reference_batch_cached_blocks` (fp64 and fp32).

## First A100 tasks (in order)
1. Fresh in-scan per-stage ablation on A100 to confirm the M2L share.
2. Run the M2L prototype on-device (drop the `interpret=True` gate); parity vs `vmap` at N-ladder.
3. Implement v1/v2 (block-diagonal + class-major shared-mem) and benchmark the delta.
4. Then tackle target #2 (near-field promotion) with real tuning.

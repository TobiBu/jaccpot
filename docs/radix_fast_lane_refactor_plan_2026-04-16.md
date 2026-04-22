# Radix-Focused Fast-Lane Refactor Plan

Date: 2026-04-16

Execution checklist:
[`docs/radix_fast_lane_implementation_checklist_2026-04-16.md`](/export/home/tbuck/jaccpot/docs/radix_fast_lane_implementation_checklist_2026-04-16.md)

## 1) Goal

Reduce the large-`N` runtime gap against `jaxfmm` by introducing a
performance-first, radix-specialized execution lane in `jaccpot` that
prioritizes:

- static array shapes
- JIT-compile stability
- fused/vectorized batched kernels
- minimal repeated gather/scatter traffic
- minimal hot-path control flow

This plan intentionally trades some generality for throughput in a tightly
scoped production path.

## 2) Current measured motivation

On the same local setup (single `autocvd` GPU), quick 1M uniform tests with
`p=4`, `theta=0.6`:

- `jaxfmm`: ~`0.443 s` (`benchmarks/jaxfmm_quick_1M_p4_theta06_s3.csv`)
- `jaccpot`: ~`4.759 s` (`benchmarks/jaccpot_quick_1M_p4_theta06.csv`)

Observed gap is ~`10.7x`.

## 3) Scope and non-scope

In scope:

- radix tree only (`yggdrax` radix producer + `jaccpot` consumer)
- large-`N` steady-eval focus
- nearfield hot path first
- JAX-native implementation first, with optional fused-kernel follow-up

Out of scope for this refactor:

- octree parity
- full general API unification
- broad adaptive policy redesign
- immediate multi-GPU distribution changes

## 4) Product shape: two runtime lanes

Keep two distinct lanes:

- `general_lane`:
  current feature-rich flexible runtime
- `radix_fast_lane`:
  constrained high-throughput path

`radix_fast_lane` should hard-fail early when unsupported options are passed
instead of branching in hot runtime code.

## 5) Fast-lane invariants (hard constraints)

Initial invariants to lock:

- `tree_type="radix"`
- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `working_dtype=float32`
- fixed `leaf_size` profile (start with `256`)
- fixed nearfield layout profile (`B`, `Lt`, `Ls`, batch sizes)
- `grouped_interactions=False`
- `streamed_far_pairs=True`
- no runtime fallback to alternative nearfield layouts in evaluate

## 6) Performance strategy

Core strategy:

1. Do expensive reindexing/sorting once during prepare.
2. Use a single static target-major nearfield payload format.
3. In evaluate, gather once per batch, accumulate locally, write back once.
4. Avoid repeated global scatter operations inside pair loops.
5. Keep hot kernels dense, static, and mask-based (not ragged dynamic loops).

## 7) Data contract (radix fast lane)

Define one canonical prepared payload for nearfield:

- `target_leaf_ids: int32[T]`
- `target_leaf_particle_ids: int32[T, Lt]`
- `target_leaf_particle_mask: bool[T, Lt]`
- `source_leaf_ids_padded: int32[T, B]`
- `source_leaf_valid_mask: bool[T, B]`
- `source_leaf_particle_ids: int32[T, B, Ls]`
- `source_leaf_particle_mask: bool[T, B, Ls]`

Where:

- `T`: number of target leaves in prepared state
- `B`: max source-neighbor leaves per target block slot (fixed profile cap)
- `Lt`: padded target particles per target leaf slot (static)
- `Ls`: padded source particles per source leaf slot (static)

Rules:

- all arrays are static-shape tensors
- overflow beyond profile caps is handled explicitly by a separate slow
  fallback path outside the fast kernel
- fast kernel never sees ragged structures

## 8) Nearfield kernel shape

Target behavior:

1. Batch target leaves (`T_batch` static).
2. Gather target/source positions/masses into dense tensors once.
3. Compute pair interactions in vectorized tile form.
4. Reduce all source contributions into target-local accumulator.
5. Scatter/write global acceleration once per target particle batch.

Design constraints:

- no per-edge global scatter in the inner loop
- no repeated sort/compaction in evaluate
- no branchy runtime dispatch in hot section
- avoid unnecessary `lax.scan` chains for tiny-step loops

## 9) JAX-native implementation guidance

Preferred primitives:

- `vmap` over static batch dimensions
- dense tensor ops and reductions along contiguous axes
- shape-stable masking (`where`) instead of dynamic slicing where possible

Avoid in hot path:

- repeated `lax.scan` over ragged edge lists
- data-dependent loop trip counts
- polymorphic shapes that trigger recompilation
- intermediate representations requiring repeated global gather/scatter

## 10) Refactor phases

### Phase 0: benchmark gate freeze

- freeze a canonical runtime command and config for A/B
- freeze one GPU-run procedure (`autocvd`, one free GPU, same env)
- establish pass/fail guard against regressions

### Phase 1: lane split and assertions

- introduce explicit `radix_fast_lane` entry path
- add invariant checks and early failure
- ensure no runtime fallback branches are hidden in this lane

### Phase 2: producer-side canonical payload

- implement/lock canonical target-major payload generation in `yggdrax` path
- store payload into `LargeNPreparedState` once
- guarantee static-shape contract for fast-kernel input

### Phase 3: consumer-side single fast kernel

- implement nearfield fast kernel for canonical payload
- accumulate local then single writeback
- keep overflow fallback outside fast kernel boundary

### Phase 4: profiling and tile tuning

- tune `T_batch`, `B`, `Lt`, `Ls`, and tile sizes
- compare memory pressure and runtime vs baseline
- reject changes that improve micro-metrics but worsen gate runtime

### Phase 5: optional fused-kernel follow-up

- if JAX-native dense kernel plateaus, layer Pallas/custom fused primitive
  over the same canonical payload

## 11) Benchmark and validation protocol

Primary performance gate:

- `N=1048576`
- uniform cube distribution
- `theta=0.6`
- `max_order=4`
- `leaf_size=256`
- same single GPU selected by `autocvd`

Collect at minimum:

- steady evaluate mean/std
- nearfield component timings
- compile time and recompilation count
- prepared-state memory footprint

Correctness checks:

- compare accelerations vs current trusted path on representative `N` ladder
- track max relative vector-norm error with fixed tolerances

## 12) File ownership and change map

Expected primary touchpoints:

- `yggdrax` traversal/neighbor payload generation
- `jaccpot/runtime/_large_n_types.py`
- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/runtime/_fmm_impl.py`
- focused benchmark worker wiring in `examples/benchmark_gpu_radix_worker.py`

Guideline:

- isolate fast-lane code to avoid accidental regressions in general lane
- keep interfaces explicit between producer payload and consumer kernel

## 13) Risks and mitigations

Risk: static padding inflates memory.

- Mitigation: profile caps per GPU class; explicit overflow path.

Risk: compile explosion from too many profile variants.

- Mitigation: keep one default profile at first; expand variants only with
  benchmark evidence.

Risk: fast lane drifts from correctness.

- Mitigation: mandatory A/B correctness checks against reference lane in CI.

Risk: premature fused-kernel work before representation is stable.

- Mitigation: finish canonical payload and JAX-native dense path first.

## 14) Exit criteria for this refactor

Minimum success criteria:

- reproducible speedup on frozen 1M gate
- nearfield runtime reduction is the dominant contributor
- no correctness regressions in defined test set
- fast-lane path is stable and documented

Stretch criteria:

- substantial narrowing of current ~10x gap to `jaxfmm`
- optional fused path integrated behind fast-lane interface

## 15) Immediate next actions

1. Add `radix_fast_lane` runtime profile and invariant assertions.
2. Finalize canonical target-major payload schema and tensor names.
3. Implement first static-shape nearfield kernel with single writeback policy.
4. Run frozen gate and compare against current baseline.
5. Iterate tile/profile tuning only after step 4 is stable.

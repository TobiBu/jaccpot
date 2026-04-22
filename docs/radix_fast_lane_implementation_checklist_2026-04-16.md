# Radix Fast-Lane Implementation Checklist

Date: 2026-04-16
Companion design doc:
[`docs/radix_fast_lane_refactor_plan_2026-04-16.md`](/export/home/tbuck/jaccpot/docs/radix_fast_lane_refactor_plan_2026-04-16.md)

## 1) How to use this checklist

- Treat each item (`RFL-XX`) as an implementation issue.
- Keep PRs small and in order where possible.
- Do not mix general-lane cleanup into fast-lane PRs.
- Each issue has:
  - files
  - concrete function signature target
  - TODOs
  - done criteria

## 1.1) Status snapshot (2026-04-20)

Legend:

- `done`: implemented and wired in primary path
- `partial`: present but not fully validated/tuned
- `todo`: not implemented yet

Current snapshot:

1. `RFL-00`: `done`
2. `RFL-01`: `partial`
3. `RFL-02`: `done`
4. `RFL-03`: `done`
5. `RFL-04`: `partial`
6. `RFL-05`: `partial`
7. `RFL-06`: `partial`
8. `RFL-07`: `partial`
9. `RFL-08`: `partial`
10. `RFL-09`: `done`

Observed code indicators for this snapshot:

- fast-lane config flag/invariants in
  `jaccpot/runtime/_large_n_nearfield.py`
- fast payload type in `jaccpot/runtime/_large_n_types.py`
- fast payload prepare wiring in `jaccpot/runtime/_large_n_pipeline.py`
- fast-lane evaluate routing and kernel entry in
  `jaccpot/runtime/_large_n_nearfield.py` and
  `jaccpot/nearfield/near_field.py`

## 2) Frozen benchmark gate for all issues

Use this gate before/after each issue:

- `N=1048576`, uniform cube
- `theta=0.6`, `max_order=4`, `leaf_size=256`
- single `autocvd` GPU
- `preset=large_n_gpu`, `basis=solidfmm`, `tree_type=radix`

Primary command (steady eval) is already documented in:
[`docs/nearfield_tonb_runbook.md`](/export/home/tbuck/jaccpot/docs/nearfield_tonb_runbook.md)

## 3) Issue breakdown

### RFL-00: Add explicit fast-lane mode and invariants

Files:

- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/runtime/_fmm_impl.py`
- `jaccpot/runtime/_large_n_types.py`

Target signatures:

```python
# jaccpot/runtime/_large_n_types.py
@dataclass(frozen=True)
class LargeNExecutionConfig:
    nearfield_mode: str
    nearfield_edge_chunk_size: int
    retain_leaf_groups: bool
    retain_pair_vectors: bool
    precompute_scatter: bool
    target_owned_block_size: int
    speed_prepared_layout: bool
    radix_fast_lane: bool  # NEW
```

```python
# jaccpot/runtime/_large_n_nearfield.py
def resolve_large_n_execution_config(
    fmm: object,
    *,
    num_particles: int,
) -> LargeNExecutionConfig:
    ...
```

TODOs:

1. Add `radix_fast_lane` flag resolution with hard assertions:
   - radix tree only
   - `large_n_gpu`, `solidfmm`, `float32`
   - `grouped_interactions=False`
2. Fail fast on unsupported options when lane enabled.
3. Keep default behavior unchanged when lane disabled.

Done criteria:

- lane can be toggled on/off deterministically
- unsupported configs raise explicit errors
- no benchmark regression when lane is off

---

### RFL-01: Introduce canonical target-major payload contract

Files:

- `/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py`
- `/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py` (`NodeNeighborList`)
- `jaccpot/runtime/_large_n_types.py`
- `jaccpot/runtime/_large_n_pipeline.py`

Current producer type:

```python
class NodeNeighborList(NamedTuple):
    offsets: Array
    neighbors: Array
    leaf_indices: Array
    counts: Array
    particle_order_leaf_indices: Array
    particle_order_to_native_leaf: Array
    neighbor_leaf_positions: Array
    target_block_leaf_ids: Array
    target_block_source_leaf_ids: Array
    target_block_valid_mask: Array
    target_block_offsets: Array
    target_block_size: int
```

Target additional fields (producer-side, static payload):

```python
# yggdrax NodeNeighborList additions (NEW)
target_leaf_particle_ids_padded: Array          # [T, Lt]
target_leaf_particle_mask_padded: Array         # [T, Lt]
target_leaf_source_leaf_ids_padded: Array       # [T, B]
target_leaf_source_valid_mask_padded: Array     # [T, B]
target_leaf_source_particle_ids_padded: Array   # [T, B, Ls]
target_leaf_source_particle_mask_padded: Array  # [T, B, Ls]
```

TODOs:

1. Add optional fields to `NodeNeighborList` (default empty arrays if disabled).
2. Build canonical payload once in `_result_to_neighbors(...)` for radix only.
3. Thread payload through `LargeNPreparedState`.
4. Keep legacy TONB fields during migration; do not remove yet.

Done criteria:

- prepare path exports static-shape payload tensors
- evaluate path can read payload without recomputing sort/group structures

---

### RFL-02: Add fast-lane payload view type in jaccpot

Files:

- `jaccpot/runtime/_large_n_types.py`
- `jaccpot/runtime/_large_n_pipeline.py`

Target signatures:

```python
# jaccpot/runtime/_large_n_types.py
@dataclass(frozen=True)
class RadixFastNearfieldPayload:
    target_leaf_ids: Array                       # [T]
    target_particle_ids: Array                   # [T, Lt]
    target_particle_mask: Array                  # [T, Lt]
    source_leaf_ids: Array                       # [T, B]
    source_leaf_valid_mask: Array                # [T, B]
    source_particle_ids: Array                   # [T, B, Ls]
    source_particle_mask: Array                  # [T, B, Ls]
    batch_tile_t: int
    batch_tile_s: int
```

```python
# jaccpot/runtime/_large_n_types.py
@dataclass(frozen=True)
class LargeNPreparedState:
    ...
    radix_fast_payload: Optional[RadixFastNearfieldPayload] = None  # NEW
```

TODOs:

1. Add new payload dataclass.
2. Add pytree flatten/unflatten support.
3. Populate payload in `prepare_large_n_state(...)` when lane is enabled.

Done criteria:

- payload survives JIT/pytree serialization round-trips
- no shape polymorphism introduced in state object

---

### RFL-03: Add dedicated fast-lane nearfield kernel entry

Files:

- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/nearfield/near_field.py`

Target signatures:

```python
# jaccpot/runtime/_large_n_nearfield.py
def evaluate_large_n_nearfield_fast_lane(
    fmm: object,
    state: LargeNPreparedState,
    *,
    return_potential: bool,
) -> Array | tuple[Array, Array]:
    ...
```

```python
# jaccpot/nearfield/near_field.py
def compute_leaf_p2p_accelerations_radix_fast_lane(
    *,
    positions_sorted: Array,
    masses_sorted: Array,
    payload: "RadixFastNearfieldPayload",
    G: Union[float, Array] = 1.0,
    softening: float = 0.0,
    return_potential: bool = False,
) -> Array | tuple[Array, Array]:
    ...
```

TODOs:

1. Add new kernel entry (do not modify old one initially).
2. Dispatch from `evaluate_large_n_nearfield(...)` when `radix_fast_lane` is true.
3. Keep current specialized/legacy branches as fallback.

Done criteria:

- new path runs end-to-end behind a single condition
- old path remains functional for non-fast-lane configs

---

### RFL-04: Eliminate repeated gather/scatter in kernel core

Files:

- `jaccpot/nearfield/near_field.py`

Target internal structure:

```python
def _radix_fast_lane_kernel_core(
    positions_sorted: Array,        # [N, 3]
    masses_sorted: Array,           # [N]
    target_particle_ids: Array,     # [T, Lt]
    target_particle_mask: Array,    # [T, Lt]
    source_particle_ids: Array,     # [T, B, Ls]
    source_particle_mask: Array,    # [T, B, Ls]
    source_leaf_valid_mask: Array,  # [T, B]
    *,
    G: Array,
    softening_sq: Array,
) -> Array:                         # [N, 3]
    ...
```

TODOs:

1. Gather target/source blocks once per static batch.
2. Compute pair interactions as dense tensor math.
3. Reduce into target-local accumulator.
4. One writeback per target batch (not per edge).
5. Avoid per-edge `lax.scan` in inner hot loop unless required and benchmarked.

Done criteria:

- global scatter count materially reduced
- kernel is shape-static for frozen profile

---

### RFL-05: Static-shape profile locking

Files:

- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/nearfield/near_field.py`

Target signatures:

```python
def resolve_radix_fast_lane_profile(
    *,
    num_particles: int,
    leaf_size: int,
) -> tuple[int, int, int, int]:
    """Return (B, Lt, Ls, target_batch_tiles) for a fixed profile."""
```

TODOs:

1. Add one default static profile (no dynamic profile switching initially).
2. Pad payload to profile limits in prepare.
3. Route overflow to explicit slow fallback path.

Done criteria:

- one stable compiled executable for repeated 1M runs
- overflow accounted for explicitly (not silently truncated)

---

### RFL-06: Pipeline and dispatch cleanup

Files:

- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/runtime/_fmm_impl.py`

Current pipeline signatures:

```python
def prepare_large_n_state(...) -> LargeNPreparedState
def evaluate_large_n_state(...) -> tuple[Array, Optional[Array]]
```

TODOs:

1. Keep `prepare_large_n_state(...)` responsible for all payload assembly.
2. Keep `evaluate_large_n_state(...)` free of payload transformation logic.
3. Ensure `_fmm_impl.py` dispatch does not branch into legacy lanes when fast-lane is active.

Done criteria:

- zero evaluate-time repacking/sorting for fast lane
- cleaner separation between prepare-time and evaluate-time responsibilities

---

### RFL-07: Instrumentation for gather/scatter accounting

Files:

- `jaccpot/nearfield/near_field.py`
- `examples/benchmark_gpu_radix_worker.py`

Target signatures:

```python
@dataclass(frozen=True)
class RadixFastLanePerfCounters:
    gather_bytes: int
    scatter_bytes: int
    scatter_ops: int
    target_batches: int
```

```python
def collect_radix_fast_lane_counters(...) -> RadixFastLanePerfCounters:
    ...
```

TODOs:

1. Add optional counters behind a diagnostics flag.
2. Print counters in worker `nearfield_components` mode.
3. Track before/after values during rollout.

Done criteria:

- counters visible in benchmark worker output
- confirms repeated scatter reduction

Status note (2026-04-17):

- implemented `collect_radix_fast_lane_counters(...)` in
  `jaccpot/nearfield/near_field.py`
- worker now emits:
  - `nearfield_radix_fast_lane_gather_bytes`
  - `nearfield_radix_fast_lane_scatter_bytes`
  - `nearfield_radix_fast_lane_scatter_ops`
  - `nearfield_radix_fast_lane_target_batches`
  - `nearfield_radix_fast_lane_source_slot_tiles`
- remaining work is rollout tracking across A/B benchmark history

---

### RFL-08: Tests for contract and correctness

Files:

- `tests/unit/core/test_near_field.py`
- `tests/integration/test_fmm.py`
- optional new file: `tests/unit/runtime/test_radix_fast_lane.py`

Target test additions:

1. payload shape contract test (prepare path)
2. fast-lane vs baseline acceleration agreement on small/medium `N`
3. overflow fallback correctness test
4. deterministic behavior test under fixed seed

Done criteria:

- tests pass on CPU and selected GPU CI slices
- accuracy tolerances explicitly documented in tests

Status note (2026-04-17):

- added unit tests in `tests/unit/core/test_near_field.py`:
  - `test_radix_fast_lane_accel_matches_large_n_specialized_small`
  - `test_collect_radix_fast_lane_counters_matches_payload_formula`
- added integration test in `tests/integration/test_fmm.py`:
  - `test_radix_fast_lane_prepared_state_matches_large_n_baseline`
  - `test_large_n_prepacked_overflow_fallback_matches_tiled_overflow`
  - `test_radix_fast_lane_fixed_seed_repeatability`
- integration tolerance now documents deterministic reduction-order drift
  (`rtol=5e-4`, `atol=5e-4`) with max error diagnostics in assertion text
- overflow fallback test compares tiled overflow vs generic overflow kernels
  with forced overflow payloads (`rtol=2e-4`, `atol=3e-4`)
- fixed-seed repeatability test validates deterministic input generation,
  radix fast payload layout equivalence, and repeat-stable accelerations
- remaining work:
  - selected GPU CI slice coverage

Audit note (2026-04-17, static/JIT closure check):

- large-N production hot path is now staticized/JIT-oriented; residual env parsing
  remains only in legacy compatibility fallbacks
- completed since last audit:
  - removed Python per-leaf overflow compaction loop in prepare path (vectorized
    block masking/compaction in `jaccpot/runtime/_large_n_pipeline.py`)
  - moved `JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD` env read out of JIT core;
    compiled evaluator now takes a static flag input
    (`jaccpot/runtime/_fmm_impl.py::_evaluate_tree_compiled_impl`)
  - switched large-N radix fast-lane eval to payload-first near-field path
    (`evaluate_large_n_state` now calls `evaluate_large_n_nearfield_fast_lane`
    when `state.radix_fast_lane and not return_potential`)
  - moved radix fast-lane scan/unroll fallback tuning reads into prepare-time payload
    fields so eval wrapper no longer reads env for these knobs
- remaining closure gap:
  - large-N hot path now passes these knobs via prepared-state static fields;
  env parsing in `compute_leaf_p2p_accelerations_large_n_accel_only` remains
  only as compatibility fallback for non-large-N/legacy direct callers

Lock-in update (2026-04-17):

- `large_n_gpu` radix path now defaults to radix fast-lane on
  and enforces radix fast-lane for production acceleration evaluate path
- acceleration eval on large-N prepared state is now locked to radix fast-lane
  payload route; non-fast-lane accel eval raises with clear guidance
- legacy `JACCPOT_LARGE_N_RADIX_FAST_LANE=0` opt-out remains accepted only as a
  no-op compatibility input; execution config is policy-locked to fast lane
- obsolete generic large-N nearfield dispatcher helper removed from
  `jaccpot/runtime/_large_n_nearfield.py`
- added lock-policy tests:
  - `tests/unit/test_large_n_fast_path_policy.py`
  - integration lock compatibility updates in `tests/integration/test_fmm.py`

Post-lock 1M performance snapshot (autocvd GPU 9, `runs=5`, `warmup=1`):

- baseline (`block=0`, fast-lane off): `evaluate_mean_seconds=1.7188`
- fast-lane (`block=8`, speed layout on): `evaluate_mean_seconds=0.3673`
- confirmed speedup: `4.68x` on steady-state evaluate
- quick fast-lane block sweep (`runs=2`) suggested best in tested set at `block=8`

---

### RFL-09: Benchmark gate automation

Files:

- `bench/` scripts (new or existing guard)
- `docs/nearfield_tonb_runbook.md` (update with fast-lane commands)

TODOs:

1. Add one benchmark guard command for fast lane at 1M.
2. Persist run CSVs with lane metadata.
3. Add pass/fail threshold per milestone.

Done criteria:

- repeatable A/B run command in docs
- guard catches regressions automatically

Status note (2026-04-20):

- added benchmark guard script:
  - `bench/guard_large_n_radix_fast_lane.py`
- guard runs frozen 1M baseline-vs-fast-lane sweep with one `autocvd`-selected
  free GPU, reused across both A/B lanes
- guard persists both `.csv` + `.json` artifacts with lane metadata into
  `benchmarks/`
- guard enforces configurable speedup threshold (`--min-speedup`, default `2.0`)
- guard defaults to `benchmark_scope=steady_eval`; full-pass sweep checks are
  supported with a lower default threshold (`1.03x`)
- runbook now includes one copy/paste guard command in
  `docs/nearfield_tonb_runbook.md`

Memory follow-up note (2026-04-20):

- reduced radix fast-lane prepared-state footprint by trimming legacy
  `neighbor_leaf_positions` from `LargeNPreparedState.neighbor_list`
  in the fast-lane path
- no new runtime flag was introduced; trim is applied by default for fast lane
- generic fallback paths that need neighbor-leaf positions rebuild them from
  `offsets` + `neighbors` when absent
- added lock-policy regression coverage:
  - `tests/unit/test_large_n_fast_path_policy.py`:
    `test_large_n_fast_lane_trims_neighbor_leaf_positions`
- latest quick validation snapshot (single selected GPU,
  `CUDA_VISIBLE_DEVICES=9`, `runs=1`, `warmup=0`):
  - steady-eval metric (`evaluate_mean_seconds`):
    - baseline (`block=0`): `1.1089s`
    - fast lane (`block=8`): `0.9834s`
    - observed speedup: `1.13x`
  - full-pass metric (`mean_seconds`) from same-GPU A/B run:
    - baseline (`block=0`): `4.9309s`
    - fast lane (`block=8`): `5.0855s`
    - observed speedup: `0.97x` (full-pass currently not improved)

## 4) Recommended implementation order

1. `RFL-00`
2. `RFL-01`
3. `RFL-02`
4. `RFL-03`
5. `RFL-04`
6. `RFL-05`
7. `RFL-06`
8. `RFL-07`
9. `RFL-08`
10. `RFL-09`

## 5) PR slicing recommendation

- PR1: `RFL-00` + scaffolding
- PR2: `RFL-01` + `RFL-02`
- PR3: `RFL-03` + basic kernel wiring
- PR4: `RFL-04` kernel optimization
- PR5: `RFL-05` + overflow fallback
- PR6: `RFL-06` cleanup
- PR7: `RFL-07` + `RFL-08`
- PR8: `RFL-09` benchmark guard/docs

## 6) Completion definition

This checklist is complete when:

1. fast lane is default-able for target production profile,
2. benchmark gate shows sustained improvement over current large-N baseline,
3. correctness tests pass with agreed tolerances,
4. repeated gather/scatter overhead is measurably reduced in counters.

# Nearfield Structural Comparison (jaxFMM vs jaccpot)

Date: 2026-04-15

## 1) Focused comparison

### What jaxFMM does structurally (direct path)

From `jaxfmm-0.2.0` source:

- Direct connectivity (`dir_cnct`) is sorted by target box id in hierarchy build.
  - `jaxfmm/hierarchy.py`: lines 143-144 (`dir_cnct = dir_cnct[jnp.argsort(dir_cnct[:,0])]`).
- Nearfield direct evaluation runs over that target-sorted connectivity and updates a
  box-major accumulator (`pot_glob`) with `indices_are_sorted=True`.
  - `jaxfmm/fmm.py`: lines 83-101 (`pot_glob.at[idcs[:,0]].add(..., indices_are_sorted=True)`).
- Data is padded in box-major form once, then flattened/reindexed at the end.
  - `jaxfmm/hierarchy.py`: lines 292-303.

Practical effect: ownership is target-box-major for most of the direct nearfield work.

### What jaccpot does structurally today (large-N hot path)

- Specialized large-N bucketed accel-only path is active for production compiled eval.
  - `jaccpot/runtime/_fmm_impl.py`: lines 8165-8193.
- Pair loop is edge-chunk based, with per-chunk gather -> pair arithmetic -> reduce ->
  repeated particle-order scatter/update.
  - `jaccpot/nearfield/near_field.py`: lines 617-775.
- Even with target-leaf reduction, the hot loop still repeatedly writes particle-order
  updates instead of owning a target buffer for longer.

From your benchmark log (`docs/small_gpu_large_n_followup_2026-03-25.md`):

- At 1M, farfield is tiny (`~0.017 s`) and nearfield dominates (`~1.17-1.56 s`).
  - lines 899-903, 920-929.
- Pair arithmetic is dominant but not all of pair-path cost; writeback/update overhead is
  still material (~18-22% in multiple probes).
  - lines 1014-1029 and 1349-1365.
- Multiple local scatter/schedule variants were regressions, confirming this is a
  representation problem, not a primitive swap problem.

## 2) Why this still lags jaxFMM

Core gap is structural ownership of nearfield accumulation:

1. jaccpot still performs many global particle-order updates inside the pair loop.
2. jaxFMM keeps work target-owned longer and flattens late.
3. Existing jaccpot target-owned probes were slower because they were expressed on top of
   the current edge-major representation and incurred extra setup/masking overhead.

## 3) Proposed structural change (recommended)

## Target-Owned Nearfield Blocks (TONB)

Introduce a new prepared nearfield representation for large-N GPU path:

- **Producer (yggdrax):** emit target-owned nearfield blocks directly from traversal
  output, not just flat edge lists.
- **Consumer (jaccpot):** run a target-owned kernel that accumulates per-target-leaf
  contributions across all neighbor blocks, then writes back once per target leaf (or once
  per target batch), avoiding repeated particle-order scatter in the inner loop.

### 3.1 yggdrax changes

Extend `NodeNeighborList` (or add a sibling structure) with pre-grouped target-owned nearfield metadata:

- `target_leaf_ids_sorted` (leaf-major)
- `target_leaf_offsets` (CSR offsets into neighbor blocks)
- `source_leaf_positions` or `source_leaf_ids` grouped per target
- optional fixed-width block packing metadata for GPU kernels (`block_size`, `block_offsets`)

Implementation point:

- Build in `_result_to_neighbors(...)` after near compact-fill, where counts/offsets are
  already present and leaf ordering is resolved.
  - `yggdrax/_interactions_impl.py`: lines 3639-3675.

This moves sorting/grouping cost to prepare time and removes repeated reshape/sort work in evaluate.

### 3.2 jaccpot changes

Add a new specialized kernel entry for this representation:

- `compute_leaf_p2p_accelerations_large_n_target_owned_blocks(...)`
- Outer loop: target leaf (or target-leaf batch)
- Inner loop: source blocks for that target
- Accumulator stays target-local until target done
- Single writeback for target leaf/batch

Integration point:

- Dispatch in `_evaluate_tree_compiled_impl(...)` when TONB metadata is present.
  - `jaccpot/runtime/_fmm_impl.py`: lines 8165-8193 (same branching site as current specialized path).

## 4) Why this is likely better than current experiments

Current target-owned v1/v2 experiments in `near_field.py` still build heavy flattened
intermediates per block and were not fed by a producer-side target-owned layout.
TONB changes both sides:

- producer emits target-owned structure once,
- consumer executes target-owned accumulation without repeated global scatter.

That is the closest architecture to the jaxFMM ownership pattern while keeping your
minimum-memory policy under explicit control.

## 5) Performance expectation for the 0.1 s goal

Based on current probes, removing only scatter/update overhead is insufficient for 0.1 s:

- pair arithmetic alone is already ~1.0-1.66 s in your measured 1M runs.

So 0.1 s at 1M likely requires **both**:

1. TONB representation (to remove update pressure), and
2. a fused GPU microkernel path (Pallas/custom) for pair arithmetic + accumulation.

In short: representation rewrite is necessary, but likely not sufficient for 0.1 s without
kernel fusion on supported GPUs.

## 6) Suggested implementation sequence

1. Add TONB metadata in yggdrax as opt-in (no behavior change for existing consumers).
2. Thread TONB through `LargeNPreparedState` in jaccpot.
3. Implement TONB kernel in jaccpot specialized nearfield path.
4. A/B vs current specialized path at 1M (`nearfield_components` + steady eval).
5. If TONB helps but plateaus, implement fused Pallas target-tile kernel on top of TONB.

## 7) Implementation status (2026-04-15 checkpoint)

Status after this session:

1. Step 1 is implemented (opt-in producer metadata in `yggdrax`).
2. Step 2 is implemented (TONB threaded through `LargeNPreparedState`).
3. Step 3 is implemented (TONB specialized nearfield branch in `jaccpot`).
4. Step 4 is pending GPU runtime A/B due no free GPU in this session.

### Active toggles

- Producer:
  - `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE`
  - default `0` (disabled)
- Consumer:
  - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE`
  - default `0` (disabled)

For TONB A/B runs, set both toggles to the same nonzero block size (start with `32`).

### Focused validation completed in this checkpoint

- cross-repo compile checks passed
- focused `yggdrax` near-neighbor slices passed
- focused `jaccpot` large-`N` nearfield and compiled-dispatch slices passed
- cross-repo producer+consumer toggle-on path passed in focused CPU-backed tests

### Next actions when GPU is available

1. Run pinned-GPU A/B with TONB off vs on at fixed 1M runtime shape.
2. Collect both `steady_eval` and `nearfield_components`.
3. Sweep block size `{16, 32, 64}` if first TONB result is neutral/positive.
4. Decide whether to keep TONB opt-in or promote to default for selected presets.

## 8) April 16, 2026 implementation outcome

Status: Steps 1-4 were completed with GPU A/B.

### What was added beyond the April 15 checkpoint

1. Target-major prepacked prepared layout in jaccpot large-N state:
- padded source-leaf ids/masks in `[leaf, block, lane]` form
- memory guard + capped fast blocks:
  - `JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS`

2. Hybrid execution strategy:
- prepacked target-major fast path for first blocks per leaf
- compact overflow fallback to existing target-block kernels
- runtime dispatch now accepts both packed and overflow TONB payloads

### Measured outcome (`N=1048576`, `autocvd` GPU `9`)

- TONB off baseline (`block_size=0`):
  - `evaluate_mean_seconds=1.7923` (`runs=3`, `warmup=2`)
- TONB on (`block_size=32`) + speed-prepared layout:
  - `evaluate_mean_seconds=7.2978` (`runs=3`, `warmup=2`)
  - `prepared_state_mb=216.55`
  - `nearfield_specialized_pairs_seconds=1.8271`

### Structural conclusion

1. The TONB architecture is now fully integrated and memory-controlled, but on
   the current backend it remains slower than baseline.
2. Current fastest production path is still non-TONB specialized bucketed.
3. Next major lever is not additional Python-level loop reshaping; it is a
   lower-overhead fused kernel path for nearfield pair/update work.

## 9) Next-day refactor plan (April 17, 2026)

Primary objective:

- attempt one more full XLA-friendly TONB refactor cycle before committing to a
  fused custom kernel path.

Execution order:

1. Freeze one benchmark gate and keep all comparisons against it.
2. Collapse TONB runtime to one canonical dense layout path plus compact
   overflow payload.
3. Refactor pair hot loop to dense static tiles only (no ragged inner control).
4. Keep accumulators target-local; single writeback per target tile.
5. Keep only static-trip-count scans in hot loops, then tune unroll.
6. Limit shape polymorphism to avoid compile churn.
7. Keep memory constrained via speed-layout caps (`MAX_MB`, `FAST_BLOCKS`).
8. Decide end-of-day:
   - continue XLA route if TONB shows clear progress toward `<1s`
   - otherwise move to fused Pallas/custom kernel on same prepared layout.

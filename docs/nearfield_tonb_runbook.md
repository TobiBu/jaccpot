# Nearfield TONB Runbook

Date: 2026-04-15

This runbook is the copy/paste path for running the new cross-repo TONB
(target-owned-nearfield-blocks) A/B checks once a GPU is free.

## Production default policy (updated 2026-04-20)

For regular `large_n_gpu` production runs, we now treat radix fast-lane as the
default execution path:

- preset: `large_n_gpu`
- tree: `radix`
- basis: `solidfmm`
- dtype: `float32`
- nearfield mode: `bucketed`
- default target-owned block size: `8` (when
  `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE` is unset or `<= 0`)
- recommended production leaf size from latest runtime sweeps: `256`

Notes:

- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE` remains an optional override for
  explicit A/B experiments.
- This document still contains older TONB A/B checkpoints below; those sections
  are useful for historical context but do not override the production default
  policy above.

## 1) Preconditions

1. Use the same environment used by current runtime checks:
   - `micromamba run -n odisseo`
   - `JAX_ENABLE_X64=1`
   - sibling `yggdrax` checkout available on `PYTHONPATH` (helper script already handles this)
2. Keep worker autotuning disabled for A/B fairness.
3. Keep the fixed 1M runtime shape unchanged between baseline and TONB runs.

## 2) Fixed runtime shape (A/B control)

Use this exact config JSON in both baseline and TONB runs:

```json
{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}
```

## 3) Baseline commands (TONB off)

`steady_eval`:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode sweep \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 3 \
    --warmup 1 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

`nearfield_components`:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode nearfield_components \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 3 \
    --warmup 1 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

## 4) TONB-on commands (block size 32)

`steady_eval`:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode sweep \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 3 \
    --warmup 1 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

`nearfield_components`:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode nearfield_components \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 3 \
    --warmup 1 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

## 5) Block-size sweep (if `32` is neutral or better)

Run the same TONB-on commands with:

- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=16`, `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=16`
- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=64`, `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=64`

Keep all other settings identical.

## 6) What to record in the checkpoint note

For each run, record:

1. Absolute date/time and physical GPU used.
2. Resolved `CUDA_VISIBLE_DEVICES` and `JACCPOT_NVIDIA_SMI_GPU_INDEX`.
3. Full toggle state (`YGGDRAX_*`, `JACCPOT_*`).
4. `evaluate_mean_seconds` and `evaluate_std_seconds` (`sweep` mode).
5. `evaluate_large_n_nearfield_seconds` and pair-path component timings
   (`nearfield_components` mode).

## 7) Focused sanity checks (non-GPU blockers)

If runtime checks fail and you need quick logic sanity:

```bash
cd /export/home/tbuck/yggdrax
PYTHONPATH=/export/home/tbuck/yggdrax JAX_ENABLE_X64=1 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_tree_interactions.py -k neighbor

cd /export/home/tbuck/jaccpot
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/core/test_near_field.py -k large_n
```

## 8) Latest GPU-0 results (2026-04-15)

Pinned environment:

- `CUDA_VISIBLE_DEVICES=0`
- `JACCPOT_NVIDIA_SMI_GPU_INDEX=0`
- `PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot`
- `JAX_ENABLE_X64=1`

`steady_eval` (`runs=3`, `warmup=1`):

- baseline (`YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0`, `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0`)
  - `evaluate_mean_seconds=1.7018`
  - `evaluate_std_seconds=0.0042`
- TONB-on (`YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32`, `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`)
  - `evaluate_mean_seconds=3.6538`
  - `evaluate_std_seconds=0.3529`

`nearfield_components` (`runs=1`, `warmup=0`, diagnostic-only):

- baseline:
  - `evaluate_large_n_nearfield_seconds=1.2679`
  - `nearfield_specialized_pair_target_leaf_owned_seconds=137.1535`
  - `prepared_state_mb=189.99`
- TONB-on:
  - `evaluate_large_n_nearfield_seconds=4.6651`
  - `nearfield_specialized_pair_target_leaf_owned_seconds=145.4640`
  - `prepared_state_mb=215.97`

Interpretation:

- current TONB-on path regresses both end-to-end steady-eval and specialized
  nearfield timings on this GPU/runtime shape.
- next step should be focused profiling inside target-owned pair kernels and
  payload movement before any block-size sweep.

## 9) April 16, 2026 status update (post structural rewrite)

Structural state:

- TONB producer payloads and offsets are now fully threaded through
  `LargeNPreparedState` and compiled runtime dispatch.
- A new speed-prepared layout path was added with controlled prepacking:
  - `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS`

Observed 1M results on `autocvd` GPU `9`:

- Baseline off (`block_size=0`): `evaluate_mean_seconds=1.7923`
- TONB on (`block_size=32`) + speed prepared layout:
  - `evaluate_mean_seconds=7.2978`
  - `prepared_state_mb=216.55`

Conclusion:

- keep TONB disabled by default for production benchmarking right now
- continue profiling from TONB-off baseline unless explicitly testing TONB
  internals
- no block-size sweep is recommended until TONB kernel overhead is reduced

Prepared-state memory note (2026-04-20):

- radix fast-lane prepared state now trims legacy
  `neighbor_leaf_positions` storage by default (no extra flag)
- this removes a large duplicated nearfield tensor from resident state memory
  when running fast-lane acceleration benchmarks
- when non-fast-lane fallback code needs neighbor-leaf positions, it rebuilds
  them from `offsets` + `neighbors`
## 10) Radix Fast-Lane 1M guard automation (April 20, 2026)

Use this command to run the frozen 1M baseline-vs-fast-lane guard on one free
GPU selected by `autocvd`:

```bash
micromamba run -n odisseo python bench/guard_large_n_radix_fast_lane.py \
  --benchmark-scope steady_eval \
  --runs 3 \
  --warmup 1 \
  --fast-block-size 8
```

Notes:

- writes both `.csv` and `.json` artifacts into `benchmarks/`
- selects one free GPU via `autocvd` once and reuses that same device for both
  baseline and fast-lane runs
- runs baseline with `target_block_size=0` and fast lane with
  `target_block_size=8`
- default speedup guard is `2.0x` for `steady_eval`
- `full` scope is also supported and uses a lower default threshold (`1.03x`)
  since it includes prepare overhead

Latest quick snapshot (2026-04-20, `CUDA_VISIBLE_DEVICES=9`, `runs=1`,
`warmup=0`):

- steady-eval:
  - baseline (`block=0`): `1.1089s`
  - fast lane (`block=8`): `0.9834s`
  - speedup: `1.13x`
- full pass:
  - baseline (`block=0`): `4.9309s`
  - fast lane (`block=8`): `5.0855s`
  - speedup: `0.97x`

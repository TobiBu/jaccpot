# Small-GPU Large-N Follow-Up

This note captures the follow-up work completed on March 25, 2026 after the
H200 memory breakthrough. The focus here was the smaller RTX 2080 Ti-class GPU
path and making the single-`N` and runtime notebooks reflect the current
minimum-memory implementation.

## Main Code Change

The main new optimization is in Yggdrax:

- [`yggdrax/_interactions_impl.py`](/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py)

What changed:

- Added a near-only bounded count-pass helper that mirrors the existing
  far-only compact-fill path.
- `build_leaf_neighbor_lists(...)` now uses that helper in the explicit
  traversal-config path when `max_interactions_per_node` is not explicitly
  overridden.
- This lets the near-only split traversal fill exact-length flat neighbor
  storage directly instead of first materializing the dense
  `(num_leaves, max_neighbors_per_leaf)` neighbor buffer.

Why this matters:

- The old near-only radix path allocated a dense staging buffer even though the
  final retained neighbor list was much smaller.
- That dense buffer was the clearest remaining warm-memory target in the
  Yggdrax near-only split builder.

Focused validation run:

```bash
python3 -m py_compile /export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py
micromamba run -n odisseo python -m pytest -q -o addopts='' /export/home/tbuck/yggdrax/tests/unit/test_tree_interactions.py -k neighbor
```

Observed test result:

- `4 passed, 7 deselected`

## Single-N Notebook Findings

Notebook:

- [`examples/benchmark_gpu_single_n_memory.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_single_n_memory.ipynb)

Latest confirmed runs were still saved against physical GPU `9`, not GPU `1`.

### 1. `N = 524288` after the near-only compact-fill change

Artifacts:

- [`benchmarks/single_n_memory/single_n_524288_summary.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_524288_summary.csv)
- [`benchmarks/single_n_memory/single_n_524288_prepare_stage_memory_split.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_524288_prepare_stage_memory_split.csv)

Key results:

- prepare peak delta: about `98 MB`
- warm prepare peak delta: about `68 MB`
- evaluate peak delta: about `202 MB`
- prepared state size: about `25.95 MB`

Most important stage split result:

- `dual_tree_split_near_only_warm`: about `6 MB`
- `dual_tree_split_build_warm`: about `30 MB`
- `dual_tree_split_far_only_warm`: about `2 MB`

Interpretation:

- the near-only split builder is no longer the dominant warm transient at this
  scale
- the new Yggdrax near-only compact-fill path is very likely active in this run

### 2. The lean notebook now fits `N = 1048576`

Artifacts:

- [`benchmarks/single_n_memory/single_n_1048576_summary.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_1048576_summary.csv)
- [`benchmarks/single_n_memory/single_n_1048576_config.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_1048576_config.csv)

Key results:

- prepare peak delta: about `172 MB`
- warm prepare peak delta: about `144 MB`
- evaluate peak delta: about `404 MB`
- prepared state size: about `52.54 MB`
- output shape: `(1048576, 3)`

Exact traversal settings in that successful run:

- `max_pair_queue=1048576`
- `process_block=256`
- `max_interactions_per_node=32768`
- `max_neighbors_per_leaf=16384`

Interpretation:

- the single-`N` minimum-memory notebook now fits at least `1048576` particles
  on this smaller GPU class in the lean configuration

## Notebook Configuration Changes

### `benchmark_gpu_single_n_memory.ipynb`

Changed defaults:

- disabled `memory_sweep_enabled`
- disabled `prepare_peak_compare_enabled`
- disabled `prepare_stage_memory_split_enabled`

Reason:

- on the smaller GPU, those extra worker analyses can fail even when the main
  prepare/evaluate path still fits
- the notebook default is now closer to a lean capacity-validation run

### `benchmark_runtime_large_N_performance.ipynb`

Updated:

- the runtime traversal seed is no longer fixed at the old
  `262144 / 8192 / 4096` baseline
- it now scales with `num_particles` and reaches the same `N=1048576`-capable
  traversal regime used by the successful single-`N` memory run

Reason:

- the runtime sweep should measure performance on the current low-memory path,
  not on stale small-capacity traversal defaults

## Main Conclusions

1. The Yggdrax near-only split traversal memory issue is materially improved.
2. The single-`N` notebook now fits `1048576` particles in lean mode on the
   small GPU path.
3. The runtime notebook needed a config refresh more than a code-path change:
   it already used `large_n_gpu` + `minimum_memory`, but its traversal seed was
   stale.

## Recommended Next Steps

1. Restart the notebook kernel before future runs so it picks up both the local
   Yggdrax patch and the updated notebook defaults.
2. Run the refreshed runtime sweep from
   [`examples/benchmark_runtime_large_N_performance.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_large_N_performance.ipynb)
   on GPU `1`.
3. Record the best runtime points at `524288` and `1048576`.
4. If `1048576` is now comfortably stable, extend the runtime sweep toward
   `2097152`.

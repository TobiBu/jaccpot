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

## Additional Results From The Same Session

### 3. Single-`N` notebook now fits `N = 16777216`

Artifact:

- [`benchmarks/single_n_memory/single_n_16777216_summary.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_16777216_summary.csv)

Key results:

- prepare time: about `129.21 s`
- warm prepare time: about `109.13 s`
- evaluate time: about `541.89 s`
- prepared state size: about `877.95 MB`
- prepare peak delta: about `2172 MB`
- warm prepare peak delta: about `2140 MB`
- evaluate peak delta: about `6470 MB`
- evaluate peak GPU used: about `8889 MB`
- output shape: `(16777216, 3)`

Interpretation:

- the current lean large-`N` path now reaches `16777216` particles on the
  RTX 2080-class GPU setup used in this notebook
- memory headroom during evaluate is now much tighter than at `524288` or
  `1048576`, so future runtime optimizations must avoid large retained
  auxiliary buffers

## Runtime Audit And Findings

After the memory breakthrough, the next focus shifted to runtime.

### 1. Configuration-level runtime wins already identified

Focused worker measurements on GPU `1` showed:

- the large-`N` GPU runtime prefers a larger nearfield edge chunk than the old
  `128` default
- around `524288`, `nearfield_edge_chunk_size=256` is better
- around `1048576`, the worker prefers `nearfield_edge_chunk_size=512`
- `m2l_chunk_size` is not a strong runtime lever in this regime
- the `1048576` traversal capacities can be trimmed relative to the first
  successful 1M fit without hurting runtime

That led to two local `jaccpot` updates:

- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- [`examples/benchmark_runtime_large_N_performance.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_large_N_performance.ipynb)

What changed:

- the auto nearfield heuristic for `large_n_gpu` on GPU now uses bucketed mode
  earlier and raises the minimum chunk size to `256` from `N >= 262144` and to
  `512` from `N >= 1000000`
- the runtime notebook now seeds larger particle counts with leaner but still
  working traversal capacities
- the runtime notebook uses `NearFieldConfig(mode="auto", ...)` in its main
  runtime sweep cells so it can benefit from the updated heuristic

Status:

- these local `jaccpot` runtime changes are currently present in the worktree
  but not yet committed

### 2. Nearfield runtime hot spot identified

The runtime audit of:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)

showed that the large-`N` minimum-memory evaluate path still spends time in the
generic bucketed nearfield machinery.

Most important finding:

- when `use_precomputed_scatter=False`, the bucketed nearfield path was
  rebuilding a per-chunk scatter schedule during evaluate with
  `_build_scatter_schedule(...)`
- on large GPU runs, precomputed scatter schedules are deliberately disabled
  above `131072` particles by policy in
  [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
  and
  [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- therefore, larger `N` runs repeatedly pay that per-chunk schedule-building
  cost at evaluate time

Why not simply retain scatter schedules:

- retained scatter schedules are stored as three dense integer arrays
  (`chunk_sort_indices`, `chunk_group_ids`, `chunk_unique_indices`)
- their retained size scales with `chunk_count * chunk_size * max_leaf_size`
- at `16777216`, evaluate already peaks near `8.9 GB`, so there is very little
  safe headroom for new retained metadata

### 3. First kernel-side runtime patch

Local code change:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

What changed:

- in the bucketed nearfield path, when `use_precomputed_scatter=False`, the
  code no longer builds a temporary per-chunk scatter schedule
- instead, it now scatters the chunk outputs directly with
  `_scatter_contributions(...)` and `_scatter_scalar_contributions(...)`
- this was applied in both nearfield implementations:
  - the tree-backed path
  - the explicit leaf-group prepared-state path

Why this was chosen:

- it directly removes repeated per-evaluate schedule-building overhead
- it does not increase prepared-state memory
- it does not change the large-`N` minimum-memory retention policy

Validation status:

- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py`
  passed
- a broad pytest run was attempted, but collection failed in unrelated
  `yggdrax` import / x64 configuration issues before this nearfield change
  could be isolated
- a direct runtime A/B script on GPU `1` showed that retained scatter schedules
  still remain disabled at `524288` under the current large-`N` GPU policy,
  which reinforces that the direct-scatter kernel path is the right place to
  optimize

Status:

- this kernel-side nearfield runtime patch is present locally but not yet
  committed

## Current Worktree State

Relevant modified files:

- [`examples/benchmark_gpu_single_n_memory.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_single_n_memory.ipynb)
- [`examples/benchmark_runtime_large_N_performance.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_large_N_performance.ipynb)
- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

Other worktree notes:

- [`docs/large_n_gpu_memory_handoff_2026-03-13.md`](/export/home/tbuck/jaccpot/docs/large_n_gpu_memory_handoff_2026-03-13.md)
  is deleted locally
- benchmark outputs under
  [`benchmarks/single_n_memory`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory)
  remain untracked
- worker autotune caches and memory profiles are also untracked

## Recommended Next Steps

1. Re-run the focused runtime worker benchmark on GPU `1` for `524288` and
   `1048576` with the current local `near_field.py` patch so the new direct
   scatter path can be compared against the earlier baseline.
2. If the direct-scatter patch helps, commit:
   - [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
   - [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
   - [`examples/benchmark_runtime_large_N_performance.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_large_N_performance.ipynb)
3. If the gain is weak, keep the memory-safe runtime config improvements but
   continue kernel work inside
   [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py),
   especially:
   - `_pair_contributions_batched(...)`
   - bucket sizing and scatter shape
   - duplicated bucketed chunk logic between the two prepared/unprepared code
     paths
4. Avoid enabling retained scatter schedules globally for large GPU runs until
   their retained size is quantified more carefully against the `16777216`
   memory envelope.

## March 26 GPU 1 Checkpoint

After reloading the current local worktree on March 26, 2026, the focused
runtime worker path was re-run on GPU `1` with `CUDA_VISIBLE_DEVICES=1`,
`JACCPOT_NVIDIA_SMI_GPU_INDEX=1`, and `JAX_ENABLE_X64=1`.

Why `JAX_ENABLE_X64=1` mattered:

- the local `yggdrax` import path still fails collection otherwise due to the
  same `uint64`/`uint32` truncation issue seen earlier
- the runtime worker itself also imports that path, so x64 needs to be enabled
  for reliable validation in this environment

### Valid worker results

#### `N = 524288`

Clean fixed-chunk `steady_eval` runs with a fresh worker runtime-cache context
showed:

- fixed `nearfield_edge_chunk_size=256`: `evaluate_mean_seconds ~= 0.6713 s`
- fixed `nearfield_edge_chunk_size=512`: `evaluate_mean_seconds ~= 0.6123 s`

Interpretation:

- on the current local code path, GPU `1` now prefers `512` over `256` at
  `524288`
- this is consistent with the first fresh autotuned run in the same session,
  which also chose `512`

#### `N = 1048576`

A fresh autotuned `steady_eval` worker run with the updated lean traversal seed

- `evaluate_mean_seconds ~= 1.4241 s`
- `prepare_mean_seconds ~= 0.7570 s`
- worker-selected traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- worker-selected `nearfield_edge_chunk_size=256`

Interpretation:

- with the current leaner `1M` traversal seed, GPU `1` selected `256`, not the
  earlier `512` note from March 25
- this suggests the best nearfield chunk is coupled to traversal capacity and
  should be treated as a configuration-level choice, not a single fixed rule

### Worker cache caveat

- the worker-side runtime autotune cache is derived as a sibling file named
  `runtime_worker_autotune_cache.json`
- if multiple checks reuse the same parent directory, prior autotuned chunk
  choices can be silently reapplied even when a later run intends to force a
  different chunk size
- future fixed-chunk A/B checks should isolate the cache directory per
  experiment

## March 26 Runtime Audit Follow-up

After the worker-side `audit` mode landed locally, the large-`N` minimum-memory
path was re-run with explicit fixed traversal / nearfield settings to isolate
nearfield runtime on current code.

Common runtime config for the focused checks:

- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `memory_objective="minimum_memory"`
- `nearfield_mode="bucketed"`
- `nearfield_edge_chunk_size=512`
- `streamed_far_pairs=true`
- `grouped_interactions=false`
- `enable_interaction_cache=false`
- `retain_traversal_result=false`
- `retain_interactions=false`

### GPU 7 fixed-path checks

Clean `steady_eval` worker runs on GPU `7` with isolated cache directories
showed:

- `N=524288`: `evaluate_mean_seconds ~= 0.9091 s`,
  `prepare_mean_seconds ~= 0.5531 s`
- `N=1048576`: `evaluate_mean_seconds ~= 2.5440 s`,
  `prepare_mean_seconds ~= 1.0302 s`

Both runs resolved to:

- `resolved_large_n_memory_path_active = true`
- `resolved_nearfield_mode = "bucketed"`
- `resolved_nearfield_edge_chunk_size = 512`

Audit-mode component timings on GPU `7` showed:

- `N=524288`:
  - `evaluate_total_seconds ~= 0.9198 s`
  - `evaluate_large_n_nearfield_seconds ~= 2.4316 s`
  - `evaluate_large_n_farfield_seconds ~= 0.0088 s`
  - `downward_m2l_seconds ~= 0.1584 s`
  - `downward_l2l_seconds ~= 0.1835 s`
  - `prepared_state_mb ~= 30.41`
- `N=1048576`:
  - `evaluate_total_seconds ~= 3.3352 s`
  - `evaluate_large_n_nearfield_seconds ~= 2.2334 s`
  - `evaluate_large_n_farfield_seconds ~= 0.0164 s`
  - `downward_m2l_seconds ~= 0.4452 s`
  - `downward_l2l_seconds ~= 0.5255 s`
  - `prepared_state_mb ~= 62.02`

Interpretation:

- farfield remains tiny
- the runtime bottleneck is still nearfield
- GPU `7` did not show an obvious win from the current arithmetic changes

### Bucketed regrouped-scatter experiment

One follow-up experiment temporarily restored per-chunk grouped scatter
reduction inside the non-precomputed bucketed path while keeping retained
scatter schedules disabled.

That version was a clear regression on GPU `1`:

- `N=524288`:
  - `evaluate_total_seconds ~= 1.4275 s`
  - `evaluate_large_n_nearfield_seconds ~= 2.9367 s`
  - `evaluate_large_n_farfield_seconds ~= 0.0107 s`
- `N=1048576`:
  - `evaluate_total_seconds ~= 6.1937 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.3054 s`
  - `evaluate_large_n_farfield_seconds ~= 0.0159 s`

Interpretation:

- rebuilding a grouped scatter schedule per chunk is not a good direction for
  the current minimum-memory large-`N` GPU path
- the direct-scatter bucketed path should remain the active baseline

### Paused code state

The regrouped-scatter experiment was reverted locally after the GPU `1`
regression.

The current uncommitted nearfield code still includes:

- the `rsqrt`-based distance rewrite in:
  - [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
    `_self_contributions(...)`
  - [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
    `_pair_contributions(...)`
  - [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
    `_pair_contributions_batched(...)`
- the direct-scatter bucketed path in both prepared/unprepared nearfield
  implementations

At the time work paused, one additional kernel-only variant was present locally
but not yet validated because the 1M GPU `1` comparison run was interrupted
when GPU memory pressure appeared:

- keep `rsqrt`
- keep direct scatter
- use explicit multiply-and-sum contractions instead of the temporary
  `einsum(...)` form

### Recommended next steps from this pause point

1. Wait for a free GPU and re-run the 1M GPU `1` audit with the current local
   `near_field.py` state to validate the paused multiply-and-sum kernel
   variant.
2. If that variant is not clearly better, narrow the next experiment to the
   direct-scatter update path rather than revisiting grouped per-chunk scatter.
3. Keep using isolated worker cache directories for every A/B because worker
   runtime autotune state can otherwise leak between runs.

## March 26 Late-Night Nearfield Iteration Log

After the checkpoint above, work continued with focused 1M audit A/B checks on
the explicit minimum-memory large-`N` path, mostly on GPU `0` and briefly on
GPU `1`.

### Stable GPU 0 reference point

A clean 1M audit on GPU `0` with the then-current direct-scatter bucketed path
gave:

- `evaluate_total_seconds ~= 2.1937 s`
- `evaluate_large_n_nearfield_seconds ~= 2.2567 s`
- `evaluate_large_n_farfield_seconds ~= 0.0099 s`

This is the reference point for the late-night iterations below.

### Experiments that clearly regressed

1. Source-tiling inside `_pair_contributions_batched(...)`

   A tiled source-accumulation version with `_PAIR_SOURCE_TILE_SIZE = 64`
   regressed badly on GPU `0` at `N=1048576`:

   - `evaluate_total_seconds ~= 6.3989 s`
   - `evaluate_large_n_nearfield_seconds ~= 3.4820 s`

   Interpretation:

   - forcing a scanned source-tile loop inside the pair kernel was much worse
     than the existing full-leaf batched contraction
   - this version was reverted locally

2. Rebuilding grouped scatter schedules per chunk

   Already recorded earlier, but reconfirmed directionally: reintroducing
   per-chunk grouped scatter in the minimum-memory path is not promising.

### Experiments that were not compelling

1. Lighter masked batched pair kernel

   A variant that removed the explicit `pair_mask` / `safe_dist_sq` flow inside
   `_pair_contributions_batched(...)` and instead zeroed invalid sources only
   through `source_mass_effective` produced a mixed result on GPU `0`:

   - `evaluate_total_seconds ~= 2.1769 s`
   - `evaluate_large_n_nearfield_seconds ~= 2.2868 s`
   - `evaluate_large_n_farfield_seconds ~= 0.0209 s`

   Interpretation:

   - total eval moved slightly in the right direction
   - isolated nearfield moved slightly in the wrong direction
   - this was not convincing enough to keep as a meaningful optimization

   This variant was reverted locally.

### Experiment that modestly helped

1. Target-local edge ordering for direct-scatter bucketed execution

   On the minimum-memory path, when pair vectors and precomputed scatter
   schedules are both absent, `compute_leaf_p2p_accelerations(...)` previously
   rebuilt leaf-pair vectors with `sort_by_source=True`.

   That was changed so bucketed runs without precomputed scatter keep
   target-local edge order instead:

   - preserve neighbor-list edge order
   - favor output-update locality for direct scatter

   GPU `0` result at `N=1048576`:

   - `evaluate_total_seconds ~= 2.1372 s`
   - `evaluate_large_n_nearfield_seconds ~= 2.2483 s`
   - `evaluate_large_n_farfield_seconds ~= 0.0251 s`

   Interpretation:

   - small but real improvement versus the GPU `0` reference point
   - this change is worth keeping as the current local baseline

### Current paused code state

At the time work paused for the night, the local uncommitted
[`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
contains all of the following:

- `rsqrt`-based distance math in:
  - `_self_contributions(...)`
  - `_pair_contributions(...)`
  - `_pair_contributions_batched(...)`
- direct-scatter bucketed execution (grouped per-chunk scatter rebuilds are not
  active)
- target-local edge ordering for bucketed runs without precomputed scatter
- a new structural change that removes the per-bucket
  `lax.cond(jnp.any(valid_edge), ...)` wrappers inside both bucketed
  implementations, so every `lax.scan` step now executes the same masked gather
  -> batched pair kernel -> scatter path

Validation status for the current paused code:

- `python3 -m py_compile /export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py /export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py`
  passed
- the new “no per-bucket branch” version has **not yet been benchmarked**

### Best next step tomorrow

Run the same 1M audit on a free GPU, preferably GPU `0`, against the current
local code state to isolate the effect of removing the per-bucket branch:

- same explicit minimum-memory large-`N` config as the other late-night runs
- `nearfield_mode="bucketed"`
- `nearfield_edge_chunk_size=512`
- fixed traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`

If the branchless chunk body helps, keep it and continue optimizing within the
bucketed scan. If not, the next experiments should stay focused on reducing
direct-scatter / gather overhead inside each bucket rather than changing the
pair kernel shape again.

## March 27 Cleanup Update

After reviewing the local worktree against the note above, two cleanup changes
were made to bring the code back in line with the most defensible validated
baseline.

### 1. GPU auto nearfield heuristic was tightened

File:

- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)

What changed:

- the `large_n_gpu` + GPU + `NearFieldConfig(mode="auto", ...)` heuristic no
  longer forces `nearfield_edge_chunk_size=512` from `N >= 1000000`
- it now keeps the simpler bucketed GPU floor of `256` from
  `N >= 262144`

Why:

- the later March 26 worker note in this same document showed that with the
  leaner 1M traversal seed, the worker-selected nearfield chunk at
  `N = 1048576` was `256`, not `512`
- keeping a hard-coded `512` auto override in code was therefore ahead of the
  newest documented evidence

Validation:

- focused integration test passed on GPU `9` with:

```bash
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
CUDA_VISIBLE_DEVICES=9 \
JAX_ENABLE_X64=1 \
micromamba run -n odisseo python -m pytest -q -o addopts='' \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py \
  -k adaptive_nearfield_edge_chunk_size_auto_policy
```

Observed result:

- `1 passed, 54 deselected`

Related test update:

- [`tests/integration/test_fmm.py`](/export/home/tbuck/jaccpot/tests/integration/test_fmm.py)
  now covers both:
  - the existing CPU auto-policy behavior
  - the `large_n_gpu` GPU auto-policy branch

### 2. Unbenchmarked branchless bucket loop was reverted

File:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

What changed:

- the temporary removal of per-bucket
  `lax.cond(jnp.any(valid_edge), ...)` guards inside the bucketed nearfield
  loops was reverted locally

What was kept:

- `rsqrt`-based distance math
- direct-scatter bucketed execution when scatter schedules are not precomputed
- target-local edge ordering for bucketed runs without precomputed scatter

Why:

- the branchless chunk-body variant was explicitly recorded above as not yet
  benchmarked
- reverting just that structural experiment reduces the risk of carrying an
  unvalidated hot-path change forward while still preserving the earlier
  validated runtime improvements

Status after cleanup:

- the local baseline now reflects:
  - validated heuristic cleanup
  - validated test coverage for the GPU auto branch
  - removal of the still-unbenchmarked branchless bucket-loop experiment

Recommended next step:

1. Re-run the focused large-`N` nearfield audit from the restored benchmarked
   bucket-loop baseline before attempting further bucketed scan rewrites.

## March 27 Specialized Nearfield Follow-Up

To narrow the remaining large-`N` runtime bottleneck further, a new local
specialization was added for the exact hot path identified above.

Files:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)

What changed:

- added a dedicated accel-only prepared-leaf nearfield kernel for the
  large-`N` bucketed direct-scatter path
- the large-`N` runtime now dispatches to that specialized kernel only when all
  of the following are true:
  - `return_potential == False`
  - nearfield mode is `bucketed`
  - explicit prepared leaf groups are available
  - scatter schedules are not precomputed

Why this path:

- this is the minimum-memory large-`N` evaluate hot path called out in the
  runtime audit above
- it removes generic nearfield branching that is irrelevant for the hot path
  while preserving the already-validated arithmetic and direct-scatter design

What it intentionally does not change:

- no grouped per-chunk scatter rebuilds
- no source tiling
- no new retained metadata
- no changes to potential-returning paths or generic nearfield entrypoints

Current validation:

- `python3 -m py_compile` passed for the edited nearfield/runtime files
- focused GPU `9` integration coverage passed with:

```bash
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
CUDA_VISIBLE_DEVICES=9 \
JAX_ENABLE_X64=1 \
micromamba run -n odisseo python -m pytest -q -o addopts='' \
  /export/home/tbuck/jaccpot/tests/integration/test_fmm.py \
  -k nearfield_bucketed_matches_baseline
```

Observed result:

- `1 passed, 54 deselected`

Status:

- correctness is checked at the public bucketed nearfield level
- a focused runtime A/B benchmark for this specialization is still the next
  needed step

## March 27 GPU 9 A/B Result For The Specialized Nearfield Path

The new specialized accel-only prepared-leaf bucketed kernel was compared
against the generic large-`N` nearfield path on GPU `9` using
`examples/benchmark_gpu_radix_worker.py` in `audit` mode with:

- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `memory_objective="minimum_memory"`
- `nearfield_mode="bucketed"`
- fixed traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- worker autotuning disabled
- isolated cache files per run

### `N = 524288`, `nearfield_edge_chunk_size = 512`

Generic path:

- `evaluate_total_seconds ~= 1.7255 s`
- `evaluate_large_n_nearfield_seconds ~= 1.7543 s`

Specialized path:

- `evaluate_total_seconds ~= 1.6555 s`
- `evaluate_large_n_nearfield_seconds ~= 1.6747 s`

Interpretation:

- nearfield improved by about `4.5%`
- total evaluate improved by about `4.1%`
- this is a real but modest win, which supports keeping the specialization

### `N = 1048576`

Two fixed-config GPU `9` audit attempts were tried:

1. `nearfield_edge_chunk_size = 512`
2. `nearfield_edge_chunk_size = 256`

In both cases:

- the generic path failed with `RESOURCE_EXHAUSTED`
- the specialized path also failed with `RESOURCE_EXHAUSTED`
- the failing allocation reported by JAX was about `2.00 GiB`

Interpretation:

- this specialization improves runtime at `524288`
- it does **not** by itself solve the current 1M fit problem for this audit
  configuration on GPU `9`
- the next speed work should stay focused on the nearfield evaluate kernel, but
  1M audit comparisons on GPU `9` may require a leaner benchmark shape than the
  one used here

## March 30 Runtime Checkpoint

Work resumed on March 30, 2026 with the goal of isolating the remaining
nearfield runtime bottleneck on currently available GPUs.

### 1. Worker audit path was made memory-safe for 1M large-`N` checks

File:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

What changed:

- the worker `audit` path now skips standalone M2L/L2L stage reconstruction for
  `LargeNPreparedState`
- it still reports:
  - prepare-time split
  - total evaluate time
  - explicit `evaluate_large_n_nearfield(...)` time
  - explicit `evaluate_large_n_farfield(...)` time
- the audit row marks this lighter path with
  `audit_stage_breakdown_mode="large_n_light"`

Why:

- the old audit path forced `need_node_interactions=True` and rebuilt
  standalone downward-stage buffers
- at `N = 1048576`, that extra benchmark-only reconstruction caused repeated
  `RESOURCE_EXHAUSTED` failures even when the normal `steady_eval` path fit

Interpretation:

- the new audit shape is now much closer to the real large-`N`
  minimum-memory runtime path
- this is the safer way to measure nearfield vs farfield at `1M`

### 2. Updated `1M` chunk result on currently available GPUs

Focused `steady_eval` worker checks with:

- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `memory_objective="minimum_memory"`
- `nearfield_mode="bucketed"`
- fixed traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- worker autotuning disabled
- isolated cache files per run

showed that `nearfield_edge_chunk_size=512` beats `256` at
`N = 1048576` on both currently used comparison GPUs:

- GPU `2`:
  - `256`: `evaluate_mean_seconds ~= 2.2934 s`
  - `512`: `evaluate_mean_seconds ~= 1.7611 s`
- GPU `3`:
  - `256`: `evaluate_mean_seconds ~= 1.7737 s`
  - `512`: `evaluate_mean_seconds ~= 1.2165 s`

Interpretation:

- in the current lean `1M` traversal regime, `512` is the better fixed
  nearfield chunk on both tested GPUs
- future runtime A/B checks should treat `512` as the current best-known
  fixed chunk unless a later kernel change changes the tradeoff

### 3. Patched `1M` audit confirmed farfield is negligible

Using the lighter `audit` path above on GPU `4` with
`nearfield_edge_chunk_size=512` gave:

- `prepare_total_seconds ~= 1.7321 s`
- `evaluate_total_seconds ~= 1.2878 s`
- `evaluate_large_n_nearfield_seconds ~= 1.5606 s`
- `evaluate_large_n_farfield_seconds ~= 0.0171 s`

Interpretation:

- the remaining runtime bottleneck is still overwhelmingly nearfield
- farfield is effectively negligible in this `1M` minimum-memory configuration
- the fact that isolated nearfield exceeds total evaluate is expected here
  because the audit measures the nearfield kernel standalone rather than as a
  perfectly additive decomposition of the whole evaluate call

### 4. GPU `1` A/B for the specialized large-`N` nearfield path

The patched light-audit path was then used on GPU `1` to compare the current
specialized accel-only nearfield path against the generic bucketed path under
the same fixed `1M` config and `nearfield_edge_chunk_size=512`.

Specialized path enabled:

- `evaluate_total_seconds ~= 1.1902 s`
- `evaluate_large_n_nearfield_seconds ~= 1.1723 s`
- `evaluate_large_n_farfield_seconds ~= 0.0215 s`

Specialized path disabled via
`JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD=1`:

- `evaluate_total_seconds ~= 1.2318 s`
- `evaluate_large_n_nearfield_seconds ~= 1.4456 s`
- `evaluate_large_n_farfield_seconds ~= 0.0332 s`

Interpretation:

- the specialized path is a real win and should be kept
- the win is still modest relative to the total remaining nearfield cost
- the specialization reduces runtime, but it does **not** eliminate the main
  bottleneck

### Main conclusion from March 30

The best current interpretation is:

1. Keep the specialized accel-only large-`N` nearfield path.
2. Keep using `nearfield_edge_chunk_size=512` as the current best fixed `1M`
   chunk in this lean traversal regime.
3. Continue runtime optimization inside
   [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py),
   especially:
   - `_pair_contributions_batched(...)`
   - `_scatter_contributions(...)`
   - the specialized bucketed scan body in
     `_compute_leaf_p2p_prepared_large_n_accel_only_impl(...)`
   - duplicated gather/scatter structure between the generic and specialized
     bucketed paths

### New instrumentation hook added locally

To answer the next runtime question more directly, the local worktree now also
contains a benchmark-only split for the specialized large-`N` nearfield path:

- the specialized nearfield kernel can now expose:
  - self-leaf contribution time
  - cross-leaf pair-bucket contribution time
- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  now has a `nearfield_components` mode for this measurement path

Why this matters:

- the next decision is no longer just “generic vs specialized”
- we now want to know whether the remaining time is dominated more by:
  - self-leaf work
  - pair-kernel arithmetic
  - bucketed direct-scatter / output update cost

Recommended next step:

1. Run the new `nearfield_components` worker mode at `N = 1048576` on an
   available GPU with the current best fixed config.
2. Use that split to decide whether the next kernel work should target:
   - self-leaf interactions
   - `_pair_contributions_batched(...)`
   - or the bucketed direct-scatter update path.

## March 30 GPU 8 Nearfield Component Follow-Up

The new `nearfield_components` worker mode was then run on GPU `8` at
`N = 1048576` using the same fixed lean runtime config and
`nearfield_edge_chunk_size = 512`.

### Specialized nearfield split result

Measured standalone timings:

- total specialized nearfield:
  `evaluate_large_n_nearfield_seconds ~= 1.2542 s`
- specialized self-leaf component:
  `nearfield_specialized_self_seconds ~= 1.2280 s`
- specialized full pair-path component:
  `nearfield_specialized_pairs_seconds ~= 1.4447 s`

Interpretation:

- the pair-path component is at least as important as self-leaf work
- the next runtime work should prioritize the pair-bucket path over self-leaf
  changes

### Pair arithmetic probe result

To distinguish pair-kernel arithmetic from pair scatter/update cost, a
benchmark-only arithmetic probe was added for the specialized pair path and
measured on the same GPU `8` config.

Measured standalone timings:

- total specialized nearfield:
  `evaluate_large_n_nearfield_seconds ~= 1.2528 s`
- specialized self-leaf component:
  `nearfield_specialized_self_seconds ~= 1.2840 s`
- specialized full pair-path component:
  `nearfield_specialized_pairs_seconds ~= 1.2868 s`
- specialized pair arithmetic probe:
  `nearfield_specialized_pair_arith_probe_seconds ~= 1.0019 s`

Interpretation:

- pair arithmetic is a large part of pair-path cost, but not all of it
- the gap between full pair-path time and arithmetic-only time is a strong sign
  that gather/scatter/update overhead is still materially present
- a rough reading of this run suggests that around one-fifth of the full
  pair-path standalone time may be outside the arithmetic core

### Rejected bucket fast-path experiment

A temporary specialized pair-loop experiment tried to add a “fully valid
bucket” fast path so the code could skip some per-bucket masking work when all
edges in the chunk were valid.

Result on GPU `8`:

- the change regressed both total nearfield time and the pair component
- it was reverted locally after measurement

Interpretation:

- extra bucket-level control flow is not a promising JAX optimization in this
  hot path
- the next work should stay focused on reducing temporary traffic and
  gather/scatter/update overhead rather than adding more `lax.cond` structure

### Updated next step

The highest-signal next kernel work is now:

1. keep the current specialized path and current fixed `512` chunk baseline
2. optimize the pair-bucket path in
   [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
   with emphasis on:
   - `_pair_contributions_batched(...)`
   - `_scatter_contributions(...)`
   - gather/update traffic around
     `_compute_leaf_p2p_prepared_large_n_pairs_only_impl(...)`
3. avoid adding extra bucket-level branching unless a future measurement gives
   a strong reason to revisit that direction

## March 31 GPU 8 Pair-Path Follow-Up

Work resumed on March 31, 2026 with a same-session GPU `8` A/B focused on the
specialized large-`N` pair-bucket path at:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- `leaf_size = 256`
- fixed lean traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- worker autotuning disabled
- isolated cache files per run

### 1. Target-leaf bucket reduction helped

Local code change:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

What changed:

- inside the specialized pair-bucket path, per-edge pair rows are first reduced
  across contiguous runs of the same target leaf before scattering back to
  particle order

Measured result on GPU `8`:

- reduced-bucket variant:
  - `evaluate_large_n_nearfield_seconds ~= 2.0955 s`
  - `nearfield_specialized_pairs_seconds ~= 1.9842 s`
  - `nearfield_specialized_pair_arith_probe_seconds ~= 1.6870 s`
- restored baseline:
  - `evaluate_large_n_nearfield_seconds ~= 2.2307 s`
  - `nearfield_specialized_pairs_seconds ~= 2.2231 s`
  - `nearfield_specialized_pair_arith_probe_seconds ~= 1.6928 s`

Interpretation:

- total specialized nearfield improved by about `6%`
- the pair-path component improved by about `11%`
- the arithmetic probe barely moved
- this is strong evidence that the change helped gather/scatter/update overhead
  rather than core pair arithmetic

Status:

- this target-leaf bucket-reduction change is worth keeping as the current
  local baseline

### 2. Component-wise pair contraction regressed badly

One follow-up experiment rewrote `_pair_contributions_batched(...)` to replace
the current `weighted[..., None] * diff` contraction with explicit
component-wise sums.

Measured result on the same GPU `8` config:

- `evaluate_large_n_nearfield_seconds ~= 3.5594 s`
- `nearfield_specialized_pairs_seconds ~= 3.6183 s`
- `nearfield_specialized_pair_arith_probe_seconds ~= 3.3838 s`

Interpretation:

- the arithmetic rewrite was a clear regression
- explicit component-wise contraction is not a good direction for this JAX/GPU
  hot path

Status:

- that arithmetic experiment was reverted locally after measurement

### Updated next step from March 31

The next kernel work should stay focused on gather/scatter structure around the
specialized pair loop, especially:

1. reusing already-gathered target metadata inside each bucket
2. reducing repeated target-id / target-mask gathers after bucket reduction
3. avoiding new bucket-level control flow unless a future measurement gives a
   strong reason to revisit it

## March 31 GPU 1 Code-Path Audit Follow-Up

Today’s slow `~7-8 s` GPU `1` runs turned out to expose a real production-path
issue, not just noisy benchmark conditions.

### Root cause found

The specialized accel-only large-`N` nearfield kernel was present and the
benchmark-only standalone nearfield helper could call it, but the main compiled
full evaluation path for `LargeNPreparedState` was not using it.

Specifically:

- `evaluate_prepared_state(...)` for `LargeNPreparedState` routes through
  [`_evaluate_tree_compiled_impl(...)`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- that compiled path was still calling the generic
  `compute_leaf_p2p_accelerations(...)` nearfield path
- so the optimized specialized kernel existed, but full production evaluation
  was bypassing it

This also explains why moving benchmark-only helper code out of
[`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
was not the real regression source.

### Important subtlety

The first attempt to patch the compiled path was too strict because the real
minimum-memory `large_n_gpu` prepared state intentionally does **not** retain
explicit pair vectors.

Observed prepared-state shape on the real path:

- `nearfield_leaf_particle_indices.shape == (4, 16)` in the small repro
- `nearfield_target_leaf_ids is None`
- `nearfield_source_leaf_ids is None`
- `nearfield_valid_pairs is None`
- `nearfield_chunk_sort_indices is None`

So the correct compiled-path fix must allow the specialized kernel to run even
when leaf-pair vectors are absent and need to be re-derived on demand.

### Fix applied locally

[`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
now dispatches `_evaluate_tree_compiled_impl(...)` to
`compute_leaf_p2p_accelerations_large_n_accel_only(...)` when all of the
following are true:

- `return_potential == False`
- `nearfield_mode == "bucketed"`
- explicit leaf particle groups are present
- no precomputed scatter schedules are present
- `JACCPOT_DISABLE_LARGE_N_SPECIALIZED_NEARFIELD` is not set

and it correctly supports both cases:

- retained pair vectors present
- minimum-memory large-`N` path with pair vectors absent

### Regression coverage added

A new focused regression test was added in
[`tests/unit/test_solver_api.py`](/export/home/tbuck/jaccpot/tests/unit/test_solver_api.py)
to verify that compiled large-`N` evaluation actually calls the specialized
nearfield function.

Validated locally with:

- `python3 -m py_compile jaccpot/runtime/_fmm_impl.py tests/unit/test_solver_api.py`
- `PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot JAX_ENABLE_X64=1 micromamba run -n odisseo python -m pytest -q -o addopts='' tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield`

Result:

- `1 passed`

### Next step for tomorrow

Now that the compiled production path is wired to the specialized nearfield
kernel again, the first thing to rerun is a **single-process** GPU benchmark on
GPU `1` using the fixed March 30 config:

- `audit`
- `nearfield_components`

with:

- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `memory_objective="minimum_memory"`
- `nearfield_mode="bucketed"`
- `nearfield_edge_chunk_size=512`
- `max_pair_queue=524288`
- `process_block=256`
- `max_interactions_per_node=16384`
- `max_neighbors_per_leaf=8192`
- worker autotuning disabled

Do not run the two heavy benchmark modes in parallel on the same GPU again,
since that contaminates the comparison.

## March 31 GPU 8 Scatter Diagnostics Follow-Up

After the GPU `1` compiled-path fix above, work continued with a sequence of
focused `nearfield_components` checks on GPU `8` using the same fixed
large-`N` runtime shape:

- `N = 1048576`
- `leaf_size = 256`
- `nearfield_edge_chunk_size = 512`
- fixed traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- worker autotuning disabled
- isolated cache files per run

The goal was to stop guessing and identify whether the remaining specialized
pair-path cost comes primarily from:

- pair arithmetic
- target/source gather traffic
- target-leaf bucket reduction
- particle-index conversion after reduction
- the final particle-order scatter/update

### Current benchmark-only instrumentation in the worker

[`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
now exposes the following specialized pair-path probes:

- `nearfield_specialized_pair_arith_probe_seconds`
- `nearfield_specialized_pair_reduction_probe_seconds`
- `nearfield_specialized_pair_particle_index_probe_seconds`
- `nearfield_specialized_pair_scatter_probe_seconds`
- `nearfield_specialized_pair_lax_scatter_probe_seconds`
- `nearfield_specialized_pair_gather_probe_seconds`

These probes are benchmark-only diagnostics. They should not be treated as
production-path changes by themselves.

### What was tried and what happened

#### 1. Reusing already-gathered target ids/masks after bucket reduction

Idea:

- avoid the second `leaf_particle_idx[...]` / `leaf_mask[...]` gather after
  `_reduce_pair_bucket_by_target_leaf(...)`

GPU `8` result:

- `evaluate_large_n_nearfield_seconds ~= 2.8088 s`
- `nearfield_specialized_pairs_seconds ~= 2.9278 s`

Interpretation:

- this was clearly worse than the kept reduced-bucket baseline
- that experiment was reverted locally

#### 2. Rebuilding a tiny local scatter schedule after bucket reduction

Idea:

- keep the reduced-bucket path but replace the direct particle scatter with a
  small per-bucket schedule built from the reduced particle ids

GPU `8` result:

- `evaluate_large_n_nearfield_seconds ~= 3.9775 s`
- `nearfield_specialized_pairs_seconds ~= 3.9042 s`

Interpretation:

- this was a major regression
- local schedule rebuilding inside the reduced-bucket loop is not a promising
  direction here
- that experiment was reverted locally

#### 3. Benchmark-only `lax.scatter_add` update path

Idea:

- compare the current particle-order update primitive
  (`.at[flat_indices].add(...)`) against a benchmark-only `jax.lax.scatter_add`
  version on the same reduced-bucket output

GPU `8` result:

- `nearfield_specialized_pair_scatter_probe_seconds ~= 2.0102 s`
- `nearfield_specialized_pair_lax_scatter_probe_seconds ~= 2.1777 s`

Interpretation:

- `lax.scatter_add` is slower than the current update primitive here
- simply swapping the scatter primitive is not the answer

### Most useful GPU 8 component split from this session

A representative final split on the current kept reduced-bucket baseline gave:

- `evaluate_large_n_nearfield_seconds ~= 2.0156 s`
- `nearfield_specialized_pairs_seconds ~= 2.0333 s`
- `nearfield_specialized_pair_arith_probe_seconds ~= 1.6623 s`
- `nearfield_specialized_pair_reduction_probe_seconds ~= 1.8523 s`
- `nearfield_specialized_pair_particle_index_probe_seconds ~= 1.7490 s`
- `nearfield_specialized_pair_scatter_probe_seconds ~= 2.0662 s`
- `nearfield_specialized_pair_gather_probe_seconds ~= 0.2085 s`

Interpretation:

- gather/setup by itself is small
- converting reduced leaf-local outputs to particle ids/masks is not the main
  remaining issue
- arithmetic still matters, but the full reduced-bucket particle-order update
  path remains closest to total pair-path cost
- the dominant remaining bottleneck is still the final scatter/update side of
  the specialized pair loop, not raw gather traffic

### Best current code baseline

Keep the following local state as the working baseline:

- target-local edge ordering for bucketed runs without precomputed scatter
- specialized accel-only large-`N` nearfield path enabled in both:
  - [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
  - [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
- target-leaf bucket reduction inside the specialized pair loop in
  [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

Do **not** keep the following experimental ideas:

- component-wise pair contraction in `_pair_contributions_batched(...)`
- reuse of already-gathered target ids/masks after bucket reduction
- local per-bucket reduced scatter schedule rebuilding
- benchmark-only `lax.scatter_add` as a presumed better replacement primitive

### Best next step for the next session

The next session should start from the current kept reduced-bucket baseline and
focus on **reducing the amount of data that reaches the final particle-order
scatter**, not on changing arithmetic or gather shape again.

Most promising next experiment:

1. Add a benchmark-only **second-stage compaction** after
   `_reduce_pair_bucket_by_target_leaf(...)` that removes invalid reduced rows
   before particle-index conversion and before the final scatter/update.
2. Re-run `nearfield_components` on GPU `8` with the same fixed `1M` config.
3. Compare:
   - current scatter probe
   - compacted scatter probe
   - full specialized pair-path time

Why this is the best next step:

- the current data says the expensive part is update volume in the final
  reduced-bucket scatter path
- the current experiments already suggest that changing the primitive or adding
  more scheduling is not enough by itself
- compaction is the clearest remaining way to test whether fewer rows into the
  same update path materially helps

## April 13 Validation Workflow Cleanup

The current local validation workflow should now be treated as:

- always run through `micromamba run -n odisseo`
- always enable `JAX_ENABLE_X64=1` for local `yggdrax`-backed checks
- prefer automatic GPU selection through Python `autocvd`, not a shell binary
- keep the sibling `yggdrax` checkout on `PYTHONPATH` for local focused tests

### What was cleaned up

Local workflow updates:

- [`tests/conftest.py`](/export/home/tbuck/jaccpot/tests/conftest.py) now adds
  the sibling local checkout at `/export/home/tbuck/yggdrax` to `sys.path`
  when it exists, so focused pytest runs no longer depend on a manual
  `PYTHONPATH=...` export.
- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  now defaults `JAX_ENABLE_X64=1` before JAX import and infers
  `JACCPOT_NVIDIA_SMI_GPU_INDEX` from `CUDA_VISIBLE_DEVICES` when it is not
  already set.
- [`examples/run_in_odisseo_with_autocvd.py`](/export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py)
  was added as the canonical repo-local runner for focused validation commands.

Environment findings confirmed on April 13, 2026:

- `autocvd` is importable inside the `odisseo` env at
  `/export/home/tbuck/micromamba/envs/odisseo/lib/python3.13/site-packages/autocvd/__init__.py`
- the earlier focused test import failure was not an `autocvd` issue; it was a
  local module-resolution issue
- the import recipe
  `PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot JAX_ENABLE_X64=1 micromamba run -n odisseo python -c "from yggdrax import build_tree; import jaccpot.runtime._fmm_impl as m; print('ok')"`
  now resolves cleanly

### Canonical focused test commands

Use the helper runner for focused local checks:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python -m pytest -q -o addopts='' \
  tests/unit/core/test_near_field.py \
  -k 'large_n_accel_only_prepared_bucketed_matches_generic or collect_neighbor_pairs_matches_neighbor_list'
```

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python -m pytest -q -o addopts='' \
  tests/integration/test_fmm.py -k adaptive_nearfield_edge_chunk_size_auto_policy \
  tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield
```

For worker-side runtime checks:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode audit \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 3 \
  --warmup 1 \
  --dtype float32 \
  --config-json '{"preset":"large_n_gpu","basis":"solidfmm","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"runtime_traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192}}'
```

Documentation hygiene note:

- when future runtime checks are re-run on a different physical GPU, record the
  exact date, whether `autocvd` or manual pinning was used, and the resolved
  `CUDA_VISIBLE_DEVICES` / `JACCPOT_NVIDIA_SMI_GPU_INDEX` values directly in
  this note instead of relying on relative references like "today" or "the
  current GPU"

### April 13 focused validation results

Using the new helper runner on April 13, 2026:

- `autocvd` selected physical GPU `8`
- resolved environment:
  - `CUDA_VISIBLE_DEVICES=8`
  - `JACCPOT_NVIDIA_SMI_GPU_INDEX=8`
  - `JAX_ENABLE_X64=1`

Focused checks:

- `tests/unit/core/test_near_field.py -k 'large_n_accel_only_prepared_bucketed_matches_generic or collect_neighbor_pairs_matches_neighbor_list'`
  - result: `2 passed, 4 deselected`
- `tests/integration/test_fmm.py -k adaptive_nearfield_edge_chunk_size_auto_policy`
  - result: `1 passed, 54 deselected`
- `tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield`
  - result: `1 passed, 84 deselected`

Interpretation:

- the local `odisseo` + `autocvd` + sibling-`yggdrax` workflow is now
  reproducible
- the focused specialized-nearfield regression slice is green again without a
  manual `PYTHONPATH=...` export

## April 13 Nearfield Runtime Checkpoint

With the cleaned local workflow in place, the next focused nearfield runtime
check was re-run on April 13, 2026 through:

- `micromamba run -n odisseo`
- the repo-local helper
  [`examples/run_in_odisseo_with_autocvd.py`](/export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py)
- automatic GPU selection via Python `autocvd`

Resolved environment for both runs:

- `CUDA_VISIBLE_DEVICES=8`
- `JACCPOT_NVIDIA_SMI_GPU_INDEX=8`
- `JAX_ENABLE_X64=1`

Common fixed runtime config:

- `preset="large_n_gpu"`
- `basis="solidfmm"`
- `tree_type="radix"`
- `leaf_target=256`
- `theta=0.6`
- `softening=0.001`
- `working_dtype="float32"`
- `memory_objective="minimum_memory"`
- `farfield_mode="pair_grouped"`
- `nearfield_mode="bucketed"`
- `nearfield_edge_chunk_size=512`
- `streamed_far_pairs=true`
- `grouped_interactions=false`
- `enable_interaction_cache=false`
- `retain_traversal_result=false`
- `retain_interactions=false`
- worker autotuning disabled:
  - `worker_autotune_traversal=false`
  - `worker_autotune_nearfield_chunk=false`
- fixed traversal:
  - `max_pair_queue=524288`
  - `process_block=256`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`

### `nearfield_components` result at `N = 1048576`

Measured with:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode nearfield_components \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 3 \
  --warmup 1 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_components_autotune_cache.json \
  --config-json '{...fixed config above...}'
```

Key results:

- `prepared_state_mb ~= 58.32`
- `specialized_path_active = true`
- `evaluate_large_n_nearfield_seconds ~= 1.3645 s`
- `nearfield_specialized_self_seconds ~= 0.0711 s`
- `nearfield_specialized_pairs_seconds ~= 1.3417 s`
- `nearfield_specialized_pair_arith_probe_seconds ~= 1.1057 s`
- `nearfield_specialized_pair_reduction_probe_seconds ~= 1.1317 s`
- `nearfield_specialized_pair_particle_index_probe_seconds ~= 1.1453 s`
- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.3528 s`
- `nearfield_specialized_pair_lax_scatter_probe_seconds ~= 1.3162 s`
- `nearfield_specialized_pair_gather_probe_seconds ~= 0.0970 s`

Interpretation:

- the specialized large-`N` nearfield path is active on the current kept
  baseline
- self-leaf work is small relative to the pair path
- gather/setup remains a minor cost
- arithmetic and reduction matter, but the final particle-order scatter/update
  path is still the closest thing to the full pair-path cost
- `lax.scatter_add` as a benchmark-only replacement still does not obviously
  beat the current kept path enough to change the direction

### Matching `audit` result at `N = 1048576`

Measured with the same fixed config and:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode audit \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 3 \
  --warmup 1 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_audit_autotune_cache.json \
  --config-json '{...fixed config above...}'
```

Key results:

- `prepared_state_mb ~= 58.32`
- `prepare_total_seconds ~= 0.7383 s`
- `evaluate_total_seconds ~= 1.3181 s`
- `evaluate_large_n_nearfield_seconds ~= 1.3455 s`
- `evaluate_large_n_farfield_seconds ~= 0.0115 s`
- `resolved_large_n_memory_path_active = true`
- `resolved_nearfield_mode = "bucketed"`
- `resolved_nearfield_edge_chunk_size = 512`

Interpretation:

- the full evaluate path is still overwhelmingly nearfield-dominated at this
  `1M` fixed configuration
- farfield time is negligible compared with nearfield time here
- the component timings and the end-to-end audit tell the same story: the next
  nearfield experiment should focus on reducing the volume or shape of the
  final scatter/update work, not on gather or farfield changes

### Best next experiment from this checkpoint

Keep the current specialized reduced-bucket baseline and test a benchmark-only
compaction step that removes invalid reduced rows before particle-index
conversion and before the final scatter/update.

Reason:

- the April 13 rerun again shows gather is small
- the pair scatter/update side remains closest to total pair-path cost
- this is still the clearest remaining experiment that might reduce update
  volume without adding retained large-`N` metadata

## April 13 Compaction Experiment Follow-Up

The benchmark-only second-stage compaction experiment was then added locally.

What changed:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  now has `_compact_reduced_pair_bucket_rows(...)`, which stably packs valid
  reduced rows to the front of the fixed-size buffers
- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  now reports two extra specialized-path probes:
  - `nearfield_specialized_pair_compacted_scatter_probe_seconds`
  - `nearfield_specialized_pair_compacted_seconds`
- [`tests/unit/core/test_near_field.py`](/export/home/tbuck/jaccpot/tests/unit/core/test_near_field.py)
  now covers the row-compaction helper

Focused validation:

- `tests/unit/core/test_near_field.py -k 'compact_reduced_pair_bucket_rows_packs_valid_prefix or large_n_accel_only_prepared_bucketed_matches_generic'`
  - result: `2 passed, 5 deselected`

### `nearfield_components` result with compaction probes

Same fixed April 13 runtime shape:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- worker autotuning disabled
- physical GPU `8` selected by `autocvd`

Key results:

- `evaluate_large_n_nearfield_seconds ~= 1.3614 s`
- `nearfield_specialized_pairs_seconds ~= 1.3402 s`
- baseline scatter probe:
  - `nearfield_specialized_pair_scatter_probe_seconds ~= 1.3258 s`
- compacted scatter probe:
  - `nearfield_specialized_pair_compacted_scatter_probe_seconds ~= 1.3228 s`
- compacted full pair-path probe:
  - `nearfield_specialized_pair_compacted_seconds ~= 1.3634 s`
- `nearfield_specialized_pair_particle_index_probe_seconds ~= 1.1348 s`
- `nearfield_specialized_pair_gather_probe_seconds ~= 0.0930 s`

Interpretation:

- the compacted scatter-only probe is only marginally better than the baseline
  scatter probe at this `1M` fixed shape
- once the compaction step is included in the full specialized pair path, the
  result is not better than the current kept reduced-bucket baseline
- this means simple second-stage row compaction is **not** a strong enough win
  to justify changing the current baseline

Updated recommendation:

- keep the current specialized reduced-bucket baseline
- do **not** promote the second-stage compaction experiment into the production
  path
- future nearfield work should focus on changing the final update structure
  more fundamentally, rather than just packing the same rows more tightly

## April 13 Leaf-Accumulation Structure Experiment

To test whether the current bottleneck is fundamentally about repeated
particle-order updates, a benchmark-only leaf-major accumulation path was added
to:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

What this experiment does:

1. Run the same specialized pair arithmetic and reduced-bucket logic.
2. Accumulate reduced pair outputs into a temporary
   `(num_leaves, max_leaf_size, 3)` leaf-major buffer with `lax.scatter_add`.
3. Do a single final leaf-to-particle scatter pass at the end.

This is intentionally benchmark-only and does **not** change the production
nearfield path.

### `nearfield_components` result with leaf-major accumulation

Same fixed benchmark shape:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- worker autotuning disabled
- physical GPU `8` selected by `autocvd`

Key results from this run:

- `evaluate_large_n_nearfield_seconds ~= 1.8000 s`
- `nearfield_specialized_pairs_seconds ~= 1.7722 s`
- baseline scatter probe:
  - `nearfield_specialized_pair_scatter_probe_seconds ~= 1.7700 s`
- `lax.scatter_add` particle-order probe:
  - `nearfield_specialized_pair_lax_scatter_probe_seconds ~= 1.7678 s`
- compacted scatter probe:
  - `nearfield_specialized_pair_compacted_scatter_probe_seconds ~= 1.8068 s`
- leaf-major accumulation probe:
  - `nearfield_specialized_pair_leaf_accum_seconds ~= 1.8378 s`

Interpretation:

- absolute times in this run were slower than the earlier April 13 checkpoint,
  so the session likely had additional device-level noise or load
- however, the **within-run comparison** is still informative:
  - the leaf-major accumulation path is slower than the current kept
    specialized pair baseline
  - it is also slower than the direct particle-order scatter probe in the same
    session

Conclusion:

- simply moving the intermediate accumulation to leaf-major storage and then
  scattering once at the end is **not** enough to beat the current baseline in
  this form
- that means the next structural nearfield experiment should probably target
  the arithmetic/update fusion boundary more aggressively, rather than just
  inserting another scatter-add stage at leaf granularity

## April 13 Delayed-Scatter Superchunk Experiment

The next benchmark-only structure change tested whether we can reduce the cost
of the global particle-order update without changing the arithmetic itself.

What changed in the worker:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  now includes a delayed-scatter probe that:
  1. computes the same reduced per-chunk pair buckets as the current baseline
  2. batches `4` consecutive reduced chunks into one superchunk
  3. performs a single particle-order scatter for the whole superchunk instead
     of one scatter per chunk

This remains benchmark-only and does **not** change the production path.

### `nearfield_components` result with delayed scatter

Environment for this run:

- date: April 13, 2026
- `autocvd` selected physical GPU `9`
- resolved environment:
  - `CUDA_VISIBLE_DEVICES=9`
  - `JACCPOT_NVIDIA_SMI_GPU_INDEX=9`
  - `JAX_ENABLE_X64=1`

Common fixed runtime config stayed the same:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- worker autotuning disabled

Key results:

- `evaluate_large_n_nearfield_seconds ~= 2.6852 s`
- `nearfield_specialized_pairs_seconds ~= 2.6306 s`
- baseline scatter probe:
  - `nearfield_specialized_pair_scatter_probe_seconds ~= 2.6585 s`
- `lax.scatter_add` particle-order probe:
  - `nearfield_specialized_pair_lax_scatter_probe_seconds ~= 2.6543 s`
- compacted scatter probe:
  - `nearfield_specialized_pair_compacted_scatter_probe_seconds ~= 2.6706 s`
- leaf-major accumulation probe:
  - `nearfield_specialized_pair_leaf_accum_seconds ~= 2.7247 s`
- delayed-scatter superchunk probe:
  - `nearfield_specialized_pair_delayed_scatter_seconds ~= 2.1404 s`

Interpretation:

- absolute wall times remain sensitive to which physical GPU was selected, so
  they should not be compared directly to earlier GPU `8` runs
- however, the **within-run comparison** on GPU `9` is strong:
  - delaying the global scatter across `4` reduced chunks beats the current
    one-scatter-per-chunk baseline by about `0.52 s`
  - this is a materially larger effect than any gain seen from compaction or
    leaf-major staging

Conclusion:

- reducing the **frequency** of the global particle-order scatter looks more
  promising than changing the scatter primitive or inserting an intermediate
  leaf-major accumulation buffer
- the next step should stay in this direction:
  - verify the delayed-scatter signal on another run or GPU
  - tune the superchunk size
  - if the gain holds, consider promoting a cleaned-up version of this idea
    into the specialized production path

## April 13 Delayed-Scatter Superchunk Sweep

The delayed-scatter idea was then re-run on the **same fixed runtime shape**
while sweeping the superchunk size to compare `2`, `4`, and `8` directly.

Environment for the sweep:

- date: April 13, 2026
- `autocvd` selected physical GPU `5` for all three runs
- resolved environment:
  - `CUDA_VISIBLE_DEVICES=5`
  - `JACCPOT_NVIDIA_SMI_GPU_INDEX=5`
  - `JAX_ENABLE_X64=1`

Common fixed config stayed the same:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- worker autotuning disabled

### Sweep results

#### Superchunk size `2`

- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.3152 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.1466 s`
- improvement vs baseline scatter probe: about `0.169 s`

#### Superchunk size `4`

- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.3136 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.2693 s`
- improvement vs baseline scatter probe: about `0.044 s`

#### Superchunk size `8`

- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.2965 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.2549 s`
- improvement vs baseline scatter probe: about `0.042 s`

### Sweep interpretation

- the delayed-scatter idea still helps at all three tested superchunk sizes
- on this GPU `5` sweep, `2` is clearly the best of the tested values
- increasing the superchunk size beyond `2` gives back most of the gain
- this suggests there is a sweet spot:
  - enough batching to reduce global scatter frequency
  - not so much batching that the larger flattened superchunk update becomes
    expensive itself

### Updated recommendation

- keep `delayed_scatter_chunks_per_superchunk = 2` as the best current
  benchmark-only delayed-scatter setting for this `1M` fixed shape
- the next concrete step should be a **production-path prototype** of the
  delayed-scatter specialized pair loop using the same basic structure with
  superchunk size `2`

## April 13 Production Delayed-Scatter Prototype

After the benchmark-only sweep, the delayed-scatter idea was prototyped in the
specialized production nearfield path in:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

Implementation note:

- the production specialized large-`N` pair loop can now read
  `JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS`
- this remains **opt-in** for now
- default behavior stays at `1`, which preserves the original one-scatter-per-
  chunk baseline until the prototype shows a clean end-to-end win

### Focused correctness checks

Validated locally with:

- `tests/unit/core/test_near_field.py -k large_n_accel_only_prepared_bucketed_matches_generic`
  - result: `1 passed, 6 deselected`
- `tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield`
  - result: `1 passed, 90 deselected`

### Production-path audit checks

Same fixed runtime config:

- `N = 1048576`
- `nearfield_edge_chunk_size = 512`
- worker autotuning disabled

#### Opt-in delayed scatter on GPU `5`

With delayed scatter enabled and `autocvd` selecting GPU `5`:

- `evaluate_total_seconds ~= 1.1188 s`
- `evaluate_large_n_nearfield_seconds ~= 1.1624 s`

This looked promising, but it was not directly comparable to the baseline
because the follow-up baseline run landed on a different physical GPU.

#### Same-GPU A/B on GPU `9`

To get a clean comparison, both paths were then checked on physical GPU `9`.

Baseline with:

- `JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS=1`

Result:

- `evaluate_total_seconds ~= 1.1380 s`
- `evaluate_large_n_nearfield_seconds ~= 1.1707 s`

Prototype with:

- `JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS=2`
- manual pinning to `CUDA_VISIBLE_DEVICES=9`

Result:

- `evaluate_total_seconds ~= 1.1484 s`
- `evaluate_large_n_nearfield_seconds ~= 1.1865 s`

### Production prototype conclusion

- the benchmark-only delayed-scatter direction is still the most promising
  structural idea we have found so far
- however, the **same-GPU production-path A/B on GPU `9` did not beat the
  current baseline**
- so the delayed-scatter production prototype is **not ready to become the
  default path**

Current status:

- keep the production delayed-scatter code available for further tuning behind
  `JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS`
- leave the default at `1`
- do not claim an end-to-end runtime breakthrough yet

Practical takeaway:

- we have a benchmark-only signal that fewer global scatters can help
- but the current production implementation still gives back that gain through
  additional overhead elsewhere
- reaching well below `1 s` at this `1M` fixed shape will likely require
  either:
  - a more carefully tuned production delayed-scatter implementation
  - or a more GPU-specific fused kernel for the pair/update stage

## April 14 Packed Unique-Update Foundation

The next step was to start from the data layout rather than the scatter API.
The goal was to make the nearfield update stage look more like a fused GPU
kernel input:

- flatten superchunk updates
- sort and reduce them to **unique particle rows**
- then apply one update per particle row instead of repeated scatter-adds

This was implemented in:

- [`jaccpot/pallas/nearfield_unique_updates.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_unique_updates.py)
- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

The new helper module provides:

- `pack_unique_particle_vector_updates(...)`
- `apply_packed_particle_vector_updates(...)`
- a backend report for whether the packed update would use:
  - Triton/Pallas
  - or the JAX gather-add-set fallback

### Important hardware constraint

On the current `odisseo` GPU pool used by `autocvd`, JAX reports:

- device kind: `NVIDIA GeForce RTX 2080 Ti`
- compute capability: `7.5`

The installed Triton-backed Pallas GPU lowering in this environment only runs
on Ampere-class GPUs (`compute capability >= 8.0`), so the new packed update
path **cannot use the Triton/Pallas kernel on the current validation hardware**.

That means the current benchmark result is testing:

- the new packed unique-update data layout
- plus a pure-JAX gather-add-set fallback backend

It is **not** yet a measurement of the Triton/Pallas kernel on supported GPU
hardware.

### Focused regression check

Validated locally with:

- `tests/unit/operators/test_pallas_nearfield_unique_updates.py`
  - result: `2 passed`

The older focused nearfield regression slice is still blocked here by the
pre-existing local `yggdrax` import issue:

- `ImportError: cannot import name 'build_tree' from 'yggdrax'`

### `nearfield_components` result with packed unique updates

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `9`
- `JAX_ENABLE_X64=1`
- the same fixed `N = 1048576` large-`N` config
- `delayed_scatter_chunks_per_superchunk = 2`

Key results:

- `nearfield_specialized_pair_packed_unique_updates_supported = false`
- `nearfield_specialized_pair_packed_unique_updates_backend = "jax_set"`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.9815 s`
- `nearfield_specialized_pair_packed_unique_scatter_seconds ~= 3.6446 s`
- `nearfield_specialized_pair_scatter_probe_seconds ~= 2.1643 s`

### Interpretation

- the **layout idea** is still useful:
  - unique particle rows are the right kind of structure for a fused update
    kernel
- but on the current GPU fleet, the path falls back to pure JAX and is much
  slower than both:
  - the delayed-scatter benchmark path
  - the original scatter probe
- so this result should be read as:
  - data-layout groundwork completed
  - Triton/Pallas kernel blocked by hardware support
  - no performance win yet on current validation devices

### Updated recommendation

- keep the packed unique-update helpers as **infrastructure**
- do not promote the packed unique-update fallback path as a runtime
  optimization on the current GPUs
- if we want to continue down the GPU-specific kernel route, the next real
  checkpoint needs one of:
  - access to Ampere-or-newer GPUs so the Triton/Pallas kernel can actually be
    benchmarked
  - or a different fused-kernel route that does not depend on Triton support
    for `compute capability 7.5`

## April 14 Grounded `jaxFMM`-Style Nearfield Check

After inspecting the public `jaxFMM` package locally, one concrete structural
difference stood out in its nearfield path:

- the direct nearfield work accumulates into a **target-box-major padded
  buffer**
- the direct connectivity is sorted by target box
- the box-major result is flattened back only at the end

That led to a more grounded benchmark in `jaccpot`:

- keep the existing leaf-major accumulation buffer
- explicitly sort the prepared nearfield edges by target leaf
- use the sorted/unique leaf update hint in the per-chunk leaf scatter

This benchmark-only probe was added in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `nearfield_specialized_pair_target_sorted_leaf_accum_seconds`

### Measurement setup

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- the same fixed `N = 1048576` large-`N` config
- `nearfield_edge_chunk_size = 512`
- `delayed_scatter_chunks_per_superchunk = 2`

### Key results

- `nearfield_specialized_pairs_seconds ~= 1.6697 s`
- `nearfield_specialized_pair_leaf_accum_seconds ~= 2.1298 s`
- `nearfield_specialized_pair_target_sorted_leaf_accum_seconds ~= 2.2204 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.7486 s`
- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.9956 s`

### Interpretation

- this is an important negative result:
  - simply copying the **target-sorted ownership idea** at the current
    leaf-major level is **not enough**
- compared within the same run:
  - target-sorted leaf-major accumulation was slower than the older leaf-accum
    probe
  - both were slower than the delayed-scatter path
- so the `jaxFMM` gap is not explained by ordering alone

Practical takeaway:

- the helpful part of the `jaxFMM` comparison is still real:
  - target-owned accumulation is a plausible direction
- but in `jaccpot`, we likely need a **deeper representation change** to see
  the benefit:
  - flatter target-owned tiles
  - less masked leaf padding
  - and ideally a tighter fused compute/writeback kernel

### Updated recommendation

- do not spend more time on target sorting by itself
- keep using the `jaxFMM` comparison as motivation for ownership/fusion ideas
- but treat the next meaningful step as a **representation rewrite**, not an
  ordering tweak

## April 14 Target-Owned Particle-Tile Prototype

To push the representation change one level deeper, a second benchmark-only
prototype was added after the target-sorted leaf-major check:

- flatten reduced per-leaf particle updates to exact particle rows
- reduce them to unique particle rows inside each chunk
- map those rows into a fixed `(num_tiles, tile_size, 3)` accumulator
- flatten the tile buffer back to particle order only once at the end

This was implemented in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `nearfield_specialized_pair_target_sorted_particle_tile_accum_seconds`

### Measurement setup

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- the same fixed `N = 1048576` large-`N` config
- `nearfield_edge_chunk_size = 512`
- `delayed_scatter_chunks_per_superchunk = 2`
- `target_tile_size = 32`

### Key results

- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.9673 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.7700 s`
- `nearfield_specialized_pair_leaf_accum_seconds ~= 2.1372 s`
- `nearfield_specialized_pair_target_sorted_leaf_accum_seconds ~= 2.1957 s`
- `nearfield_specialized_pair_target_sorted_particle_tile_accum_seconds ~= 3.7243 s`

### Interpretation

- this is another important negative result:
  - a flatter target-owned particle-tile accumulator is still **much slower**
    in its current JAX expression
- the added flatten/reduce/tile-index work costs more than it saves
- so the remaining gap is not solved by:
  - target sorting alone
  - leaf-major ownership alone
  - or a simple particle-tile buffer layered on top of the current chunked
    pair path

### Updated recommendation

- do not promote the current particle-tile accumulator idea further in this
  form
- the next serious nearfield optimization likely needs an actual **fused tile
  microkernel** shape, not just a different accumulation buffer
- in practice that means the next meaningful candidate is something like:
  - one program owns one target tile
  - loops over source tiles directly
  - accumulates in tile-local registers/shared memory
  - writes the target tile once

## April 14 Leaf-Tile Microkernel Prototype

To test the first actual **compute-shape** change, a benchmark-only tiled
microkernel analogue was added next:

- split each padded leaf into fixed-size target/source tiles
- compute tile-vs-tile interactions directly
- sum all source-tile contributions into the target tile locally
- reduce repeated target-leaf rows once per chunk
- scatter the final per-leaf tile buffer only at the end

This benchmark lives in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds`

### Measurement setup

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- the same fixed `N = 1048576` large-`N` config
- `nearfield_edge_chunk_size = 512`
- `target_tile_size = 32`
- `delayed_scatter_chunks_per_superchunk = 2`

### Key results

- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.9799 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.7771 s`
- `nearfield_specialized_pair_leaf_accum_seconds ~= 2.1184 s`
- `nearfield_specialized_pair_target_sorted_particle_tile_accum_seconds ~= 3.7318 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds ~= 2.4974 s`

### Interpretation

- this is the first tiled experiment that changed the **compute shape** as well
  as the accumulation buffer
- it is still slower than delayed scatter
- but it is much better than the earlier particle-tile buffer-only attempt
  (`~2.50 s` vs `~3.73 s`)

That suggests:

- compute shape really does matter
- but the current pure-JAX expression of the tiled microkernel still has too
  much overhead to win

### Updated recommendation

- keep this result as evidence that the remaining opportunity is probably in a
  **real fused tile kernel**, not in more scatter rearrangements
- do not promote the current JAX tiled microkernel path
- if we continue, the next version should try to reduce the extra overhead in
  one of two ways:
  - fuse the target-tile loop more tightly into one custom/Pallas-style kernel
    on supported hardware
  - or simplify the tile implementation so it avoids the large broadcasted
    batch reshapes used in this first prototype

## April 14 Fused Tile-Pair Kernel Groundwork

Following the tiled microkernel experiments, a real fused tile-pair kernel
primitive was added as a separate Pallas module:

- [`jaccpot/pallas/nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_tile_pair.py)

This module provides:

- `nearfield_tile_pair_accel_jax(...)`
  - a pure-JAX reference implementation for one target-tile x source-tile
    interaction
- `nearfield_tile_pair_accel_pallas(...)`
  - a Triton-backed Pallas kernel intended to compute the same fused tile pair
    update without going through the generic scatter-heavy path
- support detection via
  `pallas_nearfield_tile_pair_supported()`

The export surface was updated in:

- [`jaccpot/pallas/__init__.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/__init__.py)

and a focused regression test was added in:

- [`tests/unit/operators/test_pallas_nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/tests/unit/operators/test_pallas_nearfield_tile_pair.py)

### Focused validation

Validated locally with:

- `python3 -m py_compile`
  - result: passed for the new Pallas module and test
- `tests/unit/operators/test_pallas_nearfield_tile_pair.py`
  - result: `2 passed`

### Important hardware limitation

On the current `odisseo` GPU pool used by `autocvd`:

- device kind: `NVIDIA GeForce RTX 2080 Ti`
- compute capability: `7.5`

The installed Triton-backed Pallas GPU lowering still requires Ampere-class
GPUs (`compute capability >= 8.0`), so the new fused tile-pair kernel cannot
yet be benchmarked on the current validation hardware.

That means the current state is:

- fused kernel API and support plumbing implemented
- JAX reference path implemented and regression-tested
- actual Triton/Pallas fused-kernel benchmark still blocked by available GPUs

### Updated recommendation

- keep the fused tile-pair primitive as the current best technical direction
- once Ampere-or-newer GPUs are available, benchmark this kernel directly and
  then decide whether to integrate it into the large-`N` nearfield benchmark
  path

## April 14 Integrated Fused-Primitive Benchmark Path

After adding the fused tile-pair primitive, it was wired into the
`nearfield_components` worker as a benchmark-only large-`N` nearfield path.

This integration lives in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds`

The worker now also reports:

- `nearfield_specialized_pair_tile_primitive_supported`
- `nearfield_specialized_pair_tile_primitive_backend`

### Measurement setup

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- the same fixed `N = 1048576` large-`N` config
- `nearfield_edge_chunk_size = 512`
- `target_tile_size = 32`
- `delayed_scatter_chunks_per_superchunk = 2`

### Key results

- `nearfield_specialized_pair_tile_primitive_supported = false`
- `nearfield_specialized_pair_tile_primitive_backend = "jax"`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 2.4855 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds ~= 5.9072 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds ~= 7.1059 s`

### Interpretation

- the integration itself works
- but on the current RTX 2080 Ti pool, the fused primitive still falls back to
  the JAX backend
- in that fallback mode, the path is substantially slower than the delayed-
  scatter baseline

So this result should be read narrowly:

- the fused-primitive benchmark path is now ready
- the current hardware still cannot tell us whether the actual Triton/Pallas
  fused kernel is a win
- the JAX fallback is not a competitive replacement for the current nearfield
  path

### Updated recommendation

- keep the integrated fused-primitive benchmark path in place
- do not treat the current fallback measurement as evidence against the fused
  kernel idea itself
- the next meaningful checkpoint for this line is now clearly:
  - run the same worker path on Ampere-or-newer GPUs where
    `nearfield_specialized_pair_tile_primitive_supported = true`

## April 14 A100 Handoff

### Current status

The fused-kernel investigation is now at a clean checkpoint:

- the fused tile-pair primitive exists in
  [`jaccpot/pallas/nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_tile_pair.py)
- the benchmark-only large-`N` integration exists in
  [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
- the focused regression test exists in
  [`tests/unit/operators/test_pallas_nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/tests/unit/operators/test_pallas_nearfield_tile_pair.py)
- the current `odisseo` GPU pool only exposes RTX 2080 Ti devices
  (`compute capability 7.5`), so Triton-backed Pallas fused kernels still do
  not run there

What that means in practice:

- all current fused-kernel benchmark numbers are **JAX fallback** numbers
- those fallback numbers are not competitive with the delayed-scatter baseline
- the actual question is still unanswered:
  - does the fused tile-pair kernel win on Ampere-or-newer hardware?

### Minimal validation before the A100 run

These local checks already passed on April 14, 2026:

- `python3 -m py_compile jaccpot/pallas/nearfield_tile_pair.py jaccpot/pallas/__init__.py tests/unit/operators/test_pallas_nearfield_tile_pair.py`
- `micromamba run -n odisseo python -m pytest tests/unit/operators/test_pallas_nearfield_tile_pair.py -q`
  - result: `2 passed`

### Canonical A100 benchmark command

Run this on the A100 machine with the same repo state:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode nearfield_components \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 3 \
  --warmup 1 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_components_autotune_cache.json \
  --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"delayed_scatter_chunks_per_superchunk":2,"target_tile_size":32}'
```

### Fields to check first

The A100 run is only informative for the fused-kernel question if these fields
look right:

- `nearfield_specialized_pair_tile_primitive_supported`
  - should be `true`
- `nearfield_specialized_pair_tile_primitive_backend`
  - should be `"pallas"`

If either of those is still:

- `false`
- or `"jax"`

then the fused kernel still did not actually execute.

### Main comparison to record

On the A100 run, compare at least these fields:

- `nearfield_specialized_pair_delayed_scatter_seconds`
- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds`
- `nearfield_specialized_pairs_seconds`
- `evaluate_large_n_nearfield_seconds`

### Success criterion

The fused-kernel line becomes genuinely promising if the A100 run shows both:

- `nearfield_specialized_pair_tile_primitive_backend = "pallas"`
- a meaningful improvement of
  `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds`
  versus the delayed-scatter and JAX tiled baselines in the same run

### If the A100 run is promising

The next step after a positive A100 result should be:

- keep the benchmark-only fused path
- repeat once for confirmation on the same GPU class
- then decide whether to promote a cleaned-up version into the specialized
  production nearfield path

## April 14 Code-Structure Findings

Before committing to another nearfield redesign, the current large-`N` code path
was re-checked against the actual tree/runtime implementation.

### What the current large-`N` preset is doing

The current `large_n_gpu` preset does **not** use a fixed-depth Morton tree by
default. It uses:

- `tree_build_mode = "lbvh"` in
  [`jaccpot/runtime/fmm_presets.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/fmm_presets.py)
- the radix/LBVH tree build path in
  [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)

The important practical consequences are:

- particles are still Morton-ordered
- leaf particle ranges are still contiguous in that reordered storage
- leaves are bounded by `leaf_size`
- but leaf occupancy is **not** guaranteed to be exactly `leaf_size`

### What `yggdrax` actually guarantees

`yggdrax` was checked as well to make sure the earlier assumptions were not too
strong.

The fixed-depth path in:

- [`../yggdrax/yggdrax/_tree_impl.py`](/export/home/tbuck/yggdrax/yggdrax/_tree_impl.py)

does correspond to a common Morton depth, but it still does **not** imply
uniform particle occupancy per leaf. Occupancy still depends on the particle
distribution after sorting and partitioning.

So the accurate statement is:

- the nearfield already has Morton-local contiguous particle blocks
- the large-`N` path already has explicit leaf particle groups
- but the hot path still runs on padded leaf blocks with masks rather than on a
  truly target-owned dense compute loop

### What the current specialized nearfield is still doing

The current specialized large-`N` nearfield path in:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)

still has this overall shape:

- gather padded target/source leaf particle blocks
- evaluate dense leaf-pair interactions
- reduce repeated target-leaf rows inside a chunk
- update particle-order output through the current writeback path

That is why the bottleneck diagnosis has remained consistent:

- Morton ordering is already present
- the remaining problem is the compute/writeback structure
- especially repeated target handling and global update pressure

## April 14 Code Restructure Summary

By this point the nearfield investigation has already restructured the codebase
in a few important ways, even where the measured speedups were not strong enough
to become defaults.

### Runtime and validation workflow

- added
  [`examples/run_in_odisseo_with_autocvd.py`](/export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py)
  as the canonical `micromamba run -n odisseo` launcher with `autocvd`,
  `JAX_ENABLE_X64=1`, `PYTHONPATH` setup, and GPU index propagation
- updated
  [`tests/conftest.py`](/export/home/tbuck/jaccpot/tests/conftest.py)
  so local pytest picks up the sibling `yggdrax` checkout automatically

### Specialized nearfield benchmarking and instrumentation

- expanded
  [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  into the main experiment harness for:
  - scatter-path probes
  - compaction probes
  - leaf-accumulation variants
  - delayed-scatter variants
  - packed unique-update probes
  - target-sorted and tiled microkernel probes
  - fused-primitive probes

### New Pallas groundwork

- added
  [`jaccpot/pallas/nearfield_unique_updates.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_unique_updates.py)
  as packed unique-update groundwork
- added
  [`jaccpot/pallas/nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_tile_pair.py)
  as the fused tile-pair primitive module
- exported both through
  [`jaccpot/pallas/__init__.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/__init__.py)
- added focused regression tests in:
  - [`tests/unit/operators/test_pallas_nearfield_unique_updates.py`](/export/home/tbuck/jaccpot/tests/unit/operators/test_pallas_nearfield_unique_updates.py)
  - [`tests/unit/operators/test_pallas_nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/tests/unit/operators/test_pallas_nearfield_tile_pair.py)

### Current pure-JAX redesign work

A new benchmark-only target-leaf-owned pure-JAX path is now also implemented in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `_prepare_target_leaf_owned_nearfield_inputs(...)`
- `_compute_pair_target_leaf_owned_impl(...)`
- `nearfield_specialized_pair_target_leaf_owned_seconds`

This path is the first one built directly around target ownership from
`prepared_state.neighbor_list.offsets` instead of the older chunk-owned
reduction/writeback structure.

## April 14 Experiment Ledger

### Strongest insights so far

- the large-`N` nearfield bottleneck is still dominated by the update/writeback
  structure, not by lack of Morton locality
- delaying the global scatter can help in benchmark-only form, which is strong
  evidence that target ownership matters
- compute shape matters more than buffer reshaping alone
- a real fused kernel is still the clearest remaining high-upside direction,
  but current RTX 2080 Ti hardware cannot execute the Triton/Pallas path

### Experiments that did not become the answer

The following lines were explored and did not produce a compelling win on the
current validation hardware:

- second-stage compaction after target-leaf row reduction
- leaf-major accumulation with one final leaf-to-particle expansion
- target sorting alone
- particle-tile accumulation layered on top of the current pair path
- the first pure-JAX tiled microkernel variants
- packed unique-update fallback on RTX 2080 Ti
- fused tile-pair benchmark integration while still falling back to `"jax"`

### Best current practical baseline

On the current RTX 2080 Ti pool, the best practical benchmark result still comes
from the delayed-scatter line rather than from the newer tiled or fused fallback
variants.

That means:

- we have learned a lot about what does **not** help enough
- the repository is now much better structured for nearfield experiments
- but we still do not yet have the sub-`1 s` breakthrough at
  `N = 1048576`

### Current pending experiment

The newest experiment is the benchmark-only target-leaf-owned pure-JAX path
described above.

Its intended purpose is:

- keep one target-leaf accumulator resident
- scan over that target leaf's source neighbors
- avoid repeated particle-order update pressure inside the hot loop
- stay pure JAX so it can also benefit current RTX 2080 hardware

This path is implemented, but as of April 14, 2026 there is **not yet** a clean
timing result recorded for it.

The first run attempt was intentionally not treated as trustworthy because no
free GPU was available at the time, so the timing would have been distorted by
contention.

### Current recommendation

The near-term plan is now split cleanly:

- on RTX 2080-class hardware:
  - benchmark the new target-leaf-owned pure-JAX path once a genuinely free GPU
    is available
- on A100/Ampere-class hardware:
  - run the integrated fused-primitive benchmark path and verify that the
    backend flips from `"jax"` to `"pallas"`

This gives two meaningful next checkpoints:

- whether target-leaf ownership alone can materially help in pure JAX
- whether the real fused tile-pair kernel wins once suitable hardware is
  available

## April 14 Target-Leaf-Owned Pure-JAX Benchmark

Once a genuinely free GPU became available again, the new benchmark-only
target-leaf-owned pure-JAX path was rerun under the standard local workflow:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- `N = 1048576`
- `leaf_size = 256`
- `nearfield_edge_chunk_size = 512`
- `delayed_scatter_chunks_per_superchunk = 2`
- `target_tile_size = 32`

### Key results

From the clean `nearfield_components` run:

- `evaluate_large_n_nearfield_seconds ~= 1.6838 s`
- `nearfield_specialized_pairs_seconds ~= 1.6674 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.7338 s`
- `nearfield_specialized_pair_target_leaf_owned_seconds ~= 133.9086 s`
- `nearfield_specialized_pair_scatter_probe_seconds ~= 1.9892 s`
- `nearfield_specialized_pair_particle_index_probe_seconds ~= 1.7494 s`
- `nearfield_specialized_pair_gather_probe_seconds ~= 0.1155 s`

For the same run, the other benchmark-only comparison points remained in the
same rough range as before:

- `nearfield_specialized_pair_leaf_accum_seconds ~= 2.1335 s`
- `nearfield_specialized_pair_target_sorted_leaf_accum_seconds ~= 2.2641 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds ~= 2.3613 s`
- `nearfield_specialized_pair_target_sorted_particle_tile_accum_seconds ~= 3.5882 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds ~= 3.5282 s`

### Interpretation

This is a strong negative result for the first target-leaf-owned pure-JAX
implementation.

The result means:

- target ownership by itself is **not** enough when expressed as the current
  per-target-leaf scan structure
- the present implementation is far too sequential / low-throughput for GPU use
- simply removing the old chunk-owned writeback shape does not automatically
  create an efficient pure-JAX kernel

So the current target-leaf-owned path should be treated as:

- a useful structural probe
- not a viable replacement for the current specialized nearfield

### Updated takeaway

After this run, the situation is clearer:

- the delayed-scatter line remains the best practical result on the RTX 2080 Ti
  pool
- the current target-leaf-owned pure-JAX redesign is not the answer in its
  present form
- the remaining promising routes are still:
  - a much tighter pure-JAX kernel formulation with substantially more parallel
    target processing
  - or the real fused Pallas path on Ampere-or-newer hardware

## April 14 Batched Target-Leaf Pure-JAX Follow-Up

To test whether the poor result above was mainly caused by the fully sequential
per-target-leaf scan, a second benchmark-only pure-JAX variant was added to:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `_compute_pair_target_leaf_batched_impl(...)`
- `nearfield_specialized_pair_target_leaf_batched_seconds`

This variant keeps the same target-owned idea but processes multiple target
leaves in parallel per step instead of one target leaf at a time.

### Measurement setup

Measured on April 14, 2026 with:

- `micromamba run -n odisseo`
- `examples/run_in_odisseo_with_autocvd.py --use-autocvd`
- `autocvd` selecting physical GPU `1`
- `JAX_ENABLE_X64=1`
- `N = 1048576`
- `leaf_size = 256`
- `nearfield_edge_chunk_size = 512`
- `delayed_scatter_chunks_per_superchunk = 2`
- `target_leaf_batch_size = 32`

### Key results

From the clean `nearfield_components` run:

- `nearfield_specialized_pair_delayed_scatter_seconds ~= 1.7522 s`
- `nearfield_specialized_pair_target_leaf_owned_seconds ~= 132.1072 s`
- `nearfield_specialized_pair_target_leaf_batched_seconds ~= 22.8289 s`

### Interpretation

This result is directionally useful but still clearly negative overall.

It shows:

- batching target leaves does recover a large amount of lost parallelism versus
  the fully sequential target-leaf-owned prototype
- but the current batched-target formulation is still far too slow to compete
  with the delayed-scatter baseline

So the updated lesson is:

- target ownership is still not enough by itself
- the remaining pure-JAX opportunity, if any, needs a much tighter batched
  kernel shape than the current implementation

## April 14 H100 `autocvd` Checkpoint

This session finally moved the fused nearfield work off the older RTX 2080
Ti-class pool and onto Hopper-class hardware available in the current machine.

### What was confirmed

- `nvidia-smi` on this machine shows four H100 GPUs.
- [`examples/run_in_odisseo_with_autocvd.py`](/export/home/tbuck/jaccpot/examples/run_in_odisseo_with_autocvd.py)
  is working correctly here and currently reports:
  - `num_gpus=1 -> [3]`
  - `num_gpus=2 -> [0, 3]`
  - `num_gpus=3 -> [0, 2, 3]`
- the fused tile-pair primitive in
  [`jaccpot/pallas/nearfield_tile_pair.py`](/export/home/tbuck/jaccpot/jaccpot/pallas/nearfield_tile_pair.py)
  now lowers and executes on H100 after removing unsupported Pallas/Triton
  patterns from the kernel body:
  - the earlier `slice` lowering failure was removed
  - a follow-up concatenate/`stack` lowering failure was also removed
- a benchmark-only focused worker mode was added in
  [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py):
  - `--mode nearfield_fused_check`
  - this keeps the real large-`N` prepared-state path but times only the small
    set of fused-kernel-relevant probes instead of the full historical
    `nearfield_components` suite

### First meaningful H100 result

Using the focused worker mode on H100, the fused path is no longer falling back
to JAX:

- `nearfield_specialized_pair_tile_primitive_supported = true`
- `nearfield_specialized_pair_tile_primitive_backend = "pallas"`

However, the first free-GPU `autocvd` run still showed the fused path far
slower than the existing scatter-based baselines:

- `nearfield_specialized_pair_scatter_probe_seconds ~= 3.2572 s`
- `nearfield_specialized_pair_delayed_scatter_seconds ~= 3.0248 s`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds ~= 19.5294 s`

Interpretation:

- this is the first real answer to the April 14 A100/H100 handoff question
- the fused tile-pair line is now executing through real Pallas/Triton
- but the current kernel/integration shape is **not yet competitive**

### Worker config issues found during this check

Two worker-side issues were uncovered while trying to reproduce the fixed
`1M / chunk=512 / fixed traversal` benchmark shape from the note above.

#### 1. Documented traversal key mismatch

The benchmark note used:

- `runtime_traversal_config`

but the worker config builder only read:

- `traversal_config`

This is now fixed locally in
[`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
by accepting `runtime_traversal_config` as an alias.

#### 2. Runtime autotune cache still overrode “fixed” runs

Even with:

- `worker_autotune_traversal=false`
- `worker_autotune_nearfield_chunk=false`

the worker still loaded a sibling runtime cache entry from:

- `/tmp/runtime_worker_autotune_cache.json`

when `--autotune-cache /tmp/...` was provided.

That meant a stale cached nearfield chunk selection could still force:

- `worker_nearfield_edge_chunk_size = 128`

and therefore:

- `resolved_nearfield_edge_chunk_size = 128`

even when the command explicitly requested `nearfield_edge_chunk_size = 512`.

This is now fixed locally in
[`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
so that when both worker autotune toggles are `false`, the worker ignores the
runtime autotune cache entirely.

### Current paused state

At the time this note was updated:

- the focused fixed-shape `autocvd` rerun was launched again after the worker
  cache fix
- that rerun was intended to verify:
  - `resolved_nearfield_edge_chunk_size = 512`
  - `worker_nearfield_edge_chunk_size = null`
    or otherwise absent as an override
  - `nearfield_specialized_pair_tile_primitive_backend = "pallas"`
- the result of that last rerun was not yet recorded in this note

### Best next step

Resume from the fixed-shape `autocvd` rerun, not from the older accidental
`chunk=128` H100 runs.

The first fields to check from that rerun are:

- `resolved_nearfield_edge_chunk_size`
- `worker_nearfield_edge_chunk_size`
- `nearfield_specialized_pair_tile_primitive_backend`

If the rerun now really stays at fixed `chunk=512`, compare:

- `nearfield_specialized_pair_scatter_probe_seconds`
- `nearfield_specialized_pair_delayed_scatter_seconds`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds`
- `nearfield_specialized_pairs_seconds`
- `evaluate_large_n_nearfield_seconds`

Decision rule for the next session:

- if fused Pallas is still much slower than delayed scatter at fixed `512`,
  stop treating the current fused kernel as an integration candidate and focus
  the next work on kernel redesign / launch shape rather than production-path
  plumbing
- if fused Pallas closes most of the gap at fixed `512`, run one same-shape
  confirmation pass through `autocvd` before considering any larger code-path
  promotion

## April 14 Focused Benchmarking Workflow And Pending Runs

The full `nearfield_components` worker is now too expensive for routine
iteration because it still carries many probes that are already known to be
non-competitive.

To avoid paying that cost on every rerun, the worker was tightened so that the
focused mode:

- `--mode nearfield_fused_check`

now returns early after timing only the highest-signal paths:

- the specialized nearfield pair baseline
- delayed scatter
- the tiled microkernel benchmark
- the fused-primitive benchmark path
- the batched target-leaf pure-JAX path
- the newest bucketed batched target-leaf pure-JAX path

This lets later reruns compare the current best options without dragging along
the clearly losing compaction / accumulation / sequential target-leaf probes.

### Newest pure-JAX prototype

The newest benchmark-only path currently lives in:

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)

under:

- `_compute_pair_target_leaf_bucketed_batched_impl(...)`
- `nearfield_specialized_pair_target_leaf_bucketed_batched_seconds`

Its intended shape is:

- batch multiple target leaves together
- process their neighbors in dense fixed-size neighbor blocks
- flatten `(target_batch, neighbor_block)` work into one large batched pair
  kernel call
- reduce over the neighbor block in JAX before the final writeback

This is the first pure-JAX path in the current line that was explicitly written
to expose a more vectorized XLA workload rather than nested target-owned scans.

### Current status

The bucketed-batched path is implemented and `py_compile` passes, but there is
**not yet** a clean timing result recorded for it.

The first attempt to rerun the new focused mode did not complete with a usable
benchmark row before GPU availability disappeared again, so the result should
not be treated as informative.

### Next experiments to run

When a genuinely free GPU is available again on the current RTX 2080 Ti pool,
the next local rerun should be:

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode nearfield_fused_check \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 3 \
  --warmup 1 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_components_autotune_cache.json \
  --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"delayed_scatter_chunks_per_superchunk":2,"target_tile_size":32,"target_leaf_batch_size":32,"target_leaf_neighbor_block_size":16}'
```

The first fields to compare from that run are:

- `nearfield_specialized_pairs_seconds`
- `nearfield_specialized_pair_delayed_scatter_seconds`
- `nearfield_specialized_pair_target_leaf_batched_seconds`
- `nearfield_specialized_pair_target_leaf_bucketed_batched_seconds`
- `nearfield_specialized_pair_target_sorted_leaf_tile_microkernel_seconds`
- `nearfield_specialized_pair_target_sorted_leaf_tile_fused_primitive_seconds`

### Decision rule for the next checkpoint

After that focused rerun:

- if `target_leaf_bucketed_batched` is still far from delayed scatter, stop
  spending local RTX 2080 time on target-owned pure-JAX restructuring
- if it closes the gap materially, tune `target_leaf_batch_size` and
  `target_leaf_neighbor_block_size`
- independently, still run the A100 fused-kernel handoff because the real
  Pallas path remains unresolved on the current hardware

## April 15 Production-Path Iteration Checkpoint

With GPUs intermittently unavailable on April 15, 2026, work focused on
production-path nearfield restructuring plus focused correctness checks, then
quick same-GPU single-run audits when GPU `0` was briefly free.

### 1. Specialized pair-loop baseline fast-path cleanup kept

A small cleanup was kept in the specialized large-`N` pair path:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  `_compute_leaf_p2p_prepared_large_n_pairs_only_impl(...)`

What changed:

- added a dedicated `chunks_per_superchunk == 1` path that avoids the
  superchunk `vmap` staging overhead and executes a direct chunk scan

Validation:

- focused nearfield correctness checks passed

### 2. Phase A target-owned production prototype (opt-in) added

Files:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`tests/unit/core/test_near_field.py`](/export/home/tbuck/jaccpot/tests/unit/core/test_near_field.py)

What was added:

- target-owned pair accumulation prototype:
  - `_compute_leaf_p2p_prepared_large_n_pairs_target_owned_impl(...)`
  - `_compute_leaf_p2p_prepared_large_n_accel_only_target_owned_impl(...)`
- env-flag dispatch in specialized entrypoint:
  - `JACCPOT_LARGE_N_TARGET_OWNED_ACCUM`
  - `JACCPOT_LARGE_N_TARGET_LEAF_BATCH_SIZE`
  - `JACCPOT_LARGE_N_TARGET_LEAF_NEIGHBOR_BLOCK_SIZE`
- focused regression:
  - `test_large_n_accel_only_target_owned_accum_matches_baseline`

Correctness status:

- focused nearfield tests passed
- compiled large-`N` specialized-dispatch regression test passed

Quick same-GPU audit A/B on GPU `0` (`N=1048576`, fixed config, `runs=1`):

- baseline (`TARGET_OWNED_ACCUM=0`):
  - `evaluate_total_seconds ~= 1.6908 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.4145 s`
- target-owned prototype (`TARGET_OWNED_ACCUM=1`, batch `32`, block `16`):
  - `evaluate_total_seconds ~= 5.8728 s`
  - `evaluate_large_n_nearfield_seconds ~= 8.1400 s`

Interpretation:

- this first production target-owned formulation is a clear regression
- keep it opt-in only for now; do not promote to default path

### 3. Sorted writeback hint experiment (opt-in) added

Files:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`tests/unit/core/test_near_field.py`](/export/home/tbuck/jaccpot/tests/unit/core/test_near_field.py)

What was added:

- sorted scatter helper:
  - `_scatter_contributions_sorted_hint(...)`
- specialized pair path static flag:
  - `sorted_scatter_hint`
- env toggle:
  - `JACCPOT_LARGE_N_SORTED_SCATTER_HINT`
- focused regression:
  - `test_large_n_accel_only_sorted_scatter_hint_matches_baseline`

Quick same-GPU audit A/B on GPU `0` (`N=1048576`, fixed config, `runs=1`):

- baseline:
  - `evaluate_total_seconds ~= 1.6864 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.5308 s`
- sorted-hint enabled (`SORTED_SCATTER_HINT=1`):
  - `evaluate_total_seconds ~= 1.7692 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.4260 s`

Interpretation:

- no clear end-to-end win from sorted-hint alone in this quick check

### 4. Grouped sorted writeback variant (opt-in) added

Files:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
- [`tests/unit/core/test_near_field.py`](/export/home/tbuck/jaccpot/tests/unit/core/test_near_field.py)

What was added:

- grouped sorted scatter helper:
  - `_scatter_contributions_grouped_sorted(...)`
- specialized pair path option:
  - `grouped_sorted_scatter`
- env toggle:
  - `JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER`
- focused regression:
  - `test_large_n_accel_only_grouped_sorted_scatter_matches_baseline`

Correctness status:

- focused nearfield tests passed (`5 passed, 6 deselected`)

Runtime status at pause:

- same-GPU timing for grouped sorted scatter could not be recorded before GPUs
  became fully occupied again

### Current practical state after April 15

1. The default specialized production nearfield path remains unchanged in
   behavior unless opt-in env flags are set.
2. The first target-owned production prototype is currently too slow.
3. Sorted writeback hint alone is not yet a compelling improvement.
4. Grouped sorted writeback variant is implemented and correctness-checked but
   still needs clean same-GPU timing once a free GPU is available.

### First command to run when a GPU frees up

Run same-shape grouped-sorted A/B on one pinned GPU (keep `runs=1` for fast
signal first):

```bash
# Baseline
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --cuda-visible-devices 0 -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode audit \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 1 \
  --warmup 0 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_audit_sortedhint_cache.json \
  --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'

# Grouped sorted scatter
JACCPOT_LARGE_N_TARGET_OWNED_ACCUM=0 \
JACCPOT_LARGE_N_SORTED_SCATTER_HINT=1 \
JACCPOT_LARGE_N_GROUPED_SORTED_SCATTER=1 \
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --cuda-visible-devices 0 -- \
  python examples/benchmark_gpu_radix_worker.py \
  --mode audit \
  --num-particles 1048576 \
  --leaf-size 256 \
  --max-order 4 \
  --runs 1 \
  --warmup 0 \
  --dtype float32 \
  --autotune-cache /tmp/jaccpot_nf_audit_sortedhint_cache.json \
  --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

If grouped sorted scatter looks promising in the quick run, repeat with
`--runs 3 --warmup 1` on the same pinned GPU before deciding whether to keep
or revert that line.

### April 15 grouped-sorted A/B results (completed)

The exact same pinned-GPU command pair above was run on GPU `0`.

Quick signal (`--runs 1 --warmup 0`):

- baseline:
  - `evaluate_total_seconds ~= 1.6604 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.3864 s`
- grouped sorted scatter:
  - `evaluate_total_seconds ~= 1.8621 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.8465 s`

Confirmation (`--runs 3 --warmup 1`):

- baseline:
  - `evaluate_total_seconds ~= 1.6867 s`
  - `evaluate_large_n_nearfield_seconds ~= 1.7224 s`
- grouped sorted scatter:
  - `evaluate_total_seconds ~= 1.9006 s`
  - `evaluate_large_n_nearfield_seconds ~= 1.8626 s`

Conclusion:

- grouped sorted scatter remains slower than baseline in the confirmed run
- keep it opt-in only; do not promote to the default specialized path

## Why `jaccpot` Is Still Slower Than `jaxFMM` In Nearfield

The key reason is now clearer from both code inspection and repeated runtime
probes:

1. `jaccpot` still pays too much cost in repeated global particle-order updates
   inside the nearfield pair loop.
2. `jaxFMM`'s direct path is structurally more target-owned:
   - nearfield pairs are target-box sorted
   - accumulation is box-major
   - writeback uses sorted target ownership (`indices_are_sorted=True`)
   - flatten/reindex happens after the main accumulation
3. In `jaccpot`, Morton ordering is already present and target-local edge order
   is already exploited, but the current specialized path still does:
   - gather padded target/source leaf blocks per chunk
   - run pair arithmetic
   - reduce within chunk
   - scatter repeatedly to particle-order output
4. Profiling repeatedly shows gather/setup is smaller than pair writeback
   pressure; farfield is negligible in the 1M minimum-memory shape.

So the dominant gap is not “missing Morton locality”; it is the remaining
compute/writeback structure in the nearfield pair path.

### What Must Change To Improve Performance

The next meaningful performance improvement likely requires one of the
following structural changes (not just scatter primitive swaps):

1. Move to a true target-owned accumulation path in production where global
   particle-order scatter frequency is substantially reduced.
2. Keep accumulation resident in target-local buffers longer, and emit fewer,
   larger, sorted writebacks.
3. Avoid adding extra per-bucket control-flow/scheduling overhead that gives
   back the writeback gains.
4. For larger upside, move toward a fused target-tile microkernel shape
   (Pallas/custom kernel on supported GPUs) where one program owns a target
   tile, loops over source tiles, and writes once.

### Current Practical Conclusion

- Small local changes (sorted hint, regrouped schedule rebuilds, simple
  compaction, lightweight ownership reshapes) are not enough by themselves.
- The path to sub-`1 s` at `N=1048576` most likely needs a deeper nearfield
  representation + writeback redesign, ideally fused for GPU execution.

## April 15 Cross-Repo Nearfield Interop Hardening

A compatibility-safe nearfield interop contract extension was added so target
nearfield source expansion can use precomputed neighbor leaf positions directly
instead of always rebuilding that mapping from `offsets/neighbors` and
`leaf_lookup`.

`jaccpot` changes:

- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
  - `NearfieldInteropData` now includes optional `neighbor_leaf_positions`
  - `_build_nearfield_interop_data(...)` now stores `neighbor_leaf_positions`
    for both:
    - native octree-neighbor interop
    - fallback radix-style interop
  - `_build_target_nearfield_source_index_matrix(...)` now consumes
    `nearfield_interop.neighbor_leaf_positions` when present and falls back to
    legacy reconstruction when absent
- [`tests/unit/test_solver_api.py`](/export/home/tbuck/jaccpot/tests/unit/test_solver_api.py)
  - extended the native nearfield-view assertion to require
    `neighbor_leaf_positions`

`yggdrax` changes:

- [`yggdrax/_interactions_impl.py`](/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py)
  - `OctreeNativeNeighborList` now carries `neighbor_leaf_positions`
  - `_raw_to_octree_native_neighbors(...)` now computes and populates this
    array in native leaf-position space

Validation:

- `python3 -m py_compile` passed for the edited interop files
- focused `yggdrax` native octree slice passed:
  - `3 passed, 10 deselected`
- focused `jaccpot` octree nearfield slice passed:
  - `3 passed, 81 deselected`

Practical outcome:

- this does not change default large-`N` radix production behavior
- it removes a repeated reconstruction dependency in target-subset nearfield
  paths and sets up a cleaner producer/consumer nearfield data contract for
  future deeper kernel work

## April 15 Target-Owned v2 Probe (opt-in only)

A second target-owned prototype path was added as an opt-in experiment in:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  - `JACCPOT_LARGE_N_TARGET_OWNED_ACCUM_V2`
  - `_compute_leaf_p2p_prepared_large_n_pairs_target_owned_v2_impl(...)`

Main structural difference:

- target flattening for a leaf batch is precomputed once per batch and reused
  across neighbor blocks to reduce repeated setup work inside the block loop

Correctness:

- focused nearfield target-owned regressions passed:
  - `2 passed, 10 deselected`

Pinned GPU `0` runtime checks (`N=1048576`, `nearfield_edge_chunk_size=512`,
`runs=1`, `warmup=0`) gave:

- baseline (default specialized path):
  - `evaluate_total_seconds ~= 1.6626 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.5101 s`
- target-owned v1 (`TARGET_OWNED_ACCUM=1`, `B=32`, `K=16`):
  - `evaluate_total_seconds ~= 5.8137 s`
  - `evaluate_large_n_nearfield_seconds ~= 7.6093 s`
- target-owned v2 (`TARGET_OWNED_ACCUM=1`, `TARGET_OWNED_ACCUM_V2=1`,
  `B=32`, `K=16`):
  - `evaluate_total_seconds ~= 5.8995 s`
  - `evaluate_large_n_nearfield_seconds ~= 7.8105 s`

Quick v2 tuning probes (still worse):

- `B=64`, `K=8`:
  - `evaluate_total_seconds ~= 7.8875 s`
  - `evaluate_large_n_nearfield_seconds ~= 9.6316 s`
- `B=16`, `K=32`:
  - `evaluate_total_seconds ~= 8.8331 s`
  - `evaluate_large_n_nearfield_seconds ~= 10.7962 s`

Conclusion:

- this v2 target-owned formulation is not competitive on the current GPU/path
- keep it opt-in experimental only
- do not promote target-owned v1 or v2 to default production nearfield

## April 15 Superchunk Target-Reduction Probe (opt-in)

To reduce repeated particle-order updates in the default specialized pair path,
an opt-in superchunk experiment was added in:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  - env toggle: `JACCPOT_LARGE_N_SUPERCHUNK_TARGET_REDUCE`
  - applies when delayed scatter superchunking is active
  - reduction is done in target-leaf space across the whole superchunk before
    the particle-order scatter

Focused correctness:

- new nearfield regression passed via autocvd:
  - `test_large_n_accel_only_superchunk_target_reduce_matches_baseline`
- focused slice result:
  - `5 passed, 8 deselected`

Quick runtime A/B via autocvd (`N=1048576`, GPU selected by autocvd, `runs=1`,
`warmup=0`, `nearfield_edge_chunk_size=512`, `JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS=2`):

- baseline (`SUPERCHUNK_TARGET_REDUCE=0`):
  - `evaluate_total_seconds ~= 1.7340 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.5461 s`
- variant (`SUPERCHUNK_TARGET_REDUCE=1`):
  - `evaluate_total_seconds ~= 1.8050 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.8761 s`

Conclusion:

- this superchunk target-reduction formulation is currently slower
- keep it opt-in for now; do not promote to default

## April 15 Vectorization-First Branch Elimination Probe

Given the concern that loop/control overhead can dominate on JAX GPU paths, an
opt-in branch-elimination variant was added in the default specialized pair
scan path:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  - env toggle: `JACCPOT_LARGE_N_DISABLE_CHUNK_COND`
  - when enabled, the per-chunk `lax.cond(jnp.any(...))` gate is removed in the
    `chunks_per_superchunk == 1` path
  - kernel still uses mask-driven vectorized math and scatter semantics

Focused correctness:

- added regression:
  - `test_large_n_accel_only_disable_chunk_cond_matches_baseline`
- focused nearfield slice via autocvd:
  - `4 passed, 10 deselected`

Quick runtime A/B via autocvd (`N=1048576`, `nearfield_edge_chunk_size=512`,
`JACCPOT_LARGE_N_DELAYED_SCATTER_CHUNKS=1`, `runs=1`, `warmup=0`):

- baseline (`DISABLE_CHUNK_COND=0`):
  - `evaluate_total_seconds ~= 2.0158 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.8566 s`
- variant (`DISABLE_CHUNK_COND=1`):
  - `evaluate_total_seconds ~= 1.6859 s`
  - `evaluate_large_n_nearfield_seconds ~= 3.4215 s`

Interpretation:

- this is the first recent specialized-path structural probe with a positive
  quick signal on the measured GPU/run shape
- needs confirmation with `runs=3`, `warmup=1` before promoting beyond opt-in

### Same-GPU higher-iteration confirmation

To remove cross-GPU variance, both sides were re-run pinned to physical GPU `1`
with stronger timing settings (`runs=7`, `warmup=2`):

- baseline (`DISABLE_CHUNK_COND=0`):
  - `evaluate_total_seconds ~= 1.9857 s`
  - `evaluate_total_std_seconds ~= 0.0145 s`
  - `evaluate_large_n_nearfield_seconds ~= 1.9960 s`
  - `evaluate_large_n_nearfield_std_seconds ~= 0.0221 s`
- branch-elimination variant (`DISABLE_CHUNK_COND=1`):
  - `evaluate_total_seconds ~= 1.6882 s`
  - `evaluate_total_std_seconds ~= 0.0211 s`
  - `evaluate_large_n_nearfield_seconds ~= 1.7383 s`
  - `evaluate_large_n_nearfield_std_seconds ~= 0.0467 s`

Conclusion:

- the speedup persists on the same GPU with higher-iteration timing
- this remains far from sub-`1 s` at `N=1048576`, but it is a meaningful
  positive step in the default specialized nearfield path

### Promotion status

The branch-elimination path is now default-on in
[`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
for the `chunks_per_superchunk == 1` specialized pair path.

- default: `JACCPOT_LARGE_N_DISABLE_CHUNK_COND=1`
- rollback escape hatch remains:
  - `JACCPOT_LARGE_N_DISABLE_CHUNK_COND=0`

Focused regressions remain green after this default flip.

## April 15 Cross-Repo TONB Implementation Checkpoint

This section records the first end-to-end implementation of the
target-owned-nearfield-blocks (TONB) producer/consumer path across `yggdrax`
and `jaccpot`.

Quick command reference for upcoming GPU A/B runs:

- [`docs/nearfield_tonb_runbook.md`](/export/home/tbuck/jaccpot/docs/nearfield_tonb_runbook.md)

### What is now implemented

`yggdrax` (producer-side):

- [`yggdrax/_interactions_impl.py`](/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py)
  now extends neighbor payloads with:
  - `neighbor_leaf_positions` (for `NodeNeighborList`, matching existing octree-native behavior)
  - `target_block_leaf_ids`
  - `target_block_source_leaf_ids`
  - `target_block_valid_mask`
  - `target_block_size`
- target-block generation is controlled by:
  - `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE` (default `0`, disabled)

`jaccpot` (consumer-side):

- [`jaccpot/runtime/_large_n_pipeline.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py)
  now prefers `neighbor_list.target_block_*` payloads when present and matching
  the requested size, and otherwise falls back to local derivation.
- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  includes a target-block specialized pair path used by
  `compute_leaf_p2p_accelerations_large_n_accel_only(...)`.
- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
  threads precomputed target-block payloads into the compiled large-`N`
  nearfield dispatch.

### Focused validation run in this session

Because no free GPUs were available during this checkpoint, correctness was
validated via compile checks and focused unit slices, including CPU-only runs
for JAX paths that would otherwise trigger transient GPU OOM under load.

Compile checks:

```bash
python3 -m py_compile \
  /export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py \
  /export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py \
  /export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py
```

Observed: passed.

`yggdrax` focused tests:

```bash
cd /export/home/tbuck/yggdrax
PYTHONPATH=/export/home/tbuck/yggdrax JAX_ENABLE_X64=1 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_tree_interactions.py -k neighbor

PYTHONPATH=/export/home/tbuck/yggdrax JAX_ENABLE_X64=1 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_octree_topology.py
```

Observed:

- `4 passed, 7 deselected`
- `13 passed`

`yggdrax` target-block toggle check:

```bash
cd /export/home/tbuck/yggdrax
PYTHONPATH=/export/home/tbuck/yggdrax JAX_ENABLE_X64=1 \
YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_tree_interactions.py -k neighbor
```

Observed:

- `4 passed, 7 deselected`

`jaccpot` focused tests (CPU backend for stability in this session):

```bash
cd /export/home/tbuck/jaccpot
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/core/test_near_field.py -k large_n

PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield
```

Observed:

- `8 passed, 6 deselected`
- `1 passed, 83 deselected`

Cross-repo producer+consumer toggle check:

```bash
cd /export/home/tbuck/jaccpot
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/core/test_near_field.py -k large_n

PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
  micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_solver_api.py -k large_n_compiled_eval_uses_specialized_nearfield
```

Observed:

- `8 passed, 6 deselected`
- `1 passed, 83 deselected`

### Current state summary

1. Cross-repo TONB plumbing is implemented and validated in focused unit slices.
2. Default behavior remains unchanged unless block-size toggles are set.
3. No new GPU runtime numbers were collected in this checkpoint due to lack of
   free GPUs.

### Recommended next run plan once a GPU is free

1. Keep current best fixed runtime shape (`N=1048576`, `nearfield_edge_chunk_size=512`,
   fixed lean traversal, autotuning disabled).
2. Run baseline (`TONB off`) then TONB (`TONB on`) on the same pinned GPU:
   - off:
     - `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0`
     - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0`
   - on:
     - `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32`
     - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`
3. Use both:
   - `steady_eval` (real runtime delta)
   - `nearfield_components` (where gains/losses occur)
4. If `32` helps or is close, sweep `target_block_size` in `{16, 32, 64}` on the
   same pinned GPU before deciding on default promotion.

## April 15 GPU-0 TONB A/B Results (post-implementation)

This updates the prior checkpoint now that a free GPU became available.

Pinned runtime env:

- `CUDA_VISIBLE_DEVICES=0`
- `JACCPOT_NVIDIA_SMI_GPU_INDEX=0`
- `PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot`
- `JAX_ENABLE_X64=1`

Note on config:

- `examples/benchmark_gpu_radix_worker.py` currently expects a fully specified
  `--config-json` (including `tree_type`, `leaf_target`, `theta`, `softening`,
  and `working_dtype`), not just preset-style partial JSON.

### 1M `steady_eval` A/B (`runs=3`, `warmup=1`)

Baseline (TONB off):

- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0`
- `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0`
- result:
  - `evaluate_mean_seconds=1.7018`
  - `evaluate_std_seconds=0.0042`
  - `prepare_mean_seconds=1.0402`

TONB-on:

- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32`
- `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`
- result:
  - `evaluate_mean_seconds=3.6538`
  - `evaluate_std_seconds=0.3529`
  - `prepare_mean_seconds=1.0561`

Delta (TONB-on vs baseline):

- `evaluate_mean_seconds`: `+1.9520 s` (~`2.15x` slower)

### 1M `nearfield_components` A/B (diagnostic `runs=1`, `warmup=0`)

Baseline (TONB off):

- `evaluate_large_n_nearfield_seconds=1.2679`
- `nearfield_specialized_pair_target_leaf_owned_seconds=137.1535`
- `prepared_state_mb=189.99`

TONB-on:

- `evaluate_large_n_nearfield_seconds=4.6651`
- `nearfield_specialized_pair_target_leaf_owned_seconds=145.4640`
- `prepared_state_mb=215.97`

Interpretation:

- TONB-on currently regresses the nearfield path and increases prepared-state
  footprint on this exact 1M GPU shape.
- no promotion recommendation yet; the right next move is targeted profiling in
  the target-owned pair path before block-size sweeps.

Runbook link:

- [`docs/nearfield_tonb_runbook.md`](/export/home/tbuck/jaccpot/docs/nearfield_tonb_runbook.md)

## April 15 Late Checkpoint (Pure-JAX TONB Iteration, Tests Deferred)

Date: Wednesday, April 15, 2026

Scope:

- continued pure-JAX TONB restructuring toward static-shape, leaf-major,
  vectorized execution
- deferred full GPU verification set to Thursday, April 16, 2026

### Code changes completed

1. TONB pair kernel restructured to static batched execution:

- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)
  - TONB pair path now uses leaf-major batched processing driven by
    precomputed per-leaf block spans (`block_offsets`)
  - added static tile control:
    - `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SIZE` (default `8`)
  - TONB dispatch now accepts:
    - `precomputed_target_block_offsets`

2. Large-N prepare pipeline now records TONB block offsets:

- [`jaccpot/runtime/_large_n_pipeline.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py)
  - computes/persists `nearfield_target_block_offsets`
  - normalizes target-block arrays to stable leaf-major order at prepare time

3. Large-N state and runtime dispatch threaded with new metadata:

- [`jaccpot/runtime/_large_n_types.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py)
  - `LargeNPreparedState` extended with `nearfield_target_block_offsets`
- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
  - `_evaluate_tree_compiled_impl(...)` now accepts
    `precomputed_target_block_offsets`
  - specialized large-N TONB dispatch forwards offsets into
    `compute_leaf_p2p_accelerations_large_n_accel_only(...)`
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
  - TONB helper path forwards `state.nearfield_target_block_offsets`

### Verification status at checkpoint

- `py_compile` passed for updated nearfield/runtime modules.
- focused CPU nearfield test rerun was attempted but not completed in this
  checkpoint due command-approval interruption.
- first resumed pinned-GPU benchmark launch was started but interrupted before
  result collection.

### Deferred to Thursday, April 16, 2026

1. Re-run focused CPU sanity:

```bash
PYTHONPATH=/export/home/tbuck/yggdrax:/export/home/tbuck/jaccpot \
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
micromamba run -n odisseo python -m pytest -q -o addopts='' \
tests/unit/core/test_near_field.py -k large_n
```

2. Re-run pinned GPU `7` TONB-on `steady_eval` (`1M`, `runs=3`, `warmup=1`)
   with current branch.
3. If result is neutral/better, run `nearfield_components` once on the same
   GPU/config to confirm bottleneck movement.
4. Keep TONB off for default production path until the above rerun confirms a
   clear win or at least neutrality.

## April 16 Production-Only Nearfield Components Checkpoint

Date: Thursday, April 16, 2026

### Benchmark worker updates used for this run

- [`examples/benchmark_gpu_radix_worker.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_worker.py)
  now:
  - defaults `JACCPOT_LARGE_N_DISABLE_CHUNK_COND=1` at worker startup (unless
    explicitly overridden)
  - adds a lightweight mode:
    - `--mode nearfield_components_production`
    - reports production-path timings only (nearfield total, specialized self,
      specialized pairs) and skips heavy experimental probe timings

### Command run

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode nearfield_components_production \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 3 \
    --warmup 1 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

Resolved environment from worker output:

- `CUDA_VISIBLE_DEVICES=9`
- `JACCPOT_NVIDIA_SMI_GPU_INDEX=9`
- `JAX_ENABLE_X64=1`

### Result

- `nearfield_component_mode=specialized_split_production`
- `prepared_state_mb=190.02`
- `evaluate_large_n_nearfield_seconds=0.0723` (`std=0.0103`)
- `nearfield_specialized_self_seconds=0.0633` (`std=0.0062`)
- `nearfield_specialized_pairs_seconds=1.7331` (`std=0.0192`)

### Interpretation

1. The new production-only component mode runs quickly and is suitable for
   iterative profiling loops.
2. The non-TONB specialized path remains the practical profiling baseline.
3. Pair-path cost remains the dominant target for the next restructuring work.

## April 16 TONB Structural Rewrite Checkpoint

Date: Thursday, April 16, 2026

### Scope completed

- completed yggdrax-first TONB producer payload wiring plus jaccpot large-N
  prepared-state/runtime integration
- implemented target-major prepacked TONB fast path (`[leaf, block, lane]`)
  with overflow fallback and memory guards
- validated compile/tests and reran GPU A/B using `autocvd`

### New controls introduced

- `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT` (`auto|0|1`)
- `JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB` (default `256`)
- `JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS` (default `8`)

### Key GPU results (`N=1048576`, `leaf=256`, `theta=0.6`, `chunk=512`)

Resolved runtime environment for the runs below:

- `autocvd` selected physical GPU `9`
- `CUDA_VISIBLE_DEVICES=9`
- `JACCPOT_NVIDIA_SMI_GPU_INDEX=9`
- `JAX_ENABLE_X64=1`

1. Baseline (TONB off):
- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0`
- `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0`
- `sweep` (`runs=3`, `warmup=2`):
  - `evaluate_mean_seconds=1.7923`
  - `evaluate_std_seconds=0.0132`

2. TONB on + speed prepared layout:
- `YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32`
- `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`
- `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT=1`
- `nearfield_components_production`:
  - `prepared_state_mb=216.55`
  - `evaluate_large_n_nearfield_seconds=7.0094`
  - `nearfield_specialized_self_seconds=0.0697`
  - `nearfield_specialized_pairs_seconds=1.8271`
- `sweep` (`runs=3`, `warmup=2`):
  - `evaluate_mean_seconds=7.2978`
  - `evaluate_std_seconds=0.0473`

### Interpretation

1. The full structural rewrite is integrated and stable, but TONB-on remains
   significantly slower than the current fastest path.
2. Memory remained controlled after adding capped fast-block prepacking
   (`~216.6 MB` prepared state), so the rewrite did not cause major blow-up.
3. The practical fastest baseline for now is still TONB-off around `~1.79 s`
   at `1M`.

### Validation status

- `python3 -m py_compile` passed for all touched nearfield/runtime files.
- focused nearfield large-N unit slice passed:
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

## Next Step Plan (Friday, April 17, 2026)

Goal for the day:

- refactor TONB nearfield into an XLA-fusible dense static-shape path without
  adding major memory growth, and decide go/no-go for non-custom-kernel route.

### 1) Freeze benchmark gate first

Run and record baseline before edits:

- config: `N=1048576`, `leaf=256`, `theta=0.6`, `nearfield_edge_chunk_size=512`
- mode: `sweep` (`runs=3`, `warmup=2`) + `nearfield_components_production`
- compare:
  - TONB off (`block_size=0`)
  - TONB on (`block_size=32`, speed layout toggles on)

Acceptance gate:

- no refactor step is kept unless it improves TONB-on steady-eval or clearly
  lowers pair-path cost at stable memory.

### 2) Collapse to one production TONB layout path

- keep a single canonical runtime TONB representation:
  - `source_ids [leaf, block, lane]`
  - `valid_mask [leaf, block, lane]`
- keep overflow as compact secondary dense payload only
- remove/disable alternate TONB runtime branches from hot dispatch

Acceptance gate:

- compile/test parity with current branch and identical numerical output on
  focused nearfield tests.

### 3) Dense pair kernel refactor

- make pair hot loop consume only dense target/source tiles
- avoid runtime leaf-id lookups in inner loop
- keep only dense masking, no ragged control inside inner math path

Acceptance gate:

- fewer kernel fragments in profile and lower/neutral pair phase time.

### 4) Remove inner-loop global update pressure

- accumulate per-target tile locally (`[T, P, 3]`)
- perform exactly one writeback per target tile
- no repeated `at[].add` inside pair tile loop

Acceptance gate:

- no regression in correctness; pair-phase and/or total nearfield time drops.

### 5) Enforce static trip counts and tuned unroll

- prefer fixed-shape `lax.scan` loops
- no dynamic `while_loop` in pair hot path
- tune unroll only after shape cleanup

Acceptance gate:

- reduced launch/control overhead in nearfield component timings.

### 6) Control shape polymorphism

- pin `target_leaf_batch_size`, `target_block_tile_size`, block-lane shape to
  a narrow profile set
- avoid env combinations that trigger recompilation churn

Acceptance gate:

- stable compile behavior and repeatable runtime across reruns.

### 7) Memory budget check

- keep prepared-state memory in RTX-safe range (target: around current
  TONB-on `~216 MB`, no large spikes)
- tune:
  - `JACCPOT_LARGE_N_SPEED_PREPARED_MAX_MB`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_FAST_BLOCKS`

Acceptance gate:

- no OOM/regression and no major memory blow-up while testing.

### 8) End-of-day decision

If TONB-on gets to a clear improving trajectory toward `<1s`, continue XLA-only
refactor. If not, proceed to fused Pallas/custom nearfield pair kernel on top
of the same prepared layout.

## April 16 Late TONB Refactor Continuation (Shared Fixed-Shape Batch Collector)

Summary of this continuation step:

1. Added shared target-batch collection helper in nearfield hot path:
   - `jaccpot/nearfield/near_field.py`
   - new helper: `_collect_target_leaf_batch_acc(...)`
   - purpose: keep batch scan shape static and remove duplicated per-kernel scan boilerplate.

2. Refactored both TONB pair kernels to use the shared helper:
   - `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(...)`
   - `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(...)`

3. Reduced per-batch layout overhead in prepacked TONB path:
   - precompute tiled source layout once as `[tile, leaf, lane_block, lane]`
   - gather by target-leaf batch directly from pretransposed tensors
   - remove per-batch `swapaxes(...)` work from hot loop body.

4. Kept runtime behavior/API stable:
   - dynamic TONB pair path still supports overflow-oriented block offsets.
   - `block_target_leaf_ids` remains accepted for compatibility (unused in kernel body).

Validation completed:

- `python3 -m py_compile` passed for touched nearfield/runtime files.
- focused GPU nearfield unit slice via autocvd passed:
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

Benchmark note:

- A new 1M `nearfield_components_production` TONB-on run (autocvd) was started,
  but an older stale benchmark worker from a previous session was discovered
  running concurrently and consuming compute.
- Stale worker was terminated; the new long-running benchmark was then stopped to
  avoid burning GPU time in an uncertain/blocked run.
- Next timing step should be a clean single-run relaunch on an idle GPU after
  confirming no residual benchmark workers are active.

## April 16 Deep Kernel Restructuring Continuation (TONB Overflow Unroll Wiring)

Completed in this step:

1. Propagated static scan-unroll controls into the dynamic/overflow TONB pair path:
   - `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(...)`
   - now accepts:
     - `target_block_tile_scan_unroll`
     - `target_block_batch_scan_unroll`
   - tile scan and batch scan both use these static values.

2. Propagated the same controls through accel-only wrapper path:
   - `_compute_leaf_p2p_prepared_large_n_accel_only_target_blocks_impl(...)`
   - forwards both unroll knobs into pair-kernel call.

3. Updated large-N dispatch wiring so both prepacked fast path and overflow path
   share the same canonical static unroll settings from environment canonicalization.

Why this matters:

- keeps the remaining TONB overflow path aligned with the fixed-shape scan tuning
  strategy used by the prepacked path
- avoids a hidden hardcoded `unroll=1` penalty in fallback/overflow pair execution
- keeps hot-path scan structure static and compilation-friendly.

Validation:

- `python3 -m py_compile` passed for touched files.
- focused GPU slice via autocvd passed:
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`
  - resolved GPU: `CUDA_VISIBLE_DEVICES=9`.

Benchmark status:

- no new 1M runtime benchmark recorded in this step; this checkpoint focused on
  structural kernel wiring and correctness.

## April 16 Kernel Deepening + RFL-00 Kickoff (Post-Continuation)

### A) Point-2 kernel restructuring completed (before RFL)

Implemented deeper TONB kernel consolidation toward fixed-shape dense execution:

1. Added shared dense-core helper for canonical tiled tensors:
   - `_compute_target_block_pairs_from_source_tiles(...)`
   - consumes `[tile, leaf, lane_block, lane]` source tensors and performs
     batched target accumulation + single scatter writeback per batch.

2. Rewired prepacked TONB pair kernel to use shared dense core:
   - `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(...)`
   - now focuses on layout normalization/padding only, then dispatches into the
     shared core.

3. Added bounded overflow tiled kernel:
   - `_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl(...)`
   - builds canonical tiled tensors from compact overflow offsets once, then
     calls the shared dense core.

4. Added bounded overflow dispatch cap:
   - new env knob canonicalized in large-N nearfield dispatch:
     - `JACCPOT_LARGE_N_TARGET_BLOCK_OVERFLOW_FAST_MAX_BLOCKS`
     - default `65536`
     - options default `16384,32768,65536,131072`
   - if overflow blocks are within cap, use tiled overflow kernel; otherwise
     keep existing overflow fallback kernel.

Rationale:

- keeps overflow path aligned with target-major fixed-shape strategy
- reduces divergent per-batch offset/tile assembly in overflow hot path
- retains memory guardrail via explicit block cap.

### B) RFL-00 started: explicit lane flag + invariant gating

Implemented runtime config lane toggle/invariants:

1. Extended `LargeNExecutionConfig`:
   - added `radix_fast_lane: bool`

2. Added resolver gating and hard checks in
   `resolve_large_n_execution_config(...)`:
   - note: earlier iterations used `JACCPOT_LARGE_N_RADIX_FAST_LANE` (`0/1`);
     this env is now legacy/no-op and fast-lane is policy-locked
   - fail fast unless all hold:
     - `tree_type='radix'`
     - `preset='large_n_gpu'`
     - `expansion_basis='solidfmm'`
     - `working_dtype=float32`
     - `grouped_interactions=False`
     - `nearfield_mode='bucketed'`
     - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE > 0`
   - fast-lane enabled behavior also forces:
     - `speed_prepared_layout=True`
     - `precompute_scatter=False`

Validation in this step:

- `python3 -m py_compile` passed for touched files.
- focused runtime tests (autocvd) passed:
  - `tests/unit/test_solver_api.py -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"`
  - `2 passed, 82 deselected`
- focused nearfield slice (autocvd) passed:
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

Open status:

- RFL-00 lane flag/invariant resolver is in place.
- next RFL action is to thread the fast-lane condition through large-N
  nearfield dispatch entry and add a dedicated fast-lane kernel entrypoint
  skeleton (RFL-03 scaffolding) while preserving legacy fallback paths.

## April 16 RFL-03 Dispatch Scaffold (Fast-Lane Entry Routing)

Implemented in this step:

1. Threaded lane flag into prepared state:
   - `LargeNPreparedState` now carries `radix_fast_lane: bool`.
   - pytree flatten/unflatten updated to preserve the flag through JIT state
     round-trips.

2. Propagated execution config flag into state construction:
   - `prepare_large_n_state(...)` now sets
     `radix_fast_lane=bool(execution_config.radix_fast_lane)`.

3. Added dedicated nearfield fast-lane entrypoint:
   - new function:
     `evaluate_large_n_nearfield_fast_lane(fmm, state, return_potential=...)`
   - current scaffold behavior:
     - accel-only path (`return_potential=False`) uses specialized large-N nearfield
       kernel call with TONB/prepared inputs.
     - potential path currently delegates to canonical generic nearfield evaluator
       for compatibility until fast-lane potential support is implemented.

4. Routed main nearfield evaluate dispatch through fast-lane entrypoint:
   - in `evaluate_large_n_nearfield(...)`, when `state.radix_fast_lane` is true,
     dispatch now goes through `evaluate_large_n_nearfield_fast_lane(...)` first.
   - legacy specialized/general paths remain intact as fallback when lane is off.

Validation:

- `python3 -m py_compile` passed for touched runtime/nearfield files.
- focused runtime slice via autocvd passed:
  - `tests/unit/test_solver_api.py -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"`
  - `2 passed, 82 deselected`
- focused nearfield slice via autocvd passed:
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

Status:

- RFL-03 scaffold is now in place with explicit fast-lane dispatch entry.
- next RFL step is to replace the scaffold internals with a dedicated fast-lane
  payload-driven kernel entry (RFL-02 + RFL-03 integration), then benchmark on
  frozen 1M gate.

## April 16 RFL-02 Payload Threading + Fast-Lane Payload Consumer

Implemented in this step:

1. Added dedicated payload dataclass in large-N types:
   - `RadixFastNearfieldPayload` in `jaccpot/runtime/_large_n_types.py`
   - fields include target leaf ids/particle ids/masks, source leaf ids/valid
     masks, optional source-particle placeholders, and batch tile metadata.

2. Threaded payload through prepared state:
   - `LargeNPreparedState` now includes:
     - `radix_fast_payload: Optional[RadixFastNearfieldPayload]`
   - pytree flatten/unflatten updated accordingly.

3. Constructed payload in prepare pipeline when fast lane is enabled:
   - in `prepare_large_n_state(...)`, build `radix_fast_payload` from
     precomputed leaf groups + prepacked target-major TONB tensors.
   - avoids source-particle tensor materialization for now (kept as empty
     placeholders) to avoid immediate memory blow-up in this phase.

4. Added payload-driven nearfield API entry:
   - new function in nearfield module:
     - `compute_leaf_p2p_accelerations_radix_fast_lane(...)`
   - uses payload tensors directly:
     - builds prepared leaf particle blocks from payload target ids/masks
     - runs self + prepacked TONB pair path
   - currently raises `NotImplementedError` for `return_potential=True`
     (potential path still intentionally delegated in fast-lane runtime entry).

5. Updated fast-lane runtime nearfield entry to consume payload:
   - `evaluate_large_n_nearfield_fast_lane(...)` now:
     - uses payload-driven nearfield API when payload exists and
       `return_potential=False`
     - keeps generic potential fallback
     - keeps specialized legacy fallback if payload is absent (migration guard).

Validation in this step:

- `python3 -m py_compile` passed for touched files.
- focused tests via autocvd passed:
  - `tests/unit/test_solver_api.py -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"`
  - `2 passed, 82 deselected`
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

Fast-lane-on smoke confirmation (payload route):

- env:
  - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT=1`
- public solver API smoke (`working_dtype=float32`, `nearfield_mode='bucketed'`)
  completed successfully with output:
  - `ok (512, 3) float32 True True`
  where trailing booleans indicate:
  - `state.radix_fast_lane == True`
  - `state.radix_fast_payload is not None == True`

Notes:

- A fast-lane-on pytest smoke under `JAX_ENABLE_X64=1` and default dtype failed
  as expected due the new invariant `working_dtype=float32`.
- This behavior is intentional and matches RFL-00 fail-fast policy.

## April 16 RFL Dense-Kernel Upgrade (Payload Source-Particle Core)

Implemented in this step:

1. Replaced fast-lane pair path internals with dedicated dense payload kernel:
   - new JIT kernel:
     - `_compute_radix_fast_lane_payload_pairs_impl(...)`
   - consumes payload-provided source-particle tensors directly and performs
     batched target-leaf accumulation over source-slot tiles.
   - keeps static trip counts and explicit unroll controls for both source-slot
     tile scan and target-batch scan.

2. Updated public fast-lane nearfield API to use dense payload core:
   - `compute_leaf_p2p_accelerations_radix_fast_lane(...)`
   - behavior:
     - computes self term via prepared self-only kernel
     - computes pair term via new payload-dense kernel when source-particle
       payload tensors are present
     - keeps migration fallback to previous prepacked source-leaf pair kernel
       if source-particle payload tensors are absent.

3. Upgraded prepare-time payload construction to include source-particle tensors:
   - in `prepare_large_n_state(...)`, when `radix_fast_lane` is enabled,
     precompute:
     - `source_particle_ids` shape `[T, S, Ls]`
     - `source_particle_mask` shape `[T, S, Ls]`
   - where `S = fast_blocks_aligned * target_block_size` source slots.
   - source slots are derived from prepacked target-major leaf/source tensors.

4. Added payload memory guardrail for source-particle tensors:
   - new env knob:
     - `JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB` (default `1024`)
   - if estimated source-particle payload exceeds cap, payload source-particle
     tensors are left empty and runtime falls back to prepacked source-leaf pair
     kernel (safe migration behavior).

5. Added source-slot tile metadata knob:
   - new env knob:
     - `JACCPOT_LARGE_N_RADIX_FAST_SOURCE_SLOT_TILE` (default `64`)
   - stored as payload batch tile metadata for dense source-slot scan.

Validation:

- `python3 -m py_compile` passed for touched files.
- focused tests via autocvd passed:
  - `tests/unit/test_solver_api.py -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"`
  - `2 passed, 82 deselected`
  - `tests/unit/core/test_near_field.py -k large_n`
  - `8 passed, 6 deselected`

Fast-lane payload-dense smoke confirmation:

- env:
  - `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32`
  - `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT=1`
- solver API smoke (`working_dtype=float32`, `nearfield_mode='bucketed'`) returned:
  - `ok (512, 3) float32 True True (16, 256, 32) (16, 256, 32)`
- confirms:
  - fast lane active
  - payload present
  - source-particle payload tensors are populated and consumed by the dense path.

## April 16 End-of-Day Resume Pack (Status + Next Tests)

### Where we stand

Current implementation status (latest):

1. TONB structural path is refactored around target-major fixed-shape execution,
   with shared dense accumulation helpers and bounded overflow handling.
2. RFL-00/03 scaffolding is in place:
   - explicit `radix_fast_lane` config flag + fail-fast invariants
   - dedicated fast-lane runtime nearfield entry and dispatch routing.
3. RFL-02 payload threading is in place:
   - `RadixFastNearfieldPayload` is prepared and stored in `LargeNPreparedState`.
4. Fast-lane nearfield now uses a dedicated payload-dense pair kernel core:
   - consumes payload source-particle tensors directly when available.
5. Guardrails are active:
   - payload source-particle precompute memory cap
   - overflow fast-path cap
   - migration fallback path kept for safety.

Validation status at this checkpoint:

- `python3 -m py_compile` passed on touched runtime/nearfield files.
- focused GPU tests via `autocvd` passed:
  - `tests/unit/test_solver_api.py -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"`
  - `tests/unit/core/test_near_field.py -k large_n`
- fast-lane-on smoke (`working_dtype=float32`, `nearfield_mode='bucketed'`) passed,
  confirming payload source-particle tensors are populated and used.

### Important caveat before tomorrow's tests

- `radix_fast_lane` intentionally requires `working_dtype=float32`.
- in environments where `JAX_ENABLE_X64=1`, do **not** treat fast-lane failures
  under default float64 test setup as regressions unless `working_dtype=float32`
  is explicitly set for the run.

### Tomorrow: exact test sequence to run first

Run in this order.

1) Quick sanity compile:

```bash
python3 -m py_compile \
  jaccpot/nearfield/near_field.py \
  jaccpot/runtime/_large_n_types.py \
  jaccpot/runtime/_large_n_pipeline.py \
  jaccpot/runtime/_large_n_nearfield.py
```

2) Focused runtime + nearfield regression slices (GPU via autocvd):

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python -m pytest -q -o addopts='' tests/unit/test_solver_api.py \
  -k "large_n_compiled_eval_uses_specialized_nearfield or large_n_gpu_preset_applies_memory_safe_gpu_defaults"
```

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  python -m pytest -q -o addopts='' tests/unit/core/test_near_field.py -k large_n
```

3) Fast-lane payload smoke (ensure payload route still active):

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
    JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT=1 \
  python -c "import jax.numpy as jnp; from jaccpot.solver import FastMultipoleMethod; n=512; pos=jnp.reshape(jnp.linspace(-1.0,1.0,n*3,dtype=jnp.float32),(n,3)); m=jnp.ones((n,),dtype=jnp.float32)/n; fmm=FastMultipoleMethod(preset='large_n_gpu', basis='solidfmm', working_dtype=jnp.float32, grouped_interactions=False, nearfield_mode='bucketed'); st=fmm.prepare_state(pos,m,leaf_size=32,max_order=3); acc=fmm.evaluate_prepared_state(st); p=st.radix_fast_payload; print('ok',acc.shape,acc.dtype,bool(st.radix_fast_lane),p is not None,p.source_particle_ids.shape,p.source_particle_mask.shape)"
```

Expected signal:
- trailing booleans should report fast-lane and payload enabled (`True True`),
- source-particle payload shapes should be non-empty.

4) Performance gate A/B at 1M (single-run quick signal first):

Baseline (TONB off):

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=0 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=0 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode nearfield_components_production \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 1 \
    --warmup 0 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

Fast-lane on (payload-dense path):

```bash
micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
  --use-autocvd -- \
  env \
    YGGDRAX_NEARFIELD_TARGET_BLOCK_SIZE=32 \
    JACCPOT_LARGE_N_TARGET_BLOCK_SIZE=32 \
    JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT=1 \
    JACCPOT_LARGE_N_RADIX_FAST_SOURCE_SLOT_TILE=64 \
    JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB=1024 \
  python examples/benchmark_gpu_radix_worker.py \
    --mode nearfield_components_production \
    --num-particles 1048576 \
    --leaf-size 256 \
    --max-order 4 \
    --runs 1 \
    --warmup 0 \
    --dtype float32 \
    --config-json '{"preset":"large_n_gpu","basis":"solidfmm","tree_type":"radix","leaf_target":256,"theta":0.6,"softening":0.001,"working_dtype":"float32","memory_objective":"minimum_memory","nearfield_mode":"bucketed","nearfield_edge_chunk_size":512,"streamed_far_pairs":true,"grouped_interactions":false,"enable_interaction_cache":false,"retain_traversal_result":false,"retain_interactions":false,"traversal_config":{"max_pair_queue":524288,"process_block":256,"max_interactions_per_node":16384,"max_neighbors_per_leaf":8192},"worker_autotune_traversal":false,"worker_autotune_nearfield_chunk":false}'
```

If quick signal is promising, repeat both with `--runs 3 --warmup 1` on same GPU.

### Next tuning sweep (only after successful A/B quick run)

- `JACCPOT_LARGE_N_RADIX_FAST_SOURCE_SLOT_TILE`: `32, 64, 128`
- `JACCPOT_LARGE_N_TARGET_BLOCK_TILE_SCAN_UNROLL`: `1, 2, 4`
- `JACCPOT_LARGE_N_TARGET_BLOCK_BATCH_SCAN_UNROLL`: `1, 2, 4`
- keep all other runtime parameters fixed.

# Large-N GPU Memory Handoff

This note captures the current state of the `examples/benchmark_gpu_single_n_memory.ipynb`
debugging effort as of March 13, 2026. The goal is a JIT-compiled large-`N` path with
minimum memory overhead that scales toward `1e6` particles.

## Problem Summary

For `N ~= 5e5`, `prepare_state(...)` on GPU was failing with XLA compile/runtime memory
errors such as:

```text
gpu_hlo_schedule.cc:868 ... input/output arguments ... 48.09GiB
hlo_rematerialization.cc:3233 ... only reduced to 48.09GiB
RESOURCE_EXHAUSTED: Failed to allocate request for 48.09GiB
```

After some runtime changes, the dominant failure signature dropped to about `24.05 GiB`,
which turned out to be a strong clue that one static intermediate had been reduced by
roughly `2x`, but not eliminated.

## Confirmed Findings

1. The notebook itself had several misleading or harmful analysis paths.
   - Outer `jax.jit(... prepare_state ...)` compilation was measuring an artificial path.
   - Some helper cells were restoring large static traversal settings or worker-side
     `balanced` policy by accident.
   - These notebook issues have been cleaned up, but they were not the final root cause.

2. The first real root cause was in Yggdrax geometry construction under JIT.
   - In `yggdrax/_geometry_impl.py`, `_compute_leaf_bounds(...)` used:
     - `max_count = num_particles`
     when `leaf_counts` was traced.
   - That makes the geometry staging gather scale like:
     - `(num_leaves, num_particles, 3)`
     instead of:
     - `(num_leaves, max_leaf_size, 3)`
   - For the failing radix case (`524288` particles, `4096` leaves, `leaf_size=128`),
     this is exactly the kind of intermediate that can explain the fixed `~24 GiB`
     compile/request signature.

3. The failure was not primarily caused by:
   - `float32` / `int32` precision choice
   - the compact streamed far-pair path itself
   - the traversal count-pass once bounded explicit traversal config was reinstated

4. Diagnostic logging proved the giant allocation warning appears during the upward
   geometry stage, before traversal starts:

```text
[jaccpot.prepare] upward start ...
[jaccpot.upward] geometry start ...
E... 24.05GiB
[jaccpot.upward] geometry done
[jaccpot.upward] mass moments done
...
[jaccpot.prepare] dual-tree start ...
```

This means traversal exceptions were often just where the async device error surfaced,
not the true source of the bad allocation.

## Key Runtime Case

The clearest failing case so far was:

```text
particles=524288
leaf_size=128
total_nodes=8191
num_internal=4095
num_leaves=4096
theta=0.6
max_order=4
streamed_far_pairs=True
grouped_interactions=False
farfield_mode=pair_grouped
memory_objective=minimum_memory
```

## Changes Already Made

### Notebook / Benchmark Harness

- Removed outer prepare-state compile analysis from the main benchmark path.
- Restored explicit `memory_objective="minimum_memory"` in the notebook config.
- Fixed worker config reconstruction so it preserves `memory_objective`.
- Disabled misleading sweep/compile helper defaults in the notebook.
- Added prepare/traversal diagnostics via environment variables.

### Jaccpot Runtime

- Streamed large-`N` path can request compact COO far pairs directly without forcing
  a full node-interaction object when it is not needed.
- Large-`N` minimum-memory GPU runtime now uses bounded explicit traversal defaults
  instead of unbounded auto-sized traversal:
  - `max_pair_queue=32768`
  - `process_block=64`
  - `max_interactions_per_node=1024`
  - `max_neighbors_per_leaf=256`
- Upward sweep defaults to bounded leaf batching (`2048`) instead of `num_leaves`.
- Added prepare/upward diagnostics in `jaccpot/runtime/_fmm_impl.py` and
  `jaccpot/upward/solidfmm_complex_tree_expansions.py`.

### Yggdrax

- Count-pass / compact-fill traversal path got multiple dtype and overflow fixes.
- Traversal diagnostics were added to `yggdrax/_interactions_impl.py`.
- Geometry now accepts an explicit `max_leaf_size` so traced large-`N` runs do not
  pad leaf gathers out to `num_particles`.

## Most Important Code Changes

- `yggdrax/yggdrax/_geometry_impl.py`
  - `_compute_leaf_bounds(..., max_leaf_size=...)`
  - `compute_tree_geometry(..., max_leaf_size=...)`
- `yggdrax/yggdrax/geometry.py`
  - public `compute_tree_geometry(..., max_leaf_size=...)`
- `jaccpot/jaccpot/upward/solidfmm_complex_tree_expansions.py`
  - passes Jaccpot's known `max_leaf_size` into `compute_tree_geometry(...)`
- `jaccpot/jaccpot/runtime/_fmm_impl.py`
  - diagnostics and bounded traversal overrides for minimum-memory large GPU runs
- `yggdrax/yggdrax/_interactions_impl.py`
  - traversal diagnostics, bounded traversal fixes, dtype fixes

## Current Diagnostics

The following env vars are useful and have already been wired into the notebook flow:

```python
os.environ["JACCPOT_PREPARE_DIAGNOSTICS"] = "1"
os.environ["YGGDRAX_TRAVERSAL_DIAGNOSTICS"] = "1"
```

Useful marker families:

- `[jaccpot.prepare]`
- `[jaccpot.upward]`
- `[yggdrax.traversal]`

## Validation Already Run

Targeted checks that passed in the `odisseo` environment:

- Yggdrax geometry wrapper tests, including explicit leaf-cap under outer JIT
- Jaccpot targeted solver tests around the updated upward path

Representative commands:

```bash
micromamba run -n odisseo python -m pytest -q -o addopts='' \
  /export/home/tbuck/yggdrax/tests/unit/test_geometry_api_wrapper.py \
  -k 'explicit_leaf_cap_under_outer_jit or supports_outer_jit_wrapper'

JAX_ENABLE_X64=1 micromamba run -n odisseo python -m pytest -q -o addopts='' \
  tests/unit/test_solver_api.py \
  -k 'solidfmm_upward_defaults_to_bounded_leaf_batch_size or solver_matches_expanse_fast_path'
```

## Latest Confirmed State

The original large upward-memory failure is resolved for the target GPU notebook case.

For:

```text
particles=524288
leaf_size=128
theta=0.6
max_order=4
rotation=solidfmm
memory_objective=minimum_memory
streamed_far_pairs=True
grouped_interactions=False
farfield_mode=pair_grouped
```

the prepare path now completes with:

- working single-shot fail-fast traversal seed:
  - `max_pair_queue=524288`
  - `process_block=512`
  - `max_interactions_per_node=16384`
  - `max_neighbors_per_leaf=8192`
- dual-tree result:
  - `neighbor_count=1371872`
  - `far_pair_count=817210`
  - `compact_far_pairs=True`
  - `interactions_present=False`

Peak/size summary from the current notebook:

- cold prepare peak GPU delta: `4006 MB`
- warm prepare peak GPU delta: `2326 MB`
- evaluate peak GPU delta: `204 MB`
- prepared state size: `27.30 MB`
- cold prepare wall time: `~86.1 s`
- warm prepare wall time: `~37.4 s`
- evaluate wall time: `~2.29 s`

Retained prepare-state payload is small:

- dual-tree retained bytes:
  - neighbors: `5.28 MiB`
  - compact far pairs: `9.35 MiB`
- downward retained bytes:
  - locals: `1.66 MiB`
- nearfield retained bytes:
  - `0`

Conclusion:

- the catastrophic multi-10-GB allocation is gone
- the remaining prepare peak is transient workspace, not retained state
- roughly `1.68 GB` of the cold peak is compile/first-call overhead
- the main remaining optimization target is warm prepare workspace/runtime

## Traversal / Retry Policy

- `fail_fast=True` is now supported and recommended for final production-style runs.
- In fail-fast mode Jaccpot:
  - disables host-side refinement override
  - disables M2L autotune
  - disables host-side dual-tree retry growth
  - fails immediately with a capacity hint if traversal buffers are undersized
- The notebook and worker now pass explicit traversal configs through correctly for
  minimum-memory streamed runs.

## MAC / Dehnen Error Status

- The notebook now supports direct MAC comparisons for:
  - `dehnen`
  - `engblom`
  - `dehnen_error`
- `dehnen_error` previously mapped back to plain `dehnen` without activating the
  traversal policy hook.
- Jaccpot runtime now wires `mac_type=\"dehnen_error\"` into the existing
  solver-side adaptive pair policy using the Dehnen-paper error estimator during
  traversal.
- Default normalization for `dehnen_error` now sets:
  - `adaptive_error_model=\"dehnen_paper\"` when left at the default
  - `mac_force_scale_mode=\"paper\"` when left at the default
- Explicit user overrides for `adaptive_error_model` and `mac_force_scale_mode`
  still take precedence.

## Current Open Questions

1. Which traversal seed minimizes warm prepare peak while still succeeding in
   single-shot fail-fast mode.
2. Whether `engblom` or the corrected `dehnen_error` reduces far-pair count and
   warm prepare time relative to plain `dehnen` at matched accuracy.
3. How much of the remaining `~2.3 GB` warm prepare peak is dominated by dual-tree
   workspace versus downward/M2L staging.

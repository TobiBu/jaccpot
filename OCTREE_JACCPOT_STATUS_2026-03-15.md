# Octree FMM Status Handoff

Date: 2026-03-15

Repos:
- `/export/home/tbuck/jaccpot`
- `/export/home/tbuck/yggdrax`

Branches:
- `jaccpot`: `oct-tree`
- `yggdrax`: `oct-tree-updates`

Environment:
- `micromamba run -n odisseo ...`

Preferred validation GPU right now:
- `GPU 3`

Notes:
- Today’s focused validation was run on `GPU 3`.
- The work has moved past the earlier farfield boundary blocker.
- Current active blocker is the octree-native nearfield target-subset path in `jaccpot`.
- There are unrelated local changes in `jaccpot` that were intentionally not touched:
  - `examples/benchmark_gpu_n_ladder_production.ipynb`
  - untracked `benchmarks/`

## Goal

Keep moving toward a genuinely independent octree implementation:
- independent octree build in `yggdrax`
- shared tree API across radix, kd, and octree
- octree-native traversal and geometry where possible
- no host-side reorder hacks
- keep everything JAX-native and GPU-efficient
- finish a fully octree-native nearfield path without falling back to radix-style leaf contracts

## Current High-Level Status

What is done:
- octree has its own build path in `yggdrax`
- octree metadata is built from native octree leaf cells
- source-owned octree geometry and traversal view exist in `yggdrax`
- `jaccpot` consumes source-owned octree geometry and traversal packaging
- octree farfield now has a native public boundary in `yggdrax`
- octree backend in `jaccpot` consumes native octree far pairs
- octree nearfield traversal in `yggdrax` has a native execution path
- `yggdrax` now exposes a native octree near-neighbor artifact
- the new native near-neighbor compaction is JAX-only; no NumPy dependency remains in that path
- `jaccpot` now has an in-progress carrier-based nearfield interop shape for octree
- full octree execution currently still matches radix on the focused solver test

What is not done:
- `jaccpot` target-subset nearfield evaluation is still wrong for the native carrier-node path
- `NearfieldInteropData` is mid-migration from contiguous leaf ranges to explicit particle-group ownership
- the target-side source-index builder still assumes an old leaf-neighbor contract
- native nearfield is not fully wired through all prepared-state evaluation paths yet

## Important Recent Commits

### `yggdrax`

- `023b7e5` `Make leaf topology a shared tree contract`
- `7f3c0a8` `Route octree through its own build pipeline`
- `78095d8` `Build octree metadata from native leaf cells`
- `4607476` `Move octree box geometry to source module`
- `12b08c2` `Add source-owned octree traversal view`
- `3a9759d` `Add octree-specific traversal dispatch seam`
- `5a40540` `Add native octree child refinement helpers`
- `8c2fb65` `Use native octree walk for nearfield traversal`
- `9190481` `Add octree-native far-pair boundary`

### `jaccpot`

- `ab0f544` `Expose native octree box geometry in execution view`
- `86a9ed5` `Use native octree geometry and parent links`
- `76a5886` `Consume source-owned octree box geometry`
- `cc2d7d5` `Consume source-owned octree traversal view`
- `854971f` `Wire octree backend to native far pairs`

## What Changed After The Earlier Handoff

### Farfield boundary is no longer the main blocker

The earlier architectural question about public farfield layout has been resolved enough to proceed:
- `yggdrax` now exposes `CompactTaggedOctreeFarPairs`
- `build_octree_native_far_pairs(...)` returns exact-length far pairs in explicit octree node space
- `jaccpot` octree backend now consumes that native pair stream for octree farfield planning

This means the main unresolved area has shifted from farfield to nearfield.

### Native octree near-neighbor artifact now exists in `yggdrax`

New public/native pieces were added in:
- `/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py`
- `/export/home/tbuck/yggdrax/yggdrax/interactions.py`
- `/export/home/tbuck/yggdrax/yggdrax/__init__.py`

Key detail:
- the native nearfield carrier nodes are the unique values of `radix_leaf_to_oct`
- these carrier nodes are not guaranteed to be explicit structural octree leaves
- some carrier nodes can be internal octree nodes

That detail is exactly why the old `jaccpot` nearfield contract could not simply be remapped.

## Current `jaccpot` Nearfield Migration State

In-progress local edits exist in:
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- `/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py`
- `/export/home/tbuck/jaccpot/tests/unit/test_solver_api.py`

### What has already been added locally

`NearfieldInteropData` is being extended with explicit group ownership:
- `leaf_particle_indices`
- `leaf_particle_mask`
- `particle_to_leaf_position`

The octree nearfield builder in `_fmm_impl.py` now:
- accepts native octree neighbors
- groups radix leaves by carrier node via `octree.radix_leaf_to_oct`
- builds explicit per-carrier particle membership tables

The nearfield runtime in `near_field.py` now has group-based helpers:
- `prepare_bucketed_scatter_schedules_from_groups(...)`
- `_prepare_leaf_data_from_groups(...)`
- `_compute_leaf_p2p_from_prepared_leaf_data_impl(...)`

And the nearfield compute path can now accept:
- `leaf_particle_indices_override`
- `leaf_particle_mask_override`

This is the right direction because it removes the old assumption that every nearfield leaf is represented by a single disjoint `node_ranges[leaf]` interval.

## Current Blocker

The remaining regression is now narrow and well-localized:
- full octree execution matches the radix result on the focused test
- the target-subset prepared-state path is still wrong
- the mismatch is entirely in nearfield, not farfield

Current failing focused test:
- `test_octree_execution_backend_target_indices_match_full_prepared_state`

Observed behavior:
- full nearfield values are correct
- direct per-carrier local sums for sample targets match the full result
- the target-side source-index matrix is over-expanding source particles

Most likely cause:
- `_build_target_nearfield_source_index_matrix(...)` still interprets `nearfield_interop.neighbors` as if the native carrier nodes were the old leaf-position graph
- for the native carrier case, the target path should not reconstruct source positions through the old leaf lookup assumptions

## Most Likely Fix

Add an explicit neighbor-position artifact to `NearfieldInteropData` for the native carrier case.

Recommended shape:
- precompute neighbor leaf positions directly during `_build_nearfield_interop_data(...)`
- store them in the interop object in carrier-order space
- let `_build_target_nearfield_source_index_matrix(...)` use those positions directly instead of rebuilding them through `leaf_lookup[neighbors]`

This should keep the full path unchanged while fixing the target-subset path.

## Important Debugging Facts

For the 72-particle sample used during debugging:
- carrier nodes were `[0, 3, 4, 5, 6, 7, 8, 9, 10]`

Carrier membership inspection showed:
- carrier 0 node 0 owned particles `32..39`
- carrier 1 node 3 owned particles `0..7`
- carrier 2 node 4 owned particles `24..31`
- carrier 3 node 5 owned particles `40..47`
- carrier 4 node 6 owned particles `64..71`
- carrier 5 node 7 owned particles `8..15`
- carrier 6 node 8 owned particles `16..23`
- carrier 7 node 9 owned particles `48..55`
- carrier 8 node 10 owned particles `56..63`

For the failing target-subset case:
- farfield contribution was zero in the sample
- full-vs-target mismatch came entirely from target-side nearfield source expansion

## Validation Summary

### Focused `yggdrax` native nearfield slice

Command:
```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 micromamba run -n odisseo pytest -q -o addopts='' \
  /export/home/tbuck/yggdrax/tests/unit/test_octree_topology.py -k 'native'
```

Latest result:
- `3 passed, 9 deselected`

### Focused `jaccpot` octree solver slice before the latest nearfield migration

Command:
```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 micromamba run -n odisseo pytest -q -o addopts='' \
  /export/home/tbuck/jaccpot/tests/unit/test_solver_api.py \
  -k 'octree and (native_nearfield_view or target_indices_match_full_prepared_state or execution_backend_matches_radix_on_octree_tree)'
```

Current result during local migration:
- `test_octree_execution_backend_matches_radix_on_octree_tree` passed
- `test_octree_execution_backend_exposes_native_nearfield_view` passed
- `test_octree_execution_backend_target_indices_match_full_prepared_state` failed

Interpretation:
- full octree execution is still correct on the focused slice
- native nearfield view is exposed correctly
- only the target-subset nearfield path remains broken

## Main Files To Resume In Tomorrow

`jaccpot`:
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`
- `/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py`
- `/export/home/tbuck/jaccpot/tests/unit/test_solver_api.py`

`yggdrax` reference points:
- `/export/home/tbuck/yggdrax/yggdrax/_interactions_impl.py`
- `/export/home/tbuck/yggdrax/yggdrax/interactions.py`

## Practical Resume Point

Start in:
- `/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py`

Inspect:
- `_build_nearfield_interop_data(...)`
- `_build_target_nearfield_source_index_matrix(...)`
- `_evaluate_prepared_tree_targets(...)`

Then compare with:
- the native carrier ordering produced from `octree.radix_leaf_to_oct`
- the native neighbor list ordering from `yggdrax`

The likely next patch is:
- add direct native neighbor-position data to `NearfieldInteropData`
- consume it in `_build_target_nearfield_source_index_matrix(...)`

## Useful Commands For Tomorrow

Focused `yggdrax` native slice:
```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 micromamba run -n odisseo pytest -q -o addopts='' \
  /export/home/tbuck/yggdrax/tests/unit/test_octree_topology.py -k 'native'
```

Focused `jaccpot` target-subset regression:
```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 micromamba run -n odisseo pytest -q -o addopts='' \
  /export/home/tbuck/jaccpot/tests/unit/test_solver_api.py \
  -k 'octree and (native_nearfield_view or target_indices_match_full_prepared_state or execution_backend_matches_radix_on_octree_tree)'
```

Show current local changes:
```bash
git -C /export/home/tbuck/jaccpot status --short
```

## Suggested Resume Prompt

Use this in a fresh chat:

```text
We are continuing octree work across /export/home/tbuck/jaccpot and /export/home/tbuck/yggdrax.
Please read /export/home/tbuck/jaccpot/OCTREE_JACCPOT_STATUS_2026-03-15.md first.
Current status: octree farfield is now on a native public boundary and the active blocker is native nearfield target-subset evaluation in jaccpot.
The likely bug is in how _build_target_nearfield_source_index_matrix(...) interprets native carrier-node neighbors.
Please continue from the current local changes, keep the implementation JAX-native, and validate on GPU 3.
```

# Large-N Runtime Status

This note captures the current `large_n` runtime work on branch `large_N_performance`
as of March 23, 2026. It is intended as a clean handoff point before the next
near-field optimization pass.

## Branch And Commit State

Confirmed committed stack:

- `22bbe8b` `Add dedicated large-N runtime path`
- `5e3f946` `Add runtime-path controls to benchmark tooling`
- `5677676` `Add large-N benchmark notebooks and toggles`
- `e8605ef` `Tighten large-N compiled evaluation path`

Current uncommitted work:

- [`jaccpot/runtime/_large_n_types.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py)
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
- [`jaccpot/runtime/_large_n_pipeline.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py)

Those uncommitted edits are an in-progress near-field experiment. They are not yet
validated by a completed benchmark run, so all conclusions below refer to the last
confirmed committed baseline unless explicitly noted otherwise.

## What The Large-N Path Already Does

The dedicated `large_n` runtime path is now in place and dispatched through the
public API for the narrow case we care about:

- radix tree
- `solidfmm` basis
- GPU execution
- full-particle evaluation only
- no target-subset path
- no acceleration-derivative path

Main implementation files:

- [`jaccpot/runtime/_large_n_types.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py)
- [`jaccpot/runtime/_large_n_pipeline.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py)
- [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
- [`jaccpot/runtime/_large_n_farfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_farfield.py)

This path already removed a lot of general-runtime complexity:

- slim prepared state
- direct compiled evaluation path
- no retained near-field pair vectors in minimum-memory mode
- no retained leaf-group buffers in the last confirmed baseline
- no large extra payloads like full `upward` state in minimum-memory prepared states

## Confirmed Performance Findings

All measurements below were taken on GPU 9 unless stated otherwise.

### 1. The current large-N path is near-field bound

At `N=256_000` with the confirmed committed `large_n` path:

- total warm eval: about `0.918 s`
- near-field alone: about `0.915 s`
- far-field alone: about `0.005 s`

This is the single most important current result.

Conclusion:

- the large-`N` runtime is not far-field bound
- it is not primarily wrapper-bound anymore
- it is overwhelmingly near-field bound

### 2. Current practical RTX 2080-scale operating point

Confirmed warm eval timings:

- `N=256_000`: best about `0.897 s`, average about `0.898 s`
- `N=512_000`: best about `2.121 s`, average about `2.124 s`

Conclusion:

- `256k` is the right active tuning target on this GPU class
- `512k` is a good stretch validation size
- `1e6` is not yet a practical optimization target on this card

### 3. Large-N accuracy is already in good shape

Representative legacy-vs-`large_n` checks:

- `N=2048`: relative L2 about `5.12e-08`, max abs about `4.58e-05`
- `N=65536`: relative L2 about `4.65e-08`, max abs about `1.95e-03`

Conclusion:

- the current `large_n` path is already accurate enough to optimize aggressively
- runtime and memory are now the main focus

### 4. The main remaining issue at `1e6` is memory fit, not warm eval speed

At `N=1_000_000`, prepare fails on GPU 9 across tested `leaf_size` values due to OOM
inside the dual-tree traversal build before evaluation starts.

Resolved traversal settings at that point were already fairly lean:

- `max_pair_queue=32768`
- `process_block=64`
- `max_interactions_per_node=1024`
- `max_neighbors_per_leaf=256`

Conclusion:

- `1e6` is currently blocked by traversal-build memory
- the sub-second goal at `1e6` is secondary until the path fits at all

## Parameter Sweep Findings Worth Keeping

The following sweep results were useful and should not be lost:

### Working-size runtime ladder

- `100k` is the best fast-iteration size
- `256k` is the best main validation size
- `512k` is the best stretch size

### Leaf size

Earlier tuning strongly suggested:

- `leaf_size=256` was better than `128` for runtime at `256k`
- `leaf_size=384` did not look like the best runtime compromise
- shrinking near-field chunk size to `128` or `64` hurt runtime

Conclusion:

- `leaf_size=256` remains the most promising next tuning baseline
- aggressive near-field over-chunking is not promising

### Pair-vector retention experiment

One experiment retained near-field pair vectors for the `large_n` path at `256k`.
That made runtime worse:

- before: total best about `0.918 s`
- after retaining pair vectors: total best about `1.256 s`

Conclusion:

- retaining more near-field metadata is not automatically helpful
- we should avoid bringing back pair-vector retention as the next step

## Compile-Time Findings

Compile time is not the current priority, but the results are worth recording.

At `N=65536`:

- `_prepare_state_tree_and_upward()` first call: about `21.53 s`
- `_prepare_state_tree_and_upward()` second call: about `0.042 s`
- `_prepare_state_dual_and_downward()` first call: about `56.01 s`
- `_prepare_state_dual_and_downward()` second call: about `0.113 s`

Also, the raw warm compact-far-pair traversal itself was not especially slow:

- raw traversal best: about `0.079 s`
- traversal cache helper best: about `0.066 s`

Conclusion:

- the alarming stage timings were largely first-call compile cost
- warm runtime work should stay focused on near-field and memory

## Updated H200 Memory Findings

The newer H200 memory work changed the picture substantially. The main result is:

- the lean `large_n_gpu` / `minimum_memory` path now reaches `N=4_194_304`
  in the single-`N` notebook on the H200 after the Yggdrax split-traversal work
- however, prepare is still the dominant memory bottleneck

Representative `N=4_194_304` numbers from
[`benchmarks/single_n_memory/single_n_4194304_summary.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_4194304_summary.csv):

- prepared state size: about `215.84 MB`
- prepare cold peak delta: about `12.83 GB`
- prepare warm peak delta: about `8.65 GB`
- evaluate warm peak delta: about `1.37 GB`

Conclusion:

- retained prepared-state size is not the limiting factor at `4M`
- evaluation is not the limiting factor either
- the practical ceiling is still prepare-side transient memory

### What The Yggdrax Split Work Achieved

The Yggdrax-side split traversal work was decisive.

At `N=4_194_304`, the localized stage split in
[`benchmarks/single_n_memory/single_n_4194304_prepare_stage_memory_split.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_4194304_prepare_stage_memory_split.csv)
shows:

- old monolithic raw traversal build warm peak: about `41.08 GB`
- old monolithic raw traversal build cold peak: about `69.76 GB`
- new split traversal build warm peak: about `8.41 GB`
- new split traversal build cold peak: about `12.51 GB`

Conclusion:

- without the Yggdrax split-builder and compact-far-pair work, `4M` would still
  be completely impractical
- the split path is a real success, not just a cosmetic cleanup
- but the split traversal build itself is now the next dominant memory target

### Where The Remaining Memory Goes

The retained-state breakdown at `N=4_194_304` from
[`benchmarks/single_n_memory/single_n_4194304_prepared_subsystems.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_4194304_prepared_subsystems.csv)
and
[`benchmarks/single_n_memory/single_n_4194304_prepared_paths.csv`](/export/home/tbuck/jaccpot/benchmarks/single_n_memory/single_n_4194304_prepared_paths.csv)
shows:

- `tree_geometry`: about `146.19 MB`
- `nearfield`: about `56.40 MB`
- `other`: about `13.25 MB`

Largest retained arrays:

- `prepared.tree.positions_sorted`: `48 MB`
- `prepared.neighbor_list.neighbors`: about `35.78 MB`
- `prepared.tree.topology.morton_codes`: `32 MB`
- `prepared.tree.topology.particle_indices`: `32 MB`

Conclusion:

- Jaccpot retained state is relatively slim compared with the multi-GB prepare
  transient
- the current next win is not prepared-state slimming first
- the current next win is lowering the warm transient footprint of the split
  traversal build

### Updated Bottleneck Hypothesis

The old prepare-cold story was:

- raw dual-tree traversal compile spikes were the main blocker

The updated `4M` story is:

- the catastrophic compile spike was reduced enough to fit
- the remaining blocker is now the warm split traversal build itself
- tree/upward, downward, and retained state are all secondary at this point

That means the highest-value next question is now:

- inside the split traversal build, how much of the `~8.4 GB` warm footprint
  comes from the far-only compact-pair builder and how much comes from the
  near-only neighbor-list builder?

The profiler has been extended to answer exactly that, but the new `far_only` /
`near_only` split CSV rows were not yet collected at the time of writing.

## Current Hypothesis For The Next Win

The strongest current hypothesis is:

1. The committed `large_n` path is still entering the generic near-field contract.
2. In the confirmed baseline, it does not retain explicit leaf-particle groups.
3. That means warm eval still rebuilds padded per-leaf particle views before P2P.
4. A good next tradeoff is likely:
   - retain compact leaf-particle groups
   - do not retain heavy pair vectors
   - keep the full-eval-only `large_n` assumptions

That is exactly what the current uncommitted near-field experiment is trying to test.

Important:

- this experiment is not yet benchmark-confirmed
- the last benchmark run for it was interrupted before completion

## Immediate Next Steps

### 1. Finish the interrupted near-field A/B benchmark

Use the same prepared state and compare:

- baseline: zeroed leaf-group buffers
- new path: retained leaf-group buffers

Measure:

- near-field warm time only
- full warm eval time
- prepared-state size delta

Do this first at:

- `N=100_000`
- `N=256_000`

### 2. If leaf-group retention helps, keep it and promote the test

Then re-run at:

- `N=256_000`
- `N=512_000`

And compare:

- warm eval
- near-field-only time
- prepared-state bytes

### 3. If leaf-group retention does not help enough, build a narrower near-field kernel

Most likely direction:

- add a `large_n` full-eval near-field helper that bypasses as much of
  `compute_leaf_p2p_accelerations(...)` as possible
- keep assumptions narrow:
  - full-particle eval only
  - bucketed mode only
  - no target-subset path
  - no neighbor-pair collection
  - no generic override plumbing in the hot path

### 4. Split the `4M` warm traversal transient into far-only vs near-only

Run the updated prepare-stage split notebook cell and record:

- `dual_tree_split_far_only_cold`
- `dual_tree_split_far_only_warm`
- `dual_tree_split_near_only_cold`
- `dual_tree_split_near_only_warm`

This is now the highest-priority measurement because it will determine whether
the next Yggdrax optimization should focus on:

- compact far-pair construction
- near-only neighbor-list construction
- or any shared queue/buffer staging that still survives both split paths

### 5. After that attribution, continue memory scaling work in Yggdrax

If the near-only path dominates:

- reduce warm transient neighbor buffer footprint first
- inspect queue usage and near-list staging layout in the generic near builder

If the far-only path dominates:

- continue reducing queue and staging overlap in the compact far-pair builder

If both are large:

- reduce live queue/buffer overlap and shared staging inside the split traversal
  path itself

### 6. Only after the Yggdrax warm transient is smaller, revisit prepared-state slimming

Once the split traversal warm peak is lower:

- validate again at `4M`
- retry the fit path beyond `4M`
- then revisit retained-state slimming and evaluation leaf staging if the next
  ceiling is no longer traversal-dominated

### 7. If wrapper-side split traversal is not enough, patch Yggdrax itself

Recent prepare-memory profiling narrowed the remaining large cold spike to the
raw dual-tree traversal build. Splitting the Jaccpot-side far/near build path
helped, but it did not remove the underlying monolithic Yggdrax traversal
kernel allocation spike.

That means one explicit follow-up track should now be recorded:

- add a native split far/near traversal builder in `yggdrax._interactions_impl`
- or add a traversal mode that avoids materializing far and near buffers in the
  same compiled kernel
- then re-measure prepare cold/warm peaks against the current Jaccpot-side split
  helper

This work is no longer hypothetical; it already happened and it moved the fit
ceiling materially. The remaining Yggdrax work should now be understood as:

- phase 1: remove catastrophic compile/raw-build spikes
- phase 2: reduce the warm split-build transient at multi-million-particle scale

## Concrete Resume Checklist

1. Inspect:
   - [`jaccpot/runtime/_large_n_types.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_types.py)
   - [`jaccpot/runtime/_large_n_nearfield.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_nearfield.py)
   - [`jaccpot/runtime/_large_n_pipeline.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_large_n_pipeline.py)
2. Decide whether to keep or revert the uncommitted leaf-group retention experiment.
3. Re-run the interrupted GPU 9 A/B benchmark.
4. If it wins, commit it as a focused near-field optimization step.
5. If it loses, revert it and move directly to a specialized `large_n` near-field kernel.
6. If prepare cold memory still blocks scaling after the Jaccpot-side split build,
   move the next memory pass into `yggdrax._interactions_impl`.

## Bottom Line

The large-`N` refactor was worthwhile.

We now know:

- the slim `large_n` path is structurally in place
- accuracy is already good
- far-field cost is tiny
- near-field dominates runtime
- pair-vector retention is not the answer
- the next real opportunity is a cleaner, narrower near-field execution path

That is the right place to continue.

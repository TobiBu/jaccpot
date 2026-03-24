# H200 Large-N Follow-Up Plan

This note captures the current large-`N` GPU optimization status as of March 23, 2026.
It is meant as a reload point after logging out of the cluster.

## Status

- Notebook benchmark setup in [`examples/benchmark_runtime_accuracy_copy.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_accuracy_copy.ipynb) was updated to use the lean `large_n_gpu`/`engblom` baseline consistently.
- The first code-level optimization pass is implemented in [`jaccpot/upward/solidfmm_complex_tree_expansions.py`](/export/home/tbuck/jaccpot/jaccpot/upward/solidfmm_complex_tree_expansions.py).
- A first prepared-state slimming pass is implemented in [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py): minimum-memory prepared states no longer retain the original `upward` or `octree_upward` payloads.
- Verification has not been run yet in this session because we are deferring tests to another GPU server.

## Implemented Change

### 1. Streamed leaf P2M accumulation

Problem:

- `_p2m_leaves_complex(...)` previously materialized the full temporary
  `(leaf_batch_size, max_leaf_size, coeffs)` contribution tensor before reducing over particles.
- For large radix trees this is a likely source of transient HBM pressure during the upward sweep.

Change made:

- Added a bounded particle chunk path inside `_p2m_leaves_complex(...)`.
- The code now accumulates leaf coefficients over particle subchunks instead of forming one full contribution tensor for the whole leaf batch.
- Current chunk constant:
  - `_P2M_PARTICLE_CHUNK_SIZE = 32`

Expected effect:

- Lower transient GPU memory in the upward leaf P2M stage.
- Same math and same accuracy, because the operator itself is unchanged and we are only changing accumulation layout.
- Runtime should be neutral to mildly better if memory traffic was the main bottleneck, though this must still be measured.

Files changed:

- [`jaccpot/upward/solidfmm_complex_tree_expansions.py`](/export/home/tbuck/jaccpot/jaccpot/upward/solidfmm_complex_tree_expansions.py)

## Deferred Verification

Run later on the GPU server:

```bash
pytest -q tests/unit/core/test_solidfmm_complex_tree_expansions.py
```

Then re-run the notebook/runtime benchmark, especially the `N >= 2_097_152` cases.

Recommended benchmark checks:

1. Compare old vs new warm `prepare_state(...)` peak memory.
2. Compare wall time at `524288`, `1048576`, and `2097152`.
3. Check whether `4194304` becomes feasible on the H200 with the lean notebook config.

## Next Optimization Tracks

### 2. Prepared-state slimming

Goal:

- Reduce retained or long-lived prepared-state memory after downward prep without changing runtime behavior or accuracy.

Current progress:

- Implemented a conservative first pass: in minimum-memory mode, `FMMPreparedState` now stores `upward=None` and `octree_upward=None` instead of retaining the full upward bundles.
- This looks safe from a code-usage perspective because the prepared evaluation paths do not read `state.upward`; advanced source-motion helpers already rebuild what they need from `tree`, `positions_sorted`, and `masses_sorted`.
- A lightweight local compile sanity check passed:
  - `python3 -m py_compile jaccpot/runtime/_fmm_impl.py jaccpot/upward/solidfmm_complex_tree_expansions.py`

Why this matters:

- Even when transient peaks are under control, extra retained state reduces the maximum `N` that fits on device.
- The current minimum-memory path already avoids retaining some traversal artifacts, but upward/prepared payloads may still contain data that plain evaluation does not need.

What to inspect:

- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- The `FMMPreparedState` payload and `_PrepareStateTreeUpwardArtifacts`
- Any path using:
  - `upward.geometry`
  - `upward.mass_moments`
  - `upward.multipoles.centers`
  - `downward.locals`

Concrete plan:

1. Measure the effect of dropping `state.upward` on prepared-state bytes and max feasible `N`.
2. Measure the effect of dropping `octree_upward` alongside `upward`.
3. Audit whether any other retained payloads can be reconstructed on demand:
   - `nearfield_interop`
   - nearfield schedule arrays
   - `interactions` for plain evaluation-only workflows
4. Keep the richer state for derivative-heavy, topology-reuse, or adaptive-order workflows if required.

Likely next split:

- `nearfield_interop` can already be rebuilt on demand in some paths, but dropping it may trade memory for extra repeated setup work.
- The nearfield schedule arrays (`target_leaf_ids`, `source_leaf_ids`, `valid_pairs`, and chunk schedule buffers) are also optional from a correctness perspective, but they are much more likely to affect prepared-evaluation runtime if removed.

Success criteria:

- Reduced prepared-state bytes with no change in output.
- No runtime regression for repeated `evaluate_prepared_state(...)`.
- No breakage of jerk/time-derivative/adaptive-order paths.

Main risk:

- Some advanced paths still reuse geometry or mass-moment data indirectly, so field dropping must be guarded carefully.

### 3. Leaf staging cleanup in local and nearfield evaluation

Goal:

- Reduce dense `(num_leaves, max_leaf_size, ...)` staging pressure during evaluation-side kernels.

Why this matters:

- Once upward memory is reduced, evaluation-side leaf staging may become the next limiter for larger `N`.

Primary files to inspect:

- [`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py)
- [`jaccpot/nearfield/near_field.py`](/export/home/tbuck/jaccpot/jaccpot/nearfield/near_field.py)

Primary functions to inspect:

- `_evaluate_local_expansions_for_particles(...)`
- `_prepare_leaf_data(...)`
- `_prepare_leaf_data_from_groups(...)`
- `_compute_leaf_p2p_impl(...)`
- `_compute_leaf_p2p_from_prepared_leaf_data_impl(...)`

Concrete plan:

1. Measure peak memory for local-expansion evaluation vs nearfield P2P separately.
2. Identify where dense leaf particle tensors are created eagerly.
3. Introduce chunked or fused gathering where it cuts staging size without increasing scatter overhead too much.
4. Prefer changes that keep prepared-state reuse intact.

Success criteria:

- Lower evaluation-time peak memory.
- No accuracy changes.
- No runtime regression on repeated prepared-state evaluation.

Main risk:

- Over-chunking nearfield can hurt throughput if it increases scatter or launch overhead too much.

## Recommended Execution Order

1. Verify the new streamed P2M change on the other GPU server.
2. Record that the H200 now reaches `4_194_304` particles in the single-`N`
   notebook with the lean minimum-memory path.
3. Re-run the updated prepare-stage split and capture the new:
   - `dual_tree_split_far_only_*`
   - `dual_tree_split_near_only_*`
   rows
4. If `near_only` dominates, optimize the Yggdrax near-neighbor builder first.
5. If `far_only` dominates, continue on the compact far-pair builder first.
6. Only after lowering the warm split-build transient should prepared-state
   slimming or evaluation leaf staging become the primary track.

## Updated H200 Conclusions

The current `N=4_194_304` H200 single-`N` run now shows:

- prepare cold peak delta: about `12.83 GB`
- prepare warm peak delta: about `8.65 GB`
- evaluate warm peak delta: about `1.37 GB`
- prepared state size: about `215.84 MB`

Interpretation:

- the H200 fit ceiling moved materially because the old catastrophic traversal
  build path was reduced enough to fit
- the next ceiling is still prepare-side transient memory, not retained state
- the dominant remaining target is the warm split traversal build in Yggdrax,
  not Jaccpot prepared-state storage

Most important concrete lesson:

- at multi-million-particle scale, the problem is no longer primarily
  compile-time overhead
- the problem is the warm transient footprint of the split dual-tree build

## Fast Resume Checklist

When returning to this work:

1. Open this file.
2. Re-run:
   - `tests/unit/core/test_solidfmm_complex_tree_expansions.py`
   - the large-`N` notebook/runtime sweep
3. Record:
   - max successful `N`
   - warm prepare peak memory
   - evaluate peak memory
   - runtime at `524288`, `1048576`, `2097152`, and `4194304`
4. If upward memory is no longer dominant, move to prepared-state slimming.

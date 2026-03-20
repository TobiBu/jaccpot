# Runtime Traversal Comparison Notes

Date: 2026-03-20

## Summary

The remaining mismatch between pure `yggdrax` traversal timing and
`jaccpot.prepare_state()` timing is now understood well enough for practical
comparison work.

The key conclusion is:

- standalone `yggdrax` traversal should be compared against the traversal-like
  portion inside `jaccpot`, not against total `prepare_state()` wall time
- after deeper profiling, the extra `prepare_state()` time is mostly explained
  by non-traversal support work rather than a large hidden traversal penalty
- the residual unexplained gap is now small

## What Was Measured

The profiling used:

- [`examples/compare_yggdrax_jaccpot_prepare.py`](/export/home/tbuck/jaccpot/examples/compare_yggdrax_jaccpot_prepare.py)
  for broad cross-checks
- [`examples/profile_prepare_residuals.py`](/export/home/tbuck/jaccpot/examples/profile_prepare_residuals.py)
  for deeper `prepare_state()` decomposition

The deeper profiler times:

- low-level tree build
- upward total, plus `P2M` and `M2M`
- dual-tree artifact build
- downward total, plus `M2L` and `L2L`
- nearfield interop and nearfield precompute helpers
- octree-specific tail hooks when present

## GPU 3 Results

Environment used for the final run:

- `micromamba` environment: `odisseo`
- GPU selection: `CUDA_VISIBLE_DEVICES=3`
- command:

```bash
CUDA_VISIBLE_DEVICES=3 JAX_ENABLE_X64=1 micromamba run -n odisseo python \
  examples/profile_prepare_residuals.py \
  --num-particles 65536 131072 262144 \
  --warmup 1 \
  --runs 3 \
  --json
```

Measured radix-backend results:

| particles | prepare_total_ms | dual_tree_artifacts_ms | downward_non_operator_est_ms | nearfield_total_ms | residual_unexplained_ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 303.7 | 47.5 | 159.8 | 7.5 | 6.0 |
| 131,072 | 403.7 | 57.2 | 158.2 | 8.4 | 5.3 |
| 262,144 | 673.3 | 125.4 | 153.7 | 9.6 | 10.0 |

Interpretation:

- the dominant non-operator costs are dual-tree artifact construction and
  downward support work
- nearfield work is present but comparatively small in this configuration
- the remaining unexplained residual is small enough that the comparison is now
  operationally trustworthy

## Traversal Config Follow-Up

After the symmetry-path optimization, the next hotspot was
`dual_tree_artifacts_ms`. A focused GPU 3 comparison at `65,536` particles
showed that this bucket is almost entirely raw traversal build time, so the
practical lever is traversal configuration rather than Jaccpot-side packaging.

Compared configs for:

```text
num_particles=65536
tree_type=radix
execution_backend=radix
theta=0.6
leaf_size=128
max_order=4
```

| max_pair_queue | process_block | max_interactions_per_node | max_neighbors_per_leaf | dual_raw_build_ms | prepare_total_ms | status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 262144 | 256 | 8192 | 4096 | 49.72 | 161.50 | stable |
| 131072 | 128 | 8192 | 2048 | 44.34 | 159.38 | stable |
| 65536 | 64 | 4096 | 1024 | 38.15 | 149.69 | stable |

Takeaway for this runtime target:

- the larger-cap `262144 / 256 / 8192 / 4096` traversal seed is not optimal for
  `65,536` particles on GPU 3
- the older `131072 / 128 / 8192 / 2048` recommendation is better for this
  case
- an even leaner `65536 / 64 / 4096 / 1024` traversal config was stable and
  materially faster for this case

This does **not** yet replace the broader production recommendations for larger
`N`. It does suggest the runtime notes should distinguish:

- large-`N` stability-oriented seeds
- smaller-`N` prepare/runtime-optimal seeds

Additional GPU 3 follow-up at larger `N`:

### `num_particles=131072`

| max_pair_queue | process_block | max_interactions_per_node | max_neighbors_per_leaf | dual_raw_build_ms | M2L_ms | prepare_total_ms | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 262144 | 256 | 8192 | 4096 | 65.22 | 128.91 | 275.79 | stable |
| 131072 | 128 | 8192 | 2048 | 68.19 | 140.11 | 290.06 | stable |
| 65536 | 64 | 4096 | 1024 | 51.88 | 153.34 | 282.42 | stable |

Interpretation:

- the leaner config reduces raw traversal time
- but it increases downstream `M2L` enough that total prepare time gets worse
- for this case, the larger-cap `262144 / 256 / 8192 / 4096` seed is the best of
  the tested options

### `num_particles=262144`

| max_pair_queue | process_block | max_interactions_per_node | max_neighbors_per_leaf | dual_raw_build_ms | M2L_ms | prepare_total_ms | status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 262144 | 256 | 8192 | 4096 | 134.04 | 313.95 | 561.67 | stable |
| 131072 | 128 | 8192 | 2048 | — | — | — | overflow |
| 65536 | 64 | 4096 | 1024 | — | — | — | overflow |

Interpretation:

- by `262144` particles, the reduced-cap seeds are no longer stable under
  `fail_fast=True`
- the larger-cap `262144 / 256 / 8192 / 4096` seed is required among the tested
  options

Updated practical guidance:

- `65536`:
  the leaner `65536 / 64 / 4096 / 1024` seed is a good prepare-time target
- `131072`:
  use the larger-cap `262144 / 256 / 8192 / 4096` seed for best total prepare
  time among the tested options
- `262144`:
  the larger-cap `262144 / 256 / 8192 / 4096` seed is the only stable tested
  option among these three candidates

So the tuning should be documented as size-dependent rather than as a single
globally best traversal seed.

## Practical Guidance

Use these comparisons carefully:

- valid:
  pure `yggdrax` traversal vs `jaccpot` dual-tree/traversal-side buckets
- not valid:
  pure `yggdrax` traversal vs full `jaccpot.prepare_state()` total

In other words, `prepare_state()` remains a wider pipeline than traversal alone,
but the traversal-related runtime gap is no longer meaningfully mysterious.

## Bug Found During Profiling

This work also exposed a real nearfield regression in
[`jaccpot/runtime/_fmm_impl.py`](/export/home/tbuck/jaccpot/jaccpot/runtime/_fmm_impl.py):

- `_prepare_nearfield_precompute_artifacts()` referenced
  `retain_pair_vectors` without defining it locally
- the helper now receives and resolves that policy correctly

That fix was necessary to get stable profiling results on the current code path.

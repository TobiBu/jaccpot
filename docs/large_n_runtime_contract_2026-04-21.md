# Large-N Runtime Contract (2026-04-21)

## Canonical Production Path

For `preset="large_n_gpu"`, Jaccpot now canonicalizes runtime behavior to a single production path:

- `runtime_path = "large_n"`
- `memory_objective = "minimum_memory"`
- `farfield_mode = "pair_grouped"`
- `grouped_interactions = False`
- `streamed_far_pairs = True`
- `nearfield_mode = "bucketed"`
- radix fast-lane nearfield is used for acceleration evaluation

## Deprecation Notes

- `runtime_path="legacy"` is deprecated and will be removed.
- Conflicting large-N production overrides are accepted for compatibility but coerced to the canonical production values above.
- Oversized explicit traversal seeds are capped on the large-N production GPU path to avoid memory regressions.

## Benchmark Guidance

Use canonical large-N config helpers in `examples/benchmark_utils.py`.
Avoid pinning oversized explicit traversal settings in notebooks.


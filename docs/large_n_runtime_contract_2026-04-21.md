# Large-N Runtime Contract (2026-04-21)

## Canonical Production Path

For `preset="large_n_gpu"`, Jaccpot now canonicalizes runtime behavior to a single production path:

- `runtime_path = "large_n"` (since the 2026-10 cleanup, X4, a recorded value only: the
  preset alone selects the large-N lane, and `runtime_path="large_n"` under another
  preset no longer opens it)
- `memory_objective = "minimum_memory"`
- `farfield_mode = "pair_grouped"`
- `grouped_interactions = False`
- `streamed_far_pairs = True`
- `nearfield_mode = "bucketed"`
- radix fast-lane nearfield is used for acceleration evaluation

## Deprecation Notes

- `runtime_path="legacy"` was removed: only `"auto"` and `"large_n"` are accepted, and
  since the 2026-10 cleanup (X4) neither selects a lane.
- Conflicting large-N production overrides are accepted for compatibility but coerced to the canonical production values above.
- Auto-sized (preset) traversal seeds are capped on the large-N production GPU path to
  avoid memory regressions; an explicit `traversal_config` is used as given. Sizing is
  static: adaptive sizing went in the 2026-10 cleanup (X4).

## Benchmark Guidance

Use canonical large-N config helpers in `examples/benchmark_utils.py`.
Avoid pinning oversized explicit traversal settings in notebooks.


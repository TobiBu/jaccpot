# Large-N Runtime Contract (2026-04-21)

## Canonical Production Path

For `preset="large_n_gpu"` with a radix tree and the real (`"solidfmm"`) basis,
`FMMEngine._apply_large_n_gpu_production_contract` canonicalizes runtime behavior to a
single production path. The call shape is the preset alone:

```python
FastMultipoleMethod(preset="large_n_gpu", precision="fp32")
```

and the contract sets:

- `memory_objective = "minimum_memory"`
- `streamed_far_pairs = True`
- `nearfield_mode = "bucketed"`, `precompute_nearfield_scatter_schedules = False`
- mixed-order far field off
- the interaction cache on; the traversal result and the interaction lists not retained
- the preset's tree build `tree_build_mode = "static_radix"` (since the 2026-10
  cleanup, D2: the fused strict lane needs it)
- radix fast-lane near field for acceleration evaluation

The large-N lane itself (`can_use_large_n_prepare_path`) is gated on the preset alone.
The far field has one mode, a flat pair list about the nodes' centres of mass; there is
nothing to select.

## Deprecation Notes

- The 2026-10 cleanup removed the options this contract used to pin: `runtime_path`
  (X4 stopped it selecting a lane, Z-b removed it), the far-field `mode` /
  `farfield_mode` and `grouped_interactions` (X3, Z-b). Passing one now fails like any
  unknown keyword or field.
- Conflicting large-N production overrides (`memory_objective`, `nearfield_mode`,
  `streamed_far_pairs`) are accepted with a `FutureWarning` and coerced to the canonical
  values above.
- Auto-sized (preset) traversal seeds are capped on the large-N production GPU path to
  avoid memory regressions; an explicit `traversal_config` is used as given. Sizing is
  static: adaptive sizing went in the 2026-10 cleanup (X4).

## Benchmark Guidance

Use canonical large-N config helpers in `examples/benchmark_utils.py`.
Avoid pinning oversized explicit traversal settings in notebooks.


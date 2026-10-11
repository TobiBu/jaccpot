# Example Guide

The `examples/` directory keeps a small set of notebooks and scripts, each with a clear
job.

## Derivative Notebooks

- [`time_derivatives_demo.ipynb`](time_derivatives_demo.ipynb)
  API walkthrough plus direct-sum sanity checks for jerk, snap, and crackle.
- [`jerk_modes_demo.ipynb`](jerk_modes_demo.ipynb)
  Compare jerk `fast_approx` vs `accurate`, including analytic source-motion
  behavior and a small timing check.

## Focused Feature Notebook

- [`real_sh_adaptive_order.ipynb`](real_sh_adaptive_order.ipynb)
  Compact check for real-basis adaptive-order behavior.

## Benchmark Notebooks (records; they need rework to run)

These hold the outputs of earlier measurements. Most predate the 2026-10 cleanup
([`docs/cleanup_2026-10.md`](../docs/cleanup_2026-10.md), Z-c) and no longer run end to
end: they drive the deleted `benchmark_gpu_radix_worker.py` /
`profile_prepare_memory_split.py`, the removed complex basis, or private prepare helpers
whose signatures changed.

- [`adaptive_vs_fixed_benchmark.ipynb`](adaptive_vs_fixed_benchmark.ipynb)
  Adaptive-order vs fixed-order configurations.
- [`benchmark_runtime_accuracy_new.ipynb`](benchmark_runtime_accuracy_new.ipynb),
  [`benchmark_runtime_accuracy_copy.ipynb`](benchmark_runtime_accuracy_copy.ipynb)
  Broad runtime, accuracy, and prepared-state checks.
- [`benchmark_gpu_radix_runtime.ipynb`](benchmark_gpu_radix_runtime.ipynb)
  GPU/radix runtime: memory pressure, prepare/evaluate breakdowns, runtime tuning.
- [`benchmark_gpu_single_n_memory.ipynb`](benchmark_gpu_single_n_memory.ipynb)
  Single-`N` interactive GPU memory probe.
- `benchmark_runtime_large_N*.ipynb`, `benchmark_runtime_rtx2080_focused.ipynb`,
  `benchmark_runtime_breakdown_h200.ipynb`: large-`N` and per-GPU studies.

## Scripts

- [`pallas_m2l_speed.py`](pallas_m2l_speed.py): pure-JAX vs Pallas real-basis M2L
  timing (on CPU it reports the fallback).
- [`real_sh_rot_scale_demo.py`](real_sh_rot_scale_demo.py): the real-basis
  rotate+scale FMM against the direct sum.
- [`reuse_topology_demo.py`](reuse_topology_demo.py): topology and pair reuse across
  small particle motions.
- [`differentiable_fmm_overhead.py`](differentiable_fmm_overhead.py): the autodiff
  overhead of the differentiable FMM force.
- [`profile_prepare_stages.py`](profile_prepare_stages.py),
  [`benchmark_memory_pipeline.py`](benchmark_memory_pipeline.py),
  [`compare_yggdrax_jaccpot_prepare.py`](compare_yggdrax_jaccpot_prepare.py):
  prepare / evaluate stage timings and memory.
- [`run_in_odisseo_with_autocvd.py`](run_in_odisseo_with_autocvd.py): run a repo-local
  command with the standard GPU validation setup; [`benchmark_utils.py`](benchmark_utils.py):
  shared helpers.

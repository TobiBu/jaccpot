# Example Guide

The `examples/` directory now aims to keep a small set of high-signal notebooks
and scripts, each with a clear job.

## Benchmark Notebooks

- [`benchmark_runtime_accuracy.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_runtime_accuracy.ipynb)
  Main FMM status notebook. Use this first for broad runtime, accuracy, and
  prepared-state behavior checks.
- [`adaptive_vs_fixed_benchmark.ipynb`](/export/home/tbuck/jaccpot/examples/adaptive_vs_fixed_benchmark.ipynb)
  Focused comparison of adaptive-order and fixed-order configurations.
- [`benchmark_gpu_radix_runtime.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_radix_runtime.ipynb)
  Deep GPU/radix runtime notebook for memory pressure, prepare/evaluate
  breakdowns, and low-level runtime tuning.
- [`benchmark_gpu_single_n_memory.ipynb`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_single_n_memory.ipynb)
  Single-`N` interactive memory probe. Keep this as a notebook when you want
  direct access to plots and tables while debugging GPU memory behavior.

## Derivative Notebooks

- [`time_derivatives_demo.ipynb`](/export/home/tbuck/jaccpot/examples/time_derivatives_demo.ipynb)
  API walkthrough plus direct-sum sanity checks for jerk, snap, and crackle.
- [`jerk_modes_demo.ipynb`](/export/home/tbuck/jaccpot/examples/jerk_modes_demo.ipynb)
  Compare jerk `fast_approx` vs `accurate`, including analytic source-motion
  behavior and a small timing check.

## Focused Feature Notebook

- [`real_sh_adaptive_order.ipynb`](/export/home/tbuck/jaccpot/examples/real_sh_adaptive_order.ipynb)
  Compact check for real-basis adaptive-order behavior.

## Benchmark Scripts

- [`benchmark_gpu_n_ladder_production.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_n_ladder_production.py)
  Production-oriented large-`N` parameter sweep. Writes recommendation tables to
  [`benchmarks/n_ladder_production/`](/export/home/tbuck/jaccpot/benchmarks/n_ladder_production).
- [`profile_prepare_residuals.py`](/export/home/tbuck/jaccpot/examples/profile_prepare_residuals.py)
  Command-line profiler for breaking down `prepare_state()` residual runtime and
  separating traversal-like cost from surrounding support work.

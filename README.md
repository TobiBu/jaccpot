# jaccpot

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![isort](https://img.shields.io/badge/imports-isort-1674b1.svg)
![pytest](https://img.shields.io/badge/tests-pytest-0a9edc.svg)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

<p align="center">
  <img src="./jaccpot.png" alt="jaccpot Logo" width="420" />
</p>

`jaccpot` is a JAX-first Fast Multipole Method (FMM) package for N-body gravity
and related hierarchical interaction problems. It provides multipole/local
expansion operators, near/far-field kernels, and a preset-driven high-level
solver API.

Tree construction and traversal artifacts are provided by the companion package
`yggdrax`.

## Features

- High-level `FastMultipoleMethod` API with `fast`, `balanced`, `accurate`, and `large_n_gpu` presets
- Configurable expansion basis — **`real` (Dehnen) is the default** production basis (the radix large-N fast lane runs pure-real end to end, no complex↔real conversion); `complex`/`solidfmm` are retained for cross-checking; `cartesian` also available
- Pure-JAX real spherical harmonic rotate+scale M2L path
- Adaptive-order far-field evaluation with fixed `p_gears` buckets
- Optional topology reuse for multiple nearby timesteps
- Optional Pallas acceleration for the real-basis z-translation hotspot
- Modular runtime with grouped/dense interaction pathways
- Near-field and far-field execution paths with optional prepared state reuse
- Explicit octree execution backend for `basis="solidfmm"`
- Differentiable gravitational acceleration helper via JAX autodiff

## Installation

Install from source:

```bash
pip install -e .
```

`yggdrax` is not on PyPI yet. Install it from GitHub first (use the latest
`main`, which includes native `RadixTree` JAX pytree registration):

```bash
git clone https://github.com/TobiBu/yggdrax.git
cd yggdrax
pip install -e .
cd ..
```

Install with development tooling:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
import jax.numpy as jnp

from jaccpot import FastMultipoleMethod

key = jax.random.PRNGKey(0)
key_pos, key_mass = jax.random.split(key)
positions = jax.random.uniform(key_pos, (1024, 3), minval=-1.0, maxval=1.0)
masses = jax.random.uniform(key_mass, (1024,), minval=0.5, maxval=1.5)

# basis defaults to "real" (Dehnen); pass basis="solidfmm" only to cross-check.
solver = FastMultipoleMethod(preset="balanced")
accelerations = solver.compute_accelerations(positions, masses)
print(accelerations.shape)
```

Real-basis rotate+scale FMM uses the same high-level API:

```python
solver = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    adaptive_order=True,
    p_gears=(2, 3, 4),
)
accelerations = solver.compute_accelerations(positions, masses, max_order=4)
```

For very large single-GPU runs, use the dedicated throughput/memory preset:

```python
solver = FastMultipoleMethod(
    preset="large_n_gpu",
    basis="solidfmm",
    precision="fp32",
)
```

`large_n_gpu` keeps JIT enabled while applying memory-oriented runtime defaults
for large particle-count single-GPU runs (streamed far-pair mode, reduced
near-field precompute retention, and cache retention disabled).

For split-step integrators (for example active-particle substeps), you can
evaluate only a subset while still using all particles as FMM sources:

```python
active = jnp.asarray([0, 7, 11, 32], dtype=jnp.int32)
state = solver.prepare_state(positions, masses)
active_acc = solver.evaluate_prepared_state(state, target_indices=active)
```

For integrators that require jerk, use:

```python
velocities = jax.random.uniform(key_pos, (1024, 3), minval=-0.2, maxval=0.2)
acc, jerk = solver.compute_accelerations_and_jerk(
    positions,
    masses,
    velocities,
    jerk_mode="fast_approx",  # or "accurate"
)
```

See [docs/derivatives_and_jerk.md](docs/derivatives_and_jerk.md) for API details,
mode tradeoffs, and output tensor layouts.
Current higher-order status:

- jerk is available via `compute_accelerations_and_jerk(...)`
- higher total time derivatives are available via
  `compute_accelerations_with_time_derivatives(...)`
- public time-derivative support currently reaches crackle
  (`max_time_derivative_order=3`)
- the general time-derivative API currently uses the analytic `accurate` path
- orders above crackle are not yet available
- acceleration spatial derivatives (`max_acc_derivative_order > 0`) currently
  require `basis="solidfmm"`

There is also a worked example notebook for jerk, snap, and crackle, including
a small-`N` direct-sum accuracy check:
[`examples/time_derivatives_demo.ipynb`](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/time_derivatives_demo.ipynb).

### Jerk Mode Guide

| Goal | Mode | Notes |
|---|---|---|
| Lowest runtime overhead | `fast_approx` | Exact near-field jerk + far-field convective term. |
| Highest fidelity (includes source-motion effects) | `accurate` | Analytic far-field source-motion term + convective + exact near-field jerk. |
| Stable default for large production runs | `fast_approx` | Benchmark against your own workload before switching defaults. |

For ODISSEO-style primitive states `(N, 2, 3)`, you can use the adapter:

```python
from jaccpot import OdisseoFMMCoupler

coupler = OdisseoFMMCoupler(solver, leaf_size=16, max_order=4)
coupler.prepare(primitive_state, masses)  # full source tree
acc_active = coupler.accelerations(primitive_state, active_indices=active)
```

## Octree Backend

The default runtime path remains radix-oriented. To request explicit octree
execution, configure both the tree type and runtime backend:

```python
from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    RuntimePolicyConfig,
    TreeConfig,
)

solver = FastMultipoleMethod(
    preset="fast",
    basis="solidfmm",
    advanced=FMMAdvancedConfig(
        tree=TreeConfig(tree_type="octree"),
        runtime=RuntimePolicyConfig(execution_backend="octree"),
    ),
)
```

Current practical scope:

- the octree backend is validated for `basis="solidfmm"`
- prepared-state evaluation supports full outputs, target subsets, potentials,
  JIT/eager traversal, and prepared-state cache reuse
- non-default runtime modes such as baseline nearfield and class-major
  farfield are covered in the solver tests

Still worth keeping in mind:

- `execution_backend="auto"` may still resolve to the radix backend
- topology reuse remains radix-only
- validation is currently most reliable on the preferred project validation GPU

Example:

- [examples/compare_yggdrax_jaccpot_prepare.py](examples/compare_yggdrax_jaccpot_prepare.py)

## Basis Selection

- `basis="complex"` or `basis="solidfmm"`:
  default complex solidFMM-compatible path
- `basis="real"`:
  real spherical harmonic coefficient layout with rotate+scale-to-z M2L
- `basis="cartesian"`:
  cartesian multipole/local expansion path

The default remains the existing complex solidFMM-compatible path.

## Precision Control

Use `precision` to select runtime dtype explicitly:

```python
solver_fp32 = FastMultipoleMethod(
    preset="fast",
    basis="solidfmm",
    precision="fp32",
)

solver_fp64 = FastMultipoleMethod(
    preset="accurate",
    basis="solidfmm",
    precision="fp64",
)
```

`precision="fp64"` requires `jax_enable_x64=True`. You can still pass
`working_dtype` directly; if both are set, they must match.

## Adaptive Order

Use `adaptive_order=True` together with a static gear list:

```python
solver = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    adaptive_order=True,
    p_gears=(2, 3, 4),
)
```

`p_gears` must be a fixed tuple or list of orders. This keeps all hot paths
JIT-friendly and avoids shape polymorphism.

Adaptive order selection now uses yggdrax's generic `pair_policy` +
`interaction_tags` traversal hook. The tree backend only provides generic
far-pair tags; jaccpot owns the solver-side policy state, order selection, and
per-order bucketing.

The current adaptive acceptance model is solver-owned and error-aware:
- acceptance uses the highest available order as a Dehnen-style safety check
- accepted pairs are limited to a relaxed geometric cone to avoid pathological
  over-acceptance
- once accepted, the solver picks the smallest passing order from `p_gears`

In other words, the highest candidate order decides whether a pair is safe to
accept, while the first passing order decides how much far-field work is needed.
The current notebook example prints the resulting tag-derived
`far_pairs_by_gear_counts` from the solver runtime.

Adaptive traversal currently has two practical runtime modes:

- `adaptive_error_model="tail_proxy"` (default):
  the validated high-performance mode; this remains the recommended default
  when runtime matters most
- `adaptive_error_model="dehnen_paper"`:
  the paper-inspired comparison mode; for JAX-native runs pair it with
  `dehnen_geometry_mode="tree_approx"`

Other available knobs:

- `adaptive_error_model="dehnen_degree"`:
  a simplified degree-resolved Dehnen-style source-power estimator
- `dehnen_geometry_mode="exact"`:
  exact reference geometry for paper comparisons; not a throughput mode
- `dehnen_geometry_mode="tree_approx"`:
  JAX-native paper geometry based on approximate leaf spheres plus upward
  merged spheres
- `adaptive_eps=...`:
  override the default theta-derived adaptive tolerance with a direct
  solver-side tolerance scale

Examples:

- [examples/real_sh_adaptive_order.ipynb](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/real_sh_adaptive_order.ipynb)
- [examples/real_sh_rot_scale_demo.py](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/real_sh_rot_scale_demo.py)

## Reproducible Comparison Modes

Use the following three solver configurations for reproducible comparisons on this branch.
Keep the remaining benchmark settings fixed, for example:

- `preset="accurate"`
- `basis="real"`
- `theta=0.6`
- `leaf_size=16`
- `max_order=4`
- `p_gears=(2, 3, 4)` for adaptive runs
- enlarged traversal caps, as used in `examples/adaptive_vs_fixed_benchmark.ipynb`

Fixed non-adaptive baseline:

```python
fixed = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    theta=0.6,
    adaptive_order=False,
)
```

Adaptive high-performance default:

```python
tail_proxy = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    theta=0.6,
    adaptive_order=True,
    p_gears=(2, 3, 4),
    adaptive_error_model="tail_proxy",
    mac_force_scale_mode="prev",
)
```

Adaptive paper-inspired JAX-native mode:

```python
dehnen_paper = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    theta=0.6,
    adaptive_order=True,
    p_gears=(2, 3, 4),
    adaptive_error_model="dehnen_paper",
    adaptive_eps=1.0e-3,
    dehnen_geometry_mode="tree_approx",
    mac_force_scale_mode="paper",
)
```

Interpretation on the current branch:

- `fixed`: fastest non-adaptive baseline
- `tail_proxy`: best validated adaptive runtime default
- `dehnen_paper`: higher-accuracy, paper-inspired comparison mode

## Force Scale Modes For Adaptive Traversal

Adaptive traversal can weight its solver-side policy state with per-node force
scales. Select how those scales are estimated with `mac_force_scale_mode`:

- `"prev"`:
  reuse the previous full-step per-node force-scale estimate (`self._last_force_scale_nodes`).
  This is the cheapest option and is the practical default for `tail_proxy`.
- `"prepass"`:
  run a cheap lowest-order prepass for the current configuration and derive force
  scales from that pass.
- `"paper"`:
  run the stricter paper-style current-step prepass used by
  `adaptive_error_model="dehnen_paper"`.

Interpretation:

- `prev` is a runtime-oriented reuse mode.
- `paper` is the more publication/reference-oriented mode because it derives the
  threshold from a dedicated current-step prepass rather than from historical state.
- `prepass` sits between the two as a generic current-step estimate that is not
  specifically tied to the paper-style Dehnen path.

These scales stay inside jaccpot's adaptive policy state; they are no longer
exported as backend-specific traversal `node_features`.

## Pallas Acceleration

The real-basis z-translation core can be accelerated with Pallas:

```python
solver = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    use_pallas=True,
)
```

Current behavior:

- rotations stay in pure JAX
- only the real-basis z-axis M2L core is offloaded
- unsupported backends fall back to the pure-JAX kernel automatically

On the current `expanse` CPU environment, the example reports fallback rather
than true Pallas execution:

- [examples/pallas_m2l_speed.py](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/pallas_m2l_speed.py)

## Topology Reuse

For small multi-step particle motion, you can reuse cached topology and
interaction lists for a bounded number of steps:

```python
solver = FastMultipoleMethod(
    preset="accurate",
    basis="real",
    reuse_topology=True,
    rebuild_every=3,
)
```

The solver always recomputes reordered particles, geometry, upward multipoles,
and downward locals for the current state. Reuse only applies to cached
topology/traversal artifacts when the Morton ordering key remains unchanged.

Example:

- [examples/reuse_topology_demo.py](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/reuse_topology_demo.py)

## Development

Run quality gates locally:

```bash
black --check .
isort --check-only .
pytest
```

Or run pre-commit hooks:

```bash
pre-commit run --all-files
```

Coverage is enforced in CI via `pytest-cov`:

```bash
pytest --cov=jaccpot --cov-report=term-missing
```

### Performance Guard

CI also runs a benchmark regression guard based on:

- [bench/bench_parallel_paths.py](/Users/buck/Documents/Nexus/Projects/jaccpot/bench/bench_parallel_paths.py)
- [bench/ci_benchmark_guard.py](/Users/buck/Documents/Nexus/Projects/jaccpot/bench/ci_benchmark_guard.py)
- [bench/benchmark_baseline.json](/Users/buck/Documents/Nexus/Projects/jaccpot/bench/benchmark_baseline.json)

Run the lightweight runtime-path benchmark and CI guard locally:

```bash
python -m bench.bench_parallel_paths --n 512 --runs 3 --warmup 1
python -m bench.ci_benchmark_guard --n 384 --runs 2 --warmup 1
```

If a performance change is intentional, refresh the baseline:

1. Run `bench/bench_parallel_paths.py` with the CI benchmark arguments.
2. Read the `timings_s` line and update:
   `target_eval_mean_s` and `adaptive_prepare_mean_s` in
   `bench/benchmark_baseline.json`.
3. Re-run `bench/ci_benchmark_guard.py` to confirm the new baseline passes.

## Examples

- `examples/benchmark_runtime_accuracy.ipynb`: main runtime/accuracy benchmark workflow
- `examples/adaptive_vs_fixed_benchmark.ipynb`: adaptive-order vs fixed-order comparison
- `examples/benchmark_gpu_radix_runtime.ipynb`: GPU/radix runtime and memory-pressure deep dive
- `examples/benchmark_gpu_single_n_memory.ipynb`: interactive single-`N` GPU memory probe with plots/tables
- `examples/benchmark_gpu_n_ladder_production.py`: production-oriented large-`N` parameter sweep
- `examples/time_derivatives_demo.ipynb`: usage plus direct-sum accuracy checks for jerk, snap, and crackle
- `examples/jerk_modes_demo.ipynb`: compare jerk `fast_approx` vs `accurate`, including analytic source-motion behavior
- `examples/real_sh_adaptive_order.ipynb`: real-basis adaptive-order demo

## Runtime Type Checking

`jaccpot` can enable package-wide runtime checking for annotated callables using
`jaxtyping` + `beartype` at import time.

- Disabled by default.
- Enable when needed with:

```bash
export JACCPOT_RUNTIME_TYPECHECK=1
```

## Project Structure

- `jaccpot/solver.py`: preset-first user-facing FMM API
- `jaccpot/config.py`: config model for solver/runtime knobs
- `jaccpot/runtime`: execution internals and integration with yggdrax artifacts
- `jaccpot/operators`: harmonic, translation, and multipole operators
- `jaccpot/upward`, `jaccpot/downward`, `jaccpot/nearfield`: sweep and near-field modules
- `tests`: unit, integration, and performance checks

## CI

GitHub Actions runs:

- formatter checks (`black`, `isort`)
- unit/integration tests with coverage threshold
- release build and PyPI publish on version tags

Workflow files:

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`

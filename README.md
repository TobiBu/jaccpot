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

- High-level `FastMultipoleMethod` API with `fast`, `balanced`, and `accurate` presets
- Configurable expansion basis (`complex`/`solidfmm`, `real`, `cartesian`)
- Pure-JAX real spherical harmonic rotate+scale M2L path
- Adaptive-order far-field evaluation with fixed `p_gears` buckets
- Optional topology reuse for multiple nearby timesteps
- Optional Pallas acceleration for the real-basis z-translation hotspot
- Modular runtime with grouped/dense interaction pathways
- Near-field and far-field execution paths with optional prepared state reuse
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

solver = FastMultipoleMethod(preset="balanced", basis="solidfmm")
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

For split-step integrators (for example active-particle substeps), you can
evaluate only a subset while still using all particles as FMM sources:

```python
active = jnp.asarray([0, 7, 11, 32], dtype=jnp.int32)
state = solver.prepare_state(positions, masses)
active_acc = solver.evaluate_prepared_state(state, target_indices=active)
```

For ODISSEO-style primitive states `(N, 2, 3)`, you can use the adapter:

```python
from jaccpot import OdisseoFMMCoupler

coupler = OdisseoFMMCoupler(solver, leaf_size=16, max_order=4)
coupler.prepare(primitive_state, masses)  # full source tree
acc_active = coupler.accelerations(primitive_state, active_indices=active)
```

## Basis Selection

- `basis="complex"` or `basis="solidfmm"`:
  default complex solidFMM-compatible path
- `basis="real"`:
  real spherical harmonic coefficient layout with rotate+scale-to-z M2L
- `basis="cartesian"`:
  cartesian multipole/local expansion path

The default remains the existing complex solidFMM-compatible path.

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

Examples:

- [examples/real_sh_adaptive_order.ipynb](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/real_sh_adaptive_order.ipynb)
- [examples/real_sh_rot_scale_demo.py](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/real_sh_rot_scale_demo.py)

## Force Scale Modes For Adaptive Traversal

Adaptive traversal can weight its solver-side policy state with per-node force
scales. Select how those scales are estimated with `mac_force_scale_mode`:

- `"prev"`:
  reuse the previous full-step acceleration magnitudes
- `"prepass"`:
  run a cheap lowest-order prepass and derive force scales from that pass

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

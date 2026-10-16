# jaccpot

![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
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

- High-level `FastMultipoleMethod` API with `fast`, `balanced`, `accurate`, `large_n_gpu` and `galaxy` presets
- Real (Dehnen) spherical-harmonic expansions, the only basis (`"solidfmm"` is its family name; the complex and Cartesian bases were removed in the 2026-10 cleanup)
- Pure-JAX real spherical harmonic rotate+scale M2L path
- Adaptive-order far-field evaluation with fixed `p_gears` buckets
- Optional topology reuse for multiple nearby timesteps
- Pallas (Triton) kernels for the hot loops on Ampere+ GPUs, with pure-JAX fallbacks on older GPUs and CPU (see [Pallas Acceleration](#pallas-acceleration))
- Modular runtime with flat interaction pathways, streamed or materialised
- Near-field and far-field execution paths with optional prepared state reuse
- **End-to-end differentiable FMM force** — exact `jax.grad`/`jax.vjp` gradients w.r.t. positions and masses at fixed topology, verified from N=64 to N=1,000,000 (see [Differentiable FMM](#differentiable-fmm))
- **Momentum-conserving (mutual) force** for block-step individual timesteps — `sum_i m_i a_i` cancels to round-off instead of to the truncation error (see [Momentum-Conserving FMM](#momentum-conserving-fmm))
- **The fused lane's per-step force as an object** — `FusedForceProvider` runs the force `strict_run_v2` steps with inside your own jitted integrator, with `G` and the softening as compile-time literals or traced arrays (see [The fused lane's force in your own integrator](#the-fused-lanes-force-in-your-own-integrator))
- Differentiable direct-sum helper via JAX autodiff, retained as the exact-gradient oracle

## Installation

Install from source:

```bash
pip install -e .            # CPU
pip install -e ".[cuda]"    # NVIDIA GPU: adds jax's CUDA plugin
```

The base install is CPU-only; GPU runs (the Pallas kernels, the fused lane) need the
`cuda` extra, or a `jax[cuda]` you installed yourself in the same version window.

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
pip install -e ".[dev]"          # add ,cuda for a GPU machine: ".[dev,cuda]"
```

## Requirement: `jax_enable_x64`

jaccpot needs JAX's 64-bit mode. Turn it on at the top of your program, before
creating any arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)  # or set JAX_ENABLE_X64=1
```

The reasons are correctness first, then accuracy:

- The trees are built on 63-bit `uint64` Morton codes (yggdrax). Without x64, JAX
  narrows them to 32 bits with at most a warning, and the trees come out wrong:
  forces were 19 % off (rel-L2) on the float32 lanes, with no error.
- Several accuracy fixes are defined in 64 bits: the float64 centre-of-mass prefix
  sums of large-N trees, the `accum="wide"` near-field accumulator, the float64
  precompute of the rotation constants, and `precision="fp64"`. The multi-GPU
  lanes also use int64 global particle ids.

x64 makes 64-bit types *available*; it does not make jaccpot compute in float64.
The working precision follows your input dtype (or `working_dtype` /
`precision`), so float32 positions and masses stay a float32 run. Arrays you create
yourself without a dtype do become 64-bit under x64, so pass `dtype=jnp.float32`
where you want float32.

`FastMultipoleMethod`, `BlockStepFMM`, `DistributedBlockStepFMM`, `FMMEngine`,
`FusedRollout` and the `jaccpot.distributed` entry points raise a `RuntimeError`
naming the flag when it is off. Until 2026-10 yggdrax switched x64 on when it was
imported, so jaccpot programs had it on without asking for it; neither library
changes JAX's configuration any more.

## Quick Start

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # required, see above

from jaccpot import FastMultipoleMethod

key = jax.random.PRNGKey(0)
key_pos, key_mass = jax.random.split(key)
positions = jax.random.uniform(key_pos, (1024, 3), minval=-1.0, maxval=1.0)
masses = jax.random.uniform(key_mass, (1024,), minval=0.5, maxval=1.5)

# basis defaults to "real" (Dehnen), the only basis.
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
    precision="fp32",
)
```

`large_n_gpu` keeps JIT enabled while applying memory-oriented runtime defaults
for large particle-count single-GPU runs (streamed far-pair mode, reduced
near-field precompute retention, and cache retention disabled).

On a GPU this preset runs the large-N lane, which evaluates all particles'
accelerations (or potentials) and nothing finer: `target_indices`, jerk, the time
derivatives and `max_acc_derivative_order > 0` raise `NotImplementedError` there.
So the two examples below need another preset (`fast`, `balanced` or `accurate`).
On a CPU `large_n_gpu` declines the lane and runs the general path, where they work.

| preset | what it is |
|---|---|
| `fast` (default) | lowest cost; `lbvh` tree |
| `balanced` | middle ground; bucketed near field |
| `accurate` | tightest settings; what the golden suite uses |
| `large_n_gpu` | the large-N GPU lane: static-radix tree, streamed far field, minimum memory, fp32 |
| `galaxy` | the fused lane's production configuration (`jaccpot.production.GalaxySettings`): `large_n_gpu` on cell leaves of 64 (`cell_min_level` 8, `leaf_capacity="auto"`), p6, geometric Dehnen MAC at `theta` 0.8, ferrers3, fp32 -- the benches' L1-L3 configuration, same compiled programs |

`theta` defaults to the preset's: 0.8 for `galaxy`, 0.6 for the others.
`leaf_capacity="auto"` (`TreeConfig`) sizes the cell-leaf tree at the first eager
prepare: 1.15x the live cell leaves of those particles, in steps of 1024, counted on
device; it then stays fixed, so a run that clusters past it raises the capacity error.
The run-time half of `galaxy` -- `leaf_size` 64, `max_order` 6, `carry="particles"` --
is passed to `strict_run_v2` (`GALAXY_SETTINGS` holds it).

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
- acceleration spatial derivatives (`max_acc_derivative_order > 0`) differentiate
  the full acceleration: the far field through its local expansions, the near
  field exactly through its direct sum over the same pairs

There is also a worked example notebook for jerk, snap, and crackle, including
a small-`N` direct-sum accuracy check:
[`examples/time_derivatives_demo.ipynb`](examples/time_derivatives_demo.ipynb).

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

## Removed in the 2026-10 cleanup

The 2026-10 cleanup ([docs/cleanup_2026-10.md](docs/cleanup_2026-10.md)) removed
the complex and Cartesian bases, the octree execution backend, the grouped /
class-major far field, M2L autotune and adaptive sizing, the legacy strict APIs,
the treecode walk and the superseded GPU kernel variants, together with their
options (`runtime_path`, `FarFieldConfig.mode`, `execution_backend`,
`grouped_interactions`, `use_dense_interactions`, `complex_rotation`,
`jerk_fd_dt`, ...). Passing one of them now fails like any unknown keyword or value
(`TypeError` / `ValueError`); a removed `JACCPOT_*` environment value is simply not
read. Radix is the only single-GPU tree.

## Basis Selection

- `basis="real"` (**the default, and the only basis**): real spherical harmonic
  coefficient layout (Dehnen, no sqrt-2) with rotate+scale-to-z M2L. The large-N
  radix fast lane runs pure-real end to end.
- `basis="solidfmm"`: the spherical-harmonic family's name, an alias for `"real"`
  (it resolves to the same basis object and reports `basis == "real"`).

Any other name raises the validator's `ValueError`. The complex basis agreed with
the real one to round-off (4.5e-13 at N=2048/p=4/theta=0.5 in float64), so
`"real"` replaces it as is; the Cartesian basis had a relative L2 force error of
~1.8e-1 *independent of expansion order*, a divergent-series signature rather
than truncation error (see the cleanup record, phases C and K).

## Multi-GPU: the JAX floor is a correctness floor

`jaccpot` pins `jax>=0.11.2,<0.12` (`pyproject.toml`; the floor was 0.10.2 until
2026-10-05). If you import it from a checkout on an older JAX
anyway, know what you are giving up on the distributed path: **on jax < 0.9.1 the
halo exchange (`jax.lax.ragged_all_to_all`) silently returns its fill value once its
buffers move**, and input donation alone — any leapfrog loop — moves them. The
cross-domain near field then disappears from every force evaluation after the
first: rel-L2 **0.45** against an fp64 direct sum on a 17.8M-particle disc across
4×A100, with angular momentum, energy, centre of mass and every overflow flag
looking healthy. Measured in pure JAX (`bench/repro_jax_ragged_all_to_all_forward.py`,
jax 0.9.0): 36/40 calls corrupt under donation, 3/40 under allocator churn, **0/40
with identical buffers** — a run-it-twice reproducibility check cannot see it. Fixed
in the 0.9.1 GPU plugin (XLA `4e0cc7e356`); verified clean on 0.9.1, 0.10.2 and 0.11.2.

Since 2026-09-04 `halo_exchange="auto"` gates the **forward** as well as the gradient
path: below 0.9.1 (and on CPU) it resolves to the `all_gather` exchange (`"buf"`,
identical forces, more bandwidth). Two cautions. A venv built with
`--system-site-packages` can load an *older* `jax_plugins.xla_cuda12` than `pip list`
reports — check `jax_plugins.xla_cuda12.__file__`. And any multi-device result taken
on an affected JAX before this gate is unverified unless a step *after the first*
was compared with a direct sum. See `docs/differentiable_fmm_distributed_audit.md`
("Addendum: the forward is affected too").

## Reproducibility on a GPU

**GPU results are reproducible to a few ulps, not to the bit.** The near field
and M2L accumulate via scatter-add, which XLA lowers to atomics, and floating
point addition is not associative, so the summation order varies run to run.

Measured on an A100 at N=512/leaf=16/p=4/float64, 8 runs on identical inputs:
**0 of the 7 later runs were bit-identical to the first**; the worst elementwise
deviation was **3.8 eps** (8.1e-17 relative to the rms |a|). This is normal for a
GPU FMM rather than a defect, but it means a test must not assert bit equality
between two GPU runs — see `tests/unit/runtime/_reproducibility.py` for the
helpers the suite uses instead.

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` does restore bit equality, at a cost
that makes it unusable for a suite: with it set, three tests in
`tests/unit/runtime` did not finish in 50 minutes. Use it for a one-off
investigation, not as a default.

CPU results *are* bit-reproducible run to run, which is why the characterization
goldens are generated and checked there.

## Precision Control

Use `precision` to select runtime dtype explicitly:

```python
solver_fp32 = FastMultipoleMethod(
    preset="fast",
    precision="fp32",
)

solver_fp64 = FastMultipoleMethod(
    preset="accurate",
    precision="fp64",
)
```

Both need `jax_enable_x64=True` (see [the requirement](#requirement-jax_enable_x64));
it does not pick the precision for you. You can still pass `working_dtype`
directly; if both are set, they must match.

`preset="large_n_gpu"` computes the force in float32 unless you ask otherwise.
`precision="fp64"` there selects the **float64 validation lane**: the same fused
lane with every Pallas kernel natively in float64, for checking a float32 run, with no
time or memory budget (2e6 particles, the bench's defaults: 820 B/particle with
float64 state and force, against 508 in float32). The state keeps the dtype you pass,
so a float64 state under the float32 force is a supported mix. Details and measurements:
[docs/fused_provider.md](docs/fused_provider.md#the-fp64-validation-lane).

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
accept, while the first passing order is the order the pair needs.
The current notebook example prints the resulting tag-derived
`far_pairs_by_gear_counts` from the solver runtime.

**Known gap:** the downward sweep does not yet read the gear buckets. It evaluates
every accepted far pair at the full `max_order`, so today adaptive order changes
which pairs are accepted (and the `far_pairs_by_gear_counts` diagnostic), not the
order or cost of their M2L. Measure before relying on it for speed.

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

- [examples/real_sh_adaptive_order.ipynb](examples/real_sh_adaptive_order.ipynb)
- [examples/real_sh_rot_scale_demo.py](examples/real_sh_rot_scale_demo.py)

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
  reuse the previous full-step per-node force-scale estimate
  (`self._last_force_scale_nodes`, refreshed after every full-order evaluation).
  This is the cheapest option and is the practical default for `tail_proxy`.
- `"prepass"`:
  run a cheap lowest-order prepass for the current configuration and derive force
  scales from that pass, on every `prepare_state`.
- `"paper"`:
  run the stricter paper-style current-step prepass used by
  `adaptive_error_model="dehnen_paper"`, on every `prepare_state`.
- `"paper_cached"`:
  run the paper-style prepass once, on the cold call, then reuse the cached scale
  (refreshed from each full evaluation). This is what `mac_type="dehnen_error"`
  selects by default.
- `"paper_fb"` / `"paper_fb_cached"`:
  use Dehnen **eq (16b)**'s force scale `f_b = sum_{a != b} G m_a / |x_a - x_b|^2`
  instead of eq (16a)'s `|a_b|`. `f_b` is the same sum without the vector
  cancellation, so it never vanishes, and it measurably reduces the large-error
  tail -- roughly 2x over (16a) at equal interaction work. Estimated in O(N) from
  the traversal's own pair partition (exact scalar sums over near pairs, monopoles
  over far ones), so it is cheaper than the eq (16a) prepass, which runs a whole
  low-order FMM evaluation. Unlike `|a_b|`, `f_b` depends only on positions and
  masses, so a completed evaluation contributes nothing to it and the `|a_b|`
  recorder is suppressed under these modes.

Two knobs apply to whichever prepass runs:

- `mac_force_scale_prepass_theta` (default `0.5`): the opening angle of the
  prepass's own geometric traversal. This is deliberately *not* the solver's
  `theta`, which paper mode pins at 1.0 on the grounds that it does not gate
  acceptance -- true of the criterion, false of the traversal underneath it. At
  theta=1.0 the `f_b` estimate degrades to a median 0.74 of the exact value,
  against 0.997 at 0.5 (near Dehnen section 5.2's `theta_crit ~ 0.46`).
- `mac_force_scale_fb_inflation` (default `1.0`): inflates the far-field
  source-to-target distance by the target node's radius, which makes every far
  term an under-estimate and so makes the whole scale a strict lower bound on the
  exact `f_b`. That direction is deliberate: eq (16a) accepts when the estimated
  error falls below `eps * s`, so an over-large scale *loosens* acceptance and
  makes the solver faster **and** wronger -- a failure no cost measurement can
  detect. `0.0` gives the tighter, non-bounding variant; it is a measurement
  setting, not a production one.

Interpretation:

- `prev` is a runtime-oriented reuse mode.
- `paper` is the more publication/reference-oriented mode because it derives the
  threshold from a dedicated current-step prepass rather than from historical state.
  It is also the most expensive: the extra prepass on every step dominates
  `prepare_state`.
- `paper_cached` is the production form of `paper`. Dehnen §5.4 licenses the reuse
  explicitly -- the previous step's accelerations are "only very slightly worse"
  than the exact `a_b` -- and a real simulation always has them.
- `prepass` sits between the two as a generic current-step estimate that is not
  specifically tied to the paper-style Dehnen path.

These scales stay inside jaccpot's adaptive policy state; they are no longer
exported as backend-specific traversal `node_features`.

## Pallas Acceleration

On an Ampere-or-newer GPU (sm_80+) the hot loops run as Pallas (Triton) kernels;
on older GPUs (sm_75), and on CPU, the same steps run in pure JAX. The choice is
automatic:

```python
solver = FastMultipoleMethod(preset="accurate")  # use_pallas=None: auto-detect
```

`use_pallas` selects the near-field kernel (`None` auto-detects; an explicit
`True` / `False` wins). The other kernels gate on the same hardware check. What
runs as Pallas on sm_80+ (`jaccpot/pallas/`):

- **near field:** the leaf-pair kernel driven by the neighbour CSR
  (`nearfield_leafpair_csr.py`), the fused leaf-major kernel
  (`nearfield_fused_leaf.py`), and the block-step lane's mutual P2P
  (`nearfield_mutual.py`);
- **far field:** the M2L over the target-sorted far-pair CSR, one pair per lane
  (`m2l_real_csr_lanes.py`), and the fully fused rotate / z-translate / rotate-back
  M2L (`m2l_real_fused.py`, opt-in via `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1`;
  it serves the chunked M2L lanes, so on one GPU it acts only with the CSR M2L off,
  `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0`, and otherwise on the multi-GPU mesh lane);
- **the rest of the fused strict lane's step:** the dual-tree walk, one launch per
  wavefront round (`mutual_walk_pallas.py`), the directed pair CSR
  (`csr_place.py`), the COM MAC radii (`com_radii_leaf.py`), leaf P2M
  (`p2m_real_leaf.py`), the M2M / L2L cascades (`cascade_real_lanes.py`,
  `cascade_real_level.py`) and L2P (`l2p_real.py`).

The kernels on a gradient path carry a `custom_vjp`, so the Pallas lane stays
differentiable. Two forward kernels have no reverse yet, and the gradient path
takes a slower layout instead: the near field's direct layout (the gradient runs
the CSR lane's table layout) and the particle-major L2P (the gradient runs the
leaf-major XLA L2P).
[ARCHITECTURE.md](ARCHITECTURE.md) section 7 has the gates and the
version-dependent backend selection. On CPU, [examples/pallas_m2l_speed.py](examples/pallas_m2l_speed.py)
reports the fallback rather than Pallas execution.

## Differentiable FMM

`FastMultipoleMethod.differentiable_accelerations` gives **exact gradients of the
FMM force** with respect to particle positions and masses. The tree topology is
held fixed from a pre-built state while the numeric pipeline (P2M, centre-of-mass
expansion centres, the M2M/M2L/L2L translations, L2P, near-field P2P) is
re-evaluated on the live inputs, so `jax.grad` transposes it exactly.

```python
import jax, jax.numpy as jnp
from jaccpot import FastMultipoleMethod

fmm = FastMultipoleMethod(basis="real", theta=0.6, softening=1e-3)

# Build the topology ONCE, outside the differentiated function --
# prepare_state does host-side tree construction and is not traceable.
state = fmm.prepare_state(positions0, masses0, max_order=4, leaf_size=64)

def loss(positions, masses):
    return jnp.sum(fmm.differentiable_accelerations(state, positions, masses) ** 2)

grad_positions, grad_masses = jax.grad(loss, argnums=(0, 1))(positions, masses)
```

Gradients match `jax.grad` of an exact direct O(N²) sum to the FMM's own **force**
accuracy, not to machine precision.

### At galaxy scale

The `large_n_gpu` preset is differentiable and is the leanest configuration
measured. It needs `retain_far_pairs_for_grad=True` so the frozen M2L pair list
survives `prepare_state`. It is a constructor option (or
`FarFieldConfig(retain_far_pairs_for_grad=True)`, or
`JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD=1`); `prepare_state` does not take it:

```python
fmm = FastMultipoleMethod(
    preset="large_n_gpu", basis="real", retain_far_pairs_for_grad=True
)
state = fmm.prepare_state(positions0, masses0, max_order=4, leaf_size=256)
```

Measured 2026-07, A100 40 GB, fp32, θ=0.7, order 4, clustered galaxy disc
(steady-state), before the Pallas reverses of 2026-09:

| N | forward | forward+backward | reverse peak memory |
|---|---|---|---|
| 200 k | 0.86 s | 2.59 s | 2.62 GB |
| 1 M | 2.50 s | 66.4 s | 11.07 GB |

The current numbers (2026-10-13, cells of 64, θ=0.8, order 5, fp32, jitted
gradient w.r.t. positions and masses): 31.9 ms at 2e5, 101.6 ms at 1e6, and out of
memory at 8e6 on one 40 GB card ([docs/production_2026-10.md](docs/production_2026-10.md)).

**What the gradient covers.** Positions and masses only: `G` and the softening
enter as Python floats, so there is no d/dG or d/dε. Reverse mode only: every fast
kernel is a `custom_vjp`, which refuses `jax.jvp` (and so forward-mode adjoints).
One force at a time: `strict_run_v2` and the multi-GPU `FusedRollout` are
forward-only, so a rollout gradient has to differentiate its own integrator around
`differentiable_accelerations`. (The fused forward takes `G` and the softening as
traced `ForceParams`, see [below](#the-fused-lanes-force-in-your-own-integrator);
`jax.grad` in them is verified on the pure-JAX near field in float64 only.)

### Tuning

Pass a `GradConfig`. On a radix state its most important field is
`nearfield_lane`, which selects the near-field traversal. It defaults to `"auto"`,
which switches to the leaf-major fast lane at N ≥ 100 000 because the bucketed
reverse **OOMs** at that scale (30 GB peak at 200 k against the fast lane's 6.8 GB):

```python
from jaccpot import GradConfig

cfg = GradConfig(nearfield_lane="fast_lane")
accel = fmm.differentiable_accelerations(state, positions, masses, grad_config=cfg)
```

Every field falls back to a `JACCPOT_*` environment variable when left `None`, so
existing env-configured scripts keep working; an explicit field always wins. On an
Ampere+ GPU several fields do nothing: `nearfield_lane` on a `large_n_gpu` state
(it has one differentiable near field), and `fused_m2l_pallas` and the `reverse_*`
fields wherever the CSR lanes run, which is the default there (see
`GradConfig`'s docstring).

Full guide, contract, limits, and troubleshooting:
**[docs/differentiable_fmm.md](docs/differentiable_fmm.md)**.

## The fused lane's force in your own integrator

`strict_run_v2` runs velocity Verlet as one device-resident scan on the fused
`large_n_gpu` lane. Its per-step force is public: `FusedForceProvider` is prepared
eagerly and then called inside your own `jax.jit`. A caller-written velocity Verlet
reproduces `strict_run_v2(carry="particles")` bit for bit (CPU test, N = 512):

```python
from jaccpot import FusedForceProvider

# solver: preset="large_n_gpu", static-radix tree with cell leaves (see the guide)
prov, _ = FusedForceProvider.prepare(solver, x0, m, leaf_size=64, max_order=6)
dt = jnp.asarray(1e-3, x0.dtype)

def verlet(carry, m):
    x, v, a, aux = carry
    x = x + v * dt + 0.5 * a * dt**2
    a_new, aux = prov.step(x, m, aux=aux)  # rebuild + accelerations + overflow bits
    return (x, v + 0.5 * (a + a_new) * dt, a_new, aux), None

@jax.jit
def run(carry, m):  # the masses an argument, as in the lane's runner
    return jax.lax.scan(lambda c, _: verlet(c, m), carry, None, length=100)[0]

a0, aux0 = jax.jit(prov.step)(x0, m)
x, v, a, (f_b, bits, needs) = run((x0, v0, a0, aux0), m)
```

`bits` holds one bit per fixed capacity (tree, walk lists, near rows, ...); any of
bits 3-10 means a step's force was wrong, and `needs` is what a re-plan needs.
`prov.step(x, m, params)` takes `G` and the softening squared as `ForceParams(G, eps2)`
(`None`: the solver's own). Python floats are compile-time literals (the default, every
program unchanged), arrays are traced (one compile for every value; measured bitwise
the literal program on an A100). `strict_run_v2`
takes the same opt-in as `force_params=`, an array `dt` and `external_params=`.
`mac_type="dehnen_error"` refuses any `G` or softening but the solver's own, and a
traced softening beyond the one the walk was built for sets the `OVERFLOW_PARAMS` bit.

Guide, the overflow bits, the limits and the float64 validation lane:
**[docs/fused_provider.md](docs/fused_provider.md)**.

### A long production run: `ProductionRunner`

`ProductionRunner` runs a simulation as K-step `strict_run_v2(carry="particles")`
segments chained through the handle -- bit for bit one long call (CPU gates) -- with
fp64 momentum / angular momentum / centre-of-mass / kinetic-energy diagnostics at each
segment edge, host-side callbacks, snapshots and restart files that continue a run
bitwise in a fresh process:

```python
from jaccpot import FastMultipoleMethod, ProductionRunner, SegmentConfig

solver = FastMultipoleMethod(preset="galaxy", softening=1.5e-3)
runner = ProductionRunner(
    solver, segment=SegmentConfig(steps=50, restart_every=4, restart_path="restart.npz")
)
res = runner.run(state, masses, dt=5e-4, num_steps=500)    # state: [N, 2, 3]
res = runner.run(dt=5e-4, num_steps=500, restart_from="restart.npz")  # after a kill
```

Choose `steps` to divide `num_steps` (each segment length compiles its own scan).
Guide: **[docs/production.md](docs/production.md)**.

## Momentum-Conserving FMM

`FastMultipoleMethod` is target-centric: the near field gathers and the far field
sweeps downward, so every pair is evaluated **twice, independently**. The two
roundings differ, and `sum_i m_i a_i` lands at the force accuracy (~1e-3 … 1e-5).
That is fine for a global timestep and wrong for a block-step individual-timestep
KDK, where per-level antisymmetry is the defining correctness property rather
than a diagnostic.

`BlockStepFMM` evaluates each interaction **once** and applies `+f`/`-f` to both
endpoints, through the separate `jaccpot.mutual` path:

```python
from jaccpot import BlockStepFMM

fmm = BlockStepFMM(softening=1e-2, k_max=3, theta=0.7, max_order=4, leaf_size=32)

# Once per base step: build the frozen topology (host operation, not traceable).
fmm.prepare(positions, masses)

# Production path -- one mutual traversal per sub-step boundary.
velocities = fmm.boundary_kick(
    positions, velocities, masses,
    rung=rung, active_floor=1, dt_max=1e-3, half=1.0,
)
```

The force stays FMM-approximate; the momentum does not:

| order `p` | force error | momentum residual |
|---|---|---|
| 2 | 4.0e-03 | 4.1e-16 |
| 4 | 2.8e-05 | 4.1e-16 |
| 6 | 1.6e-07 | 4.2e-16 |

The force error moves five orders of magnitude across that range and the residual
does not move at all — the cancellation is algebraic, not numerical, so it is flat
in `theta` and in expansion order.

Weighting each pair by `level_weights[max(rung_i, rung_j)]` lets one traversal
serve a whole sub-step boundary, taking a base step at `k_max=3` from about 19
traversals to `n_sub + 1 = 9` — measured 2.1×. The path is differentiable at fixed
topology on the same terms as [Differentiable FMM](#differentiable-fmm).

Details, backend comparison, and the scaling study:
[docs/momentum_conserving_fmm.md](docs/momentum_conserving_fmm.md).

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

- [examples/reuse_topology_demo.py](examples/reuse_topology_demo.py)

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

CI measures branch coverage on every `test-full` shard and uploads the reports;
there is no `fail_under` threshold, so a coverage drop does not fail the build on
its own. Locally:

```bash
pytest --cov=jaccpot --cov-report=term-missing
```

### Performance Guard

CI also runs a runtime sanity guard,
[bench/ci_benchmark_guard.py](bench/ci_benchmark_guard.py), on top of
[bench/bench_parallel_paths.py](bench/bench_parallel_paths.py). It has no stored
baseline: it checks broad ratio bounds between the acceleration, the two jerk modes and
the time-derivative orders (broad, to avoid hardware-specific flakiness while still
catching major runtime regressions), and that `jerk_mode="accurate"` still
differs from `"fast_approx"` by more than 1e-6 rel-L2, a tripwire for the source-motion
term going missing. Locally, as CI runs it:

```bash
python -m bench.ci_benchmark_guard --n 384 --runs 2 --warmup 1 --preset fast \
    --basis real --theta 0.6 --leaf-size 16 --max-order 4 --dtype float32
python -m bench.bench_parallel_paths --n 512 --runs 3 --warmup 1   # the timings alone
```

## Examples

- `examples/time_derivatives_demo.ipynb`: usage plus direct-sum accuracy checks for jerk, snap, and crackle
- `examples/jerk_modes_demo.ipynb`: compare jerk `fast_approx` vs `accurate`, including analytic source-motion behavior
- `examples/real_sh_adaptive_order.ipynb`: real-basis adaptive-order demo

The `benchmark_*` and `adaptive_vs_fixed_benchmark` notebooks are records of
earlier measurements. Most predate the 2026-10 cleanup and no longer run end to
end (they drive a deleted worker script, the removed complex basis or private
helpers whose signatures changed); the cleanup record lists what each needs.

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
- `jaccpot/basis`: the real solid-harmonic basis
- `jaccpot/operators`: harmonic, translation, and multipole operators
- `jaccpot/pallas`: the Pallas kernels and their `custom_vjp` rules
- `jaccpot/distributed`: the multi-GPU drivers
- `jaccpot/upward`, `jaccpot/downward`, `jaccpot/nearfield`: sweep and near-field modules
- `jaccpot/mutual`: momentum-conserving (mutual) evaluation path — a separate lane, not a variant of the above
- `jaccpot/nornax_adapter.py`: `BlockStepFMM`, the block-step KDK facade over `jaccpot/mutual`
- `jaccpot/runtime/fused_provider.py`, `fused_topology.py`: `FusedForceProvider`, the fused lane's per-step force, and what it hands on (`FusedTopology`, `ForceParams`, `StepParams`, the overflow bits)
- `tests`: unit, integration (`tests/integration/strict/` for the fused strict lane), characterization (goldens), distributed and performance checks; `bench/tests` tests the `bench/` scripts

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs each test at most once per push,
on CPU runners. `.github/scripts/test_shards.py` defines which tests each job runs, and
the `test-partition` job checks that the shards cover the suite exactly once.

- `lint`: `black`, `isort`, and the `bench/` scripts' own tests;
- `test-full`: the suite with branch coverage, in six shards (unit, unit-runtime,
  unit-runtime-mac, integration, mutual-static-device, characterization); runtime
  type checks are on for the unit shards;
- `test-py-floor`: the `smoke`-marked tests on Python 3.12, the oldest supported;
- `test-distributed-tier`, `test-distributed-mutual`: the multi-device tests on
  forced host devices;
- `test-cross-repo-nornax`: the `BlockStepFMM` conformance suite against nornax;
- `benchmark-guard`: the performance regression guard (above);
- `test-distributed-criterion` and `test-cross-repo-nornax-distributed`: on pushes
  to `main`, nightly, on manual dispatch, and on pull requests that touch
  `jaccpot/distributed/`, `jaccpot/mutual/`, `jaccpot/nornax_adapter.py`,
  `jaccpot/pallas/nearfield_*` (or their own tests and the CI files).

There is no GPU job; `bench/gpu_gate.py` is the pre-release GPU gate (see
`CLAUDE.md`). `.github/workflows/release.yml` builds and publishes to PyPI on
version tags.

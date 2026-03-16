# Derivatives And Jerk

This page documents the higher-derivative and jerk-facing APIs in `jaccpot`.

## Current Status

The higher-order solver support is usable today, with a few explicit scope
limits:

- `compute_accelerations_and_jerk(...)` is public and supports both
  `jerk_mode="fast_approx"` and `jerk_mode="accurate"`.
- `compute_accelerations_with_time_derivatives(...)` is public and currently
  supports orders 1-3: jerk, snap, and crackle.
- the general time-derivative API currently accepts only `mode="accurate"`
  and raises for other mode strings.
- public acceleration spatial derivatives
  (`max_acc_derivative_order > 0`) currently require `basis="solidfmm"`.
- public time derivatives above crackle (`max_time_derivative_order > 3`) are
  not implemented yet.
- all of these paths work both on full solves and on prepared-state/subset
  evaluation APIs.

## Acceleration Derivatives

Use `max_acc_derivative_order` with:

- `FastMultipoleMethod.compute_accelerations(...)`
- `FastMultipoleMethod.evaluate_prepared_state(...)`

Default is `0` (disabled).

When `max_acc_derivative_order > 0`, methods return acceleration plus a tuple
of packed derivative tensors.

For `max_acc_derivative_order = 1`:

- derivative tuple length is `1`
- `derivatives[0]` has shape `(N, 3, 3)`
- this is the acceleration Jacobian

Current support:

- enabled for `basis="solidfmm"`
- requesting derivatives with `basis="cartesian"` raises `NotImplementedError`
- intended for prepared-state reuse as well as one-shot solves

## Time-Derivative APIs

Use:

- `FastMultipoleMethod.compute_accelerations_and_jerk(...)`
- `FastMultipoleMethod.evaluate_prepared_state_with_jerk(...)`
- `FastMultipoleMethod.compute_accelerations_with_time_derivatives(...)`
- `FastMultipoleMethod.evaluate_prepared_state_with_time_derivatives(...)`

`compute_accelerations_and_jerk(...)` returns:

- `accelerations`: shape `(N, 3)` (or subset shape when `target_indices` used)
- `jerk`: same shape as acceleration

`compute_accelerations_with_time_derivatives(...)` and
`evaluate_prepared_state_with_time_derivatives(...)` return:

- `accelerations`: shape `(N, 3)` (or subset shape when `target_indices` used)
- `time_derivatives`: tuple ordered as `(jerk, snap, crackle, ...)`

The higher-order API also works on prepared states, so active-particle or
substep integrators can reuse a prepared topology and still request only a
target subset.

For the currently supported public orders:

- `time_derivatives[0]`: jerk, shape `(N, 3)`
- `time_derivatives[1]`: snap, shape `(N, 3)` when
  `max_time_derivative_order >= 2`
- `time_derivatives[2]`: crackle, shape `(N, 3)` when
  `max_time_derivative_order >= 3`

## Jerk Modes

### `jerk_mode="fast_approx"`

- exact near-field pairwise jerk
- far-field convective jerk from acceleration Jacobian (`da/dx @ v_target`)
- fastest option

### `jerk_mode="accurate"`

- analytic far-field source-motion jerk via source-motion multipole/local
  contractions (`dM -> dL`) plus convective far-field and exact near-field terms
- no finite-difference solves for `solidfmm` basis
- `jerk_fd_dt` is only used as a fallback path for non-`solidfmm` configurations
- slower than `fast_approx`, but typically faster than finite-difference
  accurate-mode equivalents
- optimized implementation: builds source-motion multipoles directly for fixed
  prepared centers (avoids rebuilding full complex upward bundles)

## Higher-Order Time-Derivative Scope

- Public time-derivative runtime support currently covers:
  - order 1: jerk
  - order 2: snap
  - order 3: crackle
- `mode="accurate"` is currently the only accepted public mode for the general
  time-derivative API.
- the far-field higher time-derivative assembler currently requires
  `basis="solidfmm"`
- orders above 3 are not implemented yet
- higher-order source-motion multipole kernels are implemented internally and
  feed the public runtime assembler

## Choosing A Mode

| Priority | Recommended mode | Why |
|---|---|---|
| Throughput | `fast_approx` | No extra global solves. |
| Fidelity to total jerk | `accurate` | Includes source-motion effects analytically in the far field. |
| Conservative rollout | start `fast_approx`, compare with `accurate` | Quantify the tradeoff on your own particle distributions. |

General recommendation:

- Start with `fast_approx` when runtime is primary.
- Use `accurate` when jerk fidelity is critical (e.g. timestep control and
  close agreement to direct-sum jerk reference).

## Notes On Performance

- Derivative and jerk paths are JAX-jit compatible and GPU-friendly.
- `accurate` jerk mode adds extra far-field source-motion contractions by design.
- `accurate` mode for `solidfmm` reuses prepared interactions and topology.
- prepared-state target subsets are supported for jerk and higher total time
  derivatives, which is useful for split-step / active-particle integrators.
- Run:
  - `python -m bench.bench_parallel_paths ...`
  - `python -m bench.ci_benchmark_guard ...`
  to compare path costs on your hardware.

## Example Notebook

See
[`examples/time_derivatives_demo.ipynb`](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/time_derivatives_demo.ipynb)
for a worked example that computes and inspects jerk, snap, and crackle in the
analytic `solidfmm` path.

For a direct-sum accuracy comparison on small particle sets, see
[`examples/time_derivatives_accuracy_demo.ipynb`](/Users/buck/Documents/Nexus/Projects/jaccpot/examples/time_derivatives_accuracy_demo.ipynb).

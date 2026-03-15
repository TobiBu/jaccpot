# Derivatives And Jerk

This page documents the higher-derivative and jerk-facing APIs in `jaccpot`.

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

## Jerk APIs

Use:

- `FastMultipoleMethod.compute_accelerations_and_jerk(...)`
- `FastMultipoleMethod.evaluate_prepared_state_with_jerk(...)`
- `FastMultipoleMethod.compute_accelerations_with_time_derivatives(...)`
- `FastMultipoleMethod.evaluate_prepared_state_with_time_derivatives(...)`

Both return:

- `accelerations`: shape `(N, 3)` (or subset shape when `target_indices` used)
- `jerk`: same shape as acceleration

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

### Current Scope

- Public time-derivative runtime support currently covers:
  - order 1: jerk
  - order 2: snap
- Orders above 2 (crackle and above) are not implemented yet.
- Higher-order source-motion multipole kernels are implemented internally and
  feed the runtime assembler.

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
- Run:
  - `python -m bench.bench_parallel_paths ...`
  - `python -m bench.ci_benchmark_guard ...`
  to compare path costs on your hardware.

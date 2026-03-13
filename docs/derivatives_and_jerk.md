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

Both return:

- `accelerations`: shape `(N, 3)` (or subset shape when `target_indices` used)
- `jerk`: same shape as acceleration

## Jerk Modes

### `jerk_mode="fast_approx"`

- exact near-field pairwise jerk
- far-field convective jerk from acceleration Jacobian (`da/dx @ v_target`)
- fastest option

### `jerk_mode="accurate"`

- central finite-difference estimate of total jerk using two additional
  acceleration solves
- includes source-motion effects without dedicated time-dependent multipole
  machinery
- controlled by `jerk_fd_dt`
- slower than `fast_approx`

## Choosing A Mode

- Start with `fast_approx` when runtime is primary.
- Use `accurate` when jerk fidelity is critical (e.g. timestep control and
  close agreement to direct-sum jerk reference).

## Notes On Performance

- Derivative and jerk paths are JAX-jit compatible and GPU-friendly.
- `accurate` jerk mode costs additional solves by design.
- Run:
  - `python -m bench.bench_parallel_paths ...`
  - `python -m bench.ci_benchmark_guard ...`
  to compare path costs on your hardware.

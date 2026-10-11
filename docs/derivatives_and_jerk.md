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
  (`max_acc_derivative_order > 0`) work on the real basis, the only one
  (`"solidfmm"` is its family name). `"complex"` and `"cartesian"` were removed in
  the 2026-10 cleanup (C and K) and raise at construction.
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
- this is the acceleration Jacobian, `derivatives[0][n, i, j] = d a_i / d x_j`

In general, level `k` (`derivatives[k - 1]`) has shape `(N, 3, C_k)` with
`C_k = (k + 1)(k + 2) / 2`:
`derivatives[k - 1][n, i, c] = d^alpha a_i / d x^alpha`, where `alpha` is the
`c`-th multi-index of `symmetric_multi_indices_3d(k)`
(`jaccpot/operators/symmetric_tensors.py`). That order descends in `x`, then `y`.
At `k = 2` the columns are `xx, xy, xz, yy, yz, zz`. Only the derivative axis is
packed; the component axis `i` is kept whole.

Current support:

- enabled for `basis="real"` (default; `basis="solidfmm"` is the same basis),
  the only basis left (`"cartesian"` and `"complex"` were removed in the 2026-10
  cleanup, K and C)
- **near field and far**: the tower is the derivative of the full
  acceleration. The far half differentiates the local expansions. The near half
  is the direct sum over the same near pairs the acceleration uses, differentiated
  exactly (`jaccpot/nearfield/acc_derivatives.py`):
  - order 1 in closed form;
  - higher orders by nested forward mode through the softening kernel;
  - the self pair excluded, since its Jacobian is `-G m g(0) I`, not zero.

  It runs with every softening kernel. One case is special for the C2 Wendland
  kernel: a coincident pair of distinct particles contributes zero at order 2 and
  above. That is its exact second derivative; the third does not exist there.
  Until 2026-10 the tower was far-field only.
- the near half costs one more pass over the near pairs, only when the tower is
  requested
- the direct-sum oracle
  (`tests/integration/test_real_basis_runtime.py::test_real_basis_acceleration_derivatives_match_direct_sum`)
  checks three things:
  - the full tower against the full direct sum;
  - the far half alone against the sources the accepted far pairs cover;
  - the near half against every other source, to round-off.

  With no far pair the tower matches the direct sum to round-off, for every kernel
  at orders 1-3 (`tests/unit/nearfield/test_acc_derivatives.py`)
- the general radix path only: the large-N lane and a traced
  `compute_accelerations` raise `NotImplementedError` for
  `max_acc_derivative_order > 0`
- exact at every offset on the real basis, including a target sitting on its
  expansion centre (a single-particle leaf) or on the centre's z-axis: the tower
  lowers the coefficients with the exact Cartesian-derivative operator of the real
  harmonics (`jaccpot/operators/real_harmonic_derivatives.py`) instead of
  differentiating the polar form, which lost the curvature there before
  2026-10
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
- far-field convective jerk from the far-field acceleration Jacobian
  (`da_far/dx @ v_target`); the near half of the Jacobian is left out, because
  the near-field jerk already contains it
- fastest option

### `jerk_mode="accurate"`

- analytic far-field source-motion jerk via source-motion multipole/local
  contractions (`dM -> dL`) plus convective far-field and exact near-field terms
- no finite-difference solves
- `jerk_fd_dt` is accepted and unused, with no warning and no check on its
  value: it stepped the finite-difference fallback of `basis="cartesian"`, which
  was removed in the 2026-10 cleanup (K). It stays on all four jerk signatures
  because callers pass it (nornax's jaccpot adapter does, on every call)
- slower than `fast_approx`, but typically faster than finite-difference
  accurate-mode equivalents
- builds the source-motion multipoles `d^k M / dt^k` directly for the prepared
  (frozen) centres -- a lowered leaf P2M plus the ordinary M2M -- and runs them
  through the ordinary M2L / L2L, which are linear at fixed geometry. On the real
  basis this is `prepare_real_source_motion_multipoles`
  (`jaccpot/upward/real_tree_expansions.py`); real and complex agreed to round-off
  (8e-18 relative on the jerk at N = 96, p = 4, theta 0.6) until the complex basis
  was removed in the 2026-10 cleanup (C). The direct-sum oracle with the far field
  engaged there: jerk 4.3e-4, snap 8.3e-5, crackle 9.6e-6 (Plummer softening)

### What the time derivatives mean

All of these are `D_t^n a` for particles moving on straight lines, `x + v t`,
with the tree, the expansion centres, the interaction lists and the expansion
orders frozen at the prepared state. Accelerations of the particles themselves do
not enter: snap and crackle here are the straight-line terms only. A direct-sum
reference that includes the relative-acceleration terms (as nornax's
`DirectForce` does for Hermite-6/8) differs from them by those terms.

## Higher-Order Time-Derivative Scope

- Public time-derivative runtime support currently covers:
  - order 1: jerk
  - order 2: snap
  - order 3: crackle
- `mode="accurate"` is currently the only accepted public mode for the general
  time-derivative API.
- the far-field higher time-derivative assembler runs on the real basis (the
  only one since the 2026-10 cleanup, C)
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
- `accurate` mode reuses prepared interactions and topology.
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
analytic spherical-harmonic path, including a direct-sum accuracy comparison on small
particle sets.

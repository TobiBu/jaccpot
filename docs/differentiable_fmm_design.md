# Differentiable FMM — Phase 1 design

Scope: exact gradients of the **single-GPU** FMM force w.r.t. particle **positions**
and **masses** (self-gravity). Out of scope (deferred): external-potential-parameter
gradients, multi-step trajectory differentiation, distributed/multi-GPU gradients, and
analytic-`custom_vjp` performance splicing. Companion: `differentiable_fmm_audit.md`.

## 1. The differentiability contract (fixed topology)

Gradients are taken at **fixed topology**. The reverse pass differentiates the
*numeric* pipeline — multipole moments (P2M), COM expansion centers, the M2M/M2L/L2L
translations, L2P, and the near-field P2P — while treating **all integer index arrays
as constants**: the Morton permutation, node membership, the M2L interaction list, the
near-field neighbor CSR, and every MAC accept/reject decision.

This drops the *implicit* dependence of the force on position via "which cell a
particle lands in". That contribution is nonzero only on the measure-zero set where a
pair crosses a MAC/topology boundary — exactly the piecewise structure Dehnen (2014)
App. B.3 describes for the force itself. For a small enough finite-difference step no
particle crosses a boundary, so finite differences and autodiff must agree to FMM force
accuracy. Near θ→boundary the agreement degrades; characterizing that degradation is a
paper result, not a bug.

**COM centers are differentiated through, not held fixed.** Node/expansion centers are
center-of-mass (position- and mass-dependent). The translation vectors and the L2P
offset `leaf_positions − centers` move with the live COM, and autodiff flows through
them. Only the *pair list / near-far partition* is frozen. The finite-difference
reference must therefore perturb the **same frozen-topology function** (the differentiable
evaluate on the same `state`); FD of a full `compute_accelerations` (which rebuilds tree
+ MAC) will disagree near partition-flip boundaries — expected, and the validation
harness FDs the frozen path.

**Enforcement is by construction.** The prepared `state` is captured as a Python
constant (not a differentiated argument), so none of its children — integer topology
*or* the stale float `upward`/`downward` payloads — receive cotangents. Integer dtype
already yields a `float0` cotangent. No `stop_gradient` is required; the contract holds
because the differentiated function recomputes every numeric quantity from the live
`positions`/`masses` and never feeds `state.downward`/`state.upward` into the result.

**Interfaces kept clean for the future (not built now).** The MAC θ, the expansion
order, and the accept/reject step stay factored as `static_argname`/`state`-carried
discriminators. A later learned/continuous MAC (Jaccpot II) is *not* a matter of routing
cotangents into these — it needs a different mechanism (continuous relaxation /
straight-through / score-function estimator) and is explicitly out of scope. The
fixed-topology reverse pass here does not preclude it.

## 2. Strategy — Route A (autodiff through the pure-JAX forward pass)

Chosen because the audit shows it is both correct and near-optimal:

- **The translation cascade is linear.** M2M/M2L/L2L and the rotate→z-translate→
  rotate-back operators (with involutory B-matrices) are precomputed constant linear
  operators. A linear op is its own VJP (transpose), so XLA generates an exact, cheap
  reverse for the whole cascade for free — the same reason the existing jerk tower
  reuses these operators forward-only on derivative-coefficients.
- **L2P** already uses nested `value_and_grad`/`jacfwd` (real) or an analytic gradient
  (complex); both compose under an outer `grad`.
- **P2P** is pure-JAX with NaN-safe double-`where`/`rsqrt`.

**Route B (a hand-written adjoint FMM via `custom_vjp` on the whole solve) is NOT on the
critical path** for the paper's claim. Analytic `custom_jvp`/`custom_vjp` splicing
(complex L2P analytic gradient, P2P analytic terms, the real-basis L2P `jacfwd` tower)
is a **Phase-4 performance lever**, to be cross-checked bit-for-bit against Route A — not
the correctness mechanism.

## 3. Numeric-path changes (implemented; numerics-preserving)

1. **L2L static loop bound (B3).** `_propagate_solidfmm_locals_by_level` gains an
   optional static `num_levels`; the concrete-tree caller resolves it and passes it so
   the level cascade has static `fori_loop` bounds (reverse-mode-safe). Dynamic fallback
   unchanged. Mirrors the upward M2M pattern.
2. **M2L invalid-pair delta double-where (B4).** Substitute a nonzero displacement for
   invalid/padded pairs before the M2L apply, so `norm(delta)` has a finite cotangent;
   the contribution is still masked to 0, so the forward is byte-identical.
3. **Rotation-to-z angle NaN guards (B6).** Double-where guards in the shared angle
   builders — complex `_angles_from_delta_solidfmm`, real
   `_multipole_align_{to,from}_z_block` and the real M2L `radii` norm — so `sqrt`/
   `arctan2`/`arccos` have finite reverse cotangents at zero displacement (single-child
   COM L2L) and z-axis-aligned displacement (lattice-aligned M2L). Forward unchanged.

All three are gated by the golden characterization oracle (rtol/atol 1e-12) to prove
forward byte-stability (verified green on both bases at n256/n1024, p2/p4).

## 4. Public surface — `FastMultipoleMethod.differentiable_accelerations`

```python
state = fmm.prepare_state(pos0, mass0, max_order=p, leaf_size=...)   # concrete, once
a = fmm.differentiable_accelerations(state, positions, masses)       # (N,3), input order
g = jax.grad(lambda p, m: (fmm.differentiable_accelerations(state, p, m) ** 2).sum())(pos, mass)
```

Signature:
```python
def differentiable_accelerations(
    self, state, positions, masses, *, target_indices=None, jit_traversal=False,
) -> Array   # (N,3) accelerations in original input order
```

- `state` (built once, concretely) and the engine are captured constants; `positions`
  and `masses` are the differentiated arguments.
- **Why a method taking `state`, not a `differentiable=True` flag:** `prepare_state` is
  non-traceable (B2), so topology must be built before `jax.grad`. A flag on
  `compute_accelerations` cannot honor that and already short-circuits tracers to a
  direct sum (B1).
- **Why not differentiate `evaluate_prepared_state`:** it passes the frozen
  `state.downward` locals straight through, so its gradient would omit the entire
  P2M→M2M→M2L→L2L source-side chain — the *wrong* gradient. The differentiable path must
  **re-run the sweeps** on live inputs.

**Guards** (raise with a clear message):
- `isinstance(state, LargeNPreparedState)` → reject (keeps Pallas near-field off the
  grad path).
- `state.expansion_basis != "solidfmm"` → reject (cartesian autodiff-cleanliness
  unverified; solidfmm covers the complex and real submodes).

## 5. Implementation seam

- **New private engine method** `_evaluate_prepared_state_at_positions_and_masses_sorted`
  — a generalization of the existing `_evaluate_prepared_state_at_positions_sorted` that
  threads live `masses_sorted` into `prepare_upward_sweep` (P2M + COM) and both
  `_evaluate_prepared_tree*` calls (P2P). The existing method is refactored to *delegate*
  with `masses_sorted=state.masses_sorted` (DRY; existing behavior and forward numerics
  unchanged). Reuses `prepare_upward_sweep`/`prepare_downward_sweep` (with
  `interactions=state.interactions` frozen) and the `_evaluate_prepared_tree*` kernels
  verbatim.
- **`differentiable_accelerations`** builds a forward index from the integer
  `state.inverse_permutation` (`fwd = zeros(n,int).at[inverse_permutation].set(arange(n))`),
  gathers `positions_sorted = positions[fwd]`, `masses_sorted = masses[fwd]` (gather VJP =
  scatter-add; `fwd` integer ⇒ no cotangent), calls the seam with `jit_traversal=False`,
  and returns in original order (the seam already applies `inverse_permutation`).
- **Facade** `FastMultipoleMethod.differentiable_accelerations` delegates to
  `self._impl.differentiable_accelerations`. Added to the frozen
  `EXPECTED_FMM_PUBLIC_METHODS` set in the API-surface test.

## 6. Correctness strategy (Phase 3)

- **Primary gate:** `grad(differentiable_accelerations)` vs `grad(differentiable_gravitational_acceleration)`
  (the exactly-differentiable direct O(N²) sum) — must agree to the FMM's own **force**
  accuracy (not machine precision), for masses and positions.
- **FD vs AD** swept over θ, N, p; FD perturbs the *same frozen-topology function*
  (self-consistent to round-off) plus a small-step full-pipeline FD (no boundary
  crossing) as the physical check.
- **NaN/inf hygiene** (small separations, softening on/off), **axis-aligned M2L delta**
  (rotate-to-z kink risk), **mass-gradient traceability**, and **forward parity** vs
  `compute_accelerations`.
- Golden oracle and public-API-surface tests stay green throughout.

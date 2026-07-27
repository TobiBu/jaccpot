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

**jit limitation (known, scoped).** Bare `jax.grad`/`jax.vjp` over
`differentiable_accelerations` give exact gradients at **every N** (verified to N=4096).
Wrapping the *entire* call in an outer `jax.jit` (`jax.jit(jax.grad(loss))`) works at
moderate N but fails at large N with a `ConcretizationTypeError`: the re-run of
`prepare_upward_sweep`/`prepare_downward_sweep` still contains host-side ops
(`int(jnp.max(...))`, `device_get`, numpy level-offset construction — audit finding B2)
that JAX cannot stage once they stop being constant-folded. Two of these were made
jit-safe here (the input permutation is resolved host-side to a compile-time constant;
`_resolve_upward_num_levels` reduces on the host), but fully jit-tracing the prepared
sweeps is a separate effort. Recommended usage is therefore bare `jax.grad(loss)` (the
inner numeric kernels are already `@jax.jit`-compiled); making the whole re-eval
outer-jittable at arbitrary N is a scoped follow-up.

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

## PR-1 outcomes — reverse-pass performance (analytic rules + outer-jit)

Follow-up work (`feat/fmm-analytic-custom-vjp`) profiled and optimized the reverse pass.
The measured ~9x reverse overhead (and a startling ~360ms forward at N=1024/p=4) turned
out to be dominated **entirely by the near-field P2P**, not L2P/M2L/L2L:

| stage | fwd | reverse ratio |
|---|---|---|
| upward P2M+M2M | 3.1ms | 4.2x |
| downward M2L+L2L | 0.08ms | 1.1x (linear) |
| L2P (far) | 0.16ms | 1.56x (analytic custom_vjp) |
| **near-field P2P** | **411ms** | **7.7x** |

Delivered:

1. **Near-field `"bucketed"` mode on the differentiable path (dominant win).** The path
   was running the serialized `"baseline"` near-field (a per-leaf `lax.scan`, gated to
   large-N only); a dense O(N²) direct sum at N=1024 is 0.2ms. Forcing the vectorized
   `"bucketed"` mode (bit-identical, rel-L2 3e-16) via a `nearfield_mode_override` threaded
   from `differentiable_accelerations` → seam → `_evaluate_prepared_tree` → `evaluate_tree`
   gives **fwd 20.3ms→0.48ms (N=256), 357ms→1.11ms (N=1024)** and drops the reverse ratio
   to ~2.5–4x. Forward `compute_accelerations` is unchanged (override defaults `None`).
2. **Outer-jit at large N.** Two `int(jnp.max(node_levels…))` host ops in the re-run
   sweeps (`_resolve_upward_num_levels`, and the L2L level-count in
   `_prepare_solidfmm_downward_sweep`) are resolved on the host (device_get the *full*
   concrete array, then slice+reduce in numpy). `jax.jit(jax.grad(differentiable_accelerations))`
   now works at N=4096/8192 (gradients bit-identical to bare grad, rel ~1e-16; N=4096
   jitted reverse ratio 1.26).
3. **Real-basis L2P analytic `custom_vjp`** (`evaluate_local_real_with_grad`): a linear
   coefficient VJP + a Hessian-vector product replace the second-order-autodiff blow-up
   (isolated L2P reverse 1.56x). Byte-identical forward. Off-switch
   `JACCPOT_ANALYTIC_L2P_VJP=0`. Matters for L2P-dominated configs; negligible in the
   near-field-dominated default.
4. **P2P near-field analytic `custom_vjp`** (`_pair_contributions_batched` →
   `_pair_accel_cvjp`): the reverse is the analytic symmetric tidal tensor
   `J = -G Σ_s m_s(I/r³ − 3rrᵀ/r⁵)` contracted with the output cotangent (`Jᵀc = Jc`) in
   one extra pair pass. Robust module-level rule (all-float args, zero cotangents for the
   non-differentiated masks/softening/G — a closure over those tracers is unsupported).
   End-to-end reverse ratio (real, N=1024) 4.13→3.81; isolated near-field reverse ~4x→~2x.
   Off-switch `JACCPOT_ANALYTIC_P2P_VJP=0`. **This analytic reverse is exactly the rule a
   future fused-Pallas near-field `custom_vjp` reuses as its backward** — the main reason
   to build it now despite the modest pure-JAX ROI.
5. **`custom_vjp` parity harness** (`tests/unit/test_custom_vjp_parity.py`): bit-for-bit
   VJP-vs-autodiff verification for every analytic rule (real-L2P + P2P).

**Remaining (optional):** complex-basis L2P `custom_vjp` (analytic tower already in-tree)
— low-ROI now that the reverse is 1.3–4x. **PR-2 (Pallas fast lane):** wrap the fused
Pallas near-field/M2L kernels in `custom_vjp` (near-field reuses the P2P tidal-tensor
reverse above; M2L reuses the adjoint-M2L) and lift the `LargeNPreparedState`/Pallas
guard. Lower urgency since the bucketed pure-JAX path is already fast and differentiable.

## PR-2 outcomes — differentiable fused-Pallas M2L fast lane (`feat/fmm-pallas-vjp`)

`pallas_call` has no JVP/transpose rule, so the fused-Pallas M2L kernels were
non-differentiable and `differentiable_accelerations` rejected the
`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` flag outright. PR-2 makes the fused M2L a
differentiable **opt-in fast lane** on the grad path.

**Scope decision — near-field Pallas is wrapped for completeness, but NOT put on the
grad path (the reality check paid off).** We measured the fused-Pallas near-field forward
vs its pure-JAX twin on an A100 (`nearfield_fused_leaf_pallas` vs
`nearfield_fused_leaf_jax`, identical leaf-major inputs):

| config | pure-JAX | Pallas | speedup |
|---|---|---|---|
| N~1024, leaf 64 | 0.077 ms | 0.067 ms | 1.14x |
| N~1024, leaf 32 | 0.080 ms | 0.069 ms | 1.17x |
| N~4096, leaf 64 | 0.084 ms | 0.085 ms | 0.99x |
| N~4096, leaf 64, K=1024 | 0.099 ms | 0.145 ms | 0.68x (slower) |
| N~16384, leaf 64 | 0.160 ms | 0.087 ms | 1.85x |

At the N the reverse pass bounds (~1k) the fused near-field is only ~1.1x, and it is
*slower* at moderate N with denser sources; it wins only at large N (≥16k). The
bucketed pure-JAX near-field is already differentiable (analytic tidal-tensor
`custom_vjp`, PR-1) and essentially as fast there. So the fused near-field kernels are
wrapped with `custom_vjp` for completeness (they are now differentiable, so they *can*
be opted onto the grad path — see item 5 below), but **the grad path keeps the bucketed
near-field by default.**

**Delivered:**

1. **`custom_vjp` on the three fused Pallas M2L kernels** (`jaccpot/pallas/`):
   `m2l_core_z_real_pallas_cvjp`, `m2l_complex_fused_pallas_cvjp`,
   `m2l_real_fused_pallas_cvjp`. Each runs the Pallas kernel forward and takes the
   reverse from autodiff of the in-file pure-jnp twin — the kernel's verified literal
   port / recurrence (`bwd = grad(twin)`, correct-by-construction). Module-level rules
   following the PR-1 `_pair_accel_cvjp` pattern: statics (`order`/`interpret`/`backend`)
   in positional `nondiff_argnums`, all else explicit array args, **no closure over
   tracers**. The M2L is linear in the multipoles, non-linear in the radius, and
   b/tri-linear in the rotation blocks; autodiff of the twin handles every input exactly.
   (Started with `bwd = grad(twin)` per the plan; the analytic adjoint-M2L is a later
   refinement only if the twin autodiff ever becomes a bottleneck — it is not: M2L+L2L
   is ~0.08 ms of the forward, see the PR-1 table above.)

2. **Fast lane wired into dispatch + guard lifted.**
   `_m2l_{complex,real}_batch_kernel_fused_pallas` (`runtime/kernels/core.py`) route
   through the `_cvjp` wrappers, so the fused M2L is differentiable wherever it runs
   (the `custom_vjp` forward *is* the Pallas kernel — byte-identical forward, negligible
   overhead). `differentiable_accelerations` no longer rejects
   `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`; setting it (Ampere+) opts into the
   differentiable fused-Pallas M2L, else the default pure-JAX M2L. Off by default;
   `LargeNPreparedState` stays rejected.

3. **sm_80 gate fix.** The real-M2L routing gate `_real_m2l_pallas_active` now checks
   `pallas_m2l_real_fused_supported()` (Ampere+, matching the complex gate) instead of
   the gpu/tpu-only z-core `pallas_m2l_real_supported()`, so it no longer routes to a
   Triton lowering that fails on a pre-Ampere GPU. `pallas_m2l_real_fused_supported()`
   itself was hardened with the sm_80 check.

4. **Coverage.** Added the missing forward parity test for `m2l_real_fused`
   (`tests/test_m2l_real_fused_pallas.py`: Pallas-interpret vs the twin and vs the
   rot-scale reference it accelerates). VJP parity for all three kernels via
   `assert_vjp_matches` — interpret on CPU CI + real Pallas on the A100 (12 passed).
   Gradient-correctness + grad-vs-direct-sum re-run with the flag ON on the A100
   (17 passed: FD-vs-AD positions/masses complex+real, grad(FMM) vs grad(direct sum),
   NaN/inf hygiene).

5. **Fused-Pallas near-field `custom_vjp`** (`jaccpot/pallas/nearfield_fused_leaf.py`):
   `nearfield_fused_leaf_pallas_cvjp` (pairs lane) and `nearfield_leafpair_pallas_cvjp`
   (prepacked production lane, in-kernel leaf-id gather). Pallas forward +
   autodiff-of-twin reverse; non-diff mask/id arrays passed as floats (reconstructed
   inside) so cotangents are ordinary zeros with no closure over tracers. VJP parity
   interpret + A100 (4 passed). These make the fused near-field differentiable so it
   *can* be opted onto the grad path, but per the ROI table above the grad path keeps
   the bucketed near-field by default (the fused near-field's forward edge only shows at
   N≥16k). No grad-path wiring is added — the near-field mode stays bucketed.

**fwd/bwd consistency:** the reverse is the gradient of the pure-jnp twin, not
bit-exactly of the Pallas forward (Pallas ≈ twin to ~1e-10 for the M2L literal ports).
Both are valid FMM gradients to force accuracy — standard for a Pallas `custom_vjp`.

**End-to-end overhead (`differentiable_accelerations`, A100, order 4, jitted;
`examples/differentiable_fmm_overhead.py` run once per M2L mode):**

| N | basis | fwd: pure-JAX → fused-Pallas | fwd+bwd: pure-JAX → fused-Pallas |
|---|---|---|---|
| 256 | complex | 0.84 → 0.49 ms (1.7x) | 1.68 → 1.20 ms (1.4x) |
| 1024 | complex | 2.50 → 1.10 ms (2.3x) | 5.52 → 4.17 ms (1.3x) |
| 256 | real | 0.91 → 0.49 ms (1.9x) | 1.77 → 1.19 ms (1.5x) |
| 1024 | real | 2.33 → 1.15 ms (2.0x) | 5.68 → 4.18 ms (1.4x) |

The fused-Pallas M2L roughly **halves the forward** and cuts fwd+bwd by ~1.3–1.5x on the
grad path — a real win (the differentiable downward re-eval issues many small per-pair
rotate/z-translate/rotate-back launches that the single fused kernel collapses; the
per-stage PR-1 profile above under-counts this because it timed the M2L in isolation, not
the re-eval's launch overhead). (N≥4096 trends the same or better but was measured under
concurrent-GPU contention here, so only the clean N=256/1024 rows are quoted.)

**Fused reverse (`custom_vjp` bwd).** The reverse is a fully-fused, on-chip analytic VJP
kernel (`m2l_{real,complex}_fused_vjp_pallas`): it recomputes the forward intermediates
and walks the adjoint chain (operator transposes for the linear stages, per-block outer
products for the two rotation-block cotangents, and the analytic
`r_bar = -(1/r) lz_bar^T (Z ⊙ Zexp) mrf` for the radius), so the reverse pass runs as a
single Pallas launch rather than pure-JAX autodiff of the twin. Default ON
(`JACCPOT_FUSED_M2L_VJP=1`); `=0` falls back to autodiff of the twin (the correctness
reference — the fused reverse matches it to round-off, VJP parity 20 passed on the A100,
and grad-correctness with the fused fwd+reverse 17 passed). The complex boundary uses
JAX's conjugate convention (`out_i_bar = -imag(out_bar)`; recombine
`complex(re_bar, -im_bar)`), verified empirically against `jax.vjp`.

**Measured reverse ROI is ~0 at N=1024/p4** (A100): with the fused M2L forward engaged,
the total reverse (~3.6 ms) is **near-field-bound**, so fusing the M2L reverse moves it
only 3.62→3.58 ms (complex, 1.01×) / 3.70→3.56 ms (real, 1.04×) — within timing noise.
The fused reverse is kept default-on (validated, equal-cost, and the on-chip form should
matter at high expansion order, where the M2L share grows, or for reverse-pass memory at
scale), but the reverse bottleneck is the near-field, not M2L. Fusing the near-field
reverse (or its analytic tidal-tensor VJP) is where the remaining reverse ROI lives.

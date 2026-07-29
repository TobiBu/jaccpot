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

## PR-3 outcomes — reaching galaxy N (200k–1M)

PR-1/PR-2 optimized the reverse pass at N≲4k. Measuring at the canonical galaxy
scale (200k–1M) showed gradients are **accurate but did not scale**: FD-vs-AD on the
frozen-topology function is 1.2e-9 (N=1024), 7.9e-10 (N=4096) and **2.9e-9 at
N=16384 with the far field live** (3626 M2L pairs), with forward parity 1e-16 and
grad(FMM)-vs-grad(direct-sum) at 2.5e-4 — but the reverse pass *failed outright* at
N=65536, three independent blockers deep. All four fixes below are in this PR.

**Correction to a claim in this doc / the docstring.** "Bare `jax.grad`/`jax.vjp`
give exact gradients at every N" was false above 65536: the host-side ops that the
"jit limitation" section attributes to outer `jax.jit` also bit the *bare* reverse
once the grouped M2L auto-enabled. Fixed by (A); the docstring is corrected.

### A. The grad path must not route through the host-side grouped classifier
At `n_particles >= _GPU_LARGE_PARTICLE_THRESHOLD` (65536) the resolver auto-enables
`grouped_interactions`, whose class construction runs on the host — yggdrax
`build_grouped_interactions_from_pairs` does
`np.asarray(jax.device_get(geometry.center))`. Under `jax.grad` the centres are
tracers, so the reverse died with `TracerArrayConversionError float64[2047,3]`. The
forward survived only because eager execution keeps the centres concrete.
`_evaluate_prepared_state_at_positions_sorted` gains `force_ungrouped_farfield`
(default ON for the differentiable seam). **`center_mode` is passed through
unchanged** — the grouped classification *requires* geometric (aabb) centres, and
rewriting them would change the force the gradient is taken of.

**The grouped and ungrouped M2L are NOT the same force** (measured, and worth
recording because the obvious assumption is wrong). The grouped path quantises pair
displacements onto a lattice (`np.rint(delta / cell_size)`) and applies ONE
representative displacement per class, so it is an approximation, not a re-spelling
of the same arithmetic. At n=512, θ=0.6, p=4, `leaf_size=4` (a deliberately deep
tree, so the classes are sparsely populated and the quantisation is stressed),
against the exact direct sum:

| path | rel-L2 vs exact direct sum |
|---|---|
| grouped M2L forward | 7.08e-02 |
| **ungrouped M2L (the grad path)** | **1.07e-02** |
| grouped vs ungrouped | 6.58e-02 |

So the grad path is ~6.6x *more* accurate here, not merely different. Consequence:
if a caller explicitly requests `grouped_interactions=True`, their forward force and
the force the gradient is taken of differ at the grouped path's own accuracy. With
(D) the default path no longer enables grouping at n>=65536, so forward and grad
path agree exactly there; the mismatch is reachable only via an explicit request.
A leaf_size of 4 stresses the quantisation far harder than the production leaf
64--256, so treat 7e-02 as a worst case, not a typical figure.

### B. Reverse-pass memory was O(near-field pairs), not O(N) — the real wall
`_pair_accel_cvjp_fwd` stored `(diff, inv_dist3, inv_dist5)` = 5 doubles per particle
**pair**, and because the bucketed near-field drives the kernel from a `lax.scan` over
edge chunks, reverse mode retained *every* chunk's residual:
`40 * near_field_edges * max_leaf_size**2` bytes in total, independent of
`edge_chunk_size`. Measured at N=65536 (galaxy disk, θ=0.7, leaf 64, 569,220 edges):
a single **52.14 GiB** `diff` allocation — matching `edges*W*W*3*8` to 0.03 GiB — on a
38.2 GB A100, against **1.13 GB for the whole forward** (the forward streams; the
reverse does not). No `jax.checkpoint` existed anywhere in the tree.

The residual now carries only the O(B*W) inputs and the pair intermediates are
**rematerialized** in `_pair_accel_cvjp_bwd` via the shared `_pair_accel_pair_terms`
helper (so forward and reverse expressions cannot drift). Residual: **86.9 GiB →
1.90 GiB (46x)**; the extra pair pass lands in a backward that already materializes a
`(B, Wt, Ws, 3)` array. Gated bit-for-bit by
`test_p2p_analytic_custom_vjp_matches_autodiff` (rtol 1e-11).

### C. fp32 far-field gradients were NaN — the `1e-60` floors flushed to zero
fp32 is the precision the large-N production path uses, yet every gradient test ran
float64. Measured: fp32 gradients are finite with an empty M2L list (rel-L2 1.96e-6
vs float64, worst component 2.5e-3) but **all-NaN once the far field is active**.

Cause: `jnp.maximum(r2, 1e-60)` in the translation/harmonics operators. `1e-60`
**underflows to exactly 0.0 in float32** (smallest subnormal ~1.4e-45), so the clamp
became a no-op. That clamp is load-bearing for *reverse* safety, not just the
forward: when the floor wins, `jnp.maximum` routes a zero cotangent, which is what
keeps `d sqrt/dr2 = 1/(2r)` finite at the exact-zero displacements the fixed-topology
FMM genuinely hits (a single-child internal node shares its child's COM ⇒ a zero L2L
translation — see B6 above). With the floor gone, float32 evaluated `sqrt(0)`.

`operators/dtypes.squared_radius_floor(dtype)` returns `max(1e-60, finfo(dtype).tiny)`
— **float64 is unchanged bit-for-bit** (1e-60), float32 becomes 1.1755e-38. Applied at
all 8 real-basis and 4 complex-basis sites. Two real-basis sites paired the floor with
a `rho > 1e-30` test (`== sqrt(1e-60)`, i.e. "was the floor applied?"); that predicate
silently inverted in float32, taking the `x/rho` branch on the z axis and giving
`cos_phi = 0` instead of the correct `1.0`, so it is now expressed as
`rho2 > squared_radius_floor(...)` — identical in float64, correct in float32. The
`is_zero = r < 1e-30` sites are left alone deliberately: they are unreachable in
float64 (`r >= sqrt(1e-60)`), and the dtype-aware floor makes float32 agree.

### D. `farfield_mode="auto"` reached the kernels — breaking the FORWARD too
The `static_runtime_fixed_sizing` branch of the resolver (**on by default**,
`JACCPOT_STATIC_RUNTIME_FIXED_SIZING=1`) never resolved `farfield_mode` away from
`"auto"`, so once the grouped M2L auto-enabled at n>=65536 the grouped branch of
`_solidfmm_downward_accumulate_from_multipoles` raised
`ValueError: farfield_mode must be 'pair_grouped' or 'class_major'` — from
`prepare_state` **and `compute_accelerations`**. Pre-existing (identical on `main`)
and independent of gradients. That branch also never coupled `center_mode="aabb"` to
the grouping the way the adaptive branch does, so resolving only `farfield_mode`
would have traded a crash for *wrong forces* (the grouped path quantises pair
displacements onto a lattice and applies one representative displacement per class).
Fix: static sizing no longer inherits the adaptive auto-grouping rewrite (it is an
adaptive rewrite, which that branch exists to skip), and an *explicit*
`grouped_interactions=True` now resolves `farfield_mode` and couples `aabb` exactly
as the adaptive branch does.

**Why nothing caught A–D:** the largest real particle count in the suite is n=1500;
the `num_particles=1000000` references only unit-test chunk-size resolvers, and the
one N=65536 test uses `preset="large_n_gpu"` (which forces `pair_grouped`) and is
env-gated off. `tests/unit/test_large_n_grad_path.py` now covers the resolver policy
(host-only, instant), the grouped-vs-ungrouped grad equivalence, and fp32 far-field
gradient finiteness.

**Still open at the time of PR-3:** `LargeNPreparedState` remained rejected, so the
preset `large_n_gpu` (fp32, leaf 256) — the config the README recommends for 200k–1M —
was not differentiable; its Pallas kernels already carried `custom_vjp` (PR-2), so the
gap was the eval seam. **Since closed** by `runtime/_large_n_grad.py`; see "Galaxy scale
reached" under PR-4 below for the measurements.
`OdisseoFMMCoupler.accelerations` still yields **exactly zero** gradients silently
(it routes to `evaluate_prepared_state`, which takes no live positions/masses).
Reverse-pass *compile* time is ~10 min at N=16384 and grows (25 min at N=1M).

## PR-4 outcomes — the differentiable near field on the leaf-major fast lane

(Specified in `docs/differentiable_fmm_nearfield_fastlane_plan.md`, which calls itself
"PR-3"; the numbering diverged because the galaxy-N work above landed first.)

The plan's in-context ablation of `differentiable_accelerations` (N=1024, complex,
order 4, leaf 32, A100) attributes **~83% of the forward and ~91% of the reverse to the
near field**, with the M2L accumulate at ~0%. Its order sweep is flat (p 2→6) so it is
not the far-field FLOP cascade, and its leaf sweep (reverse 5.07 → 2.12 → 0.66 ms at
leaf 16/32/64) points at bucket/pair count, i.e. the bucketing overhead itself: ~0.9M
near-field interactions is ~2 µs of A100 f64 compute against ~2.4 ms measured, so it is
gather/scatter and padding, not arithmetic. The absolute baseline reproduces here
(N=1024: 1.04 ms fwd / 2.66 ms reverse against the plan's 1.02 / 2.61), so the
attribution carries over to this box.

**What was built.** The grad path can now take the near field leaf-major instead of
through the bucketed edge-list kernel:

1. **`runtime/_nearfield_fastlane.py`** transposes the radix state's CSR neighbour list
   into the `RadixFastNearfieldPayload` the fast lane consumes — for each target leaf, a
   padded `(leaf, block, lane)` array of source-leaf ids. The radix `FMMPreparedState`
   carries no such payload (only the large-N pipeline bakes one at `prepare_state`), so
   this is the piece that was missing. `-1` padding inside a leaf's CSR slice is masked
   exactly the way `prepare_leaf_neighbor_pairs` masks it for the bucketed lane, so both
   lanes enumerate the *same edge set*.
2. **`nearfield_mode_override="fast_lane"`** in `evaluate_tree` dispatches to
   `compute_leaf_p2p_accelerations_radix_fast_lane(..., differentiable=True)`;
   `differentiable_accelerations` selects it under
   `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`, bucketed otherwise.
3. **Which reverse you get depends on `use_pallas`.** With it on (Ampere+) the lane is
   the fused Pallas kernel behind `_radix_fast_lane_prepacked_accel_cvjp`, whose backward
   is the analytic O(N) leaf-pair reverse. With it off, the same flag still selects the
   leaf-major traversal but runs the tiled pure-JAX prepacked kernel under ordinary
   autodiff. Same force either way; only the former buys the O(N) reverse memory.

**The payload is built on the HOST, in NumPy** — this is load-bearing, not a style
choice. Under an outer `jax.jit` every `jnp` op is staged out *even on concrete
constants*, so the `max(counts)` reduction that sizes the padded block came back as a
tracer and the whole call died with `ConcretizationTypeError`. A first fix that reduced
in NumPy but returned `jnp` arrays then leaked *those* across traces via the payload
memo (`UnexpectedTracerError`), because the memo hands trace-1 arrays to trace-2. NumPy
in and NumPy out is the only stable answer; it also puts the frozen topology in the
jaxpr as a constant instead of restaging index arithmetic every call. Both failures are
pinned by `test_fast_lane_survives_an_outer_jit` and
`test_payload_builder_rejects_traced_topology`.

**Correctness (the precondition for reading any timing).** The two lanes agree to
round-off on `differentiable_accelerations` end to end — forward rel-L2 ~1.5e-16,
d/d(positions) ~2.4e-16, d/d(masses) ~1.2e-16 at N=256 and 1024, complex and real — and
every benchmark row below reports a bit-identical loss checksum across modes. This is
the check that matters most, and it is the one the grouped-M2L episode above shows you
cannot skip: an assumed equivalence there was really 7.1e-02 vs 1.1e-02 against the
exact sum.

**Reconciling the two near-field `custom_vjp`s** (the plan's open item 6). PR-2's
`nearfield_{fused_leaf,leafpair}_pallas_cvjp` are **kept as unit-level VJP oracles**, not
promoted to the grad path: their reverse is `jax.vjp` of a dense twin that materialises a
`(leaves, W_t, K, 3)` tensor (~50 TB at the fiducial large-N config). The production rule
is `_radix_fast_lane_prepacked_accel_cvjp`. The module comment in
`pallas/nearfield_fused_leaf.py` now says so, so nobody wires a second grad-path caller.

### The measurement — and why the default did NOT flip

`differentiable_accelerations`, A100, complex basis, order 4, leaf 32, θ=0.5, float64,
jitted, min of 8 steady-state reps; `reverse = (fwd+bwd) − fwd`. The loss checksum is
**bit-identical across all three modes at every N**, so these are three traversals of one
force, not three forces.

| N | bucketed fwd / rev (ms) | leaf-major pure-JAX | leaf-major fused-Pallas |
|---|---|---|---|
| 256 | 0.469 / 0.719 | 0.490 / 1.358 | **0.451 / 0.618** |
| 1024 | 1.039 / 2.656 | **0.770** / **2.419** | 1.233 / 2.544 |
| 4096 | 35.03 / 0.647 | 34.20 / 3.953 | 33.92 / n/a |

**Verdict at small N: no time win.** The plan predicted ~6× forward and ~20× reverse
from the isolated leaf-major numbers (0.147/0.113 ms pure-JAX, 0.532/0.170 fused vs
~0.85/~2.4 bucketed in-context). In context the spread is ±20%, with no mode winning at
both N: fused-Pallas is best at 256 (1.04× fwd, 1.16× rev) and *worse* than bucketed at
1024 (0.84× fwd); pure-JAX leaf-major is best at 1024 (1.35× fwd, 1.10× rev) and 1.9×
*worse* on the 256 reverse.

**But N≤4096 was the wrong scale to judge this at — see "the galaxy-scale result" below.
The payoff is reverse-pass MEMORY at 200k, where it is the difference between an OOM and
a working gradient.** The lane is wired, gated, and opt-in behind
`JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`; at galaxy N it is not optional, it is
the only radix configuration that runs.

**Why — measured, not inferred.** Re-running the plan's ablation (patch the near field to
return zeros; share = (full − ablated)/full) in *both* modes at N=1024 isolates the
near-field cost inside each lane:

| mode | full fwd / rev (ms) | near field off | near-field share | near field itself |
|---|---|---|---|---|
| bucketed | 1.050 / 2.512 | 0.186 / 0.198 | 82.3% / 92.1% | 0.864 / **2.314** |
| leaf-major fused-Pallas | 1.250 / 2.476 | 0.176 / 0.221 | 85.9% / 91.1% | 1.074 / **2.255** |

The attribution reproduces exactly (82/92% here vs the plan's 83/91%), so the near field
really is the dominant term — but **the leaf-major near field costs the same in context
as the bucketed one**, 2.26 ms of reverse against 2.31 ms, even though it measures
0.170 ms in isolation. That is a ~13× in-context inflation for leaf-major, alongside the
~30× already known for bucketed. The overhead is therefore **not the bucketing
representation**; it is the per-particle gather into leaf-major layout and the scatter
back out, which *both* lanes pay. The remaining lever is reducing near-field pair/padding
*count* (`leaf_size`, `theta`), not re-spelling the traversal.

That makes this the third time an isolated micro-benchmark has mispredicted this
subsystem — by 30× (bucketed), then by ~6–20× (leaf-major). Attribute in-context or not
at all.

**Caveats on the table.** The N=4096 row is not usable: the forward is ~34 ms in *all
three* modes (so whatever dominates there is not the near field), its run-to-run variance
exceeds the whole backward, and the subtracted reverse came out negative for
fused-Pallas. That N needs a harness that times the backward directly rather than by
subtraction. The fused-Pallas column also carries `use_pallas=True`, which additionally
selects the Pallas M2L rot-scale on the *real* basis — inert here (complex basis,
identical checksums), but not a clean A/B if these numbers are ever re-measured on real.

### Galaxy scale reached — 200k AND 1M, forward and reverse

Everything above is N≤4096, which turned out to be the wrong scale to judge the fast lane
at. Measured at the target scale (A100 40 GB, fp32, θ=0.7, order 4, clustered galaxy
disc; each stage's own peak device memory; "steady" is the min of 3 warm reps, so first
call = compile + execute):

| N | config | prepare | fwd 1st / steady | reverse 1st / steady | reverse peak |
|---|---|---|---|---|---|
| 200k | radix, leaf 64, **bucketed** | 114 s / 3.61 GB | 6.9 s / — | **OOM** | 30.11 GB, then a 7.68 GiB request failed |
| 200k | radix, leaf 64, **fast lane** | 116 s / 3.61 GB | 7.0 s / **1.60 s** | 424 s / **6.27 s** | **6.82 GB** |
| 200k | **`large_n_gpu`**, leaf 256 | 57 s / 0.29 GB | 6.1 s / **0.86 s** | 889 s / **2.59 s** | **2.62 GB** |
| 1M | **`large_n_gpu`**, leaf 256 | 79 s / 1.97 GB | 8.0 s / **2.50 s** | 1470 s / **66.4 s** | **11.07 GB** |

All gradients finite in both positions and masses at every surviving row. Forward
agreement between the two radix lanes at 200k: mean |a| 1.244721e-01 vs 1.244722e-01
(fp32 summation order).

**The radix OOM, identified.** The failing allocation is 8,243,200,000 B = exactly
`503,125 × 64 × 64` fp32 — **one scalar per particle pair over every near-field edge**.
Same residual class PR-3 fixed one lane over (§B above), and precisely what
`_leafpair_accel_analytic_vjp` never constructs: a hand-written `bwd` is never itself
differentiated, so its intermediates are tile-bounded transients rather than retained
per-scan residuals. Hence >4.4× lower reverse peak and, more to the point, **the
difference between OOM and a working gradient**.

**Correction to the framing above.** The ±20% wall-clock wash at N≤4096 is real but it is
not the figure of merit for this lane. It buys *asymptotic reverse memory*, invisible
until the pair count is large enough for the retained tile to dominate — and by then it
is not a speedup, it is feasibility. Sizing the optimisation by time at small N missed
that as thoroughly as isolated micro-benchmarks missed the in-context cost.
**Recommendation: enable `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1` for any
differentiable radix run at N ≳ 10^5.**

**The `large_n_gpu` production preset is now differentiable** — `LargeNPreparedState` is
no longer rejected (see `runtime/_large_n_grad.py`, the re-evaluation seam that re-runs
P2M→M2M→M2L→L2L on live inputs against the frozen `compact_far_pairs` list, requiring
`retain_far_pairs_for_grad=True`). It is also the *best* configuration at galaxy scale on
every axis measured: 2.62 GB reverse peak at 200k against the radix fast lane's 6.82 GB,
and 2.59 s against 6.27 s. This closes the "Still open" item recorded under PR-3.

### Why the 1M reverse is 27× the forward — attributed

`prepare_state` first: **it is not slow, it is compiling.** At N=1M the first call is
82.3 s and the second and third are **1.47 s** — faster than the 2.50 s forward. Setting
`jax_compilation_cache_dir` takes a *cold process* from 81 s to **16.8 s** (4.8×). The
stage timers (nested, so they do not partition) put the cold cost in the dual-tree
artifact build (38 s), downward compute (16 s), upward compute (13 s) and tree build
(13 s); near-field prep is 1.45 s. Nothing in the computation is worth optimising.

The reverse is a different story. Ablating the seam's own `include_near`/`include_far`
split (no monkeypatching) at both N:

| N | config | fwd ms | reverse ms |
|---|---|---|---|
| 200k | near+far | 118.1 | 1731.4 |
| 200k | near only | 90.5 | **1632.1** (94.3% of the reverse) |
| 200k | far only | 27.7 | 61.9 |
| 1M | near+far | 1659.5 | 66326.3 |
| 1M | near only | 1023.3 | **64702.4** (97.6% of the reverse) |
| 1M | far only | 638.8 | 699.1 |

So the M2L/L2L/P2M cascade is irrelevant (0.7 s of 66 s at 1M) and the near field is
everything. Three compounding causes, each measured:

**1. The reverse tracks PADDED slots, not real work.** The prepacked payload is a
rectangle padded to the *global maximum* neighbour count:

| N | leaves | slots | padded slots | valid | fill | max nbrs | mean nbrs |
|---|---|---|---|---|---|---|---|
| 200k | 782 | 1024 | 800,768 | 362,528 | 45.3% | 781 | 463.6 |
| 1M | 3907 | 8192 | 32,006,144 | 4,642,572 | 14.5% | 3906 | 1188.3 |

Padded slots grow **40×** for 5× N; valid pairs grow only 12.8×; the measured reverse
grows **38.3×**. It tracks the padding almost exactly.

**2. `max nbrs == leaves − 1`, at every N and every geometry.** Some leaf is a near-field
neighbour of *every other leaf* — verified on a flattened disc, a Plummer sphere **and a
uniform cube**, at both N. That single leaf sets the padding width for all of them. The
mean also grows with N (uniform 168→233, Plummer 339→524, disc 464→1188), so valid
near-field pairs themselves grow 6.9–12.8× per 5× in N. This is a **tree/MAC** property,
not a gradient one — it inflates the forward too, and it is accuracy-safe (an
over-inclusive near field is more exact, just slower). Worth its own investigation.

**3. The reverse had no empty-tile skip while the forward does.** The forward's
`_accumulate_target_block_tile_sequence` wraps each tile in
`lax.cond(jnp.any(tile_source_valid), ...)`; `_leafpair_accel_analytic_vjp` did not, so it
paid full price on pure padding. That is why the near-field reverse/forward ratio is 18×
at 200k and **63×** at 1M — it worsens exactly as fill drops from 45% to 14.5%.

**Fix applied (partial):** the same skip is now in `_leafpair_accel_analytic_vjp`, gated
by `JACCPOT_GRAD_REV_SKIP_EMPTY_TILES` (default on). It is semantics-preserving — every
term is already masked by `valid_slot`, so an all-invalid tile contributes exactly zero.

| | 200k reverse | 1M reverse |
|---|---|---|
| skip off | 1731.4 ms | 66326.3 ms |
| skip on | 1723.4 ms (1.00×) | **53358.6 ms (1.24×)** |

Only ~20% at 1M and nothing at 200k, because the `lax.cond` predicate spans a batch of
**8 different leaves** — a tile survives if *any* of them still has a valid slot there,
and leaves have very different neighbour counts.

**Occupancy sort was the obvious next step. It was implemented, measured, and reverted:
it is ~7× SLOWER.** The forward pairs its skip with `occupancy_sort` for exactly this
reason, so grouping leaves by neighbour count should have let whole tiles drop. Measured
on the near-only reverse, against a contention factor taken from the *far-only* row (the
change cannot touch the far field, so any movement there is the shared GPU):

| N | near-only, skip | near-only, +sort | far-only probe | corrected |
|---|---|---|---|---|
| 200k | 1632 ms | 20987 ms | 61.9 → 119.5 ms (1.9×) | **~6.8× worse** |
| 1M | 53284 ms | 66138 ms | 701.8 → 854.2 ms (1.22×) | **~2% worse (noise)** |

So it is catastrophic at 200k and a wash at 1M — the lower 14.5% fill there gives the sort
~7× more tiles to drop, and that only just pays for what it costs.
Cause: leaves arrive in **Morton order**, which makes the per-tile source
gather `leaf_positions[safe_src]` and the `.at[safe_src].add` scatter spatially coherent;
reordering by occupancy destroys that locality and the extra memory traffic dwarfs the
arithmetic the skip saves. The forward tolerates its own sort because its flattened
`batch × tile × block` gather has a different access pattern. The reverse keeps its
Morton order; see the comment block in `_leafpair_accel_analytic_vjp` so this is not
retried blind.

### Occupancy tiers — the padding fix, and its crossover

Salvaging the sort: tier the leaves by occupancy but change **only which target leaves a
pass visits and how wide a slot window it reads**. Source ids are never renumbered and
`leaf_positions` is never permuted, so the source-side locality that killed the sort is
untouched; leaves keep Morton order *within* a tier. Widths are static, computed on the
host by `build_leafpair_reverse_tiers` from the frozen validity mask (concrete there;
inside the bwd rule it is a residual, hence a tracer), and ride through the `custom_vjp`
in `nondiff_argnums`.

Predicted slot-visit reduction, canonical galaxy config:

| N | 2 tiers | 4 tiers | 6 tiers | 8 tiers | ceiling (1/fill) |
|---|---|---|---|---|---|
| 200k | 1.31× | 1.64× | 1.93× | 1.95× | 2.2× |
| 1M | 2.09× | **4.33×** | 5.02× | 5.61× | 6.9× |

Measured, near-field reverse only, contention-corrected against the untouched far-only row:

| N | baseline (skip only) | tiered | far-only probe | verdict |
|---|---|---|---|---|
| 200k | 1612 ms | 13701 ms | 62.1 → 121.7 ms (1.96×) | **4.3× worse** |
| 1M | 53284 ms | 31305 ms | 701.8 → 786.2 ms (1.12×) | **1.91× faster** |

A real crossover, and it tracks the predicted reduction. Tiering costs throughput — the
target gather goes through an index array instead of a consecutive range, and one scan
becomes several smaller ones whose fixed cost is amortised over less work — so it only
pays once the saving is large. `build_leafpair_reverse_tiers` therefore **declines unless
the predicted reduction is ≥ 3.0×** (`JACCPOT_GRAD_REV_TIER_MIN_GAIN`), which classifies
both measured points correctly: 200k declines at every tier count, 1M accepts at 4+.
Calibrated on two points on one A100 — re-measure elsewhere. Off-switch
`JACCPOT_GRAD_REV_TIERED=0`; tier count `JACCPOT_GRAD_REV_TIERS` (default 4).

**The padded near field is O(N²).** The leaf/θ sweep below makes this concrete: padded
particle-pair work is ~5.2e10 in *every* configuration, because slots ≈ leaves so
`leaves × slots × leaf² = (leaves × leaf)² = N²`. The reverse, which pays padded cost,
has been running a direct sum. That is the real content of the 38× scaling.

### Configuration: what leaf_size and theta actually buy (N=200000, disc)

| leaf | θ | leaves | mean nbrs | fill | valid pair-work | padded pair-work | far pairs |
|---|---|---|---|---|---|---|---|
| 64 | 0.5 | 3125 | 1492 | 36% | 1.91e10 | 5.24e10 | 1232126 |
| 64 | 0.7 | 3125 | 844 | 21% | 1.08e10 | 5.24e10 | 902414 |
| 64 | **0.9** | 3125 | 561 | 14% | **7.18e9** | 5.24e10 | 681994 |
| 128 | 0.7 | 1563 | 650 | 32% | 1.66e10 | 5.24e10 | 273962 |
| 256 | 0.7 (default) | 782 | 464 | 45% | 2.37e10 | 5.25e10 | 73066 |
| 256 | 0.9 | 782 | 318 | 31% | 1.63e10 | 5.25e10 | 73668 |

Real near-field work is **3.3× lower at leaf 64 / θ 0.9** than at the default leaf 256 /
θ 0.7, traded for 9.3× more M2L pairs (far pairs are far cheaper per pair — the far-field
reverse is 62 ms against the near field's 1632 ms at 200k). But note padded work is
*constant*: **leaf/θ tuning cannot help the reverse until the padding is gone**, so these
levers are sequential, not independent.

### Why the near field is so large: the MAC, not the gradient

At N=50000, uniform cube, leaf 256, θ=0.7 (196 leaves): neighbour counts min 34, **median
100**, max 195 = `leaves − 1`; **105 of 196 leaves neighbour more than half the tree**;
`corr(leaf radius, neighbour count) = 0.75`. Leaf half-extents show why — leaf 97 is
`[1.0, 0.992, 0.25]` in a `[-1,1]` box, i.e. a Morton-range leaf spanning the *entire*
domain in x and y. With median leaf radius 0.415 and θ=0.7 the MAC keeps every pair
closer than `2 × 0.415 / 0.7 ≈ 1.19`, which is ~88% of the domain volume.

Note the single all-neighbouring leaf sets the **padding width** but is only ~1% of real
pairs (leaves with radius > 3× median: 1). The cost is the *median* leaf, and that is a
**tree/MAC** property — accuracy-safe (an over-inclusive near field is more exact) but it
inflates the forward too. Not a differentiability issue; the biggest single lever left,
and untouched.

**Remaining levers.** (a) Fix the MAC / leaf compactness — Morton-range leaves are not
spatially compact, and that is upstream of padding, valid work, and the forward.
(b) CSR sources instead of a rectangle, which would remove padding at any N rather than
only where tiering pays. Reverse peak is 11.07 GB of 40 GB, so all of this is throughput,
not a memory wall. Reverse *compile* is ~25 min at 1M.

**Method note.** Three of the four optimisations attempted across PR-4 and this section
were predicted to win and did not: the leaf-major traversal (±20% at N≤4096, though it
turned out decisive for *memory* at 200k), the empty-tile skip (1.00× at 200k, 1.24× at
1M), and the occupancy sort (~7× worse). The one clear win — the fast lane's analytic
reverse turning a 200k OOM into a working gradient — was not predicted by any of the
small-N benchmarks. Measure in context, at the target scale, before and after.

**Gates.** Golden byte-stability 13/13 with the flag off (the shipped path is untouched).
With the flag **on**, on the A100: 56/56 across `test_gradient_correctness.py`,
`test_grad_fmm_vs_directsum.py`, `test_custom_vjp_parity.py` and the new
`tests/unit/test_nearfield_fastlane_grad_path.py` — FD-vs-AD in positions and masses on
both bases, `grad(FMM)` vs `grad(direct-sum)`, NaN/inf hygiene, and the fast-lane cvjp
against its tiled twin on real Pallas. The new module adds the edge-set-count identity,
bucketed-vs-fast-lane value and gradient parity (near field alone and end to end), a
fused-Pallas end-to-end comparison gated on sm_80, the two outer-`jax.jit` regressions,
and the opt-in check.

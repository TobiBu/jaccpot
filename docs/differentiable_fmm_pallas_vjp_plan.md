# PR-2 implementation spec — differentiable Pallas fast lane (custom_vjp)

**Status:** ready to implement in a fresh session. Self-contained; assumes no prior context.

## 0. Goal & context

`jaccpot`'s FMM is now end-to-end differentiable on a single GPU (PR #49) and its
reverse pass is optimized on the **pure-JAX** path (PR #50): the vectorized `"bucketed"`
near-field + analytic `custom_vjp` rules give ~1ms forward at N=1024 and a ~1.3–4×
reverse. The remaining gap is the **fused Pallas kernels**: `pallas_call` has **no autodiff
rule**, so any path that runs a fused Pallas kernel is non-differentiable, and
`FastMultipoleMethod.differentiable_accelerations` explicitly rejects it (rejects
`LargeNPreparedState`, requires `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` unset, forces the
pure-JAX bucketed near-field).

**PR-2 delivers:** `custom_vjp` wrappers on the fused Pallas near-field and M2L kernels so
the *high-performance fused forward* becomes differentiable, then lifts the guard so the
differentiable path can run the fast lane.

**Reality check (do this first, it decides scope):** the pure-JAX bucketed near-field is
already ~1ms at N=1024 and fully differentiable. Before investing in the near-field Pallas
`custom_vjp`, **measure whether the fused Pallas near-field forward is meaningfully faster
than bucketed pure-JAX at the N you care about.** If the gap is small, prioritize the M2L
Pallas wrapper (or descope). The analytic near-field reverse already exists (PR #50,
`_pair_accel_cvjp_bwd`) precisely so it can be reused here.

## 1. Setup

- Branch off `feat/differentiable-fmm` **with PR #50 merged in**, e.g.
  `feat/differentiable-fmm-pallas-vjp`.
- **Hardware:** a real Pallas forward needs Ampere sm_80+ (A100/H100). Select a free GPU
  with `autocvd` **before** `import jax`
  (`from autocvd import autocvd; autocvd(num_gpus=1, least_used=True)`), per org policy.
- **CPU CI smoke without a GPU:** every wrap-target Pallas kernel accepts `interpret=True`
  and threads it into `pl.pallas_call` (M2L sets `backend=None` under interpret). So the
  `custom_vjp` can be exercised on CPU in interpret mode; only bit-for-bit GPU parity needs
  the A100.
- Env for tests: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, small `XLA_PYTHON_CLIENT_MEM_FRACTION`
  for xdist, `JAX_ENABLE_X64=1` (conftest forces it), optional `JACCPOT_TEST_JAX_CACHE_DIR`
  for a cross-run compile cache (first cold compile of a file is minutes).
- Baseline before starting: `pytest tests/characterization/test_fmm_golden.py
  tests/unit/test_custom_vjp_parity.py tests/unit/test_gradient_correctness.py
  tests/integration/test_grad_fmm_vs_directsum.py` all green.

## 2. The custom_vjp pattern that works (hard-won in PR #50 — READ THIS)

`pallas_call` has no JVP/transpose, so every wrapped kernel MUST supply a full `custom_vjp`.
Pitfalls that cost time in PR #50:

1. **Never close over tracers in a `custom_vjp`.** A factory that builds the rule closing
   over `softening_sq`/`G`/masks fails at runtime with
   `TypeError: No constant handler for type: DynamicJaxprTracer` once those are traced.
   Use a **module-level** `@jax.custom_vjp` and pass everything as explicit args.
2. **Non-differentiated ARRAY args:** do NOT put arrays in `nondiff_argnums` (that's for
   hashable statics only). Pass them as regular args and **return zero cotangents**
   (`jnp.zeros_like(arg)`) for them in `bwd`.
3. **Avoid `float0`:** pass bool/int mask & index arrays as **0/1 floats** (or keep them in
   `nondiff_argnums` only if hashable) so their returned cotangent is an ordinary float
   zero, not a `float0` array. Reconstruct bools inside via `mask_f > 0.5`.
4. **`nondiff_argnums`** are for the hashable Pallas statics only: `order`, `interpret`,
   `backend`, `num_warps`, `num_stages`, `target_subtile`, `basis_mode`, `rotation`.
   `custom_vjp.nondiff_argnums` are positional — add a thin positional wrapper since the
   Pallas kernels take these as keyword-only.
5. **`fwd` returns the primal identically; `bwd` runs once** (it is not re-differentiated).

Canonical shape (mirror `jaccpot/nearfield/near_field.py:_pair_accel_cvjp` from PR #50):

```python
@partial(jax.custom_vjp, nondiff_argnums=(N_static...,))
def _kernel_cvjp(diff_args..., nondiff_array_args..., *statics): ...
def _kernel_fwd(...): return primal, residual
def _kernel_bwd(*statics, residual, cot):
    return (*cotangents_for_diff_args, *zeros_for_nondiff_array_args)
_kernel_cvjp.defvjp(_kernel_fwd, _kernel_bwd)
```

**Reverse-rule choice per kernel:**
- Simplest, correct-by-construction: `bwd = jax.vjp` (autodiff) of the kernel's in-file
  **pure-JAX twin**. `fwd = Pallas`. Verified against the twin's autodiff (which is exact).
- Faster: an analytic reverse — reuse PR #50's building blocks (near-field tidal tensor;
  M2L is linear so its multipole-VJP is the adjoint / `jax.linear_transpose` of the twin).
- **fwd/bwd consistency caveat:** `fwd=Pallas` but `bwd=grad(twin)`; Pallas ≈ twin only to
  ~1e-5 (near-field: sequential `fori_loop` vs `jnp.sum`) or ~1e-10 (M2L: literal port). So
  the gradient is of the twin, not bit-exactly of the Pallas forward. This is standard for
  Pallas `custom_vjp` and fine (both are valid FMM gradients to force accuracy) — **document
  it**. Start with `bwd=grad(twin)`; swap to analytic only if the twin autodiff is a
  bottleneck.

## 3. Wrap points (kernel → pure-JAX twin → parity test)

### Near-field (`jaccpot/pallas/nearfield_fused_leaf.py`)
| Pallas kernel (def / call) | pure-JAX twin | notes |
|---|---|---|
| `nearfield_fused_leaf_pallas` (L193 / call L265) | `nearfield_fused_leaf_jax` (L83) | inputs `target_positions (nl,Wt,3)`, `target_mask`, `source_positions (nl,K,3)`, `source_masses`, `source_mask`, kw `softening_sq`,`G`; out `(nl,Wt,4)` (accel 0:3, potential lane 3) |
| `nearfield_leafpair_pallas` (L476 / L544) — **production prepacked lane** | `nearfield_leafpair_jax` (L360) | gathers source leaves by `source_leaf_ids`/`source_valid` inside via `lax.cond` |
| `nearfield_leafpair_pallas_decoupled` (L585 / L665) | reduces to `nearfield_leafpair_jax` when target pool == source pool | leaf-block chunked |
- Gate `pallas_nearfield_fused_supported()` (L65): GPU + sm_80. Selector `nearfield_fused_leaf` (L299).
- Dispatch seam in `near_field.py`: `compute_leaf_p2p_accelerations_radix_fast_lane` (L3311) →
  `_radix_fast_lane_pairs_pallas` (L3073, calls `nearfield_fused_leaf_pallas`) /
  `_radix_fast_lane_prepacked_pallas` (L3150, calls `nearfield_leafpair_pallas`).
- Parity tests to clone: `tests/unit/operators/test_pallas_nearfield_fused.py`
  (`test_interpret_matches_jax_reference` L85, `test_gpu_matches_jax_reference` L210,
  leafpair variants L137/L187).
- **Analytic reverse available:** PR #50's `_pair_accel_cvjp_bwd` (symmetric tidal tensor
  `J = -G Σ_s m_s(I/r³−3rrᵀ/r⁵)`, `Jᵀc=Jc`). Reusable, but note the leafpair kernels sum
  over a *gathered-by-leaf-id* source set; the analytic reverse must contract over the SAME
  pairs. Safer first cut: `bwd = grad(nearfield_*_jax twin)`.

### M2L (`jaccpot/pallas/`)
| Pallas kernel (def / call) | pure-JAX twin | reverse strategy |
|---|---|---|
| `m2l_real_fused.py:m2l_real_fused_pallas` (L209 / L250) | `m2l_real_fused_jax` (L173, vmaps `_m2l_real_one` L133) | **adjoint-M2L** for multipole grad (linear); autodiff-of-twin for delta/block grad |
| `m2l_complex_fused.py:m2l_complex_fused_pallas` (L375 / L431) | `m2l_complex_fused_jax` (L290, vmaps `_m2l_one` L217) | same |
| `m2l_core_z_real.py:m2l_core_z_real_pallas` (L87 / L139) — z-translate only | `operators/m2l_real_rot_scale.py:m2l_core_z_real` pure path (L60) | adjoint `Zᵀ` or autodiff-of-twin |
- Signatures: `(multipoles [N,C], blocks_to_z/from_z [N,p+1,md,md], r [N], *, order, interpret, backend)` → `[N,C]`.
- **Linearity:** `out = R_from · Z(r) · R_to · mult` — linear in multipoles, so the
  multipole-VJP is the transpose `R_toᵀ Zᵀ R_fromᵀ out_bar` (another M2L-shaped op). Easiest:
  `jax.linear_transpose` of the twin w.r.t. multipoles; delta/block grad via autodiff-of-twin.
  Geometry guards already in place (PR #49/#50): double-where on M2L deltas (`core.py`) and
  NaN-safe radius (`m2l_real_rot_scale.py`).
- Dispatch (all trace-time, `runtime/kernels/core.py`): `_apply_m2l` (L1708) →
  `_apply_real_m2l` (L1569) / `_apply_complex_m2l` (L1673); gates `_real_m2l_pallas_active`
  (L1519, env `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` + sm_80) / `_fused_complex_m2l_pallas_active` (L1594).
- Parity tests: `tests/test_m2l_complex_fused_pallas.py` (**cleanest template** — clone it),
  `tests/unit/operators/test_pallas_m2l_core_z_real.py`. **GAP: `m2l_real_fused` has NO
  parity test — add one** (forward: `m2l_real_fused_pallas` interpret vs `m2l_real_fused_jax`).

**Wrap the leaf `*_pallas` functions, NOT the dispatchers** (`_apply_m2l`, `_radix_fast_lane_*`):
the dispatchers branch Pallas-vs-pure-JAX at trace time; wrapping there would double-cover the
already-differentiable pure-JAX branch.

## 4. Step-by-step increments (small, verified, one commit each)

Commit only when its parity test + golden + gradient-correctness are green.

- **Step 0 — baseline.** New branch; confirm #50 merged; baseline tests green; on the A100,
  measure fused-Pallas near-field forward vs bucketed pure-JAX (decides near-field ROI).
- **Step 1 — M2L z-core (`m2l_core_z_real_pallas`).** Smallest, linear, already has a parity
  test. `custom_vjp` fwd=Pallas, bwd=transpose/autodiff-of-`m2l_core_z_real`. Extend the
  parity harness with a VJP-vs-twin check (interpret + GPU).
- **Step 2 — complex fused M2L (`m2l_complex_fused_pallas`).** Clone `test_m2l_complex_fused_pallas.py`
  for the VJP. Reverse: linear_transpose(multipoles) + autodiff-of-twin(deltas/blocks).
- **Step 3 — real fused M2L (`m2l_real_fused_pallas`).** **Add the missing forward parity
  test first**, then the VJP. Note: the real-M2L gate lacks the sm_80 check the complex one
  has (`pallas_m2l_real_*supported()` only require GPU/TPU) — a real-M2L custom_vjp could
  route to Pallas on pre-Ampere; guard accordingly.
- **Step 4 — near-field Pallas.** If Step 0 showed a worthwhile speedup: `custom_vjp` on
  `nearfield_fused_leaf_pallas` / `nearfield_leafpair_pallas[_decoupled]`. bwd = autodiff of
  the jnp twin first; optionally swap to PR #50's analytic tidal tensor. Clone
  `test_pallas_nearfield_fused.py` for the VJP checks.
- **Step 5 — lift the guard / expose the fast lane on the grad path.** In
  `FastMultipoleMethod.differentiable_accelerations` (`runtime/fmm_evaluate.py`): allow a
  fused-Pallas mode (e.g. drop the `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`-unset assertion
  once M2L is wrapped; add a `nearfield_mode_override` value or flag that selects the Pallas
  near-field). Keep pure-JAX bucketed as the default; make Pallas opt-in until fully proven.
  **`LargeNPreparedState` is out of scope** for a first pass — the differentiable re-eval seam
  (`_evaluate_prepared_state_at_positions_sorted`) only supports the radix `FMMPreparedState`;
  differentiating the large-N pipeline is a separate, larger effort. Keep rejecting `LargeNPreparedState`.
- **Step 6 — measure + docs.** Fused-Pallas-on-grad-path forward/reverse vs bucketed pure-JAX
  (extend `examples/differentiable_fmm_overhead.py` with a Pallas toggle). Update
  `docs/differentiable_fmm_design.md` (PR-2 section) and the `differentiable_accelerations`
  docstring. Reconcile `ARCHITECTURE.md §7` (Pallas gating) to note the fused kernels are now
  differentiable via `custom_vjp`.

## 5. Verification (gates for every step)

- **VJP parity (primary):** extend `tests/unit/test_custom_vjp_parity.py` with the reusable
  `assert_vjp_matches(f_custom, f_ref, primals)` — `f_custom` = Pallas-`custom_vjp`,
  `f_ref` = the pure-JAX twin (autodiff). Note tolerances: forward Pallas-vs-twin is
  ~1e-5 (near-field) / ~1e-10 (M2L) — loosen `rtol/atol` accordingly (the twin's *autodiff*
  reverse is the exact reference for the bwd; the forward match is looser). Run interpret=True
  on CPU CI and GPU on the A100.
- **Add the real-fused-M2L forward parity test** (gap).
- **Golden oracle** (`tests/characterization/test_fmm_golden.py`): forward byte-stable. The
  Pallas forward is what it already is, so wrapping in `custom_vjp` must not change values.
- **Gradient-correctness with Pallas ON** (A100): run `tests/unit/test_gradient_correctness.py`
  and `test_grad_fmm_vs_directsum.py` with the fused-Pallas grad path enabled (new flag) —
  grad(FMM) vs grad(direct sum) to FMM force accuracy; FD-vs-AD.
- **NaN/inf hygiene** at small separations / softening on-off on the Pallas path.

## 6. Reusable assets from PR #49/#50

- `tests/unit/test_custom_vjp_parity.py::assert_vjp_matches` — the bit-for-bit VJP harness.
- `jaccpot/nearfield/near_field.py:_pair_accel_cvjp` (+ `_pair_accel_cvjp_bwd`) — the working
  module-level `custom_vjp` template AND the analytic near-field tidal-tensor reverse.
- `jaccpot/operators/real_harmonics.py:_evaluate_local_real_with_grad_cvjp` — HVP-style
  `custom_vjp` example.
- Env-toggle pattern for A/B (`JACCPOT_ANALYTIC_L2P_VJP`, `JACCPOT_ANALYTIC_P2P_VJP`) — add
  `JACCPOT_PALLAS_VJP` similarly.
- `nearfield_mode_override` threading (`differentiable_accelerations` → seam →
  `_evaluate_prepared_tree` → `evaluate_tree`) — the pattern for threading a Pallas-on flag.
- Golden + gradient-correctness + overhead-benchmark as the standing gates.

## 7. Risks / open questions

- **ROI:** bucketed pure-JAX near-field is already fast + differentiable; confirm the fused
  Pallas near-field is worth wrapping (Step 0) before doing Step 4.
- **Near-field pair-structure matching:** the leafpair Pallas kernels gather sources by leaf
  id / decouple target & source pools; an *analytic* reverse must contract over exactly those
  pairs. Prefer `bwd = grad(twin)` first (correct by construction).
- **fwd/bwd consistency** (Pallas fwd vs twin-based bwd, ~1e-5) — acceptable, document.
- **Pallas is mandatory-differentiable-only via custom_vjp** — there is no autodiff fallback
  if a kernel is on the grad path unwrapped; the guard must ensure only wrapped kernels run.
- **`LargeNPreparedState`** differentiability is a separate effort (seam support); keep rejected.
- **Real-M2L gate lacks the sm_80 check** — a real-M2L custom_vjp could route to Pallas on
  pre-Ampere; guard or match the complex gate.

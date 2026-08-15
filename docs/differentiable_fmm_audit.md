# Differentiable FMM — Phase 0 audit

**Question.** Can `jax.grad`/`jax.vjp` produce exact gradients of the assembled
single-GPU FMM force w.r.t. particle **positions** and **masses**, under a
fixed-topology contract? Where does autodiff break or silently give the wrong
answer today, and what is the cheapest fix for each break?

**Headline.** Today **no end-to-end differentiable path through the FMM exists** —
`jax.grad` of the public force silently differentiates a direct O(N²) sum instead
(finding B1). But most of the reverse pass is *latent* in the pure-JAX kernels: the
translation cascade (M2M/M2L/L2L + rotations) is linear, so autodiff transposes it
exactly and cheaply. Only **two** small numeric-path code changes plus a new
masses-threading evaluation seam are needed. The fix is "build the differentiable
evaluate seam + harden two edges + verify", not "invent a reverse pass".

## Method

Two trace scripts on `feat/differentiable-fmm` (branched from `origin/main`
@ `820dc2f`), solidfmm basis (complex + real submodes), `use_pallas=False`,
`JAX_ENABLE_X64=1`, GPU selected via `autocvd`:

- **Probe 1** — N=64, θ=0.4, leaf=8, p∈{2,4}. Small/tight-MAC ⇒ *near-field only*
  (no M2L pairs). Tested: (A) `jax.grad(compute_accelerations)`; (B) `jax.grad`
  of the existing fixed-topology re-eval seam `_evaluate_prepared_state_at_positions_sorted`
  w.r.t. positions; (C) a prototype masses-aware seam w.r.t. masses.
- **Probe 2** — N=1024, θ=0.7, leaf=16, p∈{2,4}. Loose MAC ⇒ *far-field M2L/L2L
  active*. Same grad tests + NaN hygiene (near-coincident particles, softening
  on/off).

Exercising the far field in Probe 2 was essential: it surfaced a blocker (B3) that
a purely static read of the code, and the near-field-only Probe 1, both missed.

## Ranked blockers (numeric force path)

### B1 — `jax.grad(compute_accelerations)` silently differentiates a direct O(N²) sum *(deliberate, #1 issue)*
`compute_accelerations` detects tracers (`_contains_tracer`) and `vmap`s
`reference_direct_sum` instead of the FMM
([fmm_evaluate.py:117-142](../jaccpot/runtime/fmm_evaluate.py#L117-L142)).
**Empirical:** `grad(compute_accelerations)` vs `grad(direct_sum_gravitational_acceleration)`
agree to **rel-L2 = 7.2e-13** — i.e. the "FMM gradient" *is* the direct-sum gradient.
It also raises `NotImplementedError` for `return_potential=True` / derivative orders
under tracing.
**Fix:** do not route grads through `compute_accelerations`. Add a dedicated
differentiable entry point that consumes a concretely-built prepared `state` and
re-evaluates the numeric pipeline on live inputs (see design doc). *No change to the
existing short-circuit — it stays as the drop-in direct-sum path.*

### B2 — `prepare_state` (tree build) is not traceable
Host `jax.device_get` + numpy for topology cache keys / level offsets
([fmm_prepare.py:203-284](../jaccpot/runtime/fmm_prepare.py#L203), `fmm_state.py:856`)
and the yggdrax host `build_tree`. The differentiation target therefore **cannot**
include tree construction; topology must be built once, concretely, *before*
`jax.grad`.
**Fix:** contract — build `state` concretely, capture it as a constant, differentiate
only the numeric re-evaluation. This is the fixed-topology contract, not a defect.

### B3 — L2L level cascade uses a dynamic-bound `fori_loop` *(blocks far-field grads; found empirically)*
`_propagate_solidfmm_locals_by_level` computed `max_level = jnp.max(parent_levels)`
(a dynamic device value) and used it as `jax.lax.fori_loop(0, max_level + 1, …)`
([kernels/core.py](../jaccpot/runtime/kernels/core.py) L2L cascade). Reverse-mode AD
rejects `fori_loop`/`while_loop` with dynamic start/stop:

> `ValueError: Reverse-mode differentiation does not work for lax.while_loop or
> lax.fori_loop with dynamic start/stop values.`

This fires only when the far field is active (Probe 2), which is why Probe 1
(near-field only) and the static code read both missed it. `max_level` is pure tree
**topology** (max node depth) — constant under the fixed-topology contract, so it can
be a static Python `int`. The **upward** M2M cascade already does exactly this
(`internal_level_count = max(int(num_levels)-1, 0)`, a static bound —
[solidfmm_complex_tree_expansions.py:400](../jaccpot/upward/solidfmm_complex_tree_expansions.py#L400),
[real_tree_expansions.py:216](../jaccpot/upward/real_tree_expansions.py#L216)); the
downward L2L was simply never given the same treatment.
**Fix (implemented):** thread an optional static `num_levels` into
`_propagate_solidfmm_locals_by_level`; the (non-jitted) caller
`_prepare_solidfmm_downward_sweep` resolves `int(jnp.max(node_levels[:num_internal]))`
when the tree is concrete (a captured constant, i.e. the differentiable path), else
passes `None` for the unchanged dynamic fallback. **Numerics-preserving**: the loop
runs `max_level+1` iterations either way; forward output is byte-identical (gated by
the golden oracle).

### B4 — M2L invalid/padded pairs produce a NaN cotangent at `delta = 0` *(latent; now reachable after B3)*
In `_accumulate_m2l_fullbatch` and `_accumulate_m2l_chunked_scan`, padded pairs
collapse to `safe_src == safe_tgt == 0` ⇒ `delta = 0`; the real-basis M2L rotate-to-z
`jnp.linalg.norm(delta)` ([m2l_real_rot_scale.py:114](../jaccpot/operators/m2l_real_rot_scale.py#L114))
has a `0/0` NaN reverse cotangent, and the existing post-hoc `where(valid, …, 0)`
mask does not stop it (the NaN is born inside the M2L VJP as `0·NaN`).
**Fix (implemented):** double-where — substitute a nonzero `delta` for invalid pairs
*before* the M2L apply. Forward output byte-identical (invalid contributions are still
masked to 0 in the scatter).

### B6 — Rotation-to-z angle singularities in M2M/M2L/L2L *(blocks far-field grads; found empirically)*
After B3, the far-field reverse pass compiled but returned **all-NaN** gradients (finite
forward). Stage isolation (`jax_debug_nans`) placed the NaN in the downward M2L+L2L
sweep, not the upward M2M. The rotate-to-z angle builders compute
`rho = sqrt(x²+y²)` (infinite reverse grad at 0) and `arctan2` (0/0 reverse grad at the
origin); the real path adds `arccos(R22)` (infinite grad at the poles). With **COM**
centers the fixed-topology FMM genuinely hits the degenerate directions: **zero
displacement** (a single-child internal node shares its child's COM ⇒ an exact-zero L2L
translation) and **z-axis-aligned displacement** (`rho == 0`, e.g. lattice-aligned M2L
pairs). The `jnp.maximum(·, 1e-60)` radius floors keep the *forward* finite but do not
tame the reverse. Sites:
- complex: `_angles_from_delta_solidfmm` ([complex_ops.py:961](../jaccpot/operators/complex_ops.py#L961)) — shared by all complex M2M/M2L/L2L rotations;
- real: `_multipole_align_to_z_block` / `_multipole_align_from_z_block`
  ([real_harmonics.py:1250](../jaccpot/operators/real_harmonics.py#L1250)) — shared by real M2M/M2L/L2L (local = transpose), plus the unguarded `radii = jnp.linalg.norm(delta)` at [m2l_real_rot_scale.py:114](../jaccpot/operators/m2l_real_rot_scale.py#L114).

**Fix (implemented):** NaN-safe **double-where** in each angle builder (and the norm):
substitute a safe direction/radius on the degenerate set so the cotangent is a finite
subgradient (0), leaving the forward value unchanged (`arctan2(0,0)=0`, `sqrt(0)=0`,
`arccos(±1)∈{0,π}`; `cos()` is already bounded so the `arccos` clip is a no-op). Golden
oracle stays green (both bases, n256/n1024, p2/p4).

### B5 — Pallas kernels have no autodiff rule *(kept off the grad path)*
Pallas near-field (`nearfield_fused_leaf`) is the Ampere default but only via the
`LargeNPreparedState`/radix-fast-lane pipeline — **not** the `FMMPreparedState`
prepared-eval path used here. Pallas M2L is env-gated
(`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`, default off).
**Fix:** the differentiable entry point runs on a radix `FMMPreparedState` with
`jit_traversal=False` (pure-JAX `evaluate_tree`) and a `use_pallas=False` engine;
reject `LargeNPreparedState`.

## Empirical results

| Test | Config | Result |
|---|---|---|
| **B1** grad(compute_accelerations) vs grad(direct sum) | N=64, all p/basis | rel-L2 **7.2e-13** ⇒ silently the direct sum |
| **B** grad(pos seam), near-field only | N=64, θ=0.4 | 0 non-finite; FD-vs-AD rel-err **5.3e-10**; forward parity vs `compute_accelerations` **1.8e-16** |
| **C** grad(mass seam), near-field only | N=64, θ=0.4 | 0 non-finite; FD-vs-AD rel-err **1.5e-10** |
| **B3** grad(pos seam), far-field, *before* fix | N=1024, θ=0.7 | **ValueError** (dynamic `fori_loop`) |
| **B3+B6** grad(pos seam), far-field, after B3 only | N=1024, θ=0.7 | compiles but **all-NaN** grad (rotation-angle singularity) |
| **B3+B4+B6** grad(pos), far-field, *after* all fixes | N=1024, θ=0.7, complex | 0 non-finite; FD-vs-AD rel-err **2.2e-8** (FD step error) |
| **B3+B4+B6** grad(pos & mass), far-field | N=256, θ=1.0, complex **and** real | 0 non-finite; FD-vs-AD rel-err **≈1e-9** (both bases, both inputs) |
| forward byte-stability of all fixes | golden oracle (cart/real/solidfmm, n256/n1024, p2/p4) | **13/13 green** |

## Resolved design risks

- **Mass traceability:** masses flow through `compute_tree_mass_moments`
  ([fmm_sweeps.py:184](../jaccpot/runtime/fmm_sweeps.py#L184)) with **no** host
  `device_get` keyed on mass values (the only `device_get` reads topology
  `tree.node_ranges` and is skipped when `max_leaf_size` is passed). Probe 1(C)
  confirms mass gradients are numerically clean.
- **`_resolve_upward_num_levels` stash:** with `state` captured as a concrete
  constant, `tree.parent` is concrete and the static-levels stash is not needed.
- **COM centers:** node/expansion centers are center-of-mass (position- **and**
  mass-dependent). They are differentiated *through* (not held fixed); the FD
  reference must perturb the *same* frozen-topology function (see contract).

## Conclusion

A differentiable single-GPU FMM force is achievable via Route A (autodiff through the
pure-JAX fixed-topology re-eval). Required numeric-path changes are exactly **B3**
(L2L static loop bound) and **B4** (M2L delta double-where) — both implemented and
numerics-preserving — plus a new masses-threading evaluation seam and a public entry
point (design doc). No hand-written adjoint is required for correctness.

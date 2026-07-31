# PR-3 implementation spec — differentiable FMM near-field fast lane (the real reverse win)

**Status:** ready to implement in a fresh session. Self-contained; assumes no prior context.
Companion to `docs/differentiable_fmm_pallas_vjp_plan.md` (PR-2, done) and
`docs/differentiable_fmm_design.md`.

## 0. TL;DR

`FastMultipoleMethod.differentiable_accelerations` is end-to-end differentiable and, after
PR-2, runs a differentiable fused-Pallas M2L fast lane. But **profiling shows the near-field
P2P dominates the differentiable path — ~83% of the forward and ~91% of the reverse at
N=1024 / order 4** — and the grad path runs it in the **bucketed pure-JAX** mode, which is
memory/overhead-bound (~1000× off compute peak: pure gather/scatter + dense pair-tile
sprawl, not FLOPs). The M2L and the far-field cascade are ~0–15% and are NOT the
bottleneck. **PR-3 routes the differentiable near-field off the bucketed mode and onto the
already-existing differentiable leaf-major "fast lane" (`custom_vjp`, Pallas-fwd +
analytic-O(N)/tiled reverse), which is ~6× (fwd) / ~20× (reverse) faster in isolation** —
if the win survives in-context (see §3, the mandatory caveat).

This is the highest-ROI remaining differentiable-FMM optimization. It is mostly integration
(the differentiable near-field machinery already exists), not new kernels.

## 1. What is already done (branch `feat/fmm-pallas-vjp`, PR-2 + follow-ups)

All committed and pushed on `feat/fmm-pallas-vjp` (PR open vs `feat/fmm-analytic-custom-vjp`):

- **Differentiable fused-Pallas M2L, forward AND reverse.** `custom_vjp` on all three fused
  M2L kernels (`m2l_{core_z_real,complex_fused,real_fused}_pallas_cvjp`); wired into
  `_apply_{real,complex}_m2l`; `differentiable_accelerations` opts in via
  `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` (Ampere+). The **reverse** is a fully-fused
  on-chip analytic VJP kernel (`m2l_real_fused_vjp_pallas` @ `jaccpot/pallas/m2l_real_fused.py:389`,
  `m2l_complex_fused_vjp_pallas` @ `jaccpot/pallas/m2l_complex_fused.py:613`), default on
  (`JACCPOT_FUSED_M2L_VJP=1`, `=0` → autodiff-of-twin fallback).
- **`custom_vjp` on the fused near-field kernels** (`nearfield_fused_leaf_pallas_cvjp` @
  `jaccpot/pallas/nearfield_fused_leaf.py:730`, `nearfield_leafpair_pallas_cvjp` @ `:836`) —
  Pallas fwd + autodiff-of-twin reverse, mask/id args passed as floats. **These are wrapped
  but NOT wired onto the grad path** (see §2/§5 — likely superseded by the existing fast-lane
  cvjp; reconcile).
- Validated on A100: VJP parity (M2L 20, near-field 4), grad-correctness + grad-vs-direct-sum
  with fused M2L fwd+reverse (17→15 after a test-memory rebase), golden byte-stability (63).
- Docs: `docs/differentiable_fmm_design.md` "PR-2 outcomes"; `ARCHITECTURE.md` §6/§7.

**Measured M2L reverse ROI was ~1.0× at N=1024/p4** (kept default-on anyway: validated,
equal-cost, may help at high order / reverse memory at scale). That ~0 result is what
prompted the profiling below.

## 2. The finding (measure, don't infer — see §7)

Ablation on `differentiable_accelerations`, N=1024, complex, order 4, leaf 32, A100
(patch the near-field / M2L-accumulate to no-ops, time fwd and reverse):

| config | fwd ms | reverse ms |
|---|---|---|
| FULL | 1.02 | 2.61 |
| no near-field | 0.17 | 0.23 |
| no M2L accumulate | 1.02 | 2.61 (unchanged) |

→ **near-field = 83% of fwd, 91% of reverse; M2L accumulate ≈ 0%.** Order sweep is flat
(p 2→6: reverse 3.05→3.55 ms) ⇒ p-independent ⇒ not the far-field FLOP cascade. Leaf sweep:
reverse 5.07 (leaf 16) → 2.12 (32) → 0.66 (64) ms ⇒ cost scales with **bucket/pair count**,
i.e. the near-field bucketing overhead. `cost_analysis`: 0.68 flops/byte ⇒ deeply
memory-bound. Physics check: ~0.9M near-field interactions ≈ ~2 µs of A100 f64 compute vs
~2.4 ms measured ⇒ ~1000× off peak ⇒ pure overhead (bucketing gather/scatter + unfused
dense `(B,Wt,Ws,3)` tiles + padding), NOT arithmetic.

**Isolation is a trap here (it fooled this analysis twice):** the isolated bucketed pair
kernel (`_pair_accel_cvjp`) reverse is ~0.08 ms, but the near-field is ~2.4 ms *in-context* —
the 30× gap is the gather/scatter/bucketing *around* the pair math. Always attribute
in-context (ablation), never from an isolated micro-bench.

## 3. The proposed change — and the mandatory caveat

**Change:** stop forcing `nearfield_mode_override="bucketed"` in
`differentiable_accelerations` (`jaccpot/runtime/fmm_evaluate.py:743`) and instead route the
grad-path near-field through the **existing differentiable leaf-major fast lane**:

- `compute_leaf_p2p_accelerations_radix_fast_lane(..., differentiable=True)` @
  `jaccpot/nearfield/near_field.py:3893` — "routes the fused Pallas lanes through their
  `custom_vjp` wrappers so `jax.grad` works; reverse from this lane's tiled pure-JAX
  fallback." (`differentiable` kwarg @ `:3902`.)
- `_radix_fast_lane_prepacked_accel_cvjp` @ `near_field.py:3718` (fwd `:3779`, bwd `:3831`,
  `defvjp` `:3888`) — the prepacked leaf-pair `custom_vjp` with an **analytic O(N) leaf-pair
  reverse** (commits d667d8c, 228f81f). This is very likely the right entry to use, and it
  probably **supersedes the raw-kernel wrappers added in PR-2** (`nearfield_leafpair_pallas_cvjp`)
  — reconcile / don't double-maintain.

Isolation numbers (grad-path scale, A100): pure-JAX **leaf-major** near-field
(`nearfield_leafpair_jax`) = 0.147 ms fwd / 0.113 ms reverse; fused-Pallas leaf-major =
0.532 / 0.170; bucketed in-context = ~0.85 / ~2.4. So the leaf-major *representation* is the
win, and **pure-JAX leaf-major may beat the Pallas kernel** (the Pallas per-slot `lax.cond`
gather has overhead) — evaluate both.

**MANDATORY CAVEAT — verify in-context before believing any of this.** The leaf-major path
also gathers/scatters; it could blow up in-context the same 30× the bucketed path did.
Isolation mispredicted the bucketed path by 30×. So the FIRST real step is to wire the grad
path to the fast lane and **measure `differentiable_accelerations` fwd+reverse end-to-end**
(ablation-style), not trust isolation. If it does not win in-context, stop and reassess (the
overhead may be intrinsic to the gather/scatter, in which case the lever is reducing near-field
*pair/padding count*, e.g. via `leaf_size`/`theta` — leaf 64 was ~3× faster, though at small
N that trades toward near-direct summation).

## 4. Step-by-step

- **Step 0 — reproduce the attribution** (§7). Confirm near-field ≈ 83/91% on your box, so you
  are optimizing the right thing. Baseline green:
  `tests/unit/test_custom_vjp_parity.py tests/unit/test_gradient_correctness.py
  tests/integration/test_grad_fmm_vs_directsum.py tests/characterization/test_fmm_golden.py`.
- **Step 1 — understand the two near-field paths.** Map how `differentiable_accelerations`
  reaches the near-field: `differentiable_accelerations` (`fmm_evaluate.py:543`) →
  `_evaluate_prepared_state_at_positions_and_masses_sorted` → `_evaluate_prepared_state_at_positions_sorted`
  (`:378`, forces `nearfield_mode_override="bucketed"` at `:743`) → `evaluate_tree`
  (`fmm_evaluate.py:652`) → `compute_leaf_p2p_accelerations` (`near_field.py`, bucketed mode).
  Separately, `compute_leaf_p2p_accelerations_radix_fast_lane(differentiable=True)` is the
  leaf-major differentiable entry but is NOT currently on this path. Determine whether the
  radix `FMMPreparedState` carries the fast-lane `payload` (target/source particle ids,
  `source_leaf_ids`, masks) the fast-lane entry needs — `state.nearfield_source_leaf_ids` /
  `nearfield_target_leaf_ids` / `nearfield_valid_pairs` exist; check they match the payload.
- **Step 2 — prototype + verify in-context (THE decision gate).** Thread a
  `nearfield_mode_override` value (e.g. `"fast_lane"`) or a boolean that makes the grad-path
  seam call the fast-lane differentiable near-field instead of bucketed. Prototype minimally
  (even a monkeypatch of the near-field entry to route through
  `compute_leaf_p2p_accelerations_radix_fast_lane(differentiable=True)`), then measure
  `differentiable_accelerations` fwd + reverse at N=256/1024/4096 vs bucketed. **Go/no-go:**
  proceed only if it wins in-context. Evaluate pure-JAX leaf-major vs fused-Pallas leaf-major.
- **Step 3 — wire it properly** (if Step 2 wins). Thread the fast-lane near-field mode through
  `differentiable_accelerations` → seam → `_evaluate_prepared_tree`/`evaluate_tree` → the
  near-field dispatch (mirror the M2L Step-5 pattern and the existing `nearfield_mode_override`
  threading). Keep **bucketed as the default/fallback**; make the fast lane opt-in until
  proven, then flip the default if it robustly wins. Reconcile with the PR-2 raw-kernel
  near-field `custom_vjp` wrappers (§1) — prefer the existing `_radix_fast_lane_prepacked_accel_cvjp`;
  drop/duplicate-guard the PR-2 wrappers if superseded.
- **Step 4 — gates + measure + docs.** Grad-correctness + grad-vs-direct-sum with the fast-lane
  near-field ON (A100); golden byte-stable (default path unchanged); NaN/inf hygiene at small
  separations. Extend `examples/differentiable_fmm_overhead.py` with a near-field-mode toggle.
  Update `docs/differentiable_fmm_design.md` + `ARCHITECTURE.md`.

## 5. Verification gates (every step)

- **Grad-correctness with the fast-lane near-field ON** (A100): `tests/unit/test_gradient_correctness.py`
  + `tests/integration/test_grad_fmm_vs_directsum.py` — FD-vs-AD (positions & masses,
  complex+real), grad(FMM) vs grad(direct-sum) to FMM force accuracy, NaN/inf hygiene.
- **Golden byte-stable** (`tests/characterization/test_fmm_golden.py`): default (flag-off) path
  must be unchanged — the fast lane is opt-in.
- **VJP parity**: extend `tests/unit/test_custom_vjp_parity.py::assert_vjp_matches` for the
  fast-lane near-field cvjp (interpret CPU + A100), if not already covered.
- **End-to-end overhead** (the actual point): `differentiable_accelerations` fwd + reverse,
  fast-lane vs bucketed, N sweep. This is the deliverable metric.

## 6. Leftovers from the PR-2 plan (`differentiable_fmm_pallas_vjp_plan.md`)

Done in PR-2: Steps 0–3 (M2L cvjp z-core/complex/real + missing real-fused fwd parity test +
sm_80 gate fix), Step 4 (near-field kernels wrapped), Step 5 (M2L fast lane exposed on grad
path, opt-in), Step 6 (measure + docs). Plus the beyond-plan fused M2L reverse.

Still open / deferred:

1. **Near-field on the grad path** — this doc (PR-3). The plan's Step 5 "near-field half" /
   `nearfield_mode_override` threading was left undone; PR-2 descoped it on a (misleading)
   isolated 1.1× ROI. The in-context ablation says it is actually the dominant cost.
2. **`LargeNPreparedState` differentiability** — still rejected in `differentiable_accelerations`
   (`fmm_evaluate.py:596`). The differentiable re-eval seam only supports the radix
   `FMMPreparedState`; differentiating the large-N pipeline is a separate, larger effort
   (seam support first, then its fused kernels). Out of scope.
3. **Cartesian basis** — `differentiable_accelerations` requires `expansion_basis=="solidfmm"`
   (complex/real); Cartesian is rejected. A gap, not scheduled.
4. **Complex-basis L2P analytic `custom_vjp`** — real-basis L2P has one (PR-1); complex L2P
   uses autodiff. Low ROI (L2P is a small share); optional.
5. **M2L fused reverse default** — `JACCPOT_FUSED_M2L_VJP=1` default-on despite ~0 ROI at
   N=1024/p4; revisit at high expansion order (where the M2L share grows) or keep as-is.
6. **Reconcile the two near-field `custom_vjp` implementations** — PR-2's raw-kernel wrappers
   (`nearfield_{fused_leaf,leafpair}_pallas_cvjp`) vs the pre-existing fast-lane
   `_radix_fast_lane_prepacked_accel_cvjp`. Avoid double-maintenance (§3).

## 7. How to reproduce the attribution (so you optimize the right thing)

The whole session's lesson: **attribute in-context by ablation, never from isolated
micro-benches.** Recipe (A100; select a free GPU with `autocvd` before `import jax`, org
policy; `JAX_ENABLE_X64=1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`):

- Build a state (`FastMultipoleMethod(basis="complex", use_pallas=False, theta=0.5).prepare_state(pos, mass, max_order=4, leaf_size=32)`, N=1024), define `loss = sum(differentiable_accelerations(state, p, m)**2)`, jit `fwd` and `value_and_grad`; time steady-state (min of ~8 reps, `block_until_ready`). reverse = fwd+bwd − fwd.
- **Ablate**: monkeypatch `jaccpot.runtime.fmm_evaluate.compute_leaf_p2p_accelerations` to
  return `jnp.zeros((positions_sorted.shape[0], 3))` (near-field off), and
  `jaccpot.runtime.kernels.core._accumulate_m2l_{fullbatch,chunked_scan}` to return
  `locals_coeffs` (M2L off). share = (full − ablated) / full, for fwd and reverse.
- **Localize** with a `leaf_size` sweep (near-field bucketing grows as leaf shrinks) and a
  `theta` sweep, plus `jax.jit(vg).lower(pos,mass).compile().cost_analysis()` (flops/byte).
- Env note: the suite uses `pytest -n auto`; on this 72-core box that OOMs the GPU — run with
  `-n 4` and `XLA_PYTHON_CLIENT_MEM_FRACTION≈.12–.2` (see the memory note "running-tests-locally").
  Interpret-mode (`interpret=True`) exercises Pallas kernels + their `custom_vjp` on CPU CI;
  only bit-for-bit GPU parity needs the A100.

Reusable assets: `tests/unit/test_custom_vjp_parity.py::assert_vjp_matches`;
`examples/differentiable_fmm_overhead.py` (has a fused-M2L toggle — add a near-field-mode
toggle); the analytic near-field tidal-tensor reverse `_pair_accel_cvjp_bwd`
(`jaccpot/nearfield/near_field.py`).
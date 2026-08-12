# Handoff: GPU validation for the Tier 1 refactor

Paste the block below into a fresh session on the GPU box. It is self-contained.

Why it exists: Tier 1 (PRs #79–#88) moved the fused Pallas M2L wrappers and the radix
fast lane across module boundaries. ARCHITECTURE §10's checklist has four items, and
item 4 — *"an A100 Pallas-on vs pure-JAX parity run when the change touches Pallas"* —
is the one CPU cannot supply. It was **not** run and is owed. This handoff also sweeps
up the two GPU items `docs/handoff_g10_gpu_validation.md` left open (its item 1's four
tests, and the fp32 gap on the fused lanes), because the same session can settle them.

---

Validate the Tier 1 refactor on real GPU hardware. Everything is implemented and green
on CPU; **nothing is expected to need code changes.** Your job is to run the parts CPU
cannot reach, and to report numbers.

## Context

Tier 1 of `docs/refactor_audit_2026-08.md`. Ten PRs, each a structural change that is
**not supposed to alter any number**:

| PR | what moved |
|---|---|
| #85 | `runtime/kernels/core.py` → `_shared` / `_downward_prep` / `_m2l` / `_l2l` / `_evaluate`, `core.py` becomes a re-export aggregator |
| #80, #83, #84 | `nearfield/near_field.py` 4314 → 1388 lines: the radix fast lane into `_fast_lane.py`, then `_kernels` / `_scatter` / `_schedules` / `_large_n_blocks` |
| #86 | two phases out of `PrepareMixin._prepare_state_dual_and_downward` |
| #79, #82, #87, #88, #81 | merged already: the `real_harmonics` split, the large-N env reader, thresholds, docstrings |

**The two that touch Pallas are the reason you are here.** #85 moved
`_m2l_real_batch_kernel_fused_pallas`, `_m2l_complex_batch_kernel_fused_pallas` and both
`*_pallas_active` gates into `kernels/_m2l.py`. #83 moved
`_radix_fast_lane_prepacked_accel_cvjp` and all five function-local
`pallas/nearfield_fused_leaf` imports into `nearfield/_fast_lane.py`. Every function body
moved **verbatim** — 3560 and 902 non-blank lines respectively, verified line-by-line —
so the expectation is exact equality, not "close".

Read first, in this order — binding:

- `ARCHITECTURE.md` §10 (the four-item checklist, and the warning about a defaulted
  `basis_mode` being silently satisfied by a caller that forgets it) and §7.
- `agent_guides/NUMERICS_AND_JAX.md` §1, §4, §6, and `CLAUDE.md`.
- The module docstring of `jaccpot/runtime/kernels/_m2l.py` — it states the dispatch
  invariant, the two Pallas equivalences, and the G.11 lesson about the four
  near-duplicate accumulators.
- `docs/handoff_g10_gpu_validation.md` if you get to items 5 and 6 below.

## Environment

- **Use `autocvd` to select a free GPU.** The machines are shared; do not grab a busy one.
- `conda activate jaccpot-verify` (or the box's equivalent), `JAX_ENABLE_X64=1`.
- The fused kernels need **Ampere+ (sm_80)**. Print all three support flags before
  anything else and put them in the report:

  ```python
  from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported
  from jaccpot.pallas.m2l_complex_fused import pallas_m2l_complex_fused_supported
  from jaccpot.pallas.nearfield_fused_leaf import pallas_nearfield_fused_supported
  print(pallas_m2l_real_fused_supported(),
        pallas_m2l_complex_fused_supported(),
        pallas_nearfield_fused_supported())
  ```

  If any is False, say which and stop for that item. **A run on the wrong card looks
  green while asserting nothing** — that is the single most likely way this handoff
  produces a false pass.

## Branches

The five merged PRs are on `main`. The five under review are separate branches; check
them out one at a time. If a branch is behind `main`, merge `main` into it first and say
that you did.

```
refactor/tier1-1.6-split-kernels-core                 # PR #85  <- the §10 item
refactor/tier1-1.5-split-nearfield-seams-1-3          # PR #84  (carries #83, #80)
refactor/tier1-1.7-extract-dual-downward-phases       # PR #86
```

`#84` contains `#83` and `#80`, so validating `#84` covers the whole near-field chain.

## What to run

### 1. ARCHITECTURE §10 item 4 — Pallas-on vs pure-JAX, on the branch and on `main`

This is the headline item. The two fused-M2L parity suites assert the equivalence that
§10 item 4 is about; on CPU they only run `interpret=True`, and on a GPU the
`interpret=False` halves become live.

```bash
git checkout refactor/tier1-1.6-split-kernels-core
JAX_ENABLE_X64=1 pytest -q -n 0 \
  tests/unit/operators/test_m2l_real_fused_pallas.py \
  tests/unit/operators/test_m2l_complex_fused_pallas.py \
  tests/unit/operators/test_pallas_nearfield_fused.py \
  tests/unit/test_custom_vjp_parity.py -rA
```

**Report the skip list.** `-rA` is there so you can confirm the `interpret=False`
parametrisations actually ran. Then repeat the identical command on `main` and diff the
two reports: the point is not just "green on the branch" but "the same tests, the same
results, on both sides".

### 2. The Pallas path engaged end to end, both sides, bit-compared

§10 item 1 was satisfied on CPU with 102 arrays at `rtol=0`, but only over the pure-JAX
lane. Do the same with the fused kernel switched on, and compare `main` against the
branch **by value, not by eye**:

```bash
for REF in main refactor/tier1-1.6-split-kernels-core; do
  git checkout $REF
  JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1 JAX_ENABLE_X64=1 \
    python - <<'PY'
import numpy as np, jax.numpy as jnp
from jaccpot import FastMultipoleMethod
rng = np.random.default_rng(0)
out = {}
for basis in ("real", "solidfmm"):
    for n, leaf, order in ((4096, 32, 4), (16384, 64, 6)):
        pos = jnp.asarray(rng.uniform(-1, 1, (n, 3)))
        mass = jnp.asarray(rng.uniform(0.5, 1.5, (n,)))
        fmm = FastMultipoleMethod(preset="accurate", basis=basis, theta=0.5)
        a = fmm.compute_accelerations(pos, mass, leaf_size=leaf, max_order=order)
        out[f"{basis}_n{n}_p{order}"] = np.asarray(a)
np.savez(f"/tmp/fused_{__import__('os').environ.get('REF','x')}.npz", **out)
for k, v in out.items():
    print(k, f"{np.linalg.norm(v):.17g}")
PY
done
```

Then `np.array_equal` the two `.npz`. **Expected: exactly equal.** If they differ at all,
that is a finding and the split is not what it claims — capture the max absolute and
relative difference per case.

### 3. The near-field chain, same treatment

```bash
git checkout refactor/tier1-1.5-split-nearfield-seams-1-3
JAX_ENABLE_X64=1 pytest -q -n 0 \
  tests/unit/core/test_near_field.py \
  tests/unit/test_nearfield_fastlane_grad_path.py \
  tests/unit/operators/test_pallas_nearfield_fused.py -rA
```

The fast lane's `custom_vjp` is the load-bearing piece here (NUMERICS §1). Its residuals
were verified byte-identical on CPU **on both lanes** — note the default
`audit_reverse_residuals.py` invocation does *not* reach the fast lane, so use the gate:

```bash
JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1 JAX_ENABLE_X64=1 \
  python bench/audit_reverse_residuals.py --n 4096 --leaf 64 --json /tmp/res_branch.json
```

and compare the JSON against the same command on `main`. Also run the padding audit,
which CPU could only run with its GPU picker stubbed out:

```bash
python bench/audit_nearfield_padding.py --n 50000 200000
```

Compare against `main`. CPU-side values at leaf 64 / theta 0.6, for orientation: cube
N=4096 fill 0.9419 / N=16384 fill 0.5636; plummer N=4096 0.9844 / N=16384 0.9152.

### 4. Full suite on GPU, per branch

```bash
JAX_ENABLE_X64=1 pytest
```

CPU reference on all three branches: **926 passed, 67 skipped**, and collected counts
993/1031 with 898 not-slow. On GPU the *passed* count should **rise** as GPU-gated tests
stop skipping, and failures should be **zero**. Anything that fails must be bisected
against `main` before you attribute it to a branch — several things in this repo fail on
`main` (see item 7).

### 5. The four G.10 tests that have never executed

Still open from `docs/handoff_g10_gpu_validation.md` item 1. Run them on `main` now that
the G.10 fix is merged:

```bash
JAX_ENABLE_X64=1 pytest -q -n 0 tests/unit/operators/test_transverse_degeneracy_jvp.py \
  -k "fused_pallas or production" -rA
```

Expect `test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[False]`, its complex
twin, and the two `test_the_production_*_fused_m2l_kernel_carries_the_axis_derivative`
to run rather than skip. Tolerance 1e-8. Signal size, measured from the CPU side: drop
the carrier and the real lane sits **1.98** off the pure-JAX lane, the complex one
**1.630e+00** — so this either passes clearly or fails clearly.

### 6. The fp32 gap on the fused lanes

Also still open. The analytic-branch band is `rho/r <= sqrt(eps)`: 1.5e-08 at fp64 but
**3.4e-04** at fp32, four orders wider, so ordinary tree geometry lands inside it.
`test_the_analytic_branch_holds_at_float32` pins the six *direct* cascades and **does not
cover either fused Pallas lane** — which is the lane most likely to run fp32 in
production. If you have budget, extend that test to the fused lanes and report the
residuals. If not, report that it is **still open** rather than letting a green run imply
otherwise.

### 7. Per-stage benchmark — and please interleave it

NUMERICS §4: per stage, compile time separate, uniform **and** concentrated.

```bash
python bench/profile_downward_breakdown.py
python bench/profile_fused_stage_ablation.py
python bench/profile_fused_gpu_util.py
```

Run `main` and the branch **back to back in one session**. This matters more than it
sounds: on the CPU box the same `main` worktree measured 15% apart in two windows a few
hours apart, which is larger than any effect these PRs could have, and one Tier 1 PR
description had to be corrected because of it. A non-interleaved table measures the
machine.

## What would count as a real problem

- **Any** difference in item 2's arrays. The bodies moved verbatim and the CPU grid is
  bit-identical at `rtol=0`, so a GPU difference means something about the split is not
  what it claims — not a tolerance question.
- `interpret=False` failing while `interpret=True` passes → the Triton lowering and the
  reference lowering disagree. Capture the error and the tolerance it missed by.
- A characterization golden moving. **Do not update it and do not relax the tolerance**
  (NUMERICS §3). Report it.
- A gradient-stage slowdown beyond noise on the interleaved profile.

## Traps that have already cost time

- **A GPU-gated test that self-skips looks identical to a passing one.** Always report
  the skip list; items 1, 3 and 5 all hinge on this.
- **A gradient test at a default leaf size covers no far field.** At `theta=0.5, N=128`,
  `leaf_size=4` accepts 110 M2L pairs and `leaf_size >= 8` accepts **zero**. Assert the
  pair count.
- **A symmetric construction certifies a broken gradient.** That is how G.10 survived
  review. Use asymmetric cotangents and directions.
- **`JACCPOT_RUNTIME_TYPECHECK=1 pytest tests/unit` is red on `main`** — 126 failed / 667
  passed, `fori_loop` bodies annotated `_i: int` receiving tracers (F40 in the audit). Do
  not attribute it to a branch, and do not fix it here.
- **`bench/audit_nearfield_padding.py` calls `autocvd` at import**, unconditionally. That
  is fine on the GPU box and is why item 3 can finally run it as shipped.

## What to report back

For each of items 1–7: the exact command, the pass/skip/fail counts, the skip list where
it matters, and the numbers. For item 2, the `np.array_equal` verdict per case. State the
GPU model and the three support flags at the top. If something is still open at the end,
say so explicitly — the audit would rather carry a known gap than an implied pass.

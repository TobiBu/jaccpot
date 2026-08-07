# Handoff: GPU validation for the G.10 transverse-gradient fix

Paste the block below into a fresh session on the GPU box. It is self-contained.

---

Validate the G.10 transverse-gradient fix on real GPU hardware. Everything is already
implemented and green on CPU; **nothing is expected to need code changes.** Your job is to
run the parts CPU cannot reach, and to report numbers.

## Context

Branch `fix/g10-rotation-transverse-jvp`. It fixes G.10: the rotate → z-translate →
rotate-back cascade guarded its alignment azimuth at `rho == 0` and returned a **zero**
transverse cotangent, so `d/dx` and `d/dy` of every M2L / M2M / L2L were wrong there. The
forward value was always exact. The fix supplies the derivative analytically from the
operator's rotational covariance.

Read first, in this order — they are binding and they already contain the answers:

- `docs/rotation_degeneracy_derivative.md` — the derivation, the generator calibration, and
  §5–§7 on what landed where. **§7 is your section.**
- `jaccpot/operators/_transverse_degeneracy_jvp.py` — the module docstring explains the
  scheme; `split_transverse_tangent`'s docstring explains where the switchover band goes.
- `agent_guides/NUMERICS_AND_JAX.md` §1 and §4, and `CLAUDE.md`.

Three shapes of the same rule are in play. Only the third is your concern, and it now
covers **both** bases:

1. `with_transverse_degeneracy_jvp` — the direct lanes (`m2l_real`, `m2m_real`, `l2l_real`,
   `m2l_rot_scale_real_batch`, and the complex trio).
2. The same, generalised over extra array arguments, plus
   `without_unresolvable_transverse_jvp` on the block builders — the precomputed-block
   lanes.
3. An `*_align_deltas` call **before** the radius and blocks are built, plus an
   `*_carry_axis_derivative` call **after** the kernel — the **fused Pallas M2L lanes**, in
   `jaccpot/runtime/kernels/core.py`:

   | basis | production function | the pair |
   |---|---|---|
   | real | `_m2l_real_batch_kernel_fused_pallas` | `m2l_real_fused_align_deltas` / `m2l_real_fused_carry_axis_derivative` |
   | complex | `_m2l_complex_batch_kernel_fused_pallas` | `m2l_complex_fused_align_deltas` / `m2l_complex_fused_carry_axis_derivative` |

   They need their own shape because `m2l_{real,complex}_fused_pallas_cvjp` are `custom_vjp`
   and JAX refuses `jax.jvp` through one, so a rule that differentiates the operator cannot
   wrap them. Neither piece touches the kernel; both primals return their input unchanged
   with no arithmetic, so the forward pass has no footprint at all, not even in the sign of
   zero. The complex lane shipped with only *half* the pair at first — the withdrawal
   without the carrier — and the test that caught it is item 1b below.

## Environment

- **Use `autocvd` to select a free GPU.** The machines are shared; do not grab a busy one.
- `conda activate jaccpot-verify` (or the box's equivalent), `JAX_ENABLE_X64=1`.
- The fused real kernel needs **Ampere+ (sm_80)**. If `pallas_m2l_real_fused_supported()`
  is False, say so and stop — the interesting half of this task cannot run.

## What to run

**1. The four tests that never execute on CPU.** This is the headline item — the whole
reason this handoff exists. Two of them are the `interpret=False` halves of assertions that
pass at ~2.7e-15 in interpret mode; the other two differentiate the *shipped* production
functions and are GPU-only outright, because those hardcode `interpret=False`.

```bash
JAX_ENABLE_X64=1 pytest -q -n 0 tests/unit/operators/test_transverse_degeneracy_jvp.py \
  -k "fused_pallas or production" -rA
```

  1a. `test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[False]` — real lane
      composed by hand, against `m2l_rot_scale_real_batch`.
  1b. `test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient[False]` — same,
      complex.
  1c. `test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative`
  1d. `test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative`

**Confirm none of the four skipped.** They self-skip off sm_80, so a run on the wrong card
looks green while asserting nothing — the same failure shape as the vacuous-far-field trap
below. Tolerance is 1e-8 (kernel-versus-cascade, not round-off). The signal they carry is
already measured from the CPU side: drop the carrier and the real lane sits **1.98** from
the pure-JAX lane, the complex one **1.630e+00**.

**2. Both reverse branches.** The kernel's `custom_vjp` has two backward implementations
and the env gate picks between them. Run the whole parity + transverse set under each:

```bash
JACCPOT_FUSED_M2L_VJP=1 JAX_ENABLE_X64=1 pytest -q -n 0 tests/unit/test_custom_vjp_parity.py tests/unit/operators/test_transverse_degeneracy_jvp.py
JACCPOT_FUSED_M2L_VJP=0 JAX_ENABLE_X64=1 pytest -q -n 0 tests/unit/test_custom_vjp_parity.py tests/unit/operators/test_transverse_degeneracy_jvp.py
```

`=1` is the fully-fused reverse Pallas kernel; `=0` falls back to autodiff of the pure-jnp
twin. Report both.

**3. The end-to-end lane, with the fused path actually engaged.**

```bash
JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1 JAX_ENABLE_X64=1 pytest -q tests/unit/test_gradient_correctness.py tests/characterization
```

`test_fd_vs_ad_along_a_transverse_direction_at_rho_zero` is the force-level statement of
the whole fix; on CPU it agrees with finite differences to 2.7e-10 in both bases. Report
what it gives with the fused kernel engaged.

**4. The full suite, on GPU.**

```bash
JAX_ENABLE_X64=1 pytest -q
```

CPU baseline for comparison: run `JAX_ENABLE_X64=1 pytest -q` on this branch on a CPU box
first and use *that* as the reference, because several sessions have added tests since this
document was written. What matters is the shape, not the absolute number: **0 failures, 0
xfailed**, and the passed count should *rise* on GPU because the four tests in item 1 stop
skipping. Any failure not obviously pre-existing on `main` should be bisected against `main`
before you conclude it is this branch's. Any
failure that is not obviously pre-existing on `main` should be bisected against `main`
before you conclude it is mine.

**5. Reverse-pass residuals.** Nothing in this branch changes what any `custom_vjp` saves,
but the linearised rotation-block construction now carries one extra `select`, so the
footprint around it can move:

```bash
python bench/audit_reverse_residuals.py
```

Compare against `main`. Report the delta per stage; a change in the M2L rows is expected to
be small, and a large one is a finding.

**6. Per-stage benchmark, before and after.** `NUMERICS_AND_JAX.md` §4: report per stage,
report compile time separately, and use the concentrated distribution as well as the
uniform cube. On CPU the M2L stage was unchanged within run-to-run noise (forward compile
0.494 → 0.485 s, run 1.889 → 1.870 ms; gradient compile 1.656 → 1.640 s, run 4.787 →
4.709 ms at order 4, N=512). The forward path cannot regress by construction — the primal
is untouched — so the number to watch is the **gradient**.

```bash
python bench/profile_downward_breakdown.py     # and the fused-stage ablation
```

## What would count as a real problem

- `interpret=False` fails while `interpret=True` passes → the Triton lowering and the
  reference lowering disagree in the reverse pass. That is a genuine finding; capture the
  full error and the tolerance it missed by.
- The forward pass moves *at all* on the fused path. Both new primals are identities with
  no arithmetic, so this should be impossible; if it happens, the carrier or the withdrawal
  is not doing what its docstring claims.
- A characterization golden moves. Do not update it and do not relax the tolerance — see
  `NUMERICS_AND_JAX.md` §3. Report it.
- A gradient-path slowdown beyond noise. The correction is four static block-diagonal
  matmuls plus, on the fused lane only, one pure-JAX twin application on a
  tangent-derived vector. If that shows up as more than a few percent of the M2L gradient
  stage, say so with the profile.

## Traps that already cost time on this branch

- **A gradient test at a default leaf size covers no far field.** At `theta=0.5, N=128`,
  `leaf_size=4` accepts 110 M2L pairs and `leaf_size >= 8` accepts **zero**. Assert the
  pair count, or the test passes while testing nothing.
- **A symmetric construction certifies a broken gradient.** That is exactly how G.10
  survived review: the same `(x, y)` set and masses in every cluster let the per-pair
  transverse errors cancel in the sum. Use asymmetric cotangents and directions.
- **Off-axis by one ulp is not off-axis.** Two nodes whose `(x, y)` centres are
  mathematically equal but summed in different orders differ by one ulp, giving
  `rho/r ~ 1e-17` with `rho_sq > 0`. The band exists for that case; the test batches all
  three regimes (exactly on axis, one ulp off, generic) deliberately.
- **fp32 covers the direct cascades but not the fused lanes.** The band is
  `rho/r <= sqrt(eps)`: 1.5e-08 at fp64 but **3.4e-04** at fp32, four orders wider, so
  ordinary tree geometry lands inside it. `test_the_analytic_branch_holds_at_float32` pins
  this for the six direct cascades. **It does not cover either fused Pallas lane**, which
  is the gap that matters most here, because the fused path is the one most likely to run
  fp32 in production. If you have the budget on the GPU, extend it; if not, report that it
  is still open rather than letting the green run imply otherwise.

## Out of scope

Do not restructure the rotation algebra, do not touch the JAX version bounds in
`pyproject.toml` (both ends are load-bearing and the reasoning is written there), do not
regenerate any golden, and do not change what the Pallas `custom_vjp`s save as residuals.
If you think one of those is wrong, say so and stop.

## Report back

For each numbered item: the command, whether it passed, and the numbers. Explicitly state
which GPU `autocvd` gave you, its compute capability, and whether
`pallas_m2l_real_fused_supported()` was True — a run where that was False has not tested
the thing this handoff is about.

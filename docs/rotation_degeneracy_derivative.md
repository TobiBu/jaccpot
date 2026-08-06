# The transverse derivative at `rho == 0` (G.10)

Derivation and numerical validation of the analytic transverse derivative of the
rotate → z-translate → rotate-back cascade at a z-aligned separation, where the
azimuth is undefined and the current degeneracy guards return a zero cotangent.

Written up separately from the code because the result is reusable (it applies to every
operator built on the cascade, in both bases) and because the calibration steps below
are the part that is easy to get wrong and expensive to redo.

**Status.** Derived, validated, and **wired into every M2L / M2M / L2L lane in both
bases** — §5 is what landed where, §6 is where the switchover boundary goes and why it is
wider than `rho == 0`, §7 is the precomputed-block lanes and the fused Pallas kernel,
which each needed a different shape of the same rule. §8 is the GPU validation. §9 is the
one lane this does **not** cover — the fused Pallas *complex* M2L, which now has the
builder's withdrawal and no carrier. G.10 in
[`refactor_audit_2026-08.md`](refactor_audit_2026-08.md) records the defect itself.

---

## 1. The problem

`m2l_real`, `m2m_real` and `l2l_real` (and their complex counterparts) evaluate

```
F(delta) = D_from(delta) · T_z(r) · D_to(delta)
```

where `delta = (x, y, z)`, `r = |delta|`, and the alignment rotations are built from
`az = atan2(x, y)` and `ax = atan2(rho, z)` with `rho = sqrt(x² + y²)`.

At `rho == 0` the azimuth is undefined. The guards return `az = 0` with a zero
cotangent. The **forward value is exact** there (the `m != 0` terms carry
`sin^|m|(theta) = (rho/r)^|m|`, which annihilates the arbitrary azimuth), but the
**derivative is not**: the cascade is differentiable at `rho == 0`, the limit is
direction-independent, and the true transverse derivative is nonzero.

The obstruction is that the code reaches `(x, y)` only through `rho` and `az`, so every
chain-rule route carries `x/rho` or `y/rho²`. The O(rho) coefficient the derivative
needs has already been divided out, which is why no choice of guard recovers it — see
G.10 for the three variants measured.

## 2. The derivation

`F` is **rotationally covariant**: rotating the displacement is the same as rotating
both coefficient frames,

```
F(R · delta) = D_out(R) · F(delta) · D_in(R)⁻¹                                    (1)
```

for any rotation `R`, where `D_in` / `D_out` are the representations acting on the
operator's input and output coefficient slots.

Take `delta₀ = (0, 0, z)` and differentiate (1) along a one-parameter rotation family
that sweeps `delta₀` transversally. For `R = R_y(θ)`,

```
R_y(θ) · (0, 0, z) = (z sin θ, 0, z cos θ)      so   dx/dθ = z,  dy/dθ = dz/dθ = 0
```

Differentiating both sides of (1) at `θ = 0` gives `z · ∂F/∂x` on the left and a
commutator on the right. Doing the same with `R_x(θ)`, for which
`R_x(θ) · (0, 0, z) = (0, −z sin θ, z cos θ)` and hence `dy/dθ = −z`:

```
∂F/∂x |₍₀,₀,z₎  =  (1/z) · [ G_out^y · F₀  −  F₀ · G_in^y ]                       (2)
∂F/∂y |₍₀,₀,z₎  = −(1/z) · [ G_out^x · F₀  −  F₀ · G_in^x ]                       (3)
```

where `F₀ = F(0, 0, z)` — which is just the pure z-translation, since the rotation
degenerates to the identity (or a π-turn for `z < 0`) — and `G^a = d/dθ D(R_a(θ))|₀`
are the representation generators.

`∂F/∂z` needs no special treatment: it is already correct, because `z` reaches the
result through `r` and `ax`, neither of which is degenerate here.

Applied to a coefficient vector rather than a matrix, (2) and (3) need only one extra
operator application each — `F₀ · (G_in · M)` is the same z-translation applied to a
rotated-generator image of the input:

```
∂L/∂x  =  (1/z) · [ G_out^y · F(M, delta₀)  −  F(G_in^y · M, delta₀) ]
```

## 3. The generators, calibrated rather than assumed

Sign and axis conventions are the whole difficulty here, so each of the following was
**fixed by testing against an identity the repo already verifies**, not by derivation
on paper. `Dz(ℓ, θ)` is `real_Dz_diagonal`, `B_U` is
`compute_real_B_matrix_multipole`, and `Λ(ℓ)` is `d/dθ Dz(ℓ, θ)|₀`:

```
Λ(ℓ)[ℓ+m, ℓ−m] = −m ,   Λ(ℓ)[ℓ−m, ℓ+m] = +m        (m = 1 .. ℓ; zero elsewhere)
```

| quantity | closed form | calibrated against | residual |
|---|---|---|---|
| `D^M(R_z(θ))` | `Dz(ℓ, +θ)` | `D · p2m(v) == p2m(R_z(θ)·v)` | 1.4e-17 |
| `D^M(R_x(θ))` | `B_U · Dz(ℓ, −θ) · B_U` | same, with `R_x` | 2.3e-16 |
| `D^M(R_y(θ))` | `Dz(π/2) · B_U · Dz(−θ) · B_U · Dz(−π/2)` | same, with `R_y` | 2.2e-16 |
| `G^M_x` | `−B_U · Λ · B_U` | central difference of `p2m(R_x(θ)v)` | 2.4e-11 |
| `G^M_y` | `Dz(π/2) · G^M_x · Dz(−π/2)` | central difference of `p2m(R_y(θ)v)` | 5.2e-11 |
| `D^L(R)` | `D^M(R)⁻ᵀ` (**contragredient**) | `evaluate_local_real(D^L·L, R·t) == evaluate_local_real(L, t)` | 2.8e-16 |
| `G^L` | `−(G^M)ᵀ` | follows from the row above | — |

The wrong-sign alternatives are recorded because they are the plausible mistakes:
`Dz(ℓ, −θ)` for `R_z` gives 6.6e-02, `+B_U Λ B_U` for `R_x` gives 2.5e-01, and taking
the local representation to be `D^M` or `D^Mᵀ` instead of `D^M⁻ᵀ` gives 4.1e-01 and
7.5e-02. Any of them would have looked plausible and been wrong.

The per-operator representations:

| operator | `D_in` | `D_out` |
|---|---|---|
| `m2l_real` | multipole | local |
| `m2m_real` | multipole | multipole |
| `l2l_real` | local | local |

## 4. Validation

Formula (2)/(3) against central differences of the operator itself.

**It is exact, not approximate.** Sweeping the finite-difference step at `z = 2.5`,
order 4, shows the textbook U-shape — second-order truncation falling as `eps²`, then
round-off rising — with **no error floor**:

| `eps` | 1e-03 | 1e-04 | 1e-05 | 1e-06 | 1e-07 | 1e-08 |
|---|---|---|---|---|---|---|
| relative error | 7.8e-07 | 7.8e-09 | 7.8e-11 | **3.9e-11** | 2.8e-10 | 4.7e-09 |

A formula that were merely a good approximation would bottom out at its own error
instead of tracking the finite difference down to round-off.

**It holds everywhere it must.** Worst relative error at `eps = 1e-5` over expansion
orders 1, 2, 4, 6 and `z ∈ {+2.5, −2.5, +0.4, −7.0}` (so both signs — `z < 0` makes the
degenerate rotation a π-turn rather than the identity — and both close and distant
separations):

| operator | worst over all 16 combinations |
|---|---|
| `m2l_real` | 6.3e-09 (at `z = +0.4`, worse finite-difference conditioning at close separation) |
| `m2m_real` | 3.6e-10 |
| `l2l_real` | 2.7e-10 |

## 5. What landed

The rule is attached at the **cascade** level, never at the alignment blocks. That is
forced, not stylistic: at `rho == 0` an individual alignment block *is*
direction-dependent — with `ax = 0` it reduces to `Dz(az)` for an arbitrary `az` — while
the product is not. Only the assembled operator has a derivative to supply.

The plumbing lives in `jaccpot/operators/_transverse_degeneracy_jvp.py`; each basis
owns its own generators, next to the rotation builders they are calibrated against
(`real_transverse_generators`, `complex_transverse_generators`).

| site | operators | reached from |
|---|---|---|
| `operators/m2l_real_rot_scale.py` | `m2l_rot_scale_real_batch` | **production** real M2L, via `runtime/kernels/core.py::_m2l_real_batch_kernel` |
| `operators/real_harmonics.py` | `m2l_a6_real_only` (so `m2l_real`, `m2l_optimized_real`), `m2m_real`, `l2l_real` | **production** real M2M (`upward/real_tree_expansions.py::aggregate_m2m_real_by_level`) and L2L (`runtime/kernels/core.py::_l2l_real_batch_kernel`) — these are not only reference operators |
| `operators/complex_ops.py` | `m2l_complex_reference` (so `m2l_complex_reference_batch`), `m2m_complex`, `l2l_complex` | production complex far field |

**The structure.** Only the tangent changes, and it changes as a *select*:

```
primal : unchanged                              -> forward bit-identical
tangent: where(rho_sq > 0, existing, analytic)   -> partitioned, not superposed
```

Off the axis the routed tangent is the incoming one bit-for-bit and the analytic term is
exactly zero, so every gradient that was already right is preserved to the last bit. On
it, the guards contribute exactly zero anyway, so select and add agree — the select is
written because it is the property the correctness argument needs, and because the
predicate is then the single place a future widening would go (§6).

Three regimes, as designed: `rho_sq > 0` untouched; `rho_sq == 0` with `z != 0` analytic;
`delta == 0` keeps the existing zero, because (2)/(3) divides by `z` and `|delta|` has no
derivative at the origin.

**Cost.** None in the forward pass — `custom_jvp` leaves the primal alone. None in the
tangent either, beyond four static block-diagonal matmuls: the cascade is linear in its
coefficient argument, so the two `F0 · G_in` terms fold into the incoming coefficient
tangent and ride the JVP that was going to run regardless. The naive form would have cost
one extra cascade evaluation per transverse axis.

**Complex-basis calibration.** Redone from scratch rather than transported from the real
basis, and the two signs were *searched*. At order 4, `z = 2.5`, `eps = 1e-5` on
`m2l_complex_reference` the four combinations give 2.0e+00, 1.5e+00, 9.6e-01 and
**7.4e-11**. The survivor is `G = −B_swap Λ B_swap` with `Λ = diag(i m)`, matching the
real basis' sign. One structural difference: the complex local rotation is built from its
own swap matrix `B_T`, not as the transpose of the multipole one, so each representation
takes its generator from the swap matrix its rotation blocks actually use — `G^L = −(G^M)ᵀ`
holds in the real basis and would be a coincidence to rely on here.

Reverse-mode agreement with forward mode, at `rho == 0` and with an asymmetric direction,
is asserted for all six operators in
`tests/unit/operators/test_transverse_degeneracy_jvp.py`. The rule is linear in the
tangents, so JAX transposes it; that test is what says it was written that way.

**Measured at the force level**, on the `_z_stacked_system(8, 4, 6.0, 3)` construction the
tracking test uses (6 of 24 M2L pairs exactly on axis), FD versus AD:

| basis | before | after |
|---|---|---|
| real | 1.9e-03 | 9.5e-06 |
| complex | 1.8e-05 | 9.7e-06 |

The two bases converging on the same number is the point: what is left is a cause they
share, and it is not this one.

## 6. Where the boundary goes, and why it is not `rho == 0`

The polar route does not only fail *at* `rho == 0`. It degrades on approach, and the
first implementation of §5 — which switched on the guards' own `rho_sq > 0` — left that
uncovered. Found while verifying this fix, and closed by widening the band.

`d(az)/dy = −x/rho²` grows without bound while the `(rho/r)^|m|` factor that annihilates
the azimuth shrinks, and the transverse gradient comes out of that cancellation with
relative error `~eps·r/rho`. Measured on `l2l_real`, order 4, `z = −3`:

| `rho` | 1e-17 | 1e-16 | 1e-12 | 1e-9 | 1e-6 |
|---|---|---|---|---|---|
| relative error in `d/dy` | 2.6e+02 | 1.8e+01 | 6.8e-04 | 3.5e-06 | 1.0e-09 |

That is not academic, and it does not need a contrived input. **Two tree nodes whose
`(x, y)` centres are mathematically equal — same particles, same masses, summed in a
different order — differ by one ulp instead of zero.** In the force-level tracking test's
own system, node 11's L2L displacement is `(5.551e-17, 0.0, +3.0)`: `rho_sq = 3.1e-33` is
strictly positive, the guards never fire, and the polar route computes
`d(az)/dy = −1.8e+16` — the exact derivative of a function varying that fast, evaluated
with catastrophic cancellation. It was `y`-only there because `y == 0` exactly makes
`d(az)/dx = y/rho² = 0`, which is both the computed and the true value, and it was
confined to the eight particles under node 5, with per-particle errors in equal pairs
across the two clusters as one mis-scaled delta cotangent distributed by mass must give.

**The crossover.** The analytic branch is the `rho → 0` limit, so it errs `O(rho/r)`; the
polar route errs `O(eps·r/rho)`. Equating them puts the boundary at

```
(rho/r)² == eps          i.e.   rho_sq <= eps · r_sq
```

which is what `split_transverse_tangent` codes. The choice is minimax: the worst relative
error over all `rho` becomes `~sqrt(eps)` — 1.5e-08 in float64, 3.4e-04 in float32 — and
is reached only on the boundary itself, where before it was unbounded. `z != 0` still
excludes `delta == 0`, and it is the only point the band could otherwise swallow: with
`r_sq == rho_sq` the test holds only for `rho_sq == 0`.

Widening this far is what makes the **split** load-bearing rather than decorative. Inside
the band the polar contribution is not zero, it is garbage, so it must be *removed*
rather than added to — hence a routed tangent and not just a pair of scales.

**Measured across the boundary**, worst analytic-versus-finite-difference relative error
over `z ∈ {+2.5, −3.0}` and `rho ∈ {0, 1e-17, 1e-14, 1e-12, 1e-10, 1e-8, 4.4e-8, 5e-8,
1e-7, 1e-5, 1e-3}` — so inside the band, on the boundary, and well outside it:

| operator | worst | operator | worst |
|---|---|---|---|
| `m2l_real` | 1.2e-08 | `m2l_complex_reference` | 4.0e-09 |
| `m2m_real` | 4.4e-08 | `m2m_complex` | 1.5e-07 |
| `l2l_real` | 1.1e-07 | `l2l_complex` | 2.0e-07 |

The worst cases all sit at `rho ≈ 4e-8` to `1e-7`, straddling the boundary, which is what
a minimax crossover looks like — and at those `rho` the finite difference is itself only
good to `O(step/r) ≈ 3e-07`, so this is a bound on the measurement as much as on the
branch. The `rho = 1e-17` column, which was 2.6e+02 before, now reads 4.5e-09.

**At the force level**, on `_z_stacked_system(8, 4, 6.0, 3)` (6 of 24 M2L pairs exactly on
axis, plus the one-ulp L2L displacement above), FD versus AD:

| basis | before G.10 | rho == 0 only | full band |
|---|---|---|---|
| real | 1.9e-03 | 9.5e-06 | **2.7e-10** |
| complex | 1.8e-05 | 9.7e-06 | **2.7e-10** |

Both bases now produce the *same* AD value to every digit printed, which is the tell that
the last shared cause is gone. Pinned by
`tests/unit/test_gradient_correctness.py::test_fd_vs_ad_along_a_transverse_direction_at_rho_zero`,
no longer an xfail.

**Both characterization goldens stay unmoved**, including the gradient golden. That is not
automatic once the band is wider than a measure-zero set, and it is the check to repeat if
the constant is ever changed: the golden's uniform distributions have no two nodes with
equal `(x, y)` centres, so nothing in them enters the band.

## 7. The precomputed-block lanes

Four lanes receive their rotation blocks as separate arguments, built once per
interaction class and reused, so **no single function sees both the displacement and the
operator built from it** — and the §5 rule needs both. Their forward values were
bit-identical to the direct lanes; their gradients were not. Measured before the fix, on
`m2l_rot_scale_real_batch_cached_blocks` against `m2l_rot_scale_real_batch` with the
blocks the latter would have built: the on-axis transverse gradient was `(0, 0)` where
the direct lane gave `(−0.270, −1.303)` — 1.30 apart, with exact agreement on the
off-axis rows, so no forward comparison could see it.

**The fix is two halves that only work together.**

1. The **block builder** withdraws the transverse tangent it cannot resolve
   (`without_unresolvable_transverse_jvp` on `_real_rotation_blocks_padded` and on the
   complex padded builders). An individual alignment block is the one thing here that
   genuinely has *no* transverse derivative at `rho == 0`: with `ax == 0` it reduces to
   `Dz(az)` for an arbitrary `az`, so its limit is approach-dependent even though the
   assembled cascade's is not. Inside the band it has a derivative but not one worth
   having. So it now hands its caller nothing rather than something wrong.
2. The **consumer** supplies the cascade-level term, via the same
   `with_transverse_degeneracy_jvp` the direct lanes use — generalised to pass extra
   differentiable array arguments (the blocks) through untouched.

Without (1), (2) would add the analytic term *on top of* whatever the builder's guards
produced: zero at `rho == 0`, garbage just off it. That is why the builder half exists.

Covered: `m2l_rot_scale_real_batch_cached_blocks`,
`m2m_rot_scale_real_batch_cached_blocks`, `l2l_rot_scale_real_batch_cached_blocks`,
`m2l_complex_reference_batch_cached_blocks`. Asserted against their direct twins in
`tests/unit/operators/test_transverse_degeneracy_jvp.py::test_precomputed_block_lanes_match_their_direct_twin_in_gradient`,
over a batch holding all three regimes at once — exactly on axis, one ulp off it, and
generic:

| lane | forward | gradient |
|---|---|---|
| real M2L | 0 | 3.9e-16 |
| real M2M | 7.1e-15 | 7.3e-15 |
| real L2L | 0 | 2.5e-14 |
| complex M2L | 0 | 0 |

That test also asserts the direct lane's on-axis transverse gradient is not itself ~0,
so the two cannot agree by sharing the defect.

**The fused Pallas real M2L needed a third shape.** It could not take the consumer
decorator at all: `m2l_real_fused_pallas_cvjp` is a `custom_vjp`, and JAX refuses
forward-mode through one — confirmed, not assumed ("can't apply forward-mode autodiff
(jvp) to a custom_vjp function"). What covers it instead never touches the kernel:

* `m2l_real_fused_align_deltas(deltas)` runs *before* the radius and both block
  stacks are built. Its primal returns `deltas` with no arithmetic performed on it, so it
  is invisible to a forward-only caller; what it removes is the in-band transverse
  tangent, so radius and blocks all agree on where the band is.
* `m2l_real_fused_carry_axis_derivative(out, …)` runs *after* the kernel. Its primal
  returns `out` unchanged — again no arithmetic, so no forward footprint, not even in the
  sign of zero — and its tangent adds the analytic term, computing the one operator
  application it needs with `m2l_real_fused_jax`, the same pure-JAX twin the kernel's own
  `custom_vjp` uses as its correctness reference.

Measured against the pure-JAX lane on the same three-regime batch: **1.98 apart without
the carrier** (the on-axis rows come back exactly `(0, 0)`), **2.7e-15 with it**, forward
unchanged to the bit. Asserted by
`test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient`, parametrised over
`interpret` so the same assertion covers the reference lowering on CPU and the real Triton
kernel where the hardware allows.

## 8. Validated on GPU

Run on an **A100-PCIE-40GB (sm_80)**, `pallas_m2l_real_fused_supported()` `True`, x64 on.
One caveat on the whole section: the box's environment has **JAX 0.9.0.1**, below this
project's own `>=0.10.2` floor, so the Triton lowering went through the old
`pallas_call(backend=)` API (`_compat.PALLAS_CALL_TAKES_BACKEND` `True`) rather than the
`triton.CompilerParams` path a supported install takes. Everything below should be re-run
once an in-window environment exists.

**The `interpret=False` half.** Both parametrisations pass, neither skipped:

| lowering | worst gradient difference vs the pure-JAX lane | tolerance | without the fix |
|---|---|---|---|
| `interpret=True` | 3.6e-15 | 1e-10 | 1.98 |
| `interpret=False` (real Triton) | **4.4e-15** | 1e-8 | 1.98 |

**The forward pass has no footprint**, confirmed rather than argued: the fused lane's
output is bit-identical (compared as raw `uint64`, so a `-0.0` would show) with the
withdrawal alone, the carrier alone, both, and under `jax.jit`.

**Both reverse branches.** `JACCPOT_FUSED_M2L_VJP=1` (fully-fused reverse kernel) and `=0`
(autodiff of the pure-jnp twin): 50 passed each over
`test_custom_vjp_parity.py` plus `test_transverse_degeneracy_jvp.py`, identical outcome
sets, nothing skipped.

**At the force level with the fused lane engaged**
(`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1`), FD versus AD on the `_z_stacked_system`
construction:

| basis | fused lane on | fused lane off |
|---|---|---|
| real | 1.0e-10 | 1.0e-10 |
| complex | **8.5e-07** | 1.0e-10 |

With the fused lane off both bases agree to every digit printed, as §6 claims. With it on
the real lane is unchanged and the complex one is three orders worse — see §9.

**Reverse-pass residuals** (`bench/audit_reverse_residuals.py`, and the same audit on the
real basis, which the shipped script does not cover because it hardcodes `basis="complex"`).
Complex basis: **unchanged, byte for byte**. Real basis, at N=32768 / leaf 256 / float32,
the configuration production uses:

| group | before | after | delta |
|---|---|---|---|
| m2l | 11,128.3 B/pair | 11,542.3 B/pair | +414 B/pair (+3.7%) |
| M2M by-level scan (the audit files this under `other`) | 18.947 MB | 20.290 MB | +1.343 MB (+7.1%) |
| near field, P2M, L2L, L2P, tree | — | — | unchanged |
| total | 227.157 MB | 228.518 MB | +1.361 MB (+0.6%) |

Attributed to individual buffers: four `float32[13,127,2,25]`, one `bool[13,127,2,2]` —
the extra select's predicate — and the M2L lane's own `float32[44,25]` additions. The
coefficient is **813 B per (level · internal node)** and is stable from leaf 4 to leaf 256,
so it projects to roughly 11 MB at N=200k and 60 MB at N=1M against budgets in the GB
range. Nothing about what the `custom_vjp` saves changed.

**The per-stage benchmark did not resolve.** At the scale the M2L stage actually runs here
(128–1024 pairs, order 4) the lane is eager-dispatch-bound at 0.25–1.0 ms per call, and the
harness noise exceeds the signal: an in-process A/B of the correction ON versus OFF gives
forward deltas from **−36% to +12%** on a path where the primal is provably identical work,
which is the noise floor stated outright. The gradient deltas (+12% to +19% fused,
+3.3%/−2.4% direct) sit inside it. So the honest statement is that the correction's cost is
**below what this measurement can see**, not that it is zero — and a 200k-particle
`profile_downward_breakdown.py` run is still owed, which is what would answer it.

## 9. The complex fused Pallas M2L lane is not covered

§7 covers `m2l_complex_reference_batch_cached_blocks`. It does **not** cover
`runtime/kernels/core.py::_m2l_complex_batch_kernel_fused_pallas`, and that lane now has
half the fix. It calls `complex_rotation_blocks_{to,from}_z_solidfmm_batch`, whose padded
builders carry `without_unresolvable_transverse_jvp`, and then hands the blocks straight to
`m2l_complex_fused_pallas_cvjp` with nothing after them — the withdrawal without the
carrier, which §7 says is exactly the half that does not work alone.

Measured on the three-regime batch, gradient with respect to `deltas`, worst difference
between the lane and the pure-JAX complex reference:

| | before this work | after |
|---|---|---|
| on-axis row, reference | `(0, 0)` (the defect) | `(0.469, 0.379)` |
| on-axis row, fused lane | `(0, 0)` | `(0, 0)` |
| one-ulp row, reference | `(−0.510, 6.5)` | `(−0.510, 0.391)` |
| one-ulp row, fused lane | `(−0.510, 6.5)` | `(≈0, 0)` |
| **reference versus fused** | **6.7e-16** | **5.1e-01** |

The forward values still agree to 1.3e-15. What has been lost is the reference/fused
equivalence `NUMERICS_AND_JAX.md` §1 requires: the pair agreed to round-off before and now
disagrees by 0.51. End to end this is still an improvement — the fused complex lane's
force-level FD-vs-AD went from the pre-fix 1.8e-05 to 8.5e-07 because M2M and L2L are
fixed — but its M2L transverse term is missing.

No existing test can see it. `test_m2l_complex_fused_pallas_custom_vjp_matches_twin`
differentiates with respect to the kernel's four inputs — multipoles, both block stacks, the
radius — and never with respect to `deltas`, so it is structurally blind to a degeneracy
that lives only in the `delta -> blocks` map, and `m2l_complex_fused_jax`, the twin it
compares against, shares the blindness. The real lane needed a delta-level test for exactly
this reason; the complex lane has none.

The fix is presumably the complex analogue of the real lane's pair, and it is its own
change with its own test, so it is recorded here rather than made.

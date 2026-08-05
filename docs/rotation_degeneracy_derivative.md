# The transverse derivative at `rho == 0` (G.10)

Derivation and numerical validation of the analytic transverse derivative of the
rotate → z-translate → rotate-back cascade at a z-aligned separation, where the
azimuth is undefined and the current degeneracy guards return a zero cotangent.

Written up separately from the code because the result is reusable (it applies to every
operator built on the cascade, in both bases) and because the calibration steps below
are the part that is easy to get wrong and expensive to redo.

**Status.** The formula is derived and validated (this document). It is **not yet wired
into the operators** — see "Implementation surface" at the end for what remains, and
G.10 in [`refactor_audit_2026-08.md`](refactor_audit_2026-08.md) for the defect itself.

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

## 5. Implementation surface, and the constraint that shapes it

The rule must **not** perturb anything that currently works. The structure that
guarantees this:

```
primal : unchanged                       -> forward stays bit-identical
tangent: where(rho_sq > 0, <existing JVP>, <analytic (2)/(3)>)
```

so the existing derivative is preserved exactly for every `rho > 0`, and the analytic
branch applies only on the measure-zero degenerate set the guards currently mishandle.
Three regimes, not two:

1. `rho_sq > 0` — existing behaviour, untouched.
2. `rho_sq == 0` and `z != 0` — formula (2)/(3).
3. `delta == 0` (so `z == 0` as well) — **(2)/(3) does not apply**, it divides by `z`.
   Keep the current zero. For `m2m`/`l2l` a zero displacement is the identity
   translation; for `m2l` a zero separation is unphysical.

Sites needing the rule, in priority order — the production path first, since fixing
only the reference operators would flip the operator-level tracking test while leaving
the force gradient wrong:

- `operators/m2l_real_rot_scale.py` — `m2l_rot_scale_real_batch`, the **production**
  real-basis M2L (reached from `runtime/kernels/core.py::_m2l_real_batch_kernel`).
- `operators/complex_ops.py` — `m2l_complex_reference` and the batched complex path;
  the complex basis has the same defect, measured.
- `operators/real_harmonics.py` — `m2l_real`, `m2m_real`, `l2l_real` (reference
  operators; these are what the operator-level tracking test targets).
- the M2M / L2L cascades in `upward/` and `downward/`, which share the alignment.

Two things to verify at each site beyond the obvious: that `custom_jvp` transposes
correctly for reverse mode (the rule is linear in the tangents, so it should, but the
FMM uses `jax.grad`, not `jax.jvp`), and that no new `@jit` boundary or host sync is
introduced.

The two tracking tests that must flip, both `xfail(strict=True)` so they become hard
errors the moment they start passing:

- `tests/unit/operators/test_real_harmonics.py::test_rotation_cascade_transverse_gradient_at_rho_zero`
- `tests/unit/test_gradient_correctness.py::test_fd_vs_ad_along_a_transverse_direction_at_rho_zero`

And both characterization oracles must stay byte-unmoved, since the primal is unchanged.

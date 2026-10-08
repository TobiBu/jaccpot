# Compact softening kernels and the fused lane's far field (2026-10-08)

Branch `feat/softening-kernels`; yggdrax `perf/softening-mac`; Odisseo `feat/softening-kernels` (into
`jaccpot-integration`). Module: `jaccpot/softening.py`. Tests: `tests/unit/test_softening.py`,
`tests/unit/test_softening_kernel_paths.py`, `tests/unit/runtime/test_softening_floor.py`.

## The problem

The fused lane's far field is the unsoftened multipole expansion, and the geometric MAC does not know the softening.
In the dense bulge of the disc+bulge IC it accepted cells a fraction of a softening length apart with Newton's law:
the 25M production force had rel-L2 0.30 (bulge centre ~3x too strong), the 2e6 subsample 0.58. With softening
1e-4 the same lane was in class (1.06e-3): the error was the softening, not the expansion.

## A floor alone does not fix Plummer

`softening_floor = c` refuses a far pair unless `|c_B - c_A| - r_A - r_B >= c eps` (exact COM radii: no particle
of one node is then closer than `c eps` to the other). On the 2e6 disc at the production Plummer softening 0.0268:

| c | far pairs | near leaf pairs | force ms | rel-L2 | max da/a |
| --- | --- | --- | --- | --- | --- |
| 0 | 7.4M | 2.2M | 10.9 | 0.58 | 23 |
| 1 | 9.1M | 8.6M | 31.7 | 0.11 | 0.47 |
| 3 | 22.1M | 62.6M | 105 | 2.0e-2 | 5.5e-2 |
| 5 | 38.9M | 181M | 290 | 8.4e-3 | 1.9e-2 |
| 10 | near list overflows (735M directed pairs) | | | | |

Plummer never becomes Newtonian, so the error falls only as 1/c^2; 1e-3 would need c ~ 15.

## Compact kernels, branch-free

`softening_kernel`: `"ferrers3"` (default; density `315/(64 pi h^3) (1 - r^2/h^2)^3`), `"wendland_c2"`, `"plummer"`.
`softening` stays the Plummer-EQUIVALENT length for every kernel (equal central potential): `h = 315/128 eps` and
`h = 3 eps`. With `s = 1/max(r, h)` and `c = min(r^2/h^2, 1)` (`min(r/h, 1)` for Wendland):

* force factor `g = G(c) s^3`, potential `psi = Psi(c) s`, `(1/r) dg/dr = D(c) s^5`, `dg/d(eps^2)`;
* each polynomial is `edge + (1 - c) R(c)`, so past `h` the factors are bitwise Newtonian;
* no select, finite at `r = 0`, one reciprocal square root (ferrers3 needs only `r^2`; Wendland a square root more).

The coefficients were derived symbolically; the tests re-derive them by quadrature. With a compact kernel the walk's
floor defaults to the support (`softening_floor=None`), which makes the unsoftened far field exact. Plummer's branch
keeps the kernels' historical op sequence: every Plummer force and gradient is unchanged bit for bit.

Every pair site takes the kernel: the generic jnp near field, the CSR kernels (table, sorted, direct in every
source-tile mode), the rectangle and pairs kernels, the mutual tile, every hand-written reverse rule (`g` and
`(1/r) dg/dr` in the tidal tensor, `dg/d(eps^2)` for the softening cotangent), the targeted near field's jerk/snap/
crackle (nested `jvp` for a compact kernel), the mesh and fused multi-GPU lanes, and the direct-sum references. The
floor is in yggdrax's shared MAC test, so every walk applies it; the adaptive policy applies it too.

## Choosing the softening

Against the smooth model (agama expansions fitted to the 25M particles), mass-weighted force error (Dehnen 2001):
Plummer's optimum is 5e-4 (sqrt(ASE) 1.70e-2), ferrers3 / Wendland's 7.5e-4 (1.44e-2 / 1.46e-2). The production
Plummer 0.0076 had sqrt(ASE) ~0.2 and a bulge-centre bias of ~25 %. Chosen: eps = 1.5e-3 (h = 3.7e-3), the best
bulge-centre force (9.9e-3) at sqrt(ASE) 2.8e-2. The softening-limited time step shrinks ~2x.

## Result

Fused lane, theta 0.8, p6, leaf 64, eps 1.5e-3 (shared card; times indicative, the timed A/B is below):

| N | kernel | rel-L2 | median | max | force | near leaf pairs |
| --- | --- | --- | --- | --- | --- | --- |
| 2e6 | ferrers3 | 6.1e-4 | 3.2e-4 | 8.5e-3 | 12.7 ms | 2.26M |
| 2e6 | wendland_c2 | 5.7e-4 | 3.2e-4 | 8.5e-3 | 16.9 ms | 2.29M |
| 2e6 | plummer | 2.8e-2 | 3.7e-4 | 0.22 | 12.0 ms | 2.23M |
| 25M | ferrers3 | 4.1e-4 | 2.5e-4 | 1.2e-2 | 100 ms | 37.9M |
| 25M | plummer | 2.3e-2 | 3.8e-4 | 0.12 | 76 ms | 29.9M |

The 25M row is the production IC (`disk_bulge_25m.npz`) on one A100: the force was rel-L2 0.30 with Plummer at the
old 0.0076. The floor at `h` costs ~27 % more near leaf pairs at 25M (37.9M vs 29.9M) and almost nothing at 2e6.

## Using it

```python
FastMultipoleMethod(softening=1.5e-3)                           # ferrers3, floor at its support
FastMultipoleMethod(softening=1.5e-3, softening_kernel="wendland_c2")
FastMultipoleMethod(softening=0.0076, softening_kernel="plummer")  # the historical convention
FMMAdvancedConfig(softening_floor=0.0)                          # no floor (the far field then carries the softening error)
```

`BlockStepFMM`, `DistributedBlockStepFMM`, `DistributedFMMConfig`, `direct_sum_gravitational_acceleration` and the
references take the same `softening_kernel`. Odisseo mirrors the kernels (`odisseo/softening.py`,
`SimulationConfig.softening_kernel`); `tools/mesh_galaxy_run.py` defaults to `--softening 1.5e-3`.

## Not covered

* The mutual lane's cross-device far field (`cross_theta > 0`) refuses a compact kernel: its cross walk has no floor yet.
* The treecode walk (`JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK`, `local_walk="treecode"`) refuses a floor.
* `evaluate_expansion` keeps Plummer's softened multipole terms; a compact kernel evaluates it unsoftened.
* The `dehnen_error` force-scale estimators keep their `1/(r^2 + eps^2)` regularised scale (Follow-up B).
* The softening-limited time step: at eps 1.5e-3 the bulge centre needs ~2x the steps of the old 0.0076.

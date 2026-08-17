# Operator conventions that collide

Three places in `jaccpot/operators/` hold **two things with the same shape and
incompatible meaning**. Each was found during the Tier 2.5 docstring programme
(batches 23, 24, 25), each was documented only in a body comment or not at all, and
each fails the same way: not a crash, not a shape error, but a silently degraded
force. That is the worst failure mode this library has, so they are collected here
and pinned by `tests/unit/operators/test_convention_contracts.py`.

**If you are about to change a sign or an enumeration order in these files, read the
matching section first, then run that test file.**

Section 4 was added later and is a different animal: two helpers with the same role
and the same name, one of which is simply **wrong**. It is an open defect, pinned by
a strict `xfail` rather than reconciled.

---

## 1. The two translation sign conventions are opposite

| operator family | `delta` means |
| --- | --- |
| `m2m_real`, `l2l_real` (same-type) | `source_centre - destination_centre` |
| `m2l_real`, `m2l_a6_real_only`, `m2l_optimized_real` | `destination_centre - source_centre` |

Same set of centres, opposite subtraction order. Concretely:

* **M2M** re-centres a child multipole onto its parent: `delta = child - parent`.
* **L2L** pushes a parent local down to a child: `delta = parent - child`.
* **M2L** converts a source multipole to a target local: `delta = target - source`.

M2M and L2L look opposite to each other but are not — both are `source - dest`; it is
only that the *source* is the child for M2M and the parent for L2L. M2L is the one
that genuinely runs the other way.

Measured on a single-particle, well-separated, deliberately asymmetric configuration
at order 6 in float64, comparing an order-6 round trip against the direct-sum
potential:

| chain | relative error |
| --- | --- |
| M2L with `target - source` | 4.3e-12 |
| M2L with `source - target` | 7.1e-3 |
| M2M with `child - parent` | 1.3e-12 |
| M2M with `parent - child` | 2.2e-2 |
| L2L with `parent - child` | 1.2e-13 |
| L2L with `child - parent` | 8.0e-3 |

Ten orders of magnitude between right and wrong, and the wrong answer is still a
perfectly finite, plausible-looking force.

**A third convention rides along with these:** `evaluate_local_real` takes its offset
as `centre - eval_point`. The opposite sign also runs and lands ~1.5e-2 off. It is
easy to get wrong while checking one of the above — that is how the batch 23
measurement was first mis-taken.

## 2. Two packed symmetric-tensor orderings, exact reverses

| module | ordering |
| --- | --- |
| `operators/symmetric_tensors.py` (`symmetric_multi_indices_3d`) | **descends** in `nx`, then `ny` |
| `operators/multipole_utils.py` (`multi_index_tuples`, `triangular_indices`) | **ascends** |

At order 2:

```
symmetric_multi_indices_3d(2) = (2,0,0) (1,1,0) (1,0,1) (0,2,0) (0,1,1) (0,0,2)
multi_index_tuples(2)         = (0,0,2) (0,1,1) (0,2,0) (1,0,1) (1,1,0) (2,0,0)
```

Verified as an exact reversal for orders 0–8, so it is a systematic convention
difference and not an artifact of small orders. **The two enumerate the same set and
have the same length**, so an index taken from one and used against the other reads a
real component from the wrong slot with nothing — no shape error, no bounds error —
to catch it.

`symmetric_tensors.py` serves the derivative towers; `multipole_utils.py` serves the
packed triangular multipole layout. They are separate representations that happen to
share a domain.

## 3. Four rotation builders, one signature

`operators/real_rotations.py` exports four functions with an identical
`(x, y, z, ell, dtype)` signature:

| function | direction | representation |
| --- | --- | --- |
| `real_rotation_to_z_axis_multipole` | world → z | multipole |
| `real_rotation_from_z_axis_multipole` | z → world | multipole |
| `real_rotation_to_z_axis_local` | world → z | local |
| `real_rotation_from_z_axis_local` | z → world | local |

All four return a `[2l+1, 2l+1]` real block, so nothing in the type system or the
shapes distinguishes them. All four are pairwise distinct for a generic off-axis
direction (they coincide on axis-aligned directions, which is why the test uses an
asymmetric one).

`to_z` and `from_z` are exact inverses **within** a representation. They are *not*
interchangeable across representations: the M2L sandwich deliberately crosses them —
rotate the *multipole* world→z, translate along z, rotate the *local* z→world — so
`from_z_local @ to_z_multipole` is emphatically not the identity.

## 4. Two `_angles_from_delta`, one of them wrong (OPEN DEFECT)

Unlike sections 1–3, this one is **not** a pair of locally reasonable
conventions. It is a bug, still in the tree, pinned rather than fixed.

| function | azimuth | polar |
| --- | --- | --- |
| `complex_ops._angles_from_delta_solidfmm` (production) | `atan2(x, y)` | `atan2(-rho, z)` |
| `solidfmm_reference._angles_from_delta` (reference) | `atan2(y, x)` | `atan2(rho, z)` |

Both feed a **byte-identical** rotation composition — `B_U Dz(beta) B_U Dz(alpha)`
to go into the z-aligned frame, `Dz(-alpha) B_T Dz(-beta) B_T` to bring the local
back out — so the reference differs from production by exactly these two
expressions. Both differences matter, and neither is visible on an axis-aligned
`delta`, where `alpha` is irrelevant and `beta == 0`.

`atan2(y, x)` is the same error already fixed once in the real-basis rotations:
the coded `B` (`_compute_dehnen_B_matrix_complex`) is the **x ↔ z** swap, so
`B Dz(theta) B` turns about *x*, and aligning a direction with `+z` needs the
azimuth that removes the *x* component. `atan2(y, x)` suits a y ↔ z swap, which
is not the matrix in this repo. See the convention note on
`real_rotations._multipole_align_to_z_block`, which says so at the site.

### The measurement

The right comparison is *not* a coefficient diff against `m2l_real` alone — the
two carry different normalisations. It is the evaluated potential against a
direct sum, which is basis-independent. Single unit mass at `(0.2, -0.1, 0.3)`
from the source centre, evaluated at `(0.15, 0.2, -0.1)` from the target centre,
float64, reference output rescaled by the correct channel factor (below):

| `delta` | order | `m2l_real` | reference |
| --- | --- | --- | --- |
| `(0, 0, 4)` | 2 / 4 / 6 | 5.2e-05 / 1.3e-05 / 9.8e-08 | 5.2e-05 / 1.3e-05 / 9.8e-08 |
| `(4, 1.5, -2.5)` | 2 / 4 / 6 | 4.5e-04 / 6.1e-07 / 2.7e-08 | **5.5e-02 / 5.5e-02 / 5.5e-02** |
| `(5, 0, 0)` | 2 / 4 / 6 | 1.1e-04 / 1.5e-06 / 1.8e-08 | **5.5e-02 / 5.5e-02 / 5.5e-02** |
| `(-2, -3, -1.5)` | 2 / 4 / 6 | 2.2e-04 / 5.2e-06 / 1.0e-07 | **7.5e-02 / 7.4e-02 / 7.4e-02** |
| `(1e-6, 0, 5)` | 2 / 4 / 6 | 3.5e-05 / 4.2e-06 / 2.1e-08 | 3.5e-05 / 4.2e-06 / 3.3e-08 |

**Flat in order** is the tell: truncation error falls with `p`, a wrong rotation
does not. Note the last row — at `rho ≈ 0` the `m != 0` channels carry
`(rho/r)^|m|` and suppress the defect to 1.9e-08 in the coefficients, so a
near-axis direction makes any test of this vacuous.

Restoring **both** conventions (and nothing else) makes the reference reproduce
`m2l_real` to ≤ 2.5e-16 and its field error identical to `m2l_real`'s at all
seven directions tried, orders 2/4/6. Restoring only one leaves a plateau:
1.9e-02 to 9.9e-02 for the azimuth alone, 2.2e-02 to 7.3e-02 for the polar sign
alone. Independently, the corrected complex block `B_U Dz(-ax) B_U Dz(atan2(x,y))`
conjugates by `Q` onto `real_rotation_to_z_axis_multipole` to 8.9e-16, while the
shipped one matches **none** of the four real rotation builders of section 3
(nearest mismatch 1.5 at order 2).

### The factor of two is not the problem

The reference is a factor of two smaller than `m2l_real` on every `m != 0`
channel, and it is tempting to blame *that* for the off-axis disagreement — the
argument being that the factor is per-`|m|` in the aligned frame and rotating
back mixes `m` channels, so a diagonal rescale cannot be valid off axis. **That
argument is wrong**, and it was in the `m2l_solidfmm_reference` docstring until
this change.

The factor relates the two *bases*, not the geometry, and it comes entirely from
the translate stage:

* multipole coefficients are harmonic **values** (`M_n^m = mass * U_n^m`), so
  they convert with `Q` alone — `complex_to_dehnen_real_coeffs`;
* local coefficients are the **dual** objects (`evaluate_local_real` forms the
  plain sum `Psi = sum F_n^m U_n^m`), and collapsing the complex sum over
  `m in [-n, n]` onto the `m >= 0` real channels folds each conjugate pair
  together. So locals convert with `D Q`, `D = diag(1 if m == 0 else 2)`.

Verified: `D Q Cz Q^-1 == translate_along_z_m2l_real` to 7.6e-17 for orders 2–4,
where `Cz` is `translate_along_z_m2l_complex` as a matrix; and `Q^T D Q == J`,
the `m -> -m` conjugation flip, which is *why* locals get `D` and multipoles do
not. Applied to the production complex M2L, `D Q` reproduces `m2l_real` to
≤ 3.1e-16 at every direction and order in the table above — an off-axis,
geometry-general result, so the diagonal rescale is exactly the right comparison
and the reference has nowhere left to hide.

`tests/unit/operators/test_real_harmonics.py` carries both halves:
`test_dehnen_local_channel_factor_holds_at_any_delta` (passing, with an inertness
gate asserting that dropping the factor of two breaks the identity by ≥ 1e-4) and
`test_solidfmm_reference_matches_m2l_real_off_axis` (`xfail(strict=True)`).

### Why it is not fixed here

Fixing it changes what the module computes, which CLAUDE.md makes its own
sign-off-carrying change. The blast radius is small — `solidfmm_reference` is
imported only by `tests/`, and only `translate_along_z_m2l_complex` (unaffected;
it has no rotation) is used by `test_complex_ops.py`. The one existing test of
`m2l_solidfmm_reference`, `test_solidfmm_reference_matches_z_axis_m2l`, is
on-axis and stays green either way.

---

## Why these are guarded by tests rather than unified

Unifying any of them would change what the code computes, and in two cases would
rename public API — both forbidden by `CLAUDE.md`'s non-negotiables without a
separate, sign-off-carrying change. The conventions are also not *wrong*; they are
locally reasonable and only dangerous where they meet.

So the fix is to make them impossible to break silently.
`tests/unit/operators/test_convention_contracts.py` asserts each convention **and its
plausible alternative failing**. That second half is the load-bearing one: a test
that only pins the right answer passes just as happily when both branches agree,
which is precisely when the convention has stopped mattering and the tripwire has
gone vacuous.

Those tripwires were mutation-tested when written. Each of these five deliberate
breakages was confirmed to fail the expected tests:

| mutation | tests that failed |
| --- | --- |
| negate `delta` in `m2l_a6_real_only` | both M2L tests, plus L2L/M2M chains that route through M2L, plus the collision test |
| negate `delta` in `m2m_real` | both M2M tests, plus the collision test |
| negate `delta` in `l2l_real` | both L2L tests, plus the collision test |
| make `symmetric_multi_indices_3d` ascend | the reversal and non-interchangeability tests |
| alias `from_z_local` to `to_z_multipole` | the inverse-pair test, and the translation chains that use the M2L sandwich |

The tolerances assume float64. Under float32 a correct round trip lands near 1e-5,
which is above the "tight" threshold, so the suite fails loudly rather than silently
weakening — `test_conventions_are_exercised_in_float64` says so explicitly.

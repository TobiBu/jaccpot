# Operator conventions that collide

Three places in `jaccpot/operators/` hold **two things with the same shape and
incompatible meaning**. Each was found during the Tier 2.5 docstring programme
(batches 23, 24, 25), each was documented only in a body comment or not at all, and
each fails the same way: not a crash, not a shape error, but a silently degraded
force. That is the worst failure mode this library has, so they are collected here
and pinned by `tests/unit/operators/test_convention_contracts.py`.

**If you are about to change a sign or an enumeration order in these files, read the
matching section first, then run that test file.**

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

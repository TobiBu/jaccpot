# Phase 2 module ordering, measured — 2026-08-30

**[measured 2026-08-30, status maintained to 2026-09-03 — see "Where the order stands"]**
The rollout plan orders Phase 2's modules by **bare-parameter count**. This
records what happens when they are ordered by STYLE_GUIDE §4.1's own predictor instead —
how much of each module is *already validated* — and the two orderings disagree.

Reproduce with `bench/annotation_pilot.py`; the numbers below are its output.

---

## The result

904 shape perturbations across 86 functions in the top six modules. **235 were silently
accepted.**

| module | plan's rank (bare) | tested | silently accepted | rate | functions ok/inc/unrep |
|---|---|---|---|---|---|
| `pallas/nearfield_mutual.py` | 5th (96) | 150 | **63** | **42%** | 5/0/3 |
| `operators/complex_ops.py` | 2nd (107) | 130 | **41** | **32%** | 27/4/1 |
| `runtime/_adaptive_policy.py` | 4th (97) | 86 | **25** | **29%** | 9/0/4 |
| `nearfield/_large_n_blocks.py` | 3rd (101) | 212 | **58** | **27%** | 8/0/0 |
| `nearfield/_fast_lane.py` | **1st (121)** | 124 | **32** | **26%** | 5/0/2 |
| `runtime/kernels/_m2l.py` | 6th (92) | 202 | **16** | **8%** | 14/0/4 |

`nearfield_mutual` moves from 5th to 1st. `_fast_lane` — the plan's first target — falls to
5th. `_m2l` is last either way, and by a wide margin: at 8% it is mostly validated already,
and a PR spending its budget there would be Pilot 1 again.

Ranking by *absolute* count rather than rate gives nearly the same order
(63, 58, 41, 32, 25, 16), so the conclusion does not depend on that choice.

## Read the coverage column before the rate

`ok/inc/unrep` is how many functions were actually measured, inconclusive, and
unreplayable. **`nearfield_mutual`'s 42% rests on 5 functions; `complex_ops`'s 32% rests on
27.** Those are not equally strong numbers, and 15 of the 101 targeted functions were never
called by `tests/unit` + `tests/integration` at all. All four inconclusive functions are in
`complex_ops`, so its rate is the best-covered of the six even after discounting them.

The tool refuses to guess on the other two categories: a function whose control run fails,
or one with an argument it cannot describe, is reported and **not counted in either
direction**. Every way that accounting could be sloppy would make the remaining work look
smaller.

## What is actually being accepted

| perturbation | accepted |
|---|---|
| leading axis −1 | 116 |
| trailing axis −1 | 57 |
| extra leading axis | 50 |
| flattened | 12 |

The dominant mode is a **length mismatch between arrays that must agree** — the `n` of
`positions` against `masses`, the `pairs` of a schedule against its payload. That is
exactly what a shared axis name asserts, and it is the cheapest annotation to write.

## One finding that cuts across the module structure

Thirteen of the silent acceptances are **a length-3 spatial vector truncated to length 2**,
and all thirteen are in `operators/complex_ops.py`:

```
delta[3] -> [2]      evaluate_local_complex
delta[3] -> [2]      evaluate_local_complex_derivative_tower
delta[3] -> [2]      evaluate_local_complex_grad_analytic
delta[3] -> [2]      evaluate_local_complex_with_grad
delta[3] -> [2]      evaluate_local_complex_with_grad_analytic
delta[3], direction[3] -> [2]   regular_solid_harmonic_directional_derivative
delta[3], direction[3] -> [2]   regular_solid_harmonic_directional_derivative_order
delta[3] -> [2]      rotate_complex_local_from_z_solidfmm
delta[3] -> [2]      rotate_complex_local_to_z_solidfmm
delta[3] -> [2]      rotate_complex_multipole_from_z_solidfmm
delta[3] -> [2]      rotate_complex_multipole_to_z_solidfmm
```

**This is the same defect Phase 1 fixed in `downward/local_expansions.py`**, where it was
measured to be silent rather than loud because JAX *clamps* an out-of-bounds index: `delta[2]`
on a length-2 array returns `delta[1]`, so the caller gets the answer for `(x, y, y)` with no
error. Here it reaches the rotation and local-evaluation operators — and both the complex L2L
cascade and the real-harmonic rotations have carried sign and transpose bugs before.

`Float[Array, "3"]` on `delta` and `direction` across those eleven functions is one small,
uniform, high-value change. It is a better first PR than any whole-module conversion: it is
a family with a known failure mode, it needs no new axis vocabulary, and `"3"` is a literal
so it binds nothing that could break another lane.

## Suggested revision to the plan's order

1. **The `delta`/`direction` family in `operators/complex_ops.py`** — 13 known holes, one
   annotation repeated.
2. `pallas/nearfield_mutual.py` — highest rate, but widen the capture first: 3 of its 8
   functions were unreplayable and only 5 were measured.
3. `nearfield/_large_n_blocks.py` — 58 acceptances on full coverage (8/0/0), the
   best-evidenced block of work.
4. `operators/complex_ops.py`, remainder.
5. `runtime/_adaptive_policy.py`.
6. `nearfield/_fast_lane.py`.
7. `runtime/kernels/_m2l.py` — **last**, and possibly not worth a PR: 8%.

## Where the order stands — 2026-09-03

Three of the seven are done. Counts are `bench/annotation_census.py` on `cb07ad5`, which is
the only definition of the burn-down; the package moved 387 -> 551 shaped and 1709 -> 1544
bare over these PRs, 18.5% -> 26.3%.

| # | module | bare / shaped / unann | status |
|---|---|---|---|
| 1 | `operators/complex_ops.py`, the family | 27 / 70 / 0 | **done** — #279, 26 parameters now `Float[Array, "3"]` |
| 2 | `pallas/nearfield_mutual.py` | 54 / 42 / 0 | **part done** — `fb2d0a1`; the entry points are contracted, the internals are not |
| 3 | `nearfield/_large_n_blocks.py` | 41 / 60 / 12 | open |
| 4 | `operators/complex_ops.py`, remainder | 27 / 70 / 0 | open |
| 5 | `runtime/_adaptive_policy.py` | 75 / 24 / 0 | **done** — #293 |
| 6 | `nearfield/_fast_lane.py` | 30 / 101 / 3 | **done** — #285, #289, #290 |
| 7 | `runtime/kernels/_m2l.py` | 92 / 0 / 5 | open, and this document's advice is still to skip it |

**Item 2's blocker was tooling, and the tooling landed for this module specifically.**
"Widen the capture first" was this document's own precondition. `5c6bc71` closed it: the
three UNREPLAYABLE functions were `_block_tile` and `_block_vjp_tiles`, whose positions are
`tuple[Array, Array, Array]` and fell through to opaque, and `_pad_inputs`, which takes the
working `dtype`. Both kinds are now described, and arrays inside a container are perturbed
by path rather than merely rebuilt. (#294's `Tree` and `namedtuple` kinds were for item 5,
where they turned 11 UNREPLAYABLE into 11 measured.)

**What is still outstanding is a re-record, not a code change.** The description is frozen
in the pickle, so the 2026-08-30 recording still reports those three as opaque however much
the tool now understands — an old recording replays to the identical numbers, verified.
Re-record before annotating, because the 42% rests on 5 of 8 functions.

**And the 42% no longer describes what is left of item 2.** `fb2d0a1` gave the four public
entry points -- `mutual_leafpair_block_jax`, `_pallas`, `_vjp_pallas`, `_cvjp` -- plus
`_pad_inputs` a shape contract, 42 shaped parameters, with `tests/unit/pallas/test_nearfield_mutual_shape_contracts.py`
pinning the rejections; that landed after this recording was taken. The 54 bare parameters
left are the internal tile helpers and the reverse rule: `_pair_weight_tile` (3),
`_block_tile` (6), `_block_vjp_tiles` (6), `_mutual_leafpair_block_cvjp_fwd` (11), the
`one` closure inside the twin (8), and the scalar pair (`G`, `softening_sq`) on each entry
point, which is the same family `_fast_lane` measured and left bare. So item 2 is a
*different and smaller* piece of work than this document describes, and the re-record is
what sizes it.

**Item 7 is the one open disagreement with the size-ordered plan**, and it is unresolved
rather than decided: `_m2l` is now the largest module in the package by bare count (92) and
first on the size-ordered list, against 8% measured here. Whoever picks it up owns that call.

**Two findings came out of executing items 5 and 6**, both of which sharpen §4.1 rather than
this ordering:

* `nearfield/grad.py` was recommended on a structural argument and the measurement did not
  support it — every corruption there was already caught or harmless. It shipped shapes with
  no decorators (#292), and the site note records the table.
* Item 5's `multipole_packed` was annotated `Float` from this recording, which was taken
  entirely on the **real** basis; the complex basis passes `c64`/`c128` and CI failed 27
  times. Capture coverage bounds the *dtype* as much as the shape — STYLE_GUIDE §4.3 now
  carries both halves. A pilot replay measures the lanes in the recording and nothing else,
  which is also what the last bullet of the next section says.

Modules outside the top six still have **no §4.1 evidence at all** — `runtime/kernels/_evaluate.py`
(72/23), `pallas/m2l_complex_fused.py` (64/0/19), `pallas/nearfield_fused_leaf.py` (58/0/48),
`nearfield/_kernels.py` (53), `runtime/_octree_fmm.py` (49), `nearfield/near_field.py`
(48/15/19), `pallas/m2l_real_fused.py` (41/0/17). Ranking them needs a pilot run first.
`runtime/fmm_prepare.py` (39/2) is not open work: #296 annotated the two the measurement
supported and records at class level why the rest are left alone.

## What this does not say

* A rejection is not necessarily a *good* rejection. `TypeError: mul got incompatible
  shapes for broadcasting: (8, 3), (7, 1)` counts as rejected, and it is — but it names
  neither parameter. Replacing it with a shape annotation is still worth doing; it is just
  not worth *ranking* by.
* Value-dependent validation is invisible: the replay passes zeros.
* A lane that never ran cannot be measured. `tests/perf`, `tests/distributed` and the GPU-only
  Pallas paths are all absent from this recording, which is why `nearfield_mutual` and
  `_fast_lane` have the weakest coverage of the six.

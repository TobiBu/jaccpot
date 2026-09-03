# Phase 2 module ordering, measured — 2026-08-30

**[measured 2026-08-30; every module re-recorded 2026-09-03 — read "The whole list,
re-recorded" at the bottom first, because five of the six rates below have moved]**
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

---

## The whole list, re-recorded — 2026-09-03

Items 2 and 3 each came in far below the table at the top of this document, both times
because the module had been annotated *after* that recording. So the remaining four were
re-recorded together on `main`, same test scope (`tests/unit tests/integration`), with the
three fixes in #303 — without which the numbers below are not obtainable at all.

| module | item | tested | accepted | rate | ok/inc/unrep | was |
|---|---|---|---|---|---|---|
| `runtime/_adaptive_policy.py` | 5 | 230 | **78** | **34%** | 26/0/0 | 29% on 9/0/4 |
| `runtime/kernels/_m2l.py` | 7 | 286 | 23 | 8% | 18/0/0 | 8% on 14/0/4 |
| `pallas/nearfield_mutual.py` | 2 | 213 | 32 → 1 | 15% → 0% | 8/2/0 | 42% on 5/0/3 |
| `nearfield/_large_n_blocks.py` | 3 | 212 | 11 → 9 | 5% | 8/0/1 | 27% on 8/0/0 |
| `operators/complex_ops.py` | 4 | 202 | 11 | 5% | 47/6/1 | 32% on 27/4/1 |
| `nearfield/_fast_lane.py` | 6 | 160 | 4 | 2% | 6/0/2 | 26% on 5/0/2 |

`→` is before and after the annotations that closed them.

**One caveat on the `_fast_lane` row, and it is mine.** That recording was taken before the
fourth fix on #303, which stopped the pilot replacing a `jax.custom_vjp` *object* with a
plain wrapper and thereby stripping its reverse rule. `nearfield/_fast_lane.py` owns
`_radix_fast_lane_prepacked_accel_cvjp`, so during that run its custom rule was gone. The
four acceptances found are real -- the perturbations are forward calls, and a shape
rejection does not depend on the VJP -- but the **coverage** figure (6 of 18 targeted) was
taken with one lane's dispatch altered, so read it as a lower bound. That fix is also what
the `1 failed` in the recording run was: `test_prepacked_cvjp_saves_the_documented_nine_entry_residual`,
which noticed only because it asserts the object's identity on purpose.

**Items 4 and 6 are effectively done.** `complex_ops` fell 32% → 5% and `_fast_lane`
26% → 2%, which is what the delta/direction family (#279) and the fast-lane PRs
(#285/#289/#290) were for. Neither is worth another PR on this evidence.

**Item 5 is now the top of the list, and it is not a regression.** `_adaptive_policy` reads
*worse* than in August — 34% against 29% — and the reason is coverage, not decay. The
original pass measured **9** of its functions and gave up on 4; this one measures **26**,
because #294 taught the pilot about trees and #303 stopped the xdist shards clobbering each
other. #293 closed the holes that were visible then; the module simply has ~2.9x more
measurable surface than the tool could show. 33 of the 78 acceptances sit in two functions
#293 never saw:

```
build_adaptive_policy_state    18
resolve_dehnen_geometry        15
```

The remaining 45 are spread two or three at a time over ~18 more, which is a different shape
of work from a family fix — a module pass, and at 75 bare parameters it needs a split.

**Item 7 stays last on rate and gains a target anyway.** `_m2l` is confirmed at 8%, now on
18 measured functions rather than 14, so the August verdict holds: it is mostly validated
and a whole-module conversion would be Pilot 1 again. But *where* its 23 acceptances sit is
new information, and it is not spread evenly:

```
_rotation_blocks_for_grouped_classes           5   class_keys accepts ALL FOUR
                                                   perturbations; class_deltas the
                                                   misalignment against it
_accumulate_solidfmm_m2l_grouped_class_major   4   locals_coeffs, all four
_pair_class_ids_from_offsets                   3   class_offsets and pair_indices
_accumulate_solidfmm_m2l_grouped_chunked_scan  2
_chunk_segment_scatter_add                     2
_m2l_chunk_contributions                       2
```

**That is the G.11 neighbourhood.** This module's own docstring records G.11 as a 60x
accuracy gap between two of the four accumulators, caused by `pair_grouped` *gathering
rotations with class ids from the wrong ordering* — and the two functions accepting most
freely are the rotation-blocks-by-class helper and the class-major accumulator. A
`class_keys` / `class_deltas` pair that no longer agree on their class axis is that defect
expressed as a shape. So the useful PR here is the class/rotation family, ~12 of the 23,
and not the other 80 parameters.

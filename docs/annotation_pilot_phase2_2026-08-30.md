# Phase 2 module ordering, measured — 2026-08-30

**[current]** The rollout plan orders Phase 2's modules by **bare-parameter count**. This
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

## What this does not say

* A rejection is not necessarily a *good* rejection. `TypeError: mul got incompatible
  shapes for broadcasting: (8, 3), (7, 1)` counts as rejected, and it is — but it names
  neither parameter. Replacing it with a shape annotation is still worth doing; it is just
  not worth *ranking* by.
* Value-dependent validation is invisible: the replay passes zeros.
* A lane that never ran cannot be measured. `tests/perf`, `tests/distributed` and the GPU-only
  Pallas paths are all absent from this recording, which is why `nearfield_mutual` and
  `_fast_lane` have the weakest coverage of the six.

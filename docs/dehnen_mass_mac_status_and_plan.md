# Dehnen mass-dependent MAC: status and next steps

Handoff document, 2026-07-31; Step 1 completed and folded in 2026-08-01.
Self-contained: a fresh session should be able to pick this up without prior context.

## TL;DR

jaccpot implements Dehnen (2014, arXiv:1405.2255) §5's mass-dependent multipole
acceptance criterion — eqs (12), (13), (15), (16a) plus the §5.4 low-order prepass —
reachable via `mac_type="dehnen_error"`. That transcription is now **proven** by unit
tests against independent numpy float64 references, not merely asserted.

> **⚠ Every force-error number recorded in this document before 2026-08-01 is void.**
> Two defects in the upward M2M pass were silently dropping 10–23 % of the system mass
> out of the far field on shipped default settings (97 % on the cartesian basis). All
> error benchmarks on this branch were therefore measuring that bug, not the MAC. Both
> are fixed (`8afd705`); the numbers below marked **VOID** have not yet been redone.
> The Step 1 *cost* result is unaffected — it measures prepare time, not accuracy.

Four correctness bugs in the criterion were found and fixed, plus, later, two in the
upward pass that the criterion work uncovered. Step 1 (cache the force-scale prepass)
is done: steady-state prepare overhead went from **5.87× to 0.98×** at N=16384/p=8.
Step 2 (eq 16b) is validated and **negative** — see below.

Open: the MAC comparison must be re-measured post-M2M-fix before any accuracy claim is
made; the mass MAC still cannot reach any fast lane; nothing has been measured at
Dehnen's N (10⁵–10⁷); and the cartesian basis has an unexplained ~1.8×10⁻¹ error.

## Where the work lives

Branch **`fix/dehnen-mass-mac`**, off `ce5bd36` on
`chore/pallas-call-backend-port`:

```
a342eac  perf(mac): cache the Dehnen force-scale prepass -- 5.87x prepare to 0.98x
25df4e3  docs(mac): point the handoff at the branch and its verified state
ad34a75  test(api): register mac_theta_max on the frozen facade surface
796f5cb  bench(mac): error-distribution sweep, measurements, and the four next steps
eb83875  test(mac): pin Dehnen eqs (12)/(13)/(15)/(16a) and the far/near partition
96d5a44  fix(mac): correct four defects in the Dehnen mass-dependent MAC
```

It is branched off the pallas branch rather than `main` deliberately: `main` is 69
commits behind, and the touched runtime files (`_fmm_impl.py`, `fmm_derivatives.py`,
`fmm_evaluate.py`, `fmm_policy.py`, `fmm_prepare.py`, `solver.py`) differ between `main`
and that tip, so the edits were authored against the newer versions. Rebasing onto
`main` before the pallas work lands would need conflict resolution in those files.

Regression on the final commit: `tests/unit` + `tests/characterization` +
adaptive/force-scale suites → **0 failures** (exit 0), including the characterization
goldens. `tests/unit/runtime/` is 94 cases across 46 test functions.

Note for whoever extends the public surface next: `mac_theta_max` had to be registered
in `EXPECTED_FMM_INIT_KWARGS` (`tests/unit/test_public_api_surface.py`). That
frozen-surface guard caught the widened constructor on the first full run after the
change — it is doing its job, so expect to update it deliberately rather than treating
the failure as noise.

## What is already done — do not redo

### The four correctness bugs (all fixed, all with tests that provably bite)

| # | Bug | Effect | Fix | Bite check |
|---|---|---|---|---|
| D1 | eq (12) summed `m≠0` terms once; the real (Dehnen no-√2) packing splits `\|M_n^m\|²` across two slots, so each `\|m\|≠0` magnitude contributes **twice** | `P_n` under-estimated by up to √2 ⇒ criterion too permissive, and estimator basis-dependent | `dehnen_multipole_power_by_degree` doubles `m≠0` weights when `not jnp.iscomplexobj(packed)` | `P_n/(m·dⁿ)` was 0.707–1.0, now exactly 1.0 |
| D2 | `compute_tree_merged_sphere_geometry` merged in **descending node index**; radix internal nodes are not stored in postorder | node spheres did not contain their own particles (13/63 unbounded at N=512) | span-ordered `fori_loop`, matching the pattern the sibling reduction already documents | containment test on a ≥3-level tree |
| D3 | MAC evaluated about SES centres while M2L is applied about COM | the "bound" bounded an operation never performed; over-accepts | new `dehnen_geometry_mode="com"` (now the default); all other modes re-referenced to the expansion centre and inflated by the offset | per-node bound test across all 5 modes |
| D4 | eq (16a) is asymmetric in A↔B, but the traversal accepts only on `accept_both` / `near_both`; a disagreeing **leaf-leaf** pair became `REFINE`, and leaf-leaf pairs cannot be refined | pairs **silently dropped** — no M2L *and* no P2P | symmetrize: `_dehnen_paper_directional(src,tgt) & _dehnen_paper_directional(tgt,src)` | 2688 dropped incidences → 0 |

Plus: missing `G` in `est_force_error` (criterion ran at effective `ε·G`); empty leaves
took `0` instead of the `min` identity and zeroed the force scale up to the root;
unreachable dead block in `fmm_derivatives.py`; the `safe_mass` divide-then-multiply
round-trip folded out.

### Two upward-M2M defects — found 2026-08-01, fixed in `8afd705`

Not MAC bugs, but they invalidate every accuracy measurement taken before them, so
they belong at the top of this document rather than in a changelog.

An FMM's far field substitutes a node's multipole expansion for its particles. If a
node spans particles but its expansion is zero, every M2L sourced from it contributes
nothing and that mass vanishes from those targets. Two independent defects did that. Both are fixed on this branch in `8afd705` and
submitted to `main` separately as **PR #56** (`fix/upward-m2m-mass-loss`), which is
where they should be reviewed — they are not MAC work and should not wait behind it:

| # | Where | Cause | Mass lost at N=1024 |
|---|---|---|---|
| M1 | `_aggregate_m2m_impl` (cartesian / generic) | walked internal nodes in **descending index**, assuming children are stored after parents. Radix internal nodes are not in postorder — 26 of 63 have an internal child with a *lower* index. | **97 %** (root monopole 32/1024) |
| M2 | by-level M2M in `real_tree_expansions` and `solidfmm_complex_tree_expansions` | `dynamic_slice_in_dim` **clamps** an out-of-range start rather than erroring, so the level window slides and the positional slot mask then selects the wrong nodes. Bites every level starting past `num_internal`, i.e. the deepest levels of any tree. | **23 %** |

M1 is the same defect as D2, which `96d5a44` fixed in
`compute_tree_merged_sphere_geometry`; the sibling occurrence in the M2M kernel was
missed. Both are now span-ordered — a parent's span is strictly wider than either
child's whenever every internal node has two non-empty children, which radix
guarantees.

For M2, the docstring in `prepare_solidfmm_complex_upward_sweep` *already described the
clamping* but concluded that keeping `batch_width == num_internal` avoided it. It does
not: `total_nodes = 2·num_internal + 1`, so deep levels overrun regardless. Fixed by
padding the level list; the note is corrected in place.

**Why nothing caught it.** The far/near partition invariant checks index bookkeeping
and counts a source node as covering the particles in its `node_ranges`, so a
zero-multipole node read as perfectly covered. The golden physics anchor was a single
`rel_L2 < 0.35`, which a 23 % mass loss passes comfortably — and it was that loose in
order to accommodate the cartesian basis. Error benchmarks saw a median at machine
precision with a heavy tail, which reads as ordinary truncation error.

**New guard:** `tests/unit/runtime/test_upward_node_mass_conservation.py` (6 cases) —
every node reachable from the root that spans particles must have a non-zero expansion,
and its monopole must equal the mass it spans. The golden anchor is now per-basis:
10⁻² for real and solidfmm (worst observed 7.2×10⁻⁴, so it can actually fail), 0.35
only for cartesian.

**Open, and separate: the cartesian basis is ~1.8×10⁻¹ rel-L2** at both p=2 and p=4,
before *and* after these fixes. Order-independent error is a divergent-series
signature, not truncation, and solidfmm is 8.1×10⁻⁵ at the same configuration — 2000×
better. Nobody should use the cartesian basis for anything quantitative until that is
understood. It is now documented at the anchor rather than hidden behind a blanket
tolerance.

### Guard-rails added

- **`adaptive_eps` is now required** in paper mode. It used to fall back to
  `theta**(p+2)` — a `tail_proxy` heuristic that at θ=0.6, p=4 gives **0.047**, a 4.7 %
  per-interaction tolerance. That is how a wildly loose ε reached the accuracy
  notebooks. Note θ does **not** gate acceptance in paper mode at all.
- **`mac_theta_max`** (default 1.0 = eq (16a) verbatim) caps the opening angle. It was
  added because acceptance appeared to reach opening 0.997; that turned out to be the
  M2M mass-loss bug, and post-fix eq (16a) verbatim never exceeds 0.786. The knob stays
  (it is cheap and occasionally useful) but **is no longer needed** — see trap 4.
- Silent downgrades now raise: the strict fused device-only lane
  (`fmm_prepare.py` ~:795) and an explicit `runtime_path="large_n"` request
  (`_large_n_pipeline.py` ~:1918). The large-N decline reason is surfaced in
  `get_runtime_diagnostics()` as `large_n_path_declined_reason`.

### Tests: `tests/unit/runtime/`, 94 cases / 46 functions

- `test_dehnen_mac_reference.py` (39) — pins eqs (12)/(13)/(15)/(16a) against numpy
  float64 references. The load-bearing one is
  `test_dehnen_power_of_point_mass_equals_mass_times_distance_pow_n`: for a point mass
  at distance `d`, `P_n = m·dⁿ` exactly, in **both** packings — one identity that pins
  the factorial weights, rotation invariance, and basis independence at once.
- `test_far_near_partition.py` (6) — **the highest-value invariant**: for every target
  leaf, ancestor-inherited far sources ∪ near source leaves ∪ own leaf must cover every
  source particle exactly once (`missing == 0 and double == 0`). This is the only test
  that can see a dropped pair; the suite had no such invariant before.
- `test_dehnen_mac_gradients.py` (11) — FD-vs-AD, grad-vs-direct-sum, a verified
  no-boundary-crossing mass perturbation, cotangent isolation of `force_scale_nodes`,
  and traceability of each geometry mode.
- `test_upward_node_mass_conservation.py` (6) — every node reachable from the root that
  spans particles must have a non-zero expansion and a monopole equal to the mass it
  spans. The invariant the two M2M defects violated, which nothing else could see.
- `test_force_scale_injection.py` (12) — the `force_scale_nodes` parameter (used
  verbatim, cache untouched, prepass skipped, shape validated, rejected when unusable),
  the scalar node reduction, and that eq (16b)'s injected `f_b` really reaches the
  criterion.
- `test_force_scale_prepass_cost.py` (20) — the Step 1 contracts: the prepass runs once
  under `paper_cached` and every call under `paper`, `prepare_state` is bit-idempotent,
  a prepass never recurses into itself, the prepass restores the enclosing call's
  `rebuild_every` budget and gear bookkeeping, and a stale cached scale does not move
  the δa/f error tail at a per-step displacement where the two arms provably disagree
  about the accept mask.

### Benchmarks: `bench/validation/`

`mac_error_distribution.py` sweeps fixed-θ vs the mass MAC, records the full
per-particle error distribution plus hardware-independent cost proxies, and matches the
arms at equal 90th percentile. Implements three error families
(`--metric relative|scaled|dehnen`), Plummer / uniform / mass-spectrum / bulge-halo
generators, a chunked direct-sum reference, and `chunked_force_scale` for Dehnen's `f_b`.

`force_scale_prepare_cost.py` measures warm-call `prepare_state` medians for the
geometric MAC against `paper` and `paper_cached`. It exists because the prepare-cost
figure below was originally produced by hand, and the two traps that make hand
measurement wrong — cold wall-clock and unmaterialised device arrays — are both easy to
fall into twice. It also reports the live far-pair count per arm, so an arm that has
degenerated to all-near-field cannot be compared by accident.

## Measured results

All at N=4096 unless stated. Ratios > 1 favour the mass MAC. "work" is
`far_pairs·(p+1)² + Σ n_t·n_s`, i.e. hardware-independent interaction work.

### p=8 with Dehnen's own δa/f measure — the headline — **VOID**

Measured with 10–23 % of the system mass missing from the far field. Retained only so
nobody re-derives it from the committed JSON and believes it:

| distribution | p99 | max | work |
|---|---|---|---|
| ~~Plummer (his test case), eq (16a) verbatim~~ | ~~1.34~~ | ~~1.51~~ | ~~0.89~~ |
| ~~uniform, eq (16a) verbatim~~ | ~~1.21–1.33~~ | ~~1.31–3.53~~ | ~~0.99–1.01~~ |
| ~~Plummer, `mac_theta_max=0.7`~~ | ~~1.82–1.95~~ | ~~1.53–1.65~~ | ~~0.94~~ |

### p=4 — **VOID** for the same reason

~~uniform (global-rms metric): p99 0.93–1.10, max 1.06–2.45, work 0.93–1.01.~~

### p=8 headline, POST-FIX — the valid measurement

`bench/results/validation/mac_postfix_headline_p8.json`. N=4096, p=8, softening 1e-6,
Dehnen δa/f, eq (16a) **verbatim** (no θ cap), matched at equal p90. Both arms overlap
genuinely across the whole matched range — the θ grid is dense enough that no row is
interpolated across the far-field switch-on, which is what invalidated the first
post-fix attempt.

| distribution | p99 × | **max ×** | work × |
|---|---|---|---|
| **Plummer** (Dehnen's case) | 1.32 – 2.18 | **7.5 – 57.2** | 1.00 – 1.06 |
| uniform | 1.33 – 1.46 | **3.6 – 7.7** | 1.00 – 1.05 |

**Read this on the max, not the p99.** The p99 gain is modest and roughly matches what
the pre-fix run accidentally reported (1.3×); the tail gain is large and was understated
by more than an order of magnitude (pre-fix max ratio 1.51×). Tail *shape* at comparable
accuracy on Plummer makes it concrete:

| arm | p99 | p99.99 | p99.99 / p99 |
|---|---|---|---|
| mass MAC, ε=1e-4 | 8.3×10⁻⁵ | 3.6×10⁻⁴ | **4.3** |
| fixed θ=0.62 | 6.0×10⁻⁴ | 5.0×10⁻² | **83** |

That is Dehnen §5.3's actual claim — a reduction in the large-error tail at comparable
median — and it is now supported. **Report p99.99 as the headline metric, not p99**;
Dehnen quotes rms and p99.99 for the same reason, and the bench already computes it.

Cost is a wash (work 1.00–1.06×), so the criterion buys tail accuracy rather than speed.

### What the M2M fix did to absolute accuracy

Plummer N=4096, p=8, Dehnen δa/f p99, geometric MAC, before → after `8afd705`:

| θ | far pairs | before | after | factor |
|---|---|---|---|---|
| 0.46 | 1310 | 1.2×10⁻² | **3.5×10⁻⁶** | 3400× |
| 0.54 | 4718 | 7.7×10⁻² | **6.7×10⁻⁵** | 1150× |
| 0.62 | 8774 | 1.6×10⁻¹ | **6.0×10⁻⁴** | 270× |

θ ≤ 0.38 is unchanged: it accepts ≤ 62 far pairs, all shallow, so the bug never bit.
The FMM was not functioning as an accurate method on any configuration with a deep
enough tree, which is the context for every "the criterion wins the tail" claim above.

### Prepare cost (N=16384, p=8, fp64, `real` basis, FAST preset, A100)

Warm-call medians of 5 timed `prepare_state` calls, after a cold call and one
discarded warm-up, via `bench/validation/force_scale_prepare_cost.py`
(`bench/results/validation/force_scale_prepare_cost_n16384_p8.json`):

| arm | cold | warm median | warm calls | ratio | prepasses | far pairs |
|---|---|---|---|---|---|---|
| geometric | 320.60s | 40.04s | 39.2 / 39.2 / 40.1 / 40.0 / 52.2 | 1.00× | 0 | 168388 |
| mass MAC, `paper` | 305.42s | 235.05s | 236 / 386 / 227 / 231 / 235 | **5.87×** | 7 | 82246 |
| mass MAC, `paper_cached` | 214.63s | 39.23s | 41.1 / 39.2 / 38.4 / 40.2 / 38.7 | **0.98×** | 1 | 82246 |

Step 1 took the steady-state overhead from 5.87× to **0.98×** — the prepass is no
longer a cost at all. The mass MAC's `prepare_state` is now marginally *cheaper* than
the geometric one, which is consistent with it accepting half as many far pairs
(82246 vs 168388) and so building a smaller interaction list.

Two caveats on the 5.87×. It is worse than the 3.5× previously recorded here, which
was measured by hand with no committed harness; the box is contended and the 386s
outlier among otherwise ~230s calls shows it. And this is `prepare_state` only —
end-to-end wall-clock at Dehnen's N is still Step 4's job.

### Differentiability

No code change was needed. FD-vs-AD agrees to ~7e-10 (positions) / 5.6e-10 (masses)
with the mass MAC active. The MAC sits entirely on the frozen side of the seam. The
only substantive change was the default geometry mode: the old `"tree"` ran numpy host
loops and could not be traced at all.

### What Dehnen actually uses (for matching his setup)

Plummer spheres only, equal-mass, untruncated, N = 10⁴–10⁷. **p = 8 and 10.** **No
softening.** ε = **2×10⁻⁷** for (16a), 10⁻⁷ for (16b), 10^−6.25 at p=10. Reports **rms
and the 99.99 percentile** ("the rms error is always ten times smaller"). Error measure
is the **scaled error δa/f** with `f_b ≡ Σ_{a≠b} G μ_a / |x_a−x_b|²`.

## Traps — these cost real time in the last session

1. **Never quote a cold-start wall-clock.** `prepare_state` at N=16384/p=8 is ~320s cold
   and ~40s warm; the difference is JAX compilation. A cold-vs-cold comparison across
   two processes gave a 1.29× overhead where the true steady-state figure was 5.87×.
   Always report warm-call medians, and materialise the result —
   `bench/validation/force_scale_prepare_cost.py` does both.
2. **Per-particle relative error δa/a is invalid for clustered systems.** Wherever the
   vector sum `a_b → 0` — a force null between clumps, the centre of a
   centrally-concentrated profile — it diverges for *any* MAC, identically. On
   bulge+halo it reported p99 = 0.27 where Dehnen's δa/f gave 9.6e-3, a factor 28 of
   pure artifact. Use `--metric dehnen`.
3. **`bulge_halo` order-independence — likely the M2M bug, RETRACT OR RECONFIRM.** This
   trap recorded an *order-independent* error tail (identical at p=4 and p=8) and read
   it as a divergent series. The M2M defects fixed in `8afd705` produce exactly that
   signature — dropped mass is order-independent by construction — and clustered
   distributions build the deepest trees, so they were hit hardest (clu_solidfmm went
   2.5e-2 to 3.5e-4). Re-measure bulge+halo post-fix before treating it as a bad
   discriminator, and before drawing any MAC conclusion from it either way.
4. ~~**eq (16a) alone admits near-divergent pairs.**~~ **VOID — this was the M2M bug.**
   The original observation (acceptance at opening up to 0.997, ε powerless against it)
   was made when zero-multipole nodes made eq (15)'s estimate collapse to zero, so those
   pairs passed any threshold. Re-measured post-`8afd705` on Plummer N=4096 p=8 with
   eq (16a) **verbatim** (`mac_theta_max=1.0`):

   | ε | far pairs | median opening | max opening | frac > 0.9 |
   |---|---|---|---|---|
   | 1e-4 | 7988 | 0.586 | 0.786 | 0.000 |
   | 1e-5 | 2374 | 0.469 | 0.595 | 0.000 |
   | 1e-6 | 122 | 0.381 | 0.435 | 0.000 |

   Nothing above 0.786 is ever accepted, and the maximum opening *falls* as ε tightens
   — the correct convergent behaviour. **`mac_theta_max` is therefore not needed**, and
   the "disclosed deviation from the paper" it required goes away: eq (16a) as written
   is safe. Keep the knob (default 1.0 = verbatim), but do not set it below 1 without a
   fresh reason, and do not describe it as necessary.
5. **eq (16a) is very conservative at small N.** The threshold is `ε·min_b|a_b|` over
   the target cell, so the least-accelerated particle sets the budget. At N=64/leaf=4
   it accepts *nothing* for any ε in [1e-5, 1e-2]. Gradient tests need N≥512/leaf=8 at
   ε=3e-3 to get a non-empty M2L list.
6. **A reused force scale is not unconditionally safe, and the failure is silent and
   inverted.** A cached `min_b|a_b|` that is too *large* loosens the eq (16a) threshold,
   so the criterion over-accepts — a stale scale makes the solver **faster and wronger**,
   which means no cost measurement can detect it. Measured at N=512/p=4/ε=3e-3, stepping
   a system forward and comparing `paper_cached` against `paper` on identical final
   positions with the δa/f measure (two seeds, per-step displacement as a fraction of
   r_rms):

   | displacement / r_rms | 0.006 | 0.017 | 0.028 | 0.045 | 0.062 | 0.085 | **0.114** |
   |---|---|---|---|---|---|---|---|
   | p99 error ratio, seed 7 | 1.00 | 0.50 | 1.00 | 0.99 | 1.00 | 0.99 | **41.1** |
   | p99 error ratio, seed 11 | 1.00 | 1.41 | 0.92 | 1.00 | 1.01 | 0.94 | **15.8** |

   Flat to within mask-reshuffling scatter up to ~8.5% of r_rms per step, then a cliff.
   At the cliff the cached arm accepted 198 far pairs where the fresh one accepted 150.
   Real timesteps are two or more decades below that, so `paper_cached` is the right
   default — but Dehnen's "only very slightly worse" is a claim about *small* steps, and
   nothing in the code currently detects that the scale has gone stale. If large steps
   are ever needed, the mitigation is a refresh cadence (re-prepass every K calls,
   costing 1/K of `paper`), not a tolerance. `test_a_stale_cached_force_scale_does_not_
   degrade_the_error_tail` pins the safe regime at 0.028 and asserts the two arms
   actually disagree about the accept mask there, so it cannot go vacuous.
7. **Don't pipe long-running output through `tail`** — it buffers until exit, so a
   killed job shows nothing. Redirect to a file.
8. **The fast lanes were never active in any of these measurements.** The large-N path
   requires `expansion_basis="solidfmm"` *and* `preset="large_n_gpu"`; the benchmark
   uses `basis="real"`, so everything ran the generic path. Do not attribute the slow
   baseline to a lane fallback.

## The four steps

### Step 1 — Cache the force-scale prepass — **DONE (2026-08-01)**

**Result: 5.87× → 0.98×** warm-call prepare overhead at N=16384/p=8, against an
acceptance bar of ≤ 1.3×. Table and caveats under "Prepare cost" above.

**What shipped.** `mac_force_scale_mode="paper_cached"`: the paper prepass runs on the
cold call, and every call after that reuses the cached per-node scale. It is now what
`mac_type="dehnen_error"` selects by default — the `"prev"→"paper"` promotion at
`_fmm_impl.py` ~:635 became `"prev"→"paper_cached"`, because promoting a request to
*reuse* into the most expensive mode available was the whole defect. `"paper"` is
retained deliberately as the history-free upper bound, and is now the only mode that
guarantees a given `prepare_state` depends on nothing but its arguments.

**The `"prev"` decision: given a live writer, not retired.** It was inert on the
non-paper path — its only writer was the dead block deleted in `96d5a44` — so it fell
through to a *unit* force scale and stayed there, silently, for the whole run. A unit
scale is finite and non-`None`, which is why nothing caught it. There is now one writer,
`_record_force_scale_from_evaluation`, called from `evaluate_prepared_state`: after any
full-order evaluation it min/max-reduces that step's accelerations onto the nodes. Those
accelerations *are* Dehnen's `a_b`, one step stale, so this is a strictly better estimate
than the p=1 prepass and is exactly the reuse §5.4 licenses. It is skipped while tracing
(an instance attribute must never capture a tracer), during a prepass, and for
target-subset evaluations, which do not cover every node.

Consequence worth knowing: with reuse live, `compute_accelerations` called twice on
identical inputs no longer returns identical answers — the second call sees a refreshed
scale. That is inherent to any reuse scheme and is the point of the mode; use `"paper"`
when a call must be a pure function of its arguments. `prepare_state` on its own *is*
still idempotent, and there is a test pinning it to the bit.

**Two further defects found while in this code.**

- *Unbounded recursion*, pre-existing and reachable straight from public kwargs
  (`adaptive_order=True`, `adaptive_error_model="tail_proxy"`,
  `mac_force_scale_mode="paper"`, `mac_type="dehnen"`): `"paper"` set
  `need_prepass = True` *ahead* of the `_in_force_scale_prepass` guard, and the non-paper
  prepass re-enters `prepare_state`, so the inner call requested a prepass of its own,
  and so on to `RecursionError`. Confirmed against the parent commit before fixing. The
  guard is now hoisted above the mode dispatch: an inner prepare takes the cached scale
  or unity, never another prepass.
- *Reentrancy*, as predicted: the non-paper branch bumped
  `_topology_reuse_entry.reuse_count` twice per outer call. Measured at
  `rebuild_every=4`, the pre-fix sequence was `1, 3, 1, 3, …` — the tree rebuilt every
  second call instead of every fourth — and `recent_topology_reused` reported a reuse on
  calls that had just rebuilt. Now `0, 1, 2, 3, 0, 1`. Both prepass branches share one
  `_force_scale_prepass_scope` contextmanager, which also restores
  `_recent_far_pairs_by_gear_counts` (the non-paper branch never did).

**Where.** `fmm_policy.py` (the scope + the recorder), `fmm_prepare.py` (mode dispatch,
hoisted guard), `fmm_evaluate.py` (one call site), `_fmm_impl.py` (validation + the
promotion), README force-scale section.

**The accuracy half, which the plan did not ask for and should have.** A cost-only
acceptance bar cannot see the failure mode that matters here: an over-large cached scale
loosens eq (16a)'s threshold, so a stale scale over-accepts and is *faster and wronger*.
Measured, and it is real — see trap 6 for the cliff at ~11% of r_rms per step. It sits
two or more decades beyond any real timestep, so `paper_cached` stands as the default,
but the claim is now bounded by measurement rather than by §5.4's assertion.

**Tests.** `tests/unit/runtime/test_force_scale_prepass_cost.py`, 20 cases / 15
functions. Each fix has a test that provably bit before it: the prepass-count assertions
(1 vs 3 over three `prepare_state` calls), the `0,1,2,3,0,1` cadence, the four-mode
recursion parametrisation, `_last_force_scale_nodes` going from `None` to a non-unit
array across one evaluation, and a δa/f error-tail comparison against a direct sum that
also asserts the two arms genuinely disagree about the accept mask, so it cannot pass
vacuously.

**Harness.** `bench/validation/force_scale_prepare_cost.py` — warm-call medians with
forced materialisation, a discarded warm-up call, per-step position drift so the cached
scale is genuinely stale, and a live-far-pair count so an arm that has degenerated to
all-near-field cannot be compared by accident. The 3.5× figure in this document had no
committed harness; this one is reproducible.

### Step 2 — eq (16b) — **VALIDATED, NEGATIVE (2026-08-01)**

**What eq (16b) actually is.** It replaces `min_b a_b` with `min_b f_b`, the
cancellation-free force scale `f_b = Σ_{a≠b} G m_a / |x_a−x_b|²`. Nothing else changes:
the criterion, the traversal and the eq (15) error estimator are untouched. So the
whole of (16b) is a *different per-node force scale*, which is why no traversal work
was needed to test it.

**Infrastructure that shipped** (`357814e`, keep regardless of the verdict):

- `prepare_state(..., force_scale_nodes=...)` — supplies the scale directly, skipping
  the prepass and leaving the reuse cache untouched. This replaces the
  `_last_force_scale_nodes` back door, which Step 1's live writer overwrites after
  every evaluation; an injected `f_b` survived exactly one `prepare_state`, so a naive
  prepare/evaluate loop measured (16a) while believing it measured (16b).
- `compute_node_force_scale_from_sorted_magnitudes` — scalar node reduction. `f_b` is
  already scalar per particle; the vector entry point now takes a norm and delegates.
- `--arm mass_16b` in the bench, injecting exact O(N²) `f_b` to measure the ceiling
  before building an estimator for it.

**Verdict.** Two findings, one structural and one measured.

*Structural, and the more important:* (16b) **cannot** remove the need for
`mac_theta_max`, which was the main hope for it. The θ cap addresses acceptance at the
convergence boundary, where the eq (15) bound on the criterion's **left**-hand side
stops being trustworthy. (16b) changes the **right**-hand side. Different sides of the
same inequality — no choice of force scale excludes a pair whose left-hand estimate is
already small. This holds independently of the M2M bug.

*Measured, and provisional:* with `mac_theta_max=0.7` on Plummer p=8, matched at equal
p90, (16b) did not beat (16a) — p99 ratios 1.75 / 1.10 / 0.70 / 0.54 across the
matched range, degrading as tolerance loosens. **These numbers predate the M2M fix and
must be redone before being quoted.** The structural argument above does not depend on
them.

**Do not build the O(N) estimator yet.** It was scoped at 2–3 days on the assumption
that `f_b` is near-field dominated. It is not: measured at N=4096, the 16 largest
contributors capture a median 13 % of `f_b` on Plummer (18 % uniform, 7 % bulge+halo),
and even the largest 256 capture only 41 %. In 3D the shell population grows like
`r²ρ` while each contribution falls like `1/r²`, so every logarithmic shell contributes
comparably and the sum is a global quantity. A near-field-only estimator would be wrong
by nearly an order of magnitude, so the far-field monopole pass is mandatory — Dehnen's
"p=0 suffices" is right about the *order*, not about the *locality*. Given the negative
verdict, this work is not currently justified.

**Also settled:** the bulge+halo floor is not evidence for (16b). Both criteria floored
at the identical value pre-fix because of the M2M bug, and trap 3's "order-independent
tail" observation has the same likely cause. Re-measure bulge+halo post-fix before
drawing any conclusion from it.

### Step 3 — Per-node effective θ, so the mass MAC reaches the fast lanes (~2–3 days)

**Why.** The exact criterion needs a `pair_policy`, and that vetoes every production
fast lane — `can_use_large_n_prepare_path` returns False on
`_uses_paper_style_force_scale()`, the split/streamed builds require
`pair_policy is None`, and the treecode/Pallas walks have no hook at all. This step is
a **prerequisite for Step 4**.

**The mechanism (verified).** Bounding `P_n/M ≤ ρⁿ` collapses eq (16a)'s sum to
`θ^(p+2)`, giving a per-node

```
θ_i = clip( [ ε · a_min(i) · ρ_i² / (8 M_i) ]^(1/(p+2)), θ_floor, 1 )
```

(the `1/(p+2)` exponent is the one `adaptive_policy_tolerance` already uses — the code
knows this relation, it just applies it backwards). Feeding extents
`e_i = ρ_i · θ_g/θ_i` makes the lanes' existing scalar-θ test `(e_t+e_s)² ≤ θ_g²d²`
**algebraically identical** to `ρ_t/θ_t + ρ_s/θ_s ≤ d`. So this is a per-node rescale of
`geometry.radius`, not a new comparison — no `pair_policy`, hence no lane veto.

Two facts that make it work, both checked in yggdrax `_interactions_impl.py`:
`_build_mac_extents` (~:993) reads `geometry.radius` for `mac_type="dehnen"`, and
`_propagate_extents` (~:876) only fills nodes whose extent is `<= 0` — **it takes no
maxima**, so a per-node scaling survives. `dehnen_radius_scale` is also *not* in
`static_argnames`, so it is traced and could even accept a per-node array directly.

**Do.** New `per_node_effective_theta(...)` beside `resolve_dehnen_geometry` in
`_adaptive_policy.py`; one substitution point where `tree_artifacts.upward.geometry` is
handed to the dual build; a `mac_type` value to select it. Zero yggdrax changes; reaches
the generic walk, split build, compact-streamed build, treecode **and Pallas**.

**Traps.** Zero-radius single-particle leaves hit `_compute_leaf_effective_extents`'
depth padding and would bypass the scale — clamp to a tiny positive before scaling.
Keep `dehnen_radius_scale` at 1.0 in this mode or fold it in, and pin that with a test.

**What it gives up.** Per-sink `a_b` fidelity (each node uses its own scale) and the
measured multipole spectrum. Two variants:

- *Crude bound* (`P_n/M ≤ ρⁿ`). The eq (15) sum then collapses binomially to
  `M(ρ_z+ρ_s)^p` and the whole test reduces to `θ^(p+2) < ε·a_min·ρ²/(8GM)`, so θ_i is
  **closed form** — no solve. This is the formula above. It discards the measured
  spectrum, which is where the criterion's selectivity comes from, and the win is
  specifically in the *tail*, so this is the part most at risk.
- *Measured spectrum.* Keeping the real `P_n`, the sum `Σ_n c_{p,n} P_n ρ_s^{e(p,n)}`
  does not collapse to `(ρ_z+ρ_s)^p`, so there is no closed-form inverse and θ_i must be
  solved for numerically. Recovers most of the selectivity; **treat as mandatory, not
  optional**, given where the advantage lives.

**Solve it on device, with a fixed trip count.** An earlier draft of this document
suggested "~5 Newton steps per node on the host". That is wrong and contradicts the
direction this code already took — `dehnen_geometry_mode='tree'` was *removed as a
defect* for running a numpy host loop, and `resolve_dehnen_geometry` now raises rather
than run one under trace. `prepare_state` being eager does not mean host numpy; it means
eager JAX ops on device arrays.

`F_i(θ)` is a polynomial in θ with positive coefficients, hence strictly increasing
(measured: 1.7e-8 at opening 0.2 rising monotonically to 1.6e-1 at 0.995), and the
bracket `[θ_floor, 1]` is known a priori. So use **fixed-count bisection**, vmapped over
nodes inside one `lax.fori_loop`:

```python
def per_node_effective_theta(power, mass, radius, a_min, *, order, G, eps, iters=24):
    lo = jnp.full_like(mass, THETA_FLOOR)
    hi = jnp.ones_like(mass)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        over = _eq15_force_error(power, mass, radius, mid, order=order, G=G) > eps * a_min
        return jnp.where(over, lo, mid), jnp.where(over, mid, hi)

    lo, _ = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return lo          # conservative side of the bracket
```

Bisection beats Newton here for a specific reason: it needs no derivative, cannot
diverge, and has a **static** trip count. Newton would want a convergence test, and a
convergence test is exactly what forces a device-to-host read. Returning `lo` rather
than `mid` keeps rounding on the under-accepting side, which is the right default given
that over-acceptance has been the silent failure mode in every defect found so far.

Cost: `num_nodes × iters × (p+1)` FMAs, one fused kernel, no transfers. 24 iterations
gives 2⁻²⁴ ≈ 6e-8 absolute on θ. At 1M particles: ~1.7 MFLOP at leaf 256 (~8k nodes),
~27 MFLOP at leaf 16 (~125k nodes). Free beside a p=8 FMM.

**Accept when.** Accept-mask agreement and matched-work force error against the exact
criterion on the generic lane, before claiming fast-lane parity.

### Step 4 — Measure at Dehnen's regime: 1M Plummer, p=8 (~1–2 days)

**Feasibility is established.** 1M runs on one A100-40GB: `large_n_gpu`, leaf 256,
fp32, order 4 — 11.07 GB reverse peak, 2.50 s forward, 68.9 s forward+backward.
Forward-only error measurement is well under that. Requires
`expansion_basis="solidfmm"` + the large-N lane, hence Step 3 first.

**Reference strategy.** Direct sum at 10⁶ is 10¹² pairs — infeasible. Use **target
subsampling**: exact direct force for ~10⁴ random targets against all 10⁶ sources is
10¹⁰ pairs, trivial on an A100, and gives reliable quantiles to ~p99.9. Dehnen's p99.99
needs ~10⁵ targets (10¹¹ pairs, still feasible). Add `--reference-subsample N` to the
bench.

**Config.** Plummer, untruncated, equal-mass; p ∈ {8, 10}; ε near 2×10⁻⁷; softening as
small as the runtime tolerates (Dehnen uses none); `--metric dehnen`; report rms and
p99.99. N ladder 10⁵, 3×10⁵, 10⁶.

**Accept when.** The p=8 tail advantage measured at N=4096 (p99 1.34×, max 1.51× at
0.89× work on Plummer) holds or strengthens at N ≥ 10⁵, with warm-call wall-clock
within 1.3× of the geometric MAC.

## Go/no-go for the paper

**Re-based 2026-08-01** after the M2M fix invalidated the earlier basis. Three of the
four original conditions are now met at N=4096; only the N ≥ 10⁵ scaling is open.

| condition | status |
|---|---|
| tail advantage at p ≥ 8 on Plummer, δa/f | **met at N=4096**: max 7.5–57×, p99 1.3–2.2× |
| interaction work ≤ fixed-θ at matched p90 | **met, marginally**: 1.00–1.06× — a wash, not a saving |
| warm-call prepare overhead ≤ 1.3× | **met**: 0.98× after Step 1 |
| holds at N ≥ 10⁵ | **open** — needs Step 3 then Step 4 |

**Frame the claim on the tail, and quote p99.99.** The honest statement is not "the
criterion is faster" (work is a wash) and not "p99 improves 1.3–2.2×" (true but
unremarkable). It is that the *large-error tail collapses*: p99.99/p99 is 4.3 for the
mass MAC against 83 for fixed-θ at comparable accuracy on Plummer. That is §5.3's claim
and it is the one the data supports.

**No deviation from the paper to disclose any more.** `mac_theta_max` was the one
disclosed deviation; trap 4 showed it was compensating for the M2M bug, and eq (16a)
verbatim never exceeds opening 0.786 post-fix. Run and report it verbatim.

If it loses, the bug fixes and the 94 tests stay regardless (they fix shipped defaults),
and the negative result is worth writing up with the tests as evidence that the
transcription was faithful.

## Commands

```bash
# tests (MEMORY: autocvd + XLA_PYTHON_CLIENT_PREALLOCATE=false for GPU/xdist)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/runtime/ -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
    tests/unit tests/characterization tests/unit/test_adaptive_policy_runtime.py \
    tests/integration/test_adaptive_order_runtime.py tests/integration/test_force_scale_runtime.py -q

# CPU sweep (Dehnen's metric)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution \
    --n 4096 --leaf-size 16 --order 8 --distribution plummer,uniform \
    --theta 0.30,0.38,0.46,0.54,0.62 --eps 1e-5,1e-6,2e-7,1e-7,3e-8 \
    --softening 1e-6 --metric dehnen --json-out bench/results/validation/<name>.json

# GPU (pick a free device; the box is often contended)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution ...

# prepare-cost of the force-scale prepass (warm-call medians; ~50 min at this size)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    .venv/bin/python -m bench.validation.force_scale_prepare_cost \
    --n 16384 --leaf-size 16 --order 8 --repeats 5 --eps 2e-7 \
    --json-out bench/results/validation/force_scale_prepare_cost_n16384_p8.json
```

Existing artifacts in `bench/results/validation/` — `mac_dehnen_metric_p8.json` is the
headline p=8 run; `mac_plummer_p8_cap07.json` is the `mac_theta_max=0.7` arm;
`force_scale_prepare_cost_n16384_p8.json` is the Step 1 prepare-cost measurement.

## References

- Dehnen (2014), "A fast multipole method for stellar dynamics", arXiv:1405.2255 §5.
  §5.1 expansion centres, §5.2 cost scaling and θ_crit ≈ 0.46, §5.3 eqs (16a)/(16b),
  §5.4 the practical low-order estimate of `a_b`/`f_b`.
- `docs/adaptive_traversal_design.md` — the jaccpot/yggdrax ownership boundary.
- `docs/differentiable_fmm.md` — the fixed-topology contract the MAC lives behind.
- `docs/treecode_mac_stability.md` — prior in-repo finding that MAC bound-tightness
  governs secular heating; relevant if a multi-step drift test is added.

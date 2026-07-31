# Dehnen mass-dependent MAC: status and next steps

Handoff document, 2026-07-31; Step 1 completed and folded in 2026-08-01.
Self-contained: a fresh session should be able to pick this up without prior context.

## TL;DR

jaccpot implements Dehnen (2014, arXiv:1405.2255) §5's mass-dependent multipole
acceptance criterion — eqs (12), (13), (15), (16a) plus the §5.4 low-order prepass —
reachable via `mac_type="dehnen_error"`. That transcription is now **proven** by unit
tests against independent numpy float64 references, not merely asserted.

Four real correctness bugs were found and fixed (all on shipped default paths). After
the fixes, **at p=8 with Dehnen's own error measure the criterion wins the error tail**
on his own test distribution at equal-or-less interaction work. At p=4 it is a wash,
which is why an earlier p=4-only benchmark produced a spurious negative result.

Step 1 is done: steady-state prepare overhead went from **5.87× to 0.98×** at
N=16384/p=8, and two further defects (an unbounded recursion and the predicted
reentrancy bug) were fixed on the way. Open: eq (16b) is not implemented, the mass MAC
still cannot reach any fast lane, and nothing has been measured at Dehnen's N (10⁵–10⁷).

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
goldens. `tests/unit/runtime/` is 76 cases across 33 test functions.

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

### Guard-rails added

- **`adaptive_eps` is now required** in paper mode. It used to fall back to
  `theta**(p+2)` — a `tail_proxy` heuristic that at θ=0.6, p=4 gives **0.047**, a 4.7 %
  per-interaction tolerance. That is how a wildly loose ε reached the accuracy
  notebooks. Note θ does **not** gate acceptance in paper mode at all.
- **`mac_theta_max`** (default 1.0 = eq (16a) verbatim) caps the opening angle. See
  "why this is needed" below.
- Silent downgrades now raise: the strict fused device-only lane
  (`fmm_prepare.py` ~:795) and an explicit `runtime_path="large_n"` request
  (`_large_n_pipeline.py` ~:1918). The large-N decline reason is surfaced in
  `get_runtime_diagnostics()` as `large_n_path_declined_reason`.

### Tests: `tests/unit/runtime/`, 76 cases / 33 functions

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

### p=8 with Dehnen's own δa/f measure — the headline

| distribution | p99 | max | work |
|---|---|---|---|
| **Plummer** (his test case), eq (16a) verbatim | **1.34** | **1.51** | **0.89** |
| uniform, eq (16a) verbatim | 1.21–1.33 | 1.31–3.53 | 0.99–1.01 |
| Plummer, `mac_theta_max=0.7` | **1.82–1.95** | **1.53–1.65** | **0.94** |

### p=4 — a wash, which is why the earlier negative was spurious

uniform (global-rms metric): p99 0.93–1.10, max 1.06–2.45, work 0.93–1.01.

### Prepare cost (N=16384, p=8, fp64, `real` basis, FAST preset, A100)

Warm-call medians of 5 timed `prepare_state` calls, after a cold call and one
discarded warm-up, via `bench/validation/force_scale_prepare_cost.py`
(`results/validation/force_scale_prepare_cost_n16384_p8.json`):

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
3. **`bulge_halo` is not a valid discriminator.** Its error tail is *order-independent*
   (identical at p=4 and p=8), which is the signature of a divergent series rather than
   truncation error. Do not draw MAC conclusions from it. `plummer` is the right case.
4. **eq (16a) alone admits near-divergent pairs.** Its only geometric guard is θ < 1,
   which is the *boundary of convergence*; measured acceptance reached opening 0.997,
   where a p=4 expansion has O(1) error and eq (15)'s bound — derived assuming
   convergence — under-predicts. No ε protects against this; only `mac_theta_max` does.
   Capping below 1 is a **disclosed deviation** from the paper, not a bug fix.
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

### Step 2 — Implement eq (16b) (~half a day to validate, 2–3 days for production)

**Why.** Dehnen's eq (16b) replaces `min_b a_b` with `min_b f_b`, the cancellation-free
force scale. `f_b` cannot be driven to zero by force cancellation, which is the
pathology that makes `min_b a_b` erratic and is the prime suspect for the bulge+halo
floor. It may remove the need for `mac_theta_max` entirely. Only (16a) exists today —
grep confirms no `f_b` anywhere in `jaccpot/`.

**Validation path (do this first).** `chunked_force_scale` in
`bench/validation/mac_error_distribution.py` already computes exact `f_b`. Inject it:

- construct with `adaptive_error_model="dehnen_paper"` **directly**, not
  `mac_type="dehnen_error"` — the latter force-rewrites `"prev"→"paper_cached"` at
  `jaccpot/runtime/_fmm_impl.py` ~:635;
- set `mac_force_scale_mode="prev"` and assign `fmm._last_force_scale_nodes` to the
  min-reduced `f_b`, reusing `compute_node_force_scale_from_sorted_acc(reduction="min")`;
- add a third bench arm (`--arm mass_16b`).

**Read this before using that back door.** Step 1 gave `_last_force_scale_nodes` a live
writer: `_record_force_scale_from_evaluation` overwrites it after *every* full-order
`evaluate_prepared_state`. An injected `f_b` therefore survives exactly one
`prepare_state`, and is silently replaced by `min_b |a_b|` the moment you evaluate — so
a naive prepare/evaluate loop would measure (16a) while believing it measured (16b).
Either re-inject before each `prepare_state`, or add the `force_scale_nodes` parameter
to `prepare_state` first. There is still no such parameter, and it is now clearly the
right move rather than merely the cleaner one.

**Production path.** `f_b` needs an O(N) estimate. Dehnen states p=0 suffices: a
monopole-only pass accumulating `Σ m/d²` as a **scalar** (no vector cancellation)
instead of the usual vector sum. New small kernel plus a prepass mode; reuses the tree,
traversal, and node reduction.

**Accept when.** On Plummer at p=8, (16b) matches or beats (16a) on p99/max at equal
work; and the bulge+halo error floor either disappears without `mac_theta_max` or is
shown to be unrelated.

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
measured multipole spectrum (the `P_n/M ≤ ρⁿ` bound discards it). The latter is
recoverable with ~5 Newton/bisection steps per node on the host —
`O(num_nodes · p)` over concrete arrays, negligible beside the prepass — and doing so
restores most of the criterion's selectivity. Strongly recommended.

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

Claim the mass-dependent MAC iff, at p ≥ 8 on Plummer with Dehnen's δa/f measure:

- the tail advantage (p99 and max) holds at N ≥ 10⁵, and
- interaction work is ≤ the fixed-θ arm's at matched p90, and
- warm-call wall-clock overhead is ≤ 1.3× after Step 1.

Be honest about magnitude: the measured effect is **1.2–1.9× on p99**, not the
"remarkable" reduction §5.3's language suggests. And disclose `mac_theta_max` if the
final configuration uses a cap below 1 — that is a deviation from eq (16a), justified by
the convergence-boundary argument in trap 4.

If it loses, the bug fixes and the 76 tests stay regardless (they fix shipped defaults),
and the negative result is worth writing up with the tests as evidence that the
transcription was faithful.

## Commands

```bash
# tests (MEMORY: autocvd + XLA_PYTHON_CLIENT_PREALLOCATE=false for GPU/xdist)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/runtime/ -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
    tests/unit tests/characterization tests/test_adaptive_policy_runtime.py \
    tests/test_adaptive_order_runtime.py tests/test_force_scale_runtime.py -q

# CPU sweep (Dehnen's metric)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution \
    --n 4096 --leaf-size 16 --order 8 --distribution plummer,uniform \
    --theta 0.30,0.38,0.46,0.54,0.62 --eps 1e-5,1e-6,2e-7,1e-7,3e-8 \
    --softening 1e-6 --metric dehnen --json-out results/validation/<name>.json

# GPU (pick a free device; the box is often contended)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution ...

# prepare-cost of the force-scale prepass (warm-call medians; ~50 min at this size)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    .venv/bin/python -m bench.validation.force_scale_prepare_cost \
    --n 16384 --leaf-size 16 --order 8 --repeats 5 --eps 2e-7 \
    --json-out results/validation/force_scale_prepare_cost_n16384_p8.json
```

Existing artifacts in `results/validation/` — `mac_dehnen_metric_p8.json` is the
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

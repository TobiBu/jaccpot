# Dehnen mass-dependent MAC: status and next steps

Handoff document, 2026-07-31; Steps 1–3 folded in 2026-08-01/02.
Self-contained: a fresh session should be able to pick this up without prior context.

## START HERE — remaining work, in priority order

Branch **`feat/dehnen-mass-dependent-mac`**, worktree
**`/export/home/tbuck/jaccpot-mac-wt`**, off `main`. The upward-M2M fixes are already in
`main` (PR #56). Everything below runs from the worktree with
`PYTHONPATH=/export/home/tbuck/jaccpot-mac-wt` (the venv's editable install points at
the *other* checkout, so without this you silently test the wrong code).

**Read trap 9 before running any sweep.** Every item below is one measurement away from
being answered, and each was previously derailed by a configuration trap now recorded
there.

### 1. The open question that decides the paper — does the benefit hold at Dehnen's ε?

Everything is measured at ε = 3e-4…1e-5. **Dehnen uses 2e-7.** Seven attempts failed on
configuration, not on the criterion; the run is now cheap and fully specified:

```bash
cd /export/home/tbuck/jaccpot-mac-wt
eval $(/export/home/tbuck/jaccpot/.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD JAX_ENABLE_X64=1 \
  setsid nohup /export/home/tbuck/jaccpot/.venv/bin/python -u \
  -m bench.validation.mac_error_distribution \
    --n 100000 --leaf-size 256 --order 8 --distribution plummer,bulge_halo \
    --theta 0.30,0.34,0.38,0.42,0.46,0.50 --eps 1e-5,1e-6,3e-7,2e-7,1e-7 \
    --softening 1e-6 --metric dehnen --match-on median --reference-block 64 \
    --arm fixed,mass --json-out results/validation/mac_dehnen_eps_n1e5.json \
  > /tmp/eps.log 2>&1 < /dev/null &
```

Report **rms and p99.99** (what Dehnen quotes), matched on **median** (his comparison).
Expect traversal retries on the tight-ε configs; the run prints the converged caps at the
end — feed them back via `DualTreeTraversalConfig` for the rerun and *do not round up*.

### 2. Settle the N-scaling trend properly

At matched p90 the advantage looked like it decayed 4096 → 1e5; at matched median it did
not. One seed at each end, no error bars, and the two matchings disagree — so the trend
is not established either way. Run a clean ladder **N = 3e4, 1e5, 3e5** × **3 seeds** at
leaf 256, one methodology (`--match-on median`), and report rms/p99.99 with spread. This
is cheap and it decides whether the claim is "holds at scale" or "decays with N".

### 3. Step 2 production — the O(N) `f_b` estimator

eq (16b) beats eq (16a) (~1.5× p99, ~2× tail) but only measured with *exact* O(N²) `f_b`,
which is a ceiling, not a prediction. Build the estimator (monopole-only far-field
accumulation over the existing interaction lists plus the exact near-field scalar sum;
all on device, one jit) and measure it against the exact-`f_b` arm. The far-field term is
**mandatory** — `f_b` is not near-field dominated (16 largest contributors capture ~13 %).

### 4. Step 3′ — carry the pair policy into the fast lanes

The only remaining route to 10⁶ (Step 3 is refuted; see below). Mostly jaccpot plumbing,
not a yggdrax change. **Measure `store_far_tags` memory first** — supplying a pair policy
allocates a far-tag buffer in the minimum-memory lane, where the 1M reverse peak is
already 11.07 GB of 40 GB.

### 5. Loose ends

- **`mac_type="dehnen_theta"` is refuted** and retained only behind a `FutureWarning` so
  its negative result stays reproducible. Delete it if it stops earning that keep.
- **The cartesian basis is broken** (~1.8e-1 rel-L2 at both p=2 and p=4, versus solidfmm's
  8.1e-5). Pre-existing, out of scope here, now documented at the per-basis golden anchor
  rather than hidden behind a blanket tolerance. Worth its own issue.
- Interaction **work is a wash to a win** (1.00–1.06× at N=4096, mass uses 17–51 % *less*
  at N=1e5 matched p90). Frame the claim as tail accuracy at equal-or-less cost, never as
  speed.


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
Step 2 (eq 16b) is validated and **positive** — it beats eq (16a). Step 3 (fold the
criterion into per-node opening angles) is **refuted**, structurally.

Open: whether the benefit holds at Dehnen's ε = 2×10⁻⁷ (never measured); the N-scaling
trend; the O(N) `f_b` estimator; fast-lane access; and the cartesian basis's unexplained
~1.8×10⁻¹ error. See **START HERE** above for the ordered plan.

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
3. **`bulge_halo` — HALF RETRACTED, and it is now the criterion's best case.** This trap
   recorded an *order-independent* error tail and read it as a divergent series. Post-fix
   that is only true at large θ. Fixed arm at identical θ, so identical accept masks
   (far-pair counts match exactly across orders):

   | θ | p=4 p99 | p=8 p99 | order gain | p=4 p99.99 | p=8 p99.99 |
   |---|---|---|---|---|---|
   | 0.42 | 3.6e-8 | 5.7e-10 | **63×** | 5.2e-4 | 7.0e-6 |
   | 0.46 | 8.0e-4 | 8.9e-6 | **90×** | 2.3e-2 | 1.7e-3 |
   | 0.50 | 8.3e-3 | 2.1e-4 | **40×** | 1.3e-1 | 2.1e-2 |
   | 0.54 | 3.0e-2 | 1.4e-3 | 21× | 1.2e+0 | **2.3e+0** |
   | 0.58 | 5.7e-2 | 5.2e-3 | 11× | 3.0e+0 | **7.0e+0** |
   | 0.62 | 1.1e-1 | 1.6e-2 | 7× | 3.7e+0 | **8.9e+0** |

   At θ ≲ 0.50 the error is strongly **order-dependent** (p=8 beats p=4 by 40–90×) —
   ordinary truncation error, so the original premise fails there; it was reading the M2M
   bug. At θ ≳ 0.54 the p99.99 *exceeds 1.0* and *grows* with order, which is a genuine
   divergence signature. **So bulge_halo is a valid discriminator provided you stay at
   θ ≲ 0.5 and discard any row whose absolute error approaches unity.**

   And it is where the criterion wins biggest: p99 **5.9–85×**, max **400–2000×**
   (`results/validation/mac_postfix_bulge_halo.json`), against Plummer's 1.3–2.2× / 7.5–57×.
   That fits — bulge+halo is exactly the case where accelerations span decades, which
   `ε·min_b|a_b|` exploits and a fixed opening angle cannot see. Discard the
   machine-precision matched rows (p90 ≲ 1e-9); those are the log-interpolation artifact
   described in trap 8.
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
   killed job shows nothing. Redirect to a file, and use `python -u`; see trap 9, where
   this cost several diagnostic runs their entire output.
8. **`compare_arms` will happily interpolate across the far-field switch-on, and the
   ratios it reports there are meaningless.** The fixed-θ arm's p90 jumps many orders of
   magnitude between the last θ that accepts almost nothing and the first that accepts a
   real far field — on Plummer p=8 it went 8e-16 at θ=0.38 to 2.7e-7 at θ=0.46. A
   matched-p90 target inside that gap is log-interpolated across a discontinuity, and the
   comparison then reports ratios of round-off: it produced p99 ratios of 34.8 once and
   2.9e7 on bulge+halo, both pure artifact. Guard: only trust a matched row whose target
   p90 lies **inside both arms' measured ranges**, and use a θ grid dense enough to bridge
   the switch-on (0.42–0.66 in steps of 0.04 worked at N=4096/p=8). Sanity check: a p90
   target below ~1e-9 at p=8 is almost certainly in the gap.
9. **Benchmark configuration, hard-won. Read before running any sweep.**

   - **Match on the statistic the claim is about.** Dehnen section 5.3 claims a reduced
     tail at comparable *median*; we matched *p90* for a long time. On identical N=1e5
     data that one choice is the difference between "parity" (p99 0.91-1.23) and
     "rms 2.0-23.6x, p99.99 1.7-27.2x at ~38% less work". p90 is itself partly a tail
     statistic, so equalising it surrenders the property being measured. Now
     `--match-on {p90,median}`. p90 was originally correct -- at N=4096 the far field is
     shallow and the median saturates at machine precision -- but that stopped applying
     once the far field got deep, and nobody rechecked.
   - **eps has to be Dehnen's.** Sweeps ran at 3e-4..1e-5 for a long time; Dehnen uses
     **2e-7**, i.e. 50-1500x tighter. Inherited from N=4096, where trap 5 meant tight eps
     accepted nothing. It is reachable at larger N.
   - **Tight eps is slow, not broken.** eps=2e-7 at N=16384/leaf 16 takes ~271 s with
     **6 traversal retries**, each of which recompiles. That reads exactly like a hang
     (long wall time, low GPU utilisation, no output) and cost several wasted runs.
     Converged capacities, pass them up front to skip the cycle:
     `max_pair_queue=131072`, `max_interactions_per_node=8192`. The bench now records
     these per config (`retry_final_caps`) and prints them at the end of a run.
   - **Do not round the caps up.** Traversal buffers are `num_nodes *
     interaction_capacity`. Setting `max_interactions_per_node=1<<18` (32x the converged
     8192) tried to allocate 4 GiB on top of 32 and OOMed. Bigger is not safer here.
   - **leaf_size and N are coupled, and getting it wrong fails silently.** leaf 256 is
     the production value (what the 1M runs use) and is now the bench default, but at
     N=16384 it gives only 64 leaves: the criterion accepted **zero** far pairs at
     eps=2e-7 and the run degenerated to all-to-all direct summation. It was *faster*
     (95.9 s vs 271 s) and measured nothing. Rule of thumb: keep >= 128 leaves, so leaf
     256 wants N >= ~3e4 and really N >= 1e5. The bench now warns.
   - **Use `python -u` and never pipe a long run through `tail`.** Block-buffered stdout
     is discarded when `timeout` sends SIGTERM, so a killed run reports nothing at all;
     and `tail -N` silently threw away the far/near counts a diagnostic run existed to
     produce.
   - **Pick the GPU with `autocvd`, every time.** One sweep was launched onto a device
     with 25 GB already in use and OOMed for that reason alone.
10. **The fast lanes were never active in any of these measurements.** The large-N path
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

### Step 2 — eq (16b) — **VALIDATED, POSITIVE (2026-08-01, re-measured post-M2M-fix)**

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

**Verdict: (16b) wins.** Plummer p=8, eq (16a) verbatim, matched at equal p90, both
arms against the same fixed-θ baseline
(`results/validation/mac_postfix_16b_plummer.json`):

| matched p90 | (16a) p99× | **(16b) p99×** | (16a) max× | **(16b) max×** | work× |
|---|---|---|---|---|---|
| 3.7e-8 | 1.32 | **1.82** | 7.52 | **8.86** | 1.00 |
| ~2.2e-7 | 1.61 | **2.07** | 8.53 | **20.67** | 1.00 |
| ~1.3e-6 | 2.07 | **2.90** | 7.98 | **24.13** | 1.00 |
| ~7.7e-6 | 2.18 | **3.33** | 24.22 | **47.53** | 1.01 |
| ~4.6e-5 | 1.81 | **3.20** | 57.20 | **121.79** | 1.04 |

Uniformly better than (16a) on both p99 and max at identical work — roughly 1.5× on p99
and 2× on the tail. **Build the O(N) estimator**, and carry `f_b` through Step 3
alongside `a_min`.

*An earlier pre-fix measurement said the opposite* (p99 ratios 1.75 / 1.10 / 0.70 /
0.54, degrading with tolerance) and this document recorded Step 2 as negative. That was
measured with 10–23 % of the system mass missing from the far field. Void.

*One structural note survives but is now moot:* (16b) changes the criterion's
right-hand side, so it could never have fixed a left-hand-side problem such as
acceptance at the convergence boundary — which was the original hope for it. Post-fix
there is no such problem (trap 4), so the point no longer bears on the decision.

**The measured gain uses exact O(N²) `f_b`.** A production estimator is approximate, so
part of this may not survive. Treat this arm as the *ceiling* and re-measure the
estimator against it before quoting these ratios for anything else.

**Building the O(N) estimator: the far-field pass is mandatory.** It was scoped at 2–3
days on the assumption that `f_b` is near-field dominated. It is not: measured at
N=4096, the 16 largest contributors capture a median 13 % of `f_b` on Plummer (18 %
uniform, 7 % bulge+halo), and even the largest 256 capture only 41 %. In 3D the shell
population grows like `r²ρ` while each contribution falls like `1/r²`, so every
logarithmic shell contributes comparably and the sum is a global quantity. A
near-field-only estimator would be wrong by nearly an order of magnitude — Dehnen's
"p=0 suffices" is right about the *order*, not about the *locality*.

Shape: a monopole-only far-field accumulation over the existing interaction lists
(`Σ_A G M_A / |c_A − x_b|²`, a scalar sum with no cancellation) plus the exact
near-field scalar sum. Reuses the tree, traversal and node reduction. On device, one
jit, no host round trip.

**Also settled:** the pre-fix bulge+halo "floor" was the M2M bug, not evidence for
either criterion. See trap 3, now re-measured — bulge+halo turns out to be the criterion's
*strongest* case, not an invalid one.

### Step 3 — Per-node effective θ — **REFUTED (2026-08-01). Both variants.**

**The plan was:** collapse eq (16a) into one opening angle per node, feed it as
rescaled `geometry.radius`, and the lanes' existing scalar-θ test carries the
criterion with no `pair_policy` and therefore no lane veto. All three yggdrax facts
the plan rested on were verified true (`_build_mac_extents` reads `geometry.radius`;
`_propagate_extents` only fills `<= 0` and takes no maxima; `dehnen_radius_scale` is
traced). They were not sufficient.

**One thing the plan got better than expected.** The inversion is *closed form* — in
eq (15) the distance enters only as `r^(p+2)` and `improvement` depends solely on the
two radii, so the LHS is exactly `C_i·r^-(p+2)`. Verified two ways: `LHS·(r/ρ)^(p+2)`
constant to full double precision over a 6× range of `r`, and evaluating eq (16a) at
`r = 2ρ_i/θ_i` reproduces `ε·s_i` to 2.7e-15. **No Newton, no bisection, no iteration,
and the measured multipole spectrum is retained** rather than discarded by the crude
`P_n/M ≤ ρⁿ` bound. `per_node_effective_theta` in `_adaptive_policy.py`.

**Variant A — tight on average (`mac_type="dehnen_theta"`): unsound.**
N=4096/p=8 against the exact criterion
(`results/validation/theta_fidelity_p8.json`):

| distribution | ε | far exact | far θ | p99 exact | p99 θ | p99.99 θ | work × |
|---|---|---|---|---|---|---|---|
| Plummer | 3e-4 | 10666 | 14440 | 2.83e-4 | 3.58e-2 | 5.10e+0 | 1.35 |
| Plummer | 3e-6 | 688 | 2154 | 7.71e-7 | 9.35e-6 | 1.48e-4 | 3.13 |
| bulge_halo | 3e-4 | 6492 | 10476 | 1.26e-3 | 1.28e+0 | **2.33e+2** | 1.61 |
| bulge_halo | 3e-6 | 268 | 4042 | 9.38e-7 | 8.69e-3 | 2.68e-1 | **15.08** |

12–9300× worse error at 1.35–15× *more* work. No favourable operating point. Retained
behind a `FutureWarning` only so the negative result stays reproducible.

**Variant B — provably conservative (`per_node_conservative_extent`): sound but empty.**
Sound by construction (zero violations measured) via the AM-GM split
`√(M_A/s_B) ≤ ½(M_A/λ + λ/s_B)`, giving
`e_i = max(c·ρ_i, K·M_i/λ, K·λ/s_i)`. Best achievable acceptance, optimised over 13
decades of λ and six values of `c`: **0.08 %–0.58 %** of the exact criterion's far
pairs. At that rate the far field does not exist.

**The structural reason, which is the durable finding.** eq (16a) accepts when
`r^(p+2)` exceeds a **product** of a source term (mass, power) and a sink term (force
scale). Any per-node-extent test is a **sum**, `e_A + e_B ≤ θ_g·r`. A sum cannot
represent a product: make it safe for every pair and it must absorb the full dynamic
range of `M` and `s` (variant B, empty); make it tight on average and it is unsound
exactly on the tails that matter (variant A). **Separate source/target extent arrays
would not help** — the mismatch is sum-vs-product, not role-vs-role.

Consequence: **the criterion is intrinsically a pair test.** It will always require a
`pair_policy`, so it can never be folded into the cheapest purely geometric lanes.
That is a permanent ceiling on its achievable speed, not a gap to be closed.

### Step 3′ — Carry the pair policy into the lanes instead (revised scope)

The handoff claimed "the split/streamed builds require `pair_policy is None`" and framed
this as a yggdrax change. **That is wrong.** yggdrax accepts `pair_policy` in four walks
including `_dual_tree_walk_count_impl` and `_dual_tree_walk_compact_fill_impl` — which
*are* the split/streamed build. Every `pair_policy is None` gate is jaccpot-side:

| gate | blocks | fundamental? |
|---|---|---|
| `_can_split_dual_tree_build` | the split build outright | **no** — its own docstring says "intentionally narrow" |
| `can_use_large_n_prepare_path` | the whole large-N lane, on `_uses_paper_style_force_scale()` | no — a jaccpot policy decision |
| `_interaction_cache.py` ~:443 | interaction caching | yes, correctly (the key omits the policy) — perf only |
| `strict_split_fastlane` hint | a route-probing shortcut | perf only |
| `_large_n_pipeline.py` | never threads `pair_policy` at all | plumbing |

So this is mostly threading an existing yggdrax capability through jaccpot, plus running
the force-scale prepass inside `prepare_large_n_state`.

**The real risk:** `store_far_tags = pair_policy is not None` in both passes, so supplying
a policy allocates a far-tag buffer — in the *minimum-memory streamed* regime that lane
exists for, at 1M, where the reverse peak is already 11.07 GB of 40 GB. Measure the peak
before committing to the design. Fallback: make tag storage conditional on the caller
actually needing tags (jaccpot needs the actions, possibly not the retained tags).

### Step 4 — Measure at Dehnen's regime: 1M Plummer, p=8 (~1–2 days)

**Do the N=10⁵ measurement first, on the generic lane — it needs no lane work.** This
document previously coupled the scientific claim to the engineering by asserting Step 4
"requires the large-N lane, hence Step 3 first". That is true for 10⁶, not for 10⁵: the
error measurement is forward-only, and the generic lane runs it. Since Step 3 is refuted
and Step 3′ is a multi-day effort with a memory risk, establishing whether the tail
advantage survives to 10⁵ is both the cheaper question and the one that decides whether
any of the lane work is worth doing. Nothing above N=4096 has ever been measured.

Note also that interaction work is a **wash** (1.00–1.06×), so the criterion buys tail
accuracy at equal cost, not speed. If it stays on the generic lane while fixed-θ runs
the fast lane, end-to-end it is *slower*. That tension is unquantified and bears on
whether Step 3′ is worth it even if the claim holds.

**Feasibility at 1M is established.** One A100-40GB: `large_n_gpu`, leaf 256, fp32,
order 4 — 11.07 GB reverse peak, 2.50 s forward, 68.9 s forward+backward. Forward-only
error measurement is well under that, but it does need `expansion_basis="solidfmm"` plus
the large-N lane, hence Step 3′.

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

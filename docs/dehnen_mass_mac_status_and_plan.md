# Dehnen mass-dependent MAC: status and next steps

Handoff document, 2026-07-31; Steps 1–3 folded in 2026-08-01/02; the START HERE
plan worked through on 2026-08-02.
Self-contained: a fresh session should be able to pick this up without prior context.

## What the 2026-08-02 pass settled

| plan item | outcome |
|---|---|
| **3. O(N) `f_b` estimator** | **DONE and it costs nothing.** `mac_force_scale_mode="paper_fb"`. Retains a median 99.7 % of the exact O(N²) `f_b`, is a strict lower bound, and at matched p90 the estimator arm is indistinguishable from the exact-`f_b` ceiling (rms 7.90× vs 7.87×, p99.99 23.8× vs 22.6×, inside the two-seed spread). eq (16b) is now a production path, not a ceiling. |
| **4. Step 3′ memory gate** | **MEASURED, and it clears.** A pair policy costs **+977 MiB at 1M** (split/streamed build; +969 MiB monolithic), against the 11.07 GB reverse peak on a 40 GB card. Harness: `bench/validation/pair_policy_far_tag_memory.py`; artifact: `bench/results/validation/pair_policy_far_tag_memory_1m.json`. The plumbing itself is **not** done — see Step 3′ for the cache-key hazard that has to be handled first. |
| **5. `dehnen_theta`** | **Keep.** Its retention is now pinned by `tests/unit/runtime/test_refuted_dehnen_theta_mode.py` (4 cases) instead of asserted in prose — nothing previously tested the `FutureWarning` or that the mode still ran. |
| **1. Dehnen's ε** | **ANSWERED: the benefit holds at ε=2e-7, and is larger than at N=4096.** Matched on median at N=16384/leaf 16: Plummer rms **5.8–11.1×** and p99.99 **9.6–18.9×** for eq (16a), **9.4–24.9×** and **22.8–47.3×** for eq (16b) via the O(N) estimator, at 1.00–1.04× work; bulge+halo **70–218×** rms at **0.92–0.98×** work. The specified configuration had to be corrected first — at leaf 256 the sweep is degenerate, see trap 11. |
| **1b. N=10⁵ at a production leaf** | **DONE and stronger there.** Leaf 64, 2 seeds, matched median: eq (16a) rms **18.4×** / p99.99 **22.0×**, eq (16b) estimator **35.9×** / **53.9×**, work 1.01–1.05×. |
| **2. N-scaling *trend*** | **Measured; at fixed leaf 16 it decays** (p99.99 ~12× → ~5× over N = 16384 → 65536, work 1.0 → 1.2×, all 3 seeds agreeing). |
| **2b. Leaf-size sweep** | **RESOLVED, and it is the session's main finding: the advantage is controlled by tree DEPTH, not N.** At N=10⁵, p99.99 rises 5× → 7× → 16× → 35× → 31× and work falls 1.17× → 1.14× → 1.07× → 1.02× → **1.00×** across leaf 16 → 32 → 64 → 128 → **256**. The ladder was deepening the tree, not probing N. At leaf 16 the criterion is actually **worse than fixed θ on p99** (0.76–0.83×). |

Two defects found on the way, both in shipped behaviour — and the first of them
means **eq (16a) was measurably weaker than it should have been in every prior
run**: the prepass estimated its force scale badly, so the criterion over-accepted.
Fixing it made eq (16a) simultaneously more accurate (rms 1.4×, p99.99 1.6×),
*less* work (2374 → 2166 far pairs), and 2.4× faster to cold-prepare. Details and
the table under the first bullet.

- **The force-scale prepass ran at the solver's `theta`, which paper mode pins at
  1.0.** `theta` is documented as not gating acceptance — true of the criterion,
  false of the geometric traversal underneath it. At θ=1.0 the `f_b` estimate
  degrades to a median 0.74 of the exact value against 0.997 at θ=0.5. Now
  resolved independently via `mac_force_scale_prepass_theta` (default 0.5, near
  Dehnen §5.2's θ_crit ≈ 0.46), and pinned by a test that fails if it is ever
  reconnected to the solver's `theta`. **This also affected eq (16a)**: the p=1
  acceleration prepass was running at θ=1.0 too.

  > ⚠ **This changes the eq (16a) arm, so it is a seam in the artifact record.**
  > Every `mass`-arm number recorded before 2026-08-02 — including the post-M2M-fix
  > headline table below — was measured with the θ=1.0 prepass, because
  > `paper_cached` derives the criterion's scale from the *cold-call* prepass and
  > the bench does one prepare then one evaluate. Do not compare a pre- and
  > post-change `mass` arm directly; re-run the baseline.

  **Measured, and the old prepass was over-accepting exactly as trap 6 predicts.**
  Plummer N=4096, leaf 16, p=8, ε=1e-5, δa/f:

  | mode | prepass θ | far pairs | cold prepare | warm | rms | p99.99 | `f_b` fidelity |
  |---|---|---|---|---|---|---|---|
  | `paper` (16a) | **1.0 = old** | 2374 | 52.89 s | 10.92 s | 1.10e-6 | 2.65e-5 | — |
  | `paper` (16a) | **0.5 = new** | 2166 | **22.24 s** | 10.92 s | **7.89e-7** | **1.66e-5** | — |
  | `paper` (16a) | 0.3 | 2124 | 17.53 s | 9.26 s | 7.17e-7 | 1.19e-5 | — |
  | `paper_fb` (16b) | 1.0 | 2450 | 19.43 s | 9.46 s | 9.35e-7 | 1.10e-5 | 0.736 |
  | `paper_fb` (16b) | **0.5 = default** | 2994 | 17.05 s | 8.86 s | 1.24e-6 | 1.57e-5 | **0.997** |
  | `paper_fb` (16b) | 0.3 | 3018 | 13.38 s | 8.70 s | 1.25e-6 | 1.75e-5 | 1.000 |

  For **eq (16a)** the change is a win on every axis at once: it accepts *fewer*
  far pairs (2374 → 2166) *and* is more accurate (rms 1.4×, p99.99 1.6×) *and* the
  cold prepare is 2.4× faster. So every pre-2026-08-02 `mass` arm ran a criterion
  more permissive than its nominal ε — faster and less accurate than eq (16a)
  actually is — and **the corrected arm should make the criterion's advantage larger
  than recorded, not smaller.**

  Two things this does *not* say, both checked. It is **not** a uniform
  over-estimate of `min_b |a_b|`: the per-node scale moves in both directions when
  the prepass angle is tightened (one node went 32.1 → 50.8, i.e. *up*), so the
  first draft of this note — "the old scale was simply too large" — was wrong and an
  inequality test asserting it failed. The property that does hold, and is now
  pinned against an exact direct-sum `min_b |a_b|` reduced onto nodes, is that the
  tighter prepass lands **closer to the truth**; the net effect on acceptance
  follows from that rather than from a monotone shift. And the cost worry that
  prompted the measurement — that a smaller prepass θ means a bigger prepass near
  field and so a slower prepass — was simply backwards. Warm-call prepare is
  unchanged (10.92 s both), so the "≤ 1.3× warm prepare" bar is untouched either
  way.

  For **eq (16b)** the raw error at fixed ε gets *worse* as fidelity improves
  (rms 9.35e-7 → 1.24e-6), and that is not a defect: a faithful, larger `f_b` is a
  looser `ε·f_b`, so more pairs are accepted (2450 → 2994) and less near-field work
  is done. At fixed ε the rows are not comparable — only matched accuracy is, and
  there the estimator tracks the exact-`f_b` ceiling. What θ=0.5 buys for (16b) is
  **fidelity to eq (16b) as written** (0.736 → 0.997), which is the whole point.
- **The `|a_b|` recorder would have overwritten an `f_b` scale.**
  `_record_force_scale_from_evaluation` writes each evaluation's accelerations
  into the force-scale cache, which is what makes reuse mean anything for (16a).
  Under (16b) that silently reverts to the other criterion after one evaluation —
  the same failure that made an injected `f_b` survive exactly one
  `prepare_state`. Now suppressed for `f_b` modes, with a control test asserting
  the (16a) recorder is *not* inert, so the stability test cannot pass vacuously.

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

**ANSWERED (2026-08-02): yes, decisively, and by more than the N=4096 numbers
suggested.** `bench/results/validation/mac_dehnen_eps_n16384_leaf16.json`. N=16384, leaf 16,
p=8, one seed, softening 1e-6, eq (16a) verbatim, matched at equal **median** —
Dehnen §5.3's own comparison — with ε swept 1e-5 → 1e-7 so **2e-7 sits inside the
range** rather than at an endpoint.

**Plummer** (his test case). All seven fixed-θ rows have a real far field (33 782 →
172 300 pairs), the matched targets lie inside both arms' measured ranges, and the worst
p99.99 anywhere is 7.9e-2 — so traps 3 and 8 are both satisfied and every row is usable:

| matched median | (16a) rms × | (16a) p99.99 × | (16b) est rms × | (16b) est p99.99 × | work × |
|---|---|---|---|---|---|
| 1.9e-9 | 5.80 | 9.60 | 9.41 | 22.76 | 1.00 |
| 7.7e-9 | 6.10 | 10.90 | 11.38 | 26.46 | 1.00 |
| 3.0e-8 | 5.86 | 9.72 | 13.14 | 27.70 | 1.01 |
| 1.2e-7 | 7.34 | 11.67 | 18.41 | 37.37 | 1.04 |
| 4.8e-7 | 11.10 | 18.93 | 24.90 | 47.34 | 1.04 |

**bulge+halo**, after applying the document's own guards — discard fixed θ=0.58
(p99.99 = 8.89, trap 3) and treat the two tightest matched rows as suspect because they
sit adjacent to the far-field switch-on between θ=0.30 (3072 pairs, median 1.6e-15) and
θ=0.34 (19 412 pairs, median 1.5e-10), which is trap 8's gap:

| matched median | (16a) rms × | (16a) p99.99 × | (16b) est rms × | work × |
|---|---|---|---|---|
| 9.3e-9 | 218 | 405 | 433 | 0.95 |
| 5.4e-8 | 111 | 266 | 206 | 0.96 |
| 3.2e-7 | 70 | 126 | 137 | 0.97 |

Two decades of advantage, and it comes with **less work, not more** (0.92–0.98×). The
mechanism is visible directly in the raw rows rather than only in the ratio: at ε=1e-7
the criterion reaches median 2.7e-10 / rms 4.7e-8 with 16 792 far pairs, where fixed
θ=0.34 reaches a *comparable* median 1.5e-10 with a *comparable* 19 412 far pairs but
rms 1.9e-5 — 400× worse. Same median, same cost, collapsed tail. That is §5.3's claim
stated as plainly as the data can state it.

**The O(N) estimator matches or beats the exact-`f_b` ceiling here too**, despite its
fidelity being lower at this N/leaf than at N=4096 (median 0.918–0.947, worst single
particle 0.289, and always a lower bound). Plummer at matched median 4.8e-7: estimator
rms 24.90× / p99.99 47.34× against the exact arm's 24.14× / 46.17×. An ~8 % median
under-estimate of `f_b` is absorbed by the ε sweep, which is why fidelity in that range
costs nothing at matched accuracy.

Cost note for the record: `work ×` here is the bench's full `pair_work`
(`far_pairs·(p+1)² + Σ n_t·n_s`) from the JSON, not the far-pair-only proxy an earlier
in-flight reading of this run used.

Remaining caveat: **one seed.** Item 2's ladder supplies the spread.

Retries fired on only 1 of 44 configs and converged to exactly the caps passed in
(`max_pair_queue=131072`, `max_interactions_per_node=8192`), so those values are right
for leaf 16 at this N. **Read trap 11 first**: the
previously-specified run (N=1e5, leaf 256) is degenerate — the fixed arm accepts 0/0/4/218
far pairs at θ = 0.30/0.34/0.38/0.42, so it sits at machine precision and has no error
range to match against. Eight attempts have now failed on configuration.

Use **leaf 16**, where the same N=1e5 accepts 6.19 M far pairs at θ=0.34 and 1.15 M at
θ=0.74 — a real far field across the whole grid. Start at N=16384, which reaches ε=2e-7
for a twentieth of the cost and answers the ε question on its own:

```bash
cd /export/home/tbuck/jaccpot-mac-wt
eval $(/export/home/tbuck/jaccpot/.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD JAX_ENABLE_X64=1 \
  setsid nohup /export/home/tbuck/jaccpot/.venv/bin/python -u \
  -m bench.validation.mac_error_distribution \
    --n 16384 --leaf-size 16 --order 8 --distribution plummer,bulge_halo \
    --theta 0.30,0.34,0.38,0.42,0.46,0.50,0.58 --eps 1e-5,1e-6,3e-7,2e-7,1e-7 \
    --softening 1e-6 --metric dehnen --match-on median --reference-block 256 \
    --max-pair-queue 131072 --max-interactions-per-node 8192 \
    --arm fixed,mass,mass_16b,mass_16b_est \
    --json-out bench/results/validation/mac_dehnen_eps_n16384_leaf16.json \
  > /tmp/eps.log 2>&1 < /dev/null &
```

The bench now reports **rms and p99.99 ratios directly** (they are what Dehnen quotes;
p99 alone understated the effect by more than an order of magnitude), and
`--max-pair-queue`/`--max-interactions-per-node` pre-size the traversal so the tight-ε
configs skip the retry-recompile cycle. They must be given together — pinning one leaves
the other retrying, so the run would look pinned and behave exactly as before.

### 1b. Production-scale leaf at N=10⁵ — **DONE, and the advantage is bigger there**

`bench/results/validation/mac_leaf64_n1e5_2seeds.json`. N=10⁵, **leaf 64**, p=8, Plummer,
**2 seeds**, matched at equal median, ε ∈ {1e-6, 3e-7, 2e-7}. Run because the item-1
answer came from leaf 16, and leaf 16 is not what production uses — this checks the
result survives a leaf size four times larger, at a decade more particles:

| arm | rms × | p99.99 × | work × |
|---|---|---|---|
| eq (16a) | **18.37** [15.65, 21.09] | **21.95** [19.14, 24.75] | 1.04 [1.03, 1.05] |
| eq (16b), O(N) estimator | **35.92** [30.86, 40.97] | **53.93** [47.72, 60.14] | 1.02 [1.01, 1.03] |

`median [min, max]` over the 2 seeds, at matched median 1.05e-8. The seed spread is
±15 % on rms and ±13 % on p99.99, so the effect is not a realisation artifact. Work is a
wash (1.01–1.05×), as everywhere else.

**Do not read this as the N-trend.** It is larger than the N=16384/leaf 16 numbers
(rms 5.8–11.1× → 18.4×), but N *and* leaf_size both changed, which is exactly the
confound trap 11's corollary warns about. What it does establish is that the criterion's
advantage survives — and here strengthens — at a production leaf size and at 10⁵
particles, which is what the go/no-go condition actually needed. Item 2's ladder holds
leaf fixed and is the clean trend measurement.

eq (16b)'s estimator again roughly **doubles** eq (16a)'s advantage, with fidelity
median 0.913–0.914 (worst particle 0.578, always a lower bound) — so the ~9 % median
under-estimate continues to cost nothing at matched accuracy.

Converged traversal caps differ at this leaf size: **`max_pair_queue=262144`**,
`max_interactions_per_node=8192`. Pass those, not leaf 16's 131072.

### 1c. p = 10 — **DONE. Dehnen's other order, never measured before 2026-08-03**

Every number in this document was p ≤ 8; Dehnen quotes **p = 8 and p = 10**, and Step 4's
config asks for both. `bench/results/validation/mac_dehnen_p10_n16384_leaf16.json` — N=16384,
leaf 16, p=10, one seed, matched median, ε ∈ {1e-5, 1e-6, **5.62e-7 = 10^−6.25**, 2e-7,
1e-7}, bracketing the ε Dehnen uses at p=10.

Sanity check first: the fixed arm's far-pair counts are **identical** to the p=8 run
(33 782 / 85 288 / 133 470 / 166 840 …), the expected signature of an order-independent
geometric acceptance, and the errors are ~9× lower (median 4.0e-11 at θ=0.30 against
3.7e-10 at p=8). So the extra order really is being applied.

Plummer:

| matched median | (16a) rms × | (16a) p99.99 × | (16b) est rms × | (16b) est p99.99 × | work × |
|---|---|---|---|---|---|
| 2.3e-9 | 6.65 | 12.08 | 12.29 | 30.80 | 1.03 |
| 2.7e-8 | 11.14 | 16.91 | 23.53 | 48.83 | 1.07 |
| 9.0e-8 | 14.14 | 19.34 | 29.44 | 44.88 | 1.11 |
| 3.1e-7 | 15.52 | 12.19 | 32.92 | 33.59 | 1.15 |

bulge+halo, fixed θ=0.58 discarded (p99.99 = 9.80, trap 3): eq (16a) rms **101–388×**,
p99.99 **166–712×**; the (16b) estimator **196–703×** / **327–1140×**; work 0.95–1.05×.

**The advantage survives the order increase**, mildly better than p=8 on rms (6.7–15.5×
against 5.8–11.1×). Two wrinkles worth not hiding:

- **p99.99 is not monotone in the tolerance** (12.08 → 11.55 → 16.91 → 19.34 → 12.19),
  and the loosest row is the weakest. At p=10 the fixed-θ arm's own tail is already small
  in the loose regime, so there is less tail left to collapse.
- **Work drifts above parity as the tolerance loosens** — 1.03 → 1.15× on Plummer, where
  p=8 stayed within 1.00–1.06×. At p=10 the criterion buys its tail with a few per cent
  more interaction work in the loose regime. Still a wash, but no longer free: the
  "equal-or-less cost" framing should say "at p=8", or quote 1.15× alongside it.

### 2. The N-scaling trend — **MEASURED, and at fixed leaf 16 it DECAYS (2026-08-03)**

`bench/results/validation/mac_n_ladder_leaf16_3seeds.json`. N = 16384 / 32768 / 65536, **leaf
16 held fixed**, p=8, Plummer, **3 seeds**, matched at equal median, eq (16a). Per-seed
p99.99 and work ratios, so the spread is visible rather than averaged away:

| N | leaves | p99.99 × (all 3 seeds, across the matched range) | work × |
|---|---|---|---|
| 16 384 | 1024 | 7.9 – 25.5 | 0.98 – 1.07 |
| 32 768 | 2048 | 9.6 – 15.6 | 1.01 – 1.13 |
| **65 536** | 4096 | **3.0 – 8.5** | **1.07 – 1.20** |

**Every seed shows the decline**, so it is not a realisation artifact: at fixed leaf size
the advantage shrinks and the cost grows as N rises. The question this item was written
to decide — "holds at scale" or "decays with N" — answers **decays**, at leaf 16.

> ⚠ **This conflicts with item 1b**, which measured 22× at 1.04× work at N=10⁵ with leaf
> 64. At N=65536 with leaf 16 the same statistic is ~5× at ~1.15×. The two runs differ in
> leaf size, so either leaf size is the controlling variable (and leaf 16 at large N is
> simply an unrepresentative configuration — 4096 leaves and 3.3 M far pairs, far deeper
> than production would build), or the criterion genuinely decays and item 1b's leaf-64
> point is the outlier. **Resolving this by explanation would be exactly the kind of
> story-telling this document keeps punishing**, so it is being resolved by measurement:
> a leaf-size sweep 16 → 32 → 64 → 128 → **256** (the fiducial production value) at fixed
> N. See item 2b.

Two methodology notes for whoever repeats this:

- **Do not read the go/no-go "holds at N ≥ 10⁵" row off item 1b alone.** It was recorded
  as met on that single leaf-64 point before this ladder existed, which was premature —
  the fixed-leaf trend was already identified as the clean measurement and should have
  been waited for.
- The mass arm's far-pair count stops falling with tightening ε as N grows: at N=16384
  ε 1e-5 → 1e-7 gives 207 952 → 69 776 far pairs, but at N=65536 it gives
  2 021 314 → **3 295 822**, i.e. tightening ε pushes acceptance *deeper* rather than
  reducing it. That is why the work ratio climbs with N.
- 29 of 81 configs hit traversal retries (max 11 attempts), converging to
  **`max_pair_queue=1048320`**. The 131072 that is right for N=16384 is far too small at
  N ≥ 32768; that cost wall-clock, not correctness.

`--seed` takes a comma list and prints a cross-seed aggregate as `median [min, max]`,
joined on position in the matched ladder rather than on the target value (each seed's
usable range differs slightly, so the absolute targets never coincide).

### 2b. Leaf-size sweep — **RESOLVED: tree depth is the controlling variable, not N**

`bench/results/validation/mac_leafsweep_n1e5_leaf{16,32,64,128,256}.json`. N=10⁵ fixed, p=8,
Plummer, 2 seeds, matched on median, per-leaf knob grids bracketed from a census, both
guards on (`--min-far-pairs 5000`, `--max-p9999 1.0`).

Read at a **common absolute matched median** — the same target at every leaf size, not
each leaf's own range midpoint, because the advantage grows as the tolerance loosens and
midpoint-reading would confound leaf size with accuracy level. Both seeds shown
individually; the scatter matters:

**matched median = 7.5×10⁻⁸**

| leaf | leaves | rms × | p99.99 × | p99 × | work × | fixed's far pairs |
|---|---|---|---|---|---|---|
| 16 | 6250 | 5.6, 7.4 | 5.2, 4.6 | **0.83, 0.76** | 1.16, 1.18 | 5 509 383 |
| 32 | 3125 | 7.3, 7.7 | 7.0, 5.2 | 1.30, 1.09 | 1.12, 1.16 | 1 560 266 |
| 64 | 1562 | 9.5, 16.5 | 14.2, 17.7 | 2.73, 2.19 | 1.06, 1.08 | 354 029 |
| 128 | 781 | 18.5, 20.6 | 33.3, 37.6 | 3.45, 3.27 | 1.02, 1.02 | 63 716 |
| **256** (fiducial) | 390 | 20.3, 13.4 | **41.3, 21.1** | 3.37, 3.25 | **1.00, 1.00** | 9 640 |

**Monotone in every column, and the same ordering holds at all three common targets**
(7.5e-8, 5.4e-7, 3.9e-6). At the loosest, leaf 256 reaches p99.99 165×/65× against leaf
16's 4.1×/3.6×.

**This resolves the item-2 conflict.** Nothing was decaying with N. The ladder held leaf
16 fixed while N grew, which *deepens the tree*, and depth is what erodes the advantage —
so the ladder was walking down this table rather than measuring an N effect. Item 1b's
leaf-64 point and this sweep agree once leaf size is controlled for.

**The finding that only appears if you look past the tail: at leaf 16 the p99 ratio is
0.76–0.83.** The criterion is *worse* than fixed θ on p99 there, at 16–18 % more work.
On deep trees it is a net loss on p99 and only the extreme tail still favours it. This
document's insistence on quoting p99.99 rather than p99 is what kept that hidden — the
tail ratio is 4.6–5.2× at leaf 16 and looks like a win.

**Mechanism**, visible in the last column: to reach the same median accuracy the fixed-θ
arm needs 5.5 M far pairs at leaf 16 but only 9 640 at leaf 256 — 570× less far field for
the criterion to be selective about. eq (16a) earns its advantage by *declining*
interactions a fixed angle would accept; when the far field is already thin, there is
little left to decline, and the acceptance decisions it does make are individually much
more valuable.

Caveats. **2 seeds only**, and the scatter is real — leaf 256 gives p99.99 41.3 and 21.1
on the two seeds, so **leaf 128 and leaf 256 are not separable** at this sample size, and
neither is any claim finer than the monotone trend. The **work** column is the more
reliable signal: 1.16–1.18 → 1.12–1.16 → 1.06–1.08 → 1.02 → 1.00, monotone with no
overlap between adjacent leaf sizes at all.

**Consequence for the paper.** State the claim at production leaf sizes and say so. At
leaf 256, N=10⁵: p99.99 21–41× at **1.00×** interaction work. Do not state it as a
general property of the criterion — at leaf 16 it is ~5× on the tail and a *loss* on p99.
And note that leaf 256 is only meaningful at N ≳ 10⁵ in the first place (trap 11), so the
regime where the criterion looks best is also the regime production actually runs.

### 3. Step 2 production — the O(N) `f_b` estimator — **DONE (2026-08-02)**

`estimate_particle_force_scale` in `_adaptive_policy.py`, reached via
`mac_force_scale_mode="paper_fb"` (or `"paper_fb_cached"`). Exact scalar sums over the near
pairs, monopoles over the far ones, accumulated down the tree so each particle picks up its
whole ancestor chain's far lists. One jit, on device, no host round trip.

**It gives up nothing.** Median 99.7 % of the exact O(N²) `f_b`, a strict lower bound, and
at matched p90 against the fixed-θ baseline the estimator arm and the exact-`f_b` arm agree
inside the two-seed spread:

| arm | rms × | p99.99 × | work × |
|---|---|---|---|
| eq (16a) | 5.33 [5.03, 5.62] | 10.05 [9.97, 10.13] | 1.00 |
| eq (16b), exact O(N²) `f_b` | 7.87 [7.78, 7.96] | 22.62 [21.95, 23.29] | 1.00 |
| eq (16b), **O(N) estimator** | **7.90 [7.82, 7.99]** | **23.79 [23.37, 24.21]** | 1.00 |

Plummer, N=4096, p=8, matched p90, 2 seeds, `median [min, max]`. So eq (16b) is a
production path now, and the ~2× tail gain over (16a) is real rather than a ceiling.

The far-field term is **mandatory**, as predicted: a near-only estimate captures only
53–66 % of `f_b` once there are enough leaves for the far field to matter (0.989 at 64
leaves, 0.807 at 256, 0.534 at 512 — so the obvious small test configuration is exactly
the one where a near-only bug hides). Pinned by
`tests/unit/runtime/test_fb_force_scale_estimator.py` (11 cases).

**Two design points worth not re-deriving.** The estimate errs *low* on purpose:
eq (16a)'s threshold is `ε·s`, so an over-large scale loosens acceptance and makes the
solver faster *and* wronger, which no cost measurement can detect.
`mac_force_scale_fb_inflation=0.0` gives the tighter non-bounding variant for measurement
only. And node masses come from a prefix sum over `node_ranges`, never from
`multipole_packed[:, 0]` — the two M2M defects this branch fixed left nodes that span
particles with a zero expansion, and an estimator sourced from the multipoles would have
inherited both while reading as ordinary truncation error.

### 4. Step 3′ — carry the pair policy into the fast lanes

The only remaining route to 10⁶ (Step 3 is refuted; see below).

**The memory gate is measured and it clears.** `bench/validation/pair_policy_far_tag_memory.py`,
artifact `bench/results/validation/pair_policy_far_tag_memory_1m.json`. At N=10⁶, leaf 256,
fp32, θ=0.5 (7813 nodes, capacity 8192), same far-pair count in both arms so the delta is
the policy and not a different accept mask:

| build | no policy | with policy | delta |
|---|---|---|---|
| split / streamed | 3634.0 MiB | 4610.6 MiB | **+976.6** |
| monolithic | 4041.6 MiB | 5010.6 MiB | **+968.9** |

≈ 1 GB against an 11.07 GB reverse peak on a 40 GB card — affordable. Two corrections to
the concern as recorded: the cost is **not** specific to the streamed lane (both builds pay
about the same), and it is **not** mostly the far-tag buffer (that is only 244 MiB of the
970; the rest is `_resolve_pair_actions` evaluating the policy twice, forward and reverse,
and materialising both tag arrays). Reading the allocation sites alone would have got this
wrong in both directions — the streamed fill allocates `far_tags` unconditionally, with no
`store_far_tags` gate, which looks like "a policy is free here" and is not.

Measuring it needs one subprocess per arm: `peak_bytes_in_use` is process-cumulative and
cannot be reset, so measuring both arms in one process reports the running maximum for both
and the delta is always exactly zero. That first read looked like a clean null result.

**What remains, and the hazard to handle first.** `_interaction_cache_key` takes **no**
`pair_policy` argument, so the interaction cache cannot distinguish a policy-built list
from a geometrically-built one. Today that is safe only because the cache is per-solver-
instance and `eps` is fixed on a live solver — two `dehnen_error` solvers at different
`eps` already hash identically (`_base_mac_type()` is `"dehnen"` and paper mode pins
θ=1.0). Relaxing the policy gates without keying or disabling the cache would silently
serve the wrong accept mask, which is the "faster and wronger" failure this whole document
keeps running into. So: key the cache on the policy (or disable it under one) **before**
touching the gates. The gates themselves are small — `_can_split_dual_tree_build`
(`_interaction_cache.py` ~:443) plus threading `pair_policy`/`policy_state` into
`_build_dual_tree_artifacts_split`'s three yggdrax calls, all of which already accept it —
and then `can_use_large_n_prepare_path` and `_large_n_pipeline.py`, which never thread it
at all, plus running the force-scale prepass inside `prepare_large_n_state`. This was
deliberately not started in the 2026-08-02 pass: it is untestable at 1M until the
target-subsampled reference (`--reference-subsample`, Step 4) exists, and shipping
unvalidated lane code that can bypass the criterion is worse than not shipping it.

### 5. Loose ends

- **`mac_type="dehnen_theta"`: keep.** It earns its retention, and that retention is now
  *pinned* rather than asserted: `tests/unit/runtime/test_refuted_dehnen_theta_mode.py`
  checks the `FutureWarning` fires, that `dehnen_error` does **not** warn (so the warning
  cannot be widened until users learn to ignore it), and that the mode still runs and still
  publishes the per-node angles `bench/validation/per_node_theta_fidelity.py` reads.
  Nothing tested any of that before. If that file becomes a maintenance cost, that is the
  signal to delete the mode and keep the refutation in prose.
- **The cartesian basis is broken** (~1.8e-1 rel-L2 at both p=2 and p=4, versus solidfmm's
  8.1e-5). Pre-existing, out of scope here, now documented at the per-basis golden anchor
  rather than hidden behind a blanket tolerance. **Filed as issue #62** (2026-08-02) with
  the order-independence argument and three suggested first checks. Off this document's
  critical path from here.
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
   (`bench/results/validation/mac_postfix_bulge_halo.json`), against Plummer's 1.3–2.2× / 7.5–57×.
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
11. **"≥ 128 leaves" is necessary but nowhere near sufficient. leaf 256 has no far field
   below N ≈ 10⁶.** This is trap 9's leaf/N coupling again, one level deeper, and it
   silently invalidated the run this document specified for its own top-priority item.
   The bench's guard checks *leaf count*, and N=1e5 / leaf 256 gives 390 leaves, so it
   passes — but leaf count is not the thing that matters. Bigger leaves have bigger radii,
   so the same θ accepts dramatically fewer pairs. Measured far-pair counts, N=1e5, p=8,
   Plummer, fixed arm:

   | θ | leaf 256 | leaf 16 |
   |---|---|---|
   | 0.30 | **0** | — |
   | 0.34 | **0** | 6 186 986 |
   | 0.38 | **4** | — |
   | 0.42 | **218** | 4 422 164 |
   | 0.50 | — | 3 053 358 |
   | 0.74 | — | 1 149 054 |

   Four orders of magnitude apart at θ=0.42. At leaf 256 the whole fixed arm sits at
   machine precision (median 5e-16, rms 3e-16) with nothing to match the mass arm
   against, *and* it is slow, because with no far field accepted the run degenerates to
   all-to-all direct summation — 25 minutes per config at N=1e5, which reads as "the
   tight-ε configs are just expensive". It is not expensive, it is empty.

   **Diagnose this before committing to any sweep.** A prepare-only far-pair census is
   minutes, not hours: build the tree, count `interactions.sources >= 0`, skip the O(N²)
   reference and the evaluation entirely. If the fixed arm's far count is not in the
   millions at your N, the grid is not measuring the criterion.

   **Traps 3 and 8 are now enforced in code, not just described here.** They stopped
   being avoidable by care once `leaf_size` became a swept axis, so `compare_arms` takes
   two filters and both should be on for any leaf sweep:

   - `--min-far-pairs 5000` drops all-near-field configs, whose error sits at machine
     precision and would *widen* the arm's apparent range enough to let the matched
     target be interpolated across the far-field switch-on (trap 8).
   - `--max-p9999 1.0` drops diverged configs. An absolute error at or above 1 is a
     series evaluated outside its convergence region, not a coarse point on the same
     curve, so interpolating through it is meaningless (trap 3). These sit at the *top*
     of the fixed arm's range, which is exactly where matching reaches.

   Both announce what they dropped, with knob values. And when fewer than two configs
   survive, `compare_arms` now says so and tells you to widen the grid — it used to
   return an empty table, which reads identically to "no effect found".

   **Measured need for both, at N=1e5/leaf 256:** the fixed arm has < 5000 far pairs at
   θ ≤ 0.46 *and* p99.99 of 10 and 290 at θ = 0.78 and 0.90. Only θ ∈ {0.54, 0.62, 0.70}
   is usable — a three-point basis, and the two guards are what make that visible
   instead of silently interpolating from 0 far pairs to a diverged expansion.

   **The criterion's usable ε range shifts about two decades across the leaf range.** At
   leaf 16 / N=1e5 it wants ε ≈ 1e-5…1e-7; at leaf 256 the same N wants ε ≈ 1e-3…1e-5
   (ε=1e-6 accepts 406 far pairs, 3e-7 accepts 30, 1e-7 accepts 0). So a leaf sweep needs
   a **per-leaf** knob grid bracketed from the census. That is bracketing, not
   cherry-picking, provided the methodology — matched statistic, both guards, seed count
   — is identical across leaf sizes, and provided the cross-leaf comparison is finally
   read at a **common absolute matched target** rather than at each leaf's own range
   midpoint. Each leaf reaches a different accuracy window, and the advantage grows as
   the tolerance loosens, so comparing midpoints would confound leaf size with accuracy
   level.

   **leaf 64 at N=1e5 is the usable compromise.** Censused, and every configuration has
   a real far field — so the leaf-16 result can be checked at a production-scale leaf
   rather than only at the small one:

   | arm | knob | far pairs | near leaf pairs | prepare |
   |---|---|---|---|---|
   | fixed | θ=0.38 | 281 950 | 1 965 018 | 152.5 s |
   | fixed | θ=0.46 | 380 424 | 1 469 408 | 42.0 s |
   | fixed | θ=0.54 | 359 636 | 1 068 500 | 42.8 s |
   | fixed | θ=0.62 | 303 958 | 792 436 | 45.4 s |
   | mass | ε=1e-6 | 409 574 | 1 684 898 | **1337.5 s** |
   | mass | ε=2e-7 | 284 634 | 2 025 376 | **1295.5 s** |

   Budget for the mass arm's prepare: ~22 minutes per config at this size, against ~45 s
   for the geometric arm. That is the criterion's prepass plus traversal retries, and it
   is why a leaf-64 × 3-seed × full-grid run is a many-hour job rather than an
   afternoon's. Note the far count is **not** monotone in θ — it peaks near θ=0.46 and
   falls after, because larger θ accepts higher up the tree, giving fewer but bigger
   pairs. Do not read a falling far count as a tightening criterion.

   Corollary for the N-ladder: **leaf_size must be held fixed across it.** A ladder that
   keeps leaf 256 while N grows crosses from "no far field" to "real far field" partway
   up, and reports that crossing as an N-scaling trend.

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
(`bench/results/validation/mac_postfix_16b_plummer.json`):

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

**The O(N) estimator is built, and the ceiling survives it entirely (2026-08-02).**
`estimate_particle_force_scale`, reached via `mac_force_scale_mode="paper_fb"`. The
caution below — "a production estimator is approximate, so part of this may not
survive" — turned out to be unnecessary: at matched p90 the estimator arm and the
exact-O(N²) arm agree inside the two-seed spread (rms 7.90 vs 7.87×, p99.99 23.8 vs
22.6×), because the estimate retains a median 99.7 % of the exact `f_b`. Table and
design notes under START HERE item 3.

**The far-field pass is mandatory, as predicted.** It was scoped at 2–3 days on the
assumption that `f_b` is near-field dominated. It is not: measured at
N=4096, the 16 largest contributors capture a median 13 % of `f_b` on Plummer (18 %
uniform, 7 % bulge+halo), and even the largest 256 capture only 41 %. In 3D the shell
population grows like `r²ρ` while each contribution falls like `1/r²`, so every
logarithmic shell contributes comparably and the sum is a global quantity. A
near-field-only estimator would be wrong by nearly an order of magnitude — Dehnen's
"p=0 suffices" is right about the *order*, not about the *locality*. Confirmed
end-to-end: a near-only estimate captures 0.989 / 0.807 / 0.534 of the exact `f_b` at
64 / 256 / 512 leaves.

Shape as built: a monopole-only far-field accumulation over the existing interaction
lists plus the exact near-field scalar sum. Reuses the tree, traversal and node
reduction. On device, one jit, no host round trip.

One implementation subtlety that is easy to get wrong. The natural form
`Σ_A G M_A / |c_A − x_b|²` needs a per-particle distance, which would cost
`Σ_pairs span(target)` work. Instead each far pair contributes
`G M_A / (|c_A − c_U| + ρ_U)²` **once**, and the result is accumulated *downward* so a
particle picks up its whole ancestor chain — O(pairs + nodes), and the ρ_U inflation is
what makes it a lower bound. That downward accumulation must run in **descending span
width**, not descending node index: radix internal nodes are not stored in postorder,
which is the same ordering trap as D2 and M1.

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
(`bench/results/validation/theta_fidelity_p8.json`):

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
| `_can_split_dual_tree_build` (`_interaction_cache.py` ~:443) | the split build outright | **no** — its own docstring says "intentionally narrow"; the three yggdrax calls in `_build_dual_tree_artifacts_split` all already accept `pair_policy`, they are just never passed it |
| `can_use_large_n_prepare_path` | the whole large-N lane, on `_uses_paper_style_force_scale()` | no — a jaccpot policy decision |
| `_interaction_cache_key` | nothing — and that is the problem, see below | **the one real blocker** |
| `strict_split_fastlane` hint | a route-probing shortcut | perf only |
| `_large_n_pipeline.py` | never threads `pair_policy` at all | plumbing |

So this is mostly threading an existing yggdrax capability through jaccpot, plus running
the force-scale prepass inside `prepare_large_n_state`.

**The memory risk is measured, and it is not the blocker (2026-08-02).** ≈ 1 GB at 1M
(+976.6 MiB split, +968.9 MiB monolithic) against an 11.07 GB reverse peak on 40 GB.
Table and method under START HERE item 4. Two things the concern as written got wrong:
the cost is not specific to the streamed lane, and it is not mostly the far-tag buffer
(244 MiB of 970) — most of it is `_resolve_pair_actions` evaluating the policy twice,
forward and reverse. The suggested fallback of making tag storage conditional would
therefore recover well under a third of it, and in the streamed build nothing at all:
`_dual_tree_walk_compact_fill_impl` allocates `far_tags` unconditionally, with no
`store_far_tags` gate. Note also that this is *prepare*-side traversal memory, whereas
the 11.07 GB figure is the reverse-pass peak, so the two do not simply add.

**The actual blocker is the interaction cache key.** `_interaction_cache_key` takes no
`pair_policy` argument, so a cached entry cannot record which criterion built it. It is
safe today only by accident of scoping — the cache is per-solver-instance and `eps` is
fixed on a live solver — but two `dehnen_error` solvers at different `eps` already hash
identically, since `_base_mac_type()` returns `"dehnen"` and paper mode pins θ=1.0.
Relax the policy gates without first keying the cache on the policy (or disabling it
under one) and a geometrically-built interaction list can be served to a
criterion-driven request: the accept mask is silently wrong, the run is *faster*, and
per trap 6 no cost measurement can see it. **Key or disable the cache first, then touch
the gates.**

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
| tail advantage at p ≥ 8 on Plummer, δa/f | **met at N=4096**: max 7.5–57×, p99 1.3–2.2×; on the rms/p99.99 statistics Dehnen actually quotes, 5.3× / 10.1× for (16a) and 7.9× / 22.6× for (16b) at matched p90 |
| interaction work ≤ fixed-θ at matched p90 | **met, marginally**: 1.00–1.06× — a wash, not a saving |
| warm-call prepare overhead ≤ 1.3× | **met**: 0.98× after Step 1. eq (16b) is *cheaper* still — its prepass needs only the traversal, not a low-order FMM evaluation |
| eq (16b) reachable in O(N) | **met (2026-08-02)**: the estimator retains the exact-`f_b` result inside seed spread, so the ~2× tail gain over (16a) is production-reachable |
| holds at Dehnen's ε = 2e-7 | **met (2026-08-02)**: matched on median at N=16384/leaf 16, Plummer rms 5.8–11.1× / p99.99 9.6–18.9× for (16a) and 9.4–24.9× / 22.8–47.3× for (16b); bulge+halo 70–218× rms at 0.92–0.98× work. ε=2e-7 is inside the swept range, not at an endpoint |
| holds at N ≥ 10⁵ **at a production leaf size** | **met, and this is the form the claim must take.** N=10⁵, leaf 256 (fiducial), matched median, 2 seeds: p99.99 **21–41×**, rms 13–20×, at **1.00×** interaction work. It is *not* a leaf-independent property — see the next row |
| holds at N ≥ 10⁵ on *deep* trees | **no.** At leaf 16 / N=10⁵ the tail advantage is only 4.6–5.2× and the **p99 ratio is 0.76–0.83, i.e. worse than fixed θ**, at 16–18 % more work. Quote the claim with its leaf size attached |
| the N-*trend* at fixed leaf | measured, and it decays at leaf 16 (p99.99 ~12× → ~5× over N = 16384 → 65536). **Item 2b showed this is a tree-depth effect, not an N effect** — the ladder was deepening the tree, and depth is what erodes the advantage |

**Frame the claim on the tail, and quote p99.99.** The honest statement is not "the
criterion is faster" (work is a wash) and not "p99 improves 1.3–2.2×" (true but
unremarkable). It is that the *large-error tail collapses*: p99.99/p99 is 4.3 for the
mass MAC against 83 for fixed-θ at comparable accuracy on Plummer. That is §5.3's claim
and it is the one the data supports. `compare_arms` now reports rms and p99.99 ratios
directly, so there is no longer any reason to quote p99.

**And attach the leaf size to it.** Item 2b makes this non-negotiable: the advantage is a
function of tree depth, and quoting a leaf-independent number is not supportable. At the
fiducial leaf 256 the criterion gives p99.99 21–41× at 1.00× work; at leaf 16 it gives
4.6–5.2× on the tail and is *worse than fixed θ on p99* (0.76–0.83×) at 16–18 % more
work. Both are true statements about the same criterion. The paper should state the
production configuration and disclose the deep-tree behaviour rather than let a reader
assume the headline generalises — which is exactly the mistake this document made when it
recorded "holds at N ≥ 10⁵" as met from a single leaf-64 point.

**No deviation from the paper to disclose any more.** `mac_theta_max` was the one
disclosed deviation; trap 4 showed it was compensating for the M2M bug, and eq (16a)
verbatim never exceeds opening 0.786 post-fix. Run and report it verbatim.

If it loses, the bug fixes and the tests stay regardless — they fix shipped defaults, and
two of the defects found on 2026-08-02 (the prepass running at θ=1.0, the `|a_b|` recorder
overwriting an `f_b` scale) are in that category too. The negative result is worth writing
up with the tests as evidence that the transcription was faithful.

## Commands

```bash
# tests (MEMORY: autocvd + XLA_PYTHON_CLIENT_PREALLOCATE=false for GPU/xdist)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/unit/runtime/ -q
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
    tests/unit tests/characterization tests/unit/test_adaptive_policy_runtime.py \
    tests/integration/test_adaptive_order_runtime.py tests/integration/test_force_scale_runtime.py -q

# CPU sweep (Dehnen's metric). leaf 16 -- NOT 256, see trap 11
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution \
    --n 4096 --leaf-size 16 --order 8 --distribution plummer,uniform \
    --theta 0.30,0.38,0.46,0.54,0.62 --eps 1e-5,1e-6,2e-7,1e-7,3e-8 \
    --softening 1e-6 --metric dehnen --json-out bench/results/validation/<name>.json

# GPU (pick a free device; the box is often contended)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 .venv/bin/python -m bench.validation.mac_error_distribution ...

# ALWAYS do this before a big sweep: prepare-only far-pair census (trap 11).
# Minutes, not hours -- no O(N^2) reference and no evaluation. If the fixed arm's
# far count is not in the millions, the grid is not measuring the criterion.
#   build the tree, then count `interactions.sources >= 0` per (leaf_size, knob)

# prepare-cost of the force-scale prepass (warm-call medians; ~50 min at this size)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
    .venv/bin/python -m bench.validation.force_scale_prepare_cost \
    --n 16384 --leaf-size 16 --order 8 --repeats 5 --eps 2e-7 \
    --json-out bench/results/validation/force_scale_prepare_cost_n16384_p8.json

# what a pair policy costs the traversal in memory (one subprocess per arm --
# peak_bytes_in_use is process-cumulative and cannot be reset)
eval $(.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD \
    .venv/bin/python -m bench.validation.pair_policy_far_tag_memory \
    --n 1000000 --leaf-size 256 \
    --json-out bench/results/validation/pair_policy_far_tag_memory_1m.json
```

Bench flags added 2026-08-02, all of which exist because of a trap above:

| flag | why |
|---|---|
| `--arm mass_16b_est` | eq (16b) via the O(N) estimator — the production path, versus `mass_16b`'s O(N²) ceiling. Records per-config `fb_fidelity` against the exact sum. |
| `--seed 0,1,2` | cross-seed `median [min, max]`, joined on ladder position. Any trend claim needs ≥ 3. |
| `--max-pair-queue` / `--max-interactions-per-node` | pre-size the traversal, skipping the retry-recompile cycle. Must be given together; never round up. |
| `--reference-subsample` | subsample reference *targets* (all sources) so N=1e6 is 1e10 pairs, not 1e12. Rejects `--arm mass_16b`, and warns that p99.99 of K targets rests on K/1e4 particles. |

Artifacts in `bench/results/validation/` — `mac_dehnen_metric_p8.json` is the pre-fix
headline p=8 run (**VOID**, see the M2M note); `mac_postfix_headline_p8.json` the valid
one; `mac_plummer_p8_cap07.json` the `mac_theta_max=0.7` arm;
`force_scale_prepare_cost_n16384_p8.json` the Step 1 prepare-cost measurement;
`pair_policy_far_tag_memory_1m.json` the Step 3′ memory gate;
`mac_dehnen_eps_n16384_leaf16.json` the tight-ε run (item 1) and
`mac_n_ladder_leaf16_3seeds.json` the N-ladder (item 2), both from 2026-08-02.

## References

- Dehnen (2014), "A fast multipole method for stellar dynamics", arXiv:1405.2255 §5.
  §5.1 expansion centres, §5.2 cost scaling and θ_crit ≈ 0.46, §5.3 eqs (16a)/(16b),
  §5.4 the practical low-order estimate of `a_b`/`f_b`.
- `docs/adaptive_traversal_design.md` — the jaccpot/yggdrax ownership boundary.
- `docs/differentiable_fmm.md` — the fixed-topology contract the MAC lives behind.
- `docs/treecode_mac_stability.md` — prior in-repo finding that MAC bound-tightness
  governs secular heating; relevant if a multi-step drift test is added.

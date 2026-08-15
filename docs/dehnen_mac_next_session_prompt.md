# Session prompt — after the refactor merge: close out the Dehnen mass MAC

Paste this into a fresh session. Self-contained: it names every file, gate and hazard,
and the two measurements that close the work out.

Supersedes `docs/dehnen_mac_step3prime_prompt.md`, whose goal (get the criterion onto the
large-N lane and measure at 10⁶) is **done**.

---

## Situation

The method is finished and measured. What is left is two measurements and one
one-line fix.

**All of it is merged.** PR #56 (the two upward-M2M mass-loss fixes), PR #67 (the bulk of
the mass-dependent MAC) and the 38 commits that followed it — the O(N) `f_b` estimator,
Step 3′ (the criterion on the large-N lane), the leaf-size sweep and the N=10⁶
measurement — are all in `main`. Verified commit-by-commit on 2026-08-15: every one of
the branch's 39 commits except this file had an equivalent upstream, with the
`results/validation/` → `bench/results/validation/` move of `6839a6c` already applied to
it. `feat/dehnen-mass-dependent-mac` was reset onto `origin/main` rather than replaying
38 redundant patches.

So the branch now carries no library change at all, and the remaining work below is
measurement and bench work on top of `main`.

## Read first, in this order

1. `agent_guides/STYLE_GUIDE.md` and `agent_guides/NUMERICS_AND_JAX.md` — house rules.
   `NUMERICS_AND_JAX.md` is mandatory before touching `jaccpot/operators/`, `upward/`,
   `downward/`, `nearfield/`, `pallas/` or `distributed/`.
2. `ARCHITECTURE.md` and `docs/refactor_audit_2026-08.md` — where the refactor is going.
   **The user's explicit constraint is not to interfere with it.** Prefer measurement and
   bench work over library edits; when a library edit is unavoidable, keep it surgical and
   say so in the PR.
3. `docs/dehnen_mass_mac_status_and_plan.md` — the full status. **Read all the traps
   before running anything.** Items 1, 1b, 1c, 2, 2b, 3 and 4 are closed; the open work is
   the two subsections after item 4.

## Step 0 — confirm the tree is current before measuring anything

The rebase is **done** (2026-08-15). If you are picking this up later, check you are on
top of `main` before trusting a number from this worktree:

```bash
cd /export/home/tbuck/jaccpot-mac-wt
git fetch origin
git rev-list --left-right --count origin/main...HEAD   # left = behind; want 0
```

Two conventions `main` added that this branch already had to meet once (`4f30a7b`, now
upstream as `9c2e5df`) and must keep meeting:

- **pydoclint runs on commit.** numpy style, `name : type` on its own line. One argument
  per line — `theta_floor, theta_max` on one line reads as a single argument named
  `theta_floor, theta_max`.
- **Compile-bound tests are marked `slow`.** Anything that builds a solver or runs a full
  prepare belongs behind `-m "not slow"`; leaving them unmarked timed out the
  version-compatibility matrix once already.

Then confirm the tree is sound before trusting any number from it:

```bash
PYTHONPATH=$PWD JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  /export/home/tbuck/jaccpot/.venv/bin/python -m pytest tests/unit tests/characterization -q
```

**Everything runs with `PYTHONPATH=/export/home/tbuck/jaccpot-mac-wt`.** The venv's
editable install points at the *other* checkout, so without it you silently measure the
wrong code. Pick GPUs with `autocvd`, always.

## The open work

### 1. Leaf sweep at N = 10⁶ — the measurement that decides the production figure

N=10⁶ is measured (`bench/results/validation/mac_1e6_leaf256_lane.json`) and it **does not
reproduce** the N=10⁵ / leaf-256 headline: p99.99 is 4.5–4.9× for eq (16a) and
10.8–11.9× for eq (16b), against **21–41×** at N=10⁵. The criterion is *cheaper* than
fixed θ there (12–31 % fewer interactions), so the cost half of the bar passes and the
tail half does not.

That is item 2b's mechanism, not a contradiction of it: the advantage is controlled by
**tree depth**, and holding leaf 256 fixed from 10⁵ → 10⁶ takes the tree from 390 to 7813
nodes, ~4½ levels deeper. **So the next measurement is a leaf sweep at N=10⁶ — 512, 1024,
2048 — not another N point.** If a larger leaf restores the shallow-tree regime at 10⁶,
that is the production figure the paper should quote; if it does not, the honest headline
is the 10⁶ number and the 10⁵ one must be labelled as a shallow-tree result.

Do **not** quote the N=10⁵ / leaf-256 figure as the production number until this is
settled.

Hazards specific to this run, all already paid for once:

- **fp32 median saturation.** At N=10⁶ the δa/f median floors at ≈1.9–2.4×10⁻⁶ (fp32
  summation error over a 10⁶ near field, ~√N·1.2e-7). The fixed arm's median is
  *non-monotone in θ* below that, which is impossible for truncation error and is the
  tell. **Distrust any matched median below ~3×10⁻⁶** — it is matching round-off, and one
  such row already reported p99.99 = 0.57 at work 2.02, an outlier in every column at once.
- **Retries fired on 15 of 15 configs, up to 16 attempts**, which is most of why that
  sweep took 1½ h instead of ~20 min. Converged caps for N=10⁶ / leaf 256 on the lane:
  `max_pair_queue=690804`, `max_interactions_per_node=1024`. Expect to re-size per leaf —
  those differ from leaf 256's N=10⁵ values (16384 / 8192) by 42× *up* on the queue and
  8× *down* on the interaction capacity. An oversized interaction capacity OOMs; an
  oversized queue is merely wasteful.

### 2. N = 10⁷ — blocker identified, not guessed

The 10⁷ census OOMed on a 4.77 GiB allocation inside `_dual_tree_build_raw` — the
*monolithic* dual-tree walk, on a lane whose purpose is to avoid it.

Cause: `allow_split_build` falls back to
`_streamed_minimum_memory_gpu_default_split_build`, computed in `__init__` from
`memory_objective == "minimum_memory"` and `streamed_far_pairs` — **before**
`_apply_large_n_gpu_production_contract` coerces those very fields. On
`preset="large_n_gpu"` the predicate therefore reads `memory_objective="balanced"`, comes
out `False`, and the preset silently runs the monolithic build it exists to avoid.

Both bench harnesses now pass `prepare_stage_memory_split_enabled=True` explicitly, so
the smaller runs are unaffected. For 10⁷: re-run the census with the split build on (now
the harness default) and size the caps fresh.

**The one-line fix — recompute the predicate after the coercions — belongs in its own
PR, not this one.** It flips the default build path for every `large_n_gpu` user, trading
extra prepare work for a lower peak, and that is a performance change that deserves its
own measurement. Given the refactor in flight, coordinate before landing it.

### 3. Then: PR and stop

Once the leaf sweep lands, the work is complete enough to submit. Keep the PR
measurement-and-bench-heavy and library-light so it does not collide with the refactor —
which is easy now, since the rebase left the branch carrying only measurements and docs.

## Standing hazards — the ones that have actually cost runs

- **Match on the statistic the claim is about.** Dehnen §5.3 claims a reduced tail at
  comparable **median**. `--match-on median`. Matching p90 once turned "rms 2.0–23.6×"
  into "parity".
- **A far-pair census before any sweep.** Prepare-only, count `interactions.sources >= 0`,
  skip the reference and the evaluation. If the fixed arm's far count is not in the
  millions at your N, the grid is not measuring the criterion — it is measuring direct
  summation, and it will look merely *slow* rather than wrong. Leaf 256 has no far field
  below N ≈ 10⁶; "≥ 128 leaves" is necessary and nowhere near sufficient.
- **Hold `leaf_size` fixed across an N-ladder**, or the ladder crosses from "no far field"
  to "real far field" partway up and reports that crossing as an N-scaling trend.
- **Never round traversal caps up.** Buffers are `num_nodes × interaction_capacity`.
- **`python -u`, never pipe a long run through `tail`.** SIGTERM discards block-buffered
  stdout, so a killed run reports nothing; `tail -N` has already silently discarded the
  counts a diagnostic existed to produce.
- **Be sceptical of favourable numbers.** This project has produced several that were
  artifacts: a mass-loss bug that read as truncation error, an interpolation across a
  discontinuity that reported 34× and 2.9×10⁷×, a matching statistic that inverted a
  conclusion, and an fp32 floor that produced a p99.99 below 1. Check that matched rows
  lie inside both arms' measured ranges, that the far field is non-trivial, and that the
  median is above the precision floor, before believing a ratio.

## What is already settled — do not re-measure

| | |
|---|---|
| eqs (12)/(13)/(15)/(16a) transcription | proven against numpy references |
| Dehnen's ε = 2×10⁻⁷ | benefit holds and **grows** toward it |
| p = 8 and p = 10 | both hold |
| eq (16b) + O(N) `f_b` estimator | 2–3× eq (16a)'s tail advantage, at less work |
| Step 3′ — criterion on the large-N lane | done, lane engages, verified at 10⁶ |
| what controls the advantage | **tree depth**, not N |
| `mac_theta_max` | not needed; no deviation from the paper to disclose |
| per-node effective θ (`dehnen_theta`) | refuted, structurally — a sum cannot represent a product |
| cartesian basis ~1.8×10⁻¹ error | pre-existing, filed as issue #62, off this path |

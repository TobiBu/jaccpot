# Session prompt — after the refactor merge: close out the Dehnen mass MAC

Paste this into a fresh session. Self-contained: it names every file, gate and hazard,
and the two measurements that close the work out.

Supersedes `docs/dehnen_mac_step3prime_prompt.md`, whose goal (get the criterion onto the
large-N lane and measure at 10⁶) is **done**.

---

## Situation

The method is finished and measured. **The leaf sweep that decided the production figure
is done (2026-08-16) — leaf 1024 at N=10⁶, p99.99 43× [31, 71] for eq (16b) at 1.03×
work, 3 seeds.** What is left is N=10⁷ and the one-line fix in front of it.

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

### 1. Leaf sweep at N = 10⁶ — **DONE (2026-08-16), and it settles the production figure**

The N=10⁶ / leaf-256 record did not reproduce the N=10⁵ headline (p99.99 4.5–4.9× for
eq 16a and 10.8–11.9× for eq 16b, against 21–41×). Item 2b's mechanism said that was tree
*depth*, so the test was a leaf sweep rather than another N point. It was run at 512, 1024
and 2048, plus a re-measured leaf 256, 3 seeds each.

**A larger leaf does restore it.** At a common absolute matched median of 10⁻⁵, 3 seeds,
`median [min, max]`, p99.99 goes 2.7× → 7.0× → **21.8× [13.9, 34.4]** for eq (16a) and
5.3× → 15.2× → **43.1× [31.0, 71.2]** for eq (16b) across leaf 256 → 512 → 1024, while
work falls 1.36× → 1.21× → 1.10× (`fixed/mass`, so the criterion does *less* work).
Monotone in every seed with non-overlapping adjacent bands. **Leaf 1024 at N=10⁶ is the
production configuration, and the leaf size must be quoted with the number.** Full detail,
including why leaf 2048 is not measurable in fp32, is in the leaf-sweep section of
`dehnen_mass_mac_status_and_plan.md`.

Hazards this run paid for, kept because they will recur:

- **fp32 median saturation, and the floor is not a constant.** It **rises with leaf size**
  — measured 1.86, 2.83, 4.67, 6.85 ×10⁻⁶ at leaf 256, 512, 1024, 2048 — because it is
  summation round-off over the near field and a bigger leaf means more near pairs per
  particle. So the old rule "distrust any matched median below ~3×10⁻⁶" is a **leaf-256**
  rule; per leaf it is roughly **2× that leaf's own measured floor**. The tell is a median
  that fails to rise with its knob, which no truncation error can do.
  `bench/validation/leaf_sweep_common_target.py` measures the floor, flags it, and refuses
  targets below it instead of interpolating through them. At leaf 2048 the floor has
  closed on the divergence guard and the ratio swings 13× → 50× → 112× over a 1.6× change
  of target — an artifact, not a result.
- **Retries fired on 15 of 15 configs, up to 16 attempts**, which is most of why the
  original sweep took 1½ h instead of ~20 min. Converged caps for N=10⁶ / leaf 256 on the
  lane: `max_pair_queue=690804`, `max_interactions_per_node=1024`. Expect to re-size per
  leaf — those differ from leaf 256's N=10⁵ values (16384 / 8192) by 42× *up* on the queue
  and 8× *down* on the interaction capacity. An oversized interaction capacity OOMs; an
  oversized queue is merely wasteful.
- **Pin the GPU per run; do not let each run call `autocvd` for itself.** Two simultaneous
  launches get handed the same least-used card. Ask once for as many as you need
  (`autocvd -l -n 4 -o -q`) and set `CUDA_VISIBLE_DEVICES` per run. A co-tenant arriving
  mid-run still put one 40 GB card at 37.5/40 and forced a restart.

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

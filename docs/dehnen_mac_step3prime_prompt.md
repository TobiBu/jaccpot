# Session prompt — Step 3′: get the Dehnen mass MAC into the large-N lane

Paste this into a fresh session. Self-contained: it names every file, gate and hazard,
and the measurement that closes it out.

---

## Goal

Make `mac_type="dehnen_error"` (Dehnen 2014 §5's mass-dependent MAC) work on the
**large-N GPU lane**, then measure it at **N = 10⁶ and 10⁷** — Dehnen's own range. Today
the lane refuses the criterion outright and the generic lane tops out around 10⁵.

This is the **last substantive gap** to matching Dehnen's implementation. Everything about
the method itself — eqs (12)/(13)/(15)/(16a), the §5.4 prepass, eq (16b) with an O(N) `f_b`
estimator, p = 8 and 10, ε down to 1e-7, zero softening, δa/f, rms + p99.99 — is already
implemented, tested and measured up to N=10⁵. It is engineering, not a research question.

## Read first

- `docs/dehnen_mass_mac_status_and_plan.md` — the full status. **Read the traps section
  (all 11) before running anything.** Items 1–5 are done; this prompt is its "Step 3′".
- Branch `feat/dehnen-mass-dependent-mac`, worktree `/export/home/tbuck/jaccpot-mac-wt`.
  Run everything with `PYTHONPATH=/export/home/tbuck/jaccpot-mac-wt` — the venv's editable
  install points at the *other* checkout, so without it you silently test the wrong code.

## Why this is worth doing (and why it should work)

The 2026-08-03/04 leaf-size sweep found the criterion's advantage is controlled by **tree
depth, not N**, and it is *best* on shallow trees — at N=10⁵ p99.99 goes 5× → 7× → 16× →
35× → 31× across leaf 16 → 32 → 64 → 128 → 256, while the work ratio falls 1.17× → 1.00×.
Production uses **leaf 256**, the best case, and leaf 256 only becomes meaningful at
N ≳ 10⁵ (trap 11). So the large-N regime is exactly where the criterion should look
strongest — which is also why measuring it there matters rather than extrapolating.

## Do it in this order. The first phase is a correctness blocker, not a nicety.

### Phase 0 — key the interaction cache on the pair policy (**do this first**)

`_interaction_cache_key` (`jaccpot/runtime/_interaction_cache.py:1131`) takes **no**
`pair_policy` argument. It hashes topology, `theta`, `mac_type`, `dehnen_radius_scale`,
basis, centre mode, caps and refinement — nothing about the criterion.

That is safe *today* only by accident: the cache is per-solver-instance and `eps` is fixed
on a live solver. But two `dehnen_error` solvers at different `eps` **already hash
identically**, because `_base_mac_type()` returns `"dehnen"` and paper mode pins θ=1.0.
Relax the lane gates without fixing this and a geometrically-built interaction list can be
served to a criterion-driven request: the accept mask is silently wrong, the run is
*faster*, and per trap 6 **no cost measurement can detect it**.

Either add a policy identity to the key (`eps`, `mac_force_scale_mode`, geometry mode, and
a hash of the force-scale array — it *is* part of the criterion) or refuse to cache when a
policy is installed. Caching is a perf optimisation; correctness first.

**Test that must bite before the fix:** two solvers identical except `adaptive_eps`
(e.g. 1e-5 vs 1e-7) must not produce the same cache key, and must not share an entry.
Write it, watch it fail, then fix.

### Phase 1 — let the split/streamed build take a policy

- `_can_split_dual_tree_build` (`_interaction_cache.py:439`) returns false when
  `pair_policy is not None or policy_state is not None`. Its own docstring calls the path
  "intentionally narrow". Drop those two conditions.
- Thread `pair_policy` / `policy_state` into `_build_dual_tree_artifacts_split` and its
  three yggdrax calls: `build_compact_far_pairs_and_leaf_neighbor_lists`,
  `build_interactions_and_neighbors_split`, `build_leaf_neighbor_lists`. **All three
  already accept `pair_policy`** — yggdrax exposes it on eight entry points in
  `interactions.py`; jaccpot simply never passes it.

**Verify:** `tests/unit/runtime/test_far_near_partition.py` (the only test that can see a
dropped pair — for every target leaf, ancestor far sources ∪ near leaves ∪ own leaf must
cover every source exactly once) must pass *with the split build active and a policy
installed*. Also assert the accept mask is **bit-identical** to the non-split build on the
same inputs; a differing mask means the two builds disagree about the criterion.

### Phase 2 — let the large-N lane accept the criterion

- `can_use_large_n_prepare_path` (`_large_n_pipeline.py:~1933`) declines on
  `_uses_paper_style_force_scale()` and raises a clear `RuntimeError` when
  `runtime_path="large_n"` was requested explicitly. Remove the decline once the lane can
  actually carry it — and keep the explicit-request error until then, it is doing its job.
- `_large_n_pipeline.py` contains **zero** occurrences of `pair_policy`. This is new
  plumbing, not a flag flip. Thread it through `prepare_large_n_state` (`:321`).
- Run the force-scale prepass inside `prepare_large_n_state`. **Prefer
  `mac_force_scale_mode="paper_fb"`** (eq 16b): its prepass needs only a *traversal*, no
  low-order FMM evaluation, so it is cheaper than the (16a) prepass — measured 17.05 s vs
  22.24 s cold at N=4096/p=8 — and (16b) is the better criterion anyway (roughly doubles
  (16a)'s tail advantage). `paper_fb_cached` for steady state.

## Then measure

```bash
cd /export/home/tbuck/jaccpot-mac-wt
eval $(/export/home/tbuck/jaccpot/.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD setsid nohup \
  /export/home/tbuck/jaccpot/.venv/bin/python -u -m bench.validation.mac_error_distribution \
    --n 1000000 --leaf-size 256 --order 8 --distribution plummer \
    --theta <from a census> --eps 1e-3,3e-4,1e-4,3e-5,1e-5 \
    --softening 0 --metric dehnen --match-on median \
    --reference-subsample 100000 --reference-block 64 \
    --min-far-pairs 5000 --max-p9999 1.0 \
    --seed 0,1,2 --arm fixed,mass,mass_16b_est \
    --json-out bench/results/validation/mac_1e6_leaf256.json > /tmp/1e6.log 2>&1 < /dev/null &
```

Note the ε grid: at leaf 256 the criterion's usable ε is ~**1e-3…1e-5**, roughly two
decades looser than leaf 16 wants. Do not reuse a small-leaf grid.

**Census before committing to it** (trap 11, and it has already cost this project two
wasted multi-hour runs): prepare-only, count `interactions.sources >= 0`, no reference and
no evaluation. If the fixed arm's far count is not healthy at your θ, the grid measures
nothing. At N=10⁵/leaf 256, θ=0.30 accepted **zero** far pairs.

### Acceptance

- The lane **actually engages** — assert `get_runtime_diagnostics()["large_n_path_declined_reason"]`
  is `None`. A silent fallback to the generic lane would reproduce the old numbers and look
  like success.
- At leaf 256, matched median: p99.99 ≳ 20× at ≤ 1.05× interaction work, i.e. the N=10⁵
  result holds or strengthens. (N=10⁵/leaf 256 gave p99.99 21–41× at 1.00×.)
- `tests/unit/runtime/` stays green — 110 cases today.

## Traps specific to this work

1. **The memory gate is already measured and it clears.** A pair policy costs **+977 MiB
   at 1M** (split build; +969 monolithic), against an 11.07 GB reverse peak on 40 GB.
   Harness: `bench/validation/pair_policy_far_tag_memory.py`, artifact
   `bench/results/validation/pair_policy_far_tag_memory_1m.json`. Two corrections to the old
   concern: the cost is *not* specific to the streamed lane, and it is *not* mostly the
   far-tag buffer (244 MiB of 970) — most of it is `_resolve_pair_actions` evaluating the
   policy twice, forward and reverse. So "make tag storage conditional" recovers under a
   third of it, and nothing in the streamed build, which allocates `far_tags`
   unconditionally.
2. **`peak_bytes_in_use` is process-cumulative and cannot be reset.** Measure one arm per
   subprocess or the delta is always exactly zero — which looks like a clean null result.
3. **The large-N lane runs fp32; the criterion was validated in fp64.** Unverified and
   worth checking early: eq (15)/(16a) compare an error estimate against `ε · s`, and at
   ε=2e-7 with a small `s` that product gets very small. Check for underflow or precision
   collapse in the acceptance comparison before trusting any fp32 accept mask. If it bites,
   evaluate the comparison in fp32 with a rescaled threshold, or promote just that
   reduction.
4. **Do not reuse converged traversal caps across configurations.** Measured:
   `max_pair_queue` converged to **16384** at leaf 256/N=1e5, **262144** at leaf 64/N=1e5,
   and **1048320** at leaf 16/N=65536. Oversized *interaction* capacity OOMs (buffers are
   `num_nodes × interaction_capacity`); an oversized queue is merely wasteful.
5. **`--reference-subsample` costs tail resolution.** p99.99 of K targets rests on K/10⁴
   particles, so 10⁴ targets leaves Dehnen's headline statistic resting on **one**
   particle. Use ≥ 10⁵. The flag rejects `--arm mass_16b` (it needs exact `f_b` for every
   particle) — use `mass_16b_est`, which needs no reference at all.
6. **Zero softening is fine** — verified numerically identical to 1e-6 at N=2048. Re-check
   at 10⁶, where minimum separations shrink.
7. **Report rms and p99.99, matched on median, and attach the leaf size to every claim.**
   At leaf 16 the criterion is *worse* than fixed θ on p99 (0.76–0.83×); the tail ratio
   alone hides that. `compare_arms` reports rms/p99.99 ratios directly now.

## Environment

```bash
# tests (CPU is fine and faster to iterate)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD \
  /export/home/tbuck/jaccpot/.venv/bin/python -m pytest tests/unit/runtime/ -q

# GPU: always pick the device with autocvd, the box is shared and often full
eval $(/export/home/tbuck/jaccpot/.venv/bin/autocvd -l -q)
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD JAX_ENABLE_X64=1 ...

# long runs: python -u, redirect to a file, never pipe through `tail`
# (SIGTERM discards block-buffered stdout -- this has cost whole runs their output)
```

`black --fast` + `isort` to match pre-commit.ci. No `gh` CLI on this box — there is a
token in `~/.config/gh/hosts.yml`, use the REST API.

# Prompt: fix the jaccpot issues found while measuring paper tranche 1

> Paste into a fresh session with the `jaccpot` repo checked out. Use a separate
> worktree. The full evidence for every item is in
> `docs/jaccpot_enhancements_from_paper_tranche1.md` on `paper/jaccpot-i-tranche1` --
> read it first; every number below was measured on the 8x A100-PCIE-40GB host and
> does not need re-deriving before you start.

## Context

Producing figures 01-07 and 13 for the Jaccpot I paper surfaced 14 library issues.
None were fixed then, because that tranche deliberately changed no forward-force
code. Several of them make the paper's own numbers worse than the library deserves:
the flagship preset is 139x slower than the alternative on a GPU, and the
acceleration path is overhead-bound across three decades of N.

The unifying failure mode, and the thing to keep in mind throughout: **these knobs do
not fail, they silently change what is being measured.** A wrong number that looks
plausible cost more time in that tranche than every crash combined.

## Ground rules

- **Forward numerics are frozen unless an item says otherwise.** The characterization
  goldens (`tests/characterization`) and `tests/unit/test_public_api_surface.py` stay
  green throughout. If a fix legitimately changes a force value, stop and say so
  before regoldening.
- Work on a dedicated branch off `main`; fetch first. Small verified commits.
- GPU work uses `autocvd` before importing jax. Match CI with `JAX_ENABLE_X64=1`.
  Run tests with `-n 4` and `XLA_PYTHON_CLIENT_MEM_FRACTION=.12` --
  `pyproject.toml`'s `-n auto` spawns ~72 workers on this box and OOMs the card.
- **Every performance claim needs a before/after measurement on an idle GPU**, with
  the N, preset, leaf size and basis stated. Check `nvidia-smi` first; this host is
  shared and a busy card will silently halve your numbers.
- Beware timing anything on `preset="accurate"` on a GPU (that is item 1) and beware
  supplying a partial `traversal_config` (item 4) -- both will corrupt a measurement
  of anything else.

## Tasks, in the order they pay off

### 1. `accurate` is 139x slower than `large_n_gpu` on a GPU (E1)

Measured, idle A100, N=16384/leaf=64/p=4/basis real, evaluate on a prebuilt tree:
`accurate` 27.3 s vs `large_n_gpu` 197 ms. Steady state, not compile. The same
problem on this host's CPU is 861 ms, so `accurate` on an A100 is ~30x slower than on
the CPU beside it.

Find out why `accurate` does not reach the radix fast lane on GPU. Then pick one and
justify it: route it through the lane, make preset resolution device-aware, or emit a
one-time warning naming `large_n_gpu`. Do not leave it silent. If the accuracy
presets genuinely need the slow path, that is a documentation fix plus a warning, not
a no-op.

Deliverable: before/after table at N in {4096, 16384, 65536}, both presets, and a
statement of whether any force value moved.

### 2. The acceleration path is overhead-bound (E2)

Measured, `large_n_gpu`/p=3/theta=0.77/leaf=128, evaluate on a prebuilt tree:
potential 4.5 ms at N=2048 rising to 33 ms at N=65536, while acceleration sits at
211-269 ms across the same range and ~290 ms at N=881744. Fitted over
N=27554..881744 the acceleration path gives alpha=0.20 at R^2=0.89 -- a constant, not
a complexity.

Three components plus an expansion gradient should be 3-6x the potential, not a
~210 ms floor. Profile the two paths at fixed N and find the constant. Suspects worth
eliminating first: an allocation or scatter whose size is independent of N, and a host
round-trip per call.

Deliverable: the constant identified and removed or explained, with the
potential-vs-acceleration ratio re-measured across the same N ladder. This single
number gates most of the paper's performance story.

### 3. `differentiable_accelerations` cannot be jit-wrapped, and hangs rather than raising (E3)

An outer `jax.jit` over the whole call did not finish compiling after 18 minutes at
**N=256** (85% CPU, 0% GPU); at N=16384 XLA spent 3m21s on
`jit__accumulate_m2l_fullbatch` alone and was unfinished after 30 minutes. Lowering N
does not help, so it is the unrolled order-4 rotation cascade, not the problem size.
`docs/differentiable_fmm_design.md` documents this for *large* N; it is
unconditional.

Two deliverables:

- **A jittable step seam.** `prepare_state` once, then a jitted function taking
  positions and masses as arguments, so the compile is paid once and amortised over
  repeats. This is what figure 12 needs and what anyone training through the force
  needs.
- **Fail fast.** If the call cannot be traced, raise with the reason rather than
  handing XLA an unbounded compile. A hang defeats every `try/except` fallback,
  including the one already in `bench/differentiability/autodiff_overhead.py`.

Then unblock figure 12: `python -m bench.differentiability.autodiff_overhead` should
produce jitted rows (`mode="jit"`) across N in {4096, 16384, 65536}, and
`docs/fig12_autodiff_overhead_blocked.md` should be deleted or rewritten as solved.

### 4. A partial `traversal_config` discards the preset's other tuning (E4)

Measured at N=65536 on `large_n_gpu`: passing an explicit `traversal_config` with
`max_pair_queue` set to **the value the preset already uses** took per-step time from
1085 ms to ~3200 ms and the instrumented fraction from 76% to 27%, because supplying
the object also replaces `process_block` and the interaction/neighbour caps.

Merge a user-supplied `traversal_config` onto the preset's resolved values
field-by-field, or reject a partial one outright. Add a test that overriding a single
field leaves the others at the preset's values.

### 5. Instrument the downward pass; expose the counter hierarchy (E5, E6)

`refresh_dual_m2l_compute_seconds` and `refresh_dual_l2l_compute_seconds` stay zero on
the strict non-fused refresh path -- no `dual_*` counter fires at all -- so ~30% of
per-step time is unattributable and the M2L stage cannot be reported. Cause looks like
`_record_timed_array` in `runtime/kernels/core.py` being a no-op when
`timing_recorder is None`, with the recorder not threaded through that route.

Separately, the counters are a **hierarchy**: `refresh_nearfield_seconds` is the sum of
`refresh_nearfield_*`, and `refresh_tree_upward_seconds` the sum of
`refresh_upward_*`. Measured: nearfield 80.33 ms against children summing to 79.4 ms.
Anything summing "all counters" double-counts, which figure 06 did until caught.

Deliverables: M2L and L2L separately timed on the refresh path, and the parent/child
structure exposed (nested dict, or a documented naming rule) so a consumer can sum a
partition. Then rerun `python -m bench.scaling.stage_breakdown` -- the attributed
fraction should rise well above today's 64-70%.

### 6. `upward_geometry` is 56-59% of a per-step refresh (E7)

Measured at N=32768/65536/131072: 56.7%, 59.2%, 55.6% of per-step time, against 2.1%
for the P2M and M2M expansions it feeds -- roughly 25x. Topology is reused across a
refresh, so establish what it recomputes that the frozen topology already fixes, and
cache that. Likely the largest single win available.

### 7. Capacity: pair queue and the N ~ 10^6 OOMs (E8, E9)

- `large_n_gpu`/leaf 256: N >= 262144 raises `Pair queue capacity exceeded`. Scale the
  default with particle count, or catch and re-plan with a larger queue. Note raising
  it manually is contaminated by item 4 until that is fixed.
- `preset="accurate"`/leaf 32: `prepare_state` at N=1048576 dies on an **8.00 GiB**
  single allocation on a 40 GB card. `large_n_gpu`/leaf 64 dies on 4.00 GiB. Chunk it,
  or raise a diagnostic naming the buffer and a preset that would fit.

### 8. Documentation and hygiene (E10, E11, E12, E14)

- README: GPU results are reproducible to a few ulps, not to the bit. Measured: 0 of 7
  repeat runs bit-identical, worst 3.8 eps, from atomic scatter-add. Point at
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true` and note it is far too slow for a suite.
  See `tests/unit/runtime/_reproducibility.py`.
- `basis="complex"` and `basis="solidfmm"` are **bit-identical** (measured, N=2048/p=4,
  max diff 0.0). Alias them explicitly or document that `complex` selects the solidfmm
  path, so the argument is not silently inert.
- `basis="cartesian"` is ~1.8e-1 rel-L2 independent of order -- a divergent-series
  signature, with solidfmm at 8.1e-5 on the same configuration. Fix, or mark
  experimental so it cannot be picked by accident. It is the only reason the
  characterization anchor carries a 0.35 tolerance.
- `bench/bench_jaxfmm_paper_compare.py`: when `--runner` explicitly requests a runner
  that is unavailable, exit non-zero instead of writing `status=error` rows. Its jaxfmm
  arm was dead for months because a failed import produced plausible-looking CSV.

### 9. Land the MAC criterion fixes on `main` (E13)

`fix/dehnen-mass-mac` (local, never pushed) carries four fixes to the Dehnen
mass-dependent criterion, `mac_theta_max`, the cached force-scale prepass, and 94 unit
tests. `main` still has D1 (eq 12 double-counting), D3 (COM vs SES geometry) and D4
(eq 16a asymmetry silently dropping leaf-leaf pairs from **both** M2L and P2P)
unfixed. It is merged into `paper/jaccpot-i-tranche1` and green there: 94 passed on
A100.

Push it and open a PR to `main`. Note it was branched off
`chore/pallas-call-backend-port`, so expect conflict resolution in the runtime files
if you rebase onto `main`.

## Done criteria

- Items 1-3 each have a before/after measurement on an idle GPU, with N, preset, leaf
  size and basis stated, and an explicit statement of whether any force value moved.
- `tests/characterization` and `tests/unit/test_public_api_surface.py` green
  throughout; `tests/unit/runtime` green on GPU (94 tests).
- `python -m bench.scaling.stage_breakdown` attributes materially more than today's
  64-70% of per-step wall clock.
- `python -m bench.differentiability.autodiff_overhead` produces jitted rows.
- Figures 04, 06, 07 and 12 regenerated on the paper branch afterwards, since items 1,
  2, 5 and 6 all change their numbers. `examples/jaccpot_paper/run_notebooks.py` then
  `export_to_paper_repo.py` handles that; the provenance table will show the new
  commit and clear the current `Dirty: yes` flags.

## Do not

- Do not tune a preset or a config to make a figure look better. Every number in
  `docs/jaccpot_enhancements_from_paper_tranche1.md` is reported as measured, including
  the ones that are unflattering.
- Do not time anything on `accurate` on a GPU, or with a partial `traversal_config`,
  until items 1 and 4 are done.
- Do not add a bit-equality assertion to a test that runs on GPU; see item 8.

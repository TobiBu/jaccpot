# Cleanup 2026-10: one basis, one fast lane, a smaller suite

The record for removing the code that seven fused-lane rounds left behind, and for trimming
the test suite. Each phase is one PR; its row here says what went, what replaced it, and
which gates it passed. Tests deleted under the `CLAUDE.md` exception are listed per phase
with their tag: (a) subject deleted, (b) merged into a named test, (c) duplicate of a named
owner.

## Why

On `main` at 6cca378 (2026-10-06):
- 105k lines in 130 modules, and 179 `JACCPOT_*` knobs. Most knobs select an older
  kernel or lane that a later round superseded.
- 2,054 unit tests and 204 test files. Every unit test ran about three times per push, so
  a push cost about 5.2 h of runner time. `test-mac-runtime` was cancelled at its cap on
  #367.

## Decisions (maintainer, 2026-10-07)

- **Bases:** remove the complex SH basis and the Cartesian basis; real SH is the only
  basis. Jerk and the time-derivative tower are ported to the real basis *before* the
  complex code goes, and validated against it.
- **Superseded paths:** remove the octree execution backend, the grouped / class-major
  far field, M2L autotune and adaptive sizing, the legacy strict APIs, the superseded GPU
  kernel variants and the treecode lane (including distributed `local_walk="treecode"`).
- **Fused lane:** it becomes the library default. Pre-Ampere GPUs keep the large-N lane,
  so its rectangle / target-block payloads stay, and so do all CPU pure-JAX fallbacks.
- **Protected throughout:**
  - the fused single-GPU lane;
  - both multi-GPU lanes (`distributed/fmm.py` and `distributed/fused.py` + `cross.py`
    + `rollout.py`);
  - the block-step lane (`BlockStepFMM`, `mutual/`, `nornax_adapter.py`);
  - every gradient path.
- **yggdrax:** loses only what jaccpot's removals orphan, plus distributed helpers no
  consumer uses. `applications/` and the kdtree stay.

## Phases

| Phase | PR | What | Gates | State |
| --- | --- | --- | --- | --- |
| P0 | #369 | CI: each test once per push; shard partition checked; this record | CI 16/16 | open |

### P0: CI runs each test once

Before this phase, the unit tests ran in four jobs:
- `test-full` (with coverage);
- `test-mac-runtime` (`tests/unit/runtime` again);
- `test-runtime-typecheck` (all of `tests/unit` again, under jaxtyping + beartype);
- `test-smoke` (every non-slow test again, on 3.12, in three shards).

The changes:
- **One full run.** `test-full` is now the only full run. Runtime type checks are switched
  on for its three `tests/unit` shards, so the protection the typecheck job existed for is
  kept on every unit test, in the same pass.
- **Runtime directory split in two.** `tests/unit/runtime` is now two shards of that
  matrix: the Dehnen-MAC criterion family, and the rest.
- **Python-floor job.** `test-smoke` became `test-py-floor`: one 3.12 job that runs the
  goldens and the public-API and Odisseo-contract tests.
- **One place defines the shards.** `.github/scripts/test_shards.py` holds every job's
  selection, and `test-partition` collects all of them. Measured on this branch
  (collection only, forced host devices): 2,545 tests in the default collection, 11
  shards, no overlap, no gap.
- **Concurrency.** A new push to a PR cancels that PR's run in flight; pushes to `main`
  always finish.
- **Durations in the logs.** `--durations=60` keeps per-test times in each shard's log.
  The shard split and `tests/slow_tests.txt` are tuned from those times.
- **Grid trimming is now opt-in.** `tests/unit/_typecheck_budget.trim` used to cut eight
  Pallas grids to one case whenever type checks were on. It now trims only under
  `JACCPOT_TEST_TRIM_GRIDS=1`, a local opt-in that no CI job sets. Otherwise turning the
  checks on in `test-full` would have dropped those cases from CI.

**Measured on the PR's CI run (16/16 green):** runner time about 3.2 h per push, down from
about 5.2 h. The longest job is 36 min (`test-full (unit)`, now with type checks), down
from 66.5 min (`test-runtime-typecheck`).

| job | min |
| --- | --- |
| test-full (unit) | 35.9 |
| test-full (integration) | 26.9 |
| test-full (unit-runtime) | 20.7 |
| test-full (unit-runtime-mac) | 19.2 |
| test-cross-repo-nornax | 19.5 |
| test-distributed-mutual | 15.3 |
| test-distributed-tier | 13.8 |
| test-distributed-criterion | 7.4 |
| test-full (characterization) | 7.4 |
| test-py-floor | 6.9 |
| test-full (mutual-static-device) | 6.5 |
| test-cross-repo-nornax-distributed | 3.9 |
| test-partition | 2.7 |
| benchmark-guard, lint | 2.4, 2.1 |


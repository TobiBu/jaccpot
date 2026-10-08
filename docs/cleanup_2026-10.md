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
| P1 | | Safety net: Odisseo contract test, lane goldens, inventory, gradient twins on real, GPU pins | CPU suite | in progress |

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

### P1: safety net

**Downstream contract.** `tests/unit/test_downstream_contract.py` imports every name
Odisseo's production branch imports and constructs every config object with the
keywords it passes. On its first run it found a live break: Odisseo's
`tests/test_strict_velocity_verlet_policy.py` imported `_velocity_verlet_state_update`
from `runtime._fmm_impl`, whose re-export had gone with the dead-import cleanup
(e7a2de3). Fixed in Odisseo #27, together with a stale assertion that the import error
had been hiding.

**Lane goldens.** `tests/characterization/test_lane_goldens.py` pins the large-N fast
lane, the fused `strict_run_v2` scan and the block-step lane. Each runs on CPU, at
clustered N = 512, against a direct-sum anchor.
- The fused scan runs fully fused on CPU, with no fallback. This answers the plan's open
  question: the fused scan does not need Pallas.
- The large-N lane is float32-only, so its gate is rel-L2 <= 1e-5. Measured: forcing
  `--xla_cpu_max_isa=AVX2` reproduces the golden bitwise; forcing `SSE4_2` moves it by
  1e-7 to 5e-7.
- **The first lane goldens recorded a bug, found two commits later and fixed here.**
  `_p2m_leaves_real` dropped the first leaf's multipole whenever its scan batch was
  wider than the leaf count, which the `large_n_gpu` preset (batch 2048) causes on any
  tree with fewer than 2048 leaves. It shows up on the pure-JAX real upward (CPU and
  pre-Ampere GPUs); the A100's Pallas P2M is not affected. The fix moves the large-N
  lane's direct-sum error on the golden case from 1.05e-3 to 3.9e-4. The two float32
  lane goldens were regenerated after it. The other 35 goldens did not move.

**Inventory** (`bench/cleanup_inventory.py`). It covers one full CPU run of
`tests/unit tests/integration tests/characterization` with per-test coverage contexts:
2,328 passed and 164 skipped. The one failure was a nornax cross-repo test, because the
local nornax checkout predates `block_kdk_rollout(rebuild_fn=...)`; CI uses nornax main
and passes. A test counts only if it ran the feature's body. Gatekeepers every prepare
calls (leading `if ...: return` guards, the dual-planner hint, autotune under static
sizing, the grouped budget predicates) do not count.

| family | tests | files | phase |
| --- | --- | --- | --- |
| complex basis | 423 | 52 | C (after J) |
| Cartesian basis | 134 | 17 | K |
| autotune | 60 | 8 | X4 |
| kernel variants | 58 | 10 | X5 |
| octree backend | 19 | 2 | X2 |
| legacy strict APIs | 18 | 2 | X4 |
| grouped / class-major | 17 | 6 | X3 |
| treecode | 16 | 2 | X1 |
| `_large_n_farfield` | 2 | 1 | X1 |
| dense interactions | 1 | 1 | K |
| non-fused strict loop | 1 | 1 | X6 |
| dual-downward planner | 0 | 0 | X6 (GPU-only; its gate is the GPU pins) |

**Gradient twins.** The four gradient edge-case tests that ran on the complex basis only
(near-coincident, rho == 0 degenerate azimuth, axis-aligned lattice, reorder) now also run
on real, and pass, as does a shallow real case of the FD-vs-AD positions grid.

**D1 dry run (force-real).** A test-only plugin resolved every complex request to the real
basis: the `"solidfmm"` / `"complex"` names, `ComplexSHBasis`, and an engine built
without `basis_impl`. Under it, the 52 complex-touching files ran 972 passed, 25 failed,
24 skipped. The failures, all expected:
- **19 for X3:** grouped / class-major / AABB-centre tests, which the real upward
  rejects (`center_mode='com' only`). These are the `cm_`/`pg_` mode goldens, the grouped
  cache and plateau tests, `test_nearfield_bucketed_matches_baseline` and the
  legacy-kwargs test, which switch grouped mode on.
- **3 for J:** accurate jerk with the far field engaged, and the two source-motion
  finite-difference tests. This confirms that accurate jerk does not work on the real
  basis today.
- **3 for C:** two tests that assert complex local dtypes, and
  `test_basis_object_is_accepted`, which asserts the public name `"complex"`.

So D1 flips only the implicit default (an engine without `basis_impl` runs real) and
warns on an explicit `"solidfmm"` / `"complex"`. Mapping those names to real waits for C,
after X3 and J have removed or ported the 22 tests that need them. No mass edit of the
~290 `"solidfmm"` call sites is needed before then.


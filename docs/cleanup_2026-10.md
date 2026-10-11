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
| P0 | #369 | CI: each test once per push; shard partition checked; this record | CI 16/16 | merged |
| P1 | #370 | Safety net: Odisseo contract test, lane goldens, inventory, gradient twins on real, GPU pins; leaf-P2M padding fix | CPU suite | merged |
| J | #371 | Real-basis jerk and time derivatives; exact real derivative tower | CPU suite | merged |
| X1 | #372 | Treecode walk (single-GPU + distributed), `jaccpot/experimental`, `_large_n_farfield` | CPU suite, distributed tier, GPU pins | merged |
| X2 | #373 | Octree execution backend | CPU suite, GPU pins | merged |
| D1 | #374 | Spherical-harmonic family without a basis object runs real | targeted CPU, GPU pins | merged |
| fix | #376 | The large-N lane's no-Pallas near field read the CSR lane's placeholder; two-card pins M1-M3 | targeted CPU, GPU pins (two-card: M1, M3) | merged |
| pins | #377 | GPU pins re-recorded at 15ceca4 after the softening kernels (#375) | A-vs-A bitwise | merged |
| D2 | #378 | The fused strict lane is the default; `large_n_gpu` builds `static_radix` | CPU suite, GPU defaults gate, pins, Odisseo G4 | merged |
| X3 | #380 | Grouped / class-major far field; AABB (non-COM) expansion centres | CPU suite, shard partition, GPU pins (S1-S5, M1, M3) | merged |
| X4 | #383 | M2L autotune, adaptive sizing, legacy strict APIs, `fixed_depth`, the strict cap-profile file; `runtime_path` no longer a lane switch | CPU suite, shard partition, CPU bitwise A/B, GPU pins (S1-S5, M1, M3) | merged |
| fix | #384 | The CSR lane's placeholder no longer feeds the per-particle near-field payload (default payload budget) | CPU suite, A100 bitwise budget-0 vs default | merged |
| X5 | #385 | Superseded GPU kernel variants: cascade level forwards, CSR pair / tiled M2L, per-leaf P2M, near-field sorted layout / classes / `g` / lone `p` / chunked rows, unfused walk emission (X5a); COM radii chain, MAC radius bound, degree-batched rotation, z-core M2L, three small branches (X5b). Removed env values raise | CPU suite, shard partition, CPU bitwise A/B (interpret-mode Pallas included) | merged |
| X6 | #387 | The strict lane's superseded paths: the host-routed strict refresh (`DEVICE_ONLY=0`), the dual-downward refresh planner, the unsafe compact far-pair reuse, the non-fused `strict_run_v2` loop (`FUSED_MODE=off`); `GPU_MODE` is inert. Removed env values raise, accepted ones are ignored | CPU suite, shard partition, distributed tier, CPU bitwise A/B (env unset and harness env), GPU pins S1-S6, timing A/B | merged |
| K | #388 | The Cartesian basis (`FMMEngine`'s default flips to real first: a behaviour change), `operators/multipole_utils.py`, the dense interaction buffers, the finite-difference jerk fallback. Every Cartesian spelling raises one migration error; `use_dense_interactions=True` raises; `jerk_fd_dt` is accepted and unused | CPU suite (also at the default flip alone), shard partition, distributed tier, CPU bitwise A/B, GPU pins S1-S6, M1, M3 | merged |
| C | #389 | The complex basis: the distributed driver's complex far field (C2), then single GPU (C1): `basis="complex"` raises one migration error, `basis="solidfmm"` is the real basis, the six complex modules and the runtime's complex branches go; direct-sum oracles replace the real basis's complex-only cross-checks first | CPU suite (also at the behaviour change alone), shard partition, distributed tier, CPU bitwise A/B, CI benchmark guard on real | merged |
| tower | #390 | The acceleration-derivative tower includes the near field (C's finding); `evaluate_local_real_with_grad`'s docstring sign; a GPU-only failure of K's `jerk_fd_dt` test | CPU suite, A100 tests + fp32 check, GPU pins S3/S4/S4b | merged |
| Z-a | #392 | Test consolidation (T1, rules R2-R7): the bench-tooling tests move to `bench/tests`; one signature-contract table; grid corners; `tests/integration/strict/`; one owner per gradient property; `test_fmm.py` and `test_solver_api.py` split by subject; the removed-option and shim tests collected in `test_removed_options.py` for Z-b. Tests only | CPU suite, shard partition, bench-tooling tests | merged |
| Z-b | | Shims, inert options and removal errors: `env_reject_removed` and every removal error (env values, basis migration, octree / `fixed_depth` / backend), `runtime_path`, `FarFieldConfig.mode`, `jerk_fd_dt`, the grouped / autotune / dense / `execution_backend` shims, the complex basis's `rotation` / `basis_mode` leftovers (with the large-N gate's `complex_rotation` conjunct), the always-`None` pytree fields, the locals template, the dead gear chain; benches and tests follow | CPU suite, distributed tier, shard partition, CPU bitwise A/B (110 arrays), CI benchmark guard on real, Odisseo G4, nornax adapter | open |

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
leaves an explicit `"solidfmm"` / `"complex"` on the complex basis. (An earlier version of
this record said D1 also warned on them; no such warning was ever added, as C found.)
Mapping those names to real waits for C,
after X3 and J have removed or ported the 22 tests that need them. No mass edit of the
~290 `"solidfmm"` call sites is needed before then.

### J: jerk and time derivatives on the real basis

**Two defects on the default basis, both fixed.**
- `jerk_mode="accurate"` and `compute_accelerations_with_time_derivatives` raised a
  TypeError on `basis="real"`, because only complex source-motion multipoles existed.
  nornax's jaccpot adapter defaults to `"accurate"`.
- The real derivative tower (`evaluate_local_real_derivative_tower`) was wrong at
  `delta = 0` and on the z-axis. Measured at order 4 with unit coefficients, D2 jumped
  by up to 0.5 and D3 by up to 1.5, and the U_2^0 Hessian at the centre had diagonal
  (0, 0, 1.5) instead of (-1/2, -1/2, 1). A single-particle leaf puts its target
  exactly on the centre.

**What changed.**
- `operators/real_harmonic_derivatives.py` holds the exact lowering operator: a
  Cartesian derivative maps `U_n^m` onto degree `n - 1` with coefficients 0, +-1/2, +-1.
  The rules were fitted from the code's own harmonics (residual 2e-15) and are tested
  against the Jacobian up to order 7.
- The tower is now `d^alpha phi = (A_alpha^T F) . U`, exact at every offset.
- `prepare_real_source_motion_multipoles` builds `d^k M / dt^k` with a lowered leaf P2M
  plus the ordinary real M2M; M2L and L2L run unchanged on them.

**Measured** at N = 96, leaf 4, p = 4, theta 0.6 (106 M2L pairs):
- real and complex agree to 8e-18 on the jerk and <= 6e-18 on D1-D3;
- both are at 4.338e-4 against direct summation.

The jerk / time-derivative tests in `test_solver_api.py` now run on both bases. The
source-motion multipoles are also checked against central finite differences at frozen
centres, an oracle that does not need the complex basis.

### X1: treecode, experimental, dead far field (-9.6k lines)

**Removed:**
- the per-leaf treecode walk, in both of its opt-ins: the single-GPU env switch and
  `DistributedFMMConfig(local_walk="treecode")`. Its box MAC is dynamically unstable, it
  took no pair policy, and the flat walk replaced it;
- `jaccpot/experimental/`, the `experimental` pytest marker, and everything that kept the
  prototypes out of the default run;
- `runtime/_large_n_farfield.py`, which had no importer;
- the radix benchmark worker and the three scripts that drove it.

Naming the removed walk (the env switch, or `local_walk="treecode"`) now raises with a
removal message.

**Tests:**
- Deleted, tag (a): the treecode tests, the experimental tests, the `_large_n_farfield`
  contract, the worker symbol test, and `test_octree_fmm_scaffolding.py`. That file was
  module-marked experimental, so it never ran by default; its subject is X2's.
- Converted: two tests now assert the removal errors.
- Not found by the inventory: four tests reached the treecode only through monkeypatching
  or config. They were found by grepping for the removed names.

**Gates:** CPU suite 2,334 passed and 162 skipped (the one failure is the stale local
nornax checkout); distributed driver and Dehnen-criterion tests on two forced devices,
10 passed; `test_shards.py check` 2,562 tests in exactly one shard each. The two-card
gate ran with #376: M1 and M3 are bitwise against `main`.

### GPU pins (gate G2)

Recorded at `main` 3a4bfc7 from a frozen worktree whose library is untouched; only
`bench/dce_pins.py` and the `--use-pallas` option of `bench/fused_memory_budget.py`
were copied in. One A100, deterministic GPU ops. The card was shared with another user's
~4.8 GB job, as the maintainer approved, since no card was free. Summary:
`bench/results/dce/pins_main-3a4bfc7.json`.

| pin | lane | result |
| --- | --- | --- |
| S1 | fused `strict_run_v2`, clipped Plummer 2e5, bench defaults | rel-L2 7.76e-4 vs fp64 direct |
| S2 | the same lane at 5e4, near field off Pallas | rel-L2 **0.113**, see below |
| S3 | `FastMultipoleMethod()` defaults at 3e4 and 1e5 | runs |
| S4 | Odisseo's differentiable lane, preset `fast`, 2e4 | gradients |
| S4b | the large-N differentiable lane with Odisseo's env overrides, 2e4 | gradients |
| S5 | `BlockStepFMM` with Odisseo's defaults, 20 base steps, 2e4 | momentum drift 2.5e-9 |

**The A-vs-A control is bitwise for every array of every pin**, so a phase passes only if
it reproduces them bitwise. Passed bitwise: the stack top (P1 + J + X1 + X2) and D1.

Three things the pins showed about `main`:
- **The large-N lane's no-Pallas near field is wrong (S2).** At 5e4 the Pallas near field
  gives rel-L2 1.6e-3 and the no-Pallas route 0.113, with identical far and near lists
  (668,306 / 51,294 pairs); 56 % of the particles are off by more than 1 %. This is the
  route `use_pallas=False` and `ODISSEO_FMM_USE_PALLAS=0` take on an Ampere card. The
  CPU lane golden (N = 512, leaf 16) is fine. Fixed; see "Fix: the no-Pallas near field"
  below.
- **The suspected break of the default constructor is not real (S3).**
  `FastMultipoleMethod()` on a GPU at 1e5 runs. The grouped auto-enable test does not
  fire under static sizing.
- **`BlockStepFMM(backend="pallas")` at leaf 32 fails Triton lowering** (an array of
  shape (3,)). Odisseo's default `backend="jax"` is what S5 pins.

### X2: the octree execution backend

**Removed:**
- `runtime/_octree_fmm.py` and `runtime/_octree_adapter.py`;
- the octree fields of `FMMPreparedState` (and their pytree entries) and the octree
  artifact builders in `fmm_state.py`;
- the octree prepass in `fmm_policy.py`;
- the octree branches in `fmm_prepare.py`, `kernels/_evaluate.py`, `fmm_evaluate.py`
  and `_fmm_impl.py`.

The backend was complex-basis-only, about 3x behind radix, and had no production caller.
`execution_backend="octree"`, and `tree_type="octree"` on the single-GPU solver, now raise
with a removal message; the config fields stay so old configs construct. yggdrax's octree
tree type, which the distributed lane can use, is untouched.

**Tests:**
- Deleted, tag (a): the 16 octree tests in `test_solver_api.py`,
  `test_octree_fmm_axis_contracts.py`, and the `backend_octree` constructor-state case.
- **Gates:** CPU suite 2,316 passed and 162 skipped (the one failure is the stale local
  nornax checkout); `test_shards.py check` 2,541 tests; GPU pins bitwise.

### Fix: the no-Pallas near field read the CSR lane's placeholder

**Cause.** When the CSR row-chunk near field runs, the prepare shrinks the near-field
rectangle to a one-block placeholder, and the strict runner and the multi-GPU overflow
flag skip the rectangle's capacity guard. Those three read only the hardware switch,
`_nearfield_csr_lane_enabled()`: the env var, or "an Ampere card is present". The
evaluation also needs the near field on Pallas. So with `use_pallas=False` on an A100,
the switch was on, the prepare built the placeholder and the guard was off. The pure-JAX
route then evaluated each leaf's near field from a single block of source leaves. Pre-Ampere
cards were not affected, since the switch is off there. The CPU golden missed it for the
same reason.

**Fix.** `_nearfield_csr_lane_active(use_pallas)` in `nearfield/_fast_lane.py` is now the
one predicate that the prepare, the strict runner's guard, `distributed/fused.py` and the
evaluation all use. It requires the switch, the near field on Pallas (a supported GPU or
interpret mode) and the self-leaf fold. On the Pallas path it is the same condition as
before.

**Measured:**
- CPU, the lane-golden case with the switch forced on and Odisseo's block size: rel-L2
  0.53 before the fix and 3.9e-4 after, bitwise equal to the switch-off run. This is now
  `tests/unit/runtime/test_nearfield_csr_lane_consistency.py`.
- A100, pin S2: rel-L2 0.113 before and 1.60e-3 after. The Pallas near field gives
  1.6e-3 on the same case.
- The other single-card pins (S1, S3, S4, S4b, S5) are bitwise unchanged against
  `main`. The S2 reference for later phases is now this branch's, and its A-vs-A control
  is bitwise. Summary: `bench/results/dce/pins_fix-0353eca.json`.

**Two-card pins (gate G3).** These are new in `bench/dce_pins.py` (`record --pins
M1,M2,M3`, compare with `--pins M1,M2,M3`). They ran on two A100s (GPUs 2+3), which were
shared with another user's ~4 GB jobs; these runs are untimed.

| pin | lane | result |
| --- | --- | --- |
| M1 | `distributed/fmm.py` as Odisseo's mesh lane builds it (`MeshOptions` defaults: leaf 512, theta 0.7, p6, rcb), one force at 131,072 | rel-L2 1.85e-4 vs fp64 direct, no overflow; bitwise vs `main` |
| M2 | `bench/multigpu_rollout_gate.py`, 1e5, 17 steps, repartition every 8 | **not recorded**, see below |
| M3 | `DistributedBlockStepFMM` (jax backend, cross_theta 0.5), 4 base steps at 8192 | momentum drift 6.9e-10; bitwise vs `main` |

The A-vs-A control at `main` is bitwise for M1 and M3. So X1's distributed changes and
this fix leave the mesh lane and the distributed block-step lane bitwise unchanged.

**The fused two-card lane does not run on jax 0.11.2 (M2). This is on `main` too, and it
is not part of this cleanup.**
- With `fused.RAGGED_EXCHANGE_XLA_FLAG`, the NCCL exchange jaccpot recommends, XLA
  raises: "RaggedAllToAll fallback to NCCL is not allowed".
- With the fallback allowed (`--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true`,
  which M2 now sets), the rollout hangs in its first force. One device spins at 99 % and
  the other idles. This happened on the `main` arm, its control and this branch alike,
  three times, each until the 1 h timeout.
- `bench/distributed_gate.py`'s target ran without the flag: four tests passed, then
  `test_forward_survives_a_gradient` hung the same way.

To be investigated separately. Until then, the fused multi-GPU lane (`FusedRollout`) has
no GPU gate on jax 0.11.2.

### Pin baseline re-recorded at 15ceca4

#375 made `ferrers3` the default softening kernel, so the 3a4bfc7 pins no longer describe
`main`. They were re-recorded from a frozen worktree at `main` 15ceca4 (after #375 and
#376; yggdrax 9372332), with an A-vs-A control. The control is **bitwise for every array
of every pin** (S1-S5, M1, M3). Later phases compare against `main-15ceca4` with
`--envelope main-15ceca4-b`. Summary: `bench/results/dce/pins_main-15ceca4.json`.

Two pin changes:
- **M1's reference uses the lane's own kernel.** Its direct sum was Plummer. With
  `ferrers3` it measured the kernel difference (6.7e-3), not the FMM error.
- **M3 asks for Plummer explicitly.** `cross_theta > 0` with a compact kernel raises by
  design, since the cross walk has no separation floor yet. Plummer keeps the cross-M2L
  path pinned.

What moved against 3a4bfc7, as the rel-L2 of the pinned arrays:

| pin | moved | why |
| --- | --- | --- |
| S1 | 1.7e-8 | softening 1e-7: the kernel barely matters; error vs fp64 direct unchanged (7.76e-4) |
| S2 | 0.113 | #376; error 0.113 -> 1.60e-3 |
| S3 | 2.0e-2 (3e4), 3.7e-2 (1e5) | default softening 1e-3: `ferrers3` instead of Plummer |
| S4, S4b | 6.9e-2 (d/dmass), 0.19 (d/dpos) | softening 1e-3, kernel change |
| S5 | 2.1e-4 (velocities) | `BlockStepFMM` follows the default kernel |
| M1 | 3.7e-2 | kernel change; error vs a matching direct sum 1.854e-4 -> 1.853e-4 |
| M3 | bitwise | Plummer explicitly: #375 left the Plummer distributed mutual path unchanged |

### D2: the fused strict lane is the default

Before this phase, the fused strict lane ran only when the caller set a dozen
`JACCPOT_*` variables. Odisseo did this in `jaccpot_coupling.py` for N >= 200k with
`large_n_gpu` + `static_radix`, and the bench harness did it with `apply_fast_lane_env`.
Without them, `strict_run_v2` ran the host-driven loop, which is about 10x slower. And a
default `static_radix` + `large_n_gpu` prepare raised: it required a recorded cap profile
for exactly that (leaf, N) on disk.

**A per-knob survey** (reader, default, effect; in the PR) showed that most of Odisseo's
values were already the defaults. Only these changed:

| knob | was | now | why |
| --- | --- | --- | --- |
| `JACCPOT_STATIC_STRICT_FUSED_MODE` | off | on | the switch; only `strict_run_v2` and the fused eval fn read it (the multi-GPU lane's `measure_shard_plan` needs the latter) |
| `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH` | 1 | 0 | with 1, the default static-radix prepare raised without a profile on disk; read nowhere since X4 removed the profile |
| `JACCPOT_LARGE_N_TARGET_BLOCK_SIZE` | 32 | 4 | off the CSR lane only (pre-Ampere, CPU, `use_pallas=False`); 32 left at least 256 slots per target leaf |
| `JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF` | 32 | auto | the same lane; `auto` sizes it from the densest leaf with 1.25 headroom |
| `large_n_gpu` tree mode | `lbvh` | `static_radix` | the fused lane needs it (`fmm_presets.py`, `solver.py`) |

**Not adopted, because the library's own default is better:**
- the named 131,072 far-pair cap: a hard ceiling that raises at larger N, where the default
  is sized from the count;
- `PROFILE_SET=N`: it only restricts; empty means all N.

`JACCPOT_LARGE_N_COMPILED_STATE_MODE` is read nowhere. It was removed from the tests and
benches that set it.

**Gates:**
- **CPU suite:** one failure was the stale local nornax checkout, as before. The other was
  `test_discarded_far_pairs_are_rejected_rather_than_differentiated_as_constant`. Its premise
  (the default preset discards the far-pair list) is no longer true: the `static_radix` state
  keeps it for its strict lane, so the reverse runs through the real far field. The test now
  names `lbvh` for the refusal and asserts the retention on `static_radix`.
- **Lane goldens** now run with every fused knob unset. They reproduce, within the float32
  gate, the goldens recorded with the env set.
- **Constructor-state golden:** regenerated. The diff is exactly the preset tree mode, fused
  mode and the exact-profile flag.
- **GPU, one A100, defaults gate** (`bench/fused_memory_budget.py --library-defaults` drops
  every knob above): D2 with a clean env reproduces `main` 15ceca4 with the full harness env
  **bitwise** (forces and final scan state) at 2e4, 2e5 and 8e6. Each default now equals
  the value the env set, so the same code runs; the bitwise result confirms it. That is why
  no interleaved timing A/B (G5) was run.
- **Pins:** S1-S5 bitwise against `main-15ceca4`.
- **Odisseo (G4), one A100, against `main`, D2, and D2 with Odisseo's env block removed
  (O2):** identical on all three arms.
  - `test_strict_velocity_verlet_policy`, `test_integration_api` and
    `test_dynamics_direct_sum_agreement` pass.
  - `test_blockstep_fmm` and `test_differentiable_fmm` exceed a 1 h per-file limit on
    the shared card. On CPU against D2 + O2 they run complete: the differentiable file
    passes in full.
  - The block-step file has two failures, neither caused by D2 or O2:
    - `test_the_pallas_backend_drops_most_of_the_dt_max_gradient` is a tripwire for an
      upstream defect that has since been fixed: the Pallas gradient is now exact (AD/FD
      1.0000). O2 replaces it with the exactness test on both backends.
    - `test_one_compiled_program_survives_every_topology_rebuild` fails only in some
      xdist orders and passes alone. This is a pre-existing isolation issue in Odisseo's
      suite.

**Odisseo's own production lane did not run on `main`.** At N >= 200k on a GPU, Odisseo's
defaults (`large_n_gpu`, `static_radix`, the env block) raised on the A100, before and
after D2, for two reasons:
- The coupling's neighbour-edge autosize names `16 * N + 1` = 3,200,001. That is odd, and
  the flat walk, the default since 2026-09-10, refuses odd caps.
- With the autosize off, the block's named 131,072 far-pair cap overflows.

With D2 and O2, which drops both and leaves the caps to the library, the 2e5 run completes:
4 steps, finite. There is no working baseline to compare it with bitwise. It runs the
configuration the defaults gate above checks, with unnamed caps like pin S1.

### X3: grouped / class-major far field and non-COM expansion centres

**Removed** (library -2,653 / +338 lines; most of the additions are docstrings and
the removal shims):
- `runtime/kernels/_m2l.py` (-941): the grouped and class-major accumulators
  (`_accumulate_solidfmm_m2l_grouped[_fullbatch|_chunked_scan|_class_major]`,
  `_accumulate_solidfmm_m2l_class_major_chunked_scan`), the class-rotation-block
  builder `_rotation_blocks_for_grouped_classes`, `_build_grouped_class_segments`,
  `_pair_class_ids_from_offsets`, and the cached-kernel dispatch
  (`_m2l_cached_kernel_dispatch`, `_m2l_complex_batch_cached_kernel`). The flat
  full-batch and chunked-scan accumulators and `_chunk_segment_scatter_add` stay.
  The `*_cached_blocks` operators in `operators/` and `pallas/m2l_complex_fused.py`
  stay; only this module's import of them went.
- `runtime/_interaction_cache.py` (-493): the grouped fields of the dual-tree
  artifacts, cache entry and cache hit; the grouped buffer and class-segment builders;
  `_without_grouped_class_segments`; the grouped arms of the raw build, the unpack and
  the split predicate; the grouped term of the compiled planner route.
- `runtime/fmm_caches.py` (-282): the grouped operator and segment caches, their six
  `JACCPOT_GROUPED_*` knobs, the content-digest keys and `_M2L_FULLBATCH_MAX_PAIRS`.
  `_clear_global_runtime_caches` stays; it now only clears JAX's compilation cache.
- `runtime/fmm_prepare.py` (-214): the plan field `grouped_interactions_active` and
  its uses, the two gatekeepers (`_should_precompute_grouped_class_segments`,
  `_grouped_schedule_item_budget`), and the grouped and `farfield_mode` plumbing of the
  dual/downward prepare. Where `and not grouped_interactions_active` gated the compact
  streamed pairs, the split build, the strict streamed fast path or the minimum-memory
  caps, only the term went: it was true on every surviving lane.
- `runtime/kernels/_downward_prep.py`, `_l2l.py`, `fmm_sweeps.py` (-331): the grouped
  lanes and parameters of the downward sweep (including `farfield_mode`, read only by
  the grouped lane, and the `tree` / `upward` / `interactions` arguments of the
  accumulate step, which only the grouped builder read), and the dead cached L2L
  cascade (`l2l_grouped`, `mm_class_capacity`; no runtime caller passed it).
  `_propagate_solidfmm_locals_by_level`'s call from `distributed/fmm.py` is unchanged.
- `runtime/fmm_overrides.py` (-113): the resolver's grouped auto-enable (fast preset at
  large CPU N; fast / `large_n_gpu` at large GPU N, adaptive sizing only), the class-major
  threshold, and `JACCPOT_LARGE_N_FASTLANE_GEOMETRIC_CENTERS`. The resolver now returns
  `farfield_mode="pair_grouped"` (the flat list; the name is historical) and
  `center_mode="com"` on every path, which is what every non-grouped lane already got.
- `runtime/fmm_evaluate.py` (-38): `force_ungrouped_farfield`, which forced the
  flat far field on the grad path and has nothing left to force.
- `_fmm_impl.py`, `fmm_state.py`, `fmm_constants.py`, `fmm_policy.py`,
  `fmm_derivatives.py`, `fmm_strict_run.py`, `_large_n_*.py`, `_nearfield_cache.py`,
  `kernels/{core,__init__}.py`, `fmm/__init__.py`: the grouped engine attributes, budgets
  and contract lines, `_GROUPED_SCHEDULE_BUDGET_DEFAULT`,
  `_CLASS_MAJOR_CPU_PARTICLE_THRESHOLD`, `_RuntimeExecutionOverrides.grouped_interactions`,
  `LargeNGradPlan.farfield_mode`, re-exports and constant grouped kwargs.
- `upward/tree_expansions.py`, `upward/solidfmm_complex_tree_expansions.py`: the AABB
  expansion centres. `center_mode` is `"com"` or `"explicit"` (the test seam).

The grouped far field ran only on the complex basis (its AABB centres were refused by
the real upward sweep), was opt-in except for the adaptive-sizing auto-enable, and was
less accurate than the flat list (one representative rotation per displacement class;
the residual did not shrink with order). No production lane reached it: Odisseo's
lanes, the fused lane and both multi-GPU lanes run the flat list.

**API shims** (the same pattern as X2's octree fields):
- `FarFieldConfig.grouped_interactions` stays. `None`/`False` construct; `True` raises
  `ValueError` ("grouped interactions were removed in the 2026-10 cleanup (X3)") at
  construction. So does the legacy `FastMultipoleMethod(grouped_interactions=True)`.
  The `fmm.grouped_interactions` getter returns `False`; the setter raises on `True` and
  mirrors `False` into `advanced`.
- `FarFieldConfig.mode`: `"auto"` and `"pair_grouped"` are accepted, as before, and both
  run the flat list. The block-step lane's `preset="balanced"` names `"pair_grouped"`.
  `"class_major"` (also via the legacy `farfield_mode=`) raises with a removal message.
  It stays in `config.FarFieldMode` so an old config type-checks; the engine's own
  alias drops it.
- `RuntimePolicyConfig.precompute_grouped_class_segments` and
  `.grouped_schedule_budget_bytes`, and their legacy kwargs, are accepted and ignored
  (a non-positive budget no longer raises). They go in phase Z.
- `center_mode="aabb"` / `"geometric"` raises in every upward sweep with a removal
  message.

**Tests:**

| test | tag | owner / note |
| --- | --- | --- |
| `test_fmm_golden.py::test_fmm_golden_execution_modes[cm_uni_solidfmm_n256_p4, cm_clu_…, pg_uni_…, pg_clu_…]` and their four `golden_modes/*.npz` | (a) | the `bkt_*` cases stay |
| `test_fmm_golden.py::test_grouped_farfield_plateaus_in_order[pair_grouped, class_major]` | (a) | |
| `tests/unit/runtime/test_grouped_m2l_basis_mode.py` (4 tests, 6 cases) | (a) | |
| `tests/unit/runtime/test_grouped_class_id_alignment.py` (3) | (a) | |
| `test_m2l_shape_contracts.py`: `test_matched_classes_still_build_their_blocks`, `test_class_deltas_shorter_than_class_keys_is_rejected`, `test_class_keys_shorter_than_class_deltas_is_rejected`, `test_a_class_key_of_the_wrong_width_is_rejected` | (a) | the `nodes`/`sh` and scatter contracts stay |
| `test_fmm.py`: `test_prepare_state_reuses_grouped_buffers_from_cache`, `…_reuses_grouped_class_segments_from_cache`, `test_prepare_state_cache_key_respects_center_mode`, `test_fast_preset_adaptive_class_major_threshold`, `test_solidfmm_grouped_interactions_matches_sparse_path`, `test_solidfmm_grouped_class_major_matches_pair_grouped` | (a) | `center_mode` is constant now, so the key test has no second value to tell apart |
| `test_solver_api.py`: `test_pair_grouped_mode_skips_class_major_schedule_precompute`, `test_minimum_memory_gpu_runtime_does_not_auto_enable_grouped_interactions`, `test_streamed_far_pairs_disables_grouped_interactions_runtime_override`, `test_without_grouped_class_segments_clears_cached_schedule_arrays` | (a) | |
| `test_large_n_config_thresholds.py`: `test_grouped_interactions_implies_geometric_centers` (4 cases), `test_explicit_grouped_interactions_still_resolves_under_static_sizing`, `test_grad_works_with_grouped_interactions_requested` (also out of `slow_tests.txt`), `test_ungrouped_grad_path_is_a_valid_and_more_accurate_far_field` | (a) | the radix grad path keeps its direct-sum anchor in `test_fmm_grad_golden.py` and `test_grad_fmm_vs_directsum.py` |
| `test_fastlane_geometric_centers.py::test_geometric_centers_knob_is_live` | (a) | GPU opt-in; `test_real_com_tracks_complex` stays |
| `test_tree_expansions.py::test_compute_node_multipoles_aabb_uses_geometry_center` | (a) | |
| `test_real_rot_scale_grouped.py::test_fastlane_grouped_l2l_cascade_matches_per_node` (3 cases) | (a) | the operator-level cached-block tests stay |
| constructor-state matrix cases `farfield_class_major`, `grouped_interactions` | (a) | |

No test was deleted under (b) or (c).

Adapted: `test_fast_preset_adaptive_large_cpu_policy_applies` (the policy now resolves
the flat list about COM centres), `test_nearfield_bucketed_matches_baseline` (default
far field instead of class-major), `test_advanced_config_applies_to_runtime`,
`test_large_n_gpu_preset_applies_memory_safe_gpu_defaults`,
`test_large_n_gpu_profile_coerces_conflicting_runtime_knobs`,
`test_large_n_gpu_profile_emits_deprecation_warnings_for_conflicting_knobs`,
`test_runtime_memory_policy_fields_flow_to_runtime`,
`test_accepts_legacy_expanse_kwargs_with_deprecation_warning`,
`test_static_sizing_does_not_inherit_the_adaptive_grouped_rewrite` and
`test_farfield_mode_never_resolves_to_auto_at_large_n` (now `== "pair_grouped"`, which
is stricter), `test_prepare_upward_sweep_returns_consistent_data` (COM centres), and the
seven source-motion tests in `test_solidfmm_complex_tree_expansions.py` that used AABB
centres only as the base for `"explicit"` (now COM). Four tests that call internal
builders lost their grouped keyword arguments.

Added: seven shim tests in `test_solver_api.py` (`grouped_interactions=True` through the
config, the legacy kwarg and the setter; `mode="class_major"` through both spellings;
`"auto"` and `"pair_grouped"` construct; `preset="balanced"` constructs; the two policy
fields are inert) and `test_removed_geometric_centres_raise_a_removal_error` (three
spellings, three upward sweeps).

**Goldens.** `golden/`, `golden_grad/` and `golden_lanes/` are byte-identical.
`constructor_state.json` was regenerated; its diff is exactly: the two matrix cases;
five base attributes (`_explicit_grouped_interactions`, `_fastlane_geometric_centers`,
`grouped_interactions`, `grouped_schedule_budget_bytes`,
`precompute_grouped_class_segments`) and their `preset_large_n_gpu` overrides; and the
`large_n_gpu` preset description ("streamed/grouped" -> "streamed"). The distinctness
check passes with the two cases gone.

**Left for phase Z:** the notebooks that still name grouped knobs
(`examples/benchmark_runtime_accuracy_copy.ipynb`,
`benchmark_runtime_rtx2080_focused.ipynb`, `benchmark_runtime_large_N.ipynb`,
`benchmark_gpu_radix_runtime.ipynb`, `benchmark_runtime_large_N_performance.ipynb`,
`benchmark_runtime_large_N_accuracy.ipynb`); the two inert policy fields; the
`grouped_interactions` config field and setter. `examples/benchmark_grouped_modes.py` is
deleted. `examples/profile_prepare_memory_split.py` lost its grouped arguments but still
does not import: it needs `examples/benchmark_gpu_radix_worker.py`, which X1 removed.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`): 2,354
  passed, 187 skipped, 1 failed: the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as before. An
  earlier targeted run lost one xdist worker to a segfault inside XLA's CPU runtime
  (`test_strict_prepare_refresh_and_evaluate_api_and_diagnostics`, on the loaded shared
  host); the test passes alone and in the full run.
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`, as the unit shards run in CI)
  on every touched unit test file: green.
- **Bitwise A/B against 483b810 on CPU:** 45 arrays equal. They are the forces and
  potentials of 11 configurations (the default; fast and accurate on both bases;
  balanced; an explicit `pair_grouped`; kd-tree; `dehnen_error`; `dehnen_paper`
  adaptive order; `large_n_gpu` in fp32), a prepared state with a target subset, the
  position and mass gradients through `differentiable_accelerations` on both bases,
  and the runtime overrides that three presets resolve at three N on two backends.
- **`test_shards.py check`:** 2,604 tests, each in exactly one shard. At 483b810 there
  are 2,634; this phase deletes 41 cases and adds 11.
- `bench/bench_fmm.py` and `bench/bench_real_vs_complex.py` run on CPU.
- `bench/annotation_census.py`: shape-annotated array parameters 856 -> 837, bare
  `Array` parameters 1,751 -> 1,683 (shaped share 32.8 % -> 33.2 %), `@jaxtyped`
  functions 190 -> 183. Only deletions moved them.
- **GPU pins** (frozen worktree at 1694005, against `main-15ceca4` with its A-vs-A control):
  S1-S4b on one A100 and M1, M3 on two are **bitwise**.
  - S5 (`BlockStepFMM`, 20 base steps) hit its 30 min per-pin timeout on the shared card.
  - Rerun back to back with `main` on the same card, it took 26 min on X3 and 40 min on
    `main`, and is bitwise equal to the pin. So the timeout was the card, not X3.

### X4: M2L autotune, adaptive sizing, legacy strict APIs, `fixed_depth`, the /tmp cap profile

**Removed** (library -2,067 / +186 lines; the additions are docstrings and removal
notes):
- **M2L chunk autotune (~830).** `runtime/fmm_autotune.py` (-397, `AutotuneMixin`),
  its process-global LRU cache, (de)serialisers and `_GPU_M2L_AUTOTUNE_*` constants in
  `fmm_caches.py` (-132), the engine's `autotune_m2l_chunk` attribute (and its
  `fail_fast` coupling) and `dual_m2l_autotune` timer, stage and diagnostics key,
  `export/import/save/load_m2l_autotune_cache` on the engine (-93) and the facade
  (-83), `_prepare_state_autotune_downward_chunk_size` (-73) and its two call blocks,
  and the `large_n_gpu` preset's `autotune_m2l_chunk=True`.
- **Adaptive sizing (~300).** `JACCPOT_STATIC_RUNTIME_FIXED_SIZING` (read in
  `_fmm_impl.py` and `_large_n_env.py`) and everything only its `0` reached: the
  resolver's adaptive tail (the fast preset's large-CPU traversal config and 32768 M2L
  chunk, clamping of explicit caps, the minimum-memory M2L chunk 1024 / 4096),
  `honor_explicit_traversal`, `_RuntimeExecutionOverrides.adaptive_applied`, four
  constants; in the large-N pipeline the non-static branches of
  `_size_fused_static_overflow_profile` and `_trim_radix_fast_lane_neighbor_list`, the
  `_pick_*_profile_capacity` ladders, `JACCPOT_LARGE_N_OVERFLOW_PROFILE_HEADROOM` /
  `_CAP_OPTIONS`, `JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_HEADROOM` / `_CAP_OPTIONS`
  and the `large_n_overflow_profile_reprofiles` counter. The static branches are
  unchanged: fused, a named cap or else the first build's count; not fused (what a
  pre-Ampere card runs), a named cap or else this build's count. The two
  `*_BOOTSTRAP_CAP` knobs stay as the fixed caps' defaults.
- **Legacy strict APIs (~470).** `strict_run_segmented`, `update_multipoles_only` and
  `rebuild_topology_in_place` on the engine (-279) and the facade (-180), their
  counters and the `update_multipoles_only_calls` / `rebuild_topology_in_place_calls`
  diagnostics keys. `strict_prepare_refresh_and_evaluate`, `refresh_prepared_state`
  and `_large_n_neighbor_list_matches` stay.
- **`tree_build_mode="fixed_depth"` (~25).** The valid-mode entry, its mapping onto
  yggdrax's builder, its leaf-size exemption, and the two non-radix fallback tuples
  (`fmm_prepare.py`, `fmm_strict_run.py`). Docstrings: `TreeConfig.mode`, the
  `refine_local` parameter of three methods, and the FAST preset's description,
  which said "fixed-depth builder" of a preset that builds `lbvh`.
- **The strict cap-profile file (~255).** `_strict_cap_profile_path`,
  `_strict_cap_profile_context_key`, `_maybe_load_strict_cap_profile`,
  `_apply_strict_cap_profile_for_key` and `_record_strict_cap_profile_from_retries`;
  the read in the dual/downward prepare (widening `max_pair_queue`, replacing
  `process_block`, creating an explicit config), the writes after a prepare and after
  a same-topology refresh, `JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH`,
  `JACCPOT_STATIC_STRICT_CAP_RECORD`, the read of
  `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH`, five engine attributes and
  the `strict_profiled_*` diagnostics keys. The retry logger is now
  `None if strict_mode_active else record_retry`, which is what it was by default.
  `fmm_strict_cap_profile.py` keeps the compiled-profile fingerprints and the
  `PROFILE_SET` gate.
- **Dead env reads (~15).** `JACCPOT_STATIC_STRICT_FUSED_DISABLE_REMATERIALIZE`,
  `_COMPILED_SEGMENT_LOOP` and `_JIT_REFRESH_EVAL` were read into attributes nothing
  read. Two docstrings named knobs that do not exist: `JACCPOT_UPWARD_DIAGNOSTICS`
  (the switch is `JACCPOT_PREPARE_DIAGNOSTICS`) and `JACCPOT_MUTUAL_FUSED_M2L=0`
  (`JACCPOT_MUTUAL_M2L=zcore`).

No production lane reached any of it. The autotune returned early under static
sizing, and static sizing was the default. Nothing called the three strict APIs:
Odisseo uses `strict_run_v2` and `strict_prepare_refresh_and_evaluate`. No preset
named `fixed_depth`.

**Behaviour changes:**
- **`runtime_path` no longer selects a lane.** The large-N lane is gated on
  `preset="large_n_gpu"` alone. An explicit `"large_n"` used to open it under any
  preset, and to turn the lane's `dehnen_theta` decline into a raise; on
  `large_n_gpu` the contract pins `"large_n"` anyway, so there nothing changed and the
  raise is now unconditional. Under another preset, `runtime_path="large_n"` on a GPU
  now runs the general path. Every bench and test that passes `"large_n"` also passes
  `large_n_gpu`; `bench/bench_fmm.py --runtime-path` says so in its help.
- **No /tmp cap profile.** The strict lanes no longer read or write
  `/tmp/jaccpot_static_strict_caps.json` (or the env path). A strict prepare's
  `max_pair_queue` and `process_block` no longer depend on what an earlier run --
  another session, another user -- left in that file, and a prepare that retried its
  traversal no longer writes it, the general default path included. At the time of
  writing this box has one, left by an `lbvh` run at N = 20,000. The
  in-memory carry of the same caps between prepares of one engine went with it; the
  strict lanes run `fail_fast` without a retry logger, so only retries outside them
  ever fed it.
- **The refine family is inert on radix trees.** `refine_local`, `max_refine_levels`,
  `aspect_threshold`, `TreeConfig.leaf_target` and `host_refine_mode` only shaped
  fixed-depth trees. `refine_local` still routes the build off the jitted LBVH fast
  path, and `leaf_target` still enters the topology and cache keys outside `lbvh`.

**Shims:**
- `RuntimePolicyConfig.autotune_m2l_chunk` and the legacy `autotune_m2l_chunk=`
  kwarg are accepted and ignored (examples and notebooks pass it).
- `runtime_path` stays `"auto"` | `"large_n"`: validated, stored, and part of the
  compiled-profile fingerprint.
- `TreeConfig(mode="fixed_depth")` constructs; the solver raises "tree_build_mode=
  'fixed_depth' was removed in the 2026-10 cleanup (X4); use 'lbvh' or
  'static_radix'", as does the legacy `tree_build_mode=`.
- `_gear_pairs_for_autotune` keeps its name and its three call sites (#379 edits the
  lines around one), and `far_pairs_by_gear` its plumbing. Both feed nothing now.

**Tests** (88 cases deleted or renamed, 8 added):

| test | tag | owner / note |
| --- | --- | --- |
| `tests/unit/runtime/test_autotune_chunk_selection.py` (10), `test_autotune_helpers.py` (23) | (a) | |
| `test_engine_config_resolution.py::TestTheFailFastAutotuneInteraction` (3) | (a) | module docstring rewritten |
| `test_solver_api.py`: `test_runtime_autotune_m2l_chunk_flag_flows_to_runtime`, `test_m2l_autotune_cache_roundtrip_api`, `test_large_gpu_minimum_memory_streamed_path_caps_oversized_explicit_traversal` | (a) | the last asserted an explicit cap clamped, which only adaptive sizing did |
| `test_solver_api.py::test_runtime_fail_fast_disables_autotune_and_host_refine` | (a) for its autotune half | renamed `test_runtime_fail_fast_disables_host_refine`, host-refine half kept |
| `test_traversal_clamp_honors_explicit.py::test_adaptive_sizing_still_caps_oversized_explicit_traversal` | (a) | |
| `test_fmm.py::test_fast_preset_adaptive_large_cpu_policy_applies` | (a) | |
| `test_large_n_config_thresholds.py::test_static_sizing_does_not_inherit_the_adaptive_grouped_rewrite` | (b) | its `center_mode == "com"` assertion is now in `test_farfield_mode_never_resolves_to_auto_at_large_n` (all four cases) |
| `tests/integration/test_strict_run_segmented.py` (11) | (a) | `strict_prepare_refresh_and_evaluate` keeps its owner, `test_fmm.py::test_strict_prepare_refresh_and_evaluate_api_and_diagnostics` |
| `test_strict_refresh_faces.py`: the `update_multipoles_only` and `rebuild_topology_in_place` cases of three parametrised tests (6), `test_rebuild_topology_in_place_does_not_mutate_its_input` | (a) | the `refresh_prepared_state` cases stay |
| `test_fmm.py`: `test_prepare_state_fixed_depth_tree` (also out of `slow_tests.txt`), `test_compute_accelerations_fixed_depth_matches_direct`, `..._fixed_depth_jitted_matches_eager`, `test_compute_accelerations_refined_tree_matches_non_refined` (and the helpers `_fixed_depth_sample`, `_line_cluster_sample`) | (a) | the radix lanes keep their direct-sum anchors (`test_fmm_golden.py`, the lane goldens) |
| `test_strict_cap_profile.py`: `TestContextKey` (2), `TestLoading` (5), `TestSelectionPolicy` (5), `TestRecording` (8) | (a) | the compiled-profile and `PROFILE_SET` classes stay |
| `test_fmm.py::test_strict_exact_cap_profile_match_fail_fast` | (a) | |
| `test_near_field.py::test_large_n_accel_only_env_variants_match_baseline[target_owned_accum_v2]` | (c) | `[target_leaf_batch_size]`: with the dead knobs dropped the two cases set the same env |

`JACCPOT_LARGE_N_TARGET_OWNED_ACCUM`, `..._ACCUM_V2`,
`..._TARGET_LEAF_NEIGHBOR_BLOCK_SIZE` and `JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT` are
read nowhere; the tests that set them no longer do (seven `setenv`s, and the
accel-only parity cases). The case `[target_owned_accum]` kept its only live knob and
is now `[target_leaf_batch_size]`.
`test_large_n_fast_path_policy.py::test_large_n_fast_lane_legacy_opt_out_env_is_noop`
stays: it pins that the dead `JACCPOT_LARGE_N_RADIX_FAST_LANE=0` cannot switch the
fast lane off, and it is the only test of `TARGET_BLOCK_SIZE=0` meaning the default.

**Found, not fixed:** every case of `test_large_n_accel_only_env_variants_match_baseline`
compares the default with itself. `compute_leaf_p2p_accelerations_large_n_accel_only`
takes its scatter and accumulation knobs as arguments and reads no env, and the test
passes none of them. So the env variants it names are not exercised there. (Also,
`TARGET_LEAF_BATCH_SIZE=2` snaps to 16, the default, wherever it is read.) That is its
own fix.

Adapted:
- `test_large_n_gpu_preset_applies_memory_safe_gpu_defaults`: no autotune assert.
- The four traversal-seed tests in `test_solver_api.py`
  (`..._clamps_auto_traversal_seed`, `..._seed_scales_for_xl_particle_counts`,
  `test_gpu_runtime_overrides_cap_traversal_capacities_for_large_n`,
  `test_minimum_memory_gpu_runtime_starts_with_smaller_traversal_capacities`) lost
  their `_static_runtime_fixed_sizing = False`. Static sizing gives what they assert:
  the memory-safety clamp is the same function on both paths and differed only for an
  explicit config.
- `test_static_fixed_sizing_honors_oversized_explicit_traversal`: no static-sizing
  precondition, which is now always true.
- `test_fast_preset_adaptive_policy_respects_explicit_overrides`: no `adaptive_applied`.
- `test_the_lane_still_refuses_the_folded_angle_mode`: the raise is asserted without
  the `runtime_path == "large_n"` precondition.
- `test_capacity_fixed_depth_tree_mode_is_removed`: also asserts the `fixed_depth`
  removal message.
- `test_public_api_surface.py` (seven methods), `test_mixin_engine_base_guard.py`
  (the MRO).
- `TestLaneModeNormalisation`'s `runtime_path` rows are unchanged: the value is still
  validated and normalised.

Added:
- In `test_solver_api.py`:
  - `test_fixed_depth_tree_mode_raises_removal_error` (both spellings);
  - `test_removed_autotune_m2l_chunk_is_inert` (the whole engine state equals the
    default's, through the field, the kwarg and the engine);
  - `test_runtime_path_is_accepted_and_no_longer_selects_the_lane[auto, large_n]`.
- `tests/integration/test_strict_cap_profile_removed.py`: a strict static-radix
  prepare does not open a recorded profile for its own (leaf, N), and a general
  prepare that retries its traversal does not write one. `open` is guarded to record
  and refuse, since the old reader and writer swallowed every exception.

Each fails on db02d19, except the `[auto]` case, which pins the half that did not
change.

**Goldens.** `golden/`, `golden_grad/` and `golden_lanes/` are byte-identical.
`constructor_state.json` was regenerated; its diff is exactly:
- the matrix cases `autotune_m2l` and `tree_fixed_depth` (the latter raises now);
- 16 base attributes: `autotune_m2l_chunk`, `_static_runtime_fixed_sizing`, the
  `_strict_cap_*` pair, the five `_strict_profile*` / `_strict_profiled_*`
  attributes, the three dead `_strict_fused_*` flags, the two legacy-API counters,
  `_large_n_overflow_profile_reprofiles` and `_refresh_timing_dual_m2l_autotune_seconds`;
- the `preset_large_n_gpu` override of `autotune_m2l_chunk`;
- the FAST preset description in `preset_fast` and `engine_preset_fast`.

`runtime_path_large_n` stays distinct, and the distinctness check passes.

**Left for phase Z:**
- the inert `autotune_m2l_chunk` field and kwarg;
- `runtime_path` itself;
- the refine family on radix trees;
- the name `_gear_pairs_for_autotune` and the `far_pairs_by_gear` plumbing;
- the "or None to autotune" docstring line of
  `_prepare_state_dual_and_downward_strict_streamed_fast` (#379 edits that docstring);
- `examples/benchmark_utils.py` and `examples/profile_prepare_residuals.py`, which
  still pass `autotune_m2l_chunk=True`;
- `examples/profile_prepare_memory_split.py`, which still calls the removed autotune
  helper. It has not imported since X1.
- The notebooks that name autotune (`benchmark_runtime_breakdown_h200`,
  `benchmark_runtime_accuracy_copy`, `benchmark_gpu_radix_runtime`,
  `benchmark_runtime_large_N_accuracy`, `benchmark_runtime_large_N`,
  `benchmark_gpu_single_n_memory`, `benchmark_runtime_large_N_performance`,
  `benchmark_runtime_rtx2080_focused`), `fixed_depth` (five of them) or
  `runtime_path`.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`): 2,274
  passed, 187 skipped, 1 failed: the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as before.
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`) on every touched unit test
  file: 184 passed.
- **Bitwise A/B against db02d19 on CPU** (a `git archive` export; the baseline arm's
  `JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH` pointed at an empty directory, which stayed
  empty, so no recorded profile could reach it): 37 arrays equal, and 44 resolved
  rows equal. The arrays:
  - forces and potentials of 12 configurations: the default; fast and accurate on
    both bases; balanced; an explicit `pair_grouped`; kd-tree; `dehnen_error`;
    `dehnen_paper` adaptive order; `large_n_gpu` in fp32 on the general path;
    `refine_local=True` on `lbvh`;
  - a prepared state with a target subset;
  - position and mass gradients through `differentiable_accelerations` on both bases;
  - the large-N lane on CPU with the GPU gate opened: prepare + evaluate, a
    same-topology refresh, two `strict_prepare_refresh_and_evaluate` calls, the fused
    `strict_run_v2` scan state after 3 steps (fused, no fallback), and the prepacked
    differentiable lane's forces and position / mass gradients.

  The rows are the runtime overrides that four presets (fast, balanced, accurate,
  `large_n_gpu`) resolve at five N (1e3 to 5e6) on two backends, plus the large-N
  lane's two cap diagnostics.
- **`test_shards.py check`:** 2,524 tests, each in exactly one shard. At db02d19
  there are 2,604; the export, which has no sibling nornax checkout, collects 2,568.
  This phase deletes 86 cases, renames 2 and adds 6.
- `bench/annotation_census.py`: shape-annotated array parameters 837 -> 831, bare
  `Array` parameters 1,683 -> 1,668 (shaped share 33.2 % -> 33.3 %), `@jaxtyped`
  functions 183 (unchanged). Only deletions moved them.
- **After merging `main` with #379 (2896dae; yggdrax `main` 7cab99b, which #379 needs
  for the walk's `pair_accept` hook):**
  - CPU suite 2,321 passed, 206 skipped, 1 failed (the stale nornax checkout).
    Against the older yggdrax 9372332, #379's own new walk tests fail; that is an
    environment mismatch, not this phase.
  - `test_shards.py check`: 2,590 tests, each in exactly one shard.
  - `golden/`, `golden_grad/` and `golden_lanes/` are byte-identical to `main`.
- **GPU pins** (frozen worktree at 2896dae, against `main-15ceca4` with its A-vs-A
  control): S1-S5 on one A100 and M1, M3 on two are **bitwise**. #379 and yggdrax #89
  moved no pin either, so `main-15ceca4` stays the baseline.

### Fix: the CSR lane's placeholder fed the per-particle near-field payload

Found by the X5 survey.

**Cause.** With the CSR near-field lane active (Ampere, `use_pallas`), the prepare
shrinks the target-block rectangle to a one-block placeholder. The fast lane
materialises a per-particle source payload whenever its size estimate fits
`JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB` (default 1024). The placeholder's estimate
always fits. A materialised payload makes the evaluation take the pairs kernel instead
of the CSR lane, so each leaf saw only its first block of neighbour leaves.

**Measured on an A100** with the production bench (clipped Plummer, leaf 64 cells, p6),
as rel-L2 against an fp64 direct sum on 4,096 targets:

| N | budget 0 | default budget, before | default budget, after |
| --- | --- | --- | --- |
| 5e4 | 1.6e-3 | 0.113 | 1.6e-3, bitwise equal to budget 0 |
| 2e5 | 7.8e-4 | 0.070 | 7.8e-4, bitwise equal to budget 0 |

The 5e4 value equals the #376 bug's to all printed digits: the same placeholder read
through another door.

**Why nothing caught it.** The bench harness (`compare_force.FAST_LANE_ENV`), and with
it every GPU pin, sets the budget to 0. Production callers that do not set it were
affected. That includes Odisseo's production lane since Odisseo #29, which stopped
setting jaccpot env vars, and any plain `large_n_gpu` user on Ampere.

**Fix.** `_build_radix_fast_lane_payloads` takes the prepare's `csr_lane` decision and
never materialises the payload under it.

**Checks:**
- Budget 0 is bitwise unchanged against `main`.
- Odisseo's production lane at 2e5 (4 steps, its defaults) is now bitwise equal to its
  budget-0 run. On `main` it differed by 5.7e-5 in the final state.
- CPU suite: 2,322 passed; the one failure is the stale nornax checkout.
- New test: `test_the_csr_lane_ignores_the_payload_budget` forces the CSR lane in
  Pallas interpret mode on CPU. It fails on `main`.

**Lesson for the pins.** A pin that sets an env knob the production caller does not
set is blind to that knob's default. At least one pin should run with a clean
environment, which is what D2's `--library-defaults` checks.

**Pin S6 (added after #384).** S6 is S1 run with `bench/fused_memory_budget.py
--clean-env`: no `JACCPOT_*` or `YGGDRAX_*` variable at all, and the bench's solver
configuration unchanged. Recorded at `main` 17de581 with an A-vs-A control, which is
bitwise. S6 is **bitwise equal to S1**, the full harness env, in force and scan state.
So the library's defaults now reproduce the benches' configuration exactly. Later
phases compare S6 against `main-17de581` and S1-S5 against `main-15ceca4`. Summary:
`bench/results/dce/pins_main-17de581.json`.

### X5: superseded GPU kernel variants

Each switch keeps exactly its default branch, and every pure-JAX CPU / pre-Ampere
fallback stays. A removed value no longer falls back to the default: it raises a
`ValueError` naming the removal, through the new reader `jaccpot._env.env_reject_removed`
("`NAME='v'` was removed in the 2026-10 cleanup (X5); the default is ... Unset the
variable."). `_env`'s docstring records it as the one reader that raises. Where a knob
is read at trace time inside a jitted function (most of these), the refusal fires on the
next trace, which is when the removed value used to take effect.

#### X5a: the fused single-GPU lane's kernels

**Removed** (library -1,765 / +501 lines; the additions are the removal checks, their
docstrings, and 44 lines of `_env.env_reject_removed`):

1. **Cascade level forward kernels** (`pallas/cascade_real_level.py` -467 / +95,
   `cascade_real_lanes.py` docstrings). `_m2m_level_kernel`, `_l2l_level_kernel`,
   `_level_call`, `m2m_real_levels_pallas`, `l2l_real_levels_pallas` and the dispatch
   in `_m2m_forward` / `_l2l_forward` (`_cascade_forward_kind` is now
   `_cascade_lanes_k`). The custom VJPs' forward is the lane kernel only.
2. **CSR M2L pair and tiled kernels** (-671 / +100). `pallas/m2l_real_csr_tiled.py`
   (-380, whole); `_m2l_real_csr_kernel` and `m2l_real_csr_pallas` in
   `pallas/m2l_real_csr.py`; the pair / tiled branches and the chooser
   `_m2l_csr_kernel_choice` in `runtime/kernels/_downward_prep.py` (now
   `_check_m2l_csr_kernel_env`). The tiled kernel's tuning knobs
   `JACCPOT_M2L_CSR_TILE` / `_DOT` are read nowhere.
3. **Per-leaf P2M forward** (`pallas/p2m_real_leaf.py` -138 / +57). `_p2m_leaf_kernel`
   and the `block=0` branch; `leaf_width` stays in the signature (the reverse reads it).
4. **Near-field direct-kernel variants and the `sorted` layout** (-425 / +158;
   `pallas/nearfield_leafpair_csr.py`, `nearfield/_fast_lane.py`).
   `nearfield_leafpair_csr_sorted_pallas` (the sorted ranges with per-chunk partials)
   and its `_fast_lane` branch; the kernel's wide accumulator, which only that entry
   passed; target classes (`target_classes=`, `JACCPOT_NEARFIELD_TARGET_CLASSES`); the
   source flag `g` (2D-indexed operands); `p` without `r` (`p` alongside `r` is accepted
   and adds nothing: `r` prefetches); chunked rows (`JACCPOT_NEARFIELD_DIRECT_ROWS`).
5. **Unfused walk emission** (`pallas/mutual_walk_pallas.py` -57 / +37). The
   per-child claims and unconditional overflow atomics, and the `fused_emit`
   parameter of `mutual_walk_pallas`, `_mutual_walk_jit`, `_round_kernel` and
   `_round_block`.

**Values that now raise:** `JACCPOT_CASCADE_KERNEL=level`;
`JACCPOT_M2L_CSR_KERNEL=pair|tiled`, `JACCPOT_M2L_CSR_TILED=1`;
`JACCPOT_P2M_BLOCK=0` (and negative) and `block=0`;
`JACCPOT_NEARFIELD_LAYOUT=sorted`, `JACCPOT_NEARFIELD_TARGET_CLASSES=<anything>`,
`JACCPOT_NEARFIELD_DIRECT_ROWS=chunked`, `source_flags` / `JACCPOT_NEARFIELD_SOURCE_FLAGS`
with `g` or with `p` but not `r`; `JACCPOT_WALK_FUSED_EMIT=0`.

**Behaviour change:** `JACCPOT_NEARFIELD_ACCUM=wide` routed the fused lane's CSR near
field to the `sorted` layout, because the direct kernel accumulates in the input dtype.
It now runs the `table` layout, which carries the wide accumulator and is what the
multi-GPU cross term runs for the same variable. The bitwise A/B below compares the
two.

**Kept:** the lane cascade forward and the level REVERSE kernels with everything they
use (`_full`, `_level_call2`, `_translate_rows`, `_shift_core`, `cascade_level_tables`,
`_core_tables_to_jnp`, the `*_centred_pair_jax` twins); `JACCPOT_CASCADE_PALLAS` (0 is
the pure-JAX loops, the CPU path and `GradConfig.cascade_pallas`); in
`m2l_real_csr.py` the CSR helpers, the centred layout and its tables, the per-pair
algebra, both twins and `pallas_m2l_real_csr_supported` (the sm_80 gate the other real
Pallas kernels and their lanes import); `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR`; `_DEFAULT_WARPS` and the per-leaf P2M
reverse; the near field's scalar source loop (the bitwise bridge between the direct
kernel and the table kernel), the `table` layout (forced on differentiable calls and
without leaf ranges; `distributed/cross.py` calls it directly; its custom VJP is the
grad lane), the blocked output path (the pieces of rows past `row_limit`);
`node_layout="soa"` (the only fp64 walk layout).

**Tests** (deleted or merged: 37 cases; re-anchored or re-pointed: 44; only renamed: 19;
added: 11):

| test | tag | owner / note |
| --- | --- | --- |
| `test_cascade_real_level.py`: `test_m2m_levels_interpret_matches_the_level_loop`, `test_l2l_levels_interpret_matches_the_cascade`, `test_levels_are_jittable` (5) | (b) | moved onto the lane kernel in place, same ids and tolerances; still the forward-vs-loop anchors |
| `test_cascade_real_lanes.py::test_m2m_lanes_match_the_level_kernel`, `test_l2l_lanes_match_the_level_kernel` (8) | (b) | now `test_m2m_lanes_match_the_level_loop` / `test_l2l_lanes_match_the_cascade`: the same cases and tolerance against `aggregate_m2m_real_by_level` / `_propagate_solidfmm_locals_by_level` |
| `test_cascade_real_lanes.py::test_lanes_jit_and_the_vjp_forward_switch` | (b) | now `test_lanes_jit_and_the_vjp_forward`: the `level` half asserts the removal |
| `test_m2l_real_csr_tiled.py` (5) | (a) | |
| `test_m2l_real_csr_pallas.py::test_csr_pallas_interpret_matches_twin_and_rot_scale_f32` (2) | (c) | `test_m2l_real_csr_lanes.py::test_lanes_match_the_reference_on_ragged_rows` (fp32 vs the twin, orders 2, 3, 5, 6) and `test_csr_pair_twin_matches_rot_scale_f64` (the twin vs rot-scale) |
| `test_m2l_real_csr_pallas.py`: `..._interpret_matches_rot_scale_f64` (5), `..._active_pair_count_truncates`, `..._on_axis_deltas_are_exact`, `..._under_jit_with_traced_active_count`, `..._gpu_matches_rot_scale` (3, GPU), `..._rejects_a_coefficient_count_of_another_order`, `..._rejects_misaligned_centers` | (b) | re-pointed to `m2l_real_csr_lanes_pallas`, same ids and tolerances: no lane test asserted fp64 1e-10 against rot-scale, on-axis exactness at fp64, a traced active count under `jit` or the input validation |
| `test_m2l_csr_lane_wiring.py::test_csr_lane_is_taken_and_matches_the_chunked_lane` | adapted | counts `m2l_real_csr_lanes_pallas_cvjp` (the only CPU wiring test of the lane; it pinned `KERNEL=pair`); also asserts the three removed values raise |
| `test_p2m_real_leaf.py::test_blocked_p2m_matches_the_per_leaf_kernel` (6) | (b) | now `test_blocked_p2m_matches_the_batched_reference`, against `_p2m_leaves_real` at the same atol 2e-6 (measured 1.55e-6 at p5, 1.19e-6 at p4) |
| `test_p2m_real_leaf.py::test_blocked_p2m_with_the_full_leaf_width_is_the_per_leaf_sum` | (a) | |
| `test_pallas_nearfield_leafpair_csr_sorted.py::test_sorted_ranges_equal_the_table_kernel` (4) | (a) | the table kernel's `accum="wide"` keeps `test_pallas_nearfield_leafpair_csr.py`'s test |
| `...::test_direct_equals_the_table_kernel_in_particle_order[chunked-*]` (4) | (a) | the `[whole-*]` cases stay, without the `rows` parameter |
| `...::test_source_tiles_equal_the_scalar_loop`, cases `(8, 4, 8, "p")`, `(16, 8, 8, "alg")`, `(16, None, 4, "aglr", (4, 16))` (6) | (a) | the three other class cases keep their flags in one launch (b); the rest lose only the `classes` id |
| `...::test_source_flags_are_validated`, `test_the_default_is_the_tiled_kernel` | adapted | the class check is gone; the removed flags are asserted to raise |
| `test_softening_kernel_paths.py::test_csr_sorted_direct_in_every_source_tile_mode[4-alg-*]` (4), `test_nearfield_force_scale_lane.py::test_force_scale_lane_equals_the_near_pair_sum[4-alg-*]` (6) | (a) | the `l` and `alr` cases stay |
| `test_mutual_walk_pallas.py::test_pallas_walk_lists_equal_the_flat_walk_as_sets[record-False-*]` (4) | (b) | `[record-*]` (was `[record-True-*]`); `[soa-False-*]` is now `[soa-*]`, fused |
| `test_mutual_walk_pallas.py::test_pallas_walk_flags_overflow[False]` | (a) | `[True]` stays, unparametrised, and asserts the removal |

Added: `test_env_readers.py::TestEnvRejectRemoved` (2 tests, 6 cases);
`test_p2m_real_leaf.py::test_the_removed_per_leaf_forward_raises[0, -1]`;
`test_pallas_nearfield_leafpair_csr_sorted.py::test_the_removed_layout_options_raise`
(2); `test_nearfield_csr_lane_consistency.py::test_wide_accumulation_runs_the_table_layout`
(the CSR lane in interpret mode on the lane golden's solver: `wide` enters the table
kernel and not the direct one, within 1e-5 of the default and within 1e-2 of direct
summation; `LAYOUT=sorted` raises). No test asserted that `wide` still evaluates.

**Bench:** `grad_cascade_reverse_microbench.py` times the lane forwards
(`k_lanes = 32 x warps`, as the custom VJP sets them) instead of the level forwards;
`m2l_csr_microbench.py` times `m2l_real_csr_lanes_pallas`; `nearfield_kernel_tune.py`
lost the `classes` field (its scalar default stays); `walk_tune.py` lost the
`fused_emit` field; `gpu_gate.py`'s comment names the re-pointed CSR test.

#### X5b: mutual, MAC, COM and rotation

**Removed** (library -1,038 / +225 lines):

6. **COM radii chain kernel** (`pallas/com_radii_leaf.py` -196 / +21, with item 7 in
   `runtime/_mac_geometry.py` -170 / +63). `_com_radii_chunk_kernel`,
   `com_radii_chunk_pallas`, `_com_radii_variant` and `_com_radii`'s `variant`.
7. **MAC radius bound.** `mac_radius_mode`, `_node_depths` (in `_mac_geometry.py`;
   `mutual/topology.py`'s live `_node_depths` is untouched), the bound branch of
   `_com_radii`, `_MAC_RADIUS_*`, and the `internal=` parameter of `com_mac_geometry`
   and `_com_radii`. Radii are exact on every node.
8. **Degree-batched rotation** (`operators/m2l_real_rot_scale.py` -198 / +29).
   `_degree_batched`, `_centred_degree_maps`, `_padded_dz`, `_rotate_degree_batched`, the
   branches in the two single-pair rotations, and four imports only they used. The
   switch was reachable (env only) from every lane, the mesh lane and the mutual jax
   M2L included; its default branch is unchanged.
9. **The z-core Pallas M2L** (-401 / +46). `pallas/m2l_core_z_real.py` (-357, whole,
   with `m2l_core_z_real_pallas[_cvjp]` and `pallas_m2l_real_supported`); the `zcore`
   lane of `mutual/farfield.py`; the package export. `pallas_m2l_real_supported` (a
   gpu/tpu backend check) had no library user left; its two other users, one example
   and one bench section, asked whether `use_pallas` does anything for the real basis,
   which is the sm_80 gate `pallas_m2l_real_csr_supported` -- they use that now, so the
   gpu/tpu check was not moved. Docstrings in `operators/real_translations.py`,
   `m2l_real_rot_scale.py` and `runtime/kernels/_m2l.py` no longer name the kernel.
10. **Three small branches** (-73 / +66): the gather unpermute of the fast lane
    (`runtime/_large_n_pipeline.py`; `unpermute` also left the compiled evaluation's
    cache key); the direct leaf flatten of `_evaluate_local_expansions_for_particles`
    (`runtime/kernels/_evaluate.py`; wrong for non-full leaves); the unsorted Pallas walk
    rows (`runtime/_interaction_cache.py`: `strict_walk_deterministic_rows` is gone, the
    rows are always sorted, `_reject_nondeterministic_walk_rows` refuses the switch).

**Values that now raise:** `JACCPOT_COM_RADII_VARIANT=chain`;
`JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS=bound` (and any value but `exact`, as before);
`JACCPOT_M2L_DEGREE_BATCHED=1`; `JACCPOT_MUTUAL_M2L=zcore` (where the variable is read,
i.e. with `use_pallas`); `JACCPOT_FASTLANE_UNPERMUTE=gather`;
`JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN=1`;
`JACCPOT_STATIC_STRICT_FUSED_WALK_DETERMINISTIC=0`.

**Kept:** `pallas/com_radii_leaf.py`'s table kernel, `_next_pow2` and
`JACCPOT_COM_RADII_KERNEL` (auto: Pallas on Ampere, XLA on CPU); the unrolled
per-degree rotations; the pure-JAX `m2l_core_z_real` operator; `JACCPOT_MUTUAL_M2L`'s
`auto` / `jax` / `fused`; the `JACCPOT_L2P_LAYOUT` leaf branch (the mesh lane, the grad
path and complex coefficients reach it; phase C); `LOCAL_EVAL_FLAT_ANALYTIC`,
`DTYPE_PRESERVE` and `ORDER4_UNROLLED` (complex only; phase C).

**Tests** (deleted: 15 cases; re-anchored: 6; moved or renamed: 3; added: 5):

| test | tag | owner / note |
| --- | --- | --- |
| `test_mac_geometry_com.py`: `test_com_geometry_bounds_every_node_and_is_exact_on_leaves[bound]`, `test_com_geometry_is_jittable[bound]` | (a) | the `[exact]` cases stay, unparametrised |
| `test_mac_geometry_com.py::test_internal_mode_is_validated` | (a) | the `internal=` parameter is gone; `test_mode_knob` asserts the env refusal |
| `test_mac_geometry_com.py::test_mode_knob` | adapted | the bound-vs-exact lines became: `exact` equals `com_mac_geometry`, `bound` raises |
| `test_mac_geometry_com.py::test_the_ancestor_table_kernel_is_the_chain_kernel` (6) | (b) | now `test_the_table_kernel_tiling_does_not_move_a_bit`: the table kernel at the same `(lanes, block)` tilings equals its default tiling to the bit. The survey proposed (a) with `test_pallas_chunks_equal_the_level_passes` as owner, but that test runs only the default tiling, and the multi-trip tilings had no other test |
| `test_shared_env_switches.py::TestTheDegreeBatchedSwitchIsReadAtCallTime`: `test_it_is_off_by_default`, `test_setting_it_after_import_is_honoured` | (a) | replaced by `TestTheDegreeBatchedSwitchWasRemoved::test_setting_it_after_import_raises` |
| `...::test_the_module_captures_no_env_value_at_import` | (b) | moved, unchanged, to `TestTheDegreeBatchedSwitchWasRemoved` |
| `test_shared_env_switches.py::TestTheTwoRotationPathsAgree::test_batched_matches_unrolled_at_the_noise_floor` | (a) | |
| `test_pallas_m2l_core_z_real.py` (5) | (a) | |
| `test_custom_vjp_parity.py::test_m2l_core_z_pallas_custom_vjp_matches_twin` (4) | (a) | the fused real M2L keeps its cvjp parity test |
| `test_mutual_walk_pallas.py::test_walk_backend_flag_parsing` | adapted | the deterministic half asserts the removal |

Added: `test_mac_geometry_com.py::test_the_removed_chain_variant_raises`,
`test_shared_env_switches.py::TestTheDegreeBatchedSwitchWasRemoved::test_setting_it_after_import_raises`,
`test_mutual_fmm.py::test_the_removed_zcore_lane_raises`,
`test_large_n_fast_path_policy.py::test_the_removed_gather_unpermute_raises`,
`test_l2p_particle_major.py::test_the_removed_leaf_flatten_switch_raises` (on a shape no
other test in that module traces, since the function is jitted).

**Bench and examples:** `com_radii_tune.py` lost the `chain` variant and ignores a
capture's `internal` / `variant`; `m2l_csr_microbench.py` lost the degree-batched arm;
`bench_mutual_backends.py` lost the `pallas-zcore` lane; `bench_real_vs_complex.py` and
`examples/pallas_m2l_speed.py` use `pallas_m2l_real_csr_supported`;
`profile_large_n_nearfield_stages.py`'s B5 note says the flag is gone.

**Docs:** `momentum_conserving_fmm.md` (the `zcore` lane and the degree-batched flag),
`differentiable_fmm_design.md`, `agent_guides/STYLE_GUIDE.md` (the z-core import it
listed as an open layering question), and annotations in the round records
`fused_memory_2026-10.md`, `sub10ms_2026-09.md`, `small_leaves_2026-09.md` where they
say a removed value "restores" something. ARCHITECTURE.md and README.md named none of
the removed switches. The other plan and audit documents stay as written.

#### X5 gates

- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`, at
  9979288): 2,293 passed, 198 skipped, 1 failed: the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as before.
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`) on the 16 touched unit test
  files: 255 passed, 48 skipped.
- **Distributed tiers** on four forced host devices (`tests/distributed` and
  `test_mutual_distributed.py`, `-n 6`): 95 passed, 3 skipped (two opt-in upstream
  checks and one sm_80-only case).
- **Bitwise A/B against 3d002fe on CPU** (a `git archive` export): 338 arrays equal, and
  the diagnostic rows (fused mode active, no fallback, in every large-N run). The
  arrays:
  - forces and potentials of 10 configurations on the general path: the default; fast
    and accurate on both bases; balanced; kd-tree; `dehnen_error`; `dehnen_paper`
    adaptive order; `large_n_gpu` in fp32;
  - a prepared state with a target subset; position and mass gradients through
    `differentiable_accelerations` on both bases; the real rotate/scale M2L at orders
    2, 4 and 6 (the degree-batched switch's default branch);
  - the large-N lane on CPU with the GPU gate opened (N = 2048): prepare + evaluate, a
    same-topology refresh, two `strict_prepare_refresh_and_evaluate` calls, the fused
    `strict_run_v2` state after 3 steps, and the prepacked differentiable lane's
    forces and position / mass gradients;
  - the same lane with every surviving Pallas kernel in interpret mode (N = 768,
    `use_pallas=True`, the prepacked payload), twice: without and with
    `JACCPOT_NEARFIELD_LEAFPAIR_CSR=1`. A counting run confirmed what the second one
    enters: the blocked P2M, both lane cascades, the CSR lane M2L, the direct near
    field, the Pallas walk and, in the fused scan, the COM radii table kernel (the
    L2P kernel and the differentiable lane run in the same interpret environment);
  - `JACCPOT_NEARFIELD_ACCUM=wide` through the CSR lane in interpret mode: the
    `sorted` layout at 3d002fe and the `table` layout here give the same bits;
  - kernel level, in fp32 and fp64: the lane cascades through their custom VJPs at
    orders 3 and 5 (forward and both reverse halves) and at `k_lanes=7`; the CSR lane
    M2L at orders 2 and 5 (forward, both reverse halves); the blocked P2M at orders 4
    and 5 (default and `block=4, chunk=4`, and the three reverse halves); the direct
    near field in nine configurations (the default; the scalar loop; source tiles 4
    with no flags, `a`, `r`, `l`; 8 with `alr`; 32 with `apr`; row limit 2), each for
    the acceleration, the potential and the force scale, at two leaf widths; the table
    kernel with `input` and `wide` accumulation; the COM radii from the XLA passes and
    from the table kernel; the Pallas walk's sorted far and near lists, counts, rounds
    and peak wavefront in both node layouts;
  - the block-step lane: `BlockStepFMM(backend="jax")` at N = 512 and
    `backend="pallas", pallas_interpret=True` at N = 256 -- the total accelerations and
    one base step's positions and velocities;
  - a supplement of 12 arrays for the L2L reverse with live internal locals (in the
    main run the internal locals were zero, so its centre cotangent was zero on both
    sides).
- **`test_shards.py check`:** 2,554 tests, each in exactly one shard. At 3d002fe there
  are 2,590; this phase deletes 52 cases and adds 16 (53 more are renamed).
- **Goldens.** `golden/`, `golden_grad/`, `golden_lanes/` and `golden_modes/` are
  byte-identical; `constructor_state.json` did not change and was not regenerated.
- `import jaccpot` and all 123 modules import; no file in `jaccpot/`, `tests/`,
  `bench/`, `examples/` or `notebooks/` imports a removed name (only prose mentions
  remain). pre-commit is clean on every changed file.
- `bench/annotation_census.py`: shape-annotated array parameters 833 (unchanged), bare
  `Array` parameters 1,761 -> 1,701 (shaped share 32.1 % -> 32.9 %), `@jaxtyped`
  functions 183 (unchanged). Only deletions moved them.
- **Benches on CPU:** `bench_real_vs_complex.py` (run from a copy: the script prepends
  the sibling `../yggdrax` checkout, which here is not the yggdrax under test) and
  `bench_mutual_backends.py --sizes 1000` run. The GPU-only scripts
  (`grad_cascade_reverse_microbench.py`, `m2l_csr_microbench.py`,
  `nearfield_kernel_tune.py`, `walk_tune.py`, `com_radii_tune.py`) compile and their
  jaccpot imports resolve; they were not run.
- **GPU pins** (frozen worktree at ff710d5, one A100, against `main-15ceca4` with its
  A-vs-A control): S1-S5 **bitwise**. M1 and M3 (two cards) are pending; they run when
  a second card is free under the card rules.
- **Speed (G5), one A100**, interleaved main / X5 x 3 on the same card. The card was
  shared with another user's idle job (0 % util at each sample). The timed region is
  `fused_memory_budget.py` scan min, ms/step:

  | N | main | X5 |
  | --- | --- | --- |
  | 2e5 | 6.22, 6.67, 6.25 (median 6.25) | 6.39, 6.14, 6.18 (median 6.18) |
  | 8e6 | 83.79, 83.21, 83.18 (median 83.21) | 83.10, 82.73, 83.61 (median 83.10) |

  Equal within the run-to-run spread.
  - The XLA programs around the kernels are identical: optimised HLO with source
    metadata stripped, and modules whose fusion choice the autotuner flipped compared
    before optimisation.
  - The serialised Pallas kernels differ in their embedded source-line numbers, as
    expected after editing those files.

**Left for phase Z:**
- `bench/cleanup_inventory.py`'s `kernel_variants` family still lists the removed files
  and functions, as the earlier families do: it is the record of what the phases
  removed.
- Odisseo's `tools/walltime_ab_compare.py` sets `JACCPOT_LOCAL_EVAL_DIRECT_LEAF_FLATTEN=1`
  in one arm; that arm now raises. Out of this repo.
- `bench/results/` keeps profiles that name the removed kernels.

### X6: host-routed strict refresh, the dual-downward planner, unsafe compact-pair reuse, the non-fused strict loop

The strict lane now has one path: the fused scan, whose traced refresh takes the
device-only streamed fast path and builds a fresh far-pair list every step. A
value that selected one of the removed paths raises a `ValueError` naming the
variable and X6 (`jaccpot._env.env_reject_removed`), at every door into the fused
lane: `strict_run_v2`, `strict_fused_prepared_eval_fn` (through which the
multi-GPU lane's `measure_shard_plan` prepares its shards), and a fused-device
refresh (`_refresh_large_n_same_topology(fused_device_mode=True)`, the multi-GPU
lane's per-force door). The check is `_reject_removed_strict_lane_env` in
`fmm_strict_run.py`; the two far-pair switches are also refused by their reader,
`_fresh_compact_pair_rebuild_enabled`. A switch whose other values only made
something inert, or that a caller reads for itself, is accepted and ignored.

**Removed** (library -746 / +178 lines, not counting the re-indent of two
`try` bodies; raw -974 / +406):

1. **The host-routed strict refresh** (-51 / +58 with items 5c and 5d; the
   additions are the refusal helper, its table and comments).
   `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY` and `_strict_fused_device_only`.
   `0` sent the fused lane's traced refresh through the generic dual/downward
   build, about 10x slower (the 2026-07 A100 note it carried: 1224 -> 119
   ms/step at 2e5). The hot-path predicate in `_resolve_dual_downward_plan` is
   now `suppress_host_side_effects and _strict_fused_mode_active`, and the
   mixed-order bucketing reads `_strict_fused_mode_active` alone. The generic
   dual/downward code stays (the eager prepare and the traced unsuppressed path
   use it), and so do its `suppress_host_side_effects` guards.
2. **The dual-downward refresh planner** (-367 / +2).
   `_RefreshDualPlannerHint`, `_compiled_refresh_dual_planner_route`, the
   `planner_hint` parameter of `_build_dual_tree_artifacts`,
   `_resolve_dual_downward_planner_hint`, the steady-state timing bypass, the
   two constant hints, eleven engine attributes and their counters, and the
   `JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE` and
   `..._STEADY_NO_SUBSTAGE_TIMING` reads. The planner was enabled only on the
   large-N production profile with a static-radix tree, which is where the strict
   streamed fast path returns before it. Per-test coverage of the 20 strict-lane
   test files at 17de581: no test reached its compiled route or its fast-lane
   branch; only four counter bumps on the strict fast path ran. Its route,
   `allow & ~need_traversal_result`, is `_can_split_dual_tree_build`, which now
   decides every build; the constant hints equalled `_can_split(True, False)`.
   The substage timing callback is always passed (it records only while refresh
   timing is on).
3. **Unsafe compact far-pair reuse** (-177 / +61). The reuse branch of the
   refresh (it re-used the carried far list after the drift: stale M2L pairs),
   its refusal, its reuse-only stage timing, and the
   `static_radix_compact_pair_reuse_{hits,misses}` counters and diagnostics keys
   (emitted twice). `JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH=1`
   only switched the compact streamed pairs off, which made the fused refresh
   raise; its block in the plan went too.
4. **The non-fused `strict_run_v2` loop** (-156 / +62). The host-driven
   per-step loop `JACCPOT_STATIC_STRICT_FUSED_MODE=off` selected, and the
   engine's `_strict_fused_mode_raw` / `_enabled`. `_strict_fused_mode_active` is
   now the profile-set verdict; where it only told the loop from the scan it is
   folded (`fused_device_mode=True`, the force-scale carry, the particle carry,
   the `LargeNPreparedState` guard), and the "carry='particles' needs the strict
   fused lane" raise is dead and gone. The attribute stays: prepares and
   refreshes after a run read it. Per-test coverage at 17de581: no test executed
   the loop.
5. **Other strict machinery.**
   - (a) `JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK` chose between
     re-raising a failed scan's exception and wrapping it in a `RuntimeError`
     ("... while host fallback is disallowed"). There was no host segment
     fallback either way. Both `try` blocks went; a failed scan raises its own
     exception.
   - (b) `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING`: host stage timers
     never run under `fused_device_mode` (the refresh and the large-N prepare).
   - (c) The strict plan's process-wide write of
     `YGGDRAX_DUAL_TREE_SHARED_COUNT_FILL_{ONE_SHOT,STEADY_SINGLE_QUEUE}`. Only
     yggdrax's bounded count passes read them, and those need an explicit
     traversal config; the strict lane's walks pass none. So the write reached
     only later generic builds in the same process, which is the flake
     `tests/conftest.py`'s env isolation was written for (its docstring now says
     so). No library code writes `os.environ` any more.
   - (d) `JACCPOT_STATIC_STRICT_GPU_MODE`: strict mode is now exactly its old
     `auto`, the large-N production profile on a static-radix tree.

**Env policy:**

| variable | accepted | raises |
| --- | --- | --- |
| `JACCPOT_STATIC_STRICT_FUSED_MODE` | unset, `on`, `1`, `true`, `yes` | `off`, `0`, `false`, `no` |
| `JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY` | unset, `1`, ... | `0`, `false`, `no`, `off` |
| `JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING` | unset, `1`, ... | `0`, `false`, `no`, `off` |
| `JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE` | unset, `0`, ... | `1`, `true`, `yes`, `on` |
| `JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH` | unset, `0`, ... | `1`, `true`, `yes`, `on` |
| `JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD` | unset, `1`, ... | `0`, `false`, `no`, `off` |
| `JACCPOT_STATIC_STRICT_GPU_MODE` | any value, ignored | never |
| `JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK` | any value, ignored | never |
| `JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS` | any value, ignored | never |
| `JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE`, `..._STEADY_NO_SUBSTAGE_TIMING` | any value, ignored | never |

The accepted values are what Odisseo's env block (`jaccpot_coupling.py`), the bench
harness (`compare_force.FAST_LANE_ENV`, `apply_fast_lane_env`) and pins S1, S2 and
S4b set.
`GPU_MODE` is never refused because Odisseo reads it for its own lane gate. The
planner modes gave identical forces. `REUSE_COMPACT_PAIRS` only armed the reuse
together with the unsafe opt-in.

`FRESH_COMPACT_PAIR_REBUILD=0` is refused rather than kept. With the reuse gone, `0`
would have been a newly reachable mode: the refresh returning its fresh list in the
state, so the list rides in the scan's carry instead of outside it. It was reachable
before only with `REUSE_COMPACT_PAIRS=0` as well. Nothing sets it: not jaccpot's
library, tests (except the deleted reuse refusal) or benches, not the bench harness
under `/export/home/tbuck/Odisseo-bench-multigpu`, and not the Odisseo checkouts.
Their only mention is an archived 2026-06 handoff note that describes the default.

**Behaviour changes:**
- `GPU_MODE=on` no longer opens strict mode on other engines. On those it meant
  `fail_fast`, no retry logger, and the strict streamed path when the build allowed
  it. `GPU_MODE=off` no longer closes strict mode on the production profile. Before
  X6 that made the fused scan's refresh raise ("strict_mode_inactive").
- Under `DISALLOW_HOST_SEGMENT_FALLBACK=1` (Odisseo's and the harness's setting) a
  failed fused scan used to surface as a chained `RuntimeError`. It now raises its
  own exception.
- `DISABLE_HOT_TIMING`: the refresh timers no longer run under `fused_device_mode`
  even when refresh timing is on and the variable said `0`. Only eager calls ever
  timed there: a prepare or refresh with `fused_device_mode=True` (passed, or
  inherited by `strict_prepare_refresh_and_evaluate` after the engine ran the fused
  lane).
- A generic build with an explicit traversal config, later in a process that ran a
  strict prepare, used to take yggdrax's one-shot, single-queue count pass. It now
  takes the default queue ladder, as in a process that never ran a strict prepare.
- The profile-set refusals of `strict_run_v2` and `strict_fused_prepared_eval_fn`
  no longer name a fused-mode switch or "a slower non-fused path"; both name the
  set, the N and the two remedies.

**Kept:** `strict_prepare_refresh_and_evaluate`, `refresh_prepared_state`, and the
non-fused refresh branch of `_refresh_large_n_same_topology` they take with
`fused_device_mode=False`; `_velocity_verlet_state_update` in `fmm_state.py`
(Odisseo imports it; `test_downstream_contract.py` pins it); the
`JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET` gate and its refusals; the fused lane's
jit cache-key layout (`bench/fused_memory_budget.py` reads `num_steps` at index 6);
`_strict_far_pairs_ride_outside_the_scan`; `_strict_fused_validated_caps`;
`_prepare_state_dual_and_downward_strict_streamed_fast` and
`_build_flat_walk_artifacts_strict_streamed` (the budget bench wraps both);
`strict_fused_prepared_eval_fn`; `_strict_fused_device_refresh_route_count`; the
fastlane diagnostics; `refresh_strict_mode_active_count`.

**The fused scan covers the loop.** Everything the loop did, the scan does:

| feature | non-fused loop | fused scan |
| --- | --- | --- |
| external field | yes | yes |
| `initial_self_acceleration` | yes | yes |
| `return_history` | yes | yes |
| `prepared_state` in, `return_prepared_state` | yes | yes |
| `step_callback` | ignored | yes |
| `carry="particles"` | refused | yes |
| `donate_prepared_state` / `donate_state` | ignored | yes |
| `mac_type="dehnen_error"` force-scale carry | not carried (kept the state's scale) | yes |
| capacity segment retry | no | yes |
| refresh diag modes | yes | yes |

The loop's update was `_velocity_verlet_state_update` (positions recomputed from the
old ones); the scan kicks the drifted state (`_velocity_verlet_kick_drifted`), the
same scheme.

**Diagnostics keys that disappear:** `refresh_dual_planner_{cache_hits,
cache_misses, compile_count, execute_count, steady_timing_bypass_count,
compiled_route_count}`, `strict_fused_planner_bypassed_count`,
`static_radix_compact_pair_reuse_{hits,misses}`. Odisseo's `jaccpot_coupling.py`
reads every one with `.get(..., 0)`, so its records show 0. The harness's
`probe_step.py` reads two planner keys with `.get`, which gives `None`.

**Tests** (deleted: 5; retargeted: 1; adapted: 3; added: 1 file, 24 cases):

| test | tag | owner / note |
| --- | --- | --- |
| `test_fmm.py::test_static_radix_refresh_dual_planner_mode_parity_and_diagnostics` (also out of `slow_tests.txt`) | (b) | `test_strict_refresh_faces.py::test_refresh_face_reproduces_a_fresh_prepare` owns the refresh parity (against a fresh prepare, 1e-6, stricter than the planner test's 1e-5 between two refreshes), and now also asserts that the refresh ran the strict plan (`refresh_strict_mode_active_count` + 1). Both planner modes ran the strict fast path, so the planner test compared one path with itself |
| `test_device_only_default.py`: `test_strict_fused_device_only_defaults_on`, `test_strict_fused_device_only_env_opt_out` | (b) | `test_strict_lane_removed_switches.py`: the default is `test_accepted_values_run_the_same_fused_scan` (the unset env and `DEVICE_ONLY=1` give the same scan, bitwise); the opt-out is `test_a_removed_value_raises_at_every_fused_entry[DEVICE_ONLY=0]` |
| `test_same_topology_refresh_modes.py::test_compact_pair_reuse_refuses_without_the_unsafe_opt_in`, `::test_compact_pair_reuse_is_taken_when_explicitly_allowed` | (a) | the module keeps `test_upward_only_diagnostic_returns_after_the_upward_sweep`; its docstring says what went |
| `test_strict_fused_eval_fn.py::test_requires_fused_mode_to_be_active` | (b) | now `test_requires_the_particle_count_in_the_profile_set`: the guard it reached is the profile set's (variable, N and remedies in the message, mode left inactive), and `FUSED_MODE=off` is asserted to raise |
| `test_strict_carry_helpers.py::test_fresh_rebuild_flag` | adapted | five cases: unset, `on`, `1` return `True`; `FRESH=0` and `UNSAFE=1` raise |
| `test_strict_run_fail_fast.py` (`TestFusedModeRefusesToDegradeSilently`, `TestProfileKeyAccounting`) | adapted | no longer set the removed `_strict_fused_mode_enabled`; match the reworded refusal; module docstring |
| `test_fmm.py::test_strict_run_v2_api`, `::test_strict_fused_compact_far_pair_cap_fails` | unchanged | both ran the fused scan; the second sets `DISALLOW=0`, now inert |

Added: `tests/integration/test_strict_lane_removed_switches.py`:
- `test_a_removed_value_raises_at_every_fused_entry`: seven removed values (two
  spellings of `FUSED_MODE`) times three entries, 21 cases, each before any device
  work;
- `test_the_strict_lane_is_the_production_profile_whatever_gpu_mode_says`: `off`
  leaves the strict plan on for the production profile, `on` does not open it on a
  general engine;
- `test_accepted_values_run_the_same_fused_scan[harness, inverted]`: the
  harness's values (and every other switch at its default, named), and the other
  value of each ignored switch, each give the unset environment's fused scan
  bitwise (state, history, force on the returned state).

Against 17de581, 23 of the 24 fail; the `harness` case passes there too, as it
should.

**Goldens.** `golden/`, `golden_grad/`, `golden_lanes/` and the `golden_modes/*.npz`
are byte-identical. `constructor_state.json` was regenerated; its diff is exactly 23
base attributes: `_planner_steady_timing_bypass_enabled`,
`_refresh_dual_planner_{cache, cache_hits, cache_misses, compile_count,
compiled_route_count, execute_count, mode, mode_auto, mode_on,
steady_timing_bypass_count}`, `_static_radix_compact_pair_reuse_{hits,misses}`,
`_strict_fused_device_only`, `_strict_fused_disable_hot_timing`,
`_strict_fused_disallow_host_segment_fallback`, `_strict_fused_mode_{enabled,raw}`,
`_strict_fused_planner_bypassed_count`, `_strict_gpu_mode{,_auto,_on}` and
`_strict_shared_env_applied`. No matrix case and no override moved.

**Bench:**
- `profile_refresh_stage_breakdown.py` no longer sets `FUSED_MODE=off`. It drove
  `strict_prepare_refresh_and_evaluate`, which never read the switch. Smoke-run on
  CPU with the GPU gate opened.
- `cleanup_inventory.py`: a span whose anchor its phase removed contributes
  nothing, as a removed file or function already did. So `nonfused_strict_loop`
  and `dual_planner` stay as the record. Against the coverage file above, both
  report 0 tests.
- `multigpu_fused_ndev2_probe.py`: its comment no longer names the planner as a
  cause of the cold-solver trap.
- The default-valued env lines in `profile_downward_breakdown.py`,
  `profile_fused_gpu_util.py`, `profile_fused_stage_ablation.py` and
  `bench_fused_eval_vs_jaxfmm.py` stay: they set accepted values.
- `dce_pins.py` and `fused_memory_budget.py` are unchanged. S1, S2 and S4b set
  only accepted values, and `--library-defaults` unsets them.

**Docs:**
- `phase5_pallas_plan.md` (`DEVICE_ONLY` is the only path).
- `fmm_fused_perstep_profiling_2026-07-08.md` (the breakdown script times the eager
  face).
- `multigpu_fused_2026-09.md` (the planner site of the cold-solver trap is gone).

`ARCHITECTURE.md` and `README.md` named none of it.

**Downstream:**
- Odisseo's `tools/walltime_ab_compare.py` sets `JACCPOT_STATIC_STRICT_FUSED_MODE=off`
  in its variant arm. That arm now raises, as its `LOCAL_EVAL_DIRECT_LEAF_FLATTEN=1`
  already does since X5. This goes to phase O3.
- Odisseo's env block (`jaccpot_coupling.py`), its
  `tools/agama_ic_sweep_and_render.py` and the harness set only accepted values.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`, at
  482e616): 2,314 passed, 198 skipped, 1 failed: the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as before.
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`) on the two touched unit
  test files: 28 passed.
- **Distributed tier** on four forced host devices (`tests/distributed` and
  `test_mutual_distributed.py`, `-n 6`): 95 passed, 3 skipped (two opt-in upstream
  checks and one sm_80-only case), as at X5.
- **Bitwise A/B against 17de581 on CPU** (`git archive` exports of 17de581 and of
  this branch). Every lane ran twice: with every X6 switch unset, and with the bench
  harness's values (`GPU_MODE=on`, `FUSED_MODE=on`, `DEVICE_ONLY=1`,
  `DISALLOW_HOST_SEGMENT_FALLBACK=1`, `REQUIRE_EXACT_CAP_PROFILE_MATCH=0`,
  `FLAT_COMPACT_FAR_PAIRS=1`, `PAYLOAD_IN_FUSED=1`, `PROFILE_SET=N`). In each env
  arm, 45 arrays are bitwise equal to 17de581's in the same env:
  - the general path: forces and potentials of the default, fast and accurate on
    the real basis, `dehnen_error` and `large_n_gpu` in fp32; a bare
    `FastMultipoleMethod()`; the position and mass gradients through
    `differentiable_accelerations` on both bases (15);
  - `BlockStepFMM(backend="jax")`: total accelerations and one base step (3);
  - the large-N lane with the GPU gate opened (N = 2048, 19):
    - prepare + evaluate, `refresh_prepared_state`, two
      `strict_prepare_refresh_and_evaluate` calls;
    - the fused `strict_run_v2` scan, plain;
    - the scan with history, a `step_callback` (its emitted stream),
      an external field, a given initial self acceleration and the returned
      prepared state (forces on it);
    - `carry="particles"` on cell leaves, two calls chained through the handle;
    - `strict_fused_prepared_eval_fn`;
    - the prepacked differentiable lane's forces and gradients;
  - `mac_type="dehnen_error"` in the fused scan, both carries, through the CSR
    near-field lane in Pallas interpret mode (N = 768): states, the force on the
    returned state, the handle's self-gravity and force scale (5);
  - the multi-GPU fused lane's machinery on one forced host device:
    `setup_fused_force` (`measure_shard_plan` through
    `strict_fused_prepared_eval_fn`, the plan merge, the assembled states) and the
    `shard_map`'d `fused_force_step` (the fused-device refresh with `num_valid`),
    three forces at drifted positions (3). Two devices cannot run on CPU:
    `decompose` repartitions with `ragged_all_to_all`, which XLA:CPU does not
    implement (jax 0.11.2), so the cross field has no CPU arm.

  The diagnostic rows (fused mode active, fallback count, fast-lane hits and misses,
  runner and plan counts, overflow flags, the callback's steps) are equal in the
  unset arm. In the harness arm, 6 of the 69 rows differ, all expected:
  - `refresh_strict_mode_active_count` on the general path (default, fast,
    accurate, `dehnen_error`): 1, 1, 1, 2 at 17de581, where `GPU_MODE=on` opened the
    strict plan on any engine; 0 now.
  - Two cases fail in every arm and tree: the particle carry on non-cell leaves,
    which saturates a capacity on this CPU configuration, and `dehnen_error` without
    the CSR lane, which needs it. At 17de581 under `DISALLOW=1` their messages were
    the wrapper "strict fused velocity-Verlet scan failed while host fallback is
    disallowed". Now they are the errors themselves.

  On this branch the two env arms are bitwise equal to each other, rows included.
- **`test_shards.py check`:** 2,575 tests, each in exactly one shard. The 17de581
  export, with no sibling nornax checkout, collects 2,519; adding the 36 nornax tests
  this checkout collects gives 2,555. X6 deletes 5 cases and adds 25 (the new
  file's 24 and one case of `test_fresh_rebuild_flag`).
- **Inventory:** per-test coverage of the 20 strict-lane test files at 17de581
  (`--cov-context=test`).
  - No test executed the non-fused loop, the planner's compiled route or its
    fast-lane branch, or either `DISALLOW` wrapper.
  - The reuse branch ran only in its own opt-in test, and its refusal only in its
    refusal test (both deleted).
  - The shared-env write ran in 29 tests.
- **Goldens:** `golden/`, `golden_grad/`, `golden_lanes/` and `golden_modes/*.npz`
  are byte-identical; `constructor_state.json` as above.
- `import jaccpot` and all 123 modules import. No file in `jaccpot/`, `tests/`,
  `bench/` or `examples/` imports a removed name; `cleanup_inventory.py` names one
  as a regex. pre-commit is clean on every changed file.
- `bench/annotation_census.py`: shape-annotated array parameters 833 (unchanged),
  bare `Array` parameters 1,701 -> 1,695 (the planner route's six flags), shaped
  share 32.9 % -> 33.0 %, `@jaxtyped` functions 183 (unchanged).
- **GPU pins** (frozen worktree at 18f068a, plus S6's bench changes from #386; one
  A100):
  - S1-S5 are **bitwise** against `main-15ceca4`. S1, S2 and S4b run with the
    harness env, which sets only accepted values.
  - S6 (no `JACCPOT_*` env at all) is **bitwise** against `main-17de581`.
  - M1 and M3 (two cards) are pending. They run when a second card is free under the
    card rules.
- **Speed (G5)**, interleaved main 17de581 / X6 x 3 on the same A100, which was shared
  with another user's idle job. `fused_memory_budget.py` scan min, ms/step:

  | N | main | X6 |
  | --- | --- | --- |
  | 2e5 | 6.38, 6.20, 6.45 (median 6.38) | 6.36, 6.62, 6.03 (median 6.36) |
  | 8e6 | 83.09, 83.03, 83.53 (median 83.09) | 83.54, 83.10, 83.43 (median 83.43) |

  Equal within the run-to-run spread.

**Left for phase Z:**
- the ignored switches' env lines in Odisseo, the harness and four benches;
- `_fresh_compact_pair_rebuild_enabled`, which now always returns `True` or raises;
- the name `strict_fused_device_only_hot_path`;
- `bench/results/` logs that show the removed wrapper message.

### K: the Cartesian basis, dense interactions, multipole_utils

**Behaviour change, first and alone (1c348c7).** `FMMEngine(expansion_basis=...)`
defaulted to `"cartesian"`. It now defaults to `"solidfmm"`, which D1's
`_validate_expansion_family` resolves to `RealSHBasis`. Every engine built without
a basis runs the real basis: `compute_gravitational_acceleration` and every direct
`FMMEngine(...)` caller (tests, benches). `FastMultipoleMethod` always passed a basis
and does not move. Measured on CPU in fp64, theta 0.5, leaf 16, against an fp64
direct sum (rel-L2 of the acceleration, per-particle median in brackets):

| case | Cartesian (before) | real (after) |
| --- | --- | --- |
| `compute_gravitational_acceleration` (order 2), uniform cube, N = 2,048 | 0.386 (0.644) | 4.63e-3 (5.79e-3) |
| `FMMEngine().compute_accelerations(max_order=4)`, the same | 0.386 (0.645) | 7.65e-4 (7.11e-4) |
| `compute_gravitational_acceleration`, Plummer, N = 2,041 | 0.306 (0.209) | 1.28e-2 (4.08e-3) |
| order 4, the same | 0.307 (0.208) | 3.35e-3 (4.96e-4) |
| `compute_gravitational_acceleration`, the A/B's clustered N = 1,500 | 0.171 (0.145) | 8.05e-3 (4.01e-3) |
| order 4, the same | 0.171 (0.145) | 1.46e-3 (4.74e-4) |

The Cartesian error does not fall with order, as its warning said (~1.8e-1 on the
golden configuration). One more engine moves with the default:
`FMMEngine(preset="large_n_gpu")` without a basis now satisfies the large-N production
profile (preset, radix tree, `expansion_basis == "solidfmm"`) and takes its contract
(minimum memory, streamed far pairs, `runtime_path="large_n"`, bucketed near field, no
retained interactions, upward batch 2048). The facade's `large_n_gpu` always did.

The commit is green on its own: the full CPU suite at 1c348c7 ran 2,314 passed, 198
skipped, 1 failed (the stale local nornax checkout,
`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`, as at X6). Every
`test_fmm.py` test that built an engine without a basis or called
`compute_gravitational_acceleration` passes on real with its tolerance unchanged; the
one that failed compared against the Cartesian module helper. It changed three tests:
- `test_strict_run_fail_fast.py::test_the_basis_is_part_of_the_contract` asserted that
  the preset alone left the profile closed because the default basis was Cartesian.
  It is now `test_the_preset_alone_opens_it_on_the_default_basis` (real basis, profile
  open).
- `test_fmm.py::test_prepare_downward_sweep_matches_module_helper` compared the engine
  with the Cartesian module helper, so it pinned `expansion_basis="cartesian"` until
  the next commit deleted it.
- The constructor-state matrix: its base case `cartesian_default` (`FMMEngine()`)
  became `real_default`, and the `solidfmm_real` row went because it now resolves
  byte-identically to the base (the distinctness guard caught it). The old base stayed
  as a `cartesian` row until the migration-error commit.

**Every Cartesian spelling raises one migration error (63bbda8, library -61 / +49;
the experimental-basis warning went):**

    basis='cartesian' was removed in the 2026-10 cleanup (K); see
    docs/cleanup_2026-10.md. Use basis='real' (the default).

It is `config._CARTESIAN_BASIS_REMOVED`, raised by:
- `FastMultipoleMethod(basis="cartesian")` (any case and whitespace) and the legacy
  `expansion_basis="cartesian"` kwarg (`solver._resolve_basis_input`);
- a `BasisInterface` whose `runtime_expansion_basis` is `"cartesian"`;
- `FMMEngine(expansion_basis="cartesian")` (`_validate_expansion_family`);
- `DistributedFMMConfig(basis="cartesian")`, when the evaluator is built
  (`distributed/fmm.py::_make_fn`, before any device work). The field was never
  validated: `is_real = basis == "real"`, so `"cartesian"` silently ran the complex
  far field. Valid values are unchanged; anything else still runs complex, as before.

`"cartesian"` stays in `config.Basis` and `kernels._shared.ExpansionBasis`, so a typed
caller (under `JACCPOT_RUNTIME_TYPECHECK=1`) sees this `ValueError` rather than a
type-check error; X3 kept `"class_major"` in `FarFieldMode` for the same reason. The
experimental-basis warning went, so `JACCPOT_ALLOW_CARTESIAN_BASIS` is inert. The
unknown-basis messages no longer offer `cartesian`.

**Removed** (library -3,395 / +282 lines across K; the additions are docstrings, the
migration message, the dense refusal, the moved gradient note and a `ValueError` for
an impossible basis where a Cartesian tail was):
- **Cartesian code (44cbfba, -3,134 / +155).**
  - `operators/multipole_utils.py` (-392), the copy of yggdrax's module (audit F41).
  - `downward/local_expansions.py` (-1,997 / +12): the derivative and stencil tables,
    `_accumulate_level`, the dense and CSR Cartesian M2L,
    `_translate_multipole_to_local_impl`, `translate_local_expansion`,
    `translate_multipole_to_local`, `initialize_local_expansions`,
    `accumulate_m2l_contributions`, `accumulate_dense_m2l_contributions`,
    `propagate_local_expansions` and the module-level `prepare_downward_sweep` /
    `run_downward_sweep`. `LocalExpansionData` and `TreeDownwardData` stay in place.
  - `upward/tree_expansions.py` (-466 / +27): `_aggregate_m2m_impl`,
    `compute_node_multipoles`, `_aggregate_multipoles_via_m2m` and the module-level
    `prepare_upward_sweep`. `_CENTER_MODES`, `_resolve_center_mode` (the complex upward
    imports it), `NodeMultipoleData` and `TreeUpwardData` stay.
    `NodeMultipoleData.moments` and `.component_matrix` are `Optional[Any]`: always
    `None` on the SH paths, kept so the pytree shape is unchanged
    (`test_prepare_state_does_not_retain_duplicate_component_matrix` still asserts
    `None`). So jaccpot no longer imports yggdrax's `TreeMultipoleMoments`, and five
    `# type: ignore[arg-type]` on `moments=None` went (`fmm_sweeps.py` 2,
    `fmm_derivatives.py` 2, `distributed/_force_scale.py` 1). The "COM centres are
    differentiated through" note moved to `upward/real_tree_expansions.py`'s
    docstring; `mutual/device_topology.py` points there.
  - `runtime/kernels/_evaluate.py` (-129 / +16): `_evaluate_local_cartesian_with_grad_batch`,
    the Cartesian tails of both L2P evaluators, the Cartesian order search in
    `_infer_order_from_coeff_count`, the two `order > 4` guards and the dead
    `offsets`. A basis other than `"solidfmm"` raises a `ValueError` where a tail was.
    Re-exports dropped from `kernels/core.py` and `kernels/__init__.py`.
  - `runtime/fmm_sweeps.py`: the Cartesian imports and fall-throughs of
    `prepare_upward_sweep` / `prepare_downward_sweep` (a `ValueError` now), and
    `SweepsMixin.run_downward_sweep` (Cartesian on every basis, not on the facade).
  - `runtime/fmm_prepare.py`: the `order > 4` guard. The locals-template helpers
    (`_build_locals_template_for_prepare_state`,
    `_update_locals_template_cache_after_prepare`) were Cartesian-only but are called
    from protected lanes (`fmm_strict_run.py` twice, `_large_n_pipeline.py`,
    `fmm_policy.py`, `fmm_prepare.py` twice), so they are no-ops returning what the SH
    path always got: `None`, and `_locals_template = None`. Call sites untouched.
- **Dense interactions (75e7ecd, -141 / +57).** Built only for the Cartesian basis (the
  prepare gated them on `expansion_basis != "solidfmm"`). Gone: the
  `yggdrax.dense_interactions` imports, `_DualTreeArtifacts.dense_buffers`,
  `_dual_tree_build_dense_buffers`, the `use_dense_interactions` parameter of the split
  and top-level dual-tree builders (and its terms in their payload predicates, true on
  every surviving lane), the plan field `use_dense_interactions_for_prepare` (16 -> 15
  decisions), the `dense_buffers` plumbing of `_unpack_dual_tree_artifacts` (now a
  4-tuple), `_prepare_downward_with_artifacts`, `SweepsMixin.prepare_downward_sweep`
  and `_prepare_solidfmm_downward_sweep` (which never read it), the engine attribute
  `use_dense_interactions`, and the `dual_split_dense_buffers` timing counter with its
  `STAGE_TREE` node and diagnostics key.
- **The finite-difference jerk fallback (7b672df, -58 / +17).**
  `evaluate_prepared_state_with_jerk(jerk_mode="accurate")` fell back to a central
  difference of the accelerations along `jerk_fd_dt * v` on a basis other than
  `"solidfmm"`: the Cartesian one. The analytic `dM -> dL` path is the only one, and
  unchanged.

**API shims:**
- `use_dense_interactions`: `FMMEngine(use_dense_interactions=True)` and the facade's
  legacy kwarg raise a `ValueError` naming the removal (X3's `grouped_interactions`
  pattern, refused first in `__init__` by `_reject_removed_dense_interactions`);
  `None` and `False` are accepted and ignored. `TraversalExecutionConfig` and
  `FMMPresetConfig` keep the field (always `False`) so old configs construct.
- `jerk_fd_dt` stays on all four signatures (engine and facade,
  `compute_accelerations_and_jerk` and `evaluate_prepared_state_with_jerk`),
  documented as accepted and unused, with no warning and no `<= 0` check: nornax's
  `adapters/jaccpot.py` passes `jerk_fd_dt=1e-3` on every call, and
  `bench/ci_benchmark_guard.py`, which CI runs, passes it too.
- `JACCPOT_ALLOW_CARTESIAN_BASIS`: inert.

**Kept:** `runtime/reference.py` and the engine's `compute_expansion` /
`evaluate_expansion` (the trace-free oracle); `operators/symmetric_tensors.py` (the SH
derivative towers); `pallas/l2p_real.py::l2p_real_grad_cartesian` (real SH; "Cartesian"
names its gradient components) and the "Cartesian component" wording in the SH
operators; `_resolve_center_mode`; the `moments` / `component_matrix` fields; the dead
`!= "solidfmm"` guards and tautological `== "solidfmm"` conjuncts in the protected lanes
(`fmm_evaluate.py`, `fmm_derivatives.py`, `_large_n_*`, `_fmm_impl.py`); the facade's
`mac_type` default branch; `_evaluate_prepared_state_at_positions_sorted` (the gradient
seam) and the complex-only `_evaluate_source_motion_farfield_jerk` (phase C);
everything complex.

**Tests** (deleted: 105 cases, 101 under (a) and 4 under (b); moved: 1; retargeted or
renamed: 3; added: 16). Collected (with the sibling nornax checkout) at 0375c08: 2,575;
at K: 2,486.

| test | tag | owner / note |
| --- | --- | --- |
| `tests/unit/core/test_local_expansions.py`: 21 of 22 cases | (a) | the 22nd, `test_target_local_eval_returns_potential_on_the_complex_analytic_branch`, is a complex-basis L2P test of `_evaluate.py`: moved to `tests/unit/runtime/test_l2p_particle_major.py` |
| `test_local_expansion_axis_contracts.py` (7), `test_local_expansions_shape_contracts.py` (9), `test_local_expansions_raw_moment_guard.py` (12) | (a) | |
| `tests/unit/operators/test_multipole_utils.py` (23) | (a) | yggdrax tests its own copy |
| `test_tree_expansions.py`: `test_compute_node_multipoles_com_matches_mass_moments`, `..._high_order_matches_direct`, `test_prepare_upward_sweep_returns_consistent_data` | (a) | |
| `test_convention_contracts.py`: `test_packed_orderings_are_exact_reverses` (9), `test_packed_orderings_are_not_interchangeable` (4) | (a) | the ascending packing lives only in yggdrax now; `docs/operator_conventions.md` section 2 keeps the record |
| `test_evaluate_scatter_contracts.py::test_the_cartesian_pack_is_ct_and_is_shared_with_its_offsets` | (a) | the scatter and target/leaf/node contracts stay |
| `tests/distributed/`: `test_distributed_m2l_mechanism.py` (2), `test_distributed_shardmap_local.py` (1), `test_distributed_fmm_assembly.py` (1), `test_distributed_fmm_shardmap.py` (1); two also out of `slow_tests.txt` | (a) | the Cartesian Phase-4 prototypes. The LET near-field half of `fmm_shardmap` is interleaved with its Cartesian far field (both feed every assertion), so it was not retargeted: the same `[local ; halo]` combined P2P is asserted against a direct sum by `test_distributed_solidfmm_far_shardmap.py` and through the production lane by `test_distributed_fmm_driver.py` |
| `test_fmm.py::test_prepare_downward_sweep_matches_module_helper` | (a) | |
| `test_fmm.py::test_fmm_dense_downward_matches_sparse_path` (also out of `slow_tests.txt`) | (a) | it ran the real basis from the default flip on, where the dense flag was already ignored |
| `test_fmm_golden.py::test_fmm_golden[uni_cartesian_n256_p2, uni_cartesian_n256_p4, clu_cartesian_n256_p4]` and their `golden/*.npz` | (a) | the 0.35 Cartesian anchor went with them; the anchor lookup has no fallback now (stricter) |
| `test_solver_api.py::test_compute_accelerations_and_jerk_accurate_cartesian_uses_fd_fallback` (also out of `slow_tests.txt`) | (a) | `test_jerk_fd_dt_is_accepted_silently_and_unused` owns the keyword |
| `test_gradient_correctness.py::test_rejects_non_solidfmm_basis` | (a) | no non-SH prepared state can be built; the migration error is `TestCartesianWasRemoved`'s |
| `test_basis_and_runner_hygiene.py::TestCartesianIsExperimental` (4) | (b) | `TestCartesianWasRemoved` (11 cases): the warning, its wording, its silencing switch and "it still works" became the refusal, per spelling, plus the inert switch |
| constructor-state matrix rows `cartesian`, `farfield_dense`; `solidfmm_real` | (a); (c) | `solidfmm_real` resolved byte-identically to the new base case `real_default` |

Retargeted:
- `test_tree_expansions.py`: `test_compute_node_multipoles_explicit_requires_centers` and
  `..._rejects_unknown_mode` now run on the complex upward sweep
  (`test_explicit_centre_mode_requires_centres`, `test_an_unknown_centre_mode_is_rejected`,
  the latter matching the message now); `test_removed_geometric_centres_raise_a_removal_error`
  keeps its complex-sweep assertion (3 cases).
- `test_dehnen_mac_reference.py::_upward_fixture` and the `test_dehnen_mac_gradients.py`
  host-loop test build their upward with `FMMEngine(expansion_basis="solidfmm")`
  (`resolve_dehnen_geometry` reads only the centres and the geometry).
- `test_strict_run_fail_fast.py::test_the_preset_alone_opens_it_on_the_default_basis`, as above.

Adapted: `test_type_annotation_guard.py` (two `CONVERTED_MODULES` entries and two
`DELIBERATELY_BARE` entries for deleted functions), `test_interaction_cache_retry.py`
and `test_split_build_carries_pair_policy.py` (internal builders lost the dense
keyword), `test_constructor_state_golden.py` (the matrix, above), comments in
`test_annotation_capture.py`, `test_fmm.py` and `tests/conftest.py` (the yggdrax
symbol contract: jaccpot imports 68 names from 20 yggdrax modules now, 85 from 22 at
0375c08; the comment said 72). `test_prepare_state_solidfmm_skips_prebuilt_locals_template`
is unchanged and pins the template no-op.

Added (16 cases): `TestCartesianWasRemoved` (11; 9 fail at 1c348c7, the two
`test_the_distributed_config_still_takes_its_two_bases` cases pass there too, as they
should); `test_use_dense_interactions_true_raises_removal_error` and
`test_use_dense_interactions_unset_or_false_is_accepted_and_ignored` (2 cases; the
engine attribute and the diagnostics key are asserted gone);
`test_jerk_fd_dt_is_accepted_silently_and_unused` (both jerk modes x four signatures x
steps 1e-3, 0, -1, warnings as errors, bitwise against not passing it; 2 cases).

**Goldens.** `golden/`: the three Cartesian `.npz` are deleted; the ten real and complex
ones are byte-identical. `golden_grad/`, `golden_lanes/` and `golden_modes/*.npz` are
byte-identical. `constructor_state.json` (+17 / -30 lines), per case against 0375c08:
- the base case is `real_default` (`FMMEngine()`, the real basis); `solidfmm_real` and
  `farfield_dense` are gone;
- every case loses `use_dense_interactions` and
  `_refresh_timing_dual_split_dense_buffers_seconds`;
- the 32 direct-engine cases that name no basis move `basis_impl` (`None` ->
  `RealSHBasis(...)`), `expansion_basis` (`'cartesian'` -> `'solidfmm'`) and `m2l_impl`
  (`None` -> `'rot_scale'`);
- `engine_preset_large_n_gpu` also takes the production contract (ten attributes:
  `_large_n_gpu_production_profile_cached`, `farfield_mode`, `memory_objective`,
  `nearfield_mode`, `precompute_nearfield_scatter_schedules`, `retain_interactions`,
  `retain_traversal_result`, `runtime_path`, `streamed_far_pairs`,
  `upward_leaf_batch_size`);
- the facade cases and `solidfmm_explicit_complex` move only by the two dense
  attributes.

**Bench:**
- `bench_fmm.py` runs again: no dense keyword, the 4-tuple unpack, no locals template.
  Smoke-run on CPU at N = 2,000, fixed order and adaptive order.
- `cleanup_inventory.py`: the Cartesian family excludes `_resolve_center_mode` (complex)
  and `l2p_real_grad_cartesian` (real SH; it was an overcount from P1 on); the dense
  family treats `_reject_removed_dense_interactions` as a gatekeeper. At K both
  families resolve to zero regions, so a re-run counts nothing for them.
- `annotation_capture.py`: its usage example and identity-scan note no longer name
  jaccpot's `multipole_utils`.

**Docs:** `README.md` (features; the basis section says `"cartesian"` is removed),
`ARCHITECTURE.md` (sections 5 and 7), `CLAUDE.md` (the `downward/` layout line),
`derivatives_and_jerk.md`, `differentiable_fmm.md`, `operator_conventions.md` section 2,
`agent_guides/STYLE_GUIDE.md` (`ct` retired).

**Downstream:**
- Odisseo: `fmm_basis="cartesian"` now raises at solver construction. Its defaults are
  `"real"` / `"solidfmm"`; the one place that names `"cartesian"`,
  `tests/test_integration_api.py::test_integrate_dispatches_to_fmm_coupler`,
  monkeypatches the coupler, so jaccpot never sees it.
  `jaccpot_coupling.py` reads `refresh_dual_split_dense_buffers_seconds` with
  `.get(..., 0.0)`, so its records show 0.0.
- nornax: `adapters/jaccpot.py` passes `jerk_fd_dt=1e-3` on every call; it keeps
  working, silently.
- yggdrax names jaccpot no longer imports, for phase Y2: `dense_interactions`
  (`DenseInteractionBuffers`, `densify_interactions`; `build_dense_interactions` was
  imported only by a deleted test); `multipole_utils` (`MAX_MULTIPOLE_ORDER`,
  `level_offset`, `multi_index_factorial`, `multi_index_tuples`, `multi_power`,
  `total_coefficients`); the Cartesian half of `tree_moments` (`TreeMultipoleMoments`,
  `compute_tree_multipole_moments`, `pack_multipole_expansions`,
  `translate_packed_moments`, `tree_moments_from_raw`, `multipole_from_packed`,
  `_quadrupole_from_second`, `_octupole_from_third`, `_hexadecapole_from_fourth`). From
  `tree_moments` jaccpot still imports `TreeMassMoments`, `compute_tree_mass_moments`
  and `compute_tree_mass_moments_jit`.

**Left for phase Z:**
- the locals-template helpers (now no-ops) and their seven call sites, and the engine's
  `_locals_template`;
- the inert `use_dense_interactions` fields of `TraversalExecutionConfig` and
  `FMMPresetConfig`, the engine keyword and the facade's legacy kwarg;
- `jerk_fd_dt` on the four signatures (needs nornax's adapter to stop passing it);
- `"cartesian"` in `config.Basis` and `ExpansionBasis`;
- examples that still use the removed plumbing:
  `examples/profile_prepare_memory_split.py` (dense keyword, 5-tuple unpack; it already
  does not import, see X3), `examples/profile_prepare_residuals.py` (wraps
  `_dual_tree_build_dense_buffers` by name), and the notebooks
  `adaptive_vs_fixed_benchmark.ipynb`, `benchmark_gpu_single_n_memory.ipynb`,
  `benchmark_runtime_accuracy_new.ipynb`, `benchmark_runtime_large_N_accuracy.ipynb`,
  `benchmark_runtime_large_N_performance.ipynb`.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`, at
  9be5571): 2,230 passed, 198 skipped, 1 failed: the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as before. At
  the default flip alone (1c348c7): 2,314 passed, 198 skipped, the same 1 failed. The
  84 fewer passes are the 104 node ids K removes outside `tests/distributed` less the
  20 it adds (moved, retargeted and renamed tests included).
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`) on the 15 touched unit test
  files: 289 passed.
- **Distributed tier** on four forced host devices (`tests/distributed` and
  `test_mutual_distributed.py`, `-n 6`): 90 passed, 3 skipped (two opt-in upstream
  checks and one sm_80-only case). At X6 it was 95 passed with the same skips; K
  deletes the five Cartesian prototypes.
- **Bitwise A/B against 0375c08 on CPU** (`git archive` exports of 0375c08 and of
  9be5571, every switch unset). 90 arrays and 75 diagnostic rows are bitwise equal:
  - the general path (21): forces and potentials of the default, fast, accurate and
    balanced on the real basis, fast and accurate on the complex basis (spelled
    `"solidfmm"` and `"complex"`), `dehnen_error` and `large_n_gpu` in fp32; a bare
    `FastMultipoleMethod()`; the position and mass gradients through
    `differentiable_accelerations` on both bases;
  - jerk and time derivatives (32), on both bases: `compute_accelerations_and_jerk` in
    both modes with `jerk_fd_dt=1e-3` passed (accelerations, jerk, a target subset's
    jerk), `evaluate_prepared_state_with_jerk(jerk_mode="accurate")`,
    `compute_accelerations_with_time_derivatives` to crackle,
    `evaluate_prepared_state_with_time_derivatives` on a subset, and
    `compute_accelerations(max_acc_derivative_order=2)`;
  - `BlockStepFMM(backend="jax")`: total accelerations and one base step (3);
  - the large-N lane with the GPU gate opened, N = 2,048 (19): prepare + evaluate,
    `refresh_prepared_state`, two `strict_prepare_refresh_and_evaluate` calls, the fused
    `strict_run_v2` scan plain and with history, callback, external field, initial
    acceleration and returned state, `carry="particles"` on cell leaves (two chained
    calls), `strict_fused_prepared_eval_fn`, and the prepacked differentiable lane's
    forces and gradients. The two cases that fail on this CPU configuration at X6 fail
    here too, with the same messages;
  - `mac_type="dehnen_error"` in the fused scan, both carries, through the CSR near
    field in Pallas interpret mode, N = 768 (5);
  - `distributed/fmm.py` on four forced host devices, N = 1,024 (10): forces on the real
    and complex bases with the Dehnen and BH MACs (cross-domain far pairs accepted,
    172-275 per device under BH), and the differentiable evaluator's forward and its
    position and mass gradients on both bases.

  The only differences are the expected ones: the three direct engine-default arrays
  (`FMMEngine()` at orders 2 and 4, `compute_gravitational_acceleration`) and the row
  naming the engine's basis. Against an fp64 direct sum on that clustered N = 1,500 they
  go from rel-L2 0.171 / 0.171 / 0.171 (Cartesian) to 8.05e-3 / 1.46e-3 / 8.05e-3
  (real), as in the table at the top.
- **`test_shards.py check`:** 2,486 tests, each in exactly one shard. At 0375c08 the
  same collection (with the sibling nornax checkout) is 2,575; K deletes 105 cases and
  adds 16, and moves, retargets or renames 4.
- **Inventory:** `cleanup_inventory.py`'s `cartesian` and `dense` families resolve to
  zero regions at K (40 and 1 at 0375c08).
- **Goldens** as above.
- `import jaccpot` and all 122 modules import (123 at X6, less `multipole_utils`).
  jaccpot imports 68 names from 20 yggdrax modules (85 from 22 at 0375c08), none from
  `yggdrax.dense_interactions` or `yggdrax.multipole_utils`. No file in `jaccpot/`,
  `tests/` or `bench/` imports a removed name; docstrings and comments name some as
  history, and `cleanup_inventory.py` as regexes. pre-commit is clean on every changed
  file.
- `bench/annotation_census.py`: shape-annotated array parameters 833 -> 783, bare
  `Array` parameters 1,695 -> 1,658, shaped share 33.0 % -> 32.1 %, `@jaxtyped`
  functions 183 -> 159. Only deletions moved them.
- **GPU pins** (frozen worktree at 1377ddb, one A100): S1-S5 **bitwise** against
  `main-15ceca4`, S6 **bitwise** against `main-17de581`. M1 and M3 (two cards) and an
  interleaved timing A/B against `main` are queued.
- **Two-card gate closed (2026-10-11):** M1 and M3 are **bitwise** on `main` 944e6ba
  (X1-X6 + K) against `main-15ceca4` (GPUs 5 and 6).

### C: the complex basis

The real spherical-harmonic basis is the only one. The distributed driver went first
(C2), then the single-GPU code (C1). The complex basis agreed with the real one to
round-off wherever the two were compared (4.5e-13 relative L2 in float64 at N = 2048,
p = 4, theta 0.5; 8e-18 on the jerk), so `basis="real"` is a drop-in replacement.

**The migration error** is `config._COMPLEX_BASIS_REMOVED`:

    basis='complex' was removed in the 2026-10 cleanup (C); see
    docs/cleanup_2026-10.md. Use basis='real' (the default) -- it agrees with the
    complex basis to round-off.

#### C2: the distributed driver

**Behaviour change (2200eb0).** `DistributedFMMConfig.basis` was never validated beyond
K's Cartesian refusal: `is_real = basis == "real"`, so every other string --
`"complex"`, `"solidfmm"`, a typo -- silently ran the complex far field. The evaluator
now refuses them while it is built (`_make_fn`, before any device work):
- `"complex"` and `"solidfmm"` (any case and whitespace) raise the migration error;
- any other string but `"real"` raises "`DistributedFMMConfig.basis must be 'real'`";
- `"cartesian"` keeps K's message.

Every production caller passes or defaults to `"real"`: Odisseo's mesh lane, pin M1,
the downstream contract test and the distributed tier.

**Removed (85b4cca; distributed -128 / +65, the additions docstrings).** From
`distributed/fmm.py`:
- the complex upward sweep and the complex coarse-tree M2M;
- the conjugate-symmetry fixup before L2L;
- the complex self- and cross-M2L branches and the complex working dtype;
- the module-level imports of `operators.complex_ops`,
  `upward.solidfmm_complex_tree_expansions` and `yggdrax.dtypes.complex_dtype_for_real`
  (they had to go before those modules could).

`rot = config.rotation` stays: `DistributedFMMConfig.rotation` is an inert passthrough
that the L2L cascade no longer reads.

#### C1: single GPU

**Tests first (2d1cba0, 1709c1c).** Every real-basis test that was validated only
against the complex basis got an oracle that does not need it, while the complex basis
still existed, so each new oracle was also run against complex. These are oracle
changes, not relaxations:

| test | old oracle | new oracle | measured, real (complex) | bound |
| --- | --- | --- | --- | --- |
| (i) `test_real_basis_runtime.py::test_real_basis_acceleration_derivatives_match_direct_sum` (was `..._match_complex`) | complex tower, 1e-6, at leaf 16 / theta 0.4: **no M2L pair** at N = 128, so it compared two zero towers | `jax.jacfwd` of the direct-sum field of exactly the sources the accepted far pairs cover (read off the prepared state; double coverage asserted absent), at leaf 4 / theta 0.4, 52 pairs (guarded) | p = 4: acc 2.31e-4, D1 4.51e-2, D2 2.00e-1; p = 8: acc 9.39e-6, D1 2.27e-3, D2 1.43e-2 (complex identical to the digits shown) | p = 8: D1 5e-3, D2 3e-2; acc 5e-4 / 2e-5; p = 4 -> 8 improvement > 5x (20x, 14x measured) |
| (ii) `core/test_real_source_motion.py::test_real_source_motion_with_the_far_field_matches_direct_sum` (was `..._matches_complex`) | complex, 1e-12 | `test_solver_api`'s `_direct_sum_jerk` / `_snap` / `_crackle`, N = 96, leaf 4, p = 4, theta 0.6, 106 pairs (guarded), Plummer | jerk 4.338e-4, snap 8.285e-5, crackle 9.554e-6 (complex identical); the two jerk entry points agree to 1e-12 | 1e-3, 2.5e-4, 3e-5 |
| (iii) `core/test_real_upward_sweep.py::test_real_upward_matches_direct_per_node_p2m` (was `..._matches_complex_convert`) | complex sweep + complex -> real conversion | the sum of `p2m_real_direct` over each node's particles about its centre, plus the centres asserted to be the nodes' COMs | 1.0e-16 / 1.2e-16 at p = 2 / 4, worst node 3.7e-16 (complex + conversion: 1.3e-16 / 1.4e-16) | 1e-9 (unchanged) |
| (iv) `test_fmm.py::test_m2l_ignores_padded_compact_far_pairs` (was `test_solidfmm_...`) | the complex M2L vs float64 | the real M2L (`basis_mode="real"`) vs float64 | 5.9e-7 on CPU fp32 (3.7e-4 on complex); both mutations behave as on complex | 2e-3 (unchanged, the TF32 floor) |
| (ix) `operators/test_real_harmonics.py::test_p2m_point_mass_reproduces_its_far_field_at_every_order` (new, 6 geometries) | `test_complex_to_dehnen_real_matches_p2m_real_direct` (orders 0-6, six geometries, a relative cross-check) | a point mass's multipole, M2L'd to `|X| = 9.1` and evaluated there, against the exact multipole truncation bound `(1+q)/(1-q) q^(p+1)`, p = 0-6, on the axes, `rho = 0` and a generic point | worst error/bound 0.98 (near-origin p = 0, where the bound is the leading term) | the exact bound (+1e-14); the field at 2x its bound |

Notes on them:
- **The acceleration-derivative tower is far-field only.** With no accepted M2L pair it
  is exactly zero: the near-field P2P contributes nothing to it. This is how the old
  test's two towers agreed. It is now documented in `docs/derivatives_and_jerk.md` and
  the README. Whether the near field should contribute is a separate question, not
  answered here.
- (ix) pins degree `n` to about `q^(7-n)` at p = 6 (1e-6 at degree 1, 0.1 at degree 6)
  and fails outright on a sign error. The derivative-recurrence tests anchor the
  generic-point normalisation to ~1e-3.
- (vi) `test_translation_reverse_is_finite_at_degenerate_displacements` probes `l2l_real`
  (it was `l2l_complex`). Mutation-checked: with the legacy 1e-60 floor in every dtype,
  the float32 exact-zero case goes non-finite on the real L2L as well.
- (vii) `test_fastlane_geometric_centers.py::test_real_com_tracks_complex` is GPU-only
  and opt-in, so it could not be re-measured on CPU against a direct sum. It is deleted
  under (c), with the large-N lane golden (`test_lane_goldens.py`, direct-sum anchored)
  and GPU pin S1 as owners.

1709c1c retargets the tests that ran the complex basis with no real twin to
`basis="real"`, tolerances unchanged:
- (v) `test_solver_api.py`: the bounded-leaf-batch test runs the real upward (renamed
  `test_real_upward_defaults_to_bounded_leaf_batch_size`); the basis-object test passes
  a `RealSHBasis`.
- (vi) `test_large_n_config_thresholds.py`: `_solver` and the fp32 far-field gradient
  test.
- `test_fmm.py`: the two locals-dtype tests became
  `test_real_float{32,64}_locals_keep_the_working_dtype`; bucketed-vs-baseline.
- Plain retargets: `test_grad_fmm_vs_directsum` (forward match),
  `test_nearfield_fastlane_grad_path` (`_prepared` and three sites),
  `test_gravity_vs_direct`, `test_adaptive_order_runtime`, `test_odisseo_coupling`,
  `test_grad_config`, `perf/test_perf_metrics`, `test_large_n_lane_carries_criterion`,
  `test_criterion_reaches_the_split_build`, `test_split_build_default_predicate`,
  `test_legacy_kwargs` (the legacy `expansion_basis=` too). And
  `test_upward_node_mass_conservation` loses its complex branch.

**Behaviour change (9e21f1e; library +112 / -93, the additions docstrings, the migration
message and the package `__getattr__`).**
- These raise the migration error:
  - `FastMultipoleMethod(basis="complex")` (any case and whitespace) and the legacy
    `expansion_basis="complex"`;
  - a basis object whose `name` is not `"real"`, or whose `runtime_expansion_basis`
    is `"complex"` (what a `ComplexSHBasis` was);
  - `FMMEngine(expansion_basis="complex")`, and an engine `basis_impl` not named
    `"real"`.
- **`basis="solidfmm"` and the legacy `expansion_basis="solidfmm"` resolve to
  `RealSHBasis`, public name `"real"`, silently.** `"solidfmm"` names the
  spherical-harmonic family; until now it meant the complex basis on the facade (D1 had
  already made it real on a bare engine). Odisseo's non-integration branches default
  `fmm_basis="solidfmm"` and nornax's smoke test passes `basis="solidfmm"`; both keep
  working, now on the real basis.
- `jaccpot.ComplexSHBasis` is no longer exported. `from jaccpot import ComplexSHBasis`
  raises an `ImportError` carrying the migration message, through a module
  `__getattr__`. An `AttributeError` would have lost the message in a `from ... import`.
  `jaccpot.basis` gets the same `__getattr__` with the modules (608278a).
- `fmm_policy._solidfmm_basis_mode()` always returns `"real"`.
- The three kernel `basis_mode: str = "complex"` defaults (`_l2l.py` twice,
  `_downward_prep.py`) are `"real"`. Every library caller passed it explicitly
  (checked); ARCHITECTURE section 10, which named this default as a trap, is updated.
- CI's benchmark guard runs `--basis real` and defaults to it. It was first run locally
  exactly as CI runs it, before the switch (see Gates).

The commit was gated by the full CPU suite: 2,216 passed, 197 skipped, 3 failed. Two of
the failures were downward source-motion tests in
`test_solidfmm_complex_tree_expansions.py` that built
`FMMEngine(expansion_basis="complex")`. They and their single-leaf sibling (which fed
complex multipoles to the real downward) are deleted in the commit, tag (a). The third
failure is the stale local nornax checkout, as at K.

**Removed** (library -7,863 / +573 lines across C; -7,726 / +436 ignoring the re-indent
where a branch became unconditional):
- **Runtime complex branches (4da9f68, runtime -1,168 / +255, or -1,058 / +145 ignoring
  whitespace).** The kernels keep their `basis_mode` / `rotation` keywords, because
  callers pass them. `basis_mode` other than `"real"` raises a `ValueError` where a
  complex branch was; `rotation` is accepted and unused.
  - `kernels/_m2l.py` (-253 / +56): `_apply_complex_m2l`, `_m2l_complex_batch_kernel`,
    the fused complex Pallas route (`_fused_complex_m2l_pallas_active`,
    `_m2l_complex_batch_kernel_fused_pallas`) and its cached tangent carrier.
  - `kernels/_l2l.py` (-265 / +27):
    - `_l2l_complex_batch_kernel` and the complex arms of the level cascade;
    - `_propagate_solidfmm_locals_to_children` (complex-only, no caller);
    - the conjugate-symmetry enforcement after the downward pass;
    - the source-motion-locals plumbing. Only the complex upward ever produced
      `source_motion_packed` (the runtime never passed it velocities), so those
      branches were dead on every real path.
  - `kernels/_downward_prep.py` (-77 / +46): the complex local-buffer dtype, the complex
    multipole branch and its rotation check, the source-motion payload, the symmetry step
    after the M2L pass.
  - `kernels/_evaluate.py` (-225 / +8):
    - both complex L2P tails (leaf-major and per-target, including the complex derivative
      towers);
    - the complex-only knobs `JACCPOT_LOCAL_EVAL_FLAT_ANALYTIC`, `_DTYPE_PRESERVE` and
      `_ORDER4_UNROLLED`. They never touched the real path and are inert now, ignored
      silently;
    - the particle-major L2P's complex refusal. The `iscomplexobj` gates are
      unconditional.
  - `fmm_sweeps.py`: the complex upward branch. The `upward_*` refresh-timing keys stay,
    and are always 0.
  - `fmm_derivatives.py`: the complex source-motion branch and the dead
    `_evaluate_source_motion_farfield_jerk` (-155).
  - `_large_n_pipeline.py`: only the `and not jnp.iscomplexobj(...)` of the
    particle-layout L2P test. Its `else` is the real leaf-layout L2P and stays.
  - `_adaptive_policy.py`: the complex dtype branches of the Dehnen source power.
  - `kernels/core.py` and `kernels/__init__.py`: six re-exports.
- **The modules (608278a, library -6,468 / +115).**
  - Deleted: `operators/complex_ops.py` (2,943), `operators/complex_harmonics.py` (333),
    `operators/solidfmm_reference.py` (297), `pallas/m2l_complex_fused.py` (1,171),
    `upward/solidfmm_complex_tree_expansions.py` (1,102), `basis/complex_sh.py` (158).
  - Their exports: `operators/__init__.py`, `basis/__init__.py`, and the `_fmm_impl.py`
    re-export of `enforce_conjugate_symmetry_batch`.
  - The public `jaccpot.basis.complex_to_real_coeffs` / `real_to_complex_coeffs`, and
    the private `_idx_nm` only they used.
  - The test-only `complex_to_dehnen_real_coeffs` / `_dehnen_real_Q_full`
    (`operators/real_dehnen_q.py`), with their `real_harmonics` re-exports. The
    `_compute_dehnen_B_matrix_complex` re-export went too: its only users were the
    deleted modules.
  - `runtime/dtypes.py::complex_dtype_for_real`, and `operators/dtypes.py`'s re-export
    of yggdrax's.
  - `upward/tree_expansions.py`'s `_CENTER_MODES`, `_REMOVED_CENTER_MODES` and
    `_resolve_center_mode`; only the containers are left there.
  - `config.BASIS_DOC`, which nothing used.
- **Distributed (85b4cca)**, as above.

**API shims:**
- `"complex"` stays in `config.Basis` and `kernels._shared.ExpansionBasis`, so a typed
  caller gets the migration error rather than a type-check error (K kept
  `"cartesian"` the same way).
- `basis="solidfmm"` -> real, silently, on the facade, the legacy kwarg and the engine.
  On `DistributedFMMConfig` it raises, because there it named the complex far field.
- `complex_rotation` (engine attribute, facade property, legacy kwarg),
  `FarFieldConfig.rotation` and `DistributedFMMConfig.rotation` are inert
  passthroughs; renaming them is phase Z. The large-N production gate
  (`_large_n_pipeline.py`) requires `fmm.complex_rotation == "solidfmm"`; dropping the
  attribute would have declined the large-N lane silently.
- The kernels' `basis_mode` / `rotation` keywords stay (callers pass them).
- `jaccpot.ComplexSHBasis` and `jaccpot.basis.ComplexSHBasis` raise an `ImportError`
  with the migration message.

**Kept:**
- `_compute_dehnen_B_matrix_complex`, `build_Q_dehnen_no_sqrt2` and
  `_compute_B_real_dehnen_via_Q` (`operators/real_dehnen_q.py`): complex-named, but the
  real basis's precomputed B matrices are built from them.
  `bench/cleanup_inventory.py`'s complex family excludes the first.
- The `complex_rotation` passthroughs above.
- `NodeMultipoleData.source_motion_packed` and `TreeDownwardData.source_motion_locals`:
  always `None`, but the protected lanes construct them, so the pytree shapes stay.
- The internal `"solidfmm"` family string everywhere (`expansion_basis`, the
  `_prepare_solidfmm_*` / `_propagate_solidfmm_*` names).

**Tests.** Collected at 944e6ba: 2,486; at C: 2,201. C removes 325 node ids and adds 40,
renames and moves included.

| test | tag | owner / note |
| --- | --- | --- |
| `test_distributed_fmm_driver.py::test_driver_solidfmm_matches_direct` | (a) | `test_driver_real_basis_matches_direct` (same IC, real + bh) |
| `tests/distributed/test_distributed_solidfmm_far_shardmap.py` (1) | (a) | the hand-rolled complex shard_map assembly; `test_driver_real_basis_matches_direct` and `test_driver_matches_direct` own its direct-sum assertions |
| `test_real_basis_runtime.py::test_real_basis_tracks_complex_basis` (3) | (c) | `test_real_basis_far_field_converges_with_order`, `test_real_basis_matches_direct_sum_small_n`, the real forward goldens |
| `test_fastlane_geometric_centers.py::test_real_com_tracks_complex` | (c) | GPU opt-in; the large-N lane golden and pin S1 |
| `test_fmm_golden.py::test_fmm_golden[uni_solidfmm_n256_p2, uni_solidfmm_n256_p4, clu_solidfmm_n256_p4, uni_solidfmm_n256_p6, uni_solidfmm_n1024_p4]`, `test_fmm_golden_execution_modes[bkt_uni_solidfmm_n256_p4]` and their `.npz` | (a) | the `*_real_*` cases on the same particles |
| `test_fmm_grad_golden.py::test_fmm_grad_golden[uni_complex_n128_p2, uni_complex_n128_p4]` and their `.npz` | (a) | `uni_real_n128_p2`, `uni_real_n128_p4` |
| `test_solver_api.py::test_basis_complex_alias_matches_solidfmm`, `test_prepare_state_solidfmm_complex_locals_remain_conjugate_symmetric` | (a) | |
| `test_solver_api.py`: the `[solidfmm]` cases of the seven J-phase jerk / time-derivative / acc-derivative tests | (c) | the `[real]` cases (identical once `"solidfmm"` is real) |
| `test_gradient_correctness.py`: `test_fd_vs_ad_positions[complex-0.4-64-2-16-0, complex-0.6-64-4-4-1]` and the `[complex]` cases of `test_fd_vs_ad_masses`, `test_no_nan_near_coincident` (2), `test_degenerate_azimuth_subgradient_is_correct`, `test_no_nan_axis_aligned_grid`, `test_reorders_to_original_particle_order`, `test_fd_vs_ad_along_a_transverse_direction_at_rho_zero` (9) | (a) | the `[real]` cases |
| `test_grad_fmm_vs_directsum.py`: the `[complex]` positions and masses cases (2) | (a) | the `[real]` cases |
| `test_nearfield_fastlane_grad_path.py::test_differentiable_accelerations_fast_lane_matches_bucketed[complex]` | (a) | `[real]` |
| `test_basis_and_runner_hygiene.py::TestBasisAliasing`: `test_real_is_a_genuinely_different_implementation`, `test_the_supported_bases_do_not_warn[complex]` | (a) | |
| `test_engine_config_resolution.py::test_the_spherical_harmonic_family_defaults_to_the_real_basis[kwargs1-complex, kwargs2-complex]` | (b) | `TestComplexWasRemoved::test_the_engine_expansion_basis_raises`, `test_the_engine_rejects_a_basis_object_not_named_real` |
| constructor-state matrix rows `solidfmm_explicit_complex`, `preset_accurate_complex` | (a) | they would resolve to `real_default` / `preset_accurate` under the mapping |
| `test_solidfmm_complex_tree_expansions.py`: the three downward source-motion-locals tests (9e21f1e), the rest (5) with the modules (608278a) | (a) | `core/test_real_source_motion.py` (finite differences at frozen centres, and the direct-sum engine test) |
| `test_real_harmonics.py`: `test_solidfmm_reference_matches_z_axis_m2l`, `test_dehnen_local_channel_factor_holds_at_any_delta` (3), `test_solidfmm_reference_matches_m2l_real_off_axis`, `test_complex_to_dehnen_real_matches_p2m_real_direct` (36), `test_complex_to_dehnen_real_discards_only_exact_zero_for_conforming_input` (4), `test_complex_to_dehnen_real_vjp_carries_the_imaginary_part` | (a) | the real M2L against the direct potential: `test_m2l_delta_convention_matches_direct_and_wrong_sign_is_worse`, `test_m2l_rotated_error_improves_with_order`, `test_full_rotated_pipeline_m2m_m2l_l2l_converges`; P2M: (ix) and the recurrence tests |
| `test_transverse_degeneracy_jvp.py`: the complex cascades of `test_the_rule_transposes_...` (3) and `test_the_analytic_branch_holds_at_float32` (9), `test_precomputed_block_lanes_...[m2l_complex_reference]`, `test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient` (2), `test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative` | (a) | the real cascades and the real fused lane; `bench/gpu_gate.py` drops the two complex sm_80 fragments |
| `test_custom_vjp_parity.py::test_m2l_complex_fused_pallas_custom_vjp_matches_twin` (4), `test_fused_m2l_vjp_env_off_falls_back_and_agrees[complex]` | (a) | the real fused-M2L twins |
| `test_fused_m2l_axis_contracts.py`: the five `[complex]` rows | (a) | the `[real]` rows |
| `test_l2p_particle_major.py::test_target_local_eval_returns_potential_on_the_complex_analytic_branch` | (a) | |
| `test_m2l_shape_contracts.py::test_a_complex_packed_expansion_is_still_accepted`, `test_the_chunked_scatter_accepts_both_bases[complex128]` | (a) | `_chunk_args` / `_scatter_args` are float now |
| `test_adaptive_policy_runtime.py::test_source_power_by_degree_accepts_the_complex_basis` | (a) | |
| `test_dehnen_mac_reference.py`: `test_dehnen_power_of_point_mass_...[complex-*]` (8), `test_dehnen_power_is_basis_invariant` (8) | (a) | the `[real-*]` cases and `test_dehnen_power_matches_numpy_weight_reference` |
| `test_complex_ops.py` (32), `test_complex_ops_vec3_contracts.py` (83), `test_complex_ops_packed_length_contracts.py` (23), `test_complex_harmonics.py` (2), `test_complex_sh_basis.py` (3), `test_m2l_complex_fused_pallas.py` (13), `pallas/test_fused_m2l_helper_axis_contracts.py` (8), `test_real_sh_roundtrip.py` (2) | (a) | |
| `core/test_tree_expansions.py` (5): the complex sweep's centre-mode checks | (a) / (b) | the rejection half is merged into `test_real_upward_sweep.py::test_the_real_upward_refuses_every_centre_mode_but_com` |

Renamed or moved (no assertion lost):
- `test_complex_to_dehnen_real_scales_linearly_in_mass` -> `test_p2m_real_direct_scales_linearly_in_mass`;
- `test_both_lanes_resolve_to_the_same_object` -> `test_the_lane_resolves_to_the_shared_object`;
- `test_the_distributed_config_still_takes_its_two_bases` -> `..._takes_the_real_basis`;
- `test_complex_and_solidfmm_resolve_identically` -> `test_solidfmm_resolves_to_the_real_basis`;
- the converted and retargeted tests named above;
- `test_prepare_upward_sweep_stashes_and_reuses_concrete_depth`, moved into
  `test_real_upward_sweep.py`.

Added:
- `TestComplexWasRemoved` (16): facade (3), legacy kwarg, basis object (2), engine,
  engine `basis_impl`, class import, unknown basis, distributed (4), unknown distributed
  basis;
- `TestBasisAliasing::test_solidfmm_resolves_to_the_real_basis` (3) and
  `test_the_legacy_expansion_basis_kwarg_resolves_to_real`;
- (ix) (6);
- `test_the_real_upward_refuses_every_centre_mode_but_com`.

Adapted:
- the public-API surface (`ComplexSHBasis` out of the frozen set);
- `test_type_annotation_guard.py` (the `complex_ops` entry);
- `test_gpu_gate_checks.py` fixtures (the two complex fragments; counts 8 -> 6, 5 -> 3);
- `test_shared_env_switches.py`;
- `slow_tests.txt`.

**Goldens.** `golden/`: the five `*_solidfmm_*` `.npz` are deleted, and the five
`*_real_*` are byte-identical. `golden_grad/`: the two `uni_complex_*` are deleted, the
four real ones byte-identical. `golden_modes/bkt_uni_solidfmm_n256_p4.npz` is deleted.
`golden_lanes/` and the real mode golden are byte-identical. `constructor_state.json`
loses exactly the two rows `solidfmm_explicit_complex` and `preset_accurate_complex`
(-16 lines); no other case moved.

**Bench / examples / docs (9b32502):**
- `bench_real_vs_complex.py` is deleted.
- `bench_upward_sweep.py` times the real sweep (smoke-run on CPU, parity holds); the
  stage breakdown is empty, since the real sweep has no stage callback.
- `audit_reverse_residuals.py` runs real; `bench_fmm.py` no longer offers `"complex"`.
- `examples/differentiable_fmm_overhead.py` runs real;
  `examples/real_sh_rot_scale_demo.py` compares real with the direct sum only
  (smoke-run, 2.66e-2 at N = 128).
- `examples/profile_prepare_residuals.py` is deleted: it wrapped the complex sweep's
  internals and K's removed dense buffers, and had not run since K.
- Docs: README, ARCHITECTURE, CLAUDE.md (layout), `derivatives_and_jerk.md`,
  `differentiable_fmm.md`, `operator_conventions.md` section 4,
  `real_vs_complex_gpu_plan.md` (historical) and the agent guides.

**Downstream:**
- Odisseo: `jaccpot_coupling.py`'s `fmm_basis="solidfmm"` defaults keep working, now on
  the real basis (`option_classes.py` already defaults to `"real"`).
  `fmm_basis="complex"` raises at solver construction. The mesh lane passes
  `DistributedFMMConfig(basis="real")`.
- nornax: the smoke test's and `examples/getting_started.py`'s `basis="solidfmm"` keep
  working (real).
- yggdrax names jaccpot no longer imports, for the Y phases:
  `dtypes.complex_dtype_for_real` and `tree_moments.compute_tree_mass_moments_jit` (the
  complex upward was its only user). jaccpot now imports 66 names from 20 yggdrax
  modules; it was 68 at K. From `tree_moments` it still imports `TreeMassMoments` and
  `compute_tree_mass_moments`.

**Left for phase Z:**
- `complex_rotation`, `FarFieldConfig.rotation`, `DistributedFMMConfig.rotation` and the
  kernels' `rotation` keyword (inert);
- `"complex"` in `config.Basis` and `ExpansionBasis`;
- the always-`None` fields `NodeMultipoleData.source_motion_packed` and
  `TreeDownwardData.source_motion_locals`;
- the `Inexact` (not `Float`) annotations on the M2L packed buffers;
- the `offsets_complex` local name in `_evaluate.py`;
- the notebooks that still name complex APIs: `examples/benchmark_runtime_accuracy_copy.ipynb`,
  `benchmark_runtime_accuracy_new.ipynb`, `benchmark_runtime_large_N.ipynb`,
  `benchmark_runtime_large_N_accuracy.ipynb`, `benchmark_runtime_large_N_performance.ipynb`.

**Found, not changed:**
- The acceleration-derivative tower's far-field-only scope (above). Fixed after C; see
  "Follow-up: the near field in the acceleration-derivative tower" below.
- `evaluate_local_real_with_grad`'s docstring says it returns the gradient with respect
  to the evaluation point. It returns `d phi / d delta` with `delta = centre - point`,
  which is minus that. The L2P callers use it correctly, and (ix) documents the sign.
  The docstring was fixed in the same follow-up.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`) at
  3da44fc: 1,951 passed, 193 skipped, 1 failed. The failure is the stale local nornax
  checkout (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as at K.
  The final head 9b32502 adds back one real-only test that 3da44fc had cut by mistake
  (`test_dehnen_mac_reference.py::test_dehnen_power_matches_numpy_weight_reference`);
  it passes, as does its whole file (23).
  - At the behaviour change alone (9e21f1e, before its last three test deletions):
    2,216 passed, 197 skipped, 3 failed, as described above.
  - Targeted runs at the two deletion commits: 1,354 passed at 4da9f68 (runtime,
    operators, core, pallas, characterization, `test_fmm.py`, `test_solver_api.py`),
    and 571 passed at 608278a.
- **Runtime type checks** (`JACCPOT_RUNTIME_TYPECHECK=1`) on the 27 touched unit test
  files: green.
- **Distributed tier** on four forced host devices (`tests/distributed` and
  `test_mutual_distributed.py`, `-n 6`): 88 passed, 3 skipped. The skips are the two
  opt-in upstream checks and the sm_80-only case, as at K; K's 90 less the two deleted
  complex tests.
- **Bitwise A/B against 944e6ba on CPU** (`git archive` exports of 944e6ba and of the
  final library, every switch unset): **89 arrays and 70 diagnostic rows bitwise
  equal, none different.**
  - The general path (21 arrays): forces and potentials of the default, fast, accurate
    and balanced on the real basis, `dehnen_error`, and `large_n_gpu` in fp32; a bare
    `FastMultipoleMethod()`; and the position and mass gradients through
    `differentiable_accelerations`. Also **`basis="solidfmm"` (fast, accurate and the
    gradients) against the base's `basis="real"`: bitwise.**
  - Jerk and the time derivatives (32 arrays, 16 of them `"solidfmm"` against base
    `"real"`): `compute_accelerations_and_jerk` in both modes (with `jerk_fd_dt=1e-3`
    passed, and on a target subset), `evaluate_prepared_state_with_jerk`,
    `compute_accelerations_with_time_derivatives` to crackle,
    `evaluate_prepared_state_with_time_derivatives` on a subset, and
    `compute_accelerations(max_acc_derivative_order=2)`.
  - `FMMEngine()` defaults at orders 2 and 4, and `compute_gravitational_acceleration`
    (4).
  - `BlockStepFMM(backend="jax")`: total accelerations and one base step (3).
  - The large-N lane with the GPU gate opened, N = 2,048 (19):
    - prepare + evaluate, refresh, two `strict_prepare_refresh_and_evaluate` calls;
    - the fused `strict_run_v2` scan, plain and with history, callback, external
      field, initial acceleration and returned state;
    - `strict_fused_prepared_eval_fn`;
    - the prepacked differentiable lane's forces and gradients.

    The two cases that fail on this CPU configuration at K fail here too, with the
    same messages.
  - `mac_type="dehnen_error"` in the fused scan, both carries, through the CSR near
    field in Pallas interpret mode, N = 768 (5).
  - `distributed/fmm.py` on four forced host devices, N = 1,024 (5): real forces with
    the Dehnen and BH MACs (cross-domain far pairs accepted: 3 / 0 / 1 / 29 per device
    under Dehnen, 275 / 177 / 172 / 201 under BH), and the differentiable evaluator's
    forward and its position and mass gradients.
- **`test_shards.py check`:** 2,201 tests, each in exactly one shard. At 944e6ba there
  are 2,486; C removes 325 node ids and adds 40.
- **CI benchmark guard on real**, run locally the way CI runs it
  (`--n 384 --runs 2 --warmup 1 --preset fast --basis real --theta 0.6 --leaf-size 16
  --max-order 4 --dtype float32`, `JAX_ENABLE_X64=1`; plus the jax 0.11.2 CPU
  `-vectorize-loops=false` flag this host needs). It passes every bound, three times
  (twice before the switch, once at the head):

  | ratio | bound | runs |
  | --- | --- | --- |
  | jerk_fast / acc | 0.4-8 | 0.78, 0.82, 1.03 |
  | jerk_accurate / fast | 0.9-15 | 1.83, 2.26, 1.78 |
  | td2 / jerk | 0.8-12 | 1.82, 1.93, 1.43 |
  | td3 / td2 | 0.8-12 | 1.11, 1.51, 1.31 |

  The accurate-vs-fast jerk rel-L2 is 2.116e-5 (floor 1e-6). The old setting (complex)
  failed td3 / td2 once under host load (0.749) and passed on a rerun (1.24); its jerk
  rel-L2 agrees with real to 7e-9.
- **Goldens** as above. **Imports:** `import jaccpot` works, all 116 modules import
  (122 at K, less the six), and no file in `jaccpot/`, `tests/` or `bench/` imports a
  removed name. Docstrings, comments and `cleanup_inventory.py`'s regexes name some as
  history. pre-commit is clean on every changed file.
- `bench/annotation_census.py`:
  - shape-annotated array parameters 783 -> 686;
  - bare `Array` parameters 1,658 -> 1,524;
  - shaped share 32.1 % -> 31.0 %;
  - `@jaxtyped` functions 159 -> 108;
  - modules walked 123 -> 117.

  Only deletions moved them; the complex modules were among the most annotated.
- **GPU pins** (frozen worktree at 31e7756):
  - M1 and M3 on two A100s (GPUs 2+6) are **bitwise** against `main-15ceca4`. C2
    changed `distributed/fmm.py`, so these were the key gate.
  - S1-S5 on one A100 are **bitwise** against `main-15ceca4`.
  - S6 (no `JACCPOT_*` env) is **bitwise** against `main-17de581`.
- **Speed (G5)**, interleaved main 944e6ba / C x 3 on one A100,
  `fused_memory_budget.py` scan min, ms/step:

  | N | main | C |
  | --- | --- | --- |
  | 2e5 | 6.35, 6.08, 6.18 (median 6.18) | 6.15, 6.14, 6.22 (median 6.15) |
  | 8e6 | 82.69, 82.48, 82.37 (median 82.48) | 82.70, 82.40, 82.55 (median 82.55) |

  Equal within the run-to-run spread.

**K's timing A/B (recorded here; #388 was already merged).** Interleaved main / K x 3 on
one A100, `fused_memory_budget.py` scan min, ms/step: at 2e5, 6.18 / 6.21 / 6.30 against
6.38 / 6.18 / 6.33; at 8e6, 83.20 / 83.15 / 82.90 (median 83.15) against
83.02 / 83.23 / 82.60 (median 83.02). Equal within the run-to-run spread.

### Follow-up: the near field in the acceleration-derivative tower

C found that `max_acc_derivative_order > 0` returned the derivatives of the far-field
acceleration only. The maintainer asked for the near field to be added (2026-10-11).
The tower is now `d^k a / dx^k` of the full acceleration.

**Implementation.** The new `jaccpot/nearfield/acc_derivatives.py` is the near half.
- It uses the same pairs as the acceleration: on the full route, each leaf against
  itself and its neighbour list; on the target route, the padded source matrix.
- Order 1 is the closed form `-G sum m (g I + h d d^T)`. Higher orders use nested
  forward mode through `masked_pair_factors`.
- The self pair is excluded, since its Jacobian is `-G m g(0) I`.
- For the C2 Wendland kernel, a coincident pair of distinct particles is left out at
  order 2 and above, because `sqrt(r2)` has no forward derivative at 0. Zero is the
  exact second derivative there.
- It is kept apart from the acceleration kernels: their custom VJPs block forward
  mode, and their op sequence is pinned.
- `_evaluate_prepared_tree` and `_evaluate_prepared_tree_targets` add it to the far
  half.
- The jerk path keeps the far half alone (`near_field_derivatives=False` on the
  engine's `evaluate_prepared_state`, which the facade does not expose). Its
  near-field pairwise jerk already contains the near Jacobian, so `fast_approx` would
  otherwise count it twice.

**Measured** (fp64, CPU):
- With no far pair (theta 1e-4, N = 128, leaf 4), the tower matches `jax.jacfwd` of the
  self-excluded direct sum to 3e-16 - 1e-15 at orders 1-3, for Plummer, `ferrers3`
  and `wendland_c2`. Before the change it was exactly zero there.
- At theta 0.4 (52 far pairs), the full tower's rel-L2 against the full direct sum is:

  | p | D1 | D2 |
  | --- | --- | --- |
  | 4 | 1.52e-4 | 3.10e-5 |
  | 8 | 7.67e-6 | 2.21e-6 |

  The near pairs dominate the norm. The far half alone is unchanged (D1 2.27e-3 and D2
  1.43e-2 at p = 8). The near half (full minus far) matches the direct sum over the
  sources the far pairs do not cover to round-off, so no source is missing or counted
  twice.
- A target subset matches the full route to 1e-15, also under `jit` with traced
  targets.

**Docstring.** `evaluate_local_real_with_grad` now says it returns `d phi / d delta`,
minus the gradient with respect to the evaluation point, and that the acceleration is
`-G * grad`. A numeric check of the example in `operators/real_harmonics.py`: `phi`
equals `+1/r` to 1e-9, and the returned gradient is `-grad_x(1/r)` to 1e-8.

**Tests:**
- New: `tests/unit/nearfield/test_acc_derivatives.py`. It checks the pair rule
  against autodiff for every kernel at orders 1-3, a coincident pair finite for every
  kernel, and the leaf route against the direct sum with a padded tail block.
- Rewritten: `test_real_basis_runtime.py::test_real_basis_acceleration_derivatives_match_direct_sum`.
  It now checks three oracles (full, far half, near half) and keeps the far half's
  bounds and convergence check.
- Every jerk test passes unchanged, including `fast_approx` with no far pair and the
  1e-12 `accurate` gap against the time-derivative API.

**Gates:**
- **CPU suite** (`tests/unit tests/integration tests/characterization`, `-n 12`) at
  bdcff74: 1,959 passed, 193 skipped, 1 failed. The failure is the stale local
  nornax checkout, as at K and C.
- **One A100** (GPU 2, untimed):
  - The new tests, the oracle and the solver API's tower, jerk and time-derivative
    tests pass in fp64.
  - At N = 2e4, the fp32 tower agrees with fp64 to 3.0e-6 (D1) and 3.8e-6 (D2), and
    the acceleration to 5.4e-7.
  - Pins S3, S4 and S4b are **bitwise** against C (`c-31e7756`). These cover the
    general evaluation route and the gradient lanes; at order 0 they never reach
    the new code.
- **Found on the GPU run and fixed:** `test_jerk_fd_dt_is_accepted_silently_and_unused`
  (from K) failed on a GPU, on C's base too. It turns every warning into an error,
  and jax 0.11.2 warns that the Pallas Triton backend is deprecated whenever it
  lowers a Pallas kernel (the upward P2M). CI runs it on CPU only. It now ignores
  that one jax message, and passes on the A100 and on CPU.

### Z-a: test consolidation (T1)

Tests only: nothing in `jaccpot/` changed. The plan's trimming rules R2-R7, the split of
the two grab-bag files by subject, and one file that Z-b deletes whole.

**Counts** (`pytest --collect-only -q -o addopts="" tests`):

| | fff5dfa | Z-a |
| --- | --- | --- |
| items (files) | 2,211 (180) | 2,009 (154) |
| default collection, `tests/perf` excluded | 2,208 (179) | 2,006 (153) |
| `tests/unit` | 1,836 (140) | 1,646 (122) |
| `tests/integration` | 297 (26) | 285 (18) |
| `tests/distributed`, `characterization`, `perf` | 55 (9), 20 (4), 3 (1) | unchanged |
| `-m "not slow"` | 1,969 | 1,764 |
| `bench/tests` (outside `tests/`) | -- | 93 (6) |

Of the 202 items fewer: 93 moved to `bench/tests` (R2), 100 folded by the signature table
(R3: 138 -> 38), 12 deleted under the tag rule (below), and 3 added where a test mixing a
shim with live behaviour was split. The slow-marked set grew by 3 because three
`slow_tests.txt` ids that had gone stale (`[real]` suffixes) match again. Every other test
moved unchanged, with the same name; a reconciliation of the collected ids before and after
accounts for each one.

**R2: bench tooling.** `test_gpu_gate_checks.py` (30), `test_annotation_pilot.py` (25),
`test_annotation_capture.py` (13), `test_annotation_census.py` (10),
`test_distributed_gate_checks.py` (5) and `TestComparisonBenchExitCode` (10, now
`bench/tests/test_comparison_bench_exit_code.py`) test `bench/` scripts, not the library.
They moved unchanged to `bench/tests/` and run once per push in the lint job, whose new
step reads its selection from `test_shards.py` (`EXTRA["bench-tooling"]`, outside the
partitioned `tests/` universe): `pytest -n 0 bench/tests` -> 93 passed in 26 s on this host.

**R3: one signature-contract table.** The twelve shape/axis-contract files (138 items) are
`tests/unit/test_signature_contracts.py`: one row per annotated kernel (37), each a valid
control call, the perturbations it must reject (`TypeCheckError`, or the fused near-field
leaf entry points' named `ValueError`) and the ones it must keep accepting (the free axes:
`farleaves`, the offsets lengths, `srcslots`, `srcleaves`, `tbatch`, the scatter `base`
length, the M2L accumulator's node axis; an all-zero leaf table; a length-1 level-weight
table; Python-float `G`). Every assertion of the twelve files is a case of its kernel's
row, including the reference oracle's Newtonian two-body sum and its documented `nan` at a
source; a failure names the row and the case. 38 items (37 rows and an id-uniqueness
check), 44 s serial against 69 s. The repo guards (exports, overloads, public API, mixin
base, operators-do-not-import-pallas, env readers) stay as they were.

**R4: grid corners.** At most one axis keeps full width:
- `test_real_harmonics.py`: the two `rho == 0` limit tests keep orders 1, 4 and 6 / 1 and
  4, over all three degenerate offsets.
- `test_transverse_degeneracy_jvp.py::test_the_analytic_branch_holds_at_float32` keeps
  `rho = 0` and `1e-6` over the three cascades. Mutation-checked in a scratch copy: a band
  narrowed to `rho_sq > 0` fails `1e-6` and `1e-12` on every cascade, an `eps` read from a
  float64 constant fails `1e-6` only, and neither is caught at `rho = 0`; so `1e-12` added
  no detection. The GPU fused-Pallas twin keeps all three (it is in `gpu_gate`'s registry).
- `test_custom_vjp_parity.py`'s grids were already at their corners (orders 2/4/6 on one
  axis; the `interpret` axis is a lane, not a size).
- The single-value `@pytest.mark.parametrize("basis", ["real"])` C left behind goes from
  seven `test_solver_api.py` tests and one `test_custom_vjp_parity.py` test (ids lose
  `[real]`).

**R5: `tests/integration/strict/`.** Twelve files become five:
- `test_strict_run_v2.py`: the seven `test_strict_run_v2_*.py` modules (24 items, slow,
  GPU-only), one class per former module with its docstring, env fixture and tests
  unchanged. The shared setup is `_solver`, `_state` and `_FUSED_ENV` (four modules
  imported or duplicated them byte for byte); helpers that differed carry the module's
  name. Checked by AST: all 71 tests, fixtures, helpers and constants are equivalent to
  their originals modulo the renames and `self`. They could not be run here (CPU); the
  next GPU gate (G2's integration set) runs them.
- `test_strict_fused_lane.py` (was `test_strict_fused_eval_fn.py`), plus `test_fmm.py`'s
  three fused-scan tests; `test_strict_refresh.py` (was `test_strict_refresh_faces.py`),
  plus `test_same_topology_refresh_modes.py`'s upward-only diagnostic and `test_fmm.py`'s
  three static-radix refresh tests.
- `conftest.py` holds `strict_env`, which three modules defined identically.
- `test_strict_cap_profile_removed.py` and `test_strict_lane_removed_switches.py` moved
  unchanged.

No module-scoped `fused_solver` was introduced: every test either sets env before it
builds its engine or reads that engine's own counters, so sharing one would change what
the tests observe.

**R6: one owner per gradient property.** `tests/integration/test_autodiff.py` (the
direct-sum oracle's own gradient against a dense sum, and its repeat-call stability) moved
into `test_grad_fmm_vs_directsum.py`. `test_grad_path_fast_kernels_large_n.py` and
`test_grad_path_pallas_cascades.py` are classes of `test_large_n_grad_reverse_path.py`:
`TestEveryFastPallasKernel` (its interpret-flag env is class-scoped, so the module's other
tests keep the default lane) and `TestPallasCascades` (it restores the real
`jax.default_backend`, which the module's gate patches, so it runs the CPU profile it ran
before). No assertion changed; 18 passed (x64).

**R7: the nornax adapter file** keeps protocol and schedule conformance and the three
rollout-gradient checks that need nornax's rollout and have no owner in
`test_mutual_fmm.py`: the FD check with the `checkpoint` (remat) switch on, plain-vs-remat
agreement, and AD-vs-FD through the in-scan `rebuild_fn` hook. The other two are deleted
(below). `test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`, the local
stale-nornax failure, is one of the kept three.

**The accel-only variants compared the default with itself** (X4's finding).
`test_large_n_accel_only_env_variants_match_baseline` set `JACCPOT_LARGE_N_*` variables
that `compute_leaf_p2p_accelerations_large_n_accel_only` never reads (it takes its knobs as
arguments; `runtime/_large_n_env.py` resolves the env upstream). Fixed, not deleted: the
cases pass the knobs (`..._scatter_variants_match_baseline`), and each first asserts that
the variant traces a different jaxpr from its baseline. All five agree bitwise. The sixth
case, `target_leaf_batch_size`, is deleted: only the target-block kernels read that knob,
and this edge-list route traces the same program with and without it.

**`tests/unit/test_removed_options.py` (46 items), for Z-b to delete whole.** Exactly the
tests whose subject is a removed option, an accepted-and-ignored shim or an error naming a
removal, from the four files named in the plan:
- `test_solver_api.py`: `execution_backend` (2), the octree (2), grouped / class-major /
  `FarFieldConfig.mode` (8), `fixed_depth`, `autotune_m2l_chunk`, `runtime_path` (2),
  `use_dense_interactions` (3), `jerk_fd_dt` (2), the locals template;
- `test_basis_and_runner_hygiene.py`: `TestCartesianWasRemoved` and
  `TestComplexWasRemoved` (less four live tests);
- `test_fmm.py`: the `complex_rotation` refusal, and the `fixed_depth` half of
  `test_capacity_fixed_depth_tree_mode_is_removed`;
- `test_legacy_kwargs.py`: the shim keywords (`complex_rotation`,
  `grouped_interactions=False`, `farfield_mode`) of the legacy-kwarg test.

The live halves stayed in live files: the basis aliasing, the distributed config taking
`"real"` and the unknown-basis messages (`TestBasisResolution` in
`runtime/test_engine_config_resolution.py`; `test_basis_and_runner_hygiene.py` is gone);
`tree_type` flowing from the config; an unknown tree build mode raising
(`test_an_unknown_tree_build_mode_raises`); the large-N lane gated on the preset alone
(`test_large_n_fast_path_policy.py::test_the_large_n_lane_is_gated_on_the_preset_alone`);
the live legacy keywords.

**The split by subject**, absorbed into existing files (`core/test_real_source_motion.py`
is renamed `core/test_time_derivatives.py`, as it takes the jerk APIs):
- `test_solver_api.py` keeps the facade end to end (solving, `target_indices`,
  prepared-state contents, caches; 17). Out: GPU memory and traversal caps (14) ->
  `test_large_n_config_thresholds.py`; the `large_n_gpu` preset (5) ->
  `test_large_n_fast_path_policy.py`; jerk and the time-derivative APIs (8) and the
  direct-sum jerk / snap / crackle oracles -> `core/test_time_derivatives.py` (its float64
  skip now marks only the source-motion tests); configuration reaching the engine (14) ->
  `runtime/test_engine_config_resolution.py`; the upward's default leaf batch ->
  `core/test_real_upward_sweep.py`.
- `test_fmm.py` keeps the engine's evaluation paths (near field, fast lane, caches, M2L,
  locals dtype; 25). Out: the reference expansions, physics and accuracy against the
  direct sum (16) -> `test_gravity_vs_direct.py`; preset defaults and validation (6) ->
  `runtime/test_engine_config_resolution.py`; the STF tensor test ->
  `operators/test_symmetric_tensors.py`; the strict lane (6) -> R5. The near-field-mode
  validation stays: under the unit shards' runtime type checks `NearFieldConfig` refuses
  `mode="unknown"` before the engine does (found running the moved tests with
  `JACCPOT_RUNTIME_TYPECHECK=1`).
- The heavy end-to-end tests (20-47 s each on CPU) stayed in `tests/integration`; only
  light configuration tests moved into the type-checked unit shards.

**Deleted or merged** (the tag rule):

| test | tag | owner / note |
| --- | --- | --- |
| the twelve shape/axis-contract files (136 of 138 items): `nearfield/test_large_n_block_shape_contracts.py` (37), `pallas/test_nearfield_mutual_shape_contracts.py` (28), `runtime/test_evaluate_scatter_contracts.py` (14), `pallas/test_fused_leaf_shape_guards.py` (11), `runtime/test_reference_shape_contracts.py` (8), `runtime/test_bucketed_schedule_shape_contract.py` (6), `runtime/test_m2l_shape_contracts.py` (6), `pallas/test_fused_leaf_axis_contracts.py` (6), `test_force_scale_axis_contracts.py` (5), `pallas/test_fused_m2l_axis_contracts.py` (5), `nearfield/test_near_field_axis_contracts.py` (5), `nearfield/test_kernel_axis_contracts.py` (5) | (b) | `test_signature_contracts.py::test_signature_contract[<kernel>]`: each assertion is a case of its kernel's row |
| `test_nearfield_mutual_shape_contracts.py::test_the_twin_and_the_kernel_reject_the_same_shapes` | (c) | rows `mutual_block_jax` and `mutual_block_pallas`, case "two-component xa" |
| `test_bucketed_schedule_shape_contract.py::test_the_shapes_this_pins_are_the_ones_production_passes` | (c) | row `bucketed_scatter_schedule`'s control (all three schedules built) |
| `test_real_harmonics.py::test_evaluate_local_real_grad_at_rho_zero_matches_limit[2-*]` (3) | (c) | the same test at orders 1, 4, 6, same offsets |
| `test_real_harmonics.py::test_p2m_real_direct_jacobian_at_rho_zero_matches_limit[2-*]` (3) | (c) | the same test at orders 1, 4 |
| `test_transverse_degeneracy_jvp.py::test_the_analytic_branch_holds_at_float32[deep inside the band-*]` (3) | (c) | `[exactly on axis-*]` and `[inside the band-*]`, same cascade (mutation-checked above) |
| `test_mutual_fmm_nornax.py::test_rollout_gradient_matches_finite_differences_through_the_fmm[False]` | (c) | `test_mutual_fmm.py::test_rollout_gradient_finite_difference` (FD vs AD through the adapter's base step) and `test_checkpointed_and_plain_fmm_rollout_gradients_agree` (plain = remat, remat FD-checked) |
| `test_mutual_fmm_nornax.py::test_dt_max_gradient_is_exact_through_the_fmm_rollout` | (c) | `test_mutual_fmm.py::test_the_dt_max_gradient_is_exact_on_both_backends[jax]` |
| `test_near_field.py::test_large_n_accel_only_env_variants_match_baseline[target_leaf_batch_size]` | (a) | the knob has no reader on this route (same jaxpr with and without it) |

By tag: (a) 1, (b) 136, (c) 13. No tolerance changed, and no lane lost a direct-sum
oracle (the deleted gradient checks have owners on the same lanes).

Renamed or moved without an assertion lost: the 93 R2 items; the eight `[real]` ids; the
nornax `[True]` FD case (unparametrised); the five accel-only variants; everything listed
under R5, R6 and the split. Added: the 38 signature items, and three halves of split tests
(`test_the_engine_refuses_fixed_depth_by_name` next to
`test_an_unknown_tree_build_mode_raises`, `test_the_large_n_lane_is_gated_on_the_preset_alone`,
`test_the_legacy_shim_keywords_still_parse_with_a_deprecation_warning`).

**CI and lists.** `test_shards.py` gains `EXTRA["bench-tooling"]`; no shard needed a new
path (`tests/integration/strict/` is under the integration shard,
`test_signature_contracts.py` under unit). `ci.yml`: the lint job's bench-tooling step.
`slow_tests.txt`: ten ids follow their tests, the K-deleted
`test_prepare_downward_sweep_matches_module_helper` goes, and the list is byte-sorted; none
is stale.

**Left for Z-b:**
- Removal-error assertions outside `test_removed_options.py`, mostly env values refused by
  `env_reject_removed` near their subjects (several inside otherwise live tests):
  `test_env_readers.py::TestEnvRejectRemoved`,
  `test_shared_env_switches.py::TestTheDegreeBatchedSwitchWasRemoved`,
  `test_large_n_fast_path_policy.py::test_the_removed_gather_unpermute_raises`,
  `test_mutual_fmm.py::test_the_removed_zcore_lane_raises`,
  `operators/test_p2m_real_leaf.py::test_the_removed_per_leaf_forward_raises`,
  `operators/test_pallas_nearfield_leafpair_csr_sorted.py::test_the_removed_layout_options_raise`
  (and part of `test_source_flags_are_validated`),
  `runtime/test_l2p_particle_major.py::test_the_removed_leaf_flatten_switch_raises`,
  `runtime/test_mac_geometry_com.py::test_the_removed_chain_variant_raises` (and part of
  `test_mode_knob`),
  `runtime/test_flat_walk_lane_wiring.py::test_the_removed_treecode_flag_is_refused`, parts
  of `operators/test_cascade_real_lanes.py::test_lanes_jit_and_the_vjp_forward`,
  `runtime/test_m2l_csr_lane_wiring.py::test_csr_lane_is_taken_and_matches_the_chunked_lane`,
  `runtime/test_mutual_walk_pallas.py::test_walk_backend_flag_parsing`,
  `runtime/test_nearfield_csr_lane_consistency.py::test_wide_accumulation_runs_the_table_layout`
  and `runtime/test_strict_carry_helpers.py::test_fresh_rebuild_flag`;
  `distributed/test_distributed_dehnen_error_mac.py::test_the_removed_treecode_walk_is_refused`;
  and `tests/integration/strict/test_strict_lane_removed_switches.py` (24, the X6 env
  values) with `test_strict_cap_profile_removed.py` (2).
- Live tests still pass the shims: `runtime_path=` (24 sites in 14 files),
  `FarFieldConfig(rotation="solidfmm")` (28 in 14), `grouped_interactions=False` (18 in
  8), `FarFieldConfig(mode=...)`, `execution_backend`.
- Comments in `jaccpot/` that name a moved test (this PR leaves the library alone):
  `solver.py:1216` (the jerk tests, now `core/test_time_derivatives.py`),
  `distributed/fmm.py:195` (the process-block floor, now
  `test_large_n_config_thresholds.py`), `runtime/_mac_geometry.py:82` (`test_fmm.py`'s
  order sweep, now `test_gravity_vs_direct.py`) and `runtime/strict_carry.py:8`
  (`test_strict_run_v2_particle_carry.py`, now
  `strict/test_strict_run_v2.py::TestParticleCarry`). `CLAUDE.md`'s layout block does not
  list `bench/tests/` or `tests/integration/strict/` yet.

**Left for Z-c:** ten more single-value `basis` parametrisations
(`test_gradient_correctness.py` 6, `test_grad_fmm_vs_directsum.py` 2,
`test_nearfield_fastlane_grad_path.py` 1, `test_dehnen_mac_reference.py` 1; their ids are
in `slow_tests.txt`), R9 (N <= 512, not chased), R10 (rebuild `slow_tests.txt` from
durations; smoke marker), and the shape-contract tests that live outside the twelve files
(`test_adaptive_policy_runtime.py`'s seven, `core/test_near_field.py`'s two payload-pair
rejections). The file count is 154 against the 125 target.

**Gates:**
- **CPU suite** (`pytest -p no:cacheprovider -n 12 --timeout=1800 tests/unit tests/integration
  tests/characterization -rf`, CPU, x64 off) at efcabdc: 1,757 passed, 193 skipped, 1 failed
  in 24 min. The failure is the stale local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as at K, C and the
  tower follow-up; R7 keeps that test.
- **As CI runs the shards:** every file that received tests was also run with
  `JAX_ENABLE_X64=1`, and the unit ones with `JACCPOT_RUNTIME_TYPECHECK=1`: green (that
  is how the near-field-mode check was found). `tests/integration/strict` on CPU: 42
  passed, the 24 GPU-only skip. The merged R6 module: 18 passed; the nornax FD check with
  remat: passed.
- **`test_shards.py check`** (forced four host devices, x64): 2,006 tests, each in exactly
  one of the 11 shards; no overlap, no gap.
- **Bench tooling** (the lint job's selection, `pytest -n 0 bench/tests`): 93 passed.
- **pre-commit** on the 38 changed files: clean. The goldens are untouched (no
  `tests/characterization` file changed).
- **Not run:** any GPU. The merged GPU-only `strict/test_strict_run_v2.py` and the
  sm_80-only cases are for the next G2 run.

### Z-b: shims, inert options and removal errors

Every earlier phase left a compatibility layer behind it: a removed option still
constructed and was ignored, or a removed env value or option was refused by name
(`env_reject_removed` and its hand-rolled siblings, the basis migration errors). Z-b
removes that layer. Library -1,473 / +192 lines (`jaccpot/`); benches and examples
-1,341 / +8 (one of them a 1,200-line example that had not imported since X1).

**Maintainer decisions (2026-10-11).** They override the "kept so old configs
construct" notes of X3-X6, K and C:
1. The protective removal errors go. A removed env value is simply not read; a removed
   option name hits whatever generic validation remains (an unknown basis gets the
   validator's "must be ..." message, an unknown keyword or dataclass field a
   `TypeError`). A knob whose other values are live keeps its generic validation.
2. `runtime_path` and `FarFieldConfig.mode` go (Odisseo stopped passing both, its #31).
3. `jerk_fd_dt` goes from the four jerk signatures (nornax stopped passing it, its #14).
4. The other inert shims go; none has an external caller. Kept: `basis="solidfmm"`
   (the real basis; nornax and Odisseo pass it), the refine family of `TreeConfig`
   (external callers), every live diagnostic key.
5. The X4/X6-era switches that nothing reads go from benches and tests; benches that
   passed a removed option are updated so they still run.

**Removed, by commit** (library lines):
- **The removal errors (2581498, -572 / +79).**
  - `_env.env_reject_removed` and its 15 call sites. A knob left with no live value
    besides its default is no longer read at all: `CASCADE_KERNEL`,
    `WALK_FUSED_EMIT`, `NEARFIELD_TARGET_CLASSES`, `NEARFIELD_DIRECT_ROWS`, the strict
    walk's `..._WALK_DETERMINISTIC`, `FASTLANE_UNPERMUTE`, `..._MAC_RADIUS` (with its
    single-valued "must be 'exact'" check), `COM_RADII_VARIANT`, `M2L_CSR_KERNEL` /
    `M2L_CSR_TILED` (with their "expected lanes" check),
    `LOCAL_EVAL_DIRECT_LEAF_FLATTEN`, `M2L_DEGREE_BATCHED`, and the six X6 strict-lane
    switches.
  - A knob with other live values keeps its generic validation: `NEARFIELD_LAYOUT`
    (`env_choice`: `sorted` now warns and runs the default), `MUTUAL_M2L` (`zcore` gets
    "must be one of 'auto', 'jax', 'fused'"), `P2M_BLOCK` / `block` (`<= 0` gets "must
    be positive"), the CSR kernel's `source_flags` (the letters a, l, p, r: `g` is no
    longer one of them; a lone `p`, the prefetch that is part of `r`, is ignored like
    `p` beside `r` always was), the distributed `local_walk` ("must be 'dual_tree'").
  - The hand-rolled refusals: the strict treecode walk (`RuntimeError`), the z-core
    lane, the distributed `local_walk` message.
  - The basis migration errors: `_CARTESIAN_BASIS_REMOVED`, `_COMPLEX_BASIS_REMOVED`,
    the `ComplexSHBasis` `__getattr__` of `jaccpot` and `jaccpot.basis`, and the
    `"cartesian"` / `"complex"` entries of `config.Basis` and `ExpansionBasis`. The
    names now get their validator's message: "basis must be one of 'real' or
    'solidfmm'" (the facade), "expansion_basis must be 'solidfmm'" (the engine; under
    `JACCPOT_RUNTIME_TYPECHECK=1` its `Literal["solidfmm"]` refuses first),
    "basis.name must be 'real'" / "basis_impl.name must be 'real'" (a basis object,
    facade / engine), "DistributedFMMConfig.basis must be 'real'".
  - `tree_type="octree"`, `tree_build_mode="fixed_depth"` and
    `execution_backend="octree"` reach the generic lists.
  - `_fresh_compact_pair_rebuild_enabled` (always `True` once its refusals went) is
    folded into its two callers.
- **`runtime_path` and `FarFieldConfig.mode` (cf65382, -109 / +26).** The engine
  keyword, attribute and validation, the large-N contract's `runtime_path =
  "large_n"`, its entry in the strict compiled-profile fingerprint (a diagnostic);
  `FarFieldConfig.mode`, both `FarFieldMode` aliases, `FARFIELD_MODE_DOC`, the
  engine's and facade's `farfield_mode`, the facade's legacy `farfield_mode=`, the
  presets' far-field `mode` entries, and the always-`"pair_grouped"`
  `_RuntimeExecutionOverrides.farfield_mode`.
- **`jerk_fd_dt` (7b9156b, -24 / +2)** from `compute_accelerations_and_jerk` and
  `evaluate_prepared_state_with_jerk` (facade and engine).
  `bench/ci_benchmark_guard.py` and `bench/bench_parallel_paths.py` lose
  `--jerk-fd-dt` (CI never set it).
- **Grouped, autotune, dense, backend (3d2e284, -219 / +7).**
  `RuntimePolicyConfig.autotune_m2l_chunk`, `.precompute_grouped_class_segments`,
  `.grouped_schedule_budget_bytes` and their legacy keywords;
  `FarFieldConfig.grouped_interactions`, its legacy keyword, the facade's
  always-`False` property and refusing setter, `_reject_removed_farfield_options`;
  `use_dense_interactions` (engine keyword, legacy keyword, the refusal, the
  always-`False` fields of `FMMResolvedConfig` and `FMMPresetConfig`);
  `RuntimePolicyConfig.execution_backend`, the `FMMExecutionBackend` alias, the
  engine attribute and its validation, the legacy keyword. The prepared states' own
  `execution_backend` tag (`"radix"` / `"large_n"`, a static pytree field) stays.
- **The complex basis's leftovers (69e0c1d, -274 / +40).** `complex_rotation` (engine
  attribute and validation, facade property, legacy keyword),
  `FarFieldConfig.rotation` (and the large-N preset's `rotation="solidfmm"`),
  `DistributedFMMConfig.rotation`;
  the kernels' `rotation` / `complex_rotation` keywords and static jit arguments
  (`_accumulate_m2l_fullbatch`, `_accumulate_m2l_chunked_scan`,
  `_m2l_chunk_contributions`, `_propagate_solidfmm_locals_by_level`,
  `_prepare_solidfmm_downward_sweep`, the downward-prep helpers,
  `_SolidFMMDownwardMultipoleInputs.rotation_mode`); the kernels' `basis_mode`
  keyword and its five "must be 'real'" checks; `_apply_m2l`, which only checked
  `basis_mode`
  before calling `_apply_real_m2l`; the engine's always-`"real"`
  `_solidfmm_basis_mode()`. **Load-bearing:** the large-N gate
  (`can_use_large_n_prepare_path`) required `complex_rotation == "solidfmm"` through
  `getattr(fmm, "complex_rotation", "")`; the conjunct went in the same commit, or the
  lane would silently have declined on the missing attribute.
  `test_the_lane_accepts_the_dehnen_error_criterion` and the A/B's large-N rows (the
  lane taken, its counters equal) pin it.
- **Always-`None` pytree fields (b662b00, -61 / +35).** `NodeMultipoleData.moments`,
  `.component_matrix`, `.source_motion_packed`,
  `TreeDownwardData.source_motion_locals`.
  The time-derivative tower set `component_matrix` to a copy of its packed
  source-motion multipoles, which nothing read. `Inexact` -> `Float` on the packed
  multipoles (`NodeMultipoleData.packed`, the M2L accumulators); the real L2P's
  `offsets_complex` locals are `offsets_solid` / `offsets`.
- **Template, gear chain, timings (88cdb24, -231 / +20).**
  `_build_locals_template_for_prepare_state` (always `None` since K),
  `_update_locals_template_cache_after_prepare`, their six calls,
  `_PrepareStateTreeUpwardArtifacts.locals_template`, the engine's
  `_locals_template`;
  `_gear_pairs_for_autotune` and the `far_pairs_by_gear` parameter of
  `_prepare_downward_with_artifacts`, `prepare_downward_sweep` and
  `_prepare_solidfmm_downward_sweep`, which never read it (the adaptive-order gear
  bucketing that feeds `recent_dual_far_pairs_by_gear_counts` stays);
  `strict_fused_device_only_hot_path` is `strict_fused_hot_path` (its device-only half
  is the only fused refresh since X6; it is still a real condition, the fused
  lane's
  traced refresh, so it was renamed rather than folded); the five never-written
  `refresh_upward_{geometry,mass_moments,p2m,m2m,source_motion}_seconds` diagnostics
  (`upward_compute` is a leaf of the stage-timing tree now; Odisseo reads them with
  `.get(key, 0.0)`).
- **Benches and examples (c3ffe89).** See the commit; `bench/dce_pins.py` S4/S4b drop
  `FarFieldConfig(mode="auto")`, and `fused_memory_budget.py` its `runtime_path` and
  far-field mode, so the GPU pins run on this branch with the same configuration.

`import jaccpot` and every module import; no file in `jaccpot/`, `tests/`, `bench/` or
`examples/` imports a removed name.

**Tests** (`pytest --collect-only -q -o addopts="" tests`):

| | 4f05686 | Z-b |
| --- | --- | --- |
| items (files) | 2,009 (154) | 1,913 (151) |
| default collection (`test_shards.py` universe) | 2,006 | 1,910 |
| `tests/unit` | 1,645 (122) | 1,575 (121) |
| `tests/integration` | 286 (18) | 260 (16) |
| `tests/distributed`, `characterization`, `perf` | 55 (9), 20 (4), 3 (1) | unchanged |
| `-m "not slow"` (default collection) | 1,761 | 1,668 |

105 node ids go and 9 appear: 96 deletions under the tag rule (below) and 9 renames.

Deleted (the tag rule):

| test | items | tag | owner / note |
| --- | --- | --- | --- |
| `tests/unit/test_removed_options.py` (whole file) | 46 | (a) | every test's subject is a removed option, shim or removal error; Z-a collected them for Z-b |
| `tests/integration/strict/test_strict_lane_removed_switches.py` (whole file) | 24 | (a) | the X6 removed-value refusals and the accepted-and-ignored X6 switches; nothing reads either |
| `tests/integration/strict/test_strict_cap_profile_removed.py` (whole file) | 2 | (a) | the X4 `/tmp` cap-profile file; nothing names it any more |
| `test_env_readers.py::TestEnvRejectRemoved` | 6 | (a) | `env_reject_removed` deleted |
| `test_shared_env_switches.py::TestTheDegreeBatchedSwitchWasRemoved::test_setting_it_after_import_raises` | 1 | (a) | the `M2L_DEGREE_BATCHED` refusal; the module reads no env now, which its sibling test still pins |
| `test_large_n_fast_path_policy.py::test_the_removed_gather_unpermute_raises` | 1 | (a) | the `FASTLANE_UNPERMUTE=gather` refusal |
| `operators/test_pallas_nearfield_leafpair_csr_sorted.py::test_the_removed_layout_options_raise` | 2 | (a) | `NEARFIELD_TARGET_CLASSES`, `NEARFIELD_DIRECT_ROWS=chunked`: read nowhere |
| `runtime/test_engine_config_resolution.py::TestLaneModeNormalisation`, the `runtime_path` and `execution_backend` cases of both tests | 4 | (a) | both options deleted; the `nearfield_mode` cases stay |
| `runtime/test_flat_walk_lane_wiring.py::test_the_removed_treecode_flag_is_refused` | 2 | (a) | `..._FUSED_TREECODE_WALK` read nowhere (GPU-only test) |
| `runtime/test_mac_geometry_com.py::test_the_removed_chain_variant_raises` | 1 | (a) | `COM_RADII_VARIANT` read nowhere |
| `runtime/test_strict_carry_helpers.py::test_fresh_rebuild_flag` | 5 | (a) | the always-`True` helper folded away, its two refusals deleted |
| `test_solver_api.py::test_prepare_state_does_not_retain_duplicate_component_matrix` | 1 | (a) | `NodeMultipoleData.component_matrix` deleted |
| `runtime/test_l2p_particle_major.py::test_the_removed_leaf_flatten_switch_raises` | 1 | (b) | `test_particle_major_equals_leaf_major` now also asserts its live half: the leaf-major scatter leaves dead rows zero |

By tag: (a) 95, (b) 1. No tolerance changed, and no lane lost a direct-sum oracle (only
refusal and shim tests went).

Kept on the generic validation that remains, renamed where the subject changed (the
old ids are the 9 renames):
- `test_mutual_fmm.py::test_an_unknown_mutual_m2l_lane_raises` (`zcore` -> "must be
  one of");
- `operators/test_p2m_real_leaf.py::test_a_non_positive_block_raises` (2);
- `distributed/test_distributed_dehnen_error_mac.py::test_an_unknown_local_walk_is_refused`;
- `test_large_n_config_thresholds.py::test_the_far_field_is_expanded_about_com_centres_at_large_n`
  (4; the `farfield_mode` half went, the COM-centre half stays);
- `test_large_n_fast_path_policy.py::test_a_zero_target_block_size_means_the_default`
  (the inert `LARGE_N_RADIX_FAST_LANE=0` half went).

The removal halves of eight live tests went: `test_lanes_jit_and_the_vjp_forward`,
`test_csr_lane_is_taken_and_matches_the_chunked_lane`, `test_pallas_walk_flags_overflow`,
`test_walk_backend_flag_parsing`, `test_mode_knob`,
`test_wide_accumulation_runs_the_table_layout`,
`test_requires_the_particle_count_in_the_profile_set`, and the `p` / `ap` cases of
`test_source_flags_are_validated` (its `g` cases now assert the generic letters error).

Edited without an assertion lost: live tests stop passing `runtime_path=` (23 sites
in 13 files), `FarFieldConfig(rotation=, mode=, grouped_interactions=)`, `jerk_fd_dt`,
`execution_backend`, the kernels' `rotation` / `basis_mode`, and the switches nothing
reads (`STATIC_STRICT_GPU_MODE`, `_FUSED_MODE`, `_FUSED_DEVICE_ONLY`,
`_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK`, `_REQUIRE_EXACT_CAP_PROFILE_MATCH`,
`_FUSED_FRESH_COMPACT_PAIR_REBUILD`, `_FUSED_TREECODE_WALK` in fixture clear-lists);
assertions on the removed attributes `runtime_path` / `farfield_mode` /
`_solidfmm_basis_mode()` go. `test_downstream_contract.py` follows Odisseo's
`_build_fmm_solver` on `jaccpot-integration` (ac3cb5d, after #31):
`FarFieldConfig(m2l_chunk_size=None)` without `mode`, no `runtime_path`, plus the
`softening_kernel` it now passes.
`tests/perf/test_perf_metrics.py` compared the far-field modes `pair_grouped` and
`class_major` (the latter raised since X3; the perf tier is opt-in, so nobody saw);
it compares the near-field modes it ran with. `slow_tests.txt` drops the
`test_removed_options.py` id; every remaining id matches a collected test.

**Constructor golden**, regenerated once (`JACCPOT_REGEN_GOLDEN=1`, in the record
commit): the base state goes from 236 to 226 attributes (`_locals_template`,
`complex_rotation`, `execution_backend`, `farfield_mode`, `runtime_path` and the five
upward timings), the matrix from 36 to 33 cases (`farfield_pair_grouped`,
`runtime_path_large_n`, `backend_radix`), and the `FMMResolvedConfig` /
`FMMPresetConfig` reprs lose `use_dense_interactions=False`. A semantic diff of the two
files finds nothing else. Every other golden (`golden/`, `golden_grad/`,
`golden_lanes/`, `golden_modes/*.npz`) is byte-identical.

**Bitwise A/B against 4f05686 on CPU** (`git archive` exports of main and of the final
library c3ffe89, every switch unset, separate processes, yggdrax 1f0e36f on both
sides). Each array is compared with `np.array_equal` and its dtype:

| section | what | arrays | equal |
| --- | --- | --- | --- |
| general | default, fast, accurate and balanced (real), fast and accurate spelled `"solidfmm"`, `dehnen_error`, `large_n_gpu` in fp32 (forces and potentials); a bare `FastMultipoleMethod()`; position and mass gradients through `differentiable_accelerations` on both spellings | 21 | 21 |
| deriv | `compute_accelerations_and_jerk` in both modes (and on a target subset), `evaluate_prepared_state_with_jerk`, `compute_accelerations_with_time_derivatives` to crackle, `evaluate_prepared_state_with_time_derivatives` on a subset, `compute_accelerations(max_acc_derivative_order=2)` (the near-field tower), on `"real"` and `"solidfmm"` | 32 | 32 |
| engine default | `FMMEngine()` at orders 2 and 4, `compute_gravitational_acceleration`, the direct sum | 4 | 4 |
| block step | `BlockStepFMM(backend="jax")`: total accelerations and one base step | 3 | 3 |
| large-N | the lane with the GPU gate opened, N = 2,048: prepare + evaluate, `refresh_prepared_state`, two `strict_prepare_refresh_and_evaluate` calls, the fused `strict_run_v2` scan plain and with history, callback, external field, initial acceleration and returned state, `carry="particles"` on cell leaves (two chained calls), `strict_fused_prepared_eval_fn`, the prepacked differentiable lane's forces and gradients | 19 | 19 |
| `dehnen_error` | the fused scan, both carries, through the CSR near field in Pallas interpret mode, N = 768 | 5 | 5 |
| mesh | `distributed/fmm.py` on four forced host devices, N = 1,024: forces with the Dehnen and BH MACs, the differentiable evaluator's forward and its position and mass gradients | 5 | 5 |
| fp32 | float32 inputs and working dtype: six configurations' forces and potentials, jerk in both modes, time derivatives to crackle, the acceleration-derivative tower, the gradients | 21 | 21 |

**110 arrays and 72 diagnostic rows bitwise equal, none different** (the rows: the
strict-lane and fast-lane counters, the large-N lane taken, the mesh's cross-domain
pair counts). The two cases that fail on this CPU configuration at K and C fail on both
sides with the same messages (`carry="particles"` on bucket leaves: a fixed capacity
saturates; `dehnen_error` through the non-CSR route: `return_force_scale` needs the CSR
lane). The first seven sections ran twice per side, with `JAX_ENABLE_X64=1` and `=0`:
both times 89 / 89. That second arm is a repeat, not an fp32 run: yggdrax's
`__init__` switches `jax_enable_x64` on at import, so the variable cannot turn it off
once `jaccpot` is imported. The fp32 section is the float32 coverage, with float32
inputs and `working_dtype` (the large-N, `dehnen_error` and mesh sections are float32
too).

**Gates:**
- **CPU suite** (`pytest -p no:cacheprovider -n 12 --timeout=1800 tests/unit
  tests/integration tests/characterization -rf`, CPU, at 9dccd98 with the regenerated
  golden): 1,663 passed, 191 skipped, 1 failed in 26 min. The failure is the stale
  local nornax checkout
  (`test_rollout_gradient_with_the_topology_rebuilt_inside_the_scan`), as at K, C,
  the tower follow-up and Z-a.
- **Distributed tier** (`tests/distributed`, `JAX_ENABLE_X64=1`, four forced host
  devices, `-n 4`): 53 passed, 2 skipped (the two opt-in upstream ragged-all-to-all
  checks) in 21 min.
- **`test_shards.py check`** (four forced host devices, x64): 1,910 tests, each in
  exactly one of the 11 shards; no overlap, no gap.
- **Bench tooling** (`pytest -n 0 bench/tests`): 93 passed.
- **CI benchmark guard on real**, run the way CI runs it (`--n 384 --runs 2 --warmup 1
  --preset fast --basis real --theta 0.6 --leaf-size 16 --max-order 4 --dtype
  float32`, `JAX_ENABLE_X64=1`, plus this host's jax 0.11.2 `-vectorize-loops=false`):
  passes. jerk_fast / acc 1.00 (bound 0.4-8), jerk_accurate / fast 1.84 (0.9-15),
  td2 / jerk 1.48 (0.8-12), td3 / td2 1.26 (0.8-12); accurate-vs-fast jerk rel-L2
  2.116e-5 (floor 1e-6), as at C.
- **Odisseo (G4, CPU)** against this branch, from a `git archive` of
  `jaccpot-integration` ac3cb5d (after #31), with nornax 2eab7a5 (its #14) and the
  overlay env: `test_integration_api.py`, `test_strict_velocity_verlet_policy.py`,
  `test_softening.py`, `test_dehnen_error_config.py`, `test_differentiable_fmm.py`
  (`-n 8`): 52 passed in 4 min; `test_mesh_lane.py` on two forced host devices: 15
  passed. nornax's `tests/unit/test_jaccpot_adapter.py` and
  `tests/integration/test_jaccpot_adapter_smoke.py`: 9 passed.
- **pre-commit** on the 112 changed files: clean.
- **Goldens:** as above; `test_constructor_state_golden.py` passes on the regenerated
  file, and no other golden changed.
- `bench/annotation_census.py`: shape-annotated array parameters 696 -> 696, bare
  `Array` parameters 1,543 -> 1,535, non-array parameters 3,399 -> 3,353, shaped
  share 31.1 % -> 31.2 %, `@jaxtyped` functions 111 -> 111, modules 118 -> 118.
- **Not run:** any GPU. The GPU pins (S1-S6) are next, from this branch: `dce_pins.py`
  S4 / S4b and `fused_memory_budget.py` drop only removed keywords, so the pins run
  the same configuration as `main-15ceca4` / `main-17de581`.

**Findings:**
- `tests/perf/test_perf_metrics.py` compared the far fields `pair_grouped` and
  `class_major`; the latter has raised since X3, and the perf tier is opt-in, so
  nobody saw. It now compares the near-field modes it ran with.
- `JAX_ENABLE_X64=0` is not an fp32 switch for jaccpot (above).
- Also never written, besides the five upward timings:
  `refresh_dual_final_symmetry_seconds` and `refresh_dual_source_motion_seconds`
  (always 0.0; conditional on
  `_refresh_timing_substages_measured`). Left, since the decision named the upward
  ones.
- `_prepare_solidfmm_downward_sweep` reads neither `adaptive_order`, `p_gears` nor
  `use_pallas` (the siblings of the `far_pairs_by_gear` it also ignored), so the
  adaptive-order gear buckets only feed the `recent_dual_far_pairs_by_gear_counts`
  diagnostic: the downward runs every pair at the full order. Left as decided (the
  bucketing stays); worth a look before adaptive order is claimed anywhere.

**Left for Z-c:**
- Docs and notebooks name removed options: `ARCHITECTURE.md` (the `basis_mode` /
  `rotation` seam of §5 and §10's static-discriminator invariant), 21 files in
  `docs/` (most of them dated plans and audits), and nine notebooks
  (`adaptive_vs_fixed_benchmark`, `benchmark_gpu_radix_runtime`,
  `benchmark_gpu_single_n_memory` (it also runs the deleted
  `profile_prepare_memory_split.py`), `benchmark_runtime_accuracy_copy`,
  `benchmark_runtime_accuracy_new`, `benchmark_runtime_large_N`,
  `benchmark_runtime_large_N_accuracy`, `benchmark_runtime_large_N_performance`,
  `benchmark_runtime_rtx2080_focused`; `benchmark_runtime_breakdown_h200`'s one
  source cell was cheap and is done). `README.md`'s octree and basis paragraphs are
  updated here, since they described this PR's behaviour.
- Remaining names in `jaccpot/`, `tests/`, `bench/`, `examples/` are history in
  comments and docstrings, `cleanup_inventory.py`'s regexes, the generic-validation
  tests that pass `"cartesian"` / `"complex"` as unknown names, and the prepared
  states' own `execution_backend` tag (`"radix"` / `"large_n"`, a static pytree field
  read only by the compiled-profile fingerprint): a Z-c candidate.
- `_adaptive_policy.py` keeps `Inexact` on `multipole_packed` (its docstring
  describes the complex packing); `DistributedFMMConfig.local_walk` is single-valued;
  `JACCPOT_LARGE_N_COMPILED_STATE_MODE` is read nowhere (the harness sets it).
- The four comments in `jaccpot/` that name a test Z-a moved (Z-a's list).

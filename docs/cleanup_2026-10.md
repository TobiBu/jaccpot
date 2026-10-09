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
| D2 | | The fused strict lane is the default; `large_n_gpu` builds `static_radix` | CPU suite, GPU defaults gate, pins, Odisseo G4 | open |

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
GPU gate is still to run.

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
| `JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH` | 1 | 0 | with 1, the default static-radix prepare raised without a profile on disk |
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


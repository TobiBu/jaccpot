# North-star Phase 3 — scale the treecode-walk multi-GPU FMM to 100k–1M/GPU

_Resume doc for a fresh session. Phases 1–2 are DONE and pushed; this is the remaining
scale work. Written 2026-07-20._

## Progress log

- **2026-07-20 (later): branch reconciled onto main + Phase 3a DONE (local, not committed).**
  Pulled `feat/phase5-multigpu-foldin` (was 83 behind); the fast-forward included merge
  `53f0bc4` "Merge branch 'main'…" — this IS the §"Branch / merge note" reconciliation
  (driver now imports M2L/L2L/L2P from `runtime.kernels.core`, treecode builder still from
  `runtime._interaction_cache`; `local_walk` treecode work + the `far_pair_count` 0-pad seam
  survived intact). Verified: CPU driver suite 6/6 green; GPU treecode parity ndev=2 per=1000
  `self_ovf=0`, `overflow=False`, aggL2 7.77e-4 (better than the memo's 3.2e-3 — main's gains).
  **Phase 3a (below) implemented + GPU/CPU validated.** NOT yet committed to the branch.

## Where we are (Phases 1–2, DONE + pushed)

**Goal of the North-star:** lift the distributed FMM's per-GPU-N ceiling by replacing the
overflowing self dual-tree walk with the single-GPU fast-lane **treecode walk**. Diagnosis:
the ceiling is `self_queue_overflow` — the dual-tree walk's transient *pair-queue*, not the
final lists (at leaf=128/per=20000 the final lists are tiny yet the queue blows up).

**Landed on `feat/phase5-multigpu-foldin` (origin, commit `4489d16` and its history):**
- `DistributedFMMConfig.local_walk` ∈ {`dual_tree` (default), `treecode`}. The `treecode`
  branch in `_make_fn`/`fn` (`jaccpot/distributed/fmm.py`) calls
  `_build_treecode_artifacts_strict_streamed(tree, geom, theta, mac_type="dehnen", ...)` and
  feeds its `compact_far_pairs` → the same real M2L (leaf-only targets → L2L no-op) and its
  **self-excluded** `neighbor_list` → the same combined P2P. Cross-domain LET unchanged.
- **Seam gotcha (already fixed):** treecode far pairs are **0-padded (not −1)** with the true
  count in `far_pair_count`, so `s_active` uses `far_pair_count` on the treecode path (a `>=0`
  test counts padding → degenerate delta=0 M2L → NaN).
- **Validated:** CPU pytest `test_driver_local_walk_treecode`; GPU (A100) multi-GPU parity
  ndev 2/3/4/5 (per=1000, `treecode`+`auto_scale_caps`): all overflow-free, `self_ovf=0`,
  aggL2 vs direct 3.2e-3/7.2e-3/1.8e-2/2.6e-2 — vs the dual-tree walk's **0.53 garbage** (self
  overflow) at ndev=2. The rising aggL2 with ndev is the **cross-domain far** (θ_cross=0.1 LET,
  grows with domain count), NOT the treecode (self force is per-domain + exact).

## How to resume (environment)

- **Worktree:** `/export/home/tbuck/jaccpot-phase5-wt` on `feat/phase5-multigpu-foldin`
  (isolated; shared checkout `/export/home/tbuck/jaccpot` stays on its own branch). jaccpot is
  editable-installed from the shared checkout, so import it from the worktree via the
  **sitecustomize finder-repoint**: `PYTHONPATH=<scratchpad>` where scratchpad has a
  `sitecustomize.py` doing `__editable___jaccpot_0_0_1_finder.MAPPING['jaccpot'] =
  '/export/home/tbuck/jaccpot-phase5-wt/jaccpot'` (see [[jaccpot-worktree-isolation]]).
- **Python:** `/export/home/tbuck/micromamba/envs/odisseo/bin/python`. jaccpot forces x64.
- **GPUs:** `autocvd -n N -l -o -q` → `CUDA_VISIBLE_DEVICES`. 8×A100-40GB box; free set varies.
- **Pallas M2L:** export `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1`. On Ampere the treecode
  walk auto-uses its Pallas kernel; on CPU it falls back to JAX (so CPU tests work).
- **Reusable scratchpad scripts (this session):** `treecode_swap_gpu.py` (GPU parity, argv
  ndev/per/jit/config), `treecode_swap_parity.py` (CPU parity), `inspect_walks.py` (dump
  walk outputs), `steady_time.py`/`scale_bench.py`/`leaf_sweep.py` (build-once timing +
  cap-calibration). Copies of the earlier ones live in
  `Odisseo/benchmark_a100/phase5_5c_multigpu/`.
- **Test:** `XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu pytest
  tests/test_distributed_fmm_driver.py -o addopts="" -q` (CPU); drop the env + `autocvd` for GPU.

## Phase 3 tasks (in order)

### 3a — Right-size the treecode `near_cap` (DONE 2026-07-20, local; correctness + perf)
**DONE.** `DistributedFMMConfig` gained `treecode_near_cap` / `treecode_far_cap`
(`Optional[int]`, default None). In the driver's treecode branch (`_make_fn`/`fn`,
`jaccpot/distributed/fmm.py`) the flat buffers are now right-sized from the STATIC local
tree instead of the builder's fixed defaults: `near_cap = max(1<<14,
max_neighbors_per_leaf * num_leaves)` and `far_cap = max(1<<14, max_interactions_per_node
* num_leaves)` (both keyed off `with_scaled_caps`-scaled fields, so the auto-scale retry
grows them). `_build_treecode_artifacts_strict_streamed` gained an optional `near_cap`
kwarg (explicit override wins; None keeps the old env/1<<21 default → single-GPU fast lane
byte-identical). The `_TreecodeWalkDiag` now surfaces REAL `far_overflow` (`far_pair_count
>= far_cap`, clamped) / `near_overflow` (`near_pair_count > near_cap`, unclamped counts) —
previously forced to 0, i.e. SILENT truncation with no diagnostic — so `_reduce_overflow`
→ `auto_scale_caps` self-heals both, exactly like the dual-tree caps.
VALIDATED (GPU A100 ndev=2 per=1000, `treecode_swap_gpu.py`): default caps give bit-identical
parity to the 2M-buffer baseline (aggL2 7.7657e-4, `cap_retries=0`); forcing a tiny
`treecode_near_cap=4000` → `near_overflow` fires → auto_scale retries 4000→8000→16000
(`cap_retries=2`) → clears with identical forces; same for `treecode_far_cap=100`
(`cap_retries=1`). CPU driver suite 6/6 green before AND after the edits.
NOTE: the near-source-leaf table `S_near = max_neighbors_per_leaf + cross_max_neighbors_per_leaf`
(Pallas near backend densification) is a SEPARATE truncation not touched here — watch it at scale.

_(original 3a spec, for reference:)_
### ~~3a — Right-size the treecode `near_cap`~~ (spec)
The treecode near buffer defaults to `1<<21` (env `JACCPOT_STATIC_STRICT_FUSED_TREECODE_NEAR_CAP`,
read in `_build_treecode_artifacts_strict_streamed`, `jaccpot/runtime/_interaction_cache.py`).
Two problems: (1) `_combined_neighbors` then chews a 2M-edge array per device (slow); (2) **at
~1M/GPU the actual near-pair count can EXCEED `1<<21` and the overflow guard is SKIPPED under
trace** → silent truncation → wrong forces. Fix: size it to the actual bound and make it
explicit. Options:
- Add `DistributedFMMConfig.treecode_near_cap: Optional[int]` and, in the treecode branch, set
  the env before the call (host-side, in `_make_fn`) OR — cleaner — call the lower-level
  `build_treecode_far_pairs_and_neighbors` directly (as `tests/experimental/test_treecode_
  graft_solidfmm.py` does) with explicit `near_capacity`/`far_pair_capacity`, replicating the
  `_treecode_mac_extents` + topo-padding that `_build_treecode_artifacts_strict_streamed` does.
- Safe bound: `near_cap ≈ num_leaves × max_neighbors_per_leaf` (per-device), with a factor of
  safety; `far_cap ≈ num_leaves × (far-per-leaf)`. Validate empirically: assert the returned
  `near_counts.sum()` / `far_pair_count` are **< cap** at the target N (no silent overflow).

### 3b — Cross-walk: `auto_scale_caps` or treecode-swap it too
The cross-domain LET walk (`dual_tree_walk_cross_impl`, via `yggdrax.distributed.cross_walk`)
is STILL a dual-tree walk and overflows at higher ndev / connected ICs (that is why the ndev
2–5 sweep used `auto_scale_caps`). For separated clusters the cross term is negligible so
forces stay correct; for a **connected** IC the cross-domain far matters, so decide:
- Cheapest: rely on `auto_scale_caps` (already works) — but the padded cross pair-queue is the
  same overhead-scaling problem, so it may get slow/huge.
- Proper: apply the SAME treecode-walk swap to the cross walk (a `local_walk`-style option for
  the cross traversal). Bigger, but removes the cross ceiling too. Recommended before claiming
  full scale on connected ICs.

### 3c — A realistic connected IC
Replace the separated-cluster benchmark IC with a **domain-decomposed galaxy disk** (the
single-GPU fast lane's `notebooks/scalability/galaxy_disk_fmm_large_n.py` generates one;
partition it across devices with the driver's `partition_for_devices`). This is what exercises
the cross-domain far honestly and matches the single-GPU comparison. Adversarial single dense
blobs are NOT representative (they inflate near-pairs for ANY near-field).

### 3d — Scale runs + weak/strong scaling
With 3a/3b/3c in place, push per-GPU N to 100k → 1M with `local_walk="treecode"`. Confirm
`self_ovf=0` AND `not overflow` AND (assert final list sizes < caps) AND subsampled aggL2 vs
direct < a few %. Then **build-once steady-state timing** (NOT `distributed_fmm_accelerations`,
which recompiles ~250 s/call — use `make_force_evaluator(..., jit=True)` and call the compiled
fn repeatedly; see `steady_time.py`). Weak scaling (fixed per-GPU N, ndev 2→5) + strong (fixed
total, vary ndev); compare per-GPU throughput to the single-GPU O(N) curve (fast lane does
4M/GPU). This is the payoff: demonstrate the ceiling is gone.

## Gotchas already learned (don't rediscover)
- **jit=True illegal-address crash is INTERMITTENT** (nondeterministic OOB); run each perf/probe
  eval in its OWN process. Phase-2 correctness used jit=False. Probe jit=True + treecode
  separately before relying on it for timing.
- `distributed_fmm_accelerations` **rebuilds+recompiles every call** (~50–250 s). Build once for
  timing.
- The treecode uses **dehnen** MAC for accuracy-profile parity + multi-step stability; do NOT use
  bh (dynamically unstable). θ_cross ≤ 0.1 (under-separated far-field goes garbage).
- aggL2 is worst-weighted; the median (p50 ~2.8e-4 historically) is the fair accuracy metric.
- Kill grid/sweep processes by PROCESS GROUP and verify zero before relaunch.

## Verification
- CPU pytest green (incl. `test_driver_local_walk_treecode`).
- GPU: treecode ndev 2–5 parity vs direct, overflow-free, final list sizes < caps (assert).
- Scale: 100k–1M/GPU overflow-free + accurate + weak/strong scaling curve; per-GPU throughput
  vs the single-GPU 4M/GPU reference.

## Branch / merge note
`feat/phase5-multigpu-foldin` is on the OLD `jaccpot.runtime._fmm_impl` layout; main is on
`jaccpot.runtime.kernels.core`. PR #47 (base main) has a merge conflict in `distributed/fmm.py`
to resolve before landing (re-apply the 5c + treecode changes onto main's driver). The
North-star wants main's newest fast lane, so reconcile onto main at some point.

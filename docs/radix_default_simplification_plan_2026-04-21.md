# Radix Default Simplification Plan (2026-04-21)

## Goal
Make the **high-performance + low-memory radix large-N path** the default and reduce runtime/config complexity by removing or hard-locking low-value flexibility.

Scope focus:
- Preserve/strengthen production path: radix + `large_n_gpu` + streamed pair-grouped + bucketed nearfield + minimum-memory behavior.
- Reduce branch count in runtime resolution and nearfield execution.
- Keep correctness and benchmark parity.

Out of scope (for now):
- Re-tuning physics/mac defaults for non-radix and octree research workflows.
- Removing all non-radix support globally in one shot.

## Current Complexity Hotspots

### 1) Public config surface is too wide
Files:
- `jaccpot/config.py`
- `jaccpot/runtime/_fmm_impl.py`

High-churn knobs that multiply branches:
- `runtime_path` (`auto|legacy|large_n`)
- `nearfield_mode` (`auto|baseline|bucketed`)
- `memory_objective` (`balanced|throughput|minimum_memory`)
- `grouped_interactions`, `farfield_mode`, `streamed_far_pairs`
- `traversal_config` + `max_pair_queue` + `pair_process_block` (overlapping controls)
- retention/cache toggles (`retain_interactions`, `retain_traversal_result`, `enable_interaction_cache`)

### 2) Runtime override resolver has many coupled conditionals
File:
- `jaccpot/runtime/_fmm_impl.py` (`_resolve_runtime_execution_overrides`)

Observed issues:
- Multiple interacting branches by backend, preset, tree type, memory objective, grouped mode, explicit-vs-implicit knobs.
- Easy to accidentally pin oversized traversal config and regress memory.

### 3) Dual-path nearfield logic still carries compatibility branches
Files:
- `jaccpot/runtime/_large_n_pipeline.py`
- `jaccpot/runtime/_large_n_nearfield.py`
- `jaccpot/runtime/_large_n_types.py`

Observed issues:
- `radix_fast_lane` state booleans and fallback paths still complicate control flow.
- Legacy-compatible checks stay in hot execution paths.

### 4) Benchmarks/scripts still expose old path toggles
Files under `examples/` and helper scripts.

Observed issues:
- Easy to run legacy-like configs by accident (or pin oversized traversal configs).
- Notebooks carry duplicated bootstrap blocks with different defaults.

## Target Simplified Runtime Contract

For production large-N GPU path, enforce by default:
- `preset="large_n_gpu"`
- `tree_type="radix"`
- `expansion_basis="solidfmm"`
- `working_dtype=float32`
- `memory_objective="minimum_memory"`
- `farfield_mode="pair_grouped"`
- `grouped_interactions=False`
- `streamed_far_pairs=True`
- `nearfield_mode="bucketed"`
- runtime-managed traversal caps (no user-pinned oversized traversal config by default)

Key principle:
- **One blessed default execution lane** for production, with guardrails against misconfiguration.

## Structured Commit Plan

### Commit 1: Add strict production runtime profile and central guard
**Message**: `runtime: add strict radix-largeN production profile and guardrails`

Changes:
- Add a single helper (e.g. `_resolve_production_large_n_profile`) in `_fmm_impl.py`.
- Move production defaults/locks into this helper.
- In constructor or override resolution, when profile is active, coerce/validate:
  - bucketed nearfield
  - pair_grouped farfield
  - grouped interactions off
  - minimum-memory traversal defaults
- Emit clear warning/error if conflicting explicit overrides are provided.

Validation:
- Unit tests for profile coercion and conflict errors.
- Existing fast-lane dispatch tests continue passing.

### Commit 2: Remove runtime_path branching from hot path
**Message**: `runtime: deprecate legacy runtime_path branching in large-N execution`

Changes:
- In `_large_n_pipeline.py`, collapse runtime-path conditionals to default large-N path.
- Keep temporary compatibility shim: `runtime_path` accepted but ignored (or warns once).
- Remove dead/internal conditionals where `runtime_path==legacy` is no longer used for production flow.

Validation:
- Integration tests covering prepared/evaluate flow.
- Benchmark smoke test still reports fast-lane active.

### Commit 3: Simplify nearfield mode handling to bucketed default
**Message**: `nearfield: lock large-N radix path to bucketed execution`

Changes:
- Reduce large-N path nearfield mode branching; bucketed becomes enforced path.
- Keep baseline nearfield only for non-large-N or explicit non-production contexts (if still needed).
- Trim `LargeNExecutionConfig` fields that are now invariant in production path.

Validation:
- Unit test: resolved nearfield mode for large-N profile is always `bucketed`.
- Regression test comparing accelerations vs prior known-good output.

### Commit 4: Collapse overlapping traversal controls
**Message**: `runtime: unify traversal control and disallow oversized explicit seeds by default`

Changes:
- For production profile: ignore/reject explicit `traversal_config`, `max_pair_queue`, `pair_process_block` unless explicitly in expert override mode.
- Keep one code path for traversal sizing and retry growth.
- Strengthen capping to avoid OOM-prone explicit config even if provided.

Validation:
- Test explicit oversized traversal input gets capped/rejected predictably.
- Memory notebook smoke at `N=524288` and `N=1048576` no unexpected OOM from config pinning.

### Commit 5: Reduce config API surface (soft deprecation)
**Message**: `api: soft-deprecate low-value runtime knobs for production path`

Changes:
- In `config.py`, mark knobs as advanced/deprecated in docstrings/comments:
  - `runtime_path`, `nearfield_mode=baseline`, `memory_objective!=minimum_memory`, grouped toggles in large-N profile.
- Add runtime warnings with removal timeline.
- Keep backward compatibility for one release cycle.

Validation:
- API tests confirm warnings and backward compatibility.

### Commit 6: Notebook/script cleanup to one canonical setup
**Message**: `examples: align benchmarks to canonical radix low-memory fast path`

Changes:
- Normalize benchmark notebooks/scripts to a shared helper that builds canonical FMM kwargs.
- Remove legacy path toggles from primary benchmark notebooks.
- Remove duplicated bootstrap config blocks.

Validation:
- Re-run:
  - `examples/benchmark_runtime_large_N_performance.ipynb`
  - `examples/benchmark_gpu_single_n_memory.ipynb`
- Confirm fast-lane active and expected runtime/memory envelope.

### Commit 7: Hard removal pass (after deprecation window)
**Message**: `runtime: remove deprecated legacy branches and obsolete knobs`

Changes:
- Remove deprecated conditional branches and obsolete parameter wiring.
- Simplify type literals and validation branches.
- Delete obsolete tests that only cover removed behavior; add focused tests for canonical path.

Validation:
- Full pytest suite.
- Benchmark sanity on target GPUs (production and small-GPU smoke).

## Test/Benchmark Gates Per Phase

Minimum gate for each commit batch:
1. Fast-lane dispatch tests and prepared-state cache tests pass.
2. One runtime sweep smoke (`N <= 1,048,576`) with no unexpected OOM.
3. One memory notebook smoke run confirms no regression in peak trend.

Release gate before merge:
1. Full `pytest` pass.
2. Runtime large-N sweep comparison vs last known-good baseline.
3. Memory peak comparison at leaf size 256.

## Risk Register

- Risk: silent behavior changes for existing users relying on legacy knobs.
  - Mitigation: soft deprecation + warnings + migration notes.
- Risk: simplification accidentally removes scientific/accuracy workflows.
  - Mitigation: keep non-production pathways behind explicit research profile until confirmed.
- Risk: notebook drift reintroduces pinned oversized traversal settings.
  - Mitigation: central helper + lint/check for forbidden config patterns in notebook exporter scripts.

## Suggested Ownership/Execution Order

1. Runtime core simplification (Commits 1-4).
2. API/deprecation messaging (Commit 5).
3. Benchmarks/notebooks consolidation (Commit 6).
4. Hard removals after one stabilization cycle (Commit 7).

## Immediate Next Session Checklist

- [ ] Implement Commit 1 and Commit 2 in one PR.
- [ ] Add/adjust tests for profile coercion and runtime_path deprecation behavior.
- [ ] Run targeted benchmark smoke on chosen GPU.
- [ ] If stable, proceed with Commit 3 and Commit 4.


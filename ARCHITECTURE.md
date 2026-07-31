# jaccpot architecture

This document maps the runtime structure of jaccpot: how a call to the public
`FastMultipoleMethod` reaches the numerical kernels, how the code is layered,
where the Pallas/A100 fast paths gate on, and how numerical correctness is
guarded. It is aimed at contributors touching `jaccpot/runtime/`.

For *what* the FMM computes and *how to use it*, see [`README.md`](README.md).
This document is about *where the code lives and why*.

## 1. Layering at a glance

```
jaccpot/__init__.py            public surface (12 __all__ names)
        |
jaccpot/solver.py              FastMultipoleMethod  <-- the ONLY public class
        |                      preset-first facade; resolves preset/basis/advanced
        |                      config into engine constructor args
jaccpot/runtime/fmm/           orchestrator package (re-export seam)
        |                      __init__ re-exports the engine; must NOT be the
        |                      import that forms a cycle (see section 8)
jaccpot/runtime/_fmm_impl.py   FastMultipoleMethod engine (thin coordinator)
        |                      + 10 method-cluster mixins (section 4)
        |
jaccpot/runtime/kernels/       reusable numerical core (LEAF -- never imports
        core.py                the engine); M2L / L2L / eval / nearfield kernels
```

The **hinge of the design** is that the reusable kernel library
(`runtime/kernels/`) is a true leaf: it depends only on operators, tree
artifacts, and the shared constant/cache modules — never on the orchestrator.
`distributed/` and `experimental/` reach straight past the engine into
`kernels/`, which is what proves the leaf boundary is real.

## 2. Public API contract

`jaccpot/__init__.py __all__` is the stable surface downstream code (e.g. the
ODISSEO coupling) depends on. It is frozen by
[`tests/unit/test_public_api_surface.py`](tests/unit/test_public_api_surface.py):

| Name | Kind |
|---|---|
| `FastMultipoleMethod` | the solver class (in `solver.py`) |
| `FMMPreset` | preset enum (FAST / BALANCED / ACCURATE / LARGE_N_GPU) |
| `FMMAdvancedConfig`, `FarFieldConfig`, `NearFieldConfig`, `RuntimePolicyConfig`, `TreeConfig` | advanced config dataclasses |
| `MemoryObjective` | memory-policy literal |
| `ComplexSHBasis`, `RealSHBasis` | expansion bases |
| `OdisseoFMMCoupler` | ODISSEO integration adapter |
| `differentiable_gravitational_acceleration` | autodiff-able **direct-sum** oracle (the differentiable FMM itself is `FastMultipoleMethod.differentiable_accelerations`; see section 6) |

`FastMultipoleMethod` in `solver.py` is the sole public class name; the runtime
engine class (currently also named `FastMultipoleMethod` in `_fmm_impl.py`) is
an internal implementation detail reached only through the facade.

## 3. runtime/ package map

Modules are kept **at the `runtime/` level** (not nested under `runtime/fmm/`)
on purpose: while `runtime/fmm/__init__` still re-exports from `_fmm_impl`,
placing an engine-imported submodule under `fmm/` would re-form a package-init
import cycle (section 8).

**Shared leaves** (imported by both the kernels and the orchestrator; no back-deps):

| Module | Role |
|---|---|
| `fmm_constants.py` | tuning constants + `_env_int`/`_env_flag` helpers, traversal-config templates |
| `fmm_caches.py` | the process-level `OrderedDict` caches + byte accounting + key/get/put/clear |
| `dtypes.py` | `INDEX_DTYPE` single source of truth (int32/int64 via `JACCPOT_INDEX_PRECISION`) |

**Numerical core** (`runtime/kernels/`, leaf — never imports the engine):

| Module | Role |
|---|---|
| `kernels/core.py` | the whole numerical kernel library: M2L / L2L / evaluation / nearfield-targeting / downward builders (section 5) |
| `kernels/__init__.py` | curated re-exports consumed by `distributed/`, `experimental/`, `_large_n_pipeline`, tests |

**Orchestrator scaffolding** (sibling of `_fmm_impl`):

| Module | Role |
|---|---|
| `fmm_state.py` | resolved-config dataclasses + `_resolve_fmm_config`, tree-build artifacts, `FMMPreparedState` pytree, strict-refresh diag helpers |
| `fmm_presets.py` | `FMMPresetConfig` bundles + `get_preset_config` (FAST / LARGE_N_GPU); `FMMPreset` is re-exported from `jaccpot.config` (single definition) |

**Engine coordinator + method-cluster mixins** (section 4).

## 4. The engine: coordinator + mixins

`_fmm_impl.FastMultipoleMethod` is a thin coordinator (constructor, backend
plumbing, cache lifecycle, autotune-cache IO) that inherits its behaviour from
**10 method-cluster mixins**, each a sibling `runtime/fmm_<cluster>.py` module.
Methods were moved verbatim during the god-class breakup; `self` is unchanged;
cross-cluster calls resolve through the MRO.

```python
class FastMultipoleMethod(
    PrepareMixin, EvaluateMixin, StrictRunMixin, SweepsMixin, OverridesMixin,
    AutotuneMixin, PolicyMixin, DerivativesMixin, StrictCapProfileMixin,
    DiagnosticsMixin,
):
```

| Mixin (`fmm_*.py`) | Responsibility |
|---|---|
| `PrepareMixin` (`fmm_prepare`) | build the `FMMPreparedState`: tree upward pass, downward/far-pairs, nearfield setup |
| `EvaluateMixin` (`fmm_evaluate`) | evaluate a prepared state into accelerations/potential (L2P + nearfield) |
| `StrictRunMixin` (`fmm_strict_run`) | static-radix hot path: `refresh_prepared_state`, `strict_run_v2`, same-topology refresh, velocity-Verlet update |
| `StrictCapProfileMixin` (`fmm_strict_cap_profile`) | compiled-profile persistence for the strict lane |
| `PolicyMixin` (`fmm_policy`) | adaptive execution-policy decisions |
| `OverridesMixin` (`fmm_overrides`) | resolve runtime execution knobs (farfield/nearfield mode, traversal caps) |
| `AutotuneMixin` (`fmm_autotune`) | M2L chunk-size autotuning |
| `SweepsMixin` (`fmm_sweeps`) | delta-sign / convention sweeps |
| `DerivativesMixin` (`fmm_derivatives`) | jerk / time-derivative towers |
| `DiagnosticsMixin` (`fmm_diagnostics`) | `get_runtime_diagnostics` + shape diagnostics |

## 5. Kernel-family map (`runtime/kernels/core.py`)

The M2L translation kernels are unified behind a **static `basis_mode`** seam.
Because `basis_mode` (and `order`, `rotation`, `m2l_impl`, `chunk_size`,
`total_nodes`) are `static_argname`s, XLA specialises each `jax.jit` per static
combination — the merged kernels compile to the exact HLO each single-basis
kernel did, so consolidation is source-level dedup with no numerical change.

**M2L apply seam:**

- `_apply_m2l(..., *, order, basis_mode, rotation, m2l_impl)` — dispatches to:
  - `_apply_real_m2l` → real rot-scale (`_m2l_real_batch_kernel`) or the fused
    Pallas kernel `_m2l_real_batch_kernel_fused_pallas` when gated on
  - `_apply_complex_m2l` → solidfmm rotation (`_m2l_complex_batch_kernel`) or
    `_m2l_complex_batch_kernel_fused_pallas`

**M2L accumulate (both bases, via the seam):**

- `_accumulate_m2l_fullbatch` — one full interaction batch → `segment_sum`
- `_accumulate_m2l_chunked_scan` — chunked `lax.scan` reduction (bounded memory)
- grouped / class-major variants (`_accumulate_solidfmm_m2l_grouped[_class_major]`)
  — cached class blocks; already `basis_mode`-parametrised

**L2L / downward:** `_propagate_solidfmm_locals_by_level` (unifies real+complex
behind `basis_mode`), `_propagate_{solidfmm,real}_locals_to_children`, the
solidfmm downward-sweep builders.

**Evaluation:** `_evaluate_local_expansions_for_particles` (L2P), nearfield
targeting + scatter. The public `compute_gravitational_{acceleration,potential}`
convenience wrappers live in `_fmm_impl.py` (not in `kernels/core`) because they
construct/drive the engine.

**Intentionally NOT merged** (distinct math / dispatch): the Cartesian path
(`_evaluate_local_cartesian_with_grad_batch`), the complex-only
derivative/jerk towers, and cached-vs-uncached M2L dispatch.

## 6. Config resolution, presets, and the autodiff export

- **Presets** (`config.FMMPreset`): FAST and LARGE_N_GPU resolve through
  first-class bundles in `fmm_presets.get_preset_config`; BALANCED and ACCURATE
  resolve through advanced-config defaults in `solver._default_advanced_for_preset`
  (they map to `expanse_preset=None` and never reach `get_preset_config`).
- **Advanced config** is the `FMMAdvancedConfig` group (tree / farfield /
  nearfield / runtime); `fmm_state._resolve_fmm_config` normalises constructor
  inputs into a validated `FMMResolvedConfig`.
- **End-to-end differentiable FMM (single GPU).** `FastMultipoleMethod.differentiable_accelerations(state, positions, masses)`
  gives exact gradients of the FMM force w.r.t. particle **positions** and **masses**
  under a fixed-topology contract: the tree (Morton order, node membership, MAC
  accept/reject, the M2L interaction list, near/far partition) is held constant from a
  pre-built `state` while the numeric pipeline (P2M, COM centers, the linear M2M/M2L/L2L
  translations, L2P, near-field P2P) is re-evaluated on the live inputs, so `jax.grad`/
  `jax.vjp` transpose it exactly (Route A; the translation cascade is linear so its VJP
  is free). `prepare_state` (tree build) is not traceable, so `state` must be built once,
  concretely, before `jax.grad`. Requires a radix `FMMPreparedState` with a solidfmm
  basis (complex or real submode). The M2L defaults to the pure-JAX path but can opt
  into the differentiable **fused-Pallas M2L fast lane** with
  `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` on an Ampere+ GPU (the fused kernels now
  carry a `custom_vjp`, PR-2). The near-field defaults to the bucketed pure-JAX
  edge-list kernel; `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1` re-expresses it
  **leaf-major** and routes it through the radix fast lane (PR-4) — the same edge set
  and the same force (bit-identical checksums), a different traversal. It stays
  **opt-in**, but the recommendation is N-dependent: at N≤4096 the two traversals are
  within ±20% with no consistent winner, while at **N=200000 the bucketed reverse OOMs
  (30 GB peak) and the fast lane completes in 6.8 GB** — turn it on for any
  differentiable radix run at N ≳ 10^5. `LargeNPreparedState` is differentiable via
  `runtime/_large_n_grad.py` (needs `retain_far_pairs_for_grad=True`), and the
  `large_n_gpu` preset is the leanest galaxy-scale option measured: **1M particles,
  forward 2.5 s and forward+backward 69 s at 11 GB peak**.
- **Configuring the grad path.** `GradConfig` (`jaccpot/config.py`) is the
  supported interface; `runtime/grad_options.py` resolves it under
  `explicit field > JACCPOT_* env var > measured default`, so the environment
  variables above remain a working fallback. `nearfield_lane` defaults to
  `"auto"` and switches to the fast lane at N >= 100000 — the bucketed reverse
  OOMs there, so the crossover is applied rather than documented. Three gates
  (fused-Pallas M2L, the two analytic VJP rules) are read at trace time deep in
  the kernels with no argument channel; `grad_option_overrides` installs them as
  context-locals for the duration of one call.
- **Where the reverse rules live.** `nearfield/grad.py` holds both analytic
  near-field reverses — `_pair_accel_cvjp` (bucketed, rematerialized pair terms)
  and `_leafpair_accel_analytic_vjp` (leaf-major, O(N) memory) — plus the
  occupancy-tier builder and its cache. The `custom_vjp` wrappers stay in
  `near_field.py` next to the forward kernels they wrap, which is what keeps the
  two modules acyclic.
- **Differentiable multi-GPU FMM.** `distributed.make_force_evaluator(config, ndev,
  cap, mesh, differentiable=True)` puts the `shard_map` pipeline under
  `jax.grad`/`jax.vjp` under the same fixed-topology contract. Because the tree is
  built *inside* `shard_map` (there is no host-side `prepare_state` here), the seam
  is in the body: topology from `stop_gradient`-ed inputs, numerics re-evaluated on
  the live ones. Forward values are bit-identical. Three distributed-specific
  points: the near field must go through `_radix_fast_lane_prepacked_accel_cvjp`
  (raw `pallas_call` has no autodiff rule) and `nearfield_chunk` therefore raises;
  the L2L cascade needs a **static** level bound, whose safe default
  (`num_internal - 1`) is loose, tightenable via `l2l_num_levels`, and reports
  truncation as `l2l_level_overflow`; and the halo exchange is forced onto the
  `all_gather`-based ragged path, because executing the reverse of
  `jax.lax.ragged_all_to_all` corrupts every later ragged exchange in the process
  (`_buffered_ragged_exchange`, and the guard test
  `test_forward_survives_a_gradient`). Verified on 2 GPUs at N=64: FD-vs-AD 2.7e-08
  (positions) / 5.9e-09 (masses), `grad(FMM)` vs `grad(direct sum)` 1.1e-03 /
  4.5e-03 against a 5.9e-03 forward force error. Correct, **not fast**: the reverse
  costs roughly 6-11x the forward at N=512-8192, with a 3-6 min reverse compile and
  ~30% run-to-run noise on a shared host. The reverse hotspot is **unprofiled** —
  tightening `l2l_num_levels` is worth ~6% at N=512 and nothing at N=2048, so it is
  a correctness knob, not the speed lever. Untested above 2 devices — see
  `docs/differentiable_fmm_distributed_audit.md` ("What is not covered"). The
  host-side `distributed_fmm_accelerations` stays non-differentiable (NumPy
  partition + reassembly).
- **User guide:** `docs/differentiable_fmm.md`. Engineering history and the
  measurement record: `docs/differentiable_fmm_audit.md`,
  `docs/differentiable_fmm_design.md`, and for multi-GPU
  `docs/differentiable_fmm_distributed_audit.md`.
- `differentiable_gravitational_acceleration` remains the deliberately differentiable
  **direct O(N²) sum** — retained as the simple exact-gradient reference and the
  **gradient oracle** for tests (`grad(FMM)` must match `grad(direct-sum)` to FMM force
  accuracy), not as "the" differentiable path.

## 7. Pallas / A100 gating

Pallas GPU kernels require **Ampere+ (sm_80)**; on older GPUs and CPU they are
auto-gated off and the pure-JAX path runs (CPU CI can still lower Pallas with
`interpret=True` as a smoke test).

**Backend selection is version-dependent** and funnelled through one place:
`jaccpot/pallas/_compat.pallas_backend_kwargs`. JAX <= 0.9.0.x took
`pallas_call(backend="triton")`; 0.9.1 removed that kwarg and infers the backend
from the `compiler_params` dataclass type instead. The M2L kernels must name
Triton either way -- Mosaic-GPU rejects them (small per-(pair, coeff) tiles its TMA
cannot express; fp64 in the complex kernel) -- so the shim raises rather than
letting a non-Triton request fall through to a default. The near-field and
treecode kernels already passed `pallas.triton.CompilerParams` and needed no
change.

- Fused M2L Pallas is gated by `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` **and**
  the Ampere+ hardware support check (`_real_m2l_pallas_active` /
  `_fused_complex_m2l_pallas_active`, evaluated at trace time; both now gate on the
  fused kernels' own sm_80 support functions).
- The fused M2L kernels are **differentiable** via module-level `custom_vjp` wrappers
  (`m2l_{core_z_real,complex_fused,real_fused}_pallas_cvjp`): Pallas forward +
  reverse. `_apply_{real,complex}_m2l` route through the wrappers, so the same flag
  enables the fused M2L on both the forward and the `differentiable_accelerations`
  grad path (PR-2). `pallas_call` itself still has no autodiff rule — the wrapper
  supplies it. The reverse is a **fully-fused analytic VJP kernel**
  (`m2l_{real,complex}_fused_vjp_pallas`, default; `JACCPOT_FUSED_M2L_VJP=0` falls back
  to autodiff of the twin), so both forward and reverse run as single Pallas launches.
- Nearfield Pallas is resolved from `pallas_nearfield_fused_supported()` into the
  engine's `use_pallas`. Two differentiable near-field lanes exist and they are **not**
  interchangeable:
  - `_radix_fast_lane_prepacked_accel_cvjp` (`nearfield/near_field.py`) is the
    **production** rule: Pallas forward + an analytic O(N) leaf-pair reverse. It is what
    `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1` puts on the
    `differentiable_accelerations` grad path (PR-4), driven by a leaf-major payload
    that `runtime/_nearfield_fastlane.py` transposes on the host from the radix state's
    CSR neighbour list. It engages only when the engine's `use_pallas` is on (Ampere+);
    with `use_pallas=False` the same flag still selects the leaf-major traversal but
    runs the tiled pure-JAX prepacked kernel under ordinary autodiff — same force,
    but no O(N) reverse.
  - `nearfield_fused_leaf_pallas_cvjp` / `nearfield_leafpair_pallas_cvjp`
    (`pallas/nearfield_fused_leaf.py`, PR-2) are **unit-level VJP oracles**. Their
    reverse is `jax.vjp` of a dense twin that materialises a `(leaves, W_t, K, 3)`
    tensor — fine at test scale, ~50 TB at the fiducial large-N config. Keep them for
    `tests/unit/test_custom_vjp_parity.py`; do not add a second grad-path caller.
- **fp32 matmul precision is a known accuracy floor in the M2L.** XLA lowers fp32
  matmuls on Ampere to **TF32** (~10-bit mantissa) by default, and neither M2L
  basis sets `precision=`. Measured against a float64 reference (real basis, max
  rel err): 5.7e-04 at order 4, 5.7e-04 at order 6, 5.6e-04 at order 8 —
  i.e. **~6e-04 regardless of expansion order**, so raising the order past 4 buys
  nothing in fp32. Under `jax.default_matmul_precision("highest")` the same cases
  give 1.5e-06 / 2.4e-06 / 1.8e-06 (~300x better). The complex basis behaves the
  same (3.7e-04 -> 5.6e-08 at order 2). The L2P path *does* set
  `lax.Precision.HIGHEST` (`downward/local_expansions.py`), so this is an
  inconsistency rather than a considered trade. Not changed: fixing it moves
  forward numerics package-wide (the golden oracle would need regenerating) and
  costs throughput, so it wants a preset-level decision. This floor is also why
  `tests/integration/test_fmm.py::test_solidfmm_m2l_ignores_padded_compact_far_pairs`
  compares against a float64 reference with a 2e-03 bound instead of asserting
  padded == exact — the latter passed on JAX 0.9.x only because both fp32 paths
  were identically wrong.
- **Org rule for GPU runs:** select a free GPU with autocvd *before* `import jax`:
  `from autocvd import autocvd; autocvd(num_gpus=1, least_used=True)`.

## 8. Dependency DAG and the import-cycle rule

```
fmm_constants -> fmm_caches -> kernels -> {_interaction_cache, _large_n_pipeline,
  _octree_*} -> fmm_state -> _fmm_impl (engine) -> runtime/fmm -> solver -> __init__
```

`distributed/` and `experimental/` depend only on `kernels/` (not the engine).

**Cycle rule:** `runtime/fmm/__init__.py` must not eagerly import the engine
class in a way that reforms
`_interaction_cache -> fmm -> engine -> prepare -> _interaction_cache`.
Consumers import the class explicitly rather than through a package-init that
pulls the engine in. Keep `fmm_constants`/`fmm_caches`/`fmm_state`/`fmm_*` mixins
at the `runtime/` level until the engine is fully dissolved into `fmm/engine.py`.

## 9. Validation harness

- **Golden characterization oracle** —
  [`tests/characterization/test_fmm_golden.py`](tests/characterization/test_fmm_golden.py)
  drives the FMM over a grid (N, order, basis, farfield modes, outputs) and applies
  two gates: (1) an **inertness** gate — outputs match the committed `.npz` goldens
  under `tests/characterization/golden/` to float64 round-off (`atol=0, rtol≈1e-12`),
  and (2) a **physics** gate — each output is anchored to a direct O(N²) sum to a
  loose relative-L2 bound, so a regenerated golden can never silently encode a wrong
  answer. Refactors must keep it exact-green — any drift is a wiring bug, not a
  numerical one. Regenerate goldens intentionally with `JACCPOT_REGEN_GOLDEN=1`.
- **Public-API guard** —
  [`tests/unit/test_public_api_surface.py`](tests/unit/test_public_api_surface.py)
  freezes the 12 `__all__` names + `FMMPreset` members. Red = the refactor leaked
  into the public contract.
- **Runtime typecheck** — set `JACCPOT_RUNTIME_TYPECHECK=1` to enable beartype
  runtime checks over the suite.
- **GPU/Pallas parity (A100, manual/nightly):** run the fused-Pallas M2L parity
  tests + golden with `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` under
  `JAX_ENABLE_X64=1` to confirm the Pallas paths match the pure-JAX reference.

## 10. Kernel-consolidation invariant

When merging real/complex kernel families, the merge is numerics-preserving
**only** because every discriminator (`basis_mode`, `rotation`, `m2l_impl`,
`order`, `chunk_size`) is a `static_argname`, so XLA specialises the merged
`jax.jit` per static combination. This is source-level dedup, never a runtime
branch inside a compiled kernel. Any consolidation PR must show:

1. merged-vs-old **bit-identical** output on a fixed input grid (rtol=0),
2. the golden oracle exact-green,
3. the full suite with no new failures vs the frozen baseline,
4. an A100 Pallas-on vs pure-JAX parity run when the change touches Pallas.

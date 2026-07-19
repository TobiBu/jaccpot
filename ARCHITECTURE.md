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
| `differentiable_gravitational_acceleration` | autodiff-able **direct-sum** (see section 6) |

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
- `differentiable_gravitational_acceleration` is **not** an autodiff FMM — FMM is
  forward-only here. It is the deliberately differentiable **direct O(N²) sum**,
  provided for gradient-based use where the FMM approximation is not needed.

## 7. Pallas / A100 gating

Pallas GPU kernels require **Ampere+ (sm_80)**; on older GPUs and CPU they are
auto-gated off and the pure-JAX path runs (CPU CI can still lower Pallas with
`interpret=True` as a smoke test).

- Fused M2L Pallas is gated by `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` **and**
  the hardware support check (`_real_m2l_pallas_active` /
  `_fused_complex_m2l_pallas_active`, evaluated at trace time).
- Nearfield Pallas is resolved from `pallas_nearfield_fused_supported()` into the
  engine's `use_pallas`.
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

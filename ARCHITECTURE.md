# jaccpot architecture

This document maps the runtime structure of jaccpot: how a call to the public
`FastMultipoleMethod` reaches the numerical kernels, how the code is layered,
where the Pallas/A100 fast paths gate on, and how numerical correctness is
guarded. It is aimed at contributors touching `jaccpot/runtime/`.

For *what* the FMM computes and *how to use it*, see [`README.md`](README.md).
This document is about *where the code lives and why*.

## 1. Layering at a glance

```
jaccpot/__init__.py            public surface (15 __all__ names)
        |
jaccpot/solver.py              FastMultipoleMethod  <-- the ONLY public class
        |                      preset-first facade; resolves preset/basis/advanced
        |                      config into engine constructor args
jaccpot/runtime/fmm/           orchestrator package (re-export seam)
        |                      __init__ re-exports the engine; must NOT be the
        |                      import that forms a cycle (see section 8)
jaccpot/runtime/_fmm_impl.py   FMMEngine (thin coordinator)
        |                      + 9 method-cluster mixins (section 4)
        |
jaccpot/runtime/kernels/       reusable numerical core (LEAF -- never imports
        core.py                the engine); M2L / L2L / eval / nearfield kernels
```

The **hinge of the design** is that the reusable kernel library
(`runtime/kernels/`) is a true leaf: it depends only on operators, tree
artifacts, and the shared constant/cache modules — never on the orchestrator.
`distributed/` reaches straight past the engine into `kernels/`, which is what
proves the leaf boundary is real.

The packages under `jaccpot/`, one line each:

| package | what lives there |
|---|---|
| `basis/` | the real solid-harmonic basis (`RealSHBasis`), the only one |
| `operators/` | expansion algebra: real harmonics, rotations, M2L rotate/scale, translations, precision pinning |
| `upward/` | P2M / M2M and the expansion centres |
| `downward/` | the downward pass's containers (`LocalExpansionData`, `TreeDownwardData`); the M2L / L2L / L2P sweep itself is in `runtime/kernels/` |
| `nearfield/` | P2P, its fast lane and large-N blocks, its analytic reverse rules, the acceleration-derivative tower |
| `pallas/` | the fused Pallas kernels and their `custom_vjp` wrappers (section 7) |
| `runtime/` | the engine, config resolution, lane selection, the large-N pipeline, kernel dispatch (section 3) |
| `mutual/` | the momentum-conserving lane (below) |
| `distributed/` | the multi-GPU drivers: `fmm.py` (one `shard_map`), and `fused.py` + `cross.py` + `rollout.py` (one process per GPU) |

**The one path that does not go through this stack** is the mutual force:

```
jaccpot/nornax_adapter.py      BlockStepFMM  <-- the second public class
        |                      block-step KDK facade; no nornax import
jaccpot/mutual/               momentum-conserving evaluation path
        topology.py            host-side symmetric dual traversal (frozen)
        nearfield.py           leaf-pair P2P, +F/-F to both endpoints
        farfield.py            dual M2L, one evaluation per accepted pair
        force.py               rung/level vocabulary over the two above
```

It reaches `operators/` and `pallas/` directly and never touches
`runtime/`, because the property it exists to provide — `sum_i m_i a_i` zero to
round-off rather than to the truncation error — depends on evaluating each pair
**once**, and the production runtime is target-centric by construction (each pair
evaluated twice, independently). This is a deliberate second lane, not a
duplicate: see [`docs/momentum_conserving_fmm.md`](docs/momentum_conserving_fmm.md).

## 2. Public API contract

`jaccpot/__init__.py __all__` is the stable surface downstream code (e.g. the
ODISSEO coupling) depends on. It is frozen by
[`tests/unit/test_public_api_surface.py`](tests/unit/test_public_api_surface.py):

| Name | Kind |
|---|---|
| `FastMultipoleMethod` | the solver class (in `solver.py`) |
| `FMMPreset` | preset enum (FAST / BALANCED / ACCURATE / LARGE_N_GPU) |
| `FMMAdvancedConfig`, `FarFieldConfig`, `NearFieldConfig`, `RuntimePolicyConfig`, `TreeConfig` | advanced config dataclasses |
| `TraversalOverrides` | named traversal capacities, merged onto a preset's sizing |
| `GradConfig` | the supported interface for configuring the differentiable path (section 6) |
| `MemoryObjective` | memory-policy literal |
| `RealSHBasis` | the expansion basis (the only one since the 2026-10 cleanup) |
| `OdisseoFMMCoupler` | ODISSEO integration adapter |
| `BlockStepFMM` | momentum-conserving force for a block-step individual-timestep KDK (in `nornax_adapter.py`; see section 1) |
| `DistributedBlockStepFMM` | the same contract over a device mesh, on the distributed mutual lane (`mutual/distributed.py`) |
| `FusedForceProvider` | the fused single-GPU lane's per-step force as an object: eager `prepare`, traced `rebuild` / `accelerations` / `step` (in `runtime/fused_provider.py`; section 4, [`docs/fused_provider.md`](docs/fused_provider.md)) |
| `ForceParams` | `G` and the softening squared for the fused lane: Python floats static, arrays traced (in `runtime/fused_topology.py`) |
| `direct_sum_gravitational_acceleration` | autodiff-able **direct-sum** oracle (the differentiable FMM itself is `FastMultipoleMethod.differentiable_accelerations`; see section 6) |

`FastMultipoleMethod` in `solver.py` is the facade for the production force. The
runtime engine is `FMMEngine` in `_fmm_impl.py` -- an internal implementation
detail reached only through the facade. The two used to share the name, which this
document flagged as confusing; audit 2.4 renamed the engine, so a reference to
`FastMultipoleMethod` now unambiguously means the public class.

`BlockStepFMM` (and its mesh twin `DistributedBlockStepFMM`) is the only other
public solver class, and it is **not** an alternative spelling of the same thing --
it computes a different number (momentum-exact rather than target-centric) through a
separate path. `FusedForceProvider` is not a second solver either: it is the
facade's fused lane, opened up per step (section 4).

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
| `kernels/_m2l.py` | the M2L apply / accumulate seam (section 5) |
| `kernels/_l2l.py` | L2L propagation and the downward-sweep driver |
| `kernels/_downward_prep.py` | the downward sweep's input bundles |
| `kernels/_evaluate.py` | L2P, the targeted near field, the evaluation entry points |
| `kernels/_shared.py` | the basis tag, output bundles, the strict-refresh diagnostics gate |
| `kernels/core.py` | aggregator over the five above, so `jaccpot.runtime.kernels.core` stays a working import path |
| `kernels/__init__.py` | curated re-exports consumed by `distributed/`, `_large_n_pipeline`, tests |

**Orchestrator scaffolding** (sibling of `_fmm_impl`):

| Module | Role |
|---|---|
| `fmm_state.py` | resolved-config dataclasses + `_resolve_fmm_config`, tree-build artifacts, `FMMPreparedState` pytree, strict-refresh diag helpers |
| `fmm_presets.py` | `FMMPresetConfig` bundles + `get_preset_config` (FAST / LARGE_N_GPU); `FMMPreset` is re-exported from `jaccpot.config` (single definition) |

**Engine coordinator + method-cluster mixins** (section 4).

## 4. The engine: coordinator + mixins

`_fmm_impl.FMMEngine` coordinates (constructor, backend plumbing, cache
lifecycle) and inherits its behaviour from **9 method-cluster mixins**, each a
sibling `runtime/fmm_<cluster>.py` module. (`AutotuneMixin`, the M2L chunk
autotune, and its cache IO went in the 2026-10 cleanup, X4.) It used to be described
as *thin*, which the mixin split achieved for everything except the constructor.
`__init__` was 722 lines with 60 parameters against 316 for the rest of the class;
audit 2.1 staged it into 13 private resolvers, so the body is now **141 lines** of
named phases. It still takes 60 parameters -- that is the public surface, not the
body, and is item 2.4's territory.
Methods were moved verbatim during the god-class breakup; `self` is unchanged;
cross-cluster calls resolve through the MRO.

```python
class FMMEngine(
    PrepareMixin, EvaluateMixin, StrictRunMixin, SweepsMixin, OverridesMixin,
    PolicyMixin, DerivativesMixin, StrictCapProfileMixin, DiagnosticsMixin,
):
```

| Mixin (`fmm_*.py`) | Responsibility |
|---|---|
| `PrepareMixin` (`fmm_prepare`) | build the `FMMPreparedState`: tree upward pass, downward/far-pairs, nearfield setup |
| `EvaluateMixin` (`fmm_evaluate`) | evaluate a prepared state into accelerations/potential (L2P + nearfield) |
| `StrictRunMixin` (`fmm_strict_run`) | static-radix hot path: `refresh_prepared_state`, `strict_run_v2` (its scans and runners, on the `FusedForceProvider`), same-topology refresh, velocity-Verlet update |
| `StrictCapProfileMixin` (`fmm_strict_cap_profile`) | compiled-profile fingerprints for the strict lane's compile-reuse diagnostics, and the fused lane's `PROFILE_SET` gate (the on-disk cap profile went in X4) |
| `PolicyMixin` (`fmm_policy`) | adaptive execution-policy decisions |
| `OverridesMixin` (`fmm_overrides`) | resolve runtime execution knobs (farfield/nearfield mode, traversal caps; static sizing only since X4) |
| `SweepsMixin` (`fmm_sweeps`) | delta-sign / convention sweeps |
| `DerivativesMixin` (`fmm_derivatives`) | jerk / time-derivative towers |
| `DiagnosticsMixin` (`fmm_diagnostics`) | `get_runtime_diagnostics` + shape diagnostics |

### The fused strict lane and its force provider

`strict_run_v2` runs velocity Verlet as one device-resident `lax.scan`. Each step
drifts, computes the self-force at the new positions in two halves, adds the external
field and kicks. Since S2 (2026-10) the two halves are the methods of
`runtime/fused_provider.py`'s `FusedForceProvider`, and the scan, both carries
(`carry="state"`, the default, and `carry="particles"`), the tree-order carry, the
particle carry's start program, the eval seam (`strict_fused_prepared_eval_fn`), the
compile keys (`provider.key()`) and the segment re-plan (`provider.replanned`) all go
through it:

| half | provider method | what runs |
|---|---|---|
| refresh | `rebuild` (`_rebuild_like` for the state carry, which rebuilds from the state it carries) | `_refresh_large_n_same_topology`: tree from the shape template, P2M/M2M, flat walk, M2L, L2L |
| evaluate | `accelerations` | `evaluate_large_n_state`: near P2P, L2P, scatter |
| verdict | `capacity_ok` / `FusedTopology.overflow_bits` | every fixed capacity, the cell-leaf tree's included |

`runtime/fused_topology.py` holds what the halves hand on: `FusedTopology` (the
refreshed prepared state plus the walk's and the tree's capacity reports, moved off
the engine's side channels so no tracer outlives its trace), the `OVERFLOW_*` bits, and
`ForceParams` / `StepParams`, whose Python floats are compile-time literals (the
default: every program as before) and whose arrays are traced (the opt-in:
`strict_run_v2(dt=array, force_params=..., external_params=...)`, one compile for every
value). `runtime/strict_carry.py` holds the shape template the particle carry rebuilds
from. The refactor (stages C1-C3) kept every program text identical
(`bench/provider_hlo_identity.py`, section 9); the S2 fixes on top changed only what they
meant to ([`docs/production_2026-10.md`](docs/production_2026-10.md), S2). User guide:
[`docs/fused_provider.md`](docs/fused_provider.md).

The fused multi-GPU lane (`distributed/fused.py`) still has its own force step; plan
session S11 moves it onto the provider.

## 5. Kernel-family map (`runtime/kernels/`)

There is one basis (real solid harmonics), so the kernels carry no basis
discriminator. What is static is `order`, `m2l_impl`, `chunk_size` and
`total_nodes`: they are `static_argname`s, so XLA specialises each `jax.jit` per
combination.

**M2L apply** (`kernels/_m2l.py`):

- `_apply_real_m2l(src_mult, deltas, *, order, m2l_impl)` — the real rot-scale
  kernel (`_m2l_real_batch_kernel`), or the fused Pallas kernel
  `_m2l_real_batch_kernel_fused_pallas` when `_real_m2l_pallas_active()` (section 7).

**M2L accumulate** (through `_apply_real_m2l`):

- `_accumulate_m2l_fullbatch` — one full interaction batch → `segment_sum`
- `_accumulate_m2l_chunked_scan` — chunked `lax.scan` reduction (bounded memory)

On an Ampere+ GPU the downward's accumulate step
(`_solidfmm_downward_accumulate_from_multipoles`, `kernels/_downward_prep.py`) runs
neither: it routes the flat M2L through the CSR Pallas lane kernel
(`pallas/m2l_real_csr_lanes.py`, a `custom_vjp`; gate `_m2l_csr_pallas_active`,
`JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0` restores the two above).

**L2L / downward** (`kernels/_l2l.py`, `kernels/_downward_prep.py`):
`_propagate_solidfmm_locals_by_level`, `_propagate_real_locals_to_children`, the
solidfmm downward-sweep builders ("solidfmm" names the spherical-harmonic family).

**Evaluation** (`kernels/_evaluate.py`): `_evaluate_local_expansions_for_particles`
(L2P), near-field targeting + scatter. The public
`compute_gravitational_{acceleration,potential}` convenience wrappers live in
`_fmm_impl.py` (not in `kernels/`) because they construct/drive the engine.

The grouped / class-major accumulators, the complex and Cartesian kernels and the
`basis_mode` / `rotation` seam that chose between bases were removed in the 2026-10
cleanup (`docs/cleanup_2026-10.md`: X3, K, C, Z-b).

## 6. Config resolution, presets, and the autodiff export

- **Presets** (`config.FMMPreset`): FAST and LARGE_N_GPU resolve through
  first-class bundles in `fmm_presets.get_preset_config`; BALANCED and ACCURATE
  resolve through advanced-config defaults in `solver._default_advanced_for_preset`
  (they map to `expanse_preset=None` and never reach `get_preset_config`). GALAXY
  takes both: the LARGE_N_GPU bundle (`expanse_preset="large_n_gpu"`) under advanced
  defaults built from `production.GalaxySettings` (`solver._galaxy_advanced`), which
  also sets the preset's default theta, softening kernel and precision.
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
  concretely, before `jax.grad`. Two kinds of state, both on the real basis:
  - a radix `FMMPreparedState` (`_differentiable_accelerations_radix`). Its near
    field is the bucketed pure-JAX edge list below `GradConfig.nearfield_lane`'s
    crossover (N = 100000) and the leaf-major radix fast lane at or above it — the
    same edge set and force, a different traversal. At N≤4096 the two are within
    ±20% with no consistent winner; at **N=200000 the bucketed reverse OOMs (30 GB
    peak) and the fast lane completes in 6.8 GB** (2026-07);
  - a `LargeNPreparedState` (`preset="large_n_gpu"` on a GPU), through
    `runtime/_large_n_grad.py`. It needs `retain_far_pairs_for_grad=True` (a
    constructor option, not a `prepare_state` one), and it has one near field.

  On an Ampere+ GPU (since 2026-09, `docs/fast_gradients_2026-09.md`) the gradient
  runs the same Pallas kernels as the fused forward, each with a reverse Pallas
  kernel: the leaf P2M, the M2M/L2L cascades, the M2L over the far-pair CSR, and the
  near field's CSR row-chunk lane. Two forward kernels have no reverse, so the
  gradient path bypasses them for slower layouts: the near field runs the CSR lane's
  table layout, not the production direct layout (`nearfield/_fast_lane.py:1829`),
  and the L2P runs the leaf-major padded XLA L2P with its analytic VJP, not the
  particle-major Pallas L2P. The 2026-10-13 numbers (`docs/production_2026-10.md`):
  jitted gradient 31.9 ms at 2e5 and 101.6 ms at 1e6, out of memory at 8e6 on one
  40 GB card. (The 2026-07 figure this bullet used to quote, 1M forward+backward
  69 s at 11 GB, predates the Pallas reverses.)

  **What it does not cover.** Positions and masses only: `G` and the softening are
  Python floats (`_large_n_grad.py`'s `-float(engine.G)`, the near field's
  `float(softening)`), so there is no d/dG or d/dε. (The fused forward takes them as
  traced `ForceParams` since S2, and `jax.grad` through `FusedForceProvider.accelerations`
  in them equals central differences on the pure-JAX near field in float64 (CPU). Not
  verified on the Pallas lane, whose direct near-field kernel and particle-major L2P
  have no reverse rule (below). See [`docs/fused_provider.md`](docs/fused_provider.md).) Reverse mode only: a
  `custom_vjp` refuses `jax.jvp`, so forward-mode adjoints fail. One force only:
  `strict_run_v2` evaluates the forward-only `evaluate_large_n_state`, and the
  fused multi-GPU lane's cross-domain M2L and near-field reverses raise
  (`pallas/m2l_real_csr_lanes.py:835-840`, `pallas/nearfield_leafpair_csr.py:2174-2183`),
  so neither `strict_run_v2` nor
  `FusedRollout` can be differentiated.
- **Configuring the grad path.** `GradConfig` (`jaccpot/config.py`) is the
  supported interface; `runtime/grad_options.py` resolves it under
  `explicit field > JACCPOT_* env var > measured default`, so the environment
  variables remain a working fallback. `nearfield_lane` defaults to
  `"auto"` and switches to the fast lane at N >= 100000 — the bucketed reverse
  OOMs there, so the crossover is applied rather than documented. Four gates
  (fused-Pallas M2L, the Pallas cascades, the two analytic VJP rules) are read at
  trace time deep in the kernels with no argument channel; `grad_option_overrides`
  installs them as context-locals for the duration of one call. On the default
  Ampere path several fields are inert: `fused_m2l_pallas` (the CSR M2L replaces the
  lanes it serves), the five `reverse_*` fields (they tune the prepacked rectangle
  reverse, which runs only with `JACCPOT_NEARFIELD_LEAFPAIR_CSR=0` or in interpret
  mode, and never on a `LargeNPreparedState`), and `nearfield_lane` on a
  `LargeNPreparedState`.
- **Where the reverse rules live.** `nearfield/grad.py` holds both analytic
  near-field reverses of the pure-JAX and rectangle lanes — `_pair_accel_cvjp`
  (bucketed, rematerialized pair terms) and `_leafpair_accel_analytic_vjp`
  (leaf-major, O(N) memory) — plus the occupancy-tier builder and its cache. The
  bucketed rule's whole `custom_vjp` lives there; the leaf-major rule's wrapper,
  `_radix_fast_lane_prepacked_accel_cvjp`, stays next to the kernel it wraps in
  `nearfield/_fast_lane.py`. Nothing in `grad.py` imports `near_field`, which is
  what keeps the modules acyclic. The Pallas reverses sit next to their forwards
  in `pallas/` (`p2m_real_leaf.py`, `cascade_real_level.py`,
  `m2l_real_csr_lanes.py`, `nearfield_leafpair_csr.py`).
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
- `direct_sum_gravitational_acceleration` remains the deliberately differentiable
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
cannot express; fp64 in the since-removed complex kernel) -- so the shim raises rather than
letting a non-Triton request fall through to a default. The near-field kernels
already passed `pallas.triton.CompilerParams` and needed no change.

- Fused M2L Pallas is gated by `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` **and**
  the Ampere+ hardware support check (`_real_m2l_pallas_active`, evaluated at trace
  time, gating on the fused kernel's own sm_80 support function).
- The fused M2L kernel is **differentiable** via a module-level `custom_vjp` wrapper
  (`m2l_real_fused_pallas_cvjp`): Pallas forward + reverse. `_apply_real_m2l` routes
  through the wrapper, so the same flag enables the fused M2L on both the forward and
  the `differentiable_accelerations` grad path (PR-2). `pallas_call` itself still has
  no autodiff rule — the wrapper supplies it. The reverse is a **fully-fused analytic
  VJP kernel** (`m2l_real_fused_vjp_pallas`, default; `JACCPOT_FUSED_M2L_VJP=0` falls
  back to autodiff of the twin), so both forward and reverse run as single Pallas
  launches.
- The fused M2L serves the chunked / full-batch M2L lanes (`_apply_real_m2l`). On an
  Ampere+ GPU the single-GPU downward replaces those lanes with the CSR lane kernel
  (`m2l_real_csr_lanes.py`, `custom_vjp` with a by-source reverse; default since
  2026-09-10, `_m2l_csr_pallas_active`), so there the flag acts only with
  `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0`. The multi-GPU mesh lane
  (`distributed/fmm.py`) calls `_apply_real_m2l` directly.
- Nearfield Pallas is resolved from `pallas_nearfield_fused_supported()` into the
  engine's `use_pallas`. Three differentiable near-field lanes exist and they are
  **not** interchangeable:
  - `nearfield_leafpair_csr_pallas_cvjp` (`pallas/nearfield_leafpair_csr.py`) is the
    CSR row-chunk lane's rule: Pallas forward in the table layout + a Pallas reverse
    over the same chunk table. It is what the grad path takes on an Ampere+ GPU, for a
    `LargeNPreparedState` and for a radix state's leaf-major lane alike (since
    2026-09, `docs/fast_gradients_2026-09.md`). The forward's default direct layout
    has no reverse, so `differentiable=True` forces the table layout
    (`nearfield/_fast_lane.py:1829`).
  - `_radix_fast_lane_prepacked_accel_cvjp` (`nearfield/_fast_lane.py`) is the
    prepacked rectangle's rule: Pallas forward + an analytic O(N) leaf-pair reverse,
    tuned by `GradConfig`'s `reverse_*` fields. It was the production rule until the
    CSR lane; it now runs only with `JACCPOT_NEARFIELD_LEAFPAIR_CSR=0` or in interpret
    mode. Its leaf-major payload is what `runtime/_nearfield_fastlane.py` transposes
    on the host from the radix state's CSR neighbour list. With `use_pallas=False`
    the leaf-major traversal runs the tiled pure-JAX prepacked kernel under ordinary
    autodiff — same force, but no O(N) reverse.
  - `nearfield_fused_leaf_pallas_cvjp` / `nearfield_leafpair_pallas_cvjp`
    (`pallas/nearfield_fused_leaf.py`, PR-2) are **unit-level VJP oracles**. Their
    reverse is `jax.vjp` of a dense twin that materialises a `(leaves, W_t, K, 3)`
    tensor — fine at test scale, ~50 TB at the fiducial large-N config. Keep them for
    `tests/unit/test_custom_vjp_parity.py`; do not add a second grad-path caller.
- **fp32 matmul precision is pinned in the operators.** XLA lowers fp32 matmuls on
  Ampere to **TF32** (~10-bit mantissa) by default. Measured against a float64
  reference (real basis, max rel err), that capped the M2L at 5.7e-04 at order 4,
  5.7e-04 at order 6, 5.6e-04 at order 8 — **~6e-04 regardless of expansion
  order** — against 1.5e-06 / 2.4e-06 / 1.8e-06 under highest precision (~300x
  better; the complex basis, removed in the 2026-10 cleanup, behaved the same:
  3.7e-04 -> 5.6e-08 at order 2). The expansion operators therefore run under
  `operators/_precision.highest_matmul_precision`, and the real L2P sets
  `lax.Precision.HIGHEST` (`operators/real_p2m_l2p.py`); `CLAUDE.md` forbids
  dropping either. The floor is also why
  `tests/integration/test_fmm.py::test_m2l_ignores_padded_compact_far_pairs`
  compares against a float64 reference with a 2e-03 bound instead of asserting
  padded == exact — the latter passed on JAX 0.9.x only because both fp32 paths
  were identically wrong.
- **Org rule for GPU runs:** select a free GPU with autocvd *before* `import jax`:
  `from autocvd import autocvd; autocvd(num_gpus=1, least_used=True)`.

## 8. Dependency DAG and the import-cycle rule

```
fmm_constants -> fmm_caches -> kernels -> {_interaction_cache, _large_n_pipeline}
  -> fmm_state -> _fmm_impl (engine) -> runtime/fmm -> solver -> __init__
```

(The `_octree_*` modules left this chain with the octree execution backend in the
2026-10 cleanup; `docs/cleanup_2026-10.md`, X2.)

`distributed/` depends on `kernels/` and on runtime modules below the engine
(`_adaptive_policy`, `_interaction_cache`, `_mac_geometry`, `_large_n_pipeline`, the
`capacity_*` modules), never on the engine itself.

`mutual/` sits beside the sweeps and depends on `operators/` at module scope and
on `pallas/` from **six function-local sites** (`farfield.py` ×4,
`nearfield.py` ×2) — the same deferred-heavy-import pattern
`nearfield/near_field.py` uses, and deliberate for the same reason. Two further
imports reach the `jaccpot` facade function-locally to break a cycle
(`mutual/topology.py` and `nornax_adapter.py`, both for `FastMultipoleMethod`).
Nothing in `mutual/` imports `runtime/`, and nothing in `runtime/` imports
`mutual/`; `pallas/nearfield_mutual.py` depends only on `pallas/_compat`.

**Cycle rule:** `runtime/fmm/__init__.py` must not eagerly import the engine
class in a way that reforms
`_interaction_cache -> fmm -> engine -> prepare -> _interaction_cache`.
Consumers import the class explicitly rather than through a package-init that
pulls the engine in. Keep `fmm_constants`/`fmm_caches`/`fmm_state`/`fmm_*` mixins
at the `runtime/` level until the engine is fully dissolved into `fmm/engine.py`.

## 9. Validation harness

Where the tests live:

| directory | |
|---|---|
| `tests/unit/` | one function or module against its docstring; runtime type checks on in CI |
| `tests/integration/` | end-to-end paths; `strict/` holds the fused strict lane (`strict_run_v2`, the refresh, the fused eval fn) |
| `tests/characterization/` | the goldens (below) |
| `tests/distributed/` | multi-device; every file skips below two devices, so CI runs it on forced host devices |
| `tests/perf/` | performance assertions, opt-in (`addopts` ignores it) |
| `bench/tests/` | the tests of the `bench/` scripts (gate checks, annotation tools); outside the partitioned suite, run in CI's lint job |

Which CI job runs which of them is defined once, in `.github/scripts/test_shards.py`
(`CLAUDE.md`, "Layout").

- **Gradient golden** —
  [`tests/characterization/test_fmm_grad_golden.py`](tests/characterization/test_fmm_grad_golden.py)
  snapshots `jax.grad` of a fixed-cotangent loss on `differentiable_accelerations`
  w.r.t. positions and masses, over 6 cases, with the same two gates as the forward
  oracle (inertness at `rtol=1e-12`; physics anchor against `jax.grad` of the direct
  sum, per-order bounds). The forward oracle cannot see a reverse-only regression:
  measured, scaling the analytic real L2P reverse rule by `1+1e-6` leaves the forward
  golden green and turns this one red. `leaf_size=4` is load-bearing -- at leaf >= 8
  these systems accept **zero** M2L pairs, so the far-field reverse would not be
  traced at all; a vacuity gate asserts the pair count per case.
- **Golden characterization oracle** —
  [`tests/characterization/test_fmm_golden.py`](tests/characterization/test_fmm_golden.py)
  drives the FMM over a grid of (distribution, N, basis, order) -- 13 cases, all
  `preset="accurate"`, **accelerations only** -- and applies two gates: (1) an **inertness** gate — outputs match the committed `.npz` goldens
  under `tests/characterization/golden/` to float64 round-off (`atol=0, rtol≈1e-12`),
  and (2) a **physics** gate — each output is anchored to a direct O(N²) sum to a
  loose relative-L2 bound, so a regenerated golden can never silently encode a wrong
  answer. Refactors must keep it exact-green — any drift is a wiring bug, not a
  numerical one. Regenerate goldens intentionally with `JACCPOT_REGEN_GOLDEN=1`.
- **Public-API guard** —
  [`tests/unit/test_public_api_surface.py`](tests/unit/test_public_api_surface.py)
  freezes the 17 `__all__` names + `FMMPreset` members. Red = the refactor leaked
  into the public contract.
- **Runtime typecheck** — set `JACCPOT_RUNTIME_TYPECHECK=1` to enable beartype
  runtime checks over the suite.
- **Program identity of the fused scan** —
  [`bench/provider_hlo_identity.py`](bench/provider_hlo_identity.py) records, per
  variant of `strict_run_v2` (state / particle / tree-order carry, `dehnen_error`,
  external field, history, callback, G != 1, chained calls, the segment retry, the eval
  seam; 36 on CPU at N = 512, 18 on a GPU at 2e5), the lowered text of every program
  that runs and the sha256 of every array; `compare` fails on one character or one bit.
  Built for the provider refactor (S2), it is the gate for any change that must keep
  the fused lane's programs as they are. The GPU pins of the same lane are in
  `bench/dce_pins.py`.
- **GPU/Pallas parity (A100, manual/nightly):** run the fused-Pallas M2L parity
  tests + golden with `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` under
  `JAX_ENABLE_X64=1` to confirm the Pallas paths match the pure-JAX reference.
  See `CLAUDE.md` for the worker-capped invocation — the default `-n auto` is
  unusable on one card.

### Known GPU-only failures (A100 sm_80, jax 0.10.2)

These fail on the A100 and pass on CPU, so a hit here is not necessarily yours — but check
it against this table before assuming that, because the list has been stale in **both**
directions before. Re-measured 2026-08-26, three runs per configuration, plus two whole-suite
runs at `-n 6 --xla_python_client_mem_fraction=.12`.

`XLA_FLAGS="--xla_gpu_deterministic_ops=true"` is the fast triage lever: green under it means
you were looking at reduction nondeterminism (§10), still red means a real discrepancy.
`bench/gpu_gate.py` sets it by default, so the "deterministic ops" column is the one that
decides whether the gate goes red.

| test | fails (default XLA) | fails (deterministic ops) | what it is |
|---|---|---|---|
| `test_constructor_state_matches_the_committed_golden` | 3/3 | **3/3** | platform |
| `test_every_matrix_case_resolves_to_a_distinct_state` | 3/3 | **3/3** | platform |
| `test_the_far_field_term_is_load_bearing` | 3/3 | **3/3** | resource, see below |
| `test_boundary_kick_accepts_supplied_level_weights` | 3/3 | 0/3 | assertion strength |
| `test_compiled_dispatch_is_bit_identical[None]` | 1/3 | 0/3 | reduction order |
| `test_compiled_dispatch_is_bit_identical[32]` | 2/3 | 0/3 | reduction order |

**The two `constructor_state` rows are genuine platform differences.** The constructor
golden resolves `use_pallas: False -> True` because it was generated on CPU and is being read
on an A100. (The two `potential_stays_on_fast_lane` rows this paragraph used to call platform
differences were not: see the Pallas unit files below.)

**`test_the_far_field_term_is_load_bearing` is not a numerical failure at all.** It is
`RESOURCE_EXHAUSTED` under this gate's own caps — `-n 6` at a 12% per-worker share is
1.82 GiB in use of 4.74 GiB on a 40 GB card, and the case wants ~769 MiB more. It is listed
so the gate is not red for everyone, but the honest fix is the caps, not the entry: raise
`XLA_PYTHON_CLIENT_MEM_FRACTION` or drop a worker and it clears.

**`test_boundary_kick_accepts_supplied_level_weights` is an assertion-strength bug, not a
platform one**, and is the one row here that should eventually disappear rather than be
tolerated. It is `jnp.allclose(supplied, derived, rtol=0, atol=0)` — a dispatch-identity
check, "handing in a weight row equals deriving it", whose intent has nothing to do with
numerics. §10's same-input control disqualifies it: calling `boundary_kick` twice with
identical arguments differs by the same amount and in the same fraction of elements as the
two things being compared (2..8 ULP against 2..6 ULP, ~670-770 of 1536 elements either way,
over six draws), so no tolerance makes the comparison informative about dispatch. The fix is
to split the claim — assert the weight rows exactly, the kicks to a bound — not to loosen it.

#### The Pallas unit files on an A100, jax 0.11.2 (fixed 2026-10-15)

Every `tests/unit` file that mentions Pallas (48 files, 733 tests) was run on one A100 under
`JACCPOT_RUNTIME_TYPECHECK=1`, `-n 3`, default XLA flags: 11 failed on main 63587f2. After
the fixes below none does: the rerun (`-n 2`) passed 741 of 742, the one left being the
reference fixed last (`3802a3c`, then 3/3 alone); every touched test passed three more runs,
and the same files are green on CPU. Five causes, and only the first is a library defect in
numbers:

* **A kernel bug** -- `test_fd_vs_ad_along_a_transverse_direction_at_rho_zero[real]`
  (1.62e-06 against its 1e-06 bound; CPU 1.6e-10). The Pallas reverses that replace the
  pure-JAX M2M/L2L cascades and the M2L on an Ampere card by default differentiated the
  polar route only, without the analytic transverse limit at `rho == 0` that
  `operators/_transverse_degeneracy_jvp` gives the operators they replace; bisected on the
  A100 (2.98e-06 with only the CSR M2L lane on, 2.65e-10 with both off) and reproduced on
  CPU in interpret mode. `9112733` adds the limit to both reverses (FD-vs-AD 2.65e-10 on the
  A100 after, default lanes); `test_cascade_real_level_vjp.py` and
  `test_m2l_real_csr_lanes_vjp.py` now check them on on-axis edges, where the old kernels
  missed the centre half by 0.30-2.1 relative. Off the band the reverses are unchanged: pins
  S4 and S4b (and the forward S1-S3) are bitwise against main.
* **A wrong annotation** -- both `test_softening_floor.py` GPU tests: the facade typed
  `strict_fused_prepared_eval_fn`'s state as `FMMPreparedState`, and the fused lane returns a
  `LargeNPreparedState`, so the runtime type check raised. `968f1fb`.
* **A reference behind it** -- with the type check passing,
  `test_a_cluster_smaller_than_epsilon_is_near_field` still failed (cluster median 0.2825
  against 1e-3, also without the type check on main): it compares the library's default kernel,
  ferrers3 since `5a5e0b0`, with a Plummer direct sum. `3802a3c` pins Plummer, as that commit
  did for the other Plummer-reference tests; the numbers are then the 2026-10-07 ones (cluster
  1.28e-4 / 4.91e-4 with the floor, 0.153 / 4.0 without).
* **A lane the test did not expect** (`3d879bc`) -- the GPU's default Pallas walk returns
  CSR row offsets in the far list's `targets` (`test_flat_walk_with_com_geometry...`, x2);
  its default Pallas COM-radii kernel rounds `num_levels` up to chunks of eight
  (`test_level_passes_equal_the_ancestor_table`, x2, which then also needed one ulp on the
  leaves: XLA:GPU fuses the two spellings of the sum differently, each bitwise repeatable);
  its default CSR near-field lane leaves no pairs payload to reject
  (`test_the_pairs_layout_is_rejected...`); and the engine resolves `use_pallas=True`, the
  opposite of what `test_csr_lane_enabled_without_pallas...` pins. Each test now names the
  lane it is about.
* **A premise read from the wrong field** (`3d879bc`) -- the two `potential_stays_on_fast_lane`
  rows above. The A100 builds the static target-block layout: the flat
  `nearfield_target_block_source_leaf_ids` is `(0, 4)` and the blocks are in the padded field,
  `(8, 1, 4)` with 32 valid entries. The same module's value test passed vacuously there (its
  "generic path" reference was the fused lane, since `use_pallas` resolves True) and, made
  real, needed a per-particle bound: the generic lane differs from ITSELF in 159 of 768
  components between two identical calls (up to 3.1e-05), and from the fused lane by at most
  1.9e-06 of `|a|`.

The ten failures `fused_memory_2026-10.md` counted on 2026-10-05 were not the same set:
`test_direct_equals_the_table_kernel_in_particle_order` (x2) and
`test_walk_backend_flag_parsing` pass on today's main, and the softening-floor, pairs-layout
and CSR-consistency rows are new since.

#### Resolved since the previous revision of this table

Entries that were fixed are recorded here so they are not re-added from a stale copy of this
section, and so the next person to find this list wrong knows it happens. The first three
were measured on 2026-08-12 and fixed two days later:

| test | was | fixed by |
|---|---|---|
| `test_fmm_grad_golden[clu_real_n128_p4]` | 3/3 default, 3/3 deterministic | `aacd3cf` — gate the gradient golden per particle, not per array element |
| `test_real_basis_tracks_complex_basis[nearfield-only-f32]` | 3/3 default | `86163f1` — make two fp32 path-equivalence tests actually test something |
| `test_solidfmm_chunked_m2l_matches_fullbatch` | 1/3 default | `86163f1`, as above |
| `test_the_state_under_test_actually_carries_target_blocks` | 3/3 default, 3/3 deterministic | `3d879bc` -- read the padded target-block field too (above) |
| `test_potential_does_not_delegate_to_the_generic_path` | 3/3 default, 3/3 deterministic | `3d879bc`, as above |

Those three passed 3/3 in isolation and did not fail in either whole-suite run. The
`test_fmm_grad_golden` entry is the one worth noting: it was described here as "a genuine
platform difference, not noise" and survived the determinism flag, which made it look
permanent. It was a per-element tolerance on a 384-element array, and `aacd3cf` replaced it
with a per-particle gate.

## 10. Kernel-consolidation invariant

When two kernel families are merged behind a discriminator, the merge is
numerics-preserving **only** if every discriminator (today `m2l_impl`, `order`,
`chunk_size`, `total_nodes`) is a `static_argname`, so XLA specialises the merged
`jax.jit` per static combination. That makes it source-level dedup, never a runtime
branch inside a compiled kernel. What that argument does **not** cover is the call
sites: a discriminator with a default is silently satisfied by a caller that forgets
to forward it, and the specialisation then happily compiles the wrong variant. That
is how the grouped fullbatch accumulator once applied the complex kernel to real
blocks (3.8e-01 relative error, no exception), through a `basis_mode="complex"`
default; both went in the 2026-10 cleanup. Audit the callers, not only the kernels.
Any consolidation PR must show:

1. merged-vs-old **bit-identical** output on a fixed input grid (rtol=0),
2. the golden oracle exact-green,
3. the full suite with no new failures vs the frozen baseline,
4. an A100 Pallas-on vs pure-JAX parity run when the change touches Pallas.

**Items 1 and 2 are unsatisfiable on GPU without `--xla_gpu_deterministic_ops=true`.**
Measured on an A100 (`docs/refactor_audit_2026-08.md` B.3): under default XLA the *same*
tree on the *same* card differs run-to-run by up to `2.1e-11`, which is **larger** than
the difference a consolidation PR is trying to bound — so an exact-equality check run
without the flag reports a difference that carries no information about the change. With
the flag the same comparison is exactly equal, `rtol=0`, on every case. Set it on both
sides, and say in the PR that you did.

The corollary for item 3: a GPU run of the suite has four pre-existing failures that go
green under the flag, so the "frozen baseline" for item 3 must be a baseline taken the
same way. Run the *old* code on the *same* card in the same session — a CPU baseline is
not a baseline for a GPU comparison.

**Run the same-tree-twice control first**, before reading any cross-tree number. Compare
a tree against *itself* on the same card: if A-vs-A differs as much as A-vs-B, the harness
has not yet earned the right to report A-vs-B as a difference. That control is what
identified the problem above — the same-tree difference (`4.4e-15` of typical `|a|`, ~65%
of entries affected) turned out to exceed the two-tree difference (`1.8e-15`).

Deterministic ops serialise the scatters and are measurably slower, so use the flag for
correctness comparisons only — **never for the timing runs** in `NUMERICS_AND_JAX.md` §4.

**When item 4 actually runs.** There is no GPU runner in CI — every job is
`ubuntu-latest` — so item 4 is a *manual* step, not an enforced merge gate, and it has
been discharged in arrears before (Tier 1 merged ten PRs owing it; the run is recorded in
`docs/tier1_gpu_validation_report.html`). Treat it as a release-level obligation for any
Pallas-touching batch and say in the PR whether it was run. A nightly GPU leg is tracked
as audit item F11.

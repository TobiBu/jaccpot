# Refactor and code-hygiene audit — 2026-08

Scope: `jaccpot/` excluding `jaccpot/experimental/`, plus `tests/`, repository root, and
`pyproject.toml`. Baseline commit `32857c4` (`main`, clean tree). No production code was
modified in producing this document, and none has been modified since.

**Status — updated after executing Tier 0.1 and 0.3.** Two items of section F have been
carried out on branch `test/close-rotation-and-grad-golden-gaps`, both test-only:

| Item | Commit | Result |
|---|---|---|
| 0.3 rotation blocks vs. an independent reference | `e5d8e41` | **Done, differently than specified** — see D.5. The Wigner reference turned out to be unusable (F39), so the tests assert physics identities instead. The production builders are **correct**. |
| 0.1 gradient golden | `e1f1455` | **Done** — 6 cases + a vacuity guard, `tests/characterization/golden_grad/`. See D.1. |
| G.10 fix attempt → 0.4 | `4ae5479` | **Bug not fixed; tracked instead.** The fix is a scheme change, not a guard change (three variants measured). Landed: the false comment retracted, plus 3 passing tests and 3 strict-xfail. Awaiting your call — see G.10. |
| 0.10 coverage omit | `ff4c911` | **Done.** F04 closed; `m2l_real_fused.py` measured at 92%. |
| 0.11 (part) + 0.6 | `c2874b6`, `20ba5d8` | **Done.** F05, F06, F24, D.9 closed. |
| 0.12 + 0.13 + 0.21 | `e2959eb` | **Done.** A.8, A.2, D.12 closed in the guides. |
| 0.14 (safe subset) | `934c252` | **Partly done** — F14, F15, F17, F18 closed; **F16 was wrong**, see below. |
| 0.17 repo-hygiene moves | `6839a6c` | **Done.** A.7 closed — `benchmarks/` and `results/` gone, dated handoff archived under `docs/`. |
| 0.7 decoupled Pallas parity | `26cef45` | **Done.** F25 closed, asserted bit-for-bit. |
| 0.8 `JACCPOT_FUSED_M2L_VJP=0` | `568deca` | **Done.** F26 closed; both halves (gate read per call, reverse matches the twin). |
| G.10 derivation **+ fix** | `8adc25f`, then PR #70 and #71 | **FIXED and merged** into `fix/real-rotation-transverse-gradient` (PR #72 → `main`). Went further than the handoff: both bases, the four precomputed-block lanes, both fused Pallas lanes, and an analytic/polar crossover at `rho_sq <= eps·r_sq` rather than a hard `rho == 0` switch. Both tracking xfails flipped and their markers removed; force-level FD-vs-AD 1.9e-03 → 2.7e-10 (real) and 1.8e-05 → 2.7e-10 (complex); both goldens byte-unmoved. See `docs/rotation_degeneracy_derivative.md` §5–§7. Still open: four GPU-gated tests have never executed, and fp32 does not cover the fused lanes — `docs/handoff_g10_gpu_validation.md`. |
| 0.9 staticness contracts | `92c8e73` | **Done.** D.7's two `near_field` contracts asserted. |
| 0.15 `TYPE_CHECKING` | `ad7b00c` | **Done** — but F21's stated benefit was wrong; see the corrected row. |
| 0.19 rename the misnamed large-N test | `d941805` | **Done.** F28 closed. |
| 0.18 test-layout moves | `a54f29e` | **Done.** A.6 closed — 20 files into the tiers, new `tests/distributed/`, 17 `slow_tests.txt` node ids and 32 path references retargeted from git's rename mapping. Collected counts identical (911/949; 779 not-slow), so no `slow` marker was lost. |
| 0.20 public surface | `4b2a66a` | **Done** (targeted half). All 14 public names documented; 43 dataclass fields covered. Latent pydoclint violations 2840 → 2778. The ~58 numerics functions are deferred to Tier 1.10 by your decision. |
| ~~**Tier 0 COMPLETE**~~ | — | **This line was wrong when written.** It came from the running status block, not from checking the per-item rows — and four items were still open: 0.2, 0.5, 0.11's F07 half, and 0.16. Found by re-deriving completeness from the code rather than from this table. |
| 0.5 + 0.11 (F07) | `8bd24a6` | **Done, differently than specified.** Not a Table 3 transcription — an absolute anchor from the derivative recurrences. See D.8. |
| 0.2 mode/output golden axes | `ef7ce15` | **Done.** 4 new cases in a separate golden dir; potentials goldened for the first time. Surfaced **G.11**. |
| 0.16 `__future__` / `__all__` | `23ccc9d` | **Done** (`__future__` complete at 80/80; `__all__` scoped to the public modules — the rest is the F16 star-export seam). |
| **Tier 0 complete — verified** | — | Re-checked empirically, not from this table. |
| G.4 + G.8 (Tier 2) — delete the Wigner family and the `sympy` dependence | `ab58c3d` | **Done, signed off by you.** 11 functions, 281 lines, 6 `__all__` names, plus the now-unused `import math`. Both goldens byte-unmoved; suite unchanged at 839/57. Closes F32, F39, and moots F19. |
| G.11 `pair_grouped` wrong class rotation | PR #76, #77 | **FIXED and merged.** See G.11. Also added 2 `pair_grouped` mode goldens and 11 tests, which is why the suite baseline moved — see the note below. |
| **Tier 1 — all ten items executed** | PRs #79-#88 | One PR per item, all green. Per-item outcomes in the Tier 1 table of section F, which now records what each actually did rather than what was planned. Four findings came out of the execution and are folded in below: **F40** (new), the F16 correction, the G.1 re-location, and the F23 three-part split. |

Full suite after Tier 0: **839 passed, 57 skipped**, exit 0. `black`, `isort`, `pydoclint`
clean.

**The suite baseline has moved twice since, and the second move matters for reading
this document.** After Tier 0 it was 839/57. After G.11 (PRs #76, #77) it is **926
passed / 67 skipped**, and that is the number every Tier 1 PR was verified against.
An earlier draft of the Tier 1 brief quoted 915/67, which was already stale when it
was written — the 11 tests G.11 added had landed in between. Collected counts are
**993/1031 (38 deselected), 898 under `-m "not slow"`, 38 under `-m experimental`**;
those are the numbers to compare against when checking that a move did not silently
un-mark a `slow` entry, and they replace A.6's 911/949/779.

Everything changed by that execution, so you can see exactly what is new:

- **New:** F39 (undeclared `sympy`), D.12 (a gradient test at the default leaf size covers no
  far field), G.8 (the `sympy` decision), G.9 (cross-platform bit-stability of the new golden),
  Tier 0.21, headline item 6.
- **Closed:** F03, F29, D.1, D.5, Tier 0.1, Tier 0.3, headline item 1, and two rows of D.4's
  equivalence table.
- **Corrected:** F32 (6 public `*_wigner` names, not 8 — the original count was wrong), the
  `real_harmonics.py` row of D.11's prerequisite table, and two stale line citations in G.7.
- **Reversed:** G.4. The original pass recommended keeping the Wigner family; executing 0.3
  removed two of the three premises for that. **You then decided to delete it** (`ab58c3d`) on
  the further ground that the Wigner route is the slow one and the Dehnen closed forms are what
  we want to use regardless — so G.4 and G.8 are settled, and F19, F32 and F39 are closed.

Nothing else was touched.

**How to read it.** Every claim below is either (a) a line-number citation you can open, or
(b) a measurement I made in this session with the command stated. Where I could not settle a
question, the row says so and says what evidence would settle it. I have flagged three places
where I initially agreed with a stated concern and then found the evidence contradicted it —
those are called out explicitly rather than quietly dropped, because the *reason* they were
wrong matters for the work plan.

**Measurement environment.** `jaccpot-verify` conda env, CPU backend, JAX 0.10.2,
pytest 9.1.1, 10 cores. Full suite: `JAX_ENABLE_X64=1 pytest -q --cov=jaccpot --cov-branch`
— green, 75% line coverage (14584 statements, 3149 missed, 3606 branches, 820 partial).
`black --check .` and `isort --check-only .` clean. `pydoclint --config pyproject.toml
jaccpot/` reports **0 violations**, as stated.

---

## 0. Headline

Six things dominate everything else in this audit:

1. **There was no gradient golden — now there is (`e1f1455`).** `tests/characterization/`
   contained one test over 13 forward-only acceleration cases. `NUMERICS_AND_JAX.md` §3 says,
   in as many words, that a refactor of thinly-guarded code must extend characterization first
   *"including a gradient golden, not only a forward one"*. The differentiability of this
   library is its reason to exist, and nothing in the characterization tier protected it.

   The gap is now measured, not argued: scaling the analytic real L2P reverse rule
   (`_evaluate_local_real_with_grad_cvjp_bwd`) by `1 + 1e-6` — a reverse-pass-only change,
   forward untouched by construction — leaves the **forward golden green** and turns the
   **gradient golden red**. The physics anchor cannot see it either; only the inertness gate
   can. Everything in Tier 1 and Tier 2 depended on this existing first, and no longer does.

2. **"0 pydoclint violations" is true and much weaker than it sounds.** `pyproject.toml`
   does not override pydoclint's default `--skip-checking-short-docstrings=True`, so any
   function whose docstring has no section headers is exempt entirely. Measured: with that
   default flipped off, the same config reports **2840 violations** across 584 functions
   missing `Parameters`, 600 missing `Returns`, and 161 that `raise` without a `Raises`
   section. 492 of 689 multi-parameter functions in the package have a section-less
   docstring; 73 of those take **eight or more parameters** and live in the numerics
   directories.

3. **`jaxtyping` carries no shape information anywhere in the package.** 47 modules import
   `Array` from `jaxtyping`, and **zero** occurrences of `Float[...]` / `Int[...]` /
   `Shaped[...]` exist in `jaccpot/` or `tests/`. `jaxtyping.Array` is an alias for
   `jax.Array`. So the `@jaxtyped(typechecker=beartype)` decorators (45 of them) and the
   `JACCPOT_RUNTIME_TYPECHECK=1` import hook verify "is an array" and nothing about shape or
   dtype. STYLE_GUIDE §4's *"so shape and dtype live in the signature"* describes an
   intention, not the current state.

4. **The two largest problems are functions, not modules.** The five modules you named are
   large, but the units inside them are worse: `PrepareMixin._prepare_state_dual_and_downward`
   is **883 lines**, `prepare_large_n_state` is **1253 lines**, and
   `_fmm_impl.FastMultipoleMethod.__init__` is **722 lines** with 60 parameters. A module
   split that leaves those intact buys very little.

   *Tier 1 status:* the 883-line one is now **493** (1.7). `prepare_large_n_state` is
   untouched by design — 1.8 confirmed its reverse path is still at 0% coverage, so
   A.9's leave-it-whole argument holds. The 722-line constructor is Tier 2.1 and
   remains the largest single unit in the package. The module splits did land (1.3,
   1.4, 1.5, 1.6), so the *remaining* concentration of risk is now almost entirely in
   that constructor.

5. **Two latent `NameError`s exist in `runtime/kernels/core.py`.** Both are in reachable
   branches, both are uncovered by the suite. They are in section G, not section B — bugs are
   not fixed inside a refactor. Tier 1.6 moved them to `kernels/_l2l.py:607` and
   `kernels/_evaluate.py:1285`; G.1 carries the full re-location table, and the
   before/after `pyflakes` correspondence that proves the split left them alone.

6. **Six public names in `operators/real_harmonics.py` could not execute at all — now removed
   (`ab58c3d`).** The Wigner-D
   reference family needs `sympy`, which appears exactly **once** in the repository — the
   import itself at `real_harmonics.py:1123` — and is declared in neither `dependencies` nor
   `[project.optional-dependencies].dev`. Verified: nothing in the dependency closure pulls it
   in (jax → numpy/scipy/ml_dtypes/opt_einsum; jaxlib; jaxtyping → wadler-lindig; yggdrax;
   beartype's runtime deps), and CI installs only `yggdrax` + `jaccpot[dev]`, so it is absent
   there too. All six raise `ImportError: sympy is required for Wigner-D baseline rotation`.
   This was found by trying to *use* the family for Tier 0.3 (F39, G.4, G.8).

---

## A. Structure as-is vs. as-intended

### A.1 The import graph is acyclic

Verified by AST walk over all 79 non-experimental modules, resolving relative imports and
`from X import Y` where `X.Y` is itself a module. **No cycles**, at module scope or including
function-local imports. The cycle rule in ARCHITECTURE §8 is being honoured: nothing imports
the engine through `runtime/fmm/__init__.py`, and `runtime/kernels/` imports nothing from the
orchestrator. `distributed/` and `experimental/` reach `kernels/` directly, as documented.
**No finding.** This is the healthiest structural property the codebase has and the work plan
should not disturb it.

### A.2 STYLE_GUIDE §8's upward-import inventory is incomplete

§8 states there are *"Four such relationships … across five import statements."* Measured,
there are **seven relationships across eleven statements**. The four documented ones are all
present and correct. The three undocumented ones:

| Edge | Sites | Status |
|---|---|---|
| `nearfield/near_field.py` → `pallas/nearfield_fused_leaf` | `near_field.py:3302, 3381, 3458, 3745, 4016` (all function-local) | Real and almost certainly deliberate — `_radix_fast_lane_prepacked_accel_cvjp` is the production Pallas near-field rule per ARCHITECTURE §7, and the function-local placement is the documented "defer heavy Pallas imports" pattern. **Needs listing, not fixing.** |
| `operators/m2l_real_rot_scale.py:78` → `pallas/m2l_core_z_real` | 1, function-local | This one is a genuine layering question. §8 says `operators/` is *"the mathematical algebra. Pure, jittable. No tree logic, no config parsing, no I/O"* — an accelerator import is none of those, but it is also not algebra. Section G. |
| `basis/complex_sh.py:10-11` → `operators/complex_ops`, `operators/real_harmonics` | 2, module-scope | Within the single "mathematical algebra" tier §8 defines. Benign; listing it removes a future reader's doubt. |

The fix is a docs edit to §8 (Tier 0), not an import change.

### A.3 Environment variables are read outside `runtime/`, and two guides disagree about it

STYLE_GUIDE §8: *"`runtime/` — orchestration, config resolution, lane selection, dispatch. The
only place that reads environment variables or resolves `"auto"` policies."*

`jaccpot/_env.py:21-22` (module docstring): *"This module deliberately imports nothing from
`jaccpot` so that any module, at any layer, can use it without an import cycle."*

Those cannot both be the policy. Measured env reads outside `runtime/`:

- `nearfield/near_field.py` — 13 sites (`:44` in `_large_n_nearfield_diag_mode`, then
  `:3742, 3819, 3820, 3821, 3850, 3852, 3854, 3930, 3934, 3938, 4020, 4032, 4033, 4035`)
- `pallas/m2l_real_fused.py:51` and `pallas/m2l_complex_fused.py:57`
- `upward/tree_geometry.py:54`

`_env.py` is clearly the newer, deliberate decision (its docstring records the four-divergent-
readers incident it consolidated). The likely resolution is that §8's sentence should say
*resolves policy* rather than *reads environment variables*. Section G — I am not going to
guess which document is authoritative.

### A.4 Four hand-rolled env parsers survive alongside `_env.py`

`_env.py` exists to be the single implementation. Four call sites bypass it:

- `pallas/m2l_real_fused.py:43-56` `_fused_m2l_vjp_enabled()` and
  `pallas/m2l_complex_fused.py:49-62` `_fused_m2l_vjp_enabled()` — **byte-identical bodies**,
  two copies of the same function, sharing one env var (`JACCPOT_FUSED_M2L_VJP`).
- `upward/tree_geometry.py:51-59` `_jit_enabled()`.
- `nearfield/near_field.py:42-48` `_large_n_nearfield_diag_mode()`.

**Do not dedupe these mechanically.** `_env.env_flag` returns `False` for an unparseable
value; these four return the *default* (i.e. `True` for the default-on ones). Replacing
`_fused_m2l_vjp_enabled` with `env_flag("JACCPOT_FUSED_M2L_VJP", True)` would silently flip
`JACCPOT_FUSED_M2L_VJP=garbage` from "fused VJP on" to "fused VJP off" — a change to which
reverse-mode kernel runs. Either extend `_env` with a `default-on` reader or leave them and
comment why.

### A.5 Vestigial re-export aliases and one unused import in `near_field.py`

`near_field.py:54-55` are `_env_flag = env_flag` and `_env_int = env_int` — aliases left
behind by the `_env` consolidation so call sites did not have to change. `env_float` is
imported at `near_field.py:23` and never used anywhere in the file (grep: one hit, the import
itself). Pure Tier 0.

### A.6 20 test files sat outside the five documented test tiers — CLOSED (`a54f29e`)

CLAUDE.md's Layout documents `tests/{unit,integration,characterization,perf,experimental}`.
`git ls-files` shows 20 tracked `.py` files directly under `tests/`, including *all eight*
distributed tests, both fused-Pallas M2L parity suites, and files that are unambiguously
one tier or another by content:

- integration by nature: `test_gravity_vs_direct.py`, `test_real_basis_runtime.py`,
  `test_real_upward_sweep.py`, `test_fastlane_geometric_centers.py`,
  `test_device_only_default.py`, `test_force_scale_runtime.py`,
  `test_adaptive_order_runtime.py`, `test_adaptive_policy_runtime.py`
- unit by nature: `test_real_sh_roundtrip.py`, `test_m2l_real_fused_pallas.py`,
  `test_m2l_complex_fused_pallas.py`
- a sixth tier that does not exist yet: the eight `test_distributed_*.py` +
  `test_p2p_shard_map.py`, all of which skip below 2 devices

This matters beyond tidiness: it is why `-m "not slow and not experimental"` is hard to reason
about, and why "the distributed suite" has no single path to name.

**Closed in `a54f29e`**, with one hazard worth recording because it is not obvious and the
move nearly tripped it: `slow_tests.txt` marks tests `slow` **by node id**, and a stale id does
not error — it silently stops matching. Moving these files without rewriting it would have
un-marked ~15 compile-bound tests and pushed them into the 3.11/3.12 smoke leg, which has 30
minutes and no headroom. The 17 ids and 32 downstream path references were therefore rewritten
from git's own rename detection, and the guard was the collected counts (911/949 and 779 under
`-m "not slow"`, identical before and after). CLAUDE.md now documents the coupling.

### A.7 Repository-root clutter — confirmed, with a recommendation

| Path | Contents | Recommendation |
|---|---|---|
| `bench/` | 21 files: 3 `audit_*`, 6 `bench_*`, 4 `profile_*`, `ci_benchmark_guard.py`, `guard_large_n_radix_fast_lane.py`, `nearfield_leafpair_microbench.py`, `repro_jax_ragged_all_to_all_grad.py`, `real_vs_complex_gpu_plan.md`, `validation/` (2 modules) | Keep as the one benchmark home. `real_vs_complex_gpu_plan.md` belongs in `docs/`. |
| `benchmarks/` | 1 subdir `n_ladder_production/` with a README, a `.json` and a `.md` of recommendations — **no executable code** | This is a *result*, not a benchmark. Merge into `bench/results/n_ladder_production/` or `docs/`; delete the directory. Its near-name collision with `bench/` is a trap. |
| `results/validation/` | 10 committed `.json` outputs (≈230 kB) from `bench/validation/{mac_error_distribution,force_scale_prepare_cost}.py` | These are referenced from `docs/dehnen_mass_mac_status_and_plan.md`-era work and are genuine measurement records, so do not delete them — but a top-level `results/` reads like build output. Move to `bench/results/` (next to the scripts that produce them) or `docs/measurements/`. |
| `OCTREE_JACCPOT_STATUS_2026-03-15.md` | 276-line dated handoff at root. References absolute paths on another machine (`/export/home/tbuck/jaccpot`), a branch pair (`oct-tree` / `oct-tree-updates`), a specific GPU ("GPU 3"), and a `micromamba` env. Referenced by nothing in the repo. `docs/octree_fmm_task_list.md` (60 lines) covers the same programme. | Move to `docs/octree_status_2026-03-15.md` with a one-line "superseded by" header, or delete. It should not be at root either way. |

**Done in `6839a6c`** — all four moves landed as `git mv`, plus the six invalidated path
references (both `bench/validation/` `--json-out` examples, five paths in
`docs/dehnen_mass_mac_status_and_plan.md`, and `examples/README.md`, whose link pointed at
an absolute `/export/home/tbuck/...` path on another machine).

Consolidated shape, as built:

```
bench/                 executable benchmarks, audits, guards  (unchanged)
bench/results/         benchmarks/n_ladder_production/ + results/validation/
docs/                  design notes, audits, plans, + real_vs_complex_gpu_plan.md
docs/octree_status_2026-03-15.md      (was root)
```

Only `pyproject.toml`'s `[tool.coverage]` and `bench/ci_benchmark_guard.py` need checking for
hard-coded paths; nothing else references these directories.

### A.8 ARCHITECTURE.md's own structural claims have drifted

Not vibes — specific sentences that a reader will act on:

- §4: *"`_fmm_impl.FastMultipoleMethod` is a thin coordinator (constructor, backend plumbing,
  cache lifecycle, autotune-cache IO)."* Its `__init__` is **722 lines** (`_fmm_impl.py:352-1073`)
  with **60 parameters**. Everything else in the class totals 316 lines. The coordinator is
  not thin; the *mixin split* worked and the constructor did not move.
- §2 table lists **12** `__all__` names and §9 says the guard *"freezes the 12 `__all__`
  names"*. There are **14** (`jaccpot/__init__.py:23-37`), and
  `tests/unit/test_public_api_surface.py:25-39` correctly freezes all 14. Missing from the
  table: `GradConfig`, `TraversalOverrides`.
- §9: the golden oracle *"drives the FMM over a grid (N, order, basis, farfield modes,
  outputs)."* `tests/characterization/test_fmm_golden.py:76-90` `CASES` has four axes —
  distribution, N, basis, order. There is no farfield-mode axis and no outputs axis; only
  accelerations are snapshotted (`:167`), from a single `preset="accurate"` build (`:140`).

### A.9 Oversized modules — seams, with one "leave it whole"

Line counts and the largest unit inside each:

| Module | Lines | Largest unit |
|---|---|---|
| `nearfield/near_field.py` | 4313 | `_compute_leaf_p2p_impl` 487 (`:1754`), `_compute_leaf_p2p_from_prepared_leaf_data_impl` 460 (`:2253`) |
| `runtime/kernels/core.py` | 3827 | `_prepare_solidfmm_downward_sweep` 348 (`:2165`), `_evaluate_local_expansions_for_particles` 338 (`:3439`) |
| `runtime/fmm_prepare.py` | 2953 | one class, `_prepare_state_dual_and_downward` **883** (`:719`), `_prepare_state_uncaught` 475 (`:2479`) |
| `operators/real_harmonics.py` | 2236 | `p2m_real_direct` 165 (`:334`), `evaluate_local_real` 159 (`:507`) |
| `runtime/_large_n_pipeline.py` | 2048 | `prepare_large_n_state` **1253** (`:321`) |

**All five line counts above are pre-Tier-1 and every one has moved.** Current state
(`main`, plus the near-field chain still in review as PRs #80/#83/#84):

| Module | audited | now | by |
|---|---|---|---|
| `nearfield/near_field.py` | 4313 | **1388** | 1.2, 1.4, 1.5 (PRs #80, #83, #84) |
| `runtime/kernels/core.py` | 3827 | **101** (aggregator) | 1.6 (PR #85) |
| `runtime/fmm_prepare.py` | 2953 | 3181, but its largest unit 883 → **493** | 1.7 (PR #86) |
| `operators/real_harmonics.py` | 2236 | **283** (aggregator) | 1.3 (PR #82, merged) |
| `runtime/_large_n_pipeline.py` | 2048 | **1768** | 1.8 (PR #87, merged) |

`fmm_prepare.py` growing is expected and is the point: A.9 argued the *function* was
the problem, and the module gained a bundle type plus two documented signatures in
exchange for the driver halving. Whether the mixin still wants splitting is now a
question you can answer against a 493-line driver.

**`nearfield/near_field.py` — split, along five seams.** The 46 top-level defs partition
cleanly by line range:

1. `:42-330` diagnostics gate + `RadixFastLanePerfCounters` + schedule/leaf-data preparation
   (`prepare_leaf_neighbor_pairs`, `prepare_bucketed_scatter_schedules*`, `_prepare_leaf_data*`)
   → `nearfield/_schedules.py`
2. `:331-744` the arithmetic primitives (`_self_contributions`, `_pair_contributions*`,
   `_bucketed_chunk_pair_accels`) and the scatter family (`_scatter_*`, `_build_scatter_schedule`)
   → `nearfield/_kernels.py` / `nearfield/_scatter.py`
3. `:790-1740` the large-N target-block family (10 `_compute_leaf_p2p_prepared_large_n_*`
   functions) → `nearfield/_large_n_blocks.py`
4. `:1754-2715` the two edge-list kernels + `:2716-3085` the public
   `compute_leaf_p2p_accelerations` entry point → stays in `near_field.py`
5. `:3086-4313` the radix fast lane, its Pallas wrappers and its `custom_vjp`
   (`_radix_fast_lane_*`, `compute_leaf_p2p_accelerations_radix_fast_lane`)
   → `nearfield/_fast_lane.py`

Seam 5 is the one to do first: it is self-contained, it is where all five function-local
Pallas imports live (A.2), and it is where the `custom_vjp` lives, so isolating it makes the
one thing NUMERICS §1 calls *"load-bearing"* reviewable on its own.

**`runtime/kernels/core.py` — split, along four seams.** Note ARCHITECTURE §5 already
*documents* this module as four families, so the split is making the file match the doc:

1. `:112-477` downward-sweep input preparation (`_prepare_solidfmm_downward_*`,
   `_solidfmm_downward_accumulate_from_multipoles`) → `kernels/_downward_prep.py`
2. `:932-1947` the whole M2L apply + accumulate seam (`_apply_m2l` and everything it
   dispatches to, the grouped/class-major accumulators) → `kernels/_m2l.py`
3. `:1948-2541` L2L propagation + `_prepare_solidfmm_downward_sweep` → `kernels/_l2l.py`
4. `:2542-3827` evaluation: L2P, targeted nearfield, the scatter helpers → `kernels/_evaluate.py`

The M2L seam (2) must move as one piece: `_apply_m2l`, `_apply_real_m2l`,
`_apply_complex_m2l`, the two `*_pallas_active` gates and the two fused wrappers are the
`basis_mode` dispatch invariant of ARCHITECTURE §10 and splitting *within* them would break
the "every discriminator is a `static_argname`" argument.

**`runtime/fmm_prepare.py` — split the function, not the module.** The module is one class
(`PrepareMixin`, `:118-2953`) whose 33 methods already sort into four groups by name:
topology reuse (`:175-428`), tree+upward (`:429-718`), dual-tree + downward
(`:719-1822`), nearfield artefacts (`:1823-2160`), entry points (`:2149-2953`). But splitting
the mixin four ways gains little while
`_prepare_state_dual_and_downward` is 883 lines. **Extract from that function first**
(Tier 1), then reassess whether the module still needs splitting. My guess is it will not.

**`operators/real_harmonics.py` — split, along six clean mathematical seams.** This is the
easiest of the five because the boundaries are mathematical, not incidental:

1. `:213-333` indexing/packing + azimuth helper → `operators/_sh_indexing.py`
2. `:334-863` P2M, L2P, and the L2P `custom_vjp` → `operators/real_p2m_l2p.py`
3. `:864-1119` Dehnen `Q`, the `B` matrices, complex→real conversion → `operators/real_dehnen_q.py`
4. `:1120-1239` + `:1240-1298` the Wigner reference family (see D.5 — this may not survive)
5. `:1299-1716` closed-form rotation builders → `operators/real_rotations.py`
6. `:1717-1940` z-translations, `:1941-2186` assembled M2L/M2M/L2L → `operators/real_translations.py`

**`runtime/_large_n_pipeline.py` — argue for leaving the module whole, and split the
function.** The module is three things: an env-config reader (`_read_large_n_env_config`, 271
lines), `prepare_large_n_state` (1253), and `evaluate_large_n_state` (334). Splitting the env
reader out is fine but cosmetic. The real problem is that `prepare_large_n_state` takes 26
parameters, has a **one-line docstring**, and is at **71% coverage with 0% for the reverse
path it feeds** (`_large_n_grad.py`, see D.3). I would **not** restructure this module in this
pass: it is the least-verified production lane in the tree, and moving 1253 lines of it
without a characterization net is exactly the change NUMERICS §3 warns against. Split the env
reader (Tier 0/1), and leave the rest until it has coverage. That is a legitimate "stays
whole" answer and I am giving it deliberately.

### A.10 The largest duplication in the package: ~460 lines, in the near field

`_compute_leaf_p2p_impl` (`near_field.py:1754`, 487 lines) and
`_compute_leaf_p2p_from_prepared_leaf_data_impl` (`near_field.py:2253`, 460 lines) are
**94.4% line-identical** (measured with `difflib.SequenceMatcher` over their bodies; the full
unified diff is 90 lines including context).

The *entire* semantic difference is the first 15 lines. `_compute_leaf_p2p_impl` takes
`(node_ranges, leaf_nodes, positions, masses, max_leaf_size)` and derives
`(leaf_positions, leaf_masses, leaf_mask, leaf_particle_idx)` via `_prepare_leaf_data`; the
other takes those four as arguments and recovers `max_leaf_size` from
`leaf_particle_idx.shape[1]`. Everything after that is identical modulo `black` reflow and
two `# type: ignore[arg-type]` comments present on one copy only (`:1885`, `:1954`, absent on
their twins) — which is itself evidence the copies have already drifted.

Both callers are in one function (`:3019` and `:3053` inside
`compute_leaf_p2p_accelerations`), plus `experimental/octree_fmm_uvwx.py:793`.

The delegation pattern is **already used in this file**:
`_compute_leaf_p2p_prepared_large_n_accel_only_impl` (`:1687`) calls
`_prepare_leaf_data_from_groups` and then delegates to the prepared-data kernel. So the fix is
to follow the file's own precedent, not to invent one.

One caveat that must be measured, not assumed: both functions are `@partial(jax.jit, ...)`,
and `_compute_leaf_p2p_impl` carries an extra `static_argnums=(12,)`. Turning it into a
wrapper means a jitted function calling a jitted function. XLA inlines that, but it creates a
second compilation-cache entry — and NUMERICS §1 says compile time is a first-class metric.
**Measure compile time before and after.**

---

## B. Findings

`numerics-sensitive` is per NUMERICS_AND_JAX §1: yes if the change could alter a float result,
a reduction order, a dtype, a `static_argname`, a `custom_vjp`, or a sharding annotation.

Risk = probability the change breaks something. Effort = S (<1h), M (a day), L (multi-day).

| ID | file:line | category | what | why it matters | risk | effort | num-sens? |
|---|---|---|---|---|---|---|---|
| F01 | `runtime/kernels/core.py:2457` | error-handling | `_accumulate_from_multipoles` is called but defined nowhere in the module or its imports; the call also drops the 15 keyword arguments its evident intended callee (`:362`) requires | `NameError` on a single-internal-node tree with the source-motion tower active — a case NUMERICS §3 names explicitly. Branch is uncovered (`else` of `if child_inputs.num_internal_nodes > 0` at `:2363`) | — | S | **yes** → **section G** |
| F02 | `runtime/kernels/core.py:3265` | error-handling | `evaluate_local_complex_with_grad_analytic` called; the module imports only the `_batch` variant (`:56`). The non-batch function exists at `operators/complex_ops.py:551` | `NameError` when `max_acc_derivative_order <= 0 and return_potential` on the complex derivative path. Uncovered | — | S | **yes** → **section G** |
| F03 | `tests/characterization/` | test-gap | ~~13 forward acceleration cases; **no gradient golden**~~ **CLOSED by `e1f1455`.** A potential golden and the mode/preset axes of D.2 are still open | The stated safety net did not cover the property the library exists for. Mutation-verified: a reverse-only 1e-6 break is invisible to the forward golden and caught by the new one | low | M | yes |
| F04 **✔`ff4c911`** | `pyproject.toml:191-195` | repo-hygiene | `omit` excludes `jaccpot/pallas/m2l_real_fused.py` with the reason *"has no interpret path, so CPU CI structurally cannot reach it"*. Measured: `tests/test_m2l_real_fused_pallas.py` + `tests/unit/test_custom_vjp_parity.py` alone give it **92%** (191 stmts, 12 missed) on CPU | Same defect class as `b4d09a2` fixed one commit ago for `treecode_walk.py`: a coverage omit hiding a well-covered module *and* a stated reason that is untrue | low | S | no |
| F05 **✔`c2874b6`** | `nearfield/near_field.py:2860-2867` | docstring | Caveat block describes `test_nearfield_bucketed_matches_baseline` as *"one configuration (N=96, float32, … a single PRNG seed)"* at `rtol=atol=1e-5`, *"marked `slow`"*. Since `e664c36` it is 4-way parametrised (N∈{96,256} × fp32/fp64 × chunk∈{64,128}) at 1e-6/1e-13, with the fp32/chunk-128 cases in the smoke leg | The docstring now understates its own coverage, and a reader looking for the weakest link will look at the wrong place | low | S | no |
| F06 **✔`c2874b6`** | `runtime/fmm_evaluate.py:1117` | docstring | Same staleness: *"at `rtol=atol=1e-5`"* and *"what that one configuration does and does not cover"* | Ditto | low | S | no |
| F07 | `operators/real_harmonics.py:371-382` | docstring | Says degrees 2-6 of the Dehnen Table 3 match are *"unverified in-repo"* and *"the coverage that exists cannot see a per-`m` normalisation or sign error above degree 1"*. Since `7b85028`, `test_complex_to_dehnen_real_matches_p2m_real_direct` cross-checks `p2m_real_direct` against `complex_to_dehnen_real_coeffs ∘ complex_R_solidfmm` at orders 0,1,2,3,4,6 over 6 geometries to 1e-13 rel-L2 | **This was the starting point I disagree with most.** See D.10 for the residual gap, which is real but narrower and different | low | S | no |
| F08 | `runtime/_fmm_impl.py:330-350` | docstring | The only Google-style `Args:` block left in the package, on the engine class; documents 9 of the constructor's **60** parameters | pydoclint's numpy style does not see it; a reader of the most parameter-heavy object in the codebase gets 15% of it | low | M | no |
| F09 | `runtime/_fmm_impl.py:352` | structure | `__init__` is 722 lines / 60 parameters | Unreviewable; every config-resolution bug lands here. ARCHITECTURE §4 calls the class "thin" | med | L | yes (resolution order) |
| F10 | `runtime/fmm_prepare.py:719` | structure | `_prepare_state_dual_and_downward` is 883 lines | The largest single unit under `PrepareMixin`; blocks any sensible split of the module | med | L | yes |
| F11 | `runtime/_large_n_pipeline.py:321` | structure | `prepare_large_n_state` is 1253 lines, 26 parameters, one-line docstring | Largest function in the package, in the least-covered production lane | high | L | yes |
| F12 | `nearfield/near_field.py:1754` & `:2253` | duplication | 94.4% line-identical pair, ~460 duplicated lines; already drifted (two `# type: ignore` on one copy only) | Largest duplication in the package, in the most numerics-sensitive module; a fix applied to one copy silently misses the other | med | M | yes |
| F13 | `pallas/m2l_real_fused.py:43` + `pallas/m2l_complex_fused.py:49` | duplication | `_fused_m2l_vjp_enabled()` byte-identical in two modules, one shared env var | A reverse-mode kernel selector defined twice. **Do not naively route through `_env.env_flag`** — the malformed-value semantics differ (A.4) | low | S | **yes** (selects which VJP runs) |
| F14 **✔`934c252`** | `nearfield/near_field.py:23` | dead-code | `env_float` imported, never used | — | low | S | no |
| F15 **✔`934c252`** | `nearfield/near_field.py:54-55` | dead-code | `_env_flag`/`_env_int` aliases left from the `_env` consolidation | Two names for one function in a 4313-line file | low | S | no |
| F16 | `runtime/_fmm_impl.py`, `runtime/kernels/__init__.py` | **CORRECTED** — was "13 unused imports"; there are **204** in `_fmm_impl.py` and 60 in `kernels/__init__.py`, and they are **not dead code**. `runtime/fmm/__init__.py` does `from .._fmm_impl import *` plus an explicit `build_interactions_and_neighbors`, and tests import `_build_nearfield_interop_data`, `_evaluate_local_expansions_for_particles`, `_prepare_solidfmm_downward_sweep` from `_fmm_impl` without it using them. ARCHITECTURE §3 calls `kernels/__init__.py` "curated re-exports"; pyflakes flags all of them only because neither module declares `__all__` | My original count came from a truncated pyflakes view — a reminder that `grep \| head` is not a measurement. Bulk-removing these would break consumers. 13 that are consumed nowhere were removed in `934c252` after checking every `from ... import` in `jaccpot/`, `tests/`, `bench/`, `examples/`; the rest need an explicit `__all__` and a decision about what the seam guarantees | med | M | no → **Tier 2**, not Tier 0. **CORRECTED AGAIN (Tier 1.4)** — see the sharpened statement below the table |
| F17 **✔`934c252`** | `autodiff.py:7`; `operators/complex_harmonics.py:15,17,22`; `runtime/fmm_derivatives.py:28`; `runtime/kernels/__init__.py:13` | dead-code | 6 further unused imports (`jax`, `math`, `Tuple`, `sh_offset`, `_contains_tracer`, two unused re-exports) | — | low | S | no |
| F18 **✔`934c252`** | `runtime/fmm_strict_run.py:146`; `runtime/_adaptive_policy.py:741` | dead-code | `prepare_elapsed` and `shift` assigned, never read | A dropped timing value in the strict lane is worth checking — it may be a lost diagnostic rather than dead code | low | S | no |
| F19 | ~~`operators/real_harmonics.py:1267-1271`~~ | naming | **MOOT — the containing function (`_rotation_to_z_angles`) was deleted in `ab58c3d`.** Was: `R00`/`R01`/`R10`/`R11` computed and unused, which I judged to be deliberate narration rather than a defect | Recorded because the judgement still applies elsewhere: writing a full 3×3 out is the narration STYLE_GUIDE §5 asks for, and XLA eliminates it | low | S | no |
| F20 | package-wide | typing | 1534 bare `Array` annotations; **0** shaped `jaxtyping` annotations; 350 unannotated parameters | STYLE_GUIDE §4's stated benefit is entirely unrealised; the runtime typecheck verifies nothing about shape. See §E | low | L | no (annotations only) |
| F21 **✔`ad7b00c`** | `runtime/fmm_*.py` (12 modules) | typing | 90 dangling forward refs (`FastMultipoleMethod`, `PreparedStateLike`, `LargeNGradPlan`, `NearfieldInteropData`), no `TYPE_CHECKING` block anywhere in `runtime/` | **The benefit I stated here was wrong.** I claimed the payoff was that `typing.get_type_hints` would stop raising; measured before and after, it is **identical** (50 resolvable / 63 not) — `TYPE_CHECKING` imports cannot help runtime introspection, which is inherent Python. The real payoff, measured: `pyflakes jaccpot/` undefined names **95 → 5**, which is what makes pyflakes usable as a bug-finder on this package — those 90 false positives are exactly what buried the two genuine `NameError`s of G.1 in 397 lines of noise. The static-checker benefit is real but unconsumed: neither mypy nor pyright is in `[dev]`. `kernels/core.py`'s two sites are deliberately left dangling to preserve the leaf invariant | low | M | no |
| F22 | `pyproject.toml:141-143` | test-gap | pydoclint runs with the default `--skip-checking-short-docstrings=True`; flipping it reveals **2840** violations (584 DOC101, 600 DOC201, 161 DOC501/503) | The "structural docstring compliance is done" premise holds only for functions that already have sections | low | L | no |
| F23 **◐ part 1 `PR #88`** | 73 functions in the numerics dirs (worst: `runtime/kernels/core.py:2542` 48 params/1 doc line; `nearfield/near_field.py:4067` 31/1 — a *public* name; `runtime/_large_n_pipeline.py:321` 26/1) | docstring | ≥8 parameters, no `Parameters` section | These are precisely where STYLE_GUIDE §3's "shapes, units, static args, differentiability, accuracy regime" is load-bearing. **Re-measured over NUMERICS §1's six numerics dirs: 60 functions** (so ~58 was right). 18 closed by PR #88. The remaining 42 **cannot** go in one PR, for two structural reasons, both measured — see the note below the table | low | L | no |
| F24 **✔`20ba5d8`** | `upward/real_tree_expansions.py:242-257` | docstring | `prepare_real_upward_sweep` documents only `max_leaf_size`; `static_num_levels` (`:250`, used at `:272`) is undocumented, while the complex twin's identical parameter has a full bit-identity rationale (`upward/solidfmm_complex_tree_expansions.py:480-497`) *and* a test | The real sweep's copy of a documented-and-tested optimisation is neither | low | S | yes (see D.9) |
| F25 **✔`26cef45`** | `pallas/nearfield_fused_leaf.py:667` | test-gap | *"Passing the same array as both target and source reproduces `nearfield_leafpair_pallas` bit-for-bit"* — asserted nowhere; grep for "decoupled" in `tests/` returns nothing | This is the kernel `distributed/fmm.py:575` runs via `_radix_fast_lane_prepacked_pallas_decoupled`. The distributed near-field's equivalence to the single-device one rests on an unasserted claim | low | M | yes |
| F26 **✔`568deca`** | `pallas/m2l_real_fused.py:51`, `pallas/m2l_complex_fused.py:57` | test-gap | No test sets `JACCPOT_FUSED_M2L_VJP=0`, so the documented *"correctness reference"* fallback branch never runs | A fallback that is never exercised is not a fallback | low | S | yes |
| F27 | `runtime/_large_n_grad.py` (whole file), `runtime/_large_n_farfield.py`, `distributed/cap_presets.py` | test-gap | **0% coverage**, measured. Gated behind `can_use_large_n_prepare_path`'s `jax.default_backend() != "gpu"` check (`_large_n_pipeline.py:2013`) | ARCHITECTURE §6 quotes measurements for this path ("1M particles, forward 2.5 s, forward+backward 69 s at 11 GB"). Nothing in CI touches it | low | L | yes |
| F28 **✔`d941805`** | `tests/unit/test_large_n_grad_path.py` | naming | Named for the large-N grad path; measured, it never enters it (20 tests pass, 0 skip, `_large_n_grad.py` stays at 0%). Its docstring is honest — it tests resolver policy at large-*N configuration thresholds* on the radix lane | The filename is the only documentation most readers will see, and it says the opposite of what the file does. Reinforces the false impression that F27 is covered | low | S | no |
| F29 | `operators/real_harmonics.py:1520-1523` | test-gap | **CLOSED by `e5d8e41`.** The claimed identity was unasserted; it is now asserted, and it **holds** — measured 7.6e-15, and ~2.5e-15 worst case across all four builders. The production code was right | The azimuth convention here (`atan2(x, y)`, not `atan2(y, x)`) is the documented historical bug site. Mutation check: the swap fails 8/8 new tests and **passes 4/4 pre-existing ones** | low | M | yes |
| F30 | `operators/real_harmonics.py:1526-1545`, `:1563-1580` | duplication | `_multipole_align_to_z_block` and `_multipole_align_from_z_block` share 12 identical lines of angle computation and a 9-line identical comment block | Two copies of a NaN-safe double-`where` guard is two places to get it wrong. Extractable verbatim with no expression change (do **not** add a `@jit`) | low | S | **yes** |
| F31 | `operators/real_harmonics.py:1372-1379`, `:1401-1408` | **BUG** | The rotation azimuth's *"a zero cotangent is the correct subgradient"* at `rho == 0` is not merely unasserted — **it is false, measured.** Attempting to write the Tier 0.4 test refuted the claim instead of confirming it. Promoted to **G.10**; not to be fixed inside a refactor | The `where`-guard-kills-the-gradient defect was found and fixed twice in this file (`d5cb13b`, and the P2M comment at `:404-410`). The rotation path received the guard without the test, and the guard is wrong | — | M | **yes** → **section G.10** |
| F32 | ~~`operators/real_harmonics.py:1120-1298`~~ **CLOSED `ab58c3d`** | dead-code | The Wigner reference family (`_wigner_D_complex`, `_real_wigner_rotation`, `_rotation_to_z_angles`, `_dehnen_real_basis_scale_diag_{multipole,local}`, and **6** public `*_wigner` names) is exported in `__all__` but reachable only from itself. **Corrected: 6, not 8** — the original count was wrong. `m2l_real_wigner` has zero references outside `__all__`. Grepped `jaccpot/`, `tests/`, `bench/`, `examples/`, `docs/` | `_real_wigner_rotation`'s own docstring says *"this exists to check them"* — and it never does. ~260 lines of independent 30-digit reference, unused **and unusable** (F39). The NaN-guard at `:1277-1292` was even applied to this unreachable copy | low | M | no → **section G** |
| F33 | `runtime/fmm_strict_run.py` | test-gap | **55%** coverage (502 stmts, 227 missed), including most of `strict_run_v2` (428 lines, `:389`) and `_refresh_large_n_same_topology` (515 lines, `:890`). `runtime/fmm_strict_cap_profile.py` 35%; `runtime/fmm_autotune.py` 15% | The strict/refresh lane is the per-step hot path for science runs and carries the velocity-Verlet update. Half of it is unexercised | low | L | yes |
| F34 | `distributed/fmm.py` | test-gap | **19%** coverage (520 stmts, 423 missed). All 8 distributed suites skip below 2 devices (`device_count() < 2`) | Documented and expected, but it means the whole `distributed/` layer is unrefactorable in this pass — record it, do not touch it | low | — | yes |
| F35 | `runtime/_interaction_cache.py:892` | structure | Production `distributed/fmm.py` (`local_walk="treecode"`) reaches `experimental/treecode_far_near.py`. Already documented in STYLE_GUIDE §8 as *"the weakest-covered production option in the tree"*; my measurement confirms `treecode_far_near.py` at 0% | Not a new finding, but it is the one documented layering violation that is a *correctness* risk rather than a style one. Listed so the work plan does not disturb it | — | — | yes → **section G** |
| F36 **✔`23ccc9d`** | 13 modules incl. `jaccpot/__init__.py`, `runtime/_fmm_impl.py`, `runtime/_interaction_cache.py`, `runtime/_nearfield_cache.py`, and 9 `__init__.py` | structure | Missing `from __future__ import annotations` (STYLE_GUIDE §1 says "make it 80"). 41 modules missing `__all__` | Mechanical, and the `_fmm_impl.py`/`_interaction_cache.py` cases are real modules, not just package inits | low | S | no |
| F37 | `runtime/fmm_overrides.py:253` (`num_particles <= 8192`), `:806` (`n >= 262_144`); `runtime/_large_n_types.py:269, 415, 448` (`65536` three times); `solver.py:75` + `runtime/_fmm_impl.py:1152` (`upward_leaf_batch_size=2048` twice) | magic-number | Thresholds inline rather than in `runtime/fmm_constants.py`, which has 40 named ones | Small, but the repeated `65536` and the duplicated `2048` are the ones that will drift apart | low | S | yes (policy thresholds) |
| F38 | `jaccpot/_env.py` vs STYLE_GUIDE §8 | structure | The guide says only `runtime/` reads env vars; `_env.py`'s docstring says any layer may. 16 reads outside `runtime/` | Contributors will follow whichever they read first | — | S | no → **section G** |
| F39 | ~~`operators/real_harmonics.py:1123`~~ **CLOSED `ab58c3d`** | api | **`sympy` was an undeclared dependency.** The only occurrence in the repository, absent from `dependencies` and from `[dev]`. Verified absent from the whole closure (jax, jaxlib, jaxtyping, yggdrax, beartype runtime) and therefore from CI, which installs only `yggdrax` + `jaccpot[dev]`. All 6 public `*_wigner` names raise `ImportError` when called — checked one by one | A module-public name that cannot execute in any environment the project's own install instructions produce. Found by trying to use it for Tier 0.3, which is why 0.3 had to be built differently. Also means the intended fix for F32 is not free: it costs a dependency, and CLAUDE.md forbids adding one without asking | low | S | no → **section G.8** |
| F40 **NEW** (Tier 1) | CLAUDE.md Verification block vs. `operators/complex_ops.py` and every other `lax.fori_loop` body | test-gap | **A documented verification command that has never passed.** CLAUDE.md offers `JACCPOT_RUNTIME_TYPECHECK=1 pytest -q tests/unit` as a faster inner loop. Measured on `main` (`128a0e2`, clean worktree): **126 failed, 667 passed, 44 skipped**. Dominant mode is one annotation pattern — a `lax.fori_loop` body annotated `_i: int` receives a **tracer**, so beartype raises `BeartypeCallHintParamViolation: parameter _i="JitTracer(~int64[])" violates type hint <class 'int'>`. First failure: `tests/unit/core/test_solidfmm_complex_tree_expansions.py::test_prepare_solidfmm_upward_source_motion_matches_finite_difference`, through `complex_ops.regular_solid_harmonic_directional_derivative_order` | Same defect class as F04's untrue coverage-omit reason: a stated check that does not check. Worse here, because a contributor who runs it will read 126 failures as *their* breakage — that cost time in Tier 1 before `main` was used as the control. The fix is an annotation change across many kernels (the correct hint for a loop index is not `int`), i.e. **F20/E.3, Tier 2** — and it is the same blocker that stops the `pallas/` third of F23 | low | M | no (annotations only) → **Tier 2** |

### B.1 Two findings sharpened by executing Tier 1

**F16, a third time: a module's *attribute* surface is wider than its *import*
surface.** F16 has already been corrected once (13 unused imports → 204, and not
dead code). Tier 1.4 found the next layer. After moving the radix fast lane out of
`near_field.py`, four imports there became unused; an AST sweep of every
`from ...near_field import X` in `jaccpot/`, `tests/`, `bench/` and `examples/` said
they were safe to drop. They were not:
`tests/unit/test_nearfield_fastlane_grad_path.py` reaches
`_leafpair_accel_analytic_vjp` as `nf.<attr>` after
`from jaccpot.nearfield import near_field as nf`, which **no import-statement scan
can see**.

So the rule F16 states — do not remove a re-export — needs a companion rule about
how to *check* it. The reliable test is not "who imports this name" but "does the
module's module-level surface change", which is a set difference over the AST of the
old and new file: functions, classes, module-level assignments **and every imported
name**. Tier 1.4 and 1.5 both used that, and 1.4's ended up showing exactly the ten
moved functions removed and nothing else. The four imports are restored with a
comment naming both access patterns.

Generalisation worth acting on: the same blind spot applies to
`_large_n_pipeline._read_large_n_env_config` (Tier 1.8 kept it imported for this
reason) and to anything reached through `runtime/kernels/core`. It is also why the
`kernels/core.py` split kept `fused_m2l_pallas_enabled` bound on the aggregator —
`tests/unit/test_grad_config.py` asserts `core.fused_m2l_pallas_enabled` even though
the gate is now consulted in `_m2l`.

**F23 is a three-PR programme, not one item.** The 42 functions PR #88 did not do
are blocked in two different ways, and neither is about effort:

1. **24 in `nearfield/near_field.py` — must be based on PR #84.** Tier 1.5 moves 15
   of those 24 into `_kernels.py`, `_scatter.py`, `_schedules.py` and
   `_large_n_blocks.py`. Writing their docstrings against `main` puts them in a file
   that will no longer contain the function, i.e. a hand-resolved conflict on every
   one, in the most numerics-sensitive module in the package.
2. ~~**18 in `jaccpot/pallas/`**~~ **— DONE, and it was 49 functions, not 18.**

   > **Closed.** The decision this was blocked on is made: `KernelRef = jax.Ref`,
   > aliased in `pallas/_compat.py` so the ~68 annotated ref parameters depend on one
   > line, since JAX moves the name and there is no `pallas.Ref`. Measured as accurate,
   > not just convenient: under `interpret=True` a kernel body receives a
   > `DynamicJaxprTracer` and `isinstance(that, jax.Ref)` is **True**. It is also not
   > enforced — nothing in `jaccpot/pallas/` carries `@jaxtyped`/`beartype` — so the
   > risk in choosing was nil, which is worth knowing before treating a Pallas
   > annotation as a contract.
   >
   > **The "18" was wrong.** Measured with `--skip-checking-short-docstrings=False`:
   > **49 functions, 197 violations** across five modules (`m2l_complex_fused` 63,
   > `nearfield_fused_leaf` 51, `m2l_real_fused` 48, `treecode_walk_pallas` 19,
   > `m2l_core_z_real` 16). Now **0**. Delivered in four parts to stay under the
   > ~400-line rule.
   >
   > One finding worth carrying: pydoclint reported **nothing** against the eight
   > kernel bodies, because it has no docstring to check a signature against. "Zero
   > violations" and "documented" are different predicates, and the kernels — the
   > point of the package — were the least documented things in it. Any future count
   > based on violation totals should be read with that in mind.

   The original report follows.

   **18 in `jaccpot/pallas/` — blocked on the Tier 2 typing decision.** The Pallas
   kernel bodies take unannotated references (`multipole_ref`, `bto_r`, …). Probed on
   `_m2l_core_z_real_kernel`: adding a `Parameters` section raises **DOC106, DOC107,
   DOC109 and DOC110** — `--arg-type-hints-in-signature` and
   `--arg-type-hints-in-docstring` are both `True`, so documenting one *requires
   annotating its signature*. That is F20/E.3, which this document routes to Tier 2,
   and E.1 already measures `pallas/m2l_complex_fused.py` as the highest unannotated
   count in the package (74). Deciding it inside a docstring PR would decide it by
   accident. Note **F40 is the same knot from the other side**: it shows the correct
   annotation for a `fori_loop` index is *not* `int`, so the two should be settled
   together.

### B.2 One stated verification that does not hold

Tier 1.3's "Verified by" column said *"`git log --follow` still works"*. **It does
not, and cannot, for a 1 → 5 split** — git's rename detection needs one dominant
source, and each new module is ~20% of the original, so `git log --follow` and even
`-M -C --find-copies-harder` on `operators/real_translations.py` show only the split
commit. Nothing is lost; the history is reached by the path that still exists:

```
git log --follow -- jaccpot/operators/real_harmonics.py      # full pre-split history
git log -L :m2m_real:jaccpot/operators/real_translations.py  # per-function, across the move
```

Recorded because the audit sets an expectation a reviewer would otherwise try to
check and find broken. The same applies to `nearfield/` (1.4, 1.5) and
`runtime/kernels/` (1.6).

### B.3 The A100 validation — §10 item 4, discharged, with one finding worth more than the result

Run on one **A100-PCIE-40GB** (compute capability 8.0, `autocvd(num_gpus=1,
least_used=True)` then pinned for the session so the interleaved benchmarks compare a
single card), jax/jaxlib **0.10.2**, `JAX_ENABLE_X64=1`. All three Pallas gates
reported `True`, so the fused lanes were live rather than falling back.

**Verdict: Tier 1 changed no number.** But the A/B had to be re-chosen first. The
handoff prescribes branch-vs-`main`, and by the time a GPU was free all ten Tier 1 PRs
were merged, leaving those branches 0 commits ahead of `origin/main` — that comparison
would have measured #80 and the docs PRs, not the thing under test. The run used
pre-Tier-1 `128a0e2` (PR #77, the commit before `cc8895e`) vs post-Tier-1 `da7ed57`
instead, with `PYTHONPATH` pinned per side and `jaccpot.__file__` printed on every run:
the editable-install finder appends to `meta_path`, so without that a two-worktree
comparison silently compares a tree to itself.

| Check | Result |
|---|---|
| Pallas-on vs pure-JAX parity | **65 passed, 0 skipped**, identical test-by-test both sides |
| End-to-end forward, 4 cases, `deterministic_ops=true` | **exactly equal**, max abs diff `0` |
| Reverse residuals, near-field fast lane on | JSON **identical** both sides |
| Reverse residuals at `--leaf 4` (added) | identical, and now covers m2l/m2m/l2l/p2m |
| Per-stage cost, interleaved post/base/post/base | no stage outside noise |

The empty skip list is the load-bearing part: every `interpret=False` parametrisation
ran against the Triton lowering — both `m2l_core_z` orders, both `m2l_complex_fused`,
both `m2l_real_fused`, `nearfield_fused_leaf`, `nearfield_leafpair`,
`radix_fast_lane_prepacked_accel_cvjp` — and the defensive
`pytest.skip("Pallas kernel unavailable …")` inside their `except` blocks, which would
read as a pass, did not fire.

**The finding: the forward path is nondeterministic under default XLA, by more than the
difference being measured.**

| Comparison | Max abs diff | Fraction of \|a\| |
|---|---|---|
| base vs post, default XLA | 3.6e-12 … 1.5e-11 | 1.8e-15 |
| **post vs post, same card, same code** | 3.6e-12 … 2.1e-11 | **4.4e-15** |
| base vs base, same code | 2.3e-12 … 1.5e-11 | 1.8e-15 |
| base vs post, `xla_gpu_deterministic_ops=true` | **0** | **0** |

The same tree on the same card differs run-to-run by *more* than the cross-tree
difference. So §10's exact-equality items are unsatisfiable on GPU as written,
independently of any refactor — a future run will "find a difference" and it will mean
nothing. ARCHITECTURE §10 now requires the flag; that wording fix is the durable
outcome of this run.

Two prescriptions in the handoff were also wrong in ways worth recording. Its
residual-audit config (`N=4096 leaf=64`) reports `far_pairs=0`, so it measures the M2L
coefficient **not at all** — D.12 again, in the document that warns about D.12. Re-run
at `--leaf 4`: 195,336 far pairs, 298.577 MB of M2L residuals under comparison at
1,528.5 bytes per pair. And the repo's default `-n auto` is 64 xdist workers against one
GPU: 46,000+ `CUDA_ERROR_OUT_OF_MEMORY` lines and 331/288 "failures" that are allocation
artifacts. The tell was that wall times (2760s vs 1309s) *and* failure sets differed
between sides. Re-run at `-n 6` with `XLA_PYTHON_CLIENT_MEM_FRACTION=.12`: zero OOM.

Suite, `-n 6`: base **966 passed / 24 skipped / 3 failed**, post **964 / 24 / 5**, against
the CPU reference's **926 / 67 / 0**. Passed rising and skips falling by ~43 is the
GPU-gated tests unskipping, as predicted. Every failure was re-run 3× under both XLA
configurations: four go green under the determinism flag (so: flaky, not regressions),
and the two extra on the post side are both
`test_compiled_dispatch_is_bit_identical`, whose implementation
(`jaccpot/upward/tree_geometry.py`) and test file are both untouched by Tier 1 —
`git diff 128a0e2 015c91c` on either path is empty. The fifth is real: see G.9.

The fp32 gap the G.10 handoff left open is closed with numbers rather than an
assertion. Residual `max|grad_f32 − grad_f64| / max|grad_f64|` over the transverse
components, worst over both signs of `z`: real fused **3.27e-07**, complex fused
**1.60e-06**, real *direct* control **1.14e-06**. **The fused lanes carry no float32
weakness the direct lanes do not** — the real fused lane is 3.5× better than its own
direct control. All are two to three orders inside the 3.4e-04 band and every case is
non-vacuous (the float64 reference exceeds 1e-3, so G.10's exact `(0, 0)` return could
not pass). Caveat on the constants in that test's docstring: rebuilding the real
coefficients with numpy instead of the test's `jax.random.normal` moved the real figure
by ~3× (8.9e-07 → 3.27e-07), so they are the right order but should not be read as
tight.

### B.4 The gate could not see an undefined name, and three defects used the gap

`.pre-commit-config.yaml` ran black, isort, pydoclint and pytest. **None of the four can
see a name that is used but never imported** — and neither can the test suite, because
every runtime module has `from __future__ import annotations`, so an unimported name
inside an annotation is stored as a lazy string and never evaluated. It is not a
deferred error; it is no error at all, and the annotation is simply unresolvable. E.2
("89 dangling forward references make the mixin annotations decorative") is the same
observation from the other side.

Three real defects reached `main` through the full green gate this way:

| defect | shape |
|---|---|
| G.1a `_accumulate_from_multipoles` | name does not exist anywhere |
| G.1b `evaluate_local_complex_with_grad_analytic` | exists in `complex_ops`, never imported |
| `MACTypeInput` in `fmm_sweeps.py` (PR #94) | exists in `config`, never imported |

The third is the sharpest datum, because it was introduced *during this audit's
execution*, in the PR whose subject was that the caller-facing and traversal-facing MAC
types must not be confused — reviewed, verified against the full block, and merged. A
gate that cannot see this class of error will keep admitting it.

**Closed in PR #98** with `flake8 --select=F821,F822`, scoped `files: ^jaccpot/`. Two
scoping decisions, both measured rather than assumed:

- **F811 (redefinition) was tried and dropped.** It fires on the legitimate "assign
  `None`, then conditionally `def` the same name" idiom at `fmm_sweeps.py:380`, so
  keeping it would mean annotating correct code to satisfy a linter. Every defect the
  hook exists for was an undefined name.
- **`examples/` and `bench/` are out of scope.**
  `examples/benchmark_gpu_radix_worker.py` alone carries 14 pre-existing F821s and
  `bench/bench_upward_sweep.py` one F811. CLAUDE.md exempts those trees and forbids
  reformatting files a change does not otherwise touch, so cleaning them is its own PR —
  **still open**, and the reason the hook is narrower than it looks.

No style codes are selected, so the hook cannot start disagreeing with black or isort.
Verified it actually fires by injecting an undefined name into `fmm_policy.py`.
Undefined names in `jaccpot/` go **4 → 2**, and both survivors are deliberate: `kernels/`
is a true leaf and must never name the engine, even under `TYPE_CHECKING` (§1).

---

## C. Public API inventory

All 14 names from `jaccpot/__init__.py:23-37`. "numpy sections" = the docstring has a
`Parameters` or `Attributes` section with a dashed underline. "test files" = distinct files
under `tests/` that reference the name. Renames are yours to decide; I only propose.

| Name | Defined | Doc lines | NumPy sections | Shaped jaxtyping | Test files | Note |
|---|---|---|---|---|---|---|
| `FastMultipoleMethod` | `solver.py:386` | 1 | no | n/a | 36 | The one public class. A **one-line** class docstring on a 19-parameter facade. Its *methods* are well documented (`:662, 784, 973, 1068` all have full numpy sections) — the class itself is not. Highest-value docstring gap in section C |
| `FMMPreset` | `config.py:35` | 1 | no | n/a | 11 | Enum; the four members are frozen by the API guard. One line is arguably enough, but the FAST/BALANCED/ACCURATE/LARGE_N_GPU *semantics* live only in ARCHITECTURE §6 |
| `FMMAdvancedConfig` | `config.py:301` | 1 | no | n/a | 16 | Container of the four config groups; one line for the aggregation point of ~60 knobs |
| `FarFieldConfig` | `config.py:56` | 1 | no | n/a | 4 | |
| `NearFieldConfig` | `config.py:75` | 1 | no | n/a | 3 | |
| `TreeConfig` | `config.py:44` | 1 | no | n/a | 4 | |
| `RuntimePolicyConfig` | `config.py:166` | 11 | no | n/a | 11 | Prose, no `Attributes` section |
| `TraversalOverrides` | `config.py:95` | 42 | **yes** | n/a | 3 | Exemplary — measurements in the docstring |
| `GradConfig` | `config.py:202` | 70 | **yes** | n/a | 2 | Exemplary. Only 2 test files though, and it is the supported interface for the grad path |
| `MemoryObjective` | `config.py:31` (`Literal` alias) | 0 | no | n/a | 1 | A bare `Literal` type alias with no docstring at all; `inspect.getsourcefile` cannot even locate it. Least-documented public name |
| `ComplexSHBasis` | `basis/complex_sh.py:14` | 4 | no | no | 3 | |
| `RealSHBasis` | `basis/real_sh.py:91` | 1 | no | no | 1 | One line, one test file, and it backs the **default** basis |
| `OdisseoFMMCoupler` | `odisseo.py:24` | 1 | no | no | 2 | *"Cache-oriented adapter for coupling ODISSEO and Jaccpot FMM."* This is the downstream integration contract; 59 statements at 78% coverage |
| `direct_sum_gravitational_acceleration` | `autodiff.py:13` | 15 | no | no | 4 | Good prose (it correctly disclaims being "the" differentiable path). No `Parameters` section; ARCHITECTURE §2 and §6 both re-explain it, which suggests the name is the problem, not the doc |

Observations, no decisions:

- **Documentation is inverted.** The two names added most recently (`GradConfig`,
  `TraversalOverrides`) are the two that are properly documented. The oldest and most-used
  (`FastMultipoleMethod`, `RealSHBasis`, the four config dataclasses) have one-line
  docstrings. F22 explains why pydoclint is silent about it.
- **Nothing on the public surface carries shape information.** `positions`/`masses` are
  `Array` everywhere.
- **Test breadth does not track importance**: `RealSHBasis` (the default basis) has 1
  referencing file; `MemoryObjective` has 1; `GradConfig` has 2.
- **Rename candidates, for your call only** (each would be a Tier 2 PR with an API-guard
  update): `direct_sum_gravitational_acceleration` → something that says "direct sum"
  (ARCHITECTURE §2 and §6 and its own docstring each spend a paragraph explaining it is *not*
  the differentiable FMM — three disclaimers is a naming smell); and the engine class in
  `_fmm_impl.py` sharing the name `FastMultipoleMethod` with the facade, which ARCHITECTURE §2
  already flags as confusing.

---

## D. Verification gaps

This is the section the rest of the plan depends on.

### D.1 There was no gradient golden — CLOSED (`e1f1455`)

`tests/characterization/` is one file, one test, 13 cases, **forward accelerations only**
(`test_fmm_golden.py:167`). Meanwhile:

- `NUMERICS_AND_JAX.md` §3: *"If you are refactoring something whose golden coverage is thin,
  extend the characterization suite first, in its own commit, before touching the code —
  including a gradient golden, not only a forward one."*
- `NUMERICS_AND_JAX.md` §6 item 2: *"`tests/characterization/` unmoved — **including gradient
  goldens**."* There were none to be unmoved.

There *are* good gradient tests (`tests/unit/test_gradient_correctness.py`,
`tests/unit/test_custom_vjp_parity.py` — 520 lines and genuinely adversarial,
`tests/integration/test_grad_fmm_vs_directsum.py`, `tests/unit/test_nearfield_fastlane_grad_path.py`
— 614 lines). But those are *correctness* tests with tolerances tied to FD or to a twin. A
golden is a different instrument: it catches a change in the *last three digits*, which is
what a refactor produces when it accidentally reassociates a sum. A 1e-3 FD comparison cannot
see that.

**What was added.** `tests/characterization/test_fmm_grad_golden.py` + 6 `.npz` under
`tests/characterization/golden_grad/`, mirroring the forward oracle's two-gate design:
inertness at `rtol=atol=1e-12`, plus a physics anchor against `jax.grad` of
`direct_sum_gravitational_acceleration` (the documented gradient oracle). Same
`JACCPOT_REGEN_GOLDEN=1` switch, so one command refreshes both.

Three things about it are worth carrying forward, because they were discovered while building
it rather than planned:

* **It demonstrably catches what the forward oracle cannot.** Scaling
  `_evaluate_local_real_with_grad_cvjp_bwd`'s output by `1 + 1e-6` — reverse-only, forward
  provably untouched — leaves the forward golden green (exit 0) and fails the gradient golden
  (exit 1). The 2e-3 anchor cannot see a 1e-6 perturbation either; only the 1e-12 inertness
  gate can. That is the whole argument for the tier, now measured.
* **The loss is `sum(w · a)` with a fixed cotangent, not `sum(a²)`.** `sum(a²)` weights each
  output component by its own forward value, so a component that is small in the forward pass
  contributes little to the gradient even when its reverse rule is wrong. Every existing
  gradient-correctness test in the repo uses `sum(a²)`.
* **`leaf_size = 4` is load-bearing** — see D.12, which is a new finding.

Grid: 6 cases (real/complex × order 2/4, uniform + clustered, N=128 and 256), all 9.7-24.3 s,
so all 6 plus the vacuity guard go in `slow_tests.txt` per the documented ≥8 s rule. That puts
them in `test-full`, which runs on every PR *and* runs `slow`, and keeps them out of the
3.11/3.12 smoke matrix, whose job is version compatibility and which has no timeout headroom.
Cost: ~2.1 min serial on `test-full`'s ~36 min against its 50 min cap.

Still open from the original plan: a **potential** golden, and the mode/preset axes in D.2.

### D.2 The forward golden is narrower than ARCHITECTURE §9 claims

Measured from `test_fmm_golden.py`: axes are distribution × N × basis × order. **Not**
covered, despite §9's *"farfield modes, outputs"*:

| Axis | Golden covers | Not covered |
|---|---|---|
| preset | `accurate` only (`:140`) | `fast`, `balanced`, `large_n_gpu` |
| output | accelerations (`:167`) | potentials, jerk / derivative towers |
| nearfield mode | whatever `accurate` resolves to | `bucketed`, `fast_lane` |
| farfield mode | resolver default | `pair_grouped`, `class_major` |
| M2L implementation | pure-JAX | fused Pallas (GPU-only, understandable) |
| lane | full `compute_accelerations` | strict/refresh (`strict_run_v2`), large-N |
| basis | real, solidfmm, cartesian | — (good) |

The `class_major` / `pair_grouped` gap is the one I would close first after D.1: those are
the paths `_accumulate_solidfmm_m2l_grouped_class_major` (`kernels/core.py:1335`) and friends,
which the M2L split in F/Tier-1 touches directly.

### D.3 Numerically load-bearing code with no test that could catch a *wrong answer*

Fresh measured coverage, worst first. I have separated "structurally unreachable on CPU" from
"reachable and untested", because they need different remedies.

**Structurally unreachable in the default (CPU) suite:**

| Module | Coverage | Gate |
|---|---|---|
| `runtime/_large_n_grad.py` | **0%** (72 stmts) | `can_use_large_n_prepare_path` requires `jax.default_backend() == "gpu"` (`_large_n_pipeline.py:2013`) |
| `runtime/_large_n_farfield.py` | **0%** (8) | same |
| `distributed/cap_presets.py` | **0%** (40) | ≥2 devices |
| `distributed/fmm.py` | **19%** (423 missed) | ≥2 devices |
| `experimental/treecode_far_near.py` | 0% | reachable from production only via `local_walk="treecode"` at ≥2 devices (F35) |

Remedy is a **nightly GPU leg**, not a CPU test. Until it exists, `_large_n_*` and
`distributed/` are off-limits for this refactor — which is why A.9 argues to leave
`_large_n_pipeline.py` whole.

**Reachable on CPU and still substantially untested:**

| Module | Coverage | The unexercised part |
|---|---|---|
| `runtime/fmm_autotune.py` | 15% | `autotune_m2l_chunk` defaults False; nothing turns it on |
| `runtime/fmm_strict_cap_profile.py` | 35% | compiled-profile persistence |
| `runtime/fmm_strict_run.py` | **55%** (227 missed) | most of `strict_run_v2` (428 lines, `:389`) and `_refresh_large_n_same_topology` (515 lines, `:890`) |
| `operators/complex_harmonics.py` | 61% | |
| `runtime/kernels/core.py` | 69% | includes both F01/F02 branches, `:1178-1247` (grouped rotation blocks), `:1273-1401` (class-major accumulate), `:2405-2466`, `:3185-3273` |
| `runtime/_interaction_cache.py` | 70% | |
| `runtime/_large_n_pipeline.py` | 71% | |
| `runtime/fmm_derivatives.py` | 74% | `:569-667` is one contiguous unexercised block |
| `nearfield/near_field.py` | 79% | 292 statements missed |

`fmm_strict_run.py` at 55% is the one that should worry you most of these: it is the per-step
science path and it owns the velocity-Verlet update.

### D.4 Stated path equivalences — which are asserted, and which are not

NUMERICS §1 says *"each pair has an asserted numerical equivalence."* Audited pair by pair:

| Claimed equivalence | Where claimed | Asserted? |
|---|---|---|
| fused complex M2L Pallas ≡ solidfmm reference | `kernels/core.py:1636` | **Yes**, on CPU. `tests/test_m2l_complex_fused_pallas.py` via `interpret=True`, orders 2-5 |
| fused real M2L Pallas ≡ rot-scale reference | `kernels/core.py:1691` | **Yes**, on CPU. `tests/test_m2l_real_fused_pallas.py`, `interpret=True`, orders 2-4, fp32+fp64 — which is exactly what makes F04 wrong |
| fused M2L `custom_vjp` reverse ≡ autodiff of twin | `pallas/m2l_{real,complex}_fused.py` | **Yes.** `tests/unit/test_custom_vjp_parity.py:223, 289`, `interpret` parametrised |
| Pallas z-M2L ≡ pure-JAX recurrence | `pallas/m2l_core_z_real.py` | **Yes**, `tests/unit/operators/test_pallas_m2l_core_z_real.py`, orders 1,2,4,6 |
| Pallas treecode walk ≡ pure-JAX walk, bit-identical | `pallas/treecode_walk_pallas.py:16` | **Yes**, `tests/unit/operators/test_pallas_treecode_walk.py` |
| real basis ≡ solidfmm basis | `solver.py:161-170` | **Yes**, and the docstring is honest about the tolerance being slacker than round-off (3e-2 at fp32, N=96) — worth re-measuring, per its own note |
| `"complex"` ≡ `"solidfmm"` alias | `solver.py:154-160` | **Yes**, structurally (`test_basis_and_runner_hygiene.py`) — correctly identified as the only way to pin an alias |
| bucketed ≡ baseline nearfield | `near_field.py:2855` | **Yes**, and better than the docstring says (F05) |
| complex→real conversion ≡ `p2m_real_direct` | `real_harmonics.py` | **Yes** (`7b85028`), orders 0-6 (F07) |
| `static_num_levels` concrete ≡ padded, bit-identical, **complex** sweep | `solidfmm_complex_tree_expansions.py:486` | **Yes**, `test_solidfmm_complex_tree_expansions.py:68` |
| `static_num_levels` concrete ≡ padded, **real** sweep | not claimed and not documented (`real_tree_expansions.py:250`) | **No** — F24 / D.9 |
| `nearfield_leafpair_pallas_decoupled` (same array both sides) ≡ `nearfield_leafpair_pallas`, bit-for-bit | `pallas/nearfield_fused_leaf.py:667` | **No** — F25. Grep for "decoupled" under `tests/` returns nothing |
| `JACCPOT_FUSED_M2L_VJP=0` fallback ≡ fused VJP | `pallas/m2l_real_fused.py:48` | **No** — F26, the branch never runs |
| `_multipole_align_to_z_block` ≡ the physical rotation of P2M | `real_harmonics.py:1520` | **Yes** as of `e5d8e41` — and the claim holds at 7.6e-15. Was: no |
| closed-form real rotations ≡ Wigner-D reference | `real_harmonics.py:1143-1146` | **No, and not assertable as-is** — the reference raises `ImportError` (F39). Superseded: the physics identities in `e5d8e41` check the same builders without it, which is a better reference anyway (physics, not a second implementation) |
| forward bit-identical on `large_n_gpu` | `fmm_evaluate.py:767`, `_large_n_pipeline.py:1599` | **Not in CI** (GPU-only, 0% coverage). Measurement is quoted; nothing re-checks it |
| distributed forward bit-identical / survives a gradient | `distributed/fmm.py:675` | **Not in CI** (≥2 devices). `test_forward_survives_a_gradient` exists and never runs here |

### D.5 The rotation builders were checked only against themselves — CLOSED (`e5d8e41`)

Four tests touch `real_rotation_to_z_axis_multipole` and its siblings, and **all four are
self-consistency checks**:

- `test_real_harmonics.py:331` — z-aligned input gives identity. A wrong azimuth convention
  also gives identity when `rho == 0`.
- `:351` — `ell=0` gives 1. True for any normalisation.
- `:984` `test_rotation_to_from_z_axis_are_inverses` — `D_from @ D_to == I`. An **involution
  property**: satisfied by any consistently-wrong pair.
- `:1008` `test_alignment_pipeline_steps_match_p2m` — checks the `B` and `Dz` *building
  blocks* against `p2m_real_direct` step by step, using its **own** `alpha_z = arctan2(y, x)`
  (`:1029`). The production block uses `az = arctan2(x, y)` — deliberately, per the CRITICAL
  note at `real_harmonics.py:1524-1531`. So this test never constructs
  `_multipole_align_to_z_block` and does not exercise the convention the docstring calls
  critical.

Meanwhile `_real_wigner_rotation` (`:1132`) is an independent 30-digit SymPy/NumPy reference
whose docstring says *"The closed-form production builders are
`real_rotation_to_z_axis_multipole` and friends; this exists to check them"* — and nothing
ever compares the two (F32).

**Honest calibration, because this matters for how you prioritise it.** A wrong azimuth here
*would* be caught — by `test_m2l_rotated_matches_alignment_pipeline` (`:859`),
`test_m2l_rotated_error_improves_with_order` (`:904`),
`test_full_rotated_pipeline_m2m_m2l_l2l_converges` (`:949`), and the real-basis goldens
`clu_real_n256_p4` / `uni_real_n256_p6`, all of which anchor the assembled off-axis far field
against a direct sum with convergence in `p`. So this is a **localisation** gap, not an
undetected-wrong-answer gap: the failure surfaces four layers downstream as "the real basis
does not converge", with nothing to say which of six operators is at fault. Given that this
exact bug class has already cost this project two debugging cycles, closing it is cheap
insurance, not an emergency.

**What actually closed it, and why not as planned.** The plan was to compare the production
builders against `_real_wigner_rotation`. That is impossible: the Wigner path raises
`ImportError` because `sympy` is undeclared (F39). So the tests assert **physics identities**
instead, which is strictly better — they check the builders against what the builders are *for*,
not against a second implementation that could share a convention error:

* multipole: `D_to @ p2m(s)[block] == p2m(g @ s)[block]` and `D_from` inverts it. `g` is
  rebuilt from first principles in NumPy, and `g @ direction == (0, 0, r)` is asserted inline
  so a wrong `g` cannot make the test vacuous.
* local: `evaluate_local_real(D_to @ L, g @ t) == evaluate_local_real(L, t)` — the potential is
  frame-invariant. This is what distinguishes the transpose convention from its inverse; the
  involution test cannot.

**Results.** All four builders satisfy their identities at **~2.5e-15 worst case** (~10
eps_f64) across 4 generic off-axis directions × 3 draws × `ell = 0..6`. So the production code
was **correct**, and the `atan2(x, y)` convention flagged CRITICAL at `:1524` is now a tested
fact.

**Mutation check, which is the part that justifies the test existing.** Writing the azimuth as
`atan2(y, x)` at both sites: **8 of 8 new tests fail** (the identity goes to 1.8e+00) and
**4 of 4 pre-existing rotation tests pass**. The real-basis goldens do fail, confirming the
original calibration — this was a localisation gap, not an undetected wrong answer. Tolerance
1e-12: ~400× headroom over round-off, and still sharp enough to catch a 2e-13 *relative*
azimuth error (a 1e-12 relative perturbation moves the identity to 5.1e-12).

**What this leaves for F32/G.4.** The reference family's stated purpose ("this exists to check
them") is now served by something else, so the case for keeping it is weaker than the audit
originally argued, not stronger. See G.4 as revised.

### D.6 The rotation azimuth's degenerate subgradient is unasserted — and, measured, **wrong** (→ G.10)

`real_harmonics.py:1533-1540` and `:1563-1570` (the duplicated comment, F30) claim: at
`rho == 0` *"the azimuth is undefined there and the rotation is a pure polar turn / identity,
so a zero cotangent is the correct subgradient."*

That is a differentiability claim on a NaN guard, and it is exactly the class of defect this
file has already been bitten by twice:

- `d5cb13b fix(real): azimuth degeneracy branch silently zeroed the transverse gradient` — the
  L2P case.
- `real_harmonics.py:404-410`, which says the P2M *"has the identical structure and the
  identical defect."*

Both of those now have tests: `test_evaluate_local_real_grad_at_rho_zero_matches_limit`
(`:212`, parametrised over degenerate deltas × orders 1,2,4,6) and
`test_p2m_real_direct_jacobian_at_rho_zero_matches_limit` (`:268`). Commit `09982df` even
says *"assert the degenerate azimuth subgradient, not just finiteness."*

The rotation path got the same hardening (`:1526-1545`) and **no** test. Grep for
`_rotation_to_z_angles` in `tests/` returns nothing; the only degenerate-geometry parameter in
that file targeting `rho == 0` is `_CONVERSION_DELTAS` (`:1183`), which exercises the
conversion, not the rotation.

**What happened when I tried to close it (Tier 0.4).** The planned test — "assert the reverse at
`rho == 0` matches the off-axis limit" — **refutes** the claim rather than confirming it. Full
measurement in **G.10**: the limit exists, is direction-independent, and is not zero, so the
guard silently discards two real gradient components. Tier 0.4 is therefore blocked behind a bug
fix and is no longer a characterization item.

One nuance worth keeping, because it is presumably why this survived review: the raw alignment
*block* is genuinely **discontinuous** at `rho == 0` — a jump of 2.0 across it along ±x, since
`az = atan2(x, y)` does not converge — so a zero cotangent for the block *in isolation* is
defensible. What the comment misses is that the assembled `rotate → z-translate → rotate-back`
composition **is** continuous and differentiable there (a z-translation commutes with a
z-rotation, so the jump cancels), and it is the composition whose gradient the FMM takes.

### D.7 Docstring accuracy and staticness claims that nothing pins

Beyond D.4's equivalences, the claims I could not find backing for:

| Claim | Where | Status |
|---|---|---|
| *"1M particles, forward 2.5 s and forward+backward 69 s at 11 GB peak"* | ARCHITECTURE §6 | GPU measurement, no guard. `bench/guard_large_n_radix_fast_lane.py` exists but is not in the CPU CI leg |
| *"at N=200000 the bucketed reverse OOMs (30 GB peak) and the fast lane completes in 6.8 GB"* — the reason `nearfield_lane="auto"` crosses over at N≥100000 | ARCHITECTURE §6, `config.py:226`, `solver.py:997` | The *crossover policy* is tested (`tests/unit/test_grad_config.py`); the *memory measurement* justifying it is not, and cannot be on CPU |
| *"~6e-04 regardless of expansion order"* fp32 TF32 M2L floor, and the `HIGHEST` pinning | ARCHITECTURE §7, `operators/_precision.py` | **Backed** — `tests/unit/operators/test_matmul_precision_pinned.py` (127 lines). Good |
| *"the reverse costs roughly 6-11x the forward at N=512-8192"* | ARCHITECTURE §6 | Distributed, ≥2 devices, unguarded |
| `max_leaf_size` *"Required under `jit`"*, raising `ValueError` under a tracer | `near_field.py:2829-2836` | Reachable; I did not verify a test exists for the re-raise. **Evidence needed:** grep `tests/` for that `ValueError` message |
| `softening` *"must be a concrete Python float, not a tracer"* | `near_field.py:2766` | Same — a staticness contract with no test I could find |

The last two are the pattern worth generalising: **staticness contracts are documented and
almost never tested.** `_env.py` aside, I found no test that asserts "passing a tracer here
raises". Those are cheap tests and they are the ones that break loudly when a refactor
accidentally traces a value that was static (NUMERICS §1: *"can leave runtime untouched and
triple compilation"*).

### D.8 `p2m_real_direct` / Dehnen Table 3 — reassessed

You flagged `real_harmonics.py:377` as *"the most serious test gap in the repo."* I do not
think it is, and here is why, precisely.

**What the docstring says** (`:371-382`): degree 1 only is verified; degrees 2-6 rest on
`scripts/derive_table3_polynomials.py`, which was never committed (confirmed — no `scripts/`
directory exists, in HEAD or in `git ls-files`); *"the coverage that exists cannot see a
per-`m` normalisation or sign error above degree 1."*

**What actually exists.** `tests/unit/operators/test_real_harmonics.py:1193`
`test_complex_to_dehnen_real_matches_p2m_real_direct`, landed in `7b85028` (the same PR
tranche that wrote this docstring), asserts

```
complex_to_dehnen_real_coeffs(complex_R_solidfmm(d, order=p), order=p) == p2m_real_direct(d, 1.0, order=p)
```

at **orders 0, 1, 2, 3, 4, 6** across 6 geometries (generic off-axis, z-aligned/`rho == 0`,
x-aligned, y-aligned, xy-plane, near-origin) to 1e-13 relative L2, measured worst case 8.9e-16.
The axis-aligned cases are chosen because, in the test's own words, *"the axis cases are where
a per-m sign or normalisation error shows up most cleanly."*

A per-`m` sign or normalisation error in `p2m_real_direct` at degree ≥ 2 **would** fail that
test unless it were exactly cancelled by a matching error in the `Q` matrix
(`build_Q_dehnen_no_sqrt2`) or in `complex_R_solidfmm`. So the docstring's central claim is
now false.

**The residual gap is real, and it is a different gap.** The cross-check is *relative*: it
pins `p2m_real_direct` against `Q ∘ complex_R_solidfmm`. If `Q` and `p2m_real_direct` were both
derived from the same misreading of the Dehnen convention, the error cancels and both tests
pass. Nothing in the repo compares either against Dehnen (2014) Table 3 **as literal
polynomials** above degree 1. What *does* anchor absolutely, at the assembled level, is the
convergence-in-`p` suite and the real-basis goldens (`uni_real_n256_p2/p4/p6`,
`uni_real_n1024_p4`, `clu_real_n256_p4`) against a direct O(N²) sum — a shared convention
error that survived those would have to be a similarity transform of the true basis that also
preserves the assembled potential, which is a much narrower class than "a per-`m` sign error".

**Closed in `8bd24a6`, and not the way this section proposed.** Item 2 below said to
transcribe Table 3 for degrees 2-6 as committed data. I did not: a transcription of a
paper I do not have is not checkable, and re-deriving the polynomials from the formula
`p2m_real_direct` implements would be circular. The anchor instead comes from theory —
the 1/(n+|m|)!-normalised solid harmonics satisfy `dU_n^m/dz = U_{n-1}^m` (exact,
verified to order 6) and transverse recurrences whose coefficients are exact multiples of
1/2, and those recurrences plus `U_0^0 = 1` determine every `U_n^m` **uniquely**. So it is
an absolute anchor, and a stronger one than a table: scaling any single `U_n^m` by
`1 + 1e-3` leaves the recurrences structural (held-out residual unchanged at ~8e-15) but
makes the coefficients non-half-integer in 6 entries for `U_3^2`, 6 for `U_4^{-3}` and 2
for `U_6^5`. The half-integrality check is what catches the per-`m` normalisation error
this section is about.

**What was proposed (retained for the record):**
1. **Correct the docstring** (`:371-382`) to name the test that now exists and to state the
   residual gap as *"no absolute Table 3 anchor above degree 1"* rather than *"unverified"*.
   This is a real finding — the docstring is now the misleading artefact.
2. **Commit the missing anchor as data, not a script.** Add the Table 3 scaled polynomials for
   `n = 2..6` as a literal table in the test module (a dict of `(n, m) -> callable(x, y, z)`,
   ~30 entries), and assert `p2m_real_direct` matches at several generic points. That closes
   the absolute gap permanently and does not depend on a derivation script existing. Deriving
   the entries is the only real work; validate them against the existing relative test, which
   must then agree with both.
3. Do **not** resurrect `scripts/derive_table3_polynomials.py`. A generator that is not run in
   CI has the same failure mode all over again.

### D.9 `static_num_levels` — a documented, tested optimisation whose twin has neither

`upward/solidfmm_complex_tree_expansions.py:480-497` explains `static_num_levels` in detail,
claims *"recompile-free under the fixed-shape contract and bit-identical to the padded
result"*, warns about the `dynamic_slice_in_dim` clamp overrunning at the deepest levels, and
is pinned by `tests/unit/core/test_solidfmm_complex_tree_expansions.py:68`
`test_static_num_levels_bit_identical_to_padded`.

`upward/real_tree_expansions.py:242-257` has the same parameter (`:250`), the same
`max(int(static_num_levels), 1)` handling (`:272`), a **three-line docstring** that does not
mention it, and no test. Both are called from `runtime/fmm_sweeps.py:198, 242`.

Fix: copy the test (Tier 0) and the docstring (Tier 0). Not a code change.

### D.10 pydoclint's exemption is the reason the docstring layer looks finished

Recorded here rather than only in F22 because it changes what "the package is at 0 pydoclint
violations" licenses you to conclude:

```
pydoclint --config pyproject.toml jaccpot/                                → 0 violations
pydoclint --config pyproject.toml --skip-checking-short-docstrings=False jaccpot/  → 2840
```

Breakdown: 584 × DOC101 (fewer documented args than signature), 584 × DOC103, 600 × DOC201
(no return section), 584 × DOC203, **161 × DOC501/DOC503** (an exception is raised and not
documented, or vice versa), 62 × DOC601/DOC603 (class attributes), 28 × DOC107, 14 × DOC106.

The 161 DOC501/503 are the semantically important ones: STYLE_GUIDE §9 makes loud, named
validation a policy, and *"an explicit request may not be quietly overridden"* is enforced by
`raise`. An undocumented `raise` is an undocumented part of the contract.

**Do not turn the flag off in one commit** — 2840 violations is not a PR. The useful subset is
the 73 functions with ≥8 parameters in the numerics directories (F23), and the 161
`Raises` omissions, which are independently addressable.

### D.11 What characterization must exist before section F's Tier 1 is safe

Concretely, gated on the specific Tier-1 items:

| Tier 1 item | Blocking characterization |
|---|---|
| F12 dedupe the two near-field edge-list kernels | The bucketed-vs-baseline test (now good, F05) covers the *modes*, but both kernels feed both modes. Need: a golden case with `nearfield_mode="bucketed"` **and** one exercising the `precomputed_*` path, plus a compile-time measurement (A.10) |
| Split `kernels/core.py` M2L seam | A golden case per `farfield_mode` (`pair_grouped`, `class_major`) — currently zero (D.2). Plus the ARCHITECTURE §10 bit-identity check on a fixed input grid at `rtol=0` |
| Split `near_field.py` fast-lane seam | The gradient golden (D.1) with `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`, since that seam contains the production `custom_vjp` |
| Split `real_harmonics.py` | ~~The rotation-vs-Wigner test (D.5)~~ **done** (`e5d8e41`, as physics identities). **D.6 (the degenerate subgradient) is now the only remaining blocker** — the split moves the rotation builders across a file boundary and D.6 is the untested half of what they promise |
| Extract from `_prepare_state_dual_and_downward` (F10) | A golden over the strict/refresh lane, or an explicit decision to leave `fmm_strict_run.py` alone (55%, F33) |
| Anything in `_large_n_*` or `distributed/` | A nightly GPU leg. **Not achievable in this pass** — hence A.9's "leave it whole" |

### D.12 New finding: a gradient test at the default leaf size covers no far field

Discovered while building the gradient golden, and it generalises well beyond it.

For the golden's 128-particle systems at `theta=0.5`, measured M2L pair counts:

| `leaf_size` | accepted M2L pairs |
|---|---|
| 4 | 110 |
| 8 | **0** |
| 16 | **0** |
| 32 | **0** |

At leaf ≥ 8 the tree is too shallow for the MAC to accept any box pair, so the entire
far-field reverse path (M2M → M2L → L2L) is never traced. A gradient test written at a default
leaf size therefore exercises **only the near field** — and passes, silently, looking like
far-field coverage.

This is not hypothetical: `tests/unit/test_gradient_correctness.py` already encodes it, with a
`min_far` parameter and an explicit `assert n_far >= min_far`, and its first parametrised case
is deliberately marked `min_far=0` as a cheap near-field-only point. So the trap is known in one
module and nowhere else.

Two consequences worth acting on:

1. The gradient golden asserts its own non-vacuity per case, plus
   `test_grad_golden_leaf_size_is_what_makes_the_far_field_nonempty`, which goes red if a
   shallow tree ever *starts* accepting pairs (at which point `LEAF_SIZE = 4` can be revisited).
2. **Any future gradient test needs the same guard.** Worth a line in
   `NUMERICS_AND_JAX.md` §3 alongside the existing edge-case list, since "the test passed but
   covered nothing" is exactly the failure that section exists to prevent. Candidate for Tier 0.

#### D.12 recurred twice more, and the second time it hid a fake tolerance

Two further instances, both found by accident rather than by looking:

**In the Tier 1 GPU handoff** (B.3): its prescribed residual-audit config, `N=4096
leaf=64`, reports `far_pairs=0`. The document that warns about D.12 in its own "traps"
section then specified a config that measures the M2L coefficient not at all. Re-run at
`--leaf 4`: 195,336 far pairs.

**In the source-motion downward tests** —
`tests/unit/core/test_solidfmm_complex_tree_expansions.py`, found while writing the G.1a
regression test. Its six-particle fixture accepts **zero** far pairs at `theta=0.6`, so
`test_solidfmm_downward_source_motion_locals_match_finite_difference` and its
second-derivative sibling were comparing an all-zero result against an all-zero
reference:

```python
assert np.allclose(got, ref, rtol=3e-5, atol=1e-7)   # zero vs zero
```

Neither could have detected any error in the source-motion M2L/L2L path. That path is
the jerk/time-derivative far field, i.e. D.3 territory: numerically load-bearing code
with no test that could catch a wrong answer. Fixed at `theta=0.8` (4 far pairs,
`|ref| = 1.93`, agreement `1.08e-09`), with a shared helper asserting **both** the
accepted pair count and the reference magnitude.

**The lesson beyond D.12.** Making the second test non-vacuous made it *fail*, at
`rel = 3.0e-3` against a `2e-3` bound — and the bound was not what was wrong. A central
second difference carries `eps / dt**2` roundoff, so its `dt = 1e-5` was the inaccurate
side of the comparison:

| `dt` | rel |
|---|---|
| 1e-6 | 4.0e-01 |
| **1e-5** | **3.0e-03** ← the shipped setting |
| 1e-4 | 3.9e-05 |
| **1e-3** | **2.5e-07** ← chosen, near the `eps**0.25` optimum |
| 1e-2 | 8.7e-09 |

Fixing the *reference* let the bound be **tightened 200× to 1e-5** rather than relaxed.

So: **a tolerance calibrated inside a vacuous regime is not a tolerance.** `2e-3` looked
like a considered numerical bound and was in fact an arbitrary number that had never
been compared against anything but zero. Any time a vacuity guard is added to an
existing test, its tolerance has to be re-derived, not inherited — and a guard that
turns a passing test red is doing its job.

---

## E. Typing debt

### E.1 `jaxtyping` is present and inert

Measured over all non-experimental modules by AST walk:

| | count |
|---|---|
| annotated parameters mentioning `Array` **with** shape/dtype (`Float[Array, "n 3"]` etc.) | **0** |
| annotated parameters mentioning `Array` **without** shape/dtype (bare `Array`) | **1534** |
| parameters with no annotation at all | **350** |
| `from jaxtyping import` statements | 47 (30 × `Array`, 10 × `Array, jaxtyped`, …) |
| `@jaxtyped(typechecker=beartype)` decorators | 45, across 11 modules |

`jaxtyping.Array` is an alias for `jax.Array`. So `_typecheck.py:26-27`'s *"annotated callables
are checked by beartype with jaxtyping's shape/dtype semantics"* is true only vacuously: there
are no shapes or dtypes in any annotation to check. STYLE_GUIDE §4's *"Keep axis names
consistent so `jaxtyping` can cross-check within a signature"* describes a convention that has
never been used.

Weighted by how load-bearing the shape is, the modules where bare `Array` costs the most:

| Module | bare `Array` | unannotated | Why it matters |
|---|---|---|---|
| `nearfield/near_field.py` | 317 | 81 | Every kernel juggles `[N,3]` particle arrays, `[num_leaves, max_leaf_size]` leaf tables, `[num_edges]` edge vectors and `[chunk_count, chunk_flat_size]` schedules. The `precomputed_*` contract is *shape-encoded* by design (`:2868-2878`) and none of those shapes is in a signature |
| `runtime/kernels/core.py` | 228 | 9 | `[N, (p+1)^2]` coefficient blocks vs `[num_pairs, (p+1)^2]` vs `[total_nodes, (p+1)^2]` — three different leading axes, one annotation |
| `operators/complex_ops.py` | 105 | 0 | complex vs real packing, `[(p+1)^2]` vs `[(p+1)(p+2)/2]` |
| `operators/real_harmonics.py` | 86 | 0 | `[2*ell+1, 2*ell+1]` rotation blocks vs `[(p+1)^2]` packed coefficients — a transpose bug here is in the memory record |
| `downward/local_expansions.py` | 63 | 0 | |
| `pallas/nearfield_fused_leaf.py` | 58 | **65** | `[leaves, W, 3]` / `[num_targets, S]` tile geometry, largely unannotated |
| `pallas/m2l_complex_fused.py` | 26 | **74** | The highest unannotated count in the package; Pallas kernel bodies |
| `pallas/m2l_real_fused.py` | 41 | 26 | |

The 350 unannotated parameters are all on private functions:
`tests/unit/test_type_annotation_guard.py:41` skips names starting with `_`, so the guard is
satisfied while the numerically densest code is unannotated.

### E.2 89 dangling forward references make the mixin annotations decorative

No `TYPE_CHECKING` block exists anywhere under `runtime/`. Result (pyflakes):

| Undefined name | Occurrences | Where |
|---|---|---|
| `FastMultipoleMethod` | 63 | every mixin's `self:` annotation — `fmm_evaluate.py`, `fmm_policy.py`, `fmm_strict_run.py`, `fmm_prepare.py`, `fmm_derivatives.py`, `fmm_sweeps.py`, `fmm_overrides.py`, `fmm_diagnostics.py`, `fmm_autotune.py`, `fmm_strict_cap_profile.py` |
| `PreparedStateLike` | 26 | `fmm_strict_run.py`, `fmm_overrides.py` |
| `LargeNGradPlan` | 2 | `fmm_evaluate.py:624, 931` |
| `NearfieldInteropData` | 1 | `fmm_state.py:488` |

Reproduced:

```python
import typing
from jaccpot.runtime import fmm_evaluate
typing.get_type_hints(fmm_evaluate.EvaluateMixin.differentiable_accelerations)
# NameError: name 'FastMultipoleMethod' is not defined
```

Under `from __future__ import annotations` this is harmless at runtime (beartype tolerates it;
the suite passes with `JACCPOT_RUNTIME_TYPECHECK=1`), so it is debt, not a bug.

**Closed in `ad7b00c`, and two things I wrote above are wrong — corrected here rather than
edited away, because both mistakes are instructive.**

1. *"`typing.get_type_hints` raises `NameError`"* was presented as the thing a fix would
   repair. It is true, and a `TYPE_CHECKING` fix does **not** repair it: measured before and
   after, 50 resolvable / 63 not, unchanged. `TYPE_CHECKING` names are absent from module
   globals at runtime by construction, so `get_type_hints` cannot see them. The lesson is that
   I picked the demonstration that was easy to run (`get_type_hints`) rather than the one that
   matched the claim (a static checker), and then justified the work by it.
2. *"`PreparedStateLike` (currently a name with no definition anywhere)"* — it is defined, at
   `runtime/_fmm_impl.py:305`, as `Union["FMMPreparedState", LargeNPreparedState]`. The mixins
   simply never imported it.

The measured benefit is `pyflakes jaccpot/` undefined names **95 → 5**: pyflakes becomes usable
as a bug-finder here, which matters concretely because the two genuine `NameError`s in G.1 were
found only by filtering them out of 397 lines of noise. `kernels/core.py`'s two engine
annotations stay dangling on purpose — see the comment now at the site.

### E.3 Recommended annotation policy, if you want one

A full `Float[Array, "..."]` conversion of 1534 sites is not a refactor pass; it is a project.
A cheaper policy that gets most of the value:

1. Shape-annotate the **public surface** and the **operator/kernel boundaries** only — the
   ~40 functions in section C plus `_apply_m2l`, `_accumulate_m2l_*`, the P2M/L2P entry points
   and `compute_leaf_p2p_accelerations*`. Consistent axis names (`n` particles, `p2` for
   `(p+1)^2`, `pairs`, `leaves`, `w` leaf width) make jaxtyping cross-check within each
   signature, which is where it actually catches things.
2. Everything else: leave bare `Array`, and put the shape in the docstring (which STYLE_GUIDE
   §3 already mandates and F23 says is missing anyway) — one convention, two places, rather
   than three half-conventions.
3. Extend `test_type_annotation_guard.py` to cover private functions **in the operator and
   kernel directories only**, so the 350 gap cannot grow where it matters.

---

## F. Work plan

Ordered. Every item is independently reviewable and revertible. "Verified by" is what makes it
mergeable — in addition to the standard block (`black --check`, `isort --check-only`,
`pre-commit run --all-files`, `JAX_ENABLE_X64=1 pytest -q`, `pytest -q tests/characterization`).

### Tier 0 — provably no behaviour change

Characterization first, because everything after it depends on the net existing. **0.1 and
0.3 are done** (`e1f1455`, `e5d8e41`); 0.3's ordering was wrong in the original plan and is
corrected below.

| # | Item | Touches | Closes | Verified by |
|---|---|---|---|---|
| ~~**0.1**~~ **DONE `e1f1455`** | **Gradient golden.** `tests/characterization/test_fmm_grad_golden.py` + **6** `.npz` (not 8 — the reverse compile costs ~1.1-1.5 GB per case, so the grid was trimmed and `_DIFF_FMM_TEST_FILES` in `tests/conftest.py` extended). Two gates as specified | `tests/characterization/`, `tests/conftest.py`, `tests/slow_tests.txt` | F03, D.1 | Done: full suite 839 passed / 57 skipped; forward golden unmoved; reverse-only mutation caught (D.1). **G.9 risk now RESOLVED** (per-particle inertness gate; see G.9) |
| **0.2** | **Widen the forward golden**: add a `farfield_mode` axis (`pair_grouped`, `class_major`), a `nearfield_mode` axis (`bucketed`), and a potentials output. ~8 new cases | `tests/characterization/` only | D.2, D.11 | New goldens generated with `JACCPOT_REGEN_GOLDEN=1`, each anchored to the direct sum before committing |
| ~~**0.3**~~ **DONE `e5d8e41`** — and it should have been item 0.1, since it was the only item whose *outcome* could invalidate the plan | **Rotation blocks vs. an independent reference.** ~~vs `_real_wigner_rotation`~~ — impossible, that path raises `ImportError` (F39). Implemented instead as **physics identities**: `D_to @ p2m(s) == p2m(g @ s)` for multipoles, and frame-invariance of the evaluated potential for locals | `tests/unit/operators/test_real_harmonics.py` | F29, D.5 (**not** F32 — the family stays unused, see G.4) | Green: production builders correct to ~2.5e-15. Mutation: azimuth swap fails 8/8 new, passes 4/4 pre-existing |
| **0.4** | ~~Degenerate rotation subgradient~~ **BLOCKED — became a bug report (G.10).** Writing the test refuted the claim it was meant to pin. Re-plan as: bug-fix PR first (G.10), *then* this test as its regression guard | `tests/unit/operators/test_real_harmonics.py` | F31, D.6 → **G.10** | Cannot be green against current behaviour. Do not land a test asserting the present (wrong) values |
| **0.5** | **Dehnen Table 3, degrees 2-6, as committed data.** Literal polynomial table in the test module; assert `p2m_real_direct` matches at several generic points | `tests/unit/operators/test_real_harmonics.py` | D.8 item 2 | Green **and** consistent with `test_complex_to_dehnen_real_matches_p2m_real_direct`; disagreement between the two is section G |
| **0.6** | `static_num_levels` bit-identity test for the **real** upward sweep, copied from the complex one | `tests/unit/core/` | D.9 | Green |
| **0.7** | `nearfield_leafpair_pallas_decoupled` same-array-both-sides parity test, `interpret=True` | `tests/unit/operators/test_pallas_nearfield_fused.py` | F25 | Green on CPU |
| **0.8** | `JACCPOT_FUSED_M2L_VJP=0` fallback parity test (monkeypatch the env, `interpret=True`) | `tests/unit/test_custom_vjp_parity.py` | F26 | Green; both branches now exercised |
| **0.9** | Staticness-contract tests: passing a tracer for `softening` / omitting `max_leaf_size` under `jit` raises the documented error | `tests/unit/core/test_near_field.py` | D.7 last two rows | Green |
| **0.10** | **Fix the coverage omit.** Delete `jaccpot/pallas/m2l_real_fused.py` from `[tool.coverage.run] omit`, replacing the comment with the measurement (92% from CPU tests) | `pyproject.toml` | F04 | `pytest --cov=jaccpot --cov-report=term-missing` shows the module at ~92%, TOTAL rises |
| **0.11** | **Docstring corrections.** `real_harmonics.py:371-382` (F07/D.8 item 1), `near_field.py:2862-2867` (F05), `fmm_evaluate.py:1117` (F06), `real_tree_expansions.py:242-257` (F24), plus a one-line note at `real_harmonics.py:1267` that the unused `R` entries are deliberate narration (F19) | docstrings/comments only | F05, F06, F07, F19, F24 | `pre-commit run --all-files`; diff contains no code lines |
| **0.12** | **ARCHITECTURE.md corrections.** §2/§9 twelve→fourteen names + add `GradConfig`, `TraversalOverrides`; §9 golden-grid axes; §4 drop "thin coordinator" or qualify it with the 722-line constructor | `ARCHITECTURE.md` | A.8 | Cross-checked against `test_public_api_surface.py:25-39` and `test_fmm_golden.py:76-90` |
| **0.13** | **STYLE_GUIDE §8 corrections.** Upward-import inventory 4/5 → 7/11 with the three additions and their classification | `agent_guides/STYLE_GUIDE.md` | A.2 | Re-run the import-graph script; counts match |
| **0.14** | **Dead code removal.** 19 unused imports (F16, F17, F14), the two `_env_*` aliases (F15), the two unused locals (F18 — check `prepare_elapsed` is not a dropped diagnostic first) | 8 files, import blocks only | F14-F18 | `pyflakes jaccpot` clean of "imported but unused"; suite green |
| **0.15** | **`TYPE_CHECKING` blocks.** Add per-mixin `if TYPE_CHECKING: from ._fmm_impl import FastMultipoleMethod`; define or replace `PreparedStateLike`; import `LargeNGradPlan` and `NearfieldInteropData` | 10 `runtime/fmm_*.py` + `fmm_state.py` | F21, E.2 | `typing.get_type_hints` resolves on every mixin method; `JACCPOT_RUNTIME_TYPECHECK=1 pytest tests/unit` green |
| **0.16** | `from __future__ import annotations` in the 13 modules missing it; `__all__` in the non-`__init__` modules missing it | 13 + 30 files, headers only | F36 | Suite green; no import-order change (isort clean) |
| **0.17** | **Repo-hygiene moves** (git `mv` only): `benchmarks/n_ladder_production/` → `bench/results/`, `results/validation/` → `bench/results/validation/`, `bench/real_vs_complex_gpu_plan.md` → `docs/`, `OCTREE_JACCPOT_STATUS_2026-03-15.md` → `docs/` with a superseded-by header | directory moves | A.7 | `grep -rn` for each old path returns nothing; `pytest -q` and `bench/ci_benchmark_guard.py --help` still work |
| **0.18** | **Test-layout moves** (git `mv` only): the 20 root test files into `tests/{unit,integration}/` and a new `tests/distributed/` | `tests/` moves + `tests/slow_tests.txt` paths | A.6 | Same collected test count before/after (`pytest --collect-only -q \| wc -l`); `slow_tests.txt` paths updated; CI workflow paths checked |
| **0.19** | Rename `tests/unit/test_large_n_grad_path.py` → `test_large_n_config_thresholds.py` (or similar) and add one line to its docstring saying it does **not** enter `LargeNPreparedState` | one file | F28 | Collected count unchanged |
| **0.20** | **Docstring completion, batch 1**: the 14 public-surface names in section C, and the ~15 highest-parameter numerics functions from F23 | docstrings only | F08 (partly), F23 (partly), C | `pydoclint --skip-checking-short-docstrings=False` violation count drops measurably; record before/after in the PR |
| **0.21** | **Write D.12 into `NUMERICS_AND_JAX.md` §3**: a gradient test at a default leaf size covers no far field, so it needs an explicit M2L-count assertion. One paragraph beside the existing physical-edge-case list | `agent_guides/NUMERICS_AND_JAX.md` | D.12 | Docs only. The two in-repo guards it describes already exist (`test_gradient_correctness.py`, the new golden) |

### Tier 1 — structural; no expression-level change to numerical code

Each of these is gated on the Tier 0 characterization named in its row.

| # | Item | Touches | Closes | Gated on | Verified by |
|---|---|---|---|---|---|
| **1.1** ✔ **PR #79** | Extract `_alignment_angles(x, y, z)` from `_multipole_align_{to,from}_z_block` — verbatim expression move, **no new `@jit`** | `operators/real_harmonics.py` | F30 | 0.3, 0.4 | **Done.** The duplication had grown, not shrunk: G.10 turned 12 shared lines + a 9-line comment into 6 + 38. Verified bit-identical *including all three JVP components* (`np.array_equal`) over 7 directions × `ell` 0..6 — the tangent check is the load-bearing one, since these are the guards G.10 proved wrong at `rho == 0`. |
| **1.2** ✔ **PR #80** | **Dedupe the near-field edge-list kernels.** `_compute_leaf_p2p_impl` becomes a wrapper: `_prepare_leaf_data` then delegate. Preserve `static_argnums=(12,)` and every `static_argname` | `nearfield/near_field.py` (−357 net) | F12, A.10 | 0.1, 0.2 | **Done.** 240 output arrays over the full flag grid (2 distributions × 2 (N, leaf) × 2 chunk sizes × baseline/bucketed × potential × pairs × precomputed-scatter) compare equal element-wise; 0 differ, none vacuously zero. A.10's compile-time caveat measured via AOT staging: trace +0.000–0.005 s, compile −0.012–+0.016 s, i.e. inside noise on a 0.09–0.13 s compile. |
| **1.3** ✔ **PR #82** | Split `operators/real_harmonics.py` along its mathematical seams (A.9). Pure file moves + import updates | 5 new `operators/` modules + a 283-line aggregator | F: `real_harmonics.py` size | 0.3 ✔, G.10 ✔, 0.5 ✔ | **Done, and A.9 was out of date in two ways:** every line citation had shifted (2236 → 2292 lines), and seam 4 (the Wigner family) no longer exists — `ab58c3d` deleted it. Five seams, 1698 carried lines verified verbatim. `real_harmonics.py` stays as an aggregator: 52 references across 35 files, `__all__` byte-identical at 28 names. **`git log --follow` does NOT survive this** — see the correction below. |
| **1.4** ✔ **PR #83** | Split `nearfield/near_field.py` seam 5 (the radix fast lane + `custom_vjp`) into `nearfield/_fast_lane.py` | `nearfield/` (3957 → 2969) | F: size; isolates the `custom_vjp` | 0.1, 1.2 | **Done.** 902 carried lines verbatim. `custom_vjp` residuals byte-identical on **both** lanes — and note the default `audit_reverse_residuals.py` invocation does *not* reach the fast lane (its top residual is `_compute_leaf_p2p_impl`, the bucketed kernel), so it was re-run with `JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE=1`. Surfaced the F16 correction below. |
| **1.5** ✔ **PR #84** | Split `nearfield/near_field.py` seams 1-3 | 4 new `nearfield/` modules (2969 → 1388) | ditto | 1.4 | **Done.** 1441 carried lines verbatim; seam 2 split into `_kernels.py` / `_scatter.py` as A.9 suggests. `bench/audit_nearfield_padding.py` output identical — but it calls `autocvd` unconditionally at import, so it cannot run in the CPU env; its own `audit()` was executed with the GPU picker stubbed instead, which is backend-independent because the measurement is topology-derived. |
| **1.6** ✔ **PR #85** | Split `runtime/kernels/core.py` along the four seams (A.9), M2L moving as one unit | 5 new `kernels/` modules + a 101-line aggregator | F: `core.py` size | 0.1, 0.2 | **Done.** A.9 names four seams; a **fifth** module (`_shared.py`) was needed, and that is the finding: after assigning the four, exactly five names were left over and both `_evaluate` and `_l2l` need one, so leaving them in `core` would have made `core` import a module that imports `core`. §10 item 1 satisfied with 102 arrays at `rtol=0`. **§10 item 4 discharged on an A100 — see B.3.** 65 parity tests passed with an empty skip list, forward exactly equal under `deterministic_ops=true`, residuals identical on both lanes, no stage outside noise. Its finding outlived the run: the GPU forward is nondeterministic by more than the difference being measured, so §10's exactness items now name the flag. |
| **1.7** ✔ **PR #86** | Extract from `PrepareMixin._prepare_state_dual_and_downward` (883 lines) — named private methods along its internal phases, no expression changes | `runtime/fmm_prepare.py` (883 → 493) | F10 | 0.1, 0.2 | **Done, two phases of seven.** The cut points are not guesswork: the function is already instrumented with `_record_dual_stage("..._dual_<phase>_seconds", ...)` at each boundary, and the data flow across every one was measured. Extracted the two where the interface earns a signature (317 lines → a 17-field bundle; 131 lines → 2 outputs); the PR lists the other five with the ratio that argued against them. The driver unpacks the bundle back into the same local names so no later expression changed. |
| **1.8** ✔ **PR #87** | Extract `_read_large_n_env_config` (271 lines) into `runtime/_large_n_env.py`. **Nothing else** | `runtime/` (2048 → 1768) | F11 (partly) | — | **Done.** The "re-open if GPU coverage has landed" check was run, not assumed: CI is still ubuntu-latest with no cuda leg and `_large_n_grad.py` / `_large_n_farfield.py` are still at **0%**, so A.9's leave-it-whole argument stands and F11 stays open. Datum for the cut line: the extracted reader measures **69%** from three CPU policy test files. |
| **1.9** ✔ **PR #81** | Move F37's inline thresholds into `runtime/fmm_constants.py` with `#:` comments | `runtime/` | F37 | 0.1 | **Done.** Note that **three different policies cross over at 262144** (`_CLASS_MAJOR_CPU_PARTICLE_THRESHOLD` plus the two named here), so they stay three constants: collapsing them would assert an equivalence that is not established. One site beyond F37's list was named too — the `< 262_144` fifty lines from its twin, the same drift hazard F37 describes. |
| **1.10** ◐ **PR #88 = part 1 of 3** | Docstring completion, batch 2: the remaining ~58 numerics functions with ≥8 params | docstrings only | F23 | — | **Partly done, and the item is a three-PR programme — see the F23 row.** Measured target set: **60** (confirming ~58). 18 done: `upward/` and `downward/` are clear of the finding, plus `distributed/fmm.py` and `nearfield/grad.py`. pydoclint with `--skip-checking-short-docstrings=False`: **2778 → 2710**. |

### Tier 2 — numerics-sensitive or API-affecting. One PR each, your sign-off each.

| # | Item | Why Tier 2 | Needs from you |
|---|---|---|---|
| **2.1** ✔ **PRs #103, #104** | Break up `_fmm_impl.FastMultipoleMethod.__init__` (722 lines, 60 params) into staged private resolvers | Config-resolution *order* is load-bearing (`b462e45`, `dee46d6` are both bugs in exactly this) and 0.1/0.2 do not cover every preset | **Done.** Body 653 → 141 lines, 573 lines into 13 resolvers, every one byte-verbatim and called in its original position. The "0.1/0.2 do not cover every preset" objection was answered first, in its own PR: a constructor-state golden over **46 configurations × 272 attributes**, which then gated the move. Boundaries came from a cut-cost profile, not from reading the code — and five of six step-2 boundaries landed mid-statement before being snapped to AST statement starts. Three blocks are deliberately left inline, by ratio of lines moved to parameters passed; the worst, `A6a`, would have been a 16-argument method calling a 16-argument function. |
| **2.2** ✔ **PR #105** | Consolidate the four hand-rolled env parsers (F13, A.4) | Changes malformed-value semantics — which reverse-mode M2L kernel runs. Requires deciding what a garbage value should mean | **Done. Decided: a malformed value means THE DEFAULT, whatever it is**, plus a one-time warning naming the variable and the value ignored. That is not a new convention — `env_int`/`env_float` already did it and `env_flag` was the outlier, so `_env` disagreed with itself. The "four parsers" were really *three* semantics (denylist flag ×2, allowlist flag, enum-with-fallback), which is why they could not be deduped mechanically; `env_choice` was added for the third. Verified over 14 inputs per reader: FUSED and DIAG unchanged on all 14, JIT differs on 6 malformed inputs (False → the default), which is the intended consequence. A.4's specific warning is satisfied. `_env` also had **no tests**; it has 39 now, written first. |
| **2.3** ✔ **CLOSED (`ab58c3d`)** | Delete or wire up the Wigner reference family (F32, ~260 lines, 8 `__all__` names) | Removes module-public names | **Already done; this row was stale.** G.4 records the decision (deleted) and `ab58c3d` carried it out. Verified 2026-08-15: `grep -ri wigner jaccpot/` returns **two prose mentions and no code** — `operators/_precision.py:5` and `operators/real_rotations.py:51-55`, both explaining why the closed-form Dehnen builders are the only rotation path. No `__all__` name remains. Nothing to do. |
| **2.4** | Any rename from section C | Public API; `test_public_api_surface.py` must change in the same commit | Your call, per your instruction |
| **2.5** ✖ **BLOCKED — measured 2026-08-15** | Turn off `--skip-checking-short-docstrings` in `pyproject.toml` | 2840 violations must be at zero first; this is the *last* commit of the docstring programme, not the first | **Cannot be done yet: 2403 violations across 524 distinct functions remain** with the flag off. Down from 2840 (0.20, 1.10 batches 1–2, F23 part 3), i.e. the programme is ~15% complete, not nearly finished. Worst files: `operators/complex_ops.py` 250, `runtime/fmm_prepare.py` 150, `downward/local_expansions.py` 117, `runtime/_interaction_cache.py` 109, `runtime/_adaptive_policy.py` 108, `solver.py` 100. By code: 507 DOC201 + 498 DOC203 (missing/mismatched `Returns`), 495 DOC101 + 495 DOC103 (missing/mismatched `Parameters`), 138 DOC501 + 138 DOC503 (missing `Raises`). `jaccpot/pallas/` is the only package-level directory at zero. **Measurement note:** pydoclint writes findings to **stderr**, so a `2>/dev/null` in the counting command reports 0 for any directory — that is how an early pass here produced "every subdirectory is clean" against a package total of 2403. |
| **2.6** | Nightly GPU CI leg for `_large_n_*`, `distributed/`, and Pallas non-interpret | Infrastructure, and it is what unblocks F27/F33/F34 and the deferred half of F11 | A decision on GPU budget — CLAUDE.md says confirm before occupying one |

---

## G. Decisions for you

### G.1 ~~Two latent `NameError`s~~ **FIXED (PR #98)** — and one of them was never reachable

> **Resolved 2026-08-14.** Everything below is the original report, kept because its
> reachability analysis was **wrong** in a way worth preserving.
>
> **G.1a — fixed as `jnp.zeros_like(locals_coeffs)`.** The physics question this section
> put to the reader ("is this branch supposed to be reachable at all?") was answered: with
> no internal nodes there is no M2L to do, so the source-motion far field is exactly zero.
> The value is not a new decision — the `pair_count == 0` guard at `_l2l.py:391` already
> returns zeros for the identical situation, and the two now agree by construction. Zeros
> rather than `None` because `None` already means "source motion was not requested" and
> the consumer cannot distinguish the two.
>
> **The reachability claim below is wrong.** This section says the branch is reached by
> "the `else` of `if child_inputs.num_internal_nodes > 0`, with
> `source_motion_multip_packed is not None`". That is necessary but not sufficient:
> getting there *also* needs `pair_count > 0`, and the `pair_count == 0` early return
> seventeen lines above has already claimed **every** single-leaf tree. A one-node tree
> cannot accept a far pair, so the conjunction is unsatisfiable. Measured rather than
> argued — forcing a single-leaf tree with a foreign interaction list still does not reach
> the line. **G.1a was dead code, not a live crash.** Worth noting that the first read of
> this in execution went the other way (reported as "live for any N <= leaf_size", since
> `num_internal_nodes == 0` is trivially true whenever N fits one leaf) before the guard
> was spotted; the `> 0` conditions in a chain of guards need reading together.
>
> **G.1b — fixed by adding the import.** Not a wrong name: the function exists at
> `operators/complex_ops.py:564` with exactly the signature the call site uses. Also
> effectively unreachable, for a different reason — `fmm_derivatives` passes
> `return_potential=False` at both of its call sites — so it is tested directly against
> `kernels/`, which ARCHITECTURE §1 licenses as a leaf library.
>
> **The pyflakes invariant below still holds.** PR #98 adds `# noqa: F821` to the two
> deliberately-dangling `FastMultipoleMethod` annotations, but `noqa` is a flake8
> directive: `python -m pyflakes jaccpot/` still reports all four sites. What changed is
> that two of the four are now gone for real, so the count is 2, not 4. See B.4.

**LINE NUMBERS RE-LOCATED (Tier 1.6).** Both citations below were already stale, and
the `kernels/core.py` split moved them again. Current locations:

| bug | as first written | on `main` today | after PR #85 |
|---|---|---|---|
| G.1a `_accumulate_from_multipoles` | `core.py:2457` | `core.py:2582` | `kernels/_l2l.py:607` |
| G.1b `evaluate_local_complex_with_grad_analytic` | `core.py:3265` | `core.py:3397` | `kernels/_evaluate.py:1285` |

Both survived the split untouched, deliberately. The check that establishes this is
worth reusing for any future move of that module: `pyflakes` reports **7**
non-import findings before and 7 after, in exact one-to-one correspondence (the two
above, the two deliberately-dangling `FastMultipoleMethod` annotations that preserve
the leaf contract, and three unused locals). A split that silently fixed or broke
something would not have that property.

**G.1a — `core.py:2582` (`_l2l.py:607` after #85).**

```python
source_motion_locals_updated = _accumulate_from_multipoles(
    jnp.zeros_like(locals_coeffs), source_motion_multip_packed
)
```

`_accumulate_from_multipoles` is defined nowhere in the module and is not imported. The `if`
branch 50 lines above (`:2405`) calls `_solidfmm_downward_accumulate_from_multipoles`
(`:362`) with the same two positional arguments **plus 15 required keyword arguments**. So the
name is wrong *and* the call is incomplete — fixing the name alone converts a `NameError`
into a `TypeError`.

Reachability: the `else` of `if child_inputs.num_internal_nodes > 0` (`:2363`), i.e. a tree
with no internal nodes, with `source_motion_multip_packed is not None` (the source-motion /
derivative-tower path). Uncovered — confirmed, `:2455-2466` in the missing-lines report.

The question I cannot answer for you: is this branch supposed to be reachable at all? A tree
with zero internal nodes has no M2L pairs, so "accumulate locals from multipoles" may be
meaningless there and the correct fix may be `source_motion_locals_updated = None` rather than
a call. That is a physics judgement.

**G.1b — `core.py:3397` (`_evaluate.py:1285` after #85).**

```python
grad, pot = evaluate_local_complex_with_grad_analytic(coeff_row, offset_row, order=int(order))
```

The module imports only `evaluate_local_complex_with_grad_analytic_batch` (`:56`). The
non-batch function exists at `operators/complex_ops.py:551`. Reachable when
`max_acc_derivative_order <= 0 and return_potential` on the complex derivative path (`:3264`);
`:3262-3273` uncovered. This one looks like a plain missing import, but the surrounding code
`vmap`s `eval_one` — so whether the right fix is importing the scalar function or calling the
`_batch` one directly is a decision about which is the intended kernel.

Both want a regression test that enters the branch, which is why they are not Tier 0.

### G.2 STYLE_GUIDE §8 vs `_env.py`: which is the policy?

§8 says `runtime/` is *"the only place that reads environment variables"*; `_env.py:21-22`
says any layer may. 16 reads live outside `runtime/` (A.3). My reading is that `_env.py` is the
newer decision and §8's sentence should be narrowed to *"the only place that resolves `"auto"`
policies"* — but that is your call, and it determines whether A.3 is 16 findings or zero.

### G.3 Should `operators/` be allowed to import `pallas/`?

`operators/m2l_real_rot_scale.py:78` imports `pallas/m2l_core_z_real` (function-local). §8
defines `operators/` as pure algebra. The three ways out: accept and document it (like the
`grad_options` case), move the dispatch up into `runtime/kernels/`, or note that `pallas/`
kernels are algebra-with-a-backend and belong in the same tier. I have no basis for choosing.

### G.4 Delete the Wigner reference family, or wire it up? — **DECIDED: deleted (`ab58c3d`)**

F32: ~260 lines (`real_harmonics.py:1120-1298` plus four `*_wigner` builders and two
`m2l_*_wigner`), **6** names in `__all__` (the original "8" was wrong), reachable only from
itself. `_real_wigner_rotation`'s docstring says it *"exists to check"* the production builders
and it never does. Someone even applied the NaN-safe azimuth hardening to this unreachable copy
(`:1277-1292`).

**The original pass recommended keeping it** (option (a): let Tier 0.3 give it its stated
purpose). Executing 0.3 changed two of the three premises behind that:

1. **It cannot run.** `sympy` is undeclared (F39), so option (a) is not "write one test" — it
   is "add a dependency", which CLAUDE.md forbids without your sign-off, and which puts a
   symbolic-algebra package into the closure of a GPU numerics library to serve six functions
   no production path calls.
2. **Its stated purpose is now served without it.** `e5d8e41` checks the same four builders
   against physics identities, which is a *better* reference than a second implementation —
   a shared convention error between `Q` and the Wigner path could have cancelled, and physics
   cannot cancel with itself. Measured ~2.5e-15, with a mutation check that bites.
3. Unchanged: an independent 30-digit reference for historically-fragile operators has real
   value *if someone can run it*.

**Resolved: (b), delete.** I had leaned toward (c) — move it to `bench/validation/` behind a
guarded import. You chose deletion, on a ground I had underweighted: the Wigner route is the
*slow* one, and the Dehnen closed forms are what we want to use regardless, so an offline
cross-check we would never reach for is not worth carrying at all. Landed in `ab58c3d`,
together with G.8. Kept: a comment at the rotation section recording what checks these builders
now, so the next reader does not repeat the search.

### G.5 `runtime/_interaction_cache.py:892` → `experimental/treecode_far_near.py`

Already documented in §8 as *"the weakest-covered production option in the tree"*. My
measurement confirms `treecode_far_near.py` at 0%, and it is reachable from
`distributed/fmm.py` with `local_walk="treecode"` at ≥2 devices. This is the one layering
violation that is a correctness exposure rather than a style one. Options: promote the module
out of `experimental/`, remove `local_walk="treecode"` as a supported option, or leave it and
accept the exposure. Not a refactoring decision.

### G.6 Priorities I need ranked — **partly answered**

**Answered:** the docstring programme is the *targeted* subset now (done, `4b2a66a`) with
the full sweep folded into Tier 1.10 and the `--skip-checking-short-docstrings` flip as the
closing gate at Tier 2.5. **Deferred by you:** the GPU questions, to be reported after this
phase — which leaves F27 (0% on the large-N reverse), F33 (`fmm_strict_run.py` at 55%) and
F34 (`distributed/` at 19%) open, and keeps A.9's "leave `_large_n_pipeline.py` whole"
standing as the working assumption rather than a decision.

The original questions, for the record:

- **`fmm_strict_run.py` at 55%** (F33): 227 unexercised statements in the per-step science
  hot path, including most of `strict_run_v2` and the velocity-Verlet update. Do you want
  Tier 0 characterization for the strict lane, or is it explicitly out of scope this pass?
- **GPU budget** (Tier 2.6). Without a nightly GPU leg, `_large_n_*` (0% for the grad path)
  and `distributed/` (19%) stay unrefactorable, which is why A.9 argues to leave
  `_large_n_pipeline.py` whole. CLAUDE.md says to confirm GPU use before occupying one, so I
  have not measured anything on GPU.
- **Docstring programme scope** (F22/D.10): 2840 latent violations. Tier 0.20 + 1.10 target
  the ~73 worst plus the public surface. Do you want the full sweep to zero (a multi-PR
  programme ending at Tier 2.5), or the targeted subset and the flag left as it is?

### G.7 One thing I checked and could not settle

`near_field.py:2829-2836` documents that omitting `max_leaf_size` under `jit` raises
`ValueError` (re-raised from the `TypeError` that `.item()` throws on a tracer), and it
documents at `:2766` that `softening` must be a concrete float. I could not find a test for either.
**Evidence that would settle it:** grep `tests/` for the `ValueError` message text, or run
`pytest --cov` restricted to `near_field.py` and check whether the `except TypeError` handler
around the `.item()` call is hit. If it is not, these are two more Tier 0 tests (0.9 assumes
they are missing). I flagged them rather than assert them because a staticness contract with
no test is a common pattern in this file and I did not want to over-count it.

### G.8 `sympy`: declare it, delete the code that needs it, or move that code out? — **DECIDED: deleted (`ab58c3d`)**

F39. `sympy` is imported at `real_harmonics.py:1123` and declared nowhere. Verified absent from
the entire dependency closure and therefore from CI. Three options, and this is a dependency
decision so it is explicitly yours:

- **Declare it** under `[project.optional-dependencies].dev` and make the six `*_wigner` names
  documented dev-only helpers. Cheapest diff; adds a symbolic-algebra dependency for code no
  production path calls.
- **Delete** the family (G.4 option (b)). No new dependency, removes ~260 lines and six public
  names.
- **Move** it to `bench/validation/` with a guarded import and a docstring saying
  `pip install sympy` (G.4 option (c)). My preference.

**Resolved: delete**, so `sympy` is gone from the repository entirely and the second option
applies. The closing observation still stands as a general point, and is now the *only* reason
this entry is worth keeping: a public name that raises `ImportError` on a stock install is a
broken contract, and nothing in the suite noticed for as long as it existed. Worth a thought
about whether any *other* module-public name is unreachable in a clean environment.
**Checked, and it is clean:** an AST sweep of every import in `jaccpot/` against the declared
set (`jax`, `jaxlib`, `jaxtyping`, `beartype`, `yggdrax`, plus what jax itself brings — `numpy`,
`scipy`, `ml_dtypes`, `opt_einsum`) and the stdlib finds **no other third-party import**. So
`sympy` was the only instance. Note that `import jaccpot` succeeding never proved this, since
the six broken names were reachable only by direct call — the sweep is what proves it.

### G.9 ~~Cross-platform bit-stability of the gradient golden~~ **RESOLVED — the gate's norm was the defect, fixed by option (2)**

> **Resolution.** Diagnosed in `docs/g9_grad_golden_gpu_diagnosis.md`; remediated by the
> per-particle inertness gate in `tests/characterization/test_fmm_grad_golden.py`. The
> element that failed is the **small component of a large vector**: `clu_real_n128_p4`
> particle 57's gradient is `(-1.689e5, +8.747e4, -2.619e2)`, the 6th largest of 128, with
> a z-component **726× smaller than the vector norm**. Round-off is proportional to the
> vector's magnitude, so the absolute drift is ~2e-9 on all three components; the
> elementwise relative test then divides that shared error by each component's own value.
> Divided by the norm instead it is **1.09e-14**.
>
> **Verdict H1, benign.** The accepted M2L set is bit-identical across devices (sha256
> match, 72 pairs); `--xla_gpu_deterministic_ops=true` changes the number only in its 5th
> digit, so it is deterministic reassociation and not the atomics behind B.3's four flaky
> failures; particle 57 sits 6–7 orders from any transverse-degeneracy guard
> (`rho/|dz| = 0.966` against a band of 1.49e-08), so G.10/D.6 are not implicated; and its
> direct-sum cancellation ratio is 1.69, ranking 126th of 128.
>
> The norm-scaled statistic is bounded at **55 ULP** across all six cases *and* two extra
> clustered seeds, on both devices — so a 1e-12 gate on it has **58× margin**, where the
> elementwise gate had 0.13× (i.e. it failed). It still bites: verified by mutation, the
> `1+1e-6` reverse-rule scaling of D.1 is rejected with six orders of margin, and a
> perturbation confined to one small component is rejected down to 1e-9.
>
> **Option (2) was taken, not (3)**, which is what the earlier update below predicted.
> Device-gating the elementwise assertion needs a band near **1e-8** to pass on GPU, and
> under that band a genuine 1e-9 single-component error goes undetected — the norm-scaled
> gate at 1e-12 catches exactly that. The elementwise gate is kept **on CPU**, where the
> goldens were generated and hold with ~130× margin, so its single-component sensitivity is
> retained where it is a real claim. No golden was regenerated and `INERT_RTOL` is
> unchanged. Verified: 8 passed on 2 of 2 A100 runs (was 3/3 red), 29 passed in
> `tests/characterization/` on CPU.
>
> One thing this exposed and did **not** fix: `grad_masses` at particle 57 drifts **898
> ULP** on the A100 — genuine cancellation in a scalar sum, not a small denominator (a
> scale floor does not move it). It passes at 1e-12 with only 5× margin. Left as-is
> deliberately; it is a thin margin, not a failure, and widening or restructuring it is a
> separate decision.

> **Update (A100 run, B.3).** The axis this section worried about **held**: macOS-generated
> goldens are green on ubuntu CI, across every Tier 1 PR. The one that broke is
> cross-*device*, which this section did not consider:
> `test_fmm_grad_golden[clu_real_n128_p4]` fails on the A100 with
> `Mismatched elements: 1 / 384 (0.26%)`, `grad_positions drifted from the committed
> golden`, at `rtol=atol=1e-12`. It is **reproducible 3/3 under both default XLA and
> `deterministic_ops=true`** — so unlike the four flaky failures in B.3 it is not
> nondeterminism — and it **fails identically on pre-Tier-1 `128a0e2`**, so it is not a
> Tier 1 regression. The golden and `INERT_RTOL` are untouched.
>
> This makes option (3) the live one, re-read for device rather than platform: gate the
> inertness assertion on device, or commit a per-device golden. Note that (1) does not
> apply — there is nothing wrong with the Linux CPU numbers. Wants its own investigation
> and its own PR; the first question is whether one element in 384 at 1e-12 is
> reassociation in the reverse M2L graph or something structural, and that is answered by
> finding *which* element.

The six `.npz` in `tests/characterization/golden_grad/` were generated on **macOS / CPU /
JAX 0.10.2**; CI runs **ubuntu-latest**. The forward golden sets a precedent that float64 CPU
results are reproducible across platforms at `rtol=1e-12`, but a reverse-mode program is a much
larger XLA graph with more scope for platform-dependent reassociation, and I could not test it
from here.

If `test_fmm_grad_golden` goes red on CI with small relative differences, **the correct response
is not to relax `INERT_RTOL`** — that would convert the tripwire into a formality on its first
day. The options in order of preference: (1) regenerate the goldens on Linux and commit those,
accepting that local macOS runs then need a documented skip; (2) snapshot a reduction-invariant
summary alongside the raw arrays; (3) keep the arrays but gate the inertness assertion on
platform. Decide when there is data — a green CI run makes this moot.

### G.10 ~~**BUG**~~ **FIXED** — the degeneracy guard zeroed the transverse gradient in **both** bases, and it reached the force

> **Resolved.** Everything below is the original report, kept as the record of how it was
> found and what was measured; it is not a description of current behaviour. The fix landed
> in PR #70 and #71 and is summarised in the status table above. The analysis that still
> applies is *why* no guard choice could work — that reasoning is what forced the
> `custom_jvp` approach, and it is restated at the site in
> `jaccpot/operators/_transverse_degeneracy_jvp.py`.

This is the third bug in the audit and the only one that returns a **wrong number** rather than
crashing. Found by trying to write the Tier 0.4 test: the test refuted the claim it was meant to
pin.

**The claim.** `real_harmonics.py:1372-1379` (and the identical block at `:1401-1408`) states that
at `rho == 0` *"the azimuth is undefined there and the rotation is a pure polar turn / identity,
so a zero cotangent is the correct subgradient."*

**The measurement.** `grad` of the assembled operators w.r.t. the displacement, at `z = 2.5`,
approaching `rho → 0` from **eight** directions (`rho = 1e-9`), versus the value the code returns
at `rho == 0` exactly:

| operator | true limit `(d/dx, d/dy, d/dz)` | returned at `rho == 0` | error |
|---|---|---|---|
| `m2l_real` | `(-1.502050, -0.523434, +0.834153)` | `(0, 0, +0.834153)` | **1.50, 0.52**, 0 |
| `m2m_real` | `(-6.416905, +1.769043, -9.651272)` | `(0, 0, -9.651272)` | **6.42, 1.77**, 0 |
| `l2l_real` | `(+0.305315, +0.003498, +0.072012)` | `(0, 0, +0.072012)` | **0.31, 0.0035**, 0 |

Spread of the limit across the eight approach directions: `~1.4e-07`, i.e. finite-difference
noise. **The limit is direction-independent, so the derivative exists** — it is simply being
discarded. `d/dz` is correct throughout; only the two transverse components are lost.

**It is on the production differentiable path**, not just in a standalone helper.
`operators/m2l_real_rot_scale.py:19-22, 48, 60, 254-260` — the production real-basis M2L, reached
from `runtime/kernels/core.py:1517` via `m2l_rot_scale_real_batch` — calls exactly these builders.
Verified end-to-end on the batch kernel with one z-aligned pair among generic ones: that pair's
gradient goes `(-0.655841, -0.580107, +1.120002)` → `(0, 0, +1.120002)` at `rho == 0`, while the
other pairs are unaffected. So the defect is **per-pair** and silent.

**Why nothing caught it.** The degenerate input has measure zero, so randomly-generated test
positions never hit it — including in the gradient golden I just added (D.1). The configuration is
not rare in practice, though: the comment itself names the two cases the fixed-topology FMM
actually hits, *"zero displacement (single-child COM L2L pairs)"* and *"z-axis-aligned displacement
(rho == 0, lattice-aligned M2L pairs)"*.

**Forward values are unaffected.** Both characterization oracles are byte-stable, as the comment
claims — this is a reverse-pass-only defect.

**Why the guard looks right.** The raw alignment block really is discontinuous at `rho == 0`
(measured jump of 2.0 along ±x; `az = atan2(x, y)` does not converge), so zeroing its cotangent
*in isolation* is defensible. The error is that the **composition** `rotate → z-translate →
rotate-back` is continuous and differentiable there, because a z-translation commutes with a
z-rotation and the jump cancels — measured: `m2l_real` agrees with its `rho == 0` value to
4.9e-10 at `rho = 1e-9` from every approach direction. It is the composition whose gradient is
taken, and the guard is applied to the factor.

**Precedent for the fix, and why I did not apply it.** `d5cb13b` fixed this same defect class in
L2P/P2M, and `_azimuth_from_floored_rho` documents the technique: drop the branch and let a
*floored* `rho` keep the division finite, so *"the limit comes out of the algebra instead of being
branched away."* That is **not** a mechanical copy here — flooring `rho` does not help
`arctan2(x, y)`, whose arguments are `x` and `y` directly, so the rotation case needs its own
derivation. Which is exactly why CLAUDE.md says this is its own PR with its own test, and why I
stopped.

**Suggested sequencing:** (1) a bug-fix PR deriving the correct treatment for the azimuth, with
the two tracking tests as its regression guard; (2) confirm both goldens stay byte-stable (the
forward pass must not move); (3) only then Tier 1.3's split of `real_harmonics.py`.

---

#### Update after attempting the fix (`4ae5479`, `cb83adc`) — scope is larger than written above

**It is not real-basis-only.** `operators/complex_ops.py:1004-1009` (`m2l_complex_reference`)
carries the same guard and loses the same component: at `rho == 0`, `z = 2.5`, `grad` returns
`(0, 0, +0.997445)` against a limit of `(-0.555465, ~0, +0.997445)`. Both bases, one shared
pattern.

**It reaches the force gradient — this is the part that matters.** On four z-stacked clusters
sharing one `(x, y)` point set and identical intra-cluster masses (so every cluster COM has the
same `(x, y)`, and **6 of 24** accepted M2L pairs have `rho == 0` exactly), FD-vs-AD along an
asymmetric purely-transverse direction disagree by **1.9e-03** relative (real basis) and
**1.8e-05** (complex), stable between `h = 1e-6` and `1e-7`. For the real basis that is the same
order as the FMM's own force error. Pinned by
`tests/unit/test_gradient_correctness.py::test_fd_vs_ad_along_a_transverse_direction_at_rho_zero`
(strict xfail, both bases).

**A "measured" comment asserted the opposite and has been retracted.**
`m2l_complex_reference`'s comment said the zero cotangent was correct and that this was *"now
measured"* — AD matching FD to ~10 significant digits on a `rho == 0` construction. The
measurement was real; the construction was **symmetric** (same `(x, y)` set and same
intra-cluster masses in every cluster), so the per-pair transverse errors cancelled in the sum.
With an asymmetric cotangent and direction they do not. This is the most transferable lesson in
the audit: *a symmetric test construction can certify a broken gradient.*

**Two candidate fixes measured, both fail.**

| approach | result |
|---|---|
| floor `rho` (the `_azimuth_from_floored_rho` technique from `d5cb13b`) | gradient **unchanged** — still `(0, 0, +0.834153)`. The `az` guard is the culprit, not `rho`. |
| remove both guards | `NaN`, and the zero-displacement gradient stops being finite |
| express the azimuth as ratios + a Chebyshev `Dz` (mirroring `p2m_real_direct`) | **breaks the forward** at `rho == 0` by 1.0-2.0: `(cos az, sin az) = (y/rho, x/rho)` collapses to `(0, 0)` there, which is not a rotation. Also moves generic values by ~3e-16, so goldens would shift. |

The obstruction is structural: the code reaches `(x, y)` only through `rho` and the azimuth, so
every chain-rule route carries `x/rho` or `y/rho²`. The O(rho) coefficient the true derivative
needs has already been divided out. Recovering it needs the analytic translation-derivative
identities for the Dehnen operators — a derivation, not an implementation, which is why this
stayed a bug report.

**Resolved: (a).** You chose to derive the analytic transverse derivative and attach `custom_jvp`s,
without regressing the forward pass or the currently-correct gradients.

**The derivation is done and validated** — `docs/rotation_degeneracy_derivative.md`, commit
`8adc25f`. The cascade is rotationally covariant, so differentiating
`F(R·delta) = D_out(R) F(delta) D_in(R)⁻¹` along a rotation family that sweeps `(0,0,z)`
transversally turns the unavailable derivative into a commutator with the representation
generators:

```
dF/dx = +(1/z) [ G_out^y F0 - F0 G_in^y ]
dF/dy = -(1/z) [ G_out^x F0 - F0 G_in^x ]
```

Every generator was calibrated against an identity the repo already verifies (residuals
1.4e-17 to 5.2e-11), and the formula is **exact**: the finite-difference step sweep shows
truncation falling as `eps²` to 3.9e-11 then round-off rising, with no error floor. Verified
across orders 1/2/4/6 and `z ∈ {+2.5, −2.5, +0.4, −7.0}`.

**Implementation is not started.** The derivation document maps the surface and the constraint:
the tangent must be `where(rho_sq > 0, existing JVP, analytic)` so the primal is untouched and
every currently-correct gradient is preserved bit-for-bit, with a third regime for `delta == 0`
(the formula divides by `z`). Sites in priority order — production real M2L
(`m2l_real_rot_scale.py`) first, then the complex path, then the reference operators, then the
M2M/L2L cascades in `upward/`/`downward/`. Two tracking `xfail(strict=True)` tests must flip and
both goldens must stay byte-unmoved.

### G.11 `pair_grouped` M2L applied the wrong class's rotation — **FIXED**

**Resolved.** The leading hypothesis below was right: the per-pair gather was
misaligned. `pair_grouped` now matches `class_major` to reassociation, and the plateau
that remains is the genuine class-cached trade shared by both modes.

Measured after the fix, same configuration (relative L2 versus a direct O(N²) sum):

| order | default | `pair_grouped` | `class_major` |
|---|---|---|---|
| 2 | 7.230e-04 | 8.927e-04 | 8.927e-04 |
| 4 | 8.148e-05 | 2.049e-04 | 2.049e-04 |
| 6 | 1.128e-05 | 1.887e-04 | 1.887e-04 |

Clustered N=256 at order 4: 3.94e-02 → 3.478e-03, again equal to `class_major`.

**The defect.** `GroupedInteractionBuffers` stores `class_sources` / `class_targets`
sorted by displacement class, but stores `class_ids` under the *inverse* permutation, in
the original pair order (`grouped_interactions.py:185`,
`class_ids_original = class_ids_sorted[inv_order]`). The two are therefore not
co-indexed. `pair_grouped` gathered `blocks_{to,from}_classes[grouped.class_ids]`,
handing most pairs another class's rotation; `class_major` was immune because it reads
the class id from its CSR segment table. Both the fullbatch and the chunked-scan branch
of `pair_grouped` were affected — they shared the same `class_ids` argument.

**The measurement that settled it** (the check proposed below, run on the golden's own
uniform N=256 tree, 20 far pairs): the angle between `class_displacements[class_ids[i]]`
and `centers[class_targets[i]] - centers[class_sources[i]]` was **mean 39.8°, median
19.6°, max 150.1°, with 70% of pairs beyond 10°**. Deriving the class id from
`class_offsets` instead gives **mean 0.94°, max 2.13°, none beyond 10°** — the residual
being the AABB-centre jitter inside a lattice cell, which is the approximation the mode
is supposed to make.

**The fix** (`_pair_class_ids_from_offsets`, `runtime/kernels/core.py`): derive the
per-pair class id from the CSR `class_offsets` — the same table `class_major` reads —
and pass `class_offsets` to the two accumulators in place of `class_ids`, so the
misusable array no longer reaches them. The derivation is a `searchsorted` inside the
already-jitted kernels, so the cached class rotation the mode exists for is untouched
and no per-pair rotation is built.

**Test coverage was vacuous.** `test_solidfmm_grouped_class_major_matches_pair_grouped`
already asserted these two modes agree — at 112 particles, `leaf_size=16`, `theta=0.6`,
which yields **zero** far pairs. It compared two all-zero arrays. Raised to 512
particles / `leaf_size=8` (774 far pairs) with an explicit non-vacuity assertion; it
fails at relative L2 **1.017e+00** without the fix. `pair_grouped` is now goldened in
`golden_modes/` on the same two cases and anchors as `class_major`, and
`tests/unit/runtime/test_grouped_class_id_alignment.py` pins the alignment invariant
directly.

The original analysis follows.

---

Surfaced while widening the golden (0.2, `ef7ce15`). Relative L2 versus a direct O(N²)
sum, `preset="accurate"`, solidfmm, leaf 8, theta 0.5:

| order | default | `pair_grouped` | `class_major` |
|---|---|---|---|
| 2 | 7.230e-04 | 1.253e-02 | 8.927e-04 |
| 4 | 8.148e-05 | **1.246e-02** | 2.049e-04 |
| 6 | 1.128e-05 | **1.245e-02** | 1.887e-04 |

Clustered N=256 gives 3.96e-02 / 3.94e-02 / 3.94e-02 for `pair_grouped`. The default
converges as expected; `pair_grouped` is flat to three digits, and `class_major` is flat
from order 4.

**Order-independent error is the signature of a fixed geometric approximation rather than
expansion truncation** — which is exactly what this repository's own golden file says
about the known-broken `cartesian` basis (*"a divergent-series signature, not
truncation"*).

**Narrowed further, and it is a bug, not a trade.** Read at the source level, the two
modes are *the same computation, differently batched*:

- `_accumulate_solidfmm_m2l_grouped_fullbatch` (`kernels/core.py:1163`) —
  `deltas = centers[tgt_sorted] - centers[src_sorted]`, blocks indexed **per pair** by
  `class_ids_sorted`.
- `_accumulate_solidfmm_m2l_class_major_chunked_scan` (`kernels/core.py:1255`) — the
  same `deltas = centers[tgt_chunk] - centers[src_chunk]`, the same
  `blocks_{to,from}_classes`, broadcast **per class segment**.

Both take their rotation from `_rotation_blocks_for_grouped_classes` and their
translation from the true per-pair displacement. `class_major` has no fallback path —
it builds the segment tables itself when they are not precomputed — so both genuinely
run the class-cached scheme. Two batchings of one computation should agree to
reassociation, not differ by 60x.

**Two hypotheses eliminated by measurement.** (1) An inconsistent hybrid — class
rotation paired with per-pair distance — is *not* the cause: forcing `pair_grouped` to
use the class representative displacement instead, making it fully consistent, moves the
error only from 1.246e-02 to 1.170e-02. (2) `class_major` silently falling back to an
ungrouped path — ruled out by reading the branch; it always reaches the class-major scan.

**Leading hypothesis, untested** *(confirmed — see the resolution above)*:
`grouped.class_ids` is not aligned with the ordering of
`grouped.class_sources` / `class_targets`, so `pair_grouped`'s per-pair
`blocks_to_classes[class_ids_sorted]` gather applies the wrong class's rotation to each
pair, while `class_major`'s per-segment indexing is immune because it takes the class id
from the segment table. That fits every observation: identical math, a 60x gap, and
insensitivity to which displacement is used. **What would settle it:** for each pair,
check that `class_displacements[class_ids[i]]` is parallel to
`centers[class_targets[i]] - centers[class_sources[i]]`. If the angle is large for most
pairs, the gather is misaligned and the fix is a permutation, not a scheme change.

Not fixed at the time of writing, and not encoded into a golden anchor: giving
`pair_grouped` a golden would have needed a ~4e-2 bound, which legitimises the plateau.
`test_grouped_farfield_plateaus_in_order` records it as measured behaviour for both
modes and fails if it ever changes in either direction.

**What I would want to know:** whether `pair_grouped`'s ~60x gap over `class_major` is
inherent to the coarser grouping or a defect in the representative-displacement choice.
The two differ only in how classes are batched, so a 60x accuracy gap between them is
the part that looks less like a trade and more like a bug. *(Answered: a defect. The
residual plateau that both modes share afterwards — order-independent at ~1.9e-04
uniform — is the real trade, and is now documented on `FarFieldConfig.mode`.)*

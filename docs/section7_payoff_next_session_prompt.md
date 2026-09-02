# Fresh-session prompt: implement section 7 (the payoff application)

Hand this to a new session. It is written to be self-contained, but **Step 0 is to
verify the state yourself** — do not trust this document's description of the code.

---

## What you are building and why

The `jaccpot` paper (branch `paper/jaccpot-i`) has sections 1–6 and 8 drafted and
measured. **Section 7, the payoff application, is the only section still empty**,
and it is blocked on code that does not exist rather than on a measurement.

The paper's central claim is that a fast multipole method that is differentiable
end to end makes gradient-based inference through an N-body force model practical.
Sections 3–6 establish that the force is accurate, that it scales, and that the
reverse pass costs 1.5–3.8× the forward. Section 7 has to show someone actually
using that: **recovering physical parameters from kinematics by descending a
gradient through the FMM.** Without it the paper argues capability without
demonstrating payoff.

Two figures are reserved and currently commented out in
`sections/07_case_study.tex` of the manuscript repo:

- **fig14** — energy and angular-momentum conservation over a long integration.
- **fig15** — gradient-based recovery: loss and parameter error vs. iteration.

## The design decision that has already been made

**Work in phase space with kinematics expressed as Gauss–Hermite expansions. Do
not build a synthetic IFU generator.**

The existing stub (`generate_synthetic_ifu_kinematics`) proposes mock IFU cubes
with a field of view and observational noise. That is more machinery than a proof
of concept needs and is not what the comparison should be against. Surveys such as
GECKOS reduce their data to Gauss–Hermite coefficients of the line-of-sight
velocity distribution, so **GH coefficients are the natural observable**: they are
what real data is reduced to, and they are a small, differentiable summary of a
phase-space distribution.

Concretely, the observable should be, per spatial bin, the LOSVD summarised as
$(V, \sigma, h_3, h_4, \ldots)$. The forward model maps model parameters →
phase-space realisation → binned GH coefficients; the loss compares those against
"observed" coefficients generated from a known ground truth; the gradient flows
back through the FMM force.

Rename or replace the IFU-named function accordingly — do not implement it as
specified.

## Step 0 — establish state (do this before writing anything)

Do not trust the summary below; it is what was true on 2026-08-24 and it is here
so you know where to look, not so you can skip looking.

1. `jaccpot/applications/potential_recovery/` — four files, all scaffolding:
   - `model.py` — `ParametricPotential.potential()` and `.acceleration()` raise
     `NotImplementedError`; `generate_synthetic_ifu_kinematics()` raises.
   - `recover.py` — `recover_grad_descent()` and `recover_hmc()` raise. Their
     dataclass `RecoveryResult(params_history, loss_history)` is defined and is a
     reasonable shape to keep.
   - `energy_conservation.py` — `integrate_and_track_conservation()` raises.
   - `__init__.py` — empty.
2. `bench/payoff/energy_conservation.py` and `bench/payoff/parameter_recovery_demo.py`
   — both `raise NotImplementedError` in `main()`, and both predate the paper's
   provenance pipeline (they write bare JSON with `json.dump` rather than going
   through `jsonio.write_result`). **They must be rewritten, not filled in.**
3. `bench/results/payoff/` does not exist.
4. Read `sections/07_case_study.tex` in the manuscript repo for the reserved
   figure blocks and labels.

## Answering the one TODO that is already settled

`energy_conservation.py` says "check nornax's integrator examples/tests first".
**It has been checked: nornax has what you need. Do not write a new integrator.**

`/export/home/tbuck/nornax` provides `initialize_state`, `total_energy`,
`JaccpotForceModel`, leapfrog KDK, and `hermite4_step` / `hermite6_step` /
`hermite8_step`, plus existing `tests/integration/test_energy_conservation.py`,
`test_leapfrog_energy_conservation.py` and `test_jaccpot_adapter_smoke.py`. The
adapter surface it expects on the jaccpot side is
`compute_accelerations(positions, masses, **kw)` and
`compute_accelerations_and_jerk(positions, masses, velocities, **kw)`.

jaccpot ships the other half: `jaccpot/nornax_adapter.py::BlockStepFMM`, which
faces the **momentum-conserving mutual lane** (`jaccpot/mutual/`, documented in
`docs/momentum_conserving_fmm.md`). That lane evaluates each pair once and applies
$+f/-f$, so momentum cancels algebraically — which is exactly the property a
long-integration conservation figure should be exercising. Use it.

**Blocker to resolve with the user first:** `nornax` is present at
`/export/home/tbuck/nornax` but is **not installed in jaccpot's `.venv`**.
`CLAUDE.md` forbids adding dependencies without asking. Either get agreement to
install it (editable, from the sibling path), or make `bench/payoff/energy_conservation.py`
skip cleanly with a clear message when nornax is absent. Ask; do not decide silently.

## What to build

### 1. The forward model and the mock observable (`model.py`)

- Choose and implement a parametric family with a small number of free
  parameters. Keep it simple enough that the ground truth is unambiguous and the
  recovery is interpretable — the figure has to show a parameter error going down,
  not a hero fit.
- Implement the Gauss–Hermite reduction: given phase-space coordinates, bin
  spatially, project velocities onto a line of sight, and return GH coefficients.
  This must be differentiable — it is inside the loss.
- Generate the mock "observation" from a known ground truth with a recorded seed.
- Document the returned dict's keys properly. The current docstrings say the keys
  "are not enumerated yet"; that must not survive.

### 2. The recovery (`recover.py`)

- `recover_grad_descent`: forward model → GH coefficients → loss vs. observed →
  `jax.grad` → optimiser step. Return the existing `RecoveryResult`.
- The gradient must actually flow through the FMM force. That is the entire point
  of the section; a recovery that only differentiates an analytic potential
  demonstrates nothing about this paper.
- Use `differentiable_step_fn` for the compiled seam if you take many steps —
  see `docs/fig12_autodiff_overhead_blocked.md` for why the naive routes
  (eager dispatch, an outer `jax.jit` over everything) are traps.
- Leave `recover_hmc` unimplemented unless the user asks for a posterior. Its
  docstring already says it is optional.

### 3. The two bench drivers (`bench/payoff/*.py`)

Rewrite both against the paper pipeline. Copy the shape of a script that already
works — `bench/differentiability/grad_correctness.py` is a good model.

Non-negotiable pipeline details, each of which has already cost this project time:

- `runmeta.select_gpu(args.gpu_select)` **before the first `import jax`**, then
  `runmeta.add_common_args(p)` for `--seed/--gpu-select/--dtype/--json-out`.
- Write with `jsonio.write_result(path, config=cfg, meta=runmeta.run_meta(), data=...)`.
  `config` **must** contain all of `REQUIRED_CONFIG_KEYS` = `n, theta, order,
  basis, seed, device, precision` or the write is refused.
- **`--json-out` relative paths resolve under `bench/results/`.** Passing
  `bench/results/payoff/x.json` writes to `bench/results/bench/results/payoff/x.json`.
  Pass an absolute path or a path relative to `bench/results/`.
- Timing, if any, goes through `bench/scaling/_timing.py::time_min_repeat` so every
  timing number in the paper reports the same statistic.

### 4. Register the figures in all three places

Adding an artifact is not enough; a figure that is not registered silently does
not exist:

1. `examples/jaccpot_paper/build_notebooks.py` — add `FIG14`/`FIG15` source
   strings, their `*_CAPTION`s, and entries in the registration list near the end.
   Note the figure sources are triple-quoted strings, so **a nested `"""`
   docstring inside one breaks parsing** — use a comment.
2. `examples/jaccpot_paper/export_to_paper_repo.py` — add both to `FIGURES`
   (tuples of `name, number, notebook, artifact-relative-path`) so they acquire
   provenance rows. The table is generated, never hand-edited.
3. `tests/unit/test_bench_smoke.py` — add CPU smoke cases to `CASES`. **Note its
   coverage test currently scans only `("validation", "scaling", "differentiability")`,
   so `bench/payoff/` is invisible to it.** Add `"payoff"` to that tuple so the new
   scripts cannot silently rot, which is precisely the failure that test exists to
   catch.

Then: `python examples/jaccpot_paper/build_notebooks.py`,
`run_notebooks.py fig_14 fig_15`, `export_to_paper_repo.py --paper-repo
/export/home/tbuck/jaccpot-paper-i`, and uncomment the figure blocks in
`sections/07_case_study.tex`.

Committing a new artifact needs `git add -f` — `bench/results/.gitignore` ignores
everything by default and figure summaries are added explicitly. Keep them small;
bulk per-particle arrays do not belong in git.

## Repo rules that will bite you

- **`jaccpot/applications/` is NOT in `CLAUDE.md`'s test-first exemption.** That
  exemption covers `examples/`, `bench/` and `jaccpot/experimental/` only. So
  `potential_recovery/` is production library code: **write the tests first.**
  If you think it should be exempt, ask — do not assume.
- **pydoclint runs on `^jaccpot/`**, numpy style, no baseline, and
  `skip-checking-short-docstrings = false`. Every parameter, return and raise in
  `potential_recovery/` must be documented. A one-line summary over undocumented
  parameters is a violation.
- Feature branch, never `main`. Atomic conventional commits. Changes over ~400
  lines get split.
- Run the full verification block in `CLAUDE.md` before declaring done, and paste
  the result.
- GPUs on this host are shared. Use `autocvd` (via `--gpu-select least-used`) and
  confirm with the user before occupying a card.

## Measurement discipline — learned the hard way on this paper

Section 7 is a figure that shows a loss going down, which is the easiest kind of
figure to fake accidentally. Guard against it:

- **A recovery that converges because the forward model is degenerate proves
  nothing.** Check that the loss is actually sensitive to the parameters you claim
  to recover — perturb the ground truth and confirm the loss responds.
- **Verify the gradient is the gradient.** Compare `jax.grad` of the loss against
  a finite difference of the *same* function before trusting any descent curve.
  Section 6 does exactly this and the machinery is reusable.
- **Do not report a number that is not in a JSON.** Every figure number in this
  paper comes from a committed artifact with a clean `git_dirty=false` and a
  reachable commit sha. A dirty tree invalidates an artifact; so does amending the
  commit it recorded.
- Several results in this paper were initially wrong in ways that looked exactly
  like physics — a silent 40% force error that was an indexing bug in the
  benchmark, a stage reporting zero time, a ratio below one. In each case the
  distinguishing step was checking the instrument before the code. Budget for it.

## Definition of done

- `jaccpot/applications/potential_recovery/` has no `NotImplementedError` on the
  paths section 7 needs, with tests written first.
- `bench/results/payoff/{energy_conservation,recovery}.json` exist, clean, with
  reachable commit shas.
- fig14 and fig15 build, export, and appear with provenance rows showing `Dirty: no`.
- `sections/07_case_study.tex` has prose and both figure blocks uncommented.
- The full `CLAUDE.md` verification block passes, and
  `tests/characterization/test_fmm_golden.py` has not moved.

## Ask the user before starting

1. Install `nornax` into jaccpot's `.venv`, or skip fig14 when it is absent?
2. Which parametric family — and is recovering a small number of potential
   parameters the right demonstration, or should the target be particle masses or
   an initial condition?
3. Is `jaccpot/applications/` test-first, or does it get `bench/`-style exemption?
4. Point estimate only, or is the HMC posterior wanted for the paper?

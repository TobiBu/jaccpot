# jaccpot — contributor & agent guide

`jaccpot` is a JAX Fast Multipole Method solver built on `yggdrax` tree artifacts:
GPU-native, **differentiable**, with a Pallas kernel path and a distributed path. It is
research software *and* a performance-critical numerical library. When goals conflict:

1. **Numerical accuracy** — a wrong number that looks plausible is the worst thing this
   library can produce.
2. **Readability and maintainability** — optimise for a physicist reading the code
   top-to-bottom for the first time.
3. **Performance** — near the top, not a nice-to-have.

## The two guides

- **`agent_guides/STYLE_GUIDE.md`** — house code style. Read it and apply it.
- **`agent_guides/NUMERICS_AND_JAX.md`** — invariants that must not be broken, JAX rules,
  and how we test and benchmark. **Read it before touching `jaccpot/operators/`,
  `jaccpot/upward/`, `jaccpot/downward/`, `jaccpot/nearfield/`, `jaccpot/pallas/`, or
  `jaccpot/distributed/`.**

You may draft in whatever style you find natural, but **convert touched files back to house
style before finishing**, using the checklist at the end of `STYLE_GUIDE.md`.

Also read `ARCHITECTURE.md` and `CONTRIBUTING.md`. The `docs/` directory holds design notes
and audits (`differentiable_fmm_design.md`, `differentiable_fmm_audit.md`,
`differentiable_fmm_distributed_audit.md`, the profiling and plan documents). If you are
about to reason from first principles about why something is the way it is, check `docs/`
first — the answer is usually already written down, with measurements.

## Non-negotiables

These override any instruction to "clean up", "optimise", or "simplify":

- **Do not change what the code computes.** Refactoring preserves numerics within the stated
  tolerance. If `tests/characterization/test_fmm_golden.py` moves, the change was wrong —
  revert it, do not update the golden or relax the tolerance.
- **Do not change the order or associativity of floating-point reductions**, accumulation
  dtypes, or matmul precision. See `jaccpot/operators/_precision.py` — fp32 matmul precision
  is pinned deliberately and the accuracy floor it protects is measured.
- **Do not break differentiability.** The value of this library is that you can take
  gradients through it. A change that preserves the forward pass and silently breaks the VJP
  is a broken change.
- **Do not break `jit`.** No Python control flow on traced values, no `.item()` / `bool()`,
  no host sync inside a jitted path.
- **Do not touch the JAX version floor or ceiling in `pyproject.toml`.** Both are
  load-bearing and the reasoning — CPU SIGFPE below the floor, a 2.6× CPU slowdown above the
  ceiling, the `ragged_all_to_all` reverse-pass fix — is written out there in full.
- **Do not "improve" an algorithm or numerical scheme** while doing something else. If you
  think one is wrong, say so and stop; that is its own PR with its own test.
- **Do not add, remove, or bump dependencies** without asking. The `black` and `isort` pins
  are deliberately equal to the pre-commit hook revs; unpinning them makes CI and local
  formatting disagree.
- **Do not weaken a test, relax a tolerance, or delete a test.** Ever.
- **Do not rename anything public** without asking.
- **Do not reformat files you are not otherwise touching.**

## Workflow

- Feature branch, finalise via PR. Never commit to `main`.
- **Test-first for production library code.** `examples/`, `bench/`, and anything under
  `jaccpot/experimental/` are exempt — that distinction is deliberate, do not "fix" it. The
  `experimental` marker deselects the *tests* (`tests/experimental/`), which is what keeps
  `jaccpot/experimental/` out of the default run; it is also omitted from coverage in
  `pyproject.toml`.
- Atomic commits, conventional-commit format (`feat:`, `fix:`, `refactor:`, `test:`,
  `docs:`, `perf:`, `build:`, `ci:`). One logical change per commit so `git bisect` and
  review work.
- Changes over ~400 modified lines get split. Large diffs get rubber-stamped, not reviewed.
- Keep PRs focused; include tests for behaviour changes; update README and docs when
  user-facing APIs change.
- Run the verification block below before saying you are done, and paste the result.

## Verification

```bash
black --check .
isort --check-only .
pre-commit run --all-files                     # includes pydoclint (numpy style)
JAX_ENABLE_X64=1 pytest -q                     # xdist -n auto; use -n 0 for pdb / -x
JAX_ENABLE_X64=1 pytest -q tests/characterization   # golden reference — must not move
```

pydoclint runs with `skip-checking-short-docstrings = false`, so a one-line summary over
undocumented parameters **is** a violation. `.pydoclint-baseline.txt` holds the
pre-existing tail, so only new violations fail. When a batch documents a file, drop its
entries:

```bash
pydoclint --config pyproject.toml --generate-baseline True jaccpot/
```

Check that diff **only removes lines** — the hook passes staged filenames, so
regenerating from a subset would silently discard other files' entries. That is also why
`auto-regenerate-baseline` is `false`; do not turn it on.

`pytest -q` is not the whole suite: the `addopts` in `pyproject.toml` add
`-m "not experimental"` and `--ignore=tests/perf`, so the octree/treecode prototypes and the
performance assertions are opt-in (`pytest -m experimental`, `pytest tests/perf`).

Faster inner loop while iterating:

```bash
pytest -n 2 -m "not slow and not experimental"      # the CI smoke subset
JACCPOT_RUNTIME_TYPECHECK=1 pytest -q tests/unit    # jaxtyping + beartype runtime checks
```

Coverage is measured and uploaded in CI (`--cov=jaccpot --cov-branch`), but there is no
`fail_under` threshold — a coverage drop will not fail the build on its own. Locally:
`pytest --cov=jaccpot --cov-report=term-missing`.

Prefer CPU for correctness and style work — GPUs on this machine are shared. Confirm which
device you may use before launching anything that occupies one, and before running
`bench/` or `tests/perf/`.

### Running the suite on a GPU

**Do not use the bare `pytest` above on a GPU.** The `addopts` in `pyproject.toml` carry
`-n auto`, which on this 64-core box is 64 xdist workers contending for one card: it produces
tens of thousands of `CUDA_ERROR_OUT_OF_MEMORY` lines and hundreds of allocation "failures"
that look exactly like numerical ones. Measured 2026-08-12: 331 and 288 such failures on two
trees, with differing failure sets and a 2× wall-time gap — none of them real.

Cap the workers and bound the per-worker share instead:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.12 \
  JAX_ENABLE_X64=1 pytest -n 6            # 0 OOM lines on a 40GB A100
```

Select the card with `autocvd` before `import jax` (org rule, see `ARCHITECTURE.md` §7), and
pin it for the whole session if you are comparing timings, so two runs cannot land on
different GPUs.

Expect a handful of GPU-only failures that are **not** yours: see `ARCHITECTURE.md` §9 for the
current list, and §10 for why any exact-equality assertion needs
`XLA_FLAGS="--xla_gpu_deterministic_ops=true"` before its result means anything.

## Layout

```
jaccpot/basis/         real and complex solid-harmonic bases
jaccpot/operators/     expansion algebra: harmonics, M2L rotate/scale, precision pinning
jaccpot/upward/        P2M / M2M sweep
jaccpot/downward/      M2L / L2L / L2P sweep
jaccpot/nearfield/     P2P and its gradient
jaccpot/mutual/        momentum-conserving path — a SECOND lane beside the three sweeps
                       above, not a variant of them: each pair is evaluated once and
                       applied +f/-f, so momentum cancels algebraically. Reaches
                       operators/ and pallas/, never runtime/. Faced by
                       nornax_adapter.py (BlockStepFMM), documented in
                       docs/momentum_conserving_fmm.md
jaccpot/pallas/        fused Pallas kernels + custom_vjp
jaccpot/runtime/       orchestration, config resolution, lane selection, kernel dispatch
jaccpot/distributed/   domain decomposition, halo exchange, collectives
jaccpot/experimental/  octree/treecode prototypes — NOT production, opt-in marker only

tests/unit/            does the function do what its docstring says
tests/integration/     end-to-end paths
tests/characterization/ golden references — the tripwire for silent numerics changes
                       (forward accelerations + gradients, each with an inertness
                       gate and a direct-sum physics anchor)
tests/distributed/     multi-GPU; every file skips below 2 devices, so CPU CI collects
                       and skips them. Not a tier you can rely on locally.
tests/perf/            performance assertions
tests/experimental/    prototypes; deselected by default

No test files live directly under `tests/` — only `conftest.py` and `slow_tests.txt`.
`slow_tests.txt` marks tests `slow` **by node id**, so moving or renaming a test file
silently un-marks its entries and pushes them into the smoke leg; update it in the same
change and check the collected counts under `-m "not slow"` are unchanged.

bench/                 profiling, audits, microbenchmarks, ci_benchmark_guard.py
docs/                  design notes, audits, profiling records
```

Public API surface: `jaccpot/__init__.py`. Everything exported there is a contract.

Sibling codebases sharing these conventions: `astronomix`, `yggdrax`, `nornax`.

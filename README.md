# paper/jaccpot-i (branch scaffold)

Code, benchmarks, and tests for the Jaccpot I methods paper. Kept on this
branch so paper-specific work doesn't tangle with the maintained `jaccpot`
library surface. See `PROJECT_PLAN.md` for the full phase-by-phase task
breakdown. The manuscript itself lives in the separate `jaccpot-paper-i`
repo — see that repo's README for the Overleaf sync.

## Setting this up as a branch

```bash
cd jaccpot                       # your existing clone
git checkout -b paper/jaccpot-i
# copy this scaffold's contents in:
cp -r /path/to/this/scaffold/* .
git add jaccpot/applications bench/multigpu bench/differentiability bench/payoff \
        bench/validation bench/scaling tests/unit tests/applications \
        docs/multigpu_differentiability_model.md examples/jaccpot_paper \
        PROJECT_PLAN.md README.md results/.gitignore
git commit -m "Scaffold Jaccpot I paper branch: plan, benchmarks, applications, tests"
```

## Conventions

- **Benchmarks/scaling/validation** = Python scripts in `bench/`, seeded,
  dump results as JSON to `results/`. No plotting logic in scripts.
- **Figures** = notebooks in `examples/jaccpot_paper/`, load from `results/`
  only, never recompute.
- **New application code** (the payoff case study) lives under
  `jaccpot/applications/potential_recovery/`, not the core `jaccpot/runtime/`
  public API.
- Core library changes needed along the way (e.g. the multi-GPU
  basis/MAC convergence in Phase 0) go in the normal `jaccpot/` package, same
  as any other feature work — they're not paper-specific even though this
  paper is what's motivating them.

## Running a benchmark

```bash
pip install -e .    # editable install, from the jaccpot repo root
python bench/multigpu/strong_scaling.py --n 10000000 --gpu-counts 1 2 4 8
```

Each script is standalone and argparse'd (`--help` for options) and writes
into `results/<category>/`. GPU scripts should use `autocvd` for free-GPU
selection per the org-wide JAX/GPU convention (see
`examples/run_in_odisseo_with_autocvd.py` at the repo root for the pattern).

## After running: get results into the paper repo

Copy (or symlink during active writing) the relevant `results/**/*.json`
files into `jaccpot-paper-i`'s `figures/` pipeline — see that repo's README
for exactly how notebooks there consume them, and update
`figures/README.md`'s provenance table.

## Current status (see PROJECT_PLAN.md for detail)

- **Phase 0** (multi-GPU basis/MAC convergence): blocking — check
  `docs/phase5_multigpu_pallas_foldin_plan.md`'s STATUS block before
  anything in Phase 2 is meaningful.
- **Phase 1** (validation + single-device scaling): mostly adapting existing
  `bench/` scripts at the repo root — no new engineering.
- **Phase 2** (multi-GPU scaling): blocked on `bench/multigpu/harness.py`
  (stub) — the main remaining engineering work.
- **Phase 3** (differentiability): blocked on
  `bench/differentiability/grad_bench_lib.py` (working timing/FD-check
  helpers, needs wiring to the actual FMM forward call).
- **Phase 4** (payoff case study): `jaccpot/applications/potential_recovery/`
  is fully stubbed, no implementation yet. Check `nornax` for an existing
  energy-conservation example before writing one from scratch.

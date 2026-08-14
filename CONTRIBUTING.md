# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### macOS

The command above **cannot resolve on macOS**. `jax[cuda]` is an unconditional
dependency and `jax-cuda12-plugin` has no darwin wheels, so pip fails with
`No matching distribution found for jax-cuda12-plugin`. Install the CPU stack
explicitly instead — this is the same JAX window `pyproject.toml` pins, and it is
what the CPU CI legs exercise:

```bash
pip install "jax>=0.10.2,<0.11" "jaxlib>=0.10.2,<0.11" "jaxtyping>=0.2.23" "beartype>=0.14.0" "black==26.5.1" "isort==9.0.0b1" "pydoclint==0.9.1" "pytest>=8.3.0" "pytest-cov>=5.0.0" "pytest-xdist>=3.6.0" "pre-commit>=3.8.0"
pip install -e . --no-deps
```

`yggdrax` is not on PyPI — install it from source first (`pip install
git+https://github.com/TobiBu/yggdrax.git`, or `pip install -e` a local checkout).
Keep `black`, `isort`, and `pydoclint` at those exact pins: they equal the
`.pre-commit-config.yaml` hook revs, and a floating `>=` makes local runs and the
hook disagree.

`pytest-xdist` is not optional in practice. The suite is compilation-bound and
`addopts` supplies `-n auto`; serially it takes hours rather than ~15 minutes.

## Local quality checks

Run these before opening a pull request:

```bash
black --check .
isort --check-only .
pydoclint --config pyproject.toml jaccpot/
pytest
```

`pydoclint` also runs as a pre-commit hook, so it blocks the commit rather than the PR. The
`--config pyproject.toml` form above reproduces the hook exactly.

## Pre-commit

Install and enable hooks once:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks on demand:

```bash
pre-commit run --all-files
```

## Testing and coverage

CI enforces coverage through `pytest-cov`.

```bash
pytest --cov=jaccpot --cov-report=term-missing
```

Runtime type checks (`jaxtyping` + `beartype`) are available via import-hook
instrumentation. To enable during debugging:

```bash
export JACCPOT_RUNTIME_TYPECHECK=1
```

## Pull requests

- Keep changes focused and scoped.
- Include tests for behavior changes.
- Update README and docs when user-facing APIs change.

## Writing a validation handoff

Some validation cannot run where the work happens — the A100 Pallas parity run in
`ARCHITECTURE.md` §10 item 4 is the standing example — so it gets written up in
`docs/` as a handoff for someone with the right hardware. Four rules, each of
which has already cost a real run:

**Give every command the invariant it must satisfy, not just the command.** A
handoff that says "run `audit_reverse_residuals.py --n 4096 --leaf 64`" produced a
result covering **zero** M2L pairs: at that leaf size the MAC accepts none, so the
far-field reverse it was meant to measure never ran. The audit script said so in
its own output and the handoff still had to be re-run. Write "…and confirm
`far_pairs > 0`". This is the same discipline the tests already use — see the
`min_far` parameter in `tests/unit/test_gradient_correctness.py` and the vacuity
gates in `tests/characterization/test_fmm_grad_golden.py`.

**Pin commits, not branch names.** Branches get merged while a handoff waits. One
naming three branches to compare against `main` was executed after all three had
landed, so every comparison would have been a tree against itself plus unrelated
later work. Give SHAs, or define the baseline by rule ("the last commit before the
first Tier 1 merge") so it can be recovered later.

**Say which backend each claim is for.** `rtol=0` and exact equality mean
different things on CPU and GPU; see `ARCHITECTURE.md` §10.

**Ask for the skip list, not just the pass count.** A GPU-gated test that
self-skips is indistinguishable from one that passed, and several tests here carry
a defensive `pytest.skip` inside an `except` block. `-rA` and an explicit "report
which parametrisations ran" is the difference between a validated path and an
assumption.

Then record the result somewhere durable and link it from the audit, so the next
person inherits the measurements instead of re-deriving them.

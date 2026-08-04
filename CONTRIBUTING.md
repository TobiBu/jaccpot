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

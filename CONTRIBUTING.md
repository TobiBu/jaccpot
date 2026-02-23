# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Local quality checks

Run these before opening a pull request:

```bash
black --check .
isort --check-only .
pytest
```

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

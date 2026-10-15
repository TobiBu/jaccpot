"""A nornax on the path must carry every name jaccpot's cross-repo tests import.

nornax is a test-only dependency, and the conftest used to find it in exactly one
way: a sibling checkout, whatever commit that checkout happened to be on. On
2026-10-15 the shared sibling sat on 8fe9dbd, five commits behind nornax main, and
five cross-repo tests reported "installed nornax has no conformance kit" / "...
no topology hook" as SKIPPED -- green, and indistinguishable from a suite with
nothing to run.

`tests/conftest.py` closes that in two halves, both pinned here:

* ``NORNAX_WORKTREE`` names the checkout, as ``YGGDRAX_WORKTREE`` does for yggdrax;
* a nornax that is PRESENT is held to the names the tests import, derived from
  the test sources rather than listed, and the run stops naming what it lacks.
  An ABSENT nornax is still a skip -- every CI job but the cross-repo ones runs
  without it.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import textwrap

import pytest

from tests import conftest
from tests.conftest import (
    _TESTS_DIR,
    _check_nornax,
    _missing_symbols,
    _nornax_checkout,
    _nornax_symbols_the_tests_import,
)


def test_nornax_worktree_overrides_the_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The variable wins over the sibling lookup; unset, the sibling is used."""
    monkeypatch.setenv("NORNAX_WORKTREE", str(tmp_path))
    assert _nornax_checkout() == tmp_path
    monkeypatch.delenv("NORNAX_WORKTREE")
    assert _nornax_checkout() == conftest._find_sibling_checkout("nornax")


def test_the_scan_counts_imports_under_a_handler(tmp_path: pathlib.Path) -> None:
    """In the tests a handler around a nornax import is a skip, so it still counts.

    The yggdrax rule spares a guarded import, because there the handler is jaccpot
    degrading on purpose. Here the handler IS the silent path: the conformance kit
    was imported under ``except ImportError: pytest.skip(...)``.
    """
    (tmp_path / "test_mod.py").write_text(textwrap.dedent("""
            import nornax
            from yggdrax.dtypes import INDEX_DTYPE

            try:
                from nornax.solvers.leapfrog_kdk import block_kdk_rollout
            except Exception:
                block_kdk_rollout = None


            def kit():
                try:
                    from nornax.conformance import check_mutual_force_model
                except ImportError:
                    return None
                return check_mutual_force_model
            """))
    assert _nornax_symbols_the_tests_import(tmp_path) == {
        "nornax": set(),
        "nornax.solvers.leapfrog_kdk": {"block_kdk_rollout"},
        "nornax.conformance": {"check_mutual_force_model"},
    }


def test_the_scan_finds_what_the_cross_repo_tests_need() -> None:
    """Vacuity guard: the two names the stale sibling lacked are in the contract."""
    required = _nornax_symbols_the_tests_import(_TESTS_DIR)
    assert "check_mutual_force_model" in required.get("nornax.conformance", set())
    assert "block_kdk_rollout" in required.get("nornax.solvers.leapfrog_kdk", set())


def test_the_environment_satisfies_the_contract() -> None:
    """Whatever nornax this run resolved, if any, serves the tests that import it."""
    if importlib.util.find_spec("nornax") is None:
        pytest.skip("no nornax on the path: the cross-repo modules skip")
    assert _missing_symbols(_nornax_symbols_the_tests_import(_TESTS_DIR)) == []


@pytest.fixture
def fake_nornax(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path):
    """Put a nornax without a conformance kit first on the path, then restore.

    Yields
    ------
    pathlib.Path
        The fake package's parent, i.e. what ``NORNAX_WORKTREE`` would name.
    """
    root = tmp_path / "nornax-stale"
    (root / "nornax" / "solvers").mkdir(parents=True)
    (root / "nornax" / "__init__.py").write_text("")
    (root / "nornax" / "solvers" / "__init__.py").write_text("")
    (root / "nornax" / "solvers" / "leapfrog_kdk.py").write_text(
        "def block_kdk_rollout():\n    pass\n"
    )
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "nornax" or name.startswith("nornax.")
    }
    for name in saved:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("NORNAX_WORKTREE", str(root))
    try:
        yield root
    finally:
        for name in [
            n for n in sys.modules if n == "nornax" or n.startswith("nornax.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved)


def test_a_present_but_stale_nornax_stops_the_run(
    fake_nornax: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """The run stops naming the missing name and where nornax came from."""
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_x.py").write_text(
        "from nornax.solvers.leapfrog_kdk import block_kdk_rollout\n"
        "from nornax.conformance import check_mutual_force_model\n"
    )
    monkeypatch.setattr(conftest, "_TESTS_DIR", tests_root)
    with pytest.raises(pytest.UsageError) as excinfo:
        _check_nornax()
    message = str(excinfo.value)
    assert "nornax.conformance (ModuleNotFoundError" in message
    assert "block_kdk_rollout" not in message  # present, so not reported
    assert str(fake_nornax) in message
    assert "NORNAX_WORKTREE" in message


def test_an_absent_nornax_is_a_skip_not_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """No nornax at all is a legitimate environment: nothing is checked."""
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_x.py").write_text("from nornax.state import BlockStepState\n")
    monkeypatch.setattr(conftest, "_TESTS_DIR", tests_root)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: None if name == "nornax" else real_find_spec(name, *a),
    )
    _check_nornax()  # does not raise

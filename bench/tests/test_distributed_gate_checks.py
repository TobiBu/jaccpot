"""The two-card gate's own contract, checked without two cards.

`bench/distributed_gate.py` exists to refuse a vacuous green: a distributed run
that fell back to one device skips the whole tier and prints success. A gate for
that failure mode is worth very little if its own list of what-must-run has gone
stale, so this pins the parts that can be checked on any machine.

Its `_MUST_RUN` fails in the SAFE direction already -- a fragment that matches
nothing reports "0 tests ran" and fails the gate rather than passing it. What
this file adds is finding out at unit-test time rather than eighteen minutes into
a two-card run.

The output parser and failure classifier are `gpu_gate`'s and are covered by
`test_gpu_gate_checks.py`; only what this module adds is tested here.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("yggdrax")

from bench import distributed_gate  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_every_must_run_fragment_names_a_test_that_exists():
    """A renamed test must break here, not eighteen minutes into a card run."""
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (_REPO_ROOT / "tests" / "distributed").rglob("test_*.py")
    )
    missing = [f for f in distributed_gate._MUST_RUN if f"def {f}" not in sources]
    assert not missing, (
        f"{missing} appear in the gate's _MUST_RUN but no test defines them. "
        "The gate would report '0 tests ran' and fail for the wrong reason"
    )


def test_the_must_run_list_is_not_empty():
    """A vacuity guard on the vacuity guard.

    An empty list makes `_check_not_vacuous` return no complaints for any input,
    so the gate would pass a run in which nothing at all executed -- the exact
    shape of failure it was written to refuse.
    """
    assert distributed_gate._MUST_RUN


def test_the_targets_exist():
    """Both the default and `--full` targets must be real paths."""
    for target in (*distributed_gate._TARGET, *distributed_gate._FULL_TARGET):
        assert (_REPO_ROOT / target).exists(), f"{target} does not exist"


def test_a_run_where_the_gated_tests_skipped_is_reported_as_vacuous():
    """One device skips the tier and pytest prints green; that must not pass."""
    skipped = "\n".join(
        f"tests/distributed/test_distributed_grad_correctness.py::{name} SKIPPED"
        for name in distributed_gate._MUST_RUN
    )
    complaints = distributed_gate._check_not_vacuous(skipped)
    assert len(complaints) == len(distributed_gate._MUST_RUN)


def test_a_run_where_they_ran_is_accepted():
    """The same shape, with verdicts, must produce no complaint."""
    ran = "\n".join(
        f"tests/distributed/test_distributed_grad_correctness.py::{name} PASSED"
        for name in distributed_gate._MUST_RUN
    )
    assert distributed_gate._check_not_vacuous(ran) == []

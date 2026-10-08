"""Optionally trim expensive parametrisations for a quick local typecheck run.

WHY THIS EXISTED. ``test-runtime-typecheck`` used to run the whole of
``tests/unit`` a second time, with jaxtyping and beartype enforcement, at ``-n 2``
under a 60 minute cap, and it took 56-58 minutes on ``main`` (2026-09-08 to
09-10). This helper shrank a handful of interpret-mode Pallas grids to one case
there so the job fitted.

WHAT IT DOES NOW. CI no longer has that job: runtime type checks ride on the
sharded ``test-full`` run, which runs every grid in full, under the checks, once.
So nothing in CI trims. ``JACCPOT_TEST_TRIM_GRIDS=1`` keeps the old behaviour as
an explicit opt-in for a quick local pass (``JACCPOT_RUNTIME_TYPECHECK=1
JACCPOT_TEST_TRIM_GRIDS=1 pytest tests/unit``): every test still RUNS and every
signature is still called, only the grid shrinks to its first case. It is keyed on
its own variable, not on the typecheck switch, so turning the checks on never
quietly drops cases.

WHY NOT SHRINK THE PROBLEM INSTEAD. Tried and rejected: interpret-mode Pallas
cost tracks the PAIR count, not the particle count, so 800 particles at leaf 4
ran slower than 4000 at leaf 16 (more leaves, 43k far pairs). Shrinking N also
walks into vacuity -- at leaf 16 a bucket tree of 1500 particles has zero far
pairs at theta 0.5, so the test would compare two empty lists and pass.
"""

from __future__ import annotations

import os
from typing import Sequence, TypeVar

__all__ = ["TRIM_GRIDS", "trim"]

T = TypeVar("T")

#: True when ``JACCPOT_TEST_TRIM_GRIDS=1`` (a local opt-in; never set in CI).
TRIM_GRIDS = os.environ.get("JACCPOT_TEST_TRIM_GRIDS") == "1"


def trim(values: Sequence[T], *, keep: int = 1) -> list[T]:
    """The full sequence normally; its first ``keep`` entries when trimming.

    Parameters
    ----------
    values : Sequence[T]
        Parametrisation values, most representative FIRST -- that is the one a
        trimmed run keeps.
    keep : int
        How many to keep when ``JACCPOT_TEST_TRIM_GRIDS=1``.

    Returns
    -------
    list[T]
        ``list(values)``, or its first ``keep`` entries.
    """
    return list(values[: int(keep)]) if TRIM_GRIDS else list(values)

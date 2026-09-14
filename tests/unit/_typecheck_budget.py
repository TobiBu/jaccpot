"""Trim expensive parametrisations under ``JACCPOT_RUNTIME_TYPECHECK=1``.

WHY. ``test-runtime-typecheck`` runs the whole of ``tests/unit`` with jaxtyping
and beartype enforcement at ``-n 2`` under a 60 minute cap, and on ``main`` it
already takes **56-58 minutes** (measured from three consecutive runs, 2026-09-08
to 09-10). It passes by under two minutes. Anything a branch adds there lands on
a job with no headroom: the sub-10 ms branch added roughly five minutes of
interpret-mode Pallas tests and the job hit the cap at 92 %.

WHAT THIS DOES NOT DO. It does not skip tests. That job exists to catch calls
that die at the signature -- including two vacuous tests that reported coverage
while asserting nothing -- so every test still RUNS there and every signature is
still called. Only the grid shrinks: one case instead of the full product. The
other jobs (``test-full``, ``test-mac-runtime``, ``test-smoke``) run the whole
grid for its results, which is what they are for.

WHY NOT SHRINK THE PROBLEM INSTEAD. Tried and rejected: interpret-mode Pallas
cost tracks the PAIR count, not the particle count, so 800 particles at leaf 4
ran slower than 4000 at leaf 16 (more leaves, 43k far pairs). Shrinking N also
walks into vacuity -- at leaf 16 a bucket tree of 1500 particles has zero far
pairs at theta 0.5, so the test would compare two empty lists and pass.
"""

from __future__ import annotations

import os
from typing import Sequence, TypeVar

__all__ = ["RUNTIME_TYPECHECK", "trim"]

T = TypeVar("T")

#: True inside the ``test-runtime-typecheck`` job.
RUNTIME_TYPECHECK = os.environ.get("JACCPOT_RUNTIME_TYPECHECK") == "1"


def trim(values: Sequence[T], *, keep: int = 1) -> list[T]:
    """The full sequence normally; its first ``keep`` entries under typechecking.

    Parameters
    ----------
    values : Sequence[T]
        Parametrisation values, most representative FIRST -- that is the one the
        typecheck job keeps.
    keep : int
        How many to keep there.

    Returns
    -------
    list[T]
        ``list(values)``, or its first ``keep`` entries.
    """
    return list(values[: int(keep)]) if RUNTIME_TYPECHECK else list(values)

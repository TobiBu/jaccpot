"""The accuracy verification suite's library half (plan A4).

* :mod:`jaccpot.verification.reference` -- the distributed reference force: an fp64
  direct sum on a target subset chosen by particle id, accumulated EXACTLY in integer
  limbs, so it is the same number on one card, N cards and N processes.
* :mod:`jaccpot.verification.baseline` -- the result schema and the ``record`` /
  ``check`` comparison later tracks must hold.

The standard problems, ladders and the HoreKa jobs live in ``bench/accuracy/`` and
``bench/slurm/a4/``; how to run them is in ``docs/accuracy_suite.md``.
"""

from __future__ import annotations

from jaccpot.verification.baseline import (
    SCHEMA_VERSION,
    CheckOutcome,
    Tolerance,
    check_result,
    load_result,
    make_result,
    save_result,
)
from jaccpot.verification.reference import (
    ForceLaw,
    ReferenceForce,
    force_law_of,
    gather_by_id,
    reference_force,
    score_force,
    score_mesh_result,
    score_single_result,
    select_targets,
)

__all__ = [
    "SCHEMA_VERSION",
    "CheckOutcome",
    "ForceLaw",
    "ReferenceForce",
    "Tolerance",
    "check_result",
    "force_law_of",
    "gather_by_id",
    "load_result",
    "make_result",
    "reference_force",
    "save_result",
    "score_force",
    "score_mesh_result",
    "score_single_result",
    "select_targets",
]

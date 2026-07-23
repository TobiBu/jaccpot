"""Harness runs end-to-end on tiny N / 1 GPU in CI.

This is a smoke test of bench/multigpu/harness.py's correctness, not a real
scaling measurement -- keep N tiny so this is cheap enough for every CI run.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason="Fill in once bench/multigpu/harness.py's run_once is implemented."
)
def test_harness_runs_on_tiny_n_one_gpu():
    raise NotImplementedError

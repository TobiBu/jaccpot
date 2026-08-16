"""The GPU radix benchmark worker's internal-symbol bindings must resolve.

``examples/benchmark_gpu_radix_worker.py`` drives runtime internals directly, so
it binds a fixed set of private symbols up front. That binding rotted silently:
the Tier 1.6 split of ``kernels/core.py`` moved six M2L/L2L entries out of
``jaccpot.runtime._fmm_impl``, the worker was reading them off that module as
attributes, and ``_load_jaccpot_internal_symbols`` began raising
``AttributeError`` before producing any output.

Nothing caught it because the only caller
(``bench/guard_large_n_radix_fast_lane.py``) is GPU-only and outside CI. This
test is CPU-only and imports nothing heavy -- it just resolves the bindings, so
the next move of a private symbol fails here instead of on someone's GPU run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def worker_module():
    sys.path.insert(0, str(_EXAMPLES))
    try:
        import benchmark_gpu_radix_worker as worker

        return worker
    finally:
        sys.path.remove(str(_EXAMPLES))


def test_worker_internal_symbols_all_resolve(worker_module):
    """Every symbol the worker binds must exist and be non-None."""
    symbols = worker_module._load_jaccpot_internal_symbols()

    assert symbols, "the worker binds no symbols at all"
    unbound = sorted(name for name, value in symbols.items() if value is None)
    assert not unbound, f"bound to None: {unbound}"


def test_worker_binds_the_m2l_and_l2l_entries_that_moved(worker_module):
    """Pin the six that the core.py split relocated, by name.

    Named explicitly rather than covered by the count above: these are the ones
    that actually broke, and a future move should fail on the specific symbol
    rather than on an arity check.
    """
    symbols = worker_module._load_jaccpot_internal_symbols()

    for name in (
        "_accumulate_m2l_chunked_scan",
        "_accumulate_m2l_fullbatch",
        "_accumulate_solidfmm_m2l_grouped",
        "_accumulate_solidfmm_m2l_grouped_class_major",
        "_propagate_real_locals_to_children",
        "_propagate_solidfmm_locals_to_children",
    ):
        assert callable(symbols[name]), name

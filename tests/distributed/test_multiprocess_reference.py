"""A4: the reference force on N processes x 1 device is bitwise 1 process x N devices.

``_reference_scenarios.reference`` on both sides of ``_multiprocess_harness``
(gloo collectives on localhost CPUs): the reference arrays (two kernels) are
compared bit for bit, and with them the score of the runner's multi-GPU mode
(``DirectMeshForce``) computed without a gather.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest -n 0 -m distributed_multiprocess \\
        tests/distributed/test_multiprocess_reference.py
"""

from __future__ import annotations

import os

import pytest

from tests.distributed import _multiprocess_harness as harness

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

pytestmark = pytest.mark.distributed_multiprocess


@pytest.mark.parametrize("n", [2, 4])
def test_the_reference_is_bitwise_across_processes(n: int, tmp_path) -> None:
    if n == 4 and os.environ.get("JACCPOT_RUN_MANUAL_TESTS") != "1":
        pytest.skip("4 processes: manual (JACCPOT_RUN_MANUAL_TESTS=1)")
    sp, mp, sp_json, mp_json = harness.run_both(
        "tests.distributed._reference_scenarios",
        "reference",
        n,
        str(tmp_path / f"reference-{n}"),
        timeout=1800.0,
    )
    differ = harness.differing(sp, mp)
    assert not differ, f"elements differing (sp vs mp): {differ}"
    assert set(sp) == set(mp) and len(sp) == 6
    assert sp_json == mp_json, (sp_json, mp_json)
    assert sp_json["runner_score"]["rel_l2"] < 1e-13

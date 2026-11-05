"""M2 on GPUs (manual): N processes x 1 GPU are bitwise 1 process x N GPUs.

The CPU gate (``test_multiprocess_runtime.py``) on cards: the same harness and
scenarios with ``platform="gpu"``. Both sides run on the first two cards of
``CUDA_VISIBLE_DEVICES`` -- select them with autocvd, never by hand -- in the
same order; process ``r`` owns card ``r`` (``JAX_LOCAL_DEVICE_IDS``). Every case
first repeats each side (the A-vs-A control: a float difference between the
sides means nothing unless each side reproduces itself), then compares them.

The XLA flags are the production ones (the fusion autotuner off, XLA's NCCL
ragged fallback with ``use_barrier_with_nccl=false``). Measured 2026-11-05 on
two shared A100-PCIE: every case below reproduces itself on them (A-vs-A clean)
and the two sides agree bitwise; ``JACCPOT_MP_GPU_DETERMINISTIC=1`` adds
``--xla_gpu_deterministic_ops=true`` (for sizes whose scatter-adds do not
reproduce on the production flags).

    eval "$(autocvd -q -e -l -n 2)"   # two free cards
    JACCPOT_RUN_MANUAL_TESTS=1 pytest -n 0 -m distributed_multiprocess \\
        tests/distributed/test_multiprocess_gpu.py

Each case takes minutes (two to four fresh processes compile the lane).
"""

from __future__ import annotations

import os

import pytest

from tests.distributed import _multiprocess_harness as harness

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

_CARDS = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c]

pytestmark = [
    pytest.mark.distributed_multiprocess,
    pytest.mark.skipif(
        os.environ.get("JACCPOT_RUN_MANUAL_TESTS") != "1",
        reason="manual: needs two GPUs; set JACCPOT_RUN_MANUAL_TESTS=1",
    ),
    pytest.mark.skipif(
        len(_CARDS) < 2, reason="CUDA_VISIBLE_DEVICES must list two cards (autocvd)"
    ),
]

SCENARIOS = "tests.distributed._multiprocess_scenarios"

#: the production flags (``jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS``)
PROD = (
    "--xla_gpu_experimental_enable_fusion_autotuner=false "
    "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false "
    "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true "
    "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
)
if os.environ.get("JACCPOT_MP_GPU_DETERMINISTIC") == "1":
    PROD += " --xla_gpu_deterministic_ops=true"

#: JSON values that are per process by construction
_LOCAL = ("peak_bytes",)


def _side(mode, name, out, env, timeout):
    harness.run_side(
        mode, SCENARIOS, name, 2, out, timeout=timeout, env=env, platform="gpu"
    )
    return harness.load_side(out, mode)


def _same(a, b, what):
    (xa, ja), (xb, jb) = a, b
    differ = harness.differing(xa, xb)
    assert not differ, f"{what}: elements differing {differ}"
    for key in sorted((set(ja) | set(jb)) - set(_LOCAL)):
        assert ja.get(key) == jb.get(key), (what, key, str(ja.get(key))[:1500])


def _gate(name, env, tmp_path, timeout=3600.0):
    out = {}
    for tag, mode in (("sp_a", "sp"), ("sp_b", "sp"), ("mp_a", "mp"), ("mp_b", "mp")):
        out[tag] = _side(mode, name, str(tmp_path / tag), env, timeout)
    _same(out["sp_a"], out["sp_b"], "control: 1 process x 2 GPUs, twice")
    _same(out["mp_a"], out["mp_b"], "control: 2 processes x 1 GPU, twice")
    _same(out["sp_a"], out["mp_a"], "1 process x 2 GPUs vs 2 processes x 1 GPU")
    return out["sp_a"][1]


@pytest.mark.parametrize("ragged", ["native", "exchange"])
def test_collectives_across_processes(ragged, tmp_path):
    """psum / pmax / pmin / all_gather / all_to_all / ppermute / ragged_all_to_all
    across two processes: the host references, and the single process's bits."""
    values = _gate(
        "collectives", {"XLA_FLAGS": PROD, "M2_RAGGED": ragged}, tmp_path, 900.0
    )
    for call in range(3):
        assert all(values[f"call{call}_ok"].values()), values[f"call{call}_ok"]
        assert values[f"call{call}_ragged_rows_wrong"] == 0


TREE_REPLAN = {
    "M2_MAC": "dehnen_error",
    "M2_MIGRATE": "1",
    "M2_LEAF_CAPACITY": "44",
    "M2_IC": "collapse",
}

CASES = [
    pytest.param("direct_runner", {}, PROD, id="direct"),
    pytest.param(
        "fused_runner", {"M2_MAC": "dehnen", "M2_MIGRATE": "1"}, PROD, id="dehnen-2048"
    ),
    pytest.param("fused_runner", TREE_REPLAN, PROD, id="dehnen_error-treereplan-2048"),
    pytest.param(
        "fused_runner_large", {"M2_MAC": "dehnen"}, PROD, id="dehnen-2e5-per-gpu"
    ),
    pytest.param(
        "fused_runner_large",
        {"M2_MAC": "dehnen_error"},
        PROD,
        id="dehnen_error-2e5-per-gpu",
    ),
]


@pytest.mark.parametrize("name,env,flags", CASES)
def test_runner_across_processes(name, env, flags, tmp_path):
    """The mesh runner: the start force, migrating segments with repartitions,
    a cross re-plan and retry (or a tree re-plan), the energy at every edge."""
    values = _gate(name, dict(env, XLA_FLAGS=flags), tmp_path)
    if name != "direct_runner":
        retries = [r["retries"] for r in values["reports"]]
        assert retries[0] >= 1, retries  # a capacity fired and was re-planned
        # from the first segment's edge on (the start's potential overflows the
        # undersized cross capacity, edge 0 then reports none: also in one process)
        energies = [e["diagnostics"].get("energy") for e in values["edges"][1:]]
        assert all(v is not None for v in energies), energies

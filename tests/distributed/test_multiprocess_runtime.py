"""M2: the multi-GPU mode as N processes with one device each, on localhost CPUs.

``jaccpot.distributed.multiprocess`` lets ``ProductionRunner(mesh=MeshRunConfig(
...))`` run under ``jax.distributed`` (one process per device) instead of one
process driving every device. The gate: **N processes x 1 device are bitwise 1
process x N devices** (``--xla_force_host_platform_device_count=N``), by id, on
the same machine -- at 2 and 4 processes (4 matters: from 4 on, gloo's float
all-reduce reduces in another order than XLA:CPU's in-process one, which the
mesh-order fold of the diagnostics removes).

Each side runs in fresh child processes (``_multiprocess_harness``: a free
localhost port, gloo, a timeout, every child's process group killed on failure);
the scenarios are ``_multiprocess_scenarios``. The single process passes the
whole initial condition, each of the N processes only its id range
(``ProcessLocalParticles``).

* the helpers: ``mesh_sum`` / ``mesh_moments`` / ``host_value`` and
  ``decompose_local`` == ``decompose``;
* the runner with ``DirectMeshForce``: migrating and not, a repartition before
  every segment, the energy and the moments at every edge, a continuation in
  memory; a restart file refused under several processes (M3: orbax);
* the runner with ``FusedMeshForce`` (slow), the energy at every edge: both MACs
  migrating with a cross capacity too small for the first segment (re-planned
  and retried), and ``dehnen_error`` on the A3 start whose trees outgrow a leaf
  capacity of 44 (a TREE re-plan, two devices) with the library's cross sizing
  at the start and the drift. The 4-process cases are manual
  (``JACCPOT_RUN_MANUAL_TESTS=1``).

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest -n 0 -m distributed_multiprocess \\
        tests/distributed/test_multiprocess_runtime.py
"""

from __future__ import annotations

import os

import pytest

from tests.distributed import _multiprocess_harness as harness

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

pytestmark = pytest.mark.distributed_multiprocess

SCENARIOS = "tests.distributed._multiprocess_scenarios"


def _run(name, n, tmp_path, env=None, timeout=3600.0):
    return harness.run_both(
        SCENARIOS, name, n, str(tmp_path / f"{name}-{n}"), timeout=timeout, env=env
    )


def _assert_same(sp, mp, sp_json, mp_json, skip=()):
    differ = harness.differing(sp, mp)
    assert not differ, f"elements differing (sp vs mp): {differ}"
    keys = (set(sp_json) | set(mp_json)) - set(skip)
    for key in sorted(keys):
        assert sp_json.get(key) == mp_json.get(key), (
            key,
            str(sp_json.get(key))[:2000],
            str(mp_json.get(key))[:2000],
        )


@pytest.mark.parametrize("n", [2, 4])
def test_helpers_are_the_single_process_values(n, tmp_path):
    """The mesh-order sum and moments, ``host_value`` and the decomposition from
    each process's id range are the single process's, bitwise."""
    sp, mp, sp_json, mp_json = _run("helpers", n, tmp_path, timeout=900.0)
    assert sp_json["allgather_object_ok"] and mp_json["allgather_object_ok"]
    _assert_same(sp, mp, sp_json, mp_json)


@pytest.mark.parametrize("n", [2, 4])
def test_direct_force_runner_is_the_single_process_run(n, tmp_path):
    """The runner's mesh mode with the direct force: state, acceleration and owner
    by id, every edge's diagnostics (energy, moments) and plan record, every
    segment report -- equal; restart files refused under several processes."""
    sp, mp, sp_json, mp_json = _run("direct_runner", n, tmp_path, timeout=900.0)
    assert sp_json["restart_refused"] == "one process"
    assert mp_json["restart_refused"] == "refused", mp_json["restart_refused"]
    # the energy is reported at every edge (the mesh-order fold is under test)
    for tag in ("migrate0", "migrate1"):
        energies = [e["diagnostics"]["energy"] for e in mp_json[f"{tag}_edges"]]
        assert all(v is not None for v in energies), energies
    _assert_same(sp, mp, sp_json, mp_json, skip=("restart_refused",))


#: the criterion on the A3 start: leaf capacity 44 outgrown (a tree re-plan), the
#: library's cross capacities (sized at the start and at the drift)
_TREE_REPLAN = {
    "M2_MAC": "dehnen_error",
    "M2_MIGRATE": "1",
    "M2_LEAF_CAPACITY": "44",
    "M2_IC": "collapse",
}

#: four processes compiling the fused lane at once: 16-25 min a case on a loaded
#: 64-core host, more than a hosted runner's four cores do inside the job's cap
_MANUAL = pytest.mark.skipif(
    os.environ.get("JACCPOT_RUN_MANUAL_TESTS") != "1",
    reason="manual: four fused-lane processes; set JACCPOT_RUN_MANUAL_TESTS=1",
)

FUSED_CASES = [
    pytest.param({"M2_MAC": "dehnen", "M2_MIGRATE": "1"}, 2, id="dehnen-migrate-2"),
    pytest.param(
        {"M2_MAC": "dehnen", "M2_MIGRATE": "1"},
        4,
        id="dehnen-migrate-4",
        marks=_MANUAL,
    ),
    pytest.param(_TREE_REPLAN, 2, id="dehnen_error-migrate-treereplan-2"),
    # on four devices the A3 start overflows a leaf capacity of 44 already at the
    # prepare (one process or four), so the criterion's 4-process case is the
    # cube with the undersized cross capacity
    pytest.param(
        {"M2_MAC": "dehnen_error", "M2_MIGRATE": "1"},
        4,
        id="dehnen_error-migrate-4",
        marks=_MANUAL,
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("env,n", FUSED_CASES)
def test_fused_force_runner_is_the_single_process_run(env, n, tmp_path):
    """The fused mesh force through the runner, with its re-plans and retries:
    bitwise by id, the same retries, the same re-planned capacities."""
    sp, mp, sp_json, mp_json = _run("fused_runner", n, tmp_path, env=env)
    retries = [r["retries"] for r in sp_json["reports"]]
    assert retries[0] >= 1, retries  # a capacity fired and was re-planned
    if env.get("M2_IC") == "collapse":
        grown = [e["plan"]["leaf_capacity"] for e in sp_json["edges"][1:]]
        assert grown[0] > 44, grown  # the tree re-plan grew the leaf capacity
    _assert_same(sp, mp, sp_json, mp_json)

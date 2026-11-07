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


def test_writers_write_on_process_zero(tmp_path):
    """``SnapshotWriter`` / ``DiagnosticsLog`` write their files on process 0 only
    (the same files as one process writes), keep their rows on every process, and
    with ``all_processes=True`` write on each."""
    got = {}
    for mode in ("sp", "mp"):
        out = tmp_path / mode
        harness.run_side(
            mode,
            SCENARIOS,
            "writers",
            2,
            str(out),
            timeout=900.0,
            env={"M2_WRITER_DIR": str(out)},
        )
        got[mode] = harness.load_side(str(out), mode)
    (sp, sp_json), (mp, mp_json) = got["sp"], got["mp"]
    assert not harness.differing(sp, mp)
    assert sp_json["snapshot_files"] == mp_json["snapshot_files"]
    assert len(sp_json["snapshot_files"]) == 3  # segments 0, 1, 2
    assert sp_json["log_rows"] == mp_json["log_rows"]
    assert len(mp_json["log_rows"]) == 3
    p0, p1 = mp_json["kept"]
    assert p0["paths"] and not p1["paths"]  # only process 0 wrote
    assert p0["rows"] == p1["rows"] == 3  # every process keeps the rows
    assert p0["paths_all"] and p1["paths_all"]  # all_processes: each wrote
    assert mp_json["own_dirs"] == ["p0", "p1"]
    assert sp_json["own_dirs"] == ["p0"]


def test_initialize_from_a_slurm_step(tmp_path):
    """``initialize()`` with no arguments under a SLURM step's variables (jax's
    SLURM detection): every process joins and a psum over the mesh works."""
    out = str(tmp_path / "slurm")
    harness.run_side("mp", SCENARIOS, "launch", 2, out, timeout=600.0, launcher="slurm")
    arrays, values = harness.load_side(out, "mp")
    assert [i["process"] for i in values["info"]] == [0, 1]
    assert all(i["count"] == 2 for i in values["info"])
    assert arrays["psum"].tolist() == [1.0]


@pytest.mark.parametrize(
    "visible,local_ids,want",
    [("3", None, [0]), ("2,3", None, None), (None, None, None), ("3", "0", None)],
)
def test_slurm_bound_device_ids(monkeypatch, visible, local_ids, want):
    """A SLURM task bound to one card gets ``local_device_ids=[0]`` (jax's own
    ``[SLURM_LOCALID]`` names a missing device from local rank 1 on)."""
    from jaccpot.distributed import multiprocess as mp

    for name, value in (
        ("SLURM_JOB_ID", "1"),
        ("SLURM_STEP_NODELIST", "n1"),
        ("SLURM_NTASKS", "4"),
        ("SLURM_PROCID", "1"),
        ("SLURM_LOCALID", "1"),
    ):
        monkeypatch.setenv(name, value)
    for name, value in (
        ("CUDA_VISIBLE_DEVICES", visible),
        ("JAX_LOCAL_DEVICE_IDS", local_ids),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert mp.slurm_bound_device_ids() == want
    monkeypatch.delenv("SLURM_LOCALID")
    assert mp.slurm_bound_device_ids() is None  # not a SLURM step


ESCAPER_CASES = [
    pytest.param({}, 2, id="direct-2"),
    pytest.param({}, 4, id="direct-4"),
    # the fused lane: minutes of compile a process
    *(
        pytest.param(
            {"MESC_FORCE": "fused", "MESC_MAC": mac},
            2,
            id=f"fused-{mac}-2",
            marks=_MANUAL,
        )
        for mac in ("dehnen", "dehnen_error")
    ),
]


@pytest.mark.parametrize("env,n", ESCAPER_CASES)
def test_escaper_runner_is_the_single_process_run(env, n, tmp_path):
    """The escaper path on the mesh (MESC), escapers leaving and returning at
    several edges, the run continued in memory: state, acceleration and owner by
    id, the tail's layout, the split and ledger, every edge's diagnostics --
    equal."""
    sp, mp, sp_json, mp_json = _run(
        "escaper_runner", n, tmp_path, env=env, timeout=3600.0 if env else 900.0
    )
    counts = [e["diagnostics"]["n_escapers"] for e in mp_json["edges"]]
    assert counts[0] == 1 and max(counts) > 1, counts
    assert mp_json["split"]["transfers"] > 1
    _assert_same(sp, mp, sp_json, mp_json)

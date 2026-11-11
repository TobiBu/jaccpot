"""M3: sharded checkpoints of the multi-GPU mode across devices AND processes.

The gates of the multi-node plan's track M3 on localhost CPUs (each side in fresh
child processes, ``_multiprocess_harness``): ``sp`` = one process with ``n`` forced
devices, ``mp`` = ``n`` processes with one device each (``jax.distributed``, gloo).
The scenarios are ``_checkpoint_scenarios`` (ESC's Plummer sphere with eight
escaping fliers, ``DirectMeshForce``, migrating, the energy at every edge, a
sharded checkpoint at every edge).

* **Same count** (gate 1): the run restarted from its middle checkpoint by a fresh
  runner, and a chain of three runs each restarted from the previous one's last
  checkpoint, equal the run in one go BITWISE (state, carried acceleration, every
  edge's diagnostics, the escaper ledger), with escapers on and off; and ``sp`` ==
  ``mp`` bitwise throughout.
* **Across process counts**: a checkpoint written by 2 processes restarts in 1
  process with 2 devices (and the other way) bitwise.
* **Elastic** (gate 2): written on 2 devices, restarted on 4 (``sp`` and ``mp``,
  equal to each other bitwise), and back from 4 to 2; ``DirectMeshForce`` sums in
  global-id order, so the continuation is the uninterrupted run up to the
  escapers' mesh-order coupling sums (round-off).
* **Snapshots**: ``SnapshotWriter(sharded=True)`` read back by id equals each
  edge's gathered state, from one process or two.

orbax is the ``io`` extra: the tests skip without it, except where
``JACCPOT_CHECKPOINT_TESTS_REQUIRE_ORBAX=1`` (the CI job) makes its absence an error.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 pytest -n 0 -m distributed_checkpoint \\
        tests/distributed/test_checkpoint_multiprocess.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tests.distributed import _multiprocess_harness as harness

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")
if os.environ.get("JACCPOT_CHECKPOINT_TESTS_REQUIRE_ORBAX") != "1":
    pytest.importorskip("orbax.checkpoint", reason="needs the io extra (orbax)")

pytestmark = pytest.mark.distributed_checkpoint

SCENARIOS = "tests.distributed._checkpoint_scenarios"
MID = 3
#: escapers on and off
_ESC = [pytest.param("1", id="escapers"), pytest.param("0", id="no-escapers")]


def _load(out: str, mode: str) -> tuple:
    with np.load(os.path.join(out, f"{mode}.npz")) as data:
        arrays = {k: data[k] for k in data.files}
    with open(os.path.join(out, f"{mode}.json")) as fh:
        values = json.load(fh)
    return arrays, values


def _side(mode, name, n, out, env, timeout=1800.0):
    harness.run_side(mode, SCENARIOS, name, n, out, env=env, timeout=timeout)
    return _load(out, mode)


def _same(a, b, what):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    differ = int(np.sum(a != b))
    assert differ == 0, f"{what}: {differ} of {a.size} elements differ"


@pytest.fixture(scope="module", params=_ESC)
def written(request, tmp_path_factory):
    """Both sides of ``write_and_restart`` on two devices; the checkpoints stay."""
    root = str(tmp_path_factory.mktemp(f"m3-{request.param}"))
    env = {"M3_ROOT": root, "M3_ESC": request.param}
    sides = {
        mode: _side(
            mode, "write_and_restart", 2, os.path.join(root, f"out-{mode}"), env
        )
        for mode in ("sp", "mp")
    }
    return root, env, sides


def test_a_restart_and_a_chain_are_the_run_in_one_go(written):
    _, env, sides = written
    for mode, (arrays, values) in sides.items():
        assert values["restarts"] == [f"once_{k}" for k in range(1, 7)], mode
        for tag in ("resumed", "chained"):
            restored = values[f"{tag}_restored"]
            assert restored["bitwise"], (mode, tag, restored)
            assert restored["layout"] == "checkpoint", (mode, tag, restored)
            for name in ("state", "acceleration", "force_scale"):
                if f"once_{name}" in arrays:
                    _same(
                        arrays[f"{tag}_{name}"],
                        arrays[f"once_{name}"],
                        f"{mode} {tag} {name}",
                    )
        once = values["once_edges"]
        assert values["resumed_edges"] == once[MID + 1 :], mode
        assert values["chained_edges"] == once[5:], mode
        if env["M3_ESC"] == "1":
            assert values["resumed_split"] == values["once_split"], mode
            assert values["chained_split"] == values["once_split"], mode
            assert any(row["n_escapers"] for row in once), "nobody escaped"


def test_one_process_and_n_processes_write_and_restart_alike(written):
    _, _, sides = written
    (sp, sp_json), (mp_, mp_json) = sides["sp"], sides["mp"]
    differ = harness.differing(sp, mp_)
    assert not differ, f"elements differing (sp vs mp): {differ}"
    assert sp_json == mp_json


def test_a_restart_in_another_process_count_is_bitwise(written, tmp_path):
    root, env, sides = written
    for writer, reader in (("mp", "sp"), ("sp", "mp")):
        out = str(tmp_path / f"{writer}-to-{reader}")
        got, values = _side(
            reader,
            "restore_from",
            2,
            out,
            dict(env, M3_FROM=os.path.join(root, writer, f"once_{MID}")),
        )
        assert values["restored_restored"]["bitwise"], values
        assert values["warned"] == []
        once = sides[writer][0]
        _same(got["restored_state"], once["once_state"], f"{writer}->{reader} state")
        assert values["restored_edges"] == sides[writer][1]["once_edges"][MID + 1 :]


def _gap(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def test_an_elastic_restart_continues_the_run(written, tmp_path):
    """2 -> 4 devices (one process and four alike, bitwise) and 4 -> 2."""
    root, env, sides = written
    once = sides["sp"][0]["once_state"]
    scale = float(np.max(np.abs(once)))
    four = {}
    for mode in ("sp", "mp"):
        got, values = _side(
            mode,
            "restore_from",
            4,
            str(tmp_path / f"to4-{mode}"),
            dict(env, M3_FROM=os.path.join(root, "sp", f"once_{MID}")),
        )
        restored = values["restored_restored"]
        assert not restored["bitwise"] and "mesh_layout" in restored["mismatches"]
        assert restored["layout"] == "decomposed"
        four[mode] = got
        # the direct force is partition-independent: round-off of the escapers'
        # mesh-order sums only
        assert _gap(got["restored_state"], once) <= 1e-5 * scale, mode
    differ = harness.differing(four["sp"], four["mp"])
    assert not differ, f"2->4: one process vs four differ: {differ}"
    # 4 -> 2: a checkpoint written on four devices, continued on two processes
    root4 = str(tmp_path / "root4")
    _side(
        "sp",
        "write_and_restart",
        4,
        os.path.join(root4, "out"),
        dict(env, M3_ROOT=root4),
    )
    got, values = _side(
        "mp",
        "restore_from",
        2,
        str(tmp_path / "to2"),
        dict(env, M3_FROM=os.path.join(root4, "sp", f"once_{MID}")),
    )
    assert values["restored_restored"]["layout"] == "decomposed"
    assert _gap(got["restored_state"], once) <= 1e-5 * scale


def test_sharded_snapshots_read_back_by_id(tmp_path):
    sides = {}
    for mode in ("sp", "mp"):
        sides[mode] = _side(
            mode, "snapshots", 2, str(tmp_path / mode), {"M3_ROOT": str(tmp_path)}
        )
        values = sides[mode][1]
        assert values["paths"] == [f"snapshot_{s:08d}" for s in (0, 20, 40, 60)]
        assert all(values["same"]), values
    _same(sides["sp"][0]["snap_last"], sides["mp"][0]["snap_last"], "snapshot")

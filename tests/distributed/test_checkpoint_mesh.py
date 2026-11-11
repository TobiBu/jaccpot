"""M3: sharded checkpoints in one process -- crash safety, corruption, cross-mode.

* **Crash safety** (gate 3): a process killed inside orbax's commit (every shard
  written, the rename not done) leaves no checkpoint at the new path and the
  previous one loadable, and ``latest_checkpoint`` skips the half-written one; a
  replace interrupted between its renames leaves a committed checkpoint that
  ``resolve`` finds.
* **Corruption** (gate 3): a checkpoint whose shards are not the ones its manifest
  hashed is refused by the same-layout restore (sha256 per shard), by the elastic
  restore (the set hashes) and by the by-id load; flipped bytes in a data file are
  refused too (by tensorstore's decoder or the manifest).
* **Cross mode** (gate 2): the fused CPU lane (ESC's system with escapers): a mesh
  checkpoint continued on one card and on a one-device mesh, and a single card's
  ``.npz`` continued on the mesh -- within the force-error class of the uninterrupted
  runs, whose mesh-vs-single-card gap is the yardstick.

orbax is the ``io`` extra: skipped without it unless
``JACCPOT_CHECKPOINT_TESTS_REQUIRE_ORBAX=1`` (the CI job).

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest -n 0 -m distributed_checkpoint \\
        tests/distributed/test_checkpoint_mesh.py
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import warnings

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")
if os.environ.get("JACCPOT_CHECKPOINT_TESTS_REQUIRE_ORBAX") != "1":
    pytest.importorskip("orbax.checkpoint", reason="needs the io extra (orbax)")

from jaccpot import checkpoint as ck  # noqa: E402

pytestmark = [
    pytest.mark.distributed_checkpoint,
    pytest.mark.skipif(jax.device_count() < 2, reason="needs >= 2 devices"),
]

AXIS = "gpus"


def _arrays(seed: int = 0) -> dict:
    """A two-device layout: 5 rows per device, live rows leading, ids -1 on padding."""
    mesh = Mesh(np.asarray(jax.devices()[:2]), (AXIS,))
    sh = NamedSharding(mesh, P(AXIS))
    ids = np.array([3, 0, 4, -1, -1, 1, 2, -1, -1, -1], np.int32)
    state = np.random.default_rng(seed).normal(size=(10, 2, 3)).astype(np.float32)
    state[ids < 0] = 0.0
    return {
        "state": jax.device_put(state, sh),
        "ids": jax.device_put(ids, sh),
        "count": jax.device_put(np.array([3, 2], np.int32), sh),
    }


def _write(path, seed: int = 0) -> dict:
    return ck.write_checkpoint(
        path, _arrays(seed), {"seed": seed}, id_arrays={"state": "ids"}
    )


_CRASH = r"""
import os, sys
sys.path[:0] = {path!r}
import orbax.checkpoint._src.path.atomicity as atomicity
from tests.distributed import test_checkpoint_mesh as t

async def die(self, *a, **k):
    os._exit(9)  # killed with every shard written, before the commit

atomicity.AtomicRenameTemporaryPath.finalize = die
t._write(sys.argv[1], seed=1)
"""


def test_a_write_killed_before_its_commit_leaves_the_last_checkpoint(tmp_path):
    first = str(tmp_path / "ck_00000010")
    _write(first)
    second = str(tmp_path / "ck_00000020")
    child = subprocess.run(
        [sys.executable, "-c", _CRASH.format(path=sys.path), second],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert child.returncode == 9, child.stderr[-3000:]
    assert not ck.is_checkpoint(second)
    leftovers = glob.glob(str(tmp_path / "ck_00000020*"))
    assert leftovers, "the killed write left nothing to inspect"
    assert ck.latest_checkpoint(str(tmp_path / "ck_{step:08d}")) == first
    got, meta = ck.load_checkpoint(first, ["state"])
    assert meta["seed"] == 0 and got["state"].shape == (5, 2, 3)


def test_a_replace_interrupted_between_its_renames_stays_readable(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "restart")
    _write(path, seed=0)
    calls = []

    def rename(src, dst):
        calls.append((src, dst))
        if len(calls) == 2:
            raise KeyboardInterrupt("killed between the renames")
        os.rename(src, dst)

    monkeypatch.setattr(ck, "_rename", rename)
    with pytest.raises(KeyboardInterrupt):
        _write(path, seed=1)
    # path -> path.prev happened, path.next (the new one, committed) did not move
    assert not os.path.exists(path)
    assert ck.resolve(path).endswith("restart.next")
    assert ck.read_meta(path)["seed"] == 1
    monkeypatch.setattr(ck, "_rename", os.rename)
    _write(path, seed=2)  # the next write cleans up and replaces normally
    assert ck.resolve(path) == path and ck.read_meta(path)["seed"] == 2
    assert not glob.glob(path + ".*")


def test_a_replace_keeps_one_checkpoint(tmp_path):
    path = str(tmp_path / "restart")
    for seed in (0, 1, 2):
        _write(path, seed=seed)
    assert ck.read_meta(path)["seed"] == 2
    assert sorted(os.listdir(tmp_path)) == ["restart"]


def _mixed(tmp_path) -> str:
    """A checkpoint whose array data are another write's (its manifest stays)."""
    good, other = str(tmp_path / "good"), str(tmp_path / "other")
    _write(good, seed=0)
    _write(other, seed=1)
    shutil.rmtree(os.path.join(good, "arrays"))
    shutil.copytree(os.path.join(other, "arrays"), os.path.join(good, "arrays"))
    return good


def test_shards_that_are_not_the_manifests_are_refused(tmp_path):
    path = _mixed(tmp_path)
    meta = ck.read_meta(path)
    sh = _arrays()["state"].sharding
    got = ck.restore_arrays(path, {"state": (sh, None), "ids": (sh, None)})
    with pytest.raises(ck.CheckpointCorruptError, match="sha256"):
        ck.verify_shards(got, meta["manifest"])
    with pytest.raises(ck.CheckpointCorruptError):
        ck.load_checkpoint(path, ["state"])
    # a re-sharding restore (three devices) checks the live rows' set hash
    mesh3 = Mesh(np.asarray(jax.devices()[:3]), (AXIS,))
    sh3 = NamedSharding(mesh3, P(AXIS))
    got = ck.restore_arrays(path, {"state": (sh3, (12, 2, 3)), "ids": (sh3, (12,))})
    pieces = []
    for i, s in zip(got["ids"].addressable_shards, got["state"].addressable_shards):
        ids = np.asarray(i.data).copy()
        ids[i.index[0].start + np.arange(ids.shape[0]) >= 10] = -1
        pieces.append((ids, {"state": np.asarray(s.data)}))
    with pytest.raises(ck.CheckpointCorruptError, match="set hash"):
        ck.verify_set_hashes(pieces, meta["manifest"], ["state"])


def test_flipped_bytes_in_a_shard_are_refused(tmp_path):
    path = str(tmp_path / "ck")
    _write(path)
    files = [
        f
        for f in glob.glob(os.path.join(path, "arrays", "**"), recursive=True)
        if os.path.isfile(f) and f"{os.sep}d{os.sep}" in f
    ]
    assert files
    for name in files:
        data = bytearray(open(name, "rb").read())
        for k in range(len(data) // 3, len(data), max(1, len(data) // 7)):
            data[k] ^= 0x5A
        open(name, "wb").write(bytes(data))
    with pytest.raises(Exception) as caught:
        ck.load_checkpoint(path, ["state"])
    # recorded: which layer saw it (the decoder or the manifest)
    print("flipped bytes refused by", type(caught.value).__name__)


# ----------------------------------------------------------------------------
# Cross mode on the fused CPU lane.
# ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cpu_fused_lane():
    """The fused lane's CPU switches (the MESC test's), for the module."""
    import yggdrax.distributed.comm as ygg_comm

    from tests.integration.strict import test_production_runner as tpr
    from tests.integration.strict import test_strict_runners_on_provider as r6

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        resolve = ygg_comm.resolve_ragged_method
        mp.setattr(
            ygg_comm,
            "resolve_ragged_method",
            lambda method="auto": "buf" if method == "auto" else resolve(method),
        )
        for key in sorted(set(r6._ENV_UNSET) | set(tpr._ENV_UNSET)):
            mp.delenv(key, raising=False)
        for key, val in tpr._CSR_ON_CPU.items():
            mp.setenv(key, val)
        mp.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
        mp.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
        mp.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
        yield


def _fused_runner(mac: str, ndev: int, **segment):
    from jaccpot.escapers import EscaperPolicy
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig
    from tests.distributed import test_production_runner_mesh_escapers as tme
    from tests.integration.strict import test_escapers as te

    mesh = None
    if ndev:
        mesh = MeshRunConfig(
            Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,)),
            cap=int(np.ceil(1.5 * (tme.N + tme.K + 1) / ndev)),
            num_samples=64,
            checkpoint="orbax",
        )
    return ProductionRunner(
        te._solver(mac),
        settings=te._settings(mac),
        segment=SegmentConfig(steps=tme.STEPS, diagnostics_every_energy=1, **segment),
        escapers=EscaperPolicy(tme.R_ESCAPE, tme.R_RETURN),
        mesh=mesh,
    )


def _gap(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.mark.parametrize("mac", ["dehnen", "dehnen_error"])
def test_mesh_and_single_card_continue_each_other(cpu_fused_lane, mac, tmp_path):
    """Mesh checkpoint -> one card and -> a one-device mesh; one card's ``.npz`` ->
    the mesh: each within the yardstick (the uninterrupted runs' mesh-vs-card gap)
    of BOTH uninterrupted runs, the escaper split carried over."""
    from tests.distributed import test_production_runner_mesh_escapers as tme

    state, masses = tme._system()
    total, mid = tme.STEPS * tme.SEGMENTS, tme.SEGMENTS // 2
    on_mesh = _fused_runner(
        mac, 2, restart_every=mid, restart_path=str(tmp_path / "mesh_{segment}")
    ).run(state, masses, dt=tme.DT, num_steps=total)
    on_card = _fused_runner(
        mac,
        0,
        restart_every=mid,
        restart_path=str(tmp_path / "card_{segment}.npz"),
    ).run(state, masses, dt=tme.DT, num_steps=total)
    yardstick = _gap(on_mesh.state, on_card.state)
    assert yardstick > 0.0
    cases = {
        "mesh->card": (0, str(tmp_path / f"mesh_{mid}")),
        "mesh->1-device mesh": (1, str(tmp_path / f"mesh_{mid}")),
        "card->mesh": (2, str(tmp_path / f"card_{mid}.npz")),
    }
    report = {"yardstick": yardstick}
    for what, (ndev, source) in cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = _fused_runner(mac, ndev).run(
                dt=tme.DT, num_steps=total, restart_from=source
            )
        assert not res.restored["bitwise"], (what, res.restored)
        assert res.step == total
        assert res.escapers["n_bound"] == on_mesh.escapers["n_bound"], what
        report[what] = (_gap(res.state, on_mesh.state), _gap(res.state, on_card.state))
        assert min(report[what]) <= yardstick, (what, report)
        assert max(report[what]) <= 4.0 * yardstick, (what, report)
    print(mac, report)

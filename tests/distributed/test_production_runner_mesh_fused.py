"""``ProductionRunner``'s multi-GPU mode on the FUSED mesh force, two forced CPU devices.

The CPU end-to-end pattern of ``test_mesh_segment_runner.py`` (its fixture and
operating point: N = 2048 uniform cube, leaf 64, p4, theta 0.8, eps 1e-2, the
large-N profile reached on CPU, the all-gather exchange, the Pallas kernels
interpreted). ``test_production_runner_mesh.py`` gates the mode's machinery with a
direct force; this file is what only the fused force has -- a shape template, a
merged plan, walk and cross capacities, a segment retry that grows them -- and
the one gate that needs it:

* the mode == :class:`~jaccpot.distributed.segment.MeshSegmentRunner` driven by
  hand, bitwise by id, with a cross capacity too small for the first segment (its
  flag fires, the segment is re-planned and re-run, its edge says so) and an
  energy diagnostic at every edge (``FusedMeshForce.potential_shard``: its program
  traces the whole fused rebuild; the trajectory with it is the one without);
* the restart written after the retried segment, continued in a FRESH process:
  the restore rebuilds the layout and the re-planned plan (leaf capacity, walk
  caps, level plan, depth bound, the grown cross capacities, the template's
  shapes), its force key's digest matches, and the continuation is the straight
  run, bitwise.

Minutes of compile on CPU (two runs and a resumed one), so marker
``distributed_segment`` (job ``test-distributed-segment``).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_production_runner_mesh_fused.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import jax
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.rollout import decompose  # noqa: E402
from jaccpot.distributed.segment import MeshSegmentRunner, MeshState  # noqa: E402
from jaccpot.distributed.template_force import FusedMeshForce  # noqa: E402
from jaccpot.production import (  # noqa: E402
    GalaxySettings,
    MeshRunConfig,
    ProductionRunner,
    SegmentConfig,
)
from tests.distributed import test_mesh_segment_runner as a  # noqa: E402
from tests.distributed.test_mesh_segment_runner import cpu_fused_lane  # noqa: E402,F401

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

SETTINGS = GalaxySettings(
    leaf_size=a.LEAF,
    max_order=a.ORDER,
    theta=a.THETA,
    mac_type="dehnen",
    adaptive_eps=None,
)
SAMPLES = 64
#: two cross buffers too small for the start (the near rows a device receives, the
#: particles it sends): the first segment's cross flag fires and its re-plan is
#: exact past the overflow, one retry (test_mesh_segment_runner.py)
SMALL = dict(recv_node_cap=64, send_particle_cap=512)


def _small_caps():
    return a._cross_caps()._with(**SMALL)


def _mesh_runner(solver, force=None, **segment):
    return ProductionRunner(
        solver,
        settings=SETTINGS,
        segment=SegmentConfig(steps=2, **segment),
        mesh=MeshRunConfig(
            a._mesh(),
            cap=a.CAP,
            num_samples=SAMPLES,
            cross_caps=_small_caps(),
            migrate=False,  # the by-hand reference does not migrate
            force=force,
        ),
    )


def _resume(path, out):
    """The fresh process: the CPU fused lane's switches, then continue ``path`` to
    step 6 with the library's fused force (its plan from the file)."""
    import yggdrax.distributed.comm as ygg_comm

    jax.config.update("jax_enable_x64", True)
    jax.default_backend = lambda: "gpu"
    resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = lambda method="auto": (
        "buf" if method == "auto" else resolve(method)
    )
    runner = _mesh_runner(a._solver(), diagnostics_every=0)
    # what the restore and the run warn (the solver's own construction warns of
    # its legacy keywords)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = runner.run(dt=a.DT, num_steps=6, restart_from=path)
    got = res.handle.gather()
    np.savez(
        out,
        state=res.state,
        acceleration=got["acceleration"],
        steps=np.asarray([e.step for e in res.edges]),
        retries=np.asarray([e.retries for e in res.edges]),
        restored=np.asarray(json.dumps(res.restored)),
        warned=np.asarray(json.dumps([str(w.message) for w in caught])),
    )


def test_the_fused_force_through_the_mesh_mode_and_its_restart(
    cpu_fused_lane, tmp_path
):
    """3 segments of 2 steps, a repartition before every segment after the first,
    the cross capacities too small for the first: (1) the segment runner by hand,
    on its own solver; (2) the runner's mesh mode on another, with the energy at
    every edge (the fused mesh potential: the whole fused rebuild) and a
    restart after every segment -- the first segment retried once, its edge
    saying so, the grown capacities in the plan record, step 0's energy program
    flagging the same undersized capacities -- equal to (1) bitwise by id; (3) the
    restart after the retried segment continued in a fresh process
    with the library's force: bitwise (2), the force key's digest matching."""
    pos, vel, mass = a._ic()
    mesh = a._mesh()
    shard, _ = decompose(mesh, pos, vel, mass, cap=a.CAP, num_samples=SAMPLES)
    hand = MeshSegmentRunner.fused(
        a._solver(),
        mesh,
        MeshState.from_shard_state(shard, mesh),
        leaf_size=a.LEAF,
        max_order=a.ORDER,
        cross_caps=_small_caps(),
        repartition_every=1,
        num_samples=SAMPLES,
    )
    reports = [hand.run(2, dt=a.DT) for _ in range(3)]
    assert [r.retries for r in reports] == [1, 0, 0], reports
    want = hand.gather()
    del hand

    solver = a._solver()
    force = FusedMeshForce(
        solver,
        mesh,
        leaf_size=a.LEAF,
        max_order=a.ORDER,
        cross_caps=_small_caps(),
        axis_name=a.AXIS,
    )
    runner = _mesh_runner(
        solver,
        force=force,
        diagnostics_every_energy=1,
        restart_every=1,
        restart_path=str(tmp_path / "restart_{segment}.npz"),
    )
    res = runner.run(np.stack([pos, vel], axis=1), mass, dt=a.DT, num_steps=6)
    assert [(e.step, e.retries) for e in res.edges] == [(0, 0), (2, 1), (4, 0), (6, 0)]
    # step 0's potential runs on the undersized cross capacities the first
    # segment's flag then re-plans: its verdict says so, and no energy is reported
    assert [e.diagnostics["energy_ok"] for e in res.edges] == [False, True, True, True]
    assert res.edges[0].diagnostics["energy"] is None
    for edge in res.edges[1:]:
        assert np.isfinite(edge.diagnostics["energy"])
    grown = res.edges[1].plan["cross_caps"]
    assert grown["recv_node_cap"] > 64 and grown["send_particle_cap"] > 512, grown
    assert res.edges[1].plan["leaf_capacity"] == a.LEAF_CAP
    assert res.edges[1].plan["level_plan"] is not None
    for name in ("positions", "velocities"):
        column = 0 if name == "positions" else 1
        a._assert_bitwise(res.state[:, column], want[name], name)
    a._assert_bitwise(
        res.handle.gather()["acceleration"], want["acceleration"], "acceleration"
    )

    out = tmp_path / "resumed.npz"
    path = str(tmp_path / "restart_1.npz")
    child = f"import {__name__} as m; m._resume(*{[path, str(out)]!r})"
    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(out) as got:
        restored = json.loads(str(got["restored"]))
        assert restored["bitwise"], (restored, str(got["warned"]))
        assert restored["layout"] == "file"
        assert json.loads(str(got["warned"])) == []
        assert got["steps"].tolist() == [4, 6]
        assert got["retries"].tolist() == [0, 0]
        a._assert_bitwise(got["state"], res.state, "resumed state")
        a._assert_bitwise(
            got["acceleration"], want["acceleration"], "resumed acceleration"
        )

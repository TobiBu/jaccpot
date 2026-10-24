"""``ProductionRunner``'s multi-GPU mode (S11 session 2) with a partition-independent force.

``ProductionRunner(solver, mesh=MeshRunConfig(...))`` runs the single card's loop --
segments, edges, callbacks, diagnostics with the energy, restart files -- on
:class:`~jaccpot.distributed.segment.MeshSegmentRunner`. These gates drive it with
:class:`~jaccpot.distributed.segment.DirectMeshForce` (a direct sum over the
sources in global-id order, so a particle's force and potential do not depend on
the device or row that holds it): what the mode adds is the machinery around the
segment runner, and it is tested here at N = 256 in float64 on two forced CPU
devices, in seconds. The fused force through the same mode is
``test_production_runner_mesh_fused.py``.

* the mode == the segment runner driven by hand with the same segments and
  repartition cadence, bitwise by id, and every edge's lazy ``state_host()`` is the
  hand-driven state at that step;
* chained runs (``restart_from=RunResult``) == one run, bitwise; a restart file
  continued in a FRESH process on the same mesh == the straight run, bitwise (its
  digests matching); on another mesh the restore decomposes the by-id state and
  says it is not bitwise;
* the moments are the host fp64 moments of the gathered state; the energy is the
  host fp64 direct-sum energy of the gathered state; the energy diagnostics leave
  the trajectory alone;
* a callback stops the run; a retried segment is reported at its edge; a failure
  names the last restart; what the mode cannot take is refused.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_production_runner_mesh.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

import jaccpot.distributed.rollout as rollout  # noqa: E402
from jaccpot.distributed.segment import (  # noqa: E402
    DirectMeshForce,
    MeshSegmentRunner,
    MeshState,
)
from jaccpot.production import (  # noqa: E402
    MeshRunConfig,
    ProductionRunner,
    RunResult,
    SegmentConfig,
    load_restart,
)

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the runner's mesh mode needs >= 2 devices"
)

AXIS = "gpus"
NDEV = 2
N = 256
CAP = int(np.ceil(1.6 * N / NDEV))
DT = 0.02
G, SOFT = 1e-3, 0.05
SAMPLES = 64
#: the harmonic trap's parameters, as the caller passes them (floats)
PARAMS = {"k": 0.5, "centre": (0.5, 0.5, 0.5)}


def _mesh(ndev=NDEV):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _ic():
    """``[N, 2, 3]`` and ``[N]`` float64: a uniform cube, random velocities fast
    enough that the Morton domains go stale and particles change device (a bulk
    drift alone would not: the box moves with it), and a bulk drift along x."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1.0, size=(N, 3))
    vel = 0.5 * rng.normal(size=(N, 3)) + np.array([0.6, 0.0, 0.0])
    return np.stack([pos, vel], axis=1), np.full(N, 1.0 / N)


def harmonic(state_rows, params):
    """A harmonic trap toward ``params["centre"]``, per row (module level: one
    function object, and the same name in a fresh process)."""
    return -params["k"] * (state_rows[:, 0] - jnp.asarray(params["centre"]))


def harmonic_potential(state_rows, params):
    """Its potential per unit mass."""
    d = state_rows[:, 0] - jnp.asarray(params["centre"])
    return 0.5 * params["k"] * jnp.sum(d * d, axis=1)


def _solver():
    """The runner's solver: its G and softening are the direct force's (the mode
    checks force parameters against them; the direct force computes)."""
    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(preset="large_n_gpu", G=G, softening=SOFT)


def _runner(force=None, ndev=NDEV, repartition_every=1, callbacks=(), **segment):
    config = dict(steps=2, diagnostics_every_energy=1)
    config.update(segment)
    return ProductionRunner(
        _solver(),
        segment=SegmentConfig(**config),
        external_acceleration_fn=harmonic,
        external_potential_fn=harmonic_potential,
        callbacks=callbacks,
        mesh=MeshRunConfig(
            _mesh(ndev),
            cap=CAP if ndev == NDEV else None,
            repartition_every=repartition_every,
            num_samples=SAMPLES,
            force=(
                DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS)
                if force is None
                else force
            ),
        ),
    )


def _run(num_steps=8, **kw):
    state, masses = _ic()
    return _runner(**kw).run(
        state, masses, dt=DT, num_steps=num_steps, external_params=PARAMS
    )


def _bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    assert got.shape == want.shape, (what, got.shape, want.shape)
    differ = int(np.sum(got != want))
    assert differ == 0, f"{what}: {differ} of {got.size} elements differ"


class _Snapshots:
    """A callback keeping every edge's state by id (``state_host``) and gather."""

    def __init__(self):
        self.states, self.gathers = {}, {}

    def __call__(self, edge):
        self.states[edge.step] = edge.state_host()
        self.gathers[edge.step] = edge.gather(("state", "masses", "acceleration"))


# --------------------------------------------------------------------------
# The mode is the segment runner driven by hand.
# --------------------------------------------------------------------------


def test_the_mesh_mode_is_the_segment_runner_driven_by_hand():
    """4 segments of 2 steps, a repartition before every segment after the first,
    an external field with traced parameters: the runner's final state (gathered by
    id) and every edge's ``state_host()`` equal the segment runner driven by hand
    -- ``decompose`` with the same cap and samples, ``run(2)`` four times --
    bitwise by id; the edges report the repartitions; the result's handle is the
    segment runner and the kept edges hold no arrays."""
    snaps = _Snapshots()
    res = _run(callbacks=(snaps,))
    assert res.mode == "mesh" and isinstance(res.handle, MeshSegmentRunner)
    assert [e.step for e in res.edges] == [0, 2, 4, 6, 8]
    assert [e.retries for e in res.edges] == [0] * 5
    repart = [e.plan["repartition"] for e in res.edges]
    assert repart[0] is None and not repart[1]["repartitioned"]
    assert all(r["repartitioned"] and not r["declined"] for r in repart[2:])
    assert sum(r["sent_off_device"] for r in repart[2:]) > 0, "nothing moved"
    for e in res.edges:
        assert e.plan["mesh"]["ndev"] == NDEV and e.plan["mesh"]["cap"] == CAP
        assert sum(e.plan["mesh"]["counts"]) == N
        assert e.state is None and e.layout is None and e.self_acceleration is None
    with pytest.raises(RuntimeError, match="consumed"):
        res.edges[-1].state_host()

    state, masses = _ic()
    mesh = _mesh()
    shard, _ = rollout.decompose(
        mesh,
        state[:, 0],
        state[:, 1],
        masses,
        cap=CAP,
        num_samples=SAMPLES,
        axis_name=AXIS,
    )
    hand = MeshSegmentRunner(
        mesh,
        MeshState.from_shard_state(shard, mesh),
        DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS),
        external_acceleration_fn=harmonic,
        repartition_every=1,
        num_samples=SAMPLES,
    )
    params = {"k": jnp.asarray(0.5), "centre": tuple(jnp.asarray(0.5) for _ in "xyz")}
    _bitwise(snaps.states[0], state, "step 0: the input, by id")
    for k in range(4):
        hand.run(2, dt=DT, external_params=params)
        got = hand.gather()
        step = 2 * (k + 1)
        _bitwise(snaps.states[step][:, 0], got["positions"], f"positions {step}")
        _bitwise(snaps.states[step][:, 1], got["velocities"], f"velocities {step}")
        _bitwise(
            snaps.gathers[step]["acceleration"], got["acceleration"], f"acc {step}"
        )
        _bitwise(snaps.gathers[step]["owner"], got["owner"], f"owner {step}")
    _bitwise(res.state[:, 0], got["positions"], "final positions")
    _bitwise(res.state[:, 1], got["velocities"], "final velocities")
    _bitwise(res.masses, masses, "masses by id")


def test_chained_runs_are_one_run():
    """Steps 0-4, then a run continued from the RunResult to step 8 (in memory,
    the same segment runner) == one run to step 8: state and diagnostics, bitwise.
    A consumed result cannot be continued twice."""
    straight = _run()
    first = _run(num_steps=4)
    runner = _runner()
    second = runner.run(dt=DT, num_steps=8, restart_from=first, external_params=PARAMS)
    assert second.restored == {"from": "RunResult", "step": 4, "bitwise": True}
    assert [e.step for e in second.edges] == [6, 8]
    _bitwise(second.state, straight.state, "chained state")
    assert second.edges[-1].diagnostics == straight.edges[-1].diagnostics
    assert first.handle is None and first.state is None
    with pytest.raises(ValueError, match="handle"):
        runner.run(dt=DT, num_steps=8, restart_from=first, external_params=PARAMS)


# --------------------------------------------------------------------------
# Restart files.
# --------------------------------------------------------------------------


def _resume(path, out):
    """The fresh process: continue ``path`` to step 8 on the same mesh."""
    jax.config.update("jax_enable_x64", True)
    runner = _runner()
    # what the restore and the run warn
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = runner.run(dt=DT, num_steps=8, restart_from=path, external_params=PARAMS)
    np.savez(
        out,
        state=res.state,
        acceleration=res.handle.gather()["acceleration"],
        steps=np.asarray([e.step for e in res.edges]),
        diagnostics=np.asarray(json.dumps(res.edges[-1].diagnostics)),
        restored=np.asarray(json.dumps(res.restored)),
        warned=np.asarray(json.dumps([str(w.message) for w in caught])),
    )


def test_a_restart_in_a_fresh_process_continues_bitwise(tmp_path):
    """4 segments straight (a restart after each) against 2 of them, the restart
    file, and 2 more segments in a FRESH process on the same two devices: the same
    state and carried acceleration by id, bit for bit, the same last diagnostics,
    the restore's digest matching and no warning. The file holds the by-id arrays
    and the layout."""
    straight = _run(
        restart_every=1, restart_path=str(tmp_path / "restart_{segment}.npz")
    )
    assert len(straight.restarts) == 4
    path = tmp_path / "restart_2.npz"
    arrays, meta = load_restart(path)
    assert set(arrays) == {
        "state",
        "masses",
        "acceleration",
        "mesh_ids",
        "mesh_padding",
    }
    assert arrays["state"].shape == (N, 2, 3) and arrays["mesh_ids"].shape == (
        NDEV * CAP,
    )
    assert arrays["mesh_padding"].shape == (NDEV * CAP - N, 3)
    assert meta["counters"]["step"] == 4 and meta["counters"]["segment"] == 2
    assert meta["mesh"]["ndev"] == NDEV and meta["mesh"]["cap"] == CAP
    assert meta["baselines"] == straight.baselines
    out = tmp_path / "resumed.npz"
    child = f"import {__name__} as m; m._resume(*{[str(path), str(out)]!r})"
    proc = subprocess.run(
        [sys.executable, "-c", child],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    with np.load(out) as got:
        restored = json.loads(str(got["restored"]))
        assert restored["bitwise"], (restored, str(got["warned"]))
        assert restored["layout"] == "file"
        assert json.loads(str(got["warned"])) == []
        assert got["steps"].tolist() == [6, 8]
        _bitwise(got["state"], straight.state, "resumed state")
        _bitwise(
            got["acceleration"],
            straight.handle.gather()["acceleration"],
            "resumed acceleration",
        )
        assert json.loads(str(got["diagnostics"])) == straight.edges[-1].diagnostics


def test_a_restart_on_another_mesh_continues_and_says_it_is_not_bitwise(tmp_path):
    """The two-device file continued on ONE device: the by-id state decomposed
    there, its carried acceleration placed by id (no start force), the run
    finished -- the same trajectory to round-off (the direct sum's source axis
    has another length there), and ``restored["bitwise"]`` False with the layout
    named."""
    straight = _run(
        restart_every=1, restart_path=str(tmp_path / "restart_{segment}.npz")
    )
    with pytest.warns(UserWarning, match="not bitwise"):
        other = _runner(ndev=1).run(
            dt=DT,
            num_steps=8,
            restart_from=tmp_path / "restart_2.npz",
            external_params=PARAMS,
        )
    assert not other.restored["bitwise"]
    assert other.restored["layout"] == "decomposed"
    assert "mesh_layout" in other.restored["mismatches"]
    assert other.handle.ndev == 1 and other.step == 8
    assert not other.handle.reports[0].start_force
    np.testing.assert_allclose(other.state, straight.state, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------
# Diagnostics.
# --------------------------------------------------------------------------


def _host_moments(state, masses):
    x, v, m = state[:, 0], state[:, 1], masses
    return {
        "momentum": np.sum(m[:, None] * v, axis=0),
        "angular_momentum": np.sum(m[:, None] * np.cross(x, v), axis=0),
        "com": np.sum(m[:, None] * x, axis=0) / np.sum(m),
        "kinetic": 0.5 * np.sum(m * np.sum(v * v, axis=1)),
    }


def _host_energies(state, masses):
    """fp64 on the host: ``W_self = 1/2 sum_i m_i phi_i`` by the direct sum with
    the force's kernel (ferrers3), and ``W_ext = sum m Phi_ext``."""
    from jaccpot.softening import pair_factors, softening_params_np

    x, m = state[:, 0], masses
    d = x[None, :, :] - x[:, None, :]
    r2 = np.sum(d * d, axis=-1)
    off = ~np.eye(len(m), dtype=bool)
    psi = pair_factors(
        np.where(off, r2, 1.0),
        softening_params_np("ferrers3", SOFT),
        "ferrers3",
        potential=True,
        xp=np,
    )[1]
    phi = -G * np.sum(np.where(off, psi, 0.0) * m[None, :], axis=1)
    w_ext = np.sum(
        m * 0.5 * PARAMS["k"] * np.sum((x - np.asarray(PARAMS["centre"])) ** 2, axis=1)
    )
    return 0.5 * np.sum(m * phi), w_ext


def test_the_moments_and_the_energy_are_the_gathered_states():
    """At every edge the moments (fp64 reductions over the flat sharded arrays,
    padding rows included: zero mass, zero velocity) and the energy (the direct
    force's mesh potential and the trap's, fp64 sums per device and over the mesh)
    equal the host fp64 values of the state gathered by id. Tolerance 1e-12
    relative: everything is float64, and only the summation order differs (per
    device then over the mesh, against id order on the host), a few ulps of a
    256-term sum. Step 0's energy is evaluated AT step 0 (the mesh force is
    prepared before the first segment): its drift is 0 and it is E0."""
    snaps = _Snapshots()
    res = _run(callbacks=(snaps,))
    assert res.baselines["energy"] == res.edges[0].diagnostics["energy"]
    assert res.edges[0].diagnostics["energy_drift"] == 0.0
    for edge in res.edges:
        row = edge.diagnostics
        state = snaps.gathers[edge.step]["state"]
        masses = snaps.gathers[edge.step]["masses"]
        want = _host_moments(state, masses)
        for name in ("momentum", "angular_momentum", "com"):
            np.testing.assert_allclose(row[name], want[name], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(row["kinetic"], want["kinetic"], rtol=1e-12)
        w_self, w_ext = _host_energies(state, masses)
        assert row["energy_ok"] is True
        np.testing.assert_allclose(row["potential_self"], w_self, rtol=1e-12)
        np.testing.assert_allclose(row["potential_external"], w_ext, rtol=1e-12)
        np.testing.assert_allclose(
            row["energy"], want["kinetic"] + w_self + w_ext, rtol=1e-12
        )
    # velocity Verlet in a trap at dt = 0.02: bounded, small
    assert abs(res.edges[-1].diagnostics["energy_drift"]) < 1e-5


def test_the_energy_diagnostics_leave_the_trajectory_alone():
    """With every diagnostic (the energy at every edge) and with none: the same
    final state, bitwise."""
    with_energy = _run()
    without = _run(diagnostics_every=0)
    assert all(e.diagnostics is None for e in without.edges)
    assert with_energy.edges[-1].diagnostics["energy"] is not None
    _bitwise(with_energy.state, without.state, "state")


class _NoPotential(DirectMeshForce):
    """The direct force without a mesh potential."""

    potential_shard = None


def test_a_force_without_a_potential_reports_why_the_energy_is_missing():
    """A force with no ``potential_shard`` (the fused mesh force until its potential
    lands): the moments are reported, every energy field is None, the row says why,
    and the mode warns once."""
    with pytest.warns(UserWarning, match="energy diagnostic is unavailable"):
        res = _run(force=_NoPotential(G=G, softening=SOFT, axis_name=AXIS))
    for edge in res.edges:
        row = edge.diagnostics
        assert row["kinetic"] is not None
        assert row["energy"] is None and row["energy_ok"] is None
        assert "potential_shard" in row["energy_unavailable"]
    assert res.baselines["energy"] is None


# --------------------------------------------------------------------------
# Callbacks, retries, failures, refusals.
# --------------------------------------------------------------------------


def test_a_callback_stops_the_run_and_a_continuation_finishes_it():
    """A callback returning True at step 4 stops the run after that edge; the run
    continued from the result reaches step 8 as the straight run does, bitwise."""
    straight = _run()
    stopped = _run(callbacks=(lambda edge: edge.step == 4,))
    assert stopped.stopped and stopped.step == 4
    assert [e.step for e in stopped.edges] == [0, 2, 4]
    done = _runner().run(
        dt=DT, num_steps=8, restart_from=stopped, external_params=PARAMS
    )
    assert not done.stopped and done.step == 8
    _bitwise(done.state, straight.state, "continued state")


class _Armed(DirectMeshForce):
    """The direct force with a capacity a callback can make it outgrow.

    Armed, every step flags a saturated walk (a static switch: the armed programs
    are other programs). ``retryable``: the segment retry re-plans by disarming
    it, once; otherwise the retry declines and the segment raises."""

    def __init__(self, *, retryable, **kw):
        super().__init__(**kw)
        self.retryable = bool(retryable)
        self.armed = False

    def key(self):
        return super().key() + (self.armed,)

    def shard(self, positions, masses, live, ids):
        from jaccpot.runtime.fused_topology import OVERFLOW_SATURATED

        acc, bits, needs = super().shard(positions, masses, live, ids)
        if self.armed:
            bits = jnp.uint32(OVERFLOW_SATURATED)
        return acc, bits, needs

    def retry_mode(self, needs, retries, tree_first):
        del needs, tree_first
        return "full" if self.retryable and retries == 0 else None

    def replan(self, needs, bits, *, walk_only=False):
        del needs, bits, walk_only
        self.armed = False
        return True

    def raise_overflow(self, needs, bits, *, retries, where):
        del needs
        raise RuntimeError(f"mesh {where}: armed (bits {int(bits):#x}, {retries})")


def test_a_retried_segment_is_reported_at_its_edge():
    """Armed after step 4, the third segment fails, is re-planned at its start
    (from the host copy of its donated start state) and re-run from a start force:
    its edge reports the retry and the failure, and the run is the one that never
    failed -- to round-off, not bitwise: that one carried its acceleration out of
    the second segment's scan, the retry recomputes it in the start program (the
    segment runner's own retry is gated bitwise against a fresh run in
    ``test_mesh_segment_runner.py``)."""
    force = _Armed(retryable=True, G=G, softening=SOFT, axis_name=AXIS)

    def arm(edge):
        if edge.step == 4:
            force.armed = True

    res = _run(force=force, callbacks=(arm,))
    assert [e.retries for e in res.edges] == [0, 0, 0, 1, 0]
    assert res.handle.reports[2].failures[0]["bits"] != 0
    np.testing.assert_allclose(res.state, _run().state, rtol=0, atol=1e-12)


def test_a_failed_segment_names_the_last_restart(tmp_path):
    """Armed after step 4 with no re-plan: the third segment raises a RuntimeError
    that names it, its steps and the last restart file."""
    force = _Armed(retryable=False, G=G, softening=SOFT, axis_name=AXIS)

    def arm(edge):
        if edge.step == 4:
            force.armed = True

    path = str(tmp_path / "restart_{segment}.npz")
    with pytest.raises(RuntimeError) as info:
        _run(force=force, callbacks=(arm,), restart_every=1, restart_path=path)
    message = str(info.value)
    assert "segment 3 (steps 4 to 6)" in message, message
    assert f"Last restart: {tmp_path / 'restart_2.npz'}" in message, message
    assert "armed" in message


def test_the_mesh_mode_refuses_what_it_cannot_take(tmp_path):
    """Force parameters other than the force's own (it is static in them), an
    in-scan step callback, a 2-D mesh; a multi-GPU restart file on one card. The
    force's own parameters pass."""
    from jaccpot import ForceParams

    state, masses = _ic()
    with pytest.raises(ValueError, match="static in G and the softening"):
        _runner().run(
            state,
            masses,
            dt=DT,
            num_steps=2,
            force_params=ForceParams(G=2 * G, eps2=SOFT**2),
            external_params=PARAMS,
        )
    same = _runner().run(
        state,
        masses,
        dt=DT,
        num_steps=2,
        force_params=ForceParams(G=G, eps2=SOFT**2),
        external_params=PARAMS,
    )
    assert same.step == 2
    with pytest.raises(ValueError, match="step_callback"):
        ProductionRunner(
            _solver(), mesh=_mesh(), step_callback=lambda step, state: None
        )
    mesh_2d = jax.sharding.Mesh(np.asarray(jax.devices()[:2]).reshape(2, 1), ("a", "b"))
    with pytest.raises(ValueError, match="1-D"):
        MeshRunConfig(mesh_2d)
    _run(num_steps=2, restart_every=1, restart_path=str(tmp_path / "r.npz"))
    with pytest.raises(ValueError, match="multi-GPU mode"):
        ProductionRunner(_solver()).run(
            dt=DT, num_steps=4, restart_from=tmp_path / "r.npz"
        )
    with pytest.raises(ValueError, match="'mesh' RunResult"):
        ProductionRunner(_solver()).run(
            dt=DT, num_steps=4, restart_from=RunResult(*[None] * 14, mode="mesh")
        )

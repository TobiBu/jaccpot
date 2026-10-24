"""The segment runner's potential energy and its restore hook, with the direct force.

:meth:`~jaccpot.distributed.segment.MeshSegmentRunner.potential_energy` is the
energy diagnostic's program on the mesh (S11 session 2): the force's
``potential_shard`` per device and the external potential per row, fp64 sums over
the mesh. With :class:`~jaccpot.distributed.segment.DirectMeshForce` (sources in
global-id order: partition-independent) it must be the host's fp64 direct-sum
energy of the gathered state, and calling it must change no number of the run.
:meth:`~jaccpot.distributed.segment.MeshSegmentRunner.resume` sets a runner built
from a restart's state to the counters and carried total of the run that wrote it:
the continuation is that run's, bitwise, without a start force.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_mesh_segment_energy.py -q
"""

from __future__ import annotations

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

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the mesh runner's energy needs >= 2 devices"
)

AXIS = "gpus"
NDEV = 2
N = 256
CAP = int(np.ceil(1.6 * N / NDEV))
DT = 0.02
G, SOFT = 1e-3, 0.05
PARAMS = {"k": jnp.asarray(0.5), "centre": jnp.asarray([0.5, 0.5, 0.5])}


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:NDEV]), (AXIS,))


def _state(mesh):
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1.0, size=(N, 3))
    vel = 0.5 * rng.normal(size=(N, 3))
    masses = rng.uniform(0.5, 1.5, size=N) / N
    shard, _ = rollout.decompose(
        mesh, pos, vel, masses, cap=CAP, num_samples=64, axis_name=AXIS
    )
    return MeshState.from_shard_state(shard, mesh)


def _trap(state_rows, params):
    return -params["k"] * (state_rows[:, 0] - params["centre"])


def _trap_potential(state_rows, params):
    d = state_rows[:, 0] - params["centre"]
    return 0.5 * params["k"] * jnp.sum(d * d, axis=1)


def _runner(mesh, kernel="ferrers3"):
    return MeshSegmentRunner(
        mesh,
        _state(mesh),
        DirectMeshForce(G=G, softening=SOFT, softening_kernel=kernel, axis_name=AXIS),
        external_acceleration_fn=_trap,
        repartition_every=1,
        num_samples=64,
    )


def _host_self_energy(positions, masses, kernel):
    """``1/2 sum_i m_i phi_i``, fp64 direct sum on the host, the self pair out."""
    from jaccpot.softening import pair_factors, softening_params_np

    d = positions[None, :, :] - positions[:, None, :]
    r2 = np.sum(d * d, axis=-1)
    off = ~np.eye(len(masses), dtype=bool)
    psi = pair_factors(
        np.where(off, r2, 1.0),
        softening_params_np(kernel, SOFT),
        kernel,
        potential=True,
        xp=np,
    )[1]
    phi = -G * np.sum(np.where(off, psi, 0.0) * masses[None, :], axis=1)
    return 0.5 * np.sum(masses * phi)


@pytest.mark.parametrize("kernel", ["ferrers3", "plummer"])
def test_the_mesh_potential_energy_is_the_direct_sum_of_the_gathered_state(kernel):
    """After 2 segments (one repartition): ``self`` is the host fp64 direct-sum
    ``W_self`` of the state gathered by id and ``external`` the trap's ``sum m
    Phi``, to 1e-12 relative -- float64 throughout, only the summation order
    differs (per device then over the mesh, against id order) -- and ``ok``; the
    same at the start, before any force."""
    runner = _runner(_mesh(), kernel)
    for _ in range(3):
        energy = runner.potential_energy(
            external_potential_fn=_trap_potential, external_params=PARAMS
        )
        got = runner.gather()
        x, m = got["positions"], got["masses"]
        want_ext = np.sum(
            m * 0.5 * 0.5 * np.sum((x - np.asarray(PARAMS["centre"])) ** 2, axis=1)
        )
        assert energy["ok"] is True
        np.testing.assert_allclose(
            energy["self"], _host_self_energy(x, m, kernel), rtol=1e-12
        )
        np.testing.assert_allclose(energy["external"], want_ext, rtol=1e-12)
        runner.run(2, dt=DT, external_params=PARAMS)
    assert runner.reports[-1].repartitioned
    # without an external potential, W_ext is 0
    assert runner.potential_energy()["external"] == 0.0


def test_the_potential_energy_changes_nothing():
    """A runner that evaluates its energy before and after every segment ends where
    one that never does ends, bitwise by id: the energy program reads the state
    and donates nothing."""
    with_energy, without = _runner(_mesh()), _runner(_mesh())
    for _ in range(3):
        with_energy.potential_energy(
            external_potential_fn=_trap_potential, external_params=PARAMS
        )
        with_energy.run(2, dt=DT, external_params=PARAMS)
        without.run(2, dt=DT, external_params=PARAMS)
    a, b = with_energy.gather(), without.gather()
    for name in ("positions", "velocities", "acceleration"):
        assert int(np.sum(a[name] != b[name])) == 0, name


def test_a_force_without_a_potential_says_so():
    """A force with no ``potential_shard``: ``NotImplementedError`` naming it."""

    class _NoPotential(DirectMeshForce):
        potential_shard = None

    runner = MeshSegmentRunner(
        _mesh(),
        _state(_mesh()),
        _NoPotential(G=G, softening=SOFT, axis_name=AXIS),
        num_samples=64,
    )
    with pytest.raises(NotImplementedError, match="_NoPotential has no potential"):
        runner.potential_energy()


def test_resume_continues_a_copied_state_as_the_run_that_wrote_it():
    """A second runner built from a COPY of the first's state after 2 segments
    (as a restore builds it: the layout, the carried total acceleration) and
    ``resume``d at its counters runs the next 2 segments as the first does:
    bitwise by id, no start force, the repartition cadence continued. Without
    ``resume`` it starts with a start force and misses the repartition."""
    mesh = _mesh()
    first = _runner(mesh)
    for _ in range(2):
        first.run(2, dt=DT, external_params=PARAMS)

    def copy(state):
        # through the host, as a restore builds it: new buffers, the same bits
        return MeshState(
            **{
                k: None if v is None else jax.device_put(np.asarray(v), v.sharding)
                for k, v in vars(state).items()
            }
        )

    def follower(resumed):
        runner = MeshSegmentRunner(
            mesh,
            copy(first.state),
            DirectMeshForce(G=G, softening=SOFT, axis_name=AXIS),
            external_acceleration_fn=_trap,
            repartition_every=2,
            num_samples=64,
        )
        if resumed:
            runner.resume(segment_index=2, steps_done=4, external_params=PARAMS)
        return runner

    resumed, fresh = follower(True), follower(False)
    first.repartition_every = 2
    want = [first.run(2, dt=DT, external_params=PARAMS) for _ in range(2)]
    got = [resumed.run(2, dt=DT, external_params=PARAMS) for _ in range(2)]
    late = [fresh.run(2, dt=DT, external_params=PARAMS) for _ in range(2)]
    assert [r.start_force for r in got] == [False, False]
    assert [r.repartitioned for r in got] == [r.repartitioned for r in want]
    assert [r.repartitioned for r in want] == [True, False]
    assert late[0].start_force and [r.repartitioned for r in late] == [False, False]
    assert resumed.segment_index == 4 and resumed.steps_done == 8
    a, b = first.gather(), resumed.gather()
    for name in ("positions", "velocities", "acceleration"):
        assert int(np.sum(a[name] != b[name])) == 0, name

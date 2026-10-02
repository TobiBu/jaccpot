"""The repartitioned rollout's plumbing, with a partition-independent reference force.

The reference force sums sources in GLOBAL-ID order, so a particle's acceleration does
not depend on which device or row holds it. Then a rollout that repartitions every
step and one that never does must agree BITWISE by id -- any mis-route of a position,
velocity or id breaks that. A deliberately mutated arm (velocities not routed) proves
the comparison can see a mis-route.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/unit/distributed/test_fused_rollout.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")
from yggdrax.distributed.partition import sfc_repartition  # noqa: E402,F401

from jaccpot.distributed.rollout import (  # noqa: E402
    FusedRollout,
    RolloutConfig,
    RolloutFlagError,
    decompose,
    make_reference_direct_force,
)

AXIS = "gpus"


def _mesh(n):
    devices = jax.devices()
    if len(devices) < n:
        pytest.skip(f"needs {n} devices, have {len(devices)}")
    return jax.sharding.Mesh(
        np.asarray(devices[:n]), (AXIS,), axis_types=(jax.sharding.AxisType.Auto,)
    )


def _ic(n, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(n, 3))
    # a bulk drift along x so Morton domains go stale and particles change device
    vel = 0.05 * rng.normal(size=(n, 3)) + np.array([0.6, 0.0, 0.0])
    mass = np.full(n, 1.0 / n)
    return pos, vel, mass


def _rollout(ndev, every, steps, *, n=256, route_velocities=True, force=None):
    mesh = _mesh(ndev)
    pos, vel, mass = _ic(n)
    cap = int(np.ceil(1.6 * n / ndev))
    state, _ = decompose(mesh, pos, vel, mass, cap=cap, num_samples=64, axis_name=AXIS)
    force = force or make_reference_direct_force(
        mesh, G=1e-3, softening=0.05, axis_name=AXIS
    )
    roll = FusedRollout(
        mesh,
        state,
        RolloutConfig(
            cap=cap, dt=0.02, repartition_every=every, num_samples=64, axis_name=AXIS
        ),
        force,
        route_velocities=route_velocities,
    )
    roll.start()
    reports = roll.run(steps)
    return roll, reports


@pytest.mark.parametrize("ndev", [2, 4])
def test_repartitioning_every_step_changes_nothing_but_ownership(ndev):
    a, rep_a = _rollout(ndev, every=1, steps=30)
    b, rep_b = _rollout(ndev, every=0, steps=30)
    ga, gb = a.gather(), b.gather()
    for name in ("positions", "velocities", "accel"):
        np.testing.assert_array_equal(ga[name], gb[name], err_msg=name)
    assert (
        sum(r.sent_off_device for r in rep_a) > 0
    ), "nothing moved, so nothing was tested"
    assert not np.array_equal(ga["owner"], gb["owner"]), "ownership should differ"
    n = 256
    for r in rep_a:
        assert sum(r.counts) == n
        assert max(r.counts) <= a.config.cap


def test_a_mis_routed_velocity_breaks_the_equivalence():
    """MUTATION CONTROL: velocities left in the old row order must be caught."""
    a, _ = _rollout(2, every=1, steps=10, route_velocities=False)
    b, _ = _rollout(2, every=0, steps=10)
    assert not np.array_equal(a.gather()["positions"], b.gather()["positions"])


def test_the_cadence_fires_on_schedule_and_moves_particles():
    roll, reports = _rollout(2, every=4, steps=12)
    assert [r.step for r in reports if r.repartitioned] == [4, 8, 12]
    assert all(not r.repartitioned for r in reports if r.step % 4)
    assert sum(r.sent_off_device for r in reports if r.repartitioned) > 0
    roll.gather()  # identity intact


def test_a_capacity_flag_stops_the_rollout_at_its_step():
    mesh = _mesh(2)
    ref = make_reference_direct_force(mesh, G=1e-3, softening=0.05, axis_name=AXIS)
    calls = {"n": 0}

    def flaky(positions, masses, count, ids):
        acc, _ = ref(positions, masses, count, ids)
        calls["n"] += 1
        return acc, jnp.asarray(calls["n"] == 6)  # start() is call 1 -> step 5

    with pytest.raises(RolloutFlagError, match="step 5"):
        _rollout(2, every=2, steps=10, force=flaky)

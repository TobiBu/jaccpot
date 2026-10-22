"""The segment runner's repartition and external field, with a partition-independent force.

:class:`~jaccpot.distributed.segment.DirectMeshForce` sums the sources in GLOBAL-ID
order, so a particle's acceleration does not depend on which device or row holds it.
Then a run that repartitions before every segment and one that never does must agree
BITWISE by id -- any mis-route of a position, velocity, carried acceleration or id
breaks that (the pattern of ``tests/unit/distributed/test_fused_rollout.py``). A
mutated arm (the carried acceleration left in the old row order) proves the
comparison can see a mis-route of the one array the segment runner adds to the
payload.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_repartition.py -q
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
    jax.device_count() < 2, reason="the mesh runner's repartition needs >= 2 devices"
)

AXIS = "gpus"
NDEV = 2
N = 256
CAP = int(np.ceil(1.6 * N / NDEV))
DT = 0.02


def _mesh():
    return jax.sharding.Mesh(
        np.asarray(jax.devices()[:NDEV]),
        (AXIS,),
        axis_types=(jax.sharding.AxisType.Auto,),
    )


def _state(mesh, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 1.0, size=(N, 3))
    # a bulk drift along x so Morton domains go stale and particles change device
    vel = 0.05 * rng.normal(size=(N, 3)) + np.array([0.6, 0.0, 0.0])
    mass = np.full(N, 1.0 / N)
    shard, _ = rollout.decompose(
        mesh, pos, vel, mass, cap=CAP, num_samples=64, axis_name=AXIS
    )
    return MeshState.from_shard_state(shard, mesh)


def _run(every, segments, *, steps=2, field=None, params=None):
    mesh = _mesh()
    runner = MeshSegmentRunner(
        mesh,
        _state(mesh),
        DirectMeshForce(G=1e-3, softening=0.05, axis_name=AXIS),
        external_acceleration_fn=field,
        repartition_every=every,
        num_samples=64,
    )
    reports = [
        runner.run(steps, dt=DT, external_params=params) for _ in range(segments)
    ]
    return runner, reports


def _assert_equal_by_id(a, b, names=("positions", "velocities", "acceleration")):
    ga, gb = a.gather(), b.gather()
    for name in names:
        differ = int(np.sum(ga[name] != gb[name]))
        assert differ == 0, f"{name}: {differ} of {ga[name].size} elements differ"
    return ga, gb


def test_repartitioning_every_segment_changes_nothing_but_ownership():
    a, rep_a = _run(every=1, segments=10)
    b, rep_b = _run(every=0, segments=10)
    ga, gb = _assert_equal_by_id(a, b)
    assert [r.repartitioned for r in rep_a] == [False] + [True] * 9
    assert not any(r.repartitioned for r in rep_b)
    assert sum(r.sent_off_device for r in rep_a) > 0, "nothing moved: nothing tested"
    assert not np.array_equal(ga["owner"], gb["owner"]), "ownership should differ"
    assert not any(r.declined for r in rep_a)
    for r in rep_a:
        assert sum(r.counts) == N and max(r.counts) <= CAP
    # one start force; every later segment starts from the carried acceleration,
    # repartitioned or not
    assert [r.start_force for r in rep_a] == [True] + [False] * 9


def test_a_carried_acceleration_left_in_the_old_rows_breaks_the_equivalence(
    monkeypatch,
):
    """MUTATION CONTROL: the acceleration not routed by the repartition."""
    real = rollout._repartition_shard

    def unrouted(*args, acceleration=None, **kwargs):
        out = real(*args, acceleration=acceleration, **kwargs)
        return out[:5] + (acceleration,) + out[6:]

    monkeypatch.setattr(rollout, "_repartition_shard", unrouted)
    a, rep_a = _run(every=1, segments=10)
    monkeypatch.setattr(rollout, "_repartition_shard", real)
    b, _ = _run(every=0, segments=10)
    assert sum(r.sent_off_device for r in rep_a) > 0
    assert not np.array_equal(a.gather()["positions"], b.gather()["positions"])


def _harmonic(state_rows, params):
    """A harmonic trap toward ``params["centre"]``, per row: positions only."""
    return -params["k"] * (state_rows[:, 0] - params["centre"])


def test_an_external_field_rides_in_the_carried_total_acceleration():
    """Self-gravity plus a traced external field: 3 x 2 chained segments are one of
    6, bitwise, and the carried acceleration at the end is the direct sum plus the
    field at the final positions."""
    params = {"k": jnp.asarray(0.5), "centre": jnp.asarray([0.5, 0.5, 0.5])}
    chained, reports = _run(every=0, segments=3, field=_harmonic, params=params)
    whole, _ = _run(every=0, segments=1, steps=6, field=_harmonic, params=params)
    got, _ = _assert_equal_by_id(chained, whole)
    assert [r.start_force for r in reports] == [True, False, False]

    mesh = chained.mesh
    ref = rollout.make_reference_direct_force(
        mesh, G=1e-3, softening=0.05, axis_name=AXIS
    )
    s = chained.state
    self_gravity = ref(s.state[:, 0], s.masses, s.count, s.ids)[0]
    want = np.asarray(self_gravity) + np.asarray(
        _harmonic(np.asarray(s.state), jax.tree_util.tree_map(np.asarray, params))
    )
    rows = np.concatenate(
        [
            np.arange(d * CAP, d * CAP + int(c))
            for d, c in enumerate(np.asarray(s.count))
        ]
    )
    np.testing.assert_allclose(
        np.asarray(s.acceleration)[rows],
        want[rows],
        rtol=0,
        # float64 throughout; the self-gravity is recomputed in another program, so
        # allow the round-off of a 256-source sum (~1e-14), far below the field's
        # 1e-2 scale and the self-gravity's 1e-3
        atol=1e-12,
    )

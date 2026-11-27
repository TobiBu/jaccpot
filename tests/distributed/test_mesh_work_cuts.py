"""Work-balanced rebalance cuts on the fused mesh lane (M4 session 2).

``MeshSegmentRunner(cuts="work")`` (or ``JACCPOT_MESH_CUTS=work``): the rebalance
before a segment measures every row's work in the fused force's own topology
(:func:`jaccpot.distributed.work.row_work`: its leaf's near interactions, the cross
near pairs, the device's far pairs x a weight) and cuts the Morton curve at equal
cumulative work (:func:`jaccpot.distributed.migration.work_balanced_splitters_shard`).
On two forced CPU devices (fp64, the momentum test's solver) with a dense clump in
one corner of a uniform cube -- one device holds the clump, so equal counts are
unequal work:

* the rebalance's report has the work per device before and after, and the cuts
  even it out (max / mean after <= 1.1, below before);
* the force stays flag-clean and momentum-conserving (``|sum m a| / sum m |a|`` at
  round-off);
* the run is deterministic: a second run of the same segments is bitwise equal.

Marker ``distributed_segment``, job ``test-distributed-segment-cuts``.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_mesh_work_cuts.py -q
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from tests.distributed import test_mesh_cross_momentum as tcm  # noqa: E402
from tests.distributed import test_mesh_potential as tmp  # noqa: E402
from tests.distributed.test_mesh_potential import mac  # noqa: E402,F401

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

N = 2048
CAP = int(1.6 * N / 2)
DT = 1e-3


def _clumpy_ic(seed=0):
    rng = np.random.default_rng(seed)
    n_clump = N // 3
    cube = rng.uniform(-1.0, 1.0, (N - n_clump, 3))
    clump = 0.7 + 0.05 * rng.normal(size=(n_clump, 3))
    pos = np.concatenate([cube, clump])
    vel = 0.05 * rng.normal(size=(N, 3))
    return pos, vel, np.full(N, 1.0 / N)


def _run(mac_name, segments=2):
    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    mesh = tmp._mesh(2)
    pos, vel, mass = _clumpy_ic()
    shard, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    runner = MeshSegmentRunner.fused(
        tcm._solver(mac_name),
        mesh,
        MeshState.from_shard_state(shard, mesh),
        leaf_size=tmp.LEAF,
        max_order=tmp.ORDER,
        theta=tmp.THETA,
        repartition_every=1,
        num_samples=64,
        migrate=True,
        cuts="work",
    )
    reports = [runner.run(1, dt=DT) for _ in range(segments)]
    return runner, reports


def test_work_cuts_even_the_work_and_keep_the_force(mac):
    if mac != "dehnen":
        pytest.skip("one MAC: the cuts read the topology, not the criterion")
    runner, reports = _run(mac)
    assert runner.cuts == "work"
    assert all(r.bits == 0 and r.retries == 0 for r in reports), [
        (r.bits, r.retries) for r in reports
    ]
    work = reports[-1].work
    assert work, "the rebalance before the second segment reported no work"
    before, after = (np.asarray(w, np.float64) for w in work)
    imb_before = before.max() / before.mean()
    imb_after = after.max() / after.mean()
    print(f"work per device before {before.tolist()}, after {after.tolist()}")
    assert imb_before > 1.1, imb_before  # the case is unbalanced
    assert imb_after <= 1.1 and imb_after < imb_before, (imb_before, imb_after)

    g = runner.state.gather()
    m = np.asarray(g["masses"], np.float64)
    acc = np.asarray(g["acceleration"], np.float64)
    residual = np.linalg.norm((m[:, None] * acc).sum(0)) / np.sum(
        m * np.linalg.norm(acc, axis=1)
    )
    assert residual < tcm.TOL, residual

    # deterministic: the same segments again, bitwise
    again, _ = _run(mac)
    g2 = again.state.gather()
    for name in ("positions", "velocities", "acceleration"):
        tmp._assert_bitwise(g2[name], g[name], f"work cuts, run twice: {name}")

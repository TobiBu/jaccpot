"""The LET exchange of the cross summaries equals the all_gather exchange (M4).

``make_cross_hook(exchange="let")`` sends each peer only the part of a device's
summary tree its cross walk can open (:mod:`jaccpot.distributed.let`) instead of
all_gathering every tree. Both ends of a cross pair walk the same sent and received
blocks, so the pair sets agree by construction; they equal the all_gather path's
because the rows each peer receives are found by the walk itself (a request walk
against every peer's coarse tree, :func:`jaccpot.distributed.let.let_exchange`). Here, end to end on the fused mesh lane
(the CPU route of ``test_mesh_cross_momentum.py``: fp64, uniform cube, leaf 64, p4),
the start force under both exchanges on the same decomposition:

* every device's far and near pair sets (in the gathered ``[ndev x S]`` index
  space) are the all_gather path's (the requests are exact: the request walk splits
  a cut node exactly when the full walk does);
* the accelerations are BITWISE the all_gather path's (the same pairs, the same
  shipped multipoles and particles, the same (target, source) order);
* ``|sum m a| / sum m |a|`` at round-off;
* the LET receives no more rows than the all_gather's live summary rows.

``dehnen`` on four devices and ``dehnen_error`` (eq (16a), the descriptor's binned
tables) on two. Marker ``distributed_segment``, job ``test-distributed-segment-let``.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_mesh_let_exchange.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import decompose  # noqa: E402
from jaccpot.distributed.template_force import FusedMeshForce  # noqa: E402
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

#: the devices per MAC: the geometric MAC on four (more peers, real pruning), the
#: criterion on two (its compile is the longer one)
NDEV = {"dehnen": 4, "dehnen_error": 2}
#: particles per device (the momentum test's 1024)
PER_DEVICE = 1024
#: the LET's request rounds (``CrossCapacities.let_coarse_particles``): small, so
#: the toy trees are mostly requested subtree by subtree
COARSE_PARTICLES = (8 * 64, 64)

_RECORD = (
    "walk_far_a",
    "walk_far_b",
    "walk_far_count",
    "walk_near_a",
    "walk_near_b",
    "walk_near_count",
    "summary_nodes",
)
_LET_RECORD = ("let_send_rows", "let_recv_rows", "let_overflow", "let_requested")


def _probe(force, shard, f_b, keys):
    """The start force with the hook's record per device and the momentum rows."""
    from jax.sharding import PartitionSpec as P

    record = force.cross_record

    def body(x, m, n, ids, *fb):
        live = jnp.minimum(n[0], int(x.shape[0]))
        out = force.shard(x, m, live, ids, *fb)
        rows = (jnp.arange(int(x.shape[0])) < live)[:, None]
        acc = jnp.where(rows, out[0].astype(jnp.float64), 0.0)
        mm = m.astype(jnp.float64)[:, None]
        mom = jnp.concatenate([jnp.sum(mm * acc, 0), jnp.sum(mm * jnp.abs(acc), 0)])
        rec = {k: jnp.asarray(record[k])[None] for k in keys}
        return acc, mom[None], jnp.asarray(out[1]).reshape(1), rec

    spec = P(tmp.AXIS)
    args = (shard.positions, shard.masses, shard.count, shard.ids)
    args += () if f_b is None else (f_b,)
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=force.mesh,
            in_specs=(spec,) * len(args),
            out_specs=(spec, spec, spec, {k: spec for k in keys}),
            check_vma=False,
        )
    )
    with force.context():
        force.before_trace()
        acc, mom, bits, rec = program(*args)
    return (
        np.asarray(acc),
        np.asarray(mom),
        np.asarray(bits),
        {k: np.asarray(v) for k, v in rec.items()},
    )


def _pairs(rec, device, kind):
    a = rec[f"walk_{kind}_a"][device]
    b = rec[f"walk_{kind}_b"][device]
    count = int(rec[f"walk_{kind}_count"][device])
    live = (np.arange(a.shape[0]) < count) & (a >= 0) & (b >= 0)
    return set(zip(np.minimum(a, b)[live].tolist(), np.maximum(a, b)[live].tolist()))


def test_the_let_exchange_is_the_all_gather_exchange(mac):
    ndev = NDEV[mac]
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    mesh = tmp._mesh(ndev)
    n = PER_DEVICE * ndev
    pos, vel, mass = tmp._cube(n, seed=0)
    pos, vel, mass = (np.asarray(a, np.float64) for a in (pos, vel, mass))
    shard, _ = decompose(
        mesh, pos, vel, mass, cap=int(1.3 * PER_DEVICE), num_samples=64
    )
    caps = CrossCapacities(max_cells=tmp.LEAF_CAP, leaf_width=tmp.LEAF)
    caps = caps._with(
        let_send_cap=(ndev - 1) * caps.max_summary_nodes,
        let_recv_cap=(ndev - 1) * caps.max_summary_nodes,
        # small subtrees: the requests decide most rows of the toy trees
        let_coarse_particles=COARSE_PARTICLES,
    )
    solver = tcm._solver(mac)
    got = {}
    for exchange in ("all_gather", "let"):
        force = FusedMeshForce(
            solver,
            mesh,
            leaf_size=tmp.LEAF,
            max_order=tmp.ORDER,
            theta=tmp.THETA,
            cross_caps=caps,
            cross_record={},
            cross_exchange=exchange,
        )
        force.prepare(shard.positions, shard.masses, shard.count)
        assert force.hook is not None and force.hook.exchange == exchange
        f_b = force.force_scale_seed if force.scale_carry else None
        keys = _RECORD + (_LET_RECORD if exchange == "let" else ())
        got[exchange] = _probe(force, shard, f_b, keys)
        assert not np.any(got[exchange][2]), (exchange, got[exchange][2])

    acc_g, _, _, rec_g = got["all_gather"]
    acc_l, mom_l, _, rec_l = got["let"]
    assert not np.any(rec_l["let_overflow"])
    for d in range(ndev):
        for kind in ("far", "near"):
            want, have = _pairs(rec_g, d, kind), _pairs(rec_l, d, kind)
            assert have == want, (d, kind, len(want - have), len(have - want))
        assert len(_pairs(rec_l, d, "far")) > 10
    tmp._assert_bitwise(acc_l, acc_g, "the LET exchange's acceleration")

    m = np.asarray(shard.masses, np.float64)
    scale = float(np.sum(m * np.linalg.norm(acc_l, axis=1)))
    residual = float(np.linalg.norm(mom_l[:, :3].sum(0))) / scale
    assert residual < tcm.TOL, residual

    # the rows: never more than the all_gather's live summaries, and every peer's
    # root at least
    live = rec_g["summary_nodes"].astype(np.int64)
    gathered = np.array([live.sum() - live[d] for d in range(ndev)])
    recv = rec_l["let_recv_rows"].astype(np.int64)
    print(
        f"{mac} x{ndev}: LET rows received {recv.tolist()}, all_gather {gathered.tolist()}"
    )
    assert np.all(recv <= gathered) and np.all(recv >= ndev - 1)
    # the walks asked for subtrees (the requests did something)
    assert int(rec_l["let_requested"].sum()) > 0

"""Cost-weighted cuts (M4): ``migration.work_balanced_splitters_shard``.

On forced CPU devices (``tests/unit/distributed/test_fused_rollout.py``'s job, which
forces four), each device's live rows an exact Morton range: the cuts put equal
WORK on every device (to one row's work), reduce to the count-balanced cuts for
uniform work, and with ``max_rows`` never give a device more rows than that.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from jaccpot.distributed.migration import (
    balanced_splitters_shard,
    work_balanced_splitters_shard,
)

AXIS = "gpus"
NDEV = 4
CAP = 512

pytestmark = pytest.mark.skipif(
    jax.device_count() < NDEV, reason=f"needs {NDEV} forced devices"
)


def _state(seed=0):
    """Exact Morton ranges: ``n`` clustered particles sorted by code, cut evenly."""
    from yggdrax.morton import morton_encode_impl

    from jaccpot.distributed.fused import global_mesh_bounds  # noqa: F401

    rng = np.random.default_rng(seed)
    n = 1600
    pos = np.concatenate(
        [rng.normal(0.0, 0.05, (n // 2, 3)), rng.uniform(-1.0, 1.0, (n - n // 2, 3))]
    )
    lo, hi = pos.min(0), pos.max(0)
    side = float(np.max(hi - lo))
    codes = np.asarray(
        morton_encode_impl(jnp.asarray(pos), (jnp.asarray(lo), jnp.asarray(lo + side)))
    )
    pos = pos[np.argsort(codes, kind="stable")]
    # work: the first quarter of the Morton curve costs 10x per particle (a dense
    # region a count cut leaves on one device)
    work = np.where(np.arange(n) < n // 4, 10.0, 1.0)
    shard = np.zeros((NDEV, CAP, 3))
    wk = np.zeros((NDEV, CAP))
    counts = np.full(NDEV, n // NDEV, np.int32)
    for d in range(NDEV):
        shard[d, : counts[d]] = pos[d * (n // NDEV) : (d + 1) * (n // NDEV)]
        wk[d, : counts[d]] = work[d * (n // NDEV) : (d + 1) * (n // NDEV)]
    return shard.reshape(-1, 3), wk.reshape(-1), counts, pos, work


def _cuts(fn, positions, counts, work=None, **kw):
    """The cuts, and per device the rows and work it would own under them (the
    owners from the devices' own codes, in the force's box)."""
    from jaccpot.distributed.migration import _codes

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:NDEV]), (AXIS,))
    spec = P(AXIS)

    def body(x, c, w, *use):
        cuts = (
            fn(x, c, w, cubic=True, axis_name=AXIS, **kw)
            if use
            else fn(x, c, cubic=True, axis_name=AXIS, **kw)
        )
        n = jnp.minimum(c[0], x.shape[0])
        live = jnp.arange(x.shape[0]) < n
        dest = jnp.searchsorted(
            cuts, _codes(x, n, cubic=True, axis_name=AXIS), side="right"
        )
        dest = jnp.where(live, dest, NDEV)
        rows = jax.ops.segment_sum(live.astype(jnp.int32), dest, NDEV + 1)[:NDEV]
        wsum = jax.ops.segment_sum(jnp.where(live, w, 0.0), dest, NDEV + 1)[:NDEV]
        return cuts, jax.lax.psum(rows, AXIS), jax.lax.psum(wsum, AXIS)

    use = () if fn is balanced_splitters_shard else (work,)
    args = (positions, counts, work) + use
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * len(args),
            out_specs=(P(), P(), P()),
            check_vma=False,
        )
    )
    return tuple(np.asarray(a) for a in program(*args))


def test_the_work_is_balanced_and_uniform_work_gives_the_count_cuts():
    positions, work, counts, _, _ = _state()
    _, rows0, before = _cuts(balanced_splitters_shard, positions, counts, work)
    assert before.max() / before.mean() > 1.3, before  # the test case is unbalanced
    _, rows, per_device = _cuts(work_balanced_splitters_shard, positions, counts, work)
    # equal to within one row's work
    assert per_device.max() - per_device.min() <= 2 * 10.0, per_device
    assert rows.sum() == rows0.sum()
    uniform = np.where(work > 0, 1.0, 0.0)
    _, rows_u, _ = _cuts(work_balanced_splitters_shard, positions, counts, uniform)
    assert np.abs(rows_u - rows0).max() <= 1, (rows_u, rows0)


def test_max_rows_bounds_every_device():
    positions, work, counts, _, _ = _state()
    limit = 460
    _, free, _ = _cuts(work_balanced_splitters_shard, positions, counts, work)
    assert free.max() > limit, free  # the bound binds
    _, rows, _ = _cuts(
        work_balanced_splitters_shard, positions, counts, work, max_rows=limit
    )
    assert rows.max() <= limit, rows

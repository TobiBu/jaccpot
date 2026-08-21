"""Acceptance criteria for the distributed mutual force (TobiBu/jaccpot#173).

Run with several devices, as the yggdrax distributed tests do::

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
        pytest tests/integration/test_mutual_distributed.py

The criteria from the issue, and why each is phrased the way it is:

* **Total momentum across ALL devices**, never per device. Both failure modes --
  dropping a cross pair's `-f` and double-counting it -- leave every per-device sum
  exact, because `+f`/`-f` still cancel within whatever each device did do. Only a
  global sum notices, which is exactly what makes them dangerous.
* **Force parity.** With the cross walk driven at theta = 0 every cross pair is
  leaf-leaf and summed exactly, so a configuration whose intra-domain part is also
  exact must reproduce a direct sum to round-off. That isolates plumbing bugs from
  approximation error completely.
* **Overflow raised across devices**, since a cap exceeded on one device is a wrong
  force everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jaccpot.mutual.distributed import distributed_mutual_accelerations

try:
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
except ImportError:  # pragma: no cover
    shard_map = None

from yggdrax.distributed import device_count, make_mesh
from yggdrax.distributed.sharding import AXIS_NAME

pytestmark = pytest.mark.skipif(
    device_count() < 2 or shard_map is None,
    reason="the distributed mutual force needs >= 2 devices and shard_map",
)

SOFT = 1e-2
LEAF = 4
PER_DEV = 16


def _ndev():
    return min(4, device_count())


def _complete_tree(n_leaves):
    """Child arrays for a complete binary tree, leaves last, -1 for leaf children."""
    total = 2 * n_leaves - 1
    left = np.full(total, -1, dtype=np.int32)
    right = np.full(total, -1, dtype=np.int32)
    for i in range(n_leaves - 1):
        left[i], right[i] = 2 * i + 1, 2 * i + 2
    return left, right, total


def _domain(seed, offset, n, leaf):
    """A domain's particles plus an explicit tree over them, contiguous by leaf.

    Built by hand rather than through a tree builder: the point under test is the
    cross-domain partition and the return path, so the tree only has to be a valid,
    consistent structure with centres of mass and covering radii.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.3, size=(n, 3)) + np.asarray(offset)
    mass = rng.uniform(0.5, 1.5, size=n)
    n_leaves = n // leaf
    left, right, total = _complete_tree(n_leaves)
    ranges = np.zeros((total, 2), dtype=np.int32)
    for li in range(n_leaves):
        node = n_leaves - 1 + li
        ranges[node] = (li * leaf, (li + 1) * leaf)
    for i in range(n_leaves - 2, -1, -1):
        ranges[i] = (ranges[left[i]][0], ranges[right[i]][1])
    centers = np.zeros((total, 3))
    radii = np.zeros(total)
    for i in range(total):
        s, e = ranges[i]
        m = mass[s:e]
        centers[i] = (pos[s:e] * m[:, None]).sum(0) / m.sum()
        radii[i] = np.max(np.linalg.norm(pos[s:e] - centers[i], axis=1))
    return (
        jnp.asarray(pos),
        jnp.asarray(mass),
        jnp.asarray(left),
        jnp.asarray(right),
        jnp.asarray(ranges),
        jnp.asarray(centers),
        jnp.asarray(radii),
        n_leaves,
    )


def _exact_all(pos, mass):
    """Exact softened acceleration for the whole system."""
    d = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(d * d, axis=-1) + SOFT**2
    inv3 = np.where(np.eye(len(pos), dtype=bool), 0.0, r2 ** (-1.5))
    return np.einsum("ij,j,ijk->ik", inv3, mass, d)


def _exact_within(pos, mass, owner):
    """Exact acceleration from same-domain pairs only -- the local lane's share."""
    d = pos[None, :, :] - pos[:, None, :]
    r2 = np.sum(d * d, axis=-1) + SOFT**2
    same = owner[:, None] == owner[None, :]
    inv3 = np.where(np.eye(len(pos), dtype=bool) | ~same, 0.0, r2 ** (-1.5))
    return np.einsum("ij,j,ijk->ik", inv3, mass, d)


def _run(nd, caps=None):
    doms = [_domain(7 + d, (2.0 * d, 0.0, 0.0), PER_DEV, LEAF) for d in range(nd)]
    n_leaves = doms[0][7]
    stack = lambda i: jnp.stack([d[i] for d in doms])  # noqa: E731
    pos_all = np.concatenate([np.asarray(d[0]) for d in doms])
    mass_all = np.concatenate([np.asarray(d[1]) for d in doms])
    owner_all = np.repeat(np.arange(nd), PER_DEV)
    local_ref = _exact_within(pos_all, mass_all, owner_all).reshape(nd, PER_DEV, 3)

    c = caps or dict(
        near_cap=n_leaves * n_leaves * nd,
        max_pair_queue=4096,
        recv_capacity=nd * n_leaves * n_leaves * nd * LEAF,
    )

    def body(pos, mass, left, right, ranges, centers, radii, local):
        return distributed_mutual_accelerations(
            pos[0],
            mass[0],
            left[0],
            right[0],
            ranges[0],
            centers[0],
            radii[0],
            jnp.asarray(0, dtype=jnp.int32),
            local[0],
            softening=SOFT,
            ndev=nd,
            leaf_width=LEAF,
            **c,
        )

    out = shard_map(
        body,
        mesh=make_mesh(nd),
        in_specs=(P(AXIS_NAME),) * 8,
        out_specs=P(AXIS_NAME),
        check_rep=False,
    )(
        stack(0),
        stack(1),
        stack(2),
        stack(3),
        stack(4),
        stack(5),
        stack(6),
        jnp.asarray(local_ref),
    )
    return out, pos_all, mass_all


def test_force_matches_an_exact_direct_sum():
    """With cross pairs exact and the local share exact, this must be a direct sum."""
    nd = _ndev()
    out, pos_all, mass_all = _run(nd)
    assert not bool(out.overflow.reshape(-1)[0]), "a capacity overflowed"
    got = np.asarray(out.acceleration).reshape(-1, 3)
    want = _exact_all(pos_all, mass_all)
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-12, f"relative force error {rel:.3e}"
    assert int(np.asarray(out.cross_pairs).sum()) > 0, "no cross pairs -- vacuous"


def test_total_momentum_conserves_across_devices():
    """The criterion the issue insists on: a GLOBAL sum, not a per-device one."""
    nd = _ndev()
    out, _pos, mass_all = _run(nd)
    acc = np.asarray(out.acceleration).reshape(-1, 3)
    p = (mass_all[:, None] * acc).sum(axis=0)
    scale = np.abs(mass_all[:, None] * acc).sum()
    assert (
        np.linalg.norm(p) / scale < 1e-14
    ), f"global momentum residual {np.linalg.norm(p) / scale:.3e}"


def test_a_starved_capacity_raises_across_devices():
    """A cap exceeded on ONE device must surface everywhere: the force is wrong for all."""
    nd = _ndev()
    out, _p, _m = _run(nd, caps=dict(near_cap=1, max_pair_queue=4096, recv_capacity=8))
    flags = np.asarray(out.overflow).reshape(-1)
    assert flags.all(), f"overflow not reduced across devices: {flags}"

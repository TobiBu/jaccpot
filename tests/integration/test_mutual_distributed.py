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

Force parity turns out to be the only one of the three that catches an ownership-key
mismatch -- the failure mode a locally essential tree introduces, where two devices
number the same remote node differently and so disagree about who owns a pair. Measured
here with the key deliberately dropped: the force went 8.6e-3 wrong, global momentum
stayed at 3.4e-17, and the cross-pair COUNT came out exactly right, because the pairs
claimed twice and the pairs claimed by nobody happened to balance. So a census of
emitted pairs is not a substitute for the direct-sum comparison, however much it looks
like the more direct test of a partition.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# The LET path needs two things from yggdrax that arrived with it: the reverse-halo
# return addresses, and the cross walk's `remote_index_in_owner`. Both land together in
# TobiBu/yggdrax#48.
#
# Guarded rather than imported bare because an unguarded import fails at COLLECTION
# time, which errors the whole test tier rather than skipping one module. CI installs
# yggdrax from `git+.../yggdrax.git`, i.e. its default branch -- so the window this
# closes is a jaccpot PR that needs a yggdrax change which has not merged yet, which is
# exactly the window this test was written in. The signature is checked as well as the
# import, so the skip names the real requirement instead of waiting to fail with a
# TypeError deep inside a shard_map trace.
try:
    import inspect

    from yggdrax.distributed.cross_walk import dual_tree_walk_cross_mutual

    from jaccpot.mutual.distributed import distributed_mutual_accelerations

    _needs_yggdrax = (
        None
        if "remote_index_in_owner"
        in inspect.signature(dual_tree_walk_cross_mutual).parameters
        else "a yggdrax whose cross-mutual walk takes remote_index_in_owner"
    )
except ImportError as _exc:  # pragma: no cover - yggdrax predates the LET reverse halo
    _needs_yggdrax = f"a newer yggdrax ({_exc})"

if _needs_yggdrax is not None:  # pragma: no cover - depends on the installed yggdrax
    pytest.skip(
        f"the distributed mutual force needs {_needs_yggdrax} "
        "-- see TobiBu/yggdrax#48",
        allow_module_level=True,
    )

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

    ``node_ranges`` are **inclusive** ``[start, end]``, which is what yggdrax's tree
    builders emit and what the LET's frontier records are read with -- the force now
    hands these ranges straight to ``build_coarse_frontier``, so a half-open tree here
    would name the wrong particles rather than merely disagree about a convention.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(scale=0.3, size=(n, 3)) + np.asarray(offset)
    mass = rng.uniform(0.5, 1.5, size=n)
    n_leaves = n // leaf
    left, right, total = _complete_tree(n_leaves)
    ranges = np.zeros((total, 2), dtype=np.int32)
    for li in range(n_leaves):
        node = n_leaves - 1 + li
        ranges[node] = (li * leaf, (li + 1) * leaf - 1)
    for i in range(n_leaves - 2, -1, -1):
        ranges[i] = (ranges[left[i]][0], ranges[right[i]][1])
    centers = np.zeros((total, 3))
    radii = np.zeros(total)
    for i in range(total):
        s, e = ranges[i]
        m = mass[s : e + 1]
        centers[i] = (pos[s : e + 1] * m[:, None]).sum(0) / m.sum()
        radii[i] = np.max(np.linalg.norm(pos[s : e + 1] - centers[i], axis=1))
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


def _run(nd, caps=None, dead_leaves=0):
    """Run the distributed force over ``nd`` synthetic domains.

    ``dead_leaves`` zeroes the masses of that many of domain 0's trailing leaves,
    which is what an equalised real partition looks like: the domain is padded to a
    common capacity and the padding carries no mass. It leaves the centres and radii
    as they were, which is consistent because those feed only the MAC, and the MAC is
    never consulted at theta = 0.
    """
    doms = [_domain(7 + d, (2.0 * d, 0.0, 0.0), PER_DEV, LEAF) for d in range(nd)]
    n_leaves = doms[0][7]
    mats = [[np.array(x) for x in d[:7]] for d in doms]
    for k in range(dead_leaves):
        start = (n_leaves - 1 - k) * LEAF
        mats[0][1][start : start + LEAF] = 0.0
    stack = lambda i: jnp.stack([jnp.asarray(m[i]) for m in mats])  # noqa: E731
    pos_all = np.concatenate([m[0] for m in mats])
    mass_all = np.concatenate([m[1] for m in mats])
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


def test_padding_costs_only_force_on_the_padding_itself():
    """A massless frontier leaf is dropped, and that has to cost nothing that matters.

    An equalised partition pads each domain to a common capacity, so some leaves carry
    no mass. Such a leaf is published with a placeholder centre of mass and no origin
    node id, so it has no address to return `-f` to, and both sides of any pair
    involving it drop it. The contract that makes that acceptable is narrow and worth
    pinning: forces on MASSIVE particles stay exact, and global momentum stays exact;
    only force on the padding itself is given up.
    """
    nd = _ndev()
    out, pos_all, mass_all = _run(nd, dead_leaves=1)
    assert not bool(out.overflow.reshape(-1)[0]), "a capacity overflowed"
    got = np.asarray(out.acceleration).reshape(-1, 3)
    want = _exact_all(pos_all, mass_all)
    live = mass_all > 0
    rel = np.linalg.norm(got[live] - want[live]) / np.linalg.norm(want[live])
    assert rel < 1e-12, f"massive-particle force error {rel:.3e}"

    p = (mass_all[:, None] * got).sum(axis=0)
    scale = np.abs(mass_all[:, None] * got).sum()
    assert np.linalg.norm(p) / scale < 1e-14, "global momentum broke"

    # The guard has to have actually fired, or this passes for the wrong reason: with
    # nothing dropped the run is just the all-massive case again.
    n_leaves = PER_DEV // LEAF
    census = (nd * (nd - 1) // 2) * n_leaves * n_leaves
    assert (
        int(np.asarray(out.cross_pairs).sum()) < census
    ), "no pair was dropped, so the massless-leaf path was never taken"


def test_a_starved_capacity_raises_across_devices():
    """A cap exceeded on ONE device must surface everywhere: the force is wrong for all."""
    nd = _ndev()
    out, _p, _m = _run(nd, caps=dict(near_cap=1, max_pair_queue=4096, recv_capacity=8))
    flags = np.asarray(out.overflow).reshape(-1)
    assert flags.all(), f"overflow not reduced across devices: {flags}"


# ---------------------------------------------------------------------------
# The driver. Everything above drives `distributed_mutual_accelerations`, which is
# one device's share; these drive the entry point that owns the host side.
# ---------------------------------------------------------------------------

DRIVER_N = 128
DRIVER_LEAF = 8


def _driver_config(theta=0.0):
    from jaccpot.mutual.distributed import DistributedMutualConfig

    return DistributedMutualConfig(
        leaf_size=DRIVER_LEAF, theta=theta, order=4, softening=SOFT
    )


def _random_system(seed=17, n=DRIVER_N):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)), rng.uniform(0.5, 1.5, size=n)


def test_the_driver_reproduces_an_exact_direct_sum():
    """End to end from loose particles, with no tree or mesh supplied by the caller.

    The driver owns the SFC split across devices, each domain's own tree, the
    intra-domain force and the cross-domain half. At ``theta = 0`` every part of that
    is exact -- cross pairs by construction, and intra-domain because this MAC accepts
    nothing at small theta -- so a direct sum is the reference, not an approximation
    to compare against loosely.
    """
    from jaccpot.mutual.distributed import distributed_mutual_fmm

    nd = _ndev()
    pos, mass = _random_system()
    got = distributed_mutual_fmm(
        jnp.asarray(pos), jnp.asarray(mass), config=_driver_config(), ndev=nd
    )
    assert not got.overflow, (
        f"a capacity overflowed: cross={got.cross_overflow.tolist()} "
        f"local={got.local_overflow.tolist()} "
        f"causes={got.local_overflow_causes.tolist()}"
    )
    acc = np.asarray(got.accelerations)
    want = _exact_all(pos, mass)
    rel = np.linalg.norm(acc - want) / np.linalg.norm(want)
    assert rel < 1e-12, f"relative force error {rel:.3e}"

    p = (mass[:, None] * acc).sum(axis=0)
    scale = np.abs(mass[:, None] * acc).sum()
    assert np.linalg.norm(p) / scale < 1e-14, "global momentum broke"

    # Vacuity guard: with one domain there would be no cross pairs at all, and the
    # test would be checking the single-device lane through a longer pipe.
    assert int(np.asarray(got.cross_pairs).sum()) > 0, "no cross pairs -- vacuous"


def test_the_driver_returns_accelerations_in_the_callers_order():
    """Permuting the input must permute the output, and nothing else.

    The driver unwinds TWO permutations -- the SFC split across devices and each
    domain's own tree sort -- and a bug in either is invisible to a norm against a
    reference computed in the same order, because both sides would be scrambled
    identically. Feeding a shuffled copy of the same system is what separates them:
    the physics is unchanged, so the answer has to be the same vectors, just moved.
    """
    from jaccpot.mutual.distributed import distributed_mutual_fmm

    nd = _ndev()
    pos, mass = _random_system()
    rng = np.random.default_rng(99)
    order = rng.permutation(len(pos))

    a_plain = np.asarray(
        distributed_mutual_fmm(
            jnp.asarray(pos), jnp.asarray(mass), config=_driver_config(), ndev=nd
        ).accelerations
    )
    a_shuf = np.asarray(
        distributed_mutual_fmm(
            jnp.asarray(pos[order]),
            jnp.asarray(mass[order]),
            config=_driver_config(),
            ndev=nd,
        ).accelerations
    )

    # Not exact equality: a different input order changes which domain a particle
    # lands in and therefore the summation order, so the two agree to round-off
    # rather than bit-for-bit. A permutation bug is orders of magnitude away from
    # that, so the tolerance does not blunt the test.
    rel = np.linalg.norm(a_shuf - a_plain[order]) / np.linalg.norm(a_plain)
    assert rel < 1e-12, f"output order does not follow the input order: {rel:.3e}"

    # And the shuffled run is still a direct sum, so this cannot pass by both runs
    # being wrong the same way.
    want = _exact_all(pos[order], mass[order])
    assert np.linalg.norm(a_shuf - want) / np.linalg.norm(want) < 1e-12


def test_the_driver_refuses_a_single_device():
    """One domain has no cross-domain pairs, so the lane is the wrong tool for it."""
    from jaccpot.mutual.distributed import distributed_mutual_fmm

    pos, mass = _random_system(n=32)
    with pytest.raises(ValueError, match="needs >= 2 devices"):
        distributed_mutual_fmm(
            jnp.asarray(pos), jnp.asarray(mass), config=_driver_config(), ndev=1
        )


def test_the_driver_names_which_half_starved():
    """A starved capacity is reported, and reported specifically enough to act on.

    The driver runs two lanes with two independent capacity sets, so "overflow" alone
    would leave a caller guessing which to raise. Starving the INTRA-domain caps must
    set the local flag and leave the cross flag clear -- and name the cause, since a
    truncated walk's own counters undercount and cannot be read after the fact.

    Only the intra-domain half is starved here; the cross half's own starvation is
    covered by ``test_a_starved_capacity_raises_across_devices``, and each extra
    configuration is another compile in a suite that is already the slow one.
    """
    from jaccpot.mutual.distributed import (
        DistributedMutualConfig,
        distributed_mutual_fmm,
    )
    from jaccpot.mutual.force import OVERFLOW_CAUSES, MutualCapacities

    nd = _ndev()
    pos, mass = _random_system()
    got = distributed_mutual_fmm(
        jnp.asarray(pos),
        jnp.asarray(mass),
        config=DistributedMutualConfig(
            leaf_size=DRIVER_LEAF,
            theta=0.0,
            order=4,
            softening=SOFT,
            caps=MutualCapacities(near=1, far=1, depth=4, width=4, queue=64),
        ),
        ndev=nd,
    )
    assert got.overflow, "a starved intra-domain capacity was not reported"
    assert (
        got.local_overflow.all()
    ), f"local flag not set: {got.local_overflow.tolist()}"
    assert not got.cross_overflow.any(), (
        "the CROSS flag fired for an intra-domain starvation, so the two halves are "
        f"not distinguishable: {got.cross_overflow.tolist()}"
    )
    near_bit = 1 << OVERFLOW_CAUSES.index("near")
    assert all(int(c) & near_bit for c in got.local_overflow_causes), (
        f"the 'near' cause was not named: {got.local_overflow_causes.tolist()} "
        f"against {OVERFLOW_CAUSES}"
    )

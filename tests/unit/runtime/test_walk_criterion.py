"""Dehnen eq (16a) in the flat walks: the walk table + per-pair test against the policy.

``jaccpot.runtime._walk_criterion`` re-expresses eq (15)/(16a) in a normalised,
gather-free form for the fused lane's flat walks. The general path's
``adaptive_pair_policy`` is the oracle. Pinned here:

* the per-pair test equals the policy's accept decision on every node pair of a
  real tree (float64, the policy fed the walk's own radii and centres), and the
  fixture exercises both outcomes;
* the normalised powers ``s_n`` lie in ``[0, 1]`` about the COM with exact radii;
* ``yggdrax.dual_tree_walk_mutual`` with :class:`DehnenWalkAccept` emits the same
  far and near SETS as the yggdrax dual walk running ``adaptive_pair_policy`` --
  the same criterion on the same tree through two different traversals.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax import DualTreeTraversalConfig, Tree
from yggdrax._geometry_impl import TreeGeometry
from yggdrax._interactions_impl import (
    _build_mac_extents,
    build_interactions_and_neighbors,
    dual_tree_walk_mutual,
)

from jaccpot.runtime._adaptive_policy import (
    AdaptivePolicyState,
    adaptive_pair_policy,
    dehnen_multipole_power_by_degree,
)
from jaccpot.runtime._walk_criterion import (
    DehnenWalkAccept,
    dehnen_pair_accept,
    dehnen_walk_table,
    walk_table_width,
)
from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep

ORDER = 4
LEAF = 16


def _plummer(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 0.95, n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    u = rng.normal(size=(n, 3))
    pos = r[:, None] * u / np.linalg.norm(u, axis=1, keepdims=True)
    return pos, rng.uniform(0.5, 1.5, n) / n


def _fixture(n: int = 1500, seed: int = 5):
    """A radix tree with COM centres, exact COM radii and its real-basis upward pass."""
    pos, mass = _plummer(n, seed)
    tree = Tree.from_particles(
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        tree_type="radix",
        target_leaf_particles=LEAF,
        refine_local=False,
    )
    ps = jnp.asarray(pos)[tree.particle_indices]
    ms = jnp.asarray(mass)[tree.particle_indices]
    upward = prepare_real_upward_sweep(
        tree, ps, ms, max_order=ORDER, max_leaf_size=LEAF
    )
    centers = np.asarray(upward.multipoles.centers)
    ranges = np.asarray(tree.node_ranges)
    p, m = np.asarray(ps), np.asarray(ms)
    radii = np.zeros(centers.shape[0])
    masses = np.zeros(centers.shape[0])
    for k, (lo, hi) in enumerate(ranges):
        if hi >= lo:
            radii[k] = np.sqrt(np.max(np.sum((p[lo : hi + 1] - centers[k]) ** 2, 1)))
            masses[k] = m[lo : hi + 1].sum()
    return tree, upward.multipoles.packed, masses, centers, radii


def _thresholds(num_nodes: int, seed: int, eps: float) -> np.ndarray:
    # a realistic spread of force scales: thresholds over three decades
    rng = np.random.default_rng(seed)
    return eps * 10.0 ** rng.uniform(-1.0, 2.0, num_nodes)


def _policy_state(packed, masses, centers, radii, thresholds, theta_max=1.0):
    """The general path's eq (16a) state, on the walk's own centres and radii."""
    power = dehnen_multipole_power_by_degree(multipole_packed=packed)
    p1 = ORDER + 1
    binom = np.asarray([[float(math.comb(ORDER, k)) for k in range(p1)]])
    n = int(power.shape[0])
    return AdaptivePolicyState(
        source_error_proxy_by_order=jnp.zeros((n, 1)),
        source_degree_power=jnp.zeros_like(power),
        source_dehnen_power=power,
        source_mass=jnp.asarray(np.maximum(masses, 1e-24)),
        source_mac_center=jnp.asarray(centers),
        target_mac_center=jnp.asarray(centers),
        source_radius_bound=jnp.asarray(radii),
        target_radius_bound=jnp.asarray(radii),
        target_accept_threshold=jnp.asarray(thresholds),
        order_tags=jnp.asarray([0], jnp.int32),
        order_values=jnp.asarray([ORDER], jnp.int32),
        order_values_float=jnp.asarray([float(ORDER)]),
        dehnen_binomial_masked_by_order=jnp.asarray(binom),
        dehnen_exponent_by_order=jnp.asarray(
            [[ORDER - k for k in range(p1)]], jnp.int32
        ),
        relaxed_theta_sq=jnp.asarray(1.0),
        error_model_code=jnp.asarray(2, jnp.int32),
        mac_theta_max=float(theta_max),
    )


def _table(packed, masses, radii, thresholds):
    return dehnen_walk_table(
        multipole_packed=packed,
        mass=jnp.asarray(masses),
        radius=jnp.asarray(radii),
        threshold=jnp.asarray(thresholds),
        gravitational_constant=1.0,
        order=ORDER,
    )


def test_table_width_is_a_power_of_two_holding_every_column():
    for p in range(0, 15):
        w = walk_table_width(p)
        assert w >= p + 2 and (w & (w - 1)) == 0


def test_normalised_powers_lie_in_the_unit_interval():
    tree, packed, masses, centers, radii = _fixture()
    table = np.asarray(_table(packed, masses, radii, np.ones(radii.shape[0])))
    s = table[:, 2 : 2 + ORDER]
    spans = np.asarray(tree.node_ranges)[:, 1] >= np.asarray(tree.node_ranges)[:, 0]
    assert np.all(s[spans] >= 0.0)
    assert np.all(s[spans] <= 1.0 + 1e-12), float(s[spans].max())
    # about the COM the dipole vanishes
    assert np.all(np.abs(s[spans, 0]) < 1e-10)
    # and the higher powers are not trivially zero
    assert np.max(s[spans, 1:]) > 0.1


@pytest.mark.parametrize("theta_max", [1.0, 0.7])
@pytest.mark.parametrize("eps", [3e-2, 1e-3])
def test_pair_test_equals_the_policy_on_every_node_pair(eps, theta_max):
    tree, packed, masses, centers, radii = _fixture()
    num_nodes = centers.shape[0]
    thresholds = _thresholds(num_nodes, 7, eps)
    state = _policy_state(packed, masses, centers, radii, thresholds, theta_max)
    a, b = np.triu_indices(num_nodes, k=1)
    d2 = jnp.asarray(np.sum((centers[a] - centers[b]) ** 2, 1))
    n = a.size
    actions, _ = adaptive_pair_policy(
        state,
        valid_pairs=jnp.ones((n,), bool),
        mac_ok=jnp.zeros((n,), bool),
        different_nodes=jnp.ones((n,), bool),
        target_leaf=jnp.zeros((n,), bool),
        source_leaf=jnp.zeros((n,), bool),
        same_node=jnp.zeros((n,), bool),
        target_nodes=jnp.asarray(a, jnp.int32),
        source_nodes=jnp.asarray(b, jnp.int32),
        center_target=jnp.asarray(centers[a]),
        center_source=jnp.asarray(centers[b]),
        dist_sq=d2,
        extent_target=jnp.asarray(radii[a]),
        extent_source=jnp.asarray(radii[b]),
    )
    want = np.asarray(actions) == 0
    table = _table(packed, masses, radii, thresholds)
    accept = DehnenWalkAccept(ORDER)(
        {"table": table, "theta_max": jnp.asarray(theta_max)},
        jnp.asarray(a),
        jnp.asarray(b),
        d2,
        jnp.asarray(radii[a]),
        jnp.asarray(radii[b]),
    )
    got = np.asarray(accept)
    assert want.sum() > 100 and (~want).sum() > 100, "both outcomes exercised"
    mismatch = np.nonzero(got != want)[0]
    assert mismatch.size == 0, f"{mismatch.size} of {n} pairs differ"


def test_pair_test_is_symmetric():
    tree, packed, masses, centers, radii = _fixture(seed=6)
    thresholds = _thresholds(centers.shape[0], 8, 1e-4)
    table = np.asarray(_table(packed, masses, radii, thresholds))
    a, b = np.triu_indices(centers.shape[0], k=1)
    d2 = np.sum((centers[a] - centers[b]) ** 2, 1)
    cols = range(2 + ORDER)

    def run(x, y):
        return np.asarray(
            dehnen_pair_accept(
                row_a=[jnp.asarray(table[x, k]) for k in cols],
                row_b=[jnp.asarray(table[y, k]) for k in cols],
                radius_a=jnp.asarray(radii[x]),
                radius_b=jnp.asarray(radii[y]),
                dist_sq=jnp.asarray(d2),
                order=ORDER,
                theta_max=1.0,
            )
        )

    np.testing.assert_array_equal(run(a, b), run(b, a))


def _walk_inputs(tree, centers, radii):
    topo = tree.topology
    num_internal = int(topo.left_child.shape[0])
    total = int(topo.parent.shape[0])
    geometry = TreeGeometry(
        center=jnp.asarray(centers),
        half_extent=jnp.zeros((total, 3)) + jnp.asarray(radii)[:, None],
        radius=jnp.asarray(radii),
        max_extent=jnp.asarray(radii),
    )
    extents = np.asarray(
        _build_mac_extents(topo.parent, geometry, num_internal, "dehnen", 1.0)[0]
    )
    idx = topo.parent.dtype
    left = jnp.concatenate(
        [jnp.asarray(topo.left_child, idx), jnp.full((total - num_internal,), -1, idx)]
    )
    right = jnp.concatenate(
        [jnp.asarray(topo.right_child, idx), jnp.full((total - num_internal,), -1, idx)]
    )
    root = jnp.argmin(topo.parent).astype(idx)
    return topo, geometry, extents, left, right, root


@pytest.mark.parametrize("eps", [3e-2, 1e-3])
def test_mutual_walk_with_the_criterion_equals_the_dual_walk_with_the_policy(eps):
    tree, packed, masses, centers, radii = _fixture(n=2500, seed=9)
    topo, geometry, extents, left, right, root = _walk_inputs(tree, centers, radii)
    thresholds = _thresholds(centers.shape[0], 10, eps)
    # both sides test the walk extents (for leaves the depth-padded proxy)
    state = _policy_state(packed, masses, centers, extents, thresholds)
    config = DualTreeTraversalConfig(
        max_pair_queue=1 << 17,
        process_block=256,
        max_interactions_per_node=8192,
        max_neighbors_per_leaf=8192,
    )
    _, neighbors, result = build_interactions_and_neighbors(
        topo,
        geometry,
        theta=0.5,
        traversal_config=config,
        mac_type="dehnen",
        pair_policy=adaptive_pair_policy,
        policy_state=state,
        return_result=True,
    )
    assert not (
        bool(result.queue_overflow)
        or bool(result.far_overflow)
        or bool(result.near_overflow)
    )
    src = np.asarray(result.interaction_sources)
    tgt = np.asarray(result.interaction_targets)
    live = (src >= 0) & (tgt >= 0)
    far_ref = {(min(x, y), max(x, y)) for x, y in zip(tgt[live], src[live])}
    near_ref = set()
    offsets, counts = np.asarray(neighbors.offsets), np.asarray(neighbors.counts)
    nbrs, leaves = np.asarray(neighbors.neighbors), np.asarray(neighbors.leaf_indices)
    for row, leaf in enumerate(leaves.tolist()):
        for k in range(int(counts[row])):
            other = int(nbrs[int(offsets[row]) + k])
            near_ref.add((min(leaf, other), max(leaf, other)))

    table = _table(packed, masses, extents, thresholds)

    @jax.jit
    def walk(data):
        return dual_tree_walk_mutual(
            left,
            right,
            jnp.asarray(centers),
            jnp.asarray(extents),
            0.5,
            root,
            max_pair_queue=1 << 17,
            far_cap=1 << 18,
            near_cap=1 << 18,
            pair_accept=DehnenWalkAccept(ORDER),
            pair_accept_data=data,
        )

    res = walk({"table": table, "theta_max": jnp.asarray(1.0)})
    assert not (
        bool(res.queue_overflow) or bool(res.far_overflow) or bool(res.near_overflow)
    )
    nf, nn = int(res.far_count), int(res.near_count)
    far = set(
        zip(np.asarray(res.far_a)[:nf].tolist(), np.asarray(res.far_b)[:nf].tolist())
    )
    near = set(
        zip(np.asarray(res.near_a)[:nn].tolist(), np.asarray(res.near_b)[:nn].tolist())
    )
    assert len(far) > 0 and len(near) > 0
    assert far == far_ref
    assert near == near_ref
    # and the criterion is not the geometric walk in disguise
    geo = dual_tree_walk_mutual(
        left,
        right,
        jnp.asarray(centers),
        jnp.asarray(extents),
        0.5,
        root,
        max_pair_queue=1 << 17,
        far_cap=1 << 18,
        near_cap=1 << 18,
        mac_type="dehnen",
    )
    ng = int(geo.far_count)
    assert (
        set(
            zip(
                np.asarray(geo.far_a)[:ng].tolist(), np.asarray(geo.far_b)[:ng].tolist()
            )
        )
        != far
    )


def test_dehnen_power_feeds_the_table():
    """The table's s_n are the eq (12) powers divided by M rho^n."""
    tree, packed, masses, centers, radii = _fixture(n=400, seed=2)
    table = np.asarray(_table(packed, masses, radii, np.ones(radii.shape[0])))
    power = np.asarray(dehnen_multipole_power_by_degree(multipole_packed=packed))
    mass = masses
    ok = radii > 0
    for n in range(1, ORDER + 1):
        np.testing.assert_allclose(
            table[ok, 1 + n], power[ok, n] / (mass[ok] * radii[ok] ** n), rtol=1e-12
        )
    np.testing.assert_allclose(table[:, 0], mass, rtol=1e-15)

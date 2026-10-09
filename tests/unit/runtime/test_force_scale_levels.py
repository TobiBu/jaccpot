"""Level-order reductions for the per-step force scale (jaccpot.runtime._force_scale_levels).

* ``ancestor_sum_by_level`` equals a numpy walk up the parent chain (what the
  serial ``accumulate_own_down_parent_chain`` computes);
* ``subtree_min_by_level`` equals a brute-force minimum over each node's leaves;
* ``far_force_scale_own`` pushed down equals the eager estimator's far-term formula
  (``_far_field_force_scale_by_node``) evaluated in numpy on the same far pairs;

on a bucket tree and on the capacity-padded cell tree the fused lane uses (empty
leaves included), eagerly and under ``jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax._tree_impl import build_static_cells_tree, build_static_radix_tree
from yggdrax.bounds import infer_bounds
from yggdrax.tree import get_level_offsets, get_nodes_by_level

from jaccpot.runtime._force_scale_levels import (
    ancestor_sum_by_level,
    far_force_scale_own,
    subtree_min_by_level,
)
from jaccpot.runtime._level_shapes import level_batch_width


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 0.97, n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    u = rng.normal(size=(n, 3))
    return r[:, None] * u / np.linalg.norm(u, axis=1, keepdims=True)


def _tree(kind, n=3000, leaf=16, seed=4):
    P = jnp.asarray(_plummer(n, seed), jnp.float64)
    M = jnp.asarray(np.random.default_rng(seed).uniform(0.5, 1.5, n) / n)
    if kind == "buckets":
        topo, ps, ms, _ = build_static_radix_tree(
            P, M, infer_bounds(P), leaf_size=leaf, return_reordered=True
        )
    else:
        topo, ps, ms, _ = build_static_cells_tree(
            P,
            M,
            infer_bounds(P),
            leaf_size=leaf,
            leaf_capacity=1024,
            return_reordered=True,
        )
    return topo, ps, ms


def _chain_sum(parent, own):
    out = np.array(own, dtype=np.float64)
    for k in range(parent.shape[0]):
        p = int(parent[k])
        while p >= 0:
            out[k] += own[p]
            p = int(parent[p])
    return out


def _tables(topo):
    offsets = get_level_offsets(topo)
    nodes = get_nodes_by_level(topo)
    total = int(topo.parent.shape[0])
    ni = int(topo.left_child.shape[0])
    return dict(
        left_child=topo.left_child,
        right_child=topo.right_child,
        parent=topo.parent,
        nodes_by_level=nodes,
        level_offsets=offsets,
        num_internal=ni,
        num_levels=int(offsets.shape[0] - 1),
        level_batch_width=level_batch_width(
            offsets, total_nodes=total, num_internal=ni
        ),
    )


@pytest.mark.parametrize("kind", ["buckets", "cells"])
def test_ancestor_sum_equals_the_serial_push_down(kind):
    topo, _, _ = _tree(kind)
    own = jnp.asarray(np.random.default_rng(1).uniform(0.0, 1.0, topo.parent.shape[0]))
    want = _chain_sum(np.asarray(topo.parent), np.asarray(own))
    kw = _tables(topo)
    got = np.asarray(ancestor_sum_by_level(own, **kw))
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
    jitted = jax.jit(
        lambda o: ancestor_sum_by_level(
            o,
            kw["left_child"],
            kw["right_child"],
            kw["parent"],
            kw["nodes_by_level"],
            kw["level_offsets"],
            num_internal=kw["num_internal"],
            num_levels=kw["num_levels"],
            level_batch_width=kw["level_batch_width"],
        )
    )
    np.testing.assert_allclose(np.asarray(jitted(own)), want, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("kind", ["buckets", "cells"])
def test_subtree_min_equals_the_minimum_over_each_nodes_leaves(kind):
    topo, _, _ = _tree(kind)
    ranges = np.asarray(topo.node_ranges)
    total, ni = int(topo.parent.shape[0]), int(topo.left_child.shape[0])
    rng = np.random.default_rng(2)
    leaf_vals = np.full(total, np.inf)
    spans = ranges[:, 1] >= ranges[:, 0]
    leaf_vals[ni:] = np.where(spans[ni:], rng.uniform(0.1, 10.0, total - ni), np.inf)
    got = np.asarray(subtree_min_by_level(jnp.asarray(leaf_vals), **_tables(topo)))
    lo_leaf, hi_leaf = ranges[ni:, 0], ranges[ni:, 1]
    live_leaf = spans[ni:]
    for k in range(total):
        if not spans[k]:
            continue
        inside = live_leaf & (lo_leaf >= ranges[k, 0]) & (hi_leaf <= ranges[k, 1])
        want = leaf_vals[ni:][inside].min()
        assert got[k] == want, (k, got[k], want)
    if kind == "cells":
        assert (~live_leaf).any(), "the cell tree must carry empty leaves"


def test_far_term_pushed_down_equals_the_eager_estimator_formula():
    from yggdrax.tree_moments import compute_tree_mass_moments

    from jaccpot.runtime._mac_geometry import com_mac_geometry

    topo, ps, ms = _tree("buckets", n=2000)
    mm = compute_tree_mass_moments(topo, ps, ms)
    geom = com_mac_geometry(topo, ps, mm.center_of_mass, leaf_cap=16)
    total = int(topo.parent.shape[0])
    rng = np.random.default_rng(3)
    src = rng.integers(0, total, 4000)
    tgt = rng.integers(0, total, 4000)
    src[-50:] = -1  # dead tail, as a capacity-padded list
    tgt[-50:] = -1
    G, soft_sq = 1.3, 1e-3
    c, r, m = np.asarray(geom.center), np.asarray(geom.radius), np.asarray(mm.mass)
    own_ref = np.zeros(total)
    for a, b in zip(src, tgt):
        if a < 0 or b < 0:
            continue
        reach = np.linalg.norm(c[a] - c[b]) + r[b]
        if reach > 0:
            own_ref[b] += G * m[a] / (reach * reach + soft_sq)
    want = _chain_sum(np.asarray(topo.parent), own_ref)
    own = far_force_scale_own(
        sources=jnp.asarray(src, jnp.int32),
        targets=jnp.asarray(tgt, jnp.int32),
        live=jnp.asarray((src >= 0) & (tgt >= 0)),
        node_mass=jnp.asarray(m),
        node_centers=geom.center,
        node_radii=geom.radius,
        gravitational_constant=G,
        softening_sq=jnp.asarray(soft_sq),
        num_nodes=total,
    )
    np.testing.assert_allclose(np.asarray(own), own_ref, rtol=1e-12, atol=1e-300)
    got = np.asarray(ancestor_sum_by_level(own, **_tables(topo)))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("target_sorted", [False, True])
@pytest.mark.parametrize("chunk", [7, 1000, 1 << 22])
def test_the_chunked_far_term_equals_the_whole_list_one(
    monkeypatch, target_sorted, chunk
):
    """The far term walks its list in fixed chunks; a non-dividing chunk (the last
    window clamped back over the previous one) must count every pair once, for the
    COO list and for the target-sorted one whose ``targets`` are CSR offsets."""
    from yggdrax.interactions import CompactTaggedFarPairs
    from yggdrax.tree_moments import compute_tree_mass_moments

    from jaccpot.runtime import _force_scale_levels as fsl
    from jaccpot.runtime._interaction_cache import TargetSortedFarPairs
    from jaccpot.runtime._mac_geometry import com_mac_geometry

    topo, ps, ms = _tree("buckets", n=2000)
    mm = compute_tree_mass_moments(topo, ps, ms)
    geom = com_mac_geometry(topo, ps, mm.center_of_mass, leaf_cap=16)
    total = int(topo.parent.shape[0])
    rng = np.random.default_rng(5)
    live_n, pad = 3001, 37
    src = rng.integers(0, total, live_n)
    tgt = rng.integers(0, total, live_n)
    if target_sorted:
        order = np.argsort(tgt, kind="stable")
        src, tgt = src[order], tgt[order]
        offsets = np.searchsorted(tgt, np.arange(total + 1), side="left")
        pairs = TargetSortedFarPairs(
            sources=jnp.asarray(np.concatenate([src, -np.ones(pad, int)]), jnp.int32),
            targets=jnp.asarray(offsets, jnp.int32),
            tags=jnp.zeros((live_n + pad,), jnp.int32),
            far_pair_count=jnp.asarray(live_n, jnp.int32),
        )
    else:
        pairs = CompactTaggedFarPairs(
            sources=jnp.asarray(np.concatenate([src, -np.ones(pad, int)]), jnp.int32),
            targets=jnp.asarray(np.concatenate([tgt, -np.ones(pad, int)]), jnp.int32),
            tags=jnp.zeros((live_n + pad,), jnp.int32),
            far_pair_count=jnp.asarray(live_n, jnp.int32),
        )
    leaves = jnp.arange(int(topo.left_child.shape[0]), total, dtype=jnp.int32)
    kw = dict(
        tree=topo,
        leaf_nodes=leaves,
        far_pairs=pairs,
        node_mass=jnp.asarray(mm.mass),
        node_centers=geom.center,
        node_radii=geom.radius,
        gravitational_constant=1.3,
        softening_sq=jnp.asarray(1e-3),
        num_levels=int(get_level_offsets(topo).shape[0] - 1),
        num_particles=int(ps.shape[0]),
    )
    monkeypatch.setattr(fsl, "_FAR_PAIR_CHUNK", 1 << 30)
    whole = np.asarray(fsl.far_force_scale_sorted(**kw))
    monkeypatch.setattr(fsl, "_FAR_PAIR_CHUNK", chunk)
    got = np.asarray(fsl.far_force_scale_sorted(**kw))
    assert whole.max() > 0
    np.testing.assert_allclose(got, whole, rtol=1e-12, atol=0.0)

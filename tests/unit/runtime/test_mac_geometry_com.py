"""The walk's MAC geometry about the expansion centres (plan sub-10ms, Phase 1.2).

Pinned here on a real radix tree, on CPU:

* ``com_mac_geometry`` radii bound every particle of every node about the COM
  (exact on the leaves, an upper bound on internal nodes);
* fed to the flat walk, every accepted far pair satisfies the convergence
  condition about the COMs, ``(r_A + r_B) / d <= theta`` with the EXACT radii;
* the historical box geometry does not give that guarantee on the same tree
  (the ratio about the COMs exceeds theta for some accepted pair), which is the
  defect this module exists to remove;
* the env knob selects the mode and rejects unknown values.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.runtime._interaction_cache import (
    _build_flat_walk_artifacts_strict_streamed,
)
from jaccpot.runtime._mac_geometry import (
    com_mac_geometry,
    mac_geometry_mode,
    resolve_walk_geometry,
)

_LEAF = 8
_N = 1024
from tests.unit._typecheck_budget import trim

_INTERNALS = trim(["exact", "bound"])
_THETAS = trim([0.6, 0.9])


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.fixture(scope="module")
def tree_data():
    pos = jnp.asarray(_plummer(_N), jnp.float64)
    mass = jnp.asarray(np.random.default_rng(1).uniform(0.5, 1.5, _N), jnp.float64)
    tree = Tree.from_particles(pos, mass, leaf_size=_LEAF, tree_type="radix")
    topo = tree.topology
    ps = jnp.asarray(tree.positions_sorted)
    ms = jnp.asarray(tree.masses_sorted)
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    box = compute_tree_geometry(topo, ps, max_leaf_size=_LEAF)
    return tree, topo, ps, ms, com, box


def _exact_rmax(topo, ps, centers):
    ranges = np.asarray(topo.node_ranges)
    ps = np.asarray(ps)
    c = np.asarray(centers)
    out = np.zeros(ranges.shape[0])
    for i, (a, b) in enumerate(ranges):
        seg = ps[a : b + 1]
        if seg.shape[0]:
            out[i] = np.sqrt(np.max(np.sum((seg - c[i]) ** 2, axis=1)))
    return out


@pytest.mark.parametrize("internal", _INTERNALS)
def test_com_geometry_bounds_every_node_and_is_exact_on_leaves(tree_data, internal):
    tree, topo, ps, ms, com, _box = tree_data
    geom = com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, internal=internal)
    num_internal = int(topo.left_child.shape[0])
    exact = _exact_rmax(topo, ps, com)
    r = np.asarray(geom.radius)
    assert np.allclose(np.asarray(geom.center), np.asarray(com))
    assert np.all(r + 1e-12 >= exact), "a node's particles leave its MAC sphere"
    assert np.allclose(r[num_internal:], exact[num_internal:], rtol=1e-12, atol=1e-12)
    if internal == "exact":
        # every internal node too: the ancestor walk sees all of a node's particles
        assert np.allclose(r, exact, rtol=1e-12, atol=1e-12)
    else:
        # the child-sphere bound compounds over deep chains (3x over exact seen on
        # this 1024-particle leaf-8 tree); it costs far pairs, never accuracy
        assert np.all(np.isfinite(r))
    assert np.allclose(np.asarray(geom.max_extent), r)
    assert np.allclose(np.asarray(geom.half_extent), r[:, None])


def test_internal_mode_is_validated(tree_data):
    tree, topo, ps, ms, com, _box = tree_data
    with pytest.raises(ValueError):
        com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, internal="tight")


def _far_pair_ratios(tree, geometry, centers, exact_r, *, theta):
    art = _build_flat_walk_artifacts_strict_streamed(
        tree=tree,
        geometry=geometry,
        theta=theta,
        mac_type="dehnen",
        dehnen_radius_scale=1.0,
        compact_far_pair_capacity=1 << 16,
        near_edge_capacity=1 << 16,
        max_pair_queue=1 << 14,
    )
    cfp = art.compact_far_pairs
    cnt = int(cfp.far_pair_count)
    src = np.asarray(cfp.sources)[:cnt]
    tgt = np.asarray(cfp.targets)[:cnt]
    keep = (src >= 0) & (tgt >= 0) & (src < tgt)
    src, tgt = src[keep], tgt[keep]
    c = np.asarray(centers)
    d = np.linalg.norm(c[src] - c[tgt], axis=1)
    return (exact_r[src] + exact_r[tgt]) / d, int(src.size)


@pytest.mark.parametrize("theta", _THETAS)
def test_flat_walk_with_com_geometry_accepts_only_convergent_pairs(tree_data, theta):
    tree, topo, ps, ms, com, box = tree_data
    exact = _exact_rmax(topo, ps, com)
    geom = com_mac_geometry(topo, ps, com, leaf_cap=_LEAF)
    ratios, n_pairs = _far_pair_ratios(tree, geom, com, exact, theta=theta)
    assert n_pairs > 100
    assert float(ratios.max()) <= theta + 1e-9
    # non-vacuity of the whole exercise: the box geometry lets divergent-side
    # pairs through -- about the COMs the accepted pairs overshoot theta
    ratios_box, _ = _far_pair_ratios(tree, box, com, exact, theta=theta)
    assert float(ratios_box.max()) > theta


def test_mode_knob(monkeypatch, tree_data):
    tree, topo, ps, ms, com, box = tree_data
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", raising=False)
    # The default belongs to the CALLER, not the environment: only the strict
    # fused lane passes "com". A global "com" default silently changed the
    # meaning of theta for every other lane -- the box radius is the
    # half-diagonal, so the same theta admits more far pairs under COM.
    assert mac_geometry_mode() == "aabb"
    assert mac_geometry_mode("com") == "com"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "aabb")
    assert mac_geometry_mode() == "aabb"
    assert mac_geometry_mode("com") == "aabb"  # the environment overrides the caller
    g, f = resolve_walk_geometry(
        topo, ps, box, com, leaf_cap=_LEAF, geometry_factory="factory"
    )
    assert g is box and f == "factory"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "com")
    g, f = resolve_walk_geometry(
        topo, ps, box, com, leaf_cap=_LEAF, geometry_factory="factory"
    )
    assert f is None and np.allclose(np.asarray(g.center), np.asarray(com))
    with pytest.raises(RuntimeError):
        resolve_walk_geometry(topo, ps, box, None, leaf_cap=_LEAF)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "bogus")
    with pytest.raises(ValueError):
        mac_geometry_mode()
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "com")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS", "bound")
    g_b, _ = resolve_walk_geometry(topo, ps, box, com, leaf_cap=_LEAF)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_RADIUS", "exact")
    g_e, _ = resolve_walk_geometry(topo, ps, box, com, leaf_cap=_LEAF)
    assert np.all(np.asarray(g_b.radius) + 1e-12 >= np.asarray(g_e.radius))
    assert float(np.max(np.asarray(g_b.radius) - np.asarray(g_e.radius))) > 0


@pytest.mark.parametrize("internal", _INTERNALS)
def test_com_geometry_is_jittable(tree_data, internal):
    tree, topo, ps, ms, com, box = tree_data
    f = jax.jit(
        lambda ps, com: com_mac_geometry(
            topo, ps, com, leaf_cap=_LEAF, internal=internal
        )
    )
    g = f(ps, com)
    ref = com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, internal=internal)
    assert np.allclose(np.asarray(g.radius), np.asarray(ref.radius))


def test_a_folded_per_node_criterion_survives_the_com_geometry(tree_data):
    """`dehnen_theta` folds its criterion into the radii; COM mode must keep it.

    Between the Phase 6 default switch and 2026-09-12 the COM walk geometry
    recomputed every radius from the expansion centres, which DISCARDED the
    folded criterion: two injected force scales three orders of magnitude apart
    gave byte-identical accept masks (11486 far pairs either way).
    """
    from jaccpot.runtime._mac_geometry import resolve_walk_geometry

    tree, topo, ps, _ms, com, box = tree_data
    nodes = int(box.radius.shape[0])
    plain, _ = resolve_walk_geometry(
        topo, ps, box, com, leaf_cap=_LEAF, default_mode="com"
    )
    scale = jnp.full((nodes,), 4.0, box.radius.dtype)
    scaled, _ = resolve_walk_geometry(
        topo, ps, box, com, leaf_cap=_LEAF, radius_scale=scale, default_mode="com"
    )
    assert np.allclose(np.asarray(scaled.radius), 4.0 * np.asarray(plain.radius))
    assert np.array_equal(np.asarray(scaled.center), np.asarray(plain.center))
    with pytest.raises(ValueError, match="one factor per node"):
        resolve_walk_geometry(
            topo,
            ps,
            box,
            com,
            leaf_cap=_LEAF,
            radius_scale=jnp.ones((nodes + 1,), box.radius.dtype),
            default_mode="com",
        )


def _table_radii_reference(topo, ps, centers, leaf_cap):
    """The (leaves x 64 levels) table ``com_mac_geometry`` computed until 2026-10-04."""
    from jax import lax

    ranges = jnp.asarray(topo.node_ranges)
    parent = jnp.asarray(topo.parent)
    num_nodes = int(ranges.shape[0])
    num_internal = int(jnp.asarray(topo.left_child).shape[0])
    num_leaves = num_nodes - num_internal
    n = int(ps.shape[0])
    leaf_ranges = ranges[num_internal:]
    lane = jnp.arange(leaf_cap)
    idx = leaf_ranges[:, 0][:, None] + lane[None, :]
    valid = idx <= leaf_ranges[:, 1][:, None]
    pts = ps[jnp.clip(idx, 0, n - 1)]
    d = jnp.linalg.norm(pts - centers[num_internal:][:, None, :], axis=-1)
    r_leaf = jnp.max(jnp.where(valid, d, 0.0), axis=1)
    radii = jnp.zeros((num_nodes,), ps.dtype).at[num_internal:].set(r_leaf)
    parent_safe = jnp.where(parent >= 0, parent, 0)

    def _up(anc, _):
        live = anc >= 0
        nxt = jnp.where(live, parent_safe[jnp.where(live, anc, 0)], -1)
        nxt = jnp.where(live & (parent[jnp.where(live, anc, 0)] >= 0), nxt, -1)
        return nxt, anc

    leaf_ids = jnp.arange(num_internal, num_nodes)
    _, anc_t = lax.scan(_up, parent[leaf_ids], None, length=64)
    anc = anc_t.T
    live = anc >= 0
    c = centers[jnp.where(live, anc, 0)]
    dd = jnp.linalg.norm(pts[:, :, None, :] - c[:, None, :, :], axis=-1)
    dd = jnp.max(jnp.where(valid[:, :, None], dd, 0.0), axis=1)
    d_all = jnp.where(live, dd, 0.0)

    def _seg(a, b):
        return b[0], jnp.where(a[0] == b[0], jnp.maximum(a[1], b[1]), b[1])

    _, run_max = lax.associative_scan(_seg, (anc, d_all), axis=0)
    last = jnp.concatenate([anc[1:] != anc[:-1], jnp.ones((1, 64), bool)]) & live
    target = jnp.where(last, anc, num_nodes)
    radii = jnp.concatenate([radii, jnp.zeros((1,), ps.dtype)])
    return radii.at[target.reshape(-1)].max(run_max.reshape(-1))[:num_nodes]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_level_passes_equal_the_ancestor_table(tree_data, dtype):
    """The per-level passes give the radii of the (leaves x 64) table they replaced.

    Max is exact in any order; what differs is each distance's rounding: the
    table took the norm over a ``(L, w, 4, 3)`` broadcast, which XLA sums in a
    different order than over ``(L, w, 3)`` (a few ulp, internal nodes only; the
    passes are the ones within 1 ulp of a float64 numpy max). At the tree's own
    level count (the upward sweep's bound) the radii are those of the padded 64
    levels to the bit; a bound below the depth misses the top ancestors.
    """
    from yggdrax.tree import get_node_levels

    tree, topo, ps, ms, com, _box = tree_data
    ps = ps.astype(dtype)
    com = com.astype(dtype)
    ref = np.asarray(_table_radii_reference(topo, ps, com, _LEAF))
    eps = float(jnp.finfo(dtype).eps)
    depth = int(np.asarray(get_node_levels(topo)).max()) + 1
    full = np.asarray(com_mac_geometry(topo, ps, com, leaf_cap=_LEAF).radius)
    np.testing.assert_allclose(full, ref, rtol=8 * eps, atol=0)
    num_internal = int(topo.left_child.shape[0])
    assert np.array_equal(full[num_internal:], ref[num_internal:])  # leaves: same op
    bounded = com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, num_levels=depth)
    assert np.array_equal(np.asarray(bounded.radius), full)
    # non-vacuity of the bound: a short one (rounded up to whole passes of four
    # levels) leaves the deep leaves' top ancestors short of their radius
    assert depth > 6
    short = np.asarray(
        com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, num_levels=2).radius
    )
    assert np.all(short <= full) and np.any(short < full)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_pallas_chunks_equal_the_level_passes(tree_data, dtype, monkeypatch):
    """The Pallas chunk kernel (interpret mode) gives the XLA level passes' radii.

    Same squared distances, maxima in any order, one square root per node: equal
    to a few ulp at most (the 3-term sum may contract differently), at the padded
    64 levels and at the tree's own depth (chunks of eight overshoot it).
    """
    from yggdrax.tree import get_node_levels

    tree, topo, ps, ms, com, _box = tree_data
    ps = ps.astype(dtype)
    com = com.astype(dtype)
    eps = float(jnp.finfo(dtype).eps)
    depth = int(np.asarray(get_node_levels(topo)).max()) + 1
    for levels in (None, depth):
        monkeypatch.setenv("JACCPOT_COM_RADII_KERNEL", "xla")
        ref = np.asarray(
            com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, num_levels=levels).radius
        )
        monkeypatch.setenv("JACCPOT_COM_RADII_KERNEL", "interpret")
        got = np.asarray(
            com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, num_levels=levels).radius
        )
        np.testing.assert_allclose(got, ref, rtol=4 * eps, atol=0)
        assert np.all(got > 0)  # every node of this tree holds particles

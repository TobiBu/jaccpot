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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.runtime._interaction_cache import _build_flat_walk_artifacts_strict_streamed
from jaccpot.runtime._mac_geometry import (
    com_mac_geometry,
    mac_geometry_mode,
    resolve_walk_geometry,
)

_LEAF = 8
_N = 1024


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


@pytest.mark.parametrize("internal", ["exact", "bound"])
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
        tree=tree, geometry=geometry, theta=theta, mac_type="dehnen", dehnen_radius_scale=1.0,
        compact_far_pair_capacity=1 << 16, near_edge_capacity=1 << 16, max_pair_queue=1 << 14,
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


@pytest.mark.parametrize("theta", [0.6, 0.9])
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
    assert mac_geometry_mode() == "aabb"
    g, f = resolve_walk_geometry(topo, ps, box, com, leaf_cap=_LEAF, geometry_factory="factory")
    assert g is box and f == "factory"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MAC_GEOMETRY", "com")
    g, f = resolve_walk_geometry(topo, ps, box, com, leaf_cap=_LEAF, geometry_factory="factory")
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


@pytest.mark.parametrize("internal", ["exact", "bound"])
def test_com_geometry_is_jittable(tree_data, internal):
    tree, topo, ps, ms, com, box = tree_data
    f = jax.jit(lambda ps, com: com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, internal=internal))
    g = f(ps, com)
    ref = com_mac_geometry(topo, ps, com, leaf_cap=_LEAF, internal=internal)
    assert np.allclose(np.asarray(g.radius), np.asarray(ref.radius))

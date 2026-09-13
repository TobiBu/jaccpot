"""Cell leaves in the static radix lane (plan sub-10ms, Phase 1.2): build helper, geometry, walk.

CPU-only wiring tests: ``_build_tree_with_config`` with ``leaf_partition="cells"``
returns a fixed-shape radix tree padded to ``leaf_capacity`` whose empty nodes
never enter the flat walk's lists, and the COM MAC geometry is finite on it.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.bounds import infer_bounds
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.runtime._interaction_cache import (
    _build_flat_walk_artifacts_strict_streamed,
)
from jaccpot.runtime._mac_geometry import com_mac_geometry
from jaccpot.runtime.fmm_state import TreeBuilderConfig, _build_tree_with_config


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


_CFG = TreeBuilderConfig(
    mode="static_radix",
    target_leaf_particles=16,
    refine_local=False,
    max_refine_levels=0,
    aspect_threshold=8.0,
)


def _build(n=4000, leaf=16, cap=1024, seed=0):
    P = jnp.asarray(_plummer(n, seed), jnp.float32)
    M = jnp.ones((n,), jnp.float32)
    art = _build_tree_with_config(
        P,
        M,
        infer_bounds(P),
        tree_type="radix",
        tree_config=_CFG,
        leaf_size=leaf,
        workspace=None,
        jit_tree=False,
        refine_local=False,
        max_refine_levels=0,
        aspect_threshold=8.0,
        leaf_partition="cells",
        leaf_capacity=cap,
    )
    return art, P, M


def test_build_helper_returns_a_padded_cell_tree():
    art, P, M = _build()
    tree = art.tree
    assert art.max_leaf_size == 16 and art.cache_leaf_parameter == 16
    assert art.leaf_capacity_overflow is None  # eager build raises instead
    ranges = np.asarray(tree.node_ranges)
    num_internal = int(tree.left_child.shape[0])
    assert num_internal == 1023 and ranges.shape[0] == 2047
    leaves = ranges[num_internal:]
    occ = np.where(leaves[:, 1] >= leaves[:, 0], leaves[:, 1] - leaves[:, 0] + 1, 0)
    assert occ.sum() == 4000 and occ.max() <= 16 and (occ == 0).sum() > 0
    assert np.asarray(tree.positions_sorted).shape == (4000, 3)
    with pytest.raises(RuntimeError):
        _build(cap=64)
    with pytest.raises(ValueError):
        P2 = jnp.asarray(_plummer(100), jnp.float32)
        _build_tree_with_config(
            P2,
            jnp.ones((100,), jnp.float32),
            infer_bounds(P2),
            tree_type="radix",
            tree_config=_CFG,
            leaf_size=16,
            workspace=None,
            jit_tree=False,
            refine_local=False,
            max_refine_levels=0,
            aspect_threshold=8.0,
            leaf_partition="cells",
            leaf_capacity=None,
        )


def test_com_geometry_and_flat_walk_ignore_empty_nodes():
    art, P, M = _build()
    tree = art.tree
    ps, ms = jnp.asarray(art.positions_sorted), jnp.asarray(art.masses_sorted)
    com = compute_tree_mass_moments(tree.topology, ps, ms).center_of_mass
    geom = com_mac_geometry(tree, ps, com, leaf_cap=16)
    r = np.asarray(geom.radius)
    assert np.all(np.isfinite(r)) and np.all(r >= 0)
    ranges = np.asarray(tree.node_ranges)
    empty = np.flatnonzero(ranges[:, 1] < ranges[:, 0])
    assert empty.size > 0 and np.all(r[empty] == 0)
    for theta in (0.6, 0.9):
        art_w = _build_flat_walk_artifacts_strict_streamed(
            tree=tree,
            geometry=geom,
            theta=theta,
            mac_type="dehnen",
            dehnen_radius_scale=1.0,
            compact_far_pair_capacity=1 << 18,
            near_edge_capacity=1 << 18,
            max_pair_queue=1 << 16,
        )
        cfp = art_w.compact_far_pairs
        n_far = int(cfp.far_pair_count)
        src, tgt = np.asarray(cfp.sources)[:n_far], np.asarray(cfp.targets)[:n_far]
        assert (
            n_far > 0
            and not np.isin(src, empty).any()
            and not np.isin(tgt, empty).any()
        )
        nl = art_w.neighbor_list
        counts = np.asarray(nl.counts)
        num_internal = int(tree.left_child.shape[0])
        empty_leaf_rows = empty[empty >= num_internal] - num_internal
        assert np.all(counts[empty_leaf_rows] == 0), "an empty leaf has neighbours"
        nbrs = np.asarray(nl.neighbors)[: int(counts.sum())]
        assert not np.isin(nbrs, empty).any()

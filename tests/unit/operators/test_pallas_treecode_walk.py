"""Parity tests for the Pallas per-leaf treecode walk.

The Pallas kernel replays the pure-JAX walk
(:func:`jaccpot.experimental.treecode_walk.treecode_leaf_walk`) exactly — same
LIFO stack order, same append-in-pop-order — so the emitted far/near arrays,
counts, and overflow flag must be BIT-IDENTICAL, not merely set-equal.

These run in ``interpret=True`` mode so they exercise the kernel logic on CPU /
in CI where no Ampere+ GPU is available. On a supported GPU the same kernel is
validated on-device by ``test_gpu_matches_jax_reference`` below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._geometry_impl import compute_tree_geometry
from yggdrax.tree import Tree

from jaccpot.experimental.treecode_walk import treecode_leaf_walk
from jaccpot.pallas.treecode_walk_pallas import (
    pallas_treecode_walk_supported,
    treecode_leaf_walk_backend,
    treecode_leaf_walk_pallas,
)


def _build_bundle(n, seed, leaf_size, theta):
    """Host-precomputed walk input bundle (mirrors the coverage test)."""
    key = jax.random.PRNGKey(seed)
    pts = jax.random.uniform(key, (n, 3), dtype=jnp.float64)
    mass = jnp.ones((n,), dtype=jnp.float64)
    tree = Tree.from_particles(pts, mass, leaf_size=leaf_size)
    topo = tree.topology
    pos_sorted = getattr(tree, "positions_sorted", pts)
    geom = compute_tree_geometry(topo, pos_sorted, max_leaf_size=leaf_size)

    num_internal = int(topo.num_internal_nodes)
    total_nodes = int(topo.parent.shape[0])
    num_leaves = total_nodes - num_internal
    idx = topo.parent.dtype

    left_full = jnp.concatenate(
        [topo.left_child.astype(idx), jnp.full((num_leaves,), -1, dtype=idx)]
    )
    right_full = jnp.concatenate(
        [topo.right_child.astype(idx), jnp.full((num_leaves,), -1, dtype=idx)]
    )
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=idx)
    root_idx = jnp.argmin(topo.parent).astype(idx)
    return dict(
        leaf_nodes=leaf_nodes,
        centers=geom.center,
        mac_extents=geom.max_extent,
        left_full=left_full,
        right_full=right_full,
        theta_sq=jnp.asarray(theta * theta, dtype=jnp.float64),
        root_idx=root_idx,
        num_internal=num_internal,
        total_nodes=total_nodes,
    )


def _call(fn, b, cap, **kw):
    return fn(
        b["leaf_nodes"],
        b["centers"],
        b["mac_extents"],
        b["left_full"],
        b["right_full"],
        b["theta_sq"],
        b["root_idx"],
        num_internal=b["num_internal"],
        max_far=cap,
        max_near=cap,
        max_stack=2 * cap + 4,
        max_iters=b["total_nodes"] + 1,
        **kw,
    )


def _assert_identical(ref, got):
    assert bool(ref.overflow) == bool(got.overflow)
    assert np.array_equal(np.asarray(ref.far_count), np.asarray(got.far_count))
    assert np.array_equal(np.asarray(ref.near_count), np.asarray(got.near_count))
    assert np.array_equal(np.asarray(ref.far_nodes), np.asarray(got.far_nodes))
    assert np.array_equal(np.asarray(ref.near_leaves), np.asarray(got.near_leaves))


def test_supported_returns_bool():
    assert isinstance(pallas_treecode_walk_supported(), bool)


def test_backend_returns_known_value():
    assert treecode_leaf_walk_backend(prefer_pallas=True) in {"jax", "pallas"}
    assert treecode_leaf_walk_backend(prefer_pallas=False) == "jax"


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
@pytest.mark.parametrize(
    ("n", "seed", "leaf_size", "theta"),
    [(64, 1, 8, 0.5), (256, 3, 8, 0.5), (256, 3, 8, 0.9), (500, 7, 16, 0.6)],
)
def test_interpret_matches_pure_jax_walk(n, seed, leaf_size, theta):
    b = _build_bundle(n, seed, leaf_size, theta)
    cap = b["total_nodes"]  # generous: must not overflow
    ref = _call(treecode_leaf_walk, b, cap)
    got = _call(treecode_leaf_walk_pallas, b, cap, interpret=True)
    assert not bool(got.overflow)
    _assert_identical(ref, got)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
def test_interpret_matches_when_capacity_overflows():
    """Undersized capacities: overflow flag + truncated-but-consistent counts match."""
    b = _build_bundle(256, 3, 8, 0.5)
    cap = 4  # deliberately too small -> overflow on both paths
    ref = _call(treecode_leaf_walk, b, cap)
    got = _call(treecode_leaf_walk_pallas, b, cap, interpret=True)
    assert bool(ref.overflow) and bool(got.overflow)
    _assert_identical(ref, got)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs float64 (JAX_ENABLE_X64=1)"
)
@pytest.mark.skipif(
    not pallas_treecode_walk_supported(),
    reason="treecode-walk Pallas kernel requires an Ampere+ (sm_80+) GPU",
)
@pytest.mark.parametrize(
    ("n", "seed", "leaf_size", "theta"),
    [(256, 3, 8, 0.5), (500, 7, 16, 0.6)],
)
def test_gpu_matches_pure_jax_walk(n, seed, leaf_size, theta):
    b = _build_bundle(n, seed, leaf_size, theta)
    cap = b["total_nodes"]
    ref = _call(treecode_leaf_walk, b, cap)
    got = _call(treecode_leaf_walk_pallas, b, cap, interpret=False)
    assert not bool(got.overflow)
    _assert_identical(ref, got)

"""Tests for fused static-target-block cap auto-sizing (concentrated ICs).

The fused static-radix lane packs each target leaf's neighbour source leaves into
a fixed-shape ``(num_leaves, max_blocks_per_leaf, block_size)`` payload. A fixed
small cap fails for centrally-concentrated ICs whose dense inner leaves have very
high near-neighbour counts. These tests cover the two pieces that make the cap
auto-size: the env-config parsing (``auto`` sentinel) and the builder's capacity
signal that the pipeline grows against.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._large_n_nearfield import build_large_n_target_owned_blocks_static


def _env_config():
    # Import lazily so monkeypatched env is read at call time.
    from jaccpot.runtime._large_n_pipeline import _large_n_env_config_for_fmm

    class _Fmm:
        pass

    return _large_n_env_config_for_fmm(_Fmm())


def test_static_cap_auto_sentinel(monkeypatch):
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "auto")
    cfg = _env_config()
    assert bool(cfg["static_target_blocks_auto"]) is True
    assert int(cfg["static_target_blocks_max_per_leaf"]) == 0
    # ladder is extended well beyond the old 128 ceiling
    assert max(cfg["static_target_blocks_cap_options"]) >= 1024


def test_static_cap_explicit_int(monkeypatch):
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "64")
    cfg = _env_config()
    assert bool(cfg["static_target_blocks_auto"]) is False
    assert int(cfg["static_target_blocks_max_per_leaf"]) == 64


def _concentrated_tree(leaf_size=2, theta=0.5):
    """A small centrally-concentrated point cloud + its tree/neighbour list."""
    from yggdrax.geometry import compute_tree_geometry
    from yggdrax.interactions import build_leaf_neighbor_lists
    from yggdrax.tree import build_tree

    rng = np.random.default_rng(0)
    # dense core + sparse halo -> a leaf with an outsized neighbour count
    core = rng.normal(scale=0.03, size=(48, 3))
    halo = rng.uniform(-0.9, 0.9, size=(16, 3))
    pos = jnp.asarray(np.concatenate([core, halo], axis=0), jnp.float32)
    mass = jnp.ones((pos.shape[0],), jnp.float32)
    bounds = (jnp.array([-1.0, -1.0, -1.0]), jnp.array([1.0, 1.0, 1.0]))
    tree, ps, ms, _ = build_tree(
        pos, mass, bounds, return_reordered=True, leaf_size=leaf_size
    )
    geo = compute_tree_geometry(tree, ps)
    nl = build_leaf_neighbor_lists(tree, geo, theta=theta)
    return tree, nl


def test_static_builder_capacity_signal_and_autosize():
    tree, nl = _concentrated_tree()
    offsets = np.asarray(nl.offsets)
    counts = offsets[1:] - offsets[:-1]
    max_count = int(counts.max())
    block_size = 4
    required = -(-max_count // block_size)  # ceil
    assert required >= 1

    # A cap that is too small must report capacity_ok=False (the signal the
    # pipeline grows against) rather than silently truncating.
    too_small = max(1, required - 1)
    ids_s, mask_s, ok_s = build_large_n_target_owned_blocks_static(
        tree=tree,
        neighbor_list=nl,
        block_size=block_size,
        max_blocks_per_leaf=too_small,
    )
    assert bool(ok_s) is False

    # The auto-sized cap = ceil(max_count / block_size) must fit.
    ids_f, mask_f, ok_f = build_large_n_target_owned_blocks_static(
        tree=tree, neighbor_list=nl, block_size=block_size, max_blocks_per_leaf=required
    )
    assert bool(ok_f) is True
    num_leaves = int(np.asarray(nl.leaf_indices).shape[0])
    assert tuple(np.asarray(ids_f).shape) == (num_leaves, required, block_size)
    # every valid slot references an in-range leaf id
    idsf = np.asarray(ids_f)
    maskf = np.asarray(mask_f)
    assert idsf[maskf].min() >= 0
    assert idsf[maskf].max() < num_leaves

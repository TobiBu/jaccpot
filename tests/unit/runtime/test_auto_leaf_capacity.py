"""``TreeConfig(leaf_capacity="auto")``: the cell-leaf capacity sized by the library.

The fused lane pads its cell-leaf tree to a static ``leaf_capacity``. Every caller
used to size it by hand -- the benches with a numpy count over host-sorted Morton
codes, x1.15, rounded up to 1024 (``bench/fused_memory_budget.py`` before S3). That
count moved into :mod:`jaccpot.runtime.capacity_plan`, on device, and ``"auto"``
applies it at the solver's first eager prepare. Pinned here:

* the device count is the numpy reference's, leaf for leaf, and the rounding is the
  bench's (so a bench row sized the old way and one sized by ``"auto"`` compile the
  same shapes);
* ``"auto"`` resolves once, in the prepare's own bounds, and then sticks: a later
  configuration with more leaves raises the capacity error instead of re-sizing
  (the compiled programs' shapes would move);
* the library default is unchanged: cell leaves without a capacity still raise.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
from yggdrax.morton import morton_encode

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMAdvancedConfig, TreeConfig
from jaccpot.runtime.capacity_plan import auto_leaf_capacity, live_cell_leaves
from jaccpot.runtime.kernels._evaluate import _infer_bounds


def _plummer(n, seed=0, rmax=None):
    rng = np.random.default_rng(seed)
    x_max = 1.0 if rmax is None else rmax**3 / (rmax**2 + 1.0) ** 1.5
    x = rng.uniform(0.0, x_max, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)
    return pos.astype(np.float32)


def _blobs(n, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = centres[rng.integers(0, 4, n)] + 0.08 * rng.normal(size=(n, 3))
    return pos.astype(np.float32)


def _numpy_live(pos, leaf, cml):
    """The bench's host count (``fused_memory_budget._leaf_capacity`` before S3)."""
    p = jnp.asarray(pos)
    codes = np.sort(np.asarray(morton_encode(p, _infer_bounds(p))).astype(np.uint64))
    return int(
        adaptive_cell_leaf_partition_numpy(codes, leaf_size=leaf, min_level=cml)[0].size
    )


@pytest.mark.parametrize(
    "draw",
    [
        lambda n: _plummer(n),
        lambda n: _plummer(n, rmax=20.0),
        lambda n: _blobs(n),
    ],
    ids=["plummer", "plummer_clipped", "blobs"],
)
@pytest.mark.parametrize("leaf,cml", [(64, 8), (16, 0), (32, 3)])
def test_the_device_count_is_the_numpy_count(draw, leaf, cml):
    for n in (300, 5000, 40_000):
        pos = draw(n)
        p = jnp.asarray(pos)
        live = live_cell_leaves(
            p, leaf_size=leaf, cell_min_level=cml, bounds=_infer_bounds(p)
        )
        assert live == _numpy_live(pos, leaf, cml), (n, leaf, cml)


def test_the_capacity_rounds_as_the_bench_did():
    for live in (1, 341, 890, 1023, 1024, 12_335, 154_136, 6_001_234):
        for factor in (1.0, 1.15, 1.25, 2.0):
            bench = int(-(-int(np.ceil(factor * live)) // 1024) * 1024)
            assert auto_leaf_capacity(live, factor) == bench, (live, factor)
    assert auto_leaf_capacity(890) == 1024  # ceil(1.15 x 890) = 1024
    assert auto_leaf_capacity(891) == 2048  # ceil(1.15 x 891) = 1025
    assert auto_leaf_capacity(100, 1.15, quantum=64) == 128
    for bad in (dict(live=-1), dict(live=10, headroom=0.99), dict(live=10, quantum=0)):
        with pytest.raises(ValueError):
            auto_leaf_capacity(**bad)


def _solver(**tree_kwargs):
    tree = dict(mode="static_radix", leaf_partition="cells", cell_min_level=4)
    tree.update(tree_kwargs)
    return FastMultipoleMethod(
        preset="fast",
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(**tree)),
    )


def test_auto_is_resolved_lazily_and_validated():
    impl = _solver(leaf_capacity="auto")._impl
    assert impl._tree_leaf_capacity is None
    assert impl._tree_leaf_capacity_auto == 1.15
    impl = _solver(leaf_capacity="auto", leaf_headroom=1.5)._impl
    assert impl._tree_leaf_capacity_auto == 1.5
    impl = _solver(leaf_capacity=2048)._impl
    assert (impl._tree_leaf_capacity, impl._tree_leaf_capacity_auto) == (2048, None)
    # the library default is unchanged: cell leaves need a capacity
    with pytest.raises(ValueError, match="needs leaf_capacity"):
        _solver()
    # under JACCPOT_RUNTIME_TYPECHECK=1 the Literal hint refuses it first; both
    # refusals name the value
    with pytest.raises((ValueError, jaxtyping.TypeCheckError), match="automatic"):
        _solver(leaf_capacity="automatic")
    with pytest.raises(ValueError, match="leaf_headroom needs"):
        _solver(leaf_capacity=2048, leaf_headroom=1.2)
    with pytest.raises(ValueError, match="leaf_headroom must be >= 1"):
        _solver(leaf_capacity="auto", leaf_headroom=0.9)
    with pytest.raises(ValueError, match="needs leaf_partition='cells'"):
        FastMultipoleMethod(
            preset="fast",
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_capacity="auto")
            ),
        )


def test_auto_resolves_at_the_first_eager_prepare_and_sticks():
    leaf, n = 16, 512
    solver = _solver(leaf_capacity="auto")
    masses = jnp.full((n,), 1.0 / n, jnp.float32)
    # few cells: four tight blobs
    first = jnp.asarray(_blobs(n))
    live = live_cell_leaves(
        first, leaf_size=leaf, cell_min_level=4, bounds=_infer_bounds(first)
    )
    state = solver.prepare_state(first, masses, leaf_size=leaf, max_order=2)
    capacity = solver._impl._tree_leaf_capacity
    assert capacity == auto_leaf_capacity(live) == 1024
    assert int(state.tree.leaf_codes.shape[0]) == capacity
    # more leaves (473 against 160), still within the capacity: the shapes stay
    spread = jnp.asarray(
        np.random.default_rng(0).uniform(-1.0, 1.0, (n, 3)), jnp.float32
    )
    assert (
        live_cell_leaves(
            spread, leaf_size=leaf, cell_min_level=4, bounds=_infer_bounds(spread)
        )
        > live
    )
    state = solver.prepare_state(spread, masses, leaf_size=leaf, max_order=2)
    assert solver._impl._tree_leaf_capacity == capacity
    assert int(state.tree.leaf_codes.shape[0]) == capacity
    # outgrown (2,148 leaves): the capacity error, not a silent re-size
    many = 10_000
    with pytest.raises(RuntimeError) as info:
        solver.prepare_state(
            jnp.asarray(_plummer(many)),
            jnp.full((many,), 1.0 / many, jnp.float32),
            leaf_size=leaf,
            max_order=2,
        )
    # prepare_state chains the build's error under its capacity report
    assert "overflowed the leaf capacity" in str(info.value.__cause__)
    assert solver._impl._tree_leaf_capacity == capacity


def test_a_cubic_box_builds_cubic_cells_and_counts_its_capacity_in_them(monkeypatch):
    """``TreeConfig(cubic_bounds=True)`` (``preset="galaxy"``, S3b) reaches the
    tree: the prepare's Morton box is a cube on a flat, elongated cloud, and
    ``"auto"`` counts the cell leaves in that cube -- not in the per-axis box the
    library default builds (the control), whose cells inherit the cloud's 8:2:1."""
    monkeypatch.delenv("JACCPOT_CUBIC_BOUNDS", raising=False)
    leaf, n = 16, 512
    masses = jnp.full((n,), 1.0 / n, jnp.float32)
    pos = jnp.asarray(
        np.random.default_rng(1).uniform(-1.0, 1.0, (n, 3)) * [8.0, 2.0, 1.0],
        jnp.float32,
    )
    counted = {}
    for cubic in (True, False):
        solver = _solver(leaf_capacity="auto", cubic_bounds=cubic)
        state = solver.prepare_state(pos, masses, leaf_size=leaf, max_order=2)
        edges = np.asarray(state.tree.bounds_max) - np.asarray(state.tree.bounds_min)
        live = live_cell_leaves(
            pos, leaf_size=leaf, cell_min_level=4, bounds=_infer_bounds(pos, cubic)
        )
        assert solver._impl._tree_leaf_capacity == auto_leaf_capacity(live)
        assert int(state.tree.leaf_codes.shape[0]) == auto_leaf_capacity(live)
        counted[cubic] = live
        if cubic:
            np.testing.assert_allclose(edges, edges.max(), rtol=1e-6)
        else:
            assert edges.max() / edges.min() > 7.0
    assert counted[True] != counted[False]

"""``TreeConfig(cell_min_level="auto")`` (S10 follow-ups): the level floor picked by the library.

The galaxy preset floors leaves at Morton level 8, sized for production N: on a
1e5-1e6 disc that cuts 1-3 particles per leaf (W 3.5-5.6x the gradient's
workspace at a coarser floor, S10 session 2). ``"auto"`` (opt-in) picks the level
at most 8 whose OCCUPIED cells hold, on average, nearest ``leaf_target`` particles
(in log), at the first eager prepare, before ``leaf_capacity="auto"``. Pinned here:

* the per-level occupied-cell count is numpy's ``unique`` of the shifted codes,
  padding rows excluded;
* the rule -- the level whose occupied cells' mean occupancy is nearest the leaf
  target in log (uniform grids of known occupancy, the ceiling, small N);
* ``"auto"`` resolves once at the first eager prepare -- the capacity counted at
  the picked level -- and then sticks; a traced first build and a bad value are
  refused; the mesh refuses it.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.morton import morton_encode

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMAdvancedConfig, TreeConfig
from jaccpot.runtime.capacity_plan import (
    AUTO_CELL_MIN_LEVEL_MAX,
    auto_cell_min_level,
    auto_leaf_capacity,
    live_cell_leaves,
    occupied_cells,
)
from jaccpot.runtime.kernels._evaluate import _infer_bounds


def _grid(level: int, per_cell: int, seed: int = 0) -> np.ndarray:
    """``per_cell`` particles in every cell of a level-``level`` grid of [0, 1)^3."""
    rng = np.random.default_rng(seed)
    side = 2**level
    cells = np.stack(
        np.meshgrid(*(np.arange(side),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    # inside each cell, away from its faces; the box corners pin the bounds
    offsets = rng.uniform(0.25, 0.75, size=(cells.shape[0] * per_cell, 3))
    pos = (np.repeat(cells, per_cell, axis=0) + offsets) / side
    return np.concatenate([pos, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]).astype(
        np.float32
    )


def test_the_occupied_cells_are_numpys():
    rng = np.random.default_rng(3)
    pos = jnp.asarray(
        np.concatenate([rng.normal(size=(3000, 3)), 5 * rng.normal(size=(50, 3))]),
        jnp.float32,
    )
    bounds = _infer_bounds(pos)
    codes = np.asarray(morton_encode(pos, bounds)).astype(np.uint64)
    counts = occupied_cells(pos, bounds=bounds, max_level=8)
    want = [np.unique(codes >> np.uint64(3 * (21 - lv))).size for lv in range(9)]
    assert counts.tolist() == want
    # padding rows past num_valid are in no cell
    live = occupied_cells(pos, bounds=bounds, max_level=8, num_valid=1000)
    want = [np.unique(codes[:1000] >> np.uint64(3 * (21 - lv))).size for lv in range(9)]
    assert live.tolist() == want


@pytest.mark.parametrize(
    "level, per_cell, leaf, want",
    [
        # every level-2 cell holds 64: level 2 holds a leaf's worth, level 3 holds 8
        (2, 64, 64, 2),
        (2, 64, 8, 3),
        # 80 at level 2 is nearer 64 (in log) than 10 at level 3
        (3, 10, 64, 2),
        # 64 at level 2 or 8 at level 3 for a leaf of 16: 8 is nearer
        (2, 64, 16, 3),
        # a dense grid: the ceiling binds
        (3, 1, 1, AUTO_CELL_MIN_LEVEL_MAX),
        # fewer particles than a leaf: every level holds less, the floor is off
        (0, 10, 64, 0),
    ],
)
def test_the_level_whose_occupied_cells_hold_nearest_a_leaf(level, per_cell, leaf, want):
    pos = jnp.asarray(_grid(level, per_cell))
    got = auto_cell_min_level(pos, leaf_size=leaf, bounds=_infer_bounds(pos))
    assert got == want


def _solver(**tree_kwargs):
    tree = dict(
        mode="static_radix",
        leaf_partition="cells",
        cell_min_level="auto",
        leaf_capacity="auto",
    )
    tree.update(tree_kwargs)
    return FastMultipoleMethod(
        preset="fast",
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(tree=TreeConfig(**tree)),
    )


def test_auto_is_picked_at_the_first_eager_prepare_and_sticks():
    """Until the first prepare the level is the ceiling, marked unpicked; the first
    eager prepare picks it, sizes the capacity at it, and later prepares keep both."""
    leaf, n = 16, 4096
    solver = _solver()
    impl = solver._impl
    assert impl._tree_cell_min_level == AUTO_CELL_MIN_LEVEL_MAX
    assert impl._tree_cell_min_level_auto == AUTO_CELL_MIN_LEVEL_MAX
    first = jnp.asarray(_grid(3, 8))
    bounds = _infer_bounds(first)
    want = auto_cell_min_level(first, leaf_size=leaf, bounds=bounds)
    assert want == 3
    masses = jnp.full((first.shape[0],), 1.0 / n, jnp.float32)
    solver.prepare_state(first, masses, leaf_size=leaf, max_order=2)
    assert impl._tree_cell_min_level == want and impl._tree_cell_min_level_auto is None
    capacity = auto_leaf_capacity(
        live_cell_leaves(first, leaf_size=leaf, cell_min_level=want, bounds=bounds)
    )
    assert impl._tree_leaf_capacity == capacity
    # another draw would pick another level: the first one sticks
    other = jnp.asarray(_grid(1, 64))
    assert auto_cell_min_level(other, leaf_size=leaf, bounds=_infer_bounds(other)) == 2
    solver.prepare_state(other, masses[: other.shape[0]], leaf_size=leaf, max_order=2)
    assert impl._tree_cell_min_level == want


def test_an_explicit_level_and_bad_values():
    assert _solver(cell_min_level=5)._impl._tree_cell_min_level_auto is None
    assert _solver(cell_min_level=5)._impl._tree_cell_min_level == 5
    with pytest.raises((ValueError, jaxtyping.TypeCheckError), match="automatic"):
        _solver(cell_min_level="automatic")
    with pytest.raises(ValueError, match="needs leaf_partition='cells'"):
        FastMultipoleMethod(
            preset="fast",
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", cell_min_level="auto")
            ),
        )


def test_a_traced_first_build_is_refused():
    import jax

    solver = _solver()
    pos = jnp.asarray(_grid(2, 8))

    def build(x):
        solver._impl._resolve_auto_cell_min_level(
            x, leaf_size=16, bounds=_infer_bounds(pos)
        )
        return x

    with pytest.raises(ValueError, match="picked at the first EAGER prepare"):
        jax.jit(build)(pos)


def test_the_galaxy_preset_takes_auto_and_keeps_8_by_default():
    from jaccpot.production import GALAXY_SETTINGS, GalaxySettings
    from jaccpot.solver import _galaxy_advanced as galaxy_advanced

    assert GALAXY_SETTINGS.cell_min_level == 8
    assert FastMultipoleMethod(preset="galaxy")._impl._tree_cell_min_level == 8
    advanced = galaxy_advanced(GalaxySettings(cell_min_level="auto"))
    assert advanced.tree.cell_min_level == "auto"
    solver = FastMultipoleMethod(preset="galaxy", advanced=advanced)
    assert solver._impl._tree_cell_min_level_auto == AUTO_CELL_MIN_LEVEL_MAX

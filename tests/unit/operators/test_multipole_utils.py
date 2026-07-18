"""Unit coverage for the Cartesian symmetric-tensor packing utilities.

Pure index/tensor math -- no FMM tree build or JIT, so these run in
milliseconds while covering the packed-triangular helpers end to end.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.multipole_utils import (
    LOCAL_COMBO_INV_FACTORIAL,
    LOCAL_LEVEL_COMBOS,
    MAX_MULTIPOLE_ORDER,
    level_offset,
    level_size,
    multi_index_factorial,
    multi_index_tuples,
    multi_power,
    pack_tensor,
    total_coefficients,
    triangular_index,
    triangular_indices,
    unpack_tensor,
)


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_multi_index_tuples_enumerate_all_partitions(level):
    combos = multi_index_tuples(level)
    # Count matches the closed-form size and every tuple sums to `level`.
    assert len(combos) == level_size(level)
    assert all(sum(c) == level for c in combos)
    assert len(set(combos)) == len(combos)


def test_multi_index_tuples_rejects_negative():
    with pytest.raises(ValueError, match="level must be >= 0"):
        multi_index_tuples(-1)


def test_multi_index_factorial_matches_product():
    assert multi_index_factorial((0, 0, 0)) == 1
    assert multi_index_factorial((2, 1, 0)) == 2
    assert multi_index_factorial((3, 2, 1)) == 6 * 2 * 1


def test_multi_power_selects_and_multiplies_components():
    vec = jnp.array([2.0, 3.0, 4.0])
    assert float(multi_power(vec, (0, 0, 0))) == pytest.approx(1.0)
    assert float(multi_power(vec, (1, 1, 0))) == pytest.approx(6.0)
    assert float(multi_power(vec, (2, 0, 1))) == pytest.approx(4.0 * 4.0)


@pytest.mark.parametrize(
    "level, size, offset",
    [(0, 1, 0), (1, 3, 1), (2, 6, 4), (3, 10, 10), (4, 15, 20)],
)
def test_level_size_offset_closed_forms(level, size, offset):
    assert level_size(level) == size
    # offset accumulates all lower-order sizes.
    assert level_offset(level) == offset
    assert level_offset(level) == sum(level_size(l) for l in range(level))


def test_total_coefficients_is_cumulative_level_size():
    for max_order in range(MAX_MULTIPOLE_ORDER + 1):
        assert total_coefficients(max_order) == sum(
            level_size(l) for l in range(max_order + 1)
        )


def test_triangular_index_matches_enumeration_order():
    level = 3
    idx = np.asarray(triangular_indices(level))
    assert idx.shape == (level_size(level), 3)
    # triangular_index(i, j) must return the row position of (i, j, k) in the
    # triangular_indices enumeration.
    for pos, (i, j, _k) in enumerate(idx):
        assert triangular_index(level, int(i), int(j)) == pos


def test_triangular_index_rejects_out_of_range():
    with pytest.raises(ValueError, match="Invalid triangular indices"):
        triangular_index(2, 2, 1)  # i + j = 3 > level 2
    with pytest.raises(ValueError, match="Invalid triangular indices"):
        triangular_index(2, -1, 0)


def test_pack_unpack_tensor_round_trips_symmetric_entries():
    level = 3
    rng = np.random.default_rng(0)
    dense = jnp.asarray(rng.standard_normal((level + 1, level + 1, level + 1)))
    packed = pack_tensor(level, dense)
    assert packed.shape == (level_size(level),)
    restored = np.asarray(unpack_tensor(level, packed))
    # Only the i+j+k=level entries are represented; those must round-trip.
    for i, j, k in multi_index_tuples(level):
        assert restored[i, j, k] == pytest.approx(float(dense[i, j, k]))


def test_pack_tensor_rejects_wrong_shape():
    with pytest.raises(ValueError, match="tensor must have shape"):
        pack_tensor(2, jnp.zeros((2, 2, 2)))


def test_unpack_tensor_rejects_wrong_length():
    with pytest.raises(ValueError, match="!= expected"):
        unpack_tensor(2, jnp.zeros((5,)))


def test_cached_combo_tables_are_consistent():
    assert set(LOCAL_LEVEL_COMBOS) == set(range(MAX_MULTIPOLE_ORDER + 1))
    for level, combos in LOCAL_LEVEL_COMBOS.items():
        assert combos == multi_index_tuples(level)
    for combo, inv in LOCAL_COMBO_INV_FACTORIAL.items():
        assert inv == pytest.approx(1.0 / multi_index_factorial(combo))

"""``scatter_to_input_order``: the one supported way to read a distributed result back.

Pure host-side NumPy, so unlike the rest of the distributed suite this runs in CPU CI
on a single device. That matters: the reassembly is where
``docs/distributed_padding_force_defect.md`` went wrong, and a guard that only runs on
a two-GPU box is the kind that was missing in the first place.
"""

import numpy as np
import pytest

from jaccpot.distributed import scatter_to_input_order


def _padded_layout(ndev, cap, counts, rng):
    """A ``(ndev * cap)`` gid array shaped like a real partition's output.

    Real rows carry a global id, padding rows carry ``-1``, and the ids are shuffled
    within each device so the layout is a genuine permutation rather than the
    identity -- which is exactly the case the input ``gid_flat`` gets wrong.
    """
    gid = np.full((ndev, cap), -1, np.int64)
    start = 0
    for d, c in enumerate(counts):
        ids = np.arange(start, start + c)
        gid[d, :c] = rng.permutation(ids)
        start += c
    return gid.reshape(-1)


def test_scatter_inverts_the_permutation():
    """Values land on the particle their gid names, whatever the row order."""
    rng = np.random.default_rng(0)
    ndev, cap, counts = 3, 8, [7, 7, 6]
    n = sum(counts)
    gid = _padded_layout(ndev, cap, counts, rng)
    # Encode each particle's identity in its value, so a misplaced row is visible.
    values = np.where(
        gid[:, None] >= 0, gid[:, None] * np.array([[1.0, 10.0, 100.0]]), 0.0
    )

    out = scatter_to_input_order(values, gid, n)

    expected = np.arange(n)[:, None] * np.array([[1.0, 10.0, 100.0]])
    assert out.shape == (n, 3)
    assert np.array_equal(out, expected)


def test_padding_rows_are_dropped_not_scattered():
    """A ``-1`` row contributes nothing, even when its values are not zero."""
    gid = np.array([2, -1, 0, 1], np.int64)
    values = np.array([[2.0], [999.0], [0.0], [1.0]])
    assert np.array_equal(scatter_to_input_order(values, gid, 3), [[0.0], [1.0], [2.0]])


def test_trailing_shape_is_preserved():
    """Not accelerations-only: any per-row payload keeps its trailing axes."""
    gid = np.array([1, 0, -1], np.int64)
    values = np.zeros((3, 2, 4))
    values[0] = 1.0
    out = scatter_to_input_order(values, gid, 2)
    assert out.shape == (2, 2, 4)
    assert np.array_equal(out[1], np.ones((2, 4)))


def test_gid_column_vector_is_accepted():
    """The evaluator returns ``gid`` as ``[rows, 1]``; take it as it comes."""
    gid = np.array([[1], [0]], np.int64)
    out = scatter_to_input_order(np.array([[5.0], [4.0]]), gid, 2)
    assert np.array_equal(out, [[4.0], [5.0]])


def test_float32_input_is_returned_as_float64():
    """The reassembly must not make the result the noisy side of a comparison."""
    out = scatter_to_input_order(np.ones((1, 3), np.float32), np.array([0]), 1)
    assert out.dtype == np.float64


def test_a_missing_particle_is_an_error_not_a_zero_row():
    """Silence here is what the padding defect was made of; raise instead.

    A particle with no row is a padding or capacity bug. Returning a zero force for
    it would be a physically meaningful, entirely wrong answer that no overflow
    counter reports -- which is the failure mode this whole file exists to close.
    """
    gid = np.array([0, -1, -1], np.int64)
    with pytest.raises(RuntimeError, match="2 particles missing"):
        scatter_to_input_order(np.zeros((3, 3)), gid, 3)


def test_mismatched_row_counts_are_rejected():
    """``values`` and ``gid`` must come from the same evaluator call."""
    with pytest.raises(ValueError, match="rows"):
        scatter_to_input_order(np.zeros((4, 3)), np.array([0, 1, 2]), 3)


def test_out_of_range_gid_is_rejected():
    """An id past ``n`` means the caller paired the result with the wrong ``n``."""
    with pytest.raises(ValueError, match="out of range"):
        scatter_to_input_order(np.zeros((2, 3)), np.array([0, 7]), 2)

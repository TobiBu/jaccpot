"""``_large_n_neighbor_list_matches`` -- audit item **F33**.

Lines 1765-1798 of ``fmm_strict_run.py``, unexercised on CPU and on the A100 run
#193 recorded. A pure predicate over two neighbour lists, so it needs no profile,
no tree and no GPU-shaped engine -- which makes it the cheapest remaining block
in the file and, given what it decides, one worth pinning precisely.

Its docstring states the asymmetry that motivates it: the arrays are
fixed-capacity and padded, so *"comparing them whole would report a difference
whenever the padding differs -- which says nothing about the edges"*. The two
central tests here are that differing padding reports **no** change while a
differing active edge does; get either backwards and the caller either rebuilds
constantly or reuses a stale topology.

Conservatism is the other documented property: any exception is reported as
"changed", because a false negative costs a rebuild while a false positive is a
wrong force. That is tested rather than assumed, since a bare ``except`` is
exactly the kind of thing that quietly stops catching what it was written for.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import NodeNeighborList

from jaccpot.runtime._fmm_impl import FMMEngine


@pytest.fixture(scope="module")
def engine():
    """Any engine: the predicate reads no engine state and no prepared state."""
    return FMMEngine(theta=0.6, working_dtype=jnp.float32)


def _neighbor_list(offsets, neighbors, leaf_indices, counts) -> NodeNeighborList:
    """Build a list with the four fields the predicate reads.

    The other eight are placeholders -- if the predicate ever starts reading one,
    these tests should fail loudly rather than silently compare something else.

    Parameters
    ----------
    offsets, neighbors, leaf_indices, counts : array-like
        The four fields under comparison.

    Returns
    -------
    NodeNeighborList
        A list suitable for the predicate and nothing else.
    """
    unused = jnp.zeros((0,), dtype=jnp.int32)
    return NodeNeighborList(
        offsets=jnp.asarray(offsets, dtype=jnp.int32),
        neighbors=jnp.asarray(neighbors, dtype=jnp.int32),
        leaf_indices=jnp.asarray(leaf_indices, dtype=jnp.int32),
        counts=jnp.asarray(counts, dtype=jnp.int32),
        particle_order_leaf_indices=unused,
        particle_order_to_native_leaf=unused,
        neighbor_leaf_positions=unused,
        target_block_leaf_ids=unused,
        target_block_source_leaf_ids=unused,
        target_block_valid_mask=unused,
        target_block_offsets=unused,
        target_block_size=0,
    )


BASE = dict(
    offsets=[0, 2, 4],
    neighbors=[10, 11, 12, 13, -1, -1, -1],  # 4 active edges, 3 padding slots
    leaf_indices=[0, 1],
    counts=[2, 2],
)


def test_identical_lists_match(engine):
    """The trivial case, and the one every other test is measured against."""
    assert engine._large_n_neighbor_list_matches(
        _neighbor_list(**BASE), _neighbor_list(**BASE)
    )


def test_padding_beyond_the_active_prefix_is_ignored(engine):
    """The reason this predicate exists rather than an array comparison.

    Only ``neighbors[:active_edges]`` is compared, where the active count comes
    from the CSR offsets. Whole-array equality would call these two different and
    force a rebuild on every refresh, for a difference that describes no edge.
    """
    padded_differently = dict(BASE, neighbors=[10, 11, 12, 13, 99, 77, -5])
    assert engine._large_n_neighbor_list_matches(
        _neighbor_list(**BASE), _neighbor_list(**padded_differently)
    )


def test_a_changed_active_edge_is_a_change(engine):
    """The other half: inside the prefix, a difference must be reported.

    The edge changed is the last active one, so this also pins the prefix
    boundary -- an off-by-one that compared ``[:active_edges - 1]`` would pass.
    """
    changed = dict(BASE, neighbors=[10, 11, 12, 99, -1, -1, -1])
    assert not engine._large_n_neighbor_list_matches(
        _neighbor_list(**BASE), _neighbor_list(**changed)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("offsets", [0, 3, 4]),
        ("counts", [3, 1]),
        ("leaf_indices", [0, 2]),
    ],
)
def test_structural_fields_are_compared_exactly(engine, field, value):
    """Offsets, counts and leaf indices are compared whole, not by prefix."""
    changed = dict(BASE)
    changed[field] = value
    assert not engine._large_n_neighbor_list_matches(
        _neighbor_list(**BASE), _neighbor_list(**changed)
    )


@pytest.mark.parametrize(
    "field,value", [("offsets", [0, 2]), ("counts", [2]), ("leaf_indices", [0])]
)
def test_shape_differences_are_a_change(engine, field, value):
    """Checked before the value comparison, which would raise on ragged shapes."""
    changed = dict(BASE)
    changed[field] = value
    assert not engine._large_n_neighbor_list_matches(
        _neighbor_list(**BASE), _neighbor_list(**changed)
    )


@pytest.mark.parametrize("which", ["previous", "current"])
def test_a_neighbor_array_shorter_than_the_active_count_is_a_change(engine, which):
    """Neither side may be too short to hold the edges the offsets claim.

    Both sides are checked separately in the implementation, so both are
    parametrised here; a single case would leave one of the two guards unrun.
    """
    truncated = dict(BASE, neighbors=[10, 11])  # offsets claim 4 active edges
    pair = {
        "previous": (_neighbor_list(**truncated), _neighbor_list(**BASE)),
        "current": (_neighbor_list(**BASE), _neighbor_list(**truncated)),
    }[which]
    assert not engine._large_n_neighbor_list_matches(*pair)


def test_empty_offsets_compare_as_matching(engine):
    """With no offsets the active count is zero, so there is nothing to differ."""
    empty = dict(offsets=[], neighbors=[], leaf_indices=[], counts=[])
    assert engine._large_n_neighbor_list_matches(
        _neighbor_list(**empty), _neighbor_list(**empty)
    )


def test_an_exception_is_reported_as_changed(engine):
    """Conservative by construction -- a rebuild is cheap, a stale reuse is not.

    Passing an object without the fields makes the attribute access raise. The
    documented contract is that this reports "changed" rather than propagating,
    so a malformed list can never be mistaken for an unchanged one.
    """

    # A real ``NodeNeighborList`` carrying a ragged field: the parameter type is
    # satisfied, and the exception comes from inside the comparison, which is
    # where the predicate's ``except Exception`` actually lives. An earlier
    # version passed a bare ``_NotAList`` stand-in, which beartype rejected under
    # ``JACCPOT_RUNTIME_TYPECHECK=1`` -- the test could not reach the branch it
    # was written to cover, because the call never got past the signature.
    unreadable = _neighbor_list(**BASE)._replace(offsets=[[1, 2], [3]])

    assert not engine._large_n_neighbor_list_matches(unreadable, _neighbor_list(**BASE))
    assert not engine._large_n_neighbor_list_matches(_neighbor_list(**BASE), unreadable)

"""Axis contracts for the octree lane's four complex-basis sweep kernels.

`bench/annotation_pilot.py` recorded `runtime/_octree_fmm.py` on 2026-09-04 at **137 silent
acceptances of 208 perturbations, 66%** -- the highest rate in the package. 104 of the 137
are leaves inside NamedTuple containers, which no annotation this toolchain supports can
reach, so the closable surface is 33 and it is entirely in these four functions: P2M, M2M,
M2L and L2L.

The distinction these tests exist to hold is `leaves` against `nodes`. Every recorded call
had them EQUAL -- (7, 7), (9, 9), (5, 5) -- and they are not the same axis:
`_p2m_octree_leaves_complex` takes `total_nodes` from `node_ranges.shape[0]` and then
INDEXES it with `leaf_nodes`, so the leaf list is a list of node ids whose length is the
leaf count. Naming both `nodes` would have been the `farleaves` mistake §4.3 records, where
an equality that held in all 64 captured calls broke 7 tests on the octree backend.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.runtime._octree_fmm import (
    _accumulate_octree_m2l_complex_chunked,
    _aggregate_octree_m2m_complex_by_level,
    _p2m_octree_leaves_complex,
    _propagate_octree_l2l_complex_by_level,
)

NODES, LEAVES, N, SH, PAIRS = 9, 5, 40, 16, 4
ORDER = 3


def _level_args(nodes: int = NODES, sh: int = SH):
    """Build the argument set the M2M and L2L level sweeps share.

    Parameters
    ----------
    nodes : int
        Tree-node count.
    sh : int
        Packed spherical-harmonic coefficient count.

    Returns
    -------
    dict
        Keyword arguments common to both level sweeps.
    """
    return {
        "centers": jnp.zeros((nodes, 3), dtype=jnp.float64),
        "children": jnp.full((nodes, 8), -1, dtype=jnp.int64),
        "nodes_by_level": jnp.zeros((nodes,), dtype=jnp.int64),
        "level_offsets": jnp.asarray([0, 1, nodes], dtype=jnp.int64),
    }


def test_the_level_sweeps_tie_their_node_arrays_together():
    """`centers`, `children`, `nodes_by_level` and the coefficients share `nodes`.

    Recorded at four distinct extents -- 3, 5, 7 and 9 -- across both sweeps.
    """
    args = _level_args()
    packed = jnp.zeros((NODES, SH), dtype=jnp.complex128)

    _aggregate_octree_m2m_complex_by_level(
        packed=packed, order=ORDER, num_levels=2, level_batch_width=NODES, **args
    )

    for bad in ("centers", "children", "nodes_by_level"):
        broken = dict(args)
        broken[bad] = broken[bad][:-1]
        with pytest.raises(TypeCheckError):
            _aggregate_octree_m2m_complex_by_level(
                packed=packed,
                order=ORDER,
                num_levels=2,
                level_batch_width=NODES,
                **broken,
            )

    with pytest.raises(TypeCheckError):
        _aggregate_octree_m2m_complex_by_level(
            packed=packed[:-1],
            order=ORDER,
            num_levels=2,
            level_batch_width=NODES,
            **args,
        )


def test_the_octree_branching_factor_is_a_literal_eight():
    """`children` is `(nodes, 8)`; an octree node has eight of them by definition."""
    args = _level_args()
    args["children"] = jnp.full((NODES, 4), -1, dtype=jnp.int64)
    with pytest.raises(TypeCheckError):
        _propagate_octree_l2l_complex_by_level(
            locals_coeffs=jnp.zeros((NODES, SH), dtype=jnp.complex128),
            order=ORDER,
            num_levels=2,
            level_batch_width=NODES,
            **args,
        )


def test_the_m2l_interaction_arrays_are_one_axis():
    """`source_nodes`, `target_nodes` and `valid_interactions` are parallel lists.

    They are built together from one `interactions` record, so they agree by
    construction. Worth pinning precisely BECAUSE the recording is weak evidence here:
    every recorded call had them empty, `(0,)`, so the call site is the argument and
    this test is what keeps it honest.
    """
    common = {
        "locals_coeffs": jnp.zeros((NODES, SH), dtype=jnp.complex128),
        "multipoles": jnp.zeros((NODES, SH), dtype=jnp.complex128),
        "centers": jnp.zeros((NODES, 3), dtype=jnp.float64),
        "order": ORDER,
        "chunk_size": 2,
    }
    src = jnp.zeros((PAIRS,), dtype=jnp.int64)
    valid = jnp.ones((PAIRS,), dtype=bool)

    _accumulate_octree_m2l_complex_chunked(
        target_nodes=src, source_nodes=src, valid_interactions=valid, **common
    )

    with pytest.raises(TypeCheckError):
        _accumulate_octree_m2l_complex_chunked(
            target_nodes=src, source_nodes=src[:-1], valid_interactions=valid, **common
        )
    with pytest.raises(TypeCheckError):
        _accumulate_octree_m2l_complex_chunked(
            target_nodes=src, source_nodes=src, valid_interactions=valid[:-1], **common
        )


def test_the_multipole_pair_shares_both_axes():
    """`multipoles` and `locals_coeffs` are the same `nodes sh` buffer shape."""
    common = {
        "centers": jnp.zeros((NODES, 3), dtype=jnp.float64),
        "target_nodes": jnp.zeros((PAIRS,), dtype=jnp.int64),
        "source_nodes": jnp.zeros((PAIRS,), dtype=jnp.int64),
        "valid_interactions": jnp.ones((PAIRS,), dtype=bool),
        "order": ORDER,
        "chunk_size": 2,
    }
    locals_coeffs = jnp.zeros((NODES, SH), dtype=jnp.complex128)

    with pytest.raises(TypeCheckError):
        _accumulate_octree_m2l_complex_chunked(
            locals_coeffs=locals_coeffs,
            multipoles=jnp.zeros((NODES, SH - 1), dtype=jnp.complex128),
            **common,
        )


def test_leaves_and_nodes_are_not_the_same_axis():
    """The `farleaves` distinction, in the lane where it was originally found.

    Every recorded call had the leaf count equal to the node count, so this asserts what
    the CODE says rather than what the sample showed: `_p2m_octree_leaves_complex` reads
    `total_nodes` from `node_ranges.shape[0]` and indexes it with `leaf_nodes`, so a leaf
    list shorter than the node arrays is legitimate and must go through.
    """
    kwargs = {
        "leaf_nodes": jnp.zeros((LEAVES,), dtype=jnp.int64),
        "leaf_mask": jnp.ones((LEAVES,), dtype=bool),
        "node_ranges": jnp.zeros((NODES, 2), dtype=jnp.int64),
        "positions_sorted": jnp.zeros((N, 3), dtype=jnp.float64),
        "masses_sorted": jnp.ones((N,), dtype=jnp.float64),
        "centers": jnp.zeros((NODES, 3), dtype=jnp.float64),
        "order": ORDER,
        "max_leaf_size": 8,
    }
    out = _p2m_octree_leaves_complex(**kwargs)
    # The output is per NODE, not per leaf -- which is the point: a 5-leaf list against
    # 9 nodes goes through, so the two axes are genuinely independent.
    assert out.shape[0] == NODES

    # The leaf pair still has to agree with itself.
    with pytest.raises(TypeCheckError):
        _p2m_octree_leaves_complex(**dict(kwargs, leaf_mask=kwargs["leaf_mask"][:-1]))
    # And the particle pair with itself.
    with pytest.raises(TypeCheckError):
        _p2m_octree_leaves_complex(
            **dict(kwargs, masses_sorted=kwargs["masses_sorted"][:-1])
        )

"""Axis contracts for the near-field prepared-leaf kernel.

`bench/annotation_pilot.py` recorded `nearfield/near_field.py` on 2026-09-04 at 56 silent
acceptances of 93 perturbations. Only 17 of those are reachable: the two public entry points
between them hold 39 leaves inside NamedTuple containers, which no annotation this toolchain
supports can constrain. So this pins the private
`_compute_leaf_p2p_from_prepared_leaf_data_impl`, which carries 15 of the 17, plus the
particle pair on the public large-N entry.

`offsets` is deliberately rank-only, and twice over. It is `leaves+1` in all eight recorded
calls, but it is the FIRST parameter in the signature -- nothing has bound `leaves` when
jaxtyping evaluates it -- and separately, an offsets length has proved not to be a stable
relation across lanes: `_large_n_blocks`' `block_offsets`, `_adaptive_policy`'s
`neighbor_offsets`, and `local_expansions`' `offsets`, the last of which shipped as
`nodes+1` and was caught by the distributed tier in #324.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.nearfield.near_field import (
    _compute_leaf_p2p_from_prepared_leaf_data_impl as _impl,
)

LEAVES, W, EDGES = 4, 2, 6
N = LEAVES * W


def _args():
    """Build one valid call, shaped after the recorded small cases.

    Returns
    -------
    dict
        Keyword arguments for the prepared-leaf kernel.
    """
    return {
        "offsets": jnp.zeros((LEAVES + 1,), dtype=jnp.int64),
        "neighbors": jnp.zeros((EDGES,), dtype=jnp.int64),
        "positions": jnp.zeros((N, 3), dtype=jnp.float64),
        "target_leaf_ids": jnp.zeros((EDGES,), dtype=jnp.int64),
        "source_leaf_ids": jnp.zeros((EDGES,), dtype=jnp.int64),
        "valid_pairs": jnp.zeros((EDGES,), dtype=bool),
        "precomputed_chunk_sort_indices": jnp.zeros((0, 0), dtype=jnp.int64),
        "precomputed_chunk_group_ids": jnp.zeros((0, 0), dtype=jnp.int64),
        "precomputed_chunk_unique_indices": jnp.zeros((0, 0), dtype=jnp.int64),
        "leaf_positions": jnp.zeros((LEAVES, W, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((LEAVES, W), dtype=jnp.float64),
        "leaf_mask": jnp.ones((LEAVES, W), dtype=bool),
        "leaf_particle_idx": jnp.zeros((LEAVES, W), dtype=jnp.int64),
        "G": jnp.asarray(1.0),
        "softening_sq": jnp.asarray(1e-4),
        "return_potential": False,
        "collect_neighbor_pairs": False,
        "nearfield_mode": "baseline",
        "edge_chunk_size": 4,
        "use_precomputed_scatter": False,
    }


def test_a_matched_call_still_goes_through():
    """The control. Every rejection below is worthless without it."""
    out = _impl(**_args())
    # With `return_potential=False` the kernel returns the acceleration array itself,
    # not a tuple -- asserting its shape keeps this a control rather than a claim about
    # the return structure.
    assert out.shape == (N, 3)


def test_the_four_leaf_tables_are_one_block():
    """`leaf_positions`, `leaf_masses`, `leaf_mask` and `leaf_particle_idx` share `leaves w`.

    Observed together at (5, 1), (4, 1), (1, 4), (16, 32), (32, 16), (64, 8) and (128, 32) --
    seven distinct shapes, both axes agreeing in every one.
    """
    args = _args()
    for name in ("leaf_masses", "leaf_mask", "leaf_particle_idx"):
        with pytest.raises(TypeCheckError):
            _impl(**dict(args, **{name: args[name][:, :-1]}))
        with pytest.raises(TypeCheckError):
            _impl(**dict(args, **{name: args[name][:-1]}))


def test_the_edge_list_arrays_are_one_axis():
    """`neighbors`, both leaf-id lists and `valid_pairs` are parallel per-edge arrays.

    A disagreement here pairs an edge with another edge's validity flag, which scatters a
    real contribution into the wrong target and cannot be seen downstream.
    """
    args = _args()
    for name in ("neighbors", "target_leaf_ids", "source_leaf_ids", "valid_pairs"):
        with pytest.raises(TypeCheckError):
            _impl(**dict(args, **{name: args[name][:-1]}))


def test_the_precomputed_chunk_tables_agree():
    """The three scatter-schedule tables are one `chunks chunkflat` grid."""
    args = _args()
    grid = jnp.zeros((2, 8), dtype=jnp.int64)
    args = dict(
        args,
        precomputed_chunk_sort_indices=grid,
        precomputed_chunk_group_ids=grid,
        precomputed_chunk_unique_indices=grid,
    )
    with pytest.raises(TypeCheckError):
        _impl(**dict(args, precomputed_chunk_group_ids=grid[:, :-1]))


def test_the_offsets_length_is_deliberately_free():
    """Rank is constrained; length is not, and that is the point.

    `leaves+1` holds in all eight recordings but `offsets` precedes everything that binds
    `leaves`, so the symbolic form cannot be evaluated -- and #324 showed an offsets
    relation is not stable across lanes anyway. Both lengths are asserted so nobody
    "restores" the constraint.
    """
    args = _args()
    for length in (LEAVES, LEAVES + 1):
        _impl(**dict(args, offsets=jnp.zeros((length,), dtype=jnp.int64)))

    with pytest.raises(TypeCheckError):
        _impl(**dict(args, offsets=jnp.zeros((LEAVES + 1, 1), dtype=jnp.int64)))

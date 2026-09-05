"""Axis contracts for the fused near-field leaf kernels' parallel arrays.

The `*_positions` parameters here are deliberately NOT annotated: the bodies already check
their rank and trailing 3 and say so by name -- `target_positions must have shape
(num_leaves, W_t, 3)` -- and a decorator would run first and replace that with a generic
`TypeCheckError`. `test_the_position_checks_still_fire` is the guard on that decision.

What the bodies do not check is the masks, masses, ids and validity arrays that have to
agree with those positions slot for slot, and that is exactly where this module's real
defect lived: #297, where the decoupled lane's source pool had a different width from the
target block and the kernel read the surplus out of bounds, silently.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
    nearfield_leafpair_pallas_decoupled,
)

LEAVES, W, SRCSLOTS, SRCLEAVES = 4, 8, 3, 6


def _leafpair():
    """Build one valid `nearfield_leafpair_jax` call.

    Returns
    -------
    dict
        Keyword arguments with `leaves`, `w` and `srcslots` all distinct.
    """
    return {
        "leaf_positions": jnp.zeros((LEAVES, W, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((LEAVES, W), dtype=jnp.float64),
        "leaf_mask": jnp.ones((LEAVES, W), dtype=bool),
        "source_leaf_ids": jnp.zeros((LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": jnp.asarray(1e-4),
        "G": jnp.asarray(1.0),
    }


def test_a_matched_leafpair_call_still_goes_through():
    """The control, with three genuinely different extents."""
    out = nearfield_leafpair_jax(**_leafpair())
    # The trailing axis is the kernel's packed output, not the spatial 3 -- asserting
    # only the block shape keeps this a control rather than a claim about the payload.
    assert out.shape[:2] == (LEAVES, W)


def test_the_position_checks_still_fire():
    """The bodies' own messages must survive -- that is why positions stay bare.

    Annotating `leaf_positions` would make this `ValueError` unreachable and replace a
    message naming the parameter with a generic type error. STYLE_GUIDE checklist item 13.
    """
    args = _leafpair()
    args["leaf_positions"] = jnp.zeros((LEAVES, W), dtype=jnp.float64)
    with pytest.raises(ValueError, match="leaf_positions must have shape"):
        nearfield_leafpair_pallas(**args)


def test_the_leaf_block_arrays_must_agree():
    """`leaf_masses` and `leaf_mask` are the payload and the validity of one block."""
    args = _leafpair()
    with pytest.raises(TypeCheckError):
        nearfield_leafpair_jax(**dict(args, leaf_mask=args["leaf_mask"][:, :-1]))
    with pytest.raises(TypeCheckError):
        nearfield_leafpair_jax(**dict(args, leaf_masses=args["leaf_masses"][:-1]))


def test_the_source_slot_arrays_must_agree():
    """`source_leaf_ids` and `source_valid` index the same padded neighbour list."""
    args = _leafpair()
    with pytest.raises(TypeCheckError):
        nearfield_leafpair_jax(**dict(args, source_valid=args["source_valid"][:, :-1]))


def test_the_source_slot_width_is_free_of_the_block_width():
    """`srcslots` is not `w`: a leaf's neighbour count has nothing to do with its width.

    Recorded at (4, 8) beside (4, 3), (6, 8) beside (6, 5) and (6, 8) beside (6, 6), so
    tying them would assert something false.
    """
    args = _leafpair()
    args["source_leaf_ids"] = jnp.zeros((LEAVES, SRCSLOTS + 2), dtype=jnp.int32)
    args["source_valid"] = jnp.ones((LEAVES, SRCSLOTS + 2), dtype=bool)
    assert nearfield_leafpair_jax(**args).shape[:2] == (LEAVES, W)


def test_the_decoupled_source_pool_is_its_own_leading_axis():
    """`srcleaves` is not `leaves` -- that separation is the decoupled lane's whole point.

    Recorded with `source_mask` at (4, 8) beside `target_mask` at (3, 8), so a source pool
    with a different leaf count must go through. Only the WIDTH is shared, which is #297's
    invariant.
    """
    common = {
        "target_positions": jnp.zeros((LEAVES, W, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((LEAVES, W), dtype=bool),
        "source_positions": jnp.zeros((SRCLEAVES, W, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((SRCLEAVES, W), dtype=jnp.float64),
        "source_mask": jnp.ones((SRCLEAVES, W), dtype=bool),
        "source_leaf_ids": jnp.zeros((LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": jnp.asarray(1e-4),
        "G": jnp.asarray(1.0),
    }
    # A mismatched source POOL is fine; a mismatched source mask is not.
    with pytest.raises(TypeCheckError):
        nearfield_leafpair_pallas_decoupled(
            **dict(common, source_mask=common["source_mask"][:-1])
        )

    # And #297's own guard still fires first, because `source_positions` stays bare.
    with pytest.raises(ValueError, match="same leaf width"):
        nearfield_leafpair_pallas_decoupled(
            **dict(
                common,
                source_positions=jnp.zeros((SRCLEAVES, W - 1, 3), dtype=jnp.float64),
                source_masses=jnp.ones((SRCLEAVES, W - 1), dtype=jnp.float64),
                source_mask=jnp.ones((SRCLEAVES, W - 1), dtype=bool),
            )
        )

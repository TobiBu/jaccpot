"""Mutual shape consistency in the fused near-field leaf entry points.

These are the 9 silent acceptances the 2026-09-07 pilot re-recording left in this file,
closed in the BODY rather than with annotations. The reasoning is written out in
`docs/annotation_pilot_phase2_2026-08-30.md`; the short form is that each of these
positions parameters already raises a documented `ValueError` over its shape, so a
decorator would run first and replace it -- and `DELIBERATELY_BARE`'s first entry makes
that a behaviour change rather than a docs change. Strengthening the guard delivers the
same check under the exception the docstring already promises.

WHY THE OLD CHECKS WERE NOT ENOUGH. They tested `ndim != 3 or shape[-1] != 3` and never
the LEADING extent, while the docstrings promised `ValueError` "if the input shapes are
mutually inconsistent" -- a consistency the bodies did not verify. So the docstring
over-promised, and the gap it left is the dangerous kind:

    perturbing POSITIONS shrinks the Pallas grid, so the output shape shrinks too and a
    caller who checks can see it;

    perturbing a MASK leaves the grid alone, so the BlockSpec indexes the mask out of
    bounds, JAX CLAMPS, and leaf 3 silently reuses leaf 2's mask -- real particles
    masked out or phantom ones included, at the right output shape.

Measured on `main` before this change: `target_mask` one row short returned a full
(3, 2, 4); so did `target_mask` one column short and `source_positions` one leaf short.

The reference lane needed a check too, for a different reason: `nearfield_leafpair_jax`
had none at all, and a `leaf_positions` of trailing width 2 produced a 2-wide
acceleration which the final `concatenate` turned into a 3-wide result where 4 is the
contract -- (3, 2, 2, 3) in, (3, 2, 3) out, silently.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_fused_leaf_pallas,
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
    nearfield_leafpair_pallas_decoupled,
)

LEAVES, WT, SRCSLOTS, SRCLEAVES = 3, 2, 4, 2
SOFT = jnp.asarray(1e-4)
GRAV = jnp.asarray(1.0)


def _fused(**over):
    """Build one valid `nearfield_fused_leaf_pallas` call.

    Parameters
    ----------
    **over : Any
        Arguments to replace, one per perturbation.

    Returns
    -------
    dict
        Keyword arguments with `leaves`, `w` and the source slot count all distinct.
    """
    args = {
        "target_positions": jnp.zeros((LEAVES, WT, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((LEAVES, WT), dtype=bool),
        "source_positions": jnp.zeros((LEAVES, SRCSLOTS, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((LEAVES, SRCSLOTS), dtype=jnp.float64),
        "source_mask": jnp.ones((LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
        "interpret": True,
    }
    args.update(over)
    return args


def _pair(**over):
    """Build one valid `nearfield_leafpair_pallas` call.

    Parameters
    ----------
    **over : Any
        Arguments to replace.

    Returns
    -------
    dict
        Keyword arguments for the leaf-pair entry point.
    """
    args = {
        "leaf_positions": jnp.zeros((LEAVES, WT, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((LEAVES, WT), dtype=jnp.float64),
        "leaf_mask": jnp.ones((LEAVES, WT), dtype=bool),
        "source_leaf_ids": jnp.zeros((LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
        "interpret": True,
    }
    args.update(over)
    return args


def _decoupled(**over):
    """Build one valid `nearfield_leafpair_pallas_decoupled` call.

    Parameters
    ----------
    **over : Any
        Arguments to replace.

    Returns
    -------
    dict
        Keyword arguments with the source pool a DIFFERENT leaf count from the targets,
        which is the variant's purpose and must keep working.
    """
    args = {
        "target_positions": jnp.zeros((LEAVES, WT, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((LEAVES, WT), dtype=bool),
        "source_positions": jnp.zeros((SRCLEAVES, WT, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((SRCLEAVES, WT), dtype=jnp.float64),
        "source_mask": jnp.ones((SRCLEAVES, WT), dtype=bool),
        "source_leaf_ids": jnp.zeros((LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
        "interpret": True,
    }
    args.update(over)
    return args


def test_the_valid_calls_all_still_go_through():
    """The control, on all four entry points, including the decoupled independence.

    The decoupled call has 2 source leaves against 3 target leaves on purpose: that
    separation is what the variant exists for, and a guard that "fixed" it by tying the
    two together would break the lane while looking like an improvement.
    """
    assert nearfield_fused_leaf_pallas(**_fused()).shape == (LEAVES, WT, 4)
    assert nearfield_leafpair_pallas(**_pair()).shape == (LEAVES, WT, 4)
    assert nearfield_leafpair_pallas_decoupled(**_decoupled()).shape == (LEAVES, WT, 4)
    plain = {k: v for k, v in _pair().items() if k != "interpret"}
    assert nearfield_leafpair_jax(**plain).shape == (LEAVES, WT, 4)


@pytest.mark.parametrize(
    "over, culprit",
    [
        (
            {"target_positions": jnp.zeros((LEAVES - 1, WT, 3), dtype=jnp.float64)},
            "target_mask",
        ),
        ({"target_mask": jnp.ones((LEAVES - 1, WT), dtype=bool)}, "target_mask"),
        ({"target_mask": jnp.ones((LEAVES, WT - 1), dtype=bool)}, "target_mask"),
        (
            {
                "source_positions": jnp.zeros(
                    (LEAVES - 1, SRCSLOTS, 3), dtype=jnp.float64
                )
            },
            "source_positions",
        ),
    ],
)
def test_the_fused_entry_refuses_an_operand_the_grid_would_clamp(over, culprit):
    """Each of these returned a full-size, plausible result on `main`.

    `source_mask` and `source_masses` are deliberately absent: they carry
    `srcleaves srcslots` and so already cross-check each other, which the pilot
    confirmed by REJECTING both -- they were never among the 9. Adding them here would
    claim credit for a check that predates this PR.

    Parameters
    ----------
    over : dict
        The single argument to perturb.
    culprit : str
        Substring the message must name, so the error points at the argument the caller
        actually got wrong rather than at whichever one the kernel noticed.
    """
    with pytest.raises(ValueError, match=culprit):
        nearfield_fused_leaf_pallas(**_fused(**over))


def test_the_leafpair_entry_refuses_a_short_leaf_table():
    """`leaf_positions` is both the grid source AND the gather target here.

    One leaf short, `main` returned (2, 2, 4) -- the last leaf's contributions dropped
    entirely, and every array that still had 3 rows read into the wrong block.
    """
    with pytest.raises(ValueError):
        nearfield_leafpair_pallas(
            **_pair(leaf_positions=jnp.zeros((LEAVES - 1, WT, 3)))
        )


@pytest.mark.parametrize(
    "over",
    [
        {"target_positions": jnp.zeros((LEAVES - 1, WT, 3), dtype=jnp.float64)},
        {"target_mask": jnp.ones((LEAVES, WT - 1), dtype=bool)},
        {"source_positions": jnp.zeros((SRCLEAVES - 1, WT, 3), dtype=jnp.float64)},
    ],
)
def test_the_decoupled_entry_checks_each_side_against_its_own_count(over):
    """Targets against `num_targets`, sources against `num_sources`, not one leaf count.

    Parameters
    ----------
    over : dict
        The single argument to perturb.
    """
    with pytest.raises(ValueError):
        nearfield_leafpair_pallas_decoupled(**_decoupled(**over))


def test_the_reference_lane_refuses_a_two_component_position():
    """`nearfield_leafpair_jax` had no shape check at all, and lost a component quietly.

    Trailing width 2 gives a 2-wide acceleration, and `concatenate` with the potential
    returns a 3-wide result -- the RANK is right, the contract is 4, and nothing raised.
    """
    plain = {k: v for k, v in _pair().items() if k != "interpret"}
    plain["leaf_positions"] = jnp.zeros((LEAVES, WT, 2), dtype=jnp.float64)
    with pytest.raises(ValueError, match="leaf_positions"):
        nearfield_leafpair_jax(**plain)


def test_the_new_guard_does_not_outrank_the_source_width_message():
    """#297's specific message must keep priority over the generic one.

    The source-width check explains a mechanism the generic message cannot -- a narrower
    pool reads out of bounds, a wider one silently drops real particles -- so the source
    table checks are placed AFTER it. This caught a real ordering mistake in this PR:
    with them placed before, `test_decoupled_source_pool_is_its_own_leading_axis` and
    four parametrisations of `test_decoupled_rejects_a_source_pool_of_a_different_width`
    all went red.
    """
    narrow = _decoupled(
        source_positions=jnp.zeros((SRCLEAVES, WT - 1, 3), dtype=jnp.float64),
        source_masses=jnp.ones((SRCLEAVES, WT - 1), dtype=jnp.float64),
        source_mask=jnp.ones((SRCLEAVES, WT - 1), dtype=bool),
    )
    with pytest.raises(ValueError, match="same leaf width"):
        nearfield_leafpair_pallas_decoupled(**narrow)

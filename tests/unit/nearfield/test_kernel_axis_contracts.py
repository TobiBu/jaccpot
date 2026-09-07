"""Axis contracts for the near-field pair and self kernels.

`bench/annotation_pilot.py` recorded `nearfield/_kernels.py` on 2026-09-04 at 23 silent
acceptances of 64 perturbations, and every one of them is closed by the annotations these
tests pin -- the module now measures **0**.

The judgement worth pinning is that **the target and source sides are independent axes**.
Every recorded call had them equal -- (32,) against (32,), (256, 32) against (256, 32) --
which is what a leaf-pair schedule with one padded width looks like. The body says
otherwise: it maps over targets and sums over sources with `axis=0`, so a target block of
one width against a source pool of another is a legitimate call. Tying them would repeat
the mistake #324 had to undo in `downward/local_expansions.py`, where the distributed lane
turned an apparent single node axis into two.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.nearfield._kernels import (
    _pair_contributions,
    _pair_contributions_batched,
    _self_contributions,
)

W, SW, PAIRS, LEAVES = 8, 5, 4, 3
SOFT, G = 1e-4, 1.0


def _flat(w=W, sw=SW):
    """Build one valid `_pair_contributions` call.

    Parameters
    ----------
    w : int
        Target block width.
    sw : int
        Source pool width, deliberately different from `w`.

    Returns
    -------
    dict
        Keyword arguments for `_pair_contributions`.
    """
    return {
        "target_positions": jnp.zeros((w, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((w,), dtype=bool),
        "source_positions": jnp.ones((sw, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((sw,), dtype=jnp.float64),
        "source_mask": jnp.ones((sw,), dtype=bool),
        "softening_sq": SOFT,
        "G": jnp.asarray(G),
        "compute_potential": False,
    }


def test_the_target_and_source_widths_are_independent():
    """A target block of 8 against a source pool of 5 is a legitimate call.

    The body maps over targets and sums over sources, so nothing requires the two to
    match -- and every recorded call had them equal, which is exactly the shape of
    evidence that misled the first version of `_accumulate_level` in #324.
    """
    accels, _ = _pair_contributions(**_flat())
    assert accels.shape == (W, 3)


def test_each_side_must_agree_with_itself():
    """`mask`, `masses` and `positions` describe the same slots on their own side."""
    args = _flat()
    with pytest.raises(TypeCheckError):
        _pair_contributions(**dict(args, target_mask=args["target_mask"][:-1]))
    with pytest.raises(TypeCheckError):
        _pair_contributions(**dict(args, source_masses=args["source_masses"][:-1]))
    with pytest.raises(TypeCheckError):
        _pair_contributions(**dict(args, source_mask=args["source_mask"][:-1]))


def test_a_two_component_position_is_rejected():
    """The spatial literal the whole pair kernel is built on."""
    args = _flat()
    with pytest.raises(TypeCheckError):
        _pair_contributions(
            **dict(args, source_positions=args["source_positions"][:, :-1])
        )


def test_the_batched_kernel_shares_only_the_pair_index():
    """Batched, the two sides share `pairs` and nothing else.

    Each entry is one (target block, source block) pair, so the leading axis is genuinely
    common -- observed equal in all seven recorded calls, at 2, 4, 6, 256, 1024 and 2048 --
    while the widths stay free of each other.
    """
    args = {
        "target_positions": jnp.zeros((PAIRS, W, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((PAIRS, W), dtype=bool),
        "source_positions": jnp.ones((PAIRS, SW, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((PAIRS, SW), dtype=jnp.float64),
        "source_mask": jnp.ones((PAIRS, SW), dtype=bool),
        "softening_sq": SOFT,
        "G": jnp.asarray(G),
        "compute_potential": False,
    }
    accels, _ = _pair_contributions_batched(**args)
    assert accels.shape == (PAIRS, W, 3)

    with pytest.raises(TypeCheckError):
        _pair_contributions_batched(**dict(args, source_mask=args["source_mask"][:-1]))


def test_the_self_block_is_one_leaf_table():
    """`_self_contributions` has a single block: leaves x w, shared by all three arrays."""
    args = {
        "leaf_positions": jnp.zeros((LEAVES, W, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((LEAVES, W), dtype=jnp.float64),
        "mask": jnp.ones((LEAVES, W), dtype=bool),
        "softening_sq": SOFT,
        "G": jnp.asarray(G),
        "compute_potential": False,
    }
    accels, _ = _self_contributions(**args)
    assert accels.shape == (LEAVES, W, 3)

    with pytest.raises(TypeCheckError):
        _self_contributions(**dict(args, mask=args["mask"][:, :-1]))

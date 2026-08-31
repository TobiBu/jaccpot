"""What the large-N block kernels now reject, and what they must keep taking.

`bench/annotation_pilot` measured this family accepting 58 of 212 shape
perturbations in silence. These tests pin the closure rather than the annotation
text: each feeds a malformed argument and asserts a complaint, each control feeds
the real thing and asserts it still works.

Three of the assertions are about restraint rather than protection, and they are
the ones worth reading:

* **`block_offsets` and `block_target_leaf_ids` stay rank-only on purpose.** A
  short `block_offsets` IS a silent bug -- the body reads `block_offsets[leaf + 1]`
  and JAX clamps an out-of-range index -- but the axis is the FAR-field leaf view
  (`runtime/kernels/_evaluate.py` measures radix 3 leaves against octree 5), which
  is not this signature's `leaves`. Tying them would be the `farleaves` mistake.
  `test_short_block_offsets_is_still_accepted_and_that_is_deliberate` pins that the
  gap is known, so nobody closes it by conflating the two views.
* **The prepacked rectangle's leading axis is `farleaves`, not `leaves`.** What is
  asserted is that the ids and their mask agree with *each other*, not that either
  matches the leaf table.
* **An all-zero leaf table must still pass.** The kernels early-out on it, and the
  sentinel reasoning in section 4.3 turns on this.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.nearfield._large_n_blocks import (
    _compute_leaf_p2p_prepared_large_n_pairs_only_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl,
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
)

N, LEAVES, W, PAIRS, BLOCKS, BLOCKSIZE = 8, 4, 2, 6, 5, 1


def _leaf_table(leaves: int = LEAVES, w: int = W, n: int = N):
    """Build the padded leaf table every kernel in the family takes.

    Parameters
    ----------
    leaves : int
        Number of padded leaves.
    w : int
        Leaf width.
    n : int
        Particle count.

    Returns
    -------
    dict
        `positions` plus the four per-leaf arrays.
    """
    return {
        "positions": jnp.zeros((n, 3)),
        "leaf_positions": jnp.zeros((leaves, w, 3)),
        "leaf_masses": jnp.ones((leaves, w)),
        "leaf_mask": jnp.ones((leaves, w), dtype=bool),
        "leaf_particle_idx": jnp.zeros((leaves, w), dtype=jnp.int32),
    }


def _call_self_only(**kw):
    """Call the self-leaf kernel.

    Parameters
    ----------
    **kw
        The leaf table.

    Returns
    -------
    Array
        Per-particle accelerations.
    """
    return _compute_leaf_p2p_prepared_large_n_self_only_impl(
        **kw, G=1.0, softening_sq=jnp.asarray(1e-4)
    )


#: The edge-list kernels take eight required keyword-only knobs. They are chunking
#: and scatter-strategy choices, not shapes, so they are fixed once here rather than
#: varied: `edge_chunk_size` must divide the pair list or an internal reshape fails
#: before any annotation is reached, which would make every test in the parametrised
#: set INCONCLUSIVE rather than passing or failing on the contract.
_EDGE_STATICS = {
    "G": 1.0,
    "softening_sq": jnp.asarray(1e-4),
    "edge_chunk_size": PAIRS,
    "chunks_per_superchunk": 1,
    "sorted_scatter_hint": False,
    "grouped_sorted_scatter": False,
    "superchunk_target_reduce": False,
    "disable_chunk_cond": True,
}


def _pair_list():
    """Build the edge-list triple.

    Returns
    -------
    dict
        `target_leaf_ids`, `source_leaf_ids` and `valid_pairs`.
    """
    return {
        "target_leaf_ids": jnp.zeros((PAIRS,), dtype=jnp.int32),
        "source_leaf_ids": jnp.zeros((PAIRS,), dtype=jnp.int32),
        "valid_pairs": jnp.ones((PAIRS,), dtype=bool),
    }


def _call_pairs_only(**kw):
    """Call the edge-list kernel.

    Parameters
    ----------
    **kw
        The leaf table, optionally with the pair triple already replaced.

    Returns
    -------
    Array
        Per-particle accelerations.
    """
    arguments = {**_pair_list(), **kw}
    return _compute_leaf_p2p_prepared_large_n_pairs_only_impl(
        **arguments, **_EDGE_STATICS
    )


def _call_target_blocks(**kw):
    """Call the run-length target-block kernel.

    Parameters
    ----------
    **kw
        The leaf table, optionally with a block array already replaced.

    Returns
    -------
    Array
        Per-particle accelerations.
    """
    arguments = {
        "block_offsets": jnp.zeros((LEAVES + 1,), dtype=jnp.int32),
        "block_target_leaf_ids": jnp.zeros((BLOCKS,), dtype=jnp.int32),
        "block_source_leaf_ids": jnp.zeros((BLOCKS, BLOCKSIZE), dtype=jnp.int32),
        "block_valid_mask": jnp.ones((BLOCKS, BLOCKSIZE), dtype=bool),
        **kw,
    }
    return _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(
        **arguments,
        G=1.0,
        softening_sq=jnp.asarray(1e-4),
        target_leaf_batch_size=2,
        target_block_tile_size=1,
        target_block_tile_scan_unroll=1,
        target_block_batch_scan_unroll=1,
    )


def _call_prepacked(**kw):
    """Call the prepacked-rectangle kernel.

    Parameters
    ----------
    **kw
        The leaf table, optionally with a padded array already replaced.

    Returns
    -------
    Array
        Per-particle accelerations.
    """
    arguments = {
        "block_source_leaf_ids_padded": jnp.zeros(
            (LEAVES, BLOCKS, BLOCKSIZE), dtype=jnp.int32
        ),
        "block_valid_mask_padded": jnp.ones((LEAVES, BLOCKS, BLOCKSIZE), dtype=bool),
        **kw,
    }
    return _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl(
        **arguments,
        G=1.0,
        softening_sq=jnp.asarray(1e-4),
        target_leaf_batch_size=2,
        target_block_tile_size=1,
        target_block_tile_scan_unroll=1,
        target_block_batch_scan_unroll=1,
    )


KERNELS = [
    ("self_only", _call_self_only),
    ("pairs_only", _call_pairs_only),
    ("target_blocks", _call_target_blocks),
    ("prepacked", _call_prepacked),
]
CALLS = [fn for _, fn in KERNELS]
IDS = [name for name, _ in KERNELS]


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_the_valid_leaf_table_still_goes_through(call):
    """The control. Every rejection below means nothing without it."""
    call(**_leaf_table())


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_two_component_positions_are_rejected(call):
    """`positions` is `n 3`; the pilot saw the trailing axis corrupted and accepted."""
    table = _leaf_table()
    table["positions"] = jnp.zeros((N, 2))
    with pytest.raises(TypeCheckError):
        call(**table)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_a_leaf_table_that_disagrees_with_itself_is_rejected(call):
    """`leaf_masses` one leaf short of `leaf_positions` -- accepted before.

    This is the assertion the annotation actually buys across the whole family:
    the four per-leaf arrays have to describe the same table.
    """
    table = _leaf_table()
    table["leaf_masses"] = jnp.ones((LEAVES - 1, W))
    with pytest.raises(TypeCheckError):
        call(**table)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_a_narrower_particle_index_block_is_rejected(call):
    """`leaf_particle_idx` is what fixes `W`, so a narrower one is a real mismatch."""
    table = _leaf_table()
    table["leaf_particle_idx"] = jnp.zeros((LEAVES, W - 1), dtype=jnp.int32)
    with pytest.raises(TypeCheckError):
        call(**table)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_an_extra_leading_axis_on_the_leaf_table_is_rejected(call):
    """The perturbation accepted in five of the eight measured functions."""
    table = _leaf_table()
    table["leaf_masses"] = table["leaf_masses"][None]
    with pytest.raises(TypeCheckError):
        call(**table)


@pytest.mark.parametrize("call", CALLS, ids=IDS)
def test_an_all_zero_leaf_table_is_accepted(call):
    """The kernels early-out on it, and the sentinel argument turns on this.

    `leaves` binding to 0 is fine because the four leaf arrays only have to agree
    with each other. If this ever raised, the annotations would have re-created the
    87-test failure STYLE_GUIDE section 4.3 records for `_evaluate.py`.
    """
    call(**_leaf_table(leaves=0, w=0))


def test_a_short_pair_list_is_rejected():
    """`valid_pairs` shorter than the two id arrays: the `pairs` axis is shared."""
    with pytest.raises(TypeCheckError):
        _call_pairs_only(
            **_leaf_table(), valid_pairs=jnp.ones((PAIRS - 1,), dtype=bool)
        )


def test_block_ids_and_their_mask_must_agree():
    """`blocks blocksize` binds freely but ties the two arrays together."""
    with pytest.raises(TypeCheckError):
        _call_target_blocks(
            **_leaf_table(),
            block_valid_mask=jnp.ones((BLOCKS - 1, BLOCKSIZE), dtype=bool),
        )


def test_the_prepacked_rectangle_must_agree_with_its_own_mask():
    """`farleaves blocks blocksize` on both, so all three axes are tied -- to each other."""
    with pytest.raises(TypeCheckError):
        _call_prepacked(
            **_leaf_table(),
            block_valid_mask_padded=jnp.ones(
                (LEAVES, BLOCKS, BLOCKSIZE + 1), dtype=bool
            ),
        )


def test_the_prepacked_rectangle_is_not_tied_to_the_leaf_table():
    """`farleaves` is deliberately NOT `leaves`, and this pins that it stays free.

    The leading axis is the far-field leaf view. `_evaluate.py` measures the two
    views differing -- radix 3 leaves against octree 5 -- so an annotation that made
    them equal would break the octree backend, which no test here can reach. If a
    later change "tidies" `farleaves` into `leaves`, this test is what fails.
    """
    try:
        _call_prepacked(
            **_leaf_table(),
            block_source_leaf_ids_padded=jnp.zeros(
                (LEAVES + 2, BLOCKS, BLOCKSIZE), dtype=jnp.int32
            ),
            block_valid_mask_padded=jnp.ones(
                (LEAVES + 2, BLOCKS, BLOCKSIZE), dtype=bool
            ),
        )
    except TypeCheckError as error:  # pragma: no cover - the regression this guards
        raise AssertionError(
            "`farleaves` has been tied to `leaves`; the octree backend has "
            f"5 far-field leaves against 3 near-field ones. {error}"
        ) from error
    except Exception:
        # The BODY may still reject a rectangle wider than its leaf table -- it
        # indexes one block run per leaf. That is the kernel's own business and not
        # what this test is about; only a TypeCheckError would mean the annotation
        # had made the two views equal.
        pass


def test_short_block_offsets_is_still_accepted_and_that_is_deliberate():
    """A KNOWN, UNCLOSED gap, pinned so it is not closed the wrong way.

    `block_offsets` is read at `[leaf_id + 1]` and JAX clamps, so a `leaves`-length
    array silently gives the last target leaf the wrong block run. It stays
    rank-only anyway: the axis is `farleaves+1` upstream, not this signature's
    `leaves`, and jaxtyping cannot evaluate a symbolic axis that introduces its own
    name -- `block_offsets` precedes the leaf table in the signature.

    So this asserts the CURRENT behaviour, not the desirable one. Closing it means
    binding `farleaves` earlier in the signature, not renaming `leaves`.
    """
    _call_target_blocks(
        **_leaf_table(), block_offsets=jnp.zeros((LEAVES,), dtype=jnp.int32)
    )

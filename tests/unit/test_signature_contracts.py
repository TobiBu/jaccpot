"""Signature contracts: what each annotated kernel rejects, and must keep taking.

One row per kernel. Each row is a valid call (the CONTROL -- every rejection is
worthless without it) plus the perturbations that must be REJECTED and the ones that
must still be ACCEPTED. The rows were derived by execution: `bench/annotation_pilot`
fed each kernel shape perturbations and recorded which ones went through in silence,
and the annotations (or, for the fused near-field leaf entry points, the bodies' own
`ValueError` guards) were added to close them.

The ACCEPTED cases are restraint, and they are the ones worth reading. An axis that
looks shared in every recorded call can still be free:

* the near-field target block and source pool (`w` vs `sw`, `w` vs `srcslots`,
  `leaves` vs `srcleaves`): a leaf's neighbour count has nothing to do with its width,
  and the decoupled lane exists to separate the two pools (#297);
* `farleaves` is the FAR-field leaf view, not the near-field leaf table -- the octree
  backend had 5 against 3 -- so the prepacked and tiled rectangles must not be tied to
  `leaves` (the BODY may still refuse; only a `TypeCheckError` would be the regression);
* an `offsets` / `block_offsets` length is deliberately free: it precedes everything
  that binds `leaves`, and #324 showed an offsets relation is not stable across lanes
  (the short `block_offsets` is a KNOWN gap, pinned so it is not closed the wrong way);
* the remote sources and local targets of the distributed force scale, which a
  single-device recording always shows equal;
* `tbatch` is not `leaves` and `tiles` is not `blocks` in the tile-sequence helpers;
* the M2L scatter accumulator's node axis, and `base`'s length in the `_evaluate`
  scatters (a product jaxtyping cannot bind);
* an all-zero leaf table, a length-1 level-weight table, and Python-float `G` /
  `softening` on the reference oracle.

Merged here from twelve files in the 2026-10 cleanup (Z-a):
`nearfield/test_large_n_block_shape_contracts`, `nearfield/test_kernel_axis_contracts`,
`nearfield/test_near_field_axis_contracts`,
`pallas/test_nearfield_mutual_shape_contracts`, `pallas/test_fused_leaf_shape_guards`,
`pallas/test_fused_leaf_axis_contracts`, `pallas/test_fused_m2l_axis_contracts`,
`runtime/test_evaluate_scatter_contracts`, `runtime/test_reference_shape_contracts`,
`runtime/test_bucketed_schedule_shape_contract`, `runtime/test_m2l_shape_contracts`
and `test_force_scale_axis_contracts`. Every assertion they made is a case below;
the measurements behind each are in `docs/annotation_pilot_phase2_2026-08-30.md` and
the git history of those files.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaccpot import FastMultipoleMethod
from jaccpot.distributed._force_scale import (
    coarse_source_mac_geometry,
    cross_force_scale_own,
    flatten_neighbor_csr,
)
from jaccpot.nearfield._kernels import (
    _pair_contributions,
    _pair_contributions_batched,
    _self_contributions,
)
from jaccpot.nearfield._large_n_blocks import (
    _accumulate_target_block_tile_sequence,
    _compute_leaf_p2p_prepared_large_n_pairs_only_impl,
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl,
)
from jaccpot.nearfield._large_n_blocks import (
    _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_prepacked_impl as _prepacked_impl,
)
from jaccpot.nearfield._large_n_blocks import (
    _compute_leaf_p2p_prepared_large_n_self_only_impl,
    _compute_target_block_pairs_from_source_tiles,
)
from jaccpot.nearfield.near_field import (
    _compute_leaf_p2p_from_prepared_leaf_data_impl,
)
from jaccpot.pallas.m2l_real_fused import m2l_real_fused_jax, m2l_real_fused_vjp_pallas
from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_fused_leaf_pallas,
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
    nearfield_leafpair_pallas_decoupled,
)
from jaccpot.pallas.nearfield_mutual import (
    _block_tile,
    _block_vjp_tiles,
    _pair_weight_tile,
    mutual_leafpair_block_cvjp,
    mutual_leafpair_block_jax,
    mutual_leafpair_block_pallas,
    mutual_leafpair_block_vjp_pallas,
)
from jaccpot.runtime import reference
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime.kernels._evaluate import (
    _build_target_nearfield_source_index_matrix,
    _map_targets_to_leaf_positions,
    _scatter_rank3,
    _scatter_scalars,
    _scatter_vectors,
)
from jaccpot.runtime.kernels._m2l import (
    _chunk_segment_scatter_add,
    _m2l_chunk_contributions,
)
from jaccpot.runtime.kernels._shared import NearfieldInteropData

Overrides = Callable[[dict], dict]
Check = Callable[[Any, Any], None]


@dataclasses.dataclass(frozen=True)
class Case:
    """One perturbation of a row's valid call.

    Attributes
    ----------
    label : str
        What is perturbed, and why it matters.
    overrides : callable
        Maps the valid keyword arguments to the ones this case replaces.
    raises : type or tuple of types or None
        The exception the call must raise; ``None`` means it must be ACCEPTED.
    match : str or None
        A regex the exception message must contain.
    check : callable or None
        For an accepted case, ``check(output, control_output)``.
    body_may_raise : bool
        For an accepted case: only the SIGNATURE must accept it, so any exception
        but a `TypeCheckError` is the body's own business.
    """

    label: str
    overrides: Overrides
    raises: Any = TypeCheckError
    match: Optional[str] = None
    check: Optional[Check] = None
    body_may_raise: bool = False


def reject(label, overrides, raises=TypeCheckError, match=None) -> Case:
    """A perturbation the kernel must refuse.

    Parameters
    ----------
    label : str
        What is perturbed.
    overrides : callable
        Maps the valid arguments to the replaced ones.
    raises : type or tuple of types
        The exception expected.
    match : str or None
        Regex the message must contain.

    Returns
    -------
    Case
        The rejection case.
    """
    return Case(label, overrides, raises=raises, match=match)


def accept(label, overrides, check=None, body_may_raise=False) -> Case:
    """A perturbation the kernel must keep taking.

    Parameters
    ----------
    label : str
        What is perturbed.
    overrides : callable
        Maps the valid arguments to the replaced ones.
    check : callable or None
        ``check(output, control_output)`` on the result.
    body_may_raise : bool
        Only the signature must accept it.

    Returns
    -------
    Case
        The acceptance case.
    """
    return Case(
        label, overrides, raises=None, check=check, body_may_raise=body_may_raise
    )


@dataclasses.dataclass(frozen=True)
class Contract:
    """One kernel's signature contract.

    Attributes
    ----------
    kernel : str
        The test id.
    call : callable
        Calls the kernel with keyword arguments.
    args : callable
        Builds the valid keyword arguments.
    cases : tuple of Case
        The perturbations.
    check : callable or None
        ``check(output, None)`` on the control's result.
    control : str
        ``"run"`` (the valid call must succeed), ``"signature"`` (only its signature
        must accept it; the body may refuse placeholder contents) or ``"none"`` (the
        recorded contract has no control call).
    """

    kernel: str
    call: Callable[..., Any]
    args: Callable[[], dict]
    cases: tuple
    check: Optional[Check] = None
    control: str = "run"


def _shape(*expected) -> Check:
    """A check that the output has the given shape.

    Parameters
    ----------
    *expected : int
        The expected shape.

    Returns
    -------
    callable
        ``check(output, control)``.
    """

    def check(out, _control):
        assert out.shape == tuple(expected), (out.shape, expected)

    return check


def _drop_last(name: str, axis: int = 0) -> Overrides:
    """Overrides that drop the last entry of one argument along one axis.

    Parameters
    ----------
    name : str
        The argument.
    axis : int
        The axis to shorten.

    Returns
    -------
    callable
        The overrides.
    """

    def overrides(args):
        index = [slice(None)] * args[name].ndim
        index[axis] = slice(None, -1)
        return {name: args[name][tuple(index)]}

    return overrides


def _float():
    """The working float dtype for the current x64 setting.

    Returns
    -------
    numpy.dtype
        `float64` under `JAX_ENABLE_X64`, else `float32`.
    """
    return jnp.zeros(
        (), dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    ).dtype


# ---------------------------------------------------------------------------
# nearfield/_kernels.py: the pair and self kernels. The target and source sides
# are independent axes (8 against 5 here); each side must agree with itself.
# ---------------------------------------------------------------------------

W, SW, PAIRS, LEAVES = 8, 5, 4, 3


def _pair_args():
    """One valid `_pair_contributions` call, target width != source width.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "target_positions": jnp.zeros((W, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((W,), dtype=bool),
        "source_positions": jnp.ones((SW, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((SW,), dtype=jnp.float64),
        "source_mask": jnp.ones((SW,), dtype=bool),
        "softening_sq": 1e-4,
        "G": jnp.asarray(1.0),
        "compute_potential": False,
    }


def _batched_pair_args():
    """One valid `_pair_contributions_batched` call: only `pairs` is shared.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "target_positions": jnp.zeros((PAIRS, W, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((PAIRS, W), dtype=bool),
        "source_positions": jnp.ones((PAIRS, SW, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((PAIRS, SW), dtype=jnp.float64),
        "source_mask": jnp.ones((PAIRS, SW), dtype=bool),
        "softening_sq": 1e-4,
        "G": jnp.asarray(1.0),
        "compute_potential": False,
    }


def _self_args():
    """One valid `_self_contributions` call: one `leaves w` table.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "leaf_positions": jnp.zeros((LEAVES, W, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((LEAVES, W), dtype=jnp.float64),
        "mask": jnp.ones((LEAVES, W), dtype=bool),
        "softening_sq": 1e-4,
        "G": jnp.asarray(1.0),
        "compute_potential": False,
    }


def _accels_shape(*expected) -> Check:
    """A check on the accelerations of an `(accels, potential)` pair.

    Parameters
    ----------
    *expected : int
        The expected acceleration shape.

    Returns
    -------
    callable
        ``check(output, control)``.
    """

    def check(out, _control):
        assert out[0].shape == tuple(expected)

    return check


NEARFIELD_KERNELS = [
    Contract(
        "pair_contributions",
        lambda **kw: _pair_contributions(**kw),
        _pair_args,
        check=_accels_shape(W, 3),
        cases=(
            reject("target_mask one short", _drop_last("target_mask")),
            reject("source_masses one short", _drop_last("source_masses")),
            reject("source_mask one short", _drop_last("source_mask")),
            reject("two-component source positions", _drop_last("source_positions", 1)),
        ),
    ),
    Contract(
        "pair_contributions_batched",
        lambda **kw: _pair_contributions_batched(**kw),
        _batched_pair_args,
        check=_accels_shape(PAIRS, W, 3),
        cases=(reject("source_mask one pair short", _drop_last("source_mask")),),
    ),
    Contract(
        "self_contributions",
        lambda **kw: _self_contributions(**kw),
        _self_args,
        check=_accels_shape(LEAVES, W, 3),
        cases=(reject("mask one slot narrower", _drop_last("mask", 1)),),
    ),
]


# ---------------------------------------------------------------------------
# nearfield/near_field.py: the prepared-leaf kernel. Four leaf tables share
# `leaves w`, four per-edge arrays share `edges`, the three chunk tables are one
# grid, and `offsets` is rank-only on purpose.
# ---------------------------------------------------------------------------

PL_LEAVES, PL_W, PL_EDGES = 4, 2, 6
PL_N = PL_LEAVES * PL_W


def _prepared_leaf_args():
    """One valid prepared-leaf call, shaped after the recorded small cases.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "offsets": jnp.zeros((PL_LEAVES + 1,), dtype=jnp.int64),
        "neighbors": jnp.zeros((PL_EDGES,), dtype=jnp.int64),
        "positions": jnp.zeros((PL_N, 3), dtype=jnp.float64),
        "target_leaf_ids": jnp.zeros((PL_EDGES,), dtype=jnp.int64),
        "source_leaf_ids": jnp.zeros((PL_EDGES,), dtype=jnp.int64),
        "valid_pairs": jnp.zeros((PL_EDGES,), dtype=bool),
        "precomputed_chunk_sort_indices": jnp.zeros((0, 0), dtype=jnp.int64),
        "precomputed_chunk_group_ids": jnp.zeros((0, 0), dtype=jnp.int64),
        "precomputed_chunk_unique_indices": jnp.zeros((0, 0), dtype=jnp.int64),
        "leaf_positions": jnp.zeros((PL_LEAVES, PL_W, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((PL_LEAVES, PL_W), dtype=jnp.float64),
        "leaf_mask": jnp.ones((PL_LEAVES, PL_W), dtype=bool),
        "leaf_particle_idx": jnp.zeros((PL_LEAVES, PL_W), dtype=jnp.int64),
        "G": jnp.asarray(1.0),
        "softening_sq": jnp.asarray(1e-4),
        "return_potential": False,
        "collect_neighbor_pairs": False,
        "nearfield_mode": "baseline",
        "edge_chunk_size": 4,
        "use_precomputed_scatter": False,
    }


def _chunk_grid(group_ids_short: bool):
    """Overrides that give the three chunk tables one grid.

    Parameters
    ----------
    group_ids_short : bool
        Whether `precomputed_chunk_group_ids` loses its last column.

    Returns
    -------
    callable
        The overrides.
    """

    def overrides(_args):
        grid = jnp.zeros((2, 8), dtype=jnp.int64)
        return {
            "precomputed_chunk_sort_indices": grid,
            "precomputed_chunk_group_ids": grid[:, :-1] if group_ids_short else grid,
            "precomputed_chunk_unique_indices": grid,
        }

    return overrides


def _offsets(*shape) -> Overrides:
    """Overrides that replace `offsets`.

    Parameters
    ----------
    *shape : int
        Its shape.

    Returns
    -------
    callable
        The overrides.
    """
    return lambda _args: {"offsets": jnp.zeros(shape, dtype=jnp.int64)}


PREPARED_LEAF = [
    Contract(
        "prepared_leaf_p2p",
        lambda **kw: _compute_leaf_p2p_from_prepared_leaf_data_impl(**kw),
        _prepared_leaf_args,
        # With `return_potential=False` the kernel returns the acceleration array.
        check=_shape(PL_N, 3),
        cases=(
            *(
                reject(f"{name} disagrees on {axis}", _drop_last(name, dim))
                for name in ("leaf_masses", "leaf_mask", "leaf_particle_idx")
                for axis, dim in (("w", 1), ("leaves", 0))
            ),
            *(
                reject(f"{name} one edge short", _drop_last(name))
                for name in (
                    "neighbors",
                    "target_leaf_ids",
                    "source_leaf_ids",
                    "valid_pairs",
                )
            ),
            reject("the chunk tables disagree", _chunk_grid(group_ids_short=True)),
            accept("offsets of length leaves", _offsets(PL_LEAVES)),
            accept("offsets of length leaves+1", _offsets(PL_LEAVES + 1)),
            reject("offsets of rank 2", _offsets(PL_LEAVES + 1, 1)),
        ),
    ),
]


# ---------------------------------------------------------------------------
# nearfield/_large_n_blocks.py: the four block kernels share the padded leaf
# table; the rectangles' leading axis is `farleaves`; the tile helpers have
# `tiles tbatch blocks blocksize`.
# ---------------------------------------------------------------------------

N, LN_LEAVES, LN_W, LN_PAIRS, BLOCKS, BLOCKSIZE = 8, 4, 2, 6, 5, 1


def _leaf_table(leaves: int = LN_LEAVES, w: int = LN_W, n: int = N):
    """The padded leaf table every kernel in the family takes.

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


#: Chunking and scatter-strategy knobs, not shapes. `edge_chunk_size` must divide
#: the pair list or an internal reshape fails before any annotation is reached.
_EDGE_STATICS = {
    "G": 1.0,
    "softening_sq": jnp.asarray(1e-4),
    "edge_chunk_size": LN_PAIRS,
    "chunks_per_superchunk": 1,
    "sorted_scatter_hint": False,
    "grouped_sorted_scatter": False,
    "superchunk_target_reduce": False,
    "disable_chunk_cond": True,
}
_BLOCK_STATICS = {
    "G": 1.0,
    "softening_sq": jnp.asarray(1e-4),
    "target_leaf_batch_size": 2,
    "target_block_tile_size": 1,
    "target_block_tile_scan_unroll": 1,
    "target_block_batch_scan_unroll": 1,
}


def _pairs_only_args():
    """The leaf table plus the edge-list triple.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        **_leaf_table(),
        "target_leaf_ids": jnp.zeros((LN_PAIRS,), dtype=jnp.int32),
        "source_leaf_ids": jnp.zeros((LN_PAIRS,), dtype=jnp.int32),
        "valid_pairs": jnp.ones((LN_PAIRS,), dtype=bool),
    }


def _target_blocks_args():
    """The leaf table plus the run-length block table.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        **_leaf_table(),
        "block_offsets": jnp.zeros((LN_LEAVES + 1,), dtype=jnp.int32),
        "block_target_leaf_ids": jnp.zeros((BLOCKS,), dtype=jnp.int32),
        "block_source_leaf_ids": jnp.zeros((BLOCKS, BLOCKSIZE), dtype=jnp.int32),
        "block_valid_mask": jnp.ones((BLOCKS, BLOCKSIZE), dtype=bool),
    }


def _prepacked(farleaves: int = LN_LEAVES, blocksize: int = BLOCKSIZE) -> dict:
    """The prepacked rectangle and its mask.

    Parameters
    ----------
    farleaves : int
        The rectangle's leading (far-field leaf) extent.
    blocksize : int
        The mask's trailing extent.

    Returns
    -------
    dict
        The two padded arrays.
    """
    return {
        "block_source_leaf_ids_padded": jnp.zeros(
            (farleaves, BLOCKS, BLOCKSIZE), dtype=jnp.int32
        ),
        "block_valid_mask_padded": jnp.ones((farleaves, BLOCKS, blocksize), dtype=bool),
    }


def _leaf_table_cases():
    """The cases every block kernel shares: the leaf table's own contract.

    Returns
    -------
    tuple of Case
        Two-component positions, a self-inconsistent table, a narrower index
        block, an extra leading axis (accepted in five of the eight measured
        functions), and the all-zero table the kernels early-out on (STYLE_GUIDE
        section 4.3's sentinel argument turns on it being accepted).
    """
    return (
        reject("two-component positions", lambda a: {"positions": jnp.zeros((N, 2))}),
        reject(
            "leaf_masses one leaf short",
            lambda a: {"leaf_masses": jnp.ones((LN_LEAVES - 1, LN_W))},
        ),
        reject(
            "leaf_particle_idx narrower (it fixes w)",
            lambda a: {
                "leaf_particle_idx": jnp.zeros((LN_LEAVES, LN_W - 1), dtype=jnp.int32)
            },
        ),
        reject(
            "an extra leading axis on leaf_masses",
            lambda a: {"leaf_masses": a["leaf_masses"][None]},
        ),
        accept("an all-zero leaf table", lambda a: _leaf_table(leaves=0, w=0)),
    )


TILES, TBATCH, BLOCKS_T, BSZ_T = 2, 3, 2, 1
assert TBATCH != LN_LEAVES, "tbatch-is-not-leaves needs the two extents to differ"


def _tile_sequence(tiles=TILES, tbatch=TBATCH, blocks=BLOCKS_T, bsz=BSZ_T):
    """A `tiles tbatch blocks blocksize` pair of arrays.

    Parameters
    ----------
    tiles : int
        Tile-sequence length.
    tbatch : int
        Target leaves in this scan step.
    blocks : int
        Lane blocks per tile.
    bsz : int
        Lanes per block.

    Returns
    -------
    dict
        `tile_source_ids_seq` and `tile_source_valid_seq`.
    """
    return {
        "tile_source_ids_seq": jnp.zeros((tiles, tbatch, blocks, bsz), dtype=jnp.int32),
        "tile_source_valid_seq": jnp.ones((tiles, tbatch, blocks, bsz), dtype=bool),
    }


def _accumulate_args():
    """One valid tile-sequence accumulator call.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "target_pos": jnp.zeros((TBATCH, LN_W, 3)),
        "target_mask": jnp.ones((TBATCH, LN_W), dtype=bool),
        **_tile_sequence(),
        "leaf_positions": jnp.zeros((LN_LEAVES, LN_W, 3)),
        "leaf_masses": jnp.ones((LN_LEAVES, LN_W)),
        "leaf_mask": jnp.ones((LN_LEAVES, LN_W), dtype=bool),
    }


def _tiles(farleaves: int = LN_LEAVES) -> dict:
    """The canonical tiled rectangle and its mask.

    Parameters
    ----------
    farleaves : int
        The far-field leaf extent (axis 1).

    Returns
    -------
    dict
        `source_leaf_ids_tiles` and `source_valid_tiles`.
    """
    return {
        "source_leaf_ids_tiles": jnp.zeros(
            (TILES, farleaves, BLOCKS_T, BSZ_T), dtype=jnp.int32
        ),
        "source_valid_tiles": jnp.ones((TILES, farleaves, BLOCKS_T, BSZ_T), dtype=bool),
    }


LARGE_N_BLOCKS = [
    Contract(
        "large_n_self_only",
        lambda **kw: _compute_leaf_p2p_prepared_large_n_self_only_impl(
            **kw, G=1.0, softening_sq=jnp.asarray(1e-4)
        ),
        _leaf_table,
        cases=_leaf_table_cases(),
    ),
    Contract(
        "large_n_pairs_only",
        lambda **kw: _compute_leaf_p2p_prepared_large_n_pairs_only_impl(
            **kw, **_EDGE_STATICS
        ),
        _pairs_only_args,
        cases=(
            *_leaf_table_cases(),
            reject("valid_pairs one pair short", _drop_last("valid_pairs")),
        ),
    ),
    Contract(
        "large_n_target_blocks",
        lambda **kw: _compute_leaf_p2p_prepared_large_n_pairs_target_blocks_impl(
            **kw, **_BLOCK_STATICS
        ),
        _target_blocks_args,
        cases=(
            *_leaf_table_cases(),
            reject("block_valid_mask one block short", _drop_last("block_valid_mask")),
            reject(
                "block_target_leaf_ids one block short (JAX would clamp)",
                _drop_last("block_target_leaf_ids"),
            ),
            # A KNOWN, UNCLOSED gap: read at `[leaf + 1]`, JAX clamps, so a
            # `leaves`-length array gives the last leaf the wrong run. It stays
            # rank-only because the axis is `farleaves+1` upstream and precedes the
            # leaf table; closing it means binding `farleaves` earlier.
            accept(
                "block_offsets of length leaves (deliberately free)",
                lambda a: {"block_offsets": jnp.zeros((LN_LEAVES,), dtype=jnp.int32)},
            ),
        ),
    ),
    Contract(
        "large_n_prepacked",
        lambda **kw: _prepacked_impl(**kw, **_BLOCK_STATICS),
        lambda: {**_leaf_table(), **_prepacked()},
        cases=(
            *_leaf_table_cases(),
            reject(
                "the rectangle and its mask disagree on blocksize",
                lambda a: {
                    "block_valid_mask_padded": _prepacked(blocksize=BLOCKSIZE + 1)[
                        "block_valid_mask_padded"
                    ]
                },
            ),
            accept(
                "farleaves is not tied to leaves",
                lambda a: _prepacked(farleaves=LN_LEAVES + 2),
                body_may_raise=True,
            ),
        ),
    ),
    Contract(
        "large_n_tile_sequence_accumulate",
        lambda **kw: _accumulate_target_block_tile_sequence(
            **kw,
            g_const=jnp.asarray(1.0),
            softening_sq=jnp.asarray(1e-4),
            tile_unroll=1,
        ),
        _accumulate_args,
        cases=(
            reject(
                "tbatch is not leaves: a sequence shaped on leaves",
                lambda a: _tile_sequence(tbatch=LN_LEAVES),
            ),
            reject(
                "tiles is not blocks: the rank-3 layout the docstring once claimed",
                lambda a: {
                    "tile_source_ids_seq": jnp.zeros(
                        (TILES, TBATCH, BLOCKS_T), dtype=jnp.int32
                    ),
                    "tile_source_valid_seq": jnp.ones(
                        (TILES, TBATCH, BLOCKS_T), dtype=bool
                    ),
                },
            ),
            reject(
                "the sequence and its mask disagree",
                lambda a: {
                    "tile_source_valid_seq": jnp.ones(
                        (TILES + 1, TBATCH, BLOCKS_T, BSZ_T), dtype=bool
                    )
                },
            ),
            reject(
                "a target block wider than the leaves (w is shared by construction)",
                lambda a: {"target_pos": jnp.zeros((TBATCH, LN_W + 1, 3))},
            ),
            reject(
                "target_mask disagrees with target_pos on tbatch",
                lambda a: {"target_mask": jnp.ones((TBATCH + 1, LN_W), dtype=bool)},
            ),
        ),
    ),
    Contract(
        "large_n_pairs_from_source_tiles",
        lambda **kw: _compute_target_block_pairs_from_source_tiles(
            **kw,
            g_const=jnp.asarray(1.0),
            softening_sq=jnp.asarray(1e-4),
            target_leaf_batch_size=2,
            target_block_tile_scan_unroll=1,
            target_block_batch_scan_unroll=1,
        ),
        lambda: {**_leaf_table(), **_tiles()},
        cases=(
            accept(
                "the tiled leaf axis (farleaves) is not tied to leaves",
                lambda a: _tiles(farleaves=LN_LEAVES + 2),
                body_may_raise=True,
            ),
            reject(
                "two-component positions", lambda a: {"positions": jnp.zeros((N, 2))}
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# pallas/nearfield_mutual.py: the four entry points and the three tile helpers.
# The trailing 3 matters most: `_block_tile` reads components 0, 1, 2 and JAX
# clamps, so a two-component block silently computes the force for (x, y, y).
# The twin (`jax`) rejects exactly what the kernel rejects, so the parity tests
# cannot compare a real result against a fabricated one.
# ---------------------------------------------------------------------------

MU_PAIRS, SLOTS, LEVELS = 3, 4, 2


def _mutual_blocks():
    """One valid leaf-pair block set, `b` side included.

    Returns
    -------
    dict
        Keyword arguments for any of the entry points.
    """
    dtype = _float()
    ones2 = jnp.ones((MU_PAIRS, SLOTS), dtype=dtype)
    return {
        "xa": jnp.zeros((MU_PAIRS, SLOTS, 3), dtype=dtype),
        "ma": ones2,
        "va_f": ones2,
        "xb": jnp.ones((MU_PAIRS, SLOTS, 3), dtype=dtype),
        "mb": ones2,
        "vb_f": ones2,
        "rung_a_f": jnp.zeros((MU_PAIRS, SLOTS), dtype=dtype),
        "rung_b_f": jnp.zeros((MU_PAIRS, SLOTS), dtype=dtype),
        "level_weights": jnp.ones((LEVELS,), dtype=dtype),
        "softening_sq": jnp.asarray(1e-4, dtype=dtype),
        "g_value": jnp.asarray(1.0, dtype=dtype),
    }


def _mutual_vjp_pallas(fa_bar=None, fb_bar=None, **kw):
    """Call the Pallas reverse in interpret mode, with unit cotangents by default.

    Parameters
    ----------
    fa_bar, fb_bar : Array or None
        The cotangents; ``None`` means ones shaped like ``xa``.
    **kw
        The blocks.

    Returns
    -------
    tuple
        The cotangents.
    """
    ones = jnp.ones(kw["xa"].shape, dtype=kw["xa"].dtype)
    return mutual_leafpair_block_vjp_pallas(
        **kw,
        fa_bar=ones if fa_bar is None else fa_bar,
        fb_bar=ones if fb_bar is None else fb_bar,
        interpret=True,
    )


def _mutual_cvjp(num_levels=LEVELS, **kw):
    """Call the differentiable wrapper, which takes its statics positionally.

    Parameters
    ----------
    num_levels : int
        The level count.
    **kw
        The blocks.

    Returns
    -------
    tuple
        ``(F_a, F_b)``.
    """
    names = ("xa", "ma", "va_f", "xb", "mb", "vb_f", "rung_a_f", "rung_b_f")
    return mutual_leafpair_block_cvjp(
        *(kw[name] for name in names),
        kw["level_weights"],
        kw["softening_sq"],
        kw["g_value"],
        num_levels,
        False,
        True,
        True,
    )


def _mutual_entry_cases():
    """The cases the four mutual entry points share.

    Returns
    -------
    tuple of Case
        A two-component `xa`, the two sides disagreeing on `pairs` and on `w` (a
        wider `b` reaches `jnp.pad` with a negative width on the Pallas path), and
        an extra leading axis on `ma` (`_pad_inputs` accepted it on all four arrays).
    """
    return (
        reject("two-component xa", lambda a: {"xa": a["xa"][:, :, :2]}),
        reject("xb one pair short", _drop_last("xb")),
        reject(
            "the b side one slot narrower",
            lambda a: {
                name: a[name][:, :-1] for name in ("xb", "mb", "vb_f", "rung_b_f")
            },
        ),
        reject("an extra leading axis on ma", lambda a: {"ma": a["ma"][None]}),
    )


def _tile_args():
    """One valid per-pair `_block_tile` call.

    Returns
    -------
    dict
        Keyword arguments.
    """
    dtype = _float()
    vec = jnp.zeros((SLOTS,), dtype=dtype)
    return {
        "a_xyz": (vec, vec, vec),
        "ma": jnp.ones((SLOTS,), dtype=dtype),
        "va_f": jnp.ones((SLOTS,), dtype=dtype),
        "b_xyz": (vec + 1.0, vec + 1.0, vec + 1.0),
        "mb": jnp.ones((SLOTS,), dtype=dtype),
        "vb_f": jnp.ones((SLOTS,), dtype=dtype),
        "weight": None,
        # the tiles take the two softening scalars (eps^2, 0 for Plummer)
        "soft_params": (jnp.asarray(1e-4, dtype=dtype), jnp.asarray(0.0, dtype=dtype)),
        "g_value": jnp.asarray(1.0, dtype=dtype),
        "exclude_diagonal": False,
    }


def _vjp_tile_args():
    """`_tile_args` plus per-pair cotangents.

    Returns
    -------
    dict
        Keyword arguments for `_block_vjp_tiles`.
    """
    args = _tile_args()
    vec = args["ma"]
    return {**args, "fa_bar_xyz": (vec, vec, vec), "fb_bar_xyz": (vec, vec, vec)}


def _check_tile(out, _control):
    """Three `(slots, slots)` force-component tiles."""
    fx, fy, fz = out
    assert fx.shape == fy.shape == fz.shape == (SLOTS, SLOTS)


def _check_vjp_tile(out, _control):
    """The four block cotangents, then level-weight (None), softening and G."""
    dxa, dma, dxb, dmb, _dlw, dsoft, dg = out
    assert dma.shape == dmb.shape == (SLOTS,)
    assert len(dxa) == len(dxb) == 3
    assert dsoft.shape == dg.shape == ()


MUTUAL = [
    Contract(
        "mutual_block_jax",
        lambda **kw: mutual_leafpair_block_jax(**kw),
        _mutual_blocks,
        cases=_mutual_entry_cases(),
    ),
    Contract(
        "mutual_block_pallas",
        lambda **kw: mutual_leafpair_block_pallas(**kw, interpret=True),
        _mutual_blocks,
        cases=_mutual_entry_cases(),
    ),
    Contract(
        "mutual_block_vjp_pallas",
        _mutual_vjp_pallas,
        _mutual_blocks,
        cases=(
            *_mutual_entry_cases(),
            reject(
                "a batched cotangent",
                lambda a: {
                    "fa_bar": jnp.ones((1,) + a["xa"].shape, dtype=a["xa"].dtype)
                },
            ),
        ),
    ),
    Contract(
        "mutual_block_cvjp",
        _mutual_cvjp,
        _mutual_blocks,
        cases=(
            *_mutual_entry_cases(),
            # Weighting off: `mutual/nearfield.py` passes `jnp.ones((1,))`, not None,
            # across the custom_vjp boundary, and nothing else binds `levels`.
            accept(
                "a disabled (length-1) level-weight table",
                lambda a: {
                    "level_weights": jnp.ones((1,), dtype=a["xa"].dtype),
                    "num_levels": 0,
                },
            ),
        ),
    ),
    Contract(
        "mutual_block_tile",
        lambda **kw: _block_tile(**kw),
        _tile_args,
        check=_check_tile,
        cases=(
            reject(
                "a batched ma (the caller skipped the vmap)",
                lambda a: {"ma": a["ma"][None]},
            ),
            reject(
                "a short component inside a_xyz",
                lambda a: {"a_xyz": (a["a_xyz"][0][:-1], *a["a_xyz"][1:])},
            ),
            reject(
                "a weight tile not square in the tile width",
                lambda a: {"weight": jnp.ones((SLOTS, SLOTS - 1), dtype=a["ma"].dtype)},
            ),
        ),
    ),
    Contract(
        "mutual_pair_weight_tile",
        lambda **kw: _pair_weight_tile(**kw),
        lambda: {
            "rung_a_f": jnp.zeros((SLOTS,), dtype=_float()),
            "rung_b_f": jnp.zeros((SLOTS,), dtype=_float()),
            "level_weights": jnp.ones((LEVELS,), dtype=_float()),
            "num_levels": LEVELS,
        },
        check=_shape(SLOTS, SLOTS),
        cases=(
            # Broadcast before: the wrong level weight on a column of pairs (#297).
            reject("rung_a_f one short", _drop_last("rung_a_f")),
            reject("rung_b_f one short", _drop_last("rung_b_f")),
        ),
    ),
    Contract(
        "mutual_block_vjp_tiles",
        lambda **kw: _block_vjp_tiles(**kw),
        _vjp_tile_args,
        check=_check_vjp_tile,
        cases=(
            reject(
                "a batched cotangent component",
                lambda a: {
                    "fb_bar_xyz": (a["fb_bar_xyz"][0][None], *a["fb_bar_xyz"][1:])
                },
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# runtime/kernels/_evaluate.py: the masked scatters (a mask of the wrong length
# broadcasts instead of complaining) and the target / leaf / node axes, which are
# three different extents (`nodes == 2 * leaves - 1` on the radix tree).
# ---------------------------------------------------------------------------

EV_LEAVES, EV_W = 4, 8
EV_N = EV_LEAVES * EV_W
TARGETS, NODES = 5, 2 * EV_LEAVES - 1


def _scatter_args(base_shape, values_shape):
    """One valid scatter call.

    Parameters
    ----------
    base_shape, values_shape : tuple of int
        The flat buffer's and the leaf-major values' shapes.

    Returns
    -------
    callable
        Builds the keyword arguments.
    """
    return lambda: {
        "base": jnp.zeros(base_shape, dtype=_float()),
        "indices": jnp.zeros((EV_LEAVES, EV_W), dtype=jnp.int64),
        "values": jnp.ones(values_shape, dtype=_float()),
        "mask": jnp.ones((EV_LEAVES, EV_W), dtype=bool),
    }


def _scatter_cases():
    """The mask and index cases the three scatters share.

    Returns
    -------
    tuple of Case
        The mask short on each axis, and `indices` short.
    """
    return (
        reject("mask one leaf short", _drop_last("mask")),
        reject("mask one slot narrower", _drop_last("mask", 1)),
        reject("indices one leaf short", _drop_last("indices")),
    )


def _rank3(comps: int, base_comps: Optional[int] = None) -> Overrides:
    """Overrides giving `_scatter_rank3` a component count.

    Parameters
    ----------
    comps : int
        The values' component count.
    base_comps : int or None
        The base's, if different.

    Returns
    -------
    callable
        The overrides.
    """
    return lambda _a: {
        "base": jnp.zeros((EV_N, 3, base_comps or comps), dtype=_float()),
        "values": jnp.ones((EV_LEAVES, EV_W, 3, comps), dtype=_float()),
    }


def _interop():
    """A minimally populated interop payload; only its type is checked.

    Returns
    -------
    NearfieldInteropData
        The payload.
    """
    return NearfieldInteropData(
        leaf_nodes=jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
        node_ranges=jnp.zeros((NODES, 2), dtype=jnp.int64),
        offsets=jnp.zeros((EV_LEAVES + 1,), dtype=jnp.int64),
        neighbors=jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
        counts=jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
        particle_order_node_ranges=jnp.zeros((NODES, 2), dtype=jnp.int64),
        particle_order_leaf_indices=jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
        particle_order_to_native_leaf=jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
    )


EVALUATE = [
    Contract(
        "evaluate_scatter_vectors",
        lambda **kw: _scatter_vectors(**kw),
        _scatter_args((EV_N, 3), (EV_LEAVES, EV_W, 3)),
        check=_shape(EV_N, 3),
        cases=(
            *_scatter_cases(),
            reject(
                "a scalar payload",
                lambda a: {"values": jnp.ones((EV_LEAVES, EV_W), dtype=_float())},
            ),
            # `base` is `leaves * w` long and jaxtyping cannot bind a product: the
            # length is free (the CURRENT behaviour, not the desirable one) ...
            accept(
                "a base of another length",
                lambda a: {"base": jnp.zeros((EV_N + 5, 3), dtype=_float())},
                check=lambda out, _c: _shape(EV_N + 5, 3)(out, None),
            ),
            # ... but its rank is checked.
            reject(
                "a rank-1 base", lambda a: {"base": jnp.zeros((EV_N,), dtype=_float())}
            ),
        ),
    ),
    Contract(
        "evaluate_scatter_scalars",
        lambda **kw: _scatter_scalars(**kw),
        _scatter_args((EV_N,), (EV_LEAVES, EV_W)),
        check=_shape(EV_N),
        cases=(
            *_scatter_cases(),
            reject(
                "a vector payload",
                lambda a: {"values": jnp.ones((EV_LEAVES, EV_W, 3), dtype=_float())},
            ),
        ),
    ),
    Contract(
        "evaluate_scatter_rank3",
        lambda **kw: _scatter_rank3(**kw),
        _scatter_args((EV_N, 3, 6), (EV_LEAVES, EV_W, 3, 6)),
        check=_shape(EV_N, 3, 6),
        cases=(
            *_scatter_cases(),
            # The lifted-derivative component count: 3 at level 1, 6 at level 2.
            # Pinning 3 broke the k=3 time-derivative test, whose base is (12, 3, 6).
            *(
                accept(
                    f"{comps} derivative components",
                    _rank3(comps),
                    check=lambda out, _c, comps=comps: _shape(EV_N, 3, comps)(
                        out, None
                    ),
                )
                for comps in (3, 6, 10)
            ),
            reject("base and values disagree on comps", _rank3(3, base_comps=6)),
        ),
    ),
    Contract(
        "evaluate_target_source_index_matrix",
        lambda **kw: _build_target_nearfield_source_index_matrix(**kw),
        lambda: {
            "target_sorted_indices": jnp.zeros((TARGETS,), dtype=jnp.int64),
            "target_leaf_positions": jnp.zeros((TARGETS,), dtype=jnp.int64),
            "nearfield_interop": _interop(),
        },
        control="none",
        cases=(
            reject(
                "target_leaf_positions disagrees with target_sorted_indices on t",
                _drop_last("target_leaf_positions"),
            ),
        ),
    ),
    Contract(
        "evaluate_map_targets_to_leaves",
        lambda **kw: _map_targets_to_leaf_positions(**kw),
        lambda: {
            "leaf_nodes": jnp.zeros((EV_LEAVES,), dtype=jnp.int64),
            "node_ranges": jnp.zeros((NODES, 2), dtype=jnp.int64),
            "target_sorted_indices": jnp.zeros((TARGETS,), dtype=jnp.int64),
        },
        # Three distinct extents (leaves, nodes, t) must pass the signature.
        control="signature",
        cases=(
            reject(
                "node_ranges without the (start, end) pair",
                lambda a: {"node_ranges": a["node_ranges"][:, :1]},
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# pallas/nearfield_fused_leaf.py: the positions stay bare, because the bodies
# check them by name and a decorator would replace that message; the bodies also
# check the masks and pools that must agree with them (a MASK that disagrees
# leaves the grid alone, so the BlockSpec clamps and a leaf reuses another's
# mask). The other parallel arrays are annotated.
# ---------------------------------------------------------------------------

FL_LEAVES, WT, SRCSLOTS, SRCLEAVES = 3, 2, 4, 2
SOFT = jnp.asarray(1e-4)
GRAV = jnp.asarray(1.0)


def _fused_leaf_args():
    """One valid `nearfield_fused_leaf_pallas` call; leaves, w, srcslots distinct.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "target_positions": jnp.zeros((FL_LEAVES, WT, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((FL_LEAVES, WT), dtype=bool),
        "source_positions": jnp.zeros((FL_LEAVES, SRCSLOTS, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((FL_LEAVES, SRCSLOTS), dtype=jnp.float64),
        "source_mask": jnp.ones((FL_LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
        "interpret": True,
    }


def _leafpair_args():
    """One valid leaf-pair call (no `interpret`; the Pallas row adds it).

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "leaf_positions": jnp.zeros((FL_LEAVES, WT, 3), dtype=jnp.float64),
        "leaf_masses": jnp.ones((FL_LEAVES, WT), dtype=jnp.float64),
        "leaf_mask": jnp.ones((FL_LEAVES, WT), dtype=bool),
        "source_leaf_ids": jnp.zeros((FL_LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((FL_LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
    }


def _decoupled_args():
    """One valid decoupled call, the source pool a DIFFERENT leaf count (its purpose).

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "target_positions": jnp.zeros((FL_LEAVES, WT, 3), dtype=jnp.float64),
        "target_mask": jnp.ones((FL_LEAVES, WT), dtype=bool),
        "source_positions": jnp.zeros((SRCLEAVES, WT, 3), dtype=jnp.float64),
        "source_masses": jnp.ones((SRCLEAVES, WT), dtype=jnp.float64),
        "source_mask": jnp.ones((SRCLEAVES, WT), dtype=bool),
        "source_leaf_ids": jnp.zeros((FL_LEAVES, SRCSLOTS), dtype=jnp.int32),
        "source_valid": jnp.ones((FL_LEAVES, SRCSLOTS), dtype=bool),
        "softening_sq": SOFT,
        "G": GRAV,
        "interpret": True,
    }


def _replace(**arrays) -> Overrides:
    """Overrides that replace arguments with fixed arrays.

    Parameters
    ----------
    **arrays : Array
        The replacements.

    Returns
    -------
    callable
        The overrides.
    """
    return lambda _a: arrays


FUSED_LEAF = [
    Contract(
        "fused_leaf_pallas",
        lambda **kw: nearfield_fused_leaf_pallas(**kw),
        _fused_leaf_args,
        check=_shape(FL_LEAVES, WT, 4),
        # Each returned a full-size, plausible result before its guard. The message
        # must name the argument the caller got wrong. `source_masses` / `source_mask`
        # are absent: their annotation already cross-checked them.
        cases=(
            reject(
                "target_positions one leaf short",
                lambda a: {"target_positions": a["target_positions"][:-1]},
                ValueError,
                "target_mask",
            ),
            reject(
                "target_mask one leaf short",
                _drop_last("target_mask"),
                ValueError,
                "target_mask",
            ),
            reject(
                "target_mask one slot narrower",
                _drop_last("target_mask", 1),
                ValueError,
                "target_mask",
            ),
            reject(
                "source_positions one leaf short",
                _drop_last("source_positions"),
                ValueError,
                "source_positions",
            ),
        ),
    ),
    Contract(
        "fused_leafpair_pallas",
        lambda **kw: nearfield_leafpair_pallas(**kw, interpret=True),
        _leafpair_args,
        check=_shape(FL_LEAVES, WT, 4),
        cases=(
            # The grid source AND the gather target: one leaf short dropped the last
            # leaf and read every 3-row array into the wrong block.
            reject(
                "leaf_positions one leaf short",
                _drop_last("leaf_positions"),
                ValueError,
            ),
            # The body's message survives because positions stay bare.
            reject(
                "rank-2 leaf_positions",
                _replace(leaf_positions=jnp.zeros((FL_LEAVES, WT), dtype=jnp.float64)),
                ValueError,
                "leaf_positions must have shape",
            ),
        ),
    ),
    Contract(
        "fused_leafpair_pallas_decoupled",
        lambda **kw: nearfield_leafpair_pallas_decoupled(**kw),
        _decoupled_args,
        check=_shape(FL_LEAVES, WT, 4),
        cases=(
            # Targets against `num_targets`, sources against `num_sources`.
            reject(
                "target_positions one leaf short",
                _drop_last("target_positions"),
                ValueError,
            ),
            reject(
                "target_mask one slot narrower",
                _drop_last("target_mask", 1),
                ValueError,
            ),
            reject(
                "source_positions one leaf short",
                _drop_last("source_positions"),
                ValueError,
            ),
            reject("source_mask one leaf short", _drop_last("source_mask")),
            # #297's specific message keeps priority over the generic guard.
            reject(
                "a narrower source pool",
                _replace(
                    source_positions=jnp.zeros(
                        (SRCLEAVES, WT - 1, 3), dtype=jnp.float64
                    ),
                    source_masses=jnp.ones((SRCLEAVES, WT - 1), dtype=jnp.float64),
                    source_mask=jnp.ones((SRCLEAVES, WT - 1), dtype=bool),
                ),
                ValueError,
                "same leaf width",
            ),
        ),
    ),
    Contract(
        "fused_leafpair_jax",
        lambda **kw: nearfield_leafpair_jax(**kw),
        _leafpair_args,
        check=_shape(FL_LEAVES, WT, 4),
        cases=(
            # It had no check at all: width 2 gave a 3-wide result where 4 is the
            # contract, silently.
            reject(
                "two-component leaf_positions",
                _replace(
                    leaf_positions=jnp.zeros((FL_LEAVES, WT, 2), dtype=jnp.float64)
                ),
                ValueError,
                "leaf_positions",
            ),
            reject("leaf_mask one slot narrower", _drop_last("leaf_mask", 1)),
            reject("leaf_masses one leaf short", _drop_last("leaf_masses")),
            reject("source_valid one slot narrower", _drop_last("source_valid", 1)),
            accept(
                "srcslots is free of w",
                _replace(
                    source_leaf_ids=jnp.zeros(
                        (FL_LEAVES, SRCSLOTS + 2), dtype=jnp.int32
                    ),
                    source_valid=jnp.ones((FL_LEAVES, SRCSLOTS + 2), dtype=bool),
                ),
                check=lambda out, _c: _shape(FL_LEAVES, WT, 4)(out, None),
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# runtime/reference.py: the oracle the FMM paths are checked against, so a shape
# hole here can make a wrong FMM look right. A bare `(3,)` was read as THREE
# evaluation points (it returned [-7.84, -7.91, -7.73] where -8.22 is right).
# ---------------------------------------------------------------------------

REF_N = 8


@functools.lru_cache(maxsize=None)
def _ref_sources():
    """Eight source particles.

    Returns
    -------
    tuple
        ``(positions, masses)``.
    """
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.standard_normal((REF_N, 3)))
    masses = jnp.asarray(np.abs(rng.standard_normal(REF_N)) + 0.5)
    return positions, masses


def _direct_sum_args():
    """Eight sources and one evaluation point.

    Returns
    -------
    dict
        Keyword arguments for `direct_sum`.
    """
    positions, masses = _ref_sources()
    return {
        "positions": positions,
        "masses": masses,
        "eval_point": jnp.asarray([0.1, 0.2, 0.3]),
    }


def _two_body_midpoint(_args):
    """The midpoint of two point masses: not a source, so no 0/0 self term.

    Returns
    -------
    dict
        The replaced arguments.
    """
    return {
        "positions": jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "masses": jnp.asarray([2.0, 3.0]),
        "eval_point": jnp.asarray([0.5, 0.0, 0.0]),
    }


def _check_newtonian(out, _control):
    """The hand-written Newtonian sum: the numbers must not move, only rejections."""
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = np.asarray([2.0, 3.0])
    offsets = np.asarray([0.5, 0.0, 0.0]) - positions
    distances = np.linalg.norm(offsets, axis=1)
    expected = -np.sum(masses[:, None] * offsets / distances[:, None] ** 3, axis=0)
    assert np.allclose(np.asarray(out), expected)


def _check_equals_control(out, control):
    """`G` and `softening` stay bare: Python floats and arrays give the defaults."""
    assert np.allclose(np.asarray(out), np.asarray(control))


def _check_potential(out, _control):
    """One finite value per point."""
    assert out.shape == (1,)
    assert np.all(np.isfinite(np.asarray(out)))


REFERENCE = [
    Contract(
        "reference_potential",
        lambda **kw: reference.compute_gravitational_potential(**kw),
        lambda: {
            "positions": _ref_sources()[0],
            "masses": _ref_sources()[1],
            "eval_points": jnp.asarray([[0.1, 0.2, 0.3]]),
        },
        check=_check_potential,
        cases=(
            reject(
                "a bare 3-vector is not three evaluation points",
                _replace(eval_points=jnp.asarray([0.1, 0.2, 0.3])),
                Exception,
                "(?i)eval_points|shape|dim",
            ),
            accept(
                "four points",
                _replace(eval_points=jnp.zeros((4, 3))),
                check=lambda out, _c: _shape(4)(out, None),
            ),
        ),
    ),
    Contract(
        "reference_direct_sum",
        lambda **kw: reference.direct_sum(**kw),
        _direct_sum_args,
        check=_shape(3),
        cases=(
            # Benign here (it broadcast to the same answer), closed anyway: the
            # sibling's same spelling was the wrong answer above.
            reject(
                "a (1, 3) evaluation point",
                _replace(eval_point=jnp.zeros((1, 3))),
                Exception,
                "(?i)eval_point|shape|dim",
            ),
            reject(
                "masses one short of positions",
                _drop_last("masses"),
                Exception,
                "(?i)masses|positions|shape|dim",
            ),
            accept(
                "Python-float G and softening",
                _replace(G=1.0, softening=0.0),
                check=_check_equals_control,
            ),
            accept(
                "array G and softening",
                _replace(G=jnp.asarray(1.0), softening=jnp.asarray(0.0)),
                check=_check_equals_control,
            ),
            accept(
                "the Newtonian two-body sum", _two_body_midpoint, check=_check_newtonian
            ),
            # Documented: with the default `softening=0.0` the self term is 0/0.
            accept(
                "evaluated at a source without softening: nan",
                lambda a: {"eval_point": a["positions"][0]},
                check=lambda out, _c: np.testing.assert_(
                    np.all(np.isnan(np.asarray(out)))
                ),
            ),
        ),
    ),
    Contract(
        # `SweepsMixin` delegates to `reference.py`; it must not be a way around it.
        "sweeps_facade_direct_sum",
        lambda **kw: FMMEngine().direct_sum(**kw),
        _direct_sum_args,
        check=_shape(3),
        cases=(
            reject(
                "a (1, 3) evaluation point",
                _replace(eval_point=jnp.zeros((1, 3))),
                Exception,
                "(?i)eval_point|shape|dim",
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# PrepareMixin._prepare_bucketed_scatter_schedules_safe: `_safe` swallows failures
# and returns None, so before the annotation a length-mismatched
# `target_leaf_ids` built a DIFFERENT schedule silently, and a rank mismatch
# silently dropped the schedule (a slowdown no numerical test can see).
# ---------------------------------------------------------------------------

BK_N, BK_LEAF, BK_PAIRS = 256, 32, 64


@functools.lru_cache(maxsize=None)
def _bucketed_engine():
    """A real engine and a real near-field interop payload (the helper reads it).

    Returns
    -------
    tuple
        ``(engine, nearfield_interop)``.
    """
    rng = np.random.default_rng(3)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(BK_N, 3)), dtype=jnp.float64)
    masses = jnp.asarray(np.abs(rng.standard_normal(BK_N)) + 0.5, dtype=jnp.float64)
    fmm = FastMultipoleMethod(basis="real", theta=0.5, G=1.0, softening=1e-2)
    state = fmm.prepare_state(positions, masses, max_order=2, leaf_size=BK_LEAF)
    return fmm._impl, state.nearfield_interop


def _bucketed_call(**kw):
    """Invoke the schedule builder on the cached engine.

    Parameters
    ----------
    **kw
        `target_leaf_ids` and `valid_pairs`.

    Returns
    -------
    tuple
        The three optional schedule arrays.
    """
    engine, interop = _bucketed_engine()
    return engine._prepare_bucketed_scatter_schedules_safe(
        nearfield_interop=interop, leaf_cap=BK_LEAF, edge_chunk_size=BK_PAIRS, **kw
    )


def _bucketed_args():
    """One entry per candidate leaf PAIR (not per particle).

    Returns
    -------
    dict
        Keyword arguments.
    """
    rng = np.random.default_rng(3)
    return {
        "target_leaf_ids": jnp.asarray(
            rng.integers(0, BK_N // BK_LEAF, size=(BK_PAIRS,)), dtype=jnp.int32
        ),
        "valid_pairs": jnp.ones((BK_PAIRS,), dtype=bool),
    }


def _check_all_built(out, _control):
    """All three schedules are built from the production shapes."""
    assert sum(entry is not None for entry in out) == 3


BUCKETED = [
    Contract(
        "bucketed_scatter_schedule",
        _bucketed_call,
        _bucketed_args,
        check=_check_all_built,
        cases=(
            reject("target_leaf_ids one short", _drop_last("target_leaf_ids")),
            reject(
                "target_leaf_ids with an extra leading axis",
                lambda a: {"target_leaf_ids": a["target_leaf_ids"][None, :]},
            ),
            reject("valid_pairs one short", _drop_last("valid_pairs")),
            reject(
                "valid_pairs with an extra trailing axis",
                lambda a: {"valid_pairs": a["valid_pairs"][:, None]},
            ),
            # Tightened, not a caught defect: a float 0/1 mask built an identical
            # schedule, but `valid_pairs` is a mask and every caller passes a bool.
            reject(
                "a float validity mask",
                lambda a: {"valid_pairs": jnp.ones((BK_PAIRS,), dtype=jnp.float64)},
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# runtime/kernels/_m2l.py and pallas/m2l_real_fused.py: `nodes` / `sh` on the M2L
# seam; the fused kernel and its pure-JAX twin carry the same axes (`degrees` is
# p+1, `blockdim` is 2p+1 and square, `sh` is (p+1)**2).
# ---------------------------------------------------------------------------

CHUNK, CHUNK_SH, CHUNK_NODES = 512, 25, 255
FM_PAIRS, P = 3, 2
DEGREES, BLOCKDIM, SH = P + 1, 2 * P + 1, (P + 1) ** 2


def _m2l_chunk_args():
    """One valid `_m2l_chunk_contributions` call.

    Returns
    -------
    dict
        Keyword arguments.
    """
    idx = jnp.zeros((2,), dtype=jnp.int32)
    return {
        "multip_packed": jnp.zeros((6, 9), dtype=_float()),
        "centers": jnp.zeros((6, 3), dtype=_float()),
        "src_idx": idx,
        "tgt_idx": idx,
        "valid": jnp.ones((2,), dtype=bool),
        "order": 2,
        "m2l_impl": "rot_scale",
        "out_dtype": _float(),
    }


def _chunk_scatter_args():
    """One valid chunked-scatter call.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "local_accum": jnp.zeros((CHUNK_NODES, CHUNK_SH), dtype=jnp.float64),
        "contribs": jnp.ones((CHUNK, CHUNK_SH), dtype=jnp.float64),
        "tgt_chunk": jnp.zeros((CHUNK,), dtype=jnp.int64),
        "valid": jnp.ones((CHUNK,), dtype=bool),
    }


def _fused_m2l_args():
    """One valid fused-M2L call.

    Returns
    -------
    dict
        Keyword arguments.
    """
    blocks = jnp.zeros((FM_PAIRS, DEGREES, BLOCKDIM, BLOCKDIM), dtype=jnp.float64)
    return {
        "multipoles": jnp.zeros((FM_PAIRS, SH), dtype=jnp.float64),
        "blocks_to_z": blocks,
        "blocks_from_z": blocks,
        "r": jnp.ones((FM_PAIRS,), dtype=jnp.float64),
    }


M2L = [
    Contract(
        "m2l_chunk_contributions",
        lambda **kw: _m2l_chunk_contributions(**kw),
        _m2l_chunk_args,
        control="none",
        cases=(
            reject(
                "centers disagree with the multipoles on nodes", _drop_last("centers")
            ),
            # Already refused by the displacement arithmetic; documents the contract.
            reject("a two-component centre", _drop_last("centers", 1)),
        ),
    ),
    Contract(
        "m2l_chunk_segment_scatter_add",
        lambda **kw: _chunk_segment_scatter_add(**kw, chunk_size=CHUNK),
        _chunk_scatter_args,
        check=_shape(CHUNK_NODES, CHUNK_SH),
        cases=(
            # The measured defect: an out-of-bounds gather JAX clamps (row 510 twice).
            reject("contribs one row short", _drop_last("contribs")),
            # Refused by broadcasting before the annotation, hence either exception.
            reject(
                "accumulator and contributions disagree on sh",
                _drop_last("local_accum", 1),
                (TypeCheckError, ValueError),
            ),
            # The target-node count has nothing to do with the chunk.
            accept(
                "the accumulator's node axis is free",
                _drop_last("local_accum"),
                check=lambda out, _c: _shape(CHUNK_NODES - 1, CHUNK_SH)(out, None),
            ),
        ),
    ),
    Contract(
        "m2l_real_fused_jax",
        lambda **kw: m2l_real_fused_jax(**kw, order=P),
        _fused_m2l_args,
        check=_shape(FM_PAIRS, SH),
        cases=(
            *(
                reject(f"{name} one pair short", _drop_last(name))
                for name in ("multipoles", "blocks_to_z", "blocks_from_z", "r")
            ),
            reject("a non-square rotation block", _drop_last("blocks_to_z", 3)),
            # The two halves of one rotation, built for the same order.
            reject(
                "the block stacks disagree on degrees", _drop_last("blocks_from_z", 1)
            ),
        ),
    ),
    Contract(
        # `sh` binds here: on the forward it occurs once per signature, and a short
        # multipole is caught by the body's broadcasting instead.
        "m2l_real_fused_vjp_pallas",
        lambda **kw: m2l_real_fused_vjp_pallas(**kw, order=P),
        lambda: {
            **_fused_m2l_args(),
            "out_bar": jnp.zeros((FM_PAIRS, SH), dtype=jnp.float64),
        },
        control="none",
        cases=(
            reject(
                "out_bar disagrees with the multipoles on sh", _drop_last("out_bar", 1)
            ),
            reject(
                "multipoles disagree with out_bar on sh", _drop_last("multipoles", 1)
            ),
        ),
    ),
]


# ---------------------------------------------------------------------------
# distributed/_force_scale.py: separations from CALL SITES, because the
# recording is single-device: the remote coarse tree is a different object from
# the local tree, and `cross_force_scale_own`'s sources are the remote domain.
# ---------------------------------------------------------------------------

SOURCES, FS_TARGETS, FS_PAIRS, FS_LEAVES, EDGES = 7, 4, 11, 5, 13


def _cross_args():
    """One valid `cross_force_scale_own` call with sources != targets.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "source_masses": jnp.ones((SOURCES,), dtype=jnp.float64),
        "source_centers": jnp.zeros((SOURCES, 3), dtype=jnp.float64),
        "source_radii": jnp.ones((SOURCES,), dtype=jnp.float64),
        "target_centers": jnp.zeros((FS_TARGETS, 3), dtype=jnp.float64),
        "target_radii": jnp.ones((FS_TARGETS,), dtype=jnp.float64),
        "pair_sources": jnp.zeros((FS_PAIRS,), dtype=jnp.int32),
        "pair_targets": jnp.zeros((FS_PAIRS,), dtype=jnp.int32),
        "pair_valid": jnp.ones((FS_PAIRS,), dtype=bool),
        "num_target_nodes": FS_TARGETS,
        "g": jnp.asarray(1.0),
        "eps_sq": jnp.asarray(1e-4),
        "inflation": jnp.asarray(1.0),
    }


FORCE_SCALE = [
    Contract(
        "distributed_cross_force_scale_own",
        lambda **kw: cross_force_scale_own(**kw),
        _cross_args,
        # 7 remote sources against 4 local targets go through.
        check=_shape(FS_TARGETS),
        cases=(
            reject("source_radii one short", _drop_last("source_radii")),
            reject("target_radii one short", _drop_last("target_radii")),
            reject("pair_targets one short", _drop_last("pair_targets")),
            reject("pair_valid one short", _drop_last("pair_valid")),
        ),
    ),
    Contract(
        # Per-leaf heads agree; the flattened edge list is free of them.
        "distributed_flatten_neighbor_csr",
        lambda **kw: flatten_neighbor_csr(**kw),
        lambda: {
            "counts": jnp.zeros((FS_LEAVES,), dtype=jnp.int32),
            "indices": jnp.zeros((EDGES,), dtype=jnp.int32),
            "leaf_indices": jnp.zeros((FS_LEAVES,), dtype=jnp.int32),
        },
        cases=(reject("leaf_indices one short", _drop_last("leaf_indices")),),
    ),
    Contract(
        "distributed_coarse_source_mac_geometry",
        lambda **kw: coarse_source_mac_geometry(**kw),
        lambda: {
            "expansion_centers": jnp.zeros((FS_LEAVES, 3), dtype=jnp.float64),
            "geometry_centers": jnp.zeros((FS_LEAVES, 3), dtype=jnp.float64),
            "geometry_radii": jnp.ones((FS_LEAVES,), dtype=jnp.float64),
        },
        cases=(reject("geometry_radii one short", _drop_last("geometry_radii")),),
    ),
]


CONTRACTS = [
    *NEARFIELD_KERNELS,
    *PREPARED_LEAF,
    *LARGE_N_BLOCKS,
    *MUTUAL,
    *EVALUATE,
    *FUSED_LEAF,
    *REFERENCE,
    *BUCKETED,
    *M2L,
    *FORCE_SCALE,
]


def _run_case(contract: Contract, case: Case, args: dict, control: Any) -> None:
    """Run one perturbation and assert its outcome.

    Parameters
    ----------
    contract : Contract
        The row.
    case : Case
        The perturbation.
    args : dict
        The row's valid arguments.
    control : Any
        The control's output (``None`` without one).
    """
    kwargs = {**args, **case.overrides(args)}
    if case.raises is not None:
        with pytest.raises(case.raises) as excinfo:
            contract.call(**kwargs)
        if case.match is not None:
            assert re.search(case.match, str(excinfo.value)), str(excinfo.value)
        return
    try:
        out = contract.call(**kwargs)
    except TypeCheckError as error:
        raise AssertionError(f"the signature rejected an accepted shape: {error}")
    except Exception:
        if case.body_may_raise:
            return
        raise
    if case.check is not None:
        case.check(out, control)


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.kernel)
def test_signature_contract(contract: Contract) -> None:
    """The control goes through; every rejection is refused; every acceptance is not.

    Parameters
    ----------
    contract : Contract
        One kernel's row.
    """
    args = contract.args()
    control = None
    if contract.control == "run":
        control = contract.call(**args)
        if contract.check is not None:
            contract.check(control, None)
    elif contract.control == "signature":
        try:
            contract.call(**args)
        except TypeCheckError as error:
            raise AssertionError(f"the valid call was rejected: {error}")
        except Exception:
            pass
    for case in contract.cases:
        try:
            _run_case(contract, case, args, control)
        except (Exception, pytest.fail.Exception) as error:
            raise AssertionError(f"{contract.kernel}: {case.label}") from error


def test_every_kernel_has_one_row():
    """The ids are unique, so a failure names exactly one kernel."""
    ids = [contract.kernel for contract in CONTRACTS]
    assert len(ids) == len(set(ids))

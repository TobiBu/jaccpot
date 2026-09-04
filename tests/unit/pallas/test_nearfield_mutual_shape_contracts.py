"""What the mutual leaf-pair entry points now reject, and what they must keep taking.

`bench/annotation_pilot` measured this module accepting 99 of 226 shape
perturbations in silence -- the highest rate of the six Phase 2 candidates. The
five module-level entry points are where that is closed, and these tests pin the
closure rather than the annotation text: each one feeds a malformed block and
asserts a complaint, and feeds the real thing and asserts it still works.

The perturbations are the ones the pilot found ACCEPTED, not a fresh guess. Two
matter more than the rest:

* **The trailing `3`.** `xa` arriving `(pairs, slots, 2)` was accepted by every
  entry point. `_block_tile` reads `a_xyz[0]`, `[1]`, `[2]` out of the caller's
  positions, and JAX clamps an out-of-range index, so a two-component block does
  not raise -- it silently computes the force for `(x, y, y)`. That is the same
  mechanism as the `delta` family in `operators/complex_ops.py`, and it is wrong
  physics that looks plausible.
* **`pairs` and `w` agreeing across the `a` and `b` sides.** The Pallas path pads
  both sides to `_next_pow2` of the *a*-side slot count, so a wider `b` reaches
  `jnp.pad` with a negative width.

`softening_sq` and `g_value` stay bare and are not tested here: they are scalars,
and the reason is recorded in `test_type_annotation_guard.py`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.pallas.nearfield_mutual import (
    _block_tile,
    _block_vjp_tiles,
    _pair_weight_tile,
    mutual_leafpair_block_cvjp,
    mutual_leafpair_block_jax,
    mutual_leafpair_block_pallas,
    mutual_leafpair_block_vjp_pallas,
)

PAIRS, SLOTS, LEVELS = 3, 4, 2


def _blocks(pairs: int = PAIRS, slots: int = SLOTS):
    """Build one valid leaf-pair block set.

    Parameters
    ----------
    pairs : int
        Number of leaf pairs.
    slots : int
        Leaf width.

    Returns
    -------
    dict
        Keyword arguments for any of the entry points, `b`-side included.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    ones2 = jnp.ones((pairs, slots), dtype=dtype)
    return {
        "xa": jnp.zeros((pairs, slots, 3), dtype=dtype),
        "ma": ones2,
        "va_f": ones2,
        "xb": jnp.ones((pairs, slots, 3), dtype=dtype),
        "mb": ones2,
        "vb_f": ones2,
        "rung_a_f": jnp.zeros((pairs, slots), dtype=dtype),
        "rung_b_f": jnp.zeros((pairs, slots), dtype=dtype),
        "level_weights": jnp.ones((LEVELS,), dtype=dtype),
        "softening_sq": jnp.asarray(1e-4, dtype=dtype),
        "g_value": jnp.asarray(1.0, dtype=dtype),
    }


def _call_jax(**kwargs):
    """Call the pure-jnp twin.

    Parameters
    ----------
    **kwargs
        As built by :func:`_blocks`.

    Returns
    -------
    tuple
        ``(F_a, F_b)``.
    """
    return mutual_leafpair_block_jax(**kwargs)


def _call_pallas(**kwargs):
    """Call the Pallas forward in interpret mode.

    Parameters
    ----------
    **kwargs
        As built by :func:`_blocks`.

    Returns
    -------
    tuple
        ``(F_a, F_b)``.
    """
    return mutual_leafpair_block_pallas(**kwargs, interpret=True)


def _call_vjp_pallas(**kwargs):
    """Call the Pallas reverse in interpret mode.

    Parameters
    ----------
    **kwargs
        As built by :func:`_blocks`.

    Returns
    -------
    tuple
        The four cotangent blocks, then the level-weight, softening and G
        cotangents.
    """
    shape = kwargs["xa"].shape
    return mutual_leafpair_block_vjp_pallas(
        **kwargs,
        fa_bar=jnp.ones(shape, dtype=kwargs["xa"].dtype),
        fb_bar=jnp.ones(shape, dtype=kwargs["xa"].dtype),
        interpret=True,
    )


def _call_cvjp(**kwargs):
    """Call the differentiable wrapper, which takes its statics positionally.

    Parameters
    ----------
    **kwargs
        As built by :func:`_blocks`.

    Returns
    -------
    tuple
        ``(F_a, F_b)``.
    """
    return mutual_leafpair_block_cvjp(
        kwargs["xa"],
        kwargs["ma"],
        kwargs["va_f"],
        kwargs["xb"],
        kwargs["mb"],
        kwargs["vb_f"],
        kwargs["rung_a_f"],
        kwargs["rung_b_f"],
        kwargs["level_weights"],
        kwargs["softening_sq"],
        kwargs["g_value"],
        LEVELS,
        False,
        True,
        True,
    )


ENTRY_POINTS = [
    ("jax", _call_jax),
    ("pallas", _call_pallas),
    ("vjp_pallas", _call_vjp_pallas),
    ("cvjp", _call_cvjp),
]
IDS = [name for name, _ in ENTRY_POINTS]


@pytest.mark.parametrize("call", [fn for _, fn in ENTRY_POINTS], ids=IDS)
def test_valid_blocks_still_go_through(call):
    """The control. Every rejection below is worthless if this does not hold."""
    call(**_blocks())


@pytest.mark.parametrize("call", [fn for _, fn in ENTRY_POINTS], ids=IDS)
def test_a_two_component_position_block_is_rejected(call):
    """The silent-wrong-physics case, and the reason this module was annotated.

    `_block_tile` indexes components 0, 1 and 2 out of the block. JAX clamps an
    out-of-range index, so `(pairs, slots, 2)` does not raise anywhere -- it
    returns the force field for `(x, y, y)`.
    """
    blocks = _blocks()
    blocks["xa"] = blocks["xa"][:, :, :2]
    with pytest.raises(TypeCheckError):
        call(**blocks)


@pytest.mark.parametrize("call", [fn for _, fn in ENTRY_POINTS], ids=IDS)
def test_mismatched_pair_counts_between_the_sides_are_rejected(call):
    """`xb` carrying one fewer pair than `xa`, which the pilot saw accepted."""
    blocks = _blocks()
    blocks["xb"] = blocks["xb"][:-1]
    with pytest.raises(TypeCheckError):
        call(**blocks)


@pytest.mark.parametrize("call", [fn for _, fn in ENTRY_POINTS], ids=IDS)
def test_mismatched_slot_widths_between_the_sides_are_rejected(call):
    """A wider `b` side reaches `jnp.pad` with a negative width on the Pallas path.

    The pure-jnp twin would broadcast it instead of complaining, which is exactly
    why the two sides share `w`: the twin exists to mirror the kernel.
    """
    blocks = _blocks()
    for name in ("xb", "mb", "vb_f", "rung_b_f"):
        blocks[name] = blocks[name][:, :-1]
    with pytest.raises(TypeCheckError):
        call(**blocks)


@pytest.mark.parametrize("call", [fn for _, fn in ENTRY_POINTS], ids=IDS)
def test_an_extra_leading_axis_on_the_mass_block_is_rejected(call):
    """The perturbation `_pad_inputs` accepted on all four of its arrays."""
    blocks = _blocks()
    blocks["ma"] = blocks["ma"][None]
    with pytest.raises(TypeCheckError):
        call(**blocks)


def test_a_batched_cotangent_is_rejected_by_the_reverse_kernel():
    """`fa_bar`/`fb_bar` are annotated independently of the forward blocks."""
    blocks = _blocks()
    shape = blocks["xa"].shape
    with pytest.raises(TypeCheckError):
        mutual_leafpair_block_vjp_pallas(
            **blocks,
            fa_bar=jnp.ones((1,) + shape, dtype=blocks["xa"].dtype),
            fb_bar=jnp.ones(shape, dtype=blocks["xa"].dtype),
            interpret=True,
        )


def test_a_disabled_weight_table_is_still_a_one_entry_array():
    """`levels` binds to 1 when weighting is off, and that must keep working.

    `mutual/nearfield.py` substitutes `jnp.ones((1,))` rather than `None` on the
    `custom_vjp` boundary, so `Float[Array, 'levels']` has to accept a length-1
    table. Nothing else in that signature binds `levels`, which is what makes the
    annotation safe rather than merely true today.
    """
    blocks = _blocks()
    blocks["level_weights"] = jnp.ones((1,), dtype=blocks["xa"].dtype)
    mutual_leafpair_block_cvjp(
        blocks["xa"],
        blocks["ma"],
        blocks["va_f"],
        blocks["xb"],
        blocks["mb"],
        blocks["vb_f"],
        blocks["rung_a_f"],
        blocks["rung_b_f"],
        blocks["level_weights"],
        blocks["softening_sq"],
        blocks["g_value"],
        0,
        False,
        True,
        True,
    )


def test_the_twin_and_the_kernel_reject_the_same_shapes():
    """The split that matters: the twin is the oracle, so it must not be laxer.

    A shape the kernel refuses and the twin accepts would make the parity tests
    in `test_custom_vjp_parity.py` compare a real result against a fabricated one.
    """
    blocks = _blocks()
    blocks["xa"] = blocks["xa"][:, :, :2]
    with pytest.raises(TypeCheckError):
        mutual_leafpair_block_jax(**blocks)
    with pytest.raises(TypeCheckError):
        mutual_leafpair_block_pallas(**blocks, interpret=True)


# The three tile helpers below the entry points. `bench/annotation_pilot.py` on a re-record
# of this module put every remaining silent acceptance in them -- 31 of 32, the entry points
# taking the other one -- so these are the contracts that close the module.


def _tile_args(slots: int = SLOTS):
    """Build one valid argument set for `_block_tile`, per-pair and unbatched.

    Parameters
    ----------
    slots : int
        Tile width, i.e. the padded slot count a kernel sees.

    Returns
    -------
    dict
        Keyword arguments for `_block_tile`.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    vec = jnp.zeros((slots,), dtype=dtype)
    return {
        "a_xyz": (vec, vec, vec),
        "ma": jnp.ones((slots,), dtype=dtype),
        "va_f": jnp.ones((slots,), dtype=dtype),
        "b_xyz": (vec + 1.0, vec + 1.0, vec + 1.0),
        "mb": jnp.ones((slots,), dtype=dtype),
        "vb_f": jnp.ones((slots,), dtype=dtype),
        "weight": None,
        "softening_sq": jnp.asarray(1e-4, dtype=dtype),
        "g_value": jnp.asarray(1.0, dtype=dtype),
        "exclude_diagonal": False,
    }


def test_a_valid_tile_still_goes_through():
    """The control. Every rejection below is worthless without it."""
    fx, fy, fz = _block_tile(**_tile_args())
    assert fx.shape == fy.shape == fz.shape == (SLOTS, SLOTS)


def test_a_rung_narrower_than_the_tile_is_rejected():
    """The mode that matters: a length mismatch between arrays that must agree.

    `_pair_weight_tile` accepted a rung array one entry shorter than the tile and
    broadcast it, which silently applies the wrong level weight to a column of pairs.
    #297 is the precedent for this class being reachable through a lane's internals even
    when the boundary is contracted.
    """
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    rung = jnp.zeros((SLOTS,), dtype=dtype)
    weights = jnp.ones((LEVELS,), dtype=dtype)
    # Non-vacuity: the matched call works.
    assert _pair_weight_tile(rung, rung, weights, LEVELS).shape == (SLOTS, SLOTS)

    with pytest.raises(TypeCheckError):
        _pair_weight_tile(rung[:-1], rung, weights, LEVELS)
    with pytest.raises(TypeCheckError):
        _pair_weight_tile(rung, rung[:-1], weights, LEVELS)


def test_a_batched_tile_argument_is_rejected():
    """A tile is per-pair. An extra leading axis means a caller skipped the vmap."""
    args = _tile_args()
    args["ma"] = args["ma"][None]
    with pytest.raises(TypeCheckError):
        _block_tile(**args)


def test_an_array_inside_the_coordinate_tuple_is_checked_too():
    """`a_xyz` is three separate component vectors, and each is checked by path."""
    args = _tile_args()
    ax, ay, az = args["a_xyz"]
    args["a_xyz"] = (ax[:-1], ay, az)
    with pytest.raises(TypeCheckError):
        _block_tile(**args)


def test_the_weight_tile_must_be_square_in_the_tile_width():
    """`weight` comes from `_pair_weight_tile` and shares both axes with the tile."""
    args = _tile_args()
    args["weight"] = jnp.ones((SLOTS, SLOTS - 1), dtype=args["ma"].dtype)
    with pytest.raises(TypeCheckError):
        _block_tile(**args)


def test_the_reverse_tile_checks_its_cotangents():
    """`_block_vjp_tiles` took 16 of the 31 acceptances; its cotangents are per-pair."""
    args = _tile_args()
    vec = args["ma"]
    args["fa_bar_xyz"] = (vec, vec, vec)
    args["fb_bar_xyz"] = (vec, vec, vec)
    # Non-vacuity: the matched call works and returns the documented structure --
    # the four block cotangents, then the level-weight (None here: `_tile_args`
    # passes no level ladder), softening and G cotangents.
    dxa, dma, dxb, dmb, dlw, dsoft, dg = _block_vjp_tiles(**args)
    assert dma.shape == dmb.shape == (SLOTS,)
    assert len(dxa) == len(dxb) == 3
    assert dsoft.shape == dg.shape == ()

    args["fb_bar_xyz"] = (vec[None], vec, vec)
    with pytest.raises(TypeCheckError):
        _block_vjp_tiles(**args)

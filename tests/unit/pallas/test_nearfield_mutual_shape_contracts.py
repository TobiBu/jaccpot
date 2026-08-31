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
        The four cotangent blocks.
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

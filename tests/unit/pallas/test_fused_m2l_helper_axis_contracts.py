"""Axis contracts for the fused complex-M2L inner helpers.

`m2l_complex_fused.py` carries real and imaginary parts as SEPARATE real arrays rather
than as a complex dtype, so every reduction in it is written twice and the two halves
have to agree by hand. These three helpers are where that agreement lives, and all
three reduce with an explicit broadcast rather than a matmul::

    _matvec(mat, vec)       jnp.sum(mat * vec[None, :], axis=1)
    _matvec_T(mat, vec)     jnp.sum(mat * vec[:, None], axis=0)
    _block_matmul(...)      jnp.sum(block_r * vec_r[:, None, :], axis=-1) - ...

A broadcast accepts a length-1 operand and spreads it, which is why a wrong shape here
is silent rather than loud. Measured on `origin/main` before these annotations: a
length-1 `vec` into a (32, 16) operator returned a full (32,) result, and a `vec_i` of
(4, 1) beside a `vec_r` of (4, 8) returned a full (4, 8) -- the imaginary part of every
column silently equal to the imaginary part of the first.

The axes come from the 2026-09-07 recording (`tests/unit` + `tests/integration`), read
per CALL so a relation is taken from the pairing and not from two sets of extents:

    _matvec        mat (16, 32) vec (32,)  |  mat (32, 16) vec (16,)
                   -> `vec` is `mat`'s SECOND axis, at two extents with the roles
                      swapped, so `rows` and `cols` are independent and not a
                      coincidence of one payload size
    _matvec_T      mat (16, 32) vec (16,)  |  mat (32, 16) vec (32,)
                   mat (32, 128) vec (32,) |  mat (128, 32) vec (128,)
                   -> `vec` is `mat`'s FIRST axis, at four extents
    _block_matmul  block (4, 8, 8) vec (4, 8)

`_block_matmul`'s own recording has ONE extent, so squareness is not established from
it alone; its reverse twin `_block_matmul_vjp` recorded (4, 8, 8) and (8, 16, 16) for
the same operator, and `_m2l_one` the same two, which is where the second extent comes
from. `_block_matmul_vjp` itself is deliberately left BARE: it rejected all six of the
pilot's perturbations, so section 4.1 says leave it alone.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.pallas.m2l_complex_fused import _block_matmul, _matvec, _matvec_T

ROWS, COLS = 32, 16
DEGREES, BLOCKDIM = 4, 8


def _mat(rows=ROWS, cols=COLS):
    """A non-square dense operator, so `rows` and `cols` cannot be confused.

    Parameters
    ----------
    rows : int
        Output extent.
    cols : int
        Input extent -- the length `_matvec` reduces against.

    Returns
    -------
    Array
        Shape ``(rows, cols)``, all ones.
    """
    return jnp.ones((rows, cols))


def test_the_matched_calls_still_go_through():
    """The control: both helpers and the block matmul accept their recorded shapes."""
    assert _matvec(_mat(), jnp.ones((COLS,))).shape == (ROWS,)
    assert _matvec_T(_mat(), jnp.ones((ROWS,))).shape == (COLS,)
    out_r, out_i = _block_matmul(
        jnp.ones((DEGREES, BLOCKDIM, BLOCKDIM)),
        jnp.ones((DEGREES, BLOCKDIM, BLOCKDIM)),
        jnp.ones((DEGREES, BLOCKDIM)),
        jnp.ones((DEGREES, BLOCKDIM)),
    )
    assert out_r.shape == out_i.shape == (DEGREES, BLOCKDIM)


@pytest.mark.parametrize("fn", [_matvec, _matvec_T])
def test_a_length_one_vector_is_no_longer_broadcast(fn):
    """One column spread across all of them is the silent failure this closes.

    `jnp.sum(mat * vec[None, :], axis=1)` is happy to broadcast a (1,) operand, and on
    `main` both helpers returned a full-length result from it.

    Parameters
    ----------
    fn : Callable
        `_matvec` or its adjoint; the broadcast is silent in both directions.
    """
    with pytest.raises(TypeCheckError):
        fn(_mat(), jnp.ones((1,)))


def test_matvec_reduces_against_cols_and_its_adjoint_against_rows():
    """The one difference between the pair, asserted on a NON-square operator.

    GREEN ON MAIN, deliberately, and it is the only test here that is. Swapping the
    pair does not need an annotation to be caught: a (32,) vector will not broadcast
    against a 16-long axis, so `main` already raises -- as `TypeError` rather than
    `TypeCheckError`, which is the only thing these annotations change about it. This
    test exists to pin the two axis NAMES to the right arguments, so it accepts either
    exception; asserting `TypeCheckError` alone would make an exception-type change
    look like a closed gap.

    On a square operator the swap is shape-identical and no annotation can see it,
    which is why this uses (32, 16).
    """
    with pytest.raises((TypeCheckError, TypeError)):
        _matvec(_mat(), jnp.ones((ROWS,)))  # wants COLS
    with pytest.raises((TypeCheckError, TypeError)):
        _matvec_T(_mat(), jnp.ones((COLS,)))  # wants ROWS


def test_an_extra_leading_axis_is_rejected_on_either_operand():
    """A stray batch axis changed the RANK of the result instead of raising.

    Measured on `main`: `_matvec(mat[1, 32, 16], vec[16])` returned (1, 16), and
    `_matvec(mat[32, 16], vec[1, 16])` returned (1, 16) -- neither the (32,) the
    caller's next reshape expects.
    """
    with pytest.raises(TypeCheckError):
        _matvec(jnp.ones((1, ROWS, COLS)), jnp.ones((COLS,)))
    with pytest.raises(TypeCheckError):
        _matvec(_mat(), jnp.ones((1, COLS)))


def test_a_flattened_operator_is_rejected():
    """`_matvec_T` took a flat (512,) operator and returned (512,).

    The rank check is the whole of it: with `mat` flat, `vec[:, None]` broadcasts
    against it and the reduction runs over the wrong axis entirely.
    """
    with pytest.raises(TypeCheckError):
        _matvec_T(jnp.ones((ROWS * COLS,)), jnp.ones((ROWS,)))


def test_the_real_and_imaginary_halves_must_agree():
    """The dangerous one: a real/imag mismatch inside a complex kernel.

    `_block_matmul` computes `sum(block_r * vec_r) - sum(block_i * vec_i)`. Both terms
    are reduced independently, so a `vec_i` of (4, 1) beside a `vec_r` of (4, 8)
    produced two same-shaped arrays and a plausible (4, 8) result carrying the first
    column's imaginary part in every lane. The pilot never tried this one -- its
    perturbations are leading/trailing/extra/flattened, not length-1 -- so it is a gap
    the annotation closes beyond what was measured.
    """
    blocks = jnp.ones((DEGREES, BLOCKDIM, BLOCKDIM))
    with pytest.raises(TypeCheckError):
        _block_matmul(
            blocks, blocks, jnp.ones((DEGREES, BLOCKDIM)), jnp.ones((DEGREES, 1))
        )
    with pytest.raises(TypeCheckError):
        _block_matmul(
            jnp.ones((1, DEGREES, BLOCKDIM, BLOCKDIM)),
            blocks,
            jnp.ones((DEGREES, BLOCKDIM)),
            jnp.ones((DEGREES, BLOCKDIM)),
        )


def test_the_free_axes_stay_free():
    """`rows` in `_matvec` and `cols` in `_matvec_T` are bound by ONE parameter.

    Nothing else in either signature carries them, so a (31, 16) operator with a (16,)
    vector is a perfectly well-formed matvec and is accepted. Two of the pilot's six
    acceptances on this pair were exactly that, and they are NOT defects -- the tool
    perturbed a free axis. Asserted so that a later change does not "close" them by
    cross-binding an axis the evidence does not support.
    """
    assert _matvec(_mat(rows=ROWS - 1), jnp.ones((COLS,))).shape == (ROWS - 1,)
    assert _matvec_T(_mat(cols=COLS - 1), jnp.ones((ROWS,))).shape == (COLS - 1,)

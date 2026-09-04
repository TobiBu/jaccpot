"""Shape contracts for `_evaluate`'s three masked scatter-add helpers.

`bench/annotation_pilot.py` recorded `runtime/kernels/_evaluate.py` on 2026-09-04 at 164
silent acceptances of 299 perturbations. 108 of those are leaves inside NamedTuple
containers, which no annotation this toolchain supports can reach, so the closable surface
is 58 -- and 24 of the 58 are these three near-duplicate functions.

The mode that matters is the mask. `jnp.where(flat_mask[:, None], flat_values, zero)`
BROADCASTS a mask of the wrong length rather than refusing it, so a mask that disagrees with
its values scatters the wrong slots into the accumulator and the answer is a plausible wrong
force rather than an error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.runtime.kernels._evaluate import (
    _scatter_rank3,
    _scatter_scalars,
    _scatter_vectors,
)

LEAVES, W = 4, 8
N = LEAVES * W


def _dtype():
    """Return the working float dtype for the current x64 setting.

    Returns
    -------
    numpy.dtype
        `float64` under `JAX_ENABLE_X64`, else `float32`.
    """
    return jnp.zeros(
        (), dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    ).dtype


def _common():
    """Build the `indices`/`mask` pair every helper shares.

    Returns
    -------
    tuple
        `(indices, mask)`, both `(LEAVES, W)`.
    """
    return (
        jnp.zeros((LEAVES, W), dtype=jnp.int64),
        jnp.ones((LEAVES, W), dtype=bool),
    )


CASES = [
    ("vectors", _scatter_vectors, (N, 3), (LEAVES, W, 3)),
    ("scalars", _scatter_scalars, (N,), (LEAVES, W)),
    ("rank3", _scatter_rank3, (N, 3, 3), (LEAVES, W, 3, 3)),
]
IDS = [name for name, _, _, _ in CASES]


@pytest.mark.parametrize("fn,base_shape,values_shape", [c[1:] for c in CASES], ids=IDS)
def test_a_matched_scatter_still_goes_through(fn, base_shape, values_shape):
    """The control. Every rejection below is worthless if this does not hold."""
    indices, mask = _common()
    base = jnp.zeros(base_shape, dtype=_dtype())
    out = fn(base, indices, jnp.ones(values_shape, dtype=_dtype()), mask)
    assert out.shape == base_shape


@pytest.mark.parametrize("fn,base_shape,values_shape", [c[1:] for c in CASES], ids=IDS)
def test_a_mask_that_disagrees_with_its_values_is_rejected(
    fn, base_shape, values_shape
):
    """The silent case: the mask broadcasts instead of complaining."""
    indices, mask = _common()
    base = jnp.zeros(base_shape, dtype=_dtype())
    values = jnp.ones(values_shape, dtype=_dtype())

    with pytest.raises(TypeCheckError):
        fn(base, indices, values, mask[:-1])
    with pytest.raises(TypeCheckError):
        fn(base, indices, values, mask[:, :-1])


@pytest.mark.parametrize("fn,base_shape,values_shape", [c[1:] for c in CASES], ids=IDS)
def test_indices_that_disagree_with_the_values_are_rejected(
    fn, base_shape, values_shape
):
    """`indices` addresses the same slots, so it shares both axes too."""
    indices, mask = _common()
    base = jnp.zeros(base_shape, dtype=_dtype())
    values = jnp.ones(values_shape, dtype=_dtype())

    with pytest.raises(TypeCheckError):
        fn(base, indices[:-1], values, mask)


def test_the_payload_rank_is_what_separates_the_three():
    """A vector payload handed to the scalar helper is now refused.

    The three differ only in the trailing component dims -- `()`, `(3,)`, `(3, 3)` -- which
    is exactly the kind of near-duplicate family where a wrong call site is invisible.
    """
    indices, mask = _common()
    with pytest.raises(TypeCheckError):
        _scatter_scalars(
            jnp.zeros((N,), dtype=_dtype()),
            indices,
            jnp.ones((LEAVES, W, 3), dtype=_dtype()),
            mask,
        )
    with pytest.raises(TypeCheckError):
        _scatter_vectors(
            jnp.zeros((N, 3), dtype=_dtype()),
            indices,
            jnp.ones((LEAVES, W), dtype=_dtype()),
            mask,
        )


def test_base_keeps_a_free_length_and_that_is_deliberate():
    """`base` is `leaves * w` long, and jaxtyping cannot bind a product.

    So a `base` of the wrong length is still accepted -- pinned here as the CURRENT
    behaviour, not the desirable one, the way `_large_n_blocks.py` pins `block_offsets`.
    Its rank and trailing dims are constrained; only the length is free.
    """
    indices, mask = _common()
    values = jnp.ones((LEAVES, W, 3), dtype=_dtype())
    out = _scatter_vectors(jnp.zeros((N + 5, 3), dtype=_dtype()), indices, values, mask)
    assert out.shape == (N + 5, 3)

    # The rank IS checked, which is what the annotation buys on this parameter.
    with pytest.raises(TypeCheckError):
        _scatter_vectors(jnp.zeros((N,), dtype=_dtype()), indices, values, mask)

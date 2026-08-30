"""Shape contracts for the ``delta``/``direction`` family in ``complex_ops``.

Every test here fails on the commit before the annotations landed, and it fails by
*passing*: the call returns a plausible number instead of raising. That is the
whole reason the family was worth a PR. JAX clamps an out-of-bounds index rather
than raising, so ``delta[2]`` on a length-2 array returns ``delta[1]`` and each of
these functions quietly computes the answer for ``(x, y, y)``.

The first test pins that mechanism directly, so if JAX ever changes it the reason
these annotations exist is re-measured rather than assumed.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaccpot.operators.complex_ops import (
    evaluate_local_complex,
    evaluate_local_complex_derivative_tower,
    evaluate_local_complex_grad_analytic,
    evaluate_local_complex_with_grad,
    evaluate_local_complex_with_grad_analytic,
    regular_solid_harmonic_directional_derivative,
    regular_solid_harmonic_directional_derivative_order,
    rotate_complex_local_from_z_solidfmm,
    rotate_complex_local_to_z_solidfmm,
    rotate_complex_multipole_from_z_solidfmm,
    rotate_complex_multipole_to_z_solidfmm,
)
from jaccpot.operators.real_harmonics import sh_size

ORDER = 4
DELTA = jnp.asarray([0.35, -0.25, 0.45])
DIRECTION = jnp.asarray([0.31, -0.44, 0.21])


def _local(order: int = ORDER) -> jnp.ndarray:
    """Return a deterministic packed complex expansion.

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    jnp.ndarray
        Packed complex coefficients, length ``sh_size(order)``.
    """
    rng = np.random.default_rng(0)
    size = sh_size(order)
    return jnp.asarray(rng.normal(size=size) + 1j * rng.normal(size=size))


def _call(name: str, delta: jnp.ndarray):
    """Invoke one member of the family with ``delta``.

    Collected here rather than parametrized inline because the eleven signatures
    differ in argument order and in which keywords they require, and the point of
    the test is the shared ``delta`` contract, not those differences.

    Parameters
    ----------
    name : str
        Which family member to call.
    delta : jnp.ndarray
        The displacement to pass.

    Returns
    -------
    Any
        Whatever the named function returns.

    Raises
    ------
    KeyError
        If ``name`` is not a member of the family.
    """
    local = _local()
    table = {
        "evaluate_local_complex": lambda: evaluate_local_complex(
            local, delta, order=ORDER
        ),
        "evaluate_local_complex_with_grad": lambda: evaluate_local_complex_with_grad(
            local, delta, order=ORDER
        ),
        "evaluate_local_complex_derivative_tower": (
            lambda: evaluate_local_complex_derivative_tower(
                local, delta, order=ORDER, max_derivative_order=1
            )
        ),
        "evaluate_local_complex_with_grad_analytic": (
            lambda: evaluate_local_complex_with_grad_analytic(local, delta, order=ORDER)
        ),
        "evaluate_local_complex_grad_analytic": (
            lambda: evaluate_local_complex_grad_analytic(local, delta, order=ORDER)
        ),
        "regular_solid_harmonic_directional_derivative": (
            lambda: regular_solid_harmonic_directional_derivative(
                delta, DIRECTION[: delta.shape[0]], order=ORDER
            )
        ),
        "regular_solid_harmonic_directional_derivative_order": (
            lambda: regular_solid_harmonic_directional_derivative_order(
                delta, DIRECTION[: delta.shape[0]], order=ORDER, derivative_order=2
            )
        ),
        "rotate_complex_multipole_to_z_solidfmm": (
            lambda: rotate_complex_multipole_to_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_multipole_from_z_solidfmm": (
            lambda: rotate_complex_multipole_from_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_local_to_z_solidfmm": (
            lambda: rotate_complex_local_to_z_solidfmm(local, delta, order=ORDER)
        ),
        "rotate_complex_local_from_z_solidfmm": (
            lambda: rotate_complex_local_from_z_solidfmm(local, delta, order=ORDER)
        ),
    }
    return table[name]()


FAMILY = (
    "evaluate_local_complex",
    "evaluate_local_complex_with_grad",
    "evaluate_local_complex_derivative_tower",
    "evaluate_local_complex_with_grad_analytic",
    "evaluate_local_complex_grad_analytic",
    "regular_solid_harmonic_directional_derivative",
    "regular_solid_harmonic_directional_derivative_order",
    "rotate_complex_multipole_to_z_solidfmm",
    "rotate_complex_multipole_from_z_solidfmm",
    "rotate_complex_local_to_z_solidfmm",
    "rotate_complex_local_from_z_solidfmm",
)


def test_short_delta_is_silently_the_wrong_physics_without_the_contract() -> None:
    """Pin the mechanism that makes the whole family worth annotating.

    A length-2 ``delta`` does not raise inside JAX; it is read as ``(x, y, y)``.
    This asserts that equality on the raw harmonics, so the annotations below rest
    on a measured fact rather than on a claim about indexing.
    """
    from jaccpot.operators.complex_harmonics import complex_R_solidfmm

    short = complex_R_solidfmm(DELTA[:2], order=ORDER)
    clamped = complex_R_solidfmm(
        jnp.asarray([DELTA[0], DELTA[1], DELTA[1]]), order=ORDER
    )
    full = complex_R_solidfmm(DELTA, order=ORDER)

    assert jnp.allclose(short, clamped), "short delta is not read as (x, y, y)"
    assert not jnp.allclose(short, full), "the truncation has to change the answer"


@pytest.mark.parametrize("name", FAMILY)
def test_length_two_delta_is_rejected(name: str) -> None:
    """A truncated spatial vector must raise rather than return a number.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    with pytest.raises(TypeCheckError):
        _call(name, DELTA[:2])


@pytest.mark.parametrize("name", FAMILY)
def test_full_length_delta_still_works(name: str) -> None:
    """The contract must not reject the calls these functions have always taken.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    assert _call(name, DELTA) is not None


@pytest.mark.parametrize("name", FAMILY)
def test_batched_delta_is_rejected(name: str) -> None:
    """A ``(batch, 3)`` delta belongs to the ``*_batch`` variants, not to these.

    Passing one here used to broadcast into a result of the wrong rank rather than
    raising, which is the same failure wearing a different shape.

    Parameters
    ----------
    name : str
        Family member under test.
    """
    with pytest.raises(TypeCheckError):
        _call(name, jnp.stack([DELTA, DELTA]))


def test_direction_is_shaped_independently_of_delta() -> None:
    """``direction`` carries its own contract, not one inherited from ``delta``.

    Both directional-derivative functions take two spatial vectors, so a test that
    only ever truncated ``delta`` would leave the second annotation unexercised --
    it would pass with ``direction`` still bare.
    """
    with pytest.raises(TypeCheckError):
        regular_solid_harmonic_directional_derivative(DELTA, DIRECTION[:2], order=ORDER)
    with pytest.raises(TypeCheckError):
        regular_solid_harmonic_directional_derivative_order(
            DELTA, DIRECTION[:2], order=ORDER, derivative_order=2
        )


def test_batch_variants_keep_taking_batched_deltas() -> None:
    """The ``*_batch`` siblings are deliberately NOT annotated ``'3'``.

    They take ``(batch, 3)`` and ``vmap`` the scalar function over it, so the
    per-example slice the annotation guards is exactly what they feed in. This
    pins that the PR left them alone; annotating them ``'3'`` would break every
    batched caller, and the assertion is cheap insurance against a later sweep
    doing it by pattern-match.
    """
    from jaccpot.operators.complex_ops import (
        evaluate_local_complex_grad_analytic_batch,
        regular_solid_harmonic_directional_derivative_batch,
    )

    deltas = jnp.stack([DELTA, DELTA + 0.1, DELTA - 0.2])
    grads = evaluate_local_complex_grad_analytic_batch(_local(), deltas, order=ORDER)
    assert grads.shape == (3, 3)

    packed = regular_solid_harmonic_directional_derivative_batch(
        deltas, jnp.stack([DIRECTION] * 3), order=ORDER
    )
    assert packed.shape[0] == 3

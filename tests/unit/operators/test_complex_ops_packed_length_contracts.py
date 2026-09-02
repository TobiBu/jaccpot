"""Packed-length contracts for the z-translation family in ``complex_ops``.

These tests fail on the commit before the length guard landed, and -- like their
sibling ``test_complex_ops_vec3_contracts.py`` -- they fail by *passing*: the call
returns a plausible array instead of raising. The mechanism is the same one the
module note records for ``delta[2]``. JAX **clamps** an out-of-range index rather
than raising, so ``multipole[24]`` on a 24-entry buffer returns ``multipole[23]``
and the caller gets a wrong answer with no error at all.

A shape annotation cannot express this contract: the required length is
``sh_size(order)``, and ``order`` is a static Python int rather than an array
parameter, so there is no axis for a jaxtyping spec to bind to. Hence an explicit
check, and hence these tests.

TOO LONG MUST KEEP WORKING. The module note is explicit that the bodies tolerate a
longer buffer and that callers rely on it, so every raising test here has a
longer-buffer twin asserting the guard did not become an equality check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.complex_ops import (
    evaluate_local_complex,
    evaluate_local_complex_grad_order4_unrolled,
    translate_along_z_l2l_complex,
    translate_along_z_l2l_complex_batch,
    translate_along_z_m2l_complex,
    translate_along_z_m2l_complex_batch,
    translate_along_z_m2m_complex,
    translate_along_z_m2m_complex_batch,
    translate_along_z_m2m_complex_solidfmm,
)
from jaccpot.operators.real_harmonics import sh_size

ORDER = 4
NCOEFF = sh_size(ORDER)
DELTA = jnp.asarray([0.35, -0.25, 0.45])

# (function, scalar argument, buffer parameter name) for the four single-expansion
# entry points. The scalar is a separation for M2L (it divides by ``r ** (n + k + 1)``,
# so it must stay away from zero) and a signed translation distance for the rest.
#
# The parameter name is spelled out rather than read back off ``fn.__code__``: under
# ``JACCPOT_RUNTIME_TYPECHECK=1`` these functions are wrapped, so introspection reports
# the wrapper's ``args`` and the assertion passes vacuously in the default run while
# failing in the typechecked one.
SINGLE_CASES = [
    pytest.param(translate_along_z_m2l_complex, 2.5, "multipole", id="m2l"),
    pytest.param(translate_along_z_m2m_complex, 0.5, "multipole", id="m2m"),
    pytest.param(
        translate_along_z_m2m_complex_solidfmm, 0.5, "multipole", id="m2m_solidfmm"
    ),
    pytest.param(translate_along_z_l2l_complex, 0.5, "local", id="l2l"),
]

# (batch function, scalar argument) -- these delegate through ``jax.vmap`` into the
# single-expansion functions above, so they inherit the guard rather than repeating
# it. That inheritance is what ``test_batch_*`` below pins.
BATCH_CASES = [
    pytest.param(translate_along_z_m2l_complex_batch, 2.5, id="m2l_batch"),
    pytest.param(translate_along_z_m2m_complex_batch, 0.5, id="m2m_batch"),
    pytest.param(translate_along_z_l2l_complex_batch, 0.5, id="l2l_batch"),
]


def _coeffs(size: int, seed: int) -> jnp.ndarray:
    """Return a deterministic packed complex buffer of the requested length.

    Parameters
    ----------
    size : int
        Number of coefficients to generate.
    seed : int
        Seed for the deterministic generator.

    Returns
    -------
    jnp.ndarray
        Complex coefficients of length ``size``.
    """
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=size) + 1j * rng.normal(size=size))


def test_jax_still_clamps_an_out_of_range_index() -> None:
    """Pin the mechanism the guard exists for, so a JAX change re-measures it."""
    buf = jnp.arange(4.0)

    assert float(buf[10]) == float(buf[3])
    assert buf[:10].shape == (4,)


@pytest.mark.parametrize("fn, scalar, param_name", SINGLE_CASES)
def test_single_rejects_a_too_short_buffer(fn, scalar, param_name) -> None:
    short = _coeffs(NCOEFF - 1, seed=0)

    with pytest.raises(ValueError, match=r"got length 24, need at least 25"):
        fn(short, jnp.asarray(scalar), order=ORDER)


@pytest.mark.parametrize("fn, scalar, param_name", SINGLE_CASES)
def test_single_message_names_the_parameter(fn, scalar, param_name) -> None:
    """The whole point of the guard is that the old failure was silent."""
    short = _coeffs(NCOEFF - 1, seed=1)

    with pytest.raises(ValueError) as excinfo:
        fn(short, jnp.asarray(scalar), order=ORDER)

    message = str(excinfo.value)
    assert message.startswith(param_name)
    assert "order 4" in message


@pytest.mark.parametrize("fn, scalar, param_name", SINGLE_CASES)
def test_single_still_accepts_a_longer_buffer(fn, scalar, param_name) -> None:
    """Load-bearing: the bodies slice, and callers pass over-long buffers."""
    exact = _coeffs(NCOEFF, seed=2)
    padding = jnp.asarray([3.0 + 1.0j, -2.0 + 0.5j, 7.0 - 4.0j])
    longer = jnp.concatenate([exact, padding])

    from_exact = fn(exact, jnp.asarray(scalar), order=ORDER)
    from_longer = fn(longer, jnp.asarray(scalar), order=ORDER)

    assert from_longer.shape == (NCOEFF,)
    assert np.allclose(np.asarray(from_longer), np.asarray(from_exact))


@pytest.mark.parametrize("fn, scalar", BATCH_CASES)
def test_batch_rejects_a_short_trailing_axis(fn, scalar) -> None:
    batch = 3
    short = _coeffs(batch * (NCOEFF - 1), seed=3).reshape(batch, NCOEFF - 1)
    scalars = jnp.full((batch,), scalar)

    with pytest.raises(ValueError, match=r"got length 24, need at least 25"):
        fn(short, scalars, order=ORDER)


@pytest.mark.parametrize("fn, scalar", BATCH_CASES)
def test_batch_still_accepts_a_longer_trailing_axis(fn, scalar) -> None:
    batch = 3
    exact = _coeffs(batch * NCOEFF, seed=4).reshape(batch, NCOEFF)
    longer = jnp.concatenate([exact, jnp.ones((batch, 2), dtype=exact.dtype)], axis=1)
    scalars = jnp.full((batch,), scalar)

    from_exact = fn(exact, scalars, order=ORDER)
    from_longer = fn(longer, scalars, order=ORDER)

    assert from_longer.shape == (batch, NCOEFF)
    assert np.allclose(np.asarray(from_longer), np.asarray(from_exact))


def test_guard_fires_at_every_order() -> None:
    """``sh_size(order)`` is the requirement, not a hard-coded 25."""
    for order in (0, 1, 3, 6):
        ncoeff = sh_size(order)
        short = _coeffs(max(ncoeff - 1, 0), seed=order)
        if ncoeff == 1:
            # Order 0 needs one coefficient; an empty buffer is the short case.
            assert short.shape == (0,)
        with pytest.raises(ValueError, match=rf"need at least {ncoeff}"):
            translate_along_z_m2m_complex(short, jnp.asarray(0.5), order=order)


def test_order4_unrolled_evaluator_rejects_a_too_short_buffer() -> None:
    """The one evaluator the module note was over-general about.

    The note claims a too-short buffer "already raises a domain error from the
    body (``TypeError`` from the evaluators)". That holds for every evaluator that
    reaches :func:`complex_dot`, which slices both operands and then multiplies
    mismatched lengths. It did **not** hold here: this one indexes scalar-wise,
    ``local_coeffs[ridx(n, m)]``, so it clamped like the z-translation family and
    returned a wrong gradient in silence.
    """
    short = _coeffs(NCOEFF - 1, seed=5)

    with pytest.raises(ValueError, match=r"got length 24, need at least 25"):
        evaluate_local_complex_grad_order4_unrolled(short, DELTA, order=ORDER)


def test_order4_unrolled_evaluator_still_accepts_a_longer_buffer() -> None:
    exact = _coeffs(NCOEFF, seed=6)
    longer = jnp.concatenate([exact, jnp.asarray([5.0 + 0.0j, -1.0 + 2.0j])])

    from_exact = evaluate_local_complex_grad_order4_unrolled(exact, DELTA, order=ORDER)
    from_longer = evaluate_local_complex_grad_order4_unrolled(
        longer, DELTA, order=ORDER
    )

    assert np.allclose(np.asarray(from_longer), np.asarray(from_exact))


def test_complex_dot_evaluators_still_raise_on_their_own() -> None:
    """The half of the module note's claim that does hold, pinned so it stays true.

    No guard was added to these -- :func:`complex_dot` slices both operands to
    ``ncoeff`` and then multiplies, so a short ``local`` meets a full-length
    ``regular`` and broadcasting refuses. The note is accurate for them.
    """
    short = _coeffs(NCOEFF - 1, seed=7)

    with pytest.raises(TypeError, match=r"incompatible shapes for broadcasting"):
        evaluate_local_complex(short, DELTA, order=ORDER)

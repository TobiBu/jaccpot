"""Probe the new guard's cases before implementing it."""

import jax.numpy as jnp
import pytest

from jaccpot.downward.local_expansions import (
    total_coefficients,
    translate_multipole_to_local,
)

DELTA = jnp.array([1.0, 0.5, -0.25])


def _moments(order):
    m = dict(
        raw_mass=jnp.asarray(1.0),
        raw_dipole=jnp.zeros((3,)),
    )
    if order >= 2:
        m["raw_second"] = jnp.zeros((3, 3))
    if order >= 3:
        m["raw_third"] = jnp.zeros((3, 3, 3))
    if order >= 4:
        m["raw_fourth"] = jnp.zeros((3, 3, 3, 3))
    return m


def _multipole(order):
    return jnp.ones((total_coefficients(order),))


@pytest.mark.parametrize("order", [2, 3, 4])
def test_all_raw_moments_supplied_still_works(order):
    out = translate_multipole_to_local(
        _multipole(order), DELTA, order=order, **_moments(order)
    )
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_no_raw_moments_uses_the_fallback(order):
    out = translate_multipole_to_local(_multipole(order), DELTA, order=order)
    assert jnp.all(jnp.isfinite(out))


def test_order_two_does_not_require_third_or_fourth():
    """The guard must key on `order`, not demand every moment unconditionally."""
    out = translate_multipole_to_local(_multipole(2), DELTA, order=2, **_moments(2))
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize(
    "order,missing", [(2, "raw_second"), (3, "raw_third"), (4, "raw_fourth")]
)
def test_partial_raw_moments_raise_a_clear_error(order, missing):
    kwargs = _moments(order)
    kwargs.pop(missing)
    with pytest.raises(ValueError, match=missing):
        translate_multipole_to_local(_multipole(order), DELTA, order=order, **kwargs)

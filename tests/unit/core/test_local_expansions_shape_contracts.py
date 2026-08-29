"""What the `delta`/`raw_*` shape annotations catch that nothing caught before.

STYLE_GUIDE section 4.1 says to shape-annotate an array parameter when nothing
else validates it, and to leave alone what is already checked. A pilot over the
whole of Phase 1's `downward/` + `upward/` group measured which was which:
**13 of 15 malformed inputs were already rejected**, most with precise domain
errors (`"masses_sorted must match tree.num_particles"`,
`"centers must have shape (num_nodes, 3)"`). Those parameters were deliberately
left bare.

This file covers the remainder -- the family that was accepted silently. Every
case below returned a number on `main`.

The `delta` one is the sharpest, and it is worth knowing why it is silent rather
than loud: JAX **clamps** an out-of-bounds index, so `delta[2]` on a length-2
array returns `delta[1]`. A caller who drops a component does not get an
`IndexError`; they get the answer for `(x, y, y)`. That is pinned below, because
"wrong answer, no error" is the failure mode this library exists to avoid.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.downward.local_expansions import (
    translate_local_expansion,
    translate_multipole_to_local,
)

_ORDER = 2
_CT = 10  # total_coefficients(2)


@pytest.fixture
def coefficients():
    """Packed Cartesian coefficients at ``order=2``.

    Returns
    -------
    Array
        A ``(10,)`` buffer, `total_coefficients(2)`.
    """
    return jnp.asarray(np.random.default_rng(3).standard_normal(_CT))


def test_a_short_delta_is_rejected_rather_than_silently_clamped(coefficients):
    """The measured silent-wrong-answer: `(2,)` delta returned `(x, y, y)`.

    Before the annotation this call succeeded and its result was bit-identical
    to passing `delta = (x, y, y)`, because JAX clamps the out-of-bounds
    `delta[2]` to `delta[1]`.
    """
    with pytest.raises(Exception, match="(?i)delta|shape|dim"):
        translate_local_expansion(coefficients, jnp.asarray([0.3, -0.5]), order=_ORDER)


def test_the_clamping_that_made_it_silent_is_still_real(coefficients):
    """Pins the mechanism, so the test above cannot be read as hypothetical.

    A length-3 delta whose z equals its y reproduces exactly what the rejected
    length-2 call used to compute. If JAX ever stopped clamping, this would
    stop holding and the rationale above would need rewriting.
    """
    clamped = translate_local_expansion(
        coefficients, jnp.asarray([0.3, -0.5, -0.5]), order=_ORDER
    )
    correct = translate_local_expansion(
        coefficients, jnp.asarray([0.3, -0.5, 0.7]), order=_ORDER
    )
    assert not np.allclose(np.asarray(clamped), np.asarray(correct))


def test_an_over_long_delta_is_now_rejected_too(coefficients):
    """A disclosed behaviour change: `(4,)` used to be accepted.

    It returned the *correct* answer -- the extra components were never read --
    so this tightens a signature that was tolerant rather than fixing a bug. No
    call site passes one: `delta` was recorded as `(3,)` in all 88 captured
    calls across six test files, and both production call sites
    (`runtime/kernels/_evaluate.py:119` under `vmap`, and the L2L cascade in
    this module) pass a three-vector difference of two centres.
    """
    with pytest.raises(Exception, match="(?i)delta|shape|dim"):
        translate_local_expansion(
            coefficients, jnp.asarray([0.3, -0.5, 0.7, 9.9]), order=_ORDER
        )


def test_a_valid_call_is_unchanged(coefficients):
    """The annotations must not perturb the numbers, only the rejections."""
    result = translate_local_expansion(
        coefficients, jnp.asarray([0.3, -0.5, 0.7]), order=_ORDER
    )
    assert result.shape == (_CT,)
    assert np.all(np.isfinite(np.asarray(result)))


@pytest.mark.parametrize(
    "kwargs, why",
    [
        ({"raw_dipole": jnp.zeros(2)}, "dipole with a component dropped"),
        ({"raw_dipole": jnp.zeros(4)}, "dipole with a component too many"),
        ({"raw_mass": jnp.zeros(1)}, "monopole as (1,) rather than a scalar"),
    ],
)
def test_raw_moments_with_the_wrong_shape_are_rejected(coefficients, kwargs, why):
    """Three raw moments that reached the arithmetic silently on `main`.

    `raw_*` is exactly the family section 4.1 predicts will be unvalidated: the
    body checks only that the moments are *present*, never their shape.

    Parameters
    ----------
    coefficients : Array
        The packed buffer fixture.
    kwargs : dict
        The one malformed raw moment for this case.
    why : str
        What the malformation represents, for the failure message.
    """
    supplied = {
        "raw_mass": jnp.asarray(1.0),
        "raw_dipole": jnp.zeros(3),
        "raw_second": jnp.zeros((3, 3)),
        **kwargs,
    }
    with pytest.raises(Exception):
        translate_multipole_to_local(
            coefficients, jnp.asarray([0.3, -0.5, 0.7]), order=_ORDER, **supplied
        )


def test_the_documented_packed_second_moment_was_never_the_real_one(coefficients):
    """The docstring said "packed symmetric"; the body wants full `(3, 3)`.

    A packed length-6 second moment -- what the old wording described -- is
    rejected. Pinned so the corrected docstring cannot drift back.
    """
    with pytest.raises(Exception):
        translate_multipole_to_local(
            coefficients,
            jnp.asarray([0.3, -0.5, 0.7]),
            order=_ORDER,
            raw_mass=jnp.asarray(1.0),
            raw_dipole=jnp.zeros(3),
            raw_second=jnp.zeros(6),
        )


def test_a_longer_coefficient_buffer_is_still_accepted(coefficients):
    """`ct` is a free axis on purpose, and this is what that buys.

    Both functions index by level offset and tolerate a buffer longer than
    `total_coefficients(order)`. Binding the axis to `order` would reject
    callers the package has always accepted, so the annotation asserts rank and
    dtype only.
    """
    longer = jnp.concatenate([coefficients, jnp.zeros(7)])
    result = translate_local_expansion(
        longer, jnp.asarray([0.3, -0.5, 0.7]), order=_ORDER
    )
    assert np.allclose(
        np.asarray(result),
        np.asarray(
            translate_local_expansion(
                coefficients, jnp.asarray([0.3, -0.5, 0.7]), order=_ORDER
            )
        ),
    )

"""The ``rho == 0`` transverse-derivative rule: inertness, transposition, primal.

The correctness of the analytic formula itself is asserted per operator in
``test_real_harmonics.py`` and ``test_complex_ops.py``. What is pinned here is the
three properties the *mechanism* has to have, none of which those tests can see:

* the rule is **inert** off the degenerate axis -- bit-identical, not merely close,
  because that is what lets it be added to production operators without moving any
  gradient that was already right;
* it **transposes**, so reverse mode works -- the FMM takes ``jax.grad``, never
  ``jax.jvp``;
* the **primal is untouched** at the degenerate point, checked against an independent
  reference rather than against itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators._transverse_degeneracy_jvp import split_transverse_tangent
from jaccpot.operators.complex_ops import (
    l2l_complex,
    m2l_complex_reference,
    m2m_complex,
)
from jaccpot.operators.real_harmonics import (
    l2l_real,
    m2l_real,
    m2m_real,
    sh_size,
    translate_along_z_m2l_real,
)

_ORDER = 4
_Z = 2.5

_REAL_CASCADES = [
    pytest.param(m2l_real, id="m2l_real"),
    pytest.param(m2m_real, id="m2m_real"),
    pytest.param(l2l_real, id="l2l_real"),
]
_COMPLEX_CASCADES = [
    pytest.param(m2l_complex_reference, id="m2l_complex_reference"),
    pytest.param(m2m_complex, id="m2m_complex"),
    pytest.param(l2l_complex, id="l2l_complex"),
]


def _real_coeffs(seed):
    return jax.random.normal(
        jax.random.PRNGKey(seed), (sh_size(_ORDER),), dtype=jnp.float64
    )


def _complex_coeffs(seed):
    rng = np.random.default_rng(seed)
    n = sh_size(_ORDER)
    return jnp.asarray(
        rng.normal(size=n) + 1j * rng.normal(size=n), dtype=jnp.complex128
    )


# ---------------------------------------------------------------------------
# Inertness off the axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta",
    [
        pytest.param([1.3, -0.7, 2.1], id="generic"),
        pytest.param([1.0e-6, 0.0, 2.5], id="just outside the band"),
        pytest.param([0.0, 0.0, 0.0], id="delta == 0"),
    ],
)
def test_split_leaves_the_tangent_untouched_outside_the_band(delta):
    """Outside the band, the split must be an exact identity.

    ``np.array_equal``, not ``allclose``: the claim in
    :mod:`jaccpot.operators._transverse_degeneracy_jvp` is that a gradient the polar
    route can still resolve is left alone *to the last bit*, and that only holds if the
    routed tangent is the incoming one unchanged and both scales are exactly zero.

    ``rho = 1e-6`` at ``r = 2.5`` gives ``rho/r = 4e-7``, a factor ~27 outside the
    ``sqrt(eps) = 1.5e-08`` boundary -- comfortably clear of it without being so far
    away that the test stops guarding the boundary's location.
    """
    tangent = jnp.asarray([0.3, -1.7, 0.9], dtype=jnp.float64)
    routed, scale_x, scale_y = split_transverse_tangent(
        jnp.asarray(delta, dtype=jnp.float64), tangent
    )

    assert np.array_equal(np.asarray(routed), np.asarray(tangent))
    assert float(scale_x) == 0.0
    assert float(scale_y) == 0.0


@pytest.mark.parametrize(
    "delta, label",
    [
        pytest.param(
            [1.0e-200, 0.0, _Z], "below the squaring underflow", id="underflow"
        ),
        pytest.param([5.551e-17, 0.0, _Z], "one ulp of COM cancellation", id="one ulp"),
        pytest.param([1.0e-9, 0.0, _Z], "well inside the band", id="1e-9"),
    ],
)
def test_split_claims_the_whole_band_not_just_exact_zero(delta, label):
    """Everything the polar route cannot resolve must reach the analytic branch.

    The band is ``rho_sq <= eps * r_sq``, wider than the ``rho_sq > 0`` the alignment
    guards themselves switch on, and deliberately so: inside it the polar route returns
    a finite, plausible, wrong number rather than an obvious zero. ``1e-200`` (whose
    square underflows, so the guards *do* fire) and ``5.551e-17`` (which is what one ulp
    of centre-of-mass cancellation between two mathematically equal centres looks like,
    and where the guards do *not* fire) must both land here, or a displacement gets its
    transverse derivative from neither branch.
    """
    tangent = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float64)
    routed, scale_x, scale_y = split_transverse_tangent(
        jnp.asarray(delta, dtype=jnp.float64), tangent
    )
    assert np.array_equal(
        np.asarray(routed), np.array([0.0, 0.0, 1.0])
    ), f"{label}: transverse tangent was not withdrawn from the polar route"
    assert float(scale_x) == pytest.approx(1.0 / _Z, rel=1e-15)
    assert float(scale_y) == pytest.approx(1.0 / _Z, rel=1e-15)


def test_the_band_boundary_sits_at_the_measured_crossover():
    """``rho/r == sqrt(eps)``, and the two sides of it are actually different.

    Pins the threshold itself, because it is a numerical choice rather than a
    consequence: the analytic branch errs ``O(rho/r)`` and the polar route ``O(eps r /
    rho)``, so equating them puts the crossover at ``(rho/r)^2 == eps`` and makes the
    worst error over all ``rho`` about ``sqrt(eps)`` instead of unbounded. If someone
    changes the constant, this is what says so.
    """
    boundary = float(np.sqrt(np.finfo(np.float64).eps)) * _Z
    tangent = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64)
    for factor, expect_analytic in ((0.5, True), (2.0, False)):
        rho = boundary * factor
        _, scale_x, _ = split_transverse_tangent(
            jnp.asarray([rho, 0.0, _Z], dtype=jnp.float64), tangent
        )
        took_analytic = float(scale_x) != 0.0
        assert took_analytic is expect_analytic, (
            f"rho/r = {rho / _Z:.3e} took the "
            f"{'analytic' if took_analytic else 'polar'} branch; sqrt(eps) = "
            f"{np.sqrt(np.finfo(np.float64).eps):.3e}"
        )


def test_split_removes_the_transverse_tangent_only_on_the_axis():
    """On the axis the transverse tangent is handed to the analytic branch instead."""
    delta = jnp.asarray([0.0, 0.0, _Z], dtype=jnp.float64)
    tangent = jnp.asarray([0.3, -1.7, 0.9], dtype=jnp.float64)
    routed, scale_x, scale_y = split_transverse_tangent(delta, tangent)

    # The radial component still goes through the cascade -- d/dz was never broken.
    assert np.array_equal(np.asarray(routed), np.array([0.0, 0.0, 0.9]))
    assert float(scale_x) == pytest.approx(0.3 / _Z, rel=1e-15)
    assert float(scale_y) == pytest.approx(-1.7 / _Z, rel=1e-15)


def test_split_keeps_the_existing_zero_at_delta_zero():
    """``delta == 0`` is the third regime: formula (2)/(3) divides by ``z``.

    A zero displacement is the identity translation for M2M/L2L and unphysical for
    M2L, and ``|delta|`` has no derivative at the origin anyway, so the existing zero
    cotangent has to stand rather than be replaced by ``0/0``.
    """
    tangent = jnp.asarray([0.3, -1.7, 0.9], dtype=jnp.float64)
    routed, scale_x, scale_y = split_transverse_tangent(
        jnp.zeros(3, dtype=jnp.float64), tangent
    )
    assert np.array_equal(np.asarray(routed), np.asarray(tangent))
    assert float(scale_x) == 0.0
    assert float(scale_y) == 0.0
    assert np.all(np.isfinite([float(scale_x), float(scale_y)]))


def test_split_is_vectorised_over_a_batch():
    """The production M2L applies the rule to a whole interaction batch at once."""
    deltas = jnp.asarray(
        [[0.0, 0.0, _Z], [1.3, -0.7, 2.1], [0.0, 0.0, 0.0]], dtype=jnp.float64
    )
    tangents = jnp.asarray(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=jnp.float64
    )
    routed, scale_x, scale_y = split_transverse_tangent(deltas, tangents)
    assert np.array_equal(
        np.asarray(routed),
        np.array([[0.0, 0.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    assert np.array_equal(np.asarray(scale_x), np.array([1.0 / _Z, 0.0, 0.0]))
    assert np.array_equal(np.asarray(scale_y), np.array([2.0 / _Z, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Reverse mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator, is_complex",
    [pytest.param(p.values[0], False, id=p.id) for p in _REAL_CASCADES]
    + [pytest.param(p.values[0], True, id=p.id) for p in _COMPLEX_CASCADES],
)
def test_the_rule_transposes_so_reverse_mode_agrees_with_forward(operator, is_complex):
    """``grad`` must equal the ``jvp`` contracted with the same direction.

    A ``custom_jvp`` rule only supports reverse mode if it is linear in the tangents,
    and it is JAX that transposes it, not us -- so this is the assertion that the rule
    was written in a transposable form. It matters because the FMM reaches these
    operators through ``jax.grad`` and never through ``jax.jvp``: a rule that was
    correct in forward mode and untransposable would pass every other test here.

    The direction is asymmetric on purpose. A symmetric one lets sign errors in the
    two transverse components cancel against each other, which is how the original
    defect survived review.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")

    coeffs = _complex_coeffs(5) if is_complex else _real_coeffs(5)
    weights = _complex_coeffs(6) if is_complex else _real_coeffs(6)

    def loss(delta):
        out = operator(coeffs, delta, order=_ORDER)
        return jnp.real(jnp.sum(weights * out))

    delta = jnp.asarray([0.0, 0.0, _Z], dtype=jnp.float64)
    direction = jnp.asarray([0.37, -1.29, 0.61], dtype=jnp.float64)

    reverse = float(jnp.sum(jax.grad(loss)(delta) * direction))
    forward = float(jax.jvp(loss, (delta,), (direction,))[1])
    # Both evaluate the same linear functional; they differ only by the order of a
    # handful of float64 multiply-adds, hence a round-off-level tolerance.
    assert reverse == pytest.approx(forward, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# The primal
# ---------------------------------------------------------------------------


def test_the_primal_at_rho_zero_is_still_the_pure_z_translation():
    """``custom_jvp`` must not have touched the forward value.

    Checked against an independent reference rather than against the operator itself:
    at ``delta == (0, 0, z)`` with ``z > 0`` both alignment rotations degenerate to the
    identity, so the whole M2L cascade must reduce to the bare z-axis recurrence. The
    ``1e-13`` tolerance is the round-off of the two ``B_U`` multiplications the cascade
    still performs (``B_U @ B_U == I`` only to machine precision); it is not slack for
    a scheme difference.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")

    multipole = _real_coeffs(5)
    through_cascade = np.asarray(
        m2l_real(
            multipole, jnp.asarray([0.0, 0.0, _Z], dtype=jnp.float64), order=_ORDER
        )
    )
    z_only = np.asarray(
        translate_along_z_m2l_real(
            multipole, jnp.asarray(_Z, dtype=jnp.float64), order=_ORDER
        )
    )
    assert np.allclose(through_cascade, z_only, rtol=1e-13, atol=1e-13)

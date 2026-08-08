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

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators._transverse_degeneracy_jvp import (
    split_transverse_tangent,
    withdraw_unresolvable_transverse,
)
from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    l2l_complex,
    m2l_complex_fused_align_deltas,
    m2l_complex_fused_carry_axis_derivative,
    m2l_complex_reference,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
    m2m_complex,
)
from jaccpot.operators.m2l_real_rot_scale import (
    l2l_rot_scale_real_batch_cached_blocks,
    m2l_real_fused_carry_axis_derivative,
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    m2m_rot_scale_real_batch_cached_blocks,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_from_z_multipole_batch,
    real_rotation_blocks_to_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
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


def test_the_band_boundary_follows_the_dtype_and_is_not_a_float64_constant():
    """At float32 the boundary must move to ``sqrt(eps32)``, four orders wider.

    ``split_transverse_tangent`` reads ``eps`` from the displacement's dtype, so the
    band is ``rho/r <= 1.5e-08`` at float64 but ``3.4e-04`` at float32. That is not a
    detail: the fused real M2L may well run float32 in production, and a float64
    constant hard-coded here would leave four orders of magnitude of approach --
    where the polar route errs ``O(eps r / rho)`` -- on the wrong branch. A float32
    displacement at ``rho/r = 1e-05`` is inside its own band and outside float64's,
    which is what makes this test distinguish the two.
    """
    boundary32 = float(np.sqrt(np.finfo(np.float32).eps)) * _Z
    tangent = jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32)
    for factor, expect_analytic in ((0.5, True), (2.0, False)):
        rho = boundary32 * factor
        _, scale_x, _ = split_transverse_tangent(
            jnp.asarray([rho, 0.0, _Z], dtype=jnp.float32), tangent
        )
        took_analytic = float(scale_x) != 0.0
        assert took_analytic is expect_analytic, (
            f"float32 rho/r = {rho / _Z:.3e} took the "
            f"{'analytic' if took_analytic else 'polar'} branch; sqrt(eps32) = "
            f"{np.sqrt(np.finfo(np.float32).eps):.3e}"
        )
    # And the same rho is *outside* the float64 band, so this is not just re-testing
    # the float64 threshold under a different name.
    _, scale_x64, _ = split_transverse_tangent(
        jnp.asarray([boundary32 * 0.5, 0.0, _Z], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
    )
    assert float(scale_x64) == 0.0


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
# float32
# ---------------------------------------------------------------------------
#
# Everything above runs at float64, where the band is a 1.5e-08-wide sliver that only
# exact zeros and centre-of-mass cancellation reach. At float32 it is 3.4e-04 wide, so
# ordinary tree geometry lands in it, and the fused real M2L may run float32 in
# production. A finite difference cannot referee this -- a float32 central difference is
# good to ~1e-03 at best -- so the reference is the same operator's float64 gradient.


_FP32_RHOS = [
    pytest.param(0.0, id="exactly on axis"),
    pytest.param(1.0e-12, id="deep inside the band"),
    pytest.param(1.0e-6, id="inside the band"),
]


@pytest.mark.parametrize(
    "operator, is_complex",
    [pytest.param(p.values[0], False, id=p.id) for p in _REAL_CASCADES]
    + [pytest.param(p.values[0], True, id=p.id) for p in _COMPLEX_CASCADES],
)
@pytest.mark.parametrize("rho", _FP32_RHOS)
def test_the_analytic_branch_holds_at_float32(
    operator: Callable, is_complex: bool, rho: float
) -> None:
    """Inside the float32 band the float32 transverse gradient must track float64.

    This is the claim the float64 tests cannot make, because float32's band admits
    ``rho`` values four orders of magnitude larger. Inside it the analytic branch is the
    ``rho -> 0`` limit and errs ``O(rho/r)``, so agreement with the float64 gradient
    should be at float32 round-off and nowhere near the ``3.4e-04`` worst case the
    minimax crossover permits *on the boundary*. Measured worst over these three ``rho``
    and both signs of ``z``: 1.1e-06.

    What the tolerance is really guarding is that the error does not *blow up* inside
    the band. The polar route errs ``O(eps r / rho)``, which at ``rho = 1e-12``,
    ``eps = 1.2e-07`` is unbounded -- so a regression that narrowed the band back to
    ``rho_sq > 0``, or that read ``eps`` from a float64 constant, fails here by orders of
    magnitude rather than marginally.

    Parameters
    ----------
    operator : Callable
        The cascade under test, in either basis.
    is_complex : bool
        Whether ``operator`` takes complex coefficients.
    rho : float
        Cylindrical radius, inside the float32 band at ``r ~ 2.5``.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the float64 reference requires JAX_ENABLE_X64=1")

    def transverse_gradient(dtype):
        if is_complex:
            complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
            coeffs = _complex_coeffs(5).astype(complex_dtype)
            weights = _complex_coeffs(6).astype(complex_dtype)
        else:
            coeffs = _real_coeffs(5).astype(dtype)
            weights = _real_coeffs(6).astype(dtype)

        def loss(delta):
            out = operator(coeffs, delta, order=_ORDER)
            return jnp.real(jnp.sum(weights * out))

        gradient = jax.grad(loss)(jnp.asarray([rho, 0.0, _Z], dtype=dtype))
        return np.asarray(gradient, dtype=np.float64)[:2]

    single = transverse_gradient(jnp.float32)
    double = transverse_gradient(jnp.float64)

    # Not vacuous: the defect this rule fixes returned exactly (0, 0) here, so a
    # reference that were itself ~0 would let the comparison pass without testing
    # anything -- the same way a symmetric construction let G.10 survive review.
    assert np.max(np.abs(double)) > 1.0e-3, (
        "the float64 transverse gradient is ~0, so this comparison would pass "
        "without testing anything"
    )
    relative = float(np.max(np.abs(single - double)) / np.max(np.abs(double)))
    assert relative < 1.0e-5, (
        f"float32 transverse gradient differs from float64 by {relative:.3e} at "
        f"rho = {rho:.1e}\n  float32: {single}\n  float64: {double}"
    )


@pytest.mark.parametrize("z", [_Z, -3.0])
@pytest.mark.parametrize("rho", _FP32_RHOS)
def test_the_analytic_branch_holds_at_float32_on_the_fused_pallas_lanes(
    rho: float, z: float
) -> None:
    """The same float32 claim, on the two lanes that most need it.

    The test above covers the six direct cascades. It does **not** cover either fused
    Pallas M2L, and that was the gap that mattered most: the band is
    ``rho_sq <= eps r_sq``, so its width in ``rho/r`` is ``sqrt(eps)`` -- 1.5e-08 at
    float64 but **3.4e-04** at float32, four orders wider -- and the fused path is the
    one most likely to run float32 in production. A lane can hold at float64 and still
    fail here, because at float32 the band admits ``rho`` that ordinary tree geometry
    reaches rather than only exact zeros.

    The **shipped** functions are differentiated, not a hand-composed rebuild of them,
    for the reason ``test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative``
    records: a rebuild passes whether or not ``runtime/kernels/core.py`` wires the pair
    up. GPU-only, unavoidably -- both hardcode ``interpret=False``.

    Measured worst over these three ``rho`` and both signs of ``z``: 8.9e-07 on the real
    fused lane and 1.5e-06 on the complex one, against direct-lane controls of 1.8e-06
    and 1.5e-06 -- i.e. the fused lanes carry no float32 weakness the direct ones do not.
    The grid deliberately stops short of ``rho = 3.4e-04``: that is the crossover
    boundary itself, where the minimax choice makes the worst error ``~sqrt(eps)`` **by
    design** (2.4e-04 measured there, on the fused real lane and its direct control
    alike), so a tolerance that bound would assert nothing about this branch.

    Parameters
    ----------
    rho : float
        Cylindrical radius, inside the float32 band at ``r ~ 2.5``.
    z : float
        Axial separation; both signs, because ``z < 0`` makes the degenerate alignment
        rotation a pi-turn rather than the identity.

    Raises
    ------
    Exception
        Whatever the Pallas lowering raised, re-raised unless it names a known
        hardware/runtime incompatibility -- those are skipped instead, so a card that
        cannot run the kernel does not read as a numerical failure.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the float64 reference requires JAX_ENABLE_X64=1")
    from jaccpot.pallas.m2l_complex_fused import pallas_m2l_complex_fused_supported
    from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported
    from jaccpot.runtime.kernels.core import (
        _m2l_complex_batch_kernel_fused_pallas,
        _m2l_real_batch_kernel_fused_pallas,
    )

    if not (pallas_m2l_real_fused_supported() and pallas_m2l_complex_fused_supported()):
        pytest.skip("the fused Pallas M2L lanes require an Ampere+ (sm_80) GPU")

    def transverse_gradient(operator, is_complex, dtype):
        if is_complex:
            complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
            coeffs = _complex_coeffs(5).astype(complex_dtype)
            weights = _complex_coeffs(6).astype(complex_dtype)
        else:
            coeffs = _real_coeffs(5).astype(dtype)
            weights = _real_coeffs(6).astype(dtype)

        def loss(delta):
            out = operator(coeffs[None, :], delta, order=_ORDER)
            return jnp.real(jnp.sum(weights[None, :] * out))

        gradient = jax.grad(loss)(jnp.asarray([[rho, 0.0, z]], dtype=dtype))
        return np.asarray(gradient, dtype=np.float64)[0, :2]

    lanes = (
        (
            "real fused",
            lambda c, d, order: _m2l_real_batch_kernel_fused_pallas(
                c, d, order=order, m2l_impl="rot_scale"
            ),
            False,
        ),
        ("complex fused", _m2l_complex_batch_kernel_fused_pallas, True),
    )
    for name, operator, is_complex in lanes:
        try:
            single = transverse_gradient(operator, is_complex, jnp.float32)
            double = transverse_gradient(operator, is_complex, jnp.float64)
        except Exception as exc:  # pragma: no cover - GPU/runtime dependent
            message = str(exc).lower()
            if any(t in message for t in ("warpgroup", "ptx", "triton", "mosaic")):
                pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
            raise

        # Not vacuous: the defect this rule fixes returned exactly (0, 0) here.
        assert np.max(np.abs(double)) > 1.0e-3, (
            f"{name}: the float64 transverse gradient is ~0, so this comparison "
            "would pass without testing anything"
        )
        relative = float(np.max(np.abs(single - double)) / np.max(np.abs(double)))
        assert relative < 1.0e-5, (
            f"{name}: float32 transverse gradient differs from float64 by "
            f"{relative:.3e} at rho = {rho:.1e}, z = {z}\n"
            f"  float32: {single}\n  float64: {double}"
        )


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


# ---------------------------------------------------------------------------
# The precomputed-block lanes
# ---------------------------------------------------------------------------
#
# These lanes receive their rotation blocks as separate arguments, built once per
# interaction class and reused, so no single function sees both the displacement and
# the operator built from it. The fix is therefore split in two: the block builder
# withdraws the transverse tangent it cannot resolve, and the consumer supplies the
# cascade-level term. Neither half is correct alone, so what has to be asserted is
# that they compose -- and the only convincing statement of that is equality with the
# lane that never needed splitting.


def _real_blocks(deltas, which_pair, dtype):
    to_z, from_z = which_pair
    return to_z(deltas, order=_ORDER, dtype=dtype), from_z(
        deltas, order=_ORDER, dtype=dtype
    )


def _lane_pair(name):
    """``(direct, cached)`` losses over the same displacement batch, for one lane."""
    order = _ORDER
    width = sh_size(order)
    real_coeffs = jax.random.normal(
        jax.random.PRNGKey(1), (3, width), dtype=jnp.float64
    )
    real_weights = jax.random.normal(
        jax.random.PRNGKey(2), (3, width), dtype=jnp.float64
    )
    rng = np.random.default_rng(3)
    cplx_coeffs = jnp.asarray(
        rng.normal(size=(3, width)) + 1j * rng.normal(size=(3, width)),
        dtype=jnp.complex128,
    )
    cplx_weights = jnp.asarray(
        rng.normal(size=(3, width)) + 1j * rng.normal(size=(3, width)),
        dtype=jnp.complex128,
    )

    if name == "m2l_real":

        def direct(d):
            return jnp.sum(
                real_weights * m2l_rot_scale_real_batch(real_coeffs, d, order=order)
            )

        def cached(d):
            to_z, from_z = _real_blocks(
                d,
                (
                    real_rotation_blocks_to_z_multipole_batch,
                    real_rotation_blocks_from_z_local_batch,
                ),
                real_coeffs.dtype,
            )
            return jnp.sum(
                real_weights
                * m2l_rot_scale_real_batch_cached_blocks(
                    real_coeffs, d, to_z, from_z, order=order
                )
            )

    elif name == "m2m_real":

        def direct(d):
            return jnp.sum(
                real_weights
                * jax.vmap(lambda c, dd: m2m_real(c, dd, order=order))(real_coeffs, d)
            )

        def cached(d):
            to_z, from_z = _real_blocks(
                d,
                (
                    real_rotation_blocks_to_z_multipole_batch,
                    real_rotation_blocks_from_z_multipole_batch,
                ),
                real_coeffs.dtype,
            )
            return jnp.sum(
                real_weights
                * m2m_rot_scale_real_batch_cached_blocks(
                    real_coeffs, d, to_z, from_z, order=order
                )
            )

    elif name == "l2l_real":

        def direct(d):
            return jnp.sum(
                real_weights
                * jax.vmap(lambda c, dd: l2l_real(c, dd, order=order))(real_coeffs, d)
            )

        def cached(d):
            to_z, from_z = _real_blocks(
                d,
                (
                    real_rotation_blocks_to_z_local_batch,
                    real_rotation_blocks_from_z_local_batch,
                ),
                real_coeffs.dtype,
            )
            return jnp.sum(
                real_weights
                * l2l_rot_scale_real_batch_cached_blocks(
                    real_coeffs, d, to_z, from_z, order=order
                )
            )

    else:

        def direct(d):
            return jnp.real(
                jnp.sum(
                    cplx_weights
                    * m2l_complex_reference_batch(cplx_coeffs, d, order=order)
                )
            )

        def cached(d):
            to_z = complex_rotation_blocks_to_z_solidfmm_batch(
                d, order=order, basis="multipole", dtype=cplx_coeffs.dtype
            )
            from_z = complex_rotation_blocks_from_z_solidfmm_batch(
                d, order=order, basis="local", dtype=cplx_coeffs.dtype
            )
            return jnp.real(
                jnp.sum(
                    cplx_weights
                    * m2l_complex_reference_batch_cached_blocks(
                        cplx_coeffs, d, to_z, from_z, order=order
                    )
                )
            )

    return direct, cached


@pytest.mark.parametrize(
    "lane", ["m2l_real", "m2m_real", "l2l_real", "m2l_complex_reference"]
)
def test_precomputed_block_lanes_match_their_direct_twin_in_gradient(lane):
    """The cached-blocks lanes must agree with the direct lanes on the *gradient* too.

    The forward equivalence was already asserted elsewhere and never in doubt. The
    gradient one is the load-bearing claim here, and it is what caught the gap: before
    the block builders withdrew their unresolvable transverse tangent, the real M2L lane
    returned ``(0, 0)`` for the on-axis transverse gradient where the direct lane gave
    ``(-0.270, -1.303)`` -- a discrepancy of 1.30 that no forward comparison could see.

    The batch deliberately contains all three regimes at once, because they take
    different branches: exactly on axis, one ulp off it (what centre-of-mass cancellation
    between two mathematically equal centres actually produces), and generic. Tolerance
    is round-off on a handful of float64 matmuls -- the two lanes compute the same
    quantity by different groupings, not to within a scheme difference.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")

    deltas = jnp.asarray(
        [[0.0, 0.0, 2.5], [5.551e-17, 0.0, -3.0], [1.1, -0.4, 3.0]], dtype=jnp.float64
    )
    direct, cached = _lane_pair(lane)

    assert float(direct(deltas)) == pytest.approx(
        float(cached(deltas)), rel=1e-13, abs=1e-13
    )
    grad_direct = np.asarray(jax.grad(direct)(deltas))
    grad_cached = np.asarray(jax.grad(cached)(deltas))
    worst = float(np.max(np.abs(grad_direct - grad_cached)))
    assert worst <= 1.0e-12, (
        f"{lane}: cached-blocks gradient differs from the direct lane by {worst:.3e}\n"
        f"  direct: {grad_direct}\n  cached: {grad_cached}"
    )
    # And the on-axis row is not accidentally zero in both -- otherwise the two could
    # agree by sharing the defect this whole change removes.
    assert np.max(np.abs(grad_direct[0, :2])) > 1.0e-3, (
        "the on-axis transverse gradient is ~0 in the direct lane too, so this "
        "comparison would pass without testing anything"
    )


def test_passing_blocks_by_keyword_fails_loudly():
    """An array in the keyword position must name itself, not surface as unhashable.

    The wrapper carries keyword arguments as ``custom_jvp`` ``nondiff_argnums``, so they
    must be hashable and non-differentiable. Every caller in the tree passes the rotation
    blocks positionally; someone who does not would otherwise get an unhashable-type
    failure from inside ``custom_jvp`` with nothing pointing at their call site.
    """
    width = sh_size(2)
    multipoles = jnp.ones((1, width), dtype=jnp.float64)
    deltas = jnp.asarray([[0.0, 0.0, 2.5]], dtype=jnp.float64)
    blocks = jnp.zeros((1, 3, 5, 5), dtype=jnp.float64)

    with pytest.raises(TypeError, match="'blocks_from_z', 'blocks_to_z'"):
        m2l_rot_scale_real_batch_cached_blocks(
            multipoles, deltas, blocks_to_z=blocks, blocks_from_z=blocks, order=2
        )
    # And the positional form is untouched.
    assert m2l_rot_scale_real_batch_cached_blocks(
        multipoles, deltas, blocks, blocks, order=2
    ).shape == (1, width)


@pytest.mark.parametrize("interpret", [True, False])
def test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient(interpret):
    """The fused Pallas M2L composite must agree with the pure-JAX lane on the gradient.

    This lane could not take the ``custom_jvp`` the others did: its middle is
    ``m2l_real_fused_pallas_cvjp``, a ``custom_vjp``, and JAX refuses forward-mode
    through one ("can't apply forward-mode autodiff (jvp) to a custom_vjp function"). It
    is covered instead by the pair
    :func:`~jaccpot.operators._transverse_degeneracy_jvp.withdraw_unresolvable_transverse`
    (before the blocks and radius are built) and
    :func:`~jaccpot.operators.m2l_real_rot_scale.m2l_real_fused_carry_axis_derivative`
    (after the kernel), neither of which differentiates the kernel.

    Measured without the carrier the on-axis rows come back as exactly ``(0, 0)`` and the
    lanes are **1.98** apart; with it, 2.7e-15. ``interpret=True`` runs the kernel's
    reference lowering on CPU, which is where this is verified; ``interpret=False`` runs
    the real Triton kernel and is skipped off Ampere+, so the same assertion covers the
    hardware the lane actually ships on.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")
    from jaccpot.pallas.m2l_real_fused import (
        m2l_real_fused_pallas_cvjp,
        pallas_m2l_real_fused_supported,
    )

    if not interpret and not pallas_m2l_real_fused_supported():
        pytest.skip("the fused real Pallas M2L requires an Ampere+ (sm_80) GPU")

    width = sh_size(_ORDER)
    multipoles = jax.random.normal(jax.random.PRNGKey(1), (3, width), dtype=jnp.float64)
    weights = jax.random.normal(jax.random.PRNGKey(2), (3, width), dtype=jnp.float64)
    deltas = jnp.asarray(
        [[0.0, 0.0, 2.5], [5.551e-17, 0.0, -3.0], [1.1, -0.4, 3.0]], dtype=jnp.float64
    )

    def direct(d):
        return jnp.sum(weights * m2l_rot_scale_real_batch(multipoles, d, order=_ORDER))

    def fused(d):
        aligned = withdraw_unresolvable_transverse(d)
        radii = jnp.linalg.norm(aligned, axis=1)
        to_z = real_rotation_blocks_to_z_multipole_batch(
            aligned, order=_ORDER, dtype=multipoles.dtype
        )
        from_z = real_rotation_blocks_from_z_local_batch(
            aligned, order=_ORDER, dtype=multipoles.dtype
        )
        out = m2l_real_fused_pallas_cvjp(
            multipoles, to_z, from_z, radii, _ORDER, interpret, "triton"
        )
        out = m2l_real_fused_carry_axis_derivative(
            out, multipoles, d, to_z, from_z, radii, order=_ORDER
        )
        return jnp.sum(weights * out)

    try:
        grad_direct = np.asarray(jax.grad(direct)(deltas))
        grad_fused = np.asarray(jax.grad(fused)(deltas))
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        message = str(exc).lower()
        if not interpret and any(
            token in message for token in ("warpgroup", "ptx", "triton", "mosaic")
        ):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise

    # The fused kernel and the rotate/scale cascade are separate implementations, so the
    # bound is their asserted forward agreement (<1e-10 at fp64 per
    # tests/test_m2l_real_fused_pallas.py) rather than pure round-off; measured 2.7e-15.
    tolerance = 1.0e-10 if interpret else 1.0e-8
    worst = float(np.max(np.abs(grad_direct - grad_fused)))
    assert worst <= tolerance, (
        f"fused Pallas gradient differs from the pure-JAX lane by {worst:.3e}\n"
        f"  direct: {grad_direct}\n  fused:  {grad_fused}"
    )
    assert np.max(np.abs(grad_direct[0, :2])) > 1.0e-3, (
        "the on-axis transverse gradient is ~0 in the direct lane too, so this "
        "comparison would pass without testing anything"
    )


@pytest.mark.parametrize("interpret", [True, False])
def test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient(interpret):
    """The complex fused Pallas M2L needs the same two halves the real one does.

    Its middle, ``m2l_complex_fused_pallas_cvjp``, is a ``custom_vjp`` for the same
    reason the real one is, so it takes the same pair rather than the consumer
    decorator: withdraw the unresolvable transverse tangent from the displacement before
    the radius and both block stacks are built, and carry the analytic term back onto the
    output afterwards.

    This lane already had the *withdrawal*, because
    ``_complex_rotation_blocks_{to,from}_z_solidfmm_padded`` carry
    :func:`~jaccpot.operators._transverse_degeneracy_jvp.without_unresolvable_transverse_jvp`
    for the sake of the cached-blocks lane. Half of the pair is worse than neither half:
    with the withdrawal alone the on-axis rows came back exactly ``(0, 0)`` and this lane
    sat **5.1e-01** from the pure-JAX reference, where before any of this work the two
    agreed to 6.7e-16 -- both wrong then, only one wrong after. That is the
    reference-versus-fused equivalence ``NUMERICS_AND_JAX.md`` §1 requires.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")
    from jaccpot.pallas.m2l_complex_fused import (
        m2l_complex_fused_pallas_cvjp,
        pallas_m2l_complex_fused_supported,
    )

    if not interpret and not pallas_m2l_complex_fused_supported():
        pytest.skip("the fused complex Pallas M2L requires an Ampere+ (sm_80) GPU")

    multipoles = _complex_coeffs(1)[None, :].repeat(3, axis=0)
    weights = _complex_coeffs(2)[None, :].repeat(3, axis=0)
    deltas = jnp.asarray(
        [[0.0, 0.0, 2.5], [5.551e-17, 0.0, -3.0], [1.1, -0.4, 3.0]], dtype=jnp.float64
    )

    def direct(d):
        return jnp.real(
            jnp.sum(weights * m2l_complex_reference_batch(multipoles, d, order=_ORDER))
        )

    def fused(d):
        aligned = m2l_complex_fused_align_deltas(d)
        radii = jnp.sqrt(jnp.sum(aligned * aligned, axis=-1))
        to_z = complex_rotation_blocks_to_z_solidfmm_batch(
            aligned, order=_ORDER, basis="multipole", dtype=multipoles.dtype
        )
        from_z = complex_rotation_blocks_from_z_solidfmm_batch(
            aligned, order=_ORDER, basis="local", dtype=multipoles.dtype
        )
        out = m2l_complex_fused_pallas_cvjp(
            multipoles, to_z, from_z, radii, _ORDER, interpret, "triton"
        )
        out = m2l_complex_fused_carry_axis_derivative(
            out, multipoles, d, to_z, from_z, radii, order=_ORDER
        )
        return jnp.real(jnp.sum(weights * out))

    try:
        grad_direct = np.asarray(jax.grad(direct)(deltas))
        grad_fused = np.asarray(jax.grad(fused)(deltas))
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        message = str(exc).lower()
        if not interpret and any(
            token in message for token in ("warpgroup", "ptx", "triton", "mosaic")
        ):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise

    # Separate implementations, so the bound is their asserted forward agreement rather
    # than pure round-off -- the same reasoning as the real lane's tolerance.
    tolerance = 1.0e-10 if interpret else 1.0e-8
    worst = float(np.max(np.abs(grad_direct - grad_fused)))
    assert worst <= tolerance, (
        f"complex fused Pallas gradient differs from the pure-JAX lane by {worst:.3e}\n"
        f"  direct: {grad_direct}\n  fused:  {grad_fused}"
    )
    assert np.max(np.abs(grad_direct[0, :2])) > 1.0e-3, (
        "the on-axis transverse gradient is ~0 in the direct lane too, so this "
        "comparison would pass without testing anything"
    )


def test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative():
    """The wiring in ``runtime/kernels/core.py``, not a reconstruction of it.

    The test above composes the lane itself so that ``interpret=True`` can cover it on
    CPU. That leaves the production function free to be wired differently, which is
    exactly how the gap arose: it built its blocks from the withdrawing builders and
    never carried the term back. So this asserts the shipped function.

    Nothing else can. ``test_m2l_complex_fused_pallas_custom_vjp_matches_twin``
    differentiates the kernel's four inputs -- multipoles, both block stacks, the radius
    -- and never ``deltas``, so it is structurally blind to a degeneracy that lives only
    in the ``delta -> blocks`` map, and ``m2l_complex_fused_jax``, the twin it compares
    against, shares the blindness.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")
    from jaccpot.pallas.m2l_complex_fused import pallas_m2l_complex_fused_supported
    from jaccpot.runtime.kernels.core import _m2l_complex_batch_kernel_fused_pallas

    if not pallas_m2l_complex_fused_supported():
        pytest.skip("the fused complex Pallas M2L requires an Ampere+ (sm_80) GPU")

    multipoles = _complex_coeffs(1)[None, :].repeat(3, axis=0)
    weights = _complex_coeffs(2)[None, :].repeat(3, axis=0)
    deltas = jnp.asarray(
        [[0.0, 0.0, 2.5], [5.551e-17, 0.0, -3.0], [1.1, -0.4, 3.0]], dtype=jnp.float64
    )

    def loss(operator):
        return lambda d: jnp.real(
            jnp.sum(weights * operator(multipoles, d, order=_ORDER))
        )

    try:
        grad_fused = np.asarray(
            jax.grad(loss(_m2l_complex_batch_kernel_fused_pallas))(deltas)
        )
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        message = str(exc).lower()
        if any(token in message for token in ("warpgroup", "ptx", "triton", "mosaic")):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise
    grad_direct = np.asarray(jax.grad(loss(m2l_complex_reference_batch))(deltas))

    worst = float(np.max(np.abs(grad_direct - grad_fused)))
    assert worst <= 1.0e-8, (
        f"the production complex fused M2L gradient differs from the reference batch by "
        f"{worst:.3e}\n  direct: {grad_direct}\n  fused:  {grad_fused}"
    )
    assert np.max(np.abs(grad_direct[0, :2])) > 1.0e-3, (
        "the on-axis transverse gradient is ~0 in the reference lane too, so this "
        "comparison would pass without testing anything"
    )


def test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative() -> None:
    """The real lane's counterpart of the test above, and for the same reason.

    ``test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient`` composes the real
    lane by hand so that ``interpret=True`` covers the mechanism on CPU, which means it
    passes whether or not ``runtime/kernels/core.py`` actually wires the pair up. That is
    precisely the gap that let the *complex* lane ship with only half the rule, and the
    real lane had no equivalent guard -- so it is the default basis that was the less
    protected of the two.

    GPU-only, unavoidably: the production function hardcodes ``interpret=False``, so
    there is no CPU lowering of the shipped path to differentiate. It runs wherever the
    fused kernel does.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("requires float64 (JAX_ENABLE_X64=1)")
    from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported
    from jaccpot.runtime.kernels.core import _m2l_real_batch_kernel_fused_pallas

    if not pallas_m2l_real_fused_supported():
        pytest.skip("the fused real Pallas M2L requires an Ampere+ (sm_80) GPU")

    width = sh_size(_ORDER)
    multipoles = jax.random.normal(jax.random.PRNGKey(1), (3, width), dtype=jnp.float64)
    weights = jax.random.normal(jax.random.PRNGKey(2), (3, width), dtype=jnp.float64)
    deltas = jnp.asarray(
        [[0.0, 0.0, 2.5], [5.551e-17, 0.0, -3.0], [1.1, -0.4, 3.0]], dtype=jnp.float64
    )

    def fused(d):
        return jnp.sum(
            weights
            * _m2l_real_batch_kernel_fused_pallas(
                multipoles, d, order=_ORDER, m2l_impl="rot_scale"
            )
        )

    def direct(d):
        return jnp.sum(weights * m2l_rot_scale_real_batch(multipoles, d, order=_ORDER))

    try:
        grad_fused = np.asarray(jax.grad(fused)(deltas))
    except Exception as exc:  # pragma: no cover - GPU/runtime dependent
        message = str(exc).lower()
        if any(token in message for token in ("warpgroup", "ptx", "triton", "mosaic")):
            pytest.skip(f"Pallas kernel unavailable on this GPU/runtime: {exc}")
        raise
    grad_direct = np.asarray(jax.grad(direct)(deltas))

    worst = float(np.max(np.abs(grad_direct - grad_fused)))
    assert worst <= 1.0e-8, (
        f"the production real fused M2L gradient differs from the rot-scale lane by "
        f"{worst:.3e}\n  direct: {grad_direct}\n  fused:  {grad_fused}"
    )
    assert np.max(np.abs(grad_direct[0, :2])) > 1.0e-3, (
        "the on-axis transverse gradient is ~0 in the reference lane too, so this "
        "comparison would pass without testing anything"
    )

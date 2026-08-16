"""P2M and L2P in the Dehnen real basis, plus the analytic L2P reverse rule.

The two ends of the FMM pipeline that touch particles directly: ``p2m_real_direct``
builds a multipole from a source displacement, and the ``evaluate_local_real``
family contracts a local expansion at a target. Both evaluate the real solid
harmonics ``U_n^m`` **directly as polynomials** in ``(x, y, z)`` via a Chebyshev
recurrence for ``cos(m phi)`` / ``sin(m phi)``, with no complex arithmetic and no
trigonometry -- see the theory section in :mod:`jaccpot.operators.real_harmonics`.

WHY THE ``custom_vjp`` IS HERE. ``_evaluate_local_real_with_grad_cvjp`` supplies
the L2P reverse rule analytically rather than letting autodiff walk the
recurrence, gated at trace time by
:func:`~jaccpot.runtime.grad_options.analytic_l2p_vjp_enabled`. It is
load-bearing (NUMERICS_AND_JAX §1) and it lives with the primal it differentiates.

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.operators.dtypes import floor_squared_radius
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d
from jaccpot.runtime.grad_options import analytic_l2p_vjp_enabled

from ._sh_indexing import (
    _azimuth_from_floored_rho,
    _factorial_table_jax,
    sh_index,
    sh_size,
)

__all__ = [
    "p2m_real_direct",
    "evaluate_local_real",
    "evaluate_local_real_with_grad",
    "evaluate_local_real_derivative_tower",
    "evaluate_local_real_derivative_tower_batch",
]


@partial(jax.jit, static_argnames=("order",))
def p2m_real_direct(
    delta: Array,
    mass: Array,
    *,
    order: int,
) -> Array:
    """P2M for a single point mass using direct real harmonic computation.

    This computes multipole coefficients M_n^m = mass * U_n^m(delta) directly
    in real arithmetic, without going through complex harmonics.

    Parameters
    ----------
    delta : Array
        (3,) vector from expansion center to particle position.
    mass : Array
        Particle mass (scalar).
    order : int
        Maximum SH degree p.

    Returns
    -------
    Array
        Packed real multipole coefficients of shape ((p+1)^2,).

    Notes
    -----
    Uses the Dehnen normalization in the **no-√2 real basis**:
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) cos(|m|φ) / (n+|m|)!  for m >= 0
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) sin(|m|φ) / (n+|m|)!  for m < 0

    The cos(mφ) and sin(mφ) terms are computed via Chebyshev recurrence
    from cos φ = x/ρ and sin φ = y/ρ, avoiding trigonometric functions.

    *Validation.* What actually runs is
    ``tests/unit/operators/test_real_harmonics.py::test_p2m_real_direct_dehnen_table3``,
    which checks the Dehnen (2014) Table 3 entries for **degree 1 only**
    (``U_1^{-1} = y/2``, ``U_1^{0} = z``, ``U_1^{+1} = x/2``) at the three unit
    axis directions, to ``atol=1e-10``. Degree 0 is covered separately by
    ``test_p2m_real_direct_monopole``.

    Degrees 2-6 are pinned too, by two further tests in that module rather than by a
    table. An earlier version of this docstring said they were "unverified in-repo"
    and that the coverage "cannot see a per-``m`` normalisation or sign error above
    degree 1"; both statements are now false, and the second was already too
    pessimistic when written:

    * ``test_complex_to_dehnen_real_matches_p2m_real_direct`` cross-checks orders
      0-6 against ``complex_to_dehnen_real_coeffs(complex_R_solidfmm(...))`` over six
      geometries, to 1e-13 relative L2. This is a *relative* check -- a convention
      error shared between ``Q`` and this function would cancel in it -- but it does
      catch a per-``m`` error introduced here alone.
    * ``test_solid_harmonic_z_derivative_lowers_the_degree`` and
      ``test_solid_harmonic_transverse_derivative_coefficients_are_half_integers``
      close the absolute gap. The 1/(n+|m|)!-normalised solid harmonics satisfy
      ``dU_n^m/dz = U_{n-1}^m`` and transverse recurrences whose coefficients are
      exact multiples of 1/2; those recurrences plus ``U_0^0 = 1`` determine every
      ``U_n^m`` uniquely. Scaling any single ``U_n^m`` by ``1 + 1e-3`` keeps the
      recurrences structural but makes the coefficients non-half-integer, so the
      per-``m`` normalisation is anchored without transcribing Table 3.

    The one-off derivation this used to lean on, ``scripts/derive_table3_polynomials.py``,
    was never committed and is not needed; a generator that CI never runs would have
    had the same failure mode again.

    Raises
    ------
    ValueError
        If ``order`` is negative. ``order`` is static under ``jit``, so this
        fires at trace time.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    d = jnp.asarray(delta)
    mass = jnp.asarray(mass).reshape(())
    dtype = d.dtype

    x, y, z = d[0], d[1], d[2]
    r2 = jnp.dot(d, d, precision=lax.Precision.HIGHEST)
    r = jnp.sqrt(floor_squared_radius(r2))
    rho2 = x * x + y * y
    rho = jnp.sqrt(floor_squared_radius(rho2))

    cos_theta = z / r
    sin_theta = rho / r

    # cos φ = x/ρ, sin φ = y/ρ, unconditionally -- see
    # `_azimuth_from_floored_rho` for why a degenerate branch here breaks the
    # transverse gradient. The L2P case is the one that reached production, but
    # this P2M has the identical structure and the identical defect: the gradient
    # w.r.t. a particle position exactly at its leaf's expansion centre.
    cos_phi, sin_phi = _azimuth_from_floored_rho(x, y, rho)

    # Precompute factorials
    fact = _factorial_table_jax(2 * p, dtype)

    def fact_int(k: Any) -> Array:
        return fact[jnp.asarray(k, dtype=jnp.int32)]

    # Compute cos(m*φ) and sin(m*φ) via Chebyshev recurrence:
    # cos((m+1)φ) = 2*cos(φ)*cos(mφ) - cos((m-1)φ)
    # sin((m+1)φ) = 2*cos(φ)*sin(mφ) - sin((m-1)φ)
    # Note: we use sin(mφ) = sin((m-1)φ)*cos(φ) + cos((m-1)φ)*sin(φ) instead
    cos_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    sin_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    cos_m_phi = cos_m_phi.at[0].set(1.0)  # cos(0) = 1
    sin_m_phi = sin_m_phi.at[0].set(0.0)  # sin(0) = 0
    if p >= 1:
        cos_m_phi = cos_m_phi.at[1].set(cos_phi)
        sin_m_phi = sin_m_phi.at[1].set(sin_phi)
    for m in range(2, p + 1):
        # Chebyshev recurrence
        cos_m_phi = cos_m_phi.at[m].set(
            2.0 * cos_phi * cos_m_phi[m - 1] - cos_m_phi[m - 2]
        )
        sin_m_phi = sin_m_phi.at[m].set(
            2.0 * cos_phi * sin_m_phi[m - 1] - sin_m_phi[m - 2]
        )

    # Build coefficients degree by degree
    ncoeff = sh_size(p)
    coeffs = jnp.zeros((ncoeff,), dtype=dtype)

    # Associated Legendre function P_n^m(cos_theta) WITHOUT Condon-Shortley
    # phase.
    #
    # Dehnen (2014) equation 2b includes an explicit (-1)^m, which cancels the
    # C-S phase if present. We therefore compute P_n^m without C-S here.
    #
    # Recursion (without C-S phase):
    #   P_m^m = (2m-1)!! sin^m(theta)
    #   P_{m+1}^m = cos(theta) (2m+1) P_m^m
    #   (n-m) P_n^m = cos(theta) (2n-1) P_{n-1}^m - (n+m-1) P_{n-2}^m
    for n in range(p + 1):
        r_n = r**n

        for m in range(-n, n + 1):
            m_abs = abs(m)

            # P_m^m (without Condon-Shortley phase)
            if m_abs == 0:
                Pmm = jnp.asarray(1.0, dtype=dtype)
            else:
                # (2m-1)!! = (2m)! / (2^m m!)
                double_fact = fact_int(2 * m_abs) / ((2.0**m_abs) * fact_int(m_abs))
                Pmm = double_fact * (sin_theta**m_abs)

            if m_abs == n:
                P_nm = Pmm
            elif m_abs + 1 == n:
                # P_{m+1}^m = (2m+1) cos(theta) P_m^m
                P_nm = (2.0 * m_abs + 1.0) * cos_theta * Pmm
            else:
                # General recursion
                Pnm2 = Pmm
                Pnm1 = (2.0 * m_abs + 1.0) * cos_theta * Pmm
                for k in range(m_abs + 2, n + 1):
                    numer = (2.0 * k - 1.0) * cos_theta * Pnm1
                    numer = numer - (k + m_abs - 1.0) * Pnm2
                    Pk = numer / (k - m_abs)
                    Pnm2 = Pnm1
                    Pnm1 = Pk
                P_nm = Pnm1

            # Dehnen normalization: divide by (n + |m|)!
            denom = fact_int(n + m_abs)

            # U_n^m = r^n P_n^{|m|}(cos θ) {cos(|m|φ), sin(|m|φ)} / (n+|m|)!
            # For m >= 0: multiply by cos(|m|φ)
            # For m < 0: multiply by sin(|m|φ)
            if m >= 0:
                U_nm = r_n * P_nm * cos_m_phi[m_abs] / denom
            else:
                U_nm = r_n * P_nm * sin_m_phi[m_abs] / denom

            # M_n^m = mass * U_n^m
            idx = sh_index(n, m)
            coeffs = coeffs.at[idx].set(mass * U_nm)

    return coeffs


# ===========================================================================
# L2P: Local expansion evaluation (pure real)
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
def evaluate_local_real(
    local_coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Evaluate local expansion at a point using pure real arithmetic.

    Implements Dehnen (2014) equation 3a:
        Ψ(x_b) = Σ_{n,m} Υ_n^{m*}(s_B - x_b) · F_n^m(s_B)

    where:
        - s_B is the local expansion center
        - x_b is the evaluation point
        - delta = s_B - x_b (center MINUS evaluation point, Dehnen convention)
        - F_n^m are the local expansion coefficients from M2L
        - Υ_n^m are the inner solid harmonics (eq 2b)

    The real-valued inner solid harmonics U_n^m (eq 58) are:
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) cos(|m|φ) / (n+|m|)!  for m >= 0
        U_n^m(r,θ,φ) = r^n P_n^{|m|}(cos θ) sin(|m|φ) / (n+|m|)!  for m < 0

    This normalization matches P2M (which also uses U_n^m), ensuring
    consistency through the FMM pipeline: P2M → M2M → M2L → L2L → L2P.

    Note: These locals use the Dehnen no-√2 real basis. Do not compare
    directly to unitary real tesseral coefficients without conversion.

    Note: Do not compare these coefficients to the unitary real tesseral
    basis in the real-harmonic utility layer without converting between bases.

    **Important Convention**: delta = center - eval_point
    (NOT eval_point - center)!
    This follows Dehnen's definition in eq 3a where the argument is
    (s_B - x_b).
    The sign matters because the Taylor series 1/|R+d| = (1/R) Σ_n (-d/R)^n has
    alternating signs that arise naturally when delta points from eval toward
    center.

    Parameters
    ----------
    local_coeffs : Array
        Packed real local coefficients F_n^m of shape ((p+1)^2,).
    delta : Array
        (3,) vector from evaluation point TO expansion center:
        delta = center - eval_point.
        This is the Dehnen (2014) convention from equation 3a.
    order : int
        Maximum SH degree p.

    Returns
    -------
    Array
        Scalar potential value at the evaluation point.
    """
    p = int(order)
    d = jnp.asarray(delta)
    local_coeffs = jnp.asarray(local_coeffs)
    dtype = d.dtype

    # delta follows Dehnen convention: center - eval_point (eq 3a: s_B - x_b)
    # No sign flip needed - caller provides the correct convention.

    x, y, z = d[0], d[1], d[2]
    r2 = jnp.dot(d, d, precision=lax.Precision.HIGHEST)
    r = jnp.sqrt(floor_squared_radius(r2))
    rho2 = x * x + y * y
    rho = jnp.sqrt(floor_squared_radius(rho2))

    cos_theta = z / r
    sin_theta = rho / r

    # cos φ = x/ρ, sin φ = y/ρ, unconditionally -- see
    # `_azimuth_from_floored_rho` for why a degenerate branch here breaks the
    # transverse gradient, i.e. the x and y components of the far-field force.
    cos_phi, sin_phi = _azimuth_from_floored_rho(x, y, rho)

    # Precompute factorials
    fact = _factorial_table_jax(2 * p, dtype)

    def fact_int(k: Any) -> Array:
        return fact[jnp.asarray(k, dtype=jnp.int32)]

    # Compute cos(m*φ) and sin(m*φ) via Chebyshev recurrence
    cos_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    sin_m_phi = jnp.zeros((p + 1,), dtype=dtype)
    cos_m_phi = cos_m_phi.at[0].set(1.0)
    sin_m_phi = sin_m_phi.at[0].set(0.0)
    if p >= 1:
        cos_m_phi = cos_m_phi.at[1].set(cos_phi)
        sin_m_phi = sin_m_phi.at[1].set(sin_phi)
    for m in range(2, p + 1):
        cos_m_phi = cos_m_phi.at[m].set(
            2.0 * cos_phi * cos_m_phi[m - 1] - cos_m_phi[m - 2]
        )
        sin_m_phi = sin_m_phi.at[m].set(
            2.0 * cos_phi * sin_m_phi[m - 1] - sin_m_phi[m - 2]
        )

    # Accumulate potential
    total = jnp.asarray(0.0, dtype=dtype)

    for n in range(p + 1):
        r_n = r**n

        for m in range(-n, n + 1):
            m_abs = abs(m)

            # Associated Legendre P_n^{|m|}(cos θ) WITHOUT
            # Condon-Shortley phase
            if m_abs == 0:
                Pmm = jnp.asarray(1.0, dtype=dtype)
            else:
                double_fact = fact_int(2 * m_abs) / ((2.0**m_abs) * fact_int(m_abs))
                Pmm = double_fact * (sin_theta**m_abs)

            if m_abs == n:
                P_nm = Pmm
            elif m_abs + 1 == n:
                P_nm = (2.0 * m_abs + 1.0) * cos_theta * Pmm
            else:
                Pnm2 = Pmm
                Pnm1 = (2.0 * m_abs + 1.0) * cos_theta * Pmm
                for k in range(m_abs + 2, n + 1):
                    numer = (2.0 * k - 1.0) * cos_theta * Pnm1
                    numer = numer - (k + m_abs - 1.0) * Pnm2
                    Pk = numer / (k - m_abs)
                    Pnm2 = Pnm1
                    Pnm1 = Pk
                P_nm = Pnm1

            # L2P uses the inner solid harmonic U_n^m (same normalization as
            # P2M).
            #
            # From Dehnen (2014) equation 2b:
            #   Υ_n^m(r) = (-1)^m r^n P_n^m(cos θ) e^{i m φ} / (n+m)!
            #
            # For real harmonics (eq 58), we use:
            #   U_n^m = r^n P_n^{|m|}(cos θ) {cos(|m|φ), sin(|m|φ)} / (n+|m|)!
            #
            # where the Condon-Shortley phase is not included in P_n^m.
            #
            # The L2P formula (eq 3a) uses Υ_n^{m*} (conjugate), but since the
            # local coefficients F_n^m are defined consistently with this
            # convention,
            # we just use U_n^m directly (the conjugate just affects
            # the complex
            # phase which is handled by the real/imaginary split).
            norm = fact_int(n + m_abs)  # (n + |m|)!
            if m >= 0:
                U_nm = r_n * P_nm * cos_m_phi[m_abs] / norm
            else:
                U_nm = r_n * P_nm * sin_m_phi[m_abs] / norm

            # φ += F_n^m * U_n^m
            idx = sh_index(n, m)
            total = total + local_coeffs[idx] * U_nm

    return total


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _evaluate_local_real_with_grad_cvjp(
    local_coeffs: Array, delta: Array, order: int
) -> tuple[Array, Array]:
    """``(local_coeffs, delta) -> (grad = ∇_δ φ, potential)`` with a custom
    reverse rule that avoids the second-order-autodiff blow-up.

    The primal is byte-identical to the ``value_and_grad`` path, so forward
    outputs (and the golden oracle) are unchanged; only the reverse pass differs.
    Under an outer ``grad`` the naive reverse differentiates ``value_and_grad``
    again -- reverse-over-reverse through the per-particle Chebyshev/Legendre/
    factorial recurrences, the dominant real-basis L2P reverse cost. The custom
    rule replaces that, per particle, with a single linear coefficient VJP (delta
    fixed, so no recurrence re-differentiation) plus one Hessian-vector product in
    delta. It is verified bit-for-bit against autodiff in
    ``tests/unit/test_custom_vjp_parity.py``.
    """

    def phi_fn(d: Array) -> Array:
        return evaluate_local_real(local_coeffs, d, order=order)

    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return grad, potential


def _evaluate_local_real_with_grad_cvjp_fwd(
    local_coeffs: Array, delta: Array, order: int
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    def phi_fn(d: Array) -> Array:
        return evaluate_local_real(local_coeffs, d, order=order)

    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return (grad, potential), (local_coeffs, delta)


def _evaluate_local_real_with_grad_cvjp_bwd(
    order: int,
    residual: tuple[Array, Array],
    cotangents: tuple[Array, Array],
) -> tuple[Array, Array]:
    local_coeffs, delta = residual
    grad_bar, potential_bar = cotangents

    # Coefficient leg: (grad, potential) is linear in the coefficients at fixed
    # delta, so this VJP is one cheap linear reverse pass (the harmonic
    # recurrences are constant w.r.t. the coefficients).
    def _out_of_coeffs(coeffs: Array) -> tuple[Array, Array]:
        def phi_fn(d: Array) -> Array:
            return evaluate_local_real(coeffs, d, order=order)

        potential, grad = jax.value_and_grad(phi_fn)(delta)
        return grad, potential

    _, vjp_coeffs = jax.vjp(_out_of_coeffs, local_coeffs)
    (local_bar,) = vjp_coeffs((grad_bar, potential_bar))

    # Delta leg: grad = ∇_δ φ, so its VJP is the Hessian-vector product
    # ∇²φ · grad_bar (forward-over-reverse), plus ∇φ · potential_bar for the
    # potential output. One HVP per particle -- no reverse-over-reverse.
    def _grad_phi(d: Array) -> Array:
        return jax.grad(lambda dd: evaluate_local_real(local_coeffs, dd, order=order))(
            d
        )

    primal_grad, hvp = jax.jvp(_grad_phi, (delta,), (grad_bar,))
    delta_bar = hvp + primal_grad * potential_bar
    return (local_bar, delta_bar)


_evaluate_local_real_with_grad_cvjp.defvjp(
    _evaluate_local_real_with_grad_cvjp_fwd,
    _evaluate_local_real_with_grad_cvjp_bwd,
)


@partial(jax.jit, static_argnames=("order",))
def evaluate_local_real_with_grad(
    local_coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> tuple[Array, Array]:
    """Evaluate local expansion and its gradient using autodiff.

    Computes both the potential and its gradient (for force/acceleration
    calculation).
    The gradient is computed via JAX autodiff of evaluate_local_real.

    **Important Convention**: delta = center - eval_point
    (NOT eval_point - center)!
    This follows the Dehnen (2014) convention from equation 3a.

    The returned gradient is ∇φ with respect to the evaluation point position.
    For gravitational acceleration, use a = -∇φ (note the sign!).

    Parameters
    ----------
    local_coeffs : Array
        Packed real local coefficients F_n^m of shape ((p+1)^2,).
    delta : Array
        (3,) vector from evaluation point TO expansion center:
        delta = center - eval_point.
        This is the Dehnen (2014) convention from equation 3a.
    order : int
        Maximum SH degree p.

    Returns
    -------
    tuple[Array, Array]
        (gradient, potential) where gradient is shape (3,) and potential is
        scalar.
        Note: gradient is ∇φ (not the acceleration -∇φ).
    """
    # Delegate to the custom_vjp-wrapped kernel: the forward is byte-identical to
    # value_and_grad(evaluate_local_real), but the reverse uses an analytic-
    # structured rule (linear coefficient VJP + Hessian-vector product) that
    # avoids the second-order-autodiff blow-up under an outer grad. The env
    # switch ``JACCPOT_ANALYTIC_L2P_VJP=0`` restores the plain-autodiff reverse
    # for A/B measurement (see examples/differentiable_fmm_overhead.py).
    if analytic_l2p_vjp_enabled():
        return _evaluate_local_real_with_grad_cvjp(
            jnp.asarray(local_coeffs), jnp.asarray(delta), int(order)
        )

    def phi_fn(d: Array) -> Array:
        return evaluate_local_real(local_coeffs, d, order=int(order))

    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return grad, potential


@partial(jax.jit, static_argnames=("order", "max_derivative_order"))
def evaluate_local_real_derivative_tower(
    local_coeffs: Array,
    delta: Array,
    *,
    order: int,
    max_derivative_order: int,
) -> Tuple[Array, ...]:
    """Potential and packed symmetric spatial-derivative tower ``D0..DK``.

    ``Dk`` holds the k-th partial derivatives of the real local expansion with
    respect to ``delta`` (= center - eval_point), stored as unique symmetric
    components in the deterministic order of
    :func:`~jaccpot.operators.symmetric_tensors.symmetric_multi_indices_3d`.
    ``D0`` has shape ``(1,)`` and holds the potential.

    This mirrors :func:`evaluate_local_complex_derivative_tower` (also autodiff
    based) so downstream L2P code can consume either basis's tower identically.
    """
    p = int(order)
    k_max = int(max_derivative_order)
    if k_max < 0:
        raise ValueError("max_derivative_order must be non-negative")

    def phi(d: Array) -> Array:
        return evaluate_local_real(local_coeffs, d, order=p)

    out: list[Array] = [jnp.reshape(phi(delta), (1,))]
    deriv_fn = phi
    for k in range(1, k_max + 1):
        deriv_fn = jax.jacfwd(deriv_fn)
        tensor = deriv_fn(delta)  # shape (3,) * k, fully symmetric
        components = []
        for nx, ny, nz in symmetric_multi_indices_3d(k):
            axes = tuple([0] * nx + [1] * ny + [2] * nz)
            components.append(tensor[axes])
        out.append(jnp.stack(components))
    return tuple(out)


@partial(jax.jit, static_argnames=("order", "max_derivative_order"))
def evaluate_local_real_derivative_tower_batch(
    local_coeffs: Array,
    deltas: Array,
    *,
    order: int,
    max_derivative_order: int,
) -> Tuple[Array, ...]:
    """Batched :func:`evaluate_local_real_derivative_tower` over ``deltas``."""
    return jax.vmap(
        lambda d: evaluate_local_real_derivative_tower(
            local_coeffs,
            d,
            order=order,
            max_derivative_order=max_derivative_order,
        )
    )(deltas)

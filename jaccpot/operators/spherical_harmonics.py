"""Real spherical-harmonics utilities for a Dehnen-style FMM backend.

This module is the foundation for a higher-order FMM implementation that
represents multipole and local expansions in a *real* spherical-harmonics
basis.

Design goals
------------
- Real-valued coefficients for lower memory / bandwidth.
- JAX-first implementation (vectorizable, JIT-friendly).
- Prepared for Dehnen (2014) Appendix A6 strategy:
  rotate -> translate along z -> rotate back.

NOTE
----
This file intentionally starts with indexing + small geometric helpers.
The actual rotation/translation operators will be implemented next.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped

from .real_harmonics import (
    compute_real_B_matrix_local,
    l2l_real,
    m2l_a6_real_only,
    p2m_real_direct,
    real_Dz_diagonal,
    translate_along_z_m2l_real,
    translate_along_z_m2m_real,
)


def sh_size(order: int) -> int:
    """Number of real SH coefficients up to degree ``order``.

    Using the standard packed layout: sum_{l=0..p} (2l+1) = (p+1)^2.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    return (p + 1) * (p + 1)


def sh_offset(ell: int) -> int:
    """Packed offset for degree ``ell`` in the (p+1)^2 layout."""

    ll = int(ell)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    return ll * ll


def sh_index(ell: int, m: int) -> int:
    """Packed index for coefficient (ell, m) for m in [-ell..ell]."""

    ll = int(ell)
    mm = int(m)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    if mm < -ll or mm > ll:
        raise ValueError("m must satisfy -ell <= m <= ell")
    return sh_offset(ll) + (mm + ll)


@jaxtyped(typechecker=beartype)
def packed_degree_slices(order: int) -> tuple[slice, ...]:
    """Return packed slices (one per l) for a given maximum degree."""

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    return tuple(slice(ell * ell, (ell + 1) * (ell + 1)) for ell in range(p + 1))


@dataclass(frozen=True)
class ZAxisRotation:
    r"""A rotation that maps a direction onto the +z axis.

    We represent a ZYZ Euler rotation (alpha, beta, gamma) such that
    D^\ell(alpha, beta, gamma) can be applied to spherical-harmonic
    coefficients.

    For the A6 strategy (rotate -> z-translate -> rotate back), we need a
    fully-specified rotation that maps the displacement direction onto the
    +z axis. Using only (alpha, beta) with gamma=0 leaves an arbitrary twist
    about z, which changes m!=0 phases and breaks translation equivalence.
    """

    alpha: Array  # scalar
    beta: Array  # scalar
    gamma: Array  # scalar


@jaxtyped(typechecker=beartype)
def _as_scalar(x: Array) -> Array:
    """Coerce a value into a scalar JAX array."""

    arr = jnp.asarray(x)
    if arr.shape != ():
        raise ValueError("expected a scalar")
    return arr


@partial(jax.jit, static_argnames=("order",))
def p2m_point_real_sh(
    delta: Array,
    mass: Array,
    *,
    order: int,
) -> Array:
    """P2M for a single point mass (Dehnen-style coefficient convention).

    Parameters
    ----------
    delta:
        (3,) vector from expansion center to particle position.
    mass:
        Particle mass.
    order:
        Maximum SH degree p.

    Returns
    -------
    Array
        Packed real tesseral multipole coefficients of shape ``((p+1)^2,)``.
        These are in the Dehnen real basis (no √2).

    Notes
    -----
    This is a real-only implementation that delegates to
    :func:`jaccpot.operators.real_harmonics.p2m_real_direct`.

    Normalization contract
    ----------------------
    The multipole coefficients returned by this function follow the
    Dehnen-style ``\\Upsilon_{\\ell m}`` normalization used throughout this
    module (cf. Dehnen 2014, Appendix A). Concretely we set

        M_{\\ell m} = mass * \\Upsilon_{\\ell m}(\\delta),

    with

                \\Upsilon_{\\ell m}(r,\\theta,\\phi)
                        = r^{\\ell} P_{\\ell}^{m}(\\cos\\theta)
                            e^{i m \\phi} / (\\ell + m)!.

    In the code this is implemented by dividing the associated Legendre
    values by the factorial (\\ell + m)! when forming the complex
    coefficients. This choice is made so that the z-axis specialised
    translation relation implemented in :func:`translate_along_z_m2m` has the
    simple polynomial coefficient form (see that function's docstring).
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    d = jnp.asarray(delta)
    if d.shape != (3,):
        raise ValueError("delta must have shape (3,)")

    mass = _as_scalar(mass)

    x, y, z = d[0], d[1], d[2]
    r = delta_norm(d)
    rho = jnp.sqrt(jnp.maximum(x * x + y * y, 1e-30))
    cos_theta = z / r
    # For rho ~ 0, define phi=0 (azimuth irrelevant on z-axis)
    phi = jnp.where(
        rho > 0,
        jnp.arctan2(y, x),
        jnp.asarray(0.0, dtype=d.dtype),
    )

    return p2m_real_direct(d, _as_scalar(mass), order=order)


@jaxtyped(typechecker=beartype)
def _real_tesseral_rotation_block(
    ell: int,
    alpha: Array,
    beta: Array,
    gamma: Array,
    *,
    real_dtype: jnp.dtype,
) -> Array:
    """Return real rotation block (2ell+1,2ell+1) using swap matrices.

    Uses the ZYZ factorization with the real swap matrix B and z-axis
    rotations only:
        R = Dz(alpha) @ B @ Dz(-beta) @ B @ Dz(gamma)
    """

    rdtype = jnp.dtype(real_dtype)
    alpha = _as_scalar(alpha)
    beta = _as_scalar(beta)
    gamma = _as_scalar(gamma)

    B = compute_real_B_matrix_local(ell, dtype=rdtype)
    Dz_alpha = real_Dz_diagonal(ell, alpha, dtype=rdtype)
    Dz_neg_beta = real_Dz_diagonal(ell, -beta, dtype=rdtype)
    Dz_gamma = real_Dz_diagonal(ell, gamma, dtype=rdtype)
    return (Dz_alpha @ B @ Dz_neg_beta @ B @ Dz_gamma).astype(rdtype)


@jaxtyped(typechecker=beartype)
def rotation_to_z(delta: Array, *, eps: float = 1e-30) -> ZAxisRotation:
    """Return angles that rotate vector ``delta`` onto the +z axis.

    Parameters
    ----------
    delta:
        Vector of shape (3,).

    Returns
    -------
    ZAxisRotation
        alpha = atan2(y, x)
        beta  = atan2(sqrt(x^2 + y^2), z)

    Notes
    -----
    This is the standard spherical coordinate mapping.
    """

    d = jnp.asarray(delta)
    if d.shape != (3,):
        raise ValueError("delta must have shape (3,)")

    x, y, z = d[0], d[1], d[2]
    rho = jnp.sqrt(jnp.maximum(x * x + y * y, eps))
    alpha = jnp.arctan2(y, x)
    beta = jnp.arctan2(rho, z)
    # Choose gamma so that the overall ZYZ rotation maps the azimuthal basis
    # consistently. A simple consistent choice for mapping a direction onto +z
    # is gamma = -alpha (so the total twist cancels).
    gamma = -alpha
    return ZAxisRotation(alpha=alpha, beta=beta, gamma=gamma)


@jaxtyped(typechecker=beartype)
def delta_norm(delta: Array, *, eps: float = 1e-30) -> Array:
    """Numerically safe Euclidean norm for a (3,) displacement."""

    d = jnp.asarray(delta)
    if d.shape != (3,):
        raise ValueError("delta must have shape (3,)")
    r2 = jnp.dot(d, d)
    return jnp.sqrt(jnp.maximum(r2, eps))


# -----------------------------------------------------------------------------
# Placeholders for the upcoming implementation (A6 rotations + z-translation)
# -----------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("order",))
def rotate_real_sh(
    coeffs: Array,
    alpha: Array,
    beta: Array,
    gamma: Array,
    *,
    order: int,
) -> Array:
    """Rotate real spherical-harmonic coefficients using swap matrices.

    Uses only real z-axis rotations and the swap matrix B per degree
    (no Wigner-D matrices).
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    coeffs_arr = jnp.asarray(coeffs)
    if coeffs_arr.ndim != 1:
        raise ValueError("coeffs must be a 1D packed coefficient vector")
    expected = sh_size(p)
    if coeffs_arr.shape[0] != expected:
        raise ValueError(f"coeffs must have shape ({expected},) for order={p}")

    alpha = _as_scalar(alpha)
    beta = _as_scalar(beta)
    gamma = _as_scalar(gamma)
    if jnp.asarray(gamma).dtype != coeffs_arr.dtype:
        gamma = gamma.astype(coeffs_arr.dtype)

    # By convention in this codebase, rotate_real_sh(coeffs, alpha, beta, gamma)
    # applies the *forward* ZYZ rotation with those Euler angles to the
    # coefficient vector: c' = R @ c. Callers that need the inverse/passive
    # transform should pass inverse angles explicitly (e.g., ZYZ(-gamma, -beta,
    # -alpha)).
    out = []
    for ell in range(p + 1):
        sl = slice(ell * ell, (ell + 1) * (ell + 1))

        Mr = coeffs_arr[sl]
        R = _real_tesseral_rotation_block(
            ell,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            real_dtype=coeffs_arr.dtype,
        )
        out.append(R @ Mr)

    return jnp.concatenate(out, axis=0)


@jaxtyped(typechecker=beartype)
def rotate_real_sh_with_rot(
    coeffs: Array,
    rot: ZAxisRotation,
    *,
    order: int,
) -> Array:
    """Convenience wrapper accepting :class:`ZAxisRotation`.

    This wrapper is not JIT-compiled itself; it forwards arrays into the
    JIT-compiled :func:`rotate_real_sh`.
    """

    return rotate_real_sh(coeffs, rot.alpha, rot.beta, rot.gamma, order=order)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2l(multipole: Array, r: Array, *, order: int) -> Array:
    """Translate a multipole expansion along +z into a local expansion.

    This is a stub. The full implementation will use Dehnen A6 recurrences for
    z-axis translation.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    multipole = jnp.asarray(multipole)
    if multipole.ndim != 1:
        raise ValueError("multipole must be a 1D packed coefficient vector")
    expected = sh_size(p)
    if multipole.shape[0] != expected:
        raise ValueError(
            f"multipole must have shape ({expected},) for order={p}",
        )

    r = _as_scalar(r)
    if jnp.asarray(r).dtype != multipole.dtype:
        r = r.astype(multipole.dtype)

    return translate_along_z_m2l_real(multipole, r, order=order)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2m(multipole: Array, r: Array, *, order: int) -> Array:
    r"""Translate a multipole expansion along +z to a new center.

    Parameters
    ----------
    multipole:
        Packed real tesseral multipole coefficients defined about the source
        center.
    r:
    Translation distance along +z.
    (new_center = old_center + (0, 0, r)).
    order:
        Maximum SH degree.

    Returns
    -------
    Array
    Packed real tesseral multipole coefficients about the shifted center.

    Notes
    -----
    This implements the Dehnen A6 z-axis translation in the complex SH
    basis and then maps back to our packed real tesseral representation.

    Important: the exact polynomial coefficient in the specialised z-
    translation depends on the chosen multipole normalization. In this
    module we adopt the Dehnen-style ``\\Upsilon_{\\ell m}`` form for the
    multipoles (see :func:`p2m_point_real_sh`), i.e. our multipoles are

        M_{\\ell m} = mass * \\Upsilon_{\\ell m}(delta),

    with \\Upsilon_{\\ell m} including a division by (\\ell+ m)!. With
    that convention the z-specialised accumulation used here becomes

        M_{n m}(z + r\\hat z) = \sum_{k=0}^{n} \frac{r^{k}}{k!} M_{n-k, m}(z).

    The implementation therefore uses coeff = r**k / k! when summing
    source degrees. This choice is consistent with the P2M construction in
    :func:`p2m_point_real_sh` and ensures translate_along_z_m2m(compose with
    P2M) == P2M(shifted position) for point masses on the z-axis.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    multipole = jnp.asarray(multipole)
    if multipole.ndim != 1:
        raise ValueError("multipole must be a 1D packed coefficient vector")
    expected = sh_size(p)
    if multipole.shape[0] != expected:
        raise ValueError(
            f"multipole must have shape ({expected},) for order={p}",
        )

    r = _as_scalar(r)
    if jnp.asarray(r).dtype != multipole.dtype:
        r = r.astype(multipole.dtype)
    # Keep the sign: multipole translation is a polynomial in the signed shift.
    r_signed = r

    return translate_along_z_m2m_real(multipole, r_signed, order=order)


@partial(jax.jit, static_argnames=("order",))
def translate_multipole_real_sh(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Translate a multipole expansion by an arbitrary displacement.

    Given coefficients about the source center, returns coefficients about a
    destination center offset by ``delta`` (destination = source + delta).
    """

    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)
    if delta.shape != (3,):
        raise ValueError("delta must have shape (3,)")

    rot = rotation_to_z(delta)
    m_rot = rotate_real_sh_with_rot(multipole, rot, order=order)
    # Signed displacement along z after rotation. rotation_to_z(delta) aligns
    # the direction of delta with +z, so the signed z-shift is ||delta||.
    # If delta is reversed, callers should pass -delta.
    r = delta_norm(delta)
    m_shift = translate_along_z_m2m(m_rot, r, order=order)
    # Inverse of ZYZ(alpha,beta,gamma) is ZYZ(-gamma,-beta,-alpha).
    rot_inv = ZAxisRotation(alpha=-rot.gamma, beta=-rot.beta, gamma=-rot.alpha)
    return rotate_real_sh_with_rot(m_shift, rot_inv, order=order)


@jaxtyped(typechecker=beartype)
def m2m_a6_real_sh(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2M using Dehnen A6: rotate → translate along z → rotate back.

    Parameters
    ----------
    multipole:
        Packed real tesseral multipole coefficients about the source center.
    delta:
        3-vector from *destination* center to *source* center.
        (So the source is located at ``dest + delta``.)
    order:
        Maximum SH degree.

    Returns
    -------
    Array
    Packed real tesseral multipole coefficients about the destination
    center.
    """

    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    # Here delta = source_center - dest_center.
    # To shift the expansion center from source -> dest we translate by
    # (dest - source) = -delta.
    return translate_multipole_real_sh(multipole, -delta, order=order)


@jaxtyped(typechecker=beartype)
def translate_local_real_sh(
    coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Translate a local expansion by an arbitrary displacement.

    Correctness-first implementation using collocation: evaluate the parent
    local expansion at a set of sample offsets expressed about the child
    center, then solve a small linear system to recover packed real tesseral
    local coefficients at the child center.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    coeffs = jnp.asarray(coeffs)
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be a 1D packed coefficient vector")
    expected = sh_size(p)
    if coeffs.shape[0] != expected:
        raise ValueError(f"coeffs must have shape ({expected},) for order={p}")

    delta = jnp.asarray(delta)
    if delta.shape != (3,):
        raise ValueError("delta must have shape (3,)")

    ncoef = expected

    def _make_samples(dtype):
        i = jnp.arange(ncoef, dtype=dtype)
        phi = 2.0 * jnp.pi * (i / ncoef)
        z = 1.0 - (2.0 * (i + 0.5) / ncoef)
        theta = jnp.arccos(z)
        sin_t = jnp.sin(theta)
        return jnp.stack(
            [sin_t * jnp.cos(phi), sin_t * jnp.sin(phi), z],
            axis=1,
        )

    def eval_at_offsets(local_coeffs: Array, offsets: Array) -> Array:
        from .real_harmonics import evaluate_local_real

        def phi_fn(vec: Array) -> Array:
            return evaluate_local_real(local_coeffs, -vec, order=p)

        return jax.vmap(phi_fn)(offsets)

    @lru_cache(maxsize=None)
    def _cached_collocation(order_key: int, dtype_name: str):
        samp = _make_samples(jnp.dtype(dtype_name))

        def basis_col(j: Array) -> Array:
            vec = jnp.zeros((ncoef,), dtype=samp.dtype)
            vec = vec.at[j].set(1.0)
            return eval_at_offsets(vec, samp)

        B = jax.vmap(basis_col)(jnp.arange(ncoef, dtype=jnp.int32)).T
        B_inv = jnp.linalg.inv(B)
        return jax.device_get(samp), jax.device_get(B_inv)

    samples_const, B_inv_const = _cached_collocation(p, str(coeffs.dtype))
    samples = jnp.asarray(samples_const, dtype=coeffs.dtype)
    B_inv = jnp.asarray(B_inv_const, dtype=coeffs.dtype)

    parent_offsets = samples + delta
    rhs = eval_at_offsets(coeffs, parent_offsets)
    x = B_inv @ rhs
    return x.astype(coeffs.dtype)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_l2l(
    local: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate a local expansion along +z by distance ``dz``.

    Analytic Dehnen (2014) eq 3e local shift along the z-axis:
        F'_n^m = sum_{k=0}^{p-n} (dz)^k / k! * F_{n+k}^m
    because only the m=0 inner solid harmonics survive for a z shift.
    """

    p = int(order)
    local = jnp.asarray(local)
    dz = jnp.asarray(dz)

    if local.shape[0] != sh_size(p):
        raise ValueError("local size does not match order")

    dtype = local.dtype
    coeff_count = sh_size(p)

    @lru_cache(maxsize=None)
    def _factorial_table_cached(p_key: int, dtype_key: str):
        dtype = jnp.dtype(dtype_key)
        n = jnp.arange(0, p_key + 1, dtype=dtype)
        return jnp.exp(jax.lax.lgamma(n + 1.0))

    fact = _factorial_table_cached(p, str(dtype))

    @lru_cache(maxsize=None)
    def _ell_m_table_cached(p_key: int):
        ell_list = []
        m_list = []
        for ell in range(p_key + 1):
            for m in range(-ell, ell + 1):
                ell_list.append(ell)
                m_list.append(m)
        return (
            jnp.asarray(ell_list, dtype=jnp.int32),
            jnp.asarray(m_list, dtype=jnp.int32),
        )

    ell_arr, m_arr = _ell_m_table_cached(p)
    ks_full = jnp.arange(p + 1, dtype=jnp.int32)

    def translate_single(idx: int) -> Array:
        ell = ell_arr[idx]
        m = m_arr[idx]

        ell_src = ell + ks_full
        local_idx = ell_src * ell_src + (m + ell_src)
        valid = ks_full <= (p - ell)
        local_vals = jnp.where(valid, local[local_idx], 0.0)

        # R_k^0(dz) = (dz)^k / k! in this normalization.
        coeffs = ((-dz) ** ks_full) / fact[ks_full]
        contrib = coeffs * local_vals
        return jnp.sum(contrib)

    out_flat = jax.vmap(translate_single)(jnp.arange(coeff_count, dtype=jnp.int32))
    return out_flat.astype(dtype)


@jaxtyped(typechecker=beartype)
def translate_local_real_sh_a6(
    coeffs: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """A6-style local-to-local: rotate→z-translate→rotate back.

    Uses the analytic z-axis local translator (Dehnen 2014 Appendix A6) for
    the middle step; rotation logic matches the existing A6 M2L/M2M wiring.
    """

    coeffs = jnp.asarray(coeffs)
    delta = jnp.asarray(delta)

    rot = rotation_to_z(delta)
    coeffs_rot = rotate_real_sh_with_rot(coeffs, rot, order=order)

    dz = delta_norm(delta)
    coeffs_z = translate_along_z_l2l(coeffs_rot, dz, order=order)

    rot_inv = ZAxisRotation(alpha=-rot.gamma, beta=-rot.beta, gamma=-rot.alpha)
    return rotate_real_sh_with_rot(coeffs_z, rot_inv, order=order)


# ===========================================================================
# Dehnen-specific real-only rotation for M2L
# ===========================================================================
# Real-only rotations are built from swap matrices and z-axis rotations:
#   D_y(β) = B @ D_z(-β) @ B
# where B is the swap matrix for coordinate permutation (x,y,z) → (z,y,x).
# ===========================================================================


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def m2l_a6_dehnen(
    multipole: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """M2L using Dehnen rotation: rotate → z-translate → rotate back.

    Delegates to the real-only Dehnen A6 implementation in
    :func:`jaccpot.operators.real_harmonics.m2l_a6_real_only`.

    Parameters
    ----------
    multipole : Array
        Packed real tesseral multipole coefficients of length ``(order+1)^2``.
    delta : Array
        3-vector from *target* center to *source* center.
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real tesseral local coefficients of length ``(order+1)^2``.
    """
    return m2l_a6_real_only(multipole, delta, order=order)


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def l2l_a6_dehnen(
    local: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """L2L using Dehnen rotation: rotate → z-translate → rotate back.

    Delegates to the real-only L2L implementation in
    :func:`jaccpot.operators.real_harmonics.l2l_real`.

    Parameters
    ----------
    local : Array
        Packed real tesseral local coefficients of length ``(order+1)^2``.
    delta : Array
        3-vector displacement from parent center to child center.
    order : int
        Maximum SH degree.

    Returns
    -------
    Array
        Packed real tesseral local coefficients at child center.
    """
    return l2l_real(local, delta, order=order)


__all__ = [
    "ZAxisRotation",
    "delta_norm",
    "packed_degree_slices",
    "rotation_to_z",
    "rotate_real_sh",
    "rotate_real_sh_with_rot",
    "l2l_a6_dehnen",
    "m2l_a6_dehnen",
    "m2m_a6_real_sh",
    "sh_index",
    "sh_offset",
    "sh_size",
    "translate_multipole_real_sh",
    "translate_along_z_m2l",
    "translate_along_z_m2m",
    "translate_along_z_l2l",
    "translate_local_real_sh_a6",
    "translate_local_real_sh",
]

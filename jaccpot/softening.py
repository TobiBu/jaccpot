"""Gravitational softening kernels: one branch-free form for every pair interaction.

A softening kernel replaces the point mass by a smooth density of unit mass. Three
are offered:

``"ferrers3"`` (the default)
    Density ``315 / (64 pi h^3) (1 - r^2/h^2)^3`` on ``r < h``: a spherical Ferrers
    profile of index 3, the "poly6" kernel of SPH. Its force is a polynomial in
    ``r^2``, so a pair costs the one reciprocal square root Plummer costs plus a
    few fused multiply-adds.
``"wendland_c2"``
    Wendland's C2 function, ``21 / (2 pi h^3) (1 - r/h)^4 (1 + 4 r/h)`` (Wendland
    1995; Dehnen & Aly 2012 for its use in SPH). Odd powers of ``r/h`` cost a square
    root per pair on top of ``"ferrers3"``; it is here as the literature reference.
``"plummer"``
    ``(r^2 + eps^2)^(-1/2)``, the historical convention. It never becomes Newtonian,
    so a far field computed with the unsoftened expansion always carries a
    softening error (memory record 2026-10-07: rel-L2 0.58 on the 2e6 disc).

Both compact kernels are exactly Newtonian beyond their support ``h``. A walk that
keeps every accepted far pair at least ``h`` apart (``softening_floor``) therefore
makes the unsoftened far field exact for them.

**``softening`` means the Plummer-equivalent length for every kernel.** The support
is ``h = support_factor(kernel) * softening``, which gives the compact kernel
Plummer's central potential ``-G m / eps``: ``h = 315/128 eps`` for ``"ferrers3"``,
``h = 3 eps`` for ``"wendland_c2"``. Switching kernels at one ``softening``
therefore keeps the force resolution. Measured on the 25M disc+bulge IC (2026-10-08),
the two compact kernels have equal mass-weighted force error against the smooth
model, about 15 % below Plummer's at each one's optimum.

**The branch-free form.** For a squared separation ``r2`` (no softening added),
with ``s = 1 / max(r, h)`` and the clipped variable ``c = min(r^2 / h^2, 1)``
(``min(r / h, 1)`` for Wendland):

* force factor ``g = G(c) s^3``: the pair acceleration is ``-G m x g``;
* potential factor ``psi = Psi(c) s``: the pair potential is ``-G m psi``;
* derivative ``(1/r) dg/dr = D(c) s^5``, for jerks and hand-written reverse rules.

Each polynomial is written as its edge value plus ``(1 - c) R(c)``. Past ``h`` the
clip makes ``1 - c`` exactly zero, so the factors are bitwise Newtonian (``s`` is
then ``rsqrt(r2)``). At ``r = 0`` they stay finite (``s = 1/h``), so a self pair
needs no special case beyond the mask the callers already apply. Plummer fits the
same shape with ``s = rsqrt(r^2 + eps^2)``, ``G = Psi = 1`` and ``D = -3``.

The coefficients were derived symbolically: enclosed mass, ``g = m(q)/q^3``,
``psi = m(q)/q + int_q^1 4 pi s W ds`` and ``g'(q)/q``. The tests in
``tests/unit/test_softening.py`` re-derive every one numerically.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import jax.numpy as jnp
import numpy as np
from jax import lax

SofteningKernel = Literal["ferrers3", "wendland_c2", "plummer"]

#: The kernel every entry point uses when none is named.
DEFAULT_SOFTENING_KERNEL: SofteningKernel = "plummer"

#: Every accepted kernel name.
SOFTENING_KERNELS: tuple[SofteningKernel, ...] = ("ferrers3", "wendland_c2", "plummer")

#: ``h / eps`` for equal central potential (``psi(0)``); 0 for Plummer, which has
#: no compact support.
_SUPPORT_FACTOR = {"ferrers3": 315.0 / 128.0, "wendland_c2": 3.0, "plummer": 0.0}

# R(c) of ``edge + (1 - c) R(c)``, ascending powers of c (sympy, see the module doc).
_FERRERS3_G = (89.0 / 16.0, -25.0 / 4.0, 35.0 / 16.0)
_FERRERS3_PSI = (187.0 / 128.0, -233.0 / 128.0, 145.0 / 128.0, -35.0 / 128.0)
_FERRERS3_D = (-165.0 / 8.0, 105.0 / 8.0)
_WENDLAND_G = (13.0, 13.0, -71.0, 69.0, -21.0)
_WENDLAND_PSI = (2.0, 2.0, -5.0, -5.0, 16.0, -12.0, 3.0)
_WENDLAND_D = (-165.0, 255.0, -105.0)
# H(c) = (1 - c) R_H(c) of d g / d(eps^2) = -(k^2 / 2) H(c) s^5 (zero past h).
_FERRERS3_H = (315.0 / 16.0, -315.0 / 8.0, 315.0 / 16.0)
_WENDLAND_H = (42.0, 42.0, -378.0, 462.0, -168.0)


def resolve_softening_kernel(kernel: Optional[str]) -> SofteningKernel:
    """Validate a kernel name; ``None`` gives :data:`DEFAULT_SOFTENING_KERNEL`.

    Parameters
    ----------
    kernel : Optional[str]
        A name from :data:`SOFTENING_KERNELS` (case-insensitive), or ``None``.

    Returns
    -------
    SofteningKernel
        The canonical name.

    Raises
    ------
    ValueError
        For any other name.
    """
    if kernel is None:
        return DEFAULT_SOFTENING_KERNEL
    name = str(kernel).strip().lower()
    if name not in SOFTENING_KERNELS:
        raise ValueError(
            f"softening_kernel={kernel!r}; expected one of {SOFTENING_KERNELS}"
        )
    return name  # type: ignore[return-value]


def support_factor(kernel: str) -> float:
    """``h / softening`` for the kernel: ``315/128``, ``3`` or ``0`` (Plummer).

    Parameters
    ----------
    kernel : str
        A kernel name.

    Returns
    -------
    float
        The support radius per unit Plummer-equivalent softening.
    """
    return _SUPPORT_FACTOR[resolve_softening_kernel(kernel)]


def support_radius(kernel: str, softening: float) -> float:
    """The kernel's support ``h``; ``0`` for Plummer.

    Parameters
    ----------
    kernel : str
        A kernel name.
    softening : float
        The Plummer-equivalent softening length.

    Returns
    -------
    float
        ``support_factor(kernel) * softening``.
    """
    return support_factor(kernel) * float(softening)


def softening_params(kernel: str, softening: Any, dtype: Any = None) -> Any:
    """The two scalars the pair factors read, as a ``(2,)`` array.

    ``[eps^2, 0]`` for Plummer, ``[h^2, 1/h^2]`` for ``"ferrers3"`` and
    ``[h^2, 1/h]`` for ``"wendland_c2"``. ``softening = 0`` gives Newton for every
    kernel (``1/h`` is then ``inf``, which the clip absorbs for ``r > 0``).

    Parameters
    ----------
    kernel : str
        A kernel name.
    softening : Any
        The Plummer-equivalent softening length (a float or a 0-d array).
    dtype : Any
        The array dtype; ``None`` keeps jax's default.

    Returns
    -------
    Any
        A ``(2,)`` jax array.
    """
    name = resolve_softening_kernel(kernel)
    eps = jnp.asarray(softening, dtype=dtype)
    if name == "plummer":
        return jnp.stack([eps * eps, jnp.zeros_like(eps)])
    h = jnp.asarray(_SUPPORT_FACTOR[name], dtype=eps.dtype) * eps
    if name == "ferrers3":
        return jnp.stack([h * h, 1.0 / (h * h)])
    return jnp.stack([h * h, 1.0 / h])


def softening_params_np(kernel: str, softening: float) -> np.ndarray:
    """:func:`softening_params` in float64 NumPy, for host-side references.

    Parameters
    ----------
    kernel : str
        A kernel name.
    softening : float
        The Plummer-equivalent softening length.

    Returns
    -------
    numpy.ndarray
        ``(2,)`` float64.
    """
    name = resolve_softening_kernel(kernel)
    eps = float(softening)
    if name == "plummer":
        return np.array([eps * eps, 0.0])
    h = _SUPPORT_FACTOR[name] * eps
    inv = np.inf if h == 0.0 else (1.0 / (h * h) if name == "ferrers3" else 1.0 / h)
    return np.array([h * h, inv])


def _horner(c: Any, coeffs: tuple[float, ...]) -> Any:
    out = coeffs[-1]
    for a in coeffs[-2::-1]:
        out = a + c * out
    return out


def pair_factors(
    r2: Any,
    params: Any,
    kernel: str,
    *,
    potential: bool = False,
    derivative: bool = False,
    xp: Any = jnp,
) -> tuple[Any, Optional[Any], Optional[Any]]:
    """Force factor ``g``, and optionally ``psi`` and ``(1/r) dg/dr``, for ``r2``.

    Elementwise and branch-free, so it runs unchanged inside Pallas kernels, in
    XLA and (``xp=numpy``) on the host. ``kernel`` is static.

    Parameters
    ----------
    r2 : Any
        Squared separations, softening NOT added. Callers mask inactive pairs as
        before (replace ``r2`` by a positive dummy and zero the result).
    params : Any
        ``(p0, p1)`` from :func:`softening_params`; indexable, so a ``(2,)``
        array or a Pallas ref.
    kernel : str
        A kernel name (static).
    potential : bool
        Also return the potential factor ``psi``.
    derivative : bool
        Also return ``(1/r) dg/dr``.
    xp : Any
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    tuple
        ``(g, psi or None, dg or None)``.
    """
    name = resolve_softening_kernel(kernel)
    p0, p1 = params[0], params[1]
    if xp is jnp:
        rsqrt = lax.rsqrt
    else:

        def rsqrt(x: Any) -> Any:
            return 1.0 / xp.sqrt(x)

    if name == "plummer":
        s = rsqrt(r2 + p0)
        s3 = s * s * s
        psi = s if potential else None
        dg = -3.0 * s3 * s * s if derivative else None
        return s3, psi, dg
    s = rsqrt(xp.maximum(r2, p0))
    s3 = s * s * s
    if name == "ferrers3":
        c = xp.minimum(r2 * p1, 1.0)
        rg, rpsi, rd = _FERRERS3_G, _FERRERS3_PSI, _FERRERS3_D
    else:
        c = xp.minimum(xp.sqrt(r2) * p1, 1.0)
        rg, rpsi, rd = _WENDLAND_G, _WENDLAND_PSI, _WENDLAND_D
    one_minus = 1.0 - c
    g = (1.0 + one_minus * _horner(c, rg)) * s3
    psi = (1.0 + one_minus * _horner(c, rpsi)) * s if potential else None
    dg = (-3.0 + one_minus * _horner(c, rd)) * s3 * s * s if derivative else None
    return g, psi, dg


def pair_softening_sq_derivative(
    r2: Any, params: Any, kernel: str, *, xp: Any = jnp
) -> Any:
    """``d g / d(eps^2)`` at fixed separation, for reverse rules that return it.

    Plummer: ``-1.5 (r^2 + eps^2)^(-5/2)``. Compact kernels: ``-(k^2/2) H(c) s^5``
    with ``k = support_factor(kernel)``; ``H`` vanishes past ``h``, where the force
    does not depend on the softening. Same arguments as :func:`pair_factors`.

    Parameters
    ----------
    r2 : Any
        Squared separations, softening NOT added.
    params : Any
        From :func:`softening_params` / :func:`softening_params_from_sq`.
    kernel : str
        A kernel name (static).
    xp : Any
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    Any
        The derivative, elementwise.
    """
    name = resolve_softening_kernel(kernel)
    p0, p1 = params[0], params[1]
    if xp is jnp:
        rsqrt = lax.rsqrt
    else:

        def rsqrt(x: Any) -> Any:
            return 1.0 / xp.sqrt(x)

    if name == "plummer":
        s = rsqrt(r2 + p0)
        return -1.5 * s * s * s * s * s
    s = rsqrt(xp.maximum(r2, p0))
    if name == "ferrers3":
        c = xp.minimum(r2 * p1, 1.0)
        rh = _FERRERS3_H
    else:
        c = xp.minimum(xp.sqrt(r2) * p1, 1.0)
        rh = _WENDLAND_H
    k = _SUPPORT_FACTOR[name]
    return -0.5 * k * k * (1.0 - c) * _horner(c, rh) * s * s * s * s * s


def softening_params_from_sq(kernel: str, softening_sq: Any, dtype: Any = None) -> Any:
    """:func:`softening_params` from a SQUARED softening, as the kernels carry it.

    Plummer keeps ``softening_sq`` itself as ``p0``, so its pair arithmetic is
    unchanged bit for bit; a compact kernel takes ``h`` from ``sqrt``.

    Parameters
    ----------
    kernel : str
        A kernel name.
    softening_sq : Any
        The squared Plummer-equivalent softening.
    dtype : Any
        The array dtype; ``None`` keeps that of ``softening_sq``.

    Returns
    -------
    Any
        A ``(2,)`` jax array.
    """
    name = resolve_softening_kernel(kernel)
    sq = jnp.asarray(softening_sq, dtype=dtype)
    if name == "plummer":
        return jnp.stack([sq, jnp.zeros_like(sq)])
    return softening_params(name, jnp.sqrt(sq), sq.dtype)


def masked_pair_factors(
    r2: Any,
    active: Any,
    params: Any,
    kernel: str,
    *,
    potential: bool = False,
    derivative: bool = False,
    xp: Any = jnp,
) -> tuple[Any, Optional[Any], Optional[Any]]:
    """:func:`pair_factors` with the near-field kernels' mask: zero where inactive.

    The Plummer branch is the kernels' historical sequence, op for op: ``r2 +
    eps^2``, a ``1`` substituted where inactive, ``rsqrt``, the result zeroed. That
    keeps every Plummer force bitwise what it was. Compact kernels substitute ``1``
    for ``r2`` (any positive value works) and zero the factors.

    Parameters
    ----------
    r2 : Any
        Squared separations, softening NOT added.
    active : Any
        Boolean mask of live pairs, broadcastable to ``r2``.
    params : Any
        From :func:`softening_params` / :func:`softening_params_from_sq`.
    kernel : str
        A kernel name (static).
    potential : bool
        Also return ``psi``.
    derivative : bool
        Also return ``(1/r) dg/dr``.
    xp : Any
        ``jax.numpy`` (default) or ``numpy``.

    Returns
    -------
    tuple
        ``(g, psi or None, dg or None)``, zero on inactive pairs.
    """
    name = resolve_softening_kernel(kernel)
    if name == "plummer":
        safe = xp.where(active, r2 + params[0], 1.0)
        s = lax.rsqrt(safe) if xp is jnp else 1.0 / xp.sqrt(safe)
        s = xp.where(active, s, 0.0)
        s3 = s * s * s
        return (
            s3,
            s if potential else None,
            -3.0 * s3 * s * s if derivative else None,
        )
    safe = xp.where(active, r2, 1.0)
    g, psi, dg = pair_factors(
        safe, params, name, potential=potential, derivative=derivative, xp=xp
    )
    g = xp.where(active, g, 0.0)
    psi = xp.where(active, psi, 0.0) if potential else None
    dg = xp.where(active, dg, 0.0) if derivative else None
    return g, psi, dg


__all__ = [
    "DEFAULT_SOFTENING_KERNEL",
    "SOFTENING_KERNELS",
    "SofteningKernel",
    "masked_pair_factors",
    "pair_factors",
    "pair_softening_sq_derivative",
    "resolve_softening_kernel",
    "softening_params",
    "softening_params_from_sq",
    "softening_params_np",
    "support_factor",
    "support_radius",
]

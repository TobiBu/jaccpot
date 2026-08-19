"""Packing layout and shared scalar primitives for the real solid harmonics.

The three ``sh_*`` functions define the **packed layout** every real-basis array
in this package uses: degree ``ell`` occupies the contiguous slot range
``[sh_offset(ell), sh_offset(ell) + 2*ell + 1)`` with ``m`` running
``-ell .. +ell``. Everything that indexes a ``[(p+1)^2]`` coefficient vector goes
through them, which is why they sit in their own leaf module rather than beside
any one operator: P2M, L2P, the Dehnen ``Q`` conversion, the rotation generators
and the translation tables are all consumers.

The two array helpers here are shared primitives for the same reason.
``_factorial_table_jax`` is used by P2M and by both z-translations;
``_azimuth_from_floored_rho`` carries the floored-``rho`` technique that fixed
the silently-zeroed transverse gradient in L2P and P2M (`d5cb13b`) and its
docstring is the reference for why the division is kept rather than branched
away.

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike

__all__ = ["sh_size", "sh_offset", "sh_index"]


# ===========================================================================
# Index utilities
# ===========================================================================


def sh_size(order: int) -> int:
    """Number of real SH coefficients up to degree ``order``: (p+1)^2.

    Parameters
    ----------
    order : int
        Maximum harmonic degree p.

    Returns
    -------
    int
        Total number of coefficients: (p+1)^2.

    Raises
    ------
    ValueError
        If ``order`` is negative. A pure-Python host-side check on a static
        value; nothing here is traced.
    """
    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    return (p + 1) * (p + 1)


def sh_offset(ell: int) -> int:
    """Packed offset for degree ``ell`` in the (p+1)^2 layout.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    int
        Starting index for degree ell: ell^2.

    Raises
    ------
    ValueError
        If ``ell`` is negative. A pure-Python host-side check on a static value.
    """
    ll = int(ell)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    return ll * ll


def sh_index(ell: int, m: int) -> int:
    """Packed index for coefficient (ell, m) for m in [-ell..ell].

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    m : int
        Spherical harmonic order, must satisfy -ell <= m <= ell.

    Returns
    -------
    int
        Linear index in the packed coefficient array.

    Raises
    ------
    ValueError
        If ``ell`` is negative, or ``m`` falls outside ``[-ell, ell]``. Both are
        pure-Python host-side checks on static values.
    """
    ll = int(ell)
    mm = int(m)
    if ll < 0:
        raise ValueError("ell must be >= 0")
    if mm < -ll or mm > ll:
        raise ValueError("m must satisfy -ell <= m <= ell")
    return sh_offset(ll) + (mm + ll)


# ===========================================================================
# Direct real harmonic evaluation (no complex arithmetic)
# ===========================================================================


def _factorial_table_jax(max_n: int, dtype: DTypeLike) -> Array:
    """Tabulate ``0!`` through ``max_n!`` as ``exp(lgamma(n + 1))``.

    Computed in floating point rather than exactly, so the entries are *not*
    integral beyond ``n = 1``: measured against :func:`math.factorial` up to
    ``n = 20``, the relative error peaks at 1.4e-14 in float64 and 4.4e-06 in
    float32 (``2!`` already comes out as ``1.9999999999999976`` in float64).
    That is well inside the accuracy the expansion algebra targets, but it means
    the table must not be used for exact index arithmetic or equality tests.

    Parameters
    ----------
    max_n : int
        Largest factorial to tabulate. Static under ``jit`` — it sets the array
        length. ``20!`` is the last value exactly representable in float64 range
        terms; well before that, float32 overflows to ``inf`` around ``35!``.
    dtype : DTypeLike
        Floating dtype of both the ``arange`` and the result. A real float dtype
        is required: ``lgamma`` is defined for floats only.

    Returns
    -------
    Array
        Shape ``[max_n + 1]``, ``dtype``, with entry ``n`` holding ``n!``.
    """
    n = jnp.arange(0, max_n + 1, dtype=dtype)
    return jnp.exp(jax.lax.lgamma(n + 1.0)).astype(dtype)


def _azimuth_from_floored_rho(x: Array, y: Array, rho: Array) -> tuple[Array, Array]:
    """``(cos φ, sin φ) = (x/ρ, y/ρ)`` with NO degenerate branch.

    ``rho`` must be the *floored* cylindrical radius
    ``sqrt(floor_squared_radius(x*x + y*y))``, which is a normal number in every
    supported dtype (:func:`~jaccpot.operators.dtypes.squared_radius_floor` is
    dtype-aware), so the division is always finite. ``|x|, |y| <= ρ`` also holds
    when the floor bites, so both results stay in ``[-1, 1]`` and the Chebyshev
    ``cos(mφ)``/``sin(mφ)`` recurrences downstream stay bounded.

    Deliberately branch-free. Selecting a *constant* ``(cos φ, sin φ) = (1, 0)``
    where ``ρ`` hits the floor -- what this used to do -- is harmless in the
    forward pass, because every ``|m| >= 1`` term carries a ``sin^|m| θ = (ρ/r)^|m|``
    factor that annihilates the arbitrary azimuth. Under ``jax.grad`` it is not: a
    constant has no ``x``/``y`` derivative, so the *entire* transverse gradient of
    the ``m != 0`` terms was dropped, silently zeroing the x and y components of
    the far-field force for any particle sitting exactly at its expansion centre
    (guaranteed for a one-particle leaf) or exactly on that centre's z axis.

    Keeping the division lets the limit come out of the algebra instead of being
    branched away. The floored ``ρ`` cancels analytically:
    ``sin θ cos φ = (ρ/r)(x/ρ) = x/r``, so e.g. ``U_1^1 = r sin θ cos φ / 2!``
    differentiates to ``(1/2, 0, 0)`` at ``delta == 0`` as it must. For ``|m| >= 2``
    the ``O(1/ρ_floor)`` derivative of ``cos(mφ)`` is multiplied by
    ``(ρ_floor/r)^|m|`` and vanishes -- which is also the true limit, since those
    terms are products of at least two coordinates.

    Parameters
    ----------
    x : Array
        Cartesian x displacement from the expansion centre, any shape.
    y : Array
        Cartesian y displacement, broadcastable against ``x``.
    rho : Array
        The **floored** cylindrical radius. Passing the raw
        ``sqrt(x*x + y*y)`` instead is the way to get this wrong: it reintroduces
        the ``0/0`` at ``x == y == 0`` that the floor exists to remove, and NaN
        from here propagates through every ``m != 0`` term.

    Returns
    -------
    Array
        ``cos φ = x / ρ``, in ``[-1, 1]``.
    Array
        ``sin φ = y / ρ``, in ``[-1, 1]``.
    """
    return x / rho, y / rho

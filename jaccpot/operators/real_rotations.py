"""Closed-form real-basis rotations: ``D_z``, the generators, align-to-z.

Everything needed to turn an arbitrary displacement into the ``+z``-aligned frame
where the M2L / M2M / L2L translation is a cheap z-shift, and back again. The
rotation is built as ``B_U @ Dz(-ax) @ B_U @ Dz(az)`` from the involutory ``B``
of :mod:`jaccpot.operators.real_dehnen_q` and the diagonal ``D_z`` here.

CONVENTION (CRITICAL). The coded ``B`` is the **x <-> z** swap, so
``B @ Dz @ B`` rotates about *x* and the alignment azimuth is ``atan2(x, y)``,
not ``atan2(y, x)``. That is the documented historical bug site; it is now a
tested fact (see the note above the builders).

The ``real_transverse_generators`` family supplies the representation generators
that :mod:`jaccpot.operators._transverse_degeneracy_jvp` needs to reconstruct the
transverse derivative on the ``rho == 0`` axis, where the individual alignment
block is genuinely direction-dependent but the assembled cascade is not (G.10).

Split out of ``real_harmonics.py`` (Tier 1.3); the mathematics is unchanged and
``real_harmonics`` re-exports the public names.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike

from ._precision import highest_matmul_precision
from ._sh_indexing import sh_offset, sh_size
from ._transverse_degeneracy_jvp import TransverseGenerators
from .real_dehnen_q import compute_real_B_matrix_multipole

__all__ = [
    "real_Dz_diagonal",
    "real_transverse_generators",
    "real_rotation_to_z_axis_multipole",
    "real_rotation_to_z_axis_local",
    "real_rotation_from_z_axis_local",
    "real_rotation_from_z_axis_multipole",
]


# ===========================================================================
# Real rotation via B @ D_z @ B
# ===========================================================================
#
# These closed-form Dehnen builders are the only rotation path. A SymPy Wigner-D
# baseline used to sit alongside them as a "correctness reference"; it was removed
# because it never actually checked anything (nothing called it) and it could not
# have: it imported `sympy`, which is not a dependency of this package, so every
# entry point raised `ImportError`. The Wigner route is also the slow one -- the
# whole point of the B @ D_z @ B decomposition is to avoid it.
#
# What replaces it is stronger, because it tests against physics rather than
# against a second implementation that could share a convention error:
# `tests/unit/operators/test_real_harmonics.py::test_multipole_rotation_blocks_match_p2m_of_the_rotated_source`
# asserts `D_to @ p2m(s) == p2m(g @ s)` for the physical rotation `g`, and
# `::test_local_rotation_blocks_leave_the_evaluated_potential_invariant` asserts
# that rotating a local expansion and its evaluation point cancels exactly.
# Measured agreement ~2.5e-15 (~10 eps_f64); writing the alignment azimuth as
# `atan2(y, x)` instead of the `atan2(x, y)` flagged CRITICAL below fails both at
# 1.8e+00.


def real_Dz_diagonal(ell: int, angle: Array, *, dtype: DTypeLike) -> Array:
    """Diagonal D_z rotation for REAL harmonics.

    For real harmonics, the z-rotation is block-diagonal:
    - m = 0: unchanged (coefficient 1)
    - m > 0: 2x2 rotation block [[cos(mα), -sin(mα)], [sin(mα), cos(mα)]]
             acting on (r_{+m}, r_{-m}) = (cos channel, sin channel)

    For the packed layout m = -ell..ell, think of it as:
    r'_m = cos(m*α) * r_m + sin(m*α) * r_{-m} for appropriate signs.

    For each m, the coefficient transforms as if rotating the angular part:
    cos(mφ) → cos(m(φ+α)) = cos(mφ)cos(mα) - sin(mφ)sin(mα).

    Parameters
    ----------
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    angle : Array
        Rotation angle about ``+z``, in radians.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` z-rotation block for degree ``l``. Real, and sparse
        by construction -- a z-rotation does not mix different ``|m|``.
    """
    n = 2 * ell + 1
    D = jnp.zeros((n, n), dtype=dtype)

    # For real harmonics with m=0: unchanged
    D = D.at[ell, ell].set(1.0)

    for m in range(1, ell + 1):
        c = jnp.cos(m * angle)
        s = jnp.sin(m * angle)

        # Indices in packed layout
        ip = ell + m  # +m (cos channel)
        im = ell - m  # -m (sin channel)

        # cos(m(φ+α)) = cos(mφ)cos(mα) - sin(mφ)sin(mα)
        # sin(m(φ+α)) = sin(mφ)cos(mα) + cos(mφ)sin(mα)
        # So: r'_{+m} = cos(mα) * r_{+m} - sin(mα) * r_{-m}
        #     r'_{-m} = sin(mα) * r_{+m} + cos(mα) * r_{-m}
        D = D.at[ip, ip].set(c)
        D = D.at[ip, im].set(-s)
        D = D.at[im, ip].set(s)
        D = D.at[im, im].set(c)

    return D


# --------------------------------------------------------------------------
# Rotation generators, for the analytic transverse derivative at rho == 0.
# --------------------------------------------------------------------------
#
# These are ``d/dtheta D(R_a(theta))`` at ``theta == 0`` for the real-basis
# representations, and they exist for exactly one purpose: to supply the derivative
# the azimuth guards below cannot produce. See
# :mod:`jaccpot.operators._transverse_degeneracy_jvp` for how they are used and
# ``docs/rotation_degeneracy_derivative.md`` for the derivation.
#
# Every sign here was calibrated against an identity this repository already
# verifies, not derived on paper, because the plausible alternatives are all wrong by
# O(1) and all look right: ``Dz(ell, -theta)`` for the z-rotation gives 6.6e-02,
# ``+B_U Lambda B_U`` for the x-rotation gives 2.5e-01, and taking the local
# representation to be ``D^M`` or ``D^M^T`` instead of ``D^M^-T`` gives 4.1e-01 and
# 7.5e-02. The residuals of the choices coded below are 2.4e-11 (x) and 5.2e-11 (y)
# against central differences of ``p2m(R_a(theta) v)``.


@lru_cache(maxsize=None)
def _real_z_rotation_generator(ell: int) -> np.ndarray:
    """``d/dangle`` of :func:`real_Dz_diagonal` at ``angle == 0``, degree ``ell``.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` float64. Nonzero only on the two entries per
        ``|m| >= 1`` that couple the cos and sin channels, since that is the only
        place :func:`real_Dz_diagonal`'s ``sin(m * angle)`` appears.
    """
    width = 2 * ell + 1
    generator = np.zeros((width, width), dtype=np.float64)
    for m in range(1, ell + 1):
        generator[ell + m, ell - m] = -float(m)
        generator[ell - m, ell + m] = +float(m)
    return generator


@lru_cache(maxsize=None)
@highest_matmul_precision
def _real_rotation_generator_block(
    ell: int, axis: str, representation: str
) -> np.ndarray:
    """Generator of rotation about ``axis`` for one degree, in one representation.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    axis : str
        ``'x'`` or ``'y'``. The x-generator comes from conjugating the z-generator
        with the involutory x<->z swap ``B_U`` -- the same convention
        :func:`_multipole_align_to_z_block` relies on -- and the y-generator is the
        x-generator conjugated by a quarter turn about z.
    representation : str
        ``'multipole'`` or ``'local'``. Local coefficients contract against the same
        regular harmonics P2M builds, so they transform **contragrediently**:
        ``D^L = D^M^-T``, hence ``G^L = -(G^M)^T``.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` float64.

    Raises
    ------
    ValueError
        If ``axis`` or ``representation`` is not one of the listed values.
    """
    # The matmuls below are numpy, in float64, so the pinned precision is inert
    # here -- the decorator is on for policy conformance
    # (tests/unit/operators/test_matmul_precision_pinned.py) rather than because
    # this function could drop to TF32.
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if representation not in ("multipole", "local"):
        raise ValueError(
            f"representation must be 'multipole' or 'local', got {representation!r}"
        )
    # The generators are built from this module's OWN rotation builders rather than
    # from a second closed form, because agreeing with those builders is the whole
    # calibration -- a private numpy copy of the quarter turn could drift from
    # ``real_Dz_diagonal`` and the resulting gradient error would be invisible in the
    # forward pass. Both builders return ``jnp`` arrays, and under ``jax.jit``
    # ``jnp.asarray`` of a numpy constant is a *tracer*, so pulling them back to numpy
    # (which is what lets the result be ``lru_cache``d into a compile-time constant)
    # needs the constant-folding context. This function is called from a ``custom_jvp``
    # rule, i.e. always inside a trace.
    with jax.ensure_compile_time_eval():
        B_U = np.asarray(compute_real_B_matrix_multipole(ell, dtype=jnp.float64))
        generator = -B_U @ _real_z_rotation_generator(ell) @ B_U
        if axis == "y":
            quarter = np.asarray(
                real_Dz_diagonal(ell, jnp.asarray(np.pi / 2.0), dtype=jnp.float64)
            )
            quarter_back = np.asarray(
                real_Dz_diagonal(ell, jnp.asarray(-np.pi / 2.0), dtype=jnp.float64)
            )
            generator = quarter @ generator @ quarter_back
    if representation == "local":
        generator = -generator.T
    return generator


@lru_cache(maxsize=None)
def _real_transverse_generator_packed(
    order: int, axis: str, representation: str
) -> np.ndarray:
    """Per-degree generator blocks assembled into one packed square matrix.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    axis : str
        ``'x'`` or ``'y'``, as in :func:`_real_rotation_generator_block`.
    representation : str
        ``'multipole'`` or ``'local'``, as in :func:`_real_rotation_generator_block`.

    Returns
    -------
    np.ndarray
        ``[(p+1)^2, (p+1)^2]`` float64, block-diagonal in ``ell`` with the packing
        of :func:`sh_offset`.
    """
    p = int(order)
    packed = np.zeros((sh_size(p), sh_size(p)), dtype=np.float64)
    for ell in range(p + 1):
        block = slice(sh_offset(ell), sh_offset(ell + 1))
        packed[block, block] = _real_rotation_generator_block(ell, axis, representation)
    return packed


def real_transverse_generators(
    order: int,
    dtype: DTypeLike,
    *,
    in_representation: str,
    out_representation: str,
) -> TransverseGenerators:
    """Real-basis generators for the ``rho == 0`` transverse derivative.

    Feeds :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`,
    which documents what the four matrices are for.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    dtype : DTypeLike
        Working dtype of the coefficients. The generators are built in float64 and
        cast down, for the same reason the B matrices are (see
        :func:`compute_real_B_matrix_multipole`): so the generator matmuls run in the
        working dtype instead of promoting float32 coefficients to float64.
    in_representation : str
        ``'multipole'`` or ``'local'`` -- which slot the operator's input occupies.
    out_representation : str
        Likewise for its output. M2L is multipole in, local out; M2M is multipole to
        multipole; L2L is local to local.

    Returns
    -------
    TransverseGenerators
        The four ``[(p+1)^2, (p+1)^2]`` packed generators, in ``dtype``.
    """
    return TransverseGenerators(
        in_x=jnp.asarray(
            _real_transverse_generator_packed(order, "x", in_representation),
            dtype=dtype,
        ),
        in_y=jnp.asarray(
            _real_transverse_generator_packed(order, "y", in_representation),
            dtype=dtype,
        ),
        out_x=jnp.asarray(
            _real_transverse_generator_packed(order, "x", out_representation),
            dtype=dtype,
        ),
        out_y=jnp.asarray(
            _real_transverse_generator_packed(order, "y", out_representation),
            dtype=dtype,
        ),
    )


#: :func:`real_transverse_generators` bound to each cascade operator's pair of
#: representations, ready to hand to
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`.
_M2L_TRANSVERSE_GENERATORS = partial(
    real_transverse_generators,
    in_representation="multipole",
    out_representation="local",
)
_M2M_TRANSVERSE_GENERATORS = partial(
    real_transverse_generators,
    in_representation="multipole",
    out_representation="multipole",
)
_L2L_TRANSVERSE_GENERATORS = partial(
    real_transverse_generators,
    in_representation="local",
    out_representation="local",
)


def _alignment_angles(x: Array, y: Array, z: Array) -> Tuple[Array, Array]:
    """The two NaN-safe angles that align ``(x, y, z)`` with ``+z``.

    Shared verbatim by :func:`_multipole_align_to_z_block` and
    :func:`_multipole_align_from_z_block`, which differ only in the order they
    assemble the same two rotations from these angles. Not decorated: it is
    called from inside bodies that already carry
    :func:`~jaccpot.operators._precision.highest_matmul_precision`, and it
    contains no matmul of its own.

    Parameters
    ----------
    x : Array
        First Cartesian component of the direction to align.
    y : Array
        Second Cartesian component. Broadcasts against ``x`` and ``z``.
    z : Array
        Third Cartesian component, the axis the direction is turned onto.

    Returns
    -------
    az : Array
        Azimuth about ``z`` that removes the *x*-component, ``atan2(x, y)``.
        See the convention note on :func:`_multipole_align_to_z_block` for why
        it is ``atan2(x, y)`` and not ``atan2(y, x)``.
    ax : Array
        Polar tilt about ``x``, ``atan2(rho, z)`` with
        ``rho = sqrt(x^2 + y^2)``.
    """
    # NaN-safe (double-where) alignment angles. ``sqrt`` (infinite reverse grad
    # at 0) and ``arctan2`` (0/0 grad at the origin) would inject NaNs into the
    # real-basis M2L/L2L reverse pass at the degenerate directions the
    # fixed-topology FMM hits: zero displacement (single-child COM L2L pairs) and
    # z-axis-aligned displacement (rho == 0, lattice-aligned M2L pairs). Forward
    # values are unchanged (arctan2(0,0)=0, sqrt(0)=0), so the golden oracle stays
    # byte-stable.
    #
    # WARNING: THE GUARDS ARE NOT GRADIENT-CORRECT, and this is deliberate -- the
    # missing derivative is supplied one level up rather than here. Do not try to fix
    # it at this site. Measured on the assembled cascade at z=2.5, grad w.r.t. the
    # displacement, limit taken from eight approach directions at rho=1e-9:
    #
    #     m2l_real  true (-1.502050, -0.523434, +0.834153)  returned (0, 0, +0.834153)
    #     m2m_real  true (-6.416905, +1.769043, -9.651272)  returned (0, 0, -9.651272)
    #     l2l_real  true (+0.305315, +0.003498, +0.072012)  returned (0, 0, +0.072012)
    #
    # The radial component is right; both transverse components are lost, and the
    # cascade genuinely IS differentiable here (the limit is direction-independent to
    # ~1e-07), so those zeros are wrong rather than a defensible subgradient.
    #
    # Why no guard tweak fixes it: the code reaches ``(x, y)`` only through
    # ``rho = sqrt(x^2 + y^2)`` and ``az = atan2(x, y)``, so at ``x == y == 0`` every
    # chain-rule route carries a factor ``x / rho`` or ``y / rho^2``. Flooring rho
    # (the `_azimuth_from_floored_rho` trick that fixed the same defect class in
    # L2P/P2M, `d5cb13b`) makes those exactly 0; leaving them bare makes them NaN.
    # Neither can produce the ``O(rho)`` coefficient the polar parametrisation has
    # already divided out.
    #
    # RESOLVED at the cascade level (G.10). ``m2l_a6_real_only``, ``m2m_real``,
    # ``l2l_real`` and the production ``m2l_rot_scale_real_batch`` each carry a
    # ``custom_jvp`` that supplies the transverse derivative analytically, from the
    # rotational covariance of the assembled operator -- which is available there and
    # not here, because the individual alignment block is genuinely
    # direction-dependent at rho == 0 while their product is not. See
    # :mod:`jaccpot.operators._transverse_degeneracy_jvp` and
    # ``docs/rotation_degeneracy_derivative.md``. Asserted by
    # ``test_rotation_cascade_transverse_gradient_at_rho_zero``.
    rho_sq = x * x + y * y
    rho_pos = rho_sq > 0
    rho = jnp.where(rho_pos, jnp.sqrt(jnp.where(rho_pos, rho_sq, 1.0)), 0.0)
    az = jnp.where(rho_pos, jnp.arctan2(jnp.where(rho_pos, x, 1.0), y), 0.0)
    r_pos = (rho_sq + z * z) > 0
    ax = jnp.where(r_pos, jnp.arctan2(rho, jnp.where(r_pos, z, 1.0)), 0.0)
    return az, ax


@highest_matmul_precision
def _multipole_align_to_z_block(
    x: Array, y: Array, z: Array, ell: int, *, dtype: DTypeLike
) -> Array:
    """Degree-``ell`` block that rotates a MULTIPOLE from the world frame into
    the frame where the direction ``(x, y, z)`` points along ``+z``.

    Verified identity (to machine precision):
        (this block) @ p2m(s)[block] == p2m(g @ s)[block]
    where ``g`` is the physical rotation with ``g @ (x,y,z)/|.| == +z_hat``.

    Convention (CRITICAL): the coded ``B`` matrix
    (:func:`_compute_dehnen_B_matrix_complex`) is the **x <-> z** swap, so
    ``B @ Dz(theta) @ B`` is a rotation about the *x* axis. Aligning
    ``(x, y, z)`` with ``+z`` therefore requires the azimuth that removes the
    *x*-component, ``az = atan2(x, y)`` (not ``atan2(y, x)``, which suits a
    y-rotation swap), followed by the polar tilt about x,
    ``ax = atan2(rho, z)``. In coordinate space ``g = Rx(ax) @ Rz(az)`` whose
    multipole representation is ``B_U @ Dz(-ax) @ B_U @ Dz(az)``.

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` world->z alignment block for a multipole.
    """
    # The guards inside :func:`_alignment_angles` are deliberately NOT
    # gradient-correct at rho == 0; read the WARNING there before touching them.
    az, ax = _alignment_angles(x, y, z)
    B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)
    return (
        B_U
        @ real_Dz_diagonal(ell, -ax, dtype=dtype)
        @ B_U
        @ real_Dz_diagonal(ell, az, dtype=dtype)
    )


@highest_matmul_precision
def _multipole_align_from_z_block(
    x: Array, y: Array, z: Array, ell: int, *, dtype: DTypeLike
) -> Array:
    """Inverse of :func:`_multipole_align_to_z_block` (multipole z-frame ->
    world). Equals ``Dz(-az) @ B_U @ Dz(ax) @ B_U`` with the same angles.

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` z->world block, the inverse of
        :func:`_multipole_align_to_z_block` at the same direction.
    """
    # The guards inside :func:`_alignment_angles` are deliberately NOT
    # gradient-correct at rho == 0; read the WARNING there before touching them.
    az, ax = _alignment_angles(x, y, z)
    B_U = compute_real_B_matrix_multipole(ell, dtype=dtype)
    return (
        real_Dz_diagonal(ell, -az, dtype=dtype)
        @ B_U
        @ real_Dz_diagonal(ell, ax, dtype=dtype)
        @ B_U
    )


def real_rotation_to_z_axis_multipole(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Rotation that aligns the vector (x,y,z) with the +z axis for MULTIPOLES.

    Use this to rotate multipole expansion coefficients U_n^m into the
    z-aligned frame. After applying ``D @ M``, the multipole expansion is
    expressed in the frame where the original ``(x, y, z)`` direction lies on
    ``+z``.

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` rotation taking a world-frame multipole into the
        z-aligned frame.
    """
    return _multipole_align_to_z_block(x, y, z, ell, dtype=dtype)


def real_rotation_from_z_axis_local(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Rotation for LOCAL expansions from the z-aligned frame back to world.

    Because :func:`evaluate_local_real` contracts local coefficients against
    the SAME regular solid harmonics ``U_n^m`` used by P2M, local coefficients
    transform as the (matrix) transpose of the multipole rotation -- NOT via a
    separate ``B_T`` matrix. This block is therefore the transpose of the
    multipole world->z rotation (:func:`real_rotation_to_z_axis_multipole`).

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` rotation taking a z-frame local expansion back to
        the world frame.
    """
    return _multipole_align_to_z_block(x, y, z, ell, dtype=dtype).T


def real_rotation_from_z_axis_multipole(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Inverse rotation for MULTIPOLE expansions (z-aligned frame -> world).

    This is the inverse of :func:`real_rotation_to_z_axis_multipole`; apply it
    to rotate a z-frame multipole back to the world frame (``M = D @ M_z``).

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` rotation taking a z-frame multipole back to the
        world frame.
    """
    return _multipole_align_from_z_block(x, y, z, ell, dtype=dtype)


def real_rotation_to_z_axis_local(
    x: Array,
    y: Array,
    z: Array,
    ell: int,
    *,
    dtype: DTypeLike,
) -> Array:
    """Rotation that aligns (x,y,z) with the +z axis for LOCAL expansions.

    Local coefficients transform as the transpose-inverse of the multipole
    rotation (they contract against the regular ``U_n^m`` basis in
    :func:`evaluate_local_real`). The world->z local rotation is therefore the
    transpose of the multipole z->world rotation
    (:func:`real_rotation_from_z_axis_multipole`).

    Parameters
    ----------
    x : Array
        First Cartesian component of the alignment direction.
    y : Array
        Second Cartesian component.
    z : Array
        Third Cartesian component.
    ell : int
        Degree ``l``; the returned block is ``[2l+1, 2l+1]``.
    dtype : DTypeLike
        Output dtype.

    Returns
    -------
    Array
        The ``[2l+1, 2l+1]`` rotation taking a world-frame local expansion into
        the z-aligned frame.
    """
    return _multipole_align_from_z_block(x, y, z, ell, dtype=dtype).T

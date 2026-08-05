"""The analytic transverse derivative of the rotate/translate cascade at ``rho == 0``.

WHY THIS EXISTS. Every operator built as
``F(delta) = D_from(delta) @ T_z(|delta|) @ D_to(delta)`` reaches the transverse
coordinates only through the cylindrical radius ``rho = sqrt(x^2 + y^2)`` and the
alignment azimuth ``atan2(x, y)``. At ``rho == 0`` the azimuth is undefined, and the
double-``where`` guards that keep the reverse pass finite there return a **zero**
transverse cotangent. The forward value is exact -- every ``|m| >= 1`` term carries
``sin^|m|(theta) = (rho/r)^|m|``, which annihilates the arbitrary azimuth -- but the
derivative is not: the assembled cascade *is* differentiable at ``rho == 0``, the limit
is direction-independent, and the true transverse derivative is nonzero. Measured
finite-difference-versus-autodiff disagreement at the force level was 1.9e-03 relative
in the real basis and 1.8e-05 in the complex one.

No choice of guard recovers it, and three were measured (see G.10 in
``docs/refactor_audit_2026-08.md``): flooring ``rho`` changes nothing, removing the
guards gives NaN, and a ratio-plus-Chebyshev azimuth breaks the forward value by
1.0-2.0. The obstruction is structural -- the polar parametrisation has already divided
out the ``O(rho)`` coefficient the derivative needs -- so the fix supplies that
coefficient analytically instead of trying to recover it by differentiation.

THE FORMULA. ``F`` is rotationally covariant,
``F(R @ delta) = D_out(R) @ F(delta) @ D_in(R)^-1``. Differentiating that identity
along a rotation family that sweeps ``(0, 0, z)`` transversally gives a commutator:

    dF/dx |(0,0,z) = +(1/z) [ G_out^y @ F0 - F0 @ G_in^y ]
    dF/dy |(0,0,z) = -(1/z) [ G_out^x @ F0 - F0 @ G_in^x ]

where ``F0 = F(0, 0, z)`` and ``G^a`` are the generators of the coefficient-space
representation. The derivation, the calibration of every generator sign against an
identity this repository already verifies, and the validation (an eps sweep with no
error floor, plus worst-case 6.3e-09 / 3.6e-10 / 2.7e-10 for M2L / M2M / L2L across
four orders and both signs of ``z``) are in ``docs/rotation_degeneracy_derivative.md``.
Do not re-derive it; the sign traps are recorded there with the residual each wrong
choice produces.

HOW IT IS APPLIED. Only the tangent changes:

    primal : unchanged                                    -> forward bit-identical
    tangent: where(rho_sq <= eps r_sq, analytic, polar)    -> a select, not a sum

The transverse tangent is routed to whichever branch can actually differentiate it --
see :func:`split_transverse_tangent`, which also derives where the boundary goes and
why it is wider than the ``rho_sq > 0`` the guards themselves switch on. Outside the
band the routed tangent is bit-identical to the incoming one and the analytic term is
exactly zero, so every gradient the polar route can still resolve is preserved to the
last bit and both characterization goldens stay unmoved. Inside it the polar
contribution is not zero but garbage, which is why this is a select and not a
correction added on top.

COST. Applied to a coefficient vector the formula would need one extra cascade
evaluation per transverse axis. It needs none: the cascade is linear in its coefficient
argument, so the two ``F0 @ G_in`` terms fold into the incoming coefficient tangent and
ride along the JVP that was going to run anyway. What remains is four applications of a
static, block-diagonal, mostly-zero generator matrix -- negligible against the rotation
blocks -- and the forward-only path pays nothing at all, because ``custom_jvp`` leaves
the primal untouched.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ._precision import highest_matmul_precision

__all__ = [
    "TransverseGenerators",
    "split_transverse_tangent",
    "with_transverse_degeneracy_jvp",
]


class TransverseGenerators(NamedTuple):
    """The four representation generators formula (2)/(3) needs, packed and square.

    Each entry is a ``[(p+1)^2, (p+1)^2]`` block-diagonal matrix acting on packed
    coefficients: block ``ell`` is the degree-``ell`` generator, and the blocks are
    laid out by :func:`~jaccpot.operators.real_harmonics.sh_offset`. Packing the
    per-degree blocks into one matrix trades a few wasted multiply-adds for a single
    matmul instead of ``p + 1`` unrolled ones, which is the cheaper trade here because
    compile time is the binding cost.

    ``in_*`` act on the operator's input coefficient slot, ``out_*`` on its output
    slot; for M2L those are different representations (multipole in, local out), for
    M2M and L2L they are the same one.

    Attributes
    ----------
    in_x : Array
        Generator of rotation about ``x``, acting on the input slot.
    in_y : Array
        Generator of rotation about ``y``, acting on the input slot.
    out_x : Array
        Generator of rotation about ``x``, acting on the output slot.
    out_y : Array
        Generator of rotation about ``y``, acting on the output slot.
    """

    in_x: Array
    in_y: Array
    out_x: Array
    out_y: Array


def split_transverse_tangent(
    delta: Array, delta_tangent: Array
) -> Tuple[Array, Array, Array]:
    """Route the transverse tangent to whichever branch can differentiate it.

    Returns the tangent to hand to the cascade -- with its ``x`` and ``y`` components
    removed wherever the azimuth cannot be resolved, so the polar route's unusable
    contribution never enters -- together with ``(dx/z, dy/z)``, the coefficients
    formula (2)/(3) multiplies its commutators by, nonzero on exactly the complementary
    set. The two branches partition the input rather than superpose, so this is a
    select, not a correction added on top of something wrong.

    WHERE THE BOUNDARY GOES, AND WHY IT IS NOT ``rho_sq > 0``. The polar
    parametrisation does not only fail *at* ``rho == 0``; it degrades on approach.
    ``d(az)/dy = -x/rho^2`` grows without bound while the ``(rho/r)^|m|`` factor that
    annihilates the azimuth shrinks, and the transverse gradient comes out of that
    cancellation with relative error ``~eps r / rho``. Measured on ``l2l_real`` at
    ``z = -3``, order 4:

        rho          1e-17    1e-16    1e-12    1e-9     1e-6
        rel error    2.6e+02  1.8e+01  6.8e-04  3.5e-06  1.0e-09

    The analytic branch has the complementary error: it is the ``rho -> 0`` limit, so it
    errs ``O(rho/r)``. Setting the two equal puts the crossover at ``(rho/r)^2 == eps``,
    i.e. ``rho_sq <= eps * r_sq``, which is what is coded below. That choice is minimax:
    the worst relative error over all ``rho`` becomes ``~sqrt(eps)`` -- 1.5e-08 in
    float64, 3.4e-04 in float32 -- where before it was unbounded, and it is reached only
    on the boundary itself.

    Widening the predicate this far is what makes the *split* load-bearing rather than
    decorative. Inside the band the polar contribution is not zero, it is garbage, so it
    has to be removed rather than added to. That is why this function returns a routed
    tangent instead of just the scales.

    A displacement lands inside the band for real reasons, not only synthetic ones: two
    tree nodes whose ``(x, y)`` centres are mathematically equal -- same particles, same
    masses, summed in a different order -- differ by one ulp instead of zero, which is
    ``rho/r ~ 1e-17``. That is measured; see
    ``docs/rotation_degeneracy_derivative.md``.

    The band is a superset of the set the alignment guards zero, which is the ordering
    the split needs: everything those guards hand over is covered, and so is the
    neighbourhood where they hand over a usable-looking but wrong derivative instead.

    ``z != 0`` is the third regime, not a numerical safety margin. At ``delta == 0``
    formula (2)/(3) does not apply -- it divides by ``z`` -- so the scales stay zero
    there and the existing zero cotangent stands. That case is the identity translation
    for M2M/L2L and unphysical for M2L. It is also the only point the band could
    otherwise swallow: with ``r_sq == rho_sq`` the test ``rho_sq <= eps * r_sq`` holds
    only for ``rho_sq == 0``.

    Parameters
    ----------
    delta : Array
        Displacement ``[..., 3]``. Only the last axis is interpreted.
    delta_tangent : Array
        Its tangent, same shape.

    Returns
    -------
    cascade_tangent : Array
        ``delta_tangent``'s shape, bit-identical to it outside the band and with the
        transverse components zeroed inside it.
    scale_x : Array
        ``dx / z``, shape ``delta.shape[:-1]``, exactly zero outside the band.
    scale_y : Array
        ``dy / z``, likewise.
    """
    x, y, z = delta[..., 0], delta[..., 1], delta[..., 2]
    rho_sq = x * x + y * y
    r_sq = rho_sq + z * z
    # ``eps`` is read at trace time from a static dtype, so the threshold is a compiled
    # constant -- there is no runtime cost and no host sync.
    epsilon = float(np.finfo(jnp.result_type(delta)).eps)
    on_axis = jnp.logical_and(rho_sq <= epsilon * r_sq, z != 0)
    resolvable = jnp.logical_not(on_axis)[..., None]
    cascade_tangent = jnp.concatenate(
        [
            jnp.where(resolvable, delta_tangent[..., :2], 0.0),
            delta_tangent[..., 2:],
        ],
        axis=-1,
    )
    # Double-where: the outer select supplies the zero, the inner one keeps the
    # division away from z == 0 so the *derivative* of this expression is finite too.
    inverse_z = jnp.where(on_axis, 1.0 / jnp.where(on_axis, z, 1.0), 0.0)
    return (
        cascade_tangent,
        inverse_z * delta_tangent[..., 0],
        inverse_z * delta_tangent[..., 1],
    )


@highest_matmul_precision
def _apply_generator(coeffs: Array, generator: Array) -> Array:
    """Apply a packed generator to the last axis of ``coeffs``, padding with zeros.

    ``coeffs`` may be longer than the generator when a caller passes coefficients for a
    higher order than it asks for -- :func:`~jaccpot.operators.complex_ops.m2l_complex_reference`
    slices its input, so its tangent has the untruncated width. The tail contributes
    nothing to the result, so it contributes nothing to the correction either.

    Parameters
    ----------
    coeffs : Array
        Packed coefficients ``[..., C]`` with ``C >= (p+1)^2``.
    generator : Array
        Packed generator ``[(p+1)^2, (p+1)^2]``.

    Returns
    -------
    Array
        ``G @ coeffs`` along the last axis, shaped like ``coeffs``.
    """
    width = int(generator.shape[0])
    image = coeffs[..., :width] @ generator.T
    if int(coeffs.shape[-1]) == width:
        return image
    return jnp.concatenate([image, jnp.zeros_like(coeffs[..., width:])], axis=-1)


def with_transverse_degeneracy_jvp(
    cascade: Callable[..., Array],
    *,
    generators: Callable[[int, Any], TransverseGenerators],
) -> Callable[..., Array]:
    """Give ``cascade`` the analytic transverse derivative at ``rho == 0``.

    Wraps a rotate -> z-translate -> rotate-back operator in a ``custom_jvp`` whose
    primal is the unmodified ``cascade`` and whose tangent adds the correction derived
    in this module's docstring. The wrapped function keeps ``cascade``'s signature,
    docstring, and dtypes; it is not a ``jit`` boundary, so applying it inside an
    existing ``@jax.jit`` costs no extra dispatch.

    Reverse mode works because the rule is linear in the tangents (the scales multiply
    primal-derived vectors), so JAX transposes it -- which is what matters here, since
    the FMM takes ``jax.grad``, not ``jax.jvp``.

    Parameters
    ----------
    cascade : Callable[..., Array]
        ``cascade(coeffs, delta, **static_kwargs) -> Array``. Must be linear in
        ``coeffs`` -- the whole cost saving rests on it -- and every keyword argument
        must be hashable, because they are carried as ``nondiff_argnums``. One of them
        must be ``order``.
    generators : Callable[[int, Any], TransverseGenerators]
        ``generators(order, dtype) -> TransverseGenerators`` for the basis and the pair
        of representations this operator maps between. Called at trace time with the
        coefficient dtype, so the generator matmuls run in the working dtype rather than
        promoting to float64.

    Returns
    -------
    Callable[..., Array]
        ``cascade`` with the corrected JVP, same call signature.

    Notes
    -----
    Differentiable in ``coeffs`` and ``delta``, forward and reverse. The primal is
    bit-identical to ``cascade``'s. The tangent is bit-identical outside the band
    :func:`split_transverse_tangent` defines; inside it, only ``d/dx`` and ``d/dy``
    change, and only where ``z != 0``.
    """

    @functools.partial(jax.custom_jvp, nondiff_argnums=(2,))
    def cascade_with_rule(coeffs: Array, delta: Array, static_kwargs: tuple) -> Array:
        return cascade(coeffs, delta, **dict(static_kwargs))

    @cascade_with_rule.defjvp
    @highest_matmul_precision
    def cascade_with_rule_jvp(
        static_kwargs: tuple,
        primals: Tuple[Array, Array],
        tangents: Tuple[Array, Array],
    ) -> Tuple[Array, Array]:
        coeffs, delta = primals
        coeffs_tangent, delta_tangent = tangents
        static = dict(static_kwargs)
        generator = generators(int(static["order"]), coeffs.dtype)

        cascade_tangent, scale_x, scale_y = split_transverse_tangent(
            delta, delta_tangent
        )
        # The scales come from ``delta``; the generator images come from ``coeffs``.
        # Cast once here so a mixed-dtype call fails on this line rather than as an
        # opaque tangent/primal dtype mismatch inside ``custom_jvp``.
        scale_x = scale_x[..., None].astype(coeffs.dtype)
        scale_y = scale_y[..., None].astype(coeffs.dtype)

        # ``-F0 @ G_in`` folded into the incoming coefficient tangent, which is exact
        # because the cascade is linear in its coefficient argument. Off the degenerate
        # axis both scales are exactly zero, so this subtracts a bit-exact zero.
        coeffs_tangent = coeffs_tangent - (
            scale_x * _apply_generator(coeffs, generator.in_y)
            - scale_y * _apply_generator(coeffs, generator.in_x)
        )
        primal_out, tangent_out = jax.jvp(
            lambda c, d: cascade(c, d, **static),
            (coeffs, delta),
            (coeffs_tangent, cascade_tangent),
        )
        # ``+G_out @ F0``. ``primal_out`` *is* ``F0`` on the degenerate axis, where
        # ``delta == (0, 0, z)``, so the formula needs no separate evaluation of it.
        return primal_out, tangent_out + (
            scale_x * _apply_generator(primal_out, generator.out_y)
            - scale_y * _apply_generator(primal_out, generator.out_x)
        )

    @functools.wraps(cascade)
    def cascade_corrected(coeffs: Array, delta: Array, **static_kwargs: Any) -> Array:
        return cascade_with_rule(
            jnp.asarray(coeffs),
            jnp.asarray(delta),
            tuple(sorted(static_kwargs.items())),
        )

    return cascade_corrected

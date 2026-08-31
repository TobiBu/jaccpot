"""Complex-basis operators (solidfmm-style reference) in JAX."""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jax import lax
from jaxtyping import DTypeLike as _jaxtyping_DTypeLike
from jaxtyping import Float, Inexact, jaxtyped

from ._precision import highest_matmul_precision
from ._transverse_degeneracy_jvp import (
    TransverseGenerators,
    with_transverse_degeneracy_jvp,
    with_transverse_degeneracy_tangent,
    withdraw_unresolvable_transverse,
    without_unresolvable_transverse_jvp,
)
from .complex_harmonics import complex_R_solidfmm, complex_R_solidfmm_preserve_dtype
from .dtypes import complex_dtype_for_real, floor_squared_radius
from .real_harmonics import (
    _compute_dehnen_B_matrix_complex,
    sh_offset,
    sh_size,
)
from .symmetric_tensors import (
    contract_symmetric_one_axis_3d,
    symmetric_component_count,
    symmetric_multi_indices_3d,
)

__all__ = [
    "complex_dot",
    "complex_rotation_blocks_from_z_solidfmm_batch",
    "complex_rotation_blocks_to_z_solidfmm_batch",
    "complex_transverse_generators",
    "contract_spatial_derivative_with_velocity",
    "enforce_conjugate_symmetry",
    "enforce_conjugate_symmetry_batch",
    "evaluate_local_complex",
    "evaluate_local_complex_derivative_tower",
    "evaluate_local_complex_derivative_tower_batch",
    "evaluate_local_complex_grad_analytic",
    "evaluate_local_complex_grad_analytic_batch",
    "evaluate_local_complex_grad_analytic_preserve_dtype",
    "evaluate_local_complex_grad_order4_unrolled",
    "evaluate_local_complex_with_grad",
    "evaluate_local_complex_with_grad_analytic",
    "evaluate_local_complex_with_grad_analytic_batch",
    "evaluate_local_complex_with_grad_batch",
    "l2l_complex",
    "l2l_complex_batch",
    "m2l_complex_fused_align_deltas",
    "make_m2l_complex_fused_carry_axis_derivative",
    "m2l_complex_reference",
    "m2l_complex_reference_batch",
    "m2l_complex_reference_batch_cached_blocks",
    "m2m_complex",
    "regular_solid_harmonic_directional_derivative",
    "regular_solid_harmonic_directional_derivative_batch",
    "regular_solid_harmonic_directional_derivative_order",
    "regular_solid_harmonic_directional_derivative_order_batch",
    "regular_solid_harmonic_gradient_coefficients",
    "regular_solid_harmonic_gradient_coefficients_preserve_dtype",
    "rotate_complex_local_from_z_solidfmm",
    "rotate_complex_local_to_z_solidfmm",
    "rotate_complex_multipole_from_z_solidfmm",
    "rotate_complex_multipole_to_z_solidfmm",
    "translate_along_z_l2l_complex",
    "translate_along_z_l2l_complex_batch",
    "translate_along_z_m2l_complex",
    "translate_along_z_m2l_complex_batch",
    "translate_along_z_m2m_complex",
    "translate_along_z_m2m_complex_batch",
    "translate_along_z_m2m_complex_solidfmm",
]

# The eleven functions taking a single spatial `delta` (and `direction`) carry
# `Float[Array, "3"]`, because a length-2 vector was reaching all of them silently:
# JAX *clamps* an out-of-bounds index, so `delta[2]` on a length-2 array returns
# `delta[1]` and the caller gets the answer for `(x, y, y)` with no error at all.
# Measured module-wide by `bench/annotation_pilot`: 13 of its 41 silent acceptances
# here were exactly this one mistake.
#
# NOTE: the annotation narrows the accepted input, and that is deliberate rather
# than incidental. Before it, every one of them also took a numpy array, a list or
# a tuple, because the bodies reach `jnp` ops that coerce; `Float[Array, "3"]`
# admits only `jax.Array`. `downward/local_expansions.py` already made this trade
# in the same programme. The alternative spelling,
# `Float[Union[Array, np.ndarray], "3"]`, closes the same hole while keeping numpy
# callers, and was rejected only to avoid two spellings of one idea -- so if a
# numpy caller ever has to be supported here, that is the change to make, and it
# does not reopen the hole.
#
# The coefficient parameter on the nine of them that take one is
# `Inexact[Array, "_"]`, and both halves of that are measured rather than tidy:
#
#   - `Inexact`, not `Complex`, because every one of them accepts a REAL buffer
#     today and returns a sensible answer for it. `Complex` would be the narrower
#     and more obvious-looking annotation, and it would reject callers that work.
#   - `"_"` -- one anonymous axis -- rather than a length tied to `order`, because
#     the bodies slice `[:ncoeff]` and so deliberately tolerate a LONGER buffer.
#     Binding the length would reject calls these functions have always taken.
#     Same reasoning as `coefficients` in `downward/local_expansions.py`.
#
# What that leaves asserted is rank and dtype family, which is not nothing: a
# too-SHORT buffer already raises a domain error from the body (`TypeError` from
# the evaluators, `ValueError` from the rotations), so the annotation preempts no
# message, and a 2-D `local` was reaching `evaluate_local_complex` silently before
# it.
#
# ---------------------------------------------------------------------------
#
# THE REST OF THE SPATIAL-VECTOR FAMILY, and what the measurement changed about it.
#
# Twenty-four more functions now carry the same contract: `Float[Array, "3"]` on a
# single `delta`/`direction`, `Float[Array, "_ 3"]` on the batched `deltas`/`directions`.
# Derived by execution, four calls per function -- `(3,)` in every single-vector case,
# and the batch axis observed VARYING (`(4, 3)` beside `(17, 3)`, `(6, 3)` beside
# `(30, 3)`, `(8, 3)` beside `(32, 3)`), which is why that axis is anonymous rather
# than named. Naming it would assert `deltas` and `directions` agree, and `jax.vmap`
# already raises a specific error for inconsistent batch sizes, so the annotation
# would only replace a better message.
#
# The `*_batch` siblings were left alone by the delta/direction PR, and the test that
# pinned it -- `test_batch_variants_keep_taking_batched_deltas` -- warned against
# annotating them `"3"` by pattern-match, not against annotating them at all. `"_ 3"`
# satisfies that test rather than overriding it: it still takes the `(3, 3)` the test
# feeds, and it now rejects the `(3,)` a caller might pass by mistake.
#
# FIVE OF THESE WERE INVISIBLE UNTIL THE GATES WERE OPENED, which is worth recording
# because "not reached" and "unreachable" are different findings. `_evaluate.py` picks
# between three gradient implementations on two env switches that default to "0":
# `JACCPOT_LOCAL_EVAL_ORDER4_UNROLLED` and `JACCPOT_LOCAL_EVAL_DTYPE_PRESERVE`. So
# `evaluate_local_complex_grad_order4_unrolled`,
# `evaluate_local_complex_grad_analytic_preserve_dtype`,
# `regular_solid_harmonic_gradient_coefficients_preserve_dtype` and
# `_regular_solid_harmonic_order4_scalars` are never selected in a default run.
# Re-recording with both switches set measured all four at `(3,)`.
#
# `evaluate_local_complex_with_grad_batch` STAYS BARE, and is the one thing here that
# a shape cannot be honestly derived for: it is in `__all__`, it has no caller anywhere
# in `jaccpot/`, `bench/`, `examples/` or `tests/`, and neither test tier reaches it
# with the gates open or shut. Its docstring claims `(batch, 3)`; a probe that fed it
# `(batch, 3)` would confirm only the probe. Section 4.2 says derive by execution, so
# it waits for a caller or a test.
#
# HALF OF THESE CLOSE A HOLE AND HALF MOVE AN EXISTING ERROR, and the split was
# measured rather than assumed -- the first draft of this note claimed all 24 closed
# holes, and a vacuity check on the contract tests said otherwise (only 4 of 17 failed
# against the unannotated module). Feeding each one a length-2 delta, one at a time:
#
#   SILENTLY ACCEPTED before, so the annotation is protection -- twelve of them:
#     regular_solid_harmonic_gradient_coefficients (+ _preserve_dtype),
#     evaluate_local_complex_grad_analytic_preserve_dtype,
#     evaluate_local_complex_grad_order4_unrolled,
#     _regular_solid_harmonic_order4_scalars, _angles_from_delta_solidfmm,
#     _build_complex_harmonic_derivative_coefficients,
#     _complex_rotation_blocks_{to,from}_z_solidfmm and their _padded twins,
#     complex_rotation_blocks_{to,from}_z_solidfmm_batch.
#
#   ALREADY REJECTED, so the annotation moves the complaint to the boundary and names
#   the parameter -- `m2m_complex`, `l2l_complex`, `m2l_complex_reference` and most of
#   the batched family delegate into the eleven the earlier PR annotated, so a bad
#   delta already raised `TypeCheckError` from a private callee; the unbatched-input
#   cases already raised `vmap got inconsistent sizes`.
#
# Both are worth doing -- the second group's protection is an accident of the current
# call path, and a refactor that stopped delegating would reopen it silently -- but
# only the first group is a hole being closed, and the tests are labelled accordingly.
#
# THE COEFFICIENT BUFFERS CAME ALONG, and not as scope creep: decorating a function
# brings EVERY array parameter on it under `test_type_annotation_guard.py`, which is
# section 4.1's "do it for consistency within a signature you are already annotating"
# made mechanical. Thirteen of them, all `Inexact[Array, ...]` for the two reasons the
# note above already gives -- `Inexact` because a real buffer is accepted, anonymous
# lengths because the bodies slice `[:ncoeff]` and tolerate a longer one.
#
# The RANKS are measured and one of them is counter-intuitive: `local` is 1-D even on
# the `*_batch` functions -- `(25,)` beside a `(3, 3)` batch of deltas -- because one
# shared expansion is vmapped against many displacements. Only the PLURAL names are
# 2-D (`locals` `(6, 25)` and `(30, 25)`, `multipoles` `(4, 9)` up to `(17, 25)`).
# Annotating `local` as `"_ _"` in a batch function by pattern-match would have been
# wrong, and nothing but running it would have said so.
#
# `blocks_to_z` / `blocks_from_z` are `"_ _ _ _"`, rank only. They were observed
# `(17, 3, 5, 5)`, `(17, 4, 7, 7)` and `(17, 5, 9, 9)`: the trailing pair is SQUARE at
# three distinct extents, which is a real equality that nothing else checks, and a
# non-square block array would fail later inside a matmul with an opaque message.
# Asserting it needs an axis name the section 4.3 table does not have, and adding one
# is a guide change -- so it is left as the obvious next assertion rather than smuggled
# in here. The leading axes stay anonymous on purpose: `jax.vmap` already rejects a
# batch mismatch between these and `multipoles`, with a better message.
#
# NOT EVERY ONE OF THESE NARROWS ITS INPUT the way the first eleven did.
# `_complex_rotation_blocks_to_z_solidfmm_padded` and its `from_z` twin are wrapped by
# `without_unresolvable_transverse_jvp`, which calls `jnp.asarray(deltas)` before
# delegating -- so the check sits BEHIND a coercion and a numpy or list caller still
# works there. Same annotation, different reach; not a reason to spell it differently,
# but a reason not to claim the narrowing is uniform.
#
# That last asymmetry holds in DEFAULT MODE ONLY, which is a distinction worth the
# line: under `JACCPOT_RUNTIME_TYPECHECK=1` the package-wide import hook enforces
# `_transverse_degeneracy_jvp`'s own `builder_without_unresolvable(deltas: Array)`,
# which rejects numpy before its `jnp.asarray` ever runs. So the numpy caller works
# unchecked and fails checked, and neither behaviour comes from this module. Found by
# running the contract tests under the hook, not the default suite -- the default
# suite passed.
Array = jnp.ndarray

#: `jaxtyping.DTypeLike` admits anything that names a dtype -- a `numpy.dtype`, a
#: string, and JAX's own scalar types (`jnp.complex128` is a `_ScalarMeta`, not a
#: `numpy.dtype`). Aliased here beside `Array` because this module deliberately
#: defines its own `Array` alias rather than importing jaxtyping's.
DTypeLike = _jaxtyping_DTypeLike


@lru_cache(maxsize=None)
def _conjugate_symmetry_metadata(
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return packed-index metadata for conjugate-symmetry projection.

    Parameters
    ----------
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(center_idx, pos_idx, neg_idx, signs)``: the packed index of each ``m == 0``
        entry, the indices of the ``+m`` and ``-m`` partners that must be conjugates
        of one another, and the sign each partner carries.
    """

    p = int(order)
    center_idx: list[int] = []
    pos_idx: list[int] = []
    neg_idx: list[int] = []
    signs: list[float] = []
    for ell in range(p + 1):
        base = sh_offset(ell)
        center_idx.append(base + ell)
        for m in range(1, ell + 1):
            pos_idx.append(base + ell + m)
            neg_idx.append(base + ell - m)
            signs.append(-1.0 if (m % 2) else 1.0)
    return (
        np.asarray(center_idx, dtype=np.int32),
        np.asarray(pos_idx, dtype=np.int32),
        np.asarray(neg_idx, dtype=np.int32),
        np.asarray(signs, dtype=np.float64),
    )


def enforce_conjugate_symmetry(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Project coefficients onto conjugate-symmetric form.

    Enforces C_n^{-m} = (-1)^m * conj(C_n^{m}) and Im(C_n^0)=0.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The coefficients projected onto the conjugate-symmetric subspace.
    """
    coeffs_arr = jnp.asarray(coeffs)
    return enforce_conjugate_symmetry_batch(coeffs_arr[None, :], order=order)[0]


@partial(jax.jit, static_argnames=("order",))
def enforce_conjugate_symmetry_batch(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Batch projection onto conjugate-symmetric form.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        As :func:`enforce_conjugate_symmetry`, applied over the leading batch axis.
    """

    coeffs_arr = jnp.asarray(coeffs)
    center_idx_np, pos_idx_np, neg_idx_np, signs_np = _conjugate_symmetry_metadata(
        int(order)
    )
    center_idx = jnp.asarray(center_idx_np, dtype=jnp.int32)
    pos_idx = jnp.asarray(pos_idx_np, dtype=jnp.int32)
    neg_idx = jnp.asarray(neg_idx_np, dtype=jnp.int32)
    real_dtype = jnp.real(jnp.zeros((), dtype=coeffs_arr.dtype)).dtype
    signs = jnp.asarray(signs_np, dtype=real_dtype).astype(coeffs_arr.dtype)

    out = coeffs_arr
    center_vals = jnp.real(out[..., center_idx]).astype(coeffs_arr.dtype)
    out = out.at[..., center_idx].set(center_vals)
    if pos_idx_np.size == 0:
        return out
    mirrored = signs * jnp.conjugate(out[..., pos_idx])
    out = out.at[..., neg_idx].set(mirrored)
    return out


@lru_cache(maxsize=None)
def _factorial_table_cached_impl(max_n: int, dtype_key: str) -> np.ndarray:
    dtype = np.dtype(dtype_key)
    if max_n < 0:
        raise ValueError("max_n must be >= 0")
    if max_n == 0:
        return np.ones((1,), dtype=dtype)
    n = np.arange(1, max_n + 1, dtype=dtype)
    return np.concatenate([np.ones((1,), dtype=dtype), np.cumprod(n)])


def _factorial_table_cached(max_n: int, dtype: DTypeLike) -> Array:
    dtype_key = str(jnp.dtype(dtype))
    return jnp.asarray(_factorial_table_cached_impl(max_n, dtype_key), dtype=dtype)


def complex_dot(
    left: Array,
    right: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Complex dot product for packed solid-harmonic coefficients.

    When `conjugate_left` is True, computes sum(conj(left) * right),
    which matches the standard complex inner product used in solidfmm.

    Parameters
    ----------
    left : Array
        Left operand of the contraction.
    right : Array
        Right operand of the contraction.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        The scalar contraction of the two packed coefficient vectors.
    """
    ncoeff = sh_size(int(order))
    left = jnp.asarray(left)[:ncoeff]
    right = jnp.asarray(right)[:ncoeff]
    if conjugate_left:
        left = jnp.conjugate(left)
    return jnp.sum(left * right)


@jaxtyped(typechecker=beartype)
def evaluate_local_complex(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate complex local expansion at a displacement.

    Returns the real-valued potential (solidfmm normalization).

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        The scalar potential at ``delta``.
    """
    regular = complex_R_solidfmm(delta, order=order)
    pot = complex_dot(local, regular, order=order, conjugate_left=conjugate_left)
    return jnp.real(pot)


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_with_grad(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Evaluate complex local expansion and gradient at a displacement.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, Array]
        ``(potential, gradient)``, the gradient having shape ``(3,)``.
    """
    p = int(order)

    def phi_fn(d: Array) -> Array:
        return evaluate_local_complex(local, d, order=p, conjugate_left=conjugate_left)

    potential, grad = jax.value_and_grad(phi_fn)(delta)
    return grad, potential


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
def evaluate_local_complex_with_grad_batch(
    local: Array,
    deltas: Array,
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Batch evaluate complex local expansion and gradients.

    Parameters
    ----------
    local : Array
        Packed complex local coefficients, length ``sh_size(order)``.
    deltas : Array
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, Array]
        ``(potentials, gradients)`` over the batch axis.
    """
    return jax.vmap(
        lambda d: evaluate_local_complex_with_grad(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


def _lower_complex_harmonics_one_axis(
    coeffs: Array,
    *,
    order: int,
    axis: int,
) -> Array:
    """Apply one Cartesian derivative to packed complex-harmonic coefficients.

    If ``coeffs`` represents ``f_n^m`` over ``0 <= n <= order``, this returns
    coefficients representing ``∂_{axis} f_n^m`` in the same packed layout.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    axis : int
        Cartesian axis the operation acts along.

    Returns
    -------
    Array
        The packed coefficients after one Cartesian derivative along ``axis``.

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``'x'``, ``'y'``, ``'z'``.
    """
    p = int(order)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    coeffs = jnp.asarray(coeffs)[: sh_size(p)]
    idx_a, idx_b, fac_a, fac_b = _lower_complex_harmonics_axis_maps(p, axis)
    gathered_a = coeffs[jnp.asarray(idx_a, dtype=jnp.int32)]
    gathered_b = coeffs[jnp.asarray(idx_b, dtype=jnp.int32)]
    fac_a_arr = jnp.asarray(fac_a, dtype=coeffs.dtype)
    fac_b_arr = jnp.asarray(fac_b, dtype=coeffs.dtype)
    return fac_a_arr * gathered_a + fac_b_arr * gathered_b


@lru_cache(maxsize=None)
def _lower_complex_harmonics_axis_maps(
    order: int,
    axis: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute gather/scale maps for one Cartesian derivative axis.

    Parameters
    ----------
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    axis : int
        Cartesian axis the operation acts along.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(idx_a, idx_b, fac_a, fac_b)``: the two source indices each output entry
        draws on and their scale factors, so the derivative is a gather plus a
        weighted sum rather than a matrix product.
    """
    p = int(order)
    ncoeff = sh_size(p)
    idx_a = np.zeros((ncoeff,), dtype=np.int32)
    idx_b = np.zeros((ncoeff,), dtype=np.int32)
    fac_a = np.zeros((ncoeff,), dtype=np.complex128)
    fac_b = np.zeros((ncoeff,), dtype=np.complex128)

    def _src_index(n: int, m: int) -> tuple[int, bool]:
        if n < 0 or abs(m) > n:
            return 0, False
        return sh_offset(n) + (m + n), True

    for n in range(0, p + 1):
        for m in range(-n, n + 1):
            out_idx = sh_offset(n) + (m + n)
            if n == 0:
                continue
            if axis == 0:
                idx_a_val, valid_a = _src_index(n - 1, m - 1)
                idx_b_val, valid_b = _src_index(n - 1, m + 1)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = idx_b_val
                fac_a[out_idx] = 0.5 if valid_a else 0.0
                fac_b[out_idx] = -0.5 if valid_b else 0.0
            elif axis == 1:
                idx_a_val, valid_a = _src_index(n - 1, m - 1)
                idx_b_val, valid_b = _src_index(n - 1, m + 1)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = idx_b_val
                fac_a[out_idx] = 0.5j if valid_a else 0.0
                fac_b[out_idx] = 0.5j if valid_b else 0.0
            else:
                idx_a_val, valid_a = _src_index(n - 1, m)
                idx_a[out_idx] = idx_a_val
                idx_b[out_idx] = 0
                fac_a[out_idx] = 1.0 if valid_a else 0.0
                fac_b[out_idx] = 0.0

    return idx_a, idx_b, fac_a, fac_b


@jaxtyped(typechecker=beartype)
def _build_complex_harmonic_derivative_coefficients(
    delta: Float[Array, "3"],
    *,
    order: int,
    max_derivative_order: int,
) -> tuple[Array, ...]:
    """Build packed coefficient vectors for ``D^k R`` (k=0..K).

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    max_derivative_order : int
        Highest derivative order ``K`` to build; the tower carries ``D0..DK``.

    Returns
    -------
    tuple[Array, ...]
        One packed coefficient vector per derivative order ``0..K``.

    Raises
    ------
    ValueError
        If ``max_derivative_order`` is negative.
    """
    p = int(order)
    k_max = int(max_derivative_order)
    if k_max < 0:
        raise ValueError("max_derivative_order must be non-negative")

    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    levels: list[Array] = [base[jnp.newaxis, :]]
    if k_max == 0:
        return tuple(levels)

    for deriv_order in range(1, k_max + 1):
        combos = symmetric_multi_indices_3d(deriv_order)
        prev_combos = symmetric_multi_indices_3d(deriv_order - 1)
        prev = levels[-1]
        prev_index = {combo: idx for idx, combo in enumerate(prev_combos)}
        current = jnp.zeros(
            (symmetric_component_count(deriv_order, dim=3), sh_size(p)),
            dtype=base.dtype,
        )
        for idx, combo in enumerate(combos):
            if combo[0] > 0:
                parent = (combo[0] - 1, combo[1], combo[2])
                axis = 0
            elif combo[1] > 0:
                parent = (combo[0], combo[1] - 1, combo[2])
                axis = 1
            else:
                parent = (combo[0], combo[1], combo[2] - 1)
                axis = 2
            parent_coeff = prev[prev_index[parent]]
            derived = _lower_complex_harmonics_one_axis(
                parent_coeff,
                order=p,
                axis=axis,
            )
            current = current.at[idx].set(derived)
        levels.append(current)

    return tuple(levels)


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_derivative_tower(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    max_derivative_order: int,
    conjugate_left: bool = True,
) -> tuple[Array, ...]:
    """Evaluate potential and packed spatial derivatives ``D0..DK``.

    Notes
    -----
    This is an order-generic API scaffold for derivative towers. It uses
    autodiff internally today; hot-path contraction kernels can replace the
    internals without changing downstream code.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    max_derivative_order : int
        Highest derivative order ``K`` to build; the tower carries ``D0..DK``.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, ...]
        Potential and packed spatial derivatives ``D0..DK`` at ``delta``.

    Raises
    ------
    ValueError
        If ``max_derivative_order`` is negative.
    """
    p = int(order)
    k_max = int(max_derivative_order)
    if k_max < 0:
        raise ValueError("max_derivative_order must be non-negative")

    local = jnp.asarray(local)[: sh_size(p)]
    deriv_coeffs = _build_complex_harmonic_derivative_coefficients(
        delta,
        order=p,
        max_derivative_order=k_max,
    )
    out: list[Array] = []
    for deriv_order, coeff_level in enumerate(deriv_coeffs):
        vals = jax.vmap(
            lambda coeff: jnp.real(
                complex_dot(local, coeff, order=p, conjugate_left=conjugate_left)
            )
        )(coeff_level)
        if deriv_order == 0:
            out.append(vals)
        else:
            out.append(vals[: symmetric_component_count(deriv_order, dim=3)])
    return tuple(out)


@partial(
    jax.jit,
    static_argnames=("order", "max_derivative_order", "conjugate_left"),
)
@jaxtyped(typechecker=beartype)
def evaluate_local_complex_derivative_tower_batch(
    local: Inexact[Array, "_"],
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    max_derivative_order: int,
    conjugate_left: bool = True,
) -> tuple[Array, ...]:
    """Batch evaluate packed derivative towers for one local expansion.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    max_derivative_order : int
        Highest derivative order ``K`` to build; the tower carries ``D0..DK``.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, ...]
        The derivative tower evaluated over the batch axis.
    """
    return jax.vmap(
        lambda d: evaluate_local_complex_derivative_tower(
            local,
            d,
            order=order,
            max_derivative_order=max_derivative_order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order",))
def contract_spatial_derivative_with_velocity(
    packed: Array,
    velocity: Array,
    *,
    order: int,
) -> Array:
    """Contract packed order-``order`` spatial derivatives with velocity.

    Parameters
    ----------
    packed : Array
        Packed coefficient array in this module's ``sh`` layout.
    velocity : Array
        Velocity vector ``(3,)`` contracted against the derivative tensor.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The derivative tensor contracted with ``velocity``, one order lower.
    """
    return contract_symmetric_one_axis_3d(packed, velocity, order=order)


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_gradient_coefficients(
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Return packed ``(d/dx, d/dy, d/dz)`` coefficients of ``R_n^m(delta)``.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Packed ``(d/dx, d/dy, d/dz)`` coefficients, shape ``(3, sh_size(order))``.
    """
    p = int(order)
    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    grad_x = _lower_complex_harmonics_one_axis(base, order=p, axis=0)
    grad_y = _lower_complex_harmonics_one_axis(base, order=p, axis=1)
    grad_z = _lower_complex_harmonics_one_axis(base, order=p, axis=2)
    return jnp.stack((grad_x, grad_y, grad_z), axis=0)


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_gradient_coefficients_preserve_dtype(
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Return local-gradient coefficients without widening float32 deltas.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        As above, without widening a float32 input to complex128.
    """
    p = int(order)
    base = jnp.asarray(complex_R_solidfmm_preserve_dtype(delta, order=p))
    grad_x = _lower_complex_harmonics_one_axis(base, order=p, axis=0)
    grad_y = _lower_complex_harmonics_one_axis(base, order=p, axis=1)
    grad_z = _lower_complex_harmonics_one_axis(base, order=p, axis=2)
    return jnp.stack((grad_x, grad_y, grad_z), axis=0)


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_grad_analytic_preserve_dtype(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate the analytic local gradient without float32->complex128 widening.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        The gradient ``(3,)``, computed without a float32 to complex128 promotion.
    """
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    grad_coeffs = regular_solid_harmonic_gradient_coefficients_preserve_dtype(
        delta,
        order=p,
    )[:, :ncoeff]
    return jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))


@jaxtyped(typechecker=beartype)
def _regular_solid_harmonic_order4_scalars(
    delta: Float[Array, "3"],
) -> tuple[Array, ...]:
    """Return packed order-4 regular harmonics as scalar expressions.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.

    Returns
    -------
    tuple[Array, ...]
        The order-4 regular harmonics as individual scalar expressions.
    """
    delta_arr = jnp.asarray(delta)
    real_dtype = (
        delta_arr.dtype
        if jnp.issubdtype(delta_arr.dtype, jnp.floating)
        else jnp.float32
    )
    complex_dtype = jnp.complex128 if real_dtype == jnp.float64 else jnp.complex64

    d = jnp.asarray(delta, dtype=real_dtype)
    x, y, z = d[0], d[1], d[2]
    xy = x.astype(complex_dtype) + jnp.asarray(1j, dtype=complex_dtype) * y.astype(
        complex_dtype
    )
    zc = z.astype(complex_dtype)
    r2c = (x * x + y * y + z * z).astype(complex_dtype)
    one = jnp.asarray(1.0, dtype=real_dtype).astype(complex_dtype)

    pos: dict[tuple[int, int], Array] = {}
    pos[(0, 0)] = one
    pos[(1, 0)] = zc
    pos[(1, 1)] = xy * jnp.asarray(0.5, dtype=real_dtype).astype(complex_dtype)
    pos[(2, 2)] = (
        pos[(1, 1)] * xy * jnp.asarray(0.25, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(3, 3)] = (
        pos[(2, 2)]
        * xy
        * jnp.asarray(1.0 / 6.0, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(4, 4)] = (
        pos[(3, 3)] * xy * jnp.asarray(0.125, dtype=real_dtype).astype(complex_dtype)
    )
    pos[(2, 1)] = zc * pos[(1, 1)]
    pos[(3, 2)] = zc * pos[(2, 2)]
    pos[(4, 3)] = zc * pos[(3, 3)]
    pos[(2, 0)] = (
        jnp.asarray(3.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(1, 0)]
        - r2c * pos[(0, 0)]
    ) * jnp.asarray(0.25, dtype=real_dtype).astype(complex_dtype)
    pos[(3, 0)] = (
        jnp.asarray(5.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(2, 0)]
        - r2c * pos[(1, 0)]
    ) * jnp.asarray(1.0 / 9.0, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 0)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 0)]
        - r2c * pos[(2, 0)]
    ) * jnp.asarray(1.0 / 16.0, dtype=real_dtype).astype(complex_dtype)
    pos[(3, 1)] = (
        jnp.asarray(5.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(2, 1)]
        - r2c * pos[(1, 1)]
    ) * jnp.asarray(0.125, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 1)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 1)]
        - r2c * pos[(2, 1)]
    ) * jnp.asarray(1.0 / 15.0, dtype=real_dtype).astype(complex_dtype)
    pos[(4, 2)] = (
        jnp.asarray(7.0, dtype=real_dtype).astype(complex_dtype) * zc * pos[(3, 2)]
        - r2c * pos[(2, 2)]
    ) * jnp.asarray(1.0 / 12.0, dtype=real_dtype).astype(complex_dtype)

    def get(n: int, m: int) -> Array:
        if m >= 0:
            return pos[(n, m)]
        m_abs = -m
        sign = jnp.asarray(-1.0 if (m_abs % 2) else 1.0, dtype=real_dtype).astype(
            complex_dtype
        )
        return sign * jnp.conjugate(pos[(n, m_abs)])

    return tuple(get(n, m) for n in range(5) for m in range(-n, n + 1))


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_grad_order4_unrolled(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate order-4 local gradient with scalar recurrence/contraction.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        The gradient ``(3,)`` from the unrolled order-4 recurrence.
    """
    if int(order) != 4:
        return evaluate_local_complex_grad_analytic_preserve_dtype(
            local,
            delta,
            order=order,
            conjugate_left=conjugate_left,
        )

    r = _regular_solid_harmonic_order4_scalars(delta)
    local_coeffs = jnp.asarray(local)[:25]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    cdtype = jnp.result_type(local_coeffs.dtype, r[0].dtype)
    half = jnp.asarray(0.5, dtype=jnp.real(jnp.zeros((), dtype=cdtype)).dtype).astype(
        cdtype
    )
    half_i = jnp.asarray(0.5j, dtype=cdtype)
    zero = jnp.asarray(0.0, dtype=cdtype)

    def ridx(n: int, m: int) -> int:
        return n * n + (m + n)

    def src(n: int, m: int) -> Array:
        if n < 0 or m < -n or m > n:
            return zero
        return r[ridx(n, m)]

    acc_x = zero
    acc_y = zero
    acc_z = zero
    for n in range(1, 5):
        for m in range(-n, n + 1):
            coeff = local_coeffs[ridx(n, m)].astype(cdtype)
            left = src(n - 1, m - 1)
            right = src(n - 1, m + 1)
            acc_x = acc_x + coeff * (half * left - half * right)
            acc_y = acc_y + coeff * (half_i * left + half_i * right)
            acc_z = acc_z + coeff * src(n - 1, m)
    return jnp.real(jnp.stack((acc_x, acc_y, acc_z), axis=0))


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_with_grad_analytic(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Evaluate complex local expansion and gradient without autodiff.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, Array]
        ``(gradient, potential)`` computed analytically rather than by autodiff.
    """
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    regular = jnp.asarray(complex_R_solidfmm(delta, order=p))[:ncoeff]
    grad_coeffs = regular_solid_harmonic_gradient_coefficients(delta, order=p)[
        :, :ncoeff
    ]
    potential = jnp.real(jnp.sum(local_coeffs * regular))
    grad = jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))
    return grad, potential


@jaxtyped(typechecker=beartype)
def evaluate_local_complex_grad_analytic(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Evaluate only the complex local-expansion gradient without autodiff.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        The gradient ``(3,)`` alone, skipping the potential.
    """
    p = int(order)
    ncoeff = sh_size(p)
    local_coeffs = jnp.asarray(local)[:ncoeff]
    if conjugate_left:
        local_coeffs = jnp.conjugate(local_coeffs)
    grad_coeffs = regular_solid_harmonic_gradient_coefficients(delta, order=p)[
        :, :ncoeff
    ]
    return jnp.real(jnp.sum(local_coeffs[None, :] * grad_coeffs, axis=-1))


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
@jaxtyped(typechecker=beartype)
def evaluate_local_complex_grad_analytic_batch(
    local: Inexact[Array, "_"],
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> Array:
    """Batch evaluate only complex local-expansion gradients.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    Array
        Gradients over the batch axis, shape ``(batch, 3)``.
    """
    return jax.vmap(
        lambda d: evaluate_local_complex_grad_analytic(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "conjugate_left"))
@jaxtyped(typechecker=beartype)
def evaluate_local_complex_with_grad_analytic_batch(
    local: Inexact[Array, "_"],
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    conjugate_left: bool = True,
) -> tuple[Array, Array]:
    """Batch evaluate complex local expansion gradients without autodiff.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    conjugate_left : bool
        Whether to conjugate the left operand before contracting. True is the convention these expansions are stored in; passing False contracts them as-is.

    Returns
    -------
    tuple[Array, Array]
        ``(gradients, potentials)`` over the batch axis.
    """
    return jax.vmap(
        lambda d: evaluate_local_complex_with_grad_analytic(
            local,
            d,
            order=order,
            conjugate_left=conjugate_left,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_directional_derivative(
    delta: Float[Array, "3"],
    direction: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Directional derivative of packed regular harmonics along ``direction``.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    direction : Float[Array, '3']
        Direction vector ``(3,)`` the derivative is taken along.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Packed coefficients of ``(v.grad) R``.
    """
    return regular_solid_harmonic_directional_derivative_order(
        delta,
        direction,
        order=order,
        derivative_order=1,
    )


@partial(jax.jit, static_argnames=("order", "derivative_order"))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_directional_derivative_order(
    delta: Float[Array, "3"],
    direction: Float[Array, "3"],
    *,
    order: int,
    derivative_order: int,
) -> Array:
    """Order-``k`` directional derivative ``(v·∇)^k R`` in packed form.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    direction : Float[Array, '3']
        Direction vector ``(3,)`` the derivative is taken along.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    derivative_order : int
        Derivative order ``k`` to apply.

    Returns
    -------
    Array
        Packed coefficients of ``(v.grad)^k R``.

    Raises
    ------
    ValueError
        If ``derivative_order`` is negative.
    """
    p = int(order)
    k = int(derivative_order)
    if k < 0:
        raise ValueError("derivative_order must be non-negative")
    if k == 0:
        return jnp.asarray(complex_R_solidfmm(delta, order=p))

    base = jnp.asarray(complex_R_solidfmm(delta, order=p))
    direction_arr = jnp.asarray(direction, dtype=jnp.real(base).dtype)

    # `lax.fori_loop` hands the body a TRACER, not an `int`, so the index is
    # annotated `Array` (matching `_adaptive_policy`'s `iter_idx: Array`). It was
    # `int`, which `JACCPOT_RUNTIME_TYPECHECK=1` rejects at every call (F40).
    def body(_i: Array, coeffs: Array) -> Array:
        dx = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=0)
        dy = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=1)
        dz = _lower_complex_harmonics_one_axis(coeffs, order=p, axis=2)
        return direction_arr[0] * dx + direction_arr[1] * dy + direction_arr[2] * dz

    return jax.lax.fori_loop(0, k, body, base)


@partial(jax.jit, static_argnames=("order",))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_directional_derivative_batch(
    deltas: Float[Array, "_ 3"],
    directions: Float[Array, "_ 3"],
    *,
    order: int,
) -> Array:
    """Batch directional derivatives of packed regular harmonics.

    Parameters
    ----------
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    directions : Float[Array, '_ 3']
        Batched direction vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Batched packed coefficients of ``(v.grad) R``.
    """
    return jax.vmap(
        lambda d, v: regular_solid_harmonic_directional_derivative_order(
            d,
            v,
            order=order,
            derivative_order=1,
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(deltas, directions)


@partial(jax.jit, static_argnames=("order", "derivative_order"))
@jaxtyped(typechecker=beartype)
def regular_solid_harmonic_directional_derivative_order_batch(
    deltas: Float[Array, "_ 3"],
    directions: Float[Array, "_ 3"],
    *,
    order: int,
    derivative_order: int,
) -> Array:
    """Batch order-``k`` directional derivatives of packed regular harmonics.

    Parameters
    ----------
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    directions : Float[Array, '_ 3']
        Batched direction vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    derivative_order : int
        Derivative order ``k`` to apply.

    Returns
    -------
    Array
        Batched packed coefficients of ``(v.grad)^k R``.
    """
    return jax.vmap(
        lambda d, v: regular_solid_harmonic_directional_derivative_order(
            d,
            v,
            order=order,
            derivative_order=derivative_order,
        ),
        in_axes=(0, 0),
        out_axes=0,
    )(deltas, directions)


def translate_along_z_m2l_complex(
    multipole: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole to local along +z (Dehnen series).

    Parameters
    ----------
    multipole : Array
        Packed complex multipole coefficients, length ``sh_size(order)``.
    r : Array
        Centre separation; the z-translation distance after rotation to +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The local coefficients produced by translating the multipole along +z.
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    r = jnp.asarray(r).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    # This is the complex-basis M2L: always accumulate in a complex dtype even
    # if a real-typed multipole array is passed in (defensive; a real input
    # would otherwise raise when constructing a complex accumulator).
    cdtype = jnp.result_type(multipole.dtype, jnp.complex64)
    out = jnp.zeros((ncoeff,), dtype=cdtype)
    fact = _factorial_table_cached(2 * p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=cdtype)
            for k in range(m_abs, p - n + 1):
                src_idx = sh_offset(k) + (m + k)
                coeff = ((-1.0) ** m) * fact[n + k] / (r ** (n + k + 1))
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_m2m_complex(
    multipole: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole along +z (Dehnen series).

    Parameters
    ----------
    multipole : Array
        Packed complex multipole coefficients, length ``sh_size(order)``.
    dz : Array
        Signed translation distance along +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The multipole coefficients translated along +z.
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    dz = jnp.asarray(dz).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=multipole.dtype)
    fact = _factorial_table_cached(p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=multipole.dtype)
            for k in range(0, n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_m2m_complex_solidfmm(
    multipole: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex multipole along +z (solidfmm zm2m).

    Parameters
    ----------
    multipole : Array
        Packed complex multipole coefficients, length ``sh_size(order)``.
    dz : Array
        Signed translation distance along +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        As above, using solidfmm's zm2m convention.
    """
    p = int(order)
    multipole = jnp.asarray(multipole)
    dz = jnp.asarray(dz).reshape(())
    dtype = multipole.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=multipole.dtype)
    fact = _factorial_table_cached(p, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            m_abs = abs(m)
            acc = jnp.asarray(0.0 + 0.0j, dtype=multipole.dtype)
            for k in range(0, n - m_abs + 1):
                src_n = n - k
                if m_abs > src_n:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * multipole[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def translate_along_z_l2l_complex(
    local: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Translate complex local expansion along +z (Dehnen series).

    Parameters
    ----------
    local : Array
        Packed complex local coefficients, length ``sh_size(order)``.
    dz : Array
        Signed translation distance along +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The local coefficients translated along +z.
    """
    p = int(order)
    local = jnp.asarray(local)
    dz = jnp.asarray(dz).reshape(())
    dtype = local.real.dtype

    ncoeff = sh_size(p)
    out = jnp.zeros((ncoeff,), dtype=local.dtype)
    fact = _factorial_table_cached(p + 1, dtype)

    for n in range(p + 1):
        for m in range(-n, n + 1):
            acc = jnp.asarray(0.0 + 0.0j, dtype=local.dtype)
            for k in range(0, p - n + 1):
                src_n = n + k
                if src_n > p:
                    continue
                src_idx = sh_offset(src_n) + (m + src_n)
                coeff = (dz**k) / fact[k]
                acc = acc + coeff * local[src_idx]
            out = out.at[sh_offset(n) + (m + n)].set(acc)

    return out


def _complex_Dz(ell: int, angle: Array, *, dtype: DTypeLike) -> Array:
    m_vals = jnp.arange(-ell, ell + 1, dtype=dtype)
    diag = jnp.exp(1j * m_vals * angle)
    return jnp.diag(diag)


@lru_cache(maxsize=None)
def _complex_swap_matrices_cached(
    ell: int, dtype_key: str
) -> tuple[np.ndarray, np.ndarray]:
    B = _compute_dehnen_B_matrix_complex(ell, dtype_key)
    return B, B.T


def _complex_swap_matrices(ell: int, *, dtype: DTypeLike) -> tuple[Array, Array]:
    dtype_key = str(jnp.dtype(dtype))
    B, Bt = _complex_swap_matrices_cached(ell, dtype_key)
    return jnp.asarray(B, dtype=dtype), jnp.asarray(Bt, dtype=dtype)


# --------------------------------------------------------------------------
# Rotation generators, for the analytic transverse derivative at rho == 0.
# --------------------------------------------------------------------------
#
# The complex-basis counterpart of ``real_harmonics``'s generators. See
# :mod:`jaccpot.operators._transverse_degeneracy_jvp` for what they are for and
# ``docs/rotation_degeneracy_derivative.md`` for the derivation.
#
# Calibrated exactly as the real-basis ones were -- against central differences of
# the operator itself, not on paper -- and the two sign choices were searched rather
# than assumed. At order 4, ``z = 2.5``, ``eps = 1e-5`` on
# :func:`m2l_complex_reference`, the four combinations of the two signs give
# 2.0e+00, 1.5e+00, 9.6e-01 and **7.4e-11**; the surviving one is the ``-swap Lambda
# swap`` coded below, matching the real basis' sign.
#
# One structural difference from the real basis: there the local rotation *is* the
# transpose of the multipole one, so ``G^L = -(G^M)^T``. Here the local rotation is
# built from its own swap matrix (``B_T`` rather than ``B_U``), so each
# representation gets its generator from the swap matrix the rotation blocks
# actually use. Deriving one from the other would be a coincidence to rely on.


@lru_cache(maxsize=None)
def _complex_z_rotation_generator(ell: int) -> np.ndarray:
    """``d/dangle`` of :func:`_complex_Dz` at ``angle == 0``, degree ``ell``.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` complex128, the diagonal ``i * m`` for
        ``m = -ell .. ell``.
    """
    return np.diag(1j * np.arange(-ell, ell + 1)).astype(np.complex128)


@lru_cache(maxsize=None)
@highest_matmul_precision
def _complex_rotation_generator_block(ell: int, axis: str, basis: str) -> np.ndarray:
    """Generator of rotation about ``axis`` for one degree, in one representation.

    Parameters
    ----------
    ell : int
        Spherical harmonic degree.
    axis : str
        ``'x'`` or ``'y'``. The x-generator conjugates the z-generator with the
        involutory x<->z swap this basis' rotation blocks use; the y-generator is the
        x-generator conjugated by a quarter turn about z.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``), matching
        :func:`_complex_rotation_blocks_to_z_solidfmm`.

    Returns
    -------
    np.ndarray
        ``[2*ell+1, 2*ell+1]`` complex128.

    Raises
    ------
    ValueError
        If ``axis`` or ``basis`` is not one of the listed values.
    """
    # The matmuls below are numpy, in float64, so the pinned precision is inert
    # here -- the decorator is on for policy conformance
    # (tests/unit/operators/test_matmul_precision_pinned.py) rather than because
    # this function could drop to TF32.
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if basis not in ("multipole", "local"):
        raise ValueError(f"basis must be 'multipole' or 'local', got {basis!r}")
    B_T, B_U = _complex_swap_matrices_cached(ell, "complex128")
    swap = B_U if basis == "multipole" else B_T
    generator = -swap @ _complex_z_rotation_generator(ell) @ swap
    if axis == "y":
        # Built from this module's own ``_complex_Dz`` so it cannot drift from the
        # rotation blocks. That returns a ``jnp`` array, and under ``jax.jit``
        # ``jnp.asarray`` of a constant is a tracer, so pulling it back to numpy --
        # which is what lets this be ``lru_cache``d into a compile-time constant --
        # needs the constant-folding context. This runs inside a ``custom_jvp`` rule,
        # i.e. always inside a trace.
        with jax.ensure_compile_time_eval():
            quarter, quarter_back = (
                np.asarray(
                    _complex_Dz(
                        ell,
                        jnp.asarray(sign * np.pi / 2.0, dtype=jnp.float64),
                        # ``np.dtype(...)``, not the ``jnp.complex128`` scalar type:
                        # this parameter is annotated ``jnp.dtype`` and the runtime
                        # typechecker (JACCPOT_RUNTIME_TYPECHECK=1) rejects the latter.
                        dtype=np.dtype(np.complex128),
                    )
                )
                for sign in (+1.0, -1.0)
            )
        generator = quarter @ generator @ quarter_back
    return generator


@lru_cache(maxsize=None)
def _complex_transverse_generator_packed(
    order: int, axis: str, basis: str
) -> np.ndarray:
    """Per-degree generator blocks assembled into one packed square matrix.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    axis : str
        ``'x'`` or ``'y'``, as in :func:`_complex_rotation_generator_block`.
    basis : str
        ``'multipole'`` or ``'local'``, as in :func:`_complex_rotation_generator_block`.

    Returns
    -------
    np.ndarray
        ``[(p+1)^2, (p+1)^2]`` complex128, block-diagonal in ``ell`` with the packing
        of :func:`~jaccpot.operators.real_harmonics.sh_offset`.
    """
    p = int(order)
    packed = np.zeros((sh_size(p), sh_size(p)), dtype=np.complex128)
    for ell in range(p + 1):
        block = slice(sh_offset(ell), sh_offset(ell + 1))
        packed[block, block] = _complex_rotation_generator_block(ell, axis, basis)
    return packed


def complex_transverse_generators(
    order: int,
    dtype: DTypeLike,
    *,
    in_representation: str,
    out_representation: str,
) -> TransverseGenerators:
    """Complex-basis generators for the ``rho == 0`` transverse derivative.

    Feeds :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`,
    which documents what the four matrices are for.

    Parameters
    ----------
    order : int
        Maximum SH degree ``p``.
    dtype : DTypeLike
        Working complex dtype of the coefficients. The generators are built in
        complex128 and cast down, so the generator matmuls run in the working dtype
        rather than promoting complex64 coefficients.
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
            _complex_transverse_generator_packed(order, "x", in_representation),
            dtype=dtype,
        ),
        in_y=jnp.asarray(
            _complex_transverse_generator_packed(order, "y", in_representation),
            dtype=dtype,
        ),
        out_x=jnp.asarray(
            _complex_transverse_generator_packed(order, "x", out_representation),
            dtype=dtype,
        ),
        out_y=jnp.asarray(
            _complex_transverse_generator_packed(order, "y", out_representation),
            dtype=dtype,
        ),
    )


#: :func:`complex_transverse_generators` bound to each cascade operator's pair of
#: representations, ready to hand to
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`.
_M2L_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="multipole",
    out_representation="local",
)
_M2M_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="multipole",
    out_representation="multipole",
)
_L2L_TRANSVERSE_GENERATORS = partial(
    complex_transverse_generators,
    in_representation="local",
    out_representation="local",
)


def _solidfmm_pack_m_nonneg(block: Array, *, ell: int) -> tuple[Array, Array]:
    """Extract m>=0 coefficients as (re, im) arrays.

    Used for solidfmm-style swap/rotscale operations.

    Parameters
    ----------
    block : Array
        One square rotation block for a single degree.
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    tuple[Array, Array]
        ``(re, im)`` for the ``m >= 0`` half of the block.
    """
    block = jnp.asarray(block)
    start = ell
    re = jnp.real(block[start:])
    im = jnp.imag(block[start:])
    return re, im


def _solidfmm_unpack_m_nonneg(re: Array, im: Array, *, ell: int) -> Array:
    """Reconstruct full m in [-ell, ell] block from m>=0 real/imag arrays.

    Parameters
    ----------
    re : Array
        Real parts of the ``m >= 0`` coefficients.
    im : Array
        Imaginary parts of the ``m >= 0`` coefficients.
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    Array
        The full ``m`` in ``[-ell, ell]`` block rebuilt from the ``m >= 0`` half.
    """
    re = jnp.asarray(re)
    im = jnp.asarray(im)
    dtype = complex_dtype_for_real(jnp.result_type(re, im))
    block = jnp.zeros((2 * ell + 1,), dtype=dtype)

    m_vals = jnp.arange(0, ell + 1)
    pos = ell + m_vals
    block = block.at[pos].set(re + 1j * im)

    neg_m = jnp.arange(1, ell + 1)
    neg_pos = ell - neg_m
    pos_m = ell + neg_m
    signs = (-1.0) ** neg_m
    block = block.at[neg_pos].set(signs * jnp.conjugate(block[pos_m]))
    return block


def _solidfmm_swap_mats(
    B_swap: Array,
    *,
    ell: int,
    dtype: DTypeLike,
) -> tuple[Array, Array]:
    """Build real/imag swap matrices for solidfmm's m>=0 storage.

    These implement the real-linear map induced by B on coefficients with
    conjugate symmetry.

    Parameters
    ----------
    B_swap : Array
        Involutory swap matrix exchanging the z and x axes for this degree.
    ell : int
        Spherical harmonic degree.
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    tuple[Array, Array]
        Real and imaginary swap matrices for solidfmm's ``m >= 0`` storage.
    """
    m_vals = jnp.arange(0, ell + 1)
    l_vals = jnp.arange(0, ell + 1)
    row_idx = ell + m_vals[:, None]
    col_pos = ell + l_vals[None, :]
    col_neg = ell - l_vals[None, :]

    B = jnp.asarray(B_swap, dtype=dtype)
    B_pos = B[row_idx, col_pos]
    B_neg = B[row_idx, col_neg]

    signs = (-1.0) ** l_vals
    real_mat = B_pos + signs * B_neg
    imag_mat = B_pos - signs * B_neg

    real_mat = real_mat.at[:, 0].set(B_pos[:, 0])
    imag_mat = imag_mat.at[:, 0].set(jnp.zeros((ell + 1,), dtype=dtype))
    return real_mat, imag_mat


@highest_matmul_precision
def _solidfmm_swap_apply(
    re: Array,
    im: Array,
    B_swap: Array,
    *,
    ell: int,
) -> tuple[Array, Array]:
    """Apply solidfmm-style swap to m>=0 real/imag arrays.

    Parameters
    ----------
    re : Array
        Real parts of the ``m >= 0`` coefficients.
    im : Array
        Imaginary parts of the ``m >= 0`` coefficients.
    B_swap : Array
        Involutory swap matrix exchanging the z and x axes for this degree.
    ell : int
        Spherical harmonic degree.

    Returns
    -------
    tuple[Array, Array]
        The swapped ``(re, im)`` arrays.
    """
    dtype = jnp.result_type(re, im)
    real_mat, imag_mat = _solidfmm_swap_mats(B_swap, ell=ell, dtype=dtype)
    re_out = real_mat @ re
    im_out = imag_mat @ im
    return re_out, im_out


def _solidfmm_rotscale(
    re: Array,
    im: Array,
    *,
    angle: Array,
    scale: Array,
    ell: int,
    forward: bool,
) -> tuple[Array, Array]:
    """Solidfmm rotscale for m>=0 coefficients.

    Parameters
    ----------
    re : Array
        Real parts of the ``m >= 0`` coefficients.
    im : Array
        Imaginary parts of the ``m >= 0`` coefficients.
    angle : Array
        Rotation angle about z, in radians.
    scale : Array
        Per-order scale factor applied alongside the rotation.
    ell : int
        Spherical harmonic degree.
    forward : bool
        Direction of the rotscale: True applies it, False applies its inverse.

    Returns
    -------
    tuple[Array, Array]
        The rotated and scaled ``(re, im)`` arrays.
    """
    m_vals = jnp.arange(0, ell + 1, dtype=jnp.result_type(re, im, angle))
    cos_m = jnp.cos(m_vals * angle)
    sin_m = jnp.sin(m_vals * angle)
    scale = jnp.asarray(scale)

    if forward:
        re_out = scale * (cos_m * re - sin_m * im)
        im_out = scale * (sin_m * re + cos_m * im)
    else:
        re_out = scale * (cos_m * re + sin_m * im)
        im_out = scale * (-sin_m * re + cos_m * im)
    return re_out, im_out


@jaxtyped(typechecker=beartype)
def _angles_from_delta_solidfmm(delta: Float[Array, "3"]) -> tuple[Array, Array]:
    """Angles matching solidfmm's euler() convention.

    solidfmm defines:
        cos(alpha)=y/rxy, sin(alpha)=x/rxy
        cos(beta)=z/r, sin(beta)=-rxy/r
    so alpha=atan2(x,y), beta=atan2(-rxy,z).

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.

    Returns
    -------
    tuple[Array, Array]
        The two Euler angles in solidfmm's ``euler()`` convention.
    """
    x, y, z = delta[0], delta[1], delta[2]
    # NaN-safe (double-where) angle computation. The forward values are
    # unchanged for every displacement, but the reverse-mode cotangents stay
    # finite at the degenerate directions the fixed-topology FMM genuinely hits:
    #   * zero displacement (COM of a single-child internal node == its child),
    #   * z-axis-aligned displacement (rho == 0), e.g. lattice-aligned M2L pairs.
    # At those points ``sqrt`` (infinite grad at 0) and ``arctan2`` (0/0 grad at
    # the origin) would otherwise inject NaNs into the M2L/L2L reverse pass. The
    # azimuth is undefined there and the rotation reduces to a pure polar turn /
    # identity, so returning 0 keeps the forward exact.
    #
    # THE ZERO COTANGENT IS NOT CORRECT, and this is deliberate -- the missing
    # derivative is supplied one level up rather than here. Do not try to fix it at
    # this site. The retracted history is kept because its measurement was real and
    # its scope is the instructive part: on four z-stacked clusters sharing one (x,y)
    # point set (COM displacements exactly (0, 0, +-8), so rho == 0 on every M2L
    # pair), reverse-mode AD agreed with finite differences to ~10 significant
    # digits, and the transverse one-sided derivatives converged to the same value
    # from both sides -- read at the time as "the loss is smooth and nothing is
    # dropped".
    #
    # What that construction could not see is that it is symmetric: the same (x,y)
    # set and the same intra-cluster masses in every cluster, so the per-pair
    # transverse errors cancel in the sum. Re-measured with an ASYMMETRIC random
    # cotangent and an asymmetric purely-transverse perturbation direction, on a
    # system with 6 of 24 M2L pairs at rho == 0 exactly, FD and AD disagreed by
    # 1.8e-05 relative in this (complex) basis and 1.9e-03 in the real one --
    # stable across step size, so not finite-difference noise.
    #
    # The forward argument still holds and is worth keeping: the m != 0 terms carry
    # sin^|m|(theta) = (rho/r)^|m| factors that annihilate the arbitrary azimuth, so
    # the VALUE at rho == 0 is exact. It is the derivative that is not -- the
    # cascade is differentiable there (the limit is direction-independent) and the
    # true transverse derivative is nonzero. It cannot be recovered at this site: the
    # code reaches (x, y) only through rho and the azimuth, so every chain-rule route
    # carries x/rho or y/rho^2 and the O(rho) coefficient the derivative needs has
    # already been divided out.
    #
    # RESOLVED at the cascade level (G.10). ``m2l_complex_reference``, ``m2m_complex``
    # and ``l2l_complex`` each carry a ``custom_jvp`` that supplies the transverse
    # derivative analytically, from the rotational covariance of the assembled
    # operator -- which is available there and not here, because an individual
    # alignment block is genuinely direction-dependent at rho == 0 while the product
    # is not. See :mod:`jaccpot.operators._transverse_degeneracy_jvp` and
    # ``docs/rotation_degeneracy_derivative.md``. Asserted by
    # ``tests/unit/operators/test_complex_ops.py::test_complex_cascade_transverse_gradient_at_rho_zero``.
    #
    # WARNING for anyone extending the coverage: a *uniform lattice* does not
    # exercise this guard, despite what
    # ``tests/unit/test_gradient_correctness.py::test_no_nan_axis_aligned_grid``
    # says. Centres are COM, not geometric box centres, so lattice leaf COMs are
    # generically off-axis relative to each other -- measured on that test's own
    # 5^3 grid, zero of its 22 M2L pairs have rho == 0 and the minimum rho^2 is
    # 5.64. Use the z-stacked-cluster construction above to reach the degeneracy.
    # Also note a central difference cannot validate a subgradient choice at a
    # symmetric kink: it returns 0 there whether or not 0 is correct.
    rho_sq = x * x + y * y
    rho_pos = rho_sq > 0
    rho = jnp.where(rho_pos, jnp.sqrt(jnp.where(rho_pos, rho_sq, 1.0)), 0.0)
    alpha = jnp.where(rho_pos, jnp.arctan2(jnp.where(rho_pos, x, 1.0), y), 0.0)
    r_pos = (rho_sq + z * z) > 0
    beta = jnp.where(r_pos, jnp.arctan2(-rho, jnp.where(r_pos, z, 1.0)), 0.0)
    return alpha, beta


@highest_matmul_precision
@jaxtyped(typechecker=beartype)
def _complex_rotation_blocks_to_z_solidfmm(
    delta: Float[Array, "3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> tuple[Array, ...]:
    """Rotation blocks to z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``).
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    tuple[Array, ...]
        One rotation block per degree, aligning the axis to +z.

    Raises
    ------
    ValueError
        If ``rotation`` is not a supported convention.
    """
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta_solidfmm(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, beta, dtype=dtype)
        if basis == "multipole":
            D = B_U @ Dz_beta @ B_U @ Dz_alpha
        else:
            D = B_T @ Dz_beta @ B_T @ Dz_alpha
        blocks.append(D)
    return tuple(blocks)


@highest_matmul_precision
@jaxtyped(typechecker=beartype)
def _complex_rotation_blocks_from_z_solidfmm(
    delta: Float[Array, "3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> tuple[Array, ...]:
    """Rotation blocks from z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``).
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    tuple[Array, ...]
        One rotation block per degree, rotating back from +z.

    Raises
    ------
    ValueError
        If ``rotation`` is not a supported convention.
    """
    if basis not in ("multipole", "local"):
        raise ValueError("basis must be 'multipole' or 'local'")
    p = int(order)
    delta = jnp.asarray(delta)
    alpha, beta = _angles_from_delta_solidfmm(delta)

    blocks = []
    for ell in range(p + 1):
        B_T, B_U = _complex_swap_matrices(ell, dtype=dtype)
        Dz_alpha = _complex_Dz(ell, -alpha, dtype=dtype)
        Dz_beta = _complex_Dz(ell, -beta, dtype=dtype)
        if basis == "multipole":
            D = Dz_alpha @ B_U @ Dz_beta @ B_U
        else:
            D = Dz_alpha @ B_T @ Dz_beta @ B_T
        blocks.append(D)
    return tuple(blocks)


def _pack_coeffs_by_ell(
    coeffs: Array,
    *,
    order: int,
) -> Array:
    """Pack coefficients into (p+1, 2p+1) array with zero padding.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Coefficients as ``(p+1, 2p+1)`` with zero padding.
    """
    p = int(order)
    coeffs = jnp.asarray(coeffs)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m), dtype=coeffs.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[ell, : 2 * ell + 1].set(coeffs[sl])
    return out


def _unpack_coeffs_by_ell(
    packed: Array,
    *,
    order: int,
) -> Array:
    """Unpack (p+1, 2p+1) coefficients back into packed layout.

    Parameters
    ----------
    packed : Array
        Packed coefficient array in this module's ``sh`` layout.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The ``(p+1, 2p+1)`` array flattened back to the packed layout.
    """
    p = int(order)
    dtype = jnp.asarray(packed).dtype
    out = jnp.zeros((sh_size(p),), dtype=dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[sl].set(packed[ell, : 2 * ell + 1])
    return out


def _blocks_to_padded_array(
    blocks: tuple[Array, ...],
    *,
    order: int,
    dtype: DTypeLike,
) -> Array:
    """Pad rotation blocks to (p+1, 2p+1, 2p+1).

    Parameters
    ----------
    blocks : tuple[Array, ...]
        Per-degree rotation blocks, one square matrix per degree.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    Array
        The blocks padded to ``(p+1, 2p+1, 2p+1)``.
    """
    p = int(order)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m, max_m), dtype=dtype)
    for ell in range(p + 1):
        size = 2 * ell + 1
        out = out.at[ell, :size, :size].set(blocks[ell])
    return out


# Withdraws its in-band transverse tangent: an individual alignment block has no
# transverse derivative near rho == 0 (its limit is approach-dependent, unlike the
# assembled cascade's), so it hands the caller nothing there rather than something
# wrong, and the caller supplies the cascade-level term. Applied to the *padded*
# per-delta builder, which is what the batch (precomputed-block) lanes go through; the
# unpadded one feeds m2l_complex_reference / m2m_complex / l2l_complex, which already
# carry the cascade-level rule themselves.
@without_unresolvable_transverse_jvp
@jaxtyped(typechecker=beartype)
def _complex_rotation_blocks_to_z_solidfmm_padded(
    delta: Float[Array, "3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> Array:
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta,
        order=order,
        basis=basis,
        dtype=dtype,
    )
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


# Withdraws its in-band transverse tangent: an individual alignment block has no
# transverse derivative near rho == 0 (its limit is approach-dependent, unlike the
# assembled cascade's), so it hands the caller nothing there rather than something
# wrong, and the caller supplies the cascade-level term. Applied to the *padded*
# per-delta builder, which is what the batch (precomputed-block) lanes go through; the
# unpadded one feeds m2l_complex_reference / m2m_complex / l2l_complex, which already
# carry the cascade-level rule themselves.
@without_unresolvable_transverse_jvp
@jaxtyped(typechecker=beartype)
def _complex_rotation_blocks_from_z_solidfmm_padded(
    delta: Float[Array, "3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> Array:
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta,
        order=order,
        basis=basis,
        dtype=dtype,
    )
    return _blocks_to_padded_array(blocks, order=order, dtype=dtype)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
@jaxtyped(typechecker=beartype)
def complex_rotation_blocks_to_z_solidfmm_batch(
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> Array:
    """Batch padded rotation blocks to z using solidfmm convention.

    Parameters
    ----------
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``).
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    Array
        Padded to-z rotation blocks for a batch of displacements.
    """
    return jax.vmap(
        lambda d: _complex_rotation_blocks_to_z_solidfmm_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


@partial(jax.jit, static_argnames=("order", "basis", "dtype"))
@jaxtyped(typechecker=beartype)
def complex_rotation_blocks_from_z_solidfmm_batch(
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    basis: str,
    dtype: DTypeLike,
) -> Array:
    """Batch padded rotation blocks from z using solidfmm convention.

    Parameters
    ----------
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    basis : str
        ``'multipole'`` (swap ``B_U``) or ``'local'`` (swap ``B_T``).
    dtype : DTypeLike
        Real working dtype the tables are built at.

    Returns
    -------
    Array
        Padded from-z rotation blocks for a batch of displacements.
    """
    return jax.vmap(
        lambda d: _complex_rotation_blocks_from_z_solidfmm_padded(
            d,
            order=order,
            basis=basis,
            dtype=dtype,
        )
    )(deltas)


def _apply_complex_rotation_blocks_batched(
    coeffs: Array,
    blocks: tuple[Array, ...],
    *,
    order: int,
) -> Array:
    """Apply rotation blocks using per-ell batched matvecs.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    blocks : tuple[Array, ...]
        Per-degree rotation blocks, one square matrix per degree.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The coefficients with the per-degree blocks applied.
    """
    p = int(order)
    coeffs = jnp.asarray(coeffs)
    dtype = coeffs.dtype
    blocks_array = _blocks_to_padded_array(blocks, order=p, dtype=dtype)
    packed = _pack_coeffs_by_ell(coeffs, order=p)
    rotated = jnp.einsum(
        "bij,bj->bi", blocks_array, packed, precision=lax.Precision.HIGHEST
    )
    return _unpack_coeffs_by_ell(rotated, order=p)


@partial(jax.jit, static_argnames=("order",))
def _apply_complex_rotation_blocks_padded_batch(
    coeffs: Array,
    blocks_array: Array,
    *,
    order: int,
) -> Array:
    """Apply padded rotation blocks to a batch of coefficients.

    Parameters
    ----------
    coeffs : Array
        Packed complex coefficients, length ``sh_size(order)``.
    blocks_array : Array
        Rotation blocks padded to ``(p+1, 2p+1, 2p+1)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        The batch of coefficients with padded blocks applied.
    """
    packed = jax.vmap(lambda c: _pack_coeffs_by_ell(c, order=order))(coeffs)
    rotated = jnp.einsum(
        "nbij,nbj->nbi", blocks_array, packed, precision=lax.Precision.HIGHEST
    )
    return jax.vmap(lambda c: _unpack_coeffs_by_ell(c, order=order))(rotated)


@jaxtyped(typechecker=beartype)
def rotate_complex_multipole_to_z_solidfmm(
    multipole: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Rotate multipoles to z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    multipole : Inexact[Array, '_']
        Packed complex multipole coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Multipoles rotated so the displacement lies along +z.
    """
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


@jaxtyped(typechecker=beartype)
def rotate_complex_multipole_from_z_solidfmm(
    multipole: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Rotate multipoles from z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    multipole : Inexact[Array, '_']
        Packed complex multipole coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Multipoles rotated back out of the +z frame.
    """
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta, order=order, basis="multipole", dtype=jnp.asarray(multipole).dtype
    )
    return _apply_complex_rotation_blocks_batched(multipole, blocks, order=order)


@jaxtyped(typechecker=beartype)
def rotate_complex_local_to_z_solidfmm(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Rotate locals to z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Locals rotated so the displacement lies along +z.
    """
    blocks = _complex_rotation_blocks_to_z_solidfmm(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


@jaxtyped(typechecker=beartype)
def rotate_complex_local_from_z_solidfmm(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
) -> Array:
    """Rotate locals from z using solidfmm's swap+z-rotation convention.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Locals rotated back out of the +z frame.
    """
    blocks = _complex_rotation_blocks_from_z_solidfmm(
        delta, order=order, basis="local", dtype=jnp.asarray(local).dtype
    )
    return _apply_complex_rotation_blocks_batched(local, blocks, order=order)


# The `with_transverse_degeneracy_jvp` layer on the three cascade operators below
# sits *inside* the `jax.jit`, so it adds no dispatch boundary. It leaves the primal
# bit-identical and supplies only the transverse derivative on the `rho == 0` axis,
# where `_angles_from_delta_solidfmm`'s guards return a zero cotangent; see
# :mod:`jaccpot.operators._transverse_degeneracy_jvp`.
@partial(jax.jit, static_argnames=("order", "rotation"))
@partial(with_transverse_degeneracy_jvp, generators=_M2M_TRANSVERSE_GENERATORS)
@jaxtyped(typechecker=beartype)
def m2m_complex(
    multipole: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Complex M2M using A6: rotate → z-translate → rotate back.

    Parameters
    ----------
    multipole : Inexact[Array, '_']
        Packed complex multipole coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    rotation : str
        Rotation convention; ``'solidfmm'`` is the only one wired to production.

    Returns
    -------
    Array
        The parent multipole coefficients.

    Raises
    ------
    ValueError
        If ``rotation`` is not a supported convention.
    """
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    M_rot = rotate_complex_multipole_to_z_solidfmm(multipole, delta, order=p)
    M_z = translate_along_z_m2m_complex_solidfmm(M_rot, r, order=p)
    return rotate_complex_multipole_from_z_solidfmm(M_z, delta, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
@partial(with_transverse_degeneracy_jvp, generators=_L2L_TRANSVERSE_GENERATORS)
@jaxtyped(typechecker=beartype)
def l2l_complex(
    local: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Complex L2L using A6: rotate → z-translate → rotate back.

    Parameters
    ----------
    local : Inexact[Array, '_']
        Packed complex local coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    rotation : str
        Rotation convention; ``'solidfmm'`` is the only one wired to production.

    Returns
    -------
    Array
        The child local coefficients.

    Raises
    ------
    ValueError
        If ``rotation`` is not a supported convention.
    """
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    local = jnp.asarray(local)
    delta = jnp.asarray(delta)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    L_rot = rotate_complex_local_to_z_solidfmm(local, delta, order=p)
    L_z = translate_along_z_l2l_complex(L_rot, r, order=p)
    return rotate_complex_local_from_z_solidfmm(L_z, delta, order=p)


@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
@jaxtyped(typechecker=beartype)
def m2l_complex_reference(
    multipole: Inexact[Array, "_"],
    delta: Float[Array, "3"],
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Reference M2L in complex basis (rotate → z-translate → rotate back).

    Differentiable in both arguments, forward and reverse. Near the ``rho == 0`` axis
    the ``d/dx`` and ``d/dy`` cotangents come from a ``custom_jvp`` rather than from
    differentiating ``_angles_from_delta_solidfmm``'s guarded azimuth; the analytic
    branch applies inside exactly zero outside a narrow band around that axis (``rho <= sqrt(eps) * |delta|``, the measured crossover between the two routes' errors).

    Parameters
    ----------
    multipole : Inexact[Array, '_']
        Packed complex multipole coefficients, length ``sh_size(order)``.
    delta : Float[Array, '3']
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    rotation : str
        Rotation convention; ``'solidfmm'`` is the only one wired to production.

    Returns
    -------
    Array
        The local coefficients induced by the source multipole.

    Raises
    ------
    ValueError
        If ``rotation`` is not a supported convention.
    """
    if rotation != "solidfmm":
        raise ValueError("rotation must be 'solidfmm'")
    p = int(order)
    multipole = jnp.asarray(multipole)
    delta = jnp.asarray(delta)

    ncoeff = sh_size(p)
    multipole = multipole[:ncoeff]

    M_rotated = rotate_complex_multipole_to_z_solidfmm(multipole, delta, order=p)

    r = jnp.sqrt(
        floor_squared_radius(jnp.dot(delta, delta, precision=lax.Precision.HIGHEST))
    )
    local_z = translate_along_z_m2l_complex(M_rotated, r, order=p)

    return rotate_complex_local_from_z_solidfmm(local_z, delta, order=p)


@partial(jax.jit, static_argnames=("order", "rotation"))
@jaxtyped(typechecker=beartype)
def m2l_complex_reference_batch(
    multipoles: Inexact[Array, "_ _"],
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Batch M2L in complex basis (rotate → z-translate → rotate back).

    Parameters
    ----------
    multipoles : Inexact[Array, '_ _']
        Batched packed complex multipoles, shape ``(batch, sh_size(order))``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    rotation : str
        Rotation convention; ``'solidfmm'`` is the only one wired to production.

    Returns
    -------
    Array
        Local coefficients for a batch of pairs.
    """
    return jax.vmap(
        lambda m, d: m2l_complex_reference(m, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, deltas)


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
@jaxtyped(typechecker=beartype)
def m2l_complex_reference_batch_cached_blocks(
    multipoles: Inexact[Array, "_ _"],
    deltas: Float[Array, "_ 3"],
    blocks_to_z: Inexact[Array, "_ _ _ _"],
    blocks_from_z: Inexact[Array, "_ _ _ _"],
    *,
    order: int,
) -> Array:
    """Batch M2L using precomputed rotation blocks for each pair.

    Parameters
    ----------
    multipoles : Inexact[Array, '_ _']
        Batched packed complex multipoles, shape ``(batch, sh_size(order))``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    blocks_to_z : Inexact[Array, '_ _ _ _']
        Padded rotation blocks aligning the pair axis to +z.
    blocks_from_z : Inexact[Array, '_ _ _ _']
        Padded rotation blocks rotating back from +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Local coefficients for a batch, reusing precomputed rotation blocks.
    """
    p = int(order)
    M_rot = _apply_complex_rotation_blocks_padded_batch(
        multipoles,
        blocks_to_z,
        order=p,
    )
    r = jnp.sqrt(floor_squared_radius(jnp.sum(deltas * deltas, axis=1)))
    local_z = translate_along_z_m2l_complex_batch(M_rot, r, order=p)
    return _apply_complex_rotation_blocks_padded_batch(
        local_z,
        blocks_from_z,
        order=p,
    )


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2l_complex_batch(
    multipoles: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex multipoles to locals along +z.

    Parameters
    ----------
    multipoles : Array
        Batched packed complex multipoles, shape ``(batch, sh_size(order))``.
    r : Array
        Centre separation; the z-translation distance after rotation to +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Batched multipole-to-local translation along +z.
    """
    return jax.vmap(
        lambda m, rr: translate_along_z_m2l_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, r)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_m2m_complex_batch(
    multipoles: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex multipoles along +z.

    Parameters
    ----------
    multipoles : Array
        Batched packed complex multipoles, shape ``(batch, sh_size(order))``.
    dz : Array
        Signed translation distance along +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Batched multipole translation along +z.
    """
    return jax.vmap(
        lambda m, rr: translate_along_z_m2m_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(multipoles, dz)


@partial(jax.jit, static_argnames=("order",))
def translate_along_z_l2l_complex_batch(
    locals: Array,
    dz: Array,
    *,
    order: int,
) -> Array:
    """Batch translate complex locals along +z.

    Parameters
    ----------
    locals : Array
        Batched packed complex locals, shape ``(batch, sh_size(order))``.
    dz : Array
        Signed translation distance along +z.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.

    Returns
    -------
    Array
        Batched local translation along +z.
    """
    return jax.vmap(
        lambda m, rr: translate_along_z_l2l_complex(m, rr, order=order),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, dz)


@partial(jax.jit, static_argnames=("order", "rotation"))
@jaxtyped(typechecker=beartype)
def l2l_complex_batch(
    locals: Inexact[Array, "_ _"],
    deltas: Float[Array, "_ 3"],
    *,
    order: int,
    rotation: str = "solidfmm",
) -> Array:
    """Batch L2L in complex basis.

    Parameters
    ----------
    locals : Inexact[Array, '_ _']
        Batched packed complex locals, shape ``(batch, sh_size(order))``.
    deltas : Float[Array, '_ 3']
        Batched displacement vectors, shape ``(batch, 3)``.
    order : int
        Expansion order ``p``. Static: it fixes every packed length and table shape.
    rotation : str
        Rotation convention; ``'solidfmm'`` is the only one wired to production.

    Returns
    -------
    Array
        Child local coefficients for a batch.
    """
    return jax.vmap(
        lambda l, d: l2l_complex(l, d, order=order, rotation=rotation),
        in_axes=(0, 0),
        out_axes=0,
    )(locals, deltas)


#: Re-exported so the complex fused lane's two halves come from one module and cannot be
#: reached apart, exactly as
#: :data:`~jaccpot.operators.m2l_real_rot_scale.m2l_real_fused_align_deltas` is for the
#: real one. This withdraws the unresolvable transverse tangent from the displacement
#: *before* anything is built from it.
#:
#: This lane needs it stated explicitly even though its block builders already carry
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.without_unresolvable_transverse_jvp`:
#: the withdrawal is idempotent, and naming it here is what makes the pairing visible at
#: the call site instead of being an accident of which builder happens to be used.
m2l_complex_fused_align_deltas = withdraw_unresolvable_transverse


#: Puts the analytic transverse derivative back onto the fused Pallas complex M2L's
#: output. Signature
#: ``(out, multipoles, deltas, blocks_to_z, blocks_from_z, radii, order=p)``; the primal
#: returns ``out`` untouched, so there is no forward footprint. Pair it with
#: :data:`m2l_complex_fused_align_deltas` on the displacement *before* the blocks and
#: radius are built from it -- see
#: ``runtime/kernels/core.py::_m2l_complex_batch_kernel_fused_pallas``, its only caller.
#: Using either half alone gives a wrong gradient: without the carrier this lane's
#: on-axis ``d/dx`` and ``d/dy`` come back exactly zero, measured 5.1e-01 from the
#: pure-JAX reference.
def make_m2l_complex_fused_carry_axis_derivative(
    twin: Callable[..., Array],
) -> Callable[..., Array]:
    """Build the fused complex M2L's transverse-tangent carrier around ``twin``.

    The complex counterpart of
    :func:`~jaccpot.operators.m2l_real_rot_scale.make_m2l_real_fused_carry_axis_derivative`,
    and a factory for the same reason: the pure-JAX twin lives in
    :mod:`jaccpot.pallas.m2l_complex_fused`, and ``operators/`` no longer imports
    ``pallas/`` (audit G.3). The derivative rule stays here; only the choice of
    twin moved out. Its one caller is ``runtime/kernels/_m2l.py``.

    Build the result ONCE and reuse it -- it is a ``custom_jvp`` object, so a
    fresh one per call would retrace on every call.

    Parameters
    ----------
    twin : Callable[..., Array]
        Pure-JAX twin, ``twin(coeffs, blocks_to_z, blocks_from_z, radii, order=p)``.
        Must be linear in ``coeffs``.

    Returns
    -------
    Callable[..., Array]
        ``carry(out, coeffs, delta, blocks_to_z, blocks_from_z, radii, order=p)``,
        returning ``out`` unchanged in the primal.
    """
    return with_transverse_degeneracy_tangent(
        twin,
        generators=_M2L_TRANSVERSE_GENERATORS,
    )

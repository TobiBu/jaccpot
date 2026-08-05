"""Pure-JAX real SH rotate+scale-to-z M2L kernels.

This module provides batched helpers for a solidFMM-inspired pipeline:
rotate multipoles into a z-aligned frame, apply z-axis M2L translation,
then rotate local coefficients back.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    real_rotation_from_z_axis_local,
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    real_rotation_to_z_axis_multipole,
    real_transverse_generators,
    sh_offset,
    sh_size,
    translate_along_z_l2l_real,
    translate_along_z_m2l_real,
    translate_along_z_m2m_real,
)

from ._precision import highest_matmul_precision
from ._transverse_degeneracy_jvp import (
    with_transverse_degeneracy_jvp,
    with_transverse_degeneracy_tangent,
    withdraw_unresolvable_transverse,
    without_unresolvable_transverse_jvp,
)

# NOTE: ``jaccpot.pallas.m2l_core_z_real`` is imported lazily inside
# ``m2l_core_z_real`` below. A top-level import creates a circular import
# (``jaccpot.pallas.__init__`` -> ``m2l_core_z_real`` -> ``jaccpot.operators``
# -> this module -> ``jaccpot.pallas.m2l_core_z_real``), which breaks
# ``from jaccpot.pallas import ...`` when it is the first jaccpot import.


@highest_matmul_precision
def _rotate_multipole_to_z_single(
    multipole: Array, delta: Array, *, order: int
) -> Array:
    """Rotate one real multipole expansion into the z-aligned frame."""
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(multipole)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=multipole.dtype)
        out = out.at[sl].set(block @ multipole[sl])
    return out


@highest_matmul_precision
def _rotate_local_from_z_single(local_z: Array, delta: Array, *, order: int) -> Array:
    """Rotate one real local expansion from z-frame back to world frame."""
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(local_z)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_from_z_axis_local(x, y, z, ell, dtype=local_z.dtype)
        out = out.at[sl].set(block @ local_z[sl])
    return out


def m2l_core_z_real(
    multipole_rot: Array,
    radii: Array,
    *,
    order: int,
    use_pallas: bool = False,
) -> Array:
    """Apply z-axis real M2L translation to a batch of rotated multipoles.

    When ``use_pallas=True``, the function dispatches to the optional Pallas
    kernel on supported accelerator backends and otherwise falls back to the
    pure-JAX recurrence.
    """
    from jaccpot.pallas.m2l_core_z_real import (
        m2l_core_z_real_pallas,
        pallas_m2l_real_supported,
    )

    r = jnp.maximum(jnp.asarray(radii), jnp.asarray(1.0e-30, dtype=radii.dtype))
    if bool(use_pallas) and pallas_m2l_real_supported():
        return m2l_core_z_real_pallas(multipole_rot, r, order=int(order))
    return jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=int(order)))(
        multipole_rot,
        r,
    )


def _m2l_rot_scale_real_cascade(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    use_pallas: bool = False,
) -> Array:
    """The rotate -> z-translate -> rotate-back body of :func:`m2l_rot_scale_real_batch`.

    Split out from the public entry point so the shape validation stays outside the
    ``custom_jvp`` that :func:`~jaccpot.operators._transverse_degeneracy_jvp.with_transverse_degeneracy_jvp`
    wraps around this. Takes already-``asarray``-ed inputs and does not re-check them.

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients ``[N, (p+1)^2]``.
    deltas : Array
        Source-to-target centre displacements ``[N, 3]``.
    order : int
        Maximum SH degree ``p``. Static under ``jit``.
    use_pallas : bool
        Route the z-translation through the Pallas kernel where supported. Static.

    Returns
    -------
    Array
        Real local expansion contributions ``[N, (p+1)^2]``.
    """
    # NaN-safe radius: ``linalg.norm`` has a 0/0 reverse grad at delta==0.
    # Double-where keeps the cotangent finite (0) there; forward is unchanged
    # (sqrt of the squared norm equals the norm for every input).
    r2 = jnp.sum(deltas * deltas, axis=1)
    r2_pos = r2 > 0
    radii = jnp.where(r2_pos, jnp.sqrt(jnp.where(r2_pos, r2, 1.0)), 0.0)
    mult_rot = jax.vmap(
        lambda m, d: _rotate_multipole_to_z_single(m, d, order=int(order))
    )(
        multipoles,
        deltas,
    )
    locals_z = m2l_core_z_real(
        mult_rot,
        radii,
        order=int(order),
        use_pallas=bool(use_pallas),
    )
    return jax.vmap(lambda l, d: _rotate_local_from_z_single(l, d, order=int(order)))(
        locals_z,
        deltas,
    )


#: :func:`_m2l_rot_scale_real_cascade` with the analytic transverse derivative at
#: ``rho == 0`` attached. The primal is untouched; see
#: :mod:`jaccpot.operators._transverse_degeneracy_jvp`.
_m2l_rot_scale_real_cascade_with_axis_derivative = with_transverse_degeneracy_jvp(
    _m2l_rot_scale_real_cascade,
    generators=partial(
        real_transverse_generators,
        in_representation="multipole",
        out_representation="local",
    ),
)


def m2l_rot_scale_real_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    use_pallas: bool = False,
) -> Array:
    """Batched rotate+scale real-basis M2L translation.

    Applies the rotate -> z-translate -> rotate-back decomposition per pair:
    :func:`_rotate_multipole_to_z_single`, then :func:`m2l_core_z_real`, then
    :func:`_rotate_local_from_z_single`. The ordering is load-bearing for
    cancellation and must not be restructured.

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients ``[N, (p+1)^2]``, real (no-sqrt2) Dehnen
        basis, packed by :func:`~jaccpot.operators.real_harmonics.sh_offset`.
    deltas : Array
        Source-to-target centre displacements ``[N, 3]``. Same length convention
        as the coefficients; G=1 is a caller convention, not applied here.
    order : int
        Maximum SH degree ``p``. Static under ``jit`` -- it sets the Python-level
        loop bounds in the rotation builders.
    use_pallas : bool
        Route the z-translation through the optional Pallas kernel when the
        backend supports it. Static under ``jit``. The kernel is an execution
        accelerator only; see :func:`m2l_core_z_real` for the equivalence.

    Returns
    -------
    Array
        Real local expansion contributions ``[N, (p+1)^2]``, same dtype and
        packing as ``multipoles``.

    Raises
    ------
    ValueError
        If ``multipoles`` is not 2-D, or ``deltas`` is not ``[N, 3]``. Raised at
        trace time on static shapes, so it cannot fire inside a compiled step.

    Notes
    -----
    Differentiable in ``multipoles`` and ``deltas`` under both forward and
    reverse mode. The radius is computed as a double-``where`` guarded
    ``sqrt`` rather than ``linalg.norm``, because the latter has a 0/0 reverse
    gradient at ``delta == 0``; the guard keeps that cotangent finite (zero)
    while leaving the forward value unchanged.

    A ``custom_jvp`` supplies the transverse (``d/dx``, ``d/dy``) derivative near the
    ``rho == 0`` axis, where the alignment azimuth is undefined -- the guards in
    :func:`~jaccpot.operators.real_harmonics._multipole_align_to_z_block` return a zero
    cotangent there, exact forward and wrong derivative -- and where the polar route
    degrades like ``eps / (rho / r)`` on approach. The analytic branch applies inside
    exactly zero outside a narrow band around that axis (``rho <= sqrt(eps) * |delta|``, the measured crossover between the two routes' errors); outside it the primal and the gradient are bit-identical to what this
    function computed before. See :mod:`jaccpot.operators._transverse_degeneracy_jvp`.

    ``delta == 0`` is degenerate for M2L and this function does not reject it:
    :func:`m2l_core_z_real` floors the radius at ``1e-30``, so the result is
    finite but physically meaningless. The MAC that makes a pair well-separated
    is the caller's responsibility -- nothing here checks it.

    **On accuracy: this function is the reference, and its own accuracy is not
    measured.** Every test that touches the rotate+scale path compares something
    else *to it* -- ``tests/test_m2l_real_fused_pallas.py`` checks the fused
    pure-jnp twin and the Pallas kernel against it (rel err <1e-10 at fp64 for
    orders 2, 3, 4; <3e-4 at fp32), and
    ``tests/unit/operators/test_m2l_real_rot_scale.py::test_cached_blocks_m2l_matches_direct_batch``
    checks the cached-block variant against it (order 3, 5 pairs, ``atol=1e-9``).
    Those pin *consistency*, not correctness: they would all still pass if this
    decomposition were uniformly wrong.

    Neither axis of the accuracy regime is measured anywhere: not the truncation
    error against a direct-summation or analytic reference as a function of ``p``,
    and not the dependence on the source-target separation ratio. What ``docs/``
    does record is the ~6e-04 TF32 floor this path avoids via
    :mod:`jaccpot.operators._precision` -- a floor on the arithmetic, not a bound
    on the scheme. Treat a required accuracy here as something to verify for your
    own configuration rather than something this docstring can promise.
    """
    mult = jnp.asarray(multipoles)
    delta = jnp.asarray(deltas)
    if mult.ndim != 2:
        raise ValueError("multipoles must have shape (batch, coeffs)")
    if delta.ndim != 2 or int(delta.shape[1]) != 3:
        raise ValueError("deltas must have shape (batch, 3)")

    return _m2l_rot_scale_real_cascade_with_axis_derivative(
        mult, delta, order=int(order), use_pallas=bool(use_pallas)
    )


# ---------------------------------------------------------------------------
# Grouped (per-interaction-class) real M2L: precompute the real rotation blocks
# once per class and reuse them across all pairs that share the class geometry.
# This mirrors the complex cached-blocks path and removes the per-pair rotation
# construction cost, which dominates the rotate/scale M2L.
# ---------------------------------------------------------------------------


def _pack_by_ell(coeffs: Array, *, order: int) -> Array:
    """Pack (p+1)^2 coefficients into a padded (p+1, 2p+1) array."""
    p = int(order)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m), dtype=coeffs.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[ell, : 2 * ell + 1].set(coeffs[sl])
    return out


def _unpack_by_ell(packed: Array, *, order: int) -> Array:
    """Inverse of :func:`_pack_by_ell`."""
    p = int(order)
    out = jnp.zeros((sh_size(p),), dtype=packed.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[sl].set(packed[ell, : 2 * ell + 1])
    return out


@partial(jax.jit, static_argnames=("order",))
def _apply_real_rotation_blocks_padded_batch(
    coeffs: Array, blocks_array: Array, *, order: int
) -> Array:
    """Apply padded per-degree real rotation blocks to a batch of coefficients.

    ``blocks_array`` has shape ``(batch, p+1, 2p+1, 2p+1)`` (block-diagonal per
    degree, zero-padded); ``coeffs`` has shape ``(batch, (p+1)^2)``.
    """
    packed = jax.vmap(lambda c: _pack_by_ell(c, order=order))(coeffs)
    rotated = jnp.einsum(
        "nbij,nbj->nbi", blocks_array, packed, precision=lax.Precision.HIGHEST
    )
    return jax.vmap(lambda c: _unpack_by_ell(c, order=order))(rotated)


@without_unresolvable_transverse_jvp
def _real_rotation_blocks_padded(
    deltas: Array, *, order: int, dtype: Any, which: str
) -> Array:
    """Padded real rotation blocks for a batch of displacement vectors.

    Selects the per-degree rotation op by ``which``:

    * ``'to_z_multipole'`` / ``'from_z_multipole'`` -- multipole world<->z (M2L step 1, M2M);
    * ``'to_z_local'`` / ``'from_z_local'`` -- local world<->z (L2L, M2L step 3).

    Differentiable in ``deltas``, but deliberately **not** transversally near the
    ``rho == 0`` axis: an individual alignment block has no transverse derivative there
    (its limit depends on the approach direction, unlike the assembled cascade's), so
    inside the band of
    :func:`~jaccpot.operators._transverse_degeneracy_jvp.split_transverse_tangent` this
    hands its caller nothing rather than something wrong. The caller supplies the
    cascade-level term -- see
    :func:`m2l_rot_scale_real_batch_cached_blocks`. Outside the band nothing changes, bit
    for bit, and the primal is untouched everywhere.
    """
    p = int(order)
    max_m = 2 * p + 1
    if which == "to_z_multipole":
        rot_fn = real_rotation_to_z_axis_multipole
    elif which == "from_z_multipole":
        rot_fn = real_rotation_from_z_axis_multipole
    elif which == "to_z_local":
        rot_fn = real_rotation_to_z_axis_local
    elif which == "from_z_local":
        rot_fn = real_rotation_from_z_axis_local
    else:
        raise ValueError(
            "which must be one of 'to_z_multipole', 'from_z_multipole', "
            "'to_z_local', 'from_z_local'"
        )

    def one(delta: Array) -> Array:
        x, y, z = delta[0], delta[1], delta[2]
        out = jnp.zeros((p + 1, max_m, max_m), dtype=dtype)
        for ell in range(p + 1):
            size = 2 * ell + 1
            block = rot_fn(x, y, z, ell, dtype=dtype)
            out = out.at[ell, :size, :size].set(block)
        return out

    return jax.vmap(one)(jnp.asarray(deltas))


def real_rotation_blocks_to_z_multipole_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real multipole world->z rotation blocks, one set per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="to_z_multipole"
    )


def real_rotation_blocks_from_z_local_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real local z->world rotation blocks, one set per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="from_z_local"
    )


def real_rotation_blocks_from_z_multipole_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real multipole z->world rotation blocks (M2M rotate-back), per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="from_z_multipole"
    )


def real_rotation_blocks_to_z_local_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real local world->z rotation blocks (L2L rotate-to-z), per delta."""
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="to_z_local"
    )


#: :func:`real_transverse_generators` bound to each cached-blocks lane's pair of
#: representations. Shares the generators with the direct lanes in
#: :mod:`jaccpot.operators.real_harmonics`; kept local because that module's copies are
#: private to it.
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


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2L_TRANSVERSE_GENERATORS)
def m2l_rot_scale_real_batch_cached_blocks(
    multipoles: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Batched real-basis M2L using precomputed per-pair rotation blocks.

    Equivalent to :func:`m2l_rot_scale_real_batch` but the (expensive) real
    rotation matrices are supplied precomputed (typically shared across an
    interaction class), so only the z-axis translation runs per pair.

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients ``[N, (p+1)^2]``, real Dehnen basis.
    deltas : Array
        Source-to-target centre displacements ``[N, 3]``. Used *only* for the
        translation radius here -- the direction already lives in the supplied
        blocks, so passing blocks built from different deltas is silently wrong
        and is not checked.
    blocks_to_z : Array
        Multipole world->z rotation blocks ``[N, p+1, 2p+1, 2p+1]``,
        block-diagonal per degree and zero-padded, as built by
        :func:`real_rotation_blocks_to_z_multipole_batch`.
    blocks_from_z : Array
        Local z->world rotation blocks, same shape and padding, as built by
        :func:`real_rotation_blocks_from_z_local_batch`.
    order : int
        Maximum SH degree ``p``. Static under ``jit`` (declared in
        ``static_argnames``).

    Returns
    -------
    Array
        Real local expansion contributions ``[N, (p+1)^2]``.

    Notes
    -----
    Differentiable in ``multipoles``, ``deltas``, and both block arrays.

    Equivalent to :func:`m2l_rot_scale_real_batch` when the blocks are the ones
    that function would have built for the same ``deltas``; asserted by
    ``tests/unit/operators/test_m2l_real_rot_scale.py::test_cached_blocks_m2l_matches_direct_batch``.
    It reaches the degenerate radius differently, though -- ``sqrt(maximum(r2,
    1e-60))`` here versus a double-``where`` plus the ``1e-30`` floor inside
    :func:`m2l_core_z_real` there. Both land on a ``1e-30`` radius and a zero
    ``delta`` cotangent at ``delta == 0``, so the pair still agrees, but the two
    guards are not the same expression and should not be assumed to stay in step.
    """
    p = int(order)
    mult_rot = _apply_real_rotation_blocks_padded_batch(
        jnp.asarray(multipoles), jnp.asarray(blocks_to_z), order=p
    )
    radii = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.asarray(deltas) * jnp.asarray(deltas), axis=1), 1.0e-60)
    )
    locals_z = jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=p))(
        mult_rot, radii
    )
    return _apply_real_rotation_blocks_padded_batch(
        locals_z, jnp.asarray(blocks_from_z), order=p
    )


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_M2M_TRANSVERSE_GENERATORS)
def m2m_rot_scale_real_batch_cached_blocks(
    multipoles: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Batched real-basis M2M using precomputed per-node rotation blocks.

    The real analog of :func:`m2l_rot_scale_real_batch_cached_blocks`, for the up-sweep
    (multipole -> multipole). Equivalent to :func:`~jaccpot.operators.real_harmonics.m2m_real`
    but the (expensive) rotation matrices are supplied precomputed (shared across all nodes in
    a displacement class), so only the z-axis M2M translation runs per node. ``deltas`` is the
    ``child_center - parent_center`` displacement (M2M convention, source=child); its norm is
    the +z translation distance. ``blocks_to_z`` are multipole world->z blocks
    (``real_rotation_blocks_to_z_multipole_batch``), ``blocks_from_z`` are multipole z->world
    blocks (``real_rotation_blocks_from_z_multipole_batch``).
    """
    p = int(order)
    mult_rot = _apply_real_rotation_blocks_padded_batch(
        jnp.asarray(multipoles), jnp.asarray(blocks_to_z), order=p
    )
    radii = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.asarray(deltas) * jnp.asarray(deltas), axis=1), 1.0e-60)
    )
    mult_z = jax.vmap(lambda m, rr: translate_along_z_m2m_real(m, rr, order=p))(
        mult_rot, radii
    )
    return _apply_real_rotation_blocks_padded_batch(
        mult_z, jnp.asarray(blocks_from_z), order=p
    )


@partial(jax.jit, static_argnames=("order",))
@partial(with_transverse_degeneracy_jvp, generators=_L2L_TRANSVERSE_GENERATORS)
def l2l_rot_scale_real_batch_cached_blocks(
    locals_coeffs: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Batched real-basis L2L using precomputed per-node rotation blocks.

    The real analog of :func:`m2l_rot_scale_real_batch_cached_blocks`, for the down-sweep
    (local -> local). Equivalent to :func:`~jaccpot.operators.real_harmonics.l2l_real` but the
    rotation matrices are supplied precomputed (shared across a displacement class), so only the
    z-axis L2L translation runs per node. ``deltas`` is the ``parent_center - child_center``
    displacement (L2L convention, source=parent -- OPPOSITE sign of M2M). ``blocks_to_z`` are
    local world->z blocks (``real_rotation_blocks_to_z_local_batch``), ``blocks_from_z`` are
    local z->world blocks (``real_rotation_blocks_from_z_local_batch``).
    """
    p = int(order)
    loc_rot = _apply_real_rotation_blocks_padded_batch(
        jnp.asarray(locals_coeffs), jnp.asarray(blocks_to_z), order=p
    )
    radii = jnp.sqrt(
        jnp.maximum(jnp.sum(jnp.asarray(deltas) * jnp.asarray(deltas), axis=1), 1.0e-60)
    )
    loc_z = jax.vmap(lambda l, rr: translate_along_z_l2l_real(l, rr, order=p))(
        loc_rot, radii
    )
    return _apply_real_rotation_blocks_padded_batch(
        loc_z, jnp.asarray(blocks_from_z), order=p
    )


# --------------------------------------------------------------------------
# The fused Pallas M2L's transverse derivative, added rather than differentiated.
# --------------------------------------------------------------------------


def _m2l_real_fused_twin(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    radii: Array,
    *,
    order: int,
) -> Array:
    """Pure-JAX twin of the fused Pallas real M2L, for the transverse correction only.

    Deferred import: :mod:`jaccpot.pallas.m2l_real_fused` reaches back into
    :mod:`jaccpot.operators`, so a module-scope import is a cycle. This is the same twin
    the fused kernel's own ``custom_vjp`` uses as its correctness reference, which is why
    it is the right thing to compute the correction with -- the kernel itself cannot be
    used, because the correction lives inside a JVP rule and a ``custom_vjp`` is not
    forward-differentiable.

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients ``[N, (p+1)^2]``.
    blocks_to_z : Array
        Multipole world->z blocks ``[N, p+1, 2p+1, 2p+1]``.
    blocks_from_z : Array
        Local z->world blocks, same shape.
    radii : Array
        Translation distances ``[N]``.
    order : int
        Maximum SH degree ``p``. Static under ``jit``.

    Returns
    -------
    Array
        Real local expansion contributions ``[N, (p+1)^2]``.
    """
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_jax

    return m2l_real_fused_jax(
        multipoles, blocks_to_z, blocks_from_z, radii, order=int(order)
    )


#: Re-exported so the fused lane's two halves come from one module and cannot be reached
#: apart: this one withdraws the unresolvable transverse tangent from the displacement
#: *before* anything is built from it, and :data:`m2l_real_fused_carry_axis_derivative`
#: puts the analytic term back *after* the kernel. Using either alone gives a wrong
#: gradient. See :mod:`jaccpot.operators._transverse_degeneracy_jvp` for both.
m2l_real_fused_align_deltas = withdraw_unresolvable_transverse

#: Puts the analytic transverse derivative back onto the fused Pallas M2L's output.
#: Signature ``(out, multipoles, deltas, blocks_to_z, blocks_from_z, radii, order=p)``;
#: the primal returns ``out`` untouched. Pair it with
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.withdraw_unresolvable_transverse`
#: on the displacement *before* the blocks and radius are built from it -- see
#: ``runtime/kernels/core.py::_m2l_real_batch_kernel_fused_pallas``, its only caller.
m2l_real_fused_carry_axis_derivative = with_transverse_degeneracy_tangent(
    _m2l_real_fused_twin,
    generators=_M2L_TRANSVERSE_GENERATORS,
)


__all__ = [
    "m2l_core_z_real",
    "m2l_real_fused_align_deltas",
    "m2l_real_fused_carry_axis_derivative",
    "m2l_rot_scale_real_batch",
    "m2l_rot_scale_real_batch_cached_blocks",
    "m2m_rot_scale_real_batch_cached_blocks",
    "l2l_rot_scale_real_batch_cached_blocks",
    "real_rotation_blocks_to_z_multipole_batch",
    "real_rotation_blocks_from_z_local_batch",
    "real_rotation_blocks_from_z_multipole_batch",
    "real_rotation_blocks_to_z_local_batch",
]

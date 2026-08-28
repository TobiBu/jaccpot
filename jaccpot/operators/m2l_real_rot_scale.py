"""Pure-JAX real SH rotate+scale-to-z M2L kernels.

This module provides batched helpers for a solidFMM-inspired pipeline:
rotate multipoles into a z-aligned frame, apply z-axis M2L translation,
then rotate local coefficients back.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot._env import env_flag
from jaccpot.operators.real_dehnen_q import (
    compute_real_B_matrix_multipole,
)
from jaccpot.operators.real_harmonics import (
    real_Dz_diagonal,
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

# Private, and imported from where it is DEFINED rather than through a
# re-export: `real_harmonics` used to re-export it and main has since stopped,
# which broke CI while the branch head still imported fine.
from jaccpot.operators.real_rotations import _alignment_angles

from ._precision import highest_matmul_precision
from ._transverse_degeneracy_jvp import (
    with_transverse_degeneracy_jvp,
    with_transverse_degeneracy_tangent,
    withdraw_unresolvable_transverse,
    without_unresolvable_transverse_jvp,
)


@highest_matmul_precision
def _rotate_multipole_to_z_single(
    multipole: Array, delta: Array, *, order: int
) -> Array:
    """Rotate one real multipole expansion into the z-aligned frame.

    Block-diagonal by degree: each ``ell`` block is built and applied on its own,
    because a real rotation mixes orders only within a degree.

    Parameters
    ----------
    multipole : Array
        Packed real multipole coefficients ``[(p+1)^2]``.
    delta : Array
        Displacement ``[3]`` whose direction defines the z-alignment.
    order : int
        Expansion order ``p``. A Python int -- the degree loop is unrolled at
        trace time.

    Returns
    -------
    Array
        Coefficients in the z-aligned frame, same shape and dtype.
    """
    if _degree_batched():
        return _rotate_degree_batched(
            multipole, delta, order=int(order), local_side=False
        )
    x, y, z = delta[0], delta[1], delta[2]
    out = jnp.zeros_like(multipole)
    for ell in range(int(order) + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        block = real_rotation_to_z_axis_multipole(x, y, z, ell, dtype=multipole.dtype)
        out = out.at[sl].set(block @ multipole[sl])
    return out


def _degree_batched() -> bool:
    """Whether to collapse the per-degree launches into the batched form.

    Off by default until the A/B lands; see :func:`_rotate_degree_batched`.

    Read at call time, not captured into a module-level constant. It used to be
    the latter, which is the defect ``jaccpot._env`` exists to prevent: a knob
    resolved at import cannot be changed by anyone who sets the variable after
    ``import jaccpot``, so it silently does nothing -- worse, as that module's
    docstring puts it, than not having the knob at all. Both call sites are
    trace-time branches, so this costs one lookup per trace, not per element.

    Returns
    -------
    bool
        ``True`` when the batched rotation path should be taken.
    """
    return env_flag("JACCPOT_M2L_DEGREE_BATCHED", False)


def _centred_degree_maps(order: int) -> tuple[Array, Array]:
    """Gather indices and mask putting every degree in ONE centred width-(2p+1) row.

    Within a degree the packed layout is ``m = -ell..ell`` at index ``ell + m``, so
    ``m = 0`` sits at the block *centre*, not a corner. Centring the padding is
    therefore what makes the padded blocks degree-independent: degree ``ell``
    occupies ``p - ell .. p + ell`` of a width ``2p + 1`` row, and ``m`` lands at
    ``p + m`` for every degree at once.

    Parameters
    ----------
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    tuple[Array, Array]
        ``(idx, mask)``, both ``(p + 1, 2p + 1)``: the global coefficient index of
        each padded slot, and whether that slot is a real coefficient.
    """
    p = int(order)
    width = 2 * p + 1
    idx = np.zeros((p + 1, width), dtype=np.int32)
    mask = np.zeros((p + 1, width), dtype=bool)
    for ell in range(p + 1):
        for m in range(-ell, ell + 1):
            idx[ell, p + m] = sh_offset(ell) + ell + m
            mask[ell, p + m] = True
    return jnp.asarray(idx), jnp.asarray(mask)


def _padded_dz(order: int, angle: Array, *, dtype: Any) -> Array:
    """``real_Dz_diagonal`` at full width, valid for EVERY degree at once.

    A z-rotation is block-diagonal in ``|m|`` -- it never mixes different ``|m|``
    and never mixes degrees -- so one width-``(2p+1)`` construction serves all
    degrees under the centred layout, and the blocks for ``|m| > ell`` act only on
    padded slots that are zero. That is what lets this hoist out of the degree
    axis entirely instead of being rebuilt ``p + 1`` times.

    Parameters
    ----------
    order : int
        Expansion order ``p``. Static.
    angle : Array
        Rotation angle about ``+z``.
    dtype : Any
        Output dtype.

    Returns
    -------
    Array
        ``(2p + 1, 2p + 1)`` z-rotation.
    """
    p = int(order)
    return real_Dz_diagonal(p, angle, dtype=dtype)


def _rotate_degree_batched(
    coeffs: Array, delta: Array, *, order: int, local_side: bool
) -> Array:
    """One batched einsum over degrees, instead of ``p + 1`` unrolled blocks.

    Same arithmetic, restructured. The unrolled loop is what makes M2L
    launch-bound: measured on an A100, the two rotations account for ~99% of the
    stage's compiled kernels (order 4: 108 fusions each, against 2 for the actual
    z-axis translation), and M2L's wall time tracks the kernel count rather than
    the flops -- order 2 to 6 grows the fusion count 4.32x, the measured time
    4.68x and the arithmetic 29.6x.

    The restructure is possible because of what the per-degree block actually is:
    ``B_U @ Dz(-ax) @ B_U @ Dz(az)``, where ``B_U`` depends only on the degree
    (a compile-time constant) and ``Dz`` is block-diagonal in ``|m|``. So the
    angle-dependent factors hoist out of the degree axis, the constant factors
    become one padded stack, and what remains is a single batched contraction over
    a ``p + 1 <= 7`` axis.

    The padding costs arithmetic -- dense ``(p+1)(2p+1)^2`` against the ragged
    ``sum_ell (2ell+1)^2``, i.e. 2.45x at order 4 -- which is close to free at
    0.24% of fp64 peak.

    Parameters
    ----------
    coeffs : Array
        ``(sh_size(order),)`` coefficients for one pair.
    delta : Array
        ``(3,)`` target centre minus source centre.
    order : int
        Expansion order ``p``. Static.
    local_side : bool
        Build the local from-z rotation rather than the multipole to-z one. The
        two are different matrices, not transposes of each other.

    Returns
    -------
    Array
        ``(sh_size(order),)`` rotated coefficients.
    """
    p = int(order)
    dtype = coeffs.dtype
    x, y, z = delta[0], delta[1], delta[2]
    width = 2 * p + 1
    idx, mask = _centred_degree_maps(p)

    # The whole point: assemble the block from its FACTORS rather than calling the
    # per-degree builder p+1 times.
    #
    # `_multipole_align_to_z_block` is `B_U @ Dz(-ax) @ B_U @ Dz(az)`, in which
    # `B_U` depends only on the degree and `Dz` is block-diagonal in |m|. So:
    #
    #   * the `B_U` stack is a compile-time CONSTANT -- built with numpy so it
    #     lowers to a literal and costs no kernels at all;
    #   * `Dz` is built TWICE at full width instead of 2(p+1) times, because under
    #     the centred layout its |m| blocks map the active index set to itself, so
    #     one width-(2p+1) construction restricts correctly to every degree.
    #
    # An earlier attempt batched only the *application* and moved the fusion count
    # 108 -> 100 at order 4 (~7%), because the per-degree construction is ~93% of
    # it. This is the version that actually removes the per-degree launches.
    # `ensure_compile_time_eval` is load-bearing: `compute_real_B_matrix_multipole`
    # is itself jitted, so calling it inside this trace yields tracers rather than
    # values, and the stack would become traced work instead of the literal it
    # should be. Evaluating eagerly here folds it to a constant.
    with jax.ensure_compile_time_eval():
        b_stack = jnp.stack(
            [
                jnp.zeros((width, width), dtype=dtype)
                .at[p - ell : p + ell + 1, p - ell : p + ell + 1]
                .set(compute_real_B_matrix_multipole(ell, dtype=dtype))
                for ell in range(p + 1)
            ]
        )

    az, ax = _alignment_angles(x, y, z)
    dz_az = _padded_dz(p, az, dtype=dtype)
    dz_ax = _padded_dz(p, -ax, dtype=dtype)

    # `precision=HIGHEST` on every one of these, and it is not decoration: XLA
    # lowers fp32 matmuls to TF32 on Ampere+, which measured a ~6e-04 ceiling on
    # M2L relative accuracy from order 4 up -- i.e. raising the order bought
    # nothing. The CPU parity test cannot catch it, because TF32 does not exist
    # there; `test_every_jnp_dot_call_sets_precision` is what caught it here.
    blocks = jnp.einsum("lij,jk->lik", b_stack, dz_ax, precision=lax.Precision.HIGHEST)
    blocks = jnp.einsum(
        "lik,lkm->lim", blocks, b_stack, precision=lax.Precision.HIGHEST
    )
    blocks = jnp.einsum("lim,mn->lin", blocks, dz_az, precision=lax.Precision.HIGHEST)
    if local_side:
        # The local from-z block is the transpose of the multipole world->z block.
        blocks = jnp.swapaxes(blocks, -1, -2)

    rows = jnp.where(mask, coeffs[idx], jnp.zeros((), dtype=dtype))
    out_rows = jnp.einsum("lin,ln->li", blocks, rows, precision=lax.Precision.HIGHEST)
    out = jnp.zeros_like(coeffs)
    return out.at[idx].add(jnp.where(mask, out_rows, jnp.zeros((), dtype=dtype)))


@highest_matmul_precision
def _rotate_local_from_z_single(local_z: Array, delta: Array, *, order: int) -> Array:
    """Rotate one real local expansion from z-frame back to world frame.

    The inverse partner of :func:`_rotate_multipole_to_z_single`, and a different
    matrix: the local rotation is not the multipole rotation transposed. Both are
    degree-block-diagonal.

    Parameters
    ----------
    local_z : Array
        Packed real local coefficients ``[(p+1)^2]`` in the z-aligned frame.
    delta : Array
        The same displacement ``[3]`` used to align.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Coefficients back in the world frame.
    """
    if _degree_batched():
        return _rotate_degree_batched(local_z, delta, order=int(order), local_side=True)
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
) -> Array:
    """Apply z-axis real M2L translation to a batch of rotated multipoles.

    Pure JAX. The Pallas z-core is reached by calling
    :func:`jaccpot.pallas.m2l_core_z_real.m2l_core_z_real_pallas` directly, from
    `runtime/kernels/` or from a parity test -- this module does not dispatch to
    it, so `operators/` stays free of accelerator imports (audit G.3).

    Radii are floored at 1e-30 before use. That is not cosmetic: the z-translation
    divides by ``r``, so a coincident pair would produce inf/NaN rather than a
    large-but-finite contribution, and the floor keeps the kernel total rather
    than making the caller pre-screen.

    Parameters
    ----------
    multipole_rot : Array
        Multipoles already rotated into their z-aligned frames,
        ``[N, (p+1)^2]``.
    radii : Array
        Pair separations ``[N]``, floored as above.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    Array
        Local coefficients in the z-aligned frame, ``[N, (p+1)^2]``. The two
        routes are numerically equivalent -- see the module docstring of
        ``runtime/kernels/_m2l.py`` for the equivalence the suite asserts.
    """
    r = jnp.maximum(jnp.asarray(radii), jnp.asarray(1.0e-30, dtype=radii.dtype))
    return jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=int(order)))(
        multipole_rot,
        r,
    )


def _m2l_rot_scale_real_cascade(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
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
    else *to it* -- ``tests/unit/operators/test_m2l_real_fused_pallas.py`` checks the fused
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
        mult, delta, order=int(order)
    )


# ---------------------------------------------------------------------------
# Grouped (per-interaction-class) real M2L: precompute the real rotation blocks
# once per class and reuse them across all pairs that share the class geometry.
# This mirrors the complex cached-blocks path and removes the per-pair rotation
# construction cost, which dominates the rotate/scale M2L.
# ---------------------------------------------------------------------------


def _pack_by_ell(coeffs: Array, *, order: int) -> Array:
    """Pack (p+1)^2 coefficients into a padded (p+1, 2p+1) array.

    Rectangular so a whole batch of degree blocks can go through one ``einsum``;
    degree ``ell`` uses only the first ``2*ell+1`` columns and the rest stay zero.

    Parameters
    ----------
    coeffs : Array
        Packed coefficients ``[(p+1)^2]``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        ``[p+1, 2p+1]``, zero-padded to the right within each row.
    """
    p = int(order)
    max_m = 2 * p + 1
    out = jnp.zeros((p + 1, max_m), dtype=coeffs.dtype)
    for ell in range(p + 1):
        sl = slice(sh_offset(ell), sh_offset(ell + 1))
        out = out.at[ell, : 2 * ell + 1].set(coeffs[sl])
    return out


def _unpack_by_ell(packed: Array, *, order: int) -> Array:
    """Inverse of :func:`_pack_by_ell`.

    Parameters
    ----------
    packed : Array
        Padded ``[p+1, 2p+1]`` coefficients.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Packed ``[(p+1)^2]``. The padding columns are dropped, so this is only
        the inverse for arrays that came from :func:`_pack_by_ell`.
    """
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

    One ``einsum`` at ``Precision.HIGHEST`` over the padded form, rather than a
    per-degree loop: the padding costs arithmetic on zeros but removes the
    per-degree launches, and the precision is pinned here for the same reason it
    is pinned in ``operators/_precision.py``.

    Parameters
    ----------
    coeffs : Array
        Coefficients to rotate, ``[N, (p+1)^2]``.
    blocks_array : Array
        Per-degree rotation blocks, ``[N, p+1, 2p+1, 2p+1]``.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    Array
        Rotated coefficients, ``[N, (p+1)^2]``.
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

    Parameters
    ----------
    deltas : Array
        Displacement vectors ``[N, 3]``.
    order : int
        Expansion order ``p``.
    dtype : Any
        Dtype to build the blocks in.
    which : str
        Which of the four rotations to build; see the list above.

    Returns
    -------
    Array
        Padded blocks ``[N, p+1, 2p+1, 2p+1]``.

    Raises
    ------
    ValueError
        If ``which`` is not one of the four names listed above.
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
    """Padded real multipole world->z rotation blocks, one set per delta.

    Step 1 of the M2L cascade, and the rotate-to-z of M2M.

    Parameters
    ----------
    deltas : Array
        Displacement vectors ``[N, 3]``; each defines one alignment.
    order : int
        Expansion order ``p``. Sets the block shape ``(p+1, 2p+1, 2p+1)``.
    dtype : Any
        Dtype to build the blocks in. Take it from the coefficients, so the
        cached kernel does not have to promote.

    Returns
    -------
    Array
        Padded per-degree rotation blocks, ``[N, p+1, 2p+1, 2p+1]``.
    """
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="to_z_multipole"
    )


def real_rotation_blocks_from_z_local_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real local z->world rotation blocks, one set per delta.

    Step 3 of the M2L cascade. Not the transpose of the multipole world->z
    blocks -- the local and multipole representations rotate differently.

    Parameters
    ----------
    deltas : Array
        Displacement vectors ``[N, 3]``; each defines one alignment.
    order : int
        Expansion order ``p``. Sets the block shape ``(p+1, 2p+1, 2p+1)``.
    dtype : Any
        Dtype to build the blocks in. Take it from the coefficients, so the
        cached kernel does not have to promote.

    Returns
    -------
    Array
        Padded per-degree rotation blocks, ``[N, p+1, 2p+1, 2p+1]``.
    """
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="from_z_local"
    )


def real_rotation_blocks_from_z_multipole_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real multipole z->world rotation blocks (M2M rotate-back), per delta.

    Parameters
    ----------
    deltas : Array
        Displacement vectors ``[N, 3]``; each defines one alignment.
    order : int
        Expansion order ``p``. Sets the block shape ``(p+1, 2p+1, 2p+1)``.
    dtype : Any
        Dtype to build the blocks in. Take it from the coefficients, so the
        cached kernel does not have to promote.

    Returns
    -------
    Array
        Padded per-degree rotation blocks, ``[N, p+1, 2p+1, 2p+1]``.
    """
    return _real_rotation_blocks_padded(
        deltas, order=order, dtype=dtype, which="from_z_multipole"
    )


def real_rotation_blocks_to_z_local_batch(
    deltas: Array, *, order: int, dtype: Any
) -> Array:
    """Padded real local world->z rotation blocks (L2L rotate-to-z), per delta.

    Parameters
    ----------
    deltas : Array
        Displacement vectors ``[N, 3]``; each defines one alignment.
    order : int
        Expansion order ``p``. Sets the block shape ``(p+1, 2p+1, 2p+1)``.
    dtype : Any
        Dtype to build the blocks in. Take it from the coefficients, so the
        cached kernel does not have to promote.

    Returns
    -------
    Array
        Padded per-degree rotation blocks, ``[N, p+1, 2p+1, 2p+1]``.
    """
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

    Parameters
    ----------
    multipoles : Array
        Child multipole coefficients ``[N, (p+1)^2]``.
    deltas : Array
        ``child_center - parent_center`` ``[N, 3]``. **Source is the child here**,
        the OPPOSITE sign from :func:`l2l_rot_scale_real_batch_cached_blocks` --
        the two sweeps do not share a convention, and getting it backwards is a
        wrong force rather than an error.
    blocks_to_z : Array
        Multipole world->z blocks, from
        :func:`real_rotation_blocks_to_z_multipole_batch`.
    blocks_from_z : Array
        Multipole z->world blocks, from
        :func:`real_rotation_blocks_from_z_multipole_batch`. Multipole blocks in
        both directions -- an M2M never touches the local representation.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    Array
        Parent-frame multipole contributions ``[N, (p+1)^2]``.
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

    Parameters
    ----------
    locals_coeffs : Array
        Parent local coefficients ``[N, (p+1)^2]``.
    deltas : Array
        ``parent_center - child_center`` ``[N, 3]``. **Source is the parent
        here** -- the opposite sign from
        :func:`m2m_rot_scale_real_batch_cached_blocks`. A sign error in the
        equivalent complex cascade capped accuracy at ~3e-3 for theta >= 0.5
        without failing anything, so this is worth checking rather than assuming.
    blocks_to_z : Array
        Local world->z blocks, from
        :func:`real_rotation_blocks_to_z_local_batch`.
    blocks_from_z : Array
        Local z->world blocks, from
        :func:`real_rotation_blocks_from_z_local_batch`. Local blocks in both
        directions -- an L2L never touches the multipole representation.
    order : int
        Expansion order ``p``. Static.

    Returns
    -------
    Array
        Child-frame local contributions ``[N, (p+1)^2]``.
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


#: Re-exported so the fused lane's two halves come from one module and cannot be reached
#: apart: this one withdraws the unresolvable transverse tangent from the displacement
#: *before* anything is built from it, and :func:`make_m2l_real_fused_carry_axis_derivative`
#: puts the analytic term back *after* the kernel. Using either alone gives a wrong
#: gradient. See :mod:`jaccpot.operators._transverse_degeneracy_jvp` for both.
m2l_real_fused_align_deltas = withdraw_unresolvable_transverse


#: Puts the analytic transverse derivative back onto the fused Pallas M2L's output.
#: Signature ``(out, multipoles, deltas, blocks_to_z, blocks_from_z, radii, order=p)``;
#: the primal returns ``out`` untouched. Pair it with
#: :func:`~jaccpot.operators._transverse_degeneracy_jvp.withdraw_unresolvable_transverse`
#: on the displacement *before* the blocks and radius are built from it -- see
#: ``runtime/kernels/core.py::_m2l_real_batch_kernel_fused_pallas``, its only caller.
def make_m2l_real_fused_carry_axis_derivative(
    twin: Callable[..., Array],
) -> Callable[..., Array]:
    """Build the fused real M2L's transverse-tangent carrier around ``twin``.

    A factory rather than a module-level object because the pure-JAX twin this
    needs lives in :mod:`jaccpot.pallas.m2l_real_fused`, and ``operators/`` no
    longer imports ``pallas/`` (audit G.3). The derivative rule stays here, where
    the rest of the transverse-degeneracy work lives; only the choice of twin
    moved out. Its one caller is ``runtime/kernels/_m2l.py``.

    Worth knowing before moving anything else: the twin is **pure jnp**, not the
    accelerated kernel. It is the fused kernel's own correctness reference, which
    is why it is the right thing to compute the correction with -- the kernel
    cannot be, because the correction lives inside a JVP rule and a
    ``custom_vjp`` is not forward-differentiable. So the dependency this removed
    was never on an accelerator; it was on a reference implementation that
    happens to be filed under ``pallas/``.

    Build the result ONCE and reuse it. It is a ``custom_jvp`` object, so a fresh
    one per call is a fresh primitive per call, and every call would retrace.

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


__all__ = [
    "m2l_core_z_real",
    "m2l_real_fused_align_deltas",
    "make_m2l_real_fused_carry_axis_derivative",
    "m2l_rot_scale_real_batch",
    "m2l_rot_scale_real_batch_cached_blocks",
    "m2m_rot_scale_real_batch_cached_blocks",
    "l2l_rot_scale_real_batch_cached_blocks",
    "real_rotation_blocks_to_z_multipole_batch",
    "real_rotation_blocks_from_z_local_batch",
    "real_rotation_blocks_from_z_multipole_batch",
    "real_rotation_blocks_to_z_local_batch",
]

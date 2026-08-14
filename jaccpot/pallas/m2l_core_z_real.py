"""Optional Pallas kernel for batched real-basis z-axis M2L translation."""

from __future__ import annotations

import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    sh_size,
    translate_along_z_m2l_real,
    z_m2l_translation_tables,
)
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs

try:
    import jax.experimental.pallas as pl
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None


def pallas_m2l_real_supported() -> bool:
    """Return whether the active JAX backend can run the Pallas real M2L kernel.

    Returns
    -------
    bool
        True only on GPU or TPU with ``jax.experimental.pallas`` importable. False
        on CPU, where callers must use the pure-JAX recurrence -- note this is a
        *capability* check, not a correctness one: ``interpret=True`` runs the same
        kernel logic on CPU and the parity test relies on that.
    """

    if pl is None:
        return False
    return jax.default_backend() in ("gpu", "tpu")


@lru_cache(maxsize=32)
def _m2l_translation_tables(order: int) -> tuple[np.ndarray, ...]:
    """Static lookup tables for one fixed order, derived from the SINGLE source.

    The per-term recurrence structure comes entirely from
    :func:`jaccpot.operators.real_harmonics.z_m2l_translation_tables`, so the
    Pallas kernel and the pure-JAX ``translate_along_z_m2l_real`` cannot drift.
    This function only adapts that metadata to the layout the kernel consumes
    (``power`` = radius exponent, plus a materialized factorial value table).

    Parameters
    ----------
    order : int
        Expansion order ``p``. Hashable, which is what lets this be ``lru_cache``d --
        the tables are pure functions of the order and are rebuilt per process, not
        per call.

    Returns
    -------
    tuple[np.ndarray, ...]
        ``(src_index, valid, power, fact_index, sign, factorial)``, in the order the
        kernel's ``in_specs`` list them. NumPy rather than JAX arrays on purpose: they
        are host-side static tables, converted by the caller at the dtype of the
        multipoles it was handed.
    """
    p = int(order)
    src_index, valid, fact_index, r_exponent, sign = z_m2l_translation_tables(p)
    power = r_exponent  # kernel names the radius exponent "power"
    factorial = np.asarray(
        [math.factorial(k) for k in range(2 * p + 1)],
        dtype=np.float64,
    )
    return src_index, valid, power, fact_index, sign, factorial


def _m2l_core_z_real_kernel(
    multipole_ref: KernelRef,
    radius_ref: KernelRef,
    src_index_ref: KernelRef,
    valid_ref: KernelRef,
    power_ref: KernelRef,
    fact_index_ref: KernelRef,
    sign_ref: KernelRef,
    factorial_ref: KernelRef,
    out_ref: KernelRef,
) -> None:
    """Compute one output coefficient for one pair.

    One Pallas program instance per ``(pair, coeff)``: ``program_id(axis=1)`` selects
    the output coefficient row, and the grid's axis 0 selects the pair, so every ref
    below is already narrowed to this pair by ``pallas_call``'s index maps.

    The six table refs are the ``z_m2l_translation_tables`` metadata, which is the
    single source shared with the pure-JAX ``translate_along_z_m2l_real`` -- see
    :func:`_m2l_translation_tables`. They are inputs, not constants, because Pallas
    cannot close over host arrays.

    Parameters
    ----------
    multipole_ref : KernelRef
        Rotated multipole coefficients for this pair, shape ``(1, ncoeff)``.
    radius_ref : KernelRef
        Translation distance along z for this pair, shape ``(1,)``. Floored at
        ``1e-30`` below, so a coincident pair yields a large-but-finite coefficient
        rather than a division by zero.
    src_index_ref : KernelRef
        Source coefficient index per ``(coeff, slot)``. Its ``shape[1]`` is the
        ``fori_loop`` trip count, i.e. the maximum number of source terms any output
        coefficient draws on.
    valid_ref : KernelRef
        Whether each ``(coeff, slot)`` is a real term or padding. Padded slots are
        masked out of the accumulator rather than skipped, keeping the trip count
        static.
    power_ref : KernelRef
        Radius exponent per ``(coeff, slot)``.
    fact_index_ref : KernelRef
        Index into ``factorial_ref`` per ``(coeff, slot)``.
    sign_ref : KernelRef
        Per-output-coefficient sign, shape ``(ncoeff,)``.
    factorial_ref : KernelRef
        Materialized ``k!`` for ``k`` in ``0..2p``, gathered through
        ``fact_index_ref`` rather than recomputed in the kernel.
    out_ref : KernelRef
        Output block for this ``(pair, coeff)``, shape ``(1, 1)``. Written once.

    Returns
    -------
    None
        Pallas kernels return nothing; the result is the write to ``out_ref``.
    """

    coeff_idx = pl.program_id(axis=1)
    dtype = multipole_ref.dtype
    radius = jnp.maximum(
        radius_ref[0],
        jnp.asarray(1.0e-30, dtype=radius_ref.dtype),
    )
    acc0 = jnp.asarray(0.0, dtype=dtype)

    # Loop index: a `fori_loop` tracer, not an `int` (F40).
    def body(slot: Array, acc: Array) -> Array:
        is_valid = valid_ref[coeff_idx, slot]
        src_idx = src_index_ref[coeff_idx, slot]
        exponent = power_ref[coeff_idx, slot]
        fact_num = factorial_ref[fact_index_ref[coeff_idx, slot]]
        coeff = sign_ref[coeff_idx] * fact_num / jnp.power(radius, exponent)
        contrib = coeff.astype(dtype) * multipole_ref[0, src_idx]
        return jnp.where(is_valid, acc + contrib, acc)

    acc = lax.fori_loop(0, src_index_ref.shape[1], body, acc0)
    out_ref[0, 0] = acc


def m2l_core_z_real_pallas(
    multipole_rot: Array,
    radii: Array,
    *,
    order: int,
    interpret: bool = False,
    backend: str = "triton",
) -> Array:
    """Apply batched z-axis M2L translation with Pallas when supported.

    The kernel is only enabled on supported accelerators. On other backends,
    callers should fall back to the pure-JAX implementation. ``interpret=True``
    runs the kernel in Pallas interpret mode (works on CPU) -- used by the
    parity test to exercise the kernel logic without a GPU.

    ``backend`` selects the Pallas GPU lowering; default ``"triton"`` (the
    Mosaic-GPU backend rejects the small per-(pair,coeff) tiles -- its TMA copies
    must be a multiple of the 128-byte warpgroup size, whereas one coeff row is
    (p+1)^2 elements = 100/200 bytes). Consistent with the fused complex M2L path.

    Parameters
    ----------
    multipole_rot : Array
        Multipole coefficients already rotated so the translation is along +z, shape
        ``(batch, sh_size(order))``. The rotation is the caller's job; this kernel is
        only the z-axis core.
    radii : Array
        Per-pair translation distance along z, shape ``(batch,)``.
    order : int
        Expansion order. Static: it fixes the table shapes and the grid, so each
        order compiles its own kernel (hence the ``name=`` suffix).
    interpret : bool
        Run under Pallas interpret mode, which executes CPU semantics with no
        backend lowering. How the parity test exercises this kernel without a GPU.
    backend : str
        Pallas GPU lowering, ``"triton"`` by default -- see the note above on why
        Mosaic-GPU cannot take these tiles.

    Returns
    -------
    Array
        Translated coefficients, same shape and dtype as ``multipole_rot``. An empty
        batch short-circuits to ``zeros_like`` rather than launching a zero-sized
        grid.

    Raises
    ------
    RuntimeError
        If ``jax.experimental.pallas`` could not be imported at all.
    ValueError
        If ``multipole_rot`` is not 2-D, ``radii`` is not 1-D, or the coefficient
        count disagrees with ``sh_size(order)`` -- a shape mismatch here would
        otherwise be absorbed silently by the block specs.
    """

    if pl is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    multipole_rot = jnp.asarray(multipole_rot)
    radii = jnp.asarray(radii)
    if multipole_rot.ndim != 2:
        raise ValueError("multipole_rot must have shape (batch, coeffs)")
    if radii.ndim != 1:
        raise ValueError("radii must have shape (batch,)")

    batch_size, coeff_count = multipole_rot.shape
    if batch_size == 0:
        return jnp.zeros_like(multipole_rot)
    if coeff_count != sh_size(int(order)):
        raise ValueError("multipole_rot coefficient count does not match order")

    (
        src_index_np,
        valid_np,
        power_np,
        fact_index_np,
        sign_np,
        factorial_np,
    ) = _m2l_translation_tables(int(order))
    src_index = jnp.asarray(src_index_np, dtype=jnp.int32)
    valid = jnp.asarray(valid_np, dtype=jnp.bool_)
    power = jnp.asarray(power_np, dtype=jnp.int32)
    fact_index = jnp.asarray(fact_index_np, dtype=jnp.int32)
    sign = jnp.asarray(sign_np, dtype=multipole_rot.dtype)
    factorial = jnp.asarray(factorial_np, dtype=multipole_rot.dtype)

    kernel = pl.pallas_call(
        _m2l_core_z_real_kernel,
        out_shape=jax.ShapeDtypeStruct(multipole_rot.shape, multipole_rot.dtype),
        grid=(batch_size, coeff_count),
        in_specs=[
            pl.BlockSpec(
                (1, coeff_count), lambda batch_idx, _coeff_idx: (batch_idx, 0)
            ),
            pl.BlockSpec((1,), lambda batch_idx, _coeff_idx: (batch_idx,)),
            pl.BlockSpec(src_index.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(valid.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(power.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(fact_index.shape, lambda _batch_idx, _coeff_idx: (0, 0)),
            pl.BlockSpec(sign.shape, lambda _batch_idx, _coeff_idx: (0,)),
            pl.BlockSpec(factorial.shape, lambda _batch_idx, _coeff_idx: (0,)),
        ],
        out_specs=pl.BlockSpec(
            (1, 1), lambda batch_idx, coeff_idx: (batch_idx, coeff_idx)
        ),
        interpret=bool(interpret),
        **pallas_backend_kwargs(str(backend), bool(interpret)),
        name=f"m2l_core_z_real_p{int(order)}",
    )
    return kernel(
        multipole_rot, radii, src_index, valid, power, fact_index, sign, factorial
    )


# ---------------------------------------------------------------------------
# Differentiable wrapper: fused Pallas forward + autodiff-of-twin reverse.
# ---------------------------------------------------------------------------
# ``pallas_call`` has no JVP/transpose, so the fused Pallas z-M2L is
# non-differentiable on its own. This ``custom_vjp`` supplies the missing
# reverse rule (mirrors the module-level pattern of
# ``jaccpot.nearfield.near_field._pair_accel_cvjp``): forward runs the Pallas
# kernel; reverse is autodiff through the pure-JAX recurrence twin
# ``vmap(translate_along_z_m2l_real)`` -- the SAME twin the parity test verifies
# the kernel against to ~1e-12. The z-M2L is linear in ``multipole_rot`` (so its
# multipole cotangent is the transpose) and non-linear in ``radii`` (enters as
# ``r^-(n+k+1)``); autodiff of the twin handles both exactly. ``order`` /
# ``interpret`` / ``backend`` are the hashable Pallas statics -> ``nondiff_argnums``
# (positional; the kernel itself takes them keyword-only, so this wrapper is the
# thin positional adapter).


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def m2l_core_z_real_pallas_cvjp(
    multipole_rot: Array,
    radii: Array,
    order: int,
    interpret: bool,
    backend: str,
) -> Array:
    """Differentiable fused-Pallas z-axis real M2L (see module comment above).

    Forward is byte-identical to :func:`m2l_core_z_real_pallas`; the reverse is
    autodiff of the pure-JAX twin, so the returned gradient is a faithful FMM
    gradient (the Pallas forward matches the twin to ~1e-12, not bit-exactly --
    standard for a Pallas ``custom_vjp``).

    Parameters
    ----------
    multipole_rot : Array
        Rotated multipole coefficients, shape ``(batch, sh_size(order))``. The
        z-M2L is *linear* in this argument, so its cotangent is the transpose.
    radii : Array
        Per-pair z translation distance, shape ``(batch,)``. Enters as
        ``r ** -(n + k + 1)``, i.e. non-linearly, which is why the reverse rule
        differentiates the twin instead of transposing.
    order : int
        Expansion order. ``nondiff_argnums`` -- a Pallas static.
    interpret : bool
        Whether to run the forward in interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering for the forward. ``nondiff_argnums``.

    Returns
    -------
    Array
        Translated coefficients, identical to :func:`m2l_core_z_real_pallas`.
    """
    return m2l_core_z_real_pallas(
        multipole_rot, radii, order=order, interpret=interpret, backend=backend
    )


def _m2l_core_z_real_pallas_cvjp_fwd(multipole_rot, radii, order, interpret, backend):
    out = m2l_core_z_real_pallas(
        multipole_rot, radii, order=order, interpret=interpret, backend=backend
    )
    return out, (multipole_rot, radii)


def _m2l_core_z_real_pallas_cvjp_bwd(order, interpret, backend, residual, cotangent):
    multipole_rot, radii = residual

    def _twin(mult, r):
        return jax.vmap(
            lambda m, rr: translate_along_z_m2l_real(m, rr, order=int(order))
        )(mult, r)

    _, vjp_fn = jax.vjp(_twin, multipole_rot, radii)
    return vjp_fn(cotangent)


m2l_core_z_real_pallas_cvjp.defvjp(
    _m2l_core_z_real_pallas_cvjp_fwd, _m2l_core_z_real_pallas_cvjp_bwd
)


__all__ = [
    "m2l_core_z_real_pallas",
    "m2l_core_z_real_pallas_cvjp",
    "pallas_m2l_real_supported",
]

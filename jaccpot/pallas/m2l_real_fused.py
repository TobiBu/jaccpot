"""Fully-fused real-basis M2L Pallas kernel (rotate -> z-translate -> rotate-back).

Real analog of :mod:`jaccpot.pallas.m2l_complex_fused`: fuses the ENTIRE real
rot-scale M2L (:func:`jaccpot.operators.m2l_real_rot_scale.m2l_rot_scale_real_batch`
= rotate-to-z, z-translate, rotate-back) into a single Pallas kernel per pair, so
the per-pair JAX rotation + z-core launches collapse. Real-valued (no imaginary
channel, unlike the complex kernel).

Triton-friendly like the complex kernel:
  * power-of-2 padded tile dims (Cp/Bp/mdp), padded lanes inert;
  * gather-free: pack/unpack are constant one-hot select matrices, the z-core is a
    dense operator ``Z = Zsign * Zfact * r**(-Zexp)``.

Convention: the REAL (Dehnen no-sqrt2) z-core, built from the single source of
truth :func:`jaccpot.operators.real_harmonics.z_m2l_translation_tables` (note the
factor of 2 on m != 0). Validated against ``m2l_rot_scale_real_batch``.

HARDWARE: real Pallas GPU execution needs Ampere (sm_80+); on other backends use
``interpret=True`` (correctness) or fall back to pure JAX.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.operators.real_harmonics import (
    sh_offset,
    sh_size,
    z_m2l_translation_tables,
)


def _fused_m2l_vjp_enabled() -> bool:
    """Whether the M2L custom_vjp reverse uses the fused Pallas VJP kernel.

    Default ON: the reverse runs as a single fused Pallas launch instead of pure-JAX
    autodiff of the twin, so the reverse pass also gets the fused speedup. Set
    ``JACCPOT_FUSED_M2L_VJP=0`` to fall back to autodiff of the pure-jnp twin (the
    correctness reference -- identical to round-off; useful for debugging).
    """
    return os.environ.get("JACCPOT_FUSED_M2L_VJP", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


__all__ = [
    "pallas_m2l_real_fused_supported",
    "m2l_real_fused_tables",
    "m2l_real_fused_jax",
    "m2l_real_fused_pallas",
    "m2l_real_fused_vjp_pallas",
    "m2l_real_fused_pallas_cvjp",
]


def pallas_m2l_real_fused_supported() -> bool:
    """True only on a GPU with compute capability >= 8.0 (Ampere+).

    Matches :func:`jaccpot.pallas.m2l_complex_fused.pallas_m2l_complex_fused_supported`:
    real Pallas GPU execution needs Ampere (sm_80+), so a plain ``gpu``/``tpu``
    backend check is not sufficient -- on a pre-Ampere GPU the Triton lowering
    fails at runtime. Callers must fall back to the pure-JAX rot-scale otherwise.
    """
    if pl is None:
        return False
    try:
        dev = jax.devices()[0]
    except Exception:  # pragma: no cover
        return False
    if dev.platform != "gpu":
        return False
    cc = getattr(dev, "compute_capability", None)
    if cc is None:
        return False
    try:
        major = int(str(cc).split(".")[0])
    except Exception:  # pragma: no cover
        return False
    return major >= 8


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=None)
def m2l_real_fused_tables(order: int) -> dict:
    """Pack/unpack one-hot matrices + dense real z-core operator, power-of-2 padded.

    Ppack[Bp*mdp, Cp], Uunpack[Cp, Bp*mdp] (sh layout, shared with the complex
    kernel); Zsign/Zfact/Zexp[Cp, Cp] the real (Dehnen) z-core. Padded lanes 0.
    """
    p = int(order)
    C = sh_size(p)
    md = 2 * p + 1
    Cp = _next_pow2(C)
    Bp = _next_pow2(p + 1)
    mdp = _next_pow2(md)

    # pack/unpack (identical sh layout to the complex kernel)
    Ppack = np.zeros((Bp * mdp, Cp), dtype=np.float64)
    Uunpack = np.zeros((Cp, Bp * mdp), dtype=np.float64)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            i = sh_offset(b) + c
            Ppack[b * mdp + c, i] = 1.0
            Uunpack[i, b * mdp + c] = 1.0

    # real (Dehnen no-sqrt2) z-core, from the single source of truth.
    src_index, valid, fact_index, r_exponent, sign = z_m2l_translation_tables(p)
    fact = np.asarray([math.factorial(i) for i in range(2 * p + 1)], dtype=np.float64)
    Zsign = np.zeros((Cp, Cp), dtype=np.float64)
    Zfact = np.zeros((Cp, Cp), dtype=np.float64)
    Zexp = np.zeros((Cp, Cp), dtype=np.float64)
    for out in range(C):
        for k in range(p + 1):
            if bool(valid[out, k]):
                s = int(src_index[out, k])
                Zsign[out, s] = float(sign[out])
                Zfact[out, s] = float(fact[int(fact_index[out, k])])
                Zexp[out, s] = float(r_exponent[out, k])

    return dict(
        p=p,
        C=C,
        md=md,
        Cp=Cp,
        Bp=Bp,
        mdp=mdp,
        Ppack=Ppack,
        Uunpack=Uunpack,
        Zsign=Zsign,
        Zfact=Zfact,
        Zexp=Zexp,
    )


def _tables_to_jnp(order: int, real_dtype: Any) -> dict[str, Array]:
    t = m2l_real_fused_tables(order)
    return dict(
        Ppack=jnp.asarray(t["Ppack"], dtype=real_dtype),
        Uunpack=jnp.asarray(t["Uunpack"], dtype=real_dtype),
        Zsign=jnp.asarray(t["Zsign"], dtype=real_dtype),
        Zfact=jnp.asarray(t["Zfact"], dtype=real_dtype),
        Zexp=jnp.asarray(t["Zexp"], dtype=real_dtype),
    )


def _matvec(mat: Array, vec: Array) -> Array:
    """out[i] = sum_j mat[i,j] * vec[j] (gather-free; Triton-GPU friendly)."""
    return jnp.sum(mat * vec[None, :], axis=1)


def _block_matmul_real(block: Array, vec: Array) -> Array:
    """out[b,i] = sum_j block[b,i,j] vec[b,j]  (real block-diagonal rotation)."""
    return jnp.sum(block * vec[:, None, :], axis=-1)


def _m2l_real_one(
    mult: Array, bto: Array, bfr: Array, r: Array, t: dict[str, Array]
) -> Array:
    """Full real M2L for one pair: rotate -> z-translate -> rotate-back."""
    Ppack = t["Ppack"]
    Uunpack = t["Uunpack"]
    Bp = bto.shape[0]
    mdp = bto.shape[1]
    # 1. pack [Cp] -> [Bp, mdp]
    pk = _matvec(Ppack, mult).reshape(Bp, mdp)
    # 2. rotate to z
    mr = _block_matmul_real(bto, pk)
    # 3. unpack -> [Cp]
    mrf = _matvec(Uunpack, mr.reshape(Bp * mdp))
    # 4. z-core (real Dehnen); guard r (padded pairs r=0 -> masked downstream)
    r_safe = jnp.where(r > 0.0, r, 1.0)
    Z = t["Zsign"] * t["Zfact"] * jnp.exp(-t["Zexp"] * jnp.log(r_safe))
    lz = _matvec(Z, mrf)
    # 5. pack [Cp] -> [Bp, mdp]
    pl_ = _matvec(Ppack, lz).reshape(Bp, mdp)
    # 6. rotate back
    orr = _block_matmul_real(bfr, pl_)
    # 7. unpack -> [Cp]
    return _matvec(Uunpack, orr.reshape(Bp * mdp))


def _matvec_T(mat: Array, vec: Array) -> Array:
    """out[j] = sum_i mat[i,j] * vec[i]  ==  (mat^T @ vec); the adjoint of _matvec.

    Same dense operand, opposite reduction axis -- no transpose array needed, which
    keeps the Triton lowering happy.
    """
    return jnp.sum(mat * vec[:, None], axis=0)


def _block_matmul_real_T(block: Array, vec: Array) -> Array:
    """out[b,j] = sum_i block[b,i,j] * vec[b,i]  -- adjoint of _block_matmul_real."""
    return jnp.sum(block * vec[:, :, None], axis=1)


def _m2l_real_one_vjp(
    mult: Array,
    bto: Array,
    bfr: Array,
    r: Array,
    out_bar: Array,
    t: dict[str, Array],
) -> tuple[Array, Array, Array, Array]:
    """Fused reverse of :func:`_m2l_real_one` for one pair.

    Returns cotangents ``(mult_bar, bto_bar, bfr_bar, r_bar)`` for the four
    differentiated inputs, computed ANALYTICALLY on-chip (no autodiff): recompute
    the forward intermediates, then walk the adjoint chain. The M2L is a product of
    linear maps ``out = Uunpack R_from Ppack Z(r) Uunpack R_to Ppack . mult`` so each
    step's reverse is either the operator adjoint (transpose, via ``_*_T``) or, for
    the two rotation-block inputs, the outer product of the stage cotangent with the
    stage input. ``r`` enters only through ``Z = Zsign*Zfact*r**-Zexp``, giving the
    analytic ``r_bar = -(1/r) * lz_bar^T (Z (.) Zexp) mrf``. This equals autodiff of
    :func:`_m2l_real_one` to round-off (verified in the VJP parity tests).
    """
    Ppack = t["Ppack"]
    Uunpack = t["Uunpack"]
    Bp = bto.shape[0]
    mdp = bto.shape[1]

    # --- recompute the forward intermediates needed by the reverse ---
    pk = _matvec(Ppack, mult).reshape(Bp, mdp)  # (1) pack
    mr = _block_matmul_real(bto, pk)  # (2) rotate-to-z
    mrf = _matvec(Uunpack, mr.reshape(Bp * mdp))  # (3) unpack
    r_safe = jnp.where(r > 0.0, r, 1.0)
    Z = t["Zsign"] * t["Zfact"] * jnp.exp(-t["Zexp"] * jnp.log(r_safe))
    lz = _matvec(Z, mrf)  # (4) z-core
    pl_ = _matvec(Ppack, lz).reshape(Bp, mdp)  # (5) pack

    # --- adjoint chain (out_bar -> input cotangents) ---
    orr_bar = _matvec_T(Uunpack, out_bar).reshape(Bp, mdp)  # adj (7) unpack
    pl_bar = _block_matmul_real_T(bfr, orr_bar)  # adj (6) rotate-back
    bfr_bar = orr_bar[:, :, None] * pl_[:, None, :]  # d out / d bfr
    lz_bar = _matvec_T(Ppack, pl_bar.reshape(Bp * mdp))  # adj (5) pack
    mrf_bar = _matvec_T(Z, lz_bar)  # adj (4) z-core, w.r.t. mrf
    # adj (4) z-core, w.r.t. r: dZ/dr = Z * (-Zexp / r); r_bar = sum_o lz_bar . (dZ mrf)
    dZ = Z * (-t["Zexp"] / r_safe)
    r_bar = jnp.where(r > 0.0, jnp.sum(lz_bar * _matvec(dZ, mrf)), 0.0)
    mr_bar = _matvec_T(Uunpack, mrf_bar).reshape(Bp, mdp)  # adj (3) unpack
    pk_bar = _block_matmul_real_T(bto, mr_bar)  # adj (2) rotate-to-z
    bto_bar = mr_bar[:, :, None] * pk[:, None, :]  # d out / d bto
    mult_bar = _matvec_T(Ppack, pk_bar.reshape(Bp * mdp))  # adj (1) pack
    return mult_bar, bto_bar, bfr_bar, r_bar


def _pair_pad_dims(order):
    t = m2l_real_fused_tables(order)
    return (t["C"], t["Cp"], t["p"] + 1, t["Bp"], t["md"], t["mdp"])


def _pad_pair_inputs(
    mult: Array, bto: Array, bfr: Array, dims: tuple[int, ...]
) -> tuple[Array, Array, Array]:
    C, Cp, B, Bp, md, mdp = dims
    mpad = ((0, 0), (0, Cp - C))
    bpad = ((0, 0), (0, Bp - B), (0, mdp - md), (0, mdp - md))
    return jnp.pad(mult, mpad), jnp.pad(bto, bpad), jnp.pad(bfr, bpad)


def m2l_real_fused_jax(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Pure-jnp reference for the fused real kernel (vmapped over pairs).

    multipoles: [N, C] real; blocks_*: [N, p+1, md, md] real; r: [N].
    """
    real_dtype = jnp.asarray(multipoles).dtype
    t = _tables_to_jnp(order, real_dtype)
    dims = _pair_pad_dims(order)
    C = dims[0]
    mult, bto, bfr = _pad_pair_inputs(
        jnp.asarray(multipoles),
        jnp.asarray(blocks_to_z),
        jnp.asarray(blocks_from_z),
        dims,
    )
    out = jax.vmap(lambda m, a, b, rr: _m2l_real_one(m, a, b, rr, t))(mult, bto, bfr, r)
    return out[:, :C]


def _m2l_real_fused_kernel(mult_ref, bto_ref, bfr_ref, r_ref, *table_and_out_refs):
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    (out_ref,) = table_and_out_refs[len(_TABLE_KEYS) :]
    tables = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    out_ref[0, :] = _m2l_real_one(mult_ref[0], bto_ref[0], bfr_ref[0], r_ref[0], tables)


_TABLE_KEYS = ("Ppack", "Uunpack", "Zsign", "Zfact", "Zexp")


def m2l_real_fused_pallas(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    *,
    order: int,
    interpret: bool = False,
    backend: str = "triton",
) -> Array:
    """Fully-fused real-basis M2L via a single Pallas kernel per pair.

    Same signature/semantics as :func:`m2l_real_fused_jax`. Triton backend
    (default) for the small gather-free per-pair tiles; ``interpret=True`` runs
    CPU semantics.
    """
    N = multipoles.shape[0]
    dims = _pair_pad_dims(order)
    C, Cp, _B, Bp, _md, mdp = dims
    real_dtype = jnp.asarray(multipoles).dtype
    tables = _tables_to_jnp(order, real_dtype)
    mult, btr, bfrr = _pad_pair_inputs(
        jnp.asarray(multipoles, dtype=real_dtype),
        jnp.asarray(blocks_to_z, dtype=real_dtype),
        jnp.asarray(blocks_from_z, dtype=real_dtype),
        dims,
    )
    rr = jnp.asarray(r, dtype=real_dtype)

    def bs_vec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks() -> pl.BlockSpec:
        return pl.BlockSpec((1, Bp, mdp, mdp), lambda i: (i, 0, 0, 0))

    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    out = pl.pallas_call(
        _m2l_real_fused_kernel,
        grid=(N,),
        in_specs=[
            bs_vec(Cp),  # mult [1, Cp]
            bs_blocks(),  # bto  [1, Bp, mdp, mdp]
            bs_blocks(),  # bfr
            pl.BlockSpec((1,), lambda i: (i,)),  # r [1]
            *[bs_full(a) for a in table_arrays],
        ],
        out_specs=bs_vec(Cp),
        out_shape=jax.ShapeDtypeStruct((N, Cp), real_dtype),
        interpret=interpret,
        backend=(None if interpret else backend),
        name=f"m2l_real_fused_p{int(order)}",
    )(mult, btr, bfrr, rr, *table_arrays)
    return out[:, :C]


def _m2l_real_fused_vjp_kernel(
    mult_ref, bto_ref, bfr_ref, r_ref, obar_ref, *table_and_out_refs
):
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    mult_bar_ref, bto_bar_ref, bfr_bar_ref, r_bar_ref = table_and_out_refs[
        len(_TABLE_KEYS) :
    ]
    tables = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    mb, btob, bfrb, rb = _m2l_real_one_vjp(
        mult_ref[0], bto_ref[0], bfr_ref[0], r_ref[0], obar_ref[0], tables
    )
    mult_bar_ref[0, :] = mb
    bto_bar_ref[0, :, :, :] = btob
    bfr_bar_ref[0, :, :, :] = bfrb
    r_bar_ref[0] = rb


def m2l_real_fused_vjp_pallas(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    out_bar: Array,
    *,
    order: int,
    interpret: bool = False,
    backend: str = "triton",
) -> tuple[Array, Array, Array, Array]:
    """Fully-fused reverse of :func:`m2l_real_fused_pallas` (one Pallas launch/pair).

    Given the primal inputs and the output cotangent ``out_bar`` (``[N, C]``), returns
    ``(mult_bar, bto_bar, bfr_bar, r_bar)`` with the shapes of the ORIGINAL (unpadded)
    inputs. Same analytic reverse as :func:`_m2l_real_one_vjp`, but on-chip so the
    reverse pass also runs as a single fused kernel instead of pure-JAX autodiff.
    """
    N = multipoles.shape[0]
    dims = _pair_pad_dims(order)
    C, Cp, B, Bp, md, mdp = dims
    real_dtype = jnp.asarray(multipoles).dtype
    tables = _tables_to_jnp(order, real_dtype)
    mult, btr, bfrr = _pad_pair_inputs(
        jnp.asarray(multipoles, dtype=real_dtype),
        jnp.asarray(blocks_to_z, dtype=real_dtype),
        jnp.asarray(blocks_from_z, dtype=real_dtype),
        dims,
    )
    rr = jnp.asarray(r, dtype=real_dtype)
    obar = jnp.pad(jnp.asarray(out_bar, dtype=real_dtype), ((0, 0), (0, Cp - C)))

    def bs_vec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks() -> pl.BlockSpec:
        return pl.BlockSpec((1, Bp, mdp, mdp), lambda i: (i, 0, 0, 0))

    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    mult_bar, bto_bar, bfr_bar, r_bar = pl.pallas_call(
        _m2l_real_fused_vjp_kernel,
        grid=(N,),
        in_specs=[
            bs_vec(Cp),  # mult
            bs_blocks(),  # bto
            bs_blocks(),  # bfr
            pl.BlockSpec((1,), lambda i: (i,)),  # r
            bs_vec(Cp),  # out_bar
            *[bs_full(a) for a in table_arrays],
        ],
        out_specs=[
            bs_vec(Cp),  # mult_bar
            bs_blocks(),  # bto_bar
            bs_blocks(),  # bfr_bar
            pl.BlockSpec((1,), lambda i: (i,)),  # r_bar
        ],
        out_shape=[
            jax.ShapeDtypeStruct((N, Cp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N,), real_dtype),
        ],
        interpret=interpret,
        backend=(None if interpret else backend),
        name=f"m2l_real_fused_vjp_p{int(order)}",
    )(mult, btr, bfrr, rr, obar, *table_arrays)
    # Unpad to the original input shapes.
    return (
        mult_bar[:, :C],
        bto_bar[:, :B, :md, :md],
        bfr_bar[:, :B, :md, :md],
        r_bar,
    )


# --------------------------------------------------------------------------
# Differentiable wrapper: fused Pallas forward + autodiff-of-twin reverse.
# --------------------------------------------------------------------------
# Real analogue of ``m2l_complex_fused_pallas_cvjp``. ``pallas_call`` has no
# JVP/transpose, so this ``custom_vjp`` runs the fused Pallas kernel forward and
# takes the reverse from autodiff of the pure-jnp twin ``m2l_real_fused_jax`` --
# the literal port the kernel is verified against, so the gradient is a faithful
# FMM gradient. Linear in ``multipoles``, b/tri-linear in the rotation ``blocks``,
# non-linear in ``r`` (``r**-(n+k+1)``); autodiff of the twin handles all four.
# ``order`` / ``interpret`` / ``backend`` are the hashable Pallas statics ->
# ``nondiff_argnums`` (positional adapter; the kernel takes them keyword-only).


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def m2l_real_fused_pallas_cvjp(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    order: int,
    interpret: bool,
    backend: str,
) -> Array:
    """Differentiable fully-fused real-basis M2L (see module comment above).

    Forward is byte-identical to :func:`m2l_real_fused_pallas`; the reverse is
    autodiff through :func:`m2l_real_fused_jax`.
    """
    return m2l_real_fused_pallas(
        multipoles,
        blocks_to_z,
        blocks_from_z,
        r,
        order=order,
        interpret=interpret,
        backend=backend,
    )


def _m2l_real_fused_pallas_cvjp_fwd(
    multipoles, blocks_to_z, blocks_from_z, r, order, interpret, backend
):
    out = m2l_real_fused_pallas(
        multipoles,
        blocks_to_z,
        blocks_from_z,
        r,
        order=order,
        interpret=interpret,
        backend=backend,
    )
    return out, (multipoles, blocks_to_z, blocks_from_z, r)


def _m2l_real_fused_pallas_cvjp_bwd(order, interpret, backend, residual, cotangent):
    multipoles, blocks_to_z, blocks_from_z, r = residual
    if _fused_m2l_vjp_enabled():
        # Fully-fused reverse: one Pallas launch computes all four cotangents.
        return m2l_real_fused_vjp_pallas(
            multipoles,
            blocks_to_z,
            blocks_from_z,
            r,
            cotangent,
            order=int(order),
            interpret=interpret,
            backend=backend,
        )

    # Fallback: autodiff of the pure-jnp twin (correctness reference).
    def _twin(mult, bto, bfr, rr):
        return m2l_real_fused_jax(mult, bto, bfr, rr, order=int(order))

    _, vjp_fn = jax.vjp(_twin, multipoles, blocks_to_z, blocks_from_z, r)
    return vjp_fn(cotangent)


m2l_real_fused_pallas_cvjp.defvjp(
    _m2l_real_fused_pallas_cvjp_fwd, _m2l_real_fused_pallas_cvjp_bwd
)

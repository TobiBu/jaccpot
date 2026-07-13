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

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from jaccpot.operators.real_harmonics import (
    sh_offset,
    sh_size,
    z_m2l_translation_tables,
)

__all__ = [
    "pallas_m2l_real_fused_supported",
    "m2l_real_fused_tables",
    "m2l_real_fused_jax",
    "m2l_real_fused_pallas",
]


def pallas_m2l_real_fused_supported() -> bool:
    """Return whether the active backend can run the fused real M2L Pallas kernel."""
    if pl is None:
        return False
    return jax.default_backend() in ("gpu", "tpu")


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

    return dict(p=p, C=C, md=md, Cp=Cp, Bp=Bp, mdp=mdp,
                Ppack=Ppack, Uunpack=Uunpack, Zsign=Zsign, Zfact=Zfact, Zexp=Zexp)


def _tables_to_jnp(order, real_dtype):
    t = m2l_real_fused_tables(order)
    return dict(
        Ppack=jnp.asarray(t["Ppack"], dtype=real_dtype),
        Uunpack=jnp.asarray(t["Uunpack"], dtype=real_dtype),
        Zsign=jnp.asarray(t["Zsign"], dtype=real_dtype),
        Zfact=jnp.asarray(t["Zfact"], dtype=real_dtype),
        Zexp=jnp.asarray(t["Zexp"], dtype=real_dtype),
    )


def _matvec(mat, vec):
    return jnp.sum(mat * vec[None, :], axis=1)


def _block_matmul_real(block, vec):
    """out[b,i] = sum_j block[b,i,j] vec[b,j]  (real block-diagonal rotation)."""
    return jnp.sum(block * vec[:, None, :], axis=-1)


def _m2l_real_one(mult, bto, bfr, r, t):
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


def _pair_pad_dims(order):
    t = m2l_real_fused_tables(order)
    return (t["C"], t["Cp"], t["p"] + 1, t["Bp"], t["md"], t["mdp"])


def _pad_pair_inputs(mult, bto, bfr, dims):
    C, Cp, B, Bp, md, mdp = dims
    mpad = ((0, 0), (0, Cp - C))
    bpad = ((0, 0), (0, Bp - B), (0, mdp - md), (0, mdp - md))
    return jnp.pad(mult, mpad), jnp.pad(bto, bpad), jnp.pad(bfr, bpad)


def m2l_real_fused_jax(multipoles, blocks_to_z, blocks_from_z, r, *, order):
    """Pure-jnp reference for the fused real kernel (vmapped over pairs).

    multipoles: [N, C] real; blocks_*: [N, p+1, md, md] real; r: [N].
    """
    real_dtype = jnp.asarray(multipoles).dtype
    t = _tables_to_jnp(order, real_dtype)
    dims = _pair_pad_dims(order)
    C = dims[0]
    mult, bto, bfr = _pad_pair_inputs(
        jnp.asarray(multipoles), jnp.asarray(blocks_to_z), jnp.asarray(blocks_from_z), dims
    )
    out = jax.vmap(lambda m, a, b, rr: _m2l_real_one(m, a, b, rr, t))(mult, bto, bfr, r)
    return out[:, :C]


def _m2l_real_fused_kernel(mult_ref, bto_ref, bfr_ref, r_ref, *table_and_out_refs):
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    (out_ref,) = table_and_out_refs[len(_TABLE_KEYS):]
    tables = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    out_ref[0, :] = _m2l_real_one(
        mult_ref[0], bto_ref[0], bfr_ref[0], r_ref[0], tables
    )


_TABLE_KEYS = ("Ppack", "Uunpack", "Zsign", "Zfact", "Zexp")


def m2l_real_fused_pallas(
    multipoles, blocks_to_z, blocks_from_z, r, *, order,
    interpret=False, backend="triton",
):
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

    def bs_vec(cols):
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks():
        return pl.BlockSpec((1, Bp, mdp, mdp), lambda i: (i, 0, 0, 0))

    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr):
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

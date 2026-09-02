"""Fused complex-basis M2L Pallas kernel (Phase 5 prototype, v0).

The dominant genuine-FLOP block of the FMM downward pass is the complex-basis
M2L (``operators/complex_ops.py:m2l_complex_reference_batch``), still a plain
``jax.vmap`` of rotate -> z-translate -> rotate-back. This module fuses the three
steps into a single Pallas kernel so the ``M_rot`` / ``local_z`` intermediates
stay on-chip instead of round-tripping HBM per pair.

Design (see ``docs/phase5_pallas_plan.md``):
  * consume the SAME precomputed rotation blocks as
    ``m2l_complex_reference_batch_cached_blocks`` (block-diagonal by ell),
  * carry complex as real/imag pairs (Pallas/Triton have no complex dtype),
  * grid over pairs (v0). v1/v2 (block-diagonal exploitation, class-major
    shared-memory rotation reuse + fused gather) are the A100 tuning steps.

HARDWARE: Pallas GPU needs Ampere (sm_80+). On the sm_75 dev box this can only
run via ``interpret=True`` (correctness). ``pallas_m2l_complex_fused_supported``
gates real GPU execution the same way the near-field kernel does.

The kernel arithmetic is first expressed as pure jnp in ``m2l_complex_fused_jax``
(validated against the reference) and then ported verbatim into the Pallas
kernel, so the two are trivially equivalent.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jaxtyping import Array

from jaccpot.operators.real_harmonics import sh_offset, sh_size
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas._flags import fused_m2l_vjp_enabled as _fused_m2l_vjp_enabled

__all__ = [
    "pallas_m2l_complex_fused_supported",
    "m2l_complex_fused_tables",
    "m2l_complex_fused_jax",
    "m2l_complex_fused_pallas",
    "m2l_complex_fused_vjp_pallas",
    "m2l_complex_fused_pallas_cvjp",
]


def pallas_m2l_complex_fused_supported() -> bool:
    """True only on a GPU with compute capability >= 8.0 (Ampere+).

    Returns
    -------
    bool
        True only for a GPU device reporting compute capability >= 8.0. Device
        discovery failures return False rather than raising, since this chooses a lane
        rather than validating one.
    """
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


# --------------------------------------------------------------------------
# Static index / coefficient tables (host-side, depend only on `order`).
# --------------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (>=1).

    Parameters
    ----------
    n : int
        Requested width; values below 1 clamp to 1.

    Returns
    -------
    int
        ``n`` rounded up to a power of two, which is what the Triton lowering requires
        of every operand shape.
    """
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


@functools.lru_cache(maxsize=None)
def m2l_complex_fused_tables(order: int) -> dict:
    """Precompute the pack/unpack + z-core tables for a given order.

    All array dims are padded up to the next power of 2 (Cp, Bp, mdp, Kp) so the
    Pallas *Triton* lowering (which requires power-of-2 operand shapes) accepts
    the per-pair tiles. Padding is inert: padded index entries point to 0 and the
    ``pack_valid`` / ``z_valid`` masks are 0 there, so padded lanes contribute
    nothing. Logical dims are C=(p+1)^2, md=2p+1, K=p+1 (valid extents inside the
    padded buffers). Padded returns:
      pack_flat[Bp, mdp], pack_valid[Bp, mdp]  -- flat<-(block,col) gather
      blk[Cp], col[Cp]                          -- (block,col) of each flat coeff
      z_src[Cp, Kp], z_valid/z_sign/z_fact/z_exp[Cp, Kp]

    Parameters
    ----------
    order : int
        Expansion order ``p``. Hashable, which is what makes the ``lru_cache`` sound.

    Returns
    -------
    dict
        The pack/unpack and z-core tables plus the shape scalars ``C``, ``Cp``, ``p``,
        ``Bp``, ``md``, ``mdp``. Every ``*p`` entry is the power-of-two padded width
        the Triton lowering requires.
    """
    p = int(order)
    C = sh_size(p)
    md = 2 * p + 1
    K = p + 1
    # power-of-2 padded extents for the Triton lowering
    Cp = _next_pow2(C)
    Bp = _next_pow2(p + 1)
    mdp = _next_pow2(md)
    Kp = _next_pow2(K)

    # factorials up to 2p
    fact = np.ones(2 * p + 1, dtype=np.float64)
    for i in range(1, 2 * p + 1):
        fact[i] = fact[i - 1] * i

    # pack: block b, col c (0..2b) <- flat sh_offset(b)+c
    pack_flat = np.zeros((Bp, mdp), dtype=np.int32)
    pack_valid = np.zeros((Bp, mdp), dtype=np.float64)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            pack_flat[b, c] = sh_offset(b) + c
            pack_valid[b, c] = 1.0

    # (block, col) of each flat coefficient  (for unpack)
    blk = np.zeros(Cp, dtype=np.int32)
    col = np.zeros(Cp, dtype=np.int32)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            i = sh_offset(b) + c
            blk[i] = b
            col[i] = c

    # z-core (flat): out (n,m) at o = sh_offset(n)+(m+n); sources k=|m|..p-n,
    # src (k,m) at sh_offset(k)+(m+k); coeff (-1)^m * fact[n+k] / r^(n+k+1).
    z_src = np.zeros((Cp, Kp), dtype=np.int32)
    z_valid = np.zeros((Cp, Kp), dtype=np.float64)
    z_sign = np.zeros((Cp, Kp), dtype=np.float64)
    z_fact = np.zeros((Cp, Kp), dtype=np.float64)
    z_exp = np.zeros((Cp, Kp), dtype=np.float64)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            o = sh_offset(n) + (m + n)
            j = 0
            for k in range(abs(m), p - n + 1):
                z_src[o, j] = sh_offset(k) + (m + k)
                z_valid[o, j] = 1.0
                z_sign[o, j] = (-1.0) ** m
                z_fact[o, j] = fact[n + k]
                z_exp[o, j] = float(n + k + 1)
                j += 1
    # padded z_exp lanes: keep exponent finite (r**-0 = 1), zeroed by z_valid
    z_exp[z_valid == 0.0] = 0.0

    # --- gather-free (Triton GPU) reformulation ------------------------------
    # Triton's Pallas lowering has no `gather`, so express the pack/unpack/z-core
    # index ops as constant-matrix (elementwise-multiply + reduce). Identical
    # arithmetic to the gather form; padded rows/cols are zero -> inert.
    #  pack   [Bp,mdp]<-[Cp] : Ppack[Bp*mdp, Cp] one-hot (folds in pack_valid)
    #  unpack [Cp]<-[Bp,mdp] : Uunpack[Cp, Bp*mdp] one-hot
    #  z-core [Cp]<-[Cp]     : Z = Zsign*Zfact*r**(-Zexp), dense [Cp,Cp]
    Ppack = np.zeros((Bp * mdp, Cp), dtype=np.float64)
    Uunpack = np.zeros((Cp, Bp * mdp), dtype=np.float64)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            i = sh_offset(b) + c
            Ppack[b * mdp + c, i] = 1.0
            Uunpack[i, b * mdp + c] = 1.0
    Zsign = np.zeros((Cp, Cp), dtype=np.float64)
    Zfact = np.zeros((Cp, Cp), dtype=np.float64)
    Zexp = np.zeros((Cp, Cp), dtype=np.float64)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            o = sh_offset(n) + (m + n)
            for k in range(abs(m), p - n + 1):
                s = sh_offset(k) + (m + k)
                Zsign[o, s] = (-1.0) ** m
                Zfact[o, s] = fact[n + k]
                Zexp[o, s] = float(n + k + 1)

    return dict(
        p=p,
        C=C,
        md=md,
        K=K,
        Cp=Cp,
        Bp=Bp,
        mdp=mdp,
        Kp=Kp,
        pack_flat=pack_flat,
        pack_valid=pack_valid,
        blk=blk,
        col=col,
        z_src=z_src,
        z_valid=z_valid,
        z_sign=z_sign,
        z_fact=z_fact,
        z_exp=z_exp,
        Ppack=Ppack,
        Uunpack=Uunpack,
        Zsign=Zsign,
        Zfact=Zfact,
        Zexp=Zexp,
    )


# --------------------------------------------------------------------------
# Pure-jnp reference of the exact kernel arithmetic (per batch, vectorised).
# This is validated against m2l_complex_reference_batch_cached_blocks and is
# the literal computation the Pallas kernel performs (per pair).
# --------------------------------------------------------------------------


def _block_matmul(
    block_r: Array, block_i: Array, vec_r: Array, vec_i: Array
) -> tuple[Array, Array]:
    """Complex block-diagonal matmul: out[b,i] = sum_j block[b,i,j] vec[b,j].

    Real and imaginary parts are carried as separate real arrays throughout this
    module, not as a complex dtype: the Triton lowering handles real arithmetic, and
    the complex boundary is applied only at the edges by the public wrappers.

    Parameters
    ----------
    block_r : Array
        Real part of the per-degree rotation blocks, shape ``(Bp, mdp, mdp)``.
    block_i : Array
        Imaginary part, same shape.
    vec_r : Array
        Real part of the packed coefficients, shape ``(Bp, mdp)``.
    vec_i : Array
        Imaginary part, same shape.

    Returns
    -------
    tuple[Array, Array]
        ``(out_r, out_i)``, each shape ``(Bp, mdp)``.
    """
    # block: [B, md, md], vec: [B, md]  ->  [B, md]
    out_r = jnp.sum(block_r * vec_r[:, None, :], axis=-1) - jnp.sum(
        block_i * vec_i[:, None, :], axis=-1
    )
    out_i = jnp.sum(block_r * vec_i[:, None, :], axis=-1) + jnp.sum(
        block_i * vec_r[:, None, :], axis=-1
    )
    return out_r, out_i


def _matvec(mat: Array, vec: Array) -> Array:
    """out[i] = sum_j mat[i,j] * vec[j]  (gather-free; Triton-GPU friendly).

    Parameters
    ----------
    mat : Array
        Dense operator, shape ``(rows, cols)``.
    vec : Array
        Vector, shape ``(cols,)``.

    Returns
    -------
    Array
        Shape ``(rows,)``.
    """
    return jnp.sum(mat * vec[None, :], axis=1)


def _m2l_one(
    mult_r: Array,
    mult_i: Array,
    bto_r: Array,
    bto_i: Array,
    bfrom_r: Array,
    bfrom_i: Array,
    r: Array,
    t: dict[str, Array],
) -> tuple[Array, Array]:
    """Full M2L for one pair in real/imag; `t` = tables dict of jnp arrays.

    Gather-free formulation (constant one-hot select matrices + a dense z-core
    operator) so the Pallas Triton GPU backend can lower it. Arithmetically
    identical to the index/gather form validated in interpret mode.

    Parameters
    ----------
    mult_r : Array
        Real part of the padded source multipoles, shape ``(Cp,)``.
    mult_i : Array
        Imaginary part, same shape.
    bto_r : Array
        Real part of the rotation-to-z blocks, shape ``(Bp, mdp, mdp)``.
    bto_i : Array
        Imaginary part, same shape.
    bfrom_r : Array
        Real part of the rotation-from-z blocks, same shape.
    bfrom_i : Array
        Imaginary part, same shape.
    r : Array
        Scalar centre separation for this pair.
    t : dict[str, Array]
        Tables from :func:`_tables_to_jnp`, at the working real dtype.

    Returns
    -------
    tuple[Array, Array]
        ``(out_r, out_i)``, the padded local coefficients, each shape ``(Cp,)``.
    """
    Ppack = t["Ppack"]  # [Bp*mdp, Cp]
    Uunpack = t["Uunpack"]  # [Cp, Bp*mdp]
    Zsign = t["Zsign"]
    Zfact = t["Zfact"]
    Zexp = t["Zexp"]  # [Cp, Cp]
    Bp = bto_r.shape[0]
    mdp = bto_r.shape[1]

    # 1. pack flat [Cp] -> [Bp, mdp]
    pk_r = _matvec(Ppack, mult_r).reshape(Bp, mdp)
    pk_i = _matvec(Ppack, mult_i).reshape(Bp, mdp)
    # 2. rotate to z (block-diagonal complex matmul)
    mr_r, mr_i = _block_matmul(bto_r, bto_i, pk_r, pk_i)
    # 3. unpack [Bp, mdp] -> flat [Cp]
    mrf_r = _matvec(Uunpack, mr_r.reshape(Bp * mdp))
    mrf_i = _matvec(Uunpack, mr_i.reshape(Bp * mdp))
    # 4. z-core: dense operator Z = sign * fact * r**(-exp) (padded lanes are 0).
    #    Guard r (padded pairs have r=0 -> masked downstream; keep finite here).
    r_safe = jnp.where(r > 0.0, r, 1.0)
    Z = Zsign * Zfact * jnp.exp(-Zexp * jnp.log(r_safe))
    lz_r = _matvec(Z, mrf_r)
    lz_i = _matvec(Z, mrf_i)
    # 5. pack [Cp] -> [Bp, mdp]
    pl_r = _matvec(Ppack, lz_r).reshape(Bp, mdp)
    pl_i = _matvec(Ppack, lz_i).reshape(Bp, mdp)
    # 6. rotate back
    or_r, or_i = _block_matmul(bfrom_r, bfrom_i, pl_r, pl_i)
    # 7. unpack [Bp, mdp] -> flat [Cp]
    return (
        _matvec(Uunpack, or_r.reshape(Bp * mdp)),
        _matvec(Uunpack, or_i.reshape(Bp * mdp)),
    )


def _matvec_T(mat: Array, vec: Array) -> Array:
    """out[j] = sum_i mat[i,j] * vec[i]  ==  (mat^T @ vec); the adjoint of _matvec.

    Parameters
    ----------
    mat : Array
        The SAME operand :func:`_matvec` takes, shape ``(rows, cols)`` -- reducing the
        other axis avoids materialising a transpose, which keeps Triton happy.
    vec : Array
        Cotangent, shape ``(rows,)``.

    Returns
    -------
    Array
        Shape ``(cols,)``.
    """
    return jnp.sum(mat * vec[:, None], axis=0)


def _block_matmul_vjp(
    block_r: Array,
    block_i: Array,
    vec_r: Array,
    vec_i: Array,
    or_bar: Array,
    oi_bar: Array,
) -> tuple[Array, Array, Array, Array]:
    """Reverse of :func:`_block_matmul` (complex block-diagonal matmul).

    Forward: ``o = block @ vec`` in complex (real/imag), i.e.
    ``o_r = Br vr - Bi vi``, ``o_i = Br vi + Bi vr`` (sum over j).

    Parameters
    ----------
    block_r : Array
        Real part of the rotation blocks, shape ``(Bp, mdp, mdp)``.
    block_i : Array
        Imaginary part, same shape.
    vec_r : Array
        Real part of the primal input, shape ``(Bp, mdp)``.
    vec_i : Array
        Imaginary part, same shape.
    or_bar : Array
        Cotangent of ``out_r``, shape ``(Bp, mdp)``.
    oi_bar : Array
        Cotangent of ``out_i``, same shape.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(vr_bar, vi_bar, br_bar, bi_bar)`` -- all standard real adjoints. The vector
        cotangents reduce over the output index; the block cotangents are outer
        products of the output cotangent with the primal input.
    """
    # vec cotangents: sum over output index i (axis 1).
    vr_bar = jnp.sum(block_r * or_bar[:, :, None], axis=1) + jnp.sum(
        block_i * oi_bar[:, :, None], axis=1
    )
    vi_bar = -jnp.sum(block_i * or_bar[:, :, None], axis=1) + jnp.sum(
        block_r * oi_bar[:, :, None], axis=1
    )
    # block cotangents: outer products (i,j) of output cotangent with vec.
    br_bar = (
        or_bar[:, :, None] * vec_r[:, None, :] + oi_bar[:, :, None] * vec_i[:, None, :]
    )
    bi_bar = (
        -or_bar[:, :, None] * vec_i[:, None, :] + oi_bar[:, :, None] * vec_r[:, None, :]
    )
    return vr_bar, vi_bar, br_bar, bi_bar


def _m2l_one_vjp(
    mult_r: Array,
    mult_i: Array,
    bto_r: Array,
    bto_i: Array,
    bfr_r: Array,
    bfr_i: Array,
    r: Array,
    out_r_bar: Array,
    out_i_bar: Array,
    t: dict[str, Array],
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Fused reverse of :func:`_m2l_one` in the real/imag representation, for one pair.

    Takes the cotangents of the REAL outputs ``(out_r, out_i)`` and returns the
    cotangents of the real/imag PARTS of every input:
    ``(mult_r_bar, mult_i_bar, bto_r_bar, bto_i_bar, bfr_r_bar, bfr_i_bar, r_bar)``.
    The complex<->(real,imag) boundary convention (``lax.complex``/``real``/``imag``
    transposes) is applied by the caller ``m2l_complex_fused_vjp_pallas``; this
    routine is a pure real-multilinear reverse, so it needs no complex convention.

    Parameters
    ----------
    mult_r : Array
        Real part of the primal padded multipoles, shape ``(Cp,)``.
    mult_i : Array
        Imaginary part, same shape.
    bto_r : Array
        Real part of the primal rotation-to-z blocks, shape ``(Bp, mdp, mdp)``.
    bto_i : Array
        Imaginary part, same shape.
    bfr_r : Array
        Real part of the primal rotation-from-z blocks, same shape.
    bfr_i : Array
        Imaginary part, same shape.
    r : Array
        Scalar primal centre separation.
    out_r_bar : Array
        Cotangent of the real output part, shape ``(Cp,)``.
    out_i_bar : Array
        Cotangent of the imaginary output part, same shape.
    t : dict[str, Array]
        Tables from :func:`_tables_to_jnp`.

    Returns
    -------
    tuple[Array, Array, Array, Array, Array, Array, Array]
        ``(mult_r_bar, mult_i_bar, bto_r_bar, bto_i_bar, bfr_r_bar, bfr_i_bar,
        r_bar)``.
    """
    Ppack = t["Ppack"]
    Uunpack = t["Uunpack"]
    Bp = bto_r.shape[0]
    mdp = bto_r.shape[1]

    # --- recompute forward intermediates ---
    pk_r = _matvec(Ppack, mult_r).reshape(Bp, mdp)
    pk_i = _matvec(Ppack, mult_i).reshape(Bp, mdp)
    mr_r, mr_i = _block_matmul(bto_r, bto_i, pk_r, pk_i)
    mrf_r = _matvec(Uunpack, mr_r.reshape(Bp * mdp))
    mrf_i = _matvec(Uunpack, mr_i.reshape(Bp * mdp))
    r_safe = jnp.where(r > 0.0, r, 1.0)
    Z = t["Zsign"] * t["Zfact"] * jnp.exp(-t["Zexp"] * jnp.log(r_safe))
    lz_r = _matvec(Z, mrf_r)
    lz_i = _matvec(Z, mrf_i)
    pl_r = _matvec(Ppack, lz_r).reshape(Bp, mdp)
    pl_i = _matvec(Ppack, lz_i).reshape(Bp, mdp)

    # --- adjoint chain ---
    or_r_bar = _matvec_T(Uunpack, out_r_bar).reshape(Bp, mdp)  # adj (7)
    or_i_bar = _matvec_T(Uunpack, out_i_bar).reshape(Bp, mdp)
    pl_r_bar, pl_i_bar, bfr_r_bar, bfr_i_bar = _block_matmul_vjp(  # adj (6)
        bfr_r, bfr_i, pl_r, pl_i, or_r_bar, or_i_bar
    )
    lz_r_bar = _matvec_T(Ppack, pl_r_bar.reshape(Bp * mdp))  # adj (5)
    lz_i_bar = _matvec_T(Ppack, pl_i_bar.reshape(Bp * mdp))
    mrf_r_bar = _matvec_T(Z, lz_r_bar)  # adj (4) w.r.t. mrf
    mrf_i_bar = _matvec_T(Z, lz_i_bar)
    dZ = Z * (-t["Zexp"] / r_safe)  # adj (4) w.r.t. r
    r_bar = jnp.where(
        r > 0.0,
        jnp.sum(lz_r_bar * _matvec(dZ, mrf_r)) + jnp.sum(lz_i_bar * _matvec(dZ, mrf_i)),
        0.0,
    )
    mr_r_bar = _matvec_T(Uunpack, mrf_r_bar).reshape(Bp, mdp)  # adj (3)
    mr_i_bar = _matvec_T(Uunpack, mrf_i_bar).reshape(Bp, mdp)
    pk_r_bar, pk_i_bar, bto_r_bar, bto_i_bar = _block_matmul_vjp(  # adj (2)
        bto_r, bto_i, pk_r, pk_i, mr_r_bar, mr_i_bar
    )
    mult_r_bar = _matvec_T(Ppack, pk_r_bar.reshape(Bp * mdp))  # adj (1)
    mult_i_bar = _matvec_T(Ppack, pk_i_bar.reshape(Bp * mdp))
    return (
        mult_r_bar,
        mult_i_bar,
        bto_r_bar,
        bto_i_bar,
        bfr_r_bar,
        bfr_i_bar,
        r_bar,
    )


def _tables_to_jnp(order, real_dtype):
    t = m2l_complex_fused_tables(order)
    return dict(
        Ppack=jnp.asarray(t["Ppack"], dtype=real_dtype),
        Uunpack=jnp.asarray(t["Uunpack"], dtype=real_dtype),
        Zsign=jnp.asarray(t["Zsign"], dtype=real_dtype),
        Zfact=jnp.asarray(t["Zfact"], dtype=real_dtype),
        Zexp=jnp.asarray(t["Zexp"], dtype=real_dtype),
    )


def _pad_pair_inputs(
    mr: Array,
    mi: Array,
    bto_r: Array,
    bto_i: Array,
    bfr_r: Array,
    bfr_i: Array,
    dims: tuple[int, ...],
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Zero-pad multipoles and rotation blocks to the tables' power-of-2 extents.

    ``[N, C] -> [N, Cp]`` and ``[N, p+1, md, md] -> [N, Bp, mdp, mdp]``. The padding
    is inert: the pack/unpack one-hot matrices and the z-core have exact zeros in the
    padded lanes, so padded entries contribute nothing to any reduction.

    Parameters
    ----------
    mr : Array
        Real part of the multipoles, shape ``(N, C)``.
    mi : Array
        Imaginary part, same shape.
    bto_r : Array
        Real part of the rotation-to-z blocks, shape ``(N, p+1, md, md)``.
    bto_i : Array
        Imaginary part, same shape.
    bfr_r : Array
        Real part of the rotation-from-z blocks, same shape.
    bfr_i : Array
        Imaginary part, same shape.
    dims : tuple[int, ...]
        ``(C, Cp, B, Bp, md, mdp)`` from :func:`_pair_pad_dims`.

    Returns
    -------
    tuple[Array, Array, Array, Array, Array, Array]
        The six inputs, padded, in the same order.
    """
    C, Cp, B, Bp, md, mdp = dims
    mpad = ((0, 0), (0, Cp - C))
    bpad = ((0, 0), (0, Bp - B), (0, mdp - md), (0, mdp - md))
    return (
        jnp.pad(mr, mpad),
        jnp.pad(mi, mpad),
        jnp.pad(bto_r, bpad),
        jnp.pad(bto_i, bpad),
        jnp.pad(bfr_r, bpad),
        jnp.pad(bfr_i, bpad),
    )


def _pair_pad_dims(order):
    t = m2l_complex_fused_tables(order)
    return (t["C"], t["Cp"], t["p"] + 1, t["Bp"], t["md"], t["mdp"])


def m2l_complex_fused_jax(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    *,
    order: int,
) -> Array:
    """Pure-jnp reference for the fused kernel (vmapped over pairs).

    Parameters
    ----------
    multipoles : Array
        Source multipole coefficients, shape ``(N, C)``, COMPLEX.
    blocks_to_z : Array
        Complex rotation blocks aligning each pair axis to +z, shape
        ``(N, p+1, md, md)``.
    blocks_from_z : Array
        Complex rotation blocks back from +z, same shape.
    r : Array
        Per-pair centre separation, shape ``(N,)``.
    order : int
        Expansion order, fixing the tables and padded widths.

    Returns
    -------
    Array
        Complex local contributions, shape ``(N, C)``.

    Notes
    -----
    The correctness reference for the Pallas forward, and what the ``custom_vjp``
    reverse differentiates when ``JACCPOT_FUSED_M2L_VJP=0`` -- so it must remain a
    transcription of the same operator rather than an independent implementation.
    """
    real_dtype = jnp.asarray(multipoles).real.dtype
    t = _tables_to_jnp(order, real_dtype)
    dims = _pair_pad_dims(order)
    C = dims[0]
    mr, mi, bto_r, bto_i, bfr_r, bfr_i = _pad_pair_inputs(
        jnp.real(multipoles),
        jnp.imag(multipoles),
        jnp.real(blocks_to_z),
        jnp.imag(blocks_to_z),
        jnp.real(blocks_from_z),
        jnp.imag(blocks_from_z),
        dims,
    )

    def one(
        mr: Array,
        mi: Array,
        btr: Array,
        bti: Array,
        bfr: Array,
        bfi: Array,
        rr: Array,
    ) -> tuple[Array, Array]:
        return _m2l_one(mr, mi, btr, bti, bfr, bfi, rr, t)

    or_r, or_i = jax.vmap(one)(mr, mi, bto_r, bto_i, bfr_r, bfr_i, r)
    return jax.lax.complex(or_r[:, :C], or_i[:, :C])


# --------------------------------------------------------------------------
# Pallas kernel: one program per pair. Same arithmetic as _m2l_one.
# --------------------------------------------------------------------------


_TABLE_KEYS = (
    "Ppack",
    "Uunpack",
    "Zsign",
    "Zfact",
    "Zexp",
)


def _m2l_complex_fused_kernel(
    mult_r_ref: KernelRef,
    mult_i_ref: KernelRef,
    bto_r_ref: KernelRef,
    bto_i_ref: KernelRef,
    bfr_r_ref: KernelRef,
    bfr_i_ref: KernelRef,
    r_ref: KernelRef,
    *table_and_out_refs: KernelRef,
) -> None:
    """Forward kernel: one program instance per pair, one fused complex M2L.

    Real and imaginary parts arrive as SEPARATE real refs, not a complex block: the
    Triton lowering does real arithmetic, and the complex boundary is applied by the
    wrapper.

    Parameters
    ----------
    mult_r_ref : KernelRef
        Real part of the padded source multipoles for this pair, shape ``(1, Cp)``.
    mult_i_ref : KernelRef
        Imaginary part, same shape.
    bto_r_ref : KernelRef
        Real part of the rotation-to-z blocks, shape ``(1, Bp, mdp, mdp)``.
    bto_i_ref : KernelRef
        Imaginary part, same shape.
    bfr_r_ref : KernelRef
        Real part of the rotation-from-z blocks, same shape.
    bfr_i_ref : KernelRef
        Imaginary part, same shape.
    r_ref : KernelRef
        Centre separation for this pair, shape ``(1,)``.
    *table_and_out_refs : KernelRef
        The ``_TABLE_KEYS`` operator tables followed by the TWO output refs (real and
        imaginary). Variadic because ``pallas_call`` passes inputs then outputs
        positionally; the tables are replicated inputs because Pallas forbids
        capturing them as constants.

    Returns
    -------
    None
        The results are the writes to the two output refs.
    """
    # last two refs are outputs; the rest (len(_TABLE_KEYS)) are the tables,
    # passed as replicated inputs (Pallas forbids capturing them as constants).
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    out_r_ref, out_i_ref = table_and_out_refs[len(_TABLE_KEYS) :]
    tables = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    or_r, or_i = _m2l_one(
        mult_r_ref[0],
        mult_i_ref[0],
        bto_r_ref[0],
        bto_i_ref[0],
        bfr_r_ref[0],
        bfr_i_ref[0],
        r_ref[0],
        tables,
    )
    out_r_ref[0, :] = or_r
    out_i_ref[0, :] = or_i


def m2l_complex_fused_pallas(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    *,
    order: int,
    interpret: bool = False,
    backend: str = "triton",
) -> Array:
    """Fused complex-basis M2L via a Pallas kernel (one program per pair).

    Same signature/semantics as ``m2l_complex_fused_jax``. Requires an Ampere+
    GPU unless ``interpret=True``.

    ``backend`` selects the Pallas GPU lowering. The default ``"triton"`` is
    required for this kernel: the Mosaic-GPU backend rejects it (no fp64 TMA, and
    its TMA copies must be a multiple of the 128-byte warpgroup size, whereas the
    per-pair blocks here are (p+1)^2 elements = 72/200 bytes). Triton handles the
    small, gather-heavy per-pair tiles and fp64. ``interpret=True`` ignores the
    backend and runs CPU semantics.

    Parameters
    ----------
    multipoles : Array
        Complex source multipole coefficients, shape ``(N, C)``.
    blocks_to_z : Array
        Complex rotation blocks aligning each pair axis to +z, shape
        ``(N, p+1, md, md)``.
    blocks_from_z : Array
        Complex rotation blocks back from +z, same shape.
    r : Array
        Per-pair centre separation, shape ``(N,)``.
    order : int
        Expansion order. Static: it fixes the tables and the padded tile widths.
    interpret : bool
        Run under Pallas interpret mode (CPU semantics, no lowering).
    backend : str
        Pallas GPU lowering, ``"triton"`` by default.

    Returns
    -------
    Array
        Complex local contributions, shape ``(N, C)`` -- sliced back from the padded
        ``Cp`` width, so callers never see the padding.
    """
    p = int(order)
    N = multipoles.shape[0]
    dims = _pair_pad_dims(order)  # (C, Cp, B, Bp, md, mdp)
    C, Cp, _B, Bp, _md, mdp = dims
    real_dtype = jnp.asarray(multipoles).real.dtype
    tables = _tables_to_jnp(p, real_dtype)

    mr, mi, btr, bti, bfr, bfi = _pad_pair_inputs(
        jnp.real(multipoles).astype(real_dtype),
        jnp.imag(multipoles).astype(real_dtype),
        jnp.real(blocks_to_z).astype(real_dtype),
        jnp.imag(blocks_to_z).astype(real_dtype),
        jnp.real(blocks_from_z).astype(real_dtype),
        jnp.imag(blocks_from_z).astype(real_dtype),
        dims,
    )
    rr = jnp.asarray(r, dtype=real_dtype)

    def bs_vec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks() -> pl.BlockSpec:
        return pl.BlockSpec((1, Bp, mdp, mdp), lambda i: (i, 0, 0, 0))

    # tables are identical across pairs: replicate them to every program via a
    # full-array BlockSpec whose index_map ignores the pair index.
    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    table_specs = [bs_full(a) for a in table_arrays]

    out_r, out_i = pl.pallas_call(
        _m2l_complex_fused_kernel,
        grid=(N,),
        in_specs=[
            bs_vec(Cp),
            bs_vec(Cp),  # mult r/i  [1, Cp]
            bs_blocks(),
            bs_blocks(),  # bto r/i   [1, Bp, mdp, mdp]
            bs_blocks(),
            bs_blocks(),  # bfrom r/i
            pl.BlockSpec((1,), lambda i: (i,)),  # r [1]
            *table_specs,
        ],
        out_specs=[bs_vec(Cp), bs_vec(Cp)],
        out_shape=[
            jax.ShapeDtypeStruct((N, Cp), real_dtype),
            jax.ShapeDtypeStruct((N, Cp), real_dtype),
        ],
        interpret=interpret,
        **pallas_backend_kwargs(backend, interpret),
        name=f"m2l_complex_fused_p{p}",
    )(mr, mi, btr, bti, bfr, bfi, rr, *table_arrays)

    return jax.lax.complex(out_r[:, :C], out_i[:, :C])


def _m2l_complex_fused_vjp_kernel(
    mult_r_ref: KernelRef,
    mult_i_ref: KernelRef,
    bto_r_ref: KernelRef,
    bto_i_ref: KernelRef,
    bfr_r_ref: KernelRef,
    bfr_i_ref: KernelRef,
    r_ref: KernelRef,
    obar_r_ref: KernelRef,
    obar_i_ref: KernelRef,
    *table_and_out_refs: KernelRef,
) -> None:
    """Reverse kernel: the analytic complex-M2L VJP for one pair, on-chip.

    Parameters
    ----------
    mult_r_ref : KernelRef
        Real part of the primal padded multipoles, shape ``(1, Cp)``.
    mult_i_ref : KernelRef
        Imaginary part, same shape.
    bto_r_ref : KernelRef
        Real part of the primal rotation-to-z blocks, shape ``(1, Bp, mdp, mdp)``.
    bto_i_ref : KernelRef
        Imaginary part, same shape.
    bfr_r_ref : KernelRef
        Real part of the primal rotation-from-z blocks, same shape.
    bfr_i_ref : KernelRef
        Imaginary part, same shape.
    r_ref : KernelRef
        Primal centre separation, shape ``(1,)``.
    obar_r_ref : KernelRef
        Cotangent of the real output part, shape ``(1, Cp)``.
    obar_i_ref : KernelRef
        Cotangent of the imaginary output part, same shape.
    *table_and_out_refs : KernelRef
        The ``_TABLE_KEYS`` tables followed by the SEVEN cotangent output refs, in the
        order :func:`_m2l_one_vjp` returns them. Same split-by-``len(_TABLE_KEYS)``
        convention as the forward kernel.

    Returns
    -------
    None
        The results are the writes to the cotangent refs.
    """
    table_refs = table_and_out_refs[: len(_TABLE_KEYS)]
    (
        mr_bar_ref,
        mi_bar_ref,
        btor_bar_ref,
        btoi_bar_ref,
        bfrr_bar_ref,
        bfri_bar_ref,
        r_bar_ref,
    ) = table_and_out_refs[len(_TABLE_KEYS) :]
    tables = {k: ref[...] for k, ref in zip(_TABLE_KEYS, table_refs)}
    mrb, mib, btorb, btoib, bfrrb, bfrib, rb = _m2l_one_vjp(
        mult_r_ref[0],
        mult_i_ref[0],
        bto_r_ref[0],
        bto_i_ref[0],
        bfr_r_ref[0],
        bfr_i_ref[0],
        r_ref[0],
        obar_r_ref[0],
        obar_i_ref[0],
        tables,
    )
    mr_bar_ref[0, :] = mrb
    mi_bar_ref[0, :] = mib
    btor_bar_ref[0, :, :, :] = btorb
    btoi_bar_ref[0, :, :, :] = btoib
    bfrr_bar_ref[0, :, :, :] = bfrrb
    bfri_bar_ref[0, :, :, :] = bfrib
    r_bar_ref[0] = rb


def m2l_complex_fused_vjp_pallas(
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
    """Fully-fused reverse of :func:`m2l_complex_fused_pallas` (one launch/pair).

    Returns ``(mult_bar, bto_bar, bfr_bar, r_bar)`` matching the ORIGINAL (unpadded)
    complex input shapes, computed on-chip. The complex<->(real,imag) VJP boundary is
    applied here around the real-multilinear kernel: JAX's ``lax.complex``/``imag``
    transposes use the conjugate convention, so we feed ``out_i_bar = -imag(out_bar)``
    and recombine each complex input cotangent as ``complex(re_bar, -im_bar)`` (this
    reproduces ``jax.vjp`` of the twin exactly -- verified in the parity tests).

    Parameters
    ----------
    multipoles : Array
        Complex source multipole coefficients, shape ``(N, C)``.
    blocks_to_z : Array
        Complex rotation blocks aligning each pair axis to +z, shape
        ``(N, p+1, md, md)``.
    blocks_from_z : Array
        Complex rotation blocks back from +z, same shape.
    r : Array
        Per-pair centre separation, shape ``(N,)``.
    out_bar : Array
        Cotangent of the forward output, shape ``(N, C)``, complex.
    order : int
        Expansion order. Static: it fixes the tables and the padded tile widths.
    interpret : bool
        Run under Pallas interpret mode (CPU semantics, no lowering).
    backend : str
        Pallas GPU lowering, ``"triton"`` by default.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(mult_bar, bto_bar, bfr_bar, r_bar)`` at the ORIGINAL unpadded complex
        input shapes.
    """
    p = int(order)
    N = multipoles.shape[0]
    dims = _pair_pad_dims(order)  # (C, Cp, B, Bp, md, mdp)
    C, Cp, B, Bp, md, mdp = dims
    real_dtype = jnp.asarray(multipoles).real.dtype
    tables = _tables_to_jnp(p, real_dtype)

    mr, mi, btr, bti, bfr, bfi = _pad_pair_inputs(
        jnp.real(multipoles).astype(real_dtype),
        jnp.imag(multipoles).astype(real_dtype),
        jnp.real(blocks_to_z).astype(real_dtype),
        jnp.imag(blocks_to_z).astype(real_dtype),
        jnp.real(blocks_from_z).astype(real_dtype),
        jnp.imag(blocks_from_z).astype(real_dtype),
        dims,
    )
    rr = jnp.asarray(r, dtype=real_dtype)
    obar = jnp.asarray(out_bar)
    # complex boundary: out_r_bar = real(out_bar); out_i_bar = -imag(out_bar).
    obar_r = jnp.pad(jnp.real(obar).astype(real_dtype), ((0, 0), (0, Cp - C)))
    obar_i = jnp.pad((-jnp.imag(obar)).astype(real_dtype), ((0, 0), (0, Cp - C)))

    def bs_vec(cols: int) -> pl.BlockSpec:
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks() -> pl.BlockSpec:
        return pl.BlockSpec((1, Bp, mdp, mdp), lambda i: (i, 0, 0, 0))

    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr: Array) -> pl.BlockSpec:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    outs = pl.pallas_call(
        _m2l_complex_fused_vjp_kernel,
        grid=(N,),
        in_specs=[
            bs_vec(Cp),
            bs_vec(Cp),  # mult r/i
            bs_blocks(),
            bs_blocks(),  # bto r/i
            bs_blocks(),
            bs_blocks(),  # bfr r/i
            pl.BlockSpec((1,), lambda i: (i,)),  # r
            bs_vec(Cp),
            bs_vec(Cp),  # out_bar r/i
            *[bs_full(a) for a in table_arrays],
        ],
        out_specs=[
            bs_vec(Cp),
            bs_vec(Cp),  # mult_bar r/i
            bs_blocks(),
            bs_blocks(),  # bto_bar r/i
            bs_blocks(),
            bs_blocks(),  # bfr_bar r/i
            pl.BlockSpec((1,), lambda i: (i,)),  # r_bar
        ],
        out_shape=[
            jax.ShapeDtypeStruct((N, Cp), real_dtype),
            jax.ShapeDtypeStruct((N, Cp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N, Bp, mdp, mdp), real_dtype),
            jax.ShapeDtypeStruct((N,), real_dtype),
        ],
        interpret=interpret,
        **pallas_backend_kwargs(backend, interpret),
        name=f"m2l_complex_fused_vjp_p{p}",
    )(mr, mi, btr, bti, bfr, bfi, rr, obar_r, obar_i, *table_arrays)

    mrb, mib, btorb, btoib, bfrrb, bfrib, rb = outs
    # Unpad and recombine with the conjugate convention: complex(re_bar, -im_bar).
    mult_bar = jax.lax.complex(mrb[:, :C], -mib[:, :C])
    bto_bar = jax.lax.complex(btorb[:, :B, :md, :md], -btoib[:, :B, :md, :md])
    bfr_bar = jax.lax.complex(bfrrb[:, :B, :md, :md], -bfrib[:, :B, :md, :md])
    return mult_bar, bto_bar, bfr_bar, rb


# --------------------------------------------------------------------------
# Differentiable wrapper: fused Pallas forward + autodiff-of-twin reverse.
# --------------------------------------------------------------------------
# ``pallas_call`` has no JVP/transpose, so the fused complex M2L is
# non-differentiable on its own. This ``custom_vjp`` (mirroring the module-level
# pattern of ``jaccpot.nearfield.near_field._pair_accel_cvjp``) runs the Pallas
# kernel forward and takes the reverse from autodiff of the pure-jnp twin
# ``m2l_complex_fused_jax`` -- the literal port the Pallas kernel is verified
# against to ~1e-10, so the gradient is a faithful FMM gradient (fwd Pallas ==
# twin to the port tolerance, not bit-exactly). The M2L is linear in
# ``multipoles`` and b/tri-linear in the rotation ``blocks`` and non-linear in
# ``r`` (enters as ``r**-(n+k+1)``); autodiff of the twin handles all four
# exactly. ``order`` / ``interpret`` / ``backend`` are the hashable Pallas
# statics -> ``nondiff_argnums`` (positional; the kernel takes them keyword-only,
# so this wrapper is the thin positional adapter).


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6))
def m2l_complex_fused_pallas_cvjp(
    multipoles: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    r: Array,
    order: int,
    interpret: bool,
    backend: str,
) -> Array:
    """Differentiable fused complex-basis M2L (see module comment above).

    Forward is byte-identical to :func:`m2l_complex_fused_pallas`; the reverse is
    autodiff through :func:`m2l_complex_fused_jax`.

    Parameters
    ----------
    multipoles : Array
        Complex source multipole coefficients, shape ``(N, C)``.
    blocks_to_z : Array
        Complex rotation blocks aligning each pair axis to +z, shape
        ``(N, p+1, md, md)``.
    blocks_from_z : Array
        Complex rotation blocks back from +z, same shape.
    r : Array
        Per-pair centre separation, shape ``(N,)``.
    order : int
        Expansion order. ``nondiff_argnums`` -- a Pallas static, hence positional
        here even though the wrapped function takes it keyword-only.
    interpret : bool
        Interpret mode for the forward. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering for the forward. ``nondiff_argnums``.

    Returns
    -------
    Array
        Complex local contributions, shape ``(N, C)``.
    """
    return m2l_complex_fused_pallas(
        multipoles,
        blocks_to_z,
        blocks_from_z,
        r,
        order=order,
        interpret=interpret,
        backend=backend,
    )


def _m2l_complex_fused_pallas_cvjp_fwd(
    multipoles, blocks_to_z, blocks_from_z, r, order, interpret, backend
):
    out = m2l_complex_fused_pallas(
        multipoles,
        blocks_to_z,
        blocks_from_z,
        r,
        order=order,
        interpret=interpret,
        backend=backend,
    )
    return out, (multipoles, blocks_to_z, blocks_from_z, r)


def _m2l_complex_fused_pallas_cvjp_bwd(order, interpret, backend, residual, cotangent):
    multipoles, blocks_to_z, blocks_from_z, r = residual
    if _fused_m2l_vjp_enabled():
        # Fully-fused reverse: one Pallas launch computes all four cotangents.
        return m2l_complex_fused_vjp_pallas(
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
        return m2l_complex_fused_jax(mult, bto, bfr, rr, order=int(order))

    _, vjp_fn = jax.vjp(_twin, multipoles, blocks_to_z, blocks_from_z, r)
    return vjp_fn(cotangent)


m2l_complex_fused_pallas_cvjp.defvjp(
    _m2l_complex_fused_pallas_cvjp_fwd, _m2l_complex_fused_pallas_cvjp_bwd
)

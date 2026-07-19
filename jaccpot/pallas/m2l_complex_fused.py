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

from jaccpot.operators.real_harmonics import sh_offset, sh_size

__all__ = [
    "pallas_m2l_complex_fused_supported",
    "m2l_complex_fused_tables",
    "m2l_complex_fused_jax",
    "m2l_complex_fused_pallas",
]


def pallas_m2l_complex_fused_supported() -> bool:
    """True only on a GPU with compute capability >= 8.0 (Ampere+)."""
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
    """Smallest power of 2 >= n (>=1)."""
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


def _block_matmul(block_r, block_i, vec_r, vec_i):
    """Complex block-diagonal matmul: out[b,i] = sum_j block[b,i,j] vec[b,j]."""
    # block: [B, md, md], vec: [B, md]  ->  [B, md]
    out_r = jnp.sum(block_r * vec_r[:, None, :], axis=-1) - jnp.sum(
        block_i * vec_i[:, None, :], axis=-1
    )
    out_i = jnp.sum(block_r * vec_i[:, None, :], axis=-1) + jnp.sum(
        block_i * vec_r[:, None, :], axis=-1
    )
    return out_r, out_i


def _matvec(mat, vec):
    """out[i] = sum_j mat[i,j] * vec[j]  (gather-free; Triton-GPU friendly)."""
    return jnp.sum(mat * vec[None, :], axis=1)


def _m2l_one(mult_r, mult_i, bto_r, bto_i, bfrom_r, bfrom_i, r, t):
    """Full M2L for one pair in real/imag; `t` = tables dict of jnp arrays.

    Gather-free formulation (constant one-hot select matrices + a dense z-core
    operator) so the Pallas Triton GPU backend can lower it. Arithmetically
    identical to the index/gather form validated in interpret mode.
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


def _tables_to_jnp(order, real_dtype):
    t = m2l_complex_fused_tables(order)
    return dict(
        Ppack=jnp.asarray(t["Ppack"], dtype=real_dtype),
        Uunpack=jnp.asarray(t["Uunpack"], dtype=real_dtype),
        Zsign=jnp.asarray(t["Zsign"], dtype=real_dtype),
        Zfact=jnp.asarray(t["Zfact"], dtype=real_dtype),
        Zexp=jnp.asarray(t["Zexp"], dtype=real_dtype),
    )


def _pad_pair_inputs(mr, mi, bto_r, bto_i, bfr_r, bfr_i, dims):
    """Zero-pad multipoles [N,C]->[N,Cp] and blocks [N,p+1,md,md]->[N,Bp,mdp,mdp]
    to the power-of-2 extents the tables use (inert; masked by pack/z valid)."""
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
    multipoles: jax.Array,
    blocks_to_z: jax.Array,
    blocks_from_z: jax.Array,
    r: jax.Array,
    *,
    order: int,
) -> jax.Array:
    """Pure-jnp reference for the fused kernel (vmapped over pairs).

    multipoles: [N, C] complex; blocks_*: [N, p+1, md, md] complex; r: [N].
    Returns [N, C] complex local contributions.
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
        mr: jax.Array,
        mi: jax.Array,
        btr: jax.Array,
        bti: jax.Array,
        bfr: jax.Array,
        bfi: jax.Array,
        rr: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
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
    mult_r_ref,
    mult_i_ref,
    bto_r_ref,
    bto_i_ref,
    bfr_r_ref,
    bfr_i_ref,
    r_ref,
    *table_and_out_refs,
):
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
    multipoles: jax.Array,
    blocks_to_z: jax.Array,
    blocks_from_z: jax.Array,
    r: jax.Array,
    *,
    order: int,
    interpret: bool = False,
    backend: str = "triton",
) -> jax.Array:
    """Fused complex-basis M2L via a Pallas kernel (one program per pair).

    Same signature/semantics as ``m2l_complex_fused_jax``. Requires an Ampere+
    GPU unless ``interpret=True``.

    ``backend`` selects the Pallas GPU lowering. The default ``"triton"`` is
    required for this kernel: the Mosaic-GPU backend rejects it (no fp64 TMA, and
    its TMA copies must be a multiple of the 128-byte warpgroup size, whereas the
    per-pair blocks here are (p+1)^2 elements = 72/200 bytes). Triton handles the
    small, gather-heavy per-pair tiles and fp64. ``interpret=True`` ignores the
    backend and runs CPU semantics.
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

    def bs_full(arr: jax.Array) -> pl.BlockSpec:
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
        backend=(None if interpret else backend),
        name=f"m2l_complex_fused_p{p}",
    )(mr, mi, btr, bti, bfr, bfi, rr, *table_arrays)

    return jax.lax.complex(out_r[:, :C], out_i[:, :C])

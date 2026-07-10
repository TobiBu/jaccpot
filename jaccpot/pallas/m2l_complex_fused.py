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


@functools.lru_cache(maxsize=None)
def m2l_complex_fused_tables(order: int) -> dict:
    """Precompute the pack/unpack + z-core tables for a given order.

    Returns a dict of int/float numpy arrays:
      pack_flat[p+1, md], pack_valid[p+1, md]  -- flat<-(block,col) gather
      blk[C], col[C]                            -- (block,col) of each flat coeff
      z_src[C, K], z_valid[C, K], z_sign[C, K], z_fact[C, K], z_exp[C, K]
    where md = 2p+1, C = (p+1)^2, K = p+1 (max #sources per output coeff).
    """
    p = int(order)
    C = sh_size(p)
    md = 2 * p + 1
    K = p + 1

    # factorials up to 2p
    fact = np.ones(2 * p + 1, dtype=np.float64)
    for i in range(1, 2 * p + 1):
        fact[i] = fact[i - 1] * i

    # pack: block b, col c (0..2b) <- flat sh_offset(b)+c
    pack_flat = np.zeros((p + 1, md), dtype=np.int32)
    pack_valid = np.zeros((p + 1, md), dtype=np.float64)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            pack_flat[b, c] = sh_offset(b) + c
            pack_valid[b, c] = 1.0

    # (block, col) of each flat coefficient  (for unpack)
    blk = np.zeros(C, dtype=np.int32)
    col = np.zeros(C, dtype=np.int32)
    for b in range(p + 1):
        for c in range(2 * b + 1):
            i = sh_offset(b) + c
            blk[i] = b
            col[i] = c

    # z-core (flat): out (n,m) at o = sh_offset(n)+(m+n); sources k=|m|..p-n,
    # src (k,m) at sh_offset(k)+(m+k); coeff (-1)^m * fact[n+k] / r^(n+k+1).
    z_src = np.zeros((C, K), dtype=np.int32)
    z_valid = np.zeros((C, K), dtype=np.float64)
    z_sign = np.zeros((C, K), dtype=np.float64)
    z_fact = np.zeros((C, K), dtype=np.float64)
    z_exp = np.zeros((C, K), dtype=np.float64)
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

    return dict(
        p=p,
        C=C,
        md=md,
        K=K,
        pack_flat=pack_flat,
        pack_valid=pack_valid,
        blk=blk,
        col=col,
        z_src=z_src,
        z_valid=z_valid,
        z_sign=z_sign,
        z_fact=z_fact,
        z_exp=z_exp,
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


def _m2l_one(mult_r, mult_i, bto_r, bto_i, bfrom_r, bfrom_i, r, t):
    """Full M2L for one pair in real/imag; `t` = tables dict of jnp arrays."""
    pack_flat = t["pack_flat"]
    pack_valid = t["pack_valid"]
    blk = t["blk"]
    col = t["col"]
    z_src = t["z_src"]
    z_valid = t["z_valid"]
    z_sign = t["z_sign"]
    z_fact = t["z_fact"]
    z_exp = t["z_exp"]

    # 1. pack flat -> [B, md]
    pk_r = mult_r[pack_flat] * pack_valid
    pk_i = mult_i[pack_flat] * pack_valid
    # 2. rotate to z (block-diagonal complex matmul)
    mr_r, mr_i = _block_matmul(bto_r, bto_i, pk_r, pk_i)
    # 3. unpack -> flat [C]
    mrf_r = mr_r[blk, col]
    mrf_i = mr_i[blk, col]
    # 4. z-core (real coefficients on real & imag independently)
    coeff = z_valid * z_sign * z_fact * (r ** (-z_exp))  # [C, K]
    lz_r = jnp.sum(coeff * mrf_r[z_src], axis=1)
    lz_i = jnp.sum(coeff * mrf_i[z_src], axis=1)
    # 5. pack -> [B, md]
    pl_r = lz_r[pack_flat] * pack_valid
    pl_i = lz_i[pack_flat] * pack_valid
    # 6. rotate back
    or_r, or_i = _block_matmul(bfrom_r, bfrom_i, pl_r, pl_i)
    # 7. unpack -> flat [C]
    return or_r[blk, col], or_i[blk, col]


def _tables_to_jnp(order, real_dtype):
    t = m2l_complex_fused_tables(order)
    return dict(
        pack_flat=jnp.asarray(t["pack_flat"]),
        pack_valid=jnp.asarray(t["pack_valid"], dtype=real_dtype),
        blk=jnp.asarray(t["blk"]),
        col=jnp.asarray(t["col"]),
        z_src=jnp.asarray(t["z_src"]),
        z_valid=jnp.asarray(t["z_valid"], dtype=real_dtype),
        z_sign=jnp.asarray(t["z_sign"], dtype=real_dtype),
        z_fact=jnp.asarray(t["z_fact"], dtype=real_dtype),
        z_exp=jnp.asarray(t["z_exp"], dtype=real_dtype),
    )


def m2l_complex_fused_jax(multipoles, blocks_to_z, blocks_from_z, r, *, order):
    """Pure-jnp reference for the fused kernel (vmapped over pairs).

    multipoles: [N, C] complex; blocks_*: [N, p+1, md, md] complex; r: [N].
    Returns [N, C] complex local contributions.
    """
    real_dtype = jnp.asarray(multipoles).real.dtype
    t = _tables_to_jnp(order, real_dtype)
    mr = jnp.real(multipoles)
    mi = jnp.imag(multipoles)
    bto_r = jnp.real(blocks_to_z)
    bto_i = jnp.imag(blocks_to_z)
    bfr_r = jnp.real(blocks_from_z)
    bfr_i = jnp.imag(blocks_from_z)

    def one(mr, mi, btr, bti, bfr, bfi, rr):
        return _m2l_one(mr, mi, btr, bti, bfr, bfi, rr, t)

    or_r, or_i = jax.vmap(one)(mr, mi, bto_r, bto_i, bfr_r, bfr_i, r)
    return jax.lax.complex(or_r, or_i)


# --------------------------------------------------------------------------
# Pallas kernel: one program per pair. Same arithmetic as _m2l_one.
# --------------------------------------------------------------------------


_TABLE_KEYS = (
    "pack_flat",
    "pack_valid",
    "blk",
    "col",
    "z_src",
    "z_valid",
    "z_sign",
    "z_fact",
    "z_exp",
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
    multipoles, blocks_to_z, blocks_from_z, r, *, order, interpret=False
):
    """Fused complex-basis M2L via a Pallas kernel (one program per pair).

    Same signature/semantics as ``m2l_complex_fused_jax``. Requires an Ampere+
    GPU unless ``interpret=True``.
    """
    p = int(order)
    N = multipoles.shape[0]
    C = sh_size(p)
    md = 2 * p + 1
    real_dtype = jnp.asarray(multipoles).real.dtype
    tables = _tables_to_jnp(p, real_dtype)

    mr = jnp.real(multipoles).astype(real_dtype)
    mi = jnp.imag(multipoles).astype(real_dtype)
    btr = jnp.real(blocks_to_z).astype(real_dtype)
    bti = jnp.imag(blocks_to_z).astype(real_dtype)
    bfr = jnp.real(blocks_from_z).astype(real_dtype)
    bfi = jnp.imag(blocks_from_z).astype(real_dtype)
    rr = jnp.asarray(r, dtype=real_dtype)

    def bs_vec(cols):
        return pl.BlockSpec((1, cols), lambda i: (i, 0))

    def bs_blocks():
        return pl.BlockSpec((1, p + 1, md, md), lambda i: (i, 0, 0, 0))

    # tables are identical across pairs: replicate them to every program via a
    # full-array BlockSpec whose index_map ignores the pair index.
    table_arrays = [tables[k] for k in _TABLE_KEYS]

    def bs_full(arr):
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    table_specs = [bs_full(a) for a in table_arrays]

    out_r, out_i = pl.pallas_call(
        _m2l_complex_fused_kernel,
        grid=(N,),
        in_specs=[
            bs_vec(C),
            bs_vec(C),  # mult r/i  [1, C]
            bs_blocks(),
            bs_blocks(),  # bto r/i   [1, p+1, md, md]
            bs_blocks(),
            bs_blocks(),  # bfrom r/i
            pl.BlockSpec((1,), lambda i: (i,)),  # r [1]
            *table_specs,
        ],
        out_specs=[bs_vec(C), bs_vec(C)],
        out_shape=[
            jax.ShapeDtypeStruct((N, C), real_dtype),
            jax.ShapeDtypeStruct((N, C), real_dtype),
        ],
        interpret=interpret,
        name=f"m2l_complex_fused_p{p}",
    )(mr, mi, btr, bti, bfr, bfi, rr, *table_arrays)

    return jax.lax.complex(out_r, out_i)

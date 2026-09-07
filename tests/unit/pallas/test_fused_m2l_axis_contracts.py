"""Axis contracts for the two fused M2L kernels and their pure-JAX twins.

`runtime/kernels/_m2l.py`'s module docstring pins an equivalence: the fused Pallas kernels
must equal their references, `m2l_real_fused_pallas == _m2l_real_batch_kernel` and the
complex pair likewise. Signatures are part of that -- a twin that accepts a shape its
reference rejects is not a drop-in replacement -- so both halves carry the SAME axes and
these tests assert that rather than each in isolation.

The shapes come from the 2026-09-04 recording, at four expansion orders:

    p=2   blocks (pairs, 3, 5, 5)     multipoles (pairs, 9)
    p=3   blocks (pairs, 4, 7, 7)     multipoles (pairs, 16)
    p=4   blocks (pairs, 5, 9, 9)     multipoles (pairs, 25)
    p=6   blocks (pairs, 7, 13, 13)   multipoles (pairs, 49)

so `degrees` is `p+1`, `blockdim` is `2p+1` and square, and `sh` is `(p+1)**2`.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.pallas.m2l_complex_fused import (
    m2l_complex_fused_jax,
    m2l_complex_fused_vjp_pallas,
)
from jaccpot.pallas.m2l_real_fused import (
    m2l_real_fused_jax,
    m2l_real_fused_vjp_pallas,
)

PAIRS, P = 3, 2
DEGREES, BLOCKDIM, SH = P + 1, 2 * P + 1, (P + 1) ** 2


def _args(dtype):
    """Build one valid fused-M2L argument set.

    Parameters
    ----------
    dtype : Any
        Element dtype -- complex for the complex twin, float for the real one.

    Returns
    -------
    dict
        Keyword arguments for either `*_fused_jax`.
    """
    return {
        "multipoles": jnp.zeros((PAIRS, SH), dtype=dtype),
        "blocks_to_z": jnp.zeros((PAIRS, DEGREES, BLOCKDIM, BLOCKDIM), dtype=dtype),
        "blocks_from_z": jnp.zeros((PAIRS, DEGREES, BLOCKDIM, BLOCKDIM), dtype=dtype),
        "r": jnp.ones((PAIRS,), dtype=jnp.float64),
    }


CASES = [
    ("complex", m2l_complex_fused_jax, jnp.complex128),
    ("real", m2l_real_fused_jax, jnp.float64),
]
IDS = [name for name, _, _ in CASES]


@pytest.mark.parametrize("fn,dtype", [(f, d) for _, f, d in CASES], ids=IDS)
def test_a_matched_call_still_goes_through(fn, dtype):
    """The control. Both twins take the same axes, so both must accept the same input."""
    out = fn(**_args(dtype), order=P)
    assert out.shape == (PAIRS, SH)


@pytest.mark.parametrize("fn,dtype", [(f, d) for _, f, d in CASES], ids=IDS)
def test_the_pair_axis_is_shared_by_every_argument(fn, dtype):
    """`multipoles`, both block stacks and `r` are one interaction list."""
    for name in ("multipoles", "blocks_to_z", "blocks_from_z", "r"):
        args = _args(dtype)
        args[name] = args[name][:-1]
        with pytest.raises(TypeCheckError):
            fn(**args, order=P)


@pytest.mark.parametrize("fn,dtype", [(f, d) for _, f, d in CASES], ids=IDS)
def test_the_rotation_blocks_are_square_in_blockdim(fn, dtype):
    """`blockdim` is `2p+1` a side, and both trailing axes carry the same name.

    A block that is not square still contracts against the multipole vector on one side
    and produces a wrong-length result on the other, which is why the two axes share a
    name rather than being two anonymous ones.
    """
    args = _args(dtype)
    args["blocks_to_z"] = args["blocks_to_z"][:, :, :, :-1]
    with pytest.raises(TypeCheckError):
        fn(**args, order=P)


@pytest.mark.parametrize("fn,dtype", [(f, d) for _, f, d in CASES], ids=IDS)
def test_the_two_block_stacks_must_agree_on_degrees(fn, dtype):
    """`blocks_to_z` and `blocks_from_z` are the two halves of one rotation.

    They are built for the same order, so a `degrees` disagreement means one half was
    built at a different `p` -- the kind of mismatch that produces a plausible wrong
    translation rather than an error.
    """
    args = _args(dtype)
    args["blocks_from_z"] = args["blocks_from_z"][:, :-1]
    with pytest.raises(TypeCheckError):
        fn(**args, order=P)


VJP_CASES = [
    ("complex", m2l_complex_fused_vjp_pallas, jnp.complex128),
    ("real", m2l_real_fused_vjp_pallas, jnp.float64),
]


@pytest.mark.parametrize(
    "fn,dtype", [(f, d) for _, f, d in VJP_CASES], ids=[n for n, _, _ in VJP_CASES]
)
def test_the_cotangent_shares_sh_with_the_multipoles(fn, dtype):
    """`out_bar` is the cotangent OF `multipoles`, so they are one shape.

    This is where `sh` actually binds. On the forward twins it occurs once per signature
    and so asserts nothing about length -- a short multipole there is caught by the body,
    which raises `mul got incompatible shapes for broadcasting` rather than a
    `TypeCheckError`. Verified, not assumed: that is what the forward call does today,
    and it is why this test targets the reverse entry points instead.
    """
    args = _args(dtype)
    args["out_bar"] = jnp.zeros((PAIRS, SH), dtype=dtype)

    with pytest.raises(TypeCheckError):
        fn(**dict(args, out_bar=args["out_bar"][:, :-1]), order=P)
    with pytest.raises(TypeCheckError):
        fn(**dict(args, multipoles=args["multipoles"][:, :-1]), order=P)

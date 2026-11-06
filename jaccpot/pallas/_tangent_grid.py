"""Tangents on the grid: how many tangents one tangent-kernel program carries.

Every Pallas tangent kernel (P2M, M2M, L2L, M2L, L2P, near field) used to unroll
all ``K`` tangents of a launch in one program body. At ``p = 6`` and ``K = 8`` the
M2M body is ~40k straight-line lane operations with ``8 C`` live tangent values:
register spills and a Triton compile that grows superlinearly in ``K`` (2e6
``jacfwd``: ``K = 1`` 4 min, ``K = 4`` 23 min, ``K = 8`` ~75 min, measured
2026-11-05).

So each tangent kernel has a grid axis over *tangent groups*: a program carries
``tpp`` (*tangents per program*) consecutive tangents ``g tpp .. g tpp + tpp - 1``
of the stacked ``[K, ...]`` operands, and the body unrolls over ``tpp`` only. The
operands and outputs stay ``K``-stacked; the group's program id selects their rows.
A ``K`` that is not a multiple of ``tpp`` leaves a partial last group: its rows past
``K`` are clamped on load (they read tangent ``K - 1``) and masked on store, so no
padded copy of a tangent operand is made. ``tpp = K`` is the old fused body.

``tpp`` is a static choice per seam: the op parameter ``tangents_per_program``, else
the environment ``JACCPOT_TANGENT_TPP_<SEAM>`` (``SEAM`` one of :data:`TANGENT_SEAMS`),
else :data:`TANGENT_TPP_DEFAULT`; it is clamped to the power of two at or above
``K`` (a ``K = 1`` launch is one tangent per program whatever the setting). The
primal intermediates a program shares between its tangents are recomputed once per
group, so a smaller ``tpp`` trades compile time and registers for repeated primal
work; the production defaults come from the HoreKa sweep.
"""

from __future__ import annotations

from typing import Any, Optional

#: The tangents-per-program values a tangent kernel is compiled for.
TANGENT_TPP_CHOICES: tuple[int, ...] = (1, 2, 4, 8)

#: Tangents per program when neither the op nor the environment chooses
#: (provisional, until the HoreKa sweep).
TANGENT_TPP_DEFAULT: int = 2

#: The seams with a Pallas tangent kernel, as spelled in ``JACCPOT_TANGENT_TPP_<SEAM>``.
TANGENT_SEAMS: tuple[str, ...] = ("P2M", "M2M", "L2L", "M2L", "L2P", "NEAR")


def _next_pow2(value: int) -> int:
    out = 1
    while out < value:
        out *= 2
    return out


def tangents_per_program(seam: str, k: int, override: Optional[int] = None) -> int:
    """The tangents one program of ``seam``'s tangent kernel carries for ``K = k``.

    Parameters
    ----------
    seam : str
        One of :data:`TANGENT_SEAMS`.
    k : int
        Tangents of the launch (``>= 1``).
    override : Optional[int]
        The op's ``tangents_per_program`` parameter; ``None`` reads
        ``JACCPOT_TANGENT_TPP_<SEAM>`` (default :data:`TANGENT_TPP_DEFAULT`).

    Returns
    -------
    int
        The chosen value, clamped to the power of two at or above ``k``.

    Raises
    ------
    ValueError
        For an unknown seam, ``k < 1`` or a value outside :data:`TANGENT_TPP_CHOICES`.
    """
    from jaccpot._env import env_int

    name = str(seam).upper()
    if name not in TANGENT_SEAMS:
        raise ValueError(f"unknown tangent seam {seam!r}; one of {TANGENT_SEAMS}")
    k = int(k)
    if k < 1:
        raise ValueError(f"a tangent launch carries K >= 1 tangents, got {k}")
    if override is None:
        value = env_int(f"JACCPOT_TANGENT_TPP_{name}", TANGENT_TPP_DEFAULT)
        source = f"JACCPOT_TANGENT_TPP_{name}"
    else:
        value = int(override)
        source = "tangents_per_program"
    if value not in TANGENT_TPP_CHOICES:
        raise ValueError(f"{source}={value}: must be one of {TANGENT_TPP_CHOICES}")
    return min(value, _next_pow2(k))


def tangent_k(*tangents: Any) -> int:
    """``K`` of a launch: the largest leading axis of its tangent operands.

    A dead tangent may arrive as a ``(1, ...)`` placeholder
    (``make_tangent_op(..., dead_placeholders=True)``), so ``K`` is never read off
    one fixed operand.

    Parameters
    ----------
    *tangents : Any
        The tangent operands (arrays or shape/dtype structs).

    Returns
    -------
    int
        ``max_i tangents[i].shape[0]``.
    """
    return max(int(t.shape[0]) for t in tangents)


def tangent_op_options() -> dict:
    """``make_tangent_op`` keywords of the six Pallas tangent ops.

    ``dead_placeholders=True``: the implementations take ``K`` from
    :func:`tangent_k` and never read a dead tangent, so a dead one stays its
    ``(1, ...)`` placeholder. ``any_k=True``: the tangent-group grid axis masks a
    partial last group, so the op is called once at the operands' ``K`` (no bucket
    padding, no chunks of 8).

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {"dead_placeholders": True, "any_k": True}


#: Lanes per program of :func:`late_zeros`.
_ZEROS_BLOCK = 1024


def late_zeros(
    shape: tuple[int, ...],
    dtype: Any,
    dep: Any,
    *,
    interpret: bool,
    backend: str = "triton",
) -> Any:
    """A zero table for a tangent kernel to update in place, made AFTER ``dep``.

    ``jnp.zeros`` is an operand-free broadcast: XLA hoists it and merges the
    same-shaped zero tables of every force evaluation in a program (the two forces
    of a KDK step) into ONE multi-output fusion at the program's start, so every
    such table is live from the start (2e6 ``jacfwd``, ``K = 8``: two
    ``f32[8, nodes, 49]`` P2M tables and four ``f32[8, N, 3]`` L2P / near tables,
    measured 2026-11-05). This is a Pallas fill whose operand is a one-element
    slice of ``dep`` (an input of the consuming call): it can neither run before
    ``dep`` exists nor be merged with another evaluation's fill.

    Parameters
    ----------
    shape : tuple[int, ...]
        The table's shape.
    dtype : Any
        Its dtype.
    dep : Any
        An array the consumer reads (e.g. its first primal).
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.

    Returns
    -------
    Any
        Zeros of ``shape`` and ``dtype``.
    """
    import math

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    from jaccpot.pallas._compat import pallas_backend_kwargs

    size = int(math.prod(shape))
    if size == 0:
        return jnp.zeros(shape, dtype)
    wide = size >= 2**31 - _ZEROS_BLOCK
    idx_dtype = jnp.int64 if wide else jnp.int32

    def kernel(dep_ref: Any, out_ref: Any) -> None:
        del dep_ref
        base = pl.program_id(0).astype(idx_dtype) * _ZEROS_BLOCK
        idx = base + lax.broadcasted_iota(idx_dtype, (_ZEROS_BLOCK,), 0)
        plgpu.store(out_ref.at[idx], jnp.zeros((_ZEROS_BLOCK,), dtype), mask=idx < size)

    token = jnp.reshape(jnp.asarray(dep), (-1,))[:1]
    out = pl.pallas_call(
        kernel,
        grid=(-(-size // _ZEROS_BLOCK),),
        in_specs=[pl.BlockSpec((1,), lambda i: (0,))],
        out_specs=pl.BlockSpec((size,), lambda i: (0,)),
        out_shape=jax.ShapeDtypeStruct((size,), dtype),
        interpret=bool(interpret),
        name="tangent_late_zeros",
        **pallas_backend_kwargs(backend, interpret),
    )(token)
    return jnp.reshape(out, shape)


def tangent_groups(k: int, tpp: int) -> int:
    """Programs along the tangent-group grid axis: ``ceil(k / tpp)``.

    Parameters
    ----------
    k : int
        Tangents of the launch.
    tpp : int
        Tangents per program.

    Returns
    -------
    int
        The group count (the last group may be partial).
    """
    return -(-int(k) // int(tpp))


__all__ = [
    "TANGENT_SEAMS",
    "TANGENT_TPP_CHOICES",
    "TANGENT_TPP_DEFAULT",
    "late_zeros",
    "tangent_groups",
    "tangent_k",
    "tangent_op_options",
    "tangents_per_program",
]

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

from typing import Optional

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
    "tangent_groups",
    "tangents_per_program",
]

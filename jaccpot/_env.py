"""Canonical environment-variable readers for Jaccpot's runtime knobs.

Jaccpot exposes a number of execution knobs through ``JACCPOT_*`` environment
variables. Four near-identical private readers had accumulated across
``runtime/fmm_constants``, ``nearfield/near_field``,
``upward/solidfmm_complex_tree_expansions`` and ``runtime/_interaction_cache``,
and they did **not** agree: one clamped integers to a minimum of 1, the others
did not, so the same env value meant different things depending on which module
read it. This module is the single implementation; the per-module names remain
as thin aliases so no call site changes semantics.

Two rules that matter for correctness, not style:

* **Read at call time, never at import time.** A knob captured into a
  module-level constant cannot be changed by a user who sets the variable after
  importing ``jaccpot`` -- it silently does nothing, which is worse than not
  having the knob. Call these functions from inside the code that needs the
  value.
* **Never raise.** A malformed value falls back to the default. These are
  performance and diagnostic switches; a typo in one must not take down a run.

This module deliberately imports nothing from ``jaccpot`` so that any module,
at any layer, can use it without an import cycle.
"""

from __future__ import annotations

import os
from typing import Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})

__all__ = ["env_flag", "env_float", "env_int"]


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean switch from the environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : bool
        Value to use when the variable is unset.

    Returns
    -------
    bool
        ``True`` when the value is one of ``1``/``true``/``yes``/``on``
        (case-insensitive, surrounding whitespace ignored). Any other value --
        including an unparseable one -- reads as ``False``, so an explicit
        ``=0`` reliably turns a default-on knob off.
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUTHY


def env_int(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    """Read an integer knob from the environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : int
        Value to use when the variable is unset or unparseable.
    minimum : Optional[int]
        When given, clamp the result upward to this bound. Use it for knobs
        that index or size a buffer, where 0 or a negative value would be a
        shape error rather than a slow configuration. Leave it ``None`` when 0
        is meaningful -- several call sites use 0 as "unset, pick a default".

    Returns
    -------
    int
        The parsed value, clamped when ``minimum`` is given.
    """
    raw = os.environ.get(name)
    if raw is None:
        value = int(default)
    else:
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            value = int(default)
    if minimum is not None:
        return max(value, int(minimum))
    return value


def env_float(name: str, default: float, *, minimum: Optional[float] = None) -> float:
    """Read a float knob from the environment.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : float
        Value to use when the variable is unset or unparseable.
    minimum : Optional[float]
        When given, clamp the result upward to this bound.

    Returns
    -------
    float
        The parsed value, clamped when ``minimum`` is given.
    """
    raw = os.environ.get(name)
    if raw is None:
        value = float(default)
    else:
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            value = float(default)
    if minimum is not None:
        return max(value, float(minimum))
    return value

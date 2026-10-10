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
* **Never raise.** A malformed value falls back to the default -- *the* default,
  whatever it is, not a fixed ``False``. That makes a typo a no-op instead of a
  silent behaviour change. ``env_flag`` used to differ here: it read anything
  outside ``{1,true,yes,on}`` as ``False``, so ``JACCPOT_ANALYTIC_P2P_VJP=ture``
  quietly *disabled* the analytic VJP while ``env_int``/``env_float`` in this same
  module already fell back to their default. Audit 2.2; the hand-rolled readers it
  replaced used the default-preserving rule, and that is the one kept.
* **Warn once per variable on malformed input.** Falling back silently is what lets
  a typo survive a whole session. The warning names the variable and the value it
  ignored, and is emitted at most once per variable per process so a knob read
  inside a hot path cannot spam.

The one reader that raises is :func:`env_reject_removed`, and it raises on purpose:
a value that selected a kernel or lane the 2026-10 cleanup deleted is not a typo.
Falling back to the default would run something other than what the caller asked
for, so it is refused by name.

This module deliberately imports nothing from ``jaccpot`` so that any module,
at any layer, can use it without an import cycle.
"""

from __future__ import annotations

import os
import warnings
from typing import Iterable, Optional

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSEY = frozenset({"0", "false", "no", "off"})

#: Variables already warned about, so a knob read per call does not spam. Process
#: wide on purpose; `_reset_malformed_warning_cache` exists for tests.
_WARNED: set[str] = set()

__all__ = [
    "env_choice",
    "env_flag",
    "env_flag_optional",
    "env_float",
    "env_int",
    "env_reject_removed",
    "env_text",
]


def _reset_malformed_warning_cache() -> None:
    """Forget which variables have already warned.

    Test-only. Without it the second test to read a given variable would assert
    nothing, because the warning would have been suppressed by the first.

    Returns
    -------
    None
        Clears process-wide state.
    """
    _WARNED.clear()


def _warn_malformed(name: str, raw: str, expected: str, default: object) -> None:
    """Warn once that a value could not be parsed and the default was used.

    Parameters
    ----------
    name : str
        Environment variable name.
    raw : str
        The value that could not be parsed, quoted into the message so the user can
        see the typo rather than being told one exists.
    expected : str
        Human-readable description of what would have been accepted.
    default : object
        The value used instead.

    Returns
    -------
    None
        Emits at most one warning per variable per process.
    """
    if name in _WARNED:
        return
    _WARNED.add(name)
    warnings.warn(
        f"{name}={raw!r} is not a valid value ({expected}); using the default "
        f"{default!r}. The setting has had no effect.",
        RuntimeWarning,
        stacklevel=3,
    )


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
        ``True`` for ``1``/``true``/``yes``/``on`` and ``False`` for
        ``0``/``false``/``no``/``off``, case-insensitively and ignoring surrounding
        whitespace. **Anything else is malformed and yields ``default``**, with a
        one-time warning -- so an explicit ``=0`` still reliably turns a default-on
        knob off, while a typo changes nothing.
    """
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    _warn_malformed(
        name, str(raw), "expected one of 1/true/yes/on/0/false/no/off", bool(default)
    )
    return bool(default)


def env_flag_optional(name: str) -> Optional[bool]:
    """Read a boolean switch that also has an "unset" meaning.

    Some knobs are tri-state: unset means "decide automatically", while an
    explicit ``0``/``1`` overrides that decision
    (``JACCPOT_NEARFIELD_LEAFPAIR_CSR``). Call sites used to test
    ``os.environ`` directly for that; this keeps the read inside ``_env``.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    Optional[bool]
        ``None`` when the variable is unset, else :func:`env_flag`'s reading of
        it (a malformed value warns once and yields ``False``).
    """
    if os.environ.get(name) is None:
        return None
    return env_flag(name, False)


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
            _warn_malformed(name, str(raw), "expected an integer", int(default))
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
            _warn_malformed(name, str(raw), "expected a number", float(default))
            value = float(default)
    if minimum is not None:
        return max(value, float(minimum))
    return value


def env_choice(name: str, default: str, choices: Iterable[str]) -> str:
    """Read a string knob restricted to a fixed set of values.

    The reader ``_env`` was missing: several diagnostic knobs are enums rather than
    flags (``JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE``,
    ``JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE``), and each had hand-rolled the same
    normalise-then-fall-back logic.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Value to use when the variable is unset or holds an unrecognised mode. Must
        itself be one of ``choices``.
    choices : Iterable[str]
        The accepted values, compared lowercased with surrounding whitespace removed.

    Returns
    -------
    str
        The normalised value when recognised, otherwise ``default``.

    Raises
    ------
    ValueError
        If ``default`` is not in ``choices``. That is a call-site bug rather than
        user input, so it surfaces immediately -- the module's "never raise" rule is
        about *environment* values, which this is not.
    """
    allowed = {str(c).strip().lower() for c in choices}
    normalised_default = str(default).strip().lower()
    if normalised_default not in allowed:
        raise ValueError(
            f"env_choice({name!r}) default {default!r} is not one of {sorted(allowed)}"
        )
    raw = os.environ.get(name)
    if raw is None:
        return normalised_default
    text = str(raw).strip().lower()
    if text in allowed:
        return text
    _warn_malformed(
        name, str(raw), f"expected one of {sorted(allowed)}", normalised_default
    )
    return normalised_default


def env_text(name: str, default: str = "") -> str:
    """Read a variable verbatim, for checks on settings jaccpot does not own.

    The other readers parse a ``JACCPOT_*`` knob into a value. This one only hands
    back the text -- e.g. ``XLA_FLAGS``, which jaccpot inspects to warn about a slow
    configuration but never parses or sets.

    Parameters
    ----------
    name : str
        Environment variable name.
    default : str
        Value to use when the variable is unset.

    Returns
    -------
    str
        The raw value, or ``default``.
    """
    raw = os.environ.get(name)
    return default if raw is None else str(raw)


def env_reject_removed(
    name: str, removed: Iterable[str], *, phase: str, default: str
) -> None:
    """Raise if a knob holds a value whose kernel or lane the 2026-10 cleanup removed.

    The other readers fall back to the default on a value they do not know. A
    removed value is not that: it asked for a code path that no longer exists, and
    running the default instead would silently change what the caller measured.

    Parameters
    ----------
    name : str
        Environment variable name.
    removed : Iterable[str]
        The removed values, compared lowercased with surrounding whitespace removed.
    phase : str
        The cleanup phase that removed them (``docs/cleanup_2026-10.md``), e.g.
        ``"X5"``.
    default : str
        What runs instead, for the message: the surviving default.

    Raises
    ------
    ValueError
        If the variable is set to one of ``removed``. Unset, or any other value,
        returns quietly; the caller's own reader decides what else is valid.
    """
    raw = os.environ.get(name)
    if raw is None:
        return
    text = str(raw).strip().lower()
    if text in {str(r).strip().lower() for r in removed}:
        raise ValueError(
            f"{name}={raw!r} was removed in the 2026-10 cleanup ({phase}); the "
            f"default is {default}. Unset the variable."
        )

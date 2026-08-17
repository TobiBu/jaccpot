"""Runtime type-check configuration for jaccpot."""

from __future__ import annotations

import os
from typing import Any

_TYPECHECK_HOOK: Any = None


def _runtime_typecheck_enabled() -> bool:
    raw = os.getenv("JACCPOT_RUNTIME_TYPECHECK", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enable_runtime_typecheck() -> bool:
    """Enable package-wide jaxtyping+beartype checks for annotated callables.

    Called once from :mod:`jaccpot.__init__` before any submodule is imported.
    ``jaxtyping.install_import_hook`` only instruments modules imported *after*
    the hook is installed, so calling this later — from a test, say — leaves
    everything already in ``sys.modules`` unchecked.

    Returns
    -------
    bool
        Whether the hook is installed. ``True`` also on repeat calls that found
        it already installed, so this is "checks are active", not "this call
        installed them". ``False`` means ``JACCPOT_RUNTIME_TYPECHECK`` is unset
        or one of ``0``/``false``/``no``/``off`` and nothing was instrumented.
    """
    global _TYPECHECK_HOOK

    if _TYPECHECK_HOOK is not None:
        return True
    if not _runtime_typecheck_enabled():
        return False

    from jaxtyping import install_import_hook

    # Instruments submodule imports under `jaccpot` so annotated callables are
    # checked by beartype with jaxtyping's shape/dtype semantics.
    _TYPECHECK_HOOK = install_import_hook("jaccpot", typechecker="beartype.beartype")
    return True


__all__ = ["enable_runtime_typecheck"]

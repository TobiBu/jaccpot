"""Runtime type-check configuration for jaccpot."""

from __future__ import annotations

import os
from typing import Any

_TYPECHECK_HOOK: Any = None


def _runtime_typecheck_enabled() -> bool:
    raw = os.getenv("JACCPOT_RUNTIME_TYPECHECK", "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def enable_runtime_typecheck() -> bool:
    """Enable package-wide jaxtyping+beartype checks for annotated callables."""
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

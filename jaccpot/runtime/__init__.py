"""Runtime policies, caches, and kernel dispatch namespace."""

from __future__ import annotations

from typing import Any

#: forward-mode helpers, imported on first use (``jaccpot.runtime`` stays light)
_LAZY = {
    "apply_in_chunks": "jaccpot.runtime.forward_batching",
    "forward_memory_estimate": "jaccpot.runtime.forward_batching",
    "jacfwd_batched": "jaccpot.runtime.forward_batching",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """The lazily imported helpers of :data:`_LAZY`.

    Parameters
    ----------
    name : str
        The attribute.

    Returns
    -------
    Any
        The helper.

    Raises
    ------
    AttributeError
        For any other name.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'jaccpot.runtime' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)

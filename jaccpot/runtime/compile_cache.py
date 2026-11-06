"""Opt-in persistent compilation cache (``JACCPOT_COMPILATION_CACHE_DIR``).

The forward-mode programs compile for minutes to over an hour (the 2e6 ``jacfwd``
at ``K = 8``: ~75 min on an A100, round 1), and every process compiles them again.
JAX's persistent compilation cache stores the compiled executables on disk, keyed
by the program, the backend and the compile options; a second process with the
same program loads it instead.

It is OFF unless the environment names a directory::

    export JACCPOT_COMPILATION_CACHE_DIR=/scratch/$USER/jax-cache

Then :func:`maybe_enable_compilation_cache` -- called by the
:class:`~jaccpot.production.ProductionRunner`, by
:meth:`~jaccpot.production.ProductionRunner.gradient_plan` and by the forward-mode
benches -- sets ``jax_compilation_cache_dir`` once per process (idempotent), with
``jax_persistent_cache_min_compile_time_secs`` from
``JACCPOT_COMPILATION_CACHE_MIN_COMPILE_S`` (default 1 s: the eager prepare's many
small programs are not worth a file each). Setting ``jax_compilation_cache_dir``
yourself works as well; this is the same thing behind one variable.

JAX decides whether the cache is used at the FIRST compile of a process and never
again; if a program was compiled before the directory was set (an import, an
eager prepare), the cache is reset so the next compile sees the directory.

Notes: entries are only valid for the same jax / jaxlib / CUDA / driver and the
same program text (a different N, order, leaf capacity or ``K`` is a different
program); the directory grows without bound unless
``JACCPOT_COMPILATION_CACHE_MAX_BYTES`` is set (``jax_compilation_cache_max_size``,
least recently used entries are evicted). Use a node-local or a per-job directory
on a shared file system if several jobs compile the same program at once.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

__all__ = [
    "COMPILATION_CACHE_ENV",
    "maybe_enable_compilation_cache",
]

#: the environment variable naming the cache directory (unset: no cache)
COMPILATION_CACHE_ENV = "JACCPOT_COMPILATION_CACHE_DIR"
_MIN_COMPILE_ENV = "JACCPOT_COMPILATION_CACHE_MIN_COMPILE_S"
_MAX_BYTES_ENV = "JACCPOT_COMPILATION_CACHE_MAX_BYTES"

_log = logging.getLogger(__name__)
_applied: Optional[str] = None


def maybe_enable_compilation_cache(directory: Optional[str] = None) -> Optional[str]:
    """Point JAX's persistent compilation cache at ``JACCPOT_COMPILATION_CACHE_DIR``.

    Idempotent: a second call with the same directory does nothing.

    Parameters
    ----------
    directory : Optional[str]
        The directory; ``None`` reads :data:`COMPILATION_CACHE_ENV` (unset or
        empty: nothing is done).

    Returns
    -------
    Optional[str]
        The cache directory in use after the call (``None``: no cache set here).
    """
    global _applied
    path = directory if directory is not None else os.environ.get(COMPILATION_CACHE_ENV)
    if not path:
        return _applied
    path = os.path.abspath(os.path.expanduser(path))
    if _applied == path:
        return path
    import jax

    os.makedirs(path, exist_ok=True)
    min_s = float(os.environ.get(_MIN_COMPILE_ENV, "1.0"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", min_s)
    max_bytes = os.environ.get(_MAX_BYTES_ENV)
    if max_bytes:
        jax.config.update("jax_compilation_cache_max_size", int(max_bytes))
    jax.config.update("jax_compilation_cache_dir", path)
    try:  # JAX fixes the cache at the first compile: re-arm it if that was earlier
        from jax._src import compilation_cache as _cc

        if getattr(_cc, "_cache_initialized", False) or getattr(
            _cc, "_cache_checked", False
        ):
            _cc.reset_cache()
    except Exception as exc:  # noqa: BLE001 -- a private API; the dir is set anyway
        _log.warning("could not reset jax's compilation cache state: %s", exc)
    _applied = path
    _log.info("persistent compilation cache: %s (min compile %.1f s)", path, min_s)
    return path

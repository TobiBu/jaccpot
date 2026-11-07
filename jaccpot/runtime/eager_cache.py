"""Opt-in: the eager prepare's per-call programs compiled once (S10 follow-ups).

The eager prepare (``prepare_state`` and the runner's caps-only re-prepare) runs a
few pieces op by op, outside any ``jax.jit``, whose dispatch compiles a FRESH
program on every call, because each call builds new Python functions for them:

* yggdrax's ``_build_mac_extents`` -- four eager ``lax.while_loop`` extent
  propagations, each a new body closure (``jit(while)``), per walk;
* the P2M leaf kernel's ``pallas_call`` (a new kernel closure: ``jit(wrapped)``);
* the f_b seed's sorted CSR near field: its ``pallas_call`` and the eager
  ``lax.cond`` over the long rows' pieces (``jit(wrapped)``, ``jit(cond)``).

Measured on track F's 1e5 disc (A100, galaxy preset): 11 compiles, ~1.5 s, per WARM
runner pre-pass, every pre-pass. Under :func:`eager_jit_cache` (the runner's
``gradient_plan(freeze_cache=...)`` turns it on) those call sites run through one
``jax.jit`` per function and static arguments, kept in the cache it is given, so a
warm pre-pass compiles none of them. Outside it, and whenever an argument is a
tracer (a traced rebuild), :func:`eager_cached` calls the function as before: the
default programs are unchanged.

The cached program is the same computation jitted as a whole rather than
dispatched piece by piece; its outputs are the eager ones bit for bit
(``tests/integration/strict/test_runner_gradient_mode.py``, the freeze-cache
test: plans, seeds and the forward compared bitwise).
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterator, Optional

from jaccpot._env import env_snapshot

__all__ = ["eager_cached", "eager_jit_cache"]

_CACHE: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "jaccpot_eager_jit_cache", default=None
)


@contextlib.contextmanager
def eager_jit_cache(cache: dict) -> Iterator[dict]:
    """Run the eager prepare's per-call programs through ``cache`` (module docstring).

    Parameters
    ----------
    cache : dict
        Where the jitted programs are kept (reused across calls; one per function,
        static arguments and ``JACCPOT_*`` environment).

    Yields
    ------
    dict
        ``cache``.
    """
    token = _CACHE.set(cache)
    try:
        yield cache
    finally:
        _CACHE.reset(token)


def _environment() -> tuple:
    # the functions read JACCPOT_* switches when they trace: part of the key
    # (with the context's `_env.scoped_env_defaults`, which those reads see too)
    return tuple(env_snapshot(("JACCPOT_",)).items())


def eager_cached(
    fn: Callable[..., Any], *trailing: Any, **static: Any
) -> Callable[..., Any]:
    """``fn`` with trailing positional and keyword STATIC arguments bound.

    Under :func:`eager_jit_cache` the bound function is one ``jax.jit`` per
    ``(fn, trailing, static, environment)``, kept in the active cache, and called
    with the dynamic arguments; otherwise -- and when any dynamic argument is a
    tracer -- ``fn(*args, *trailing, **kwargs, **static)`` is called directly.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function.
    *trailing : Any
        Static positional arguments after the dynamic ones (hashable).
    **static : Any
        Static keyword arguments (hashable).

    Returns
    -------
    Callable[..., Any]
        ``call(*args, **kwargs)``: the dynamic arguments (arrays, pytrees of them).
    """

    def direct(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, *trailing, **kwargs, **static)

    def call(*args: Any, **kwargs: Any) -> Any:
        import jax

        from jaccpot._jax_compat import Tracer

        cache = _CACHE.get()
        if cache is None:
            return direct(*args, **kwargs)
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        if any(isinstance(leaf, Tracer) for leaf in leaves):
            return direct(*args, **kwargs)
        key = (fn, trailing, tuple(sorted(static.items())), _environment())
        program = cache.get(key)
        if program is None:
            program = cache[key] = jax.jit(direct)
        return program(*args, **kwargs)

    return call

"""The largest array a traced function materialises outside its Pallas kernels.

A memory property of a reverse rule -- "no per-particle coefficient buffer", "no
per-leaf-slot block" -- is a property of the arrays XLA allocates, which are the
outputs of the jaxpr's equations (and of its sub-jaxprs); the values inside a
``pallas_call`` body are registers and kernel refs.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.extend as jex
import numpy as np


def largest_xla_array(fn: Callable[..., Any], *args: Any) -> int:
    """Elements of the largest array the traced ``fn`` materialises outside kernels.

    Walks the jaxpr and its sub-jaxprs (``pjit``, ``custom_vjp``, loops), not the
    bodies of ``pallas_call``.

    Parameters
    ----------
    fn : Callable[..., Any]
        The function to trace.
    *args : Any
        Its arguments.

    Returns
    -------
    int
        The largest ``size`` of any equation output.
    """

    def walk(jaxpr: Any) -> int:
        best = 0
        for eqn in jaxpr.eqns:
            for v in eqn.outvars:
                best = max(best, int(np.prod(getattr(v.aval, "shape", ()) or (1,))))
            if eqn.primitive.name == "pallas_call":
                continue
            for p in eqn.params.values():
                for sub in p if isinstance(p, (list, tuple)) else (p,):
                    inner = getattr(sub, "jaxpr", sub)
                    if isinstance(inner, jex.core.Jaxpr):
                        best = max(best, walk(inner))
        return best

    return walk(jax.make_jaxpr(fn)(*args).jaxpr)

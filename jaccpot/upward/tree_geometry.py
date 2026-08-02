"""Compiled dispatch for ``compute_tree_geometry``.

``compute_tree_geometry`` is a large graph with two data-dependent loops: a
pointer-doubling ``lax.while_loop`` that derives node depths from the parent
array, and a level-by-level ``lax.fori_loop`` merging child bounding boxes whose
trip count is itself traced. Dispatched op by op, that structure costs far more
than the arithmetic in it.

Measured on an idle A100, ``large_n_gpu``/radix/real, leaf 256, evaluating on a
prebuilt tree -- the same call, eager against one ``jax.jit``:

=======  =======  ==========  ========  =========
N        nodes    op-by-op    compiled  factor
=======  =======  ==========  ========  =========
32768        255   571.67 ms   0.72 ms       794x
65536        511   585.73 ms   0.92 ms       637x
131072      1023   605.89 ms   0.96 ms       631x
=======  =======  ==========  ========  =========

Bit-identical in centres and radii (max abs diff 0.0): compiling a graph does not
reassociate it. Note the eager cost tracks the *node count*, which is ~2x the
leaf count and independent of the particles per leaf -- it is the loop structure
being re-dispatched, not the geometry being recomputed.

That is where the 56-59% of per-step refresh time attributed to
``upward_geometry`` went (measured 634 ms of a 1079 ms step at N=65536, against
2.0% for the P2M and M2M expansions it feeds).

The one-time compile is ~0.6-0.7 s and is cached per (tree type, leaf cap, input
avals), so a refresh loop on frozen topology recompiles nothing.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import jax
from yggdrax.geometry import TreeGeometry, compute_tree_geometry

__all__ = ["compute_tree_geometry_compiled"]

# Keyed by (tree class, leaf cap). jax.jit keys on the pytree structure and leaf
# avals itself, so this only has to separate the Python-level statics. Bounded
# because an unbounded cache of compiled executables is a memory leak: a caller
# sweeping leaf caps would accumulate one entry per value.
_CACHE: dict[tuple[Any, ...], Callable[..., TreeGeometry]] = {}
_CACHE_MAX_ENTRIES = 64


def _jit_enabled() -> bool:
    """Read per call, so the flag can be set after import (notebooks)."""

    return str(os.environ.get("JACCPOT_TREE_GEOMETRY_JIT", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def compute_tree_geometry_compiled(
    tree: Any,
    positions_sorted: Any,
    *,
    max_leaf_size: Optional[int] = None,
) -> TreeGeometry:
    """``compute_tree_geometry`` through a cached ``jax.jit``.

    Falls back to the uncompiled call, with identical semantics, when compiling
    would be wrong or pointless:

    * an outer trace is already in charge (tracers in the arguments), so nesting
      a second ``jax.jit`` would only re-trace;
    * ``JACCPOT_TREE_GEOMETRY_JIT=0``, the escape hatch for A/B measurement.
    """

    if not _jit_enabled():
        return compute_tree_geometry(
            tree, positions_sorted, max_leaf_size=max_leaf_size
        )

    leaves = jax.tree_util.tree_leaves((tree, positions_sorted))
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return compute_tree_geometry(
            tree, positions_sorted, max_leaf_size=max_leaf_size
        )

    cap = None if max_leaf_size is None else int(max_leaf_size)
    key = (type(tree), cap)
    fn = _CACHE.get(key)
    if fn is None:
        if len(_CACHE) >= _CACHE_MAX_ENTRIES:
            _CACHE.clear()

        def _compute(tree_in: Any, positions_in: Any) -> TreeGeometry:
            return compute_tree_geometry(tree_in, positions_in, max_leaf_size=cap)

        fn = jax.jit(_compute)
        _CACHE[key] = fn
    return fn(tree, positions_sorted)

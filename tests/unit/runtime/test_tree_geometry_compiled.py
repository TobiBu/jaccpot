"""``compute_tree_geometry`` must go through the compiled dispatch, and agree.

Dispatched op by op, ``compute_tree_geometry`` cost 56-59% of a per-step refresh
-- measured 634 ms of a 1079 ms step at N=65536, against 2.0% for the P2M and
M2M expansions it feeds. It is a large graph with two data-dependent loops
(pointer-doubling for node depths, then a level-by-level bounding-box merge whose
trip count is itself traced), and re-dispatching that structure is what cost the
time, not the arithmetic in it. Measured on an idle A100, leaf 256:

=======  ==========  =========
N        op-by-op    compiled
=======  ==========  =========
32768     571.67 ms    0.72 ms
65536     585.73 ms    0.92 ms
131072    605.89 ms    0.96 ms
=======  ==========  =========

These tests do not assert on timings -- a shared GPU makes that flaky, and the
speedup is recorded in the commit and the module docstring. They assert the two
things a wrong version would break: that the compiled dispatch returns exactly
what the uncompiled one does, and that no jaccpot call site bypasses it.
"""

from __future__ import annotations

import pathlib
import re

import jax
import jax.numpy as jnp
import pytest
from yggdrax.geometry import compute_tree_geometry

from jaccpot import FastMultipoleMethod
from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def prepared():
    n = 2048
    key = jax.random.PRNGKey(n)
    k1, k2 = jax.random.split(key)
    pts = jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
    q = jax.random.uniform(k2, (n,), minval=0.1, maxval=1.0, dtype=jnp.float32)
    solver = FastMultipoleMethod(preset="fast", basis="real", theta=0.6)
    state = solver.prepare_state(pts, q, leaf_size=32, max_order=3, theta=0.6)
    return state.tree, state.positions_sorted


@pytest.mark.parametrize("max_leaf_size", (None, 32))
def test_compiled_dispatch_is_bit_identical(prepared, max_leaf_size) -> None:
    """Compiling a graph must not reassociate it."""

    tree, positions = prepared
    reference = compute_tree_geometry(tree, positions, max_leaf_size=max_leaf_size)
    compiled = compute_tree_geometry_compiled(
        tree, positions, max_leaf_size=max_leaf_size
    )
    for field in ("center", "half_extent", "radius", "max_extent"):
        got = jnp.asarray(getattr(compiled, field))
        want = jnp.asarray(getattr(reference, field))
        assert got.shape == want.shape
        assert bool(jnp.all(got == want)), f"{field} differs with leaf={max_leaf_size}"


def test_the_escape_hatch_returns_the_same_answer(prepared, monkeypatch) -> None:
    tree, positions = prepared
    monkeypatch.setenv("JACCPOT_TREE_GEOMETRY_JIT", "0")
    off = compute_tree_geometry_compiled(tree, positions, max_leaf_size=32)
    monkeypatch.setenv("JACCPOT_TREE_GEOMETRY_JIT", "1")
    on = compute_tree_geometry_compiled(tree, positions, max_leaf_size=32)
    assert bool(jnp.all(jnp.asarray(off.center) == jnp.asarray(on.center)))


def test_it_falls_back_under_an_outer_trace(prepared) -> None:
    """Nesting a jit inside a trace would only re-trace; it must not try.

    The check is that tracing through it works at all -- a ``jax.jit`` call on
    tracer arguments is legal but pointless, while a *cache* keyed on concrete
    values would be a leak. Compare against the uncompiled result to make sure
    the fallback is the same computation.
    """

    tree, positions = prepared

    @jax.jit
    def centers(pos):
        return compute_tree_geometry_compiled(tree, pos, max_leaf_size=32).center

    got = centers(positions)
    want = compute_tree_geometry(tree, positions, max_leaf_size=32).center
    assert jnp.allclose(got, want, rtol=0, atol=0)


def test_no_jaccpot_module_calls_the_uncompiled_function_directly() -> None:
    """The dispatch is only a fix if every call site uses it.

    A new call site importing ``compute_tree_geometry`` straight from yggdrax
    would silently reintroduce ~590 ms per step, and nothing else would notice.
    """

    offenders = []
    for path in (REPO_ROOT / "jaccpot").rglob("*.py"):
        if path.name == "tree_geometry.py":
            continue  # the dispatch itself
        text = path.read_text()
        if re.search(r"(?<![_\w])compute_tree_geometry\s*\(", text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "these call compute_tree_geometry directly instead of "
        f"compute_tree_geometry_compiled: {offenders}"
    )

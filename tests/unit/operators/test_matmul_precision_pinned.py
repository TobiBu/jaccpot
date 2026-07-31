"""fp32 matmuls in the expansion operators must not silently fall back to TF32.

XLA lowers fp32 matmuls on Ampere+ to TF32 (~10-bit mantissa) by default. The
rotation/translation algebra is built from small dense matmuls, and at TF32 that
capped M2L relative accuracy at ~6e-04 from order 4 up, *regardless of expansion
order* -- so raising the order bought nothing. See
``jaccpot/operators/_precision.py`` for the measurements.

Two invariants are enforced here, because the two matmul spellings need different
mechanisms:

* ``jnp.einsum`` / ``jnp.dot`` / ``jnp.matmul`` take a ``precision=`` argument, so
  every such call in the package must pass one.
* the ``@`` operator takes no ``precision``, so any function containing one must
  be decorated with ``@highest_matmul_precision``.

Both checks are AST-based and need no GPU.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[3] / "jaccpot"
DOT_CALLS = {"einsum", "dot", "matmul", "tensordot", "vdot", "inner"}
DECORATOR = "highest_matmul_precision"


def _python_files():
    assert PACKAGE.is_dir(), f"package not found at {PACKAGE}"
    return sorted(PACKAGE.rglob("*.py"))


def test_every_jnp_dot_call_sets_precision():
    """A ``jnp.einsum``/``dot``/``matmul`` without ``precision=`` gets TF32."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr in DOT_CALLS
                and getattr(func.value, "id", None) == "jnp"
            ):
                continue
            if not any(kw.arg == "precision" for kw in node.keywords):
                offenders.append(
                    f"{path.relative_to(PACKAGE.parent)}:{node.lineno} "
                    f"jnp.{func.attr}"
                )
    assert not offenders, (
        "these matmul calls would use TF32; pass "
        "precision=lax.Precision.HIGHEST (see jaccpot/operators/_precision.py):\n  "
        + "\n  ".join(offenders)
    )


def test_functions_using_the_matmul_operator_are_decorated():
    """``@`` takes no precision kwarg, so its enclosing function must be pinned."""
    offenders = []
    for path in _python_files():
        if path.name == "_precision.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # only count '@' matmuls in THIS function, not nested ones, so a
            # decorated outer function does not excuse an undecorated inner one
            nested = {
                inner
                for child in ast.walk(node)
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child is not node
                for inner in ast.walk(child)
            }
            has_matmul = any(
                isinstance(x, ast.BinOp)
                and isinstance(x.op, ast.MatMult)
                and x not in nested
                for x in ast.walk(node)
            )
            if not has_matmul:
                continue
            decorated = any(
                (isinstance(d, ast.Name) and d.id == DECORATOR)
                or (isinstance(d, ast.Attribute) and d.attr == DECORATOR)
                for d in node.decorator_list
            )
            if not decorated:
                offenders.append(
                    f"{path.relative_to(PACKAGE.parent)}:{node.lineno} {node.name}"
                )
    assert not offenders, (
        "these functions use the '@' matmul operator without pinning precision; "
        f"add @{DECORATOR} (see jaccpot/operators/_precision.py):\n  "
        + "\n  ".join(offenders)
    )


def test_decorator_actually_pins_precision():
    """Guard the mechanism itself, not just its application."""
    jax = pytest.importorskip("jax")
    from jaccpot.operators._precision import highest_matmul_precision

    seen = {}

    @highest_matmul_precision
    def probe():
        # a flag with a contextmanager must be read as an attribute, not via read()
        seen["inside"] = jax.config.jax_default_matmul_precision
        return 0

    outside_before = jax.config.jax_default_matmul_precision
    probe()
    assert (
        str(seen["inside"]).lower().endswith("highest")
    ), f"decorator did not pin precision (saw {seen['inside']!r})"
    assert (
        jax.config.jax_default_matmul_precision == outside_before
    ), "decorator leaked the precision setting outside the call"

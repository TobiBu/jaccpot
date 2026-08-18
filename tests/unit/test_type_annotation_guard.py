"""Guardrails for runtime type-check coverage.

This test enforces that non-private callables in the jaccpot package keep
explicit parameter and return annotations so jaxtyping+beartype can validate
contracts at runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "jaccpot"


def _is_fully_annotated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    args = node.args
    all_args = args.posonlyargs + args.args + args.kwonlyargs

    args_ok = all(arg.annotation is not None for arg in all_args)
    if args.vararg is not None:
        args_ok = args_ok and args.vararg.annotation is not None
    if args.kwarg is not None:
        args_ok = args_ok and args.kwarg.annotation is not None

    return_ok = node.returns is not None
    return args_ok and return_ok


def _iter_missing_annotations() -> list[tuple[str, int, str, bool, bool]]:
    missing: list[tuple[str, int, str, bool, bool]] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue

        tree = ast.parse(path.read_text())

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue

            args = node.args
            all_args = args.posonlyargs + args.args + args.kwonlyargs
            args_ok = all(arg.annotation is not None for arg in all_args)
            if args.vararg is not None:
                args_ok = args_ok and args.vararg.annotation is not None
            if args.kwarg is not None:
                args_ok = args_ok and args.kwarg.annotation is not None
            ret_ok = node.returns is not None

            if not _is_fully_annotated(node):
                missing.append(
                    (
                        str(path.relative_to(PROJECT_ROOT)),
                        node.lineno,
                        node.name,
                        args_ok,
                        ret_ok,
                    )
                )

    return missing


def test_all_non_private_callables_are_fully_annotated() -> None:
    missing = _iter_missing_annotations()
    if not missing:
        return

    details = "\n".join(
        f"- {path}:{lineno} `{name}` (args={args_ok}, return={ret_ok})"
        for path, lineno, name, args_ok, ret_ok in missing
    )
    raise AssertionError(
        "Found callables with incomplete type annotations. "
        "Add parameter and return annotations so runtime type-checking remains comprehensive.\n"
        f"{details}"
    )


# The public facade: the modules a user's own code imports from. Shape
# annotations start here because this is where the shape is least guessable from
# surrounding context, and where getting it wrong is most expensive to debug.
# See STYLE_GUIDE.md section 4 for the axis vocabulary.
FACADE_MODULES = (
    "jaccpot/solver.py",
    "jaccpot/autodiff.py",
    "jaccpot/odisseo.py",
    "jaccpot/nornax_adapter.py",
)

# Parameters carrying particle-indexed arrays, which therefore have a knowable
# shape. Deliberately a name list rather than "every Array parameter": several
# facade parameters are genuinely shape-agnostic (`compile_now` is a sample tuple,
# `carry`/`s` are `lax.scan` slices), and asserting a shape on those would be a
# lie the runtime typechecker would then enforce.
SHAPED_FACADE_PARAMS = frozenset(
    {
        "positions",
        "positions_sorted",
        "velocities",
        "initial_self_acceleration",
        "masses",
        "masses_sorted",
        "state",
        "target_indices",
        "active_indices",
        "rung",
        "level_weights",
        "bounds",
    }
)

_SHAPE_MARKERS = ("Float[", "Int[", "Bool[", "Shaped[", "Complex[", "Num[")


def _iter_unshaped_facade_params() -> list[tuple[str, int, str, str, str]]:
    """Find facade parameters that should carry a shape but do not.

    Returns
    -------
    list[tuple[str, int, str, str, str]]
        One ``(path, lineno, function, parameter, annotation)`` per offender.
    """
    offenders: list[tuple[str, int, str, str, str]] = []

    for rel in FACADE_MODULES:
        path = PROJECT_ROOT / rel
        tree = ast.parse(path.read_text())

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue

            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                if arg.arg not in SHAPED_FACADE_PARAMS or arg.annotation is None:
                    continue
                text = ast.unparse(arg.annotation)
                if "Array" not in text:
                    continue
                if any(marker in text for marker in _SHAPE_MARKERS):
                    continue
                offenders.append((rel, arg.lineno, node.name, arg.arg, text))

    return offenders


def test_public_facade_array_params_carry_shapes() -> None:
    """Facade array parameters must be shaped, not bare ``Array``.

    ``jaxtyping.Array`` is an alias for ``jax.Array``, so a bare annotation
    asserts only "is an array" -- which is why the audit (E.1) called the
    package's jaxtyping usage inert. This keeps the facade from regressing to it.
    """
    offenders = _iter_unshaped_facade_params()
    details = "\n".join(
        f"- {path}:{lineno} `{func}` parameter `{param}` is `{text}`"
        for path, lineno, func, param, text in offenders
    )
    assert not offenders, (
        "Public facade array parameters must carry a jaxtyping shape "
        "(see STYLE_GUIDE.md section 4 for the axis vocabulary):\n" + details
    )

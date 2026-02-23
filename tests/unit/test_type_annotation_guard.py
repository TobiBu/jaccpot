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

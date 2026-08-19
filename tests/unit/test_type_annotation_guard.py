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

# Modules converted by the E.3 shape programme, in the order they landed. Listing
# a module here says "its unvalidated array parameters carry shapes, and must keep
# them" -- not "every array parameter in it is shaped". Some are deliberately bare
# with the reason recorded at the site: `explicit_centers` in tree_expansions would
# preempt a documented `ValueError` and make that branch unreachable for shape
# errors, so it is left alone (and it is outside the families below anyway).
#
# Add a module here in the same PR that annotates it. See STYLE_GUIDE.md section 4.
CONVERTED_MODULES = (
    "jaccpot/upward/tree_expansions.py",
    "jaccpot/nearfield/near_field.py",
    "jaccpot/runtime/fmm_evaluate.py",
)

# The parameter families the E.3 pilots measured as validated by NOTHING else, and
# therefore the ones worth pinning. Before the pilots, a wrong dtype, wrong rank or
# mismatched length in any `precomputed_*` argument was accepted silently and
# reached the kernels.
#
# `farfield_*` is deliberately NOT here: pilot 3 measured it as already rejected
# with a domain `ValueError`, so requiring a shape would assert a guarantee the
# package makes elsewhere and better. The families are the unit of this policy,
# not the modules.
UNVALIDATED_PARAM_PREFIXES = ("precomputed_",)
UNVALIDATED_PARAM_SUFFIXES = ("_override",)

# Array parameters in those families that are deliberately bare. EMPTY as measured
# 2026-08-19, and that is the intended steady state: the three bare parameters
# matching these names (`precomputed_geometry`, and `nearfield_mode_override` twice)
# are not arrays at all -- `Optional[TreeGeometry]` and `Optional[str]` -- so they
# never reach the shape check. Kept so a future exemption has somewhere to go that
# is checked rather than assumed.
DELIBERATELY_BARE: frozenset[tuple[str, str]] = frozenset()

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


def _is_unvalidated_family(name: str) -> bool:
    """Whether a parameter belongs to a family nothing else validates.

    Parameters
    ----------
    name : str
        Parameter name.

    Returns
    -------
    bool
        ``True`` for the ``precomputed_*`` and ``*_override`` families that the
        E.3 pilots measured as unchecked before annotation.
    """
    return name.startswith(UNVALIDATED_PARAM_PREFIXES) or name.endswith(
        UNVALIDATED_PARAM_SUFFIXES
    )


def _iter_unshaped_unvalidated_params() -> list[tuple[str, int, str, str, str]]:
    """Find unvalidated-family array parameters that lost their shape.

    Only ``@jaxtyped``-decorated functions are examined, because those are the
    ones whose annotations are enforced on every call. An undecorated function's
    annotation is inert outside ``JACCPOT_RUNTIME_TYPECHECK=1``, so requiring one
    there would assert a guarantee the package does not actually make.

    Returns
    -------
    list[tuple[str, int, str, str, str]]
        One ``(path, lineno, function, parameter, annotation)`` per offender.
    """
    offenders: list[tuple[str, int, str, str, str]] = []

    for rel in CONVERTED_MODULES:
        tree = ast.parse((PROJECT_ROOT / rel).read_text())

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorated = any(
                "jaxtyped" in ast.unparse(dec) for dec in node.decorator_list
            )
            if not decorated:
                continue

            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                if not _is_unvalidated_family(arg.arg):
                    continue
                if (rel, arg.arg) in DELIBERATELY_BARE:
                    continue
                if arg.annotation is None:
                    continue
                text = ast.unparse(arg.annotation)
                if "Array" not in text:
                    continue
                if any(marker in text for marker in _SHAPE_MARKERS):
                    continue
                offenders.append((rel, arg.lineno, node.name, arg.arg, text))

    return offenders


def test_converted_modules_keep_shapes_on_unvalidated_params() -> None:
    """The ``precomputed_*`` / ``*_override`` families must stay shaped.

    These are the parameters the E.3 pilots measured as validated by nothing else:
    before annotation, a wrong dtype, wrong rank or mismatched length in any of
    them was accepted silently and reached the kernels. Losing the annotation
    restores that hole without any test going red, which is why this guard exists.
    """
    offenders = _iter_unshaped_unvalidated_params()
    details = "\n".join(
        f"- {path}:{lineno} `{func}` parameter `{param}` is `{text}`"
        for path, lineno, func, param, text in offenders
    )
    assert not offenders, (
        "Unvalidated-family parameters must carry a jaxtyping shape "
        "(STYLE_GUIDE.md section 4). Nothing else checks these:\n" + details
    )


def test_deliberately_bare_list_has_not_rotted() -> None:
    """Every ``DELIBERATELY_BARE`` entry must still exist and still be bare.

    Asserted in both directions so the exemption list cannot become a place
    where entries accumulate unexamined: an entry naming a parameter that no
    longer exists, or one that has since been shaped, fails here.
    """
    stale: list[str] = []
    for rel, param in sorted(DELIBERATELY_BARE):
        tree = ast.parse((PROJECT_ROOT / rel).read_text())
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                if arg.arg != param or arg.annotation is None:
                    continue
                found = True
                text = ast.unparse(arg.annotation)
                if any(marker in text for marker in _SHAPE_MARKERS):
                    stale.append(
                        f"- {rel} `{param}` is shaped now (`{text}`); remove the exemption"
                    )
        if not found:
            stale.append(f"- {rel} has no parameter `{param}`; remove the exemption")

    assert not stale, "\n".join(["DELIBERATELY_BARE has rotted:", *stale])

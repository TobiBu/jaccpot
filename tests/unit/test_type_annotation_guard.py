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


def _annotation_status(node: ast.FunctionDef | ast.AsyncFunctionDef):
    """Return ``(args_ok, return_ok)`` for one function definition.

    A leading ``self`` or ``cls`` is exempt. Python supplies it implicitly, so
    beartype has never validated it -- and for the ``runtime/`` mixins annotating
    it is not merely redundant but *wrong*: ``self: "FMMEngine"`` declares a
    subtype of the mixin as its own ``self``, which a type checker rejects (119
    errors, measured; see the audit's E.2). Those mixins reach the engine's
    attributes by inheriting ``_EngineBase`` instead, which is guarded by
    ``test_mixin_engine_base_guard.py``.

    This exemption is not a relaxation of what the file's docstring asks for.
    The stated purpose is that "jaxtyping+beartype can validate contracts at
    runtime", and an annotation on ``self`` bought none of that: it was an
    unresolvable forward reference under ``TYPE_CHECKING``, which beartype skips.
    The requirement was inert for its own purpose before it was removed.

    Parameters
    ----------
    node
        The function definition to inspect.

    Returns
    -------
    tuple of (bool, bool)
        Whether every non-implicit parameter is annotated, and whether the
        return is annotated.
    """
    args = node.args
    positional = args.posonlyargs + args.args
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]

    args_ok = all(arg.annotation is not None for arg in positional + args.kwonlyargs)
    if args.vararg is not None:
        args_ok = args_ok and args.vararg.annotation is not None
    if args.kwarg is not None:
        args_ok = args_ok and args.kwarg.annotation is not None

    return args_ok, node.returns is not None


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

            args_ok, ret_ok = _annotation_status(node)
            if not (args_ok and ret_ok):
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
    # Added late, and the delay is the point: this module was annotated and given
    # its `@jaxtyped` decorator without being listed here, so for two commits the
    # one module whose shapes are the ONLY check on its inputs was the one module
    # this guard did not hold. STYLE_GUIDE section 4 says to add a module in the
    # same PR that annotates it; that is the rule this omission broke.
    "jaccpot/runtime/kernels/_evaluate.py",
    # Phase 1 of the rollout. Only the `delta`/`raw_*` family was unvalidated
    # here; the `upward/solidfmm_complex_*` functions in the same phase were
    # measured and left alone, because yggdrax and their own bodies already
    # reject every malformed input tried (13 of 15 across the group).
    "jaccpot/downward/local_expansions.py",
    # Phase 1, `runtime/` half. Only the reference oracle and its sweeps facade
    # were unvalidated: `fmm_derivatives.py`, `fmm_prepare.py` and `_fmm_impl.py`
    # were measured and left alone, because each already rejects every malformed
    # input tried, with a domain message an annotation would replace.
    "jaccpot/runtime/reference.py",
    "jaccpot/runtime/fmm_sweeps.py",
    # Phase 2, first change. NOT a module conversion: only the eleven functions
    # taking a single spatial `delta` (and `direction`) are shaped, because those
    # are the ones `bench/annotation_pilot` measured accepting a length-2 vector
    # silently. The rest of this 2400-line module is still bare and is its own
    # work -- listing it here says the `"3"` family must not regress, not that
    # the module is finished.
    "jaccpot/operators/complex_ops.py",
    # Phase 2, second change, and the FIRST enforced annotations anywhere in
    # `jaccpot/pallas/`. Not a module conversion: the five module-level entry
    # points carry shapes, the three tile helpers below them do not, and the split
    # is on whether the check would execute inside a `pallas_call` body. The
    # helpers hold 33 of the module's 99 measured silent acceptances and are the
    # bigger prize; they wait for a GPU run, because interpret mode is not the
    # Triton lowering and there is no GPU leg in CI. Reason recorded at the site.
    "jaccpot/pallas/nearfield_mutual.py",
)

# Array parameters deliberately left bare, each with its reason recorded at the
# site. The rot check below asserts every entry still exists AND is still bare, so
# this cannot become a place where exemptions accumulate unexamined.
DELIBERATELY_BARE: frozenset[tuple[str, str]] = frozenset(
    {
        # Annotating the shape would raise a beartype violation BEFORE the body,
        # replacing the `ValueError` both functions document in their Raises
        # section and making that branch unreachable for shape errors. Changing
        # which exception a caller sees is a behaviour change, not a docs change.
        ("jaccpot/upward/tree_expansions.py", "explicit_centers"),
        # `Union[float, Array]`: a scalar that legitimately accepts a Python
        # number, so `Float[Array, ""]` would break every defaulted call.
        ("jaccpot/nearfield/near_field.py", "G"),
        # `initialize_local_expansions` already raises
        # `ValueError("centers must have shape (num_nodes, 3)")`, and it fires on
        # both a wrong node count and a wrong trailing dimension -- measured, not
        # read off the docstring. Annotating would preempt that message with a
        # generic `TypeCheckError`, which STYLE_GUIDE section 4.1 calls out as a
        # loss rather than a gain.
        ("jaccpot/downward/local_expansions.py", "centers"),
        # Same `Union[float, Array]` reason as `near_field.py`'s `G` above, and
        # here it is load-bearing for BOTH names: every one of these signatures
        # defaults them to Python floats (`G=1.0`, `softening=0.0`), so
        # `Float[Array, ""]` would reject the default path itself.
        ("jaccpot/runtime/reference.py", "G"),
        ("jaccpot/runtime/reference.py", "softening"),
        # Scalars, and the third instance of the same reason: `Float[Array, ""]`
        # buys nothing a scalar can get wrong, and both are reshaped to `(1,)` by
        # the Pallas wrappers anyway, so the only shape a caller could pass that
        # the annotation would catch is one the next line would catch too.
        ("jaccpot/pallas/nearfield_mutual.py", "softening_sq"),
        ("jaccpot/pallas/nearfield_mutual.py", "g_value"),
    }
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

# Every jaxtyping dtype FAMILY, not the six that happened to be in use when this
# guard was written. The short list was a false-negative generator in one
# direction and a false positive in the other: `Inexact[Array, '_']` -- the right
# annotation for a coefficient buffer that legitimately arrives real or complex --
# was reported as unshaped, so the guard would have rejected a correct annotation
# and pushed the author toward a wrong one.
#
# Family names only, deliberately: STYLE_GUIDE section 4.4 forbids the
# width-suffixed spellings (`Int32`, `Float64`), so `Int32[Array, "n"]` still
# reads as unshaped here and fails. That is the intended behaviour, not an
# oversight -- the same set `bench/annotation_census.py` classifies on.
_SHAPE_MARKERS = (
    "Bool[",
    "Complex[",
    "Float[",
    "Inexact[",
    "Int[",
    "Key[",
    "Num[",
    "Real[",
    "Shaped[",
    "UInt[",
)


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


def _iter_unshaped_converted_params() -> list[tuple[str, int, str, str, str]]:
    """Find array parameters in converted modules that lost their shape.

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


def test_converted_modules_keep_shapes_on_array_params() -> None:
    """Every array parameter in the converted modules must stay shaped.

    These are the parameters the E.3 pilots measured as validated by nothing else:
    before annotation, a wrong dtype, wrong rank or mismatched length in any of
    them was accepted silently and reached the kernels. Losing the annotation
    restores that hole without any test going red, which is why this guard exists.
    """
    offenders = _iter_unshaped_converted_params()
    details = "\n".join(
        f"- {path}:{lineno} `{func}` parameter `{param}` is `{text}`"
        for path, lineno, func, param, text in offenders
    )
    assert not offenders, (
        "Array parameters in converted modules must carry a jaxtyping shape "
        "(STYLE_GUIDE.md section 4). Nothing else checks these:\n" + details
    )


def test_shape_markers_accept_families_and_still_reject_widths() -> None:
    """Pin both edges of ``_SHAPE_MARKERS``, because it was widened.

    Widening a guard's accept-list is where a silent hole gets introduced, so the
    two directions are asserted separately: every jaxtyping dtype family must read
    as shaped, and a bare ``Array`` or a width-suffixed spelling must not. The
    width case is the one that matters -- ``Int32[Array, 'n']`` violates
    STYLE_GUIDE section 4.4 and has to keep failing this guard, even though it
    looks shaped and would satisfy a naive "contains a bracket" test.
    """
    families = (
        "Bool",
        "Complex",
        "Float",
        "Inexact",
        "Int",
        "Key",
        "Num",
        "Real",
        "Shaped",
        "UInt",
    )
    for family in families:
        text = f'{family}[Array, "n 3"]'
        assert any(
            marker in text for marker in _SHAPE_MARKERS
        ), f"{family} is a jaxtyping dtype family and must read as shaped"

    for text in (
        "Array",
        "Optional[Array]",
        "Int32[Array, 'n']",
        "Float64[Array, '3']",
    ):
        assert not any(
            marker in text for marker in _SHAPE_MARKERS
        ), f"`{text}` must not count as a shaped annotation"


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

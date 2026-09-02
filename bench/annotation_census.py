"""The annotation burn-down counter: one definition, so the number stops moving.

WHY THIS EXISTS
---------------
The jaxtyping rollout (STYLE_GUIDE section 4, audit E.1/E.3/F20) is a burn-down, and a
burn-down needs a counter that two people can run and agree on. It did not have one.
E.1's table has been restated by hand and gone stale twice, and F20's row and an AST
walk over the same tree disagreed by ~70% -- 3174 bare against 1855.

Neither number was wrong. They were answers to different questions, and nobody had
written the question down. This module writes it down and answers it, so the audit can
quote a command instead of a remembered figure.

WHAT IT COUNTS, EXACTLY
-----------------------
The unit is a **function parameter**, not an occurrence of the word ``Array``:

* One parameter counts once, however many ``Array``s its annotation mentions.
  ``tuple[Array, Array]`` is one bare parameter, not two.
* A leading ``self``/``cls`` is exempt. Python supplies it, beartype never sees it --
  the same exemption ``tests/unit/test_type_annotation_guard.py`` makes, and for the
  same reason.
* **Return annotations are excluded.** Not an oversight: STYLE_GUIDE section 4.4 records
  that shaped returns are effectively unavailable here -- pydoclint 0.9.1 crashes on a
  multi-token axis spec, and an optional ``target_indices`` makes the leading axis
  caller-dependent in a way jaxtyping cannot express. Counting a population that policy
  forbids anyone to convert would put a floor under the burn-down that no PR can lift.
* ``jaccpot/experimental/`` is excluded, consistent with every other measurement in the
  audit -- it is prototype code, deselected by the default test run and omitted from
  coverage. ``--include-experimental`` overrides it.

Each parameter lands in exactly one bucket:

``shaped``
    Annotation carries a jaxtyping shape spec: a dtype family subscripted with a string
    axis spec, e.g. ``Float[Array, "n 3"]``. This is the goal state.
``bare``
    Annotation mentions ``Array`` but no shape spec. This is the burn-down population.
``unannotated``
    No annotation at all.
``other``
    Annotated, but not with an array type (``int``, ``TreeConfig``, ...). Reported for
    completeness so the four buckets sum to the parameter count and a reader can see
    nothing was dropped.

RECONCILIATION, AND WHAT IT TURNED UP
-------------------------------------
``--reconcile`` prints this definition beside the looser ones, because the gap is the
part that caused the confusion. Each rung loosens exactly one choice. Measured on
``main`` at 2026-08-29:

    bare Array parameters (this module's definition)                 1855
      + occurrences, so tuple[Array, Array] counts twice             1935
      + return annotations                                           2887
      + jaccpot/experimental/                                        2955
      + the Array inside a shaped annotation                         3169
    F20's reported figure                                            3174

    unannotated parameters (this module's definition)                 217
      + implicit self/cls                                             371
      + jaccpot/experimental/                                         411
    the plan's reported figure                                        368

So both figures are close to reproducible, and the plan's suspected causes -- returns,
nesting, ``experimental/`` -- are real but are not the whole story. **Two of the rungs
are worth more than the arithmetic:**

**F20's "bare" count includes the shaped ones.** ``Float[Array, "n 3"]`` contains the
token ``Array``, so a scan counting mentions counts every converted annotation in the
unconverted column too. The last rung is worth 214 of the gap, and it means the row's
two halves are not disjoint: converting a parameter moves it *within* one number rather
than between two, so that number cannot show the burn-down at all.

**42% of the "368 unannotated" is ``self``/``cls``** -- 154 implicit parameters that
must **never** be annotated. Audit E.2 measured 119 type-checker errors from doing
exactly that on the ``runtime/`` mixins, and
``tests/unit/test_type_annotation_guard.py`` exempts them for the same reason. A
backlog figure that counts work nobody may do overstates the remaining work by nearly
half, and no PR can ever reduce it.

**The shaped half does not reconcile by scope at all, and that is the sharpest find.**
No AST definition reaches F20's ``381 shaped``: the widest -- occurrences over
parameters and returns including ``experimental/`` -- gives **214**. A text scan does
reach it. ``grep -Eo '(Float|Int|...)\\[Array' jaccpot/`` returns **423**, almost exactly
twice 214, because STYLE_GUIDE section 4.5 requires the docstring parameter type to
mirror the annotation verbatim -- so every shaped annotation appears in the source
**twice**, once in the signature and once in the ``Parameters`` block. At the 179 shaped
parameters standing on 2026-08-27, when F20 was re-measured, twice is 358.

The two halves of that row were therefore taken by different methods, one structural and
one textual, which is why no single correction reconciles both. It is the same lesson as
"count failures, not grep hits": a text scan counts what the file says, and pydoclint
requires the file to say it twice.

**What is not reproduced, stated because it is not:** the ladder lands 5 short of 3174 on
today's tree and 47 short at ``24d8f2e``, the commit F20 was measured at. The residual is
small and its direction is not stable, so it is not commit drift alone. The method behind
that figure was never written down, which is the whole argument for this file: an
unrecorded measurement cannot be reproduced, only approximated.

USAGE
-----
    python -m bench.annotation_census                 # headline + per-module table
    python -m bench.annotation_census --reconcile     # the table above, recomputed
    python -m bench.annotation_census --json          # for a script or a diff

No jax import, no yggdrax import, no device -- an AST walk over the source tree, 0.8 s,
which is what makes it cheap enough to quote in a PR. ``--reconcile`` re-walks the tree
once per rung and costs ~9 s; it is a diagnostic for when two people disagree about a
number, not something to put in a loop.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ModuleCensus",
    "PACKAGE_ROOT",
    "Totals",
    "census_package",
    "classify_parameter",
    "main",
    "reconciliation",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = _REPO_ROOT / "jaccpot"

# The jaxtyping dtype families. Deliberately the FAMILY names only: STYLE_GUIDE
# section 4.4 forbids the width-suffixed spellings (`Int32`, `Float64`) because
# INDEX_DTYPE is selectable via JACCPOT_INDEX_PRECISION and one call was observed
# carrying int32 and int64 index arrays at the same time. A width-suffixed
# annotation is a defect, so counting it as progress would be wrong; it should be
# caught in review, and `--strict-families` reports any that slipped through.
_DTYPE_FAMILIES = frozenset(
    {
        "Bool",
        "Complex",
        "Inexact",
        "Int",
        "Float",
        "Key",
        "Num",
        "Real",
        "Shaped",
        "UInt",
    }
)

_WIDTH_SUFFIXED = frozenset(
    {
        "BFloat16",
        "Complex64",
        "Complex128",
        "Float16",
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
    }
)

SHAPED = "shaped"
BARE = "bare"
UNANNOTATED = "unannotated"
OTHER = "other"


def _subscript_base(node: ast.Subscript) -> str | None:
    """Return the dotted-name tail of a subscript's base, if it has one.

    Handles both ``Float[...]`` and ``jaxtyping.Float[...]``.

    Parameters
    ----------
    node : ast.Subscript
        The subscript expression to inspect.

    Returns
    -------
    str or None
        The base name (``"Float"`` for either spelling above), or None when the
        base is neither a name nor an attribute access.
    """
    base = node.value
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _has_shape_spec(annotation: ast.expr, families: frozenset[str]) -> bool:
    """Report whether an annotation carries a jaxtyping shape specification.

    Walks the whole annotation rather than matching its outermost node, so that
    ``Optional[Float[Array, 'n 3']]`` and ``tuple[Float[Array, 'n 3'], ...]``
    both count.

    Parameters
    ----------
    annotation : ast.expr
        The annotation expression.
    families : frozenset of str
        The dtype family names to accept as jaxtyping shape carriers.

    Returns
    -------
    bool
        True when some dtype family is subscripted with a string axis spec.
    """
    for node in ast.walk(annotation):
        if not isinstance(node, ast.Subscript):
            continue
        if _subscript_base(node) not in families:
            continue
        index = node.slice
        elements = index.elts if isinstance(index, ast.Tuple) else [index]
        # The axis spec is the string literal: `Float[Array, "n 3"]`. A dtype
        # family subscripted with no string at all -- `Float[Array]` -- asserts a
        # dtype and no shape, so it does not count as shaped.
        if any(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements
        ):
            return True
    return False


def _mentions_array(annotation: ast.expr) -> bool:
    """Report whether an annotation mentions an array type anywhere inside it.

    Parameters
    ----------
    annotation : ast.expr
        The annotation expression.

    Returns
    -------
    bool
        True when the name ``Array`` appears, plain or dotted (``jax.Array``).
    """
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id == "Array":
            return True
        if isinstance(node, ast.Attribute) and node.attr == "Array":
            return True
    return False


def classify_parameter(
    annotation: ast.expr | None, families: frozenset[str] = _DTYPE_FAMILIES
) -> str:
    """Put one parameter in exactly one bucket.

    Parameters
    ----------
    annotation : ast.expr or None
        The parameter's annotation, or None when it has none.
    families : frozenset of str, optional
        The dtype family names to accept as jaxtyping shape carriers. Defaults to
        the family names STYLE_GUIDE section 4.4 permits.

    Returns
    -------
    str
        One of ``"shaped"``, ``"bare"``, ``"unannotated"`` or ``"other"``.
    """
    if annotation is None:
        return UNANNOTATED
    if not _mentions_array(annotation):
        return OTHER
    return SHAPED if _has_shape_spec(annotation, families) else BARE


def _explicit_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Return the parameters beartype can actually see.

    A leading ``self``/``cls`` is dropped: Python supplies it implicitly, so it has
    never been validated, and for the ``runtime/`` mixins annotating it is
    positively wrong (audit E.2). ``*args``/``**kwargs`` are included -- they carry
    annotations and beartype checks them.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function definition to inspect.

    Returns
    -------
    list of ast.arg
        The parameters that count towards the census.
    """
    args = node.args
    positional = args.posonlyargs + args.args
    if positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
    variadic = [a for a in (args.vararg, args.kwarg) if a is not None]
    return positional + args.kwonlyargs + variadic


def _is_jaxtyped(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a function carries a ``@jaxtyped`` decorator.

    Matched on the source text of the decorator rather than on its structure,
    because the call is spelled ``@jaxtyped(typechecker=beartype)`` in this
    package but the bare ``@jaxtyped`` form is legal and should still count.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The function definition to inspect.

    Returns
    -------
    bool
        True when some decorator names ``jaxtyped``.
    """
    return any("jaxtyped" in ast.unparse(d) for d in node.decorator_list)


@dataclass
class ModuleCensus:
    """The census of one module.

    Attributes
    ----------
    path : str
        Module path relative to ``jaccpot/``.
    shaped : int
        Parameters carrying a jaxtyping shape spec.
    bare : int
        Parameters annotated with an array type but no shape.
    unannotated : int
        Parameters with no annotation.
    other : int
        Parameters annotated with a non-array type.
    decorated : int
        Functions carrying ``@jaxtyped``.
    vacuous_decorated : int
        Functions carrying ``@jaxtyped`` with no shaped parameter -- so the
        decorator checks nothing about shape. Phase 1's burn-down population.
    jaxtyping_imports : int
        ``from jaxtyping import ...`` statements.
    width_suffixed : list of str
        Parameter names annotated with a width-suffixed dtype
        (``Float32[Array, ...]``), which section 4.4 forbids.
    """

    path: str
    shaped: int = 0
    bare: int = 0
    unannotated: int = 0
    other: int = 0
    decorated: int = 0
    vacuous_decorated: int = 0
    jaxtyping_imports: int = 0
    width_suffixed: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Give each instance its own list rather than sharing a default."""
        if self.width_suffixed is None:
            self.width_suffixed = []

    @property
    def array_parameters(self) -> int:
        """int: Parameters annotated with an array type, shaped or not."""
        return self.shaped + self.bare


@dataclass
class Totals:
    """Package-level headline numbers -- the four the plan tracks, plus context.

    Attributes
    ----------
    modules : int
        Modules walked.
    shaped : int
        Shape-annotated array parameters.
    bare : int
        Bare ``Array`` parameters -- the burn-down population.
    unannotated : int
        Parameters with no annotation.
    other : int
        Parameters annotated with a non-array type.
    decorated : int
        Functions carrying ``@jaxtyped``.
    vacuous_decorated : int
        ``@jaxtyped`` functions with no shaped parameter.
    jaxtyping_imports : int
        ``from jaxtyping import ...`` statements.
    """

    modules: int = 0
    shaped: int = 0
    bare: int = 0
    unannotated: int = 0
    other: int = 0
    decorated: int = 0
    vacuous_decorated: int = 0
    jaxtyping_imports: int = 0

    @property
    def shaped_fraction(self) -> float:
        """float: Shaped share of all array parameters, 0.0 when there are none."""
        denominator = self.shaped + self.bare
        return self.shaped / denominator if denominator else 0.0


def _iter_module_paths(package_root: Path, include_experimental: bool) -> list[Path]:
    """List the modules the census walks.

    Parameters
    ----------
    package_root : Path
        The ``jaccpot/`` package directory.
    include_experimental : bool
        Whether to include ``jaccpot/experimental/``.

    Returns
    -------
    list of Path
        Sorted module paths.

    Raises
    ------
    FileNotFoundError
        If the root contains no Python modules at all.
    """
    paths = sorted(package_root.rglob("*.py"))
    if not paths:
        # A counter that reports "0 bare, 0 shaped" because it walked the wrong
        # directory reads exactly like a finished burn-down. This tool exists to
        # stop a number being taken on trust, so it must not produce the most
        # flattering possible number by accident.
        raise FileNotFoundError(
            f"no Python modules under {package_root} -- wrong package root? "
            "Refusing to report a zero census that would read as a finished "
            "burn-down."
        )
    if include_experimental:
        return paths
    return [p for p in paths if "experimental" not in p.relative_to(package_root).parts]


def census_module(
    path: Path,
    package_root: Path = PACKAGE_ROOT,
    families: frozenset[str] = _DTYPE_FAMILIES,
) -> ModuleCensus:
    """Walk one module and count its parameters.

    Parameters
    ----------
    path : Path
        The module to walk.
    package_root : Path, optional
        Root the reported path is made relative to.
    families : frozenset of str, optional
        The dtype family names to accept as jaxtyping shape carriers.

    Returns
    -------
    ModuleCensus
        The module's counts.
    """
    result = ModuleCensus(path=str(path.relative_to(package_root)))
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "jaxtyping":
            result.jaxtyping_imports += 1
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        parameters = _explicit_parameters(node)
        any_shaped = False
        for parameter in parameters:
            kind = classify_parameter(parameter.annotation, families)
            setattr(result, kind, getattr(result, kind) + 1)
            any_shaped = any_shaped or kind == SHAPED
            if parameter.annotation is not None and _has_shape_spec(
                parameter.annotation, _WIDTH_SUFFIXED
            ):
                result.width_suffixed.append(f"{node.name}({parameter.arg})")

        if _is_jaxtyped(node):
            result.decorated += 1
            if not any_shaped:
                result.vacuous_decorated += 1

    return result


def census_package(
    package_root: Path = PACKAGE_ROOT, include_experimental: bool = False
) -> tuple[list[ModuleCensus], Totals]:
    """Walk the package and return per-module counts and the totals.

    Parameters
    ----------
    package_root : Path, optional
        The ``jaccpot/`` package directory.
    include_experimental : bool, optional
        Whether to include ``jaccpot/experimental/``. Defaults to False, matching
        every other measurement in the audit.

    Returns
    -------
    tuple of (list of ModuleCensus, Totals)
        Per-module counts sorted by bare count descending, and the package totals.

    Raises
    ------
    FileNotFoundError
        If ``package_root`` contains no Python modules at all.
    """
    modules = [
        census_module(path, package_root)
        for path in _iter_module_paths(package_root, include_experimental)
    ]
    totals = Totals(modules=len(modules))
    for module in modules:
        totals.shaped += module.shaped
        totals.bare += module.bare
        totals.unannotated += module.unannotated
        totals.other += module.other
        totals.decorated += module.decorated
        totals.vacuous_decorated += module.vacuous_decorated
        totals.jaxtyping_imports += module.jaxtyping_imports
    modules.sort(key=lambda m: (-m.bare, -m.unannotated, m.path))
    return modules, totals


def _bare_ladder(package_root: Path) -> dict[str, int]:
    """Recompute the bare-``Array`` count under progressively looser definitions.

    Parameters
    ----------
    package_root : Path
        The ``jaccpot/`` package directory.

    Returns
    -------
    dict of str to int
        Rung label to count, in ladder order.
    """

    def _count(
        *,
        occurrences: bool,
        returns: bool,
        experimental: bool,
        count_inside_shaped: bool = False,
    ) -> int:
        total = 0
        for path in _iter_module_paths(package_root, experimental):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                annotations = [p.annotation for p in _explicit_parameters(node)]
                if returns:
                    annotations.append(node.returns)
                for annotation in annotations:
                    if annotation is None:
                        continue
                    if not count_inside_shaped and _has_shape_spec(
                        annotation, _DTYPE_FAMILIES
                    ):
                        continue
                    if not _mentions_array(annotation):
                        continue
                    total += _array_occurrences(annotation) if occurrences else 1
        return total

    rungs: dict[str, int] = {}
    rungs["bare Array parameters (this module's definition)"] = _count(
        occurrences=False, returns=False, experimental=False
    )
    rungs["  + occurrences, so tuple[Array, Array] counts twice"] = _count(
        occurrences=True, returns=False, experimental=False
    )
    rungs["  + return annotations"] = _count(
        occurrences=True, returns=True, experimental=False
    )
    rungs["  + jaccpot/experimental/"] = _count(
        occurrences=True, returns=True, experimental=True
    )
    # The last rung is not a loosening of scope but of MEANING: `Float[Array, "n 3"]`
    # contains the token `Array`, so a scan that counts mentions counts every shaped
    # annotation as bare as well. That is why the two halves of F20's row cannot both
    # be right at once -- the same annotation is in both of its numbers.
    rungs["  + the Array inside a shaped annotation ('bare' now means 'mentions')"] = (
        _count(
            occurrences=True,
            returns=True,
            experimental=True,
            count_inside_shaped=True,
        )
    )
    return rungs


def _unannotated_ladder(package_root: Path) -> dict[str, int]:
    """Recompute the unannotated count with and without the implicit parameters.

    Parameters
    ----------
    package_root : Path
        The ``jaccpot/`` package directory.

    Returns
    -------
    dict of str to int
        Rung label to count, in ladder order.
    """

    def _count(*, implicit: bool, experimental: bool) -> int:
        total = 0
        for path in _iter_module_paths(package_root, experimental):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                counted = list(_explicit_parameters(node))
                if implicit:
                    positional = node.args.posonlyargs + node.args.args
                    if positional and positional[0].arg in ("self", "cls"):
                        counted.append(positional[0])
                total += sum(1 for a in counted if a.annotation is None)
        return total

    rungs: dict[str, int] = {}
    rungs["unannotated parameters (this module's definition)"] = _count(
        implicit=False, experimental=False
    )
    rungs["  + implicit self/cls, which must NOT be annotated (audit E.2)"] = _count(
        implicit=True, experimental=False
    )
    rungs["  + jaccpot/experimental/"] = _count(implicit=True, experimental=True)
    return rungs


def reconciliation(package_root: Path = PACKAGE_ROOT) -> dict[str, dict[str, int]]:
    """Recompute both burn-down populations under competing definitions.

    Each rung loosens exactly one thing, so a reader can see which choice accounts
    for which part of the gap rather than taking a docstring's word for it. See
    this module's own docstring for what the ladders showed.

    Parameters
    ----------
    package_root : Path, optional
        The ``jaccpot/`` package directory.

    Returns
    -------
    dict of str to dict
        Population name to its ladder of rung label and count, both in order
        (dicts preserve insertion order).
    """
    return {
        "bare Array": _bare_ladder(package_root),
        "unannotated": _unannotated_ladder(package_root),
    }


def _array_occurrences(annotation: ast.expr) -> int:
    """Count how many times an array type is named inside one annotation.

    Parameters
    ----------
    annotation : ast.expr
        The annotation expression.

    Returns
    -------
    int
        Occurrences of ``Array``, plain or dotted.
    """
    return sum(
        1
        for node in ast.walk(annotation)
        if (isinstance(node, ast.Name) and node.id == "Array")
        or (isinstance(node, ast.Attribute) and node.attr == "Array")
    )


def _format_report(modules: list[ModuleCensus], totals: Totals, limit: int) -> str:
    """Render the headline block and the per-module table.

    Parameters
    ----------
    modules : list of ModuleCensus
        Per-module counts, already sorted.
    totals : Totals
        Package totals.
    limit : int
        Rows to show; 0 shows every module with at least one array parameter.

    Returns
    -------
    str
        The report.
    """
    lines = [
        "jaccpot annotation census",
        f"  modules walked                       {totals.modules:6d}",
        f"  shape-annotated array parameters     {totals.shaped:6d}",
        f"  bare Array parameters                {totals.bare:6d}",
        f"  unannotated parameters               {totals.unannotated:6d}",
        f"  non-array parameters                 {totals.other:6d}",
        f"  shaped share of array parameters     {totals.shaped_fraction:6.1%}",
        "",
        f"  @jaxtyped functions                  {totals.decorated:6d}",
        f"    of which no shaped parameter       {totals.vacuous_decorated:6d}",
        f"  from jaxtyping import statements     {totals.jaxtyping_imports:6d}",
        "",
        f"{'bare':>6} {'shaped':>6} {'unann':>6}  module",
    ]
    interesting = [m for m in modules if m.array_parameters or m.unannotated]
    shown = interesting if limit == 0 else interesting[:limit]
    for module in shown:
        lines.append(
            f"{module.bare:6d} {module.shaped:6d} {module.unannotated:6d}  {module.path}"
        )
    if len(shown) < len(interesting):
        lines.append(
            f"{'':>6} {'':>6} {'':>6}  ... {len(interesting) - len(shown)} more"
        )

    offenders = [(m.path, name) for m in modules for name in m.width_suffixed]
    if offenders:
        lines.append("")
        lines.append("WIDTH-SUFFIXED dtypes (STYLE_GUIDE 4.4 forbids these):")
        lines.extend(f"  {path}: {name}" for path, name in offenders)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the census from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status. Non-zero only when a width-suffixed dtype was found,
        since that is a style violation rather than a measurement.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="include jaccpot/experimental/ (excluded by default, as in the audit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="module rows to print; 0 for all (default: 20)",
    )
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="print the ladder from this definition to the audit's F20 figure",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead")
    args = parser.parse_args(argv)

    modules, totals = census_package(include_experimental=args.include_experimental)

    if args.json:
        payload = {
            "totals": {
                "modules": totals.modules,
                "shaped": totals.shaped,
                "bare": totals.bare,
                "unannotated": totals.unannotated,
                "other": totals.other,
                "decorated": totals.decorated,
                "vacuous_decorated": totals.vacuous_decorated,
                "jaxtyping_imports": totals.jaxtyping_imports,
            },
            "modules": [
                {
                    "path": m.path,
                    "shaped": m.shaped,
                    "bare": m.bare,
                    "unannotated": m.unannotated,
                    "other": m.other,
                    "decorated": m.decorated,
                    "vacuous_decorated": m.vacuous_decorated,
                    "width_suffixed": m.width_suffixed,
                }
                for m in modules
            ],
        }
        if args.reconcile:
            payload["reconciliation"] = reconciliation()
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(modules, totals, args.limit))
        if args.reconcile:
            for population, ladder in reconciliation().items():
                print("")
                print(f"reconciliation -- {population}, under looser definitions:")
                for label, count in ladder.items():
                    print(f"  {count:6d}  {label}")
            print("")
            print("  F20 reports 3174 bare; the plan reports 368 unannotated.")
            print("  See this module's docstring, including for the shaped half.")

    return 1 if any(m.width_suffixed for m in modules) else 0


if __name__ == "__main__":
    sys.exit(main())

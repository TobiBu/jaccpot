"""Guardrails for ``__all__`` on the package's non-``__init__`` modules.

Why this exists: after the Tier 1 splits, several modules are aggregators whose
remaining job is to re-export what moved out of them. Without ``__all__`` there
is no mechanical way to tell a deliberate re-export from a dead import, which is
what leaves audit item 0.14 ("no dead imports remain") resting on a ``# noqa``
count nobody can re-derive. See ``docs/refactor_audit_2026-08.md`` A.11.

The three checks below are deliberately different in kind:

1. **Accuracy** -- every name in an ``__all__`` is actually bound by its module.
   ``flake8 --select=F822`` already catches this statically; asserting it here
   means it also holds for names bound dynamically, and it costs nothing.
2. **Coverage** -- every public function or class a module *defines* appears in
   its ``__all__``. This is the one that makes ``__all__`` mean something: a list
   that omits half the surface documents nothing. It stops at callables on
   purpose -- see ``_defined_public_callables`` for the 24 module-level constants
   that would otherwise force a public-surface widening.
3. **Completeness** -- the set of modules still without ``__all__`` is asserted
   to equal a literal list, in both directions.

Check 3 is an assertion, not a suppression, and the distinction is the reason it
is written this way. It fails when a module is converted and not removed from the
list, *and* when a new module lands without ``__all__``. Nothing can be added to
it silently: growing the list is an explicit edit that shows up in review. That
is what a retired ``.pydoclint-baseline.txt`` was not (see ``pyproject.toml``'s
``[tool.pydoclint]`` comment) -- a generated file that absorbs new violations in
any file it still lists.
"""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "jaccpot"

# `jaccpot/experimental/` is opt-in prototype code and is out of scope for the
# audit that motivates this guard, exactly as it is for pydoclint and coverage.
EXCLUDED_PARTS = frozenset({"experimental"})

# Modules that do not yet declare `__all__` -- the deferred half of audit item
# 0.16, tracked in A.11. Measured 2026-08-18: 34 of the 90 in-scope modules.
#
# They are NOT one job. 26 of them have no unused imports at all, so `__all__` is
# a plain declaration with nothing to disambiguate. The other 8 re-export names
# that other modules import from them, and 83 of the package's 114 non-`__init__`
# unused imports are PRIVATE names (leading underscore) that `__all__` does not
# govern at all -- 62 of them in `runtime/kernels/core.py` alone. Those 8 need a
# convention for private re-exports before a list can mean anything for them.
#
# Shrink this list; do not grow it. A new module belongs here only with a reason
# in review, and `test_no_module_silently_lacks_exports` fails either way round.
MODULES_WITHOUT_ALL = frozenset(
    {
        "jaccpot/nearfield/near_field.py",
        "jaccpot/runtime/_fmm_impl.py",
        "jaccpot/runtime/_interaction_cache.py",
        "jaccpot/runtime/_large_n_pipeline.py",
        "jaccpot/runtime/fmm_autotune.py",
        "jaccpot/runtime/fmm_sweeps.py",
        "jaccpot/runtime/kernels/core.py",
        "jaccpot/upward/solidfmm_complex_tree_expansions.py",
    }
)


def _iter_module_paths() -> list[Path]:
    """Collect the package modules this guard applies to.

    Returns
    -------
    list[Path]
        Every ``*.py`` under ``jaccpot/`` that is not an ``__init__.py`` and not
        under an excluded directory, sorted by path.
    """
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name != "__init__.py" and not EXCLUDED_PARTS.intersection(path.parts)
    )


def _rel(path: Path) -> str:
    """Render a module path relative to the repository root.

    Parameters
    ----------
    path : Path
        Absolute path to a module inside the package.

    Returns
    -------
    str
        The path relative to the repository root, with forward slashes, so the
        literal set above is stable across platforms.
    """
    return path.relative_to(PROJECT_ROOT).as_posix()


def _all_entries(tree: ast.Module) -> list[str] | None:
    """Read a module's ``__all__`` as a list of names.

    Both spellings used in this package are recognised: a plain
    ``__all__ = [...]`` and the annotated ``__all__: list[str] = []`` that the
    1.4/1.5/1.8 splits introduced for modules with no public surface.

    Parameters
    ----------
    tree : ast.Module
        Parsed module.

    Returns
    -------
    list[str] | None
        The string entries of ``__all__``, or ``None`` when the module does not
        declare one. An empty list is a declaration and is not ``None``.
    """
    for node in tree.body:
        target_is_all = False
        if isinstance(node, ast.Assign):
            target_is_all = any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            )
        elif isinstance(node, ast.AnnAssign):
            target_is_all = (
                isinstance(node.target, ast.Name) and node.target.id == "__all__"
            )
        if not target_is_all or node.value is None:
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return []
        return [
            element.value
            for element in node.value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
    return None


def _bound_names(tree: ast.Module) -> set[str]:
    """Collect every top-level name a module binds.

    Imports are included, because a deliberate re-export is a legitimate
    ``__all__`` entry that the module does not define itself.

    Parameters
    ----------
    tree : ast.Module
        Parsed module.

    Returns
    -------
    set[str]
        Names bound at module scope by ``def``, ``class``, assignment, or import.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _defined_public_callables(tree: ast.Module) -> set[str]:
    """Collect the public functions and classes a module defines itself.

    Two exclusions, both deliberate:

    * **Imported names.** Importing something is not exporting it, and counting
      it as such would make ``__all__`` grow with every dependency.
    * **Module-level constants.** This package already treats a public-named
      constant as internal in several places, and enforcing coverage would mean
      widening a public surface to satisfy a linter. Measured 2026-08-18: 24 such
      constants across 7 modules that *do* declare ``__all__`` -- ``config.py``'s
      five ``*_DOC`` strings, ``grad_options.py``'s ten ``ENV_*`` variable names,
      ``local_expansions.DEFAULT_M2L_CHUNK_SIZE`` and others. Each is a judgement
      call belonging to its own module, so the invariant here stops at callables,
      which are unambiguous.

    Parameters
    ----------
    tree : ast.Module
        Parsed module.

    Returns
    -------
    set[str]
        Names without a leading underscore that the module binds by ``def``,
        ``async def``, or ``class`` at module scope.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return {name for name in names if not name.startswith("_")}


def test_all_entries_are_bound_by_their_module() -> None:
    """Every ``__all__`` entry must be a name the module actually binds."""
    dangling: list[str] = []
    for path in _iter_module_paths():
        tree = ast.parse(path.read_text())
        entries = _all_entries(tree)
        if entries is None:
            continue
        bound = _bound_names(tree)
        for name in entries:
            if name not in bound:
                dangling.append(
                    f"{_rel(path)}: __all__ names {name!r}, which is unbound"
                )

    assert not dangling, "\n".join(
        ["__all__ entries with no corresponding binding:", *dangling]
    )


def test_all_covers_every_public_callable_the_module_defines() -> None:
    """A module's ``__all__`` must not omit a public function or class it defines."""
    omitted: list[str] = []
    for path in _iter_module_paths():
        tree = ast.parse(path.read_text())
        entries = _all_entries(tree)
        if entries is None:
            continue
        missing = sorted(_defined_public_callables(tree) - set(entries))
        if missing:
            omitted.append(f"{_rel(path)}: {', '.join(missing)}")

    assert not omitted, "\n".join(
        [
            "Public callables defined by a module but absent from its __all__.",
            "Add them, or rename them private if they are not exported:",
            *omitted,
        ]
    )


def test_no_module_silently_lacks_exports() -> None:
    """The set of modules without ``__all__`` must equal ``MODULES_WITHOUT_ALL``.

    Asserted in both directions on purpose. A converted module left in the list
    fails just as loudly as a new module that never got one, so the list cannot
    drift into being a suppression file.
    """
    actual = {
        _rel(path)
        for path in _iter_module_paths()
        if _all_entries(ast.parse(path.read_text())) is None
    }

    newly_missing = sorted(actual - MODULES_WITHOUT_ALL)
    now_converted = sorted(MODULES_WITHOUT_ALL - actual)

    assert not newly_missing, "\n".join(
        [
            "These modules declare no __all__ and are not on the known list.",
            "Add __all__ to them rather than extending MODULES_WITHOUT_ALL:",
            *newly_missing,
        ]
    )
    assert not now_converted, "\n".join(
        [
            "These modules now declare __all__ but are still listed as missing it.",
            "Remove them from MODULES_WITHOUT_ALL:",
            *now_converted,
        ]
    )

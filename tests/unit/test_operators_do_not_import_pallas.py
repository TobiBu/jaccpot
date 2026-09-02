"""``operators/`` is pure algebra: it must not import ``pallas/``.

Audit item **G.3** asked whether ``operators/`` should be allowed to import
``pallas/``, and the decision was option (b) -- move the dispatch up into
``runtime/kernels/`` -- so that ARCHITECTURE section 8's description of
``operators/`` as pure algebra is true rather than true-with-exceptions.

That is now the case: the three function-local imports the audit measured
(``m2l_real_rot_scale.py`` twice, ``complex_ops.py`` once) are gone, and the
dispatch lives in ``runtime/kernels/_m2l.py``, which builds the fused carriers
lazily from the kernels it imports. This file is what keeps it that way.

Why a *static* check rather than an import-time one: every edge G.3 measured was
**function-local**, deferred precisely so it would not fire at import. A runtime
probe would import ``jaccpot.operators`` and see nothing, which is exactly how
three of these accumulated under a rule that was supposed to forbid them. Only
reading the source finds an import that has not run yet.

The direction is asserted in both senses. ``pallas/ -> operators/`` must keep
existing: the point of G.3 was to make the dependency acyclic, not to sever it,
and a test that only checked for the absence of one edge would pass just as
happily if someone deleted the other.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "jaccpot"


def _imports_of(package: str) -> set[tuple[str, str]]:
    """Return ``(file, imported module)`` for every ``jaccpot.*`` import in a package.

    Relative imports are resolved against the package so that ``from ..pallas
    import x`` inside ``operators/`` is caught alongside the absolute form.

    Parameters
    ----------
    package : str
        Package directory under ``jaccpot/``, e.g. ``"operators"``.

    Returns
    -------
    set of tuple of (str, str)
        One entry per import edge, module-scope or function-local alike.
    """
    edges: set[tuple[str, str]] = set()
    root = PACKAGE_ROOT / package
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("jaccpot."):
                        edges.add((path.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # `from ..pallas import x` inside jaccpot/<package>/
                    parts = ["jaccpot", package][: 2 - (node.level - 1)] or ["jaccpot"]
                    resolved = ".".join(parts + ([node.module] if node.module else []))
                elif node.module:
                    resolved = node.module
                else:
                    continue
                if resolved.startswith("jaccpot."):
                    edges.add((path.name, resolved))
    return edges


def test_operators_does_not_import_pallas() -> None:
    """No import anywhere under ``operators/`` may target ``pallas/``."""
    offenders = sorted(
        (f, m) for f, m in _imports_of("operators") if m.startswith("jaccpot.pallas")
    )
    assert not offenders, (
        "`operators/` is pure algebra (ARCHITECTURE section 8) and must not import "
        "`pallas/`; the kernel dispatch belongs in `runtime/kernels/`. Found: "
        f"{offenders}"
    )


def test_pallas_still_imports_operators() -> None:
    """The dependency must stay directed, not severed.

    G.3 made the edge one-way. If this ever reaches zero, the guard above has
    become vacuous and the layering claim it protects is no longer being tested
    by anything.
    """
    edges = sorted(
        (f, m) for f, m in _imports_of("pallas") if m.startswith("jaccpot.operators")
    )
    assert edges, (
        "`pallas/` should still build on `operators/` -- G.3 removed the reverse "
        "edge to make the dependency acyclic, not to cut it"
    )


@pytest.mark.parametrize(
    "first", ["from jaccpot.pallas import m2l_real_fused", "import jaccpot.operators"]
)
def test_either_package_may_be_imported_first(first: str) -> None:
    """Neither package may depend on the other having been imported already.

    The comment this replaces recorded that a module-scope `operators/ ->
    pallas/` import broke ``from jaccpot.pallas import ...`` when that was the
    first ``jaccpot`` import. Run in a clean subprocess, because import order
    cannot be tested inside a session that has already imported both.
    """
    result = subprocess.run(
        [sys.executable, "-c", first],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert (
        result.returncode == 0
    ), f"`{first}` failed as the first jaccpot import:\n{result.stderr[-1500:]}"

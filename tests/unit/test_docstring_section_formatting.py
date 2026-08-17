"""A numpydoc section header must be preceded by a blank line.

pydoclint does not check this, and it is not cosmetic: numpydoc and Sphinx's
napoleon only recognise a section when a blank line separates it from the text
above. Without one the whole ``Parameters`` block renders as ordinary prose, so
the docstring passes every gate in this repo while documenting nothing a reader
or a doc build can use.

This is a self-inflicted-defect guard. Fifteen sites were introduced during the
Tier 2.5 docstring programme by appending a section directly onto a one-line
summary -- for a single-line docstring the closing ``\"\"\"`` sits on the summary
line, so an insert before it lands with no separating blank line. Four batches
shipped that way before it was noticed.
"""

from __future__ import annotations

import ast
import pathlib

# Section names numpydoc recognises. Only those actually used in this package,
# plus the ones a future docstring is likely to reach for.
_SECTIONS = frozenset(
    {
        "Attributes",
        "Examples",
        "Notes",
        "Parameters",
        "Raises",
        "References",
        "Returns",
        "See Also",
        "Warnings",
        "Yields",
    }
)

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "jaccpot"


def _offenders(source: str) -> list[tuple[str, str]]:
    """Find docstrings whose section header lacks a preceding blank line.

    Parameters
    ----------
    source : str
        Python source text.

    Returns
    -------
    list[tuple[str, str]]
        ``(owner name, section name)`` per offending docstring. At most one entry
        per docstring -- the first offence is enough to fail, and reporting every
        one would bury the useful signal.
    """
    tree = ast.parse(source)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            continue
        lines = docstring.split("\n")
        for index in range(1, len(lines)):
            if lines[index].strip() in _SECTIONS and lines[index - 1].strip() != "":
                found.append((getattr(node, "name", "<module>"), lines[index].strip()))
                break
    return found


def test_every_docstring_section_has_a_blank_line_before_it() -> None:
    """Scan the whole package, so a new offender cannot hide in an untouched file."""
    failures: list[str] = []
    scanned = 0
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        scanned += 1
        for owner, section in _offenders(path.read_text()):
            failures.append(
                f"{path.relative_to(_PACKAGE_ROOT.parent)}::{owner} -> {section}"
            )

    assert scanned > 0, "found no package files to scan; the root path is wrong"
    assert not failures, (
        "these docstrings have a section header with no blank line above it, so "
        "numpydoc will not recognise the section:\n  " + "\n  ".join(failures)
    )


def test_the_guard_detects_a_known_offender() -> None:
    """Non-vacuity: the scanner must actually fire on the shape it targets."""
    offending = '''
def f(a):
    """One-line summary.
    Parameters
    ----------
    a : int
        Something.
    """
'''
    assert _offenders(offending) == [("f", "Parameters")]

    well_formed = '''
def f(a):
    """One-line summary.

    Parameters
    ----------
    a : int
        Something.
    """
'''
    assert _offenders(well_formed) == []

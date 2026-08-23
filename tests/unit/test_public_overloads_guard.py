"""Guardrails for the ``@overload`` contract on the public solver methods.

Why this exists: ``compute_accelerations`` and ``evaluate_prepared_state``
return one of four shapes depending on ``return_potential`` and on whether
``max_acc_derivative_order`` is zero. Without overloads every caller receives the
four-member union and has to narrow it by hand. Measured on a probe that mimics
downstream code -- assigning the results to ``Array``-typed targets -- pyright
reports **6 errors without the overloads and 0 with them**.

That benefit is invisible to ``pyright jaccpot/``: every *internal* caller passes
runtime values for those flags, so no overload can be selected and the package
error count does not move. The audit's E.4 records why the internal multi-flag
producers were therefore left alone. This surface is the exception, because it is
what downstream code calls and downstream calls with literals -- so the contract
needs a guard here rather than a number to watch.

The third overload is the one to be careful about. It is the honest fallback for
callers whose flags are runtime values, and deleting it would not fail any type
check -- it would silently make pyright resolve such calls to a ``Literal[0]``
overload and promise a derivative tuple where a bare ``Array`` is returned. So it
is asserted explicitly, by return type, not merely counted.

**Read from the source, not from ``typing.get_overloads``.** The first version of
this file used the runtime registry, and it was red under
``JACCPOT_RUNTIME_TYPECHECK=1``: that lane installs a package-wide import hook
which wraps every function, so all three stubs register under the wrapper's code
object and the registry reports one overload instead of three. Reading the AST is
immune to that, and is the better match anyway -- an overload set is a *static*
type-surface contract, and it is how the other guards in this suite
(``test_module_exports_guard``, ``test_mixin_engine_base_guard``) already work.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOLVER = PROJECT_ROOT / "jaccpot" / "solver.py"
CLASS_NAME = "FastMultipoleMethod"
OVERLOADED = ("compute_accelerations", "evaluate_prepared_state")


def _overload_stubs(name: str) -> list[ast.FunctionDef]:
    """Return the ``@overload`` stubs declared for one method, in source order.

    Parameters
    ----------
    name : str
        The method name to collect stubs for.

    Returns
    -------
    list of ast.FunctionDef
        Every declaration of ``name`` in the class body carrying an
        ``@overload`` decorator.
    """
    tree = ast.parse(SOLVER.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS_NAME
    )
    return [
        item
        for item in cls.body
        if isinstance(item, ast.FunctionDef)
        and item.name == name
        and any("overload" in ast.unparse(d) for d in item.decorator_list)
    ]


def _flag_annotation(stub: ast.FunctionDef) -> str:
    """Return the ``return_potential`` annotation of one stub, unparsed."""
    for arg in stub.args.posonlyargs + stub.args.args + stub.args.kwonlyargs:
        if arg.arg == "return_potential" and arg.annotation is not None:
            return ast.unparse(arg.annotation)
    return ""


@pytest.mark.parametrize("name", OVERLOADED)
def test_public_method_declares_three_overloads(name: str) -> None:
    """Each overloaded method keeps its two narrow cases and the fallback."""
    stubs = _overload_stubs(name)
    assert len(stubs) == 3, (
        f"{name} should declare exactly three overloads -- the two literal cases "
        f"and the union fallback; found {len(stubs)}"
    )


@pytest.mark.parametrize("name", OVERLOADED)
def test_narrow_overloads_promise_narrow_returns(name: str) -> None:
    """``return_potential`` literals map to the concrete return shapes."""
    by_flag = {
        _flag_annotation(s): ast.unparse(s.returns) if s.returns else ""
        for s in _overload_stubs(name)
    }
    false_ret = next(v for k, v in by_flag.items() if "False" in k)
    true_ret = next(v for k, v in by_flag.items() if "True" in k)
    assert false_ret == "Array", f"{name}: Literal[False] should return Array"
    assert (
        true_ret.replace(" ", "") == "Tuple[Array,Array]"
    ), f"{name}: Literal[True] should return Tuple[Array, Array]; got {true_ret}"


@pytest.mark.parametrize("name", OVERLOADED)
def test_the_union_fallback_survives(name: str) -> None:
    """The non-literal overload must stay, or the narrow ones start lying.

    A caller passing a runtime ``bool`` cannot select either literal overload.
    With the fallback removed, pyright would fall back to a ``Literal[0]``
    signature and promise a shape the call does not return.
    """
    fallbacks = [
        s for s in _overload_stubs(name) if "Literal" not in _flag_annotation(s)
    ]
    assert len(fallbacks) == 1, (
        f"{name} must keep exactly one non-literal fallback overload; "
        f"found {len(fallbacks)}"
    )
    ret = ast.unparse(fallbacks[0].returns) if fallbacks[0].returns else ""
    assert (
        "Union" in ret or "|" in ret
    ), f"{name}: the fallback must return the full union, not {ret}"

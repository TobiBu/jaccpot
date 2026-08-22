"""Guardrails for the ``@overload`` contract on the public solver methods.

Why this exists: ``compute_accelerations`` and ``evaluate_prepared_state``
return one of four shapes depending on ``return_potential`` and on whether
``max_acc_derivative_order`` is zero. Without overloads every caller receives the
four-member union and has to narrow it by hand. Measured on a probe that mimics
downstream code -- assigning the results to ``Array``-typed targets -- pyright
reports **6 errors without the overloads and 0 with them**.

That benefit is invisible to ``pyright jaccpot/``: every *internal* caller passes
runtime values for those flags, so no overload can be selected and the package
error count does not move (293 before, 293 after). The audit's E.4 records why
the internal multi-flag producers were therefore left alone. This surface is the
exception, because it is what downstream code calls and downstream calls with
literals -- so the contract needs a guard here rather than a number to watch.

The third overload is the one to be careful about. It is the honest fallback for
callers whose flags are runtime values, and deleting it would not fail any type
check -- it would silently make pyright resolve such calls to a ``Literal[0]``
overload and promise a derivative tuple where a bare ``Array`` is returned. So it
is asserted explicitly, by return type, not merely counted.
"""

from __future__ import annotations

import typing

import pytest

from jaccpot.solver import FastMultipoleMethod

OVERLOADED = ("compute_accelerations", "evaluate_prepared_state")


@pytest.mark.parametrize("name", OVERLOADED)
def test_public_method_declares_three_overloads(name: str) -> None:
    """Each overloaded method keeps its two narrow cases and the fallback."""
    fn = getattr(FastMultipoleMethod, name)
    overloads = typing.get_overloads(fn)
    assert len(overloads) == 3, (
        f"{name} should declare exactly three overloads -- the two literal cases "
        f"and the union fallback; found {len(overloads)}"
    )


@pytest.mark.parametrize("name", OVERLOADED)
def test_narrow_overloads_promise_narrow_returns(name: str) -> None:
    """``return_potential`` literals map to the concrete return shapes."""
    overloads = typing.get_overloads(getattr(FastMultipoleMethod, name))
    by_flag = {
        str(o.__annotations__.get("return_potential")): str(
            o.__annotations__.get("return")
        )
        for o in overloads
    }
    false_ret = next(v for k, v in by_flag.items() if "False" in k)
    true_ret = next(v for k, v in by_flag.items() if "True" in k)
    assert false_ret == "Array", f"{name}: Literal[False] should return Array"
    assert "Tuple[Array, Array]" in true_ret.replace(" ", "").replace(
        "Tuple[Array,Array]", "Tuple[Array, Array]"
    ) or "Tuple[Array,Array]" in true_ret.replace(
        " ", ""
    ), f"{name}: Literal[True] should return Tuple[Array, Array]; got {true_ret}"


@pytest.mark.parametrize("name", OVERLOADED)
def test_the_union_fallback_survives(name: str) -> None:
    """The non-literal overload must stay, or the narrow ones start lying.

    A caller passing a runtime ``bool`` cannot select either literal overload.
    With the fallback removed, pyright would fall back to a ``Literal[0]``
    signature and promise a shape the call does not return.
    """
    overloads = typing.get_overloads(getattr(FastMultipoleMethod, name))
    fallbacks = [
        o
        for o in overloads
        if "Literal" not in str(o.__annotations__.get("return_potential"))
    ]
    assert len(fallbacks) == 1, (
        f"{name} must keep exactly one non-literal fallback overload; "
        f"found {len(fallbacks)}"
    )
    ret = str(fallbacks[0].__annotations__.get("return"))
    assert (
        "Union" in ret or "|" in ret
    ), f"{name}: the fallback must return the full union, not {ret}"

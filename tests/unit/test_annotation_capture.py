"""The shape-capture helper's own contract.

`bench/annotation_capture.py` exists so that STYLE_GUIDE §4.2 -- derive the shape
by execution, never from the docstring -- has a reusable tool instead of a
throwaway script per PR. Two of its behaviours are not obvious and both were
wrong on the first attempt, so both are pinned here.

**It must follow `from X import Y`.** That form binds a SEPARATE name in the
importing module's globals, and Python resolves the call against that one. A
wrapper installed only on the defining module never runs, and the tool reports
"never called" for a function called on every step -- which is worse than no
tool, because the silence looks like evidence.

**It must not confuse same-named functions from different modules.** This is not
hypothetical here: `jaccpot/operators/multipole_utils.py` and
`yggdrax/multipole_utils.py` define eight identically-named functions, and
production imports yggdrax's. The binding scan therefore matches on object
identity, never on name.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

pytest.importorskip("yggdrax")

import pathlib

from bench import annotation_capture  # noqa: E402


def _make_module(name: str, **attributes: object) -> types.ModuleType:
    """Register a throwaway module in ``sys.modules``.

    Parameters
    ----------
    name : str
        Module name.
    **attributes : object
        Attributes to set on it.

    Returns
    -------
    types.ModuleType
        The registered module.
    """
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture
def two_modules(request):
    """Provide a defining module and an importer that did ``from X import Y``.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Used to register cleanup of the synthetic modules.

    Returns
    -------
    tuple
        ``(defining_module, importing_module, original_function)``.
    """

    def target(x):
        return x

    defining = _make_module("_cap_defining", target=target)
    importing = _make_module("_cap_importing", target=target)
    request.addfinalizer(lambda: sys.modules.pop("_cap_defining", None))
    request.addfinalizer(lambda: sys.modules.pop("_cap_importing", None))
    return defining, importing, target


def test_it_patches_the_importers_binding_not_only_the_definition(
    two_modules, tmp_path
):
    """The defect that made the tool report "never called" for a hot function."""
    defining, importing, _ = two_modules
    out = tmp_path / "caps.jsonl"

    with annotation_capture.capture_shapes({"_cap_defining": ["target"]}, out):
        # Call through the IMPORTER's binding, which is how real call sites reach it.
        importing.target(_FakeArray((7, 3), "float64"))

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records, "the call through the importing module was not observed"
    assert records[0]["parameters"]["x"]["shape"] == [7, 3]


def test_it_restores_every_binding_it_patched(two_modules, tmp_path):
    """A capture that leaked a wrapper would poison the rest of the session."""
    defining, importing, original = two_modules
    with annotation_capture.capture_shapes(
        {"_cap_defining": ["target"]}, tmp_path / "caps.jsonl"
    ):
        assert importing.target is not original
    assert defining.target is original
    assert importing.target is original


def test_a_same_named_function_from_another_module_is_left_alone(tmp_path, request):
    """Identity, not name -- jaccpot and yggdrax share eight function names."""

    def target(x):
        return x

    def impostor(x):
        return x

    _make_module("_cap_defining2", target=target)
    other = _make_module("_cap_other", target=impostor)
    request.addfinalizer(lambda: sys.modules.pop("_cap_defining2", None))
    request.addfinalizer(lambda: sys.modules.pop("_cap_other", None))

    with annotation_capture.capture_shapes(
        {"_cap_defining2": ["target"]}, tmp_path / "c.jsonl"
    ):
        assert other.target is impostor, "a same-named different function was patched"


def test_only_array_like_arguments_are_recorded(two_modules, tmp_path):
    """Scalars and None have no axes, so they cannot inform a shape annotation."""
    _, importing, _ = two_modules
    out = tmp_path / "caps.jsonl"
    with annotation_capture.capture_shapes({"_cap_defining": ["target"]}, out):
        importing.target(4)
        importing.target(None)
        importing.target(_FakeArray((5,), "int32"))
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["parameters"] for r in records] == [
        {},
        {},
        {"x": {"shape": [5], "dtype": "int32"}},
    ]


def test_constant_axes_are_reported_as_literals_and_never_paired():
    """Pairing them emitted 55 lines of "3 == 3" on one real signature.

    Every `(3,)` in a signature is equal to every other `(3,)`, so pairing the
    constant axes buries the one equality a reader actually has to adjudicate.
    They are reported as literals instead.
    """
    calls = [
        {"parameters": {"delta": {"shape": [3]}, "dipole": {"shape": [3]}}},
        {"parameters": {"delta": {"shape": [3]}, "dipole": {"shape": [3]}}},
    ]
    lines = annotation_capture._equality_report(calls)
    assert any("CONSTANT axes" in line for line in lines)
    assert not any(line.startswith("equal:") for line in lines)


def test_an_equality_seen_at_one_extent_is_marked_unproven():
    """Two axes that each vary, but were only ever observed together once.

    The realistic shape of this is an optional parameter: `a` is present on
    every call, `b` on few, and the calls carrying both happen to sit at one
    problem size. The equality is then one observation, not three, and naming
    the two axes alike rests on that single call.
    """
    calls = [
        {"parameters": {"a": {"shape": [5, 2]}, "b": {"shape": [5]}}},
        {"parameters": {"a": {"shape": [9, 2]}}},
        {"parameters": {"b": {"shape": [7]}}},
    ]
    lines = annotation_capture._equality_report(calls)
    assert any("a[0] == b[0]" in line and "UNPROVEN" in line for line in lines)


def test_an_equality_seen_at_several_extents_is_reported_as_evidence():
    """The same equality across varying extents is what makes an axis name safe."""
    calls = [
        {"parameters": {"a": {"shape": [5, 2]}, "b": {"shape": [5]}}},
        {"parameters": {"a": {"shape": [9, 2]}, "b": {"shape": [9]}}},
        {"parameters": {"a": {"shape": [17, 2]}, "b": {"shape": [17]}}},
    ]
    lines = annotation_capture._equality_report(calls)
    assert any(
        "a[0] == b[0]" in line and "at 3 distinct values" in line for line in lines
    )
    assert not any("UNPROVEN" in line for line in lines)


def test_targets_parse_into_modules_and_names():
    """The env-var spec, which is how CI and distributed runs configure it."""
    assert annotation_capture.parse_targets("m:a,b;n:c") == {
        "m": ["a", "b"],
        "n": ["c"],
    }
    with pytest.raises(ValueError):
        annotation_capture.parse_targets("no_colon_here")


class _FakeArray:
    """A stand-in carrying only what the recorder reads.

    Deliberately not a real `jax` array: the recorder's contract is "anything
    exposing `.shape` and `.dtype`", and pinning it against a real array would
    test jax rather than this module.

    Parameters
    ----------
    shape : tuple of int
        The shape to report.
    dtype : str
        The dtype to report.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = shape
        self.dtype = dtype


# ---------------------------------------------------------------------------
# Method targets. They were the tool's largest blind spot rather than an edge case: the
# `runtime/` mixins hold 174 of the burn-down's remaining parameters, and while only
# module attributes were rebound, every one of them reported no observations -- which is
# indistinguishable from "never called".
# ---------------------------------------------------------------------------


class _Mixin:
    """A stand-in for the `runtime/` mixins: defines the method, never instantiated."""

    def measure(self, values):
        """Return the argument unchanged.

        Parameters
        ----------
        values : object
            Anything.

        Returns
        -------
        object
            ``values``.
        """
        return values


class _Engine(_Mixin):
    """A stand-in for `FMMEngine`: inherits the method and does not override it."""


def _make_class_module(request):
    """Register a throwaway module exposing ``_Mixin`` and ``_Engine``.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Used to deregister the synthetic module.

    Returns
    -------
    types.ModuleType
        The registered module.
    """
    module = _make_module("_cap_methods", _Mixin=_Mixin, _Engine=_Engine)
    request.addfinalizer(lambda: sys.modules.pop("_cap_methods", None))
    return module


def test_a_method_target_resolves_to_the_defining_class(request):
    """``Class.method`` names the class as the holder, not the module."""
    module = _make_class_module(request)
    holder, attribute, original = annotation_capture.resolve_target(
        module, "_Mixin.measure"
    )
    assert holder is _Mixin
    assert attribute == "measure"
    assert original is _Mixin.__dict__["measure"]


def test_patching_the_mixin_observes_a_call_made_through_a_subclass(request, tmp_path):
    """The reason patching the CLASS is enough, and the reason it is the right level.

    `FMMEngine` inherits `DerivativesMixin` without overriding, so a call on an engine
    resolves through the MRO to the patched attribute. Patching every subclass would be
    both unnecessary and fragile.
    """
    module = _make_class_module(request)
    out = tmp_path / "caps.jsonl"
    with annotation_capture.capture_shapes({"_cap_methods": ["_Mixin.measure"]}, out):
        _Engine().measure(_FakeArray((4, 3), "float64"))
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["function"] == "_cap_methods:_Mixin.measure"
    # `self` carries no `.shape`, so it is dropped without needing an exemption.
    assert list(records[0]["parameters"]) == ["values"]
    assert records[0]["parameters"]["values"]["shape"] == [4, 3]


def test_the_method_binding_is_restored_afterwards(request):
    """A leaked wrapper would make every later test record into a closed file."""
    module = _make_class_module(request)
    original = _Mixin.__dict__["measure"]
    with annotation_capture.capture_shapes(
        {"_cap_methods": ["_Mixin.measure"]}, pathlib.Path("/dev/null")
    ):
        assert _Mixin.__dict__["measure"] is not original
    assert _Mixin.__dict__["measure"] is original


@pytest.mark.parametrize(
    "target,match",
    [("_Mixin.inner.measure", "dots"), ("_Mixin.NOT_CALLABLE", "not callable")],
)
def test_an_unresolvable_method_target_raises_rather_than_recording_nothing(
    request, target, match
):
    """The whole point: "no observations" must keep meaning "never called".

    A typo'd or too-deeply-nested target that silently recorded nothing would read as
    "this lane is not exercised", which is the conclusion the tool exists to make
    trustworthy.
    """
    module = _make_class_module(request)
    _Mixin.NOT_CALLABLE = 3
    try:
        with pytest.raises(ValueError, match=match):
            annotation_capture.resolve_target(module, target)
    finally:
        del _Mixin.NOT_CALLABLE

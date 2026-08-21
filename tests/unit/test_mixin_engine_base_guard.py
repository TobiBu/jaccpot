"""Guardrails for how the ``FMMEngine`` mixins reach the engine's attributes.

Why this exists: the ten ``*Mixin`` classes under ``jaccpot/runtime/`` are only
ever mixed into ``FMMEngine``, so every method body reads ``self.<engine
attribute>`` for attributes the mixin itself does not define. A type checker
cannot resolve those unless it is told what ``self`` is, and there are two ways
to tell it -- one of which is wrong:

* ``self: "FMMEngine"`` on each method. This *does* resolve the attributes, but
  ``FMMEngine`` is a **subtype** of every mixin, and pyright rejects a declared
  ``self`` that is not a supertype of its class. Applied consistently it produced
  119 errors of its own.
* Inheriting a ``TYPE_CHECKING``-only ``_EngineBase`` alias, which is
  ``FMMEngine`` to a checker and ``object`` at runtime. This is what the tree
  does, and it took pyright from 705 errors to 408 with the MRO untouched.

See ``docs/refactor_audit_2026-08.md`` E.2 for the measurements.

The checks are deliberately different in kind:

1. **Every mixin inherits the alias** -- read statically, so a new mixin that
   forgets it fails here rather than silently losing attribute resolution.
2. **The alias is ``object`` at runtime** -- the property that makes this free.
   If it ever evaluated to ``FMMEngine`` at runtime it would form the import
   cycle ARCHITECTURE section 8 forbids.
3. **The MRO is pinned** -- the whole argument for this pattern is that it does
   not perturb attribute lookup, and an explicit base is where that could go
   wrong. Asserted as a literal list, so a reordering has to be an edit.
4. **The rejected convention stays rejected** -- no mixin method annotates
   ``self``. This is the check that stops the 119 errors coming back one
   well-meaning method at a time.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "jaccpot" / "runtime"

# The engine's own module is exempt from check 4: there `self: "FMMEngine"` names
# the class itself, which is a valid (equal) supertype, so the nine annotations
# it carries are correct rather than tolerated.
SELF_ANNOTATION_EXEMPT = frozenset({"_fmm_impl.py"})

# Asserted, not derived, for the reason check 3 exists at all.
EXPECTED_ENGINE_MRO = (
    "FMMEngine",
    "PrepareMixin",
    "EvaluateMixin",
    "StrictRunMixin",
    "SweepsMixin",
    "OverridesMixin",
    "AutotuneMixin",
    "PolicyMixin",
    "DerivativesMixin",
    "StrictCapProfileMixin",
    "DiagnosticsMixin",
    "object",
)


def _mixin_classes():
    """Yield ``(path, ClassDef)`` for every ``*Mixin`` defined under ``runtime/``."""
    for path in sorted(RUNTIME_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Mixin"):
                yield path, node


def test_every_mixin_inherits_the_engine_base_alias():
    """Each ``*Mixin`` must declare ``_EngineBase`` as its only base."""
    found = list(_mixin_classes())
    assert found, "no mixins discovered -- the glob or the layout moved"

    offenders = {
        f"{path.name}::{node.name}": [ast.unparse(b) for b in node.bases]
        for path, node in found
        if [ast.unparse(b) for b in node.bases] != ["_EngineBase"]
    }
    assert not offenders, (
        "every runtime mixin must inherit the TYPE_CHECKING-only `_EngineBase` "
        f"alias so `self.<engine attribute>` resolves; these do not: {offenders}"
    )


def test_engine_base_alias_is_object_at_runtime():
    """``_EngineBase`` must be ``object`` outside a type checker."""
    modules = {path.stem for path, _ in _mixin_classes()}
    assert modules, "no mixin modules discovered"

    for stem in sorted(modules):
        module = importlib.import_module(f"jaccpot.runtime.{stem}")
        alias = getattr(module, "_EngineBase", None)
        assert alias is object, (
            f"jaccpot.runtime.{stem}._EngineBase is {alias!r}, not `object`; at "
            "runtime the alias must not resolve to the engine or it forms the "
            "import cycle ARCHITECTURE section 8 forbids"
        )


def test_engine_mro_is_unchanged_by_the_alias():
    """The explicit base must leave ``FMMEngine.__mro__`` exactly as it was."""
    from jaccpot.runtime._fmm_impl import FMMEngine

    assert tuple(c.__name__ for c in FMMEngine.__mro__) == EXPECTED_ENGINE_MRO


def test_no_mixin_method_annotates_self():
    """``self: "FMMEngine"`` is the rejected convention and must not return."""
    offenders = []
    for path, node in _mixin_classes():
        if path.name in SELF_ANNOTATION_EXEMPT:
            continue
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = item.args.posonlyargs + item.args.args
            if not args or args[0].arg != "self":
                continue
            if args[0].annotation is not None:
                offenders.append(
                    f"{path.name}::{node.name}.{item.name}: "
                    f"self: {ast.unparse(args[0].annotation)}"
                )
    assert not offenders, (
        "annotating `self` on a mixin method declares a subtype of the mixin as "
        "its own `self`, which pyright rejects -- inheritance from `_EngineBase` "
        f"is what resolves the attributes instead. Found: {offenders}"
    )


def test_mixin_methods_are_actually_the_engine_surface():
    """Guard the premise: the mixins supply most of the engine's own methods.

    Checks 1-4 are all about *how* the mixins see the engine. This one asserts
    the thing that makes it worth doing -- that the mixin methods really are the
    bulk of ``FMMEngine``, so a checker that cannot resolve them cannot check the
    engine at all.
    """
    from jaccpot.runtime._fmm_impl import FMMEngine

    mixin_methods = {
        item.name
        for _, node in _mixin_classes()
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    engine_methods = {
        name for name, _ in inspect.getmembers(FMMEngine, inspect.isfunction)
    }
    assert len(mixin_methods & engine_methods) > 100, (
        "expected the mixins to supply >100 of the engine's methods; got "
        f"{len(mixin_methods & engine_methods)} -- has the god-class split moved?"
    )

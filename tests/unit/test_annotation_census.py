"""The burn-down counter's own contract.

`bench/annotation_census.py` exists so that the jaxtyping rollout has one number
two people can reproduce. A counter is worth very little if its classifier is
wrong in a way nobody notices: an over-count makes finished work look unfinished,
and an under-count -- the dangerous direction -- makes the burn-down appear to
progress while nothing was converted.

So this pins the classifier against hand-written signatures whose correct bucket
is obvious by inspection, plus the two structural exemptions the census inherits
from `test_type_annotation_guard.py` and audit E.2.

It deliberately does NOT pin the package totals. Those move with every PR of the
rollout, which is the point of measuring them; a test asserting `bare == 1855`
would fail on the first conversion and teach the next contributor to update the
number rather than to read it.
"""

from __future__ import annotations

import ast
import textwrap

from bench import annotation_census


def _census_of(source: str) -> annotation_census.ModuleCensus:
    """Run the census over a source string by classifying its parameters.

    Parameters
    ----------
    source : str
        Module source to classify.

    Returns
    -------
    ModuleCensus
        Counts for the synthetic module.
    """
    result = annotation_census.ModuleCensus(path="<synthetic>")
    tree = ast.parse(textwrap.dedent(source))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "jaxtyping":
            result.jaxtyping_imports += 1
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        any_shaped = False
        for parameter in annotation_census._explicit_parameters(node):
            kind = annotation_census.classify_parameter(parameter.annotation)
            setattr(result, kind, getattr(result, kind) + 1)
            any_shaped = any_shaped or kind == annotation_census.SHAPED
        if annotation_census._is_jaxtyped(node):
            result.decorated += 1
            if not any_shaped:
                result.vacuous_decorated += 1
    return result


def _classify(annotation_source: str) -> str:
    """Classify a single annotation given as source text.

    Parameters
    ----------
    annotation_source : str
        The annotation, e.g. ``"Float[Array, 'n 3']"``.

    Returns
    -------
    str
        The census bucket.
    """
    return annotation_census.classify_parameter(
        ast.parse(annotation_source, mode="eval").body
    )


def test_a_shape_spec_is_what_separates_shaped_from_bare():
    """The distinction the whole burn-down is measuring."""
    assert _classify("Float[Array, 'n 3']") == annotation_census.SHAPED
    assert _classify("Array") == annotation_census.BARE
    assert _classify("jax.Array") == annotation_census.BARE
    assert _classify("int") == annotation_census.OTHER


def test_a_dtype_family_without_an_axis_spec_is_not_shaped():
    """`Float[Array]` asserts a dtype and no shape, so it is not the goal state.

    Counting it as shaped would let a conversion that pinned nothing about shape
    register as progress -- the under-count direction, which is the one that
    hides work rather than inventing it.
    """
    assert _classify("Float[Array]") == annotation_census.BARE


def test_a_shape_nested_inside_a_wrapper_still_counts():
    """`Optional[...]` and `tuple[...]` are the common spellings here."""
    assert _classify("Optional[Float[Array, 'n 3']]") == annotation_census.SHAPED
    assert _classify("tuple[Float[Array, 'n 3'], Int[Array, 'leaves']]") == (
        annotation_census.SHAPED
    )


def test_one_parameter_counts_once_however_many_arrays_it_names():
    """The unit is the parameter, not the token.

    This is the choice that separates this module's 1855 from the audit F20
    row's 3174, and it is the one that makes the number a burn-down: converting
    a parameter must move it from one column to another exactly once.
    """
    census = _census_of("""
        def f(pair: tuple[Array, Array], single: Array) -> None: ...
        """)
    assert census.bare == 2
    assert census.shaped == 0


def test_implicit_self_and_cls_are_not_counted_as_unannotated_work():
    """154 of the plan's 368 unannotated parameters are these, and they are not work.

    Audit E.2 measured 119 type-checker errors from annotating `self` on the
    runtime mixins, so this population must never be converted. Counting it
    would put a floor under the burn-down that no PR can lift.
    """
    census = _census_of("""
        class Engine:
            def method(self, positions: Array) -> None: ...

            @classmethod
            def build(cls, order) -> None: ...
        """)
    assert census.unannotated == 1  # `order`, and not `self` or `cls`
    assert census.bare == 1


def test_a_parameter_actually_named_self_outside_a_method_is_still_exempt():
    """The exemption is positional, matching `test_type_annotation_guard.py`.

    Pinned so the shared rule stays visible: both walk the AST with no class
    context, so both key off the leading position rather than off the enclosing
    scope. If that ever diverges, the two files stop measuring the same package.
    """
    census = _census_of("def free(self, other) -> None: ...")
    assert census.unannotated == 1


def test_a_decorated_function_is_vacuous_until_some_parameter_has_a_shape():
    """Phase 1's counter: `@jaxtyped` that checks nothing about shape."""
    census = _census_of("""
        @jaxtyped(typechecker=beartype)
        def unshaped(coefficients: Array) -> None: ...

        @jaxtyped(typechecker=beartype)
        def shaped(positions: Float[Array, "n 3"]) -> None: ...
        """)
    assert census.decorated == 2
    assert census.vacuous_decorated == 1


def test_experimental_is_excluded_by_default_as_everywhere_else_in_the_audit():
    """`jaccpot/experimental/` is prototype code, deselected and uncovered."""
    included = annotation_census._iter_module_paths(
        annotation_census.PACKAGE_ROOT, include_experimental=False
    )
    assert included, "the package walk found no modules at all"
    assert not any("experimental" in p.parts for p in included)

    with_experimental = annotation_census._iter_module_paths(
        annotation_census.PACKAGE_ROOT, include_experimental=True
    )
    assert any(
        "experimental" in p.parts for p in with_experimental
    ), "jaccpot/experimental/ was not found, so the exclusion above proves nothing"


def test_the_reconciliation_ladders_only_ever_loosen():
    """Each rung adds a population, so the counts must be non-decreasing.

    A rung that went down would mean a rung excludes something an earlier one
    counted, which would make the ladder unable to attribute the gap -- its only
    purpose.
    """
    for population, ladder in annotation_census.reconciliation().items():
        counts = list(ladder.values())
        assert counts == sorted(
            counts
        ), f"{population} ladder is not monotone: {counts}"
        assert counts[0] < counts[-1], f"{population} ladder never widens: {counts}"


def test_the_package_census_agrees_with_its_own_first_ladder_rung():
    """The headline number and the reconciliation must not drift apart.

    They are computed by separate walks -- deliberately, so the ladder is an
    independent check rather than a restatement -- which is exactly why they need
    pinning to each other.
    """
    _, totals = annotation_census.census_package()
    ladder = annotation_census.reconciliation()
    assert totals.bare == next(iter(ladder["bare Array"].values()))
    assert totals.unannotated == next(iter(ladder["unannotated"].values()))

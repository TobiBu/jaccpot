"""The pilot's accounting, which is the part that can flatter.

`bench/annotation_pilot.py` measures how much of a module is already validated,
and that number decides how much annotation work the rollout still has. Every
way this tool can be wrong makes the remaining work look SMALLER: a control
failure counted as a rejection, an unreplayable function counted as validated, a
typo'd target that simply never fires. So the three "uncounted" paths are pinned
here, in the direction that matters.

The perturbation set is pinned too. It is the difference between measuring
something and measuring nothing, and a perturbation that produces a degenerate
array -- shrinking an axis of extent 1 to 0 -- would be accepted by kernels that
legitimately handle empty input, inflating the silently-accepted count with
false positives.
"""

from __future__ import annotations

import pytest

from bench import annotation_pilot


class _FakeArray:
    """A stand-in carrying only `.shape` and `.dtype`.

    Parameters
    ----------
    shape : tuple of int
        Shape to report.
    dtype : str
        Dtype to report.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str = "float64") -> None:
        self.shape = shape
        self.dtype = dtype


def test_an_array_is_described_by_shape_not_value():
    """A tracer has shape and dtype and no value, which is why values are unused."""
    kind, shape, dtype = annotation_pilot.describe_argument(_FakeArray((4, 3), "int32"))
    assert (kind, shape, dtype) == ("array", (4, 3), "int32")


def test_scalars_survive_but_objects_make_a_call_unreplayable():
    """Non-array arguments stay concrete inside a trace; arbitrary objects do not."""
    assert annotation_pilot.describe_argument(7)[:2] == ("scalar", 7)
    assert annotation_pilot.describe_argument((2, 4))[:2] == ("tuple", (2, 4))
    assert annotation_pilot.describe_argument(object())[0] == "opaque"


def test_a_function_whose_control_fails_is_counted_in_neither_direction():
    """The failure mode that would understate the remaining work.

    Synthesized zeros are not always a valid stand-in. A kernel that rejects them
    for its own reasons must be reported INCONCLUSIVE, never silently folded into
    "already validated" -- that would make the annotation backlog look smaller
    than it is, which is the one error this measurement must not make.
    """

    def always_raises(x):
        raise RuntimeError("value-dependent kernel")

    import bench.annotation_pilot as module

    module.always_raises = always_raises
    recorded = {
        f"{module.__name__}:always_raises": [({"x": ("array", (4,), "float64")}, True)]
    }
    report, tally = annotation_pilot.replay(recorded)

    assert tally["inconclusive"] == 1
    assert tally["accepted"] == 0 and tally["rejected"] == 0
    assert "INCONCLUSIVE" in report


def test_a_function_with_an_opaque_argument_is_counted_in_neither_direction():
    """Same direction: unreplayable must not read as validated."""

    def takes_an_object(cfg, x):
        return x

    import bench.annotation_pilot as module

    module.takes_an_object = takes_an_object
    recorded = {
        f"{module.__name__}:takes_an_object": [
            (
                {
                    "cfg": ("opaque", None, "TreeConfig"),
                    "x": ("array", (4,), "float64"),
                },
                False,
            )
        ]
    }
    report, tally = annotation_pilot.replay(recorded)

    assert tally["unreplayable"] == 1
    assert tally["accepted"] == 0 and tally["rejected"] == 0
    assert "UNREPLAYABLE" in report


def test_a_permissive_function_is_reported_as_silently_accepting():
    """The measurement itself: a function that checks nothing must score 0% rejected."""

    def checks_nothing(x):
        return 0

    import bench.annotation_pilot as module

    module.checks_nothing = checks_nothing
    recorded = {
        f"{module.__name__}:checks_nothing": [
            ({"x": ("array", (4, 3), "float64")}, True)
        ]
    }
    report, tally = annotation_pilot.replay(recorded)

    assert tally["rejected"] == 0
    assert tally["accepted"] == len(annotation_pilot.shape_perturbations((4, 3)))
    assert ">> ACCEPTED" in report


def test_a_strict_function_is_reported_as_rejecting():
    """The other direction, so the tool cannot pass by calling everything permissive."""

    def insists_on_the_shape(x):
        if x.shape != (4, 3):
            raise ValueError("x must be (4, 3)")
        return 0

    import bench.annotation_pilot as module

    module.insists_on_the_shape = insists_on_the_shape
    recorded = {
        f"{module.__name__}:insists_on_the_shape": [
            ({"x": ("array", (4, 3), "float64")}, True)
        ]
    }
    _, tally = annotation_pilot.replay(recorded)

    assert tally["accepted"] == 0
    assert tally["rejected"] == len(annotation_pilot.shape_perturbations((4, 3)))


def test_no_perturbation_produces_a_degenerate_array():
    """An axis of extent 1 must not be shrunk to 0.

    Many kernels accept empty input legitimately, so a zero-extent perturbation
    would be counted as a silent acceptance that is really correct behaviour --
    inflating the number in the direction that argues for more annotation work.
    """
    for shape in [(1,), (1, 3), (4, 1), (1, 1), (5, 3)]:
        for _, perturbed in annotation_pilot.shape_perturbations(shape):
            assert all(dim > 0 for dim in perturbed), f"{shape} -> {perturbed}"


def test_every_perturbation_actually_changes_the_shape():
    """A no-op perturbation would be counted as an acceptance and mean nothing."""
    for shape in [(3,), (4, 3), (2, 5, 3)]:
        for label, perturbed in annotation_pilot.shape_perturbations(shape):
            assert perturbed != shape, f"{label} left {shape} unchanged"


def test_a_tuple_of_arrays_is_a_container_not_an_opaque_object():
    """`_block_tile` takes `a_xyz: tuple[Array, Array, Array]`, and it is describable.

    The old rule accepted a tuple only when every element was a scalar, so a tuple
    of arrays fell through to `opaque` and took the whole function out of the
    measurement. Nothing about it is actually opaque: each element has a shape and
    a dtype, which is the entire input the replay needs.
    """
    kind, elements, meta = annotation_pilot.describe_argument(
        (_FakeArray((4,)), _FakeArray((4,)), _FakeArray((4,)))
    )

    assert (kind, meta) == ("container", "tuple")
    assert elements == [("array", (4,), "float64")] * 3


def test_a_tuple_of_scalars_is_still_a_tuple():
    """The narrower description is kept where it applies, so old recordings replay."""
    assert annotation_pilot.describe_argument((2, 4))[:2] == ("tuple", (2, 4))


def test_a_container_holding_an_opaque_element_is_still_unreplayable():
    """Widening must not swallow the thing it was meant to report.

    The opaque check has to recurse, or a tuple with one un-describable element
    reads as fully replayable and its control fails later as INCONCLUSIVE -- the
    same understatement, relabelled.
    """
    kind, elements, _ = annotation_pilot.describe_argument((_FakeArray((4,)), object()))

    assert kind == "container"
    assert annotation_pilot.is_replayable(("container", elements, "tuple")) is False
    assert annotation_pilot.is_replayable(("container", elements[:1], "tuple")) is True


def test_a_dtype_argument_is_described_rather_than_making_the_call_opaque():
    """`_pad_inputs` takes a working `dtype`, which is neither array nor scalar."""
    import numpy as np

    for spelling in [np.dtype("float64"), np.float32]:
        kind, value, _ = annotation_pilot.describe_argument(spelling)
        assert kind == "dtype", spelling
        assert value in ("float64", "float32"), spelling


def test_a_string_is_still_a_scalar_and_not_mistaken_for_a_dtype():
    """`np.dtype('float64')` accepts a string, so ordering is load-bearing here.

    `None` matters more: `np.dtype(None)` is `float64`, so an `Optional[Array]`
    argument recorded as `None` would come back as a dtype and be rebuilt as one.
    """
    assert annotation_pilot.describe_argument("float64")[:2] == ("scalar", "float64")
    assert annotation_pilot.describe_argument(None)[:2] == ("scalar", None)


def test_arrays_inside_a_container_are_perturbed_and_not_merely_rebuilt():
    """The failure mode this widening could introduce, in the same direction as the rest.

    A container that rebuilds but is never perturbed makes an UNREPLAYABLE function
    -- which is reported, and uncounted -- into one that passes its control, prints
    "control OK", and contributes nothing. That is strictly worse: it converts a
    visible gap into an invisible one. So the perturbation has to address arrays by
    path, and this pins that it does.
    """

    def container_checks_nothing(xyz, m):
        return 0

    import bench.annotation_pilot as module

    module.container_checks_nothing = container_checks_nothing
    per_array = len(annotation_pilot.shape_perturbations((4,)))
    recorded = {
        f"{module.__name__}:container_checks_nothing": [
            (
                {
                    "xyz": ("container", [("array", (4,), "float64")] * 3, "tuple"),
                    "m": ("array", (4,), "float64"),
                },
                True,
            )
        ]
    }
    report, tally = annotation_pilot.replay(recorded)

    assert tally["rejected"] == 0
    assert tally["accepted"] == 4 * per_array, "a container's arrays were not perturbed"
    assert (
        "xyz[0]" in report and "xyz[2]" in report
    ), "the report cannot name the element"


def test_a_strict_function_still_rejects_when_the_bad_shape_is_inside_a_container():
    """The other direction, so the container path cannot pass by accepting everything."""

    def insists_on_the_component(xyz):
        for component in xyz:
            if component.shape != (4,):
                raise ValueError("each component must be (4,)")
        return 0

    import bench.annotation_pilot as module

    module.insists_on_the_component = insists_on_the_component
    recorded = {
        f"{module.__name__}:insists_on_the_component": [
            ({"xyz": ("container", [("array", (4,), "float64")] * 3, "tuple")}, True)
        ]
    }
    _, tally = annotation_pilot.replay(recorded)

    assert tally["accepted"] == 0
    assert tally["rejected"] == 3 * len(annotation_pilot.shape_perturbations((4,)))

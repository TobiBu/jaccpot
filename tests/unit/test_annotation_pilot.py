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

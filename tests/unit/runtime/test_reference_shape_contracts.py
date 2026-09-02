"""The oracle's shape contract, and the wrong answer it used to return.

`runtime/reference.py` says of itself: *"this is the oracle the FMM paths are
checked against."* That is what makes a silent shape hole here worse than one
anywhere else -- a reference that accepts a malformed evaluation point and
returns numbers can only make a wrong FMM look right, or a right one look wrong.

A pilot over Phase 1's whole `runtime/` group measured 23 malformed inputs
against `main`. Twenty were already rejected, most with precise domain errors
(`"velocities must have shape (8, 3), got (7, 3)"`,
`"target_indices must contain integer values"`), so `fmm_derivatives.py`,
`fmm_prepare.py` and `_fmm_impl.py` were deliberately left alone -- annotating
them would replace those messages with a generic `TypeCheckError`.

The three that were accepted silently were all here, and one of them was a
wrong answer rather than a tolerated spelling. That one is pinned first.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime import reference
from jaccpot.runtime._fmm_impl import FMMEngine

_N = 8


@pytest.fixture
def sources():
    """Eight source particles.

    Returns
    -------
    tuple
        ``(positions, masses)`` with shapes ``(8, 3)`` and ``(8,)``.
    """
    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.standard_normal((_N, 3)))
    masses = jnp.asarray(np.abs(rng.standard_normal(_N)) + 0.5)
    return positions, masses


def test_a_bare_three_vector_is_not_three_evaluation_points(sources):
    """The measured wrong answer, and the reason this file exists.

    `compute_gravitational_potential` is vectorised over `eval_points`, so a
    `(3,)` argument was read as THREE points whose coordinates were the
    components of the one point intended. Measured on `main` it returned
    `[-7.84, -7.91, -7.73]` where the correct answer is the single value
    `-8.22` -- three numbers, not one of them right.
    """
    positions, masses = sources
    with pytest.raises(Exception, match="(?i)eval_points|shape|dim"):
        reference.compute_gravitational_potential(
            positions, masses, jnp.asarray([0.1, 0.2, 0.3])
        )


def test_the_correct_spelling_still_returns_one_value_per_point(sources):
    """The counterpart, so the rejection above cannot pass by breaking the API."""
    positions, masses = sources
    one = reference.compute_gravitational_potential(
        positions, masses, jnp.asarray([[0.1, 0.2, 0.3]])
    )
    assert one.shape == (1,)
    four = reference.compute_gravitational_potential(
        positions, masses, jnp.zeros((4, 3))
    )
    assert four.shape == (4,)
    assert np.all(np.isfinite(np.asarray(one)))


def test_a_single_evaluation_point_must_be_rank_one(sources):
    """`direct_sum` had the mirror hole; it was benign, and is closed anyway.

    A `(1, 3)` point broadcast to the same answer here, so this tightens a
    tolerant signature rather than fixing a wrong result. It is closed because
    the same spelling in the sibling above was **not** benign, and a reference
    module whose two entry points disagree about what "a point" means is a trap
    regardless of which one currently returns the right number.
    """
    positions, masses = sources
    with pytest.raises(Exception, match="(?i)eval_point|shape|dim"):
        reference.direct_sum(positions, masses, jnp.zeros((1, 3)))


def test_sources_must_agree_in_length(sources):
    """`n` is shared across `positions` and `masses`, so this is now one check.

    On `main` this surfaced as `TypeError: mul got incompatible shapes for
    broadcasting: (8, 3), (7, 1)` -- an XLA message naming neither parameter
    nor the mismatch a caller would recognise.
    """
    positions, masses = sources
    with pytest.raises(Exception, match="(?i)masses|positions|shape|dim"):
        reference.direct_sum(positions, masses[:-1], jnp.asarray([0.1, 0.2, 0.3]))


def test_the_scalar_constants_still_take_python_numbers(sources):
    """`G` and `softening` stay bare, and this is why.

    Both default to Python floats. `Float[Array, ""]` would reject the default
    path itself, which is the same reason `near_field.py`'s `G` is exempt.
    Pinned so a later consistency sweep cannot quietly annotate them.
    """
    positions, masses = sources
    point = jnp.asarray([0.1, 0.2, 0.3])
    from_defaults = reference.direct_sum(positions, masses, point)
    explicit = reference.direct_sum(positions, masses, point, G=1.0, softening=0.0)
    assert np.allclose(np.asarray(from_defaults), np.asarray(explicit))
    as_arrays = reference.direct_sum(
        positions, masses, point, G=jnp.asarray(1.0), softening=jnp.asarray(0.0)
    )
    assert np.allclose(np.asarray(from_defaults), np.asarray(as_arrays))


def test_the_oracle_still_computes_the_newtonian_sum(sources):
    """The numbers must not move -- only the rejections.

    Checked against a hand-written Newtonian sum rather than against a stored
    value, so the assertion states the physics instead of restating the
    implementation.

    The evaluation point is deliberately NOT one of the sources: with the
    default `softening=0.0` the self term is `0/0`, so `direct_sum` evaluated
    at a particle returns `nan`. That is the function's documented behaviour and
    not this file's subject, but it is worth recording, because an "evaluate at
    every particle" test written the obvious way asserts against `nan` and can
    only pass by accident.
    """
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    masses = jnp.asarray([2.0, 3.0])
    midpoint = jnp.asarray([0.5, 0.0, 0.0])

    got = reference.direct_sum(positions, masses, midpoint)

    offsets = np.asarray(midpoint) - np.asarray(positions)
    distances = np.linalg.norm(offsets, axis=1)
    expected = -np.sum(
        np.asarray(masses)[:, None] * offsets / distances[:, None] ** 3, axis=0
    )
    assert np.allclose(np.asarray(got), expected)


def test_evaluating_at_a_source_is_still_nan_without_softening(sources):
    """Pins the caveat above, so the test that avoids it keeps its reason."""
    positions, masses = sources
    at_a_source = reference.direct_sum(positions, masses, positions[0])
    assert np.all(np.isnan(np.asarray(at_a_source)))


def test_the_sweeps_facade_inherits_the_same_contract(sources):
    """`SweepsMixin` delegates to these functions, so it must not be a way around them.

    Its three diagnostic methods pass `positions`, `masses` and `eval_point`
    straight through to `reference.py`, which is why annotating the reference
    functions covers both -- the capture helper cannot patch class methods, so
    the shapes here were derived from the delegated calls rather than from the
    methods themselves. This test is what makes that reasoning checkable instead
    of merely stated.
    """
    positions, masses = sources
    engine = FMMEngine()

    with pytest.raises(Exception, match="(?i)eval_point|shape|dim"):
        engine.direct_sum(positions, masses, jnp.zeros((1, 3)))

    good = engine.direct_sum(positions, masses, jnp.asarray([0.1, 0.2, 0.3]))
    assert good.shape == (3,)

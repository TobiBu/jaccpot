"""Tripwires for the three sign/ordering conventions that collide in this package.

Each convention here is a pair of things with the SAME shape and INCOMPATIBLE
meaning. Getting one backwards is not a crash and not a shape error -- it is a
silently degraded force, which is the worst failure mode this library has
(CLAUDE.md's first principle). The audit that turned them up is
``docs/operator_conventions.md``.

Every test below is written in two halves:

* the documented convention reproduces an exact direct-sum reference (or an
  algebraic identity) to a tight tolerance; and
* **the plausible alternative does not**, asserted explicitly.

The second half is the point. A test that only pins the right answer passes just
as happily when both branches agree, which is exactly the situation where the
convention has stopped being load-bearing and the tripwire has gone vacuous.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.operators.multipole_utils import multi_index_tuples
from jaccpot.operators.real_harmonics import evaluate_local_real, p2m_real_direct
from jaccpot.operators.real_rotations import (
    real_rotation_from_z_axis_local,
    real_rotation_from_z_axis_multipole,
    real_rotation_to_z_axis_local,
    real_rotation_to_z_axis_multipole,
)
from jaccpot.operators.real_translations import l2l_real, m2l_real, m2m_real
from jaccpot.operators.symmetric_tensors import symmetric_multi_indices_3d

# Enough order that a correct translation is limited by float64, not truncation,
# so "correct" and "sign-flipped" differ by ten orders of magnitude rather than
# by a factor of a few.
ORDER = 6

# A well-separated, deliberately asymmetric configuration. Asymmetric matters: on
# an axis-aligned or symmetric layout several of these conventions coincide, and
# the test would pass under a flipped sign.
SOURCE_CENTER = np.array([0.10, -0.05, 0.02])
PARENT_CENTER = np.array([0.0, 0.0, 0.0])
PARTICLE = np.array([0.13, -0.02, 0.06])
MASS = 0.83
TARGET_CENTER = np.array([4.0, 1.5, -2.5])
EVAL_POINT = np.array([4.03, 1.47, -2.52])

# A correct round trip lands at ~1e-12; a flipped sign at ~1e-2. Two decades of
# slack on each side, so neither half of a test is brittle.
TIGHT = 1e-9
CLEARLY_WRONG = 1e-4


def _exact_potential() -> float:
    """Direct-sum potential of the single source particle at the eval point.

    Returns
    -------
    float
        ``m / |eval - particle|``, the reference every translation chain below is
        compared against.
    """
    return float(MASS / np.linalg.norm(EVAL_POINT - PARTICLE))


def _multipole_about(center: np.ndarray) -> jnp.ndarray:
    """P2M the single source particle about ``center``.

    Parameters
    ----------
    center : np.ndarray
        Expansion centre.

    Returns
    -------
    jnp.ndarray
        Packed real multipole coefficients.
    """
    return p2m_real_direct(
        jnp.asarray(PARTICLE - center), jnp.asarray(MASS), order=ORDER
    )


def _evaluate_local(local: jnp.ndarray, center: np.ndarray) -> float:
    """Evaluate a local expansion at :data:`EVAL_POINT`.

    The L2P offset is ``centre - eval``, itself a convention worth stating: the
    opposite sign also runs and lands ~1.5e-2 off.

    Parameters
    ----------
    local : jnp.ndarray
        Packed real local coefficients.
    center : np.ndarray
        Centre the expansion is about.

    Returns
    -------
    float
        The evaluated potential.
    """
    return float(
        evaluate_local_real(local, jnp.asarray(center - EVAL_POINT), order=ORDER)
    )


def _relative_error(got: float) -> float:
    """Relative error of ``got`` against the direct-sum reference.

    Parameters
    ----------
    got : float
        Potential produced by a translation chain.

    Returns
    -------
    float
        ``|got - exact| / exact``.
    """
    exact = _exact_potential()
    return abs(got - exact) / abs(exact)


# ---------------------------------------------------------------------------
# Convention 1: the two translation sign conventions are OPPOSITE.
#
#   M2M, L2L (same-type)  delta = source_centre - destination_centre
#   M2L      (multipole -> local)  delta = destination_centre - source_centre
# ---------------------------------------------------------------------------


def _m2l_then_evaluate(delta: np.ndarray) -> float:
    """M2L a source multipole with ``delta``, then evaluate the local.

    Parameters
    ----------
    delta : np.ndarray
        Displacement handed to :func:`m2l_real`.

    Returns
    -------
    float
        The evaluated potential.
    """
    local = m2l_real(_multipole_about(SOURCE_CENTER), jnp.asarray(delta), order=ORDER)
    return _evaluate_local(local, TARGET_CENTER)


def test_m2l_delta_is_destination_minus_source() -> None:
    right = _m2l_then_evaluate(TARGET_CENTER - SOURCE_CENTER)
    assert _relative_error(right) < TIGHT


def test_m2l_delta_reversed_is_clearly_wrong() -> None:
    """The tripwire half: ``source - destination`` must NOT also work."""
    flipped = _m2l_then_evaluate(SOURCE_CENTER - TARGET_CENTER)
    assert _relative_error(flipped) > CLEARLY_WRONG


def _m2m_then_evaluate(delta: np.ndarray) -> float:
    """Re-centre a multipole onto the parent with ``delta``, then evaluate.

    Parameters
    ----------
    delta : np.ndarray
        Displacement handed to :func:`m2m_real`.

    Returns
    -------
    float
        The evaluated potential.
    """
    parent = m2m_real(_multipole_about(SOURCE_CENTER), jnp.asarray(delta), order=ORDER)
    local = m2l_real(parent, jnp.asarray(TARGET_CENTER - PARENT_CENTER), order=ORDER)
    return _evaluate_local(local, TARGET_CENTER)


def test_m2m_delta_is_source_minus_destination() -> None:
    right = _m2m_then_evaluate(SOURCE_CENTER - PARENT_CENTER)
    assert _relative_error(right) < TIGHT


def test_m2m_delta_reversed_is_clearly_wrong() -> None:
    """The tripwire half: ``destination - source`` must NOT also work."""
    flipped = _m2m_then_evaluate(PARENT_CENTER - SOURCE_CENTER)
    assert _relative_error(flipped) > CLEARLY_WRONG


def _l2l_then_evaluate(delta: np.ndarray, child_center: np.ndarray) -> float:
    """Push a local from the target centre down to ``child_center``.

    Parameters
    ----------
    delta : np.ndarray
        Displacement handed to :func:`l2l_real`.
    child_center : np.ndarray
        Centre the pushed-down expansion is about.

    Returns
    -------
    float
        The evaluated potential.
    """
    parent_local = m2l_real(
        _multipole_about(SOURCE_CENTER),
        jnp.asarray(TARGET_CENTER - SOURCE_CENTER),
        order=ORDER,
    )
    child_local = l2l_real(parent_local, jnp.asarray(delta), order=ORDER)
    return _evaluate_local(child_local, child_center)


def test_l2l_delta_is_source_minus_destination() -> None:
    child = TARGET_CENTER + np.array([0.05, -0.03, 0.02])
    right = _l2l_then_evaluate(TARGET_CENTER - child, child)
    assert _relative_error(right) < TIGHT


def test_l2l_delta_reversed_is_clearly_wrong() -> None:
    """The tripwire half: ``destination - source`` must NOT also work."""
    child = TARGET_CENTER + np.array([0.05, -0.03, 0.02])
    flipped = _l2l_then_evaluate(child - TARGET_CENTER, child)
    assert _relative_error(flipped) > CLEARLY_WRONG


def test_the_two_translation_conventions_really_are_opposite() -> None:
    """State the collision itself, not just each side of it.

    If someone ever unifies the two conventions, the individual tests above still
    pass on whichever side they unified to -- this one is what notices that the
    asymmetry is gone.
    """
    child = TARGET_CENTER + np.array([0.05, -0.03, 0.02])

    # Same-type translations want source - destination ...
    assert _relative_error(_m2m_then_evaluate(SOURCE_CENTER - PARENT_CENTER)) < TIGHT
    assert _relative_error(_l2l_then_evaluate(TARGET_CENTER - child, child)) < TIGHT

    # ... while M2L wants destination - source, i.e. the other way round.
    assert _relative_error(_m2l_then_evaluate(TARGET_CENTER - SOURCE_CENTER)) < TIGHT

    # And the collision stated as such: applying either family's rule to the
    # other family's operator is wrong. This is the assertion that fails if the
    # two conventions are ever unified, whichever side they are unified to.
    m2l_rule_applied_to_m2m = _m2m_then_evaluate(PARENT_CENTER - SOURCE_CENTER)
    same_type_rule_applied_to_m2l = _m2l_then_evaluate(SOURCE_CENTER - TARGET_CENTER)
    assert _relative_error(m2l_rule_applied_to_m2m) > CLEARLY_WRONG
    assert _relative_error(same_type_rule_applied_to_m2l) > CLEARLY_WRONG


# ---------------------------------------------------------------------------
# Convention 2: two packed symmetric-tensor orderings, exact reverses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", list(range(9)))
def test_packed_orderings_are_exact_reverses(order: int) -> None:
    """``symmetric_tensors`` descends where ``multipole_utils`` ascends.

    Both enumerate the same multi-index set with the same length, so an index
    from one used against the other reads a real component from the wrong slot
    with nothing to catch it.

    Parameters
    ----------
    order : int
        Tensor order under test.
    """
    descending = symmetric_multi_indices_3d(order)
    ascending = multi_index_tuples(order)

    assert set(descending) == set(ascending)
    assert len(descending) == len(ascending)
    assert descending == tuple(reversed(ascending))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_packed_orderings_are_not_interchangeable(order: int) -> None:
    """The tripwire half: they must NOT coincide.

    Parameters
    ----------
    order : int
        Tensor order under test.
    """
    assert symmetric_multi_indices_3d(order) != multi_index_tuples(order)


# ---------------------------------------------------------------------------
# Convention 3: four rotation builders, one signature.
# ---------------------------------------------------------------------------

# Off-axis and non-degenerate: on an axis-aligned direction several of these
# blocks coincide and the distinctness assertions below would pass vacuously.
_ROT_DIRECTION = (0.7, -1.3, 2.1)


def _rotation_blocks(ell: int) -> dict[str, np.ndarray]:
    """Build all four rotation blocks for one degree.

    Parameters
    ----------
    ell : int
        Degree ``l``.

    Returns
    -------
    dict[str, np.ndarray]
        The four ``[2l+1, 2l+1]`` blocks, keyed by direction and representation.
    """
    # As arrays, because that is what the builders declare and what production
    # passes: every call site in `operators/` supplies `delta[0], delta[1],
    # delta[2]` -- traced scalars, never Python floats. Passing floats here made
    # these six the last failures of `JACCPOT_RUNTIME_TYPECHECK=1 pytest
    # tests/unit` (audit F40); beartype rejected them at the signature, so the
    # test never reached the builders at all. Verified bit-identical to the
    # float form, so the contracts below assert exactly what they did before.
    x, y, z = (jnp.asarray(v, dtype=jnp.float64) for v in _ROT_DIRECTION)
    return {
        name: np.asarray(fn(x, y, z, ell, dtype=jnp.float64))
        for name, fn in (
            ("to_z_multipole", real_rotation_to_z_axis_multipole),
            ("from_z_multipole", real_rotation_from_z_axis_multipole),
            ("to_z_local", real_rotation_to_z_axis_local),
            ("from_z_local", real_rotation_from_z_axis_local),
        )
    }


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_the_four_rotation_builders_are_pairwise_distinct(ell: int) -> None:
    """No two of the four are the same matrix.

    They share the ``(x, y, z, ell, dtype)`` signature and differ only in
    direction (world->z vs z->world) and representation (multipole vs local), so
    a caller can silently pick the wrong one. If any pair ever coincides, the
    four-way distinction has stopped being real and callers can no longer rely on
    the docstrings that name them.

    Parameters
    ----------
    ell : int
        Degree ``l``.
    """
    blocks = _rotation_blocks(ell)
    names = sorted(blocks)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            spread = np.max(np.abs(blocks[first] - blocks[second]))
            assert spread > 1e-3, f"{first} and {second} coincide at ell={ell}"


@pytest.mark.parametrize("ell", [1, 2, 3])
def test_to_and_from_z_are_inverses_within_each_representation(ell: int) -> None:
    """``from_z @ to_z == I``, separately for multipoles and locals.

    This is what makes "to" and "from" meaningful names. It also pins that the
    pairing is WITHIN a representation: the M2L sandwich deliberately crosses
    them (multipole world->z, then local z->world), so those two are not
    inverses and must not be assumed interchangeable.

    Parameters
    ----------
    ell : int
        Degree ``l``.
    """
    blocks = _rotation_blocks(ell)
    identity = np.eye(2 * ell + 1)

    for representation in ("multipole", "local"):
        to_z = blocks[f"to_z_{representation}"]
        from_z = blocks[f"from_z_{representation}"]
        assert np.max(np.abs(from_z @ to_z - identity)) < 1e-12
        assert np.max(np.abs(to_z @ from_z - identity)) < 1e-12

    # The M2L sandwich pair is NOT an inverse pair.
    cross = blocks["from_z_local"] @ blocks["to_z_multipole"]
    assert np.max(np.abs(cross - identity)) > 1e-3


def test_conventions_are_exercised_in_float64() -> None:
    """Guard the guards: these tolerances assume x64 is enabled.

    Under float32 a correct round trip lands near ``1e-5``, which is still below
    ``CLEARLY_WRONG`` but far above ``TIGHT`` -- so the suite would fail loudly
    rather than silently weaken, and this test says why.
    """
    assert jax.config.jax_enable_x64, (
        "these convention tripwires are calibrated for float64; run the suite "
        "with JAX_ENABLE_X64=1"
    )

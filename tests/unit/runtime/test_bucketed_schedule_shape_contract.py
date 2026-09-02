"""The bucketed scatter schedule's shape contract, which was enforced by nothing.

`PrepareMixin._prepare_bucketed_scatter_schedules_safe` builds a `precomputed_*` schedule
-- the family STYLE_GUIDE section 4.1 names as this package's unvalidated one -- from
`target_leaf_ids` and `valid_pairs`, one entry per candidate leaf pair. The two must agree
in length, and nothing checked that.

It matters more than a schedule usually would because the method is `_safe` by design: it
swallows failures and returns `None`, so the caller falls back to the unscheduled path.
Measured on the parent commit, all five malformations were accepted, in three different
ways:

    valid_pairs one element short      3 of 3 built, output IDENTICAL (harmless)
    valid_pairs (pairs, 1)             schedule SILENTLY DROPPED, 0 of 3
    target_leaf_ids one element short  3 of 3 built, output DIFFERENT
    target_leaf_ids (1, pairs)         schedule SILENTLY DROPPED, 0 of 3
    valid_pairs float, not bool        3 of 3 built, output IDENTICAL (harmless)

The third is the one this file exists for: a length-mismatched `target_leaf_ids` builds a
DIFFERENT schedule and says nothing, which is a wrong optimisation rather than a slow one.
The two "silently dropped" rows are a performance regression that no test would notice
either, since the fallback produces the same numbers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from jaccpot import FastMultipoleMethod

N = 256
LEAF = 32
PAIRS = 64


@pytest.fixture(scope="module")
def engine_and_interop():
    """A real engine and a real near-field interop payload.

    A stand-in is not usable here: the helper reads the interop payload's own arrays, so
    the schedule it builds depends on them. See ``jaccpot-test-doubles-should-be-real``.

    Returns
    -------
    tuple
        ``(engine, nearfield_interop)``.
    """
    rng = np.random.default_rng(3)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(N, 3)), dtype=jnp.float64)
    masses = jnp.asarray(np.abs(rng.standard_normal(N)) + 0.5, dtype=jnp.float64)
    fmm = FastMultipoleMethod(basis="real", theta=0.5, G=1.0, softening=1e-2)
    state = fmm.prepare_state(positions, masses, max_order=2, leaf_size=LEAF)
    return fmm._impl, state.nearfield_interop


def _run(engine, interop, **overrides):
    """Invoke the schedule builder, substituting ``overrides``.

    Parameters
    ----------
    engine : Any
        The engine holding the mixin.
    interop : Any
        The near-field interop payload.
    **overrides : Array
        Arguments to replace, by parameter name.

    Returns
    -------
    tuple
        The three optional schedule arrays.
    """
    rng = np.random.default_rng(3)
    arguments = dict(
        target_leaf_ids=jnp.asarray(
            rng.integers(0, N // LEAF, size=(PAIRS,)), dtype=jnp.int32
        ),
        valid_pairs=jnp.ones((PAIRS,), dtype=bool),
    )
    arguments.update(overrides)
    return engine._prepare_bucketed_scatter_schedules_safe(
        nearfield_interop=interop,
        leaf_cap=LEAF,
        edge_chunk_size=PAIRS,
        **arguments,
    )


def test_the_well_formed_call_still_builds_all_three_schedules(engine_and_interop):
    """Non-vacuity: every rejection below is meaningless if the good call builds nothing."""
    engine, interop = engine_and_interop
    built = _run(engine, interop)
    assert sum(entry is not None for entry in built) == 3


@pytest.mark.parametrize(
    "label,override",
    [
        ("target_leaf_ids one element short", "short_ids"),
        ("target_leaf_ids with an extra leading axis", "rank_ids"),
        ("valid_pairs one element short", "short_valid"),
        ("valid_pairs with an extra trailing axis", "rank_valid"),
    ],
)
def test_a_mismatched_schedule_input_is_rejected(engine_and_interop, label, override):
    """A length or rank mismatch must not reach the schedule builder.

    ``target_leaf_ids`` one element short is the sharp case: on the parent it built a
    schedule that differed from the correct one, silently. The two rank cases were
    swallowed by the ``_safe`` wrapper and returned ``None``, losing the schedule
    altogether -- which no numerical test can see, because the fallback path computes the
    same answer more slowly.
    """
    engine, interop = engine_and_interop
    rng = np.random.default_rng(3)
    ids = jnp.asarray(rng.integers(0, N // LEAF, size=(PAIRS,)), dtype=jnp.int32)
    valid = jnp.ones((PAIRS,), dtype=bool)
    overrides = {
        "short_ids": dict(target_leaf_ids=ids[:-1]),
        "rank_ids": dict(target_leaf_ids=ids[None, :]),
        "short_valid": dict(valid_pairs=valid[:-1]),
        "rank_valid": dict(valid_pairs=valid[:, None]),
    }[override]
    with pytest.raises(TypeCheckError):
        _run(engine, interop, **overrides)


def test_the_validity_mask_must_be_boolean(engine_and_interop):
    """The one thing this change makes stricter rather than safer, pinned deliberately.

    A float 0/1 mask worked on the parent and produced an identical schedule, so this
    rejection is a tightened contract and not a caught defect. It is asserted so the
    trade is visible: `valid_pairs` is a mask, every caller passes a bool, and the
    alternative -- `Num[...]` -- would accept the integer and float spellings of a
    mask that the rest of the near field never uses.
    """
    engine, interop = engine_and_interop
    with pytest.raises(TypeCheckError):
        _run(engine, interop, valid_pairs=jnp.ones((PAIRS,), dtype=jnp.float64))


def test_the_shapes_this_pins_are_the_ones_production_passes(engine_and_interop):
    """Guard against the annotation being right about a shape nothing uses.

    ``target_leaf_ids`` is one entry per candidate leaf PAIR, not per particle -- the
    docstring said particle until the capture measured it at 4032 to 55550 against
    particle counts of 512 to 4096. This pins the agreement rather than the extent.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip("the fixture builds a float64 state")
    engine, interop = engine_and_interop
    built = _run(engine, interop)
    assert all(entry is not None for entry in built)

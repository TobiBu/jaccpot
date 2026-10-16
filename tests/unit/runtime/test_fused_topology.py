"""``FusedTopology`` and the force parameters, as containers (S2).

No solver here: the topology's capacity arithmetic runs on stand-in states that
hold only the fields it reads, so each overflow bit is set by the one capacity it
names and the verdict mask is checked bit by bit. The provider tests
(``test_fused_provider.py``) check the same bits on real rebuilds.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime import fused_topology as ft
from jaccpot.runtime.capacity_guard import WALK_NEEDS_FIELDS


def _prepared(*, offsets=(0, 2, 5), far_count=None, far_width=8, padded=None):
    """What ``overflow_bits`` reads of a prepared state: two leaves with 2 and 3
    near-field rows, optionally a far list and a rectangle layout."""
    far = None
    if far_count is not None:
        far = SimpleNamespace(
            far_pair_count=jnp.asarray(far_count, jnp.int32),
            sources=jnp.zeros((far_width,), jnp.int32),
        )
    return SimpleNamespace(
        neighbor_list=SimpleNamespace(offsets=jnp.asarray(offsets, jnp.int32)),
        compact_far_pairs=far,
        nearfield_target_block_source_leaf_ids_padded=(
            None if padded is None else jnp.zeros(padded, jnp.int32)
        ),
    )


def _needs(**flags):
    row = dict.fromkeys(WALK_NEEDS_FIELDS, 0) | flags
    return jnp.asarray([row[k] for k in WALK_NEEDS_FIELDS], jnp.int32)


def _rebuilt(**kw):
    """A topology as a rebuild returns it, every capacity fitting unless ``kw``."""
    fields = dict(
        prepared=_prepared(),
        refresh_ok=jnp.asarray(True),
        walk_needs=_needs(),
        tree_overflow=jnp.asarray(False),
        fresh=True,
        caps=(None, None, False),
    )
    fields.update(kw)
    return ft.FusedTopology(**fields)


@pytest.mark.parametrize(
    "case, expected",
    [
        (dict(), 0),
        # the walk's own flags inform a re-plan; the far list it saturated fails
        (
            dict(walk_needs=_needs(far_overflow=1), refresh_ok=jnp.asarray(False)),
            ft.OVERFLOW_WALK_FAR | ft.OVERFLOW_SATURATED,
        ),
        (
            dict(walk_needs=_needs(near_overflow=1, queue_overflow=1)),
            ft.OVERFLOW_WALK_NEAR | ft.OVERFLOW_WALK_QUEUE,
        ),
        (dict(tree_overflow=jnp.asarray(True)), ft.OVERFLOW_TREE),
        # rows of 2 and 3 against a per-leaf cap of 3
        (dict(caps=(3, None, False)), ft.OVERFLOW_NEIGHBOR_ROW),
        # the returned state's far list full: structurally and against the cap
        (
            dict(prepared=_prepared(far_count=8, far_width=8), caps=(None, 8, False)),
            ft.OVERFLOW_SATURATED | ft.OVERFLOW_FAR_CAP,
        ),
        # a rectangle of 1 x 2 slots per target against a row of 3
        (
            dict(prepared=_prepared(padded=(2, 1, 2)), caps=(None, None, True)),
            ft.OVERFLOW_RECTANGLE,
        ),
    ],
)
def test_each_capacity_sets_its_own_bit(case, expected):
    bits = int(_rebuilt(**case).overflow_bits())
    assert bits == expected, f"{bits:#x} != {expected:#x}"
    # the verdict: any of bits 3-10; the walk's flags alone decide nothing
    fails = bool(expected & ft.OVERFLOW_FAIL_MASK)
    assert bool(ft.overflow_ok(jnp.asarray(bits, jnp.uint32))) is not fails


def test_the_caller_adds_its_own_bits():
    extra = ft.OVERFLOW_LEVEL_PLAN | ft.OVERFLOW_CROSS
    bits = _rebuilt().overflow_bits(extra=extra)
    assert int(bits) == extra and not bool(ft.overflow_ok(bits))


def test_an_eager_topology_has_no_rebuild_verdict():
    topo = ft.FusedTopology.from_prepared(_prepared())
    assert not topo.fresh and topo.refresh_ok is None and topo.walk_needs is None
    np.testing.assert_array_equal(topo.needs(), np.zeros(len(WALK_NEEDS_FIELDS)))
    assert int(topo.overflow_bits()) == 0


def test_the_topology_is_a_pytree_with_static_flags():
    topo = ft.FusedTopology(
        prepared={"x": jnp.arange(3.0)},
        refresh_ok=jnp.asarray(True),
        walk_needs=_needs(),
        tree_overflow=None,
        fresh=True,
        caps=(None, 4096, False),
    )
    leaves, treedef = jax.tree_util.tree_flatten(topo)
    assert len(leaves) == 3  # x, refresh_ok, walk_needs; no tree flag
    back = jax.jit(lambda t: t)(topo)
    assert back.fresh and back.caps == (None, 4096, False)
    np.testing.assert_array_equal(back.prepared["x"], topo.prepared["x"])
    # the static fields key the program
    other = jax.tree_util.tree_structure(
        ft.FusedTopology(prepared={"x": jnp.arange(3.0)}, fresh=False)
    )
    assert other != treedef


def test_the_parameters_split_static_from_traced():
    static = ft.ForceParams(G=1.0, eps2=1e-4)
    leaves, treedef = jax.tree_util.tree_flatten(static)
    assert leaves == []  # Python floats are aux: compile-time literals
    mixed = ft.StepParams(
        force=ft.ForceParams(G=4.3e-6, eps2=jnp.asarray(1e-4)),
        dt=1e-3,
        external={"m": 10.0, "rs": jnp.ones(())},
    )
    leaves, treedef = jax.tree_util.tree_flatten(mixed)
    assert len(leaves) == 2  # eps2 and rs; G, dt and m are static
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.force.G == 4.3e-6 and back.dt == 1e-3 and back.external["m"] == 10.0
    assert back.force.is_static() is False and static.is_static() is True
    # a different static value is a different program
    other = jax.tree_util.tree_structure(ft.ForceParams(G=2.0, eps2=1e-4))
    assert other != jax.tree_util.tree_structure(static)

"""The large-N state types under ``JACCPOT_RUNTIME_TYPECHECK=1``.

Two constructions the import hook used to reject, both on ``strict_run_v2``'s
particle carry (the caps-only prepare, then its template under ``jax.eval_shape``):

* the caps-only prepare's stub, a ``LargeNPreparedState`` holding the tree and the
  static fields with ``local_data`` / ``neighbor_list`` / the leaf arrays ``None``;
* ``tree_unflatten`` of the three large-N pytrees with abstract leaves
  (``jax.ShapeDtypeStruct``), which no field annotation describes.

The shard runs with the hook on (ci.yml), so these fail there if either comes back.
Without the hook they pin that the round trip keeps every field.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.tree import Tree

from jaccpot.runtime._large_n_types import (
    LargeNCompiledState,
    LargeNPreparedState,
    RadixFastNearfieldPayload,
    _unflattened,
)
from jaccpot.runtime.dtypes import INDEX_DTYPE


def _tree() -> Tree:
    points = jnp.asarray(np.random.default_rng(3).uniform(-1.0, 1.0, (64, 3)))
    masses = jnp.full((64,), 1.0 / 64)
    return Tree.from_particles(points, masses, leaf_size=8, tree_type="radix")


def _caps_stub(tree: Tree) -> LargeNPreparedState:
    # what `PrepareMixin.prepare_state_caps_only` builds (fmm_prepare.py)
    return LargeNPreparedState(
        tree=tree,
        local_data=None,
        neighbor_list=None,
        **dict.fromkeys(_REQUIRED_NEARFIELD),
        nearfield_target_block_size=0,
        max_leaf_size=8,
        local_order=2,
        input_dtype=jnp.dtype(jnp.float64),
        working_dtype=jnp.dtype(jnp.float64),
        theta=0.6,
        topology_key=None,
        retry_events=(),
    )


# the near-field fields without a dataclass default
_REQUIRED_NEARFIELD = {
    f.name
    for f in dataclasses.fields(LargeNPreparedState)
    if f.name.startswith("nearfield_")
    and f.default is dataclasses.MISSING
    and f.name != "nearfield_target_block_size"
}


def _payload() -> RadixFastNearfieldPayload:
    leaves, width, slots = 4, 8, 3
    return RadixFastNearfieldPayload(
        target_leaf_ids=jnp.arange(leaves, dtype=INDEX_DTYPE),
        target_particle_ids=jnp.zeros((leaves, width), INDEX_DTYPE),
        target_particle_mask=jnp.ones((leaves, width), bool),
        source_leaf_ids=jnp.zeros((leaves, slots), INDEX_DTYPE),
        source_leaf_valid_mask=jnp.ones((leaves, slots), bool),
        source_particle_ids=jnp.zeros((leaves, slots, width), INDEX_DTYPE),
        source_particle_mask=jnp.ones((leaves, slots, width), bool),
        batch_tile_t=2,
        batch_tile_s=4,
    )


def test_the_caps_only_stub_constructs():
    stub = _caps_stub(_tree())
    assert stub.local_data is None and stub.neighbor_list is None
    assert stub.nearfield_leaf_particle_indices is None
    assert stub.nearfield_leaf_particle_mask is None


@pytest.mark.parametrize("which", ["payload", "stub", "compiled"])
def test_unflatten_takes_abstract_leaves(which):
    if which == "payload":
        state = _payload()
    else:
        state = _caps_stub(_tree())
        if which == "compiled":
            state = LargeNCompiledState(
                prepared=state,
                positions_sorted=state.positions_sorted,
                masses_sorted=state.masses_sorted,
                inverse_permutation=jnp.asarray(
                    state.inverse_permutation, dtype=INDEX_DTYPE
                ),
                topology_key=None,
                max_leaf_size=state.max_leaf_size,
                local_order=state.local_order,
            )
    shapes = jax.eval_shape(lambda s: s, state)
    assert type(shapes) is type(state)
    for field in dataclasses.fields(state):
        got, want = getattr(shapes, field.name), getattr(state, field.name)
        got_leaves, got_def = jax.tree_util.tree_flatten(got)
        want_leaves, want_def = jax.tree_util.tree_flatten(want)
        assert got_def == want_def, field.name
        for g, w in zip(got_leaves, want_leaves):
            if isinstance(w, (jax.Array, np.ndarray)):  # a child: abstract now
                assert isinstance(g, jax.ShapeDtypeStruct), field.name
                assert (g.shape, g.dtype) == (w.shape, w.dtype), field.name
            else:  # aux: static, unchanged
                assert g == w, field.name


def test_unflattened_fills_defaults_and_refuses_what_init_refuses():
    payload = _payload()
    kw = {
        f.name: getattr(payload, f.name)
        for f in dataclasses.fields(payload)
        if f.default is dataclasses.MISSING
    }
    rebuilt = _unflattened(RadixFastNearfieldPayload, **kw)
    assert rebuilt.source_slot_scan_unroll == payload.source_slot_scan_unroll == 1
    with pytest.raises(TypeError, match="missing field 'batch_tile_s'"):
        _unflattened(
            RadixFastNearfieldPayload,
            **{k: v for k, v in kw.items() if k != "batch_tile_s"},
        )
    with pytest.raises(TypeError, match="unknown fields"):
        _unflattened(RadixFastNearfieldPayload, **kw, not_a_field=1)

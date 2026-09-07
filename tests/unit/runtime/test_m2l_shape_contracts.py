"""Shape contracts for the M2L seam's two axes: `classes`, and `nodes`/`sh`.

`bench/annotation_pilot.py` re-recorded 2026-09-03 put 23 silent acceptances of 286
perturbations in this module, concentrated in the class/rotation family and the
accumulators. The annotations that close 19 of them are pinned here.

`classes` is the interesting one. This module's docstring records G.11 as a 60x accuracy
gap between two of the four accumulators, caused by `pair_grouped` gathering rotations
with class ids from the **wrong ordering**. `_rotation_blocks_for_grouped_classes` takes
its class count from `class_deltas` while using `class_keys` as the rotation cache's
identity, so a length disagreement between them keys the cache on a different class count
than the blocks it returns -- G.11 expressed as a shape.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.runtime.kernels._m2l import (
    _m2l_chunk_contributions,
    _rotation_blocks_for_grouped_classes,
)

CLASSES, KEYS, ORDER = 7, 5, 2


def _dtype():
    """Return the working float dtype for the current x64 setting.

    Returns
    -------
    numpy.dtype
        `float64` under `JAX_ENABLE_X64`, else `float32`.
    """
    # A dtype INSTANCE, not the scalar type: `_rotation_blocks_for_grouped_classes`
    # declares `dtype: jnp.dtype` and the decorator added with these annotations now
    # enforces it, which is how this helper was caught passing `jnp.float64` itself.
    return jnp.zeros(
        (), dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    ).dtype


def _class_args(classes: int = CLASSES):
    """Build a matched `class_keys` / `class_deltas` pair.

    Parameters
    ----------
    classes : int
        Number of displacement classes.

    Returns
    -------
    dict
        Keyword arguments for `_rotation_blocks_for_grouped_classes`.
    """
    return {
        "order": ORDER,
        "rotation": "solidfmm",
        "class_keys": jnp.zeros((classes, KEYS), dtype=jnp.int32),
        "class_deltas": jnp.ones((classes, 3), dtype=_dtype()),
        "dtype": _dtype(),
        "basis_mode": "complex",
    }


def test_matched_classes_still_build_their_blocks():
    """The control. Every rejection below is worthless without it."""
    to_blocks, from_blocks = _rotation_blocks_for_grouped_classes(**_class_args())
    assert to_blocks.shape[0] == CLASSES
    assert from_blocks.shape[0] == CLASSES


def test_class_deltas_shorter_than_class_keys_is_rejected():
    """G.11 as a shape: the cache identity and the class count disagreeing."""
    args = _class_args()
    args["class_deltas"] = args["class_deltas"][:-1]
    with pytest.raises(TypeCheckError):
        _rotation_blocks_for_grouped_classes(**args)


def test_class_keys_shorter_than_class_deltas_is_rejected():
    """The same disagreement from the other side."""
    args = _class_args()
    args["class_keys"] = args["class_keys"][:-1]
    with pytest.raises(TypeCheckError):
        _rotation_blocks_for_grouped_classes(**args)


def test_a_class_key_of_the_wrong_width_is_rejected():
    """The key width is a literal 5, so a changed key layout fails loudly.

    It was 5 in all eight recorded calls, across three problem sizes and two orders.
    Failing here is the point: a mis-keyed rotation cache is silent.
    """
    args = _class_args()
    args["class_keys"] = args["class_keys"][:, :-1]
    with pytest.raises(TypeCheckError):
        _rotation_blocks_for_grouped_classes(**args)


def _chunk_args(nodes: int = 6, sh: int = 9):
    """Build one valid `_m2l_chunk_contributions` call.

    Parameters
    ----------
    nodes : int
        Tree-node count.
    sh : int
        Packed spherical-harmonic coefficient count, `(p+1)**2`.

    Returns
    -------
    dict
        Keyword arguments for `_m2l_chunk_contributions`.
    """
    idx = jnp.zeros((2,), dtype=jnp.int32)
    return {
        "multip_packed": jnp.zeros((nodes, sh), dtype=jnp.complex128),
        "centers": jnp.zeros((nodes, 3), dtype=_dtype()),
        "src_idx": idx,
        "tgt_idx": idx,
        "valid": jnp.ones((2,), dtype=bool),
        "order": ORDER,
        "basis_mode": "complex",
        "rotation": "solidfmm",
        "m2l_impl": None,
        "out_dtype": jnp.complex128,
    }


def test_a_complex_packed_expansion_is_still_accepted():
    """`Inexact` and not `Float`: `basis_mode="complex"` is a live lane.

    Narrowing this pair to `Float` is the mistake #293 made one module over, where a
    real-basis-only recording cost 27 CI failures on the complex lane.
    """
    out = _m2l_chunk_contributions(**_chunk_args())
    assert out.shape[0] == 2


def test_centers_that_disagree_with_the_multipoles_on_nodes_are_rejected():
    """`nodes` is shared: ~50 recorded calls, eight distinct extents."""
    args = _chunk_args()
    args["centers"] = args["centers"][:-1]
    with pytest.raises(TypeCheckError):
        _m2l_chunk_contributions(**args)


def test_a_two_component_centre_is_rejected():
    """The spatial literal -- and this one was ALREADY rejected before the annotation.

    Kept because it documents the contract, not because it closes a hole: it passes
    against `main` too, so the M2L displacement arithmetic was already refusing a
    2-component centre on its own. The four tests above are the ones that go red
    without the annotations.
    """
    args = _chunk_args()
    args["centers"] = args["centers"][:, :-1]
    with pytest.raises(TypeCheckError):
        _m2l_chunk_contributions(**args)

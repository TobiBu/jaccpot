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
    _chunk_segment_scatter_add,
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


# ---------------------------------------------------------------------------
# The chunked scatter, from the 2026-09-07 re-recording.
#
# That run put this module at 2 silent acceptances of 244 perturbations -- 1%, down
# from the 8% on 286 that the section above was written against, because the class
# and accumulator families are now closed. BOTH remaining acceptances are in
# `_chunk_segment_scatter_add`, and only ONE of them is a defect.
#
# The defect: `contribs[sort_idx]` gathers with a `sort_idx` whose length comes from
# `tgt_chunk`, so a `contribs` one row short is an out-of-bounds gather, and JAX
# CLAMPS it -- the last row is silently used twice and one pair's contribution is
# scattered into the wrong target.
#
# The non-defect: `local_accum`'s leading axis. 12 recorded calls show it at 7, 15,
# 31, 127, 255, 511 and 1023 against an unchanged `contribs`, so it is genuinely free
# and the pilot was perturbing a free axis. It is asserted below to stay accepted.
# ---------------------------------------------------------------------------

CHUNK, CHUNK_SH, CHUNK_NODES = 512, 25, 255


def _scatter_args(dtype=jnp.complex128):
    """Build one valid chunked-scatter argument set.

    Parameters
    ----------
    dtype : Any
        Coefficient dtype. The recording shows complex128, complex64 AND float64 here,
        which is why the annotation is `Inexact` and not `Float`.

    Returns
    -------
    dict
        Keyword arguments for :func:`_chunk_segment_scatter_add`.
    """
    return {
        "local_accum": jnp.zeros((CHUNK_NODES, CHUNK_SH), dtype=dtype),
        "contribs": jnp.ones((CHUNK, CHUNK_SH), dtype=dtype),
        "tgt_chunk": jnp.zeros((CHUNK,), dtype=jnp.int64),
        "valid": jnp.ones((CHUNK,), dtype=bool),
    }


@pytest.mark.parametrize("dtype", [jnp.complex128, jnp.float64])
def test_the_chunked_scatter_accepts_both_bases(dtype):
    """The control, and the dtype half of it.

    Parameters
    ----------
    dtype : Any
        Complex for the complex basis, float for the real one. `Float` here would
        reject the complex basis outright -- the mistake #293 shipped.
    """
    args = _scatter_args(dtype)
    out = _chunk_segment_scatter_add(**args, chunk_size=CHUNK)
    assert out.shape == (CHUNK_NODES, CHUNK_SH)


def test_contributions_shorter_than_their_target_list_are_rejected():
    """The one measured defect: an out-of-bounds gather that JAX clamps.

    On `main` this returned a full (255, 25) accumulator, having silently gathered
    row 510 twice.
    """
    args = _scatter_args()
    args["contribs"] = jnp.ones((CHUNK - 1, CHUNK_SH), dtype=jnp.complex128)
    with pytest.raises(TypeCheckError):
        _chunk_segment_scatter_add(**args, chunk_size=CHUNK)


def test_the_accumulator_and_the_contributions_must_agree_on_sh():
    """Adding coefficients of two different expansion orders.

    Already rejected on `main`, by broadcasting rather than by annotation, so this
    accepts either exception: `sh` is pinned here for the name, not for a new check.
    The recording agrees on it in all 9 distinct combinations, at 4, 9, 25 and 81.
    """
    args = _scatter_args()
    args["local_accum"] = jnp.zeros((CHUNK_NODES, CHUNK_SH - 1), dtype=jnp.complex128)
    with pytest.raises((TypeCheckError, ValueError)):
        _chunk_segment_scatter_add(**args, chunk_size=CHUNK)


def test_the_accumulators_node_axis_stays_free():
    """`nodes` is bound by `local_accum` alone and must NOT be cross-checked.

    The second of the pilot's two acceptances is this, and it is not a defect: the
    accumulator's length is the target-node count, which has nothing to do with the
    chunk. Asserted so nobody "closes" it later on the strength of the pilot's report.
    """
    args = _scatter_args()
    args["local_accum"] = jnp.zeros((CHUNK_NODES - 1, CHUNK_SH), dtype=jnp.complex128)
    out = _chunk_segment_scatter_add(**args, chunk_size=CHUNK)
    assert out.shape == (CHUNK_NODES - 1, CHUNK_SH)

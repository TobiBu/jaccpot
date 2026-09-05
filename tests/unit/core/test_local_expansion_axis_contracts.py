"""Axis contracts for the downward sweep's M2L and L2L kernels.

`bench/annotation_pilot.py` recorded `downward/local_expansions.py` on 2026-09-04 at 53
silent acceptances of 132 perturbations, 40%, on full 13/0/0 coverage -- and unusually for
this programme, **all 53 were plain parameters**, with no NamedTuple container leaves. That
made it the largest genuinely closable block left in the package.

Two families are pinned here. The Cartesian moment tensors are pure rank contracts: a
`third` passed where a `second` belongs is a different physical term, and nothing downstream
notices because the contraction still broadcasts. And `ct` -- the Cartesian packed
coefficient count, `(p+1)(p+2)(p+3)/6`, recorded at 10 and 20 -- ties the expansion buffers
to the multipole they translate.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.downward.local_expansions import (
    _accumulate_level,
    _build_component_vector,
    _propagate_local_expansions_impl,
)

NODES, CT, EDGES, INTERNAL = 7, 10, 6, 3
ORDER = 2  # ct == (p+1)(p+2)(p+3)/6 == 10 at p=2


def _f(*shape):
    """Return a zero float array of the given shape.

    Parameters
    ----------
    *shape : int
        The shape to build.

    Returns
    -------
    jax.Array
        Zeros of that shape in float64.
    """
    return jnp.zeros(shape, dtype=jnp.float64)


def test_the_cartesian_moment_ranks_are_fixed():
    """A moment of the wrong rank is a different physical term, silently.

    `_build_component_vector` contracts `dipole`, `second`, `third` and `fourth` into one
    component vector. Passing a rank-3 array where the rank-2 quadrupole belongs still
    broadcasts, so the answer is wrong rather than refused.
    """
    good = {
        "mass": jnp.asarray(1.0),
        "dipole": _f(3),
        "second": _f(3, 3),
        "third": _f(3, 3, 3),
        "fourth": _f(3, 3, 3, 3),
    }
    assert _build_component_vector(**good, order=ORDER).shape[0] > 0

    for name, wrong in (
        ("dipole", _f(3, 3)),
        ("second", _f(3, 3, 3)),
        ("third", _f(3, 3)),
        ("fourth", _f(3, 3, 3)),
    ):
        with pytest.raises(TypeCheckError):
            _build_component_vector(**dict(good, **{name: wrong}), order=ORDER)


def test_a_two_component_dipole_is_rejected():
    """The spatial literal, which the whole translation is built on."""
    with pytest.raises(TypeCheckError):
        _build_component_vector(
            mass=jnp.asarray(1.0),
            dipole=_f(2),
            second=_f(3, 3),
            third=_f(3, 3, 3),
            fourth=_f(3, 3, 3, 3),
            order=ORDER,
        )


def _level_kwargs():
    """Build one valid `_accumulate_level` call.

    Returns
    -------
    dict
        Keyword arguments with `nodes`, `nodes+1` and `edges` all distinct.
    """
    return {
        "coeffs": _f(NODES, CT),
        "component_matrix": _f(NODES, CT),
        "centers_target": _f(NODES, 3),
        "centers_source": _f(NODES, 3),
        "sources": jnp.zeros((EDGES,), dtype=jnp.int64),
        "offsets": jnp.zeros((NODES + 1,), dtype=jnp.int64),
        "counts": jnp.zeros((NODES,), dtype=jnp.int64),
    }


def test_the_csr_offsets_length_is_deliberately_free():
    """`offsets` is rank-only, and the distributed lane is why.

    It is `nodes+1` in every call the pilot recorded -- (8,) against 7 nodes, (32,)
    against 31 -- so the symbolic form looked safe and was used. CI disagreed:
    `tests/distributed/test_distributed_m2l_mechanism.py` passes `offsets` at 9 against 9
    target nodes, so the relation is not stable across lanes and the annotation had to go
    back to `_`. Both lengths are asserted here so the next reader does not "restore" it.
    """
    kwargs = _level_kwargs()
    for length in (NODES, NODES + 1):
        _accumulate_level(
            **dict(kwargs, offsets=jnp.zeros((length,), dtype=jnp.int64)),
            order=ORDER,
            chunk_size=2,
        )

    # Rank is still constrained, which is what the annotation buys on this parameter.
    with pytest.raises(TypeCheckError):
        _accumulate_level(
            **dict(kwargs, offsets=jnp.zeros((NODES, 1), dtype=jnp.int64)),
            order=ORDER,
            chunk_size=2,
        )


def test_the_target_and_source_node_sets_are_different_axes():
    """The distributed lane makes them differ, and the single-device recording could not.

    Every recorded call had `coeffs` and `component_matrix` on equal extents -- (7, 7),
    (31, 31) -- so one `nodes` axis looked right. `tests/distributed` passes `coeffs` at 9
    against `component_matrix` at 11, because the source side is a remote tree. A source
    set of a different size must go through.
    """
    kwargs = _level_kwargs()
    wider = {
        "component_matrix": _f(NODES + 2, CT),
        "centers_source": _f(NODES + 2, 3),
    }
    _accumulate_level(**dict(kwargs, **wider), order=ORDER, chunk_size=2)

    # The two source-side arrays still have to agree with each other.
    with pytest.raises(TypeCheckError):
        _accumulate_level(
            **dict(kwargs, component_matrix=_f(NODES + 2, CT)),
            order=ORDER,
            chunk_size=2,
        )


def test_the_level_node_arrays_share_one_axis():
    """`coeffs`, `centers_target` and `counts` are the target side of one level."""
    kwargs = _level_kwargs()
    for name in ("centers_target", "counts"):
        with pytest.raises(TypeCheckError):
            _accumulate_level(
                **dict(kwargs, **{name: kwargs[name][:-1]}), order=ORDER, chunk_size=2
            )


def test_the_edge_list_is_its_own_axis():
    """`sources` is `edges` and must NOT be tied to `nodes`.

    Recorded at 6, 10 and 12 against node counts of 7, 31 and 7, so they vary
    independently -- a shorter edge list is legitimate and has to go through.
    """
    kwargs = _level_kwargs()
    _accumulate_level(
        **dict(kwargs, sources=jnp.zeros((EDGES - 2,), dtype=jnp.int64)),
        order=ORDER,
        chunk_size=2,
    )


def test_the_child_arrays_are_internal_nodes_not_nodes():
    """`left_child`/`right_child` are `internal`, which is not `nodes`.

    Recorded at (5, 2), (7, 3) and (3, 1) -- `internal == (nodes - 1) / 2` for the radix
    tree -- so naming them `nodes` would assert something false, the `farleaves` mistake
    in a different lane.
    """
    kwargs = {
        "coeffs": _f(NODES, CT),
        "centers": _f(NODES, 3),
        "left_child": jnp.zeros((INTERNAL,), dtype=jnp.int64),
        "right_child": jnp.zeros((INTERNAL,), dtype=jnp.int64),
    }
    _propagate_local_expansions_impl(**kwargs, order=ORDER, num_internal=INTERNAL)

    # The pair still has to agree with itself.
    with pytest.raises(TypeCheckError):
        _propagate_local_expansions_impl(
            **dict(kwargs, right_child=kwargs["right_child"][:-1]),
            order=ORDER,
            num_internal=INTERNAL,
        )

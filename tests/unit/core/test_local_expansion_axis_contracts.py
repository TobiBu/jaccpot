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


def test_the_csr_offsets_are_one_longer_than_the_nodes():
    """`offsets` is `nodes+1`, and it is expressible here.

    Unlike `_large_n_blocks.py`'s `block_offsets` and `_adaptive_policy`'s
    `neighbor_offsets`, this parameter comes AFTER `coeffs` in the signature, so `nodes`
    is already bound when jaxtyping evaluates the expression. The symbolic form works
    where those two had to settle for a rank-only `_`.
    """
    kwargs = _level_kwargs()
    _accumulate_level(**kwargs, order=ORDER, chunk_size=2)

    with pytest.raises(TypeCheckError):
        _accumulate_level(
            **dict(kwargs, offsets=jnp.zeros((NODES,), dtype=jnp.int64)),
            order=ORDER,
            chunk_size=2,
        )


def test_the_level_node_arrays_share_one_axis():
    """`coeffs`, `component_matrix`, both centre arrays and `counts` are all per-node."""
    kwargs = _level_kwargs()
    for name in ("component_matrix", "centers_target", "centers_source", "counts"):
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

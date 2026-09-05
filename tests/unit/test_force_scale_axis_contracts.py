"""Axis contracts for the distributed force-scale helpers.

This module is the one where the pilot's evidence is weakest by construction. Its
recording is taken on a SINGLE device, so every local/remote pair reads equal -- 255 nodes
and 1024 particles on both sides of `distributed_force_scale_nodes` -- and the production
path is multi-device. `tests/distributed` cannot supply better evidence either: every file
there skips below two devices, so it is not in the pilot's scope at all.

So the separations below come from CALL SITES, not from the sample. `coarse_centers` is
`rct.geometry.center` where `rct` is the remote coarse tree, a different object from the
local `tree` whose geometry feeds `node_centers`; `cross_force_scale_own`'s sources are the
remote domain and its targets the local one. Tying either pair would repeat the mistake
#324 had to undo in `downward/local_expansions.py`, where exactly this kind of
single-device equality shipped and the distributed tier caught it.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from jaccpot.distributed._force_scale import (
    coarse_source_mac_geometry,
    cross_force_scale_own,
    flatten_neighbor_csr,
)

SOURCES, TARGETS, PAIRS, LEAVES, EDGES = 7, 4, 11, 5, 13


def _f(*shape):
    """Return zeros of the given shape in float64.

    Parameters
    ----------
    *shape : int
        Shape to build.

    Returns
    -------
    jax.Array
        The zero array.
    """
    return jnp.zeros(shape, dtype=jnp.float64)


def _cross():
    """Build one valid `cross_force_scale_own` call with sources != targets.

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {
        "source_masses": jnp.ones((SOURCES,), dtype=jnp.float64),
        "source_centers": _f(SOURCES, 3),
        "source_radii": jnp.ones((SOURCES,), dtype=jnp.float64),
        "target_centers": _f(TARGETS, 3),
        "target_radii": jnp.ones((TARGETS,), dtype=jnp.float64),
        "pair_sources": jnp.zeros((PAIRS,), dtype=jnp.int32),
        "pair_targets": jnp.zeros((PAIRS,), dtype=jnp.int32),
        "pair_valid": jnp.ones((PAIRS,), dtype=bool),
        "num_target_nodes": TARGETS,
        "g": jnp.asarray(1.0),
        "eps_sq": jnp.asarray(1e-4),
        "inflation": jnp.asarray(1.0),
    }


def test_the_remote_sources_and_local_targets_are_independent():
    """7 remote sources against 4 local targets must go through.

    Every recorded call had them equal at 255, because the recording is single-device.
    The call site says they are different domains.
    """
    out = cross_force_scale_own(**_cross())
    assert out.shape == (TARGETS,)


def test_each_domain_must_agree_with_itself():
    """Within a domain, centres, radii and masses describe the same nodes."""
    args = _cross()
    with pytest.raises(TypeCheckError):
        cross_force_scale_own(**dict(args, source_radii=args["source_radii"][:-1]))
    with pytest.raises(TypeCheckError):
        cross_force_scale_own(**dict(args, target_radii=args["target_radii"][:-1]))


def test_the_pair_lists_are_one_axis():
    """`pair_sources`, `pair_targets` and `pair_valid` are parallel lists."""
    args = _cross()
    with pytest.raises(TypeCheckError):
        cross_force_scale_own(**dict(args, pair_targets=args["pair_targets"][:-1]))
    with pytest.raises(TypeCheckError):
        cross_force_scale_own(**dict(args, pair_valid=args["pair_valid"][:-1]))


def test_the_csr_head_arrays_agree_and_the_edge_list_does_not():
    """`counts` and `leaf_indices` are per leaf; `indices` is the flattened edge list.

    Recorded at (4,)/(4,)/(8,) and (128,)/(128,)/(524288,), so the edge count is free.
    """
    counts = jnp.zeros((LEAVES,), dtype=jnp.int32)
    leaf_indices = jnp.zeros((LEAVES,), dtype=jnp.int32)
    indices = jnp.zeros((EDGES,), dtype=jnp.int32)
    flatten_neighbor_csr(counts=counts, indices=indices, leaf_indices=leaf_indices)

    with pytest.raises(TypeCheckError):
        flatten_neighbor_csr(
            counts=counts, indices=indices, leaf_indices=leaf_indices[:-1]
        )


def test_the_coarse_geometry_triple_is_one_node_set():
    """`coarse_source_mac_geometry` takes three views of the SAME coarse nodes."""
    args = {
        "expansion_centers": _f(LEAVES, 3),
        "geometry_centers": _f(LEAVES, 3),
        "geometry_radii": jnp.ones((LEAVES,), dtype=jnp.float64),
    }
    coarse_source_mac_geometry(**args)

    with pytest.raises(TypeCheckError):
        coarse_source_mac_geometry(
            **dict(args, geometry_radii=args["geometry_radii"][:-1])
        )

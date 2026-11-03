"""The floor under eq (16a)'s thresholds (``mac_force_scale_floor``, #487).

* ``ancestor_max_by_level`` equals a numpy walk up the parent chain taking the max;
* ``force_scale_floor_nodes`` at factor 1 is a LOWER BOUND on the exact eq (16b)
  ``f_b`` of every particle under every node (brute force), so a floor at any
  factor <= 1 can never make eq (16a) looser than the exact force scale would;
* it is zero where no ancestor holds more than the largest particle's mass;

on a bucket tree and on the capacity-padded cell tree the fused lane uses (empty
leaves included), eagerly and under ``jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._force_scale_levels import (
    ancestor_max_by_level,
    force_scale_floor_nodes,
)
from tests.unit.runtime.test_force_scale_levels import _tables, _tree

SOFTENING = 1e-2


def _chain_max(parent, own):
    out = np.array(own, dtype=np.float64)
    for k in range(parent.shape[0]):
        p = int(parent[k])
        while p >= 0:
            out[k] = max(out[k], own[p])
            p = int(parent[p])
    return out


@pytest.mark.parametrize("kind", ["buckets", "cells"])
def test_ancestor_max_equals_the_serial_chain_max(kind):
    topo, _, _ = _tree(kind)
    own = jnp.asarray(np.random.default_rng(2).uniform(0.0, 1.0, topo.parent.shape[0]))
    want = _chain_max(np.asarray(topo.parent), np.asarray(own))
    kw = _tables(topo)
    np.testing.assert_array_equal(np.asarray(ancestor_max_by_level(own, **kw)), want)
    jitted = jax.jit(
        lambda o: ancestor_max_by_level(
            o,
            kw["left_child"],
            kw["right_child"],
            kw["parent"],
            kw["nodes_by_level"],
            kw["level_offsets"],
            num_internal=kw["num_internal"],
            num_levels=kw["num_levels"],
            level_batch_width=kw["level_batch_width"],
        )
    )
    np.testing.assert_array_equal(np.asarray(jitted(own)), want)


def _node_mass_and_com_radius(topo, ps, ms):
    ranges = np.asarray(topo.node_ranges)
    ps, ms = np.asarray(ps), np.asarray(ms)
    mass = np.zeros(ranges.shape[0])
    radius = np.zeros(ranges.shape[0])
    for k, (lo, hi) in enumerate(ranges):
        if hi < lo:
            continue
        m = ms[lo : hi + 1]
        x = ps[lo : hi + 1]
        mass[k] = m.sum()
        com = (m[:, None] * x).sum(0) / m.sum()
        radius[k] = np.sqrt(((x - com) ** 2).sum(1)).max()
    return mass, radius


@pytest.mark.parametrize("kind", ["buckets", "cells"])
def test_the_floor_is_a_lower_bound_on_the_exact_force_scale(kind):
    topo, ps, ms = _tree(kind, n=1500)
    mass, radius = _node_mass_and_com_radius(topo, ps, ms)
    kw = _tables(topo)
    floor = force_scale_floor_nodes(
        tree=topo,
        node_mass=jnp.asarray(mass),
        node_radii=jnp.asarray(radius),
        max_particle_mass=jnp.max(ms),
        gravitational_constant=1.0,
        softening_sq=jnp.asarray(SOFTENING**2),
        factor=1.0,
        num_levels=kw["num_levels"],
    )
    floor = np.asarray(floor)
    x, m = np.asarray(ps), np.asarray(ms)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    terms = m[None, :] / (d2 + SOFTENING**2)
    np.fill_diagonal(terms, 0.0)
    f_b = terms.sum(1)  # the exact eq (16b), every particle
    ranges = np.asarray(topo.node_ranges)
    live = ranges[:, 1] >= ranges[:, 0]
    exact_min = np.array(
        [f_b[lo : hi + 1].min() if hi >= lo else np.inf for lo, hi in ranges]
    )
    assert np.all(floor[live] <= exact_min[live])
    # it is not vacuous: it binds within a small factor somewhere, and every leaf
    # under a node of many particles gets a positive floor from its ancestors
    assert np.max(floor[live] / exact_min[live]) > 0.05
    leaves = np.arange(int(topo.left_child.shape[0]), ranges.shape[0])
    assert np.all(floor[leaves[live[leaves]]] > 0.0)


def test_no_floor_where_no_ancestor_outweighs_one_particle():
    topo, ps, ms = _tree("cells", n=600)
    mass, radius = _node_mass_and_com_radius(topo, ps, ms)
    kw = _tables(topo)
    floor = force_scale_floor_nodes(
        tree=topo,
        node_mass=jnp.asarray(mass),
        node_radii=jnp.asarray(radius),
        max_particle_mass=jnp.asarray(10.0 * float(np.sum(mass[:1]))),
        gravitational_constant=1.0,
        softening_sq=jnp.asarray(SOFTENING**2),
        factor=1.0,
        num_levels=kw["num_levels"],
    )
    # a "largest particle" heavier than the whole system: every bound is zero
    assert np.all(np.asarray(floor) == 0.0)

"""The floor under eq (16a)'s thresholds (``mac_force_scale_floor``, #487).

* ``force_scale_floor_nodes`` at factor 1 is a LOWER BOUND on the exact eq (16b)
  ``f_b`` of every particle under every node (brute force), so a floor at any
  factor <= 1 can never make eq (16a) looser than the exact force scale would;
* it is tight enough to matter: within a factor 50 of the exact f_b at the
  median leaf;

on a bucket tree and on the capacity-padded cell tree the fused lane uses (empty
leaves included), eagerly and under ``jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._force_scale_levels import force_scale_floor_nodes
from tests.unit.runtime.test_force_scale_levels import _tables, _tree

SOFTENING = 1e-2


def _node_mass_and_com_radius(topo, ps, ms):
    ranges = np.asarray(topo.node_ranges)
    ps, ms = np.asarray(ps), np.asarray(ms)
    mass = np.zeros(ranges.shape[0])
    radius = np.zeros(ranges.shape[0])
    centers = np.zeros((ranges.shape[0], 3))
    for k, (lo, hi) in enumerate(ranges):
        if hi < lo:
            continue
        m = ms[lo : hi + 1]
        x = ps[lo : hi + 1]
        mass[k] = m.sum()
        com = (m[:, None] * x).sum(0) / m.sum()
        radius[k] = np.sqrt(((x - com) ** 2).sum(1)).max()
        centers[k] = com
    return mass, centers, radius


@pytest.mark.parametrize("kind", ["buckets", "cells"])
def test_the_floor_is_a_lower_bound_on_the_exact_force_scale(kind):
    topo, ps, ms = _tree(kind, n=1500)
    mass, centers, radius = _node_mass_and_com_radius(topo, ps, ms)
    kw = _tables(topo)
    floor = force_scale_floor_nodes(
        tree=topo,
        node_mass=jnp.asarray(mass),
        node_centers=jnp.asarray(centers),
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
    # it is not vacuous: at the leaves, where eq (16a)'s rows are decided, it is
    # within a factor 50 of the exact value for the median leaf (measured: 0.032
    # median, 0.011-0.10, on this leaf-16 Plummer cell tree)
    leaves = np.arange(int(topo.left_child.shape[0]), ranges.shape[0])
    leaves = leaves[live[leaves]]
    ratio = floor[leaves] / exact_min[leaves]
    assert np.all(ratio > 0.0)
    assert np.median(ratio) > 0.02, np.quantile(ratio, [0.0, 0.5, 1.0])

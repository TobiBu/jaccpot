"""The system of ``bench/hybrid_clusters_runs.py`` (HoreKa jobs hh4a and hh4c).

The bench draws HH3b gate 5's / HH4a item 4's / HH4c's GPU gate's system itself: a
Hernquist galaxy and K Plummer clusters on circular orbits. Its shapes, masses and
centring are what the rows' K x M and the stars' totals assume.
"""

from __future__ import annotations

import numpy as np

from bench import hybrid_clusters_runs as runs


def test_system_shapes_masses_and_centring():
    xg, vg, mg, xc, vc, mc = runs._system(256, 3, 40, seed=7)
    assert xg.shape == vg.shape == (256, 3) and mg.shape == (256,)
    assert xc.shape == vc.shape == (3, 40, 3) and mc.shape == (3, 40)
    assert np.isclose(mg.sum(), 1.0) and np.allclose(mc.sum(axis=1), runs.MC)
    assert all(np.isfinite(a).all() for a in (xg, vg, xc, vc))
    # each cluster sits at radius 0.3-1.5, moving on the galaxy's circular speed there
    r = np.linalg.norm(xc.mean(axis=1), axis=1)
    assert ((r > 0.29) & (r < 1.51)).all()
    v = np.linalg.norm(vc.mean(axis=1), axis=1)
    assert np.allclose(v, np.sqrt(r / (1 + r) ** 2), rtol=1e-6)
    # the internal Plummer velocities stay bound (below the escape speed at the centre)
    dv = np.linalg.norm(vc - vc.mean(axis=1, keepdims=True), axis=2)
    assert dv.max() < 2 * np.sqrt(2 * runs.MC / runs.RC)


def test_system_is_seeded():
    a = runs._system(64, 2, 16, seed=3)
    b = runs._system(64, 2, 16, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))

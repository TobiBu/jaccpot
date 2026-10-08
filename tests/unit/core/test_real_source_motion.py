"""Real-basis source-motion multipoles: the far field's time derivatives.

``prepare_real_source_motion_multipoles`` returns ``d^k M / dt^k`` for every node
when the particles move on straight lines about frozen centres. These tests use a
basis-independent oracle -- central finite differences of the ordinary real
multipoles at those same centres -- so they outlive the complex basis. The
engine-level check against the complex implementation (while it exists) is in
``test_real_source_motion_matches_complex``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.dtypes import INDEX_DTYPE
from yggdrax.tree import get_level_offsets, get_nodes_by_level

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMPreset
from jaccpot.runtime._level_shapes import level_batch_width
from jaccpot.upward.real_tree_expansions import (
    _p2m_leaves_real,
    aggregate_m2m_real_by_level,
    prepare_real_source_motion_multipoles,
)

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="finite differences need float64"
)

ORDER = 4


def _problem(n=96, seed=11):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(scale=0.6, size=(n, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=n))
    velocities = jnp.asarray(rng.normal(scale=0.3, size=(n, 3)))
    return positions, masses, velocities


def _frozen(state):
    tree = state.tree
    return dict(
        tree=tree,
        centers=jnp.asarray(state.upward.multipoles.centers),
        total=int(jnp.asarray(tree.parent).shape[0]),
        internal=int(jnp.asarray(tree.left_child).shape[0]),
        leaf=int(state.max_leaf_size),
    )


def _multipoles_at(f, positions_sorted, masses_sorted):
    """The ordinary real P2M + M2M at the FROZEN centres of ``f``."""
    tree = f["tree"]
    offsets = get_level_offsets(tree)
    packed = _p2m_leaves_real(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        f["centers"],
        order=ORDER,
        max_leaf_size=f["leaf"],
        num_internal=f["internal"],
        total_nodes=f["total"],
        leaf_batch_size=64,
    )
    return aggregate_m2m_real_by_level(
        packed,
        f["centers"],
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(get_nodes_by_level(tree), dtype=INDEX_DTYPE),
        jnp.asarray(offsets, dtype=INDEX_DTYPE),
        order=ORDER,
        num_internal=f["internal"],
        num_levels=max(int(offsets.shape[0] - 1), 1),
        level_batch_width=level_batch_width(
            offsets, total_nodes=f["total"], num_internal=f["internal"]
        ),
    )


def test_source_motion_multipoles_are_the_time_derivatives():
    positions, masses, velocities = _problem()
    fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    state = fmm.prepare_state(positions, masses, max_order=ORDER, leaf_size=4)
    f = _frozen(state)
    order_idx = jnp.asarray(state.tree.particle_indices, dtype=INDEX_DTYPE)
    x = jnp.asarray(state.positions_sorted)
    m = jnp.asarray(state.masses_sorted)
    v = velocities[order_idx]

    def moved(t):
        return _multipoles_at(f, x + t * v, m)

    h = 1e-3
    m0, mp, mm = moved(0.0), moved(h), moved(-h)
    fd1 = (mp - mm) / (2.0 * h)
    fd2 = (mp - 2.0 * m0 + mm) / (h * h)
    for k, fd in ((1, fd1), (2, fd2)):
        got = prepare_real_source_motion_multipoles(
            f["tree"],
            x,
            m,
            v,
            max_order=ORDER,
            centers=f["centers"],
            time_derivative_order=k,
            max_leaf_size=f["leaf"],
        )
        err = float(jnp.linalg.norm(got - fd) / jnp.linalg.norm(got))
        # central differences: O(h^2) truncation, plus round-off / h^k
        assert err < (1e-6 if k == 1 else 1e-4), f"k={k}: rel {err:.3e}"


def test_a_resting_system_has_no_source_motion():
    positions, masses, _ = _problem(n=40)
    fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis="real")
    state = fmm.prepare_state(positions, masses, max_order=ORDER, leaf_size=4)
    f = _frozen(state)
    got = prepare_real_source_motion_multipoles(
        f["tree"],
        jnp.asarray(state.positions_sorted),
        jnp.asarray(state.masses_sorted),
        jnp.zeros_like(state.positions_sorted),
        max_order=ORDER,
        centers=f["centers"],
        time_derivative_order=1,
        max_leaf_size=f["leaf"],
    )
    assert float(jnp.max(jnp.abs(got))) == 0.0


def test_real_source_motion_matches_complex():
    """Jerk and the K = 3 tower: real against complex on one configuration.

    The complex basis computed these first; the real path now does it natively
    (lowered leaf P2M, real M2M / M2L / L2L, exact derivative tower). Measured at
    N = 96, leaf 4, p = 4, theta 0.6 (106 M2L pairs): 8e-18 relative on the jerk,
    <= 6e-18 on D1-D3. Goes with the complex basis (cleanup phase C); the
    direct-sum tests in ``test_solver_api.py`` and the finite-difference test
    above then own these properties.
    """
    from tests.unit.test_solver_api import _sample_problem, _sample_velocities

    positions, masses = _sample_problem(n=96)
    velocities = _sample_velocities(n=96)
    positions, masses, velocities = (
        jnp.asarray(a, jnp.float64) for a in (positions, masses, velocities)
    )
    kwargs = dict(leaf_size=4, max_order=ORDER, theta=0.6)
    out = {}
    for basis in ("complex", "real"):
        fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis=basis)
        _, jerk = fmm.compute_accelerations_and_jerk(
            positions, masses, velocities, jerk_mode="accurate", **kwargs
        )
        fmm = FastMultipoleMethod(preset=FMMPreset.ACCURATE, basis=basis)
        _, derivs = fmm.compute_accelerations_with_time_derivatives(
            positions, masses, velocities, max_time_derivative_order=3, **kwargs
        )
        out[basis] = (jerk, *derivs)
    for got, want in zip(out["real"], out["complex"]):
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < 1e-12, rel

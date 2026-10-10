"""Real-basis source-motion multipoles: the far field's time derivatives.

``prepare_real_source_motion_multipoles`` returns ``d^k M / dt^k`` for every node
when the particles move on straight lines about frozen centres. These tests use a
basis-independent oracle -- central finite differences of the ordinary real
multipoles at those same centres. The engine-level check, with the far field
engaged, is against direct sums:
``test_real_source_motion_with_the_far_field_matches_direct_sum``.
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


def test_real_source_motion_with_the_far_field_matches_direct_sum():
    """Jerk and the K = 3 tower with the far field engaged, against direct sums.

    The engine-level owner of the far-field time derivatives: the direct-sum tests
    in ``test_solver_api.py`` run the tower at ``theta=1e-4`` (no M2L pair), and
    ``test_jerk_modes_diverge_against_direct_sum_when_far_field_engaged`` covers
    the jerk only. Here snap and crackle go through the far field too.

    Replaces ``test_real_source_motion_matches_complex`` (2026-10 cleanup, C), which
    compared the same quantities with the complex basis to 1e-12. Measured at
    N = 96, leaf 4, p = 4, theta 0.6 (106 M2L pairs), fp64, Plummer softening 1e-3
    (the direct sums' kernel), relative L2 against the direct sums -- identical on
    the complex basis before it was removed:

        jerk     4.338e-4   (both entry points)
        snap     8.285e-5
        crackle  9.554e-6

    The bounds give each ~2.3-3x headroom. The two jerk entry points
    (``compute_accelerations_and_jerk`` and the tower's first level) run the same
    analytic path and must agree to round-off.
    """
    from tests.unit.test_solver_api import (
        _direct_sum_crackle,
        _direct_sum_jerk,
        _direct_sum_snap,
        _sample_problem,
        _sample_velocities,
    )

    positions, masses = _sample_problem(n=96)
    velocities = _sample_velocities(n=96)
    positions, masses, velocities = (
        jnp.asarray(a, jnp.float64) for a in (positions, masses, velocities)
    )
    kwargs = dict(leaf_size=4, max_order=ORDER, theta=0.6)

    def solver():
        return FastMultipoleMethod(
            preset=FMMPreset.ACCURATE, softening=1e-3, softening_kernel="plummer"
        )

    state = solver().prepare_state(positions, masses, **kwargs)
    far_pairs = int(jnp.sum(state.interactions.counts))
    assert far_pairs > 0, f"the far field must be engaged (got {far_pairs} M2L pairs)"

    _, jerk = solver().compute_accelerations_and_jerk(
        positions, masses, velocities, jerk_mode="accurate", **kwargs
    )
    _, derivs = solver().compute_accelerations_with_time_derivatives(
        positions, masses, velocities, max_time_derivative_order=3, **kwargs
    )
    assert len(derivs) == 3
    jerk_gap = float(jnp.linalg.norm(jerk - derivs[0]) / jnp.linalg.norm(derivs[0]))
    assert jerk_gap < 1e-12, jerk_gap

    references = [
        oracle(positions, masses, velocities, G=1.0, softening=1e-3)
        for oracle in (_direct_sum_jerk, _direct_sum_snap, _direct_sum_crackle)
    ]
    bounds = (1e-3, 2.5e-4, 3e-5)
    for name, got, want, bound in zip(
        ("jerk", "snap", "crackle"), derivs, references, bounds
    ):
        rel = float(jnp.linalg.norm(got - want) / jnp.linalg.norm(want))
        assert rel < bound, f"{name}: rel-L2 {rel:.3e} >= {bound:.1e}"

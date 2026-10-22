"""The large-N far field at live inputs, on the production layout, forward and reverse.

``large_n_farfield_locals_at`` re-runs P2M -> M2M -> M2L -> L2L at frozen topology,
on the production refresh's operands, and ``large_n_farfield_l2p_at`` evaluates the
locals as the production eval does (its L2P layout and kernel: the particle-major
Pallas L2P through its ``custom_vjp`` here, in interpret mode). Together they are
the far half of the differentiable forward that plan S8 makes the production one.

* forward: at the state's own positions and masses the pair reproduces the
  production evaluation's far field BITWISE (``JACCPOT_LARGE_N_EVAL_DIAG_MODE=
  far_only``, sorted output);
* reverse: ``jax.grad`` in float64 through every far-field kernel's reverse
  (Pallas P2M, M2M, M2L, L2L and L2P, interpret mode) agrees with central finite
  differences, in the positions AND the masses (the masses only reach the force
  through the far chain's source side). The tree has single-particle leaves, whose
  expansion centre is the particle itself (``delta = 0``): the leaf-major L2P's
  reverse returned NaN there.
* the whole force (plan S8, the wiring): ``differentiable_accelerations`` jitted
  with the state and the grad plan as ARGUMENTS is the production evaluation
  bitwise, on the far list as the GPU's walk hands it over (target-sorted and
  checked symmetric).
"""

from __future__ import annotations

import functools
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import (
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    TreeConfig,
)
from jaccpot.runtime._large_n_grad import (
    large_n_farfield_l2p_at,
    large_n_farfield_locals_at,
    prepare_large_n_grad_plan,
)
from jaccpot.runtime._large_n_types import LargeNPreparedState
from jaccpot.runtime.dtypes import INDEX_DTYPE

_N = 1024
_LEAF = 8
_ORDER = 4
_G = 1.0
_SOFTENING = 1e-2

#: every Pallas kernel of the lane in interpret mode, the prepacked near layout
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and ``can_use_large_n_prepare_path``'s backend gate open.

    The backend check is profile selection, not a hardware requirement (see
    ``test_large_n_grad_reverse_path._open_the_backend_gate``).

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    for key, value in _ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()


def _make_lane() -> tuple[Any, Any, Any]:
    """A float64 large-N state with a load-bearing far field, and its grad plan.

    Returns
    -------
    tuple[Any, Any, Any]
        Solver, prepared state, plan.
    """
    rng = np.random.default_rng(11)
    positions = jnp.asarray(rng.normal(size=(_N, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=_N))
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        G=_G,
        softening=_SOFTENING,
        use_pallas=True,
        working_dtype=jnp.float64,
        advanced=FMMAdvancedConfig(
            # cell leaves, as in production: single-particle leaves occur
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity="auto",
            ),
            farfield=FarFieldConfig(retain_far_pairs_for_grad=True),
        ),
    )
    state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
    assert isinstance(state, LargeNPreparedState)
    assert state.positions_sorted.dtype == jnp.float64
    return fmm, state, prepare_large_n_grad_plan(fmm, state)


@pytest.fixture(scope="module")
def lane(_lane_on_cpu) -> tuple[Any, Any, Any]:
    """The far list as a COO list (the CPU walk's default here).

    Parameters
    ----------
    _lane_on_cpu : None
        The environment fixture.

    Returns
    -------
    tuple[Any, Any, Any]
        Solver, prepared state, plan.
    """
    return _make_lane()


@pytest.fixture(scope="module")
def sorted_lane(_lane_on_cpu) -> tuple[Any, Any, Any]:
    """The far list as the GPU's walk hands it over: target-sorted, symmetric.

    The Pallas walk (``JACCPOT_WALK_PALLAS_INTERPRET``, the GPU's default walk in
    interpret mode) emits its far list as a ``TargetSortedFarPairs`` CSR; the plan
    checks it symmetric, and the M2L reverse then sorts nothing.

    Parameters
    ----------
    _lane_on_cpu : None
        The environment fixture.

    Returns
    -------
    tuple[Any, Any, Any]
        Solver, prepared state, plan.
    """
    patcher = pytest.MonkeyPatch()
    patcher.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")
    try:
        return _make_lane()
    finally:
        patcher.undo()


def _far_acceleration(fmm, state, plan, positions, masses):
    locals_live = large_n_farfield_locals_at(fmm, state, positions, masses, plan=plan)
    return -_G * large_n_farfield_l2p_at(state, locals_live, positions)


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_tree_is_one_the_test_needs(lane):
    """Non-vacuous: far pairs exist, and so do single-particle leaves."""
    _, state, plan = lane
    assert int(plan.far_pair_active_count) > 0
    ranges = np.asarray(state.tree.node_ranges)[
        np.asarray(state.neighbor_list.leaf_indices)
    ]
    assert np.any(ranges[:, 1] == ranges[:, 0])


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_far_forward_is_the_production_far_field(lane, monkeypatch):
    """At the state's own inputs: the production evaluation's far field, bitwise."""
    fmm, state, plan = lane
    monkeypatch.setenv("JACCPOT_LARGE_N_EVAL_DIAG_MODE", "far_only")
    production = np.asarray(fmm.evaluate_prepared_state(state))
    got_sorted = _far_acceleration(
        fmm, state, plan, state.positions_sorted, state.masses_sorted
    )
    # the production evaluation scatters its sorted rows to the input order
    got = np.asarray(got_sorted)[np.asarray(state.inverse_permutation)]
    assert np.any(production)
    assert np.array_equal(got, production)


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_far_gradient_matches_finite_differences(lane):
    """``jax.grad`` through the far chain's reverse kernels vs central differences.

    Directional derivatives along random directions, in the positions and the
    masses separately; float64, so the differences agree to ~1e-8 (measured:
    every component within 5e-7 of a per-entry check at N=1024).
    """
    fmm, state, plan = lane
    rng = np.random.default_rng(5)
    weight = jnp.asarray(rng.normal(size=(_N, 3)))
    ps, ms = state.positions_sorted, state.masses_sorted

    def loss(p, m):
        return jnp.sum(weight * _far_acceleration(fmm, state, plan, p, m))

    grad_p, grad_m = jax.grad(loss, argnums=(0, 1))(ps, ms)
    assert np.all(np.isfinite(np.asarray(grad_p)))
    assert np.all(np.isfinite(np.asarray(grad_m)))
    value = jax.jit(loss)
    for _ in range(2):
        v_p = jnp.asarray(rng.normal(size=(_N, 3)))
        v_m = jnp.asarray(rng.normal(size=(_N,)))
        h = 1e-5
        fd_p = (float(value(ps + h * v_p, ms)) - float(value(ps - h * v_p, ms))) / (
            2 * h
        )
        fd_m = (float(value(ps, ms + h * v_m)) - float(value(ps, ms - h * v_m))) / (
            2 * h
        )
        got_p = float(jnp.sum(grad_p * v_p))
        got_m = float(jnp.sum(grad_m * v_m))
        assert abs(got_p - fd_p) <= 1e-6 * abs(fd_p), (got_p, fd_p)
        assert abs(got_m - fd_m) <= 1e-6 * abs(fd_m), (got_m, fd_m)


def _input_order(state):
    """The state's positions and masses back in the caller's order."""
    inverse = state.inverse_permutation
    return state.positions_sorted[inverse], state.masses_sorted[inverse]


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_differentiable_force_is_the_production_force(sorted_lane):
    """The whole force, jitted with the state and the plan as arguments: bitwise."""
    fmm, state, plan = sorted_lane
    positions, masses = _input_order(state)

    @jax.jit
    def force(st, pl, p, m):
        return fmm.differentiable_accelerations(st, p, m, grad_plan=pl)

    production = np.asarray(fmm.evaluate_prepared_state(state))
    got = np.asarray(force(state, plan, positions, masses))
    assert np.any(production)
    assert np.array_equal(got.view(np.uint64), production.view(np.uint64))


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_step_fn_takes_the_topology_as_arguments(sorted_lane):
    """``differentiable_step_fn`` binds the state and its plan as arguments.

    Closed over, ``jax.jit`` would embed every array of the topology in the program
    as a constant. The bound program is the production force (bitwise) and
    differentiable.
    """
    from jaccpot.runtime._large_n_grad import LargeNGradPlan

    fmm, state, _ = sorted_lane
    positions, masses = _input_order(state)
    step = fmm.differentiable_step_fn(state)
    assert isinstance(step, functools.partial)
    assert step.args[0] is state and isinstance(step.args[1], LargeNGradPlan)
    production = np.asarray(fmm.evaluate_prepared_state(state))
    got = np.asarray(step(positions, masses))
    assert np.array_equal(got.view(np.uint64), production.view(np.uint64))
    g = jax.grad(lambda p: jnp.sum(step(p, masses)[:, 0]))(positions)
    assert np.all(np.isfinite(np.asarray(g))) and np.any(np.asarray(g) != 0.0)


@pytest.mark.xdist_group("large_n_far_chain_grad")
def test_the_plan_checks_that_the_lists_are_symmetric(sorted_lane):
    """The walk's lists hold every directed pair both ways -- the near reverse
    needs it, and on the far list the M2L reverse then sorts nothing -- and the
    plan's check tells a list that does not."""
    from jaccpot.runtime._large_n_grad import (
        _csr_pairs_symmetric,
        _directed_pairs_symmetric,
        _near_list_symmetric,
    )

    _, state, plan = sorted_lane
    assert plan.far_pairs_target_sorted and plan.far_pairs_symmetric
    assert _near_list_symmetric(state.neighbor_list)
    a, b = jnp.asarray([0, 1, 2, 3, 2]), jnp.asarray([1, 0, 3, 2, 0])
    both = jnp.asarray([True, True, True, True, False])
    assert _directed_pairs_symmetric(a, b, both)
    # one pair without its reverse, live
    assert not _directed_pairs_symmetric(a, b, jnp.ones((5,), bool))
    # a duplicated pair is not its reverse either (a multiset test)
    assert not _directed_pairs_symmetric(
        jnp.asarray([0, 0, 1]), jnp.asarray([1, 1, 0]), jnp.ones((3,), bool)
    )
    # as CSR rows: offsets with or without the end, rows padded, a padded tail
    # (rows 0: [1], 1: [0, 2], 2: [1]; 9 is padding nobody reads)
    entries = jnp.asarray([1, 9, 0, 2, 9, 1, 9, 9])
    offsets, counts = jnp.asarray([0, 2, 5, 6]), jnp.asarray([1, 2, 1])
    assert _csr_pairs_symmetric(entries, offsets, counts)
    assert _csr_pairs_symmetric(entries, offsets[:-1], counts)
    assert not _csr_pairs_symmetric(entries, offsets, jnp.asarray([2, 2, 1]))


def test_the_plan_keeps_the_walks_list_as_walked(lane):
    """A target-sorted walk list stays row offsets in the plan (no per-pair targets)."""
    _, state, plan = lane
    from jaccpot.runtime._interaction_cache import TargetSortedFarPairs

    compact = state.compact_far_pairs
    sorted_list = isinstance(compact, TargetSortedFarPairs)
    assert plan.far_pairs_target_sorted == sorted_list
    assert np.array_equal(
        np.asarray(plan.far_pair_targets),
        np.asarray(jnp.asarray(compact.targets, INDEX_DTYPE)),
    )

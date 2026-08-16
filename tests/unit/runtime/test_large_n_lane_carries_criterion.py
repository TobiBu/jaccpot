"""The large-N lane must accept the Dehnen criterion -- and refuse what it cannot run.

``can_use_large_n_prepare_path`` used to decline on ``_uses_paper_style_force_scale``
outright, which shut ``mac_type="dehnen_error"`` out of the only lane that reaches
N = 10^6. The lane now carries it: ``prepare_large_n_state`` resolves the per-node
force scale between the tree/upward build and the dual build, through the *same*
``_resolve_force_scale_nodes_for_prepare`` the generic lane uses.

Getting the threading wrong is silent rather than loud. ``_prepare_state_dual_and_downward``
builds the policy state from whatever ``force_scale_nodes`` it is handed, and
``build_adaptive_policy_state`` substitutes ``jnp.ones(...)`` for ``None`` -- so a
lane that forgot the prepass would run a threshold of ``eps * 1`` instead of
``eps * min_b |a_b|``, accept far more, run *faster*, and report nothing.

The gate is tested with the GPU predicate stubbed, since the rest of the
selection logic is backend-independent and the lane itself needs a device.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FMMAdvancedConfig
from jaccpot.runtime import _large_n_pipeline
from jaccpot.runtime._large_n_pipeline import can_use_large_n_prepare_path
from jaccpot.solver import FastMultipoleMethod

# Compile-bound: builds solvers and runs full prepares. Marked slow to match the
# sibling MAC tests (see test_dehnen_mac_gradients.py).
pytestmark = pytest.mark.slow

N = 512
LEAF = 16
ORDER = 4
EPS = 3e-3


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(N, 3))
    mass = rng.uniform(0.5, 1.5, size=N)
    return (
        jnp.asarray(pos, dtype=jnp.float64),
        jnp.asarray(mass, dtype=jnp.float64),
    )


def _large_n_solver(mac_type: str, *, runtime_path: str = "auto"):
    """A solver configured the way the large-N GPU lane requires."""

    advanced = FMMAdvancedConfig()
    advanced = replace(
        advanced,
        mac_type=mac_type,
        tree=replace(advanced.tree, tree_type="radix"),
    )
    return FastMultipoleMethod(
        theta=0.6,
        adaptive_eps=EPS,
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        runtime_path=runtime_path,
        advanced=advanced,
    )


@pytest.fixture
def _pretend_gpu(monkeypatch):
    monkeypatch.setattr(_large_n_pipeline.jax, "default_backend", lambda: "gpu")


def _gate(fmm):
    positions, masses = _problem()
    engine = getattr(fmm, "_impl", fmm)
    return can_use_large_n_prepare_path(
        engine,
        positions_arr=positions,
        masses_arr=masses,
        allow_stateful_cache=True,
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_lane_accepts_the_dehnen_error_criterion(_pretend_gpu):
    fmm = _large_n_solver("dehnen_error")
    engine = getattr(fmm, "_impl", fmm)
    assert _gate(fmm) is True, (
        "the large-N lane still declines mac_type='dehnen_error'; declined reason "
        f"= {getattr(engine, '_large_n_path_declined_reason', None)!r}"
    )
    assert getattr(engine, "_large_n_path_declined_reason", None) is None


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_lane_still_refuses_the_folded_angle_mode(_pretend_gpu):
    """`dehnen_theta` has no folding step in this lane, so it must not engage.

    Its criterion lives in ``geometry.radius``, applied before the dual build by
    ``_apply_per_node_effective_theta``. Running it here would fall back to the
    geometric MAC at the solver's ``theta``, which paper mode pins at 1.0 -- so
    acceptance would be wildly loose, not merely different.

    It raises rather than declining quietly, and under the production preset that
    is the only possible outcome: ``_apply_large_n_gpu_production_contract`` pins
    ``runtime_path="large_n"``, which turns every decline into an explicit
    request. The reason is still recorded first, for
    ``get_runtime_diagnostics()``.
    """

    fmm = _large_n_solver("dehnen_theta")
    engine = getattr(fmm, "_impl", fmm)
    assert str(engine.runtime_path) == "large_n", (
        "the large_n_gpu preset no longer pins runtime_path, so a decline here "
        "would be a silent fallback rather than an error"
    )
    with pytest.raises(RuntimeError, match="dehnen_theta"):
        _gate(fmm)
    assert (
        getattr(engine, "_large_n_path_declined_reason", None)
        == "per_node_effective_theta"
    )


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_the_lane_resolves_the_same_force_scale_as_the_generic_lane():
    """One resolver, not two copies.

    The lane calls ``_resolve_force_scale_nodes_for_prepare`` directly, so this
    pins that the shared entry point returns what the generic ``prepare_state``
    path puts in ``_last_force_scale_nodes`` -- the property that would break
    first if the lane grew its own copy of the prepass.
    """

    positions, masses = _problem()
    advanced = FMMAdvancedConfig()
    advanced = replace(advanced, mac_type="dehnen_error")
    fmm = FastMultipoleMethod(theta=0.6, adaptive_eps=EPS, advanced=advanced)
    engine = getattr(fmm, "_impl", fmm)

    fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    from_prepare_state = np.asarray(jax.device_get(engine._last_force_scale_nodes))

    tree_artifacts = engine._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=LEAF,
        max_order=ORDER,
        refine_local_val=False,
        max_refine_levels_val=0,
        aspect_threshold_val=16.0,
        jit_tree_override=None,
        upward_center_mode="com",
        allow_stateful_cache=False,
    )
    engine._last_force_scale_nodes = None
    from_resolver = np.asarray(
        jax.device_get(
            engine._resolve_force_scale_nodes_for_prepare(
                tree_artifacts=tree_artifacts,
                supplied_force_scale=None,
                positions_arr=positions,
                masses_arr=masses,
                bounds=None,
                leaf_size=LEAF,
                max_order=ORDER,
                jit_tree=None,
                upward_center_mode="com",
                runtime_traversal_config=None,
                runtime_m2l_chunk_size=None,
                runtime_l2l_chunk_size=None,
                grouped_interactions=False,
                farfield_mode="pair_grouped",
                record_retry=lambda _event: None,
                refine_local_val=False,
                max_refine_levels_val=0,
                aspect_threshold_val=16.0,
            )
        )
    )

    assert from_resolver.shape == from_prepare_state.shape
    assert np.all(np.isfinite(from_resolver))
    assert not np.allclose(from_resolver, 1.0), (
        "the resolved force scale is all ones, which is the silent fallback "
        "build_adaptive_policy_state substitutes for None -- so this test cannot "
        "tell a real prepass from a missing one"
    )
    np.testing.assert_allclose(from_resolver, from_prepare_state, rtol=1e-12)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_an_injected_force_scale_reaches_the_lane_resolver():
    """``force_scale_nodes=`` must not be dropped on the way into the lane.

    ``prepare_state`` dispatches to the lane before it resolves the scale, so the
    supplied array has to be threaded through ``prepare_large_n_state``. If it
    were dropped, a prepare/evaluate loop would measure the prepass's scale while
    believing it measured the injected one.
    """

    positions, masses = _problem()
    advanced = FMMAdvancedConfig()
    advanced = replace(advanced, mac_type="dehnen_error")
    fmm = FastMultipoleMethod(theta=0.6, adaptive_eps=EPS, advanced=advanced)
    engine = getattr(fmm, "_impl", fmm)
    probe = fmm.prepare_state(positions, masses, leaf_size=LEAF, max_order=ORDER)
    node_count = int(probe.tree.parent.shape[0])

    tree_artifacts = engine._prepare_state_tree_and_upward(
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=LEAF,
        max_order=ORDER,
        refine_local_val=False,
        max_refine_levels_val=0,
        aspect_threshold_val=16.0,
        jit_tree_override=None,
        upward_center_mode="com",
        allow_stateful_cache=False,
    )
    injected = jnp.full((node_count,), 7.0, dtype=positions.dtype)
    resolved = engine._resolve_force_scale_nodes_for_prepare(
        tree_artifacts=tree_artifacts,
        supplied_force_scale=injected,
        positions_arr=positions,
        masses_arr=masses,
        bounds=None,
        leaf_size=LEAF,
        max_order=ORDER,
        jit_tree=None,
        upward_center_mode="com",
        runtime_traversal_config=None,
        runtime_m2l_chunk_size=None,
        runtime_l2l_chunk_size=None,
        grouped_interactions=False,
        farfield_mode="pair_grouped",
        record_retry=lambda _event: None,
        refine_local_val=False,
        max_refine_levels_val=0,
        aspect_threshold_val=16.0,
    )
    np.testing.assert_allclose(np.asarray(jax.device_get(resolved)), 7.0)

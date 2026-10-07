"""What Odisseo's production branch imports and calls from jaccpot -- pinned.

Odisseo (``jaccpot-integration``) is the production consumer and has no test CI of
its own, so a jaccpot change that drops a name, a keyword or a config field it
uses would only surface in a user's run. These tests import every name it imports
and construct every config object with the exact keywords it passes
(``odisseo/jaccpot_coupling.py`` ``_build_solver``, ``odisseo/blockstep_coupling.py``,
``odisseo/mesh_coupling.py``, ``odisseo/differentiable.py``). The method keywords
it calls with are checked against the signatures; nothing is solved here.

When a jaccpot change has to break one of these, the same change lands with an
Odisseo PR, and this file is updated with it.
"""

from __future__ import annotations

import dataclasses
import inspect

import jax.numpy as jnp
import pytest
from yggdrax import DualTreeTraversalConfig

import jaccpot
from jaccpot import (
    BlockStepFMM,
    FarFieldConfig,
    FastMultipoleMethod,
    FMMAdvancedConfig,
    GradConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters)


def _odisseo_solver(
    *,
    preset: str = "fast",
    traversal_config=None,
    retain_far_pairs_for_grad: bool = False,
) -> FastMultipoleMethod:
    """``odisseo.jaccpot_coupling._build_solver`` with SimulationConfig defaults."""
    farfield_kwargs = dict(mode="auto", m2l_chunk_size=None)
    if retain_far_pairs_for_grad:
        farfield_kwargs["retain_far_pairs_for_grad"] = True
    return FastMultipoleMethod(
        preset=preset,
        basis="real",
        runtime_path="auto",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        use_pallas=None,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=32),
            farfield=FarFieldConfig(**farfield_kwargs),
            nearfield=NearFieldConfig(mode="auto", edge_chunk_size=256),
            runtime=RuntimePolicyConfig(
                jit_tree=None,
                jit_traversal=True,
                max_pair_queue=None,
                pair_process_block=None,
                traversal_config=traversal_config,
                prepare_stage_memory_split_enabled=None,
                upward_leaf_batch_size=None,
                fixed_order=None,
                fixed_max_leaf_size=32,
            ),
            mac_type="dehnen",
        ),
    )


@pytest.mark.parametrize("preset", ["fast", "large_n_gpu"])
def test_odisseo_solver_construction(preset):
    solver = _odisseo_solver(preset=preset)
    assert isinstance(solver, FastMultipoleMethod)


def test_odisseo_solver_with_traversal_capacities_and_grad_retention():
    traversal = DualTreeTraversalConfig(
        max_pair_queue=4096,
        process_block=256,
        max_interactions_per_node=64,
        max_neighbors_per_leaf=64,
    )
    solver = _odisseo_solver(traversal_config=traversal, retain_far_pairs_for_grad=True)
    assert isinstance(solver, FastMultipoleMethod)
    # differentiable.py probes the field by name before passing it
    assert "retain_far_pairs_for_grad" in {
        f.name for f in dataclasses.fields(FarFieldConfig)
    }


def test_odisseo_solver_method_keywords():
    assert {"leaf_size", "max_order", "theta", "fused_device_mode", "bounds"} <= (
        _params(FastMultipoleMethod.prepare_state)
    )
    assert {"leaf_size", "max_order", "theta", "fused_device_mode"} <= _params(
        FastMultipoleMethod.refresh_prepared_state
    )
    assert {"target_indices", "return_potential"} <= _params(
        FastMultipoleMethod.evaluate_prepared_state
    )
    assert {"leaf_size", "max_order", "theta", "jit_traversal"} <= _params(
        FastMultipoleMethod.strict_prepare_refresh_and_evaluate
    )
    assert {
        "state",
        "masses",
        "dt",
        "num_steps",
        "refresh_every",
        "leaf_size",
        "max_order",
        "theta",
        "prepared_state",
        "initial_self_acceleration",
        "jit_traversal",
        "add_external",
        "external_acceleration_fn",
        "rematerialize_between_refresh",
        "return_history",
        "step_callback",
        "step_callback_stride",
        "return_prepared_state",
    } <= _params(FastMultipoleMethod.strict_run_v2)
    assert {"grad_plan", "grad_config"} <= _params(
        FastMultipoleMethod.differentiable_accelerations
    )
    assert callable(FastMultipoleMethod.get_runtime_diagnostics)


def test_odisseo_grad_config():
    config = GradConfig(nearfield_lane="auto", fused_m2l_pallas=None)
    assert isinstance(config, GradConfig)


def test_odisseo_large_n_grad_plan_names():
    from jaccpot.runtime._large_n_grad import (
        LargeNPreparedState,
        prepare_large_n_grad_plan,
    )

    assert inspect.isclass(LargeNPreparedState)
    assert callable(prepare_large_n_grad_plan)


def test_odisseo_blockstep_construction():
    """``odisseo.blockstep_coupling`` builds the force model like this."""
    kwargs = dict(
        softening=1e-3,
        k_max=3,
        theta=0.6,
        max_order=4,
        G=1.0,
        basis="real",
        backend="jax",
        leaf_size=32,
        near_chunk_size=None,
        pallas_interpret=False,
    )
    params = _params(BlockStepFMM.__init__)
    assert set(kwargs) <= params
    # passed conditionally, after the same signature probe
    assert {"static_shapes", "topology_backend"} <= params
    force = BlockStepFMM(**kwargs, static_shapes=True, topology_backend="device")
    assert isinstance(force, BlockStepFMM)


def test_odisseo_mutual_force_names():
    from jaccpot.mutual.force import (
        OVERFLOW_CAUSES,
        level_weights_from_floor,
        mutual_weighted_accelerations,
    )

    assert callable(mutual_weighted_accelerations)
    assert callable(level_weights_from_floor)
    assert len(tuple(OVERFLOW_CAUSES)) > 0


def test_odisseo_mesh_lane_config_and_evaluator():
    """``odisseo.mesh_coupling``: the distributed config and evaluator."""
    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
        scatter_to_input_order,
    )

    cfg = DistributedFMMConfig(
        leaf_size=32,
        theta=0.6,
        order=4,
        softening=1e-3,
        G=1.0,
        m2l_chunk=None,
        nearfield_chunk=None,
        nearfield_accum="wide",
        mac_type="dehnen",
        adaptive_eps=None,
        mac_cross_criterion=False,
    ).resolved_for(1024, 2)
    assert isinstance(cfg, DistributedFMMConfig)
    assert {"jit", "halo_exchange"} <= _params(make_force_evaluator)
    assert callable(partition_for_devices)
    assert callable(scatter_to_input_order)
    assert all(isinstance(name, str) for name in DIAG_FIELDS)


def test_odisseo_velocity_verlet_helper():
    """Imported by Odisseo's ``tests/test_strict_velocity_verlet_policy.py``.

    From where it is defined: the god-class split moved it to ``fmm_state``, and the
    ``_fmm_impl`` re-export Odisseo's test used went with the dead-import cleanup
    (e7a2de3), which broke that test unnoticed -- the gap this file closes.
    """
    from jaccpot.runtime.fmm_state import _velocity_verlet_state_update

    assert callable(_velocity_verlet_state_update)


def test_odisseo_top_level_names_stay_exported():
    for name in (
        "BlockStepFMM",
        "FarFieldConfig",
        "FastMultipoleMethod",
        "FMMAdvancedConfig",
        "GradConfig",
        "NearFieldConfig",
        "RuntimePolicyConfig",
        "TreeConfig",
    ):
        assert name in jaccpot.__all__

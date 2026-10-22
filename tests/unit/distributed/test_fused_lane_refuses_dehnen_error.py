"""The fused multi-GPU lane refuses Dehnen's error criterion until it carries it.

Review G7 (2026-10-13): ``mac_type='dehnen_error'`` was neither supported nor
refused on this lane. Its force step refreshes without eq (16b)'s per-particle
force scale ``f_b`` (the single-card scan carries one from step to step), and its
cross-domain walks test the geometric MAC at ``theta`` -- so a user asking for the
criterion silently got another one. Every entry point now says so before it builds
anything; the dev plan's session S12 brings the criterion to the lane.

The mesh lane (``jaccpot.distributed.fmm``) carries the criterion and is untouched:
its own tests (``tests/distributed/test_distributed_dehnen_error_mac.py``, pin M1)
cover it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.runtime.capacity_plan as capacity_plan
from jaccpot.config import FMMAdvancedConfig, TreeConfig
from jaccpot.distributed.cross import make_cross_hook
from jaccpot.distributed.fused import (
    _refuse_error_criterion,
    fused_force_step,
    make_fused_force_evaluator,
)
from jaccpot.distributed.rollout import ShardState, setup_fused_force
from jaccpot.solver import FastMultipoleMethod

AXIS = "gpus"


def _solver(mac_type="dehnen", **kwargs):
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=0.8,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix", leaf_partition="cells", leaf_capacity=64
            ),
            mac_type=mac_type,
        ),
        fixed_order=4,
        **kwargs,
    )


@pytest.fixture
def criterion_solver():
    return _solver("dehnen_error", adaptive_eps=1e-3)


def _mesh():
    return jax.sharding.Mesh(np.asarray(jax.devices()[:1]), (AXIS,))


def _state(n=8):
    f32 = jnp.zeros((n, 3), jnp.float32)
    return ShardState(
        positions=f32,
        velocities=f32,
        masses=jnp.ones((n,), jnp.float32),
        ids=jnp.arange(n, dtype=jnp.int32),
        count=jnp.asarray([n], jnp.int32),
        accel=f32,
    )


def test_setup_refuses_before_preparing_a_shard(criterion_solver, monkeypatch):
    def prepared_anyway(*args, **kwargs):
        raise AssertionError("a shard was prepared before the refusal")

    monkeypatch.setattr(capacity_plan, "measure_shard_plan", prepared_anyway)
    with pytest.raises(NotImplementedError) as excinfo:
        setup_fused_force(
            criterion_solver, _mesh(), _state(), leaf_size=64, max_order=4, theta=0.8
        )
    message = str(excinfo.value)
    # actionable: the entry point, the criterion, the lane, and where it does run
    for needle in (
        "setup_fused_force",
        "mac_type='dehnen_error'",
        "fused multi-GPU lane",
        "single-card",
        "jaccpot.distributed.fmm",
        "mac_type='dehnen'",
    ):
        assert needle in message, needle


def test_the_evaluator_and_the_force_step_refuse(criterion_solver):
    with pytest.raises(NotImplementedError, match="make_fused_force_evaluator"):
        make_fused_force_evaluator(
            criterion_solver, None, mesh=_mesh(), plan=None, leaf_size=64, max_order=4
        )
    positions = jnp.zeros((8, 3), jnp.float32)
    with pytest.raises(NotImplementedError, match="fused_force_step"):
        fused_force_step(
            criterion_solver,
            None,
            positions,
            jnp.ones((8,), jnp.float32),
            bounds=(positions[0], positions[0] + 1.0),
            leaf_size=64,
            max_order=4,
        )


@pytest.mark.parametrize("mac_type", ["dehnen_error", "dehnen_theta"])
def test_the_cross_hook_refuses_the_criterion_names(mac_type):
    with pytest.raises(NotImplementedError, match=f"mac_type='{mac_type}'"):
        make_cross_hook(ndev=2, theta=0.8, mac_type=mac_type)


def test_the_paper_error_model_is_the_same_criterion_and_is_refused():
    """``adaptive_error_model='dehnen_paper'`` runs eq (16a) under a geometric
    ``mac_type`` too (``_flat_walk_criterion_active``); the message names it."""
    solver = _solver("dehnen", adaptive_eps=1e-3, adaptive_error_model="dehnen_paper")
    with pytest.raises(
        NotImplementedError, match="adaptive_error_model='dehnen_paper'"
    ):
        _refuse_error_criterion(solver, entry="setup_fused_force")


def test_geometric_criteria_pass():
    for mac_type in ("dehnen", "bh"):
        assert _refuse_error_criterion(_solver(mac_type), entry="x") is None
    make_cross_hook(ndev=2, theta=0.8, mac_type="dehnen")


def test_the_segment_runner_refuses_before_preparing_a_shard(
    criterion_solver, monkeypatch
):
    """The template route's entry points (S11): the segment runner's fused force and
    its setup refuse before any shard is prepared."""
    from jaccpot.distributed import template_force
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    engine = criterion_solver._impl

    def prepared_anyway(*args, **kwargs):
        raise AssertionError("a shard was prepared before the refusal")

    monkeypatch.setattr(engine, "prepare_state_caps_only", prepared_anyway)
    shard = _state()
    state = MeshState(
        state=jnp.stack([shard.positions, shard.velocities], axis=1),
        masses=shard.masses,
        ids=shard.ids,
        count=shard.count,
    )
    with pytest.raises(NotImplementedError, match="FusedMeshForce"):
        MeshSegmentRunner.fused(
            criterion_solver, _mesh(), state, leaf_size=64, max_order=4, theta=0.8
        )
    with pytest.raises(NotImplementedError, match="prepare_mesh_force"):
        template_force.prepare_mesh_force(
            criterion_solver,
            _mesh(),
            shard.positions,
            shard.masses,
            shard.count,
            leaf_size=64,
            max_order=4,
        )

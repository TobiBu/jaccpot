"""The fused multi-GPU lane's STACKED route refuses Dehnen's error criterion.

Review G7 (2026-10-13): ``mac_type='dehnen_error'`` was neither supported nor
refused on this lane. Its force step refreshes without eq (16b)'s per-particle
force scale ``f_b`` (the single-card scan carries one from step to step), and its
cross-domain walks test the geometric MAC at ``theta`` -- so a user asking for the
criterion silently got another one. Every entry point of the stacked route
(``setup_fused_force``, ``make_fused_force_evaluator``, ``fused_force_step``) says
so before it builds anything, and still does.

S12 brought the criterion to the TEMPLATE route (``MeshSegmentRunner.fused``,
``FusedMeshForce``, ``prepare_mesh_force``): f_b rides its segment carry and its
cross walks evaluate eq (16a) (``tests/distributed/test_mesh_segment_dehnen_error.py``).
There only the variants the single card's flat walk does not run either are
refused (the refuted ``dehnen_theta``), and the cross hook takes the criterion as
``error_criterion=True`` on the symmetric exchange only.

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


class _Prepared(Exception):
    """Raised by a stubbed prepare: the entry point got past its refusal."""


def _template_route_entries(solver, monkeypatch):
    """Call the template route's entry points with a prepare that raises
    :class:`_Prepared`; yield what each raised."""
    from jaccpot.distributed import template_force
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    def prepared(*args, **kwargs):
        raise _Prepared

    monkeypatch.setattr(solver._impl, "prepare_state_caps_only", prepared)
    shard = _state()
    state = MeshState(
        state=jnp.stack([shard.positions, shard.velocities], axis=1),
        masses=shard.masses,
        ids=shard.ids,
        count=shard.count,
    )
    calls = {
        "FusedMeshForce": lambda: MeshSegmentRunner.fused(
            solver, _mesh(), state, leaf_size=64, max_order=4, theta=0.8
        ),
        "prepare_mesh_force": lambda: template_force.prepare_mesh_force(
            solver,
            _mesh(),
            shard.positions,
            shard.masses,
            shard.count,
            leaf_size=64,
            max_order=4,
        ),
    }
    for entry, call in calls.items():
        try:
            call()
        except Exception as exc:  # noqa: BLE001 - what it raised is the result
            yield entry, exc
        else:  # pragma: no cover - the stub always raises
            yield entry, None


def test_the_segment_runner_takes_dehnen_error_since_s12(criterion_solver, monkeypatch):
    """The template route's entry points (S11) refused the criterion until S12
    carried f_b on it: they now reach the prepare."""
    for entry, exc in _template_route_entries(criterion_solver, monkeypatch):
        assert isinstance(exc, _Prepared), (entry, exc)


def test_the_segment_runner_refuses_dehnen_theta_before_preparing(monkeypatch):
    """The refuted ``dehnen_theta`` folds a per-node opening angle the cross walks
    would not apply (the single card's flat walk does not run it either): refused
    before any shard is prepared."""
    solver = _solver("dehnen_theta", adaptive_eps=1e-3)
    for entry, exc in _template_route_entries(solver, monkeypatch):
        assert isinstance(exc, NotImplementedError), (entry, exc)
        assert entry in str(exc) and "mac_type='dehnen_error'" in str(exc)


def test_the_cross_hook_takes_the_criterion_on_the_symmetric_exchange_only(
    monkeypatch,
):
    """``error_criterion=True`` builds on the default (symmetric) exchange and says
    so to the refresh (``hook.error_criterion``); a control exchange, whose walks
    re-test imported rows that carry no eq (16a) table, and the box MAC geometry
    are refused."""
    hook = make_cross_hook(ndev=2, theta=0.8, error_criterion=True)
    assert hook.error_criterion is True
    assert make_cross_hook(ndev=2, theta=0.8).error_criterion is False
    with pytest.raises(NotImplementedError, match="symmetric exchange only"):
        make_cross_hook(ndev=2, theta=0.8, near_theta=0.5, error_criterion=True)
    monkeypatch.setenv("JACCPOT_CROSS_SYMMETRIC", "0")
    with pytest.raises(NotImplementedError, match="symmetric exchange only"):
        make_cross_hook(ndev=2, theta=0.8, error_criterion=True)
    monkeypatch.delenv("JACCPOT_CROSS_SYMMETRIC")
    monkeypatch.setenv("JACCPOT_CROSS_MAC_GEOMETRY", "aabb")
    with pytest.raises(NotImplementedError, match="COM MAC geometry"):
        make_cross_hook(ndev=2, theta=0.8, error_criterion=True)

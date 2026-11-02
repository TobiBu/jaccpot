"""``GradientPlan.rollout(start_force="adjoint")`` (S10 follow-ups): the start force inside the adjoint.

By default (``"outside"``) a seeded segment's start force -- kinds ``"start"``,
``"seed_kept"``, ``"eager_self"`` -- runs under ``jax.checkpoint`` beside the
rollout, its field added eagerly behind a barrier, so the chain's forward is the
runner's bit for bit. ``"adjoint"`` hands it to ``differentiable_rollout``
(``acceleration=None``; a ``"seed_kept"`` / ``"eager_self"`` segment's first step
gets the seed again through ``start_aux``), whose reverse runs it as unit 0 of
its one reverse loop: S9 session 2's memory lever (W 883 -> ~406 B/p at the S9
2e6 Plummer calibration on GPU). Gates, on ``test_runner_gradient_mode``'s
system (CPU, interpret, N = 512, float64, p2, 3 segments x 2 steps, the second
re-planned):

* the forward of the ``"adjoint"`` chain against the runner's trajectory, in
  every lane: Plummer + harmonic field (start / self / carried),
  ``dehnen_error`` + field (start / self / carried), ``dehnen_error`` without
  (start / seed_kept / carried), and ``dehnen_error`` without under
  ``JACCPOT_STRICT_CAPS_ONLY_PREPARE=0`` (start / eager_self / carried);
* a plan without a seeded segment (a restart's) lowers to the same program in
  both modes;
* MANUAL (``JACCPOT_RUN_MANUAL_TESTS=1``; one gradient compile is 19-21 GB of
  RSS on CPU): the value and every gradient of the ``"adjoint"`` chain against
  the ``"outside"`` chain's, in every lane, to round-off.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.strict import test_grad_rollout_differentiable as g
from tests.integration.strict import test_runner_gradient_mode as r

_MANUAL = pytest.mark.skipif(
    os.environ.get("JACCPOT_RUN_MANUAL_TESTS") != "1",
    reason="manual: 19-38 GB of RSS on CPU; set JACCPOT_RUN_MANUAL_TESTS=1",
)

#: the lanes: test_runner_gradient_mode's, and its field-free dehnen_error lane
#: with a full re-prepare at the retry (the "eager_self" kind)
LANES = ("plummer", "dehnen_error", "dehnen_error_nofield", "eager")
#: the second segment's start kind per lane
RETRIED = {
    "plummer": "self",
    "dehnen_error": "self",
    "dehnen_error_nofield": "seed_kept",
    "eager": "eager_self",
}
_PLANS: dict = {}
_GRADS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and the large-N profile's backend gate open.

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    for key in g._ENV_UNSET:
        patcher.delenv(key, raising=False)
    for key, value in g._ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()
    _PLANS.clear()
    _GRADS.clear()


def _lanes(*names: str) -> list:
    return [
        pytest.param(name, marks=pytest.mark.xdist_group(f"runner_start_force_{name}"))
        for name in names
    ]


def _plan(name: str) -> Any:
    """The lane's gradient plan (the pre-pass with the forced re-plan), once."""
    if name not in _PLANS:
        base = "dehnen_error_nofield" if name == "eager" else name
        mac, field, grow_last = r._LANES[base]
        solver = g._solver(mac)
        runner = r._runner(base, solver)
        runner.callbacks = (r._replanning(solver, grow_last),)
        x, v, m, _ = g._system()
        with pytest.MonkeyPatch.context() as patcher:
            if name == "eager":
                patcher.setenv("JACCPOT_STRICT_CAPS_ONLY_PREPARE", "0")
            _PLANS[name] = runner.gradient_plan(
                jnp.stack([x, v], axis=1),
                m,
                dt=g.DT,
                num_steps=r.STEPS,
                external_params={"k": g.K_EXT} if field else None,
            )
    return _PLANS[name]


def _wrt(name: str) -> tuple:
    return r._wrt("dehnen_error_nofield" if name == "eager" else name)


def _gradients(name: str) -> tuple:
    """The ``"outside"`` and the ``"adjoint"`` gradient of the lane's chain."""
    if name not in _GRADS:
        plan = _plan(name)
        _GRADS[name] = tuple(
            plan.value_and_grad(r._loss, wrt=_wrt(name), start_force=mode)
            for mode in ("outside", "adjoint")
        )
    return _GRADS[name]


def _rel(got: Any, want: Any) -> float:
    """The largest difference over the largest magnitude of ``want``."""
    got, want = np.asarray(got), np.asarray(want)
    scale = float(np.max(np.abs(want)))
    return float(np.max(np.abs(got - want))) / (scale if scale else 1.0)


@pytest.mark.parametrize("name", _lanes(*LANES))
def test_the_adjoint_start_forces_forward_is_the_runners_to_round_off(name):
    """The ``"adjoint"`` chain's jitted forward against the runner: the verdict
    clean, the f_b bitwise, the state and the total acceleration to round-off
    (the start total's field is added inside the program there, not eagerly)."""
    from jaccpot.runtime.grad_rollout import check_rollout

    plan = _plan(name)
    kinds = [(s.step, s.kind) for s in plan.segments]
    assert kinds == [(0, "start"), (2, RETRIED[name]), (4, "carried")]
    out = jax.jit(lambda inputs: plan.rollout(inputs, start_force="adjoint"))(
        plan.inputs()
    )
    check_rollout(out, gradient=True)
    forward = plan.forward
    got = jnp.stack([out.positions, out.velocities], axis=1)
    state_bytes = r._bitwise(got, forward.state)
    acc_bytes = r._bitwise(out.acceleration, forward.handle.acceleration)
    state_rel = _rel(got, forward.state)
    acc_rel = _rel(out.acceleration, forward.handle.acceleration)
    print(
        f"\n{name}: adjoint forward vs runner: state {state_bytes} bytes differ "
        f"(rel {state_rel:.2e}), acceleration {acc_bytes} (rel {acc_rel:.2e})"
    )
    assert state_rel <= 1e-14 and acc_rel <= 1e-13
    if forward.handle.force_scale is not None:
        assert r._bitwise(out.aux[0], forward.handle.force_scale) == 0


@pytest.mark.xdist_group("runner_start_force_restart")
def test_a_plan_without_a_seeded_segment_is_the_same_program(tmp_path):
    """A restart's plan (one ``"carried"`` segment from the file's handle) has no
    start force to move: both modes lower to the same program text."""
    x, v, m, _ = g._system()
    r._runner(
        "plummer",
        g._solver("dehnen"),
        restart_every=1,
        restart_path=str(tmp_path / "restart_{segment}.npz"),
    ).run(
        jnp.stack([x, v], axis=1),
        m,
        dt=g.DT,
        num_steps=2 * r.SEGMENT,
        external_params={"k": g.K_EXT},
    )
    resumed = r._runner("plummer", g._solver("dehnen")).gradient_plan(
        dt=g.DT,
        num_steps=2 * r.SEGMENT,
        restart_from=str(tmp_path / "restart_1.npz"),
        external_params={"k": g.K_EXT},
    )
    assert [s.kind for s in resumed.segments] == ["carried"]
    texts = [
        jax.jit(lambda inputs, mode=mode: resumed.rollout(inputs, start_force=mode))
        .lower(resumed.inputs())
        .as_text()
        for mode in ("outside", "adjoint")
    ]
    assert texts[0] == texts[1]


def test_an_unknown_start_force_is_refused():
    """``start_force`` is one of ``START_FORCE_MODES``."""
    from jaccpot.production_grad import START_FORCE_MODES, GradientPlan

    assert START_FORCE_MODES == ("outside", "adjoint")
    plan = GradientPlan(
        engine=None,
        segments=(),
        start={},
        params={},
        boundary=None,
        external_fn=None,
        forward=None,
    )
    with pytest.raises(ValueError, match="start_force must be one of"):
        plan.rollout({}, start_force="inside")


@_MANUAL
@pytest.mark.parametrize("name", _lanes(*LANES))
def test_the_adjoint_start_forces_gradient_is_the_outside_ones(name):
    """The value and every gradient of the ``"adjoint"`` chain against the
    ``"outside"`` chain's: the same function, its start force's reverse run in
    another place; to round-off (the reverse sums are grouped otherwise)."""
    outside, adjoint = _gradients(name)
    assert int(adjoint.result.aux[1]) == 0
    value_rel = _rel(adjoint.value, outside.value)
    rows = []
    for (path, want), got in zip(
        jax.tree_util.tree_leaves_with_path(outside.grads),
        jax.tree_util.tree_leaves(adjoint.grads),
    ):
        rows.append((jax.tree_util.keystr(path), _rel(got, want)))
    print(f"\n{name}: value rel {value_rel:.2e}; grads {rows}")
    assert value_rel <= 1e-14
    for key, rel in rows:
        assert rel <= 1e-12, (key, rel)

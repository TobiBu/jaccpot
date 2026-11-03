"""``GradientPlan.rollout(inputs, arrays=plan.arrays())`` on the fused lane (S10 follow-ups).

The plan's array data (the array leaves of ``dt`` / the force parameters, every
segment's f_b seed, a restart's restored handle) passed as ARGUMENTS of the
jitted chain instead of constants of it. On ``test_runner_gradient_mode``'s
system (CPU, interpret, N = 512, float64, p2):

* the ``dehnen_error`` lane without a field (start / seed_kept / carried: a seed
  in two segments): the forward with the arrays as arguments is the runner's
  trajectory bitwise, its program holds fewer ``[N]`` constants; MANUAL
  (``JACCPOT_RUN_MANUAL_TESTS=1``, ~20 GB of RSS per gradient compile on CPU):
  the gradient with the arrays as arguments is the one without to round-off (the
  value bitwise);
* a restart's plan (its boundary from the file): the forward through
  ``arrays`` is the runner's, bitwise.
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
    reason="manual: ~20 GB of RSS per gradient compile on CPU; set "
    "JACCPOT_RUN_MANUAL_TESTS=1",
)
LANE = "dehnen_error_nofield"


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
    r._PLANS.clear()


def _forward(plan: Any, with_arrays: bool) -> tuple:
    """The chain's jitted forward, the arrays as arguments or not, and its text."""
    if with_arrays:
        fn = jax.jit(lambda inputs, arrays: plan.rollout(inputs, arrays=arrays))
        args: tuple = (plan.inputs(), plan.arrays())
    else:
        fn = jax.jit(lambda inputs: plan.rollout(inputs))
        args = (plan.inputs(),)
    text = fn.lower(*args).as_text()
    return fn(*args), text


def _n_constants(text: str) -> int:
    """The ``[N]`` float64 constants of a program's text."""
    return sum(
        "stablehlo.constant" in line and f"tensor<{g.N}xf64>" in line
        for line in text.splitlines()
    )


@pytest.mark.xdist_group("runner_arrays")
def test_the_forward_with_the_plans_arrays_as_arguments_is_the_runners():
    """Seeds as arguments: the same trajectory (state, acceleration, f_b) as the
    runner's, bitwise, and fewer ``[N]`` constants in the program."""
    plan = r._plan(LANE)
    assert [s.kind for s in plan.segments] == ["start", "seed_kept", "carried"]
    assert sum(s.seed is not None for s in plan.segments) == 2
    forward = plan.forward
    outs = {}
    for with_arrays in (False, True):
        out, text = _forward(plan, with_arrays)
        got = jnp.stack([out.positions, out.velocities], axis=1)
        assert r._bitwise(got, forward.state) == 0
        assert r._bitwise(out.acceleration, forward.handle.acceleration) == 0
        assert r._bitwise(out.aux[0], forward.handle.force_scale) == 0
        outs[with_arrays] = _n_constants(text)
    print(f"\n[N] constants: plan's arrays {outs[False]}, as arguments {outs[True]}")
    assert outs[True] < outs[False]


@pytest.mark.xdist_group("runner_arrays_restart")
def test_a_restarts_boundary_through_arrays_is_the_runners(tmp_path):
    """A plan from a restart file: its boundary (the restored handle) passed
    through ``arrays``; the forward is the runner's, bitwise."""
    x, v, m, _ = g._system()
    kw = dict(dt=g.DT, num_steps=2 * r.SEGMENT, external_params={"k": g.K_EXT})
    whole = r._runner(
        "plummer",
        g._solver("dehnen"),
        restart_every=1,
        restart_path=str(tmp_path / "restart_{segment}.npz"),
    ).run(jnp.stack([x, v], axis=1), m, **kw)
    resumed = r._runner("plummer", g._solver("dehnen")).gradient_plan(
        restart_from=str(tmp_path / "restart_1.npz"), **kw
    )
    assert resumed.boundary is not None
    out, _ = _forward(resumed, True)
    got = jnp.stack([out.positions, out.velocities], axis=1)
    assert r._bitwise(got, whole.state) == 0


@_MANUAL
@pytest.mark.xdist_group("runner_arrays")
def test_the_gradient_with_the_arrays_as_arguments_is_the_same():
    """``jax.value_and_grad`` of the chain with the arrays as arguments against
    the plan's own (constants): the value bitwise, the gradients to round-off (XLA
    folds what depends on constant seeds alone at compile time)."""
    from jaccpot.production import _trace_state_kept
    from jaccpot.production_grad import _install_plan

    plan = r._plan(LANE)
    wrt = r._wrt(LANE)
    inputs = plan.inputs()
    diff = {k: inputs[k] for k in wrt}
    rest = {k: val for k, val in inputs.items() if k not in diff}
    rest.pop("dt")  # a Python float: a literal, from the plan through arrays

    def objective(diff_in, arrays):
        every = {**rest, **diff_in}
        if arrays is None:
            every["dt"] = plan.params["dt"]
        out = plan.rollout(every, arrays=arrays)
        return r._loss(out, every)

    grads = {}
    for name, arrays in (("constants", None), ("arguments", plan.arrays())):
        program = jax.jit(jax.value_and_grad(objective))
        with _trace_state_kept(plan.engine):
            _install_plan(plan.engine, plan.segments[-1].plan)
            grads[name] = program(diff, arrays)
    (v0, g0), (v1, g1) = grads["constants"], grads["arguments"]
    assert r._bitwise(v1, v0) == 0
    rows = []
    for (path, b), a in zip(
        jax.tree_util.tree_leaves_with_path(g0), jax.tree_util.tree_leaves(g1)
    ):
        a, b = np.asarray(a), np.asarray(b)
        rel = float(np.max(np.abs(a - b))) / float(np.max(np.abs(b)))
        rows.append((jax.tree_util.keystr(path), r._bitwise(a, b), rel))
    print(f"\narguments vs constants: (leaf, bytes differing, rel) {rows}")
    # not bitwise: with the seeds as constants XLA folds what depends on them
    # alone at compile time (the f_b thresholds), evaluated otherwise than at run
    # time; round-off of the largest component
    for key, _, rel in rows:
        assert rel <= 1e-12, (key, rel)

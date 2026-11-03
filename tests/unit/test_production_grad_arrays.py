"""``GradientPlan.arrays()`` and ``rollout(inputs, arrays=)``'s binding (S10 follow-ups).

The plan's array data -- the array leaves of ``(dt, force_params)``, every
segment's f_b seed, a restart's restored handle -- placed once, for a program to
take as arguments instead of constants. Here the bookkeeping on a plan without an
engine (nothing is traced); the fused-lane gate is
``tests/integration/strict/test_runner_gradient_arrays.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _plan(boundary=None, dt=0.01, seeds=(np.arange(4.0), None)):
    from jaccpot.production_grad import GradientPlan, GradSegment

    segments = tuple(
        GradSegment(
            step=2 * k,
            steps=2,
            kind="start" if k == 0 else "carried",
            seed=seed,
            retries=0,
            provider=None,
            plan={},
        )
        for k, seed in enumerate(seeds)
    )
    return GradientPlan(
        engine=None,
        segments=segments,
        start={},
        params={
            "dt": dt,
            "force_params": {"G": jnp.asarray(1.0), "order": 2},
            "external_params": None,
        },
        boundary=boundary,
        external_fn=None,
        forward=None,
    )


def test_arrays_holds_the_array_leaves_the_seeds_and_the_boundary():
    """A Python number (``dt``, a static force parameter) is ``None`` in
    ``params`` -- it stays a literal of the program -- and every array is placed
    (a ``jax.Array``) with the plan's value."""
    boundary = {"acceleration": np.ones((4, 3)), "force_scale": None}
    plan = _plan(boundary=boundary)
    arrays = plan.arrays()
    leaves = jax.tree_util.tree_leaves(
        {"dt": plan.params["dt"], "force_params": plan.params["force_params"]}
    )
    assert len(arrays["params"]) == len(leaves) == 3
    assert [p is None for p in arrays["params"]] == [True, False, True]
    assert isinstance(arrays["params"][1], jax.Array)
    assert isinstance(arrays["seeds"][0], jax.Array) and arrays["seeds"][1] is None
    assert np.array_equal(np.asarray(arrays["seeds"][0]), np.arange(4.0))
    assert isinstance(arrays["boundary"]["acceleration"], jax.Array)
    assert arrays["boundary"]["force_scale"] is None
    placed = plan.arrays(jax.devices()[0])
    assert placed["seeds"][0].devices() == {jax.devices()[0]}


def test_bound_fills_the_inputs_and_reads_the_given_arrays():
    """``dt`` / ``force_params`` the inputs leave out come from ``arrays`` (the
    plan's Python numbers kept); the inputs' own win; the seeds and the boundary
    are the given ones."""
    plan = _plan(dt=jnp.asarray(0.01))
    arrays = plan.arrays()
    new_g = jnp.asarray(2.0)
    arrays = {**arrays, "params": (arrays["params"][0], new_g, None)}
    inputs, seeds, boundary = plan._bound({"positions": 1}, arrays)
    assert inputs["positions"] == 1
    assert inputs["force_params"]["G"] is new_g and inputs["force_params"]["order"] == 2
    assert inputs["dt"] is arrays["params"][0]
    assert seeds is not None and seeds[0] is arrays["seeds"][0] and boundary is None
    own_dt = jnp.asarray(0.5)
    inputs, _, _ = plan._bound({"dt": own_dt}, arrays)
    assert inputs["dt"] is own_dt


@pytest.mark.parametrize(
    "change",
    ["params", "seeds", "boundary"],
)
def test_arrays_of_another_plan_are_refused(change):
    plan = _plan()
    arrays = plan.arrays()
    if change == "params":
        arrays["params"] = arrays["params"][:2]
    elif change == "seeds":
        arrays["seeds"] = arrays["seeds"] + (None,)
    else:
        arrays["boundary"] = {"acceleration": jnp.zeros((4, 3))}
    with pytest.raises(ValueError, match="arrays do not match this plan"):
        plan._bound({}, arrays)

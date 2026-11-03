"""``ProductionRunner``'s gradient mode (S9 session 2): a pre-pass, then one gradient.

``runner.gradient_plan(...)`` runs the production runner as always and records, per
segment, the plan its scan traced with, its provider and how its start was made;
``GradientPlan.value_and_grad`` differentiates the chain of those segments in one
program (:mod:`jaccpot.production_grad`). The gates, on CPU with every Pallas kernel
and the GPU's walk in interpret mode, N = 512, float64, order 2, 3 segments of 2
steps, total mass ~1 (``test_grad_rollout_differentiable.py``'s system):

* G1 -- the gradient run's forward IS the runner's trajectory, bitwise (state,
  total acceleration, f_b), through a segment the pre-pass re-planned: a callback
  halves the far cap after the first segment, so the second overflows and is
  retried by ``strict_run_v2`` (the start kinds ``"self"`` with a field and
  ``"seed_kept"`` without), and in the Plummer lane the third segment runs on caps
  doubled at its edge (another plan: its gradient statics are checked against the
  installed one at trace time). Plummer with the harmonic field and traced ``G`` /
  ``eps2``; ``dehnen_error`` with and without the field. The primal inside the
  gradient program equals it too;
* FD -- central differences of the whole chain (every rebuild in it) in x0, v0, m,
  the field's constant (and ``G``, ``eps2`` on Plummer), with every force's lists
  checked identical on the +-h runs (session 1's rule and steps);
* A-vs-A -- the gradient on a second call and on a second compile, bitwise;
* restart -- the gradient run from a restart file the forward runner wrote, in a
  fresh runner: its forward is the runner's; its gradient, pulled back through the
  first segment, is the uninterrupted gradient to round-off;
* refusal -- inputs the settled caps do not hold give a failed verdict, refused.

``jax.default_backend`` reports ``"gpu"``. Interpret-mode reverses compile for
minutes, so the plans are module-scoped and every test is in
``tests/slow_tests.txt``.

Only G1 runs by default. The other tests are MANUAL (``JACCPOT_RUN_MANUAL_TESTS=1``):
one gradient compile of the chain peaks at 18.7 GB (Plummer) / 20.9 GB
(``dehnen_error``) of RSS on CPU, 32.8 / 38.4 GB per test with the second compile,
more than a hosted 16 GB runner holds (2026-11-01). Run them on a large host before
changing :mod:`jaccpot.production_grad` or the rollout's reverse
(``grad_rollout._adjoint_rollout``)::

    JACCPOT_RUN_MANUAL_TESTS=1 pytest -n 2 --dist loadgroup \\
        tests/integration/strict/test_runner_gradient_mode.py
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.strict import test_grad_rollout_differentiable as g

_MANUAL = pytest.mark.skipif(
    os.environ.get("JACCPOT_RUN_MANUAL_TESTS") != "1",
    reason="manual: 19-38 GB of RSS on CPU; set JACCPOT_RUN_MANUAL_TESTS=1",
)

SEGMENT, STEPS = 2, 6
#: the far cap the callback leaves for the second segment (every lane's need is
#: above it: 2048 / 8192 at the start)
SHRUNK_FAR_CAP = 1024


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


def _settings(mac: str) -> Any:
    from jaccpot.production import GalaxySettings

    return GalaxySettings(
        leaf_size=g.LEAF,
        max_order=g.ORDER,
        theta=g.THETA,
        mac_type=mac,
        adaptive_eps=1e-5 if mac == "dehnen_error" else None,
        precision="fp64",
    )


def _replanning(solver: Any, grow_last: bool) -> Any:
    """The callback that forces the pre-pass's re-plan: the far cap halved below
    the second segment's need after the first, and (``grow_last``) doubled after the
    second, so the third segment traces another plan."""

    def callback(edge: Any) -> None:
        if edge.segment not in (1, 2):
            return
        engine = solver._impl
        caps = dict(engine._strict_fused_validated_caps)
        if edge.segment == 1:
            caps["compact_far_pair_capacity"] = SHRUNK_FAR_CAP
        elif grow_last:
            caps["compact_far_pair_capacity"] *= 2
        engine._strict_fused_validated_caps = caps

    return callback


#: the lanes: (MAC, external field, a third plan)
_LANES = {
    "plummer": ("dehnen", True, True),
    "dehnen_error": ("dehnen_error", True, False),
    "dehnen_error_nofield": ("dehnen_error", False, False),
}
_PLANS: dict = {}


def _lanes(*names: str) -> list:
    """``names`` as parameters, each in its lane's xdist group (the lane's plan and
    gradient are cached per worker)."""
    return [
        pytest.param(name, marks=pytest.mark.xdist_group(f"runner_gradient_{name}"))
        for name in names
    ]


def _runner(name: str, solver: Any, **kw: Any) -> Any:
    from jaccpot.production import ProductionRunner, SegmentConfig

    mac, field, _ = _LANES[name]
    return ProductionRunner(
        solver,
        settings=_settings(mac),
        segment=SegmentConfig(steps=SEGMENT, diagnostics_every=0, **kw),
        external_acceleration_fn=g._harmonic if field else None,
    )


def _plan(name: str) -> Any:
    """The lane's gradient plan (the pre-pass with the forced re-plan), once."""
    if name not in _PLANS:
        mac, field, grow_last = _LANES[name]
        solver = g._solver(mac)
        runner = _runner(name, solver)
        runner.callbacks = (_replanning(solver, grow_last),)
        x, v, m, _ = g._system()
        _PLANS[name] = runner.gradient_plan(
            jnp.stack([x, v], axis=1),
            m,
            dt=g.DT,
            num_steps=STEPS,
            external_params={"k": g.K_EXT} if field else None,
        )
    return _PLANS[name]


def _loss(result: Any, inputs: dict) -> Any:
    del inputs
    w = g._system()[3]
    return jnp.sum(w * result.positions) + jnp.sum(w * result.velocities)


def _wrt(name: str) -> tuple:
    mac, field, _ = _LANES[name]
    names = ("positions", "velocities", "masses")
    names += ("external_params",) if field else ()
    # dehnen_error takes the solver's own G and softening only
    return names + (("force_params",) if mac != "dehnen_error" else ())


_GRADS: dict = {}


def _gradient(name: str) -> Any:
    if name not in _GRADS:
        _GRADS[name] = _plan(name).value_and_grad(_loss, wrt=_wrt(name))
    return _GRADS[name]


def _bitwise(got: Any, want: Any) -> int:
    return g._bitwise(np.asarray(got), np.asarray(want))


# ---------------------------------------------------------------------------
# G1: the gradient run's forward is the runner's.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _lanes(*_LANES))
def test_the_gradient_runs_forward_is_the_runners(name):
    """The pre-pass re-planned the second segment (one retry, from its start), and
    the jitted chain over the recorded segments ends on the runner's state, total
    acceleration and f_b, bit for bit."""
    from jaccpot.runtime.grad_rollout import check_rollout

    plan = _plan(name)
    _, field, grow_last = _LANES[name]
    kinds = [(s.step, s.kind, s.retries) for s in plan.segments]
    retried = "self" if field else "seed_kept"
    assert kinds == [(0, "start", 0), (2, retried, 1), (4, "carried", 0)]
    caps = [s.plan["walk_caps"]["compact_far_pair_capacity"] for s in plan.segments]
    statics = [s.provider.grad_statics.num_far_pairs for s in plan.segments]
    assert statics == caps
    if grow_last:
        assert caps[2] == 2 * caps[1]  # a third plan, installed for its trace
    out = jax.jit(lambda inputs: plan.rollout(inputs))(plan.inputs())
    check_rollout(out, gradient=True)
    forward = plan.forward
    assert forward.step == STEPS
    got = jnp.stack([out.positions, out.velocities], axis=1)
    assert _bitwise(got, forward.state) == 0
    assert _bitwise(out.acceleration, forward.handle.acceleration) == 0
    if forward.handle.force_scale is not None:
        assert _bitwise(out.aux[0], forward.handle.force_scale) == 0


@pytest.mark.xdist_group("runner_gradient_freeze_cache")
def test_a_shared_freeze_cache_compiles_the_freeze_once(monkeypatch):
    """``gradient_plan(..., freeze_cache=)`` (opt-in, S10: one plan per shooting
    segment, re-planned many times): the first pre-pass with the cache compiles the
    freeze rebuild and keeps it; a second one traces no rebuild at all (its scans
    are the engine's, cached by the same key). Both plans are the plan without the
    cache: the same start, plan snapshot, frozen statics and forward, bitwise."""
    from jaccpot.runtime.fused_provider import FusedForceProvider

    runner = _runner("plummer", g._solver("dehnen"))
    x, v, m, _ = g._system()
    state = jnp.stack([x, v], axis=1)
    kw = dict(dt=g.DT, num_steps=SEGMENT, external_params={"k": g.K_EXT})
    plain = runner.gradient_plan(state, m, **kw)
    cache: dict = {}
    first = runner.gradient_plan(state, m, freeze_cache=cache, **kw)
    assert len(cache) == 1
    entry = next(iter(cache.values()))
    assert entry[0] is runner.engine
    traced: list = []
    rebuild = FusedForceProvider.rebuild

    def counting(self, *args, **kwargs):
        traced.append(1)
        return rebuild(self, *args, **kwargs)

    monkeypatch.setattr(FusedForceProvider, "rebuild", counting)
    second = runner.gradient_plan(state, m, freeze_cache=cache, **kw)
    assert traced == [] and next(iter(cache.values())) is entry
    want = plain.segments[0]
    for plan in (first, second):
        (seg,) = plan.segments
        assert (seg.kind, seg.retries) == (want.kind, want.retries) == ("start", 0)
        assert seg.plan == want.plan
        assert seg.provider.grad_statics == want.provider.grad_statics
        assert _bitwise(plan.forward.state, plain.forward.state) == 0


@_MANUAL
@pytest.mark.parametrize("name", _lanes("plummer", "dehnen_error"))
def test_the_gradient_is_finite_and_repeats_bitwise(name):
    """Every gradient finite and nonzero, the verdict clean, the primal inside the
    gradient program the runner's state; a second call and a second compile of the
    gradient, bitwise."""
    plan = _plan(name)
    res = _gradient(name)
    assert int(res.result.aux[1]) == 0
    for key, value in res.grads.items():
        for leaf in jax.tree_util.tree_leaves(value):
            leaf = np.asarray(leaf)
            assert np.all(np.isfinite(leaf)) and np.any(leaf != 0), key
    got = jnp.stack([res.result.positions, res.result.velocities], axis=1)
    assert _bitwise(got, plan.forward.state) == 0
    again = plan.value_and_grad(_loss, wrt=_wrt(name))
    fresh = dataclasses.replace(plan, _programs={}).value_and_grad(
        _loss, wrt=_wrt(name)
    )
    for other in (again, fresh):
        pairs = zip(
            jax.tree_util.tree_leaves(other.grads), jax.tree_util.tree_leaves(res.grads)
        )
        assert sum(_bitwise(a, b) for a, b in pairs) == 0
        assert _bitwise(other.value, res.value) == 0


# ---------------------------------------------------------------------------
# FD: central differences of the whole chain, one map.
# ---------------------------------------------------------------------------


class _Recording:
    """A provider whose differentiable step also sends its lists' digest to the
    host (``jax.debug.callback``), for the identity check of the +-h runs."""

    def __init__(self, provider: Any, sink: list) -> None:
        self.provider = provider
        self.sink = sink

    def differentiable_step(self, x, m, params, aux):
        f_b = None if aux is None else aux[0]
        stop = jax.lax.stop_gradient
        prepared = self.provider.rebuild(
            stop(x),
            stop(m),
            force_scale=None if f_b is None else stop(f_b),
            keep_far_list=True,
        ).prepared
        jax.debug.callback(
            lambda d: self.sink.append(np.asarray(d)),
            g._list_digest(prepared),
            ordered=True,
        )
        return self.provider.differentiable_step(x, m, params, aux)


def _recording(plan: Any) -> tuple:
    """The chain's jitted forward loss with every force's lists recorded."""
    sink: list = []
    segments = tuple(
        dataclasses.replace(s, provider=_Recording(s.provider, sink))
        for s in plan.segments
    )
    recorded = dataclasses.replace(plan, segments=segments, _programs={})

    @jax.jit
    def run(inputs):
        out = recorded.rollout(inputs, checkpoint=None)
        return _loss(out, inputs), out.aux[1]

    def call(inputs: dict) -> tuple:
        sink.clear()
        value, bits = run(inputs)
        jax.effects_barrier()
        return float(value), int(bits), np.stack(sink)

    return call


#: session 1's measured steps (``test_grad_rollout_differentiable._FD_STEPS``):
#: absolute along unit-variance directions for the arrays, relative for scalars
_FD_STEPS = {
    "plummer": {
        "positions": 1e-7,
        "velocities": 1e-4,
        "masses": 1e-7,
        "k": 1e-3,
        "G": 1e-3,
        "eps2": 1e-4,
    },
    "dehnen_error": {"positions": 1e-7, "velocities": 1e-4, "masses": 1e-7, "k": 1e-3},
}


def _moved(inputs: dict, which: str, delta: Any) -> dict:
    out = dict(inputs)
    if which in ("positions", "velocities", "masses"):
        out[which] = inputs[which] + delta
    elif which == "k":
        out["external_params"] = {"k": inputs["external_params"]["k"] + delta}
    else:
        out["force_params"] = dataclasses.replace(
            inputs["force_params"],
            **{which: getattr(inputs["force_params"], which) + delta},
        )
    return out


@_MANUAL
@pytest.mark.parametrize("name", _lanes("plummer", "dehnen_error"))
def test_the_chains_gradient_matches_finite_differences(name):
    """Central differences of the 6-step chain (3 segments, one re-planned) in
    float64, every force's lists (the start forces and every step) identical on
    both +-h runs; relative error at most 1e-7."""
    from jaccpot.runtime.fused_topology import OVERFLOW_PARAMS

    plan = _plan(name)
    res = _gradient(name)
    run = _recording(plan)
    centre = plan.inputs()
    value0, bits0, digests0 = run(centre)
    assert bits0 == 0 and value0 == pytest.approx(float(res.value), rel=1e-13)
    # every force of the chain recorded: the start forces (the first segment's; the
    # retried one starts from the previous self-gravity plus the field) and the steps
    starts = sum(s.kind in ("start", "seed_kept") for s in plan.segments)
    assert starts == 1 and digests0.shape[0] == starts + STEPS
    rng = np.random.default_rng(7)
    for which, h in _FD_STEPS[name].items():
        if which in ("positions", "velocities", "masses"):
            d = jnp.asarray(rng.normal(size=centre[which].shape))
            exact = float(jnp.sum(res.grads[which] * d))
        elif which == "k":
            d = jnp.asarray(1.0)
            h = h * g.K_EXT
            exact = float(res.grads["external_params"]["k"])
        else:
            d = jnp.asarray(1.0)
            h = h * float(getattr(centre["force_params"], which))
            exact = float(getattr(res.grads["force_params"], which))
        values = []
        for sign in (1.0, -1.0):
            value, bits, digests = run(_moved(centre, which, sign * h * d))
            # eps2 + h is past the softening the walk was built for: the
            # verdict says so (OVERFLOW_PARAMS), and the identical lists below say
            # the map is the same one
            assert bits & ~OVERFLOW_PARAMS == 0, (which, sign, bits)
            assert np.array_equal(
                digests, digests0
            ), f"{which}: the {'+' if sign > 0 else '-'}h run rebuilt other lists"
            values.append(value)
        fd = (values[0] - values[1]) / (2.0 * h)
        assert exact != 0.0, which
        assert abs(exact - fd) <= 1e-7 * abs(fd), (which, exact, fd)


# ---------------------------------------------------------------------------
# Restart: the gradient run from a restart file.
# ---------------------------------------------------------------------------


@_MANUAL
@pytest.mark.xdist_group("runner_gradient_restart")
def test_a_gradient_from_a_restart_continues_the_uninterrupted_one(tmp_path):
    """The forward runner writes a restart after its first segment; a FRESH runner
    (another solver) makes a gradient plan from that file. Its forward is the
    runner's final state, bitwise. Its gradient with respect to the restart state,
    pulled back through the uninterrupted chain's first segment, plus its own
    gradient in the masses and the field's constant, is the uninterrupted gradient
    to round-off (1e-12 of each array's largest component): the start
    acceleration's value is the file's, its tangent the force recomputed at the
    restart positions (``production_grad._keep_value``), and the reverse sums are
    grouped otherwise (measured 1.8e-16). Plummer lane, no forced re-plan. Under
    ``dehnen_error`` it is 3e-8 / 6e-8 (positions / masses), not round-off, and not
    gated here: the file's f_b is the restart force's OUTPUT, so the recompute's
    thresholds pick neighbouring lists (``production_grad``'s docstring)."""
    from jaccpot.production import _trace_state_kept

    x, v, m, _ = g._system()
    whole = _runner(
        "plummer",
        g._solver("dehnen"),
        restart_every=1,
        restart_path=str(tmp_path / "restart_{segment}.npz"),
    ).gradient_plan(
        jnp.stack([x, v], axis=1),
        m,
        dt=g.DT,
        num_steps=2 * SEGMENT,
        external_params={"k": g.K_EXT},
    )
    assert [s.kind for s in whole.segments] == ["start", "carried"]
    resumed = _runner("plummer", g._solver("dehnen")).gradient_plan(
        dt=g.DT,
        num_steps=2 * SEGMENT,
        restart_from=str(tmp_path / "restart_1.npz"),
        external_params={"k": g.K_EXT},
    )
    assert resumed.forward.restored["bitwise"], resumed.forward.restored
    assert [(s.step, s.kind) for s in resumed.segments] == [(SEGMENT, "carried")]
    out = jax.jit(lambda inputs: resumed.rollout(inputs))(resumed.inputs())
    got = jnp.stack([out.positions, out.velocities], axis=1)
    assert _bitwise(got, whole.forward.state) == 0
    assert _bitwise(got, resumed.forward.state) == 0

    wrt = ("positions", "velocities", "masses", "external_params")
    full = whole.value_and_grad(_loss, wrt=wrt)
    part = resumed.value_and_grad(_loss, wrt=wrt)
    assert _bitwise(part.value, full.value) == 0
    head = dataclasses.replace(whole, segments=whole.segments[:1], _programs={})

    def to_edge(x0, v0, m0, k0):
        inputs = {
            **whole.inputs(),
            "positions": x0,
            "velocities": v0,
            "masses": m0,
            "external_params": {"k": k0},
        }
        out = head.rollout(inputs)
        return out.positions, out.velocities

    centre = whole.inputs()

    @jax.jit
    def pulled(cotangent, x0, v0, m0, k0):
        return jax.vjp(to_edge, x0, v0, m0, k0)[1](cotangent)

    with _trace_state_kept(whole.engine):
        gx, gv, gm, gk = pulled(
            (part.grads["positions"], part.grads["velocities"]),
            centre["positions"],
            centre["velocities"],
            centre["masses"],
            centre["external_params"]["k"],
        )
    composed = {
        "positions": gx,
        "velocities": gv,
        "masses": gm + part.grads["masses"],
        "k": gk + part.grads["external_params"]["k"],
    }
    wanted = {**full.grads, "k": full.grads["external_params"]["k"]}
    for key, value in composed.items():
        want = np.asarray(wanted[key])
        scale = float(np.max(np.abs(want)))
        err = float(np.max(np.abs(np.asarray(value) - want)))
        assert err <= 1e-12 * scale, (key, err, scale)


# ---------------------------------------------------------------------------
# The refusal.
# ---------------------------------------------------------------------------


@_MANUAL
@pytest.mark.xdist_group("runner_gradient_plummer")
def test_inputs_the_settled_caps_do_not_hold_are_refused():
    """The Plummer lane's compiled gradient at a clipped Plummer sphere (far more
    pairs than the four blobs' caps): the verdict fails and the gradient is
    refused, not returned."""
    from jaccpot.runtime.grad_rollout import RolloutVerdictError

    plan = _plan("plummer")
    _gradient("plummer")  # compiled
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, g.N)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu, phi = rng.uniform(-1.0, 1.0, g.N), rng.uniform(0.0, 2.0 * np.pi, g.N)
    st = np.sqrt(1.0 - mu * mu)
    sphere = jnp.asarray(
        np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    )
    with pytest.raises(RolloutVerdictError, match="gradient verdict failed"):
        plan.value_and_grad(_loss, wrt=_wrt("plummer"), inputs={"positions": sphere})

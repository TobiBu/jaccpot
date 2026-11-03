"""Forward mode (``ad_mode="forward"``) through the production gradient lane: the F gates.

``jax.jvp``, ``jax.jacfwd`` and ``jax.linearize`` through
``FusedForceProvider.differentiable_step`` and ``differentiable_rollout(...,
ad_mode="forward")`` (a plain scan), on the S9 G2 system of
``test_grad_rollout_differentiable.py`` (N = 512, four blobs, total mass ~1, order 2,
leaf 16, theta 0.6, float64, 3 steps; every Pallas kernel and the GPU's walk in
interpret mode on CPU):

* F1 -- the jvp's primal IS the production forward, bitwise: the step's against
  ``FusedForceProvider.step`` and the rollout's against
  ``strict_run_v2(carry="particles")``, Plummer and ``mac_type='dehnen_error'``;
* F2 -- the adjoint identity ``<w, J d> == <J^T w, d>`` against the reverse path
  (``checkpoint="step"``: the hand-written rollout adjoint over the custom_vjp
  kernels), to round-off, for the start positions, velocities, masses, ``G``,
  ``eps2``, ``dt`` and the external field's parameter, one direction each;
* F3 -- central differences of the rollout along those directions, each +-h run's
  lists identical to the centre's (``_FD_STEPS``, ``_list_digest``), 1e-7 relative;
* F5 -- ``jax.jacfwd`` over a few parameters (``G``, ``eps2``, ``dt``, a mass scale)
  equals the reverse path's ``jax.jacrev``, and the batched tangents equal single
  jvps to round-off;
* F7 -- caps sized on a configuration with fewer pairs: the step's verdict fails
  and its tangent is NaN (never a plausible wrong number);
* F9 -- ``jax.linearize`` applied to three directions equals three jvps.

Every force of the rollout has far pairs (F3's recorded digests). The tangents run
on whatever :func:`~jaccpot.runtime._forward_seams.resolve_tangent` picks for the
environment's ``JACCPOT_GRAD_FORWARD_TANGENT_KERNELS`` (default ``"auto"``: the
Pallas tangent kernels); ``=jnp`` runs every gate on the jnp references. Interpret-mode programs compile
for minutes, so every test is in ``tests/slow_tests.txt`` and the file is a CI shard
of its own (``integration-strict-grad-forward``).
"""

from __future__ import annotations

import gc
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.strict.test_grad_rollout_differentiable import (
    _ENV,
    _ENV_UNSET,
    _FD_STEPS,
    DT,
    K_EXT,
    LEAF,
    ORDER,
    SOFTENING,
    STEPS,
    THETA,
    _field_kw,
    _lane,
    _recorded,
    _solver,
    _system,
)

#: the inputs the gates differentiate, in the programs' argument order
_INPUTS = ("x0", "v0", "m", "G", "eps2", "k", "dt")

_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and the large-N profile's backend gate open.

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    # JACCPOT_GRAD_FORWARD_TANGENT_KERNELS is kept: the gates run on any choice
    for key in (*_ENV_UNSET, "JACCPOT_GRAD_AD_MODE"):
        patcher.delenv(key, raising=False)
    for key, value in _ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()
    _CACHE.clear()
    _drop_executables()


def _drop_executables() -> None:
    jax.clear_caches()
    gc.collect()


def _primals(lane: Any) -> tuple:
    return (
        lane.positions,
        lane.velocities,
        lane.masses,
        jnp.asarray(1.0),
        jnp.asarray(SOFTENING**2),
        jnp.asarray(K_EXT),
        jnp.asarray(DT),
    )


def _rollout(lane: Any, ad_mode: str, checkpoint: Any = None):
    """``(loss, RolloutResult)`` of the rollout as a function of :data:`_INPUTS`."""
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def run(x, v, m, g, eps2, k, dt):
        out = differentiable_rollout(
            lane.provider.differentiable_step,
            x,
            v,
            m,
            dt,
            STEPS,
            aux=lane.aux0,
            checkpoint=checkpoint,
            ad_mode=ad_mode,
            **_field_kw(lane, g, eps2, k),
        )
        value = jnp.sum(lane.weight * out.positions)
        return value + jnp.sum(lane.weight * out.velocities), out

    return run


def _directions(lane: Any, seed: int = 7) -> list:
    """One direction per input (the others zero): unit-variance normal arrays,
    a unit scalar."""
    primals = _primals(lane)
    rng = np.random.default_rng(seed)
    out = []
    for i, p in enumerate(primals):
        d = [jnp.zeros_like(q) for q in primals]
        d[i] = jnp.asarray(rng.normal(size=p.shape) if p.ndim else 1.0, p.dtype)
        out.append(tuple(d))
    return out


def _forward(name: str) -> Any:
    """The jitted jvp of the rollout's loss and state (compiled once per lane)."""
    key = ("forward", name)
    if key not in _CACHE:
        lane = _lane(name)
        run = _rollout(lane, "forward")

        @jax.jit
        def program(primals: tuple, tangents: tuple) -> tuple:
            (value, out), (t_value, t_out) = jax.jvp(run, primals, tangents)
            return (
                value,
                out.positions,
                out.velocities,
                out.aux[1],
                t_value,
                t_out.positions,
                t_out.velocities,
            )

        _CACHE[key] = program
    return _CACHE[key]


def _reverse_gradient(name: str) -> tuple:
    """The reverse path's gradient of the loss in every input (``"step"``)."""
    key = ("reverse", name)
    if key not in _CACHE:
        lane = _lane(name)
        run = _rollout(lane, "reverse", "step")
        grads = jax.jit(
            jax.grad(lambda *p: run(*p)[0], argnums=tuple(range(len(_INPUTS))))
        )(*_primals(lane))
        _CACHE[key] = tuple(np.asarray(g) for g in grads)
        _drop_executables()
    return _CACHE[key]


def _bitwise_equal(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# F1: the primal is production's.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_the_jvp_primal_is_the_production_forward(name):
    """The step's jvp primal (accelerations and aux) is ``step``'s bitwise; the
    rollout's is ``strict_run_v2(carry="particles")``'s, and its tangent finite."""
    from jaccpot.runtime.grad_options import ad_mode_override

    lane = _lane(name)
    prov, x, m = lane.provider, lane.positions, lane.masses
    aux_in = lane.aux0
    want_acc, want_aux = jax.jit(lambda y: prov.step(y, m, None, aux_in))(x)
    d = _directions(lane)[0][0]
    with ad_mode_override("forward"):
        (acc, aux), (t_acc, _) = jax.jit(
            lambda y, t: jax.jvp(
                lambda z: prov.differentiable_step(z, m, None, aux_in), (y,), (t,)
            )
        )(x, d)
    _drop_executables()
    assert _bitwise_equal(acc, want_acc)
    for got, want in zip(
        jax.tree_util.tree_leaves(aux), jax.tree_util.tree_leaves(want_aux)
    ):
        assert _bitwise_equal(got, want)
    assert bool(jnp.all(jnp.isfinite(t_acc))) and float(jnp.max(jnp.abs(t_acc))) > 0

    # the rollout with literal parameters and no field (the runner's) against
    # strict_run_v2(carry="particles")
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    def literal(y, v):
        out = differentiable_rollout(
            prov.differentiable_step, y, v, m, DT, STEPS, aux=aux_in, ad_mode="forward"
        )
        return out.positions, out.velocities, out.aux[1]

    (xt, vt, bits), (tx, _, _) = jax.jit(
        lambda y, v, t: jax.jvp(literal, (y, v), (t, jnp.zeros_like(v)))
    )(x, lane.velocities, d)
    _drop_executables()
    assert int(bits) == 0
    solver = _solver("dehnen_error" if name == "dehnen_error" else "dehnen")
    state = jnp.stack([lane.positions, lane.velocities], axis=1)
    want, _, _ = solver.strict_run_v2(
        state=state,
        masses=lane.masses,
        dt=DT,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        carry="particles",
    )
    _drop_executables()
    got = np.asarray(jnp.stack([xt, vt], axis=1))
    assert _bitwise_equal(
        got, want
    ), f"{int(np.sum(got != np.asarray(want)))} of {got.size} elements differ"
    assert bool(jnp.all(jnp.isfinite(tx))) and float(jnp.max(jnp.abs(tx))) > 0

    # every input traced (G, eps2, the field's constant, dt): the jvp program's
    # primal against the same rollout's default (reverse-mode) forward
    primals = _primals(lane)
    value, xt, vt, bits, t_value, tx, tv = _forward(name)(primals, _directions(lane)[0])
    assert int(bits) == 0
    want_value, want_out = jax.jit(_rollout(lane, "reverse"))(*primals)
    _drop_executables()
    assert _bitwise_equal(xt, want_out.positions)
    assert _bitwise_equal(vt, want_out.velocities)
    assert _bitwise_equal(value, want_value)
    assert np.all(np.isfinite(np.asarray(tx))) and float(t_value) != 0.0


# ---------------------------------------------------------------------------
# F2 + F3: the adjoint identity, and finite differences on identical lists.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_the_tangent_is_the_adjoints_transpose(name):
    """``<w, J d>`` (forward mode) against ``<J^T w, d>`` (the reverse path's
    gradient), per input, to round-off in float64."""
    lane = _lane(name)
    primals = _primals(lane)
    grads = _reverse_gradient(name)
    program = _forward(name)
    for input_name, d, g in zip(_INPUTS, _directions(lane), grads):
        if name == "dehnen_error" and input_name in ("G", "eps2"):
            # mac_type='dehnen_error' takes the solver's own G and eps2
            continue
        *_, bits, t_value, _, _ = program(primals, d)
        assert int(bits) == 0
        i = _INPUTS.index(input_name)
        reverse = float(np.sum(g * np.asarray(d[i])))
        forward = float(t_value)
        assert reverse != 0.0, input_name
        assert abs(forward - reverse) <= 1e-11 * abs(reverse), (
            input_name,
            forward,
            reverse,
        )


@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_the_tangent_matches_finite_differences(name):
    """Central differences of the 3-step rollout along each input's direction,
    every +-h run's lists the centre's (#316), 1e-7 relative (the G2 gate's)."""
    lane = _lane(name)
    primals = _primals(lane)
    program = _forward(name)
    run = _recorded(lane)
    centre = primals[:6]  # the recording rollout's inputs (dt is its literal)
    _, digests0, bits0 = run(*centre)
    assert int(bits0) == 0
    assert all(int(c) > 0 for c in np.asarray(digests0)[:, -1]), "no far pairs"
    for input_name, d in zip(_INPUTS[:6], _directions(lane)):
        if input_name not in _FD_STEPS[name]:
            continue
        i = _INPUTS.index(input_name)
        a, di = primals[i], d[i]
        h = _FD_STEPS[name][input_name] * (abs(float(a)) if a.ndim == 0 else 1.0)
        values = []
        for sign in (1.0, -1.0):
            moved = list(centre)
            moved[i] = a + sign * h * di
            value, digests, _ = run(*moved)
            assert np.array_equal(np.asarray(digests), np.asarray(digests0)), (
                f"{input_name}: the {'+' if sign > 0 else '-'}h run rebuilt "
                "other lists"
            )
            values.append(float(value))
        fd = (values[0] - values[1]) / (2.0 * h)
        exact = float(program(primals, d)[4])
        assert exact != 0.0, input_name
        assert abs(exact - fd) <= 1e-7 * abs(fd), (input_name, exact, fd)


# ---------------------------------------------------------------------------
# F5: jacfwd is jacrev; batched tangents are single ones.
# ---------------------------------------------------------------------------


def test_jacfwd_over_parameters_is_jacrev():
    """``jax.jacfwd`` of the final state's two weighted sums over ``(G, eps2, dt,
    mass scale)`` against the reverse path's ``jax.jacrev``; and each column
    against a single jvp."""
    lane = _lane("plummer")
    primals = _primals(lane)

    def of_theta(ad_mode: str, checkpoint: Any) -> Any:
        run = _rollout(lane, ad_mode, checkpoint)

        def f(theta):
            g, eps2, dt, scale = theta[0], theta[1], theta[2], theta[3]
            _, out = run(
                primals[0],
                primals[1],
                primals[2] * scale,
                g,
                eps2,
                primals[5],
                dt,
            )
            return jnp.stack(
                [
                    jnp.sum(lane.weight * out.positions),
                    jnp.sum(lane.weight * out.velocities),
                ]
            )

        return f

    theta = jnp.asarray([1.0, SOFTENING**2, DT, 1.0])
    jf = np.asarray(jax.jit(jax.jacfwd(of_theta("forward", None)))(theta))
    _drop_executables()
    jr = np.asarray(jax.jit(jax.jacrev(of_theta("reverse", "step")))(theta))
    _drop_executables()
    assert jf.shape == jr.shape == (2, 4)
    assert np.all(jr != 0.0)
    assert np.max(np.abs(jf - jr) / np.abs(jr)) <= 1e-11, (jf, jr)
    from jaccpot.runtime.grad_options import ad_mode_override

    f = of_theta("forward", None)
    single = jax.jit(lambda t, d: jax.jvp(f, (t,), (d,))[1])
    with ad_mode_override("forward"):
        for k in range(4):
            col = np.asarray(single(theta, jnp.eye(4)[k]))
            assert np.max(np.abs(col - jf[:, k])) <= 1e-13 * np.max(np.abs(jf[:, k]))
    _drop_executables()


# ---------------------------------------------------------------------------
# F7: an overflow makes the tangent NaN.
# ---------------------------------------------------------------------------


def test_undersized_caps_make_the_tangent_nan():
    """Caps sized on the four blobs, the step on a clipped Plummer sphere (more
    pairs): the verdict bits are set and the jvp's tangent is NaN, every element;
    the same step on its own blobs has a finite tangent."""
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_topology import overflow_ok
    from jaccpot.runtime.grad_options import ad_mode_override

    solver = _solver()
    x, _, m, _ = _system()
    small, _ = FusedForceProvider.prepare(
        solver, x, m, leaf_size=LEAF, max_order=ORDER, theta=THETA, differentiable=True
    )
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, x.shape[0])
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu, phi = rng.uniform(-1.0, 1.0, x.shape[0]), rng.uniform(
        0.0, 2.0 * np.pi, x.shape[0]
    )
    st = np.sqrt(1.0 - mu * mu)
    plummer = jnp.asarray(
        np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    )
    d = jnp.asarray(np.random.default_rng(3).normal(size=x.shape))
    with ad_mode_override("forward"):
        program = jax.jit(
            lambda y, t: jax.jvp(lambda z: small.differentiable_step(z, m), (y,), (t,))
        )
        (_, (_, bits, _)), (t_acc, _) = program(plummer, d)
        (_, (_, bits_ok, _)), (t_ok, _) = program(x, d)
    _drop_executables()
    assert not bool(overflow_ok(bits))
    assert bool(jnp.all(jnp.isnan(t_acc)))
    assert bool(overflow_ok(bits_ok))
    assert bool(jnp.all(jnp.isfinite(t_ok)))


# ---------------------------------------------------------------------------
# F9: linearize is jvp.
# ---------------------------------------------------------------------------


def test_linearize_over_three_directions_is_three_jvps():
    """``jax.linearize`` once, its linear map on three directions (x0, m, G),
    against the jvp program on the same directions."""
    from jaccpot.runtime.grad_options import ad_mode_override

    lane = _lane("plummer")
    primals = _primals(lane)
    run = _rollout(lane, "forward")
    directions = _directions(lane)
    picked = [directions[_INPUTS.index(n)] for n in ("x0", "m", "G")]

    def lin_three(p: tuple, ds: list) -> list:
        _, f_lin = jax.linearize(lambda *q: run(*q)[0], *p)
        return [f_lin(*d) for d in ds]

    with ad_mode_override("forward"):
        got = jax.jit(lin_three)(primals, picked)
    _drop_executables()
    program = _forward("plummer")
    for g, d in zip(got, picked):
        want = float(program(primals, d)[4])
        assert abs(float(g) - want) <= 1e-13 * abs(want), (float(g), want)


# ---------------------------------------------------------------------------
# GradientPlan: the production runner's gradient run, forward mode.
# ---------------------------------------------------------------------------


def _plan() -> Any:
    """A two-segment plan (2 + 2 steps, Plummer, the harmonic field), once."""
    if "plan" not in _CACHE:
        from jaccpot.production import GalaxySettings, ProductionRunner, SegmentConfig
        from tests.integration.strict.test_grad_rollout_differentiable import (
            _harmonic,
        )

        runner = ProductionRunner(
            _solver("dehnen"),
            settings=GalaxySettings(
                leaf_size=LEAF,
                max_order=ORDER,
                theta=THETA,
                mac_type="dehnen",
                precision="fp64",
            ),
            segment=SegmentConfig(steps=2, diagnostics_every=0),
            external_acceleration_fn=_harmonic,
        )
        x, v, m, _ = _system()
        _CACHE["plan"] = runner.gradient_plan(
            jnp.stack([x, v], axis=1),
            m,
            dt=DT,
            num_steps=4,
            external_params={"k": K_EXT},
        )
    return _CACHE["plan"]


def _plan_loss(result: Any, inputs: dict) -> Any:
    del inputs
    w = _system()[3]
    return jnp.sum(w * result.positions) + jnp.sum(w * result.velocities)


def test_the_gradient_plans_forward_twins():
    """``GradientPlan.jvp``: its value is the runner's final state and total
    acceleration, bitwise; ``linearize``'s map and ``jacfwd``'s column are the
    jvp's tangent; inputs the settled caps do not hold are refused."""
    from jaccpot.runtime.grad_rollout import RolloutVerdictError

    plan = _plan()
    assert [s.kind for s in plan.segments] == ["start", "carried"]
    dx = jnp.asarray(np.random.default_rng(5).normal(size=(512, 3)))
    res = plan.jvp(None, {"positions": dx})
    positions, velocities, acceleration = res.value
    forward = plan.forward
    got = np.asarray(jnp.stack([positions, velocities], axis=1))
    assert _bitwise_equal(got, forward.state)
    assert _bitwise_equal(acceleration, forward.handle.acceleration)
    assert all(bool(jnp.all(jnp.isfinite(t))) for t in res.tangent)
    _drop_executables()

    lin = plan.linearize(wrt=("positions",), fn=_plan_loss)
    from_lin = float(lin.tangent({"positions": dx}))
    by_jvp = plan.jvp(None, {"positions": dx}, fn=_plan_loss)
    assert from_lin != 0.0
    assert abs(from_lin - float(by_jvp.tangent)) <= 1e-12 * abs(from_lin)
    _drop_executables()

    k_array = {"external_params": {"k": jnp.asarray(K_EXT)}}
    jac = plan.jacfwd(_plan_loss, wrt=("external_params",), inputs=k_array)
    column = float(jac.tangent["external_params"]["k"])
    one = plan.jvp(
        k_array,
        {"external_params": {"k": jnp.asarray(1.0)}},
        fn=_plan_loss,
    )
    assert abs(column - float(one.tangent)) <= 1e-12 * abs(column)
    assert float(jac.value) == float(one.value)

    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, 512)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu, phi = rng.uniform(-1.0, 1.0, 512), rng.uniform(0.0, 2.0 * np.pi, 512)
    st = np.sqrt(1.0 - mu * mu)
    plummer = jnp.asarray(
        np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    )
    with pytest.raises(RolloutVerdictError):
        plan.jvp({"positions": plummer}, {"positions": dx}, fn=_plan_loss)
    _drop_executables()

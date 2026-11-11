"""The field at points that are not sources (``target_points``, #302), on CPU.

``FusedForceProvider.target_field`` / ``step_with_targets``
(:mod:`jaccpot.runtime.target_points`) on the fused lane's CPU route
(``RuntimePolicyConfig(cpu_fused_lane=True)``), the galaxy preset at leaf 16, p4:

* against a float64 direct sum with the engine's softening kernel, under both
  criteria: the geometric opening angle and ``dehnen_error`` (eq 16a with the
  points' own force scale) -- points inside the sources, a tight clump, points far
  outside the source box, and points on top of sources (the softened kernel's
  finite value, the source included);
* the sources' accelerations of ``step_with_targets`` are ``step``'s, bit for bit:
  the points enter neither the tree nor the walk nor the force scale;
* moving points recompile nothing (one trace for three point sets);
* a plan too small for the points says so (``ok=False``, the needs), and the plan
  grown from those needs fits;
* ``dehnen_error`` without the points' force scale is refused.

N = 1024 sources, M = 160 points (CPU, about a minute a criterion).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._lane_platform import CPU_ROUTE_PALLAS_ENV

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="the fused lane's CPU route"
)

N, LEAF, ORDER, THETA, EPS = 1024, 16, 4, 0.6, 1e-2
_ENV_UNSET = (
    *CPU_ROUTE_PALLAS_ENV,
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_WALK_SEPARATION_FLOOR",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)


def _solver(mac: str):
    from jaccpot import FastMultipoleMethod

    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = dataclasses.replace(
        base,
        tree=dataclasses.replace(base.tree, leaf_target=LEAF, cell_min_level=0),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=ORDER, cpu_fused_lane=True
        ),
        mac_type=mac,
    )
    criterion = (
        dict(
            adaptive_eps=1e-4,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="galaxy",
        theta=THETA,
        G=1.0,
        softening=EPS,
        precision="fp32",
        use_pallas=mac == "dehnen_error",
        advanced=advanced,
        **criterion,
    )


def _sources():
    rng = np.random.default_rng(0)
    x = 0.3 * rng.normal(size=(N, 3))
    m = np.abs(rng.normal(size=N)) / N + 0.2 / N
    return x, m


def _points(x):
    rng = np.random.default_rng(1)
    return np.concatenate(
        [
            0.3 * rng.normal(size=(64, 3)),  # inside
            0.4 + 0.02 * rng.normal(size=(32, 3)),  # a tight clump
            rng.uniform(3.0, 5.0, size=(32, 3)),  # far outside the box
            x[:32],  # on top of sources
        ]
    )


_SLICES = {
    "inside": slice(0, 64),
    "clump": slice(64, 96),
    "far": slice(96, 128),
    "coincident": slice(128, 160),
}


def _direct(points, x, m, engine):
    from jaccpot.softening import pair_factors, softening_params_np

    params = softening_params_np(engine.softening_kernel, float(engine.softening))
    dx = points[:, None, :] - x[None, :, :]
    r2 = np.sum(dx * dx, axis=-1)
    g, psi, _ = pair_factors(r2, params, engine.softening_kernel, potential=True, xp=np)
    acc = -np.sum((m[None, :] * g)[..., None] * dx, axis=1)
    return acc, -np.sum(m[None, :] * psi, axis=1)


def _rel_l2(got, want):
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.fixture(scope="module", params=["dehnen", "dehnen_error"])
def lane(request):
    from jaccpot import FusedForceProvider

    mac = request.param
    x, m = _sources()
    xf, mf = jnp.asarray(x, jnp.float32), jnp.asarray(m, jnp.float32)
    prov, topo = FusedForceProvider.prepare(
        _solver(mac), xf, mf, leaf_size=LEAF, max_order=ORDER
    )
    pts = jnp.asarray(_points(x), jnp.float32)
    plan = prov.plan_targets(topo, pts)
    fb = None
    if mac == "dehnen_error":
        from jaccpot.runtime.target_points import bootstrap_target_force_scale

        fb, _ = bootstrap_target_force_scale(prov, topo, pts, plan=plan)
    return dict(
        mac=mac, x=x, m=m, xf=xf, mf=mf, prov=prov, topo=topo, pts=pts, plan=plan, fb=fb
    )


# the criterion's tolerance on these sources: the geometric angle at theta 0.6, p4
# (the source lane itself is at 2.3e-3 on such a set), and eq (16a) at 1e-4
_BOUND = {"dehnen": 2e-3, "dehnen_error": 3e-4}


def test_field_matches_a_float64_direct_sum(lane):
    field = jax.jit(
        lambda t, p, f: lane["prov"].target_field(
            t, p, lane["plan"], force_scale=f, potential=True, return_force_scale=True
        )
    )(lane["topo"], lane["pts"], lane["fb"])
    assert bool(field.ok)
    acc = np.asarray(field.acc, np.float64)
    pot = np.asarray(field.potential, np.float64)
    want_acc, want_pot = _direct(
        np.asarray(lane["pts"], np.float64), lane["x"], lane["m"], lane["prov"].engine
    )
    assert np.isfinite(acc).all() and np.isfinite(pot).all()
    bound = _BOUND[lane["mac"]]
    for name, sl in _SLICES.items():
        assert _rel_l2(acc[sl], want_acc[sl]) < bound, name
        assert np.max(np.abs(pot[sl] - want_pot[sl])) < bound * np.max(
            np.abs(want_pot[sl])
        ), name
    # far outside, the root's multipole alone: well inside the bound
    assert _rel_l2(acc[_SLICES["far"]], want_acc[_SLICES["far"]]) < 1e-4
    fb = np.asarray(field.force_scale)
    assert np.isfinite(fb).all() and (fb > 0).all()


def test_sources_are_untouched_bitwise(lane):
    prov, xf, mf = lane["prov"], lane["xf"], lane["mf"]
    aux = (
        None
        if lane["fb"] is None
        else (prov.accelerations(lane["topo"], force_scale=True)[1], None, None)
    )
    want, want_aux = jax.jit(lambda x, m, a: prov.step(x, m, aux=a))(xf, mf, aux)
    got, got_aux, field = jax.jit(
        lambda x, m, p, a, f: prov.step_with_targets(
            x, m, p, lane["plan"], aux=a, target_force_scale=f
        )
    )(xf, mf, lane["pts"], aux, lane["fb"])
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got_aux[1]), np.asarray(want_aux[1]))
    assert bool(field.ok)


def test_moving_points_recompile_nothing(lane):
    traces = []

    def run(t, p, f):
        traces.append(1)
        return lane["prov"].target_field(t, p, lane["plan"], force_scale=f).acc

    fn = jax.jit(run)
    rng = np.random.default_rng(2)
    for _ in range(3):
        moved = lane["pts"] + jnp.asarray(
            1e-3 * rng.normal(size=lane["pts"].shape), jnp.float32
        )
        assert np.isfinite(np.asarray(fn(lane["topo"], moved, lane["fb"]))).all()
    assert len(traces) == 1


def test_overflow_is_reported_and_the_grown_plan_fits(lane):
    small = dataclasses.replace(lane["plan"], far_capacity=8, near_capacity=8)
    field = lane["prov"].target_field(
        lane["topo"], lane["pts"], small, force_scale=lane["fb"]
    )
    assert not bool(field.ok)
    needs = np.asarray(field.needs)
    assert needs[0] > 8 or needs[1] > 8
    grown = small.grown(needs, 1.0)
    again = lane["prov"].target_field(
        lane["topo"], lane["pts"], grown, force_scale=lane["fb"]
    )
    assert bool(again.ok)
    np.testing.assert_array_equal(np.asarray(again.needs), needs)


def test_dehnen_error_needs_the_points_force_scale(lane):
    if lane["mac"] != "dehnen_error":
        pytest.skip("geometric criterion: no force scale")
    with pytest.raises(ValueError, match="force scale"):
        lane["prov"].target_field(lane["topo"], lane["pts"], lane["plan"])


def test_plan_rejects_points_of_another_count(lane):
    with pytest.raises(ValueError, match="points must be"):
        lane["prov"].target_field(
            lane["topo"], lane["pts"][:-1], lane["plan"], force_scale=lane["fb"]
        )

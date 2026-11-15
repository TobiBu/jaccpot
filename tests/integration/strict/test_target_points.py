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


def test_capacities_are_totals_so_a_dense_clump_costs_only_its_own_pairs(lane):
    """A few points inside the sources and many far outside: the plan's capacities
    are TOTALS, so the far groups do not pay for the clump's near/far lists, and
    the compiled temporaries stay at one walk block plus one pair chunk (the first
    version gave every group the worst group's capacities: 14.9 GiB at 1e6 points
    on an A100, 2026-11-11)."""
    from jaccpot.runtime.target_points import evaluate_target_points

    rng = np.random.default_rng(5)
    pts = np.concatenate(
        [0.02 * rng.normal(size=(32, 3)), rng.uniform(4.0, 6.0, size=(4064, 3))]
    )
    pts = jnp.asarray(pts, jnp.float32)
    prov, topo = lane["prov"], lane["topo"]
    fb = None
    if lane["mac"] == "dehnen_error":
        from jaccpot.runtime.target_points import bootstrap_target_force_scale

        probe = prov.plan_targets(topo, pts)
        fb, _ = bootstrap_target_force_scale(prov, topo, pts, plan=probe)
    plan = prov.plan_targets(topo, pts, force_scale=fb)
    field = prov.target_field(topo, pts, plan, force_scale=fb)
    assert bool(field.ok)
    needs = np.asarray(field.needs)
    # 127 far groups accept a handful of nodes each; one clump group needs many
    assert needs[0] < 127 * 64 and needs[1] < plan.near_capacity + 1
    compiled = (
        jax.jit(
            lambda t, p, f: evaluate_target_points(
                prov, t, p, plan=plan, force_scale=f
            ).acc
        )
        .lower(topo, pts, fb)
        .compile()
    )
    temp = compiled.memory_analysis().temp_size_in_bytes
    assert temp < 256 * 2**20, temp


def test_a_softening_override_gives_another_kernel_from_a_small_softening_tree(lane):
    """Sources built with a SMALL softening (their own walk stays O(n log n)) give
    the field of the engine's EPS at the points with ``softening=EPS``: against the
    float64 direct sum at EPS, at the criterion's tolerance (the hybrid scheme's
    back-reaction: a compact cluster's field on heavy galaxy particles)."""
    if lane["mac"] != "dehnen":
        pytest.skip("one criterion suffices for the override")
    from jaccpot import FastMultipoleMethod, FusedForceProvider

    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = dataclasses.replace(
        base,
        tree=dataclasses.replace(base.tree, leaf_target=LEAF, cell_min_level=0),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=ORDER, cpu_fused_lane=True
        ),
        mac_type="dehnen",
    )
    tiny = FastMultipoleMethod(
        preset="galaxy",
        theta=THETA,
        G=1.0,
        softening=1e-5,
        precision="fp32",
        advanced=advanced,
    )
    prov, topo = FusedForceProvider.prepare(
        tiny, lane["xf"], lane["mf"], leaf_size=LEAF, max_order=ORDER
    )
    plan = prov.plan_targets(topo, lane["pts"], softening=EPS)
    field = prov.target_field(topo, lane["pts"], plan, softening=EPS)
    assert bool(field.ok)
    want, _ = _direct(
        np.asarray(lane["pts"], np.float64), lane["x"], lane["m"], lane["prov"].engine
    )
    assert _rel_l2(np.asarray(field.acc, np.float64), want) < _BOUND["dehnen"]
    # without the override the same tree gives the near-Newtonian field
    bare = prov.target_field(topo, lane["pts"], prov.plan_targets(topo, lane["pts"]))
    coincident = _SLICES["coincident"]
    assert (
        _rel_l2(np.asarray(bare.acc, np.float64)[coincident], want[coincident])
        > 10 * _BOUND["dehnen"]
    )
    with pytest.raises(ValueError, match="softening"):
        prov.target_field(topo, lane["pts"], plan, softening=-1.0)


# --------------------------------------------------------------- session 2 (HH2a)
def _walk_sets(lane, plan, points=None, mask=None):
    from jaccpot.runtime.target_points import _target_walk

    tw = _target_walk(
        lane["prov"],
        lane["topo"],
        lane["pts"] if points is None else points,
        plan=plan,
        force_scale=lane["fb"],
        mac=None,
        softening=None,
        point_mask=mask,
    )
    fg, fn, ng, nn = (np.asarray(a) for a in tw.lists)
    far, near = (int(v) for v in np.asarray(tw.totals))
    return (
        set(zip(fg[:far].tolist(), fn[:far].tolist())),
        set(zip(ng[:near].tolist(), nn[:near].tolist())),
        np.asarray(tw.stats),
    )


@pytest.mark.parametrize("far", ["m2p", "local"])
def test_the_frontier_walk_accepts_the_dense_walks_pair_set(lane, far):
    """The frontier walk (each group descends from the root, opening only the
    rejected nodes) uses exactly the dense sweep's (group, node) far pairs and
    (group, leaf) near pairs, under both criteria and both far-field forms, with
    far fewer MAC tests."""
    plan = dataclasses.replace(lane["plan"], far=far, walk="frontier")
    f_far, f_near, f_stats = _walk_sets(lane, plan)
    d_far, d_near, d_stats = _walk_sets(lane, dataclasses.replace(plan, walk="dense"))
    assert f_far == d_far and f_near == d_near
    assert len(f_far) > 0 and len(f_near) > 0
    assert 0 < f_stats[1] < d_stats[1]
    assert 0 < f_stats[0] <= plan.queue_capacity
    assert int(f_stats[2]) == lane["pts"].shape[0]


def test_a_full_frontier_queue_is_reported_and_the_grown_plan_fits(lane):
    small = dataclasses.replace(lane["plan"], queue_capacity=8, walk_chunk=8)
    field = lane["prov"].target_field(
        lane["topo"], lane["pts"], small, force_scale=lane["fb"]
    )
    assert not bool(field.ok)
    assert int(np.asarray(field.stats)[0]) > 8
    grown = small
    for _ in range(8):
        grown = grown.grown(field.needs, 1.0, field.stats)
        field = lane["prov"].target_field(
            lane["topo"], lane["pts"], grown, force_scale=lane["fb"]
        )
        if bool(field.ok):
            break
    assert bool(field.ok)
    want = lane["prov"].target_field(
        lane["topo"], lane["pts"], lane["plan"], force_scale=lane["fb"]
    )
    np.testing.assert_array_equal(np.asarray(field.needs), np.asarray(want.needs))
    np.testing.assert_allclose(
        np.asarray(field.acc), np.asarray(want.acc), rtol=1e-5, atol=1e-6
    )


def test_group_local_expansions_match_a_float64_direct_sum(lane):
    """``far='local'``: one M2L per (group, node) about the group's centre and an
    L2P per point, with the group radius in the criterion. On compact groups
    (inside, the clump, on top of sources) it is at the criterion's tolerance
    against the direct sum, like the per-point M2P; on the SPARSE far-box groups
    (32 points spread over a 2-unit box) eq (16a)'s per-interaction bound adds up
    over the many nodes such a wide group accepts (dehnen_error: 6.8e-4 there
    against M2P's 5e-6), so that slice gets 3x the bound -- why M2P stays the
    default."""
    from jaccpot.runtime.target_points import plan_target_points

    plan = plan_target_points(
        lane["prov"], lane["topo"], lane["pts"], force_scale=lane["fb"], far="local"
    )
    field = lane["prov"].target_field(
        lane["topo"], lane["pts"], plan, force_scale=lane["fb"], potential=True
    )
    assert bool(field.ok)
    want_acc, _ = _direct(
        np.asarray(lane["pts"], np.float64), lane["x"], lane["m"], lane["prov"].engine
    )
    acc = np.asarray(field.acc, np.float64)
    assert np.isfinite(acc).all()
    for name, sl in _SLICES.items():
        bound = _BOUND[lane["mac"]] * (3.0 if name == "far" else 1.0)
        assert _rel_l2(acc[sl], want_acc[sl]) < bound, name


def test_a_point_mask_evaluates_only_the_live_points(lane):
    """Dead points get zeros and no walk; the live points' field is the field of
    the live subset alone (same groups up to the dead tail)."""
    from jaccpot.runtime.target_points import evaluate_target_points

    pts = lane["pts"]
    live = np.zeros(pts.shape[0], bool)
    live[::3] = True
    field = evaluate_target_points(
        lane["prov"],
        lane["topo"],
        pts,
        plan=lane["plan"],
        force_scale=lane["fb"],
        point_mask=jnp.asarray(live),
    )
    acc = np.asarray(field.acc, np.float64)
    assert np.all(acc[~live] == 0.0)
    assert int(np.asarray(field.stats)[2]) == int(live.sum())
    want, _ = _direct(
        np.asarray(pts, np.float64)[live], lane["x"], lane["m"], lane["prov"].engine
    )
    assert _rel_l2(acc[live], want) < _BOUND[lane["mac"]]


@pytest.fixture(scope="module")
def back_lane():
    """A compact geometric source set (the cluster) and points spread far around
    it (the galaxy), with the points' softening larger than the sources' own."""
    from jaccpot import FastMultipoleMethod, FusedForceProvider

    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = dataclasses.replace(
        base,
        tree=dataclasses.replace(base.tree, leaf_target=LEAF, cell_min_level=0),
        runtime=dataclasses.replace(
            base.runtime, fixed_order=ORDER, cpu_fused_lane=True
        ),
        mac_type="dehnen",
    )
    solver = FastMultipoleMethod(
        preset="galaxy",
        theta=0.5,
        G=1.0,
        softening=1e-4,
        precision="fp32",
        advanced=advanced,
    )
    rng = np.random.default_rng(11)
    xs = 0.02 * rng.normal(size=(512, 3))
    ms = np.full(512, 1e-3 / 512)
    prov, topo = FusedForceProvider.prepare(
        solver,
        jnp.asarray(xs, jnp.float32),
        jnp.asarray(ms, jnp.float32),
        leaf_size=LEAF,
        max_order=ORDER,
    )
    pts = np.concatenate(
        [0.03 * rng.normal(size=(96, 3)), rng.uniform(-3.0, 3.0, size=(1952, 3))]
    )
    return dict(prov=prov, topo=topo, xs=xs, ms=ms, pts=jnp.asarray(pts, jnp.float32))


def test_the_back_reaction_matches_a_direct_sum_and_full_target_points(back_lane):
    """Outer points take the root multipole, inner ones the walk: the result is
    at the criterion's tolerance against the float64 direct sum (at the points'
    softening EPS), and equals the full ``target_points`` evaluation to the
    multipole's own tolerance; the provider's ``target_field`` dispatches a
    ``BackReactionPlan``; moving points recompile nothing."""
    from jaccpot.runtime.target_back_reaction import (
        evaluate_back_reaction,
        plan_back_reaction,
    )

    prov, topo, pts = back_lane["prov"], back_lane["topo"], back_lane["pts"]
    plan = plan_back_reaction(prov, topo, pts, softening=EPS)
    assert plan.mp_order == ORDER and plan.theta_mp == pytest.approx(0.5)
    assert 0 < plan.inner_capacity < pts.shape[0]
    traces = []

    def run(t, p):
        traces.append(1)
        return prov.target_field(t, p, plan, softening=EPS, potential=True)

    fn = jax.jit(run)
    field = fn(topo, pts)
    assert bool(field.ok)
    stats = np.asarray(field.stats)
    assert 96 <= int(stats[2]) <= plan.inner_capacity
    for k in range(2):
        moved = pts + jnp.asarray(
            1e-4 * np.random.default_rng(k).normal(size=pts.shape), jnp.float32
        )
        assert np.isfinite(np.asarray(fn(topo, moved).acc)).all()
    assert len(traces) == 1

    from jaccpot.softening import pair_factors, softening_params_np

    eng = prov.engine
    params = softening_params_np(eng.softening_kernel, EPS)
    p64 = np.asarray(pts, np.float64)
    dx = p64[:, None, :] - back_lane["xs"][None, :, :]
    g, psi, _ = pair_factors(
        np.sum(dx * dx, -1), params, eng.softening_kernel, potential=True, xp=np
    )
    want = -np.sum((back_lane["ms"][None, :] * g)[..., None] * dx, axis=1)
    want_pot = -np.sum(back_lane["ms"][None, :] * psi, axis=1)
    acc = np.asarray(field.acc, np.float64)
    assert _rel_l2(acc, want) < _BOUND["dehnen"]
    assert _rel_l2(acc[96:], want[96:]) < _BOUND["dehnen"]
    assert np.max(np.abs(np.asarray(field.potential) - want_pot)) < (
        _BOUND["dehnen"] * np.max(np.abs(want_pot))
    )
    full_plan = prov.plan_targets(topo, pts, softening=EPS)
    full = prov.target_field(topo, pts, full_plan, softening=EPS)
    assert _rel_l2(acc, np.asarray(full.acc, np.float64)) < _BOUND["dehnen"]
    direct_call = evaluate_back_reaction(prov, topo, pts, plan=plan, softening=EPS)
    np.testing.assert_array_equal(np.asarray(direct_call.acc), np.asarray(field.acc))


def test_the_back_reactions_inner_overflow_is_reported(back_lane):
    from jaccpot.runtime.target_back_reaction import (
        evaluate_back_reaction,
        plan_back_reaction,
    )

    prov, topo, pts = back_lane["prov"], back_lane["topo"], back_lane["pts"]
    plan = plan_back_reaction(prov, topo, pts, softening=EPS, min_inner=32)
    crowded = jnp.concatenate([0.03 * pts[:1024] / 3.0, pts[1024:]])
    field = evaluate_back_reaction(prov, topo, crowded, plan=plan, softening=EPS)
    assert not bool(field.ok)
    assert int(np.asarray(field.stats)[2]) > plan.inner_capacity
    with pytest.raises(ValueError, match="mp_order"):
        plan_back_reaction(prov, topo, pts, mp_order=ORDER + 1)


def test_planning_from_sources_rebuilds_in_program(lane):
    """``TargetSources`` in place of a topology: each planning round rebuilds the
    sources' topology inside its program (no eager state needs to stay resident:
    at 1e8 it did not leave room on an A100), and sizes the same plan."""
    from jaccpot.runtime.target_points import (
        TargetSources,
        bootstrap_target_force_scale,
        plan_target_points_from_sources,
    )

    prov = lane["prov"]
    fb_src = (
        prov.accelerations(lane["topo"], force_scale=True)[1]
        if prov.scale_carry
        else None
    )
    plan = plan_target_points_from_sources(
        prov, lane["xf"], lane["mf"], lane["pts"], source_force_scale=fb_src
    )
    assert plan == lane["plan"]
    if lane["fb"] is not None:
        fb, _ = bootstrap_target_force_scale(
            prov, TargetSources(lane["xf"], lane["mf"], fb_src), lane["pts"], plan=plan
        )
        np.testing.assert_allclose(
            np.asarray(fb), np.asarray(lane["fb"]), rtol=1e-6, atol=0
        )

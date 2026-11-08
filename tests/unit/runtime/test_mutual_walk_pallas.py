"""jaccpot's use of yggdrax's Pallas mutual walk.

The walk itself -- one Pallas launch per wavefront round, the same pair SETS as
``dual_tree_walk_mutual`` -- moved to yggdrax (``yggdrax.pallas.mutual_walk``,
2026-11) together with its tests (yggdrax ``tests/unit/test_mutual_walk_pallas.py``:
set parity on both layouts and tree kinds, overflow, separation floor, read counts).
What stays here is what jaccpot adds around it:

* eq (16a) from jaccpot's multipole table (``dehnen_walk_table``) gives the flat
  walk's sets under ``DehnenWalkAccept``;
* the ``JACCPOT_WALK_*`` environment switches reach the yggdrax walk as its keyword
  arguments, under the ``fmm_walk`` named scope;
* the walk backend switch and the (target, source) sort of the walk's lists.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax")
from yggdrax.interactions import dual_tree_walk_mutual, mutual_walk_pallas
from yggdrax.tree_moments import compute_tree_mass_moments

from jaccpot.runtime._mac_geometry import com_mac_geometry


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _sets(res):
    nf, nn = int(res.far_count), int(res.near_count)
    fa, fb = np.asarray(res.far_a)[:nf], np.asarray(res.far_b)[:nf]
    na, nb = np.asarray(res.near_a)[:nn], np.asarray(res.near_b)[:nn]
    return set(zip(fa.tolist(), fb.tolist())), set(zip(na.tolist(), nb.tolist()))


@pytest.mark.parametrize(
    "node_layout, dtype", [("record", "float32"), ("soa", "float64")]
)
@pytest.mark.parametrize("eps", [1e-2, 1e-4])
def test_pallas_walk_with_eq16a_equals_the_flat_walk_with_it(node_layout, dtype, eps):
    """``error_order > 0``: the Pallas walk accepts by Dehnen's eq (16a) from the walk
    table, exactly as ``dual_tree_walk_mutual`` does with ``DehnenWalkAccept``."""
    from jaccpot.runtime._walk_criterion import DehnenWalkAccept, dehnen_walk_table
    from jaccpot.upward.real_tree_expansions import prepare_real_upward_sweep

    n, leaf, order = 4000, 16, 4
    P = jnp.asarray(_plummer(n, 7), dtype)
    M = jnp.full((n,), 1.0 / n, dtype)
    # a full Tree (the upward sweep is typed for one; CI runs the type checks)
    from yggdrax.tree import Tree

    tree = Tree.from_particles(P, M, leaf_size=leaf, tree_type="radix")
    topo = tree.topology
    ps, ms = P[tree.particle_indices], M[tree.particle_indices]
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=leaf)
    upward = prepare_real_upward_sweep(
        tree, ps, ms, max_order=order, max_leaf_size=leaf
    )
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    ranges = np.asarray(topo.node_ranges)
    spans = ranges[:, 1] >= ranges[:, 0]
    active = jnp.asarray(spans)
    root = jnp.argmin(topo.parent).astype(idx)
    mass = np.asarray(compute_tree_mass_moments(topo, ps, ms).mass)
    c = np.asarray(geom.center, np.float64)
    # a smooth stand-in for eps * min f_b: the Plummer field strength at the node
    fscale = 1.0 / (np.sum(c * c, axis=1) + 1.0)
    table = dehnen_walk_table(
        multipole_packed=upward.multipoles.packed,
        mass=jnp.asarray(mass, dtype),
        radius=geom.radius,
        threshold=jnp.asarray(eps * fscale, dtype),
        gravitational_constant=1.0,
        order=order,
    )
    kw = dict(
        max_pair_queue=1 << 15, far_cap=1 << 17, near_cap=1 << 17, node_active=active
    )
    ref = dual_tree_walk_mutual(
        left,
        right,
        geom.center,
        geom.radius,
        0.5,
        root,
        pair_accept=DehnenWalkAccept(order),
        pair_accept_data={"table": table, "theta_max": jnp.asarray(1.0, dtype)},
        **kw,
    )
    got = mutual_walk_pallas(
        left,
        right,
        geom.center,
        geom.radius,
        0.5,
        root,
        block=64,
        interpret=True,
        node_layout=node_layout,
        error_table=table,
        error_order=order,
        theta_max=1.0,
        **kw,
    )
    for res in (ref, got):
        assert not (
            bool(res.queue_overflow)
            or bool(res.far_overflow)
            or bool(res.near_overflow)
        )
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
    far_r, near_r = _sets(ref)
    far_g, near_g = _sets(got)
    assert far_g == far_r
    assert near_g == near_r
    assert int(got.rounds) == int(ref.rounds)
    # the criterion is live: it differs from the geometric walk at the same theta
    geo = dual_tree_walk_mutual(
        left, right, geom.center, geom.radius, 0.5, root, mac_type="dehnen", **kw
    )
    assert _sets(geo)[0] != far_r


def test_lex_sorted_orders_by_target_then_source_in_both_branches():
    from jaccpot.runtime._interaction_cache import _lex_sorted

    rng = np.random.default_rng(7)
    prim = jnp.asarray(rng.integers(0, 50, 500), jnp.int32)
    sec = jnp.asarray(rng.integers(0, 70000, 500), jnp.int32)
    expect = np.lexsort((np.asarray(sec), np.asarray(prim)))
    for primary_bound in (50, 50000):  # int32 composite / int64 composite
        p, s = _lex_sorted(
            prim, sec, primary_bound=primary_bound, secondary_bound=70000
        )
        assert p.dtype == prim.dtype and s.dtype == sec.dtype
        assert np.array_equal(np.asarray(p), np.asarray(prim)[expect])
        assert np.array_equal(np.asarray(s), np.asarray(sec)[expect])


def test_walk_backend_flag_parsing(monkeypatch):
    from yggdrax.interactions import pallas_mutual_walk_supported

    from jaccpot.runtime._interaction_cache import strict_walk_backend

    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_WALK", raising=False)
    monkeypatch.delenv("JACCPOT_WALK_PALLAS_INTERPRET", raising=False)
    # unset: the Pallas walk wherever it lowers (an Ampere+ GPU), else yggdrax's
    default = "pallas" if pallas_mutual_walk_supported() else "flat"
    assert strict_walk_backend() == default
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK", "Pallas")
    assert strict_walk_backend() == "pallas"
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_WALK", "cuda")
    with pytest.raises(ValueError):
        strict_walk_backend()


def _tiny_walk_args():
    left = jnp.asarray([1, -1, -1], jnp.int32)
    right = jnp.asarray([2, -1, -1], jnp.int32)
    cent = jnp.asarray(
        [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], jnp.float32
    )
    rad = jnp.asarray([2.0, 0.1, 0.1], jnp.float32)
    return left, right, cent, rad, 0.5, jnp.asarray(0, jnp.int32)


@pytest.mark.parametrize(
    "env, expect",
    [
        ({}, {"interpret": False, "node_layout": "record", "max_programs": 2048}),
        (
            {
                "JACCPOT_WALK_PALLAS_INTERPRET": "1",
                "JACCPOT_WALK_NODE_LAYOUT": "soa",
                "JACCPOT_WALK_MAX_PROGRAMS": "16",
            },
            {"interpret": True, "node_layout": "soa", "max_programs": 16},
        ),
    ],
)
def test_jaccpot_walk_switches_reach_the_yggdrax_walk(monkeypatch, env, expect):
    """yggdrax reads no environment: jaccpot passes its switches as keywords."""
    import yggdrax.interactions as yi

    from jaccpot.runtime._interaction_cache import pallas_mutual_walk

    for name in (
        "JACCPOT_WALK_PALLAS_INTERPRET",
        "JACCPOT_WALK_NODE_LAYOUT",
        "JACCPOT_WALK_MAX_PROGRAMS",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    seen = {}

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return "walked"

    monkeypatch.setattr(yi, "mutual_walk_pallas", spy)
    out = pallas_mutual_walk(
        *_tiny_walk_args(), max_pair_queue=8, far_cap=8, near_cap=8
    )
    assert out == "walked"
    assert {k: seen[k] for k in expect} == expect
    assert seen["max_pair_queue"] == 8


def test_the_walk_runs_under_the_fmm_walk_scope(monkeypatch):
    """``bench/analyse_trace_by_stage.py`` credits the walk by its ``fmm_*`` scope."""
    from jaccpot.runtime._interaction_cache import pallas_mutual_walk

    monkeypatch.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")
    jaxpr = jax.make_jaxpr(
        lambda: pallas_mutual_walk(
            *_tiny_walk_args(), max_pair_queue=8, far_cap=8, near_cap=8
        )
    )()
    stacks = [str(e.source_info.name_stack) for e in jaxpr.jaxpr.eqns]
    assert any("fmm_walk" in st for st in stacks), stacks

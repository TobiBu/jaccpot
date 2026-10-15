"""The drift margin: frozen lists that stay valid over a base step, and a measured guard.

A block-step base step evaluates every sub-step boundary against interaction lists
built at its start while the particles drift. Particles then sit inside cells whose
far expansions they still use: on a 1e6 prefix of the 25M disc the frozen-list force
error was 17x the fresh-list one at ``k_max = 3`` and diverged from ``k_max = 5`` (S0,
``docs/production_2026-10.md``), and neither momentum nor the overflow flag saw it.

``BlockStepFMM(drift_margin=m)`` builds the MAC on radii inflated by
``(2 + theta) m``, so the lists stay inside the MAC while no particle moves further
than ``m``, and keeps the build positions so the drift is measured and flagged on the
state's overflow flag when it is exceeded. See ``jaccpot.mutual.topology`` ("Drift
margin").
"""

from __future__ import annotations

import dataclasses
import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaccpot.nornax_adapter as adapter
from jaccpot.mutual import build_mutual_topology
from jaccpot.mutual.force import OVERFLOW_CAUSES, drift_distance, flag_drift
from jaccpot.nornax_adapter import (
    BlockStepFMM,
    raise_on_overflow,
    suggest_drift_margin,
)

SOFTENING = 1.0e-2
THETA = 0.6
LEAF = 16
DRIFT_BIT = 1 << OVERFLOW_CAUSES.index("drift_margin")


def _two_clumps(n=512, seed=23, separation=8.0, sigma=0.6):
    """Two well-separated clumps, with far pairs inside and between them."""
    rng = np.random.default_rng(seed)
    half = n // 2
    centres = ([-separation / 2.0, 0.0, 0.0], [separation / 2.0, 0.0, 0.0])
    positions = np.concatenate([rng.normal(c, sigma, (half, 3)) for c in centres])
    velocities = rng.normal(0.0, 0.5, (n, 3))
    masses = rng.uniform(0.5, 1.5, n) / n
    return (
        jnp.asarray(positions, jnp.float64),
        jnp.asarray(velocities, jnp.float64),
        jnp.asarray(masses, jnp.float64),
    )


def _blob(n=1024, seed=1):
    """A Gaussian blob: at leaf 8 its far pairs span every scale, down to leaf pairs."""
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(0.0, 1.0, (n, 3)), jnp.float64),
        jnp.asarray(rng.uniform(0.5, 1.5, n) / n, jnp.float64),
    )


def _exploding_clumps(side=8, per=16, sigma=0.02, seed=4):
    """Tight clumps on a unit grid, every particle at unit speed in a random direction.

    Over ``dt`` each clump swells from ``sigma`` to about ``dt``, so a pair of
    neighbouring clumps accepted as far at the build is inside each other's
    convergence sphere by ``dt = 0.25`` -- the S0 failure at test scale -- while
    clumps further apart stay far, so a margin keeps a far field.
    """
    rng = np.random.default_rng(seed)
    grid = np.stack(
        np.meshgrid(np.arange(side), np.arange(side), [0.0], indexing="ij"), -1
    ).reshape(-1, 3)
    positions = grid[:, None, :] + rng.normal(0.0, sigma, (grid.shape[0], per, 3))
    positions = positions.reshape(-1, 3)
    velocities = rng.normal(size=positions.shape)
    velocities /= np.linalg.norm(velocities, axis=1, keepdims=True)
    masses = rng.uniform(0.5, 1.5, positions.shape[0]) / positions.shape[0]
    return (
        jnp.asarray(positions, jnp.float64),
        jnp.asarray(velocities, jnp.float64),
        jnp.asarray(masses, jnp.float64),
    )


def _model(**kwargs):
    defaults = dict(
        softening=SOFTENING,
        softening_kernel="plummer",
        k_max=2,
        theta=THETA,
        max_order=4,
        leaf_size=LEAF,
    )
    defaults.update(kwargs)
    return BlockStepFMM(**defaults)


def _node_geometry(topology, positions, masses):
    """Centre of mass and radius of every node at ``positions`` (original order)."""
    order = np.asarray(topology.forward_permutation)
    x = np.asarray(positions, np.float64)[order]
    m = np.asarray(masses, np.float64)[order]
    ranges = np.asarray(topology.node_particle_ranges)
    centres = np.zeros((topology.num_nodes, 3))
    radii = np.zeros(topology.num_nodes)
    for node, (start, end) in enumerate(ranges):
        if end < start:
            continue
        xs, ms = x[start : end + 1], m[start : end + 1]
        centres[node] = (ms[:, None] * xs).sum(0) / ms.sum()
        radii[node] = np.sqrt(((xs - centres[node]) ** 2).sum(1)).max()
    return centres, radii


def _worst_mac_ratio(topology, positions, masses):
    """``max (R_A + R_B) / |c_B - c_A|`` over the far pairs; the MAC needs < theta."""
    centres, radii = _node_geometry(topology, positions, masses)
    a, b = np.asarray(topology.far_a), np.asarray(topology.far_b)
    dist = np.linalg.norm(centres[b] - centres[a], axis=1)
    return float(((radii[a] + radii[b]) / dist).max())


def _drifts(topology, positions, masses, margin, seed=5):
    """Displacement fields with ``|dx_i| <= margin`` that push the MAC hardest.

    Every particle flying straight out of its leaf's centre of mass swells every
    cell by about the margin; contraction towards the origin brings the clumps
    together; random kicks do some of both. Each is scaled to the full margin,
    less a hair for round-off.
    """
    x = np.asarray(positions)
    unit = lambda d: d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-300)
    centres, _ = _node_geometry(topology, positions, masses)
    order = np.asarray(topology.forward_permutation)
    ranges = np.asarray(topology.node_particle_ranges)
    leaf_centre = np.zeros_like(x)
    for leaf in topology.leaf_nodes:
        start, end = ranges[leaf]
        leaf_centre[order[start : end + 1]] = centres[leaf]
    explode = unit(x - leaf_centre)
    kicks = unit(np.random.default_rng(seed).normal(size=x.shape))
    fields = [explode, -unit(x), kicks, unit(explode - unit(x))]
    return [(1.0 - 1e-9) * margin * f for f in fields]


# --- the MAC -----------------------------------------------------------------------


def test_margin_lists_stay_inside_the_mac_for_every_drift_within_the_margin():
    """The guarantee itself: the frozen far pairs pass the plain MAC after the drift.

    Without the margin the same drifts push accepted pairs past ``theta`` -- which is
    what makes the check non-vacuous.
    """
    positions, masses = _blob()
    margin = 0.05
    built, _ = build_mutual_topology(
        positions, masses, theta=THETA, order=4, leaf_size=8, drift_margin=margin
    )
    plain, _ = build_mutual_topology(
        positions, masses, theta=THETA, order=4, leaf_size=8
    )
    assert built.num_far_pairs > 0, "vacuous: the margin left no far pairs"

    # At the build positions the inflated MAC holds with the full pad.
    centres, radii = _node_geometry(built, positions, masses)
    a, b = np.asarray(built.far_a), np.asarray(built.far_b)
    pad = (2.0 + THETA) * margin
    dist = np.linalg.norm(centres[b] - centres[a], axis=1)
    assert np.all(THETA * dist > radii[a] + radii[b] + 2.0 * pad)

    broken = 0
    for delta in _drifts(built, positions, masses, margin):
        moved = positions + jnp.asarray(delta)
        assert _worst_mac_ratio(built, moved, masses) < THETA
        broken += _worst_mac_ratio(plain, moved, masses) >= THETA
    assert broken > 0, "the plain lists survived every drift: the test proves nothing"


def test_margin_lists_are_a_symmetric_partition_and_conserve_momentum():
    """Inflating the radii only makes acceptance stricter: still one visit per pair.

    Mutual accumulation hides a double count from the momentum check (a doubled
    pair still cancels), so the partition is checked structurally, then momentum.
    """
    positions, masses = _blob()
    n = positions.shape[0]
    topology, _ = build_mutual_topology(
        positions, masses, theta=THETA, order=4, leaf_size=8, drift_margin=0.05
    )
    assert topology.num_far_pairs > 0
    assert np.all(topology.far_a < topology.far_b)
    assert np.all(topology.near_a < topology.near_b)
    ranges = topology.node_particle_ranges
    count = np.zeros((n, n), dtype=np.int32)

    def members(node):
        start, end = ranges[node]
        return np.arange(start, end + 1)

    for pairs in ((topology.far_a, topology.far_b), (topology.near_a, topology.near_b)):
        for a, b in zip(*pairs):
            count[np.ix_(members(a), members(b))] += 1
            count[np.ix_(members(b), members(a))] += 1
    for leaf in topology.leaf_nodes:
        count[np.ix_(members(leaf), members(leaf))] += 1
    off_diagonal = count[~np.eye(n, dtype=bool)]
    assert off_diagonal.min() == 1 and off_diagonal.max() == 1

    fmm = _model(leaf_size=8, drift_margin=0.05)
    state = fmm.prepare(positions, masses)
    assert int(state.num_far_pairs) > 0
    rung = jnp.asarray(np.arange(n) % 3, jnp.int32)
    for acc in (
        fmm.total_accelerations(positions, masses),
        fmm.boundary_kick_at(
            positions, jnp.zeros_like(positions), masses, rung=rung, s=1, dt_max=0.1
        ),
    ):
        terms = masses[:, None] * acc
        residual = jnp.linalg.norm(terms.sum(0)) / jnp.linalg.norm(
            jnp.abs(terms).sum(0)
        )
        assert float(residual) < 1e-14


# --- the force ---------------------------------------------------------------------


def _direct_sum(positions, masses):
    dr = positions[None, :, :] - positions[:, None, :]
    r2 = jnp.sum(dr * dr, axis=-1) + SOFTENING**2
    live = ~jnp.eye(positions.shape[0], dtype=bool)
    inv_r3 = jnp.where(live, jnp.where(live, r2, 1.0) ** -1.5, 0.0)
    return jnp.sum((masses[None, :] * inv_r3)[..., None] * dr, axis=1)


def _errors(acc, ref):
    rel = np.linalg.norm(np.asarray(acc - ref), axis=1) / np.linalg.norm(
        np.asarray(ref), axis=1
    )
    return float(jnp.linalg.norm(acc - ref) / jnp.linalg.norm(ref)), float(rel.max())


def test_frozen_lists_with_a_margin_keep_the_fresh_list_error():
    """The end of a base step: lists built at x0, force at x1 = x0 + v dt.

    Without a margin the frozen lists' error is a multiple of the fresh lists' (the
    S0 failure, at test scale); with ``suggest_drift_margin`` it stays within the
    fresh lists' error -- the drifted configuration still satisfies the plain MAC.
    Measured: frozen rel-L2 / max 6.2e-4 / 4.9e-2, fresh 1.4e-4 / 9.5e-3, frozen
    with the margin 1.2e-5 / 5.0e-4 on 147 far pairs (plain lists: 409).
    """
    positions, velocities, masses = _exploding_clumps()
    dt = 0.25
    moved = positions + velocities * dt
    ref = _direct_sum(moved, masses)

    plain = _model()
    frozen_plain = plain.total_accelerations(
        moved, masses, topology=plain.prepare(positions, masses)
    )
    fresh = plain.total_accelerations(
        moved, masses, topology=_model().prepare(moved, masses)
    )
    margin = suggest_drift_margin(velocities, dt, safety=1.01)
    guarded = _model(drift_margin=margin)
    state = guarded.prepare(positions, masses)
    assert int(state.num_far_pairs) > 0
    frozen_margin = guarded.total_accelerations(moved, masses, topology=state)
    assert not bool(guarded.check_drift(state, moved).topology_overflow)

    l2_plain, max_plain = _errors(frozen_plain, ref)
    l2_fresh, max_fresh = _errors(fresh, ref)
    l2_margin, max_margin = _errors(frozen_margin, ref)
    assert max_plain > 3.0 * max_fresh and l2_plain > 2.0 * l2_fresh, (
        f"no frozen-list error to guard against at this drift: {l2_plain:.2e} / "
        f"{max_plain:.2e} against fresh {l2_fresh:.2e} / {max_fresh:.2e}"
    )
    assert l2_margin <= l2_fresh and max_margin <= max_fresh, (
        f"margin rel-L2 {l2_margin:.2e} / max {max_margin:.2e} against fresh "
        f"{l2_fresh:.2e} / {max_fresh:.2e}"
    )


# --- the guard ---------------------------------------------------------------------


@pytest.mark.parametrize("scan_boundaries", [False, True])
def test_advance_base_step_raises_once_the_drift_leaves_the_margin(scan_boundaries):
    """Every boundary is measured; a base step that outruns the margin is refused."""
    positions, velocities, masses = _two_clumps(n=256)
    rung = jnp.asarray(np.arange(positions.shape[0]) % 2, jnp.int32)
    dt_max = 0.05
    margin = suggest_drift_margin(velocities, dt_max, safety=1.5)
    fmm = _model(k_max=1, drift_margin=margin)
    fmm.prepare(positions, masses)
    x, v, _ = fmm.advance_base_step(
        positions,
        velocities,
        masses,
        rung=rung,
        dt_max=dt_max,
        scan_boundaries=scan_boundaries,
    )
    assert float(drift_distance(fmm.state, x)) <= margin
    # The same lists, a second base step on: further than the margin from the build.
    with pytest.raises(RuntimeError, match="drifted further than drift_margin"):
        fmm.advance_base_step(
            x, v, masses, rung=rung, dt_max=dt_max, scan_boundaries=scan_boundaries
        )


def test_check_drift_flags_the_state_under_jit_and_raise_on_overflow_names_it():
    """Traceable: the flag is the overflow flag, so it leaves a jit like one."""
    positions, velocities, masses = _two_clumps()
    margin = 0.05
    fmm = _model(drift_margin=margin)
    state = fmm.prepare(positions, masses)
    np.testing.assert_array_equal(np.asarray(state.drift_reference), positions)

    check = jax.jit(fmm.check_drift)
    kicks = np.random.default_rng(1).normal(size=positions.shape)
    kicks *= 0.99 * margin / np.linalg.norm(kicks, axis=1, keepdims=True)
    inside = check(state, positions + jnp.asarray(kicks))
    assert not bool(inside.topology_overflow)
    raise_on_overflow(inside, fmm)

    outside = check(state, positions.at[7].add(jnp.asarray([2.0 * margin, 0.0, 0.0])))
    assert bool(outside.topology_overflow)
    assert int(outside.overflow_causes) == DRIFT_BIT
    with pytest.raises(RuntimeError) as excinfo:
        raise_on_overflow(outside, fmm)
    message = str(excinfo.value)
    assert f"drift_margin={margin}" in message
    assert "capacity" not in message  # a drift is not a capacity overflow

    # Sticky, and composes with a capacity cause: both are reported.
    far_bit = 1 << OVERFLOW_CAUSES.index("far")
    both = flag_drift(
        dataclasses.replace(outside, overflow_causes=outside.overflow_causes | far_bit),
        0.0,
        margin,
    )
    with pytest.raises(RuntimeError, match="drifted further.*far exceeded"):
        raise_on_overflow(both, fmm)


def test_a_drift_flag_rides_a_scan_with_the_topology_rebuilt_inside():
    """The device lane: rebuild + check inside one jitted scan, raise after it."""
    positions, velocities, masses = _two_clumps()
    dt = 0.05
    margin = suggest_drift_margin(velocities, dt, safety=1.5)
    fmm = _model(drift_margin=margin, topology_backend="device")
    fmm.prepare(positions, masses)

    @jax.jit
    def rollout(x, step):
        def body(x, _):
            state = fmm.rebuild_state(x, masses)
            x = x + velocities * step
            state = fmm.check_drift(state, x)
            return x, (state.topology_overflow, state.overflow_causes)

        return jax.lax.scan(body, x, xs=None, length=3)

    _, (flags, causes) = rollout(positions, dt)
    assert not bool(jnp.any(flags))
    _, (flags, causes) = rollout(positions, 3.0 * dt)
    assert bool(jnp.all(flags))
    record = SimpleNamespace(
        topology_overflow=bool(jnp.any(flags)),
        overflow_causes=int(np.bitwise_or.reduce(np.asarray(causes))),
    )
    assert record.overflow_causes == DRIFT_BIT
    with pytest.raises(RuntimeError, match="drifted further than drift_margin"):
        raise_on_overflow(record, fmm)


def test_the_device_walk_applies_the_same_margin_as_the_host_traversal():
    """Same tree, same margin: the device far list is the host's, pair for pair."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.mutual.device_topology import build_mutual_state_device
    from jaccpot.mutual.force import resolve_mutual_capacities
    from jaccpot.mutual.topology import build_mutual_topology_from_tree

    positions, masses = _blob()
    margin = 0.05
    prepared = FastMultipoleMethod(
        preset="balanced", basis="real", softening_kernel="plummer"
    ).prepare_state(positions, masses, leaf_size=8, max_order=4, theta=THETA)
    parent = np.asarray(prepared.tree.parent)
    root = int(np.flatnonzero(parent < 0)[0])
    for drift_margin in (None, margin):
        host = build_mutual_topology_from_tree(
            prepared.tree,
            np.asarray(prepared.positions_sorted),
            np.asarray(prepared.masses_sorted),
            theta=THETA,
            order=4,
            drift_margin=drift_margin,
        )
        device = build_mutual_state_device(
            jnp.asarray(prepared.positions_sorted),
            jnp.asarray(prepared.masses_sorted),
            parent=jnp.asarray(parent),
            left_child=jnp.asarray(host.left_child),
            right_child=jnp.asarray(host.right_child),
            node_ranges=jnp.asarray(host.node_particle_ranges),
            inverse_permutation=jnp.asarray(host.inverse_permutation),
            root=jnp.asarray(root),
            theta=THETA,
            order=4,
            leaf_size=int(host.max_leaf_size),
            caps=resolve_mutual_capacities(host)._replace(queue=1 << 16),
            softening=SOFTENING,
            softening_kernel="plummer",
            max_pair_queue=1 << 16,
            drift_margin=drift_margin,
        )
        assert not bool(device.topology_overflow)
        count = int(device.num_far_pairs)
        got = set(
            zip(
                np.asarray(device.far_a)[:count].tolist(),
                np.asarray(device.far_b)[:count].tolist(),
            )
        )
        assert got == set(zip(host.far_a.tolist(), host.far_b.tolist()))
        assert int(device.num_near_pairs) == host.num_near_pairs
        if drift_margin is None:
            plain_far = count
    assert 0 < count < plain_far


# --- the knobs ---------------------------------------------------------------------


def test_suggest_drift_margin_is_the_furthest_reach():
    v = jnp.asarray([[3.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    a = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 40.0, 0.0]])
    assert suggest_drift_margin(v, 0.1) == pytest.approx(0.3)
    # |v| dt + |a| dt^2 / 2: 0.3 for the first, 0.1 + 0.2 = 0.3 for the second
    assert suggest_drift_margin(v, 0.1, a, safety=2.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        suggest_drift_margin(jnp.zeros((4, 3)), 0.1)
    with pytest.raises(ValueError):
        suggest_drift_margin(v, 0.0)


def test_without_a_margin_there_is_no_reference_and_nothing_to_check():
    positions, _, masses = _two_clumps()
    fmm = _model()
    state = fmm.prepare(positions, masses)
    assert state.drift_reference is None
    with pytest.raises(ValueError, match="drift margin"):
        fmm.check_drift(state, positions)
    with pytest.raises(ValueError, match="without a drift margin"):
        drift_distance(state, positions)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite positive"):
            _model(drift_margin=bad)


def test_k_max_3_without_a_margin_warns_once(monkeypatch):
    """The limitation is stated once per process, not once per model."""
    monkeypatch.setattr(adapter, "_FROZEN_LIST_WARNING_ISSUED", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _model(k_max=2)
        _model(k_max=3, drift_margin=0.1)
        assert not caught
        _model(k_max=3)
        _model(k_max=4)
    assert len(caught) == 1
    assert "drift_margin" in str(caught[0].message)

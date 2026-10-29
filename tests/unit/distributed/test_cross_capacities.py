"""The cross capacities sized in the library, and the needs they are sized from.

A production run cannot depend on capacity formulas that live in the benches. The
library sizes every capacity of the default (symmetric) exchange from what a force
NEEDED (:data:`jaccpot.distributed.cross.CROSS_NEEDS_FIELDS`), so these pin two things
without a device mesh:

* the needs survive the overflow they are read after: the summary is recounted from the
  tree (the cut saturates its count at its capacity), and the exchange rows are counted
  on the presence map, which no row capacity truncates;
* the sizing: a priori from the shard (``for_shard``), from a measurement
  (``from_needs``), and the segment retry's re-plan (``replanned``), which never narrows
  a capacity and doubles one whose need is only a lower bound;
* the template route's sizing (``for_measurement``, ``size_cross_capacities``): a
  measurement on buffers far below the demand, sized from its exact needs.

The traced half -- the needs against the probes' record counts, and an undersized
capacity re-planned to the same force -- is ``tests/distributed/
test_cross_capacities_cpu_e2e.py``.

    JAX_PLATFORMS=cpu pytest tests/unit/distributed/test_cross_capacities.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.summary")
from yggdrax._tree_impl import build_static_cells_tree  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed.cross import (  # noqa: E402
    _CROSS_CAP_GRANULE,
    CROSS_NEEDS_FIELDS,
    CrossCapacities,
    _rows_by_peer,
    _summary_needs,
    _summary_rows,
    cross_needs,
    plan_cross_capacities,
    size_cross_capacities,
)
from jaccpot.runtime._mac_geometry import com_mac_geometry  # noqa: E402

LEAF = 16
LEAF_CAPACITY = 2048

# (two_sided, max_leaves_per_cell): the one-sided control, two-sided over 4-leaf cells,
# and the default -- two-sided over LEAVES, the symmetric exchange's summary
MODES = {
    "one_sided": (False, 4),
    "two_sided_cells": (True, 4),
    "two_sided_leaves": (True, 1),
}


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 30.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


@pytest.fixture(scope="module")
def shard():
    """Half of a Plummer sphere as one device trees it: cell tree and COM geometry."""
    pts = _plummer(6000, 11)
    pts = pts[np.argsort(pts[:, 0])[:3000]]
    P = jnp.asarray(pts, jnp.float32)
    M = jnp.ones((P.shape[0],), jnp.float32)
    topo, ps, ms, _ = build_static_cells_tree(
        P,
        M,
        infer_bounds(P),
        leaf_size=LEAF,
        leaf_capacity=LEAF_CAPACITY,
        return_reordered=True,
    )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    return topo, com_mac_geometry(topo, ps, com, leaf_cap=LEAF)


def _caps(max_leaves, *, cells, nodes, cell_level=None):
    return CrossCapacities(
        max_cells=cells,
        max_summary_nodes=nodes,
        max_leaves_per_cell=max_leaves,
        summary_cell_level=cell_level,
    )


@pytest.mark.parametrize("cell_level", [None, 3])
@pytest.mark.parametrize("mode", list(MODES))
def test_the_summary_needs_are_the_published_counts(shard, mode, cell_level):
    topo, geom = shard
    two_sided, max_leaves = MODES[mode]
    cap = _caps(max_leaves, cells=LEAF_CAPACITY, nodes=2 * LEAF_CAPACITY)
    cap.summary_cell_level = cell_level
    pub = _summary_rows(topo, geom, cap, two_sided=two_sided)
    assert not bool(pub.overflow)
    cells, nodes = _summary_needs(topo, cap, two_sided=two_sided)
    assert int(cells) == int(pub.cut.num_cells)
    assert int(nodes) == int(pub.num_entries)
    # not vacuous: a summary with structure, and (over leaves) one cell per live leaf
    assert int(cells) > 50
    if max_leaves == 1:
        live = np.asarray(topo.node_ranges)[int(topo.left_child.shape[0]) :]
        assert int(cells) == int(np.sum(live[:, 1] >= live[:, 0]))


@pytest.mark.parametrize("mode", ["two_sided_leaves", "two_sided_cells"])
def test_the_summary_needs_survive_the_overflow_they_report(shard, mode):
    """Undersized, the cut and the summary tree saturate their counts; the needs are
    what the generous capacities published."""
    topo, geom = shard
    two_sided, max_leaves = MODES[mode]
    big = _caps(max_leaves, cells=LEAF_CAPACITY, nodes=2 * LEAF_CAPACITY)
    pub = _summary_rows(topo, geom, big, two_sided=two_sided)
    want_cells, want_nodes = int(pub.cut.num_cells), int(pub.num_entries)

    small = _caps(max_leaves, cells=want_cells // 3, nodes=want_nodes // 3)
    pub_small = _summary_rows(topo, geom, small, two_sided=two_sided)
    assert bool(pub_small.overflow)
    assert int(pub_small.cut.num_cells) == want_cells // 3  # saturated
    cells, nodes = _summary_needs(topo, small, two_sided=two_sided)
    assert (int(cells), int(nodes)) == (want_cells, want_nodes)

    # the tree alone overflowing (the cut fits) is seen too
    tight = _caps(max_leaves, cells=LEAF_CAPACITY, nodes=want_nodes - 1)
    assert bool(_summary_rows(topo, geom, tight, two_sided=two_sided).overflow)
    assert int(_summary_needs(topo, tight, two_sided=two_sided)[1]) == want_nodes


def test_the_row_needs_survive_the_row_capacity():
    """``_rows_by_peer`` counts unique (peer, node) on its presence map: the count and
    the map are the same whatever the capacity, and the flag says when it overflowed."""
    rng = np.random.default_rng(3)
    ndev, S, P = 3, 64, 400
    peer = jnp.asarray(rng.integers(0, ndev, P), jnp.int32)
    node = jnp.asarray(rng.integers(0, S, P), jnp.int32)
    live = jnp.asarray(rng.uniform(size=P) < 0.8)
    big = _rows_by_peer(peer, node, live, ndev=ndev, S=S, capacity=ndev * S)
    keys = {
        (int(p), int(n))
        for p, n, ok in zip(np.asarray(peer), np.asarray(node), np.asarray(live))
        if ok
    }
    assert int(big.needed) == len(keys) and not bool(big.overflow)
    small = _rows_by_peer(peer, node, live, ndev=ndev, S=S, capacity=len(keys) // 2)
    assert bool(small.overflow)
    assert int(small.needed) == len(keys)
    np.testing.assert_array_equal(np.asarray(small.present), np.asarray(big.present))


def _needs(**values):
    vec = np.zeros(len(CROSS_NEEDS_FIELDS), np.int64)
    for k, v in values.items():
        vec[CROSS_NEEDS_FIELDS.index(k)] = v
    return vec


CAP_FIELDS = CROSS_NEEDS_FIELDS[:10]


def test_the_needs_name_the_capacities_they_size():
    caps = CrossCapacities()
    for name in CAP_FIELDS:
        assert isinstance(getattr(caps, name), int), name
    assert CROSS_NEEDS_FIELDS[10:] == (
        "overflow",
        "walk_lower_bound",
        "pairs_lower_bound",
        "mismatch",
    )


def test_the_defaults_are_unchanged():
    """Existing tests and benches build ``CrossCapacities()``; sizing is opt-in."""
    assert vars(CrossCapacities()) == dict(
        max_cells=1024,
        max_leaves_per_cell=None,
        export_far_cap=1 << 16,
        export_near_cap=1 << 16,
        send_node_cap=1 << 13,
        send_csr_cap=1 << 16,
        recv_node_cap=1 << 13,
        recv_csr_cap=1 << 16,
        walk_queue=1 << 16,
        leaf_width=64,
        recv_far_cap=1 << 17,
        recv_near_cap=1 << 17,
        recv_near_csr_cap=1 << 16,
        export_walk_queue=1 << 16,
        send_particle_cap=(1 << 13) * 64,
        recv_particle_cap=(1 << 13) * 64,
        summary_cell_level=None,
        max_summary_nodes=2048,
    )


def test_for_shard_bounds_the_summary_by_the_tree_and_scales_with_leaves_and_peers():
    two = CrossCapacities.for_shard(
        leaf_capacity=4096, ndev=2, leaf_width=64, live_leaves=3000
    )
    assert two.max_cells == 4096 and two.max_summary_nodes == 2 * 4096 - 1
    assert two.leaf_width == 64 and two.summary_cell_level is None
    # 16 far pairs per live leaf per peer, x2.5
    assert two.export_far_cap == -(-16 * 3000 * 25 // 10 // 1024) * 1024
    four = CrossCapacities.for_shard(
        leaf_capacity=4096, ndev=4, leaf_width=64, live_leaves=3000
    )
    for name in CAP_FIELDS[2:9]:
        assert getattr(four, name) % _CROSS_CAP_GRANULE == 0, name
        # three peers instead of one, up to the rounding of each
        lo = 3 * getattr(two, name) - 3 * _CROSS_CAP_GRANULE
        assert lo <= getattr(four, name) <= 3 * getattr(two, name), name
    # no live count: the leaf capacity bounds the live leaves
    upper = CrossCapacities.for_shard(leaf_capacity=4096, ndev=2, leaf_width=64)
    assert upper.export_far_cap > two.export_far_cap


def test_from_needs_sizes_every_capacity_from_its_need():
    like = CrossCapacities(send_csr_cap=12345, summary_cell_level=8, leaf_width=64)
    need = dict(zip(CAP_FIELDS, (33, 65, 5000, 900, 700, 120, 130, 1500, 1600, 51)))
    caps = CrossCapacities.from_needs(_needs(**need), like=like, headroom=1.5)
    for name in CAP_FIELDS[:9]:
        want = -(-int(np.ceil(1.5 * need[name])) // 1024) * 1024
        assert getattr(caps, name) == want, name
    # the leaf width is a bound, not a count: the widest shipped leaf or the lane's
    assert caps.leaf_width == 64
    # what the symmetric exchange does not read comes from `like`
    assert caps.send_csr_cap == 12345 and caps.summary_cell_level == 8
    assert vars(like)["max_cells"] == 1024  # `like` is not modified

    clamped = CrossCapacities.from_needs(
        _needs(**need), like=like, headroom=1.5, leaf_capacity=40
    )
    assert clamped.max_cells == 40 and clamped.max_summary_nodes == 79


def test_from_needs_reads_the_local_walks_headroom(monkeypatch):
    monkeypatch.setenv("JACCPOT_FLAT_WALK_CAP_HEADROOM", "3.0")
    caps = CrossCapacities.from_needs(_needs(export_far_cap=1000))
    assert caps.export_far_cap == 3072


def test_replanned_never_narrows_and_covers_the_need():
    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [100] * 9 + [40]))), headroom=1.0
    )
    assert all(getattr(caps, name) == 1024 for name in CAP_FIELDS[:9])
    # an exact need past its capacity: headroom x need
    grown = caps.replanned(
        _needs(recv_node_cap=3000, send_particle_cap=10, overflow=1), headroom=1.5
    )
    assert grown.recv_node_cap == 5120
    assert grown.send_particle_cap == 1024  # never narrower
    assert grown.leaf_width == 64  # the lane's width; the widest leaf was 40
    # a leaf wider than the tiles: the width grows to it, exactly
    assert caps.replanned(_needs(leaf_width=70, overflow=1)).leaf_width == 70
    # nothing needed more: the same capacities
    same = caps.replanned(_needs(export_far_cap=500), headroom=1.5)
    assert vars(same) == vars(caps)


def test_a_lower_bound_need_that_overflowed_at_least_doubles():
    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [10_000] * 9 + [64]))), headroom=1.0
    )
    far = caps.export_far_cap
    # yggdrax's traced walk stopped at its first overflow: the count is a lower bound
    lower = _needs(export_far_cap=far + 1, walk_lower_bound=1, overflow=1)
    assert caps.replanned(lower, headroom=1.5).export_far_cap == 2 * far
    # the Pallas walk's count is exact past a full list: headroom x need
    exact = _needs(export_far_cap=far + 1, overflow=1)
    assert caps.replanned(exact, headroom=1.5).export_far_cap < 2 * far
    # pairs lost to the walk make the exchange's counts lower bounds; a buffer that
    # did NOT overflow is not doubled
    rows = _needs(send_node_cap=caps.send_node_cap + 1, pairs_lower_bound=1, overflow=1)
    grown = caps.replanned(rows, headroom=1.0)
    assert grown.send_node_cap == 2 * caps.send_node_cap
    assert grown.recv_node_cap == caps.recv_node_cap


def test_an_overflowed_walk_queue_doubles_whatever_its_peak_reads():
    """The Pallas walk stops at a queue overflow and reports the widest wavefront it
    admitted, which can sit BELOW the queue it overflowed: the queue must still
    double, or the re-plan grows nothing that stopped the walk and the next
    measurement overflows the same way (the disc + bulge at 2e5 on two cards)."""
    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [10_000] * 9 + [64]))), headroom=1.0
    )
    queue = caps.export_walk_queue
    stopped = _needs(export_walk_queue=queue // 2, walk_lower_bound=1, overflow=1)
    assert caps.replanned(stopped, headroom=1.5).export_walk_queue == 2 * queue
    # without the lower-bound bit the peak is exact and nothing needed more
    exact = _needs(export_walk_queue=queue // 2, overflow=1)
    assert caps.replanned(exact, headroom=1.5).export_walk_queue == queue


def test_plan_measures_until_clean_then_sizes_from_the_clean_measurement():
    clean = _needs(
        **dict(zip(CAP_FIELDS, (30, 59, 4000, 800, 300, 90, 90, 900, 900, 50)))
    )
    seen = []

    def measure(caps):
        seen.append(caps)
        if caps.export_far_cap < 4000:  # the first guess is too narrow
            return _needs(export_far_cap=4000, overflow=1)
        return clean

    initial = CrossCapacities.from_needs(_needs(), headroom=1.0)
    caps, needs = plan_cross_capacities(measure, initial, headroom=1.5)
    assert len(seen) == 2 and seen[1].export_far_cap >= 4000
    np.testing.assert_array_equal(needs, clean)
    assert vars(caps) == vars(
        CrossCapacities.from_needs(clean, like=seen[1], headroom=1.5)
    )


def test_plan_refuses_a_flag_no_capacity_explains():
    initial = CrossCapacities()
    with pytest.raises(RuntimeError, match="mismatch"):
        plan_cross_capacities(
            lambda caps: _needs(mismatch=1, overflow=1), initial, headroom=1.5
        )
    grow = iter(range(1, 100))
    with pytest.raises(RuntimeError, match="still overflowing"):
        plan_cross_capacities(
            lambda caps: _needs(
                export_far_cap=caps.export_far_cap + next(grow), overflow=1
            ),
            initial,
            max_rounds=2,
        )


ROWS = ("send_node_cap", "recv_node_cap", "send_particle_cap", "recv_particle_cap")


@pytest.mark.parametrize("exact_walk_needs", [False, True])
def test_the_measurement_holds_no_buffer_it_only_counts(exact_walk_needs):
    """``for_measurement``: the rows and particles (counted on the presence maps)
    one granule, and the walk's lists too when the walk counts past a full list
    (the Pallas walk); the summary at the tree's bound, the queue at least the
    floor; everything else ``for_shard``'s."""
    kw = dict(leaf_capacity=4096, ndev=2, leaf_width=64, live_leaves=3000)
    guess = CrossCapacities.for_shard(**kw)
    caps = CrossCapacities.for_measurement(
        **kw, exact_walk_needs=exact_walk_needs, queue_floor=500_000
    )
    for name in ROWS:
        assert getattr(caps, name) == _CROSS_CAP_GRANULE, name
    lists = ("export_far_cap", "export_near_cap")
    for name in lists:
        want = _CROSS_CAP_GRANULE if exact_walk_needs else getattr(guess, name)
        assert getattr(caps, name) == want, name
    assert caps.export_walk_queue == -(-500_000 // 1024) * 1024
    assert caps.export_walk_queue > guess.export_walk_queue
    small_floor = CrossCapacities.for_measurement(**kw, queue_floor=10)
    assert small_floor.export_walk_queue == guess.export_walk_queue
    changed = (
        set(ROWS) | {"export_walk_queue"} | (set(lists) if exact_walk_needs else set())
    )
    for name, value in vars(guess).items():
        if name not in changed:
            assert getattr(caps, name) == value, name


def _clean_needs():
    return _needs(
        **dict(zip(CAP_FIELDS, (30, 59, 4000, 800, 300, 90, 90, 900, 900, 50)))
    )


def test_an_overflowed_measurement_with_exact_needs_sizes_in_one():
    """The rows and particles overflowed, nothing is a lower bound: ONE measurement
    sizes every capacity from its needs -- the clean measurement's sizing -- where
    ``plan_cross_capacities`` would measure again. The measured capacities come
    back with it (their program is compiled)."""
    exact = _clean_needs()
    exact[CROSS_NEEDS_FIELDS.index("overflow")] = 1
    seen = []

    def measure(caps):
        seen.append(caps)
        return exact

    initial = CrossCapacities.for_measurement(
        leaf_capacity=4096, ndev=2, leaf_width=64, live_leaves=3000
    )
    sized, measured, needs = size_cross_capacities(measure, initial, headroom=1.5)
    assert len(seen) == 1 and measured is initial
    np.testing.assert_array_equal(needs, exact)
    assert vars(sized) == vars(
        CrossCapacities.from_needs(_clean_needs(), like=initial, headroom=1.5)
    )
    assert sized.send_particle_cap == 2048 and sized.export_far_cap == 6144


def test_lower_bound_needs_grow_the_walk_and_measure_again():
    """A walk that lost pairs (its far list full) leaves the row counts lower
    bounds: the walk's lists grow (to headroom x the exact walk need) and the force
    is measured again; the rows and particles keep their granule (their next count
    is exact), and the exact second measurement sizes them."""
    clean = _clean_needs()
    seen = []

    def measure(caps):
        seen.append(caps)
        if caps.export_far_cap < 4000:  # the list lost pairs: rows lower bounds
            return _needs(
                export_far_cap=4000, send_node_cap=40, pairs_lower_bound=1, overflow=1
            )
        out = clean.copy()
        out[CROSS_NEEDS_FIELDS.index("overflow")] = 1  # rows still overflow
        return out

    initial = CrossCapacities.for_measurement(
        leaf_capacity=4096,
        ndev=2,
        leaf_width=64,
        live_leaves=3000,
        exact_walk_needs=True,
    )
    sized, measured, _ = size_cross_capacities(measure, initial, headroom=1.5)
    assert len(seen) == 2 and measured is seen[1]
    assert seen[1].export_far_cap == 6144  # 1.5 x the exact walk need
    for name in ROWS:
        assert getattr(seen[1], name) == _CROSS_CAP_GRANULE, name
    assert vars(sized) == vars(
        CrossCapacities.from_needs(clean, like=seen[1], headroom=1.5)
    )
    # room for the next measurements (the drift probes): the measured walk grows by
    # its own factor, the sized capacities keep theirs
    seen.clear()
    roomy, _, _ = size_cross_capacities(
        measure, initial, headroom=1.5, walk_headroom=2.0
    )
    assert len(seen) == 2 and seen[1].export_far_cap == 8192  # 2 x 4000
    assert vars(roomy) == vars(
        CrossCapacities.from_needs(clean, like=seen[1], headroom=1.5)
    )
    # and a floor under the near list (the drift's intruder pairs), for the next
    # measurement only
    seen.clear()
    floored, _, _ = size_cross_capacities(
        measure, initial, headroom=1.5, near_floor=50_000
    )
    assert len(seen) == 2 and seen[1].export_near_cap == 49 * 1024  # >= 50,000
    assert floored.export_near_cap == 2048  # 1.5 x the near need of 800


def test_size_refuses_a_mismatch_and_bounds_that_never_settle():
    initial = CrossCapacities.for_measurement(leaf_capacity=4096, ndev=2, leaf_width=64)
    with pytest.raises(RuntimeError, match="mismatch"):
        size_cross_capacities(lambda caps: _needs(mismatch=1, overflow=1), initial)
    # walks that stopped (a queue overflow) can leave the two ends disagreeing:
    # that is a lower bound to re-plan, not a mismatch to refuse
    rounds = []

    def stopped_then_exact(caps):
        rounds.append(caps)
        if len(rounds) == 1:
            return _needs(
                export_walk_queue=caps.export_walk_queue,
                walk_lower_bound=1,
                pairs_lower_bound=1,
                mismatch=1,
                overflow=1,
            )
        return _clean_needs()

    size_cross_capacities(stopped_then_exact, initial, headroom=1.5)
    assert len(rounds) == 2
    assert rounds[1].export_walk_queue == 2 * initial.export_walk_queue
    grow = iter(range(1, 100))
    with pytest.raises(RuntimeError, match="still lower bounds"):
        size_cross_capacities(
            lambda caps: _needs(
                export_far_cap=caps.export_far_cap + next(grow),
                walk_lower_bound=1,
                overflow=1,
            ),
            initial,
            max_rounds=2,
        )


def test_cross_needs_refuses_a_hook_that_measured_nothing():
    def hook(tree_artifacts):
        return None

    hook.needs_sink = {}
    with pytest.raises(ValueError, match="symmetric exchange"):
        cross_needs(hook)
    with pytest.raises(ValueError, match="symmetric exchange"):
        cross_needs(None)


def test_no_need_grows_a_capacity_past_its_structural_bound():
    """What a NaN state's degenerate tree read on two A100s at 8e6 -- the root of
    each shard one summary cell, near the other's: 4,002,489 particles in one near
    row -- sized a leaf width of a whole shard (an OOM of 282.72 GiB). A cell leaf
    holds at most the template's leaf size or, at the finest Morton level, a few
    times it (``_FINEST_CELL_WIDTH_FACTOR``), and a near row ships at most a leaf
    width of particles, so neither re-plan nor sizing goes past those bounds."""
    from jaccpot.distributed.cross import _FINEST_CELL_WIDTH_FACTOR

    bound = _FINEST_CELL_WIDTH_FACTOR * 64
    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [1000] * 9 + [64]))), headroom=1.0, leaf_size=64
    )
    garbage = _needs(
        leaf_width=4_002_489,
        send_particle_cap=4_002_489,
        recv_particle_cap=4_002_489,
        send_node_cap=60,
        recv_node_cap=60,
        overflow=1,
        pairs_lower_bound=1,
    )
    grown = caps.replanned(garbage, headroom=1.5, leaf_size=64)
    assert grown.leaf_width == bound, vars(grown)
    assert grown.send_particle_cap <= grown.send_node_cap * bound, vars(grown)
    assert grown.recv_particle_cap <= grown.recv_node_cap * bound, vars(grown)
    sized = CrossCapacities.from_needs(garbage, headroom=1.5, leaf_size=64)
    assert sized.leaf_width == bound
    assert sized.send_particle_cap <= sized.send_node_cap * bound, vars(sized)
    # the bounds never narrow what a run already has
    wide = caps._with(leaf_width=2 * bound, send_particle_cap=10**7)
    kept = wide.replanned(garbage, leaf_size=64)
    assert kept.leaf_width == 2 * bound and kept.send_particle_cap == 10**7
    # a leaf width the leaf size allows still grows
    assert (
        caps._with(leaf_width=32)
        .replanned(_needs(leaf_width=50, overflow=1), leaf_size=64)
        .leaf_width
        == 50
    )


def test_a_finest_level_cell_wider_than_the_leaf_size_grows_the_leaf_width():
    """Plummer 8e7 on four A100s (2e7 per card): the box is 35,148 wide (the
    sphere's outliers), so a finest-level Morton cell is 0.017 across and holds up
    to 121 particles in the core -- a leaf whatever it holds. One near row shipped
    144 particles against a leaf width of 64; the re-plan, clamped to the leaf
    size, grew nothing and the run stopped ("re-plan grew nothing"). The width
    grows to the need, by both doors, and the particle caps with it."""
    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [1000] * 9 + [64]))), headroom=1.0, leaf_size=64
    )
    wide_leaf = _needs(leaf_width=144, overflow=1)
    assert caps.replanned(wide_leaf, headroom=1.5, leaf_size=64).leaf_width == 144
    sized = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [1000] * 9 + [144]))), headroom=1.0, leaf_size=64
    )
    assert sized.leaf_width == 144
    assert sized.send_particle_cap >= 1000, vars(sized)


def test_a_run_whose_tree_overflowed_replans_only_the_cross_summary():
    """The segment runner's cross re-plan after a run whose TREE overflowed: that
    run's walks read a truncated tree (the nodes the level loops dropped carry no
    multipole or geometry), so its walk and exchange needs are no demand, and the
    monotone re-plan would keep whatever they grew. Only the summary grows (it
    follows the live leaves); the re-run on a tree that fits measures the rest."""
    from types import SimpleNamespace

    from jaccpot.distributed.template_force import FusedMeshForce
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS

    caps = CrossCapacities.from_needs(
        _needs(**dict(zip(CAP_FIELDS, [1000] * 9 + [64]))),
        headroom=1.0,
        leaf_size=64,
        leaf_capacity=4096,
    )
    force = SimpleNamespace(
        cross_caps=caps,
        cross_replan=None,
        engine=SimpleNamespace(_tree_leaf_capacity=4096),
        leaf_size=64,
    )
    broken = _needs(
        max_cells=2000,
        max_summary_nodes=3999,
        export_near_cap=550_592,
        send_node_cap=179_966,
        send_particle_cap=3_507_599,
        overflow=1,
        walk_lower_bound=1,
        pairs_lower_bound=1,
        mismatch=1,
    )
    needs = np.concatenate([np.zeros(len(NEEDS_FIELDS), np.int64), broken])
    grown = FusedMeshForce._replanned_cross(
        force, needs, leaf_needed=0, tree_overflowed=True
    )
    changed = {k for k in vars(caps) if getattr(grown, k) != getattr(caps, k)}
    assert changed == {"max_cells", "max_summary_nodes"}, changed
    # on a tree that fit, the same needs are a demand
    trusted = FusedMeshForce._replanned_cross(force, needs, leaf_needed=0)
    assert trusted.send_node_cap > caps.send_node_cap

"""A padded shard's summary cells ship their LIVE particles, not the dead rows.

A capacity-padded cells tree (a mesh shard: ``num_valid`` live rows, the dead ones
sorted to the end) gives its padding leaves the empty range ``(n, n - 1)``, and an
internal node the (min start, max end) of its children. So the parent of the last
live leaf and the padding subtree spans ``[that leaf's start, n - 1]``: the leaf's
particles AND every dead row. Under one-leaf summary cells without a
``summary_cell_level`` bound that parent holds one live leaf, so it IS a cut cell, and
the symmetric exchange laid its near particles out over the whole range -- the
near half's ``full > W`` check then fired whatever the width (found on two A100s with
the rollout gate's capacities at 2e5 per card; the dead rows carry zero mass, so the
force was not wrong, but the run could not continue). The level bound hid it: the
dead rows' sentinel Morton code makes that parent's cell edge huge, so the cut takes
the leaf itself.

The pipeline runs as the hook runs it -- ``_summary_rows``, the all_gather,
``_symmetric_exchange`` with its ragged exchange -- on two "devices" mapped with
``jax.vmap(axis_name=...)``, so it needs no device mesh.

    JAX_PLATFORMS=cpu pytest tests/unit/distributed/test_cross_padded_shard.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.summary")
from yggdrax._tree_impl import build_static_cells_tree  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed.cross import (  # noqa: E402
    CROSS_NEEDS_FIELDS,
    CrossCapacities,
    _leaf_under,
    _summary_rows,
    _symmetric_exchange,
)
from jaccpot.runtime._mac_geometry import com_mac_geometry  # noqa: E402

LEAF = 16
LEAF_CAPACITY = 1024
N_LIVE = 3000
ROWS = 3450  # per shard: 450 dead rows, sorted behind the live ones
AXIS = "gpus"


def _shard(points, bounds):
    """One padded shard as the lane trees it: dead rows on a live particle."""
    dead = np.repeat(points[:1], ROWS - len(points), 0)
    P = jnp.asarray(np.concatenate([points, dead]), jnp.float32)
    M = jnp.asarray(
        np.concatenate([np.ones(len(points)), np.zeros(ROWS - len(points))]),
        jnp.float32,
    )
    topo, ps, ms, _ = build_static_cells_tree(
        P,
        M,
        bounds,
        leaf_size=LEAF,
        leaf_capacity=LEAF_CAPACITY,
        return_reordered=True,
        num_valid=len(points),
    )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=LEAF)
    return SimpleNamespace(topo=topo, ps=ps, ms=ms, geom=geom)


@pytest.fixture(scope="module")
def shards():
    """Two halves of a Gaussian blob, split along x, one box, both padded."""
    pts = np.random.default_rng(0).normal(size=(2 * N_LIVE, 3)).astype(np.float32)
    pts = pts[np.argsort(pts[:, 0])]
    bounds = infer_bounds(jnp.asarray(pts))
    return [_shard(pts[:N_LIVE], bounds), _shard(pts[N_LIVE:], bounds)]


def _caps(level=None):
    return CrossCapacities(
        max_cells=LEAF_CAPACITY,
        max_summary_nodes=2 * LEAF_CAPACITY,
        max_leaves_per_cell=1,
        summary_cell_level=level,
        export_far_cap=1 << 16,
        export_near_cap=1 << 16,
        export_walk_queue=1 << 14,
        send_node_cap=2048,
        recv_node_cap=2048,
        leaf_width=LEAF,
    )


def _padding_cells(shard, cap):
    """The cut cells whose node range reaches past the shard's live rows."""
    pub = _summary_rows(shard.topo, shard.geom, cap, two_sided=True)
    rows = np.asarray(pub.rows)
    nodes = np.asarray(pub.block_nodes)
    nr = np.asarray(shard.topo.node_ranges)
    cell = (rows[:, 4] < 0) & (rows[:, 6] > 0.5)
    reach = cell & (nodes >= 0) & (nr[np.maximum(nodes, 0), 1] >= N_LIVE)
    return pub, np.flatnonzero(reach)


def test_the_padding_cell_exists_without_the_level_bound(shards):
    """The precondition, on base and after the fix: without ``summary_cell_level``
    the last live leaf's parent is a cut cell whose range covers the dead rows; with
    the bound (the probes' 8) it is not."""
    _, reach = _padding_cells(shards[0], _caps())
    assert len(reach) == 1
    assert len(_padding_cells(shards[0], _caps(level=8))[1]) == 0


def test_a_summary_cell_publishes_its_live_particle_count(shards):
    """The count column the receiver lays the near particles out by."""
    for shard in shards:
        topo = shard.topo
        pub = _summary_rows(topo, shard.geom, _caps(), two_sided=True)
        rows = np.asarray(pub.rows)
        nodes = np.asarray(pub.block_nodes)
        cell = (rows[:, 4] < 0) & (rows[:, 6] > 0.5)
        nr = jnp.asarray(topo.node_ranges)
        leaf = np.asarray(
            _leaf_under(jnp.asarray(nodes[cell]), nr, int(topo.left_child.shape[0]))
        )
        live = np.asarray(nr)[leaf, 1] - np.asarray(nr)[leaf, 0] + 1
        np.testing.assert_array_equal(np.round(rows[cell, 7]).astype(int), live)


def _exchange(shards, cap, walk_fn=None):
    """The hook's publish, all_gather and symmetric exchange on two mapped devices
    (``walk_fn``: the cross walk, ``None`` for yggdrax's traced one)."""

    def stack(get):
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *map(get, shards)
        )

    fields = ("parent", "left_child", "right_child", "node_ranges", "morton_codes")
    topo = stack(lambda s: {k: getattr(s.topo, k) for k in fields})
    geom = stack(lambda s: (s.geom.center, s.geom.radius))
    sorted_pm = stack(lambda s: (s.ps, s.ms))

    def one(topo, geom, sorted_pm):
        g = SimpleNamespace(center=geom[0], radius=geom[1])
        tree = SimpleNamespace(
            **topo, positions_sorted=sorted_pm[0], masses_sorted=sorted_pm[1]
        )
        pub = _summary_rows(tree, g, cap, two_sided=True)
        gathered = jax.lax.all_gather(pub.rows, AXIS, tiled=False)
        n_nodes = int(tree.parent.shape[0])
        mp = SimpleNamespace(packed=jnp.zeros((n_nodes, 4)), centers=g.center)
        sym = _symmetric_exchange(
            gathered,
            jax.lax.axis_index(AXIS),
            pub.block_nodes,
            tree,
            mp,
            theta=0.8,
            cap=cap,
            ndev=2,
            mac_type="dehnen",
            walk_fn=walk_fn,
            axis_name=AXIS,
            with_near=True,
        )
        needs = jnp.stack(
            [jnp.asarray(sym["needs"].get(k, 0), jnp.int32) for k in CROSS_NEEDS_FIELDS]
        )
        return sym["overflow"], sym["far_overflow"], needs, jnp.sum(sym["masses"] != 0)

    return jax.vmap(one, axis_name=AXIS)(topo, geom, sorted_pm)


def test_the_symmetric_exchange_ships_live_particles_only(shards):
    """The sender's layout: the padding cell is near-exported, and the near half
    stays within the leaf width and flag-clean."""
    cap = _caps()
    near_ovf, far_ovf, needs, received = jax.jit(lambda: _exchange(shards, cap))()
    need = {k: np.asarray(needs)[:, i] for i, k in enumerate(CROSS_NEEDS_FIELDS)}
    assert not np.any(np.asarray(far_ovf))
    assert np.all(np.asarray(received) > 0), "vacuous: no near particles exchanged"
    assert not np.any(np.asarray(near_ovf)), "the near half's flag fired"
    # the widest shipped leaf is a live leaf (at most LEAF particles), not the
    # padding cell's range of live and dead rows
    assert np.all(need["leaf_width"] <= LEAF), need["leaf_width"]


@pytest.mark.parametrize("walk", ["traced", "pallas"])
def test_a_stopped_walk_reads_lower_bounds_not_internal_nodes(
    shards, walk, monkeypatch
):
    """A cross walk that stops on a queue overflow keeps only near pairs between
    summary LEAVES (both walks emit a near pair only when neither node can be
    split), so every near-side need it reports is a lower bound of the full walk's
    -- never a whole-shard row -- and the bits say so. (On two A100s at 8e6 the
    whole-shard rows came from steps run AFTER a failed one, on a NaN state: the
    segment fold, ``segment._build_segment``.) The Pallas walk runs interpreted."""
    from jaccpot.distributed import cross

    if walk == "pallas":
        monkeypatch.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")
        monkeypatch.setenv("JACCPOT_CROSS_WALK", "pallas")
    walk_fn = cross.cross_walk_fn("dehnen")
    assert (walk_fn is None) == (walk == "traced")
    stopped = _caps()
    stopped.export_walk_queue = 256
    _, _, needs_full, _ = jax.jit(lambda: _exchange(shards, _caps(), walk_fn))()
    _, _, needs_stop, _ = jax.jit(lambda: _exchange(shards, stopped, walk_fn))()
    f = {k: np.asarray(needs_full)[:, i] for i, k in enumerate(CROSS_NEEDS_FIELDS)}
    s = {k: np.asarray(needs_stop)[:, i] for i, k in enumerate(CROSS_NEEDS_FIELDS)}
    assert np.all(f["walk_lower_bound"] == 0) and np.all(s["walk_lower_bound"] == 1)
    assert np.all(s["pairs_lower_bound"] == 1)
    assert np.all(s["export_walk_queue"] > 256)  # it says what stopped it
    for name in (
        "export_near_cap",
        "send_node_cap",
        "recv_node_cap",
        "send_particle_cap",
        "recv_particle_cap",
        "leaf_width",
    ):
        assert np.all(s[name] <= f[name]), (name, s[name], f[name])
    assert np.all(s["leaf_width"] <= LEAF)
    assert np.all(f["send_particle_cap"] > 0), "vacuous: no near particles"

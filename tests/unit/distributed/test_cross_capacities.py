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
  a capacity and doubles one whose need is only a lower bound.

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
    CROSS_NEEDS_FIELDS,
    CrossCapacities,
    _rows_by_peer,
    _summary_needs,
    _summary_rows,
    cross_needs,
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


def test_cross_needs_refuses_a_hook_that_measured_nothing():
    def hook(tree_artifacts):
        return None

    hook.needs_sink = {}
    with pytest.raises(ValueError, match="symmetric exchange"):
        cross_needs(hook)
    with pytest.raises(ValueError, match="symmetric exchange"):
        cross_needs(None)

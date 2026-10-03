"""The cross hook's walks through the seeded Pallas walk: the SAME pair sets as yggdrax's.

The export walk (this device's tree against another device's summary cells) and the
receiver walk (local cell roots against imported rows) are one-sided: the other side
is a block of childless nodes indexed above the local tree. The Pallas kernel needs no
change for that -- only a seeded start -- and these tests pin that the sets it emits
are exactly the traced walk's. Interpret mode, so they run on CPU.

    JAX_PLATFORMS=cpu pytest tests/unit/distributed/test_cross_pallas_walk.py -q
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.export")
from yggdrax._tree_impl import build_static_cells_tree  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.export import export_walk  # noqa: E402
from yggdrax.distributed.import_cells import receiver_interaction_lists  # noqa: E402
from yggdrax.distributed.summary import occupancy_cut  # noqa: E402
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed.cross import cross_walk_fn  # noqa: E402
from jaccpot.runtime._mac_geometry import com_mac_geometry  # noqa: E402

LEAF = 16
MAX_CELLS = 256


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 30.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _domain(points, bounds):
    """One device's cell tree, COM geometry, full child arrays and its summary cells."""
    P = jnp.asarray(points, jnp.float32)
    M = jnp.ones((P.shape[0],), jnp.float32)
    topo, ps, ms, _ = build_static_cells_tree(
        P, M, bounds, leaf_size=LEAF, leaf_capacity=2048, return_reordered=True
    )
    com = compute_tree_mass_moments(topo, ps, ms).center_of_mass
    geom = com_mac_geometry(topo, ps, com, leaf_cap=LEAF)
    ni = int(topo.left_child.shape[0])
    tot = int(topo.parent.shape[0])
    idx = topo.parent.dtype
    left = jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)])
    right = jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)])
    ranges = np.asarray(topo.node_ranges)
    active = jnp.asarray(ranges[:, 1] >= ranges[:, 0])
    summary = occupancy_cut(
        jnp.asarray(topo.parent),
        jnp.asarray(topo.node_ranges),
        ni,
        max_leaves=4,
        capacity=MAX_CELLS,
    )
    live = jnp.arange(MAX_CELLS) < summary.num_cells
    safe = jnp.where(live, summary.cells, 0)
    return dict(
        topo=topo,
        geom=geom,
        left=left,
        right=right,
        active=active,
        root=jnp.argmin(topo.parent).astype(idx),
        cells=summary.cells,
        cell_cen=jnp.where(live[:, None], jnp.asarray(geom.center)[safe], 0.0),
        cell_rad=jnp.where(live, jnp.asarray(geom.radius)[safe], 0.0),
        cell_live=live,
    )


@pytest.fixture(scope="module")
def two_domains():
    pts = _plummer(6000, 11)
    order = np.argsort(pts[:, 0])
    half = len(pts) // 2
    bounds = infer_bounds(jnp.asarray(pts, jnp.float32))
    return _domain(pts[order[:half]], bounds), _domain(pts[order[half:]], bounds)


def _pairs(a, b, n):
    n = int(n)
    return set(zip(np.asarray(a)[:n].tolist(), np.asarray(b)[:n].tolist()))


@pytest.fixture()
def pallas_walk(monkeypatch):
    monkeypatch.setenv("JACCPOT_CROSS_WALK", "pallas")
    monkeypatch.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")
    fn = cross_walk_fn("dehnen")
    assert fn is not None
    return fn


@pytest.mark.parametrize("theta", [0.8, 0.5])
def test_the_export_walk_emits_the_same_sets(two_domains, pallas_walk, theta):
    me, other = two_domains
    cen = jnp.stack([me["cell_cen"], other["cell_cen"]])
    rad = jnp.stack([me["cell_rad"], other["cell_rad"]])
    act = jnp.stack([me["cell_live"], other["cell_live"]])
    kw = dict(max_pair_queue=1 << 15, far_cap=1 << 16, near_cap=1 << 16)
    args = (
        me["left"],
        me["right"],
        me["geom"].center,
        me["geom"].radius,
        me["root"],
        cen,
        rad,
        act,
        float(theta),
        jnp.asarray(0),
    )
    ref = export_walk(*args, node_active=me["active"], **kw)
    got = export_walk(*args, node_active=me["active"], walk_fn=pallas_walk, **kw)
    for res in (ref, got):
        assert not (
            bool(res.far_overflow)
            or bool(res.near_overflow)
            or bool(res.queue_overflow)
        )
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
    far_r = _pairs(ref.far_cell, ref.far_node, ref.far_count)
    far_g = _pairs(got.far_cell, got.far_node, got.far_count)
    near_r = _pairs(ref.near_cell, ref.near_node, ref.near_count)
    near_g = _pairs(got.near_cell, got.near_node, got.near_count)
    assert len(far_g) == int(got.far_count), "duplicate far emission"
    assert far_g == far_r
    assert near_g == near_r
    # cells of MY OWN device never appear: the walk was one-sided onto the other
    assert all(c >= MAX_CELLS for c, _ in far_g | near_g)


@pytest.mark.parametrize("max_leaves", [4, 1])
@pytest.mark.parametrize("theta", [0.8, 0.5])
def test_the_two_sided_export_walk_emits_the_same_sets(
    two_domains, pallas_walk, theta, max_leaves
):
    """Both trees refine: the summary tree's internal nodes are just more nodes with
    children to the kernel, so its sets must equal the traced walk's here too."""
    from jaccpot.distributed.cross import (
        CrossCapacities,
        _export_from_rows,
        _summary_rows,
    )

    me, other = two_domains
    cap = CrossCapacities(
        max_cells=MAX_CELLS if max_leaves > 1 else 2048,  # 2048 = the leaf capacity
        max_leaves_per_cell=max_leaves,
        export_far_cap=1 << 16,
        export_near_cap=1 << 16,
        export_walk_queue=1 << 15,
    )
    pubs = [
        _summary_rows(d["topo"], d["geom"], cap, two_sided=True) for d in (me, other)
    ]
    assert not any(bool(p.overflow) for p in pubs)
    rows = jnp.stack([p.rows for p in pubs])
    args = (rows, me["left"], me["right"], me["geom"], me["root"], float(theta))
    kw = dict(two_sided=True, mac_type="dehnen")
    ref = _export_from_rows(*args, jnp.asarray(0), cap, walk_fn=None, **kw)
    got = _export_from_rows(*args, jnp.asarray(0), cap, walk_fn=pallas_walk, **kw)
    for res in (ref, got):
        assert not (
            bool(res.far_overflow)
            or bool(res.near_overflow)
            or bool(res.queue_overflow)
        )
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
    far_g = _pairs(got.far_cell, got.far_node, got.far_count)
    assert len(far_g) == int(got.far_count), "duplicate far emission"
    assert far_g == _pairs(ref.far_cell, ref.far_node, ref.far_count)
    assert _pairs(got.near_cell, got.near_node, got.near_count) == _pairs(
        ref.near_cell, ref.near_node, ref.near_count
    )
    S = int(cap.max_summary_nodes)
    assert all(c >= S for c, _ in far_g), "exported to its own block"


def test_the_receiver_walk_emits_the_same_sets(two_domains, pallas_walk):
    """Seeds (my cell root, imported row) over a combined [local ; imported] space."""
    me, other = two_domains
    n_local = int(me["left"].shape[0])
    # import the OTHER domain's summary cells as childless rows behind my nodes
    n_imp = MAX_CELLS
    left = jnp.concatenate([me["left"], jnp.full((n_imp,), -1, me["left"].dtype)])
    right = jnp.concatenate([me["right"], jnp.full((n_imp,), -1, me["right"].dtype)])
    cen = jnp.concatenate([jnp.asarray(me["geom"].center), other["cell_cen"]])
    rad = jnp.concatenate([jnp.asarray(me["geom"].radius), other["cell_rad"]])
    active = jnp.concatenate([me["active"], other["cell_live"]])
    # every (my cell, imported row) combination that is live, as a CSR
    my_cells = np.flatnonzero(np.asarray(me["cell_live"]))
    imp_rows = np.flatnonzero(np.asarray(other["cell_live"]))
    cc, rr = np.meshgrid(my_cells, imp_rows, indexing="ij")
    cc, rr = cc.ravel(), rr.ravel()
    K = 1 << int(np.ceil(np.log2(len(cc) + 1)))
    csr_cell = np.full(K, -1, np.int64)
    csr_row = np.full(K, -1, np.int64)
    csr_cell[: len(cc)] = cc
    csr_row[: len(rr)] = rr
    kw = dict(max_pair_queue=2 * K, far_cap=1 << 17, near_cap=1 << 17)
    args = (
        left,
        right,
        cen,
        rad,
        n_local,
        me["cells"],
        jnp.asarray(csr_cell),
        jnp.asarray(csr_row),
        jnp.asarray(len(cc)),
        0.8,
    )
    ref = receiver_interaction_lists(*args, node_active=active, **kw)
    got = receiver_interaction_lists(
        *args, node_active=active, walk_fn=pallas_walk, **kw
    )
    for res in (ref, got):
        assert not (
            bool(res.far_overflow)
            or bool(res.near_overflow)
            or bool(res.queue_overflow)
        )
    assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
    assert _pairs(got.far_target, got.far_source, got.far_count) == _pairs(
        ref.far_target, ref.far_source, ref.far_count
    )
    assert _pairs(got.near_target, got.near_source, got.near_count) == _pairs(
        ref.near_target, ref.near_source, ref.near_count
    )


def test_a_walk_cut_off_at_max_rounds_is_flagged(two_domains):
    """CONTROL for the new flag: stopping with pairs still queued is an overflow."""
    from jaccpot.pallas.mutual_walk_pallas import mutual_walk_pallas

    me, _ = two_domains
    kw = dict(max_pair_queue=1 << 15, far_cap=1 << 16, near_cap=1 << 16)
    args = (me["left"], me["right"], me["geom"].center, me["geom"].radius, 0.8)
    full = mutual_walk_pallas(*args, me["root"], interpret=True, **kw)
    assert not bool(full.queue_overflow) and int(full.rounds) > 2
    cut = mutual_walk_pallas(
        *args, me["root"], interpret=True, max_rounds=2, rounds_per_check=1, **kw
    )
    assert bool(cut.queue_overflow), "an unfinished walk must not look complete"


def test_the_traced_walk_is_kept_for_other_macs_and_on_request(monkeypatch):
    monkeypatch.setenv("JACCPOT_CROSS_WALK", "pallas")
    assert cross_walk_fn("engblom") is None
    monkeypatch.setenv("JACCPOT_CROSS_WALK", "flat")
    assert cross_walk_fn("dehnen") is None


def test_the_far_receiver_walk_is_a_pass_through(two_domains):
    """What the hook now skips: walking the receiver's cell roots against the nodes a
    sender ACCEPTED against those cells re-runs the sender's MAC on the same numbers,
    so every seed comes back as a far pair and nothing is refined. B exports to A; A
    walks the result; that must equal reading the CSR off directly."""
    from types import SimpleNamespace

    from jaccpot.distributed.cross import _direct_far_lists

    a, b = two_domains
    cen = jnp.stack([a["cell_cen"], b["cell_cen"]])
    rad = jnp.stack([a["cell_rad"], b["cell_rad"]])
    act = jnp.stack([a["cell_live"], b["cell_live"]])
    kw = dict(max_pair_queue=1 << 15, far_cap=1 << 16, near_cap=1 << 16)
    ex = export_walk(
        b["left"],
        b["right"],
        b["geom"].center,
        b["geom"].radius,
        b["root"],
        cen,
        rad,
        act,
        0.8,
        jnp.asarray(1),
        node_active=b["active"],
        **kw,
    )
    nf = int(ex.far_count)
    assert nf > 0, "vacuous: B exported no far pairs to A"
    # B ships its accepted nodes as childless imported rows behind A's nodes
    csr_cell = np.asarray(ex.far_cell)[:nf]  # A's cell index (device 0 block)
    b_nodes = np.asarray(ex.far_node)[:nf]
    rows, csr_row = np.unique(b_nodes, return_inverse=True)
    n_local = int(a["left"].shape[0])
    n_imp = len(rows)
    idx = a["left"].dtype
    left = jnp.concatenate([a["left"], jnp.full((n_imp,), -1, idx)])
    right = jnp.concatenate([a["right"], jnp.full((n_imp,), -1, idx)])
    cen_c = jnp.concatenate(
        [jnp.asarray(a["geom"].center), jnp.asarray(b["geom"].center)[rows]]
    )
    rad_c = jnp.concatenate(
        [jnp.asarray(a["geom"].radius), jnp.asarray(b["geom"].radius)[rows]]
    )
    K = 1 << int(np.ceil(np.log2(nf + 1)))
    cc = np.full(K, -1, np.int64)
    cr = np.full(K, -1, np.int64)
    cc[:nf], cr[:nf] = csr_cell, csr_row
    walked = receiver_interaction_lists(
        left,
        right,
        cen_c,
        rad_c,
        n_local,
        a["cells"],
        jnp.asarray(cc),
        jnp.asarray(cr),
        jnp.asarray(nf),
        0.8,
        max_pair_queue=2 * K,
        far_cap=2 * K,
        near_cap=2 * K,
    )
    assert int(walked.near_count) == 0, "the receiver refined a sender-accepted pair"
    assert int(walked.far_count) == nf
    got = SimpleNamespace(
        csr_cell=jnp.asarray(cc), csr_row=jnp.asarray(cr), num_csr=jnp.asarray(nf)
    )
    direct = _direct_far_lists(a["cells"], got)
    assert int(direct.far_count) == nf
    assert _pairs(direct.far_target, direct.far_source, direct.far_count) == _pairs(
        walked.far_target, walked.far_source, walked.far_count
    )


def test_the_symmetric_walk_emits_the_same_sets(two_domains, pallas_walk):
    """The walk between two gathered trees (both blocks refine) on the Pallas kernel."""
    from jaccpot.distributed.cross import (
        CrossCapacities,
        _summary_rows,
        _symmetric_walk,
    )

    me, other = two_domains
    cap = CrossCapacities(
        max_cells=2048,
        max_leaves_per_cell=1,
        export_far_cap=1 << 16,
        export_near_cap=1 << 16,
        export_walk_queue=1 << 15,
    )
    pubs = [
        _summary_rows(d["topo"], d["geom"], cap, two_sided=True) for d in (me, other)
    ]
    gathered = jnp.stack([p.rows for p in pubs])
    for dev in (0, 1):
        ref = _symmetric_walk(
            gathered, jnp.asarray(dev), 0.8, cap, mac_type="dehnen", walk_fn=None
        )
        got = _symmetric_walk(
            gathered, jnp.asarray(dev), 0.8, cap, mac_type="dehnen", walk_fn=pallas_walk
        )
        for res in (ref, got):
            assert not (
                bool(res.far_overflow)
                or bool(res.near_overflow)
                or bool(res.queue_overflow)
            )
        assert int(ref.far_count) > 0 and int(ref.near_count) > 0, "vacuous"
        assert _pairs(got.far_a, got.far_b, got.far_count) == _pairs(
            ref.far_a, ref.far_b, ref.far_count
        )
        assert _pairs(got.near_a, got.near_b, got.near_count) == _pairs(
            ref.near_a, ref.near_b, ref.near_count
        )

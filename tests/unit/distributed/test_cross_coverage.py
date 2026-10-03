"""Every cross (receiver particle, sender particle) pair is covered EXACTLY once.

The cross field is assembled from three lists: the far pairs the SENDER accepted
against the receiver's summary (shipped as multipoles), and the near pairs it could not
accept, which the receiver refines into far pairs of its own (M2L against the leaf
multipoles that travel with the particles) and leaf-leaf direct sums. A pair dropped
or double-counted anywhere in that chain applies +f and -f all the same, so **global
momentum is structurally blind to it** (memory ``cross-m2l-theta-lift-design``: the
ownership bug that passed a momentum check at 1e-17). This test counts instead: every
emitted pair is expanded to the particle block it stands for, and the count matrix
over (receiver particle, sender particle) must be exactly one everywhere.

The pipeline is the hook's, piece by piece, minus the collectives: the receiver's
published rows (``cross._summary_rows``, one-sided cells or the two-sided summary
tree), the sender's export walk over the gathered rows (``cross._export_from_rows``),
the grouped send buffers, the receiver's slice of them (what the ragged exchange would
deliver), the far lists read straight off the CSR, and the receiver's near walk. Traced
walks: the Pallas walk's pair SETS are pinned equal to these in
``test_cross_pallas_walk.py``.

    JAX_PLATFORMS=cpu pytest tests/unit/distributed/test_cross_coverage.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.export")
from yggdrax._tree_impl import build_static_cells_tree  # noqa: E402
from yggdrax.bounds import infer_bounds  # noqa: E402
from yggdrax.distributed.export import build_send_buffers  # noqa: E402
from yggdrax.distributed.import_cells import receiver_interaction_lists  # noqa: E402
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed.cross import (  # noqa: E402
    CrossCapacities,
    _direct_far_lists,
    _direct_near_lists,
    _export_from_rows,
    _summary_rows,
)
from jaccpot.runtime._mac_geometry import com_mac_geometry  # noqa: E402

LEAF = 16
MAX_CELLS = 256
LEAF_CAPACITY = 2048  # the fixture trees' leaf slots: bounds the live leaves

# (two_sided, max_leaves_per_cell): the one-sided control, the two-sided walk over
# 4-leaf cells, and the default -- two-sided over the receiver's LEAVES
MODES = {
    "one_sided": (False, 4),
    "two_sided_cells": (True, 4),
    "two_sided_leaves": (True, 1),
}
CAP = 1 << 16


def _plummer(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 30.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    s = np.sqrt(1.0 - mu * mu)
    return np.stack([r * s * np.cos(phi), r * s * np.sin(phi), r * mu], axis=1)


def _domain(points, bounds):
    """One device: its cell tree, COM walk geometry and full child arrays."""
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
    ranges = np.asarray(topo.node_ranges)
    return SimpleNamespace(
        n=int(points.shape[0]),
        topo=topo,
        geom=geom,
        ranges=ranges,
        num_internal=ni,
        left=jnp.concatenate([topo.left_child, jnp.full((tot - ni,), -1, idx)]),
        right=jnp.concatenate([topo.right_child, jnp.full((tot - ni,), -1, idx)]),
        active=jnp.asarray(ranges[:, 1] >= ranges[:, 0]),
        root=jnp.argmin(topo.parent).astype(idx),
    )


@pytest.fixture(scope="module")
def two_domains():
    """Two Plummer halves split along x, one Morton box, as two devices see them."""
    pts = _plummer(6000, 11)
    order = np.argsort(pts[:, 0])
    half = len(pts) // 2
    bounds = infer_bounds(jnp.asarray(pts, jnp.float32))
    return _domain(pts[order[:half]], bounds), _domain(pts[order[half:]], bounds)


def _caps(summary_cell_level=None, max_leaves=4):
    cells = LEAF_CAPACITY if max_leaves == 1 else MAX_CELLS
    return CrossCapacities(
        max_cells=cells,
        max_summary_nodes=2 * cells,
        max_leaves_per_cell=max_leaves,
        export_far_cap=CAP,
        export_near_cap=CAP,
        export_walk_queue=CAP,
        summary_cell_level=summary_cell_level,
    )


def _export(send, recv, theta, *, mode, drop_entry=None, cell_level=None):
    """The hook's publish + gather + export, sender = device 1, receiver = device 0.

    ``drop_entry`` (MUTATION) marks one of the receiver's published entries inactive:
    a cell nobody exports to, or (two-sided) a whole subtree of the summary.
    """
    two_sided, max_leaves = MODES[mode]
    cap = _caps(cell_level, max_leaves)
    pub_r = _summary_rows(recv.topo, recv.geom, cap, two_sided=two_sided)
    pub_s = _summary_rows(send.topo, send.geom, cap, two_sided=two_sided)
    assert not (bool(pub_r.overflow) or bool(pub_s.overflow))
    rows_r = pub_r.rows
    if drop_entry is not None:
        rows_r = rows_r.at[drop_entry, -1].set(0.0)  # the active column is last
    ex = _export_from_rows(
        jnp.stack([rows_r, pub_s.rows]),
        send.left,
        send.right,
        send.geom,
        send.root,
        float(theta),
        jnp.asarray(1),
        cap,
        two_sided=two_sided,
        mac_type="dehnen",
        walk_fn=None,
    )
    return ex, pub_r.block_nodes, int(pub_r.block_nodes.shape[0])


def _deliver(cell, node, count, *, block, n_send):
    """build_send_buffers, then the slice device 0 receives from device 1."""
    sb = build_send_buffers(
        cell,
        node,
        count,
        ndev=2,
        max_cells=block,
        num_nodes=n_send,
        node_capacity=CAP,
        csr_capacity=CAP,
    )
    assert not (bool(sb.node_overflow) or bool(sb.csr_overflow))
    # device 1 sends to device 0 only: its device-0 block is the whole live prefix
    n_rows, n_csr = int(sb.node_sizes[0]), int(sb.csr_sizes[0])
    assert int(sb.node_sizes[1]) == 0 and int(sb.csr_sizes[1]) == 0, "self-export"
    rows = np.asarray(sb.node_rows)[:n_rows]
    got = SimpleNamespace(
        csr_cell=sb.csr_cell, csr_row=sb.csr_row, num_csr=jnp.asarray(n_csr)
    )
    return rows, got


def cross_pairs(recv, send, theta, *, mode="one_sided", direct_near=False, **kw):
    """Every pair device 0 (receiver) evaluates against device 1 (sender).

    ``direct_near`` reads the near lists straight off the CSR
    (``cross._direct_near_lists``, what the hook does under a leaf summary) instead of
    walking them.

    Returns
    -------
    list of (np.ndarray, np.ndarray)
        ``(receiver nodes, sender nodes)`` blocks: far pairs read off the CSR, and the
        receiver near walk's far and near pairs.
    """
    ex, block_nodes, block = _export(send, recv, theta, mode=mode, **kw)
    assert not (
        bool(ex.far_overflow) or bool(ex.near_overflow) or bool(ex.queue_overflow)
    )
    n_send = int(send.left.shape[0])
    n_local = int(recv.left.shape[0])

    # FAR: a pass-through -- target = the receiver's node behind the CSR index
    rows, got = _deliver(
        ex.far_cell, ex.far_node, ex.far_count, block=block, n_send=n_send
    )
    rl = _direct_far_lists(block_nodes, got)
    nf = int(rl.far_count)
    out = [
        (np.asarray(rl.far_target)[:nf], rows[np.asarray(rl.far_source)[:nf]]),
    ]

    # NEAR: imported sender leaves as childless rows behind the receiver's nodes,
    # seeded (receiver node of the CSR index, imported row), refined by the receiver
    rows_n, got_n = _deliver(
        ex.near_cell, ex.near_node, ex.near_count, block=block, n_send=n_send
    )
    n_imp = max(len(rows_n), 1)
    idx = recv.left.dtype
    rows_safe = np.zeros(n_imp, np.int64)
    rows_safe[: len(rows_n)] = rows_n
    imp_live = np.zeros(n_imp, bool)
    imp_live[: len(rows_n)] = True
    left = jnp.concatenate([recv.left, jnp.full((n_imp,), -1, idx)])
    right = jnp.concatenate([recv.right, jnp.full((n_imp,), -1, idx)])
    cen = jnp.concatenate(
        [jnp.asarray(recv.geom.center), jnp.asarray(send.geom.center)[rows_safe]]
    )
    rad = jnp.concatenate(
        [jnp.asarray(recv.geom.radius), jnp.asarray(send.geom.radius)[rows_safe]]
    )
    active = jnp.concatenate([recv.active, jnp.asarray(imp_live)])
    if direct_near:
        direct = _direct_near_lists(
            block_nodes, jnp.asarray(recv.ranges), recv.num_internal, got_n
        )
        c = int(direct.near_count)
        out.append((np.zeros(0, np.int64), np.zeros(0, np.int64)))  # no far pairs
        out.append(
            (
                np.asarray(direct.near_target)[:c],
                rows_n[np.asarray(direct.near_source)[:c]],
            )
        )
        return out
    walked = receiver_interaction_lists(
        left,
        right,
        cen,
        rad,
        n_local,
        block_nodes,
        got_n.csr_cell,
        got_n.csr_row,
        got_n.num_csr,
        float(theta),
        max_pair_queue=CAP,
        far_cap=CAP,
        near_cap=CAP,
        node_active=active,
    )
    assert not (
        bool(walked.far_overflow)
        or bool(walked.near_overflow)
        or bool(walked.queue_overflow)
    )
    for t, s, c in (
        (walked.far_target, walked.far_source, walked.far_count),
        (walked.near_target, walked.near_source, walked.near_count),
    ):
        c = int(c)
        out.append((np.asarray(t)[:c], rows_n[np.asarray(s)[:c]]))
    return out


def coverage(recv, send, pairs):
    """Count matrix over (receiver particle, sender particle) in tree order."""
    C = np.zeros((recv.n, send.n), np.int32)
    rr, sr = recv.ranges, send.ranges
    for tgt, src in pairs:
        for t, s in zip(tgt.tolist(), src.tolist()):
            t0, t1 = rr[t]
            s0, s1 = sr[s]
            if t0 <= t1 and s0 <= s1:
                C[t0 : t1 + 1, s0 : s1 + 1] += 1
    return C


def _report(C):
    return (
        f"uncovered {int(np.sum(C == 0))}, multiply covered {int(np.sum(C > 1))} "
        f"of {C.size} particle pairs"
    )


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("theta", [0.8, 0.5])
@pytest.mark.parametrize("direction", ["right_to_left", "left_to_right"])
def test_the_cross_field_covers_every_cross_pair_once(
    two_domains, mode, theta, direction
):
    a, b = two_domains
    recv, send = (a, b) if direction == "right_to_left" else (b, a)
    pairs = cross_pairs(recv, send, theta, mode=mode)
    nf = len(pairs[0][0])
    assert nf > 0 and len(pairs[2][0]) > 0, "vacuous: no far or no direct pairs"
    C = coverage(recv, send, pairs)
    assert np.all(C == 1), _report(C)


@pytest.mark.parametrize("mode", list(MODES))
def test_a_size_bounded_summary_covers_too(two_domains, mode):
    """``summary_cell_level`` (the production default is 8) changes the cut."""
    a, b = two_domains
    C = coverage(a, b, cross_pairs(a, b, 0.8, mode=mode, cell_level=3))
    assert np.all(C == 1), _report(C)


def test_over_receiver_leaves_the_near_walk_is_a_pass_through(two_domains):
    """Published per leaf, a near pair the sender emits is (receiver leaf, sender leaf)
    on the same geometry, so the receiver's near walk refines nothing: no far pairs out
    of it, and its near pairs are exactly the received CSR."""
    a, b = two_domains
    pairs = cross_pairs(a, b, 0.8, mode="two_sided_leaves")
    assert len(pairs[1][0]) == 0, "the near walk found far pairs under a leaf summary"
    assert len(pairs[2][0]) > 0, "vacuous"
    assert np.all(pairs[2][0] >= a.num_internal), "a near target that is no leaf"


@pytest.mark.parametrize("theta", [0.8, 0.5])
def test_the_hook_reads_the_leaf_summary_near_lists_off_the_csr(two_domains, theta):
    """What the hook does instead of the near walk under a leaf summary: the same
    near pairs, and the same exact coverage."""
    a, b = two_domains
    for recv, send in ((a, b), (b, a)):
        walked = cross_pairs(recv, send, theta, mode="two_sided_leaves")
        direct = cross_pairs(
            recv, send, theta, mode="two_sided_leaves", direct_near=True
        )
        assert len(walked[1][0]) == 0, "the walk was not a pass-through"
        as_set = lambda p: set(zip(p[0].tolist(), p[1].tolist()))  # noqa: E731
        assert as_set(direct[2]) == as_set(walked[2])
        assert len(direct[2][0]) == len(walked[2][0]) > 0
        C = coverage(recv, send, direct)
        assert np.all(C == 1), _report(C)


def test_two_sided_far_pairs_land_above_the_cells_and_are_fewer(two_domains):
    a, b = two_domains
    one = cross_pairs(a, b, 0.8)
    two = cross_pairs(a, b, 0.8, mode="two_sided_cells")
    assert len(two[0][0]) < len(one[0][0]), (len(two[0][0]), len(one[0][0]))
    ni = a.num_internal
    cut_cells = set(
        np.asarray(
            _summary_rows(a.topo, a.geom, _caps(), two_sided=False).block_nodes
        ).tolist()
    )
    targets = set(two[0][0].tolist())
    assert targets - cut_cells, "vacuous: no far target above the cut"
    assert any(t < ni for t in targets - cut_cells)


@pytest.mark.parametrize("mode", list(MODES))
def test_the_coverage_check_sees_a_dropped_entry(two_domains, mode):
    """MUTATION: one receiver entry left out of the export loses its particles' rows."""
    a, b = two_domains
    C = coverage(a, b, cross_pairs(a, b, 0.8, mode=mode, drop_entry=3))
    assert np.sum(C == 0) > 0 and np.sum(C > 1) == 0, _report(C)


def test_the_coverage_check_sees_a_double_count(two_domains):
    """MUTATION: one far pair emitted twice is counted twice -- momentum would not care."""
    a, b = two_domains
    pairs = cross_pairs(a, b, 0.8)
    tgt, src = pairs[0]
    pairs[0] = (np.append(tgt, tgt[:1]), np.append(src, src[:1]))
    C = coverage(a, b, pairs)
    assert np.sum(C > 1) > 0 and np.sum(C == 0) == 0, _report(C)

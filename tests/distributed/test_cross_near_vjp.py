"""Seam B of the multi-GPU gradients: the cross NEAR term's reverse (gate MG3b).

``jaccpot.distributed.cross_vjp.cross_near_acceleration_direct_cvjp`` is the
production cross near term (``cross.cross_near_acceleration_direct``) under a
``custom_vjp`` whose backward sends each sink's acceleration cotangent along the
forward's near exchange (same direction, same layout) and runs the direct near
reverse over the local leaves (``num_target_leaves=L``, no self term). Pinned here on
forced CPU devices (2 and 4), float64, the Pallas kernels interpreted, yggdrax's
``buf`` exchange:

* the per-device layouts are the production ones: a GLOBAL symmetric set of cross
  leaf pairs (every pair across devices, both devices list it, the walk's
  orientation ``a`` in the lower device's block), each device's near exchange built
  by ``cross._symmetric_exchange``'s near half on its own helpers
  (:func:`_near_exchange`, line for line) -- the send rows, the ragged exchange of
  ``[x, m]``, the received rows, the pairs in (target, source) order -- and the
  device's sorted rows a random permutation of its input rows, a padding leaf and
  dead rows included;
* the forward equals the production call BITWISE (the same program, both in it) and
  the global direct sum over the cross particle pairs to round-off;
* ``jax.grad`` of ``sum(W * acc)`` (the loss outside the ``shard_map``) with respect
  to positions, masses, ``softening_sq`` and ``G`` equals, to ~1e-12 relative,
  (1) ``jax.grad`` of a pure-JAX global pair sum over the same particle pairs, and
  (2) the single-device production reverse (``_direct_cvjp``) on all particles
  concatenated with the global symmetric leaf CSR, its self term subtracted by a
  second call over an empty CSR;
* the cross pairs are not vacuous: every device has some, and some leaf none.

Plummer at 2 and 4 devices, the solver's default ``ferrers3`` once (with pairs on
both sides of its support), and the force-scale lane once (passed through, no
cotangent). The replicated scalars' cotangents are per-device partials; with
``P()`` in_specs and ``check_vma=False`` the ``shard_map`` transpose sums them, which
the oracles measure.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_cross_near_vjp.py -q -n 0
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map

from jaccpot._searchsorted import searchsorted_method
from jaccpot.distributed.cross import (
    _in_pair_order,
    _leaf_under,
    _live_node_ends,
    _row_of_slot,
    _rows_by_peer,
    _symmetric_pair_sides,
    cross_near_acceleration_direct,
)
from jaccpot.distributed.cross_vjp import cross_near_acceleration_direct_cvjp
from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import (
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp,
)
from jaccpot.softening import pair_factors, softening_params_from_sq, support_factor

pytestmark = pytest.mark.slow

AXIS = "gpus"
W = 4  # leaf width
LIVE_LEAVES = 4  # per device, then one padding leaf (empty range past the rows)
L = LIVE_LEAVES + 1
S = L  # summary index s IS leaf row s (one-leaf cells)
NUM_INTERNAL = 3  # not L - 1: a row/node mixup would show
N = LIVE_LEAVES * W + 3  # local rows: the leaves' particles, then dead rows
SOFT = 0.5  # ferrers3: a support of 1.23 against pair distances ~0.3-3
G_VALUE = 1.3
#: measured <= 5.5e-16 against both oracles at 2 and 4 devices (float64 round-off);
#: dropping the cotangent exchange gives 0.51, a self term in the reverse 1.19
TOL = 1e-12


@pytest.fixture(autouse=True)
def _buf_exchange(monkeypatch):
    """yggdrax's all-gather ``buf`` exchange, as the forced-CPU mesh tests run it."""
    resolve = ygg_comm.resolve_ragged_method
    monkeypatch.setattr(
        ygg_comm,
        "resolve_ragged_method",
        lambda method="auto": "buf" if method == "auto" else resolve(method),
    )


def _case(ndev, seed):
    """Per-device sorted rows, leaves, permutations; the global symmetric pair set."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, W + 1, size=(ndev, LIVE_LEAVES))
    x_in = 0.6 * rng.standard_normal((ndev, N, 3))
    m_in = np.zeros((ndev, N))
    perm = np.stack([rng.permutation(N) for _ in range(ndev)]).astype(np.int32)
    leaf_ranges = np.zeros((ndev, L, 2), np.int32)
    node_ranges = np.zeros((ndev, NUM_INTERNAL + L, 2), np.int32)
    for d in range(ndev):
        start = np.concatenate([[0], np.cumsum(counts[d])[:-1]])
        n_live = int(counts[d].sum())
        leaf_ranges[d, :LIVE_LEAVES, 0] = start
        leaf_ranges[d, :LIVE_LEAVES, 1] = start + counts[d] - 1
        leaf_ranges[d, LIVE_LEAVES] = (N, N - 1)  # a padding leaf, as a shard's
        node_ranges[d, :NUM_INTERNAL] = (0, N - 1)
        node_ranges[d, NUM_INTERNAL:] = leaf_ranges[d]
        # live sorted rows carry mass; their input rows are perm[sorted row]
        m_in[d, perm[d, :n_live]] = rng.uniform(0.5, 1.5, n_live)
    # every unordered cross leaf pair with probability 0.35; leaf (0, 1) none
    pairs = []
    for d1 in range(ndev):
        for d2 in range(d1 + 1, ndev):
            for s1 in range(LIVE_LEAVES):
                for s2 in range(LIVE_LEAVES):
                    if (d1, s1) == (0, 1) or rng.random() >= 0.35:
                        continue
                    pairs.append((d1 * S + s1, d2 * S + s2))
    cap = (ndev - 1) * LIVE_LEAVES * LIVE_LEAVES
    pair_a = -np.ones((ndev, cap), np.int32)
    pair_b = -np.ones((ndev, cap), np.int32)
    pair_count = np.zeros((ndev,), np.int32)
    for d in range(ndev):
        mine = [p for p in pairs if p[0] // S == d or p[1] // S == d]
        mine = [mine[i] for i in rng.permutation(len(mine))]  # the walk's order
        pair_count[d] = len(mine)
        if mine:
            pair_a[d, : len(mine)], pair_b[d, : len(mine)] = np.asarray(mine).T
    # what each device publishes per summary node: its live particles (column 7)
    published = np.concatenate([counts, np.zeros((ndev, 1), counts.dtype)], axis=1)
    return dict(
        ndev=ndev,
        counts=counts,
        x_in=x_in,
        m_in=m_in,
        perm=perm,
        leaf_ranges=leaf_ranges,
        node_ranges=node_ranges,
        pairs=pairs,
        pair_a=pair_a,
        pair_b=pair_b,
        pair_count=pair_count,
        published=published.astype(np.int32),
        cap=dict(node=ndev * S, send_particle=ndev * S * W, recv_particle=ndev * S * W),
    )


def _near_exchange(
    me, x_sorted, m_sorted, nr, pair_a, pair_b, pair_count, published, *, ndev, cap
):
    """``cross._symmetric_exchange``'s near half, line for line, on its helpers.

    Returns the near sink, the forward exchange's send layout (``src``, ``live_q``,
    ``part_sizes``) and the receive-size mismatch flag.
    """
    idx = jnp.int32
    tree = SimpleNamespace(node_ranges=nr, left_child=jnp.zeros((NUM_INTERNAL,), idx))
    nodes = NUM_INTERNAL + jnp.arange(S, dtype=idx)
    nl, n_mine, n_peer, n_theirs = _symmetric_pair_sides(
        pair_a, pair_b, pair_count, me, S
    )
    exp_n = _rows_by_peer(n_peer, n_mine, nl, ndev=ndev, S=S, capacity=cap["node"])
    imp_n = _rows_by_peer(n_peer, n_theirs, nl, ndev=ndev, S=S, capacity=cap["node"])
    s_ok = exp_n.rows >= 0
    s_node = nodes[jnp.maximum(exp_n.rows, 0)]
    starts = nr[s_node, 0].astype(idx)
    ends = _live_node_ends(tree)
    full = jnp.where(s_ok, ends[s_node].astype(idx) - starts + 1, 0)
    counts = jnp.clip(full, 0, W)
    row, within, live_q, total = _row_of_slot(counts, cap["send_particle"])
    src = jnp.clip(starts[row] + within, 0, x_sorted.shape[0] - 1)
    parts = jnp.where(
        live_q[:, None],
        jnp.concatenate([x_sorted[src], m_sorted[src][:, None]], axis=1),
        0.0,
    )
    row_bound = jnp.concatenate(
        [jnp.zeros((1,), idx), jnp.cumsum(exp_n.sizes, dtype=idx)]
    )
    count_bound = jnp.concatenate([jnp.zeros((1,), idx), jnp.cumsum(counts, dtype=idx)])
    part_sizes = count_bound[row_bound[1:]] - count_bound[row_bound[:-1]]
    recv_p, recv_psizes, _ = ygg_comm.ragged_all_to_all_exchange(
        parts, part_sizes, output_capacity=cap["recv_particle"], axis_name=AXIS
    )
    r_ok = imp_n.rows >= 0
    r_peer = jnp.searchsorted(
        jnp.cumsum(imp_n.sizes, dtype=idx),
        jnp.arange(imp_n.rows.shape[0], dtype=idx),
        side="right",
        method=searchsorted_method(),
    )
    r_peer = jnp.minimum(r_peer, ndev - 1)
    r_count = jnp.where(r_ok, published[r_peer, jnp.maximum(imp_n.rows, 0)], 0)
    r_count = jnp.clip(r_count, 0, W)
    expect = jax.ops.segment_sum(
        r_count, jnp.where(r_ok, r_peer, ndev), num_segments=ndev + 1
    )[:ndev]
    mismatch = jnp.any(expect != recv_psizes.astype(idx))
    r_start = jnp.cumsum(r_count, dtype=idx) - r_count
    ok_n = nl & (imp_n.pair_row >= 0) & (imp_n.pair_row < cap["node"])
    leaf = _leaf_under(nodes[n_mine], nr, NUM_INTERNAL).astype(idx)
    leaf, row_n = _in_pair_order(
        ok_n,
        leaf,
        imp_n.pair_row,
        num_targets=NUM_INTERNAL + L + 1,
        num_sources=ndev * S,
    )
    near = dict(
        particles=recv_p,
        row_start=r_start,
        row_count=r_count,
        leaf_width=W,
        target_node=leaf,
        source_row=row_n,
        count=pair_count,
    )
    overflow = exp_n.overflow | imp_n.overflow | (total > cap["send_particle"])
    return near, (src, live_q, part_sizes), mismatch | overflow


def _mesh_program(c, kernel, with_force_scale):
    """``jit(shard_map)``: per device the production term and the seam's, input order."""
    ndev = c["ndev"]
    mesh = Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
    published = jnp.asarray(c["published"])
    cap = c["cap"]

    def shard(x_in, m_in, perm, leaf_ranges, nr, pair_a, pair_b, pair_count, eps2, g):
        x_in, m_in, perm = x_in[0], m_in[0], perm[0]
        leaf_ranges, nr = leaf_ranges[0], nr[0]
        pair_a, pair_b, pair_count = pair_a[0], pair_b[0], pair_count[0]
        me = lax.axis_index(AXIS)
        x_sorted, m_sorted = x_in[perm], m_in[perm]  # differentiable gathers
        # the topology from stop_gradient'd particles, as the gradient's rebuild
        near, (src, live_q, part_sizes), flag = _near_exchange(
            me,
            lax.stop_gradient(x_sorted),
            lax.stop_gradient(m_sorted),
            nr,
            pair_a,
            pair_b,
            pair_count,
            published,
            ndev=ndev,
            cap=cap,
        )
        common = dict(
            leaf_width=W,
            softening_sq=eps2,
            G=g,
            interpret=True,
            softening_kernel=kernel,
            with_force_scale=with_force_scale,
        )
        stop = lax.stop_gradient
        prod = cross_near_acceleration_direct(
            near,
            stop(x_sorted),
            stop(m_sorted),
            leaf_ranges,
            perm,
            NUM_INTERNAL,
            **dict(common, softening_sq=stop(eps2), G=stop(g)),
        )
        mine = cross_near_acceleration_direct_cvjp(
            near,
            x_sorted,
            m_sorted,
            leaf_ranges,
            perm,
            NUM_INTERNAL,
            src=src,
            live_q=live_q,
            part_sizes=part_sizes,
            axis_name=AXIS,
            **common,
        )
        if not with_force_scale:
            prod, mine = (prod, jnp.zeros((N,), x_in.dtype)), (
                mine,
                jnp.zeros((N,), x_in.dtype),
            )
        # target-leaf pairs this device evaluates (cross pairs it holds)
        held = jnp.sum(
            (jnp.arange(near["target_node"].shape[0]) < pair_count)
            & (near["target_node"] >= 0)
        )
        return (
            mine[0][None],
            mine[1][None],
            prod[0][None],
            prod[1][None],
            held[None],
            flag[None],
        )

    spec = P(AXIS)
    return jax.jit(
        shard_map(
            shard,
            mesh=mesh,
            in_specs=(spec,) * 8 + (P(), P()),
            out_specs=(spec,) * 6,
            check_vma=False,
        )
    )


def _particle_pairs(c):
    """(target, source) flat particle ids (``d * N + input row``) of the cross pairs."""
    pi, pj = [], []
    lr, perm = c["leaf_ranges"], c["perm"]

    def members(g):
        d, s = divmod(g, S)
        return [d * N + int(perm[d, r]) for r in range(lr[d, s, 0], lr[d, s, 1] + 1)]

    for a, b in c["pairs"]:
        for p in members(a):
            for q in members(b):
                pi += [p, q]
                pj += [q, p]
    return np.asarray(pi), np.asarray(pj)


def _pair_sum_oracle(c, kernel):
    """Oracle 1: a pure-JAX global sum over the cross particle pairs, flat ids."""
    pi, pj = (jnp.asarray(p) for p in _particle_pairs(c))
    total = c["ndev"] * N

    def acc(x, m, eps2, g):
        params = softening_params_from_sq(kernel, eps2, x.dtype)
        d = x[pi] - x[pj]
        gk, _, _ = pair_factors(jnp.sum(d * d, axis=1), params, kernel)
        return jax.ops.segment_sum((-g * m[pj] * gk)[:, None] * d, pi, total)

    return acc


def _direct_cvjp_oracle(c, kernel):
    """Oracle 2: the single-card production reverse (``_direct_cvjp``) on every
    particle at once -- the global symmetric leaf CSR, minus its self term (the same
    call over an empty CSR)."""
    ndev = c["ndev"]
    lr, perm = c["leaf_ranges"], c["perm"]
    # global sorted table: device d's sorted rows at d * N; flat input id of each
    sorted_ids = np.concatenate([d * N + perm[d] for d in range(ndev)])
    starts, counts = [], []
    for d in range(ndev):
        for s in range(LIVE_LEAVES):
            starts.append(d * N + lr[d, s, 0])
            counts.append(lr[d, s, 1] - lr[d, s, 0] + 1)
    nl = len(starts)

    def gid(g):
        d, s = divmod(g, S)
        return d * LIVE_LEAVES + s

    rows = [[] for _ in range(nl)]
    for a, b in c["pairs"]:
        rows[gid(a)].append(gid(b))
        rows[gid(b)].append(gid(a))
    rows = [sorted(r) for r in rows]
    cnt = np.asarray([len(r) for r in rows], np.int32)
    nbr = np.asarray(sum(rows, []) + [0], np.int32)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
    topo = (jnp.asarray(starts, jnp.int32), jnp.asarray(counts, jnp.int32))
    full = (jnp.asarray(nbr), jnp.asarray(off), jnp.asarray(cnt))
    empty = (
        jnp.zeros((1,), jnp.int32),
        jnp.zeros((nl + 1,), jnp.int32),
        jnp.zeros((nl,), jnp.int32),
    )
    ids = jnp.asarray(sorted_ids)

    def acc(x, m, eps2, g):
        xs, ms = x[ids], m[ids]

        def near(csr):
            out, _ = nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
                xs,
                ms,
                *topo,
                *csr,
                leaf_width=W,
                softening_sq=eps2,
                G=g,
                chunk=64,
                softening_kernel=kernel,
                interpret=True,
            )
            return out

        cross = near(full) - near(empty)
        return jnp.zeros_like(x).at[ids].set(cross)

    return acc


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


@pytest.mark.parametrize(
    "ndev, kernel, with_force_scale",
    [
        (2, "plummer", False),
        (4, "plummer", False),
        (2, "ferrers3", False),
        (2, "plummer", True),
    ],
)
def test_cross_near_seam_gradient_matches_the_global_oracles(
    ndev, kernel, with_force_scale
):
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    c = _case(ndev, seed=ndev + (7 if kernel == "ferrers3" else 0))
    assert len(c["pairs"]) > 0
    if kernel != "plummer":
        pi, pj = _particle_pairs(c)
        x = c["x_in"].reshape(-1, 3)
        r = np.linalg.norm(x[pi] - x[pj], axis=1)
        assert 0.05 < np.mean(r < support_factor(kernel) * SOFT) < 0.95
    program = _mesh_program(c, kernel, with_force_scale)
    weights = np.random.default_rng(3).standard_normal((ndev, N, 3))
    scale_w = np.random.default_rng(4).standard_normal((ndev, N))
    ints = tuple(
        jnp.asarray(c[k])
        for k in (
            "perm",
            "leaf_ranges",
            "node_ranges",
            "pair_a",
            "pair_b",
            "pair_count",
        )
    )

    def loss(x, m, eps2, g):
        acc, fb, acc_p, fb_p, held, flag = program(x, m, *ints, eps2, g)
        # the force-scale lane is in the loss: its cotangent must be dropped
        val = jnp.sum(jnp.asarray(weights) * acc) + jnp.sum(jnp.asarray(scale_w) * fb)
        return val, (acc, fb, acc_p, fb_p, held, flag)

    primals = (
        jnp.asarray(c["x_in"]),
        jnp.asarray(c["m_in"]),
        jnp.asarray(SOFT**2),
        jnp.asarray(G_VALUE),
    )
    (_, aux), grads = jax.value_and_grad(loss, argnums=(0, 1, 2, 3), has_aux=True)(
        *primals
    )
    acc, fb, acc_p, fb_p, held, flag = aux
    # not vacuous: every device holds cross pairs, the layout is consistent
    assert np.all(np.asarray(held) > 0), np.asarray(held)
    assert not np.any(np.asarray(flag))
    # forward: the production call's bits
    assert np.array_equal(np.asarray(acc), np.asarray(acc_p))
    if with_force_scale:
        assert np.array_equal(np.asarray(fb), np.asarray(fb_p))
        assert np.max(np.abs(np.asarray(fb))) > 0

    flat = (primals[0].reshape(-1, 3), primals[1].reshape(-1), primals[2], primals[3])
    w_flat = jnp.asarray(weights.reshape(-1, 3))
    for oracle in (_pair_sum_oracle(c, kernel), _direct_cvjp_oracle(c, kernel)):
        ref_acc = oracle(*flat)
        assert _rel(np.asarray(acc).reshape(-1, 3), ref_acc) < TOL
        ref = jax.grad(lambda *a: jnp.sum(w_flat * oracle(*a)), argnums=(0, 1, 2, 3))(
            *flat
        )
        got = (
            np.asarray(grads[0]).reshape(-1, 3),
            np.asarray(grads[1]).reshape(-1),
            grads[2],
            grads[3],
        )
        for name, x, y in zip(("positions", "masses", "softening_sq", "G"), got, ref):
            assert np.max(np.abs(np.asarray(y))) > 0, name
            assert _rel(x, y) < TOL, (name, _rel(x, y))
    # rows of no leaf (dead rows) get nothing
    dead = np.concatenate(
        [d * N + c["perm"][d, int(c["counts"][d].sum()) :] for d in range(ndev)]
    )
    assert not np.any(np.asarray(grads[0]).reshape(-1, 3)[dead])
    assert not np.any(np.asarray(grads[1]).reshape(-1)[dead])

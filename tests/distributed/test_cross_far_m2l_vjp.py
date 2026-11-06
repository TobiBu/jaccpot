"""Gate MG3a: the cross-domain M2L seam's gradient vs one device over the global pairs.

WP-G seam A (:func:`jaccpot.pallas.m2l_real_csr_lanes_cross_vjp.cross_far_m2l_cvjp`)
on forced CPU devices (2 and 4), fp64, the M2L CSR kernels interpreted, the
all-gather ``"buf"`` exchange. A synthetic but production-shaped case: per device a
handful of tree nodes with random multipoles and well-separated centres, and a
GLOBAL symmetric far pair set over (device, node) -- local pairs and cross pairs,
every device holding cross pairs (asserted). Inside ``shard_map`` each device lays
out its far exchange exactly as :func:`jaccpot.distributed.cross._symmetric_exchange`
does (``_rows_by_peer`` for the send and receive rows, the ragged exchange of
``[multipoles | centres]``, ``_in_pair_order``, ``merge_into_csr`` into its local
target-sorted CSR), the imported rows ``stop_gradient``-ed, and runs the seam.

* forward: the seam's locals equal the production launch
  (``m2l_real_csr_lanes_pallas_cvjp(..., n_targets=n_local, presorted=True)``) on
  every device BITWISE, and the one-device oracle to ~1e-14;
* gradient of ``sum(W * locals)`` wrt every device's multipoles and centres, mapped
  to global node ids: the one-device symmetric reverse over the global CSR
  (``m2l_real_csr_lanes_pallas_cvjp(symmetric=True)``, validated against the
  per-pair twin in ``tests/unit/operators/test_m2l_real_csr_lanes_vjp.py``) to
  1e-12 relative, and the pure-JAX twin's ``jax.vjp`` (``m2l_real_csr_jax``) to
  1e-10; dropping the cotangent exchange (zero imported cotangents) fails it;
* the ``dehnen_error`` variant (``cross_far_m2l_fb_cvjp``): forward bitwise the
  force-scale production launch, gradient bitwise the plain seam's.

Measured (2026-11-06, CPU, 64-core shared host): ~1 min for the file at 4 devices
(~25 s of compile per ``ndev``), 1.1 GB peak RSS; gradient vs the oracle 1e-16
relative, without the cotangent exchange 0.5-1. The ``distributed-tier`` job runs
2 devices, so its ``ndev4`` cases skip there; run them locally with 4.

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_cross_far_m2l_vjp.py -q -n 0
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jaccpot.distributed.cross import _in_pair_order, _rows_by_peer
from jaccpot.pallas.m2l_real_csr import m2l_real_csr_jax, merge_into_csr
from jaccpot.pallas.m2l_real_csr_lanes import (
    m2l_real_csr_lanes_fb_pallas_cvjp,
    m2l_real_csr_lanes_pallas_cvjp,
)
from jaccpot.pallas.m2l_real_csr_lanes_cross_vjp import (
    cross_far_m2l_cvjp,
    cross_far_m2l_fb_cvjp,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the cross seam needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_grad,
]

AXIS = "gpus"
ORDER = 3
K = 8
N_LOCAL = 6
C = (ORDER + 1) ** 2
#: the fb variant's statics
G_FB = 1.0


def _rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _global_case(ndev: int, seed: int) -> dict:
    """Nodes, a global symmetric pair set and its per-device views (numpy)."""
    rng = np.random.default_rng(seed)
    G = ndev * N_LOCAL
    # well separated: jittered points of a grid of spacing 3, radius-free nodes
    grid = np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), -1).reshape(-1, 3)
    cent = grid[rng.permutation(len(grid))[:G]] * 3.0 + rng.uniform(-0.4, 0.4, (G, 3))
    mult = rng.standard_normal((G, C))
    adj = np.triu(rng.random((G, G)) < 0.45, 1)
    adj = adj | adj.T
    owner = np.arange(G) // N_LOCAL
    # every device: a local pair and a cross pair with every peer
    for d in range(ndev):
        a = d * N_LOCAL
        adj[a, a + 1] = adj[a + 1, a] = True
        for e in range(ndev):
            if e != d:
                b = e * N_LOCAL + (d % N_LOCAL)
                adj[a + 2, b] = adj[b, a + 2] = True
    np.fill_diagonal(adj, False)
    assert np.array_equal(adj, adj.T)
    # the oracle's global CSR: rows by target, sources ascending
    g_rows = [np.flatnonzero(adj[t]) for t in range(G)]
    g_off = np.concatenate([[0], np.cumsum([len(r) for r in g_rows])])
    g_src = np.concatenate(g_rows)

    loc_src, loc_off, cross = [], [], []
    for d in range(ndev):
        lo = d * N_LOCAL
        rows = [np.flatnonzero(adj[lo + t, lo : lo + N_LOCAL]) for t in range(N_LOCAL)]
        loc_off.append(np.concatenate([[0], np.cumsum([len(r) for r in rows])]))
        loc_src.append(np.concatenate(rows))
        # cross pairs as the symmetric walk hands them out: (mine, peer, theirs),
        # in a scrambled order (the GPU walk's atomics)
        mine, theirs = np.nonzero(adj[lo : lo + N_LOCAL] & (owner[None, :] != d))
        perm = rng.permutation(len(mine))
        cross.append(
            np.stack([mine[perm], owner[theirs[perm]], theirs[perm] % N_LOCAL])
        )
    pl_cap = max(len(s) for s in loc_src) + 3
    pc_cap = max(c.shape[1] for c in cross) + 4

    def pad(a, n, v):
        return np.concatenate([a, np.full(n - len(a), v, a.dtype)])

    return dict(
        ndev=ndev,
        mult=mult,
        cent=cent,
        g_src=g_src,
        g_off=g_off,
        loc_src=np.stack([pad(s, pl_cap, -1) for s in loc_src]).astype(np.int32),
        loc_off=np.stack(loc_off).astype(np.int32),
        f_mine=np.stack([pad(c[0], pc_cap, 0) for c in cross]).astype(np.int32),
        f_peer=np.stack([pad(c[1], pc_cap, 0) for c in cross]).astype(np.int32),
        f_theirs=np.stack([pad(c[2], pc_cap, 0) for c in cross]).astype(np.int32),
        f_live=np.stack([pad(np.ones(c.shape[1], bool), pc_cap, False) for c in cross]),
        n_cross=np.array([c.shape[1] for c in cross]),
        # unique (peer, node) rows are at most (ndev - 1) * N_LOCAL; slack rows on top
        cap=(ndev - 1) * N_LOCAL + 3,
        node_mass=rng.uniform(0.5, 1.5, G),
        node_radii=rng.uniform(0.1, 0.4, G),
    )


def _device_layout(
    mult, cent, mass, loc_src, loc_off, f_mine, f_peer, f_theirs, f_live, *, ndev, cap
):
    """One device's far exchange + merged CSR, as ``_symmetric_exchange`` lays it out."""
    from yggdrax.distributed.comm import ragged_all_to_all_exchange

    S = N_LOCAL
    nodes = jnp.arange(S, dtype=jnp.int32)
    loc_src, loc_off = loc_src[0], loc_off[0]
    f_mine, f_peer, f_theirs, fl = f_mine[0], f_peer[0], f_theirs[0], f_live[0]
    exp_f = _rows_by_peer(f_peer, f_mine, fl, ndev=ndev, S=S, capacity=cap)
    imp_f = _rows_by_peer(f_peer, f_theirs, fl, ndev=ndev, S=S, capacity=cap)
    alive = exp_f.rows >= 0
    send_node = nodes[jnp.maximum(exp_f.rows, 0)]
    a2 = alive[:, None]
    payload = jax.lax.stop_gradient(
        jnp.concatenate(
            [
                jnp.where(a2, mult[send_node], 0.0),
                jnp.where(a2, cent[send_node], 0.0),
                jnp.where(a2, mass[send_node][:, None], 0.0),
            ],
            axis=1,
        )
    )
    recv, recv_sizes, _ = ragged_all_to_all_exchange(
        payload, exp_f.sizes, output_capacity=cap, axis_name=AXIS
    )
    tgt_f = nodes[f_mine]
    ok_f = fl & (imp_f.pair_row >= 0) & (imp_f.pair_row < cap)
    tgt_f, row_f = _in_pair_order(
        ok_f, tgt_f, imp_f.pair_row, num_targets=S, num_sources=ndev * S
    )
    src_f = jnp.where(row_f >= 0, S + row_f, -1)
    src, off, live = merge_into_csr(loc_src, loc_off, src_f, tgt_f, total_nodes=S)
    flags = jnp.stack(
        [
            jnp.sum(ok_f).astype(jnp.int32),
            (exp_f.overflow | imp_f.overflow).astype(jnp.int32),
            jnp.any(recv_sizes.astype(jnp.int32) != imp_f.sizes).astype(jnp.int32),
            jnp.sum(exp_f.sizes).astype(jnp.int32),
        ]
    )
    mult_full = jnp.concatenate([mult, recv[:, :C]])
    cent_full = jnp.concatenate([cent, recv[:, C : C + 3]])
    mass_full = jnp.concatenate([mass, recv[:, C + 3]])
    return dict(
        mult=mult_full,
        cent=cent_full,
        mass=mass_full,
        src=src,
        off=off,
        live=live,
        send_node=send_node,
        alive=alive,
        send_sizes=exp_f.sizes,
        flags=flags,
    )


def _seam(lay, cap):
    return cross_far_m2l_cvjp(
        lay["mult"],
        lay["cent"],
        lay["src"],
        lay["off"],
        lay["live"],
        lay["send_node"],
        lay["alive"],
        lay["send_sizes"],
        ORDER,
        K,
        True,
        "triton",
        1,
        N_LOCAL,
        cap,
        AXIS,
    )


def _fb_args(lay, radii):
    rad_full = jnp.concatenate([radii, jnp.zeros((lay["mass"].shape[0] - N_LOCAL,))])
    return (lay["mass"], rad_full, jnp.asarray(1e-4, jnp.float64), None)


class _Run:
    """The jitted shard_map programs of one ``ndev``."""

    def __init__(self, case: dict) -> None:
        self.case = case
        ndev, cap = case["ndev"], case["cap"]
        self.mesh = Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        spec = P(AXIS)
        lay_fn = functools.partial(_device_layout, ndev=ndev, cap=cap)
        self.topo = tuple(
            jnp.asarray(case[k])
            for k in ("loc_src", "loc_off", "f_mine", "f_peer", "f_theirs", "f_live")
        )

        def shard(fn, n_in, out_specs):
            return jax.jit(
                jax.shard_map(
                    fn,
                    mesh=self.mesh,
                    in_specs=(spec,) * n_in,
                    out_specs=out_specs,
                    check_vma=False,
                )
            )

        def forward(mult, cent, mass, radii, *topo):
            lay = lay_fn(mult, cent, mass, *topo)
            seam = _seam(lay, cap)
            prod = m2l_real_csr_lanes_pallas_cvjp(
                lay["mult"],
                lay["cent"],
                lay["src"],
                lay["off"],
                lay["live"],
                ORDER,
                K,
                True,
                "triton",
                1,
                N_LOCAL,
                True,
                False,
            )
            fb_args = _fb_args(lay, radii)
            seam_fb = cross_far_m2l_fb_cvjp(
                lay["mult"],
                lay["cent"],
                lay["src"],
                lay["off"],
                lay["live"],
                lay["send_node"],
                lay["alive"],
                lay["send_sizes"],
                *fb_args,
                ORDER,
                K,
                True,
                "triton",
                1,
                N_LOCAL,
                cap,
                AXIS,
                G_FB,
            )
            prod_fb = m2l_real_csr_lanes_fb_pallas_cvjp(
                lay["mult"],
                lay["cent"],
                lay["src"],
                lay["off"],
                lay["live"],
                *fb_args,
                ORDER,
                K,
                True,
                "triton",
                1,
                N_LOCAL,
                True,
                G_FB,
            )
            return (
                seam,
                prod,
                seam_fb[0],
                seam_fb[1],
                prod_fb[0],
                prod_fb[1],
                lay["flags"][None],
            )

        self.forward = shard(forward, 10, (spec,) * 7)

        def locals_(mult, cent, mass, radii, *topo):
            return _seam(lay_fn(mult, cent, mass, *topo), cap)

        def locals_fb(mult, cent, mass, radii, *topo):
            lay = lay_fn(mult, cent, mass, *topo)
            return cross_far_m2l_fb_cvjp(
                lay["mult"],
                lay["cent"],
                lay["src"],
                lay["off"],
                lay["live"],
                lay["send_node"],
                lay["alive"],
                lay["send_sizes"],
                *_fb_args(lay, radii),
                ORDER,
                K,
                True,
                "triton",
                1,
                N_LOCAL,
                cap,
                AXIS,
                G_FB,
            )[0]

        self.locals = shard(locals_, 10, spec)
        self.locals_fb = shard(locals_fb, 10, spec)

    def args(self):
        c = self.case
        return (
            jnp.asarray(c["mult"]),
            jnp.asarray(c["cent"]),
            jnp.asarray(c["node_mass"]),
            jnp.asarray(c["node_radii"]),
            *self.topo,
        )

    def grad(self, W, fn):
        def loss(m, x):
            a = self.args()
            return jnp.sum(W * fn(m, x, *a[2:]))

        a = self.args()
        return jax.jit(jax.grad(loss, argnums=(0, 1)))(a[0], a[1])


def _oracle(case: dict, W):
    """One device, all nodes, the global symmetric CSR: locals and gradients."""
    mult, cent = jnp.asarray(case["mult"]), jnp.asarray(case["cent"])
    src = jnp.asarray(np.concatenate([case["g_src"], -np.ones(5, int)]), jnp.int32)
    off = jnp.asarray(case["g_off"], jnp.int32)
    live = off[-1]

    def lanes(m, x):
        return m2l_real_csr_lanes_pallas_cvjp(
            m, x, src, off, live, ORDER, K, True, "triton", 1, None, True, True
        )

    # the twin on the live pairs only: its padding lanes (a masked r = 0 pair) give
    # a NaN centre cotangent through the mask's transpose
    tgt = np.repeat(np.arange(len(case["g_off"]) - 1), np.diff(case["g_off"]))
    tgt = jnp.asarray(tgt, jnp.int32)
    src_live = jnp.asarray(case["g_src"], jnp.int32)

    def twin(m, x):
        return m2l_real_csr_jax(m, x, src_live, tgt, order=ORDER)

    out = lanes(mult, cent)
    g_lanes = jax.grad(lambda m, x: jnp.sum(W * lanes(m, x)), argnums=(0, 1))(
        mult, cent
    )
    g_twin = jax.grad(lambda m, x: jnp.sum(W * twin(m, x)), argnums=(0, 1))(mult, cent)
    return out, g_lanes, g_twin


@pytest.fixture(scope="module", params=[2, 4], ids=["ndev2", "ndev4"])
def run(request):
    ndev = request.param
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} devices")
    case = _global_case(ndev, seed=10 + ndev)
    r = _Run(case)
    rng = np.random.default_rng(99 + ndev)
    W = jnp.asarray(rng.standard_normal((ndev * N_LOCAL, C)))
    # the all-gather "buf" exchange, as the forced-device tier runs it (what
    # "auto" resolves to on CPU anyway; pinned so a patched backend cannot flip it)
    mp = pytest.MonkeyPatch()
    mp.setattr(ygg_comm, "resolve_ragged_method", lambda method="auto": "buf")
    try:
        fwd = [np.asarray(o) for o in r.forward(*r.args())]
        grad = r.grad(W, r.locals)
        grad_fb = r.grad(W, r.locals_fb)
        # the control: the seam with the imported cotangents dropped
        grad_nox = _grad_without_exchange(r, W)
    finally:
        mp.undo()
    oracle = _oracle(case, W)
    return dict(
        case=case, fwd=fwd, grad=grad, grad_fb=grad_fb, grad_nox=grad_nox, oracle=oracle
    )


def _grad_without_exchange(r: _Run, W):
    """The seam's gradient with zero imported cotangents (a broken backward)."""
    import jaccpot.pallas.m2l_real_csr_lanes_cross_vjp as mod

    real = mod.imported_local_cotangents

    def zero(loc_bar, send_node, alive, send_sizes, *, recv_cap, axis_name):
        return jnp.zeros((recv_cap, loc_bar.shape[1]), loc_bar.dtype)

    mod.imported_local_cotangents = zero
    try:
        r2 = _Run(r.case)
        return r2.grad(W, r2.locals)
    finally:
        mod.imported_local_cotangents = real


def test_cross_pairs_are_live_and_the_layout_is_clean(run):
    flags = run["fwd"][6]
    ndev = run["case"]["ndev"]
    assert flags.shape == (ndev, 4)
    # every device applies cross pairs and ships multipoles
    assert np.all(flags[:, 0] > 0), flags
    assert np.array_equal(flags[:, 0], run["case"]["n_cross"])
    assert np.all(flags[:, 3] > 0)
    assert not np.any(flags[:, 1]) and not np.any(flags[:, 2])


def test_forward_bitwise_production_and_oracle(run):
    seam, prod, seam_fb, fb, prod_fb, prod_fb_term, _ = run["fwd"]
    assert np.array_equal(seam, prod)
    assert np.array_equal(seam_fb, prod_fb) and np.array_equal(fb, prod_fb_term)
    assert np.array_equal(seam_fb, seam)
    out = np.asarray(run["oracle"][0])
    assert np.linalg.norm(out) > 0
    print(f"ndev={run['case']['ndev']}: forward vs oracle {_rel(seam, out):.3e}")
    assert _rel(seam, out) < 1e-14


def test_gradient_matches_the_one_device_oracle(run):
    mb, cb = (np.asarray(g) for g in run["grad"])
    _, (mb_o, cb_o), (mb_t, cb_t) = run["oracle"]
    assert np.all(np.isfinite(mb)) and np.all(np.isfinite(cb))
    assert np.linalg.norm(np.asarray(cb_o)) > 0
    rm, rc = _rel(mb, mb_o), _rel(cb, cb_o)
    print(f"ndev={run['case']['ndev']}: mult rel {rm:.3e}, centre rel {rc:.3e}")
    assert rm < 1e-12 and rc < 1e-12
    tm, tc = _rel(mb, mb_t), _rel(cb, cb_t)
    print(f"ndev={run['case']['ndev']}: vs twin mult {tm:.3e}, centre {tc:.3e}")
    assert tm < 1e-10 and tc < 1e-10


def test_fb_variant_gradient_is_the_plain_seams(run):
    for x, y in zip(run["grad_fb"], run["grad"]):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_dropping_the_cotangent_exchange_fails(run):
    """The control: without the peers' cotangents the cross half goes missing."""
    _, (mb_o, cb_o), _ = run["oracle"]
    mb, cb = (np.asarray(g) for g in run["grad_nox"])
    print(
        f"ndev={run['case']['ndev']}: no cotangent exchange: mult "
        f"{_rel(mb, mb_o):.3e}, centre {_rel(cb, cb_o):.3e}"
    )
    assert _rel(mb, mb_o) > 1e-2 and _rel(cb, cb_o) > 1e-2

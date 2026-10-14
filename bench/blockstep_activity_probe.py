"""How much of the fused lane's work stays active at each block-step boundary.

usage: python bench/blockstep_activity_probe.py OUT.json [--kmax 3,4,5,6]
           -- <fused_memory_budget.py arguments, with --no-scan --eval-repeats 1 --no-analysis>

Runs ``bench/fused_memory_budget.py`` in this process and keeps its eager prepared
state (far pairs retained) and first force. From the force (plus the disc's NFW halo)
it assigns power-of-two rungs, ``dt_i = sqrt(eps / |a_i|)`` up to the criterion's
eta, which cancels: the fastest particle sits on the finest rung. On the lane's own
tree and lists it then counts, per base step of ``2**k_max`` boundaries, the work a
momentum-exact block scheme would still do at each boundary:

* MAX map (the mutual lane's: a pair's level is the max rung of its two ends,
  a cell's rung is the max inside it). Active near tiles, near particle pairs,
  far pairs, and the KICKED fraction of particles -- every particle with an active
  pair, which needs its L2L/L2P (and all particles drift and need P2M/M2M);
* MAX map with a multipole per rung (the record's option B1): far work counts
  the non-empty (rung, rung) blocks of an active pair;
* HOLD map (level = min rung, a fresh tree per level subset): work at a boundary
  is the subset rung >= floor, so it scales the whole step by ``N(>=floor)/N``.

It predicts the time per base step against a shared ``dt_min`` from the round-7
stage costs (``docs/fused_memory_2026-10.md``, "Where a step goes now"), and
reports the particle-step bound ``N 2**k_max / sum_i 2**rung_i`` no scheme can beat.
Counts only: nothing here integrates.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time

out, rest = sys.argv[1], sys.argv[2:]
kmax_list = [3, 4, 5, 6]
if rest and rest[0] == "--kmax":
    kmax_list = [int(k) for k in rest[1].split(",")]
    rest = rest[2:]
if rest and rest[0] == "--":
    rest = rest[1:]

# the far list is dropped after the downward pass unless a gradient wants it
os.environ["JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD"] = "1"

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from jaccpot import FastMultipoleMethod  # noqa: E402

_kept: dict = {}
_orig = FastMultipoleMethod.strict_fused_prepared_eval_fn


def _keeping(self, *args, **kwargs):
    prepared, eval_fn = _orig(self, *args, **kwargs)
    _kept["prepared"] = prepared

    def _eval(p):
        a = eval_fn(p)
        _kept.setdefault("force", np.asarray(a, np.float64))
        return a

    return prepared, _eval


FastMultipoleMethod.strict_fused_prepared_eval_fn = _keeping

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, "fused_memory_budget.py")
sys.argv = [_BENCH, *rest]
try:
    harness = runpy.run_path(_BENCH, run_name="__main__")
except SystemExit:
    harness = {}
if "prepared" not in _kept or "force" not in _kept:
    raise SystemExit("the harness never ran its eager prepare + eval (pass --no-scan)")

from jaccpot.runtime._interaction_cache import far_pair_targets  # noqa: E402

t0 = time.perf_counter()
prepared = _kept["prepared"]
acc = _kept["force"]
args_n = acc.shape[0]


def _arg(name: str, default: str | None = None) -> str | None:
    return rest[rest.index(name) + 1] if name in rest else default


# the external field the harness adds to the disc (an NFW halo, as tools/mesh_galaxy_run.py)
softening = float(prepared.softening) if hasattr(prepared, "softening") else None
if _arg("--ic") == "disc":
    ic = np.load("/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz")
    pos_in = np.asarray(ic["state0"][:args_n, 0], np.float64)
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    hm, rs = float(ic["halo_mass_code"]), float(ic["halo_rs_code"])
    r = np.sqrt(np.maximum(np.sum(pos_in * pos_in, 1), 1e-30))
    x = r / rs
    big = np.log1p(x) / np.maximum(r**3, 1e-30) - 1.0 / np.maximum(
        r * r * (rs + r), 1e-30
    )
    ser = (1.0 / (2.0 * rs**3)) * (1.0 - (4.0 / 3.0) * x)
    acc = acc + (-(g * hm) * np.where(x < 1e-3, ser, big))[:, None] * pos_in
eps = float(_arg("--softening") or (softening if softening else 1e-7))

tree = prepared.tree
perm = np.asarray(tree.particle_indices)  # sorted slot -> input index
ranges = np.asarray(tree.node_ranges).astype(
    np.int64
)  # inclusive (start, end), sorted order
# padding nodes (leaf capacity > live leaves) carry empty or out-of-range spans
bad = (ranges[:, 0] < 0) | (ranges[:, 1] < ranges[:, 0]) | (ranges[:, 1] >= args_n)
ranges[bad] = (0, -1)
n_nodes = ranges.shape[0]

nl = prepared.neighbor_list
leaf_nodes = np.asarray(nl.leaf_indices).astype(np.int64)
offsets = np.asarray(nl.offsets).astype(np.int64)
neighbors = np.asarray(nl.neighbors).astype(np.int64)
counts = np.asarray(nl.counts).astype(np.int64)
counts = np.maximum(counts[: leaf_nodes.shape[0]], 0)
row_of_entry = np.repeat(np.arange(leaf_nodes.shape[0]), counts)
starts = np.cumsum(counts) - counts
entry = np.repeat(offsets[: leaf_nodes.shape[0]], counts) + (
    np.arange(int(counts.sum())) - np.repeat(starts, counts)
)
near_t = leaf_nodes[row_of_entry]
near_s = neighbors[entry]
keep = (near_s >= 0) & (near_s < n_nodes) & (ranges[near_t, 1] >= 0)
near_t, near_s = near_t[keep], near_s[keep]
leaf_nodes = leaf_nodes[(leaf_nodes >= 0) & (leaf_nodes < n_nodes)]
leaf_nodes = leaf_nodes[ranges[leaf_nodes, 1] >= ranges[leaf_nodes, 0]]

compact = prepared.compact_far_pairs
far_s = np.asarray(compact.sources).astype(np.int64)
far_t = np.asarray(far_pair_targets(compact)).astype(np.int64)
n_far = getattr(compact, "far_pair_count", None)
if n_far is not None:
    far_s, far_t = far_s[: int(np.asarray(n_far))], far_t[: int(np.asarray(n_far))]
live = (far_s >= 0) & (far_t >= 0) & (far_s < n_nodes) & (far_t < n_nodes)
far_s, far_t = far_s[live], far_t[live]
print(
    f"[probe] N={args_n} nodes {n_nodes} leaves {leaf_nodes.size} "
    f"near entries {near_t.size} far pairs {far_s.size} (directed)",
    flush=True,
)

a_norm = np.linalg.norm(acc, axis=1)
dt_req = np.sqrt(eps / np.maximum(a_norm, 1e-300))
lg = np.log2(dt_req / dt_req.min())  # >= 0; the fastest particle is 0

# the round-7 stage costs (kernel ms per step), interpolated per particle by N
_STAGES = {
    8_000_000: dict(
        near=19.6,
        m2l=14.4,
        lists=7.8,
        walk=8.4,
        tree=4.3,
        com=4.9,
        upward=3.8,
        l2l=2.9,
        l2p=0.8,
        scatter=1.3,
        integ=1.5,
    ),
    100_000_000: dict(
        near=246.0,
        m2l=140.0,
        lists=96.0,
        walk=90.0,
        tree=52.0,
        com=48.0,
        upward=34.0,
        l2l=23.0,
        l2p=10.6,
        scatter=18.6,
        integ=17.0,
    ),
}
ref_n = min(_STAGES, key=lambda k: abs(np.log(k / args_n)))
st = {k: v * args_n / ref_n for k, v in _STAGES[ref_n].items()}
t_topo = st["lists"] + st["walk"] + st["tree"] + st["com"]
t_kick = st["l2l"] + st["l2p"] + st["scatter"] + st["integ"]
t_full = t_topo + st["near"] + st["m2l"] + st["upward"] + t_kick


def _floor(s: int, k: int) -> int:
    if s % (1 << k) == 0:
        return 0
    v = (s & -s).bit_length() - 1
    return k - v


def _mark_ranges(nodes: np.ndarray, n: int) -> np.ndarray:
    """Per sorted particle: covered by any of ``nodes``' ranges."""
    diff = np.zeros(n + 1, np.int64)
    if nodes.size:
        u = np.unique(nodes)
        np.add.at(diff, ranges[u, 0], 1)
        np.add.at(diff, ranges[u, 1] + 1, -1)
    return np.cumsum(diff[:-1]) > 0


report = {"n": int(args_n), "softening": eps, "stage_ref_n": ref_n, "kmax": {}}
for k in kmax_list:
    rung_in = np.clip(np.ceil(k - lg), 0, k).astype(np.int64)
    rs = rung_in[perm]
    hist = np.bincount(rung_in, minlength=k + 1)
    n_ge = np.cumsum(hist[::-1])[::-1]  # N(rung >= f)
    pref = np.zeros((args_n + 1, k + 1), np.int32)
    np.cumsum(np.eye(k + 1, dtype=np.int32)[rs], axis=0, out=pref[1:])
    nh = pref[ranges[:, 1] + 1] - pref[ranges[:, 0]]  # per-node rung histogram
    size = nh.sum(1)
    has = nh > 0
    rmax = np.where(has.any(1), (k - np.argmax(has[:, ::-1], axis=1)), -1)
    nbins = has.sum(1)
    below = np.concatenate([np.zeros((n_nodes, 1), np.int64), np.cumsum(nh, 1)], 1)

    tile_lvl = np.maximum(rmax[near_t], rmax[near_s])
    self_lvl = rmax[leaf_nodes]
    far_lvl = np.maximum(rmax[far_t], rmax[far_s])
    n_tiles = tile_lvl.size + self_lvl.size
    pairs_full = float(
        np.sum(size[near_t] * size[near_s]) + np.sum(size[leaf_nodes] ** 2)
    )
    b1_full = float(np.sum(nbins[far_t] * nbins[far_s]))

    per_floor = {}
    for f in range(k + 1):
        near_tiles = (np.sum(tile_lvl >= f) + np.sum(self_lvl >= f)) / n_tiles
        lo_t, lo_s = below[near_t, f], below[near_s, f]
        lo_l = below[leaf_nodes, f]
        near_pairs = (
            np.sum(size[near_t] * size[near_s] - lo_t * lo_s)
            + np.sum(size[leaf_nodes] ** 2 - lo_l**2)
        ) / pairs_full
        far = float(np.mean(far_lvl >= f)) if far_lvl.size else 0.0
        nb_lo_t = has[far_t, :f].sum(1) if f else np.zeros(far_t.size, np.int64)
        nb_lo_s = has[far_s, :f].sum(1) if f else np.zeros(far_s.size, np.int64)
        b1 = float(np.sum(nbins[far_t] * nbins[far_s] - nb_lo_t * nb_lo_s)) / b1_full
        kicked = (
            _mark_ranges(far_t[far_lvl >= f], args_n)
            | _mark_ranges(near_t[tile_lvl >= f], args_n)
            | _mark_ranges(leaf_nodes[self_lvl >= f], args_n)
            | (rs >= f)
        )
        per_floor[f] = dict(
            near_tiles=float(near_tiles),
            near_pairs=float(near_pairs),
            far=far,
            far_b1=b1,
            far_b1_full_over_one=b1_full / max(far_s.size, 1),
            kicked=float(kicked.mean()),
            hold_subset=float(n_ge[f] / args_n),
        )

    # one base step = 2**k boundaries (the last one, floor 0, is every level)
    floors = [_floor(s, k) for s in range(1, (1 << k) + 1)]
    shared = (1 << k) * t_full
    t_max = t_topo + sum(
        st["upward"]
        + t_kick * per_floor[f]["kicked"]
        + st["near"] * per_floor[f]["near_tiles"]
        + st["m2l"] * per_floor[f]["far"]
        for f in floors
    )
    t_b1 = t_topo + sum(
        st["upward"]
        + t_kick * per_floor[f]["kicked"]
        + st["near"] * per_floor[f]["near_tiles"]
        + st["m2l"] * per_floor[f]["far_b1"] * per_floor[f]["far_b1_full_over_one"]
        for f in floors
    )
    t_hold = sum(t_full * per_floor[f]["hold_subset"] for f in floors)
    bound = args_n * (1 << k) / float(np.sum(2.0**rung_in))
    leaf_rungs = nbins[leaf_nodes]
    report["kmax"][k] = dict(
        rung_hist=hist.tolist(),
        particle_step_bound=bound,
        leaf_single_rung_fraction=float(np.mean(leaf_rungs == 1)),
        leaf_mean_rungs=float(np.mean(leaf_rungs)),
        per_floor=per_floor,
        speedup_max=shared / t_max,
        speedup_max_b1=shared / t_b1,
        speedup_hold=shared / t_hold,
    )
    print(
        f"[probe] k_max {k}: rungs {hist.tolist()} | bound {bound:.2f}x | "
        f"predicted vs shared dt_min: max {shared / t_max:.2f}x, max+B1 {shared / t_b1:.2f}x, "
        f"HOLD {shared / t_hold:.2f}x | leaves single-rung {np.mean(leaf_rungs == 1):.2f}",
        flush=True,
    )
    for f in range(k + 1):
        p = per_floor[f]
        print(
            f"    floor {f}: near tiles {p['near_tiles']:.3f} pairs {p['near_pairs']:.3f} "
            f"far {p['far']:.3f} (B1 {p['far_b1']:.3f}) kicked {p['kicked']:.3f} "
            f"HOLD subset {p['hold_subset']:.4f}",
            flush=True,
        )

report["probe_seconds"] = time.perf_counter() - t0
with open(out, "w") as fh:
    json.dump(report, fh, indent=1)
print(f"[probe] wrote {out}", flush=True)

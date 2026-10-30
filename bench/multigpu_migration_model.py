"""Host model of the multi-GPU lane's cross near field under drift: migration or not.

The cross-domain near volume of the fused multi-GPU lane is set by its INTRUDER
leaves (S11 session 2, track F): a decomposition leaves exact Morton domains; one
drift later a domain's boundary particles sit in cells the neighbour fills densely,
the owner's cell-leaf tree does not subdivide there (a handful of its particles),
and one coarse sparse leaf fails the MAC against much of the neighbour's core. This
script models, on the HOST and without a force, what a device's leaves look like
after ``s`` drift steps under several ownership rules, and what that costs:

* ``none-axis`` -- today: the decomposition's ownership (Morton ranges in the
  per-axis box, ``rollout._repartition_shard``) kept for the whole segment;
* ``none-cube`` -- the same, the decomposition in the force's CUBE;
* ``mig-axis-frozen`` / ``mig-axis-now`` -- after every drift each particle moves to
  the device whose splitter range holds its key, the keys in the per-axis box frozen
  at the decomposition / of the drifted positions;
* ``mig-cube-frozen`` / ``mig-cube-now`` -- the same with the decomposition and the
  keys in the cube; ``-now`` is the force's own box at that force, so every device's
  particles are an exact Morton range in the box its tree is built in;
* ``split-axis`` / ``split-cube`` -- no migration, but a device's particles outside
  its own splitter range (the decomposition's box) each become a leaf of their own:
  the intruder-aware leaves, at their finest;
* ``owner-axis`` / ``owner-cube`` -- no migration, a device's intruders grouped by
  the owner's leaf cell they fall in (the owner's level there).

Per drift it reports the cross near leaf pairs (``theta |c_a - c_b| <= r_a + r_b``
over every pair of devices' leaves, COM centres and exact particle radii: what the
GPU's ``export_near_cap`` reads, within ~2 % at 2e5 and 8e6 Plummer -- track F),
the particles each device ships (its particles in leaves with a cross near pair:
``send_particle_cap``), the migrants of that step (the extra exchange; for the
``mig-*`` rules), the counts per device, and the sparse leaves.

The leaves are the galaxy preset's: cell leaves of 64, no leaf coarser than level 8
(``yggdrax._cell_partition.adaptive_cell_leaf_partition_numpy``, the device
version's reference), Morton codes in the force's box (a cube of the drifted live
positions, ``jaccpot.distributed.fused.cube_bounds``, float32 as on the cards). The
decomposition is an exact rank split of the sorted keys (the cards' sample sort is
within ~1 % of it). Drift is ``x0 + s dt v0`` (the force's ``a dt^2 / 2`` is ~1e-3
of it at a stable step).

    JAX_PLATFORMS=cpu python bench/multigpu_migration_model.py --ic plummer \\
        --n 200000 --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

_DISC_IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
_RULES = (
    "none-axis",
    "none-cube",
    "mig-axis-frozen",
    "mig-axis-now",
    "mig-cube-frozen",
    "mig-cube-now",
    "split-axis",
    "split-cube",
    "owner-axis",
    "owner-cube",
)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ic", choices=("plummer", "disc"), default="plummer")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--ndev", type=int, default=2)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--steps", type=int, default=8, help="drift steps (migrants: all)")
    p.add_argument("--pairs-at", default="0,1,2,4,8", help="drifts to count pairs at")
    p.add_argument("--rules", default=",".join(_RULES))
    p.add_argument("--theta", type=float, default=0.8)
    p.add_argument("--leaf", type=int, default=64)
    p.add_argument("--min-level", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _plummer_velocities(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Isotropic equilibrium speeds, G = M = a = 1 (bench/multigpu_segment_gate.py)."""
    r = np.linalg.norm(pos, axis=1)
    n = r.shape[0]
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < x * x * (1.0 - x * x) ** 3.5
        q[todo[ok]] = x[ok]
        todo = todo[~ok]
    speed = q * np.sqrt(2.0) * (1.0 + r * r) ** -0.25
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    return speed[:, None] * np.stack([st * np.cos(phi), st * np.sin(phi), mu], 1)


def _ic(args: argparse.Namespace) -> tuple:
    """``(pos, vel, dt)`` float32, as ``bench/multigpu_segment_gate.py`` builds them."""
    if args.ic == "plummer":
        sys.path.insert(
            0,
            os.environ.get(
                "BENCH_DIR",
                "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu",
            ),
        )
        from common.ic import IC_GENERATORS

        pos, _ = IC_GENERATORS["plummer"](args.n, seed=args.seed)
        pos = np.asarray(pos, np.float64)
        vel = _plummer_velocities(pos, np.random.default_rng(7))
        dt = 5e-3 if args.dt is None else args.dt
        return pos.astype(np.float32), vel.astype(np.float32), dt
    ic = np.load(_DISC_IC)
    state = np.asarray(ic["state0"][: args.n], np.float32)
    dt = 5e-4 if args.dt is None else args.dt
    return state[:, 0], state[:, 1], dt


def _box(x: np.ndarray, cubic: bool) -> tuple:
    """The lane's Morton box of live positions (``global_mesh_bounds``), float32."""
    from jaccpot.distributed.fused import cube_bounds

    lo = jnp.asarray(x.min(0), jnp.float32)
    hi = jnp.asarray(x.max(0), jnp.float32)
    return tuple(
        jnp.asarray(b, jnp.float32) for b in cube_bounds(lo, hi, pad=1e-6, cubic=cubic)
    )


def _codes(x: np.ndarray, box: tuple) -> np.ndarray:
    from yggdrax.morton import morton_encode

    return np.asarray(morton_encode(jnp.asarray(x, jnp.float32), box)).astype(np.uint64)


def _leaves(x: np.ndarray, codes: np.ndarray, leaf: int, min_level: int) -> dict:
    """Cell leaves of one device's particles: COM, exact radius, size, depth, cell."""
    from yggdrax._cell_partition import (
        MORTON_LEVELS,
        adaptive_cell_leaf_partition_numpy,
    )

    order = np.argsort(codes, kind="stable")
    sc = codes[order]
    starts, ends, depths = adaptive_cell_leaf_partition_numpy(
        sc, leaf_size=leaf, min_level=min_level
    )
    xs = x[order].astype(np.float64)
    n = ends - starts
    com = np.add.reduceat(xs, starts, axis=0) / n[:, None]
    seg = np.repeat(np.arange(len(starts)), n)
    r = np.zeros(len(starts))
    np.maximum.at(r, seg, np.linalg.norm(xs - com[seg], axis=1))
    shift = (3 * (MORTON_LEVELS - np.asarray(depths))).astype(np.uint64)
    return dict(
        com=com,
        r=r,
        n=n,
        depth=np.asarray(depths),
        cell=sc[starts] >> shift,
        first=sc[starts],
        order=order,
        seg=seg,
    )


def _singletons(x: np.ndarray) -> dict:
    n = x.shape[0]
    return dict(
        com=x.astype(np.float64),
        r=np.zeros(n),
        n=np.ones(n, np.int64),
        depth=np.full(n, 21),
    )


def _grouped(x: np.ndarray, labels: np.ndarray) -> dict:
    """Leaves of particles grouped by an integer label."""
    if not len(labels):
        return _singletons(x)
    order = np.argsort(labels, kind="stable")
    lab = labels[order]
    xs = x[order].astype(np.float64)
    change = np.concatenate([[True], lab[1:] != lab[:-1]])
    starts = np.flatnonzero(change)
    n = np.diff(np.concatenate([starts, [len(lab)]]))
    com = np.add.reduceat(xs, starts, axis=0) / n[:, None]
    seg = np.repeat(np.arange(len(starts)), n)
    r = np.zeros(len(starts))
    np.maximum.at(r, seg, np.linalg.norm(xs - com[seg], axis=1))
    return dict(com=com, r=r, n=n, depth=np.zeros(len(starts), np.int64))


def _concat(*parts: dict) -> dict:
    keys = ("com", "r", "n", "depth")
    return {k: np.concatenate([p[k] for p in parts if len(p["r"])]) for k in keys}


def _near_pairs(la: dict, lb: dict, theta: float) -> np.ndarray:
    """Per leaf of ``la``: its leaves of ``lb`` with ``theta d <= r_a + r_b``.

    ``lb`` is binned by radius (factors of two), one k-d tree per bin, each queried
    at ``(r_a + r_bin_max) / theta``: the candidates stay within a few times the
    hits however wide the radius spread (Plummer outskirts: leaves of r ~ 20).
    """
    from scipy.spatial import cKDTree

    per_a = np.zeros(len(la["r"]), np.int64)
    if not len(la["r"]) or not len(lb["r"]):
        return per_a
    rb_all = lb["r"]
    floor = max(float(np.quantile(rb_all, 0.01)), 1e-12)
    edges = [0.0]
    while edges[-1] < rb_all.max():
        edges.append(floor if edges[-1] == 0.0 else 2.0 * edges[-1])
    edges[-1] = np.nextafter(edges[-1], np.inf)
    for lo_r, hi_r in zip(edges[:-1], edges[1:]):
        sel = (rb_all >= lo_r) & (rb_all < hi_r)
        if not sel.any():
            continue
        cb = lb["com"][sel]
        rb = rb_all[sel]
        tree = cKDTree(cb)
        cut = float(rb.max())
        # chunks keep the candidate lists bounded
        for lo in range(0, len(la["r"]), 20_000):
            sl = slice(lo, lo + 20_000)
            cand = tree.query_ball_point(
                la["com"][sl], (la["r"][sl] + cut) / theta + 1e-12, workers=4
            )
            lens = np.fromiter((len(c) for c in cand), np.int64, len(cand))
            if not lens.sum():
                continue
            flat = np.concatenate([np.asarray(c, np.int64) for c in cand if len(c)])
            owner = np.repeat(np.arange(len(cand)), lens)
            d = np.linalg.norm(cb[flat] - la["com"][sl][owner], axis=1)
            hit = theta * d <= la["r"][sl][owner] + rb[flat]
            per_a[sl] += np.bincount(owner[hit], minlength=len(cand))
    return per_a


def _cross(leaves: list, theta: float) -> dict:
    """Cross near pairs and shipped particles per device over every device pair."""
    ndev = len(leaves)
    pairs = np.zeros(ndev, np.int64)
    shipped = np.zeros(ndev, np.int64)
    worst: list = []
    for a in range(ndev):
        for b in range(ndev):
            if a == b:
                continue
            per = _near_pairs(leaves[a], leaves[b], theta)
            pairs[a] += int(per.sum())
            shipped[a] += int(leaves[a]["n"][per > 0].sum())
            i = int(np.argmax(per)) if len(per) else 0
            if len(per):
                worst.append(
                    dict(
                        dev=a,
                        peer=b,
                        pairs=int(per[i]),
                        n=int(leaves[a]["n"][i]),
                        r=float(leaves[a]["r"][i]),
                        dist=float(np.linalg.norm(leaves[a]["com"][i])),
                    )
                )
    worst.sort(key=lambda w: -w["pairs"])
    return dict(
        near_pairs=pairs.tolist(),
        shipped=shipped.tolist(),
        worst_leaf=worst[0] if worst else None,
        sparse_leaves=[int((lv["n"] <= 8).sum()) for lv in leaves],
        leaves=[int(len(lv["r"])) for lv in leaves],
    )


def _owners(codes: np.ndarray, splitters: np.ndarray) -> np.ndarray:
    """``sfc_repartition``'s rule: ``searchsorted(pivots, code, side='right')``."""
    return np.searchsorted(splitters, codes, side="right").astype(np.int32)


def _rank_splitters(codes: np.ndarray, ndev: int) -> np.ndarray:
    sc = np.sort(codes)
    n = len(sc)
    return np.asarray([sc[(d * n) // ndev] for d in range(1, ndev)], np.uint64)


def _model(args: argparse.Namespace, pos: np.ndarray, vel: np.ndarray, dt: float):
    ndev = int(args.ndev)
    pairs_at = sorted({int(s) for s in args.pairs_at.split(",")})
    steps = max(int(args.steps), max(pairs_at))
    rules = [r for r in args.rules.split(",") if r]
    # the decompositions at the start positions, in both boxes
    boxes0 = {"axis": _box(pos, False), "cube": _box(pos, True)}
    split0 = {}
    owner0 = {}
    for kind, box in boxes0.items():
        c0 = _codes(pos, box)
        split0[kind] = _rank_splitters(c0, ndev)
        owner0[kind] = _owners(c0, split0[kind])
    out: dict = {r: [] for r in rules}
    prev: dict = {r: None for r in rules}
    for s in range(0, steps + 1):
        x = pos + np.float32(s * dt) * vel
        force_box = _box(x, True)
        fcodes = _codes(x, force_box)
        cache: dict = {}
        for rule in rules:
            fam, kind = rule.split("-")[0], rule.split("-")[1]
            t0 = time.perf_counter()
            if fam == "mig":
                frame = rule.split("-")[2]
                if frame == "frozen":
                    codes = _codes(x, boxes0[kind])
                elif kind == "cube":
                    codes = fcodes  # the force's own box
                else:
                    codes = _codes(x, _box(x, False))
                owner = _owners(codes, split0[kind])
            else:
                owner = owner0[kind]
            migrants = 0 if prev[rule] is None else int(np.sum(owner != prev[rule]))
            prev[rule] = owner
            row = dict(
                drift=s,
                migrants=migrants,
                counts=np.bincount(owner, minlength=ndev).tolist(),
            )
            if s in pairs_at:
                if fam in ("split", "owner"):
                    # intruders: outside the device's splitter range in the
                    # decomposition's (frozen) box
                    key = ("dcodes", kind)
                    if key not in cache:
                        cache[key] = _owners(_codes(x, boxes0[kind]), split0[kind])
                    home = cache[key]
                    intr = home != owner
                    row["intruders"] = [
                        int(np.sum(intr & (owner == d))) for d in range(ndev)
                    ]
                    own_leaves = []
                    for d in range(ndev):
                        sel = (owner == d) & ~intr
                        own_leaves.append(
                            _leaves(x[sel], fcodes[sel], args.leaf, args.min_level)
                        )
                    leaves = []
                    for d in range(ndev):
                        isel = np.flatnonzero((owner == d) & intr)
                        if fam == "split":
                            extra = _singletons(x[isel])
                        else:
                            # the owner's leaf cell holding each intruder
                            lab = np.empty(len(isel), np.int64)
                            for h in range(ndev):
                                hsel = home[isel] == h
                                lv = own_leaves[h]
                                j = np.searchsorted(
                                    lv["first"], fcodes[isel][hsel], "right"
                                )
                                lab[hsel] = h * (1 << 40) + np.maximum(j - 1, 0)
                            extra = _grouped(x[isel], lab)
                        leaves.append(_concat(own_leaves[d], extra))
                else:
                    leaves = []
                    for d in range(ndev):
                        sel = owner == d
                        leaves.append(
                            _leaves(x[sel], fcodes[sel], args.leaf, args.min_level)
                        )
                row.update(_cross(leaves, args.theta))
                row["model_s"] = time.perf_counter() - t0
                print(rule, json.dumps(row), flush=True)
            out[rule].append(row)
    return out


def main() -> None:
    args = _args()
    pos, vel, dt = _ic(args)
    t0 = time.perf_counter()
    rows = _model(args, pos, vel, dt)
    json.dump(
        dict(args=vars(args), dt=dt, wall_s=time.perf_counter() - t0, rules=rows),
        open(args.out, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()

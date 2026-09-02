"""A1 -- test the traced-wavefront diagnosis behind the distributed per-device ceiling.

``docs/plan_2026-08_A_ceiling.md`` starts from a diagnosis rather than an
investigation, and asks for it to be falsified sharply before anything is
changed. The diagnosis: under ``shard_map`` the dual-tree walk's capacity-retry
ladder cannot read its own overflow flag (it is a tracer), so it breaks on the
first rung and declares success. The first rung of
``yggdrax._interactions_impl._queue_candidates_bounded`` is
``max(1024, process_block * 16)``, so on the traced path the wavefront capacity
would be ``process_block * 16`` and *not* the caller's ``max_pair_queue``.

Two predictions follow, and this module measures both:

``--sweep process-block``
    Vary only ``process_block``. ``32`` and ``64`` must give **identical**
    results (both floor to 1024) -- the sharpest prediction in the set, because
    nothing else would produce it -- and each doubling above 64 must widen the
    reachable wavefront by 2x.

``--sweep pair-queue``
    Hold ``process_block`` and vary ``max_pair_queue`` over four decades.
    Nothing may change. If something does, the diagnosis is wrong or incomplete.

``--sweep neighbors``
    Not in the plan, added because the first run of the other two demanded it:
    at the shipped distributed ``max_neighbors_per_leaf=128`` the *neighbour*
    capacity binds before the queue does, so a queue sweep run at the default
    measures the neighbour cap instead. Sweep it to find where each cap starts
    binding before reading either of the other two.

TWO LEVELS, AND WHY THE CHEAP ONE IS THE REAL TEST
--------------------------------------------------
``--level walk`` (default) runs the yggdrax walk directly, once eagerly and once
under ``jax.jit``, on the same tree with the same capacities. Eager is ground
truth: the flags are concrete, the ladder works, the answer is the untruncated
one. ``jit`` is the distributed path's arithmetic exactly -- the flag is a
tracer either way, and ``shard_map`` adds nothing to *this* mechanism. So the
walk level isolates the mechanism, costs seconds instead of the ~200 s per point
a distributed compile costs, and gives an untruncated reference to compare
against, which the driver level cannot.

``--level driver`` runs the real thing through ``make_force_evaluator`` and reads
``self_near_pairs`` out of the per-device diagnostics, to confirm the mechanism
survives the trip through ``shard_map``. It has no ground-truth column.

THE CACHE THAT INVALIDATES THE OBVIOUS EXPERIMENT
-------------------------------------------------
``_run_dual_tree_walk_raw`` memoises the capacity the ladder settled on, keyed on
tree shape and capacities, in a process-global dict. So running eager *before*
jit at the same point primes the cache and the jit run reads the *converged*
capacity rather than the first rung -- and reports the untruncated answer,
apparently refuting the diagnosis. The first draft of this measurement did
exactly that. Every point here clears ``_DUAL_TREE_QUEUE_CACHE`` before each arm
and runs the jit arm first, which is also what a real distributed run sees: the
cache is cold, because nothing ran eagerly to warm it.

USAGE
-----
    python -u -m bench.distributed_ceiling_sweep --sweep process-block
    python -u -m bench.distributed_ceiling_sweep --sweep pair-queue
    python -u -m bench.distributed_ceiling_sweep --sweep neighbors
    python -u -m bench.distributed_ceiling_sweep --sweep process-block --level driver --ndev 2

Results land in ``bench/results/distributed_ceiling/`` as JSON.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

_RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "distributed_ceiling"

#: The four decades of ``max_pair_queue`` the plan asks for.
_QUEUE_DECADES = (1 << 12, 1 << 15, 1 << 18, 1 << 21, 1 << 24)

#: The plan's process_block set. 32 and 64 are the pair that must agree exactly.
_PROCESS_BLOCKS = (32, 64, 128, 256, 512)

#: Neighbour caps spanning the shipped distributed default (128) and well past it.
_NEIGHBOR_CAPS = (128, 256, 512, 1024, 2048, 4096)


def _disc(
    n: int, radius: float = 10.0, thickness: float = 0.2, seed: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    """A thin exponential-ish disc: the geometry the track's target load uses.

    Separated clusters make the cross-domain term negligible and so exercise
    nothing; a disc keeps every device's frontier adjacent to its neighbours'.
    """
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack(
        [r * np.cos(th), r * np.sin(th), rng.normal(scale=thickness, size=n)], axis=1
    )
    return pos.astype(np.float64), rng.uniform(0.8, 1.2, n).astype(np.float64)


# ---------------------------------------------------------------------------
# walk level -- the mechanism, in isolation, with an untruncated reference
# ---------------------------------------------------------------------------


def _clear_queue_cache() -> None:
    """Drop the memoised ladder capacities so every point starts cold.

    Imported from ``_interactions_impl`` rather than a re-export: the module that
    defines a private name is the only place it is guaranteed to still exist.
    """
    from yggdrax._interactions_impl import _DUAL_TREE_QUEUE_CACHE

    _DUAL_TREE_QUEUE_CACHE.clear()


def _build_tree(n: int, leaf: int, seed: int) -> tuple[Any, Any, int]:
    from yggdrax.tree import Tree

    from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled

    pos, mass = _disc(n, seed=seed)
    tree = Tree.from_particles(
        jnp.asarray(pos), jnp.asarray(mass), leaf_size=leaf, tree_type="radix"
    )
    geom = compute_tree_geometry_compiled(
        tree, tree.positions_sorted, max_leaf_size=leaf
    )
    topo = tree.topology
    num_leaves = int(topo.parent.shape[0]) - int(topo.left_child.shape[0])
    return topo, geom, num_leaves


def _walk_once(
    topo: Any,
    geom: Any,
    *,
    theta: float,
    mac_type: str,
    process_block: int,
    max_pair_queue: int,
    max_neighbors_per_leaf: int,
    max_interactions_per_node: int,
    jit: bool,
) -> dict:
    """One walk, cold cache, returning the counts and every overflow flag."""
    from yggdrax.interactions import (
        DualTreeTraversalConfig,
        build_interactions_and_neighbors,
    )

    cfg = DualTreeTraversalConfig(
        max_interactions_per_node=max_interactions_per_node,
        max_neighbors_per_leaf=max_neighbors_per_leaf,
        max_pair_queue=max_pair_queue,
        process_block=process_block,
    )
    ladder: list[tuple[str, int, int]] = []

    def body(topo: Any, geom: Any) -> tuple:
        _far, _near, res = build_interactions_and_neighbors(
            topo,
            geom,
            theta=theta,
            traversal_config=cfg,
            mac_type=mac_type,
            return_result=True,
            retry_logger=(
                None
                if jit
                else (
                    lambda ev: ladder.append(
                        (
                            ev.status,
                            int(ev.queue_capacity),
                            int(ev.interaction_capacity),
                        )
                    )
                )
            ),
        )
        return (
            res.near_pair_count,
            res.far_pair_count,
            res.queue_overflow,
            res.far_overflow,
            res.near_overflow,
        )

    _clear_queue_cache()
    fn = jax.jit(body) if jit else body
    t0 = time.perf_counter()
    near, far, q_ovf, f_ovf, n_ovf = fn(topo, geom)
    out = {
        "near_pairs": int(near),
        "far_pairs": int(far),
        "queue_overflow": int(q_ovf),
        "far_overflow": int(f_ovf),
        "near_overflow": int(n_ovf),
        "seconds": round(time.perf_counter() - t0, 3),
    }
    if not jit:
        out["ladder"] = ladder
    return out


def _walk_point(topo: Any, geom: Any, *, label: dict, **caps: Any) -> dict:
    """The jit arm (the distributed path) against the eager arm (ground truth).

    jit runs first: it is the arm whose answer the cache would corrupt.
    """
    traced = _walk_once(topo, geom, jit=True, **caps)
    eager = _walk_once(topo, geom, jit=False, **caps)
    truncated = traced["near_pairs"] != eager["near_pairs"]
    return {
        **label,
        **{k: v for k, v in caps.items() if k not in ("theta", "mac_type")},
        "traced": traced,
        "eager": eager,
        "truncated": truncated,
        # A truncation nobody can see from the return value is the failure mode
        # the whole track is about, so name it as its own column.
        "silent": truncated
        and not (
            traced["queue_overflow"]
            or traced["far_overflow"]
            or traced["near_overflow"]
        ),
        "near_ratio": (
            round(traced["near_pairs"] / eager["near_pairs"], 6)
            if eager["near_pairs"]
            else None
        ),
    }


# ---------------------------------------------------------------------------
# occupancy -- what capacities the geometry actually needs (feeds A3's sizing)
# ---------------------------------------------------------------------------

#: Per-device particle counts to characterise. The top of the range is the
#: track's target; the bottom is where the distributed tier's tests live.
_OCCUPANCY_N = (2048, 8192, 32768, 131072)


def _occupancy(args: argparse.Namespace) -> dict:
    """Measure the per-node far and per-leaf near occupancy an eager walk needs.

    ``DistributedFMMConfig``'s buffers are ``[total_nodes, max_interactions_per_node]``
    and ``[num_leaves, max_neighbors_per_leaf]``, so their cost is the *product*
    and only the second factor is a free choice. Guessing that factor is how the
    shipped defaults ended up at 512/128; measuring it says what the geometry
    needs, and whether it grows with N (it should not -- both are per-node
    budgets) or only with leaf size (it should -- a bigger leaf has more
    particles behind each neighbour).
    """
    from yggdrax.interactions import (
        DualTreeTraversalConfig,
        build_interactions_and_neighbors,
    )

    rows = []
    for n in _OCCUPANCY_N:
        if n < args.leaf * 4:
            continue
        topo, geom, num_leaves = _build_tree(n, args.leaf, args.seed)
        total_nodes = int(topo.parent.shape[0])
        # The *true* upper bounds, not an arbitrary large number: a node cannot
        # have more far interactions than there are nodes, nor a leaf more
        # neighbours than there are leaves. That keeps the observed maxima the
        # geometry's own while keeping the buffers from going quadratic -- a flat
        # ``1 << 15`` asks for 4 GB of neighbour buffer at 16 384 leaves.
        config = DualTreeTraversalConfig(
            max_interactions_per_node=total_nodes,
            max_neighbors_per_leaf=num_leaves,
            max_pair_queue=1 << 22,
            process_block=256,
        )
        _clear_queue_cache()
        interactions, neighbors, result = build_interactions_and_neighbors(
            topo,
            geom,
            theta=args.theta,
            traversal_config=config,
            mac_type=args.mac_type,
            return_result=True,
        )
        # What wavefront the walk actually needs, straight from the eager ladder:
        # it climbs 1024, 2048, 4096, ... and reports the rung it stopped on, so the
        # requirement is bracketed to within 2x for free. Started from
        # ``process_block=32`` (first rung 1024) precisely so it has to climb --
        # from a rung that already fits, no event is emitted and there is nothing
        # to read. This is the number the distributed ``max_pair_queue`` default has
        # to clear, and guessing it is how the shipped 32768 got there.
        ladder: list[int] = []
        _clear_queue_cache()
        build_interactions_and_neighbors(
            topo,
            geom,
            theta=args.theta,
            traversal_config=DualTreeTraversalConfig(
                max_interactions_per_node=total_nodes,
                max_neighbors_per_leaf=num_leaves,
                max_pair_queue=1 << 24,
                process_block=32,
            ),
            mac_type=args.mac_type,
            return_result=True,
            retry_logger=lambda ev: ladder.append(int(ev.queue_capacity)),
        )
        row = {
            "n": n,
            "leaf_size": args.leaf,
            "num_leaves": num_leaves,
            "total_nodes": total_nodes,
            "converged_pair_queue": ladder[-1] if ladder else None,
            "pair_queue_per_leaf": (
                round(ladder[-1] / num_leaves, 1) if ladder else None
            ),
            "far_pairs": int(result.far_pair_count),
            "near_pairs": int(result.near_pair_count),
            "max_far_per_node": int(jnp.max(interactions.counts)),
            "max_near_per_leaf": int(jnp.max(neighbors.counts)),
            "overflow": bool(
                result.queue_overflow | result.far_overflow | result.near_overflow
            ),
        }
        rows.append(row)
        print(
            f"n={n:8d} leaf={args.leaf:4d} leaves={num_leaves:7d} "
            f"nodes={total_nodes:7d} | max_far/node={row['max_far_per_node']:6d} "
            f"max_near/leaf={row['max_near_per_leaf']:6d} "
            f"far={row['far_pairs']:9d} near={row['near_pairs']:11d} "
            f"queue={row['converged_pair_queue']} "
            f"(={row['pair_queue_per_leaf']}/leaf) ovf={row['overflow']}",
            flush=True,
        )
    return {"points": rows}


def _walk_sweep(args: argparse.Namespace) -> dict:
    topo, geom, num_leaves = _build_tree(args.per_device, args.leaf, args.seed)
    base = dict(
        theta=args.theta,
        mac_type=args.mac_type,
        process_block=args.process_block,
        max_pair_queue=args.max_pair_queue,
        max_neighbors_per_leaf=args.max_neighbors_per_leaf,
        max_interactions_per_node=args.max_interactions_per_node,
    )
    label = {"n": args.per_device, "leaf_size": args.leaf, "num_leaves": num_leaves}
    points = []
    if args.sweep == "process-block":
        varied = [dict(base, process_block=pb) for pb in _PROCESS_BLOCKS]
    elif args.sweep == "pair-queue":
        varied = [dict(base, max_pair_queue=q) for q in _QUEUE_DECADES]
    elif args.sweep == "neighbors":
        varied = [dict(base, max_neighbors_per_leaf=nb) for nb in _NEIGHBOR_CAPS]
    else:  # pragma: no cover -- argparse constrains this
        raise ValueError(args.sweep)
    for caps in varied:
        point = _walk_point(topo, geom, label=label, **caps)
        points.append(point)
        print(_walk_line(point), flush=True)
    return {"points": points, "num_leaves": num_leaves}


def _walk_line(point: dict) -> str:
    t, e = point["traced"], point["eager"]
    flags = "".join(
        name
        for name, key in (
            ("q", "queue_overflow"),
            ("f", "far_overflow"),
            ("n", "near_overflow"),
        )
        if t[key]
    )
    verdict = (
        "ok" if not point["truncated"] else ("SILENT" if point["silent"] else "cut")
    )
    return (
        f"pb={point['process_block']:5d} queue={point['max_pair_queue']:9d} "
        f"nbr={point['max_neighbors_per_leaf']:5d} | "
        f"traced_near={t['near_pairs']:9d} eager_near={e['near_pairs']:9d} "
        f"ratio={point['near_ratio']} flags={flags or '-':3s} {verdict}"
    )


# ---------------------------------------------------------------------------
# driver level -- the real shard_map pipeline, confirming the mechanism survives
# ---------------------------------------------------------------------------


def _driver_point(*, ndev: int, args: argparse.Namespace, **caps: Any) -> dict:
    """One distributed evaluation; ``self_near_pairs`` summed over devices.

    ``make_force_evaluator``, not ``distributed_fmm_accelerations``: the latter
    recompiles per call and retries on overflow, which turns a truncation into a
    timeout and makes the point unreadable either way.
    """
    import dataclasses

    from jax.sharding import Mesh

    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
    )

    config = dataclasses.replace(
        DistributedFMMConfig(
            leaf_size=args.leaf,
            theta=args.theta,
            mac_type=args.mac_type,
            order=args.order,
            nearfield_backend=args.nearfield_backend,
        ),
        **caps,
    )
    pos, mass = _disc(args.per_device * ndev, seed=args.seed)
    part = partition_for_devices(
        pos, mass, ndev, leaf_size=args.leaf, partitioner=config.partitioner
    )
    mesh = Mesh(np.asarray(jax.devices()[:ndev]).reshape(ndev), ("gpus",))
    evaluate = make_force_evaluator(config, ndev, part["cap"], mesh, jit=False)
    t0 = time.perf_counter()
    _accel, _gid, diag = evaluate(
        part["pos_flat"], part["mass_flat"], part["gid_flat"], part["counts"]
    )
    diag = np.asarray(diag)
    out = {
        **{k: v for k, v in caps.items()},
        "n_total": int(part["n"]),
        "cap": int(part["cap"]),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    for i, name in enumerate(DIAG_FIELDS):
        out[name] = [float(v) for v in diag[:, i]]
    return out


def _driver_sweep(args: argparse.Namespace) -> dict:
    ndev = args.ndev
    if len(jax.devices()) < ndev:
        raise SystemExit(
            f"need {ndev} devices, saw {len(jax.devices())}; force host devices with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=2"
        )
    base = dict(
        process_block=args.process_block,
        max_pair_queue=args.max_pair_queue,
        max_neighbors_per_leaf=args.max_neighbors_per_leaf,
        max_interactions_per_node=args.max_interactions_per_node,
    )
    if args.sweep == "process-block":
        varied = [dict(base, process_block=pb) for pb in _PROCESS_BLOCKS]
    elif args.sweep == "pair-queue":
        varied = [dict(base, max_pair_queue=q) for q in _QUEUE_DECADES]
    else:
        varied = [dict(base, max_neighbors_per_leaf=nb) for nb in _NEIGHBOR_CAPS]
    points = []
    for caps in varied:
        point = _driver_point(ndev=ndev, args=args, **caps)
        points.append(point)
        print(
            f"pb={point['process_block']:5d} queue={point['max_pair_queue']:9d} "
            f"nbr={point['max_neighbors_per_leaf']:5d} | "
            f"self_near={sum(point['self_near_pairs']):11.0f} "
            f"self_far={sum(point['self_far_pairs']):10.0f} "
            f"q_ovf={sum(point['self_queue_overflow']):.0f} "
            f"n_ovf={sum(point['self_near_overflow']):.0f} "
            f"({point['seconds']}s)",
            flush=True,
        )
    return {"points": points}


def main(argv: Optional[list[str]] = None) -> int:
    """Run one sweep and write it to ``bench/results/distributed_ceiling``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sweep",
        choices=("process-block", "pair-queue", "neighbors", "occupancy"),
        required=True,
    )
    ap.add_argument("--level", choices=("walk", "driver"), default="walk")
    ap.add_argument("--ndev", type=int, default=2)
    ap.add_argument("--per-device", type=int, default=16384)
    ap.add_argument("--leaf", type=int, default=8)
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--mac-type", default="dehnen")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--nearfield-backend", default="baseline")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--process-block", type=int, default=64)
    ap.add_argument("--max-pair-queue", type=int, default=1 << 15)
    ap.add_argument("--max-neighbors-per-leaf", type=int, default=128)
    ap.add_argument("--max-interactions-per-node", type=int, default=512)
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    print(
        f"# level={args.level} sweep={args.sweep} per_device={args.per_device} "
        f"leaf={args.leaf} mac={args.mac_type} theta={args.theta} "
        f"devices={[d.platform for d in jax.devices()]}",
        flush=True,
    )
    if args.sweep == "occupancy":
        body = _occupancy(args)
    else:
        body = _walk_sweep(args) if args.level == "walk" else _driver_sweep(args)
    record = {
        "sweep": args.sweep,
        "level": args.level,
        "args": vars(args),
        "jax": jax.__version__,
        "x64": bool(jax.config.read("jax_enable_x64")),
        "platform": [d.platform for d in jax.devices()],
        "host": platform.node(),
        **body,
    }
    _RESULTS.mkdir(parents=True, exist_ok=True)
    stem = (
        f"occupancy_leaf{args.leaf}"
        if args.sweep == "occupancy"
        else f"{args.level}_{args.sweep}_n{args.per_device}_leaf{args.leaf}"
    )
    path = _RESULTS / f"{stem}{('_' + args.tag) if args.tag else ''}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"# wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

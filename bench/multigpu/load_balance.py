"""Figure 11 data: per-device work on the mesh lane, clustered vs uniform.

The mesh lane's partitions balance particle COUNT: ``decompose`` cuts Morton
ranges with the weighted-pivot ``sfc_repartition`` (#492: each device's samples
weigh its live count), and under migration (the production default) every
repartition cuts at exact global ranks. Neither weighs a particle by its work. On
a uniform cube equal counts are equal work by construction; on a centrally
concentrated distribution they need not be: the devices holding the dense centre
get more interaction pairs per particle. This bench measures how far apart the two
are on the configuration figures 08 and 09 time.

What it records, per device, from the force's own capacity census
------------------------------------------------------------------
The mesh force (``FusedMeshForce.shard`` -> ``mesh_force_shard``) returns each
device's ``needs`` vector before the segment runner ``pmax``-es it over the mesh.
This bench runs that force in its own ``shard_map`` with the needs sharded
``P(axis)`` instead, so every device's row comes back:

* ``count``: live particles (rows) on the device;
* ``far_pairs_local`` / ``near_pairs_local``: the local walk's far (cell) and
  near (leaf) pair lists, DIRECTED (twice the walk's symmetric pairs;
  ``NEEDS_FIELDS`` ``far_needed`` / ``near_needed``);
* ``far_pairs_cross`` / ``near_pairs_cross``: the cross-domain walk's far and near
  pairs on this device (``CROSS_NEEDS_FIELDS`` ``export_far_cap`` /
  ``export_near_cap``);
* ``particles_imported`` / ``particles_exported``, ``nodes_imported`` /
  ``nodes_exported``: what the cross exchange shipped (``recv_*`` / ``send_*``);
* the overflow ``bits`` and the lower-bound flags of each device.

These are LIST ENTRIES (cell pairs, leaf pairs), not particle-particle
interactions and not time: a near leaf pair costs ``n_i * n_j`` kernel
evaluations, and no per-device timer exists inside the fused program.
``per_device_work_total`` sums the four pair lists; it is a proxy, and the per-field
lists are there so the figure can show each. A census is ``valid`` only when every
device fitted its capacities and no count is a lower bound; pair counts are
deterministic, so a CPU census is a real census (at the N it ran), unlike a CPU
timing.

When
----
At the start force (after ``decompose`` and the runner's own repartition into the
force's box) and after each of ``--segments`` segments of ``--steps`` steps (drift,
migration, the rebalance before the next segment). The top-level fields of a
point are its LAST census; ``censuses`` holds all of them.

``--dtype`` (the paper benches' common flag) is accepted and IGNORED: the
galaxy preset sets the working precision, as in the scaling matrix.

Usage
-----
Every (distribution, device count) point runs in its own process::

    python -m bench.multigpu.load_balance --ndevs 2,4 --per-device-n 4000000 \\
        --distributions plummer,uniform --segments 2 --steps 4

CPU smoke (forced host devices, the bundle's CPU route)::

    JAX_PLATFORMS=cpu python -m bench.multigpu.load_balance --ndevs 2,4 \\
        --per-device-n 1024 --segments 0 --mac dehnen \\
        --json-out /tmp/load_balance_smoke.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.multigpu import _mesh_common as mc  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "multigpu/load_balance.json"
MODULE = "bench.multigpu.load_balance"

#: per-device census fields: (record name, needs-vector field)
CENSUS = (
    ("far_pairs_local", "far_needed"),
    ("near_pairs_local", "near_needed"),
    ("peak_wavefront_local", "peak_wavefront"),
    ("far_pairs_cross", "export_far_cap"),
    ("near_pairs_cross", "export_near_cap"),
    ("particles_imported", "recv_particle_cap"),
    ("particles_exported", "send_particle_cap"),
    ("nodes_imported", "recv_node_cap"),
    ("nodes_exported", "send_node_cap"),
    ("summary_cells", "max_cells"),
)
WORK_FIELDS = (
    "far_pairs_local",
    "near_pairs_local",
    "far_pairs_cross",
    "near_pairs_cross",
)
#: a nonzero value on any device means some count is not exact. The local walk's
#: ``lower_bound`` is only a property of the walk (the traced flat walk stops at its
#: first overflow; the Pallas walk counts on): its counts are inexact only together
#: with one of :data:`LOCAL_OVERFLOW_FLAGS` (``_interaction_cache``'s walk_needs).
INEXACT_FLAGS = (
    "tree_overflow",
    "walk_lower_bound",
    "pairs_lower_bound",
    "overflow",
    "mismatch",
)
LOCAL_OVERFLOW_FLAGS = ("far_overflow", "near_overflow", "queue_overflow")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="See the module docstring for what is (and is not) measured.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    runmeta.add_common_args(p)
    p.add_argument("--ndevs", default="2,4", help="device counts, one process each")
    p.add_argument(
        "--per-device-n",
        type=int,
        default=4_000_000,
        help="particles per device (default: the weak tier's 4e6 per card)",
    )
    p.add_argument(
        "--distributions",
        default="plummer,uniform",
        help="plummer, uniform, disc (comma-separated; the contrast is the point)",
    )
    p.add_argument("--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error")
    p.add_argument("--segments", type=int, default=2, help="censuses after the start")
    p.add_argument("--steps", type=int, default=4, help="steps per segment")
    p.add_argument(
        "--no-migrate",
        action="store_true",
        help="MeshSegmentRunner(migrate=False): the weighted-pivot sfc_repartition "
        "rebalances between segments instead of migration + exact-rank cuts",
    )
    p.add_argument("--disc-ic", default=None, help="the disc + bulge .npz (disc only)")
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--ndev", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--distribution", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


def _census(runner: Any, fail_mask: int, fields: tuple[str, ...]) -> dict[str, Any]:
    """Every device's needs and bits of the start force at the runner's state.

    The runner's own start program reduces both over the mesh; this one keeps them
    per device (``out_specs=P(axis)``). Same force, same plan and capacities
    (``force.context()``), same ``f_b`` input (``runner._scale_args``).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    force, state, axis = runner.force, runner.state, runner.axis_name
    scale = tuple(runner._scale_args(state))

    def body(st: Any, masses: Any, count: Any, ids: Any, *fb: Any) -> tuple:
        cap = int(st.shape[0])
        live = jnp.minimum(count[0], cap)
        out = force.shard(
            st[:, 0, :],
            masses,
            live,
            ids,
            **({"force_scale": fb[0]} if fb else {}),
        )
        bits, needs = out[1], out[2]
        return (
            jnp.asarray(bits, jnp.uint32).reshape(1),
            jnp.asarray(needs, jnp.int32).reshape(1, -1),
            jnp.asarray(count[:1], jnp.int32),
        )

    spec = P(axis)
    program = jax.jit(
        jax.shard_map(
            body,
            mesh=runner.mesh,
            in_specs=(spec,) * (4 + len(scale)),
            out_specs=(spec, spec, spec),
            check_vma=False,
        )
    )
    with force.context():
        force.before_trace()
        bits, needs, count = program(
            state.state, state.masses, state.count, state.ids, *scale
        )
    bits = np.asarray(jax.device_get(bits)).astype(np.int64)
    needs = np.asarray(jax.device_get(needs)).astype(np.int64)
    count = np.asarray(jax.device_get(count)).astype(np.int64)
    if needs.shape[1] != len(fields):
        raise RuntimeError(
            f"needs vector has {needs.shape[1]} entries, the library names "
            f"{len(fields)}: refusing to label them"
        )
    col = {name: needs[:, i] for i, name in enumerate(fields)}
    per_device: dict[str, list[int]] = {"count": count.tolist()}
    for out_name, field in CENSUS:
        if field in col:
            per_device[out_name] = col[field].tolist()
    inexact = {
        f: col[f].tolist() for f in INEXACT_FLAGS if f in col and bool(col[f].any())
    }
    local_over = np.zeros(len(count), bool)
    for f in LOCAL_OVERFLOW_FLAGS:
        local_over |= col[f] != 0
    if bool(((col["lower_bound"] != 0) & local_over).any()):
        inexact["lower_bound with a local walk overflow"] = (
            (col["lower_bound"] != 0) & local_over
        ).tolist()
    capacity_ok = bool(((bits & fail_mask) == 0).all())
    total = [sum(per_device[f][d] for f in WORK_FIELDS) for d in range(len(count))]

    def imbalance(vals: list[int]) -> Any:
        mean = sum(vals) / len(vals)
        return (max(vals) / mean) if mean > 0 else None

    reasons = []
    if not capacity_ok:
        reasons.append(f"capacity verdict failed (bits {sorted(set(bits.tolist()))})")
    if inexact:
        reasons.append(f"counts not exact: {sorted(inexact)}")
    return {
        "bits": bits.tolist(),
        "capacity_ok": capacity_ok,
        "inexact_flags": inexact,
        "per_device": per_device,
        "per_device_work_total": total,
        "work_imbalance": imbalance(total),
        "count_imbalance": imbalance(per_device["count"]),
        "imbalance_by_field": {
            f: imbalance(per_device[f]) for f in WORK_FIELDS if f in per_device
        },
        "needs_fields": list(fields),
        "needs_per_device": needs.tolist(),
        "valid": not reasons,
        "invalid_reasons": reasons,
    }


def _report(rep: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: getattr(rep, k) for k in keys if hasattr(rep, k)}


def _worker(args: argparse.Namespace) -> int:
    ndev = int(args.ndev)
    chosen = mc.worker_preamble(ndev, args.gpu_select)

    import jax

    jax.config.update("jax_enable_x64", True)

    from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS, cross_walk_backend
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS, OVERFLOW_FAIL_MASK

    fields = (*NEEDS_FIELDS, *CROSS_NEEDS_FIELDS)
    n = int(args.per_device_n) * ndev
    pos, vel, mass, g, soft, dt = mc.make_ic(
        args.distribution, n, int(args.seed), args.disc_ic
    )
    solver = mc.make_solver(args.mac, g, soft)
    t0 = time.perf_counter()
    runner, decomposition = mc.build_runner(
        solver, pos, vel, mass, ndev, migrate=not args.no_migrate
    )
    start = runner.start()
    jax.block_until_ready(runner.state.acceleration)
    keys = (
        "retries",
        "bits",
        "counts",
        "repartitioned",
        "declined",
        "sent_off_device",
        "migrated",
    )
    censuses = [
        {
            "when": "start",
            "segment": 0,
            "runner_report": _report(start, keys),
            **_census(runner, int(OVERFLOW_FAIL_MASK), fields),
        }
    ]
    for _ in range(int(args.segments)):
        rep = runner.run(int(args.steps), dt=dt)
        jax.block_until_ready(runner.state.state)
        censuses.append(
            {
                "when": f"after segment {rep.segment}",
                "segment": int(rep.segment),
                "runner_report": _report(rep, keys),
                **_census(runner, int(OVERFLOW_FAIL_MASK), fields),
            }
        )
    last = censuses[-1]
    reasons = [f"{c['when']}: {r}" for c in censuses for r in c["invalid_reasons"]]
    caps = runner.force.cross_caps
    record = {
        "distribution": args.distribution,
        "ndev": ndev,
        "n": n,
        "per_device_n": int(args.per_device_n),
        "mac": args.mac,
        "migrate": not args.no_migrate,
        "partition": (
            "decompose (weighted-pivot sfc_repartition) + migration with exact-rank "
            "rebalances"
            if not args.no_migrate
            else "decompose + weighted-pivot sfc_repartition before every segment"
        ),
        "segments": int(args.segments),
        "steps": int(args.steps),
        "dt": dt,
        "softening": soft,
        "seed": int(args.seed),
        "decomposition": decomposition,
        "cross_walk_backend": cross_walk_backend(),
        "leaf_capacity": int(runner.force.setup.leaf_capacity),
        "cross_caps_final": (
            None
            if caps is None
            else {
                k: v
                for k, v in vars(caps).items()
                if isinstance(v, (int, float)) or v is None
            }
        ),
        "censuses": censuses,
        # the figure's keys: the last census
        "per_device_work": {f: last["per_device"].get(f) for f in WORK_FIELDS},
        "per_device_count": last["per_device"]["count"],
        "per_device_work_total": last["per_device_work_total"],
        "work_imbalance": last["work_imbalance"],
        "count_imbalance": last["count_imbalance"],
        "valid": not reasons,
        "invalid_reasons": reasons,
        "wall_s": time.perf_counter() - t0,
        "cuda_visible_devices": chosen,
        **mc.platform_of_devices(),
        "cpu_route": (
            "bench/slurm/h0/smoke_site (H0_SMOKE_CPU=1)" if mc.on_cpu() else None
        ),
        "meta": runmeta.run_meta({"argv": sys.argv[1:]}),
    }
    print(json.dumps(record, default=_jsonable))
    return 0 if record["valid"] else 1


def _jsonable(x: Any) -> Any:
    import numpy as np

    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# --------------------------------------------------------------------------- #
# parent
# --------------------------------------------------------------------------- #


def main() -> int:
    """Sweep (distribution, device count), one process each; write the artifact.

    Returns
    -------
    int
        0 when at least one point is valid, else 1.
    """
    args = _parse_args()
    if args.worker:
        return _worker(args)
    ndevs = [int(v) for v in str(args.ndevs).split(",") if v.strip()]
    if any(d < 2 for d in ndevs):
        raise SystemExit("--ndevs must all be >= 2: one device has nothing to balance")
    dists = [d.strip() for d in str(args.distributions).split(",") if d.strip()]
    common = [
        "--per-device-n",
        str(args.per_device_n),
        "--mac",
        args.mac,
        "--segments",
        str(args.segments),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--gpu-select",
        args.gpu_select,
        "--dtype",
        args.dtype,
    ]
    if args.no_migrate:
        common.append("--no-migrate")
    if args.disc_ic:
        common += ["--disc-ic", args.disc_ic]
    points = [
        {"ndev": d, "distribution": dist, "argv": ["--distribution", dist]}
        for dist in dists
        for d in ndevs
    ]
    records = mc.run_points(MODULE, points, common)
    for r in records:
        flag = "" if r.get("valid") else f"  [INVALID: {r.get('invalid_reasons')}]"
        print(
            f"{r.get('distribution')} ndev={r.get('ndev')} "
            f"work_imbalance={r.get('work_imbalance')} "
            f"count_imbalance={r.get('count_imbalance')}{flag}"
        )
    worker_meta = next((r.get("meta") for r in records if r.get("meta")), {}) or {}
    platform = next((r.get("platform") for r in records if r.get("platform")), None)
    config = {
        "n": [r.get("n") for r in records],
        "theta": mc.THETA,
        "order": mc.MAX_ORDER,
        "basis": "real",
        "seed": args.seed,
        "device": worker_meta.get("device_kind", "unknown"),
        # positions and masses enter as float32 (the segment gate's ICs); the
        # runner does not record the preset's working dtype
        "precision": None,
        "ndevs": ndevs,
        "distributions": dists,
        "per_device_n_fixed": int(args.per_device_n),
        "leaf_size": mc.LEAF_SIZE,
        "preset": "galaxy",
        "mac": args.mac,
        "migrate": not args.no_migrate,
        "segments": args.segments,
        "steps": args.steps,
        "platform": platform,
        "work_fields": list(WORK_FIELDS),
        "work_is": "pair-list entries (cell / leaf pairs), not interactions or time",
        "n_valid": sum(1 for r in records if r.get("valid")),
        "n_invalid": sum(1 for r in records if not r.get("valid")),
    }
    meta = dict(runmeta.git_meta())
    meta.update(
        {
            "argv": sys.argv[1:],
            "sweep_parent": True,
            "parent_imported_jax": False,
            "worker_jax": worker_meta.get("jax_version"),
            "worker_device_kind": worker_meta.get("device_kind"),
            "measured_at": worker_meta.get("measured_at"),
        }
    )
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=meta,
        data={"records": records},
    )
    print(f"wrote {out}")
    return 0 if any(r.get("valid") for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

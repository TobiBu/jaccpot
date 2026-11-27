"""Multi-GPU gradient correctness on the fused mesh lane (WP-G) -- the numbers
behind the multi-GPU row of figure 13 and the differentiability paragraph.

The mesh counterpart of ``grad_correctness.py``. The force is the production
multi-GPU lane, ``FusedMeshForce`` (Morton domains placed by
``rollout.decompose``, cross far and near terms exchanged between devices), and
its gradient is ``jaccpot.distributed.grad.make_mesh_grad_fn`` (merged #510): the
per-device rebuild on stopped inputs, S8's differentiable evaluation at the live
positions and masses, the cross terms through their seams, the loss
``L = sum(w . acc)`` built OUTSIDE the ``shard_map``. It writes the values the
gates in ``tests/distributed/test_mesh_grad_gates.py`` only assert on (MG1, MG4,
MG5/6), so the manuscript can quote a number.

Records (``data.records``, one per ``check``)
---------------------------------------------
``forward_bit_identity``
    The gradient program's forward against the production force
    (``FusedMeshForce.shard``) on the live rows: elements whose bits differ
    (expected 0), plus the rel-L2 drift. Also both programs' verdict bits
    (``overflow_ok`` / ``grad_ok`` on every device) and the cross far / near
    pair counts per device: a mesh with no cross pairs would check only the
    single-card path. ``cross_engaged`` (cross far pairs on every device, cross
    NEAR pairs on at least one) and ``local_near_engaged`` (local near pairs on
    every device) are REQUIRED for ``valid``. ``topology_record_reproducible``
    and ``centre_checks`` re-read the centre's topology record (a second call;
    after the gradient; at ``x0 + 0 v`` and ``m0 (1 + 0 u)``): a stencil is only
    called switched against a centre that reproduces.
``fd_vs_ad`` (positions, masses)
    Central differences of the full pipeline (rebuild included) along one
    random direction (positions) and one relative direction ``m (1 + s u)``
    (masses), against the reverse. The topology is checked unchanged at
    ``+-h``: every device's local list counts (its needs vector) and its cross
    pairs' order-free hashes (the trace's ``CrossGradPlan``, read through a spy
    on ``CrossGradPlan.from_sink`` as the gate does), per ``--fd-steps``. A
    switched stencil is recorded (``topology_identical: false``, with
    ``topology_diff_plus`` / ``_minus``: which device's needs entries moved and
    which hash entries differ) and makes the run invalid: it is not an autodiff
    result.
``oracle``
    The mesh gradient in x and m, matched by global id, against the gradient of
    an exact fp64 direct sum with the solver's pair law
    (``grad_bench_lib.direct_loss_grad``), on a loss whose weights live on
    ``--check-targets`` seeded targets; next to the mesh FORCE's error at the
    same targets against the A4 reference (``jaccpot.verification.
    reference_force``, an exact-order fp64 direct sum, run on the sharded arrays).
``forward_survives_gradient``
    The production forward and the gradient program's forward after a gradient
    call, bitwise against their first calls. (Kept from the old ``distributed/
    fmm`` path, where a gradient once silently disabled the halo exchange.)

Configuration: ``preset="galaxy"`` (``dehnen_error`` 1e-4, p6, cell leaves of
64, ferrers3) by default, or ``--mac dehnen`` (geometric at ``--theta``,
``--order``); the leaf capacity fixed (LEAVES per tree: ``auto_leaf_capacity`` of
the global live cell leaves unless ``--leaf-capacity``), the cell-leaf floor
``--cell-min-level`` (``"auto"`` by default here, resolved on the host from all N:
the preset's pinned 8 gives ~1 particle per leaf at small N, no cross near pairs,
and -- measured at N = 2048, p4, theta 0.5, 2 devices -- a centre on a walk
decision boundary that every displacement down to 1e-10, in x or in m, crosses:
device 1's near list 18 -> 20 or 26 entries, FD off by 20% to 1e6x) and the cross
capacities
``CrossCapacities(max_cells=leaf capacity)`` (the CPU gates'); the far pairs kept.
fp64 by default (``--dtype``): the FD check needs it (fp32 FD bottoms out near
4e-4). Correctness only, no timing (figure 12's ``--mode mesh`` times the mesh).

CPU: the only route is the gates' -- forced host devices, the S9 interpret set,
``jax.default_backend`` reporting ``"gpu"``, the ``buf`` ragged exchange
(``grad_bench_lib``); interpret-mode programs compile for minutes.

Usage
-----
CPU smoke (2 emulated devices)::

    JAX_PLATFORMS=cpu python -m bench.differentiability.distributed_grad_correctness \\
        --ndev 2 --n 2048 --distribution cube --mac dehnen --theta 0.8 --order 2 \\
        --check-targets 256 --gpu-select none --json-out /tmp/dist_grad_smoke.json

Paper run::

    python -m bench.differentiability.distributed_grad_correctness --ndev 4 --n 65536
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.differentiability import grad_bench_lib as L  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "differentiability/distributed_grad_correctness.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ndev", type=int, default=2, help="device count (mesh size)")
    p.add_argument("--n", type=int, default=4096, help="total particles")
    p.add_argument(
        "--distribution",
        choices=("cube", "plummer"),
        default="cube",
        help="cube: the gates' uniform cube, masses U(0.5, 1.5) / N; plummer: "
        "bench.validation._harness",
    )
    p.add_argument("--softening", type=float, default=1e-2)
    p.add_argument("--mac", choices=("preset", "dehnen"), default="preset")
    p.add_argument("--theta", type=float, default=None, help="with --mac dehnen")
    p.add_argument("--order", type=int, default=None, help="with --mac dehnen")
    p.add_argument("--leaf-capacity", type=int, default=None)
    p.add_argument(
        "--cell-min-level",
        default="auto",
        help="the cell-leaf tree's level floor: 'auto' (default here; resolved on "
        "the host from all N in the global box, auto_cell_min_level) or an int. "
        "The preset pins 8, which at small N gives ~1 particle per leaf: nearly "
        "every pair is far and the cross NEAR term is never exercised. Recorded "
        "in off_preset",
    )
    p.add_argument("--cap-factor", type=float, default=1.3, help="rows / (N / ndev)")
    p.add_argument(
        "--fd-steps",
        default="1e-6",
        help="comma-separated central-difference steps, one fd_vs_ad record "
        "each per target (the gates' MG4: 1e-6 in fp64); every one must keep "
        "the topology pinned for the run to be valid",
    )
    p.add_argument("--check-targets", type=int, default=2048)
    runmeta.add_common_args(p)
    args = p.parse_args()
    if args.ndev < 2:
        p.error(
            "--ndev must be >= 2: a one-device mesh is not the multi-GPU "
            "measurement (grad_correctness.py is the single card)"
        )
    if args.mac == "preset" and (args.theta is not None or args.order is not None):
        p.error("--theta / --order move the geometric criterion: --mac dehnen")
    args.fd_step_list = [float(v) for v in args.fd_steps.split(",") if v.strip()]
    if args.cell_min_level != "auto":
        args.cell_min_level = int(args.cell_min_level)
    return args


def _ic(args: argparse.Namespace, seed: int) -> tuple[Any, Any]:
    import numpy as np

    if args.distribution == "cube":
        rng = np.random.default_rng(seed)
        pos = rng.uniform(-1.0, 1.0, (args.n, 3))
        return pos, rng.uniform(0.5, 1.5, args.n) / args.n
    from bench.validation import _harness as H

    return H.make_distribution("plummer", args.n, seed)


def _global_tree_sizes(fmm: Any, pos: Any, dtype: Any, cell_min_level: Any) -> tuple:
    """``(cell_min_level, leaf capacity)`` from all N in the global box.

    ``"auto"`` is resolved here, on the host, with ``auto_cell_min_level`` (the
    engine would pick it at its first eager prepare; the mesh needs one level
    for every device). The leaf capacity is ``auto_leaf_capacity`` of all N's
    live cell leaves at that level: a device's particles are a subset, so its
    cell leaves in the same box are no more, and one capacity covers every
    device.
    """
    import jax.numpy as jnp

    from bench.scaling import _galaxy
    from jaccpot.distributed.fused import host_mesh_bounds
    from jaccpot.runtime.capacity_plan import (
        AUTO_CELL_MIN_LEVEL_MAX,
        auto_cell_min_level,
        auto_leaf_capacity,
        live_cell_leaves,
    )

    engine = fmm._impl
    leaf = int(_galaxy.GALAXY.leaf_size)
    box = host_mesh_bounds([pos], [len(pos)], cubic=bool(engine._tree_cubic_bounds))
    x = jnp.asarray(pos, dtype)
    level = cell_min_level
    if level == "auto":
        level = int(
            auto_cell_min_level(
                x, leaf_size=leaf, bounds=box, max_level=AUTO_CELL_MIN_LEVEL_MAX
            )
        )
    live = live_cell_leaves(x, leaf_size=leaf, cell_min_level=int(level), bounds=box)
    return int(level), int(auto_leaf_capacity(live, 1.15)), int(live)


def _topology_diff(out: Any, hashes0: Any, needs0: Any, fields: list) -> dict:
    """Where a displaced run's topology record differs from the centre's.

    ``needs``: every (device, field) whose value moved; ``hashes``: per device,
    how many cross pair-hash entries differ and the first few indices.
    """
    import numpy as np

    hashes, needs = np.asarray(out[1][1]), np.asarray(out[1][2])
    moved = []
    for d, j in zip(*np.nonzero(needs != needs0)):
        name = fields[j] if j < len(fields) else f"needs[{j}]"
        moved.append([int(d), name, int(needs0[d, j]), int(needs[d, j])])
    hdiff = {}
    for d in range(hashes.shape[0]):
        idx = np.nonzero(hashes[d] != hashes0[d])[0]
        if idx.size:
            hdiff[int(d)] = {"count": int(idx.size), "first": idx[:8].tolist()}
    return {"needs": moved, "hashes": hdiff}


def main() -> int:  # noqa: C901 - one linear measurement script
    """Measure the mesh gradient's correctness and write the artifact."""
    args = _parse_args()
    # select_gpu claims the whole mesh at once; the env before the jax import
    runmeta.select_gpu(args.gpu_select, num_gpus=int(args.ndev))
    env_set = L.prepare_env(host_devices=int(args.ndev))
    import jax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    L.patch_cpu_lane(mesh=True)
    import jax.numpy as jnp
    import numpy as np
    import yggdrax.distributed.comm as ygg_comm
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from bench.scaling import _galaxy
    from jaccpot.distributed import grad as GR
    from jaccpot.distributed.cross import CrossCapacities
    from jaccpot.distributed.fused import AXIS_NAME
    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.template_force import FusedMeshForce
    from jaccpot.runtime.fused_topology import GRAD_FAIL_MASK, OVERFLOW_FAIL_MASK
    from jaccpot.verification import force_law_of, reference_force, select_targets

    ndev = int(args.ndev)
    if len(jax.devices()) < ndev:
        raise SystemExit(
            f"need {ndev} devices, JAX sees {len(jax.devices())}: a distributed "
            "measurement on another device count is another measurement"
        )
    dt = L.jnp_dtype(args.dtype)
    seed = runmeta.seed_sequence(args.seed, args.distribution, str(args.n))
    pos, mass = _ic(args, seed)
    n = int(args.n)
    mac_type = None if args.mac == "preset" else "dehnen"
    probe_fmm, _ = L.solver(softening=args.softening, dtype=args.dtype)
    level, auto_cap, live_leaves = _global_tree_sizes(
        probe_fmm, pos, dt, args.cell_min_level
    )
    leaf_cap = args.leaf_capacity or auto_cap
    print(
        f"cell_min_level {args.cell_min_level} -> {level}; {live_leaves} live cell "
        f"leaves over all N; leaf capacity {leaf_cap}",
        flush=True,
    )
    fmm, scfg = L.solver(
        softening=args.softening,
        dtype=args.dtype,
        mac_type=mac_type,
        theta=args.theta,
        order=args.order,
        leaf_capacity=leaf_cap,
        cell_min_level=level,
    )
    scfg["off_preset"]["cell_min_level"] = {
        "requested": args.cell_min_level,
        "resolved": level,
    }
    leaf = int(_galaxy.GALAXY.leaf_size)
    order = int(scfg["max_order"])

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS_NAME,))
    spec = P(AXIS_NAME)
    sh = NamedSharding(mesh, spec)
    cap = int(np.ceil(n / ndev * args.cap_factor))
    shard, _ = decompose(
        mesh,
        pos.astype(np.dtype(args.dtype)),
        np.zeros_like(pos, dtype=np.dtype(args.dtype)),
        mass.astype(np.dtype(args.dtype)),
        cap=cap,
        num_samples=64,
    )
    force = FusedMeshForce(
        fmm,
        mesh,
        leaf_size=leaf,
        max_order=order,
        theta=scfg["theta"],
        cross_caps=CrossCapacities(max_cells=int(leaf_cap), leaf_width=leaf),
    )
    force.prepare(shard.positions, shard.masses, shard.count)
    force.with_grad_plan(shard.positions, shard.masses, shard.count)
    seed_fb = force.force_scale_seed if force.scale_carry else None
    fb = () if seed_fb is None else (seed_fb,)

    ids = np.asarray(shard.ids)
    counts = np.asarray(shard.count).astype(np.int64)
    rows = np.concatenate(
        [np.arange(d * cap, d * cap + counts[d]) for d in range(ndev)]
    )
    gid = ids[rows]
    assert sorted(gid.tolist()) == list(range(n)), "decompose lost or duplicated ids"

    def by_id(g: Any) -> Any:
        g = np.asarray(g, np.float64)
        out = np.zeros((n,) + g.shape[1:])
        out[gid] = g[rows]
        return out

    def mesh_layout(a: Any) -> Any:
        """A global-id-ordered host array in the mesh's padded row layout."""
        out = np.zeros((ndev * cap,) + a.shape[1:], np.dtype(args.dtype))
        out[rows] = a[gid]
        return jax.device_put(jnp.asarray(out), sh)

    w_np = L.weights(n, seed)
    w_all = mesh_layout(w_np)

    # the cross pairs' hashes of THIS trace: the gate's spy on the plan's builder
    captured: dict = {}
    real_from_sink = GR.CrossGradPlan.from_sink.__func__

    def spy(cls, sink):
        plan = real_from_sink(cls, sink)
        captured["plan"] = plan
        return plan

    fn = GR.make_mesh_grad_fn(force)

    def prod_body(x, m, cnt, i, *f):
        out = force.shard(x, m, jnp.minimum(cnt[0], x.shape[0]), i, *f)
        return out[0], out[1].reshape(1)

    prod = jax.jit(
        jax.shard_map(
            prod_body,
            mesh=mesh,
            in_specs=(spec,) * (4 + len(fb)),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )

    def probe_body(x, m, n_, i, *f):
        out = force.differentiable_shard(x, m, jnp.minimum(n_[0], x.shape[0]), i, *f)
        plan = captured["plan"]
        far = jnp.sum(plan.far_sources >= 0)
        live = jnp.arange(plan.near_targets.shape[0]) < plan.near_count
        near = jnp.sum(live & (plan.near_targets >= 0))
        return (
            out[0],
            out[1].reshape(1),
            far.reshape(1),
            near.reshape(1),
            plan.pair_hashes.reshape(1, -1),
            out[2].reshape(1, -1),
        )

    probe = jax.shard_map(
        probe_body,
        mesh=mesh,
        in_specs=(spec,) * (4 + len(fb)),
        out_specs=(spec,) * 6,
        check_vma=False,
    )

    def value(x, m, w):
        acc, bits, far, near, hashes, needs = probe(x, m, shard.count, shard.ids, *fb)
        return jnp.sum(acc * w), (bits, hashes, needs, far, near)

    def loss(x, m, w):
        out = fn(x, m, shard.count, shard.ids, None, seed_fb)
        return jnp.sum(out[0] * w), out[1]

    def diff_fwd(x, m):
        return fn(x, m, shard.count, shard.ids, None, seed_fb)[:2]

    x0, m0 = shard.positions, shard.masses
    pargs = (x0, m0, shard.count, shard.ids, *fb)
    records: list[dict] = []
    GR.CrossGradPlan.from_sink = classmethod(spy)
    try:
        with force.context():
            force.before_trace()
            acc_p, bits_p = prod(*pargs)
            force.before_trace()
            diff_j = jax.jit(diff_fwd)
            acc_d, bits_d = diff_j(x0, m0)
            force.before_trace()
            value_j = jax.jit(value)
            _, (_, hashes0, needs0, far_d, near_d) = value_j(x0, m0, w_all)
            far_d, near_d = np.asarray(far_d), np.asarray(near_d)
            # is the topology record itself reproducible? (same program, same input)
            _, (_, hashes_r, needs_r, _, _) = value_j(x0, m0, w_all)
            force.before_trace()
            vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))
            (l0, gbits), (gx, gm) = vg(x0, m0, w_all)
            gx, gm = np.asarray(gx, np.float64), np.asarray(gm, np.float64)
            hashes0, needs0 = np.asarray(hashes0), np.asarray(needs0)
            repeat_ok = bool(
                np.array_equal(np.asarray(hashes_r), hashes0)
                and np.array_equal(np.asarray(needs_r), needs0)
            )
            from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
            from jaccpot.runtime.fused_topology import NEEDS_FIELDS

            fields = list(NEEDS_FIELDS) + list(CROSS_NEEDS_FIELDS)
            local_near = needs0[:, list(NEEDS_FIELDS).index("near_needed")]

            pb = np.asarray(bits_p).astype(np.uint64)
            gb = np.asarray(gbits).astype(np.uint64)
            force_ok = bool(np.all((pb & OVERFLOW_FAIL_MASK) == 0))
            grad_ok = bool(np.all((gb & GRAD_FAIL_MASK) == 0))
            finite = bool(np.all(np.isfinite(gx)) and np.all(np.isfinite(gm)))
            a_p = np.asarray(acc_p)[rows]
            a_d = np.asarray(acc_d)[rows]
            records.append(
                {
                    "check": "forward_bit_identity",
                    "bit_identical": bool(L.differing(a_d, a_p) == 0),
                    "differing_elements": L.differing(a_d, a_p),
                    "elements": int(a_p.size),
                    "rel_l2_drift": L.rel_l2(a_d, a_p),
                    "force_bits": pb.tolist(),
                    "grad_bits": gb.tolist(),
                    "force_ok": force_ok,
                    "grad_ok": grad_ok,
                    "grad_finite": finite,
                    "cross_far_pairs": far_d.reshape(-1).tolist(),
                    "cross_near_pairs": near_d.reshape(-1).tolist(),
                    # cross far on every device, cross near on at least one
                    "cross_engaged": bool(np.all(far_d > 0) and np.any(near_d > 0)),
                    "local_near_needed": local_near.tolist(),
                    "local_near_engaged": bool(np.all(local_near > 0)),
                    "topology_record_reproducible": repeat_ok,
                }
            )
            print(records[-1], flush=True)

            # MG4: the full pipeline's central differences, topology pinned
            rng = np.random.default_rng(seed + 100)
            v = np.zeros((ndev * cap, 3))
            v[rows] = rng.normal(size=(rows.size, 3))
            u = np.zeros(ndev * cap)
            u[rows] = rng.normal(size=rows.size)
            vj = jax.device_put(jnp.asarray(v, dt), sh)
            uj = jax.device_put(jnp.asarray(u, dt), sh)
            # the centre's record again, after the gradient's trace and call, and
            # at a fresh array equal to x0 (x0 + 0 v): a stencil can only be
            # called switched against a centre that is itself reproducible
            centre_checks = {
                "after_gradient": _topology_diff(
                    value_j(x0, m0, w_all), hashes0, needs0, fields
                ),
                "x0_plus_zero": _topology_diff(
                    value_j(x0 + 0.0 * vj, m0, w_all), hashes0, needs0, fields
                ),
                "m0_times_one": _topology_diff(
                    value_j(x0, m0 * (1.0 + 0.0 * uj), w_all), hashes0, needs0, fields
                ),
            }
            records[0]["centre_checks"] = centre_checks
            print({"centre_checks": centre_checks}, flush=True)
            fd_cases = [
                (h, wrt, move, analytic)
                for h in args.fd_step_list
                for wrt, move, analytic in (
                    ("positions", lambda s: (x0 + s * vj, m0), float(np.sum(gx * v))),
                    (
                        "masses",
                        lambda s: (x0, m0 * (1 + s * uj)),
                        float(np.sum(gm * np.asarray(m0, np.float64) * u)),
                    ),
                )
            ]
            for h, wrt, move, analytic in fd_cases:
                outs = [value_j(*move(s), w_all) for s in (h, -h)]
                diffs = [_topology_diff(o, hashes0, needs0, fields) for o in outs]
                same = all(not d["needs"] and not d["hashes"] for d in diffs)
                numeric = (float(outs[0][0]) - float(outs[1][0])) / (2 * h)
                records.append(
                    {
                        "check": "fd_vs_ad",
                        "wrt": wrt,
                        "direction": "random" if wrt == "positions" else "m (1 + s u)",
                        "fd": numeric,
                        "ad": analytic,
                        "rel_err": abs(numeric - analytic) / (abs(numeric) + 1e-300),
                        "fd_step": h,
                        "topology_identical": bool(same),
                        "topology_diff_plus": diffs[0],
                        "topology_diff_minus": diffs[1],
                        "grad_norm": float(
                            np.linalg.norm(gx if wrt == "positions" else gm)
                        ),
                        "all_finite": finite,
                    }
                )
                print(records[-1], flush=True)

            # MG5/6: against the exact fp64 direct sum on seeded targets
            law = force_law_of(fmm)
            targets = select_targets(n, min(args.check_targets, n), seed=seed)
            w_t = np.zeros_like(w_np)
            w_t[targets] = w_np[targets]
            force.before_trace()
            _, (tx, tm) = vg(x0, m0, mesh_layout(w_t))
            direct = L.direct_loss_grad(pos, mass, w_t, law, targets=targets)
            ref = reference_force(
                shard.positions,
                shard.masses,
                shard.ids,
                targets,
                law,
                mesh=mesh,
                axis_name=AXIS_NAME,
            )
            mesh_acc = by_id(acc_p)[targets]
            records.append(
                {
                    "check": "oracle",
                    "targets": int(len(targets)),
                    "grad_rel_l2_positions": L.rel_l2(by_id(tx), direct["grad_x"]),
                    "grad_rel_l2_masses": L.rel_l2(by_id(tm), direct["grad_m"]),
                    # the reference the gradient errors should be read against
                    "forward_force_rel_l2": L.rel_l2(mesh_acc, ref.acceleration),
                    "direct_vs_reference_force": L.rel_l2(
                        direct["acc"], ref.acceleration
                    ),
                    "reference": "jaccpot.verification.reference_force (A4)",
                    "law": law.as_dict(),
                }
            )
            print(records[-1], flush=True)

            # a gradient must not change a later forward
            force.before_trace()
            after_p = np.asarray(prod(*pargs)[0])[rows]
            after_d = np.asarray(diff_j(x0, m0)[0])[rows]
            records.append(
                {
                    "check": "forward_survives_gradient",
                    "forward_bit_identical": bool(L.differing(after_p, a_p) == 0),
                    "grad_path_bit_identical": bool(L.differing(after_d, a_d) == 0),
                    "forward_rel_l2_drift": L.rel_l2(after_p, a_p),
                    "grad_path_rel_l2_drift": L.rel_l2(after_d, a_d),
                }
            )
            print(records[-1], flush=True)
    finally:
        GR.CrossGradPlan.from_sink = classmethod(real_from_sink)

    engaged = records[0]["cross_engaged"] and records[0]["local_near_engaged"]
    ok = (
        force_ok
        and grad_ok
        and finite
        and engaged
        and records[0]["topology_record_reproducible"]
        and all(r.get("topology_identical", True) for r in records)
    )
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config={
            "n": n,
            "theta": scfg["theta"],
            "order": order,
            "basis": "real",
            "seed": int(args.seed),
            "device": runmeta.device_label(),
            "precision": args.dtype,
            "lane": "fused_mesh",
            "solver": scfg,
            "mac_type": scfg["mac_type"],
            "ndev": ndev,
            "per_device_n": [int(c) for c in counts],
            "cap": cap,
            "leaf_size": leaf,
            "leaf_capacity": int(leaf_cap),
            "cross_caps": "CrossCapacities(max_cells=leaf_capacity, leaf_width=leaf)",
            "ragged_exchange": ygg_comm.resolve_ragged_method("auto"),
            "distribution": args.distribution,
            "softening": float(args.softening),
            "fd_steps": args.fd_step_list,
            "cell_min_level": level,
            "check_targets": int(min(args.check_targets, n)),
            "valid": bool(ok),
            "cpu_route_env": env_set or None,
        },
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records},
    )
    print(f"wrote {out} (valid={ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

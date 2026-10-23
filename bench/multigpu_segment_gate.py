"""S11 track A on cards: the mesh segment runner against the stacked route and one card.

Three modes, each in its OWN process (the peak-memory statistic is a per-process
high-water mark, and XLA reads its flags once):

* ``--mode segment`` (2 cards): ``decompose`` + ``MeshSegmentRunner.fused`` (the
  template route: caps-only prepares, one shape template, one program per segment)
  for ``--segments`` segments of ``--steps`` steps. Reports the flags, the retries,
  the rel-L2 of the start force and of the carried acceleration at the end against
  an fp64 direct sum on 512 fixed ids, the per-device peak bytes per particle, and
  the segment wall times; dumps the step-0 and final positions by id.
* ``--mode stacked`` (2 cards): ``setup_fused_force`` + ``FusedRollout`` (the old
  route: full eager prepares, stacked prepared states, 3-4 programs and a host sync
  per step) for the same steps, at the leaf capacity the segment mode sized: the
  peak bytes per particle and the step times of the reference route.
* ``--mode solo`` (1 card): the single-card fused lane (``FusedForceProvider``) on
  the positions the segment mode dumped, scored on the same ids.

The lists are the LIBRARY's (``preset="galaxy"``: ``leaf_capacity="auto"``, no
fast-lane environment, no Odisseo harness caps), the cross capacities too
(``cross_caps="auto"``: sized from the start force's measured needs, on both
routes). ``--mac`` picks the MAC: ``dehnen_error`` (the default since S12: the
preset's own, eq (16a) at ``adaptive_eps = 1e-4``, f_b carried on the mesh and
seeded per device; segment and solo modes) or ``dehnen`` (the geometric theta 0.8
the S11 rows ran; the only one the stacked route takes). ``--cross formula`` takes
``bench/multigpu_rollout_gate.py``'s formulas instead (x ``--cross-scale``), with
``max_cells`` the leaf capacity, as the S11 track-A rows were measured. The solo mode
also times the single card's force (``prov.step``, jitted; the median of
``--force-repeats``), next to the segment mode's ``force_only_ms``.

    eval $(autocvd -q -l -n 2 -e -t 120)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/multigpu_segment_gate.py \\
        --mode segment --ic plummer --n 200000 --out OUT.json --dump DUMPDIR

Not a timing gate unless both cards were free (the JSON records ``nvidia-smi``
before and after).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402

_FUSION = "--xla_gpu_experimental_enable_fusion_autotuner=false"
if _FUSION.split("=")[0] not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _FUSION).strip()
add_ragged_exchange_flags()

# the Odisseo harness's ICs and fp64 reference only (common/); BENCH_DIR elsewhere
sys.path.insert(
    0,
    os.environ.get(
        "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
    ),
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

_DISC_IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("segment", "stacked", "solo"), required=True)
    p.add_argument("--ic", choices=("plummer", "disc"), default="plummer")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--steps", type=int, default=8, help="steps per segment")
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--repartition-every", type=int, default=1)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dump", required=True, help="positions by id, segment -> solo")
    p.add_argument(
        "--leaf-capacity", type=int, default=None, help="stacked: from the segment run"
    )
    p.add_argument(
        "--cross",
        choices=("library", "formula"),
        default="library",
        help="cross capacities: the library's (cross_caps='auto', the default) or "
        "bench/multigpu_rollout_gate.py's formulas (the S11 track-A runs)",
    )
    p.add_argument(
        "--cross-json",
        default=None,
        help="explicit cross capacities: a segment run's JSON (its cross_caps_final), "
        "so the process's peak holds no sizing measurement",
    )
    p.add_argument(
        "--cross-scale",
        type=float,
        default=1.0,
        help="multiplies the gate formulas' leaves per device (the disc needs more)",
    )
    p.add_argument(
        "--mac",
        choices=("dehnen_error", "dehnen"),
        default="dehnen_error",
        help="the galaxy preset's eq (16a) (default) or the geometric theta 0.8",
    )
    p.add_argument("--force-repeats", type=int, default=5)
    p.add_argument(
        "--migrate",
        action="store_true",
        help="segment mode: migrate to fixed cuts after every drift "
        "(MeshSegmentRunner(migrate=True), jaccpot.distributed.migration)",
    )
    p.add_argument(
        "--migration-cap",
        default="auto",
        help="--migrate: the buffer's rows ('auto', an int, or 'json' to take "
        "--cross-json's migration_cap_final)",
    )
    return p.parse_args()


ARGS = _args()
NDEV = 1 if ARGS.mode == "solo" else 2
CAP = int(ARGS.n / NDEV * 1.15) if NDEV > 1 else ARGS.n


def _smi() -> str:
    try:
        return subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"nvidia-smi failed: {exc!r}"


def _plummer_velocities(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Isotropic equilibrium speeds for a G = M = a = 1 Plummer sphere (AHW 1974)."""
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


def _ic() -> tuple:
    """``(pos, vel, mass, G, softening, dt)`` as float32 arrays and floats."""
    if ARGS.ic == "plummer":
        from common.ic import IC_GENERATORS

        pos, mass = IC_GENERATORS["plummer"](ARGS.n, seed=ARGS.seed)
        pos = np.asarray(pos, np.float64)
        vel = _plummer_velocities(pos, np.random.default_rng(7))
        dt = 5e-3 if ARGS.dt is None else ARGS.dt
        return (
            pos.astype(np.float32),
            vel.astype(np.float32),
            np.asarray(mass, np.float32),
            1.0,
            1e-7,
            dt,
        )
    ic = np.load(_DISC_IC)
    n_all = int(ic["state0"].shape[0])
    state = np.asarray(ic["state0"][: ARGS.n], np.float32)
    # the IC is shuffled: a prefix is a random subsample; rescaled to the same galaxy
    mass = np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n)
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(ARGS.n / 1e5))
    dt = 5e-4 if ARGS.dt is None else ARGS.dt
    return state[:, 0], state[:, 1], mass.astype(np.float32), g, soft, dt


def _solver(g: float, soft: float, leaf_capacity=None):
    from jaccpot import FastMultipoleMethod

    advanced = FastMultipoleMethod(preset="galaxy").advanced
    if ARGS.mac == "dehnen":
        # docs/production_2026-10.md, "The galaxy preset on dehnen_error": the
        # geometric MAC back
        advanced = dataclasses.replace(advanced, mac_type="dehnen")
    if leaf_capacity is not None:
        advanced = dataclasses.replace(
            advanced,
            tree=dataclasses.replace(
                advanced.tree, leaf_capacity=int(leaf_capacity), leaf_headroom=None
            ),
        )
    solver = FastMultipoleMethod(
        preset="galaxy", G=g, softening=soft, advanced=advanced
    )
    engine = solver._impl
    want = ARGS.mac == "dehnen_error"
    assert bool(engine._flat_walk_criterion_active()) == want, (
        ARGS.mac,
        engine.mac_type,
    )
    return solver


def _cross_caps(leaves_per_device: float, max_cells: int):
    """``bench/multigpu_rollout_gate.py``'s formulas, ``max_cells`` the leaf cap."""
    from jaccpot.distributed.cross import CrossCapacities

    per = float(leaves_per_device) * float(ARGS.cross_scale)

    def pow2(x: float, floor: int) -> int:
        return 1 << max(floor, int(np.ceil(np.log2(max(x, 1)))))

    return CrossCapacities(
        max_cells=int(max_cells),
        export_far_cap=pow2(2.5 * 53 * per, 21),
        export_near_cap=pow2(2.5 * 10 * per, 21),
        send_node_cap=pow2(2.5 * 1.25 * per, 15),
        send_csr_cap=pow2(2.5 * 53 * per, 21),
        recv_node_cap=pow2(2.5 * 1.25 * per, 15),
        recv_csr_cap=pow2(2.5 * 53 * per, 19),
        walk_queue=2 * pow2(2.5 * 53 * per, 19),
        recv_far_cap=pow2(2.5 * 106 * per, 21),
        recv_near_cap=pow2(2.5 * 60 * per, 21),
        leaf_width=64,
    )


def _direct_fp64(pos, mass, idx, g: float, soft: float, kernel: str) -> np.ndarray:
    """fp64 direct sum at the targets ``idx`` through the solver's pair kernel
    (``bench/fused_memory_budget.py``'s ``_direct_kernel_fp64``)."""
    from jaccpot.softening import pair_factors, softening_params

    p = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)
    n = int(p.shape[0])
    chunk = 1 << 17
    npad = -(-n // chunk) * chunk
    sp = jnp.zeros((npad, 3), jnp.float64).at[:n].set(p).reshape(-1, chunk, 3)
    sm = jnp.zeros((npad,), jnp.float64).at[:n].set(m).reshape(-1, chunk)
    sid = jnp.arange(npad).reshape(-1, chunk)
    params = softening_params(kernel, soft, jnp.float64)

    @jax.jit
    def block(tp, tid):
        def body(c, acc):
            d = tp[:, None, :] - sp[c][None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            f = pair_factors(jnp.where(r2 > 0, r2, 1.0), params, kernel)[0]
            w = jnp.where(tid[:, None] == sid[c][None, :], 0.0, f * sm[c][None, :])
            return acc - jnp.einsum("ij,ijk->ik", w, d)

        return jax.lax.fori_loop(0, sp.shape[0], body, jnp.zeros_like(tp))

    out = []
    for b in range(0, len(idx), 128):
        ib = jnp.asarray(idx[b : b + 128])
        out.append(np.asarray(block(p[ib], ib)))
    return g * np.concatenate(out)


def _rel_l2(acc_by_id, pos_by_id, mass, g, soft, kernel, ids) -> float:
    ref = _direct_fp64(pos_by_id, mass, ids, g, soft, kernel)
    got = np.asarray(acc_by_id, np.float64)[ids]
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def _peaks() -> list:
    out = []
    for d in jax.local_devices():
        stats = d.memory_stats() or {}
        out.append(int(stats.get("peak_bytes_in_use", 0)))
    return out


def _checkpoint(result: dict, tag: str) -> None:
    """Per-device peak and live bytes now: the peak is a process high-water mark,
    so the phase that raised it is the first checkpoint where it grew."""
    stats = [d.memory_stats() or {} for d in jax.local_devices()]
    result.setdefault("memory", {})[tag] = dict(
        peak=[int(st.get("peak_bytes_in_use", 0)) for st in stats],
        in_use=[int(st.get("bytes_in_use", 0)) for st in stats],
    )


def _leaf_capacity_like_the_mesh(solver, shard) -> tuple[int, float]:
    """``prepare_mesh_force``'s auto capacity (its eager count), and leaves / device."""
    from jaccpot.distributed.fused import host_mesh_bounds
    from jaccpot.distributed.rollout import _mesh_devices, _pad_onto_live_row
    from jaccpot.distributed.template_force import _by_device
    from jaccpot.runtime.capacity_plan import auto_leaf_capacity, live_cell_leaves

    engine = solver._impl
    devices = _mesh_devices(MESH)
    counts = [int(c) for c in np.asarray(shard.count)]
    pos_d = [
        _pad_onto_live_row(jnp.asarray(p), jnp.arange(CAP) < counts[d])
        for d, p in enumerate(_by_device(shard.positions, devices))
    ]
    box = host_mesh_bounds(
        [np.asarray(p) for p in pos_d], counts, cubic=bool(engine._tree_cubic_bounds)
    )
    live = [
        live_cell_leaves(
            jnp.asarray(p, engine.working_dtype),
            leaf_size=64,
            cell_min_level=int(engine._tree_cell_min_level),
            bounds=box,
        )
        for p in pos_d
    ]
    return auto_leaf_capacity(max(live), 1.15), float(np.mean(live))


def main() -> None:
    from jaccpot.distributed.rollout import decompose

    devices = jax.devices()
    assert devices[0].platform == "gpu", devices
    assert len(devices) >= NDEV, devices
    smi_before = _smi()
    pos0, vel0, mass0, g, soft, dt = _ic()
    rng = np.random.default_rng(12345)
    ids = np.sort(rng.choice(ARGS.n, 512, replace=False))
    os.makedirs(ARGS.dump, exist_ok=True)
    result: dict = dict(
        mode=ARGS.mode,
        mac=ARGS.mac,
        ic=ARGS.ic,
        n=ARGS.n,
        ndev=NDEV,
        cap=CAP,
        steps=ARGS.steps,
        segments=ARGS.segments,
        cross_scale=ARGS.cross_scale,
        dt=dt,
        softening=soft,
        G=g,
        devices=[str(d) for d in devices[:NDEV]],
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        smi_before=smi_before,
    )

    if ARGS.mode == "solo":
        from jaccpot import FusedForceProvider

        solver = _solver(g, soft)
        kernel = solver._impl.softening_kernel
        for tag in ("start", "end"):
            f = os.path.join(ARGS.dump, f"positions_{tag}.npy")
            pos = jnp.asarray(np.load(f).astype(np.float32))
            m = jnp.asarray(mass0)
            prov, topology = FusedForceProvider.prepare(
                solver,
                pos,
                m,
                leaf_size=64,
                max_order=6,
                theta=0.8,
                caps_only=True,
            )
            aux = None
            if prov.scale_carry:
                # the caps-only prepare's eq (16b) seed, in input order (as the
                # particle carry's start force reads it)
                perm = jnp.asarray(topology.prepared.stub.tree.particle_indices)
                fb_sorted = jnp.asarray(solver._impl._last_force_scale_particles)
                seed = jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted)
                aux = (seed, None, None)
            del topology
            step = jax.jit(prov.step)
            acc, (_, bits, _) = step(pos, m, None, aux)
            jax.block_until_ready(acc)
            force_s = []
            for _ in range(int(ARGS.force_repeats)):
                t0 = time.perf_counter()
                jax.block_until_ready(step(pos, m, None, aux)[0])
                force_s.append(time.perf_counter() - t0)
            from jaccpot.runtime.fused_topology import overflow_ok

            result[f"solo_{tag}"] = dict(
                ok=bool(overflow_ok(bits)),
                rel_l2=_rel_l2(
                    np.asarray(acc), np.asarray(pos), mass0, g, soft, kernel, ids
                ),
                leaf_capacity=int(solver._impl._tree_leaf_capacity),
                force_ms=[1e3 * x for x in force_s],
            )
            print(tag, result[f"solo_{tag}"], flush=True)
            del prov, step, acc, aux
        result["peak_bytes"] = _peaks()
        result["smi_after"] = _smi()
        json.dump(result, open(ARGS.out, "w"), indent=1)
        return

    from yggdrax.distributed.sharding import make_mesh

    global MESH
    MESH = make_mesh(NDEV)
    shard, _ = decompose(MESH, pos0, vel0, mass0, cap=CAP, num_samples=256)
    leaf_cap, leaves_per_device = _leaf_capacity_like_the_mesh(_solver(g, soft), shard)
    result["leaf_capacity_expected"] = leaf_cap
    result["leaves_per_device"] = leaves_per_device
    caps = (
        "auto" if ARGS.cross == "library" else _cross_caps(leaves_per_device, leaf_cap)
    )
    result["cross"] = ARGS.cross
    if ARGS.cross_json:
        from jaccpot.distributed.cross import CrossCapacities

        final = json.load(open(ARGS.cross_json))["cross_caps_final"]
        caps = CrossCapacities()._with(
            **{k: v for k, v in final.items() if v is not None}
        )
        result["cross"] = f"json:{ARGS.cross_json}"
    per_device = [int(c) for c in np.asarray(shard.count)]
    result["counts0"] = per_device
    times = []

    if ARGS.mode == "segment":
        from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

        solver = _solver(g, soft)
        kernel = solver._impl.softening_kernel
        migration_cap = ARGS.migration_cap
        if migration_cap == "json":
            migration_cap = json.load(open(ARGS.cross_json))["migration_cap_final"]
        elif migration_cap != "auto":
            migration_cap = int(migration_cap)
        result["migrate"] = bool(ARGS.migrate)
        t0 = time.perf_counter()
        runner = MeshSegmentRunner.fused(
            solver,
            MESH,
            MeshState.from_shard_state(shard, MESH),
            leaf_size=64,
            max_order=6,
            theta=0.8,
            cross_caps=caps,
            repartition_every=ARGS.repartition_every,
            migrate=bool(ARGS.migrate),
            migration_cap=migration_cap,
        )
        del shard
        result["setup_s"] = time.perf_counter() - t0
        _checkpoint(result, "after_setup")
        result["leaf_capacity"] = int(runner.force.setup.leaf_capacity)
        result["walk_caps"] = {
            k: v
            for k, v in (runner.force.setup.walk_caps or {}).items()
            if isinstance(v, (int, float, bool))
        }
        result["plan"] = list(runner.force.setup.plan.fingerprint())
        result["cross_caps"] = {
            k: v
            for k, v in vars(runner.force.cross_caps).items()
            if isinstance(v, (int, float)) or v is None
        }
        if runner.force.cross_needs_measured is not None:
            from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS

            result["cross_needs_measured"] = dict(
                zip(
                    CROSS_NEEDS_FIELDS,
                    np.asarray(runner.force.cross_needs_measured).tolist(),
                )
            )
        t0 = time.perf_counter()
        start = runner.start()
        jax.block_until_ready(runner.state.acceleration)
        result["start_s"] = time.perf_counter() - t0
        _checkpoint(result, "after_start")
        result["start_report"] = dataclasses.asdict(start)
        got_start = runner.gather()
        np.save(os.path.join(ARGS.dump, "positions_start.npy"), got_start["positions"])
        print("start", result["start_report"], flush=True)
        reports = []
        for _ in range(ARGS.segments):
            t0 = time.perf_counter()
            rep = runner.run(ARGS.steps, dt=dt)
            jax.block_until_ready(runner.state.state)
            times.append(time.perf_counter() - t0)
            reports.append(dataclasses.asdict(rep))
            _checkpoint(result, f"after_segment_{rep.segment}")
            print(
                "segment",
                rep.segment,
                f"{times[-1]:.3f}s",
                rep.retries,
                rep.bits,
                rep.counts,
                f"migrated {rep.migrated} need {rep.migration_need}",
                flush=True,
            )
        result["reports"] = reports
        result["migration_cap_final"] = runner.migration_cap
        result["cross_caps_final"] = {
            k: v
            for k, v in vars(runner.force.cross_caps).items()
            if isinstance(v, (int, float)) or v is None
        }
        drifted = getattr(runner.force, "cross_needs_drifted", None)
        if drifted is not None:
            # the library's sizing at the first segment's drifted positions
            from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS

            result["cross_needs_drifted"] = dict(
                zip(CROSS_NEEDS_FIELDS, np.asarray(drifted).tolist())
            )
        # the force alone (the start program: the template route's force at the
        # current positions), to tell the force from the integrator's overhead
        state = runner.state
        force_s = []
        for _ in range(int(ARGS.force_repeats)):
            t0 = time.perf_counter()
            out = runner._call(
                "start",
                0,
                None,
                state.state,
                state.masses,
                state.count,
                state.ids,
                # under the criterion: the carried f_b (the start program's input)
                *runner._scale_args(state),
            )
            jax.block_until_ready(out[0])
            force_s.append(time.perf_counter() - t0)
        result["force_only_ms"] = [1e3 * x for x in force_s]
        del out, state
        got_end = runner.gather()
        np.save(os.path.join(ARGS.dump, "positions_end.npy"), got_end["positions"])
        # the peak BEFORE the fp64 references, whose temporaries would set it
        result["peak_bytes"] = _peaks()
        del runner
        for tag, got in (("start", got_start), ("end", got_end)):
            result[f"rel_l2_{tag}"] = _rel_l2(
                got["acceleration"], got["positions"], mass0, g, soft, kernel, ids
            )
            print(tag, "rel-L2", result[f"rel_l2_{tag}"], flush=True)
    else:
        from jaccpot.distributed.rollout import (
            FusedRollout,
            RolloutConfig,
            setup_fused_force,
        )

        # the stacked route refuses the criterion (fused._refuse_error_criterion)
        assert ARGS.mac == "dehnen", "--mode stacked runs the geometric MAC only"

        cap_leaf = ARGS.leaf_capacity or leaf_cap
        result["leaf_capacity"] = int(cap_leaf)
        sibling = ARGS.out.replace("stacked_", "segment_")
        if caps == "auto" and sibling != ARGS.out and os.path.exists(sibling):
            # FusedRollout has no segment retry, and capacities sized at the start
            # (cross_caps="auto") can be outgrown in the first steps (the Morton
            # domains interpenetrate until the first repartition): the stacked
            # route runs on the capacities the segment runner ended with
            from jaccpot.distributed.cross import CrossCapacities

            final = json.load(open(sibling)).get("cross_caps_final")
            if final:
                caps = CrossCapacities()._with(
                    **{k: v for k, v in final.items() if v is not None}
                )
                result["cross"] = "segment_final"
        solver = _solver(g, soft, leaf_capacity=cap_leaf)
        t0 = time.perf_counter()
        force, plan, walk = setup_fused_force(
            solver, MESH, shard, leaf_size=64, max_order=6, theta=0.8, cross_caps=caps
        )
        _checkpoint(result, "after_setup")
        roll = FusedRollout(
            MESH,
            shard,
            RolloutConfig(
                cap=CAP,
                dt=dt,
                repartition_every=ARGS.repartition_every * ARGS.steps,
                num_samples=256,
            ),
            force,
        )
        result["plan"] = list(plan.fingerprint())
        measured = getattr(force, "cross_needs", None)
        if measured is not None:
            from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS

            result["cross_needs_measured"] = dict(
                zip(CROSS_NEEDS_FIELDS, np.asarray(measured).tolist())
            )
        chosen = getattr(force, "cross_caps", None)
        if chosen is not None:
            result["cross_caps"] = {
                k: v
                for k, v in vars(chosen).items()
                if isinstance(v, (int, float)) or v is None
            }
        # what the route sized, even when its rollout then stops on a flag
        json.dump(result, open(ARGS.out, "w"), indent=1)
        roll.start()
        jax.block_until_ready(roll.state.accel)
        result["setup_s"] = time.perf_counter() - t0
        _checkpoint(result, "after_start")
        for k in range(ARGS.segments):
            t0 = time.perf_counter()
            roll.run(ARGS.steps)
            jax.block_until_ready(roll.state.positions)
            times.append(time.perf_counter() - t0)
            _checkpoint(result, f"after_segment_{k + 1}")
            print("segment", f"{times[-1]:.3f}s", flush=True)
        result["overflow"] = any(r.overflow for r in roll.reports)
        st = roll.state
        force_s = []
        for _ in range(5):
            t0 = time.perf_counter()
            out = force(st.positions, st.masses, st.count, st.ids)
            jax.block_until_ready(out[0])
            force_s.append(time.perf_counter() - t0)
        result["force_only_ms"] = [1e3 * x for x in force_s]

    result["segment_s"] = times
    # the first two segments compile (the segment program, then the repartition's);
    # the rest are steady state
    if len(times) > 2:
        result["step_ms_steady"] = 1e3 * float(np.median(times[2:])) / ARGS.steps
    result.setdefault("peak_bytes", _peaks())
    result["peak_bytes_per_particle"] = [
        b / max(c, 1) for b, c in zip(result["peak_bytes"], per_device)
    ]
    result["smi_after"] = _smi()
    print(json.dumps({k: v for k, v in result.items() if "smi" not in k}, indent=1))
    json.dump(result, open(ARGS.out, "w"), indent=1)


if __name__ == "__main__":
    main()

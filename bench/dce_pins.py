"""GPU pins of the protected lanes for the 2026-10 cleanup (``docs/cleanup_2026-10.md``).

A deletion phase must not move a number on a lane it is not meant to touch. The
CPU suite and its goldens cover the pure-JAX routes; these pins cover what only a
GPU runs (Pallas kernels, the fused scan as Odisseo drives it, the
differentiable lane's reverse kernels, the block-step lane on device).

Each pin runs in a FRESH process (``case``), with deterministic GPU ops, and saves
its arrays under ``--root/<tag>/<pin>.npz`` plus scalars in ``<pin>.json``:

=====  ===========================================================================
S1     fused ``strict_run_v2``, clipped Plummer 2e5 (``fused_memory_budget.py``
       defaults: leaf 64 cells, theta 0.8, p6): first force, final scan state,
       fp64 rel-L2 on 4096 targets
S2     the same lane at 5e4 with ``--use-pallas off`` (the near field's pure-JAX
       rectangle / target-block route)
S3     ``FastMultipoleMethod()`` defaults at 3e4 and 1e5 (a raise is recorded as
       the pin)
S4     Odisseo's differentiable lane below its large-N profile (preset "fast"):
       d/dpos and d/dmass of a weighted force sum, 2e4
S4b    the same through the large-N lane and its grad plan, with Odisseo's
       large-N env overrides, 2e4
S5     ``BlockStepFMM`` with Odisseo's ``BlockStepOptions`` defaults (jax
       backend, leaf 64, static shapes, device topology), 20 base steps at 2e4
M1     two cards: ``distributed/fmm.py`` built as Odisseo's mesh lane builds it
       (``MeshOptions`` defaults: leaf 512, theta 0.7, p6, rcb), one force at
       131,072, in input order
M2     two cards: ``bench/multigpu_rollout_gate.py`` (``FusedRollout``,
       repartitioned on device) at 1e5, 17 steps, repartition every 8: positions
       at steps 1, 16 and 17
M3     two cards: ``DistributedBlockStepFMM`` (jax backend, cross_theta 0.5), 4
       base steps at 8192
=====  ===========================================================================

The M pins need two cards (``autocvd -n 2``) and are recorded separately:
``record --pins M1,M2,M3``.

Usage (book the card with autocvd first; the org rule)::

    export CUDA_VISIBLE_DEVICES=$(autocvd -n 1 -o -q -i 20)
    python bench/dce_pins.py record --tag 3a4bfc7 [--pins S1,S3]
    python bench/dce_pins.py record --tag 3a4bfc7-b       # A-vs-A control
    python bench/dce_pins.py compare 3a4bfc7 3a4bfc7-b    # the envelope
    python bench/dce_pins.py compare 3a4bfc7 <new> --envelope 3a4bfc7-b

``compare`` with ``--envelope`` passes a pin when it is bitwise equal to the
reference, or -- if the A-vs-A control itself was not bitwise -- within twice the
control's max difference with the rel-L2 scalars equal to two significant figures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = "/export/scratch/tbuck/dce_pins"
PINS = ("S1", "S2", "S3", "S4", "S4b", "S5")
PINS_TWO_CARD = ("M1", "M2", "M3")


# --------------------------------------------------------------------- cases


def _plummer(n: int, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    r = np.minimum(r, 20.0)
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    vel = rng.normal(0.0, 0.3, (n, 3))
    mass = np.full(n, 1.0 / n)
    return pos.astype(np.float32), vel.astype(np.float32), mass.astype(np.float32)


def _fused_bench(out: Path, n: int, extra: list[str]) -> dict:
    """S1 / S2: the production bench, unchanged, with its bitwise save."""
    import numpy as np

    js = out.with_suffix(".bench.json")
    npz = out.with_suffix(".bench.npz")
    cmd = [
        sys.executable,
        str(HERE / "fused_memory_budget.py"),
        "--n",
        str(n),
        "--ic",
        "plummer_clipped",
        "--caps",
        "unnamed",
        "--steps",
        "4",
        "--reps",
        "2",
        "--eval-repeats",
        "2",
        "--accuracy-targets",
        "4096",
        "--no-analysis",
        "--save-forces",
        str(npz),
        "--out",
        str(js),
        *extra,
    ]
    subprocess.run(cmd, check=True)
    data = np.load(npz)
    np.savez(out, force=data["force"], state=data["state"])
    result = json.loads(js.read_text())
    return {"rel_l2": result["accuracy"]["rel_l2"], "n": n}


def case_s1(out: Path) -> dict:
    return _fused_bench(out, 200_000, [])


def case_s2(out: Path) -> dict:
    return _fused_bench(out, 50_000, ["--use-pallas", "off"])


def case_s3(out: Path) -> dict:
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import FastMultipoleMethod

    arrays, meta = {}, {}
    for n in (30_000, 100_000):
        pos, _, mass = _plummer(n)
        try:
            acc = FastMultipoleMethod().compute_accelerations(
                jnp.asarray(pos), jnp.asarray(mass)
            )
            arrays[f"acc_{n}"] = np.asarray(acc)
            meta[f"n{n}"] = "ok"
        except Exception as exc:  # noqa: BLE001 -- the raise IS the pin
            meta[f"n{n}"] = f"raises {type(exc).__name__}: {str(exc)[:300]}"
    np.savez(out, **arrays)
    return meta


def _grad_pin(out: Path, solver, n: int, plan_fn=None) -> dict:
    """d/dpos and d/dmass of a fixed weighting of the forces, through
    ``differentiable_accelerations`` (Odisseo's differentiable lane)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    pos, _, mass = _plummer(n)
    p = jnp.asarray(pos)
    m = jnp.asarray(mass)
    prepared = solver.prepare_state(p, m, leaf_size=32, max_order=4, theta=0.6)
    plan = None if plan_fn is None else plan_fn(solver, prepared)
    w = jnp.asarray(np.random.default_rng(2).normal(size=(n, 3)), jnp.float32)

    def loss(x, mm):
        acc = solver.differentiable_accelerations(prepared, x, mm, grad_plan=plan)
        return jnp.sum(acc * w)

    value, (g_pos, g_mass) = jax.value_and_grad(loss, argnums=(0, 1))(p, m)
    np.savez(out, g_pos=np.asarray(g_pos), g_mass=np.asarray(g_mass))
    return {"loss": float(value), "n": n}


def case_s4(out: Path) -> dict:
    """Odisseo's differentiable lane below its large-N profile: preset "fast"."""
    import jax.numpy as jnp

    from jaccpot import FarFieldConfig, FastMultipoleMethod, FMMAdvancedConfig

    solver = FastMultipoleMethod(
        preset="fast",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            farfield=FarFieldConfig(mode="auto", retain_far_pairs_for_grad=True),
            mac_type="dehnen",
        ),
    )
    return _grad_pin(out, solver, 20_000)


def case_s4b(out: Path) -> dict:
    """The large-N differentiable lane, with Odisseo's large-N env overrides."""
    import os

    sys.path.insert(
        0,
        os.environ.get(
            "BENCH_DIR",
            "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu",
        ),
    )
    from codes.compare_force import apply_fast_lane_env

    n = 20_000
    apply_fast_lane_env(n)  # what Odisseo sets for its large_n_gpu profile
    import jax.numpy as jnp

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )
    from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=32),
            farfield=FarFieldConfig(mode="auto", retain_far_pairs_for_grad=True),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
    )
    return _grad_pin(out, solver, n, plan_fn=prepare_large_n_grad_plan)


def case_s5(out: Path) -> dict:
    """``BlockStepFMM`` with Odisseo's ``BlockStepOptions`` defaults, on device."""
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import BlockStepFMM

    n = 20_000
    pos, vel, mass = _plummer(n)
    force = BlockStepFMM(
        softening=1e-3,
        k_max=2,
        theta=0.6,
        max_order=4,
        G=1.0,
        basis="real",
        backend="jax",
        leaf_size=64,
        static_shapes=True,
        topology_backend="device",
    )
    x, v, m = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(mass)
    rung = jnp.asarray(np.arange(n) % 3, jnp.int32)
    force.prepare(x, m)
    for _ in range(20):
        x, v, _ = force.advance_base_step(x, v, m, rung=rung, dt_max=1e-3)
    p0 = np.sum(mass[:, None] * vel, axis=0)
    p1 = np.sum(mass[:, None] * np.asarray(v, np.float64), axis=0)
    np.savez(out, positions=np.asarray(x), velocities=np.asarray(v))
    return {"momentum_drift": float(np.linalg.norm(p1 - p0)), "n": n}


def _need_two_devices() -> None:
    import jax

    if jax.local_device_count() < 2:
        raise RuntimeError("the M pins need two cards: book them with autocvd -n 2")


def case_m1(out: Path) -> dict:
    """``distributed/fmm.py`` as Odisseo's mesh lane builds it."""
    _need_two_devices()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P
    from yggdrax.distributed import make_mesh

    from jaccpot.distributed.fmm import (
        DIAG_FIELDS,
        DistributedFMMConfig,
        make_force_evaluator,
        partition_for_devices,
        scatter_to_input_order,
    )

    ndev, leaf = 2, 512
    n = ndev * leaf * 128  # the mesh lane wants N = ndev * k * leaf: no padding rows
    pos, _, mass = _plummer(n)
    part = partition_for_devices(pos, mass, ndev, leaf_size=leaf, partitioner="rcb")
    cap = int(part["cap"])
    # odisseo.mesh_coupling.integrate_mesh with MeshOptions() defaults
    cfg = DistributedFMMConfig(
        leaf_size=leaf,
        theta=0.7,
        order=6,
        softening=1e-3,
        G=1.0,
        m2l_chunk=65_536,
        nearfield_chunk=512,
        nearfield_accum="wide",
        mac_type="dehnen",
        adaptive_eps=None,
        mac_cross_criterion=True,
    ).resolved_for(cap, ndev)
    mesh = make_mesh(ndev)
    evaluate = make_force_evaluator(
        cfg, ndev, cap, mesh, jit=True, halo_exchange="auto"
    )
    x = jax.device_put(
        jnp.asarray(part["pos_flat"]), NamedSharding(mesh, P("gpus", None))
    )
    m = jax.device_put(jnp.asarray(part["mass_flat"]), NamedSharding(mesh, P("gpus")))
    a_raw, gid, diag = evaluate(
        x, m, jnp.asarray(part["gid_flat"]), jnp.asarray(part["counts"])
    )
    acc = scatter_to_input_order(np.asarray(a_raw), np.asarray(gid), n)
    d = np.asarray(diag)
    overflow = {
        name: float(d[:, i].sum())
        for i, name in enumerate(DIAG_FIELDS)
        if name.endswith("overflow") and i < d.shape[1]
    }
    # fp64 direct sum on 1024 targets (informative; the pin is the array)
    targets = np.random.default_rng(3).choice(n, 1024, replace=False)
    p64, m64 = pos.astype(np.float64), mass.astype(np.float64)
    ref = np.empty((targets.size, 3))
    for s in range(0, targets.size, 32):
        t = targets[s : s + 32]
        dx = p64[None, :, :] - p64[t, None, :]
        r2 = np.sum(dx * dx, -1) + 1e-6
        ref[s : s + 32] = np.sum(m64[None, :, None] * dx * r2[..., None] ** -1.5, 1)
    rel = float(np.linalg.norm(acc[targets] - ref) / np.linalg.norm(ref))
    np.savez(out, acc=acc)
    return {"n": n, "cap": cap, "rel_l2": rel, "overflow": overflow}


def case_m2(out: Path) -> dict:
    """The fused two-card rollout gate, short: the forward, the cross field and the
    on-device repartition all feed the positions it dumps."""
    _need_two_devices()
    import numpy as np

    n, steps, every = 100_000, 17, 8
    gate_dir = out.parent / f"{out.stem}_gate"
    gate_dir.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        GATE_MODE="mesh",
        GATE_N=str(n),
        GATE_STEPS=str(steps),
        GATE_EVERY=str(every),
        GATE_OUT=str(gate_dir),
    )
    # The gate selects XLA's NCCL ragged exchange (fused.RAGGED_EXCHANGE_XLA_FLAG);
    # from jax 0.11 XLA refuses that fallback unless it is allowed explicitly.
    allow = "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true"
    if allow.split("=")[0] not in env.get("XLA_FLAGS", ""):
        env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + allow).strip()
    subprocess.run(
        [sys.executable, str(HERE / "multigpu_rollout_gate.py")], env=env, check=True
    )
    arrays = {
        f"positions_step{k}": np.load(gate_dir / f"positions_step{k}.npy")
        for k in (1, 16, 17)
    }
    np.savez(out, **arrays)
    result = json.loads((gate_dir / f"gate_mesh_every{every}.json").read_text())
    return {
        "n": n,
        "probes": result["probes"],
        "repartitions": result["repartitions"],
        "moved_total": result["moved_total"],
    }


def case_m3(out: Path) -> dict:
    """``DistributedBlockStepFMM``: the distributed mutual lane, block steps."""
    _need_two_devices()
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import DistributedBlockStepFMM
    from jaccpot.mutual.force import MutualCapacities

    n = 8192
    pos, vel, mass = _plummer(n)
    # explicit: the heuristic's depth (16) is too shallow for the clipped Plummer core
    caps = MutualCapacities(near=16384, far=16384, depth=32, width=1024, queue=1 << 17)
    force = DistributedBlockStepFMM(
        softening=1e-3,
        k_max=2,
        theta=0.5,
        cross_theta=0.5,
        max_order=4,
        G=1.0,
        leaf_size=32,
        backend="jax",
        ndev=2,
        caps=caps,
    )
    x, v, m = jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(mass)
    rung = jnp.asarray(np.arange(n) % 3, jnp.int32)
    force.prepare(x, m)
    for _ in range(4):
        x, v, _ = force.advance_base_step(x, v, m, rung=rung, dt_max=1e-3)
    p0 = np.sum(mass[:, None].astype(np.float64) * vel, axis=0)
    p1 = np.sum(mass[:, None].astype(np.float64) * np.asarray(v, np.float64), axis=0)
    np.savez(out, positions=np.asarray(x), velocities=np.asarray(v))
    return {"momentum_drift": float(np.linalg.norm(p1 - p0)), "n": n}


CASES = {
    "M1": case_m1,
    "M2": case_m2,
    "M3": case_m3,
    "S1": case_s1,
    "S2": case_s2,
    "S3": case_s3,
    "S4": case_s4,
    "S4b": case_s4b,
    "S5": case_s5,
}


# --------------------------------------------------------------------- driver


def _record(args) -> int:
    out_dir = Path(args.root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
    ).strip()
    failed = []
    for pin in args.pins.split(","):
        print(f"[{args.tag}] {pin} ...", flush=True)
        rc = subprocess.run(
            [sys.executable, __file__, "case", pin, "--out", str(out_dir / pin)],
            env=env,
            check=False,
        ).returncode
        if rc != 0:
            failed.append(pin)
    print(f"[{args.tag}] done; failed: {failed or 'none'}")
    return 1 if failed else 0


def _case(args) -> int:
    out = Path(args.out).with_suffix(".npz")
    meta = CASES[args.pin](out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return 0


def _diff(a: Path, b: Path) -> dict:
    import numpy as np

    da, db = np.load(a), np.load(b)
    rows = {}
    for key in sorted(set(da.files) | set(db.files)):
        if key not in da.files or key not in db.files:
            rows[key] = {"missing": True}
            continue
        x, y = da[key], db[key]
        if x.shape != y.shape:
            rows[key] = {"shape": [list(x.shape), list(y.shape)]}
            continue
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        ref = np.linalg.norm(x.astype(np.float64))
        rows[key] = {
            "bitwise": bool(np.array_equal(x, y)),
            "max_abs": float(d.max()) if d.size else 0.0,
            "rel_l2": float(np.linalg.norm(d) / ref) if ref else 0.0,
        }
    return rows


def _compare(args) -> int:
    root = Path(args.root)
    ref, new = root / args.ref, root / args.new
    env = root / args.envelope if args.envelope else None
    report, ok = {}, True
    wanted = args.pins.split(",")
    for pin in wanted:
        if not (ref / f"{pin}.npz").exists():
            report[pin] = {"missing_in_ref": True}
            ok = False
    for npz in sorted(ref.glob("*.npz")):
        if npz.name.endswith(".bench.npz") or npz.stem not in wanted:
            continue
        pin = npz.stem
        if not (new / npz.name).exists():
            report[pin] = {"missing": True}
            ok = False
            continue
        rows = _diff(npz, new / npz.name)
        meta_ref = json.loads(npz.with_suffix(".json").read_text())
        meta_new = json.loads((new / npz.name).with_suffix(".json").read_text())
        entry = {"arrays": rows, "meta": {"ref": meta_ref, "new": meta_new}}
        if env is not None:
            control = _diff(npz, env / npz.name)
            passed = True
            for key, row in rows.items():
                c = control.get(key, {})
                if row.get("bitwise"):
                    continue
                if c.get("bitwise", True) or "max_abs" not in row:
                    passed = False
                elif row["max_abs"] > 2.0 * c["max_abs"]:
                    passed = False
            for key, val in meta_ref.items():
                other = meta_new.get(key)
                if isinstance(val, float) and isinstance(other, float):
                    if f"{val:.2g}" != f"{other:.2g}":
                        passed = False
                elif val != other:
                    passed = False
            entry["pass"] = passed
            ok = ok and passed
        report[pin] = entry
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--tag", required=True)
    rec.add_argument("--pins", default=",".join(PINS))
    rec.add_argument("--root", default=ROOT_DEFAULT)
    case = sub.add_parser("case")
    case.add_argument("pin", choices=sorted(CASES))
    case.add_argument("--out", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("ref")
    cmp_.add_argument("new")
    cmp_.add_argument("--envelope", default=None)
    cmp_.add_argument(
        "--pins", default=",".join(PINS), help="M1,M2,M3 for the two-card set"
    )
    cmp_.add_argument("--root", default=ROOT_DEFAULT)
    cmp_.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return {"record": _record, "case": _case, "compare": _compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

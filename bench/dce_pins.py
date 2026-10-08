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
=====  ===========================================================================

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


CASES = {
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
    for npz in sorted(ref.glob("*.npz")):
        if npz.name.endswith(".bench.npz"):
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
    cmp_.add_argument("--root", default=ROOT_DEFAULT)
    cmp_.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return {"record": _record, "case": _case, "compare": _compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

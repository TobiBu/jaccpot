"""S11 session 2 track G on cards: is the mesh start force bitwise run to run?

One process, one compiled program -- ``MeshSegmentRunner``'s start force (the
template route, ``FusedMeshForce.shard``) -- run ``--repeats`` times on the
identical state; every run is compared with the first element by element (the
acceleration, and ``f_b`` under ``dehnen_error``). The control is the last run's
buffers read twice. Then ``--time-repeats`` more calls are timed (the median and the
minimum, in ms; contended unless both cards were free: the JSON records
``nvidia-smi`` before and after).

On a GPU the cross walk (the Pallas mutual walk) emits its pairs through atomic
cursors, in a different order every run; until the cross hook put its pair lists in
(target, source) order (``cross._in_pair_order``) the CSR M2L and the cross near
kernel summed each target's cross pairs in that order, and at 4e5 on two A100s
~600k of 1.38M acceleration components differed between two calls (track G's stage
probe: the gathered summary, the exchanged payloads and the local multipoles
bitwise, the walk's ordered lists the first thing that differs). Forced CPU devices
cannot show it: the interpreted walk emits in one order (the CPU check is
``tests/distributed/test_mesh_segment_runner.py``'s emission-order case).

    eval $(autocvd -q -l -x 1 -n 2 -e -t 120)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/multigpu_force_repeat.py \\
        --mac dehnen --n 400000 --out OUT.json

Exits 1 if any run differs from the first. ``--save`` / ``--compare`` set the first
run against another process's (a fresh compile of the same program). Plummer sphere from the Odisseo harness
(``BENCH_DIR``), ``preset="galaxy"`` (leaf 64, p6), the library's cross capacities.
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

sys.path.insert(
    0,
    os.environ.get(
        "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
    ),
)

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ndev", type=int, default=2)
    p.add_argument("--mac", choices=("dehnen_error", "dehnen"), default="dehnen")
    p.add_argument("--n", type=int, default=400_000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--time-repeats", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--save",
        default=None,
        help="save the first run's outputs (.npz), for --compare in another process",
    )
    p.add_argument(
        "--compare",
        default=None,
        help="compare the first run with a --save of another process (same flags)",
    )
    return p.parse_args()


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


def _differ(got: np.ndarray, want: np.ndarray) -> dict:
    mask = got != want
    gap = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {
        "differ": int(mask.sum()),
        "of": int(mask.size),
        "max_abs": float(gap.max()) if mask.any() else 0.0,
    }


def main() -> int:
    """Run the start force repeatedly and write the JSON record.

    Returns
    -------
    int
        ``0`` when every run is bitwise the first, else ``1``.
    """
    from common.ic import IC_GENERATORS
    from yggdrax.distributed.sharding import make_mesh

    import jaccpot
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    args = _args()
    devices = jax.devices()
    assert devices[0].platform == "gpu", devices
    assert jax.device_count() >= args.ndev, jax.device_count()
    smi_before = _smi()

    pos, mass = IC_GENERATORS["plummer"](args.n, seed=0)
    pos = np.asarray(pos, np.float32)
    mass = np.asarray(mass, np.float32)
    base = FastMultipoleMethod(preset="galaxy").advanced
    adv = (
        base
        if args.mac == "dehnen_error"
        else dataclasses.replace(base, mac_type="dehnen")
    )
    solver = FastMultipoleMethod(preset="galaxy", G=1.0, softening=1e-7, advanced=adv)
    mesh = make_mesh(args.ndev)
    cap = int(np.ceil(args.n / args.ndev * 1.15))
    shard, _ = decompose(mesh, pos, np.zeros_like(pos), mass, cap=cap, num_samples=256)
    runner = MeshSegmentRunner.fused(
        solver, mesh, MeshState.from_shard_state(shard, mesh), leaf_size=64, max_order=6
    )
    s = runner.state
    inputs = (s.state, s.masses, s.count, s.ids, *runner._scale_args(s))

    def call() -> tuple:
        out = runner._call("start", 0, None, *inputs)
        jax.block_until_ready(out)
        return out

    runs = []
    for _ in range(args.repeats):
        out = call()
        assert bool(np.asarray(out[-3])), "the start force overflowed"
        runs.append([np.asarray(x) for x in out[:-3]])
    names = ["acceleration"] + (["f_b"] if runner.scale_carry else [])
    control = {
        name: _differ(np.asarray(x), ref)
        for name, x, ref in zip(names, out[:-3], runs[-1])
    }
    compared = [
        {name: _differ(got, ref) for name, got, ref in zip(names, run, runs[0])}
        for run in runs[1:]
    ]
    times = []
    for _ in range(args.time_repeats):
        t0 = time.perf_counter()
        call()
        times.append(1e3 * (time.perf_counter() - t0))
    bitwise = all(c[name]["differ"] == 0 for c in compared for name in names)
    if args.save:
        np.savez(args.save, **dict(zip(names, runs[0])))
    other_process = None
    if args.compare:
        saved = np.load(args.compare)
        other_process = {
            name: _differ(got, saved[name]) for name, got in zip(names, runs[0])
        }
    record = {
        "jaccpot": os.path.dirname(jaccpot.__file__),
        "ndev": args.ndev,
        "mac": args.mac,
        "n": args.n,
        "devices": [str(d) for d in devices[: args.ndev]],
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "cross_caps": (
            None if runner.force.cross_caps is None else vars(runner.force.cross_caps)
        ),
        "control_same_buffers": control,
        "runs_vs_first": compared,
        "bitwise": bitwise,
        "first_vs_other_process": other_process,
        "start_force_ms": {
            "median": float(np.median(times)),
            "min": float(np.min(times)),
            "all": times,
        },
        "nvidia_smi_before": smi_before,
        "nvidia_smi_after": _smi(),
    }
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=1)
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("runs_vs_first", "bitwise", "first_vs_other_process")
            }
        )
    )
    print(json.dumps(record["start_force_ms"]))
    return 0 if bitwise else 1


if __name__ == "__main__":
    sys.exit(main())

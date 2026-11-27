"""Shared plumbing for the paper's mesh-lane benches (figures 10 and 11).

Two halves, and only the second imports JAX:

* the PARENT half (:func:`run_points`): one fresh process per point, because a
  process's device count is fixed when JAX starts (and XLA reads its flags once).
  The parent never imports JAX, so it holds no card its children need.
* the WORKER half (:func:`worker_preamble` before the jax import, then
  :func:`make_ic`, :func:`make_solver`, :func:`build_runner`): the same mesh lane,
  ICs and solver settings as ``bench/multigpu_segment_gate.py`` (the scaling
  matrix's row), so figures 10 and 11 describe the configuration figures 08 and 09
  time: ``preset="galaxy"``, ``decompose`` (Morton domains, weighted-pivot
  ``sfc_repartition``), ``MeshSegmentRunner.fused(leaf_size=64, max_order=6,
  theta=0.8, cross_caps="auto")``.

On the CPU (``JAX_PLATFORMS=cpu``) a worker runs on forced host devices
(``--xla_force_host_platform_device_count``) through the bundle's CPU route,
``bench/slurm/h0/smoke_site`` with ``H0_SMOKE_CPU=1`` (the Pallas kernels in
interpret mode, the ragged exchange on its CPU route), which the parent puts on the
child's ``PYTHONPATH``. That is a plumbing check: emulated devices share one CPU.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
#: the Odisseo harness's ICs (``common.ic``), vendored with the HoreKa bundle
VENDORED_BENCH_DIR = (
    REPO_ROOT / "bench" / "slurm" / "h0" / "harness" / "benchmark_multigpu"
)
SMOKE_SITE = REPO_ROOT / "bench" / "slurm" / "h0" / "smoke_site"
AUTOTUNER_OFF = "--xla_gpu_experimental_enable_fusion_autotuner=false"
#: what the segment gate passes the runner (its source literals)
LEAF_SIZE, MAX_ORDER, THETA = 64, 6, 0.8


def on_cpu() -> bool:
    """``JAX_PLATFORMS=cpu``: a smoke on forced host devices, not a measurement."""
    return os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"


# --------------------------------------------------------------------------- #
# parent half: no jax
# --------------------------------------------------------------------------- #


def child_env(ndev: int) -> dict[str, str]:
    """The environment of one point's worker process.

    Parameters
    ----------
    ndev : int
        The point's device count (forced host devices on the CPU).

    Returns
    -------
    dict
        ``os.environ`` plus: no preallocation, ``BENCH_DIR`` defaulting to the
        vendored harness, and on the CPU the forced device count and the smoke
        shim. The ragged-exchange flags are added by the worker itself
        (``bench/_ragged_exchange_flags.py``) before its jax import.
    """
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("BENCH_DIR", str(VENDORED_BENCH_DIR))
    path = [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(":") if p]
    if on_cpu():
        flags = env.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            flags = f"{flags} --xla_force_host_platform_device_count={ndev}".strip()
        env["XLA_FLAGS"] = flags
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("H0_SMOKE_CPU", "1")
        if str(SMOKE_SITE) not in path:
            path.insert(0, str(SMOKE_SITE))
    env["PYTHONPATH"] = ":".join(path)
    return env


def run_points(
    module: str, points: list[dict[str, Any]], common_argv: list[str]
) -> list[dict[str, Any]]:
    """Run each point as ``python -m module --worker ...`` and collect its record.

    Parameters
    ----------
    module : str
        The bench module (its worker prints its record as the last JSON line).
    points : list of dict
        Each with ``ndev`` and ``argv`` (the point's own flags).
    common_argv : list of str
        Flags every worker gets.

    Returns
    -------
    list of dict
        One record per point, in order. A worker that died is recorded
        ``valid=False`` with its return code and stderr tail, so a partial sweep
        cannot pass for a complete one.
    """
    out: list[dict[str, Any]] = []
    for p in points:
        cmd = [sys.executable, "-m", module, "--worker", "--ndev", str(p["ndev"])]
        cmd += list(p.get("argv", [])) + list(common_argv)
        print("[point]", " ".join(cmd[3:]), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=child_env(int(p["ndev"])),
            capture_output=True,
            text=True,
        )
        record = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = None
                break
        if record is None or proc.returncode not in (0, 1):
            record = {
                **{k: v for k, v in p.items() if k != "argv"},
                "valid": False,
                "invalid_reasons": [f"worker exited {proc.returncode}"],
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr or "")[-3000:],
                "stdout_tail": (proc.stdout or "")[-1500:],
            }
            print(f"[point] FAILED (exit {proc.returncode})", file=sys.stderr)
            print((proc.stderr or "")[-1500:], file=sys.stderr)
        out.append(record)
    return out


# --------------------------------------------------------------------------- #
# worker half
# --------------------------------------------------------------------------- #


def worker_preamble(ndev: int, gpu_select: str) -> Optional[str]:
    """Everything that must happen before the worker's first ``import jax``.

    autocvd claims ``ndev`` free cards in ONE selection
    (``runmeta.select_gpu(num_gpus=ndev)``; skipped on the CPU), the fusion
    autotuner goes off (production), the ragged-exchange flags go on
    (``bench/_ragged_exchange_flags.py``, as every library multi-GPU bench), and
    ``BENCH_DIR`` (the ICs) goes on ``sys.path``.

    Parameters
    ----------
    ndev : int
        Cards this point needs.
    gpu_select : str
        ``least-used`` / ``first`` / ``none``.

    Returns
    -------
    str or None
        The selected ``CUDA_VISIBLE_DEVICES``.
    """
    sys.path.insert(0, str(REPO_ROOT / "bench"))
    from _ragged_exchange_flags import add_ragged_exchange_flags

    from examples.jaccpot_paper.common import runmeta

    chosen = None
    if not on_cpu():
        chosen = runmeta.select_gpu(gpu_select, num_gpus=int(ndev))
    else:
        flags = os.environ.get("XLA_FLAGS", "")
        if "xla_force_host_platform_device_count" not in flags:
            os.environ["XLA_FLAGS"] = (
                f"{flags} --xla_force_host_platform_device_count={ndev}"
            ).strip()
    flags = os.environ.get("XLA_FLAGS", "")
    if AUTOTUNER_OFF.split("=")[0] not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {AUTOTUNER_OFF}".strip()
    add_ragged_exchange_flags()
    sys.path.insert(0, os.environ.get("BENCH_DIR", str(VENDORED_BENCH_DIR)))
    return chosen


def _plummer_velocities(pos: Any, rng: Any) -> Any:
    """Isotropic equilibrium speeds for a G = M = a = 1 Plummer sphere (AHW 1974).

    The segment gate's draw (``bench/multigpu_segment_gate.py``), seeded the same.
    """
    import numpy as np

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


def make_ic(
    name: str, n: int, seed: int, disc_ic: Optional[str]
) -> tuple[Any, Any, Any, float, float, float]:
    """``(pos, vel, mass, G, softening, dt)``, float32 arrays, as the segment gate.

    Parameters
    ----------
    name : str
        ``plummer`` (the gate's: Odisseo harness draw, AHW speeds, softening 1e-7,
        dt 5e-3), ``uniform`` (the harness's cube, at rest, same softening and dt;
        the contrast case: a Morton split of a cube is balanced by construction) or
        ``disc`` (the gate's 25M disc + bulge prefix, its softening formula, dt
        5e-4).
    n : int
        Particles.
    seed : int
        The draw's seed (the gate uses 0).
    disc_ic : str or None
        The disc + bulge ``.npz`` (``state0``, ``mass``, ``rdisk_code``, ``G_code``).

    Returns
    -------
    tuple
        As above.
    """
    import numpy as np

    if name in ("plummer", "uniform"):
        from common.ic import IC_GENERATORS

        pos, mass = IC_GENERATORS[name](n, seed=seed)
        pos = np.asarray(pos, np.float64)
        if name == "plummer":
            vel = _plummer_velocities(pos, np.random.default_rng(7))
        else:
            vel = np.zeros_like(pos)
        return (
            pos.astype(np.float32),
            vel.astype(np.float32),
            np.asarray(mass, np.float32),
            1.0,
            1e-7,
            5e-3,
        )
    if name == "disc":
        if not disc_ic or not os.path.exists(disc_ic):
            raise SystemExit(f"--distribution disc needs --disc-ic (got {disc_ic!r})")
        ic = np.load(disc_ic)
        n_all = int(ic["state0"].shape[0])
        state = np.asarray(ic["state0"][:n], np.float32)
        mass = np.asarray(ic["mass"][:n], np.float64) * (n_all / n)
        g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
        soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(n / 1e5))
        return state[:, 0], state[:, 1], mass.astype(np.float32), g, soft, 5e-4
    raise SystemExit(f"unknown distribution {name!r} (plummer, uniform, disc)")


def make_solver(mac: str, g: float, soft: float) -> Any:
    """The segment gate's solver: ``preset="galaxy"``, ``dehnen_error`` or ``dehnen``."""
    import dataclasses

    from jaccpot import FastMultipoleMethod

    advanced = FastMultipoleMethod(preset="galaxy").advanced
    if mac == "dehnen":
        advanced = dataclasses.replace(advanced, mac_type="dehnen")
    elif mac != "dehnen_error":
        raise SystemExit(f"--mac {mac!r}: dehnen_error or dehnen")
    return FastMultipoleMethod(preset="galaxy", G=g, softening=soft, advanced=advanced)


def build_runner(
    solver: Any, pos: Any, vel: Any, mass: Any, ndev: int, *, migrate: bool
) -> tuple[Any, dict[str, Any]]:
    """``decompose`` + ``MeshSegmentRunner.fused`` as the segment gate builds them.

    Parameters
    ----------
    solver : Any
        From :func:`make_solver`.
    pos, vel, mass : array
        Host ICs.
    ndev : int
        Mesh devices.
    migrate : bool
        ``MeshSegmentRunner(migrate=...)``: on (production) migrates after every
        drift and rebalances by exact rank; off lets the weighted-pivot
        ``sfc_repartition`` rebalance before every segment.

    Returns
    -------
    tuple
        ``(runner, decomposition)``: the runner and ``decompose``'s diagnostics
        (``recv_counts``, ``sent_off_device``, ``max_util``) as lists.
    """
    import jax
    import numpy as np
    from yggdrax.distributed.sharding import make_mesh

    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

    if jax.device_count() < ndev:
        raise SystemExit(
            f"need {ndev} devices, JAX sees {jax.device_count()}: a point on another "
            "device count is not the point asked for"
        )
    n = int(pos.shape[0])
    cap = int(n / ndev * 1.15) if ndev > 1 else n
    mesh = make_mesh(ndev)
    shard, diag = decompose(mesh, pos, vel, mass, cap=cap, num_samples=256)
    runner = MeshSegmentRunner.fused(
        solver,
        mesh,
        MeshState.from_shard_state(shard, mesh),
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        theta=THETA,
        cross_caps="auto",
        repartition_every=1,
        migrate=bool(migrate),
    )
    decomposition = {
        k: np.asarray(v).tolist() for k, v in diag.items() if np.asarray(v).size < 4096
    }
    decomposition["cap"] = cap
    return runner, decomposition


def platform_of_devices() -> dict[str, Any]:
    """The devices the worker ran on (after the jax import)."""
    import jax

    devs = jax.devices()
    return {
        "platform": devs[0].platform,
        "device_kind": devs[0].device_kind,
        "devices": [str(d) for d in devs],
    }

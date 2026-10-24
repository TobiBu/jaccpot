"""S11 session 2 track D on cards: ``ProductionRunner``'s multi-GPU mode, and its restart.

Each mode in its OWN process (XLA reads its flags once; the peak is a per-process
high-water mark):

* ``--mode run`` (2 cards): ``ProductionRunner(solver, mesh=MeshRunConfig(...))`` on
  a Plummer sphere (the S11/S12 gates' draw, ``preset="galaxy"``: leaf 64, p6,
  ``dehnen_error`` at 1e-4 unless ``--mac dehnen``; the library's cross capacities
  and shard capacity), ``--segments`` segments of ``--steps`` steps with a
  repartition before every segment after the first, every diagnostic at every
  edge (the energy is ``None`` with a reason until the fused mesh force has its
  potential), a restart file after every segment. Records the edges (step,
  retries, flags, repartitions, plan, diagnostics), the per-device peak bytes per
  live particle (BEFORE the fp64 reference), the rel-L2 of the carried total
  acceleration against an fp64 direct sum on 512 fixed ids at the first edge
  after a segment and at the end, and saves the final state and acceleration by
  id. Run it twice (``--tag a``, ``--tag b``) for the A-vs-A control.
* ``--mode resume`` (2 cards): a fresh process continues run ``a``'s restart file
  after segment ``--from-segment`` to the end and compares its final state and
  acceleration with run ``a``'s, element by element; the control (``b`` against
  ``a``) is reported next to it. A second resume (``--tag resumed2``) is the
  control of the continuation itself.

On the production flags the lane is not deterministic run to run on a GPU (the
``dehnen_error`` seed's scatter-add, and the cross field's): add
``--xla_gpu_deterministic_ops=true`` to ``XLA_FLAGS`` for a bitwise restart gate.

    eval $(autocvd -q -l -x 1 -n 2 -e -t 120)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/multigpu_runner_gate.py \\
        --mode run --tag a --out-dir OUT

Not a timing gate (the JSON records ``nvidia-smi`` before and after).
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

# the Odisseo harness's Plummer draw (common/), as bench/multigpu_segment_gate.py
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


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=("run", "resume"), required=True)
    p.add_argument(
        "--tag",
        default=None,
        help="run: which run (a, or b the control; default a); resume: its outputs' "
        "name (default resumed)",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--steps", type=int, default=8, help="steps per segment")
    p.add_argument("--segments", type=int, default=3)
    p.add_argument("--dt", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from-segment", type=int, default=2, help="resume: the restart")
    p.add_argument("--mac", choices=("dehnen_error", "dehnen"), default="dehnen_error")
    return p.parse_args()


ARGS = _args()
if ARGS.tag is None:
    ARGS.tag = "a" if ARGS.mode == "run" else "resumed"
NDEV = 2


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
    """``(state [N, 2, 3], masses [N])`` float32, the segment gate's Plummer."""
    from common.ic import IC_GENERATORS

    pos, mass = IC_GENERATORS["plummer"](ARGS.n, seed=ARGS.seed)
    pos = np.asarray(pos, np.float64)
    vel = _plummer_velocities(pos, np.random.default_rng(7))
    state = np.stack([pos, vel], axis=1).astype(np.float32)
    return state, np.asarray(mass, np.float32)


G, SOFTENING = 1.0, 1e-7


def _solver_and_settings():
    from jaccpot import FastMultipoleMethod
    from jaccpot.production import GALAXY_SETTINGS, GalaxySettings

    if ARGS.mac == "dehnen_error":
        return (
            FastMultipoleMethod(preset="galaxy", G=G, softening=SOFTENING),
            GALAXY_SETTINGS,
        )
    base = FastMultipoleMethod(preset="galaxy").advanced
    solver = FastMultipoleMethod(
        preset="galaxy",
        G=G,
        softening=SOFTENING,
        advanced=dataclasses.replace(base, mac_type="dehnen"),
    )
    return solver, GalaxySettings(mac_type="dehnen", adaptive_eps=None)


def _runner(callbacks=()):
    from yggdrax.distributed.sharding import make_mesh

    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig

    solver, settings = _solver_and_settings()
    restarts = os.path.join(ARGS.out_dir, f"restart_{ARGS.tag}_{{segment}}.npz")
    return ProductionRunner(
        solver,
        settings=settings,
        segment=SegmentConfig(
            steps=ARGS.steps,
            diagnostics_every=1,
            diagnostics_every_energy=1,
            restart_every=1,
            restart_path=restarts,
        ),
        callbacks=callbacks,
        mesh=MeshRunConfig(make_mesh(NDEV)),
    )


def _direct_fp64(pos, mass, idx, kernel: str) -> np.ndarray:
    """fp64 direct sum at the targets ``idx`` through the solver's pair kernel
    (``bench/multigpu_segment_gate.py``'s)."""
    from jaccpot.softening import pair_factors, softening_params

    p = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)
    n = int(p.shape[0])
    chunk = 1 << 17
    npad = -(-n // chunk) * chunk
    sp = jnp.zeros((npad, 3), jnp.float64).at[:n].set(p).reshape(-1, chunk, 3)
    sm = jnp.zeros((npad,), jnp.float64).at[:n].set(m).reshape(-1, chunk)
    sid = jnp.arange(npad).reshape(-1, chunk)
    params = softening_params(kernel, SOFTENING, jnp.float64)

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
    return G * np.concatenate(out)


def _rel_l2(acc, pos, mass, kernel, ids) -> float:
    ref = _direct_fp64(pos, mass, ids, kernel)
    got = np.asarray(acc, np.float64)[ids]
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def _peaks() -> list:
    return [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()[:NDEV]
    ]


class _FirstEdge:
    """Keeps the first post-segment edge's state and total acceleration by id."""

    def __init__(self):
        self.kept = None

    def __call__(self, edge):
        if edge.segment == 1:
            self.kept = edge.gather(("state", "acceleration"))


def _differ(got: np.ndarray, want: np.ndarray) -> dict:
    got, want = np.asarray(got), np.asarray(want)
    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    return {
        "differ": int(np.sum(got != want)),
        "of": int(got.size),
        "max_abs": float(np.nanmax(diff)) if diff.size else 0.0,
    }


def main() -> None:
    devices = jax.devices()
    assert devices[0].platform == "gpu", devices
    assert len(devices) >= NDEV, devices
    os.makedirs(ARGS.out_dir, exist_ok=True)
    result: dict = dict(
        mode=ARGS.mode,
        tag=ARGS.tag,
        mac=ARGS.mac,
        n=ARGS.n,
        steps=ARGS.steps,
        segments=ARGS.segments,
        dt=ARGS.dt,
        devices=[str(d) for d in devices[:NDEV]],
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        smi_before=_smi(),
    )
    num_steps = ARGS.steps * ARGS.segments
    if ARGS.mode == "run":
        state, masses = _ic()
        first = _FirstEdge()
        runner = _runner(callbacks=(first,))
        begun = time.perf_counter()
        res = runner.run(state, masses, dt=ARGS.dt, num_steps=num_steps)
        result["wall_s"] = time.perf_counter() - begun
        from jaccpot.distributed.mesh_runner import gather_by_id

        final = gather_by_id(res.handle.state, ("state", "acceleration"))
        result["peak_bytes"] = _peaks()
        counts = res.edges[-1].plan["mesh"]["counts"]
        result["peak_bytes_per_particle"] = [
            b / max(c, 1) for b, c in zip(result["peak_bytes"], counts)
        ]
        result["edges"] = [
            dict(
                step=e.step,
                segment=e.segment,
                retries=e.retries,
                wall_s=e.wall_s,
                diagnostics=e.diagnostics,
                plan={k: v for k, v in e.plan.items() if k != "walk_caps"},
            )
            for e in res.edges
        ]
        result["restarts"] = res.restarts
        # the kept runs' bits (a retried segment's failures are in its report)
        result["flag_clean"] = all(int(r.bits) == 0 for r in res.handle.reports)
        result["retries"] = [int(r.retries) for r in res.handle.reports]
        result["reports"] = [dataclasses.asdict(r) for r in res.handle.reports]
        np.save(os.path.join(ARGS.out_dir, f"state_{ARGS.tag}.npy"), res.state)
        np.save(
            os.path.join(ARGS.out_dir, f"acc_{ARGS.tag}.npy"), final["acceleration"]
        )
        kernel = runner.engine.softening_kernel
        ids = np.sort(np.random.default_rng(12345).choice(ARGS.n, 512, replace=False))
        del res, runner
        result["rel_l2_first_edge"] = _rel_l2(
            first.kept["acceleration"], first.kept["state"][:, 0], masses, kernel, ids
        )
        result["rel_l2_end"] = _rel_l2(
            final["acceleration"], final["state"][:, 0], masses, kernel, ids
        )
        print("rel-L2", result["rel_l2_first_edge"], result["rel_l2_end"], flush=True)
    else:
        path = os.path.join(ARGS.out_dir, f"restart_a_{ARGS.from_segment}.npz")
        runner = _runner()
        begun = time.perf_counter()
        res = runner.run(dt=ARGS.dt, num_steps=num_steps, restart_from=path)
        result["wall_s"] = time.perf_counter() - begun
        result["restored"] = res.restored
        result["edges"] = [dict(step=e.step, retries=e.retries) for e in res.edges]
        from jaccpot.distributed.mesh_runner import gather_by_id

        acc = gather_by_id(res.handle.state, ("acceleration",))["acceleration"]
        want_state = np.load(os.path.join(ARGS.out_dir, "state_a.npy"))
        want_acc = np.load(os.path.join(ARGS.out_dir, "acc_a.npy"))
        result["resume_vs_a"] = {
            "state": _differ(res.state, want_state),
            "acceleration": _differ(acc, want_acc),
        }
        control = os.path.join(ARGS.out_dir, "state_b.npy")
        if os.path.exists(control):
            result["control_b_vs_a"] = {
                "state": _differ(np.load(control), want_state),
                "acceleration": _differ(
                    np.load(os.path.join(ARGS.out_dir, "acc_b.npy")), want_acc
                ),
            }
        print(json.dumps({k: result[k] for k in result if "vs" in k}), flush=True)
    result["smi_after"] = _smi()
    out = os.path.join(ARGS.out_dir, f"{ARGS.mode}_{ARGS.tag}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)
    print(
        json.dumps({k: v for k, v in result.items() if "smi" not in k}, default=str)[
            :6000
        ]
    )


if __name__ == "__main__":
    main()

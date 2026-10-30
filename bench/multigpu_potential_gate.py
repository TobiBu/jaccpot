"""S11 session 2 track E on cards: the mesh potential against fp64 direct and one card.

Two modes, each in its OWN process (the peak-memory statistic is a per-process
high-water mark, and XLA reads its flags once):

* ``--mode mesh`` (2 cards): ``decompose`` + ``MeshSegmentRunner.fused`` (library
  cross capacities, ``preset="galaxy"``) and its start force, then the mesh potential
  (``FusedMeshForce.potential_shard`` inside a ``shard_map``, the runner's carried
  ``f_b`` under ``dehnen_error``) at the same positions. Reports the potential's
  rel-L2 and max error against an fp64 direct sum (the solver's pair kernel) on 512
  fixed ids, ``W = 1/2 sum m phi``, both programs' times (the median of
  ``--repeats``, after the first call) and temporaries
  (``compiled.memory_analysis()``), the per-device peak after each phase, and an
  A-vs-A control (the program run again: elements that differ); dumps the positions
  and ``phi`` by id.
* ``--mode solo`` (1 card): the single card on the dumped positions -- its force
  (``prov.step`` from the caps-only prepare's seed) for ``f_b``, then its potential
  (a rebuild with that ``f_b`` and ``FusedForceProvider.potential``, the runner's
  energy program) -- scored on the same ids, with its ``W``.

The ICs, solver and capacities are ``bench/multigpu_segment_gate.py``'s (Plummer
from the Odisseo harness's ``common.ic``, the disc + bulge prefix of the 25M IC).

    eval $(autocvd -q -l -x 1 -n 2 -e -t 120)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python bench/multigpu_potential_gate.py \\
        --mode mesh --ic plummer --n 2000000 --out OUT.json --dump DUMPDIR

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
    p.add_argument("--mode", choices=("mesh", "solo"), required=True)
    p.add_argument("--ic", choices=("plummer", "disc"), default="plummer")
    p.add_argument("--n", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dump", required=True, help="positions and phi by id")
    p.add_argument(
        "--mac",
        choices=("dehnen_error", "dehnen"),
        default="dehnen_error",
        help="the galaxy preset's eq (16a) (default) or the geometric theta 0.8",
    )
    p.add_argument("--repeats", type=int, default=7)
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


def _ic() -> tuple:
    """``(pos, mass, G, softening)``: ``bench/multigpu_segment_gate.py``'s ICs."""
    if ARGS.ic == "plummer":
        from common.ic import IC_GENERATORS

        pos, mass = IC_GENERATORS["plummer"](ARGS.n, seed=ARGS.seed)
        return (
            np.asarray(pos, np.float32),
            np.asarray(mass, np.float32),
            1.0,
            1e-7,
        )
    ic = np.load(_DISC_IC)
    n_all = int(ic["state0"].shape[0])
    pos = np.asarray(ic["state0"][: ARGS.n, 0], np.float32)
    mass = np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n)
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(ARGS.n / 1e5))
    return pos, mass.astype(np.float32), g, soft


def _solver(g: float, soft: float):
    from jaccpot import FastMultipoleMethod

    advanced = FastMultipoleMethod(preset="galaxy").advanced
    if ARGS.mac == "dehnen":
        advanced = dataclasses.replace(advanced, mac_type="dehnen")
    solver = FastMultipoleMethod(
        preset="galaxy", G=g, softening=soft, advanced=advanced
    )
    assert bool(solver._impl._flat_walk_criterion_active()) == (
        ARGS.mac == "dehnen_error"
    )
    return solver


def _direct_potential_fp64(pos, mass, idx, g: float, soft: float, kernel: str):
    """fp64 ``-G sum_{j != i} m_j psi(r_ij)`` at the targets ``idx``, the solver's
    pair kernel (``_direct_fp64`` of ``bench/multigpu_segment_gate.py``, potential)."""
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
            psi = pair_factors(
                jnp.where(r2 > 0, r2, 1.0), params, kernel, potential=True
            )[1]
            w = jnp.where(tid[:, None] == sid[c][None, :], 0.0, psi * sm[c][None, :])
            return acc - jnp.sum(w, axis=1)

        return jax.lax.fori_loop(0, sp.shape[0], body, jnp.zeros(tp.shape[:1]))

    out = []
    for b in range(0, len(idx), 128):
        ib = jnp.asarray(idx[b : b + 128])
        out.append(np.asarray(block(p[ib], ib)))
    return g * np.concatenate(out)


def _score(phi_by_id, pos, mass, g, soft, kernel, ids) -> dict:
    ref = _direct_potential_fp64(pos, mass, ids, g, soft, kernel)
    got = np.asarray(phi_by_id, np.float64)[ids]
    return dict(
        rel_l2=float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
        max_rel=float(np.max(np.abs(got - ref) / np.abs(ref))),
        w_self=float(0.5 * np.sum(np.asarray(mass, np.float64) * phi_by_id)),
    )


def _peaks() -> list:
    return [
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.local_devices()
    ]


def _checkpoint(result: dict, tag: str) -> None:
    stats = [d.memory_stats() or {} for d in jax.local_devices()]
    result.setdefault("memory", {})[tag] = dict(
        peak=[int(st.get("peak_bytes_in_use", 0)) for st in stats],
        in_use=[int(st.get("bytes_in_use", 0)) for st in stats],
    )


def _temps(lowered) -> dict:
    """A program's ``memory_analysis()`` (per device), from a fresh compile."""
    ma = lowered.compile().memory_analysis()
    if ma is None:
        return {}
    return {
        k: int(getattr(ma, k))
        for k in (
            "temp_size_in_bytes",
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if hasattr(ma, k)
    }


def _timed(call, repeats: int) -> list:
    out = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        out.append(1e3 * (time.perf_counter() - t0))
    return out


def _potential_program(force, mesh, axis: str, with_scale: bool):
    """The mesh potential inside a ``shard_map``: ``(phi flat, bits per device)``."""
    from jax.sharding import PartitionSpec as P

    spec = P(axis)

    def body(x, m, count, ids, *fb):
        live = jnp.minimum(count[0], int(x.shape[0]))
        phi, bits = force.potential_shard(x, m, live, ids, *fb)
        return phi, bits.reshape(1)

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(spec,) * (5 if with_scale else 4),
            out_specs=(spec, spec),
            check_vma=False,
        )
    )


def main() -> None:
    from jaccpot.runtime.fused_topology import overflow_ok

    devices = jax.devices()
    assert devices[0].platform == "gpu", devices
    assert len(devices) >= NDEV, devices
    pos0, mass0, g, soft = _ic()
    ids = np.sort(np.random.default_rng(12345).choice(ARGS.n, 512, replace=False))
    os.makedirs(ARGS.dump, exist_ok=True)
    result: dict = dict(
        mode=ARGS.mode,
        mac=ARGS.mac,
        ic=ARGS.ic,
        n=ARGS.n,
        ndev=NDEV,
        cap=CAP,
        softening=soft,
        G=g,
        devices=[str(d) for d in devices[:NDEV]],
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_flags=os.environ.get("XLA_FLAGS"),
        smi_before=_smi(),
    )

    if ARGS.mode == "solo":
        from jaccpot import FusedForceProvider

        solver = _solver(g, soft)
        kernel = solver._impl.softening_kernel
        pos = jnp.asarray(np.load(os.path.join(ARGS.dump, "positions.npy")))
        m = jnp.asarray(mass0)
        prov, topology = FusedForceProvider.prepare(
            solver, pos, m, leaf_size=64, max_order=6, theta=0.8, caps_only=True
        )
        aux = None
        if prov.scale_carry:
            perm = jnp.asarray(topology.prepared.stub.tree.particle_indices)
            fb_sorted = jnp.asarray(solver._impl._last_force_scale_particles)
            aux = (jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted), None, None)
        del topology
        step = jax.jit(prov.step)
        acc, (f_b, bits_f, _) = step(pos, m, None, aux)
        jax.block_until_ready(acc)
        _checkpoint(result, "after_force")
        force_ms = _timed(lambda: step(pos, m, None, aux)[0], ARGS.repeats)

        @jax.jit
        def energy(x, masses, fb):
            topo = prov.rebuild(x, masses, force_scale=fb)
            return prov.potential(topo), prov.capacity_ok(topo)

        phi, ok = energy(pos, m, f_b)
        jax.block_until_ready(phi)
        _checkpoint(result, "after_potential")
        pot_ms = _timed(lambda: energy(pos, m, f_b)[0], ARGS.repeats)
        result["peak_bytes"] = _peaks()
        result.update(
            force_ok=bool(overflow_ok(bits_f)),
            potential_ok=bool(ok),
            force_ms=force_ms,
            potential_ms=pot_ms,
            leaf_capacity=int(solver._impl._tree_leaf_capacity),
            **_score(np.asarray(phi), np.asarray(pos), mass0, g, soft, kernel, ids),
        )
    else:
        from yggdrax.distributed.sharding import make_mesh

        from jaccpot.distributed.rollout import decompose
        from jaccpot.distributed.segment import MeshSegmentRunner, MeshState

        mesh = make_mesh(NDEV)
        axis = mesh.axis_names[0]
        vel0 = np.zeros_like(pos0)
        shard, _ = decompose(mesh, pos0, vel0, mass0, cap=CAP, num_samples=256)
        solver = _solver(g, soft)
        kernel = solver._impl.softening_kernel
        runner = MeshSegmentRunner.fused(
            solver,
            mesh,
            MeshState.from_shard_state(shard, mesh),
            leaf_size=64,
            max_order=6,
            theta=0.8,
            cross_caps="auto",
        )
        del shard
        _checkpoint(result, "after_setup")
        start = runner.start()
        jax.block_until_ready(runner.state.acceleration)
        _checkpoint(result, "after_force")
        result["start_report"] = dataclasses.asdict(start)
        result["leaf_capacity"] = int(runner.force.setup.leaf_capacity)
        result["cross_caps"] = {
            k: v
            for k, v in vars(runner.force.cross_caps).items()
            if isinstance(v, (int, float)) or v is None
        }
        force = runner.force
        st = runner.state
        scale = runner._scale_args(st)
        force_args = (st.state, st.masses, st.count, st.ids, *scale, None)
        start_program, _ = runner._program("start", 0, None)
        prog = _potential_program(force, mesh, axis, bool(scale))
        pos_flat = jax.jit(lambda s: s[:, 0])(st.state)
        pot_args = (pos_flat, st.masses, st.count, st.ids, *scale)
        with force.context():
            force.before_trace()
            t0 = time.perf_counter()
            phi, bits = prog(*pot_args)
            jax.block_until_ready(phi)
            result["potential_first_call_s"] = time.perf_counter() - t0
            _checkpoint(result, "after_potential")
            bits_all = int(np.bitwise_or.reduce(np.asarray(bits)))
            result["potential_bits"] = bits_all
            result["potential_ok"] = bool(overflow_ok(jnp.uint32(bits_all)))
            result["peak_bytes"] = _peaks()
            result["potential_ms"] = _timed(lambda: prog(*pot_args)[0], ARGS.repeats)
            # A-vs-A: the same program again, elementwise
            again = np.asarray(prog(*pot_args)[0])
            result["potential_repeat_differ"] = int(np.sum(again != np.asarray(phi)))
            del again
            result["force_ms"] = _timed(
                lambda: start_program(*force_args)[0], ARGS.repeats
            )
            force.before_trace()
            result["potential_temps"] = _temps(prog.lower(*pot_args))
            force.before_trace()
            result["force_temps"] = _temps(start_program.lower(*force_args))
        rows = np.concatenate(
            [
                np.arange(d * CAP, d * CAP + int(c))
                for d, c in enumerate(np.asarray(st.count))
            ]
        )
        got_ids = np.asarray(st.ids)[rows]
        phi_by_id = np.zeros(ARGS.n, np.float32)
        phi_by_id[got_ids] = np.asarray(phi)[rows]
        pos_by_id = np.zeros((ARGS.n, 3), np.float32)
        pos_by_id[got_ids] = np.asarray(st.state[:, 0])[rows]
        pad = np.setdiff1d(np.arange(NDEV * CAP), rows)
        result["padding_phi_zero"] = bool(np.all(np.asarray(phi)[pad] == 0.0))
        np.save(os.path.join(ARGS.dump, "positions.npy"), pos_by_id)
        np.save(os.path.join(ARGS.dump, "phi.npy"), phi_by_id)
        result["counts"] = [int(c) for c in np.asarray(st.count)]
        del runner, st, phi
        result.update(_score(phi_by_id, pos_by_id, mass0, g, soft, kernel, ids))
    result["peak_bytes_per_particle"] = [
        b / max(1, ARGS.n // NDEV) for b in result["peak_bytes"]
    ]
    result["smi_after"] = _smi()
    print(json.dumps({k: v for k, v in result.items() if "smi" not in k}, indent=1))
    json.dump(result, open(ARGS.out, "w"), indent=1)


if __name__ == "__main__":
    main()

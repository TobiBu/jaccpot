"""Reverse-mode gradients of the fused multi-GPU force: one row of WP-G's G5 matrix.

usage: python bench/multigpu_grad_gate.py --n N | --per-card NPC [--ndev K]
       [--mode mesh|solo] [--ic plummer|disc|cube] [--seed S] [--galaxy]
       [--order P] [--leaf L] [--leaf-capacity auto|INT] [--theta T]
       [--softening EPS] [--kernel K] [--dtype float32|float64] [--params]
       [--cross auto|default] [--reps R] [--grad-only] [--no-forward] [--no-time]
       [--check] [--cpu-smoke] --out OUT.json

One process, one row (XLA reads its flags once; the device peak is a per-process
high-water mark), on the FIRST ``K`` devices. ``--mode mesh`` (the default): the
particles placed on a ``K``-device mesh as Morton domains (``rollout.decompose``,
``cap = N / K * --cap-factor``; ``K = 1``: a one-device mesh, ``cap = N``), the
template-route force (``FusedMeshForce``: caps-only prepares, the library's cross
sizing ``cross_caps="auto"`` from a measured start force) and its gradient
(``jaccpot.distributed.grad.make_mesh_grad_fn``, the loss ``sum(w . acc)`` built
OUTSIDE the ``shard_map``). ``--mode solo``: the single card's provider path
(``FusedForceProvider.prepare(differentiable=True)``: ``step`` against
``differentiable_step``, as ``bench/grad_rollout_timing.py``), the 1-card column.

The operating point: ``--galaxy`` is ``preset="galaxy"`` (``dehnen_error`` 1e-4,
p6, leaf 64, theta 0.8, the ``auto`` leaf capacity; ``f_b`` seeded by the prepare
as the mesh runner seeds it: ``FusedMeshForce.force_scale_seed``); without it the
G1 pin's geometric point (``large_n_gpu``, ``mac_type="dehnen"``, ``--order``
default 5, ``--theta`` 0.8). Under ``dehnen_error`` the force accepts only the
engine's own ``G`` / ``eps2``, so ``--params`` (``G`` and ``eps2`` as traced
``ForceParams``, differentiated too) is refused with ``--galaxy``.

Recorded to ``--out`` (written after every phase; ``complete`` at the end):

* MG1: the differentiable forward (the gradient's user-facing function) against
  the production force (``FusedMeshForce.shard``; solo: ``step``), elements whose
  bits differ on the live rows (expected 0);
* the verdict bits of the production force and of the gradient program (its aux),
  ``bits_ok`` / ``grad_ok``; the cross far / near pair needs per device from the
  production force's needs vector (``export_far_cap`` / ``export_near_cap``:
  non-empty on >= 2 devices) and the particles it shipped; the gradient finite;
* compile seconds and ``memory_analysis()`` of every program (per device:
  arguments, outputs, temporaries; per particle of the fullest card) and the
  largest arrays the gradient program defines;
* per-device ``peak_bytes_in_use`` and ``bytes_in_use`` at each phase
  (``prepared``, ``forward``, ``grad``, ``timed``) and the peak in bytes per live
  particle of each card (the ``grad`` phase's: after one gradient call);
* timings: the production force, ``value_and_grad`` (and ``grad`` alone, which
  drops the near field's forward: the loss is linear in it), min / median of
  ``--reps`` warm calls each, interleaved twice as ``bench/grad_production_gate.py``;
  the ratios of the minima. NOT a claim unless the cards were free: ``nvidia-smi``'s
  other compute processes on this row's cards before and after are recorded
  (``contended``);
* A-vs-A: two gradient calls, elements whose bits differ (per gradient leaf);
* ``--check`` (small N): the gradient in x and m by global id against the single
  card's S8 gradient on the mesh's Morton box (``FusedForceProvider`` rebuild +
  ``differentiable_accelerations``) and against an fp64 direct sum restricted to
  ``--check-targets`` weighted targets (all sources; the loss's weights live on
  those targets only, for all three), rel-L2 and the error ratio mesh / single;
  with ``--params`` MG9 (``G * G_bar == sum(m * m_bar)``).

Flags: the fusion autotuner off and, on two or more devices, the NCCL ragged-exchange
flags (``bench/_ragged_exchange_flags.py``) are added to ``XLA_FLAGS`` before jax
starts unless the caller set them; a run on >= 2 GPUs whose flags differ from the
library's (an override) is refused unless ``--allow-flag-override``.

``--cpu-smoke``: forced host devices (``K``, at least 2), the S9 interpret set of
``tests/distributed/test_mesh_grad_gates.py`` and its ``default_backend`` / ``buf``
exchange patches; for tiny N only (interpret-mode reverse kernels compile for
minutes and ~10 GB per program):

    JAX_PLATFORMS=cpu python bench/multigpu_grad_gate.py --cpu-smoke --n 2048 \\
        --ndev 2 --ic cube --order 4 --leaf-capacity 128 --cross default \\
        --softening 1e-2 --params --check --check-targets 256 --reps 2 --out OUT.json

ICs: ``plummer`` (Odisseo's ``common.ic.IC_GENERATORS`` under ``BENCH_DIR``, the
scaling matrix's; an in-file sampler, recorded, if it is not importable), ``disc``
(the 25M disc + bulge, ``--disc-ic`` / ``DISC_IC``: rows ``[seed * N, (seed + 1) *
N)`` of the shuffled file, masses rescaled to the whole galaxy; ``N <= 25M / (seed
+ 1)``), ``cube`` (the CPU gates' uniform cube with unequal masses). Default
softening: plummer 1e-3, disc the segment gate's formula, cube 1e-2.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _ragged_exchange_flags import add_ragged_exchange_flags  # noqa: E402

AUTOTUNER_OFF = "--xla_gpu_experimental_enable_fusion_autotuner=false"
#: the 25M disc+bulge IC; DISC_IC (or --disc-ic) on any other machine
_DISC_IC = os.environ.get(
    "DISC_IC", "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
)
_BENCH_DIR = os.environ.get(
    "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
)
#: tests/distributed/test_mesh_grad_gates.py's CPU set (the S9 lane interpreted, the
#: GPU's target-sorted symmetric walk interpreted)
SMOKE_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
SMOKE_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_CROSS_WALK",
)
#: the largest N ``--check`` runs at (the single card's S8 gradient and the direct
#: sum on one card; --force-check lifts it)
CHECK_N_MAX = 2_000_000


def _args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--n", type=float, help="total particles")
    size.add_argument("--per-card", type=float, help="particles per card (x ndev)")
    p.add_argument("--ndev", type=int, default=1, help="the first K devices")
    p.add_argument("--mode", choices=("mesh", "solo"), default="mesh")
    p.add_argument("--ic", choices=("plummer", "disc", "cube"), default="plummer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disc-ic", default=_DISC_IC)
    p.add_argument("--galaxy", action="store_true", help='preset="galaxy" (p6)')
    p.add_argument("--order", type=int, default=None, help="5 (6 with --galaxy)")
    p.add_argument("--leaf", type=int, default=64)
    p.add_argument("--leaf-capacity", default="auto", help="'auto' or an int")
    p.add_argument("--theta", type=float, default=0.8)
    p.add_argument("--softening", type=float, default=None)
    p.add_argument("--kernel", default=None, help="softening kernel (solver's)")
    p.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--params", action="store_true", help="also d/dG and d/deps2")
    p.add_argument(
        "--cross",
        choices=("auto", "default"),
        default="auto",
        help="cross capacities: the library's sizing (auto) or CrossCapacities' "
        "defaults with max_cells the leaf capacity (the CPU gates'; needs an int "
        "--leaf-capacity)",
    )
    p.add_argument("--cap-factor", type=float, default=1.15, help="mesh rows / (N/K)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument(
        "--grad-only",
        action="store_true",
        help="compile and run jax.grad only (no value_and_grad): the memory rows",
    )
    p.add_argument("--no-forward", action="store_true", help="skip MG1")
    p.add_argument("--no-time", action="store_true", help="no timings")
    p.add_argument("--check", action="store_true", help="vs single card + direct")
    p.add_argument("--check-targets", type=int, default=2048)
    p.add_argument(
        "--force-check", action="store_true", help=f"--check past {CHECK_N_MAX:.0e}"
    )
    p.add_argument("--allow-flag-override", action="store_true")
    p.add_argument("--cpu-smoke", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    args.n = int(args.n if args.n is not None else args.per_card * args.ndev)
    if args.mode == "solo" and args.ndev != 1:
        p.error("--mode solo runs on one card (--ndev 1)")
    if args.galaxy and args.params:
        p.error("--params with --galaxy: dehnen_error accepts only the engine's G/eps2")
    if args.galaxy and args.order not in (None, 6):
        p.error("--galaxy is p6")
    args.order = args.order or (6 if args.galaxy else 5)
    if args.leaf_capacity != "auto":
        args.leaf_capacity = int(args.leaf_capacity)
    if args.cross == "default" and args.leaf_capacity == "auto":
        p.error("--cross default needs an int --leaf-capacity (max_cells)")
    if args.check and args.mode == "solo":
        p.error("--check compares the mesh's gradient: --mode mesh")
    if args.check and args.n > CHECK_N_MAX and not args.force_check:
        p.error(
            f"--check at N > {CHECK_N_MAX:.0e} (single card + direct sum): --force-check"
        )
    if args.ic == "disc" and args.n * (args.seed + 1) > 25_165_824:
        p.error("the disc IC has 25,165,824 rows: N * (seed + 1) must fit")
    return args


def prepare_env(args: argparse.Namespace, env: Optional[dict] = None) -> None:
    """``XLA_FLAGS`` and the lane's environment, before jax starts.

    Parameters
    ----------
    args : argparse.Namespace
        The row.
    env : dict or None
        The environment to edit; ``None`` edits ``os.environ``.
    """
    env = os.environ if env is None else env
    flags = env.get("XLA_FLAGS", "")
    if AUTOTUNER_OFF.split("=")[0] not in flags:
        env["XLA_FLAGS"] = f"{flags} {AUTOTUNER_OFF}".strip()
    if args.ndev >= 2 or args.cpu_smoke:
        add_ragged_exchange_flags(env)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.cpu_smoke:
        for key in SMOKE_ENV_UNSET:
            env.pop(key, None)
        env.update(SMOKE_ENV)
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
        if "xla_force_host_platform_device_count" not in env["XLA_FLAGS"]:
            count = max(args.ndev, 2)
            env["XLA_FLAGS"] += f" --xla_force_host_platform_device_count={count}"


def _patch_cpu_lane() -> None:
    """The CPU gates' patches: the large-N GPU profile, the ``buf`` exchange."""
    import jax
    import yggdrax.distributed.comm as ygg_comm

    from jaccpot.runtime import _level_shapes

    jax.default_backend = lambda: "gpu"
    resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = lambda method="auto": (
        "buf" if method == "auto" else resolve(method)
    )
    _level_shapes._WIDTHS = {}
    _level_shapes._LEVELS = {}


# --------------------------------------------------------------------------
# The host side: ICs, nvidia-smi.
# --------------------------------------------------------------------------


def _plummer_fallback(n: int, seed: int) -> tuple:
    """A G = M = a = 1 Plummer sphere (unclipped), equal masses."""
    rng = np.random.default_rng(seed)
    r = 1.0 / np.sqrt(rng.uniform(1e-12, 1.0, n) ** (-2.0 / 3.0) - 1.0)
    v = rng.normal(size=(n, 3))
    pos = r[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
    return pos, np.full(n, 1.0 / n)


def make_ic(args: argparse.Namespace) -> dict:
    """``dict(pos, mass, G, softening, source)``: float64 host arrays."""
    n, seed = args.n, args.seed
    if args.ic == "plummer":
        try:
            sys.path.insert(0, _BENCH_DIR)
            from common.ic import IC_GENERATORS

            pos, mass = IC_GENERATORS["plummer"](n, seed=seed)
            source = f"IC_GENERATORS[plummer] ({_BENCH_DIR})"
        except ImportError:
            pos, mass = _plummer_fallback(n, seed)
            source = "in-file Plummer sampler (common.ic not importable)"
        soft, g = 1e-3, 1.0
    elif args.ic == "cube":
        rng = np.random.default_rng(seed)
        pos = rng.uniform(-1.0, 1.0, (n, 3))
        mass = rng.uniform(0.5, 1.5, n) / n
        soft, g, source = 1e-2, 1.0, "uniform cube, masses U(0.5, 1.5) / N"
    else:
        ic = np.load(args.disc_ic, mmap_mode="r")
        n_all = int(ic["state0"].shape[0])
        lo = seed * n
        pos = np.asarray(ic["state0"][lo : lo + n, 0], np.float64)
        mass = np.asarray(ic["mass"][lo : lo + n], np.float64) * (n_all / n)
        g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
        soft = float(0.5 * float(ic["rdisk_code"]) / np.sqrt(n / 1e5))
        source = f"{args.disc_ic} rows [{lo}, {lo + n})"
    if args.softening is not None:
        soft = args.softening
    return dict(
        pos=np.asarray(pos, np.float64),
        mass=np.asarray(mass, np.float64),
        G=g,
        softening=soft,
        source=source,
    )


def _visible_cards(ndev: int) -> list:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cards = [c for c in visible.split(",") if c] if visible else []
    return cards[:ndev] if cards else [str(i) for i in range(ndev)]


def card_processes(cards: list) -> list:
    """Compute processes other than this one on ``cards`` (nvidia-smi indices)."""
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        uuids = {}
        for line in q.splitlines():
            idx, uuid = (x.strip() for x in line.split(","))
            if idx in cards:
                uuids[uuid] = idx
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return [f"nvidia-smi failed: {exc!r}"]
    out = []
    for line in apps.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 3 and parts[1] in uuids and parts[0] != str(os.getpid()):
            out.append(dict(pid=int(parts[0]), card=uuids[parts[1]], mib=parts[2]))
    return out


# --------------------------------------------------------------------------
# The device side.
# --------------------------------------------------------------------------


def _solver(args: argparse.Namespace, g: float, soft: float):
    import dataclasses

    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod
    from jaccpot.config import (
        FarFieldConfig,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    extra = {} if args.kernel is None else {"softening_kernel": args.kernel}
    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    if args.galaxy:
        advanced = FastMultipoleMethod(preset="galaxy").advanced
        if args.leaf_capacity != "auto":
            advanced = dataclasses.replace(
                advanced,
                tree=dataclasses.replace(
                    advanced.tree,
                    leaf_capacity=int(args.leaf_capacity),
                    leaf_headroom=None,
                ),
            )
        return FastMultipoleMethod(
            preset="galaxy",
            G=g,
            softening=soft,
            working_dtype=dtype,
            advanced=advanced,
            **extra,
        )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=args.theta,
        G=g,
        softening=soft,
        working_dtype=dtype,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=args.leaf,
                leaf_partition="cells",
                leaf_capacity=args.leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            farfield=FarFieldConfig(m2l_chunk_size=512),
            mac_type="dehnen",
        ),
        fixed_order=args.order,
        **extra,
    )


def _memory_now(devices: list) -> dict:
    stats = [d.memory_stats() or {} for d in devices]
    return dict(
        peak=[int(s.get("peak_bytes_in_use", 0)) for s in stats],
        in_use=[int(s.get("bytes_in_use", 0)) for s in stats],
    )


def _memory_analysis(compiled: Any, per: int, largest: bool) -> Optional[dict]:
    try:
        ma = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001 - not every backend reports it
        return dict(error=repr(exc))
    if ma is None:
        return None
    out = {
        "argument_bytes": int(ma.argument_size_in_bytes),
        "output_bytes": int(ma.output_size_in_bytes),
        "temp_bytes": int(ma.temp_size_in_bytes),
        "alias_bytes": int(getattr(ma, "alias_size_in_bytes", 0)),
    }
    total = out["argument_bytes"] + out["output_bytes"] + out["temp_bytes"]
    out["total_B_per_p"] = total / max(per, 1)
    out["temp_B_per_p"] = out["temp_bytes"] / max(per, 1)
    if largest:
        from grad_production_gate import _largest_arrays

        try:
            out["largest_arrays"] = _largest_arrays(compiled.as_text(), max(per, 1))
        except Exception as exc:  # noqa: BLE001
            out["largest_arrays"] = repr(exc)
    return out


def _differing(a: Any, b: Any) -> int:
    a, b = np.asarray(a), np.asarray(b)
    view = {4: np.uint32, 8: np.uint64, 2: np.uint16}[a.dtype.itemsize]
    return int(np.sum(a.view(view) != b.view(view)))


def _rel(a: Any, b: Any) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _bits_summary(bits: Any) -> dict:
    from jaccpot.runtime.fused_topology import GRAD_FAIL_MASK, OVERFLOW_FAIL_MASK

    b = np.asarray(bits).astype(np.uint64).reshape(-1)
    return dict(
        bits=b.tolist(),
        bits_ok=bool(np.all((b & OVERFLOW_FAIL_MASK) == 0)),
        grad_ok=bool(np.all((b & GRAD_FAIL_MASK) == 0)),
        bits_clean=bool(int(np.bitwise_or.reduce(b)) == 0),
    )


def _direct_grads(pos, mass, targets, w_t, g, eps2, kernel, chunk=64):
    """fp64 direct-sum gradient of ``sum_k w_k . a_k`` over the ``targets``.

    All sources; chunked over the targets (a ``[chunk, N, 3]`` pair block at a
    time). Returns ``(L, d/dx, d/dm)`` as host float64.
    """
    import jax
    import jax.numpy as jnp

    from jaccpot.softening import masked_pair_factors, softening_params_from_sq

    x = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)

    @jax.jit
    def block(x, m, tj, wt):
        def loss(x, m):
            xt = x[tj]
            d = x[None, :, :] - xt[:, None, :]
            r2 = jnp.sum(d * d, -1)
            params = softening_params_from_sq(kernel, eps2, x.dtype)
            inv = masked_pair_factors(r2, r2 > 0, params, kernel)[0]
            acc = g * jnp.einsum(
                "ij,ijk->ik", m[None, :] * inv, d, precision=jax.lax.Precision.HIGHEST
            )
            return jnp.sum(acc * wt)

        return jax.value_and_grad(loss, argnums=(0, 1))(x, m)

    total, gx, gm = 0.0, np.zeros(pos.shape), np.zeros(mass.shape)
    for lo in range(0, len(targets), chunk):
        tj = np.zeros(chunk, np.int64)
        wt = np.zeros((chunk, 3))
        part = targets[lo : lo + chunk]
        tj[: len(part)] = part
        wt[: len(part)] = w_t[lo : lo + chunk]
        val, (bx, bm) = block(x, m, jnp.asarray(tj), jnp.asarray(wt))
        total += float(val)
        gx += np.asarray(bx)
        gm += np.asarray(bm)
    return total, gx, gm


class _Row:
    """The JSON record, written after every phase."""

    def __init__(self, path: str, record: dict) -> None:
        self.path, self.r = path, record

    def save(self) -> None:
        Path(self.path).write_text(json.dumps(self.r, indent=1, default=_jsonable))

    def phase(self, name: str, devices: list) -> None:
        self.r.setdefault("memory", {})[name] = _memory_now(devices)
        self.r["phase"] = name
        self.save()
        print(
            f"[{time.time() - _T0:8.1f}s] {name}",
            json.dumps(self.r["memory"][name]),
            flush=True,
        )


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return repr(x)


_T0 = time.time()


def _log(*a: Any) -> None:
    print(f"[{time.time() - _T0:8.1f}s]", *a, flush=True)


def _compile(fn: Any, args: tuple, ctx: Any, before: Any) -> tuple:
    with ctx():
        before()
        t0 = time.perf_counter()
        compiled = fn.lower(*args).compile()
    return compiled, time.perf_counter() - t0


def _timed(fn: Any, args: tuple, reps: int, ctx: Any) -> tuple:
    import jax

    with ctx():
        jax.block_until_ready(fn(*args))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
    return 1e3 * min(ts), 1e3 * float(np.median(ts))


def _measure(
    row: _Row,
    args: argparse.Namespace,
    *,
    devices: list,
    counts: np.ndarray,
    live_rows: np.ndarray,
    ctx: Any,
    before: Any,
    prod: Any,
    pargs: tuple,
    diff: Any,
    dargs: tuple,
    loss: Any,
    gargs: tuple,
    argnums: tuple,
) -> Optional[tuple]:
    """MG1, the gradient's memory, A-vs-A and the timings (both modes).

    ``prod(*pargs) -> (acc, bits, needs, ...)``, ``diff(*dargs) -> (acc, bits,
    ...)``, ``loss(*gargs) -> (L, bits)``. Returns the first gradient's host leaves
    (for ``--check``), or ``None``.
    """
    import jax

    r = row.r
    per = int(max(counts))
    r["compile_s"] = {}
    prod_c, r["compile_s"]["force"] = _compile(jax.jit(prod), pargs, ctx, before)
    r["memory_analysis"] = {"force": _memory_analysis(prod_c, per, False)}
    with ctx():
        out = prod_c(*pargs)
    a_p = np.asarray(out[0])[live_rows]
    r["force"] = _bits_summary(out[1])
    needs = np.asarray(out[2]).reshape(len(counts), -1)
    del out
    if args.mode == "mesh" and args.ndev > 1:
        from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS
        from jaccpot.runtime.fused_topology import NEEDS_FIELDS

        base = len(NEEDS_FIELDS)
        col = {
            f: needs[:, base + CROSS_NEEDS_FIELDS.index(f)] for f in CROSS_NEEDS_FIELDS
        }
        r["cross"] = dict(
            far_pairs=col["export_far_cap"].tolist(),
            near_pairs=col["export_near_cap"].tolist(),
            shipped_particles=col["send_particle_cap"].tolist(),
            nonempty=bool(
                np.all(col["export_far_cap"] > 0) and np.all(col["export_near_cap"] > 0)
            ),
        )
        _log("cross", r["cross"])
    _log("force bits", r["force"])
    if not args.no_forward:
        diff_c, r["compile_s"]["diff_forward"] = _compile(
            jax.jit(diff), dargs, ctx, before
        )
        with ctx():
            out = diff_c(*dargs)
        a_d = np.asarray(out[0])[live_rows]
        r["MG1"] = dict(
            differing=_differing(a_d, a_p),
            elements=int(a_p.size),
            max_abs=float(np.max(np.abs(a_d.astype(np.float64) - a_p))),
            diff_bits=_bits_summary(out[1]),
        )
        _log("MG1", r["MG1"])
        del out, diff_c, a_d
    del a_p
    gc.collect()
    row.phase("forward", devices)

    # the gradient alone (the near field's forward is dead code: the loss is linear
    # in it), then value_and_grad (the main program unless --grad-only)
    go = jax.jit(jax.grad(loss, argnums=argnums, has_aux=True))
    go_c, r["compile_s"]["grad"] = _compile(go, gargs, ctx, before)
    r["memory_analysis"]["grad"] = _memory_analysis(go_c, per, True)
    if args.grad_only:
        main_c = go_c
    else:
        vg = jax.jit(jax.value_and_grad(loss, argnums=argnums, has_aux=True))
        main_c, r["compile_s"]["value_and_grad"] = _compile(vg, gargs, ctx, before)
        r["memory_analysis"]["value_and_grad"] = _memory_analysis(main_c, per, False)
    row.save()
    _log("compiled", r["compile_s"])

    def split(out: Any) -> tuple:
        if args.grad_only:
            grads, bits = out
            return None, bits, grads
        (value, bits), grads = out
        return value, bits, grads

    with ctx():
        out1 = main_c(*gargs)
        jax.block_until_ready(out1)
    row.phase("grad", devices)
    value, bits, grads = split(out1)
    r["grad_bits"] = _bits_summary(bits)
    r["L"] = None if value is None else float(value)
    host1 = [np.asarray(g) for g in jax.tree_util.tree_leaves(grads)]
    del out1, value, bits, grads
    r["grad_finite"] = bool(all(np.all(np.isfinite(g)) for g in host1))
    peak = r["memory"]["grad"]["peak"]
    r["peak_B_per_p"] = [b / max(int(c), 1) for b, c in zip(peak, counts)]
    with ctx():
        out2 = main_c(*gargs)
        host2 = [np.asarray(g) for g in jax.tree_util.tree_leaves(split(out2)[2])]
    del out2
    r["grad_AvsA_differing"] = [_differing(a, b) for a, b in zip(host1, host2)]
    del host2
    _log(
        f"grad finite {r['grad_finite']}, bits {r['grad_bits']}, "
        f"A-vs-A {r['grad_AvsA_differing']}, peak B/p {r['peak_B_per_p']}"
    )
    row.save()

    if not args.no_time:
        reps = int(args.reps)
        t = r.setdefault("time_ms", {})
        t["force"] = _timed(prod_c, pargs, reps, ctx)
        t["main"] = _timed(main_c, gargs, reps, ctx)
        t["force_b"] = _timed(prod_c, pargs, reps, ctx)
        t["main_b"] = _timed(main_c, gargs, reps, ctx)
        if not args.grad_only:
            t["grad"] = _timed(go_c, gargs, reps, ctx)
        t["main_is"] = "grad" if args.grad_only else "value_and_grad"
        force_min = min(t["force"][0], t["force_b"][0])
        main_min = min(t["main"][0], t["main_b"][0])
        r["ratio_main_over_force"] = main_min / force_min
        if not args.grad_only:
            r["ratio_value_and_grad_over_force"] = main_min / force_min
            r["ratio_grad_over_force"] = t["grad"][0] / force_min
        else:
            r["ratio_grad_over_force"] = main_min / force_min
        _log(
            "time ms",
            t,
            "ratios",
            r.get("ratio_value_and_grad_over_force"),
            r["ratio_grad_over_force"],
        )
        row.phase("timed", devices)
    return host1


def _run_mesh(row: _Row, args: argparse.Namespace, ic: dict) -> None:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from jaccpot.distributed import grad as GR
    from jaccpot.distributed.cross import CrossCapacities
    from jaccpot.distributed.fused import AXIS_NAME, host_mesh_bounds
    from jaccpot.distributed.rollout import decompose
    from jaccpot.distributed.template_force import FusedMeshForce
    from jaccpot.runtime.fused_topology import NEEDS_FIELDS, ForceParams, overflow_ok

    r, n, ndev = row.r, args.n, args.ndev
    devices = jax.devices()[:ndev]
    dtype = np.float64 if args.dtype == "float64" else np.float32
    mesh = jax.sharding.Mesh(np.asarray(devices), (AXIS_NAME,))
    cap = n if ndev == 1 else int(np.ceil(n / ndev * args.cap_factor))
    pos = ic["pos"].astype(dtype)
    mass = ic["mass"].astype(dtype)
    t0 = time.perf_counter()
    shard, _ = decompose(mesh, pos, np.zeros_like(pos), mass, cap=cap, num_samples=256)
    x0, m0, counts_d, ids_d = shard.positions, shard.masses, shard.count, shard.ids
    counts = np.asarray(counts_d).astype(np.int64)
    r.update(cap=cap, counts=counts.tolist(), decompose_s=time.perf_counter() - t0)
    solver = _solver(args, ic["G"], ic["softening"])
    caps = "auto"
    if args.cross == "default":
        caps = CrossCapacities(max_cells=int(args.leaf_capacity), leaf_width=args.leaf)
    force = FusedMeshForce(
        solver,
        mesh,
        leaf_size=args.leaf,
        max_order=args.order,
        theta=args.theta,
        cross_caps=caps,
    )
    t0 = time.perf_counter()
    force.prepare(x0, m0, counts_d)
    seed = force.force_scale_seed if force.scale_carry else None
    fb = () if seed is None else (seed,)
    spec = P(AXIS_NAME)
    if force.needs_cross_sizing:
        from jaccpot.distributed.segment import _mesh_verdict

        def measure_body(x, m, cnt, i, *f):
            out = force.shard(x, m, jnp.minimum(cnt[0], x.shape[0]), i, *f)
            needs = _mesh_verdict(
                overflow_ok(out[1]), out[1], out[2], jnp.int32(0), axis_name=AXIS_NAME
            )[2]
            return needs

        def measure() -> np.ndarray:
            prog = jax.jit(
                jax.shard_map(
                    measure_body,
                    mesh=mesh,
                    in_specs=(spec,) * (4 + len(fb)),
                    out_specs=P(),
                    check_vma=False,
                )
            )
            with force.context():
                force.before_trace()
                needs = prog(x0, m0, counts_d, ids_d, *fb)
            return np.asarray(jax.device_get(needs))[len(NEEDS_FIELDS) :]

        force.plan_cross(measure)
    force.with_grad_plan(x0, m0, counts_d)
    r["prepare_s"] = time.perf_counter() - t0
    r["scale_carry"] = bool(force.scale_carry)
    r["leaf_capacity_used"] = int(force.setup.leaf_capacity)
    r["live_leaves"] = int(force.setup.live_leaves)
    r["cross_caps"] = (
        None
        if force.cross_caps is None
        else {
            k: v
            for k, v in vars(force.cross_caps).items()
            if isinstance(v, (int, float)) or v is None
        }
    )
    if force.cross_needs_measured is not None:
        from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS

        r["cross_needs_measured"] = dict(
            zip(CROSS_NEEDS_FIELDS, np.asarray(force.cross_needs_measured).tolist())
        )
    r["grad_statics"] = str(force.grad_statics)
    row.phase("prepared", devices)

    live_rows = np.concatenate(
        [np.arange(d * cap, d * cap + counts[d]) for d in range(ndev)]
    )
    gid = np.asarray(ids_d)[live_rows]
    rng = np.random.default_rng(args.seed + 1)
    if args.check:
        k = min(args.check_targets, n)
        targets = np.sort(rng.choice(n, k, replace=False))
        w_global = np.zeros((n, 3))
        w_global[targets] = rng.normal(size=(k, 3))
    else:
        w_global = rng.normal(size=(n, 3))
    w_mesh = np.zeros((ndev * cap, 3), dtype)
    w_mesh[live_rows] = w_global[gid]
    w = jax.device_put(jnp.asarray(w_mesh), NamedSharding(mesh, spec))
    del w_mesh
    p0 = None
    if args.params:
        p0 = (jnp.asarray(ic["G"], dtype), jnp.asarray(ic["softening"] ** 2, dtype))

    def as_params(p):
        return None if p is None else ForceParams(G=p[0], eps2=p[1])

    fn = GR.make_mesh_grad_fn(force)

    def prod_body(x, m, cnt, i, *f):
        out = force.shard(x, m, jnp.minimum(cnt[0], x.shape[0]), i, *f)
        return (out[0], out[1].reshape(1), out[2].reshape(1, -1), *out[3:])

    prod = jax.shard_map(
        prod_body,
        mesh=mesh,
        in_specs=(spec,) * (4 + len(fb)),
        out_specs=(spec,) * (3 + len(fb)),
        check_vma=False,
    )

    def diff(x, m, cnt, i, p, f):
        return fn(x, m, cnt, i, as_params(p), f)

    def loss(x, m, p, cnt, i, ww, f):
        out = fn(x, m, cnt, i, as_params(p), f)
        return jnp.sum(out[0] * ww), out[1]

    argnums = (0, 1, 2) if args.params else (0, 1)
    host = _measure(
        row,
        args,
        devices=devices,
        counts=counts,
        live_rows=live_rows,
        ctx=force.context,
        before=force.before_trace,
        prod=prod,
        pargs=(x0, m0, counts_d, ids_d, *fb),
        diff=diff,
        dargs=(x0, m0, counts_d, ids_d, p0, seed),
        loss=loss,
        gargs=(x0, m0, p0, counts_d, ids_d, w, seed),
        argnums=argnums,
    )
    row.save()
    if not args.check:
        return

    # ---- --check: the single card (the mesh's box) and the fp64 direct sum
    gx, gm = host[0], host[1]
    mgx = np.zeros((n, 3))
    mgx[gid] = gx[live_rows]
    mgm = np.zeros(n)
    mgm[gid] = gm[live_rows]
    check: dict = {"targets": int(len(targets))}
    if args.params:
        g_bar = float(host[2])
        lhs = float(ic["G"]) * g_bar
        rhs = float(np.sum(np.asarray(m0, np.float64)[live_rows] * gm[live_rows]))
        check["MG9"] = dict(G_Gbar=lhs, sum_m_mbar=rhs, rel=abs(lhs - rhs) / abs(lhs))
        check["eps2_bar"] = float(host[3])
    # free the mesh arrays (rebound, not del: the cross sizing closure names them)
    host = x0 = m0 = w = None
    gc.collect()
    from jaccpot import FusedForceProvider

    box = host_mesh_bounds([pos], [n], cubic=bool(force.engine._tree_cubic_bounds))
    dev0 = devices[0]
    s_solver = _solver(args, ic["G"], ic["softening"])
    xs = jax.device_put(jnp.asarray(pos), dev0)
    ms = jax.device_put(jnp.asarray(mass), dev0)
    t0 = time.perf_counter()
    single, topo = FusedForceProvider.prepare(
        s_solver,
        xs,
        ms,
        leaf_size=args.leaf,
        max_order=args.order,
        theta=args.theta,
        bounds=box,
        differentiable=True,
    )
    s_seed = None
    if single.scale_carry:
        fb_sorted = jnp.asarray(s_solver._impl._last_force_scale_particles, xs.dtype)
        perm = jnp.asarray(topo.prepared.tree.particle_indices)
        s_seed = jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted)
    del topo
    wg = jax.device_put(jnp.asarray(w_global, dtype), dev0)

    def single_loss(x, m, p, ww, f):
        stop = jax.lax.stop_gradient
        kw = {} if f is None else {"force_scale": stop(f)}
        topo = single.rebuild(stop(x), stop(m), bounds=box, keep_far_list=True, **kw)
        plan, _ = single.grad_plan(topo)
        out = single.differentiable_accelerations(
            topo,
            x,
            m,
            as_params(p),
            plan=plan,
            **({} if f is None else {"force_scale": True}),
        )
        acc = out if f is None else out[0]
        return jnp.sum(acc * ww)

    sp0 = None if p0 is None else tuple(jax.device_put(a, dev0) for a in p0)
    s_l, s_grads = jax.jit(jax.value_and_grad(single_loss, argnums=(0, 1)))(
        xs, ms, sp0, wg, s_seed
    )
    sgx, sgm = np.asarray(s_grads[0]), np.asarray(s_grads[1])
    check["single_s"] = time.perf_counter() - t0
    single = s_grads = xs = ms = wg = None
    gc.collect()
    t0 = time.perf_counter()
    kernel = getattr(force.engine, "softening_kernel", None)
    with jax.default_device(dev0):
        d_l, dgx, dgm = _direct_grads(
            ic["pos"].astype(dtype).astype(np.float64),
            ic["mass"].astype(dtype).astype(np.float64),
            targets,
            w_global[targets],
            float(ic["G"]),
            float(ic["softening"]) ** 2,
            kernel,
        )
    check["direct_s"] = time.perf_counter() - t0
    check.update(
        L=[r.get("L"), float(s_l), d_l],
        x_mesh_vs_direct=_rel(mgx, dgx),
        x_single_vs_direct=_rel(sgx, dgx),
        x_mesh_vs_single=_rel(mgx, sgx),
        m_mesh_vs_direct=_rel(mgm, dgm),
        m_single_vs_direct=_rel(sgm, dgm),
        m_mesh_vs_single=_rel(mgm, sgm),
    )
    check["x_ratio"] = check["x_mesh_vs_direct"] / check["x_single_vs_direct"]
    check["m_ratio"] = check["m_mesh_vs_direct"] / check["m_single_vs_direct"]
    r["check"] = check
    _log("check", json.dumps(check))


def _run_solo(row: _Row, args: argparse.Namespace, ic: dict) -> None:
    import jax
    import jax.numpy as jnp

    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_topology import ForceParams

    r, n = row.r, args.n
    dev = jax.devices()[0]
    dtype = np.float64 if args.dtype == "float64" else np.float32
    solver = _solver(args, ic["G"], ic["softening"])
    x = jax.device_put(jnp.asarray(ic["pos"].astype(dtype)), dev)
    m = jax.device_put(jnp.asarray(ic["mass"].astype(dtype)), dev)
    t0 = time.perf_counter()
    prov, topo = FusedForceProvider.prepare(
        solver,
        x,
        m,
        leaf_size=args.leaf,
        max_order=args.order,
        theta=args.theta,
        differentiable=True,
    )
    aux = None
    if prov.scale_carry:
        # strict_run_v2's first f_b: the eager prepare's eq (16b) prepass, input order
        fb_sorted = jnp.asarray(solver._impl._last_force_scale_particles, x.dtype)
        perm = jnp.asarray(topo.prepared.tree.particle_indices)
        aux = (jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted), None, None)
    del topo
    # the solver's per-call side channels of the eager prepare
    for holder in (prov.engine, getattr(prov.engine, "_impl", None)):
        for name in list(vars(holder) if holder is not None else ()):
            if name.startswith("_last_"):
                setattr(holder, name, None)
    gc.collect()
    r["prepare_s"] = time.perf_counter() - t0
    r["scale_carry"] = bool(prov.scale_carry)
    r.update(cap=n, counts=[n])
    row.phase("prepared", [dev])
    w = jax.device_put(
        jnp.asarray(np.random.default_rng(args.seed + 1).normal(size=(n, 3)), dtype),
        dev,
    )
    p0 = None
    if args.params:
        p0 = (jnp.asarray(ic["G"], dtype), jnp.asarray(ic["softening"] ** 2, dtype))

    def as_params(p):
        return None if p is None else ForceParams(G=p[0], eps2=p[1])

    def prod(xx, mm, a):
        acc, (_, bits, needs) = prov.step(xx, mm, None, a)
        return acc, bits, needs

    def diff(xx, mm, p, a):
        acc, (_, bits, _) = prov.differentiable_step(xx, mm, as_params(p), a)
        return acc, bits

    def loss(xx, mm, p, ww, a):
        acc, (_, bits, _) = prov.differentiable_step(xx, mm, as_params(p), a)
        return jnp.sum(acc * ww), bits

    _measure(
        row,
        args,
        devices=[dev],
        counts=np.asarray([n]),
        live_rows=np.arange(n),
        ctx=contextlib.nullcontext,
        before=lambda: None,
        prod=prod,
        pargs=(x, m, aux),
        diff=diff,
        dargs=(x, m, p0, aux),
        loss=loss,
        gargs=(x, m, p0, w, aux),
        argnums=(0, 1, 2) if args.params else (0, 1),
    )


def main(argv: Optional[list] = None) -> int:
    args = _args(argv)
    prepare_env(args)
    import jax

    jax.config.update("jax_enable_x64", True)
    if args.cpu_smoke:
        _patch_cpu_lane()
    from jaccpot.distributed.fused import _missing_ragged_exchange_flags

    devices = jax.devices()
    cards = _visible_cards(args.ndev) if not args.cpu_smoke else []
    record = dict(
        mode=args.mode,
        ic=args.ic,
        n=args.n,
        ndev=args.ndev,
        n_per_card=args.n / args.ndev,
        seed=args.seed,
        galaxy=args.galaxy,
        mac="dehnen_error" if args.galaxy else "dehnen",
        order=args.order,
        leaf=args.leaf,
        leaf_capacity=args.leaf_capacity,
        theta=args.theta,
        dtype=args.dtype,
        params=args.params,
        cross=args.cross,
        grad_only=args.grad_only,
        reps=args.reps,
        cpu_smoke=args.cpu_smoke,
        argv=sys.argv[1:] if argv is None else list(argv),
        devices=[str(d) for d in devices[: args.ndev]],
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        allocator=dict(
            preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
            allocator=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        ),
        others_before=card_processes(cards) if cards else [],
        complete=False,
    )
    row = _Row(args.out, record)
    missing = _missing_ragged_exchange_flags(record["xla_flags"])
    record["ragged_flags_missing"] = missing
    on_gpu = devices[0].platform == "gpu"
    if not (on_gpu or args.cpu_smoke):
        record["error"] = f"not a GPU: {devices}"
        row.save()
        print(record["error"], file=sys.stderr)
        return 2
    if len(devices) < args.ndev:
        record["error"] = f"{args.ndev} devices wanted, {len(devices)} visible"
        row.save()
        print(record["error"], file=sys.stderr)
        return 2
    if on_gpu and args.ndev >= 2 and missing:
        msg = (
            f"XLA_FLAGS lacks the ragged-exchange flags {missing} (jax 0.11's "
            "use_barrier_with_nccl deadlock; bench/_ragged_exchange_flags.py)"
        )
        if not args.allow_flag_override:
            record["error"] = "refused: " + msg
            row.save()
            print(record["error"], file=sys.stderr)
            return 2
        print("WARNING: " + msg, file=sys.stderr, flush=True)
    ic = make_ic(args)
    record.update(G=ic["G"], softening=ic["softening"], ic_source=ic["source"])
    _log(json.dumps({k: v for k, v in record.items() if k != "xla_flags"}))
    if args.mode == "mesh":
        _run_mesh(row, args, ic)
    else:
        _run_solo(row, args, ic)
    record["others_after"] = card_processes(cards) if cards else []
    others = {p.get("pid") for p in record["others_before"] + record["others_after"]}
    record["contended"] = bool(others) if cards else None
    record["wall_s"] = time.time() - _T0
    record["complete"] = True
    row.save()
    print(
        "RESULT "
        + json.dumps(
            {k: v for k, v in record.items() if k not in ("memory", "xla_flags")},
            default=_jsonable,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

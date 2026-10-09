"""Benchmark the real (Dehnen) basis vs the complex/solidfmm basis.

Runnable end-to-end comparison for the real-basis FMM work: correctness first
(so timings are trustworthy), then compute wall-clock, the Pallas real M2L
path, and coefficient memory. (The grouped/class-major arm went with the grouped
far field in the 2026-10 cleanup, X3.)

Free-GPU selection uses ``autocvd`` and MUST run before ``import jax`` (mirrors
the other bench scripts). On CPU the script still runs (useful as a smoke test),
but the Pallas path falls back to pure-JAX and the timing story is GPU-specific.

Quick start on a GPU box (after pulling the branch)::

    # ensure yggdrax is importable (installed, or a sibling ../yggdrax checkout
    # on a branch that has rebuild_static_radix_tree_from_template)
    python bench/bench_real_vs_complex.py --output bench_real_vs_complex.md

Common variations::

    python bench/bench_real_vs_complex.py --n 8000,50000,200000 --orders 4,6,8
    python bench/bench_real_vs_complex.py --gpu-select first        # pin GPU 0
    python bench/bench_real_vs_complex.py --skip-pallas

What to look for
----------------
* Correctness: ``real`` must converge with order like ``complex``/``solidfmm``
  (rel-L2 dropping toward ~1e-6). If it does not, STOP -- do not trust timings.
* Compute: ``real/solidfmm`` wall-clock ratio < 1 means real is faster; the CPU
  baseline was ~0.5-0.9x, and the gap is expected to widen on GPU (real is half
  the FLOPs/bandwidth of complex).
* Pallas: ``use_pallas=True`` should beat pure-JAX for the real z-M2L core on
  GPU; on CPU it is identical (silent fallback).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Fall back to a sibling yggdrax checkout if it is not installed (matches
# tests/conftest.py).
_YGGDRAX = REPO_ROOT.parent / "yggdrax"
if _YGGDRAX.exists() and str(_YGGDRAX) not in sys.path:
    sys.path.insert(0, str(_YGGDRAX))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", default="3000,8000,20000", help="Comma-separated N values")
    p.add_argument("--orders", default="4,8", help="Comma-separated expansion orders")
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--leaf-size", type=int, default=16)
    p.add_argument("--softening", type=float, default=1.0e-3)
    p.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--gpu-select",
        choices=("least-used", "first", "none"),
        default="least-used",
        help="autocvd free-GPU selection (before importing jax)",
    )
    p.add_argument("--correctness-n", type=int, default=1500)
    p.add_argument("--skip-correctness", action="store_true")
    p.add_argument("--skip-pallas", action="store_true")
    p.add_argument("--output", default=None, help="Write a markdown report here")
    return p.parse_args()


def _select_gpu(args: argparse.Namespace) -> None:
    if args.gpu_select == "none" or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    try:
        from autocvd import autocvd
    except Exception as exc:  # pragma: no cover - env-dependent
        print(f"[bench] autocvd unavailable ({exc}); using default device")
        return
    autocvd(num_gpus=1, least_used=(args.gpu_select == "least-used"))


def main() -> None:
    args = _parse_args()
    _select_gpu(args)
    if args.dtype == "float64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot import FastMultipoleMethod
    from jaccpot.operators.real_harmonics import sh_size

    dtype = jnp.float64 if args.dtype == "float64" else jnp.float32
    ns = [int(x) for x in str(args.n).split(",") if x]
    orders = [int(x) for x in str(args.orders).split(",") if x]
    backend = jax.default_backend()
    device = jax.devices()[0]
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# real vs complex/solidfmm benchmark")
    emit()
    emit(f"- backend: `{backend}`  device: `{device}`")
    emit(
        f"- dtype: `{args.dtype}`  theta: {args.theta}  leaf_size: {args.leaf_size}"
        f"  softening: {args.softening}"
    )
    emit(f"- warmup: {args.warmup}  runs (best-of): {args.runs}")
    emit()

    def make_positions(n: int, seed: int):
        key = jax.random.PRNGKey(seed)
        kp, km = jax.random.split(key)
        pos = jax.random.uniform(kp, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype)
        mass = jnp.abs(jax.random.normal(km, (n,), dtype=dtype)) + jnp.asarray(
            0.5, dtype=dtype
        )
        return pos, mass

    def build_solver(
        basis: str,
        *,
        use_pallas: Optional[bool] = None,
    ) -> FastMultipoleMethod:
        kwargs: dict[str, Any] = dict(
            preset="accurate",
            basis=basis,
            theta=args.theta,
            softening=args.softening,
        )
        if use_pallas is not None:
            kwargs["use_pallas"] = use_pallas
        return FastMultipoleMethod(**kwargs)

    def time_solver(fmm: FastMultipoleMethod, pos, mass, order: int) -> float:
        def call():
            return fmm.compute_accelerations(
                pos, mass, leaf_size=args.leaf_size, max_order=order
            )

        for _ in range(max(1, args.warmup)):
            call().block_until_ready()
        best = float("inf")
        for _ in range(max(1, args.runs)):
            t0 = time.perf_counter()
            out = call()
            out.block_until_ready()
            best = min(best, time.perf_counter() - t0)
        return best

    def direct_accelerations(pos, mass) -> np.ndarray:
        pos_np = np.asarray(pos, dtype=np.float64)
        mass_np = np.asarray(mass, dtype=np.float64)
        n = pos_np.shape[0]
        out = np.zeros_like(pos_np)
        s2 = float(args.softening) ** 2
        for i in range(n):
            d = pos_np[i] - pos_np
            r2 = np.sum(d * d, axis=1) + s2
            inv = 1.0 / (r2 * np.sqrt(r2))
            inv[i] = 0.0
            out[i] = -np.sum((mass_np[:, None] * inv[:, None]) * d, axis=0)
        return out

    # ---- Correctness (trust gate) -----------------------------------------
    if not args.skip_correctness:
        emit("## Correctness (rel-L2 vs direct sum)")
        emit()
        cn = int(args.correctness_n)
        pos, mass = make_positions(cn, args.seed)
        ref = direct_accelerations(pos, mass)
        emit(f"N = {cn}")
        emit()
        emit("| order | real | complex | solidfmm |")
        emit("|------:|-----:|--------:|---------:|")
        for order in orders:
            row = [f"| {order} "]
            for basis in ("real", "complex", "solidfmm"):
                try:
                    acc = np.asarray(
                        build_solver(basis).compute_accelerations(
                            pos, mass, leaf_size=args.leaf_size, max_order=order
                        )
                    )
                    rel = float(
                        np.linalg.norm(acc - ref) / (np.linalg.norm(ref) + 1e-30)
                    )
                    row.append(f"| {rel:.2e} ")
                except Exception as exc:  # pragma: no cover
                    row.append(f"| ERR:{str(exc)[:18]} ")
            emit("".join(row) + "|")
        emit()

    # ---- Compute wall-clock (flat path) -----------------------------------
    emit("## Compute wall-clock (best-of, flat far-field)")
    emit()
    emit("| N | order | solidfmm (ms) | complex (ms) | real (ms) | real/solidfmm |")
    emit("|--:|------:|--------------:|-------------:|----------:|--------------:|")
    for n in ns:
        pos, mass = make_positions(n, args.seed)
        for order in orders:
            times: dict[str, float] = {}
            for basis in ("solidfmm", "complex", "real"):
                try:
                    times[basis] = time_solver(build_solver(basis), pos, mass, order)
                except Exception as exc:  # pragma: no cover
                    print(f"[bench] {basis} N={n} p={order} failed: {exc}")
                    times[basis] = float("nan")
            ratio = (
                times["real"] / times["solidfmm"] if times["solidfmm"] else float("nan")
            )
            emit(
                f"| {n} | {order} | {times['solidfmm']*1e3:.1f} "
                f"| {times['complex']*1e3:.1f} | {times['real']*1e3:.1f} "
                f"| {ratio:.2f} |"
            )
    emit()

    # ---- Pallas real z-M2L core -------------------------------------------
    if not args.skip_pallas:
        from jaccpot.pallas.m2l_core_z_real import pallas_m2l_real_supported

        emit("## Real z-M2L core: pure-JAX vs Pallas")
        emit()
        if not pallas_m2l_real_supported():
            emit(
                f"Pallas not supported on backend `{backend}` "
                "(runs only on gpu/tpu); use_pallas silently falls back to "
                "pure-JAX, so these numbers would be identical. Skipping."
            )
        else:
            emit("| N | order | pure-JAX (ms) | Pallas (ms) | pallas/pureJAX |")
            emit("|--:|------:|--------------:|------------:|---------------:|")
            for n in ns:
                pos, mass = make_positions(n, args.seed)
                for order in orders:
                    try:
                        t_pure = time_solver(
                            build_solver("real", use_pallas=False), pos, mass, order
                        )
                        t_pal = time_solver(
                            build_solver("real", use_pallas=True), pos, mass, order
                        )
                        emit(
                            f"| {n} | {order} | {t_pure*1e3:.1f} | {t_pal*1e3:.1f} "
                            f"| {t_pal/t_pure:.2f} |"
                        )
                    except Exception as exc:  # pragma: no cover
                        print(f"[bench] pallas N={n} p={order} failed: {exc}")
        emit()

    # ---- Coefficient memory ------------------------------------------------
    emit("## Coefficient memory per node (real float vs complex)")
    emit()
    bytes_real = 8 if args.dtype == "float64" else 4
    bytes_complex = 2 * bytes_real
    emit("| order | n_coeffs | real (B) | complex (B) | complex/real |")
    emit("|------:|---------:|---------:|------------:|-------------:|")
    for order in orders:
        nc = sh_size(order)
        emit(
            f"| {order} | {nc} | {nc*bytes_real} | {nc*bytes_complex} "
            f"| {bytes_complex/bytes_real:.1f} |"
        )
    emit()

    if args.output:
        out_path = pathlib.Path(args.output)
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[bench] wrote report to {out_path}")


if __name__ == "__main__":
    main()

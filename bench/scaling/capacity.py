"""Memory per particle on the fused lane, and the largest N one card holds (capacity panel).

REBASE_PLAN's new figure: the capacity story of ``preset="galaxy"`` (D3;
``bench/scaling/_galaxy.py``). For each N it records, from the COMPILED programs
(``lower().compile().memory_analysis()``, the method of
``bench/fused_memory_budget.py`` and ``bench/grad_production_gate.py``):

* **forward** -- ``FusedForceProvider.step`` jitted (rebuild + evaluate, the
  program a simulation runs every step): argument, output, alias and temporary
  bytes, and ``step_program_B_per_p = (argument + output + temp - alias) / N``.
  Arguments are the positions, masses and f_b (jit arguments, as everywhere);
  the prepared topology is not an argument of ``step`` (it reads only the
  template's shapes).
* **gradient temporaries** (``--grad``, on by default) -- ``jax.grad`` of
  ``sum(w . acc)`` through ``FusedForceProvider.differentiable_step`` in the
  positions and masses, on a solver that keeps its far pairs
  (``retain_far_pairs=True``, the gradient plan reads them; that is the only field
  off the preset): ``temp_B_per_p``. This is S9's differentiable step, which also
  rebuilds (with stopped gradients), so it is not byte-for-byte S8's whole-force
  gradient that ``docs/production_2026-10.md`` tabulates.
* **device memory** (GPU only; ``memory_stats()``): bytes in use after the warm
  step (what stays resident: the prepared state, the inputs) and the process's
  peak. The peak is a high-water mark that is never reset, so it includes the
  eager prepare and every earlier row of this process: run one N per process
  (``--n 1e8``) for a clean peak, as ``fused_memory_budget.py`` does. A row says
  whether its peak may carry an earlier row's (``peak_includes_earlier_rows``).

Largest N per card
------------------
A linear fit ``bytes(N) = a + b N`` over the rows with ``N >= --fit-min-n``, for
the step program and (GPU) for the device peak, solved for the card's limit
(``memory_stats()['bytes_limit']``, or ``--card-bytes``). That is an
EXTRAPOLATION from the fitted rows, labelled as such; the measured ceiling is
only the largest N whose row ran (``largest_measured_n``). Neither covers
fragmentation (``fused_memory_budget.py --prealloc`` is that bench's lever).

Reference values to check against (REBASE_PLAN, "new: memory per particle"; not
copied into any output): about 175 B/particle forward and about 150 B/particle
of gradient temporaries at N = 1e8 (``docs/production_2026-10.md``, S8 memory
table: 149.4 B/p of ``jax.grad`` temporaries at 1e8; ``docs/production.md``: the
1e8 clipped-Plummer device peak 257 B/p geometric, 279 B/p ``dehnen_error``).

On CPU the numbers are XLA:CPU's buffer assignment of the interpreted kernels at
tiny N (fixed buffers dominate): smoke only. The gradient through the CPU route
is expected to fail at trace time (a level loop with dynamic bounds has no
reverse there; the GPU lane's cascades are Pallas kernels with their own
reverses): the row records the error instead of a number.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.scaling.capacity --n 2048,4096 \\
        --fit-min-n 0 --gpu-select none --json-out /tmp/smoke.json

Paper rows (one free A100; one N per process for a clean device peak)::

    for n in 1e6 1e7 3e7 1e8; do
        python -m bench.scaling.capacity --n $n --json-out scaling/capacity_$n.json
    done
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "scaling/capacity.json"
_MA_FIELDS = (
    "argument_size_in_bytes",
    "output_size_in_bytes",
    "alias_size_in_bytes",
    "temp_size_in_bytes",
    "generated_code_size_in_bytes",
    "peak_memory_in_bytes",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--n",
        default="1e6,1e7,3e7,1e8",
        help="Comma-separated particle counts, 1e6 notation allowed",
    )
    p.add_argument("--distribution", default="plummer_clipped")
    p.add_argument("--rmax", type=float, default=20.0)
    p.add_argument("--softening", type=float, default=1e-7)
    p.add_argument("--G", dest="g", type=float, default=1.0)
    p.add_argument(
        "--grad",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compile the gradient through differentiable_step (default on)",
    )
    p.add_argument(
        "--card-bytes",
        type=float,
        default=None,
        help="Card memory for the largest-N solve (default: memory_stats bytes_limit)",
    )
    p.add_argument("--fit-min-n", type=int, default=1_000_000)
    runmeta.add_common_args(p)
    p.set_defaults(dtype="float32")
    return p.parse_args()


def _analysis(compiled: Any, n: int) -> dict[str, Any]:
    ma = compiled.memory_analysis()
    out: dict[str, Any] = {}
    for f in _MA_FIELDS:
        v = getattr(ma, f, None)
        if v is not None:
            out[f] = int(v)
    arg = out.get("argument_size_in_bytes", 0)
    res = out.get("output_size_in_bytes", 0)
    tmp = out.get("temp_size_in_bytes", 0)
    alias = out.get("alias_size_in_bytes", 0)
    out["program_bytes"] = arg + res + tmp - alias
    out["program_B_per_p"] = out["program_bytes"] / n
    out["temp_B_per_p"] = tmp / n
    out["argument_B_per_p"] = arg / n
    out["output_B_per_p"] = res / n
    return out


def _mem() -> Optional[dict[str, int]]:
    import jax

    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return None
    return {
        "bytes_in_use": int(stats.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", 0)),
        "bytes_limit": int(stats.get("bytes_limit", 0)),
    }


def _largest_n(
    rows: list[dict[str, Any]], field: str, limit: Optional[float], fit_min_n: int
) -> dict[str, Any]:
    """Solve ``a + b N = limit`` from a least-squares fit over the rows."""
    import numpy as np

    pts = [(r["n"], r[field]) for r in rows if r.get(field) and r["n"] >= fit_min_n]
    if len(pts) < 2 or not limit:
        return {
            "n_points": len(pts),
            "largest_n_extrapolated": None,
            "why_null": "fewer than 2 rows" if len(pts) < 2 else "no card limit",
        }
    ns = np.asarray([a for a, _ in pts], float)
    bs = np.asarray([b for _, b in pts], float)
    b, a = np.polyfit(ns, bs, 1)
    return {
        "n_points": len(pts),
        "fit_min_n": int(ns.min()),
        "fit_max_n": int(ns.max()),
        "fixed_bytes": float(a),
        "bytes_per_particle": float(b),
        "card_bytes": float(limit),
        "largest_n_extrapolated": int((limit - a) / b) if b > 0 else None,
        "note": "extrapolated from a linear fit; not measured",
    }


def main() -> int:
    args = _parse_args()
    if args.dtype != "float32":
        print("[capacity] preset='galaxy' is fp32 (D3); --dtype float64 is not this")
        return 2
    runmeta.select_gpu(args.gpu_select)
    from bench.scaling import _timing as T  # noqa: E402  (imports no jax)

    env = T.launch_env()

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from bench.scaling import _galaxy as GX  # noqa: E402
    from jaccpot.runtime.fused_topology import grad_ok  # noqa: E402

    ns = [int(float(v)) for v in str(args.n).split(",") if v.strip()]
    rows: list[dict[str, Any]] = []
    galaxy: dict[str, Any] = {}
    for i, n in enumerate(ns):
        key = jax.random.PRNGKey(runmeta.seed_sequence(args.seed, args.distribution, n))
        x, m = T.distribution(key, n, args.distribution, jnp.float32, rmax=args.rmax)
        x, m = jax.block_until_ready((x, m))
        row: dict[str, Any] = {
            "n": int(n),
            "card_load_before": GX.card_load(),
            "peak_includes_earlier_rows": i > 0,
        }
        # -- forward: the step program ----------------------------------------
        try:
            solver = GX.galaxy_solver(softening=float(args.softening), G=float(args.g))
            galaxy = galaxy or GX.describe(solver)
            fs, setup = T.warm_fused_step(solver, x, m)
            row["setup"] = setup
            row["device_after_step"] = _mem()
            compiled = fs._step.lower(x, m, None, fs.aux).compile()
            row["forward"] = _analysis(compiled, n)
            row["step_program_B_per_p"] = row["forward"]["program_B_per_p"]
            row["step_program_bytes"] = row["forward"]["program_bytes"]
            acc, aux = fs.step(x, m)
            jax.block_until_ready(acc)
            row["valid"] = bool(fs.ok(aux)) and bool(setup["first_step_ok"])
            dev = _mem()
            if dev:
                row["device_peak_bytes"] = dev["peak_bytes_in_use"]
                row["device_peak_B_per_p"] = dev["peak_bytes_in_use"] / n
                row["device_resident_B_per_p"] = (
                    row["device_after_step"]["bytes_in_use"] / n
                )
            del fs, compiled, acc
        except Exception as exc:
            row["valid"] = False
            row["error"] = str(exc)[:400]
            print(f"N={n:<10d} forward FAILED: {str(exc)[:200]}")

        # -- gradient temporaries ---------------------------------------------
        if args.grad and row.get("valid"):
            try:
                solver_g = GX.galaxy_solver(
                    softening=float(args.softening),
                    G=float(args.g),
                    retain_far_pairs=True,
                )
                fg, _ = T.warm_fused_step(solver_g, x, m, differentiable=True)
                prov = fg.provider
                w = jax.random.normal(jax.random.PRNGKey(1), (n, 3), jnp.float32)

                def loss(xx, mm, ww, fb):
                    acc_g, aux_g = prov.differentiable_step(
                        xx, mm, None, (fb, None, None)
                    )
                    return jnp.sum(ww * acc_g), aux_g[1]

                gfun = jax.jit(jax.grad(loss, argnums=(0, 1), has_aux=True))
                compiled_g = gfun.lower(x, m, w, fg.aux[0]).compile()
                row["gradient"] = _analysis(compiled_g, n)
                row["grad_temp_B_per_p"] = row["gradient"]["temp_B_per_p"]
                grads, bits = compiled_g(x, m, w, fg.aux[0])
                jax.block_until_ready(grads)
                row["gradient"]["grad_ok"] = bool(grad_ok(bits))
                dev = _mem()
                if dev:
                    row["device_peak_after_grad_B_per_p"] = dev["peak_bytes_in_use"] / n
                del fg, compiled_g, grads
            except Exception as exc:
                row["gradient"] = {"error": str(exc)[:400]}
                print(f"N={n:<10d} gradient not measured: {str(exc)[:160]}")
        row["card_load_after"] = GX.card_load()
        if row.get("valid") and not GX.card_free(
            row.get("card_load_before"), row["card_load_after"]
        ):
            row["valid"] = False
            row["invalid_reason"] = "contended card"
        rows.append(row)
        if row.get("step_program_B_per_p") is not None:
            print(
                f"N={n:<10d} valid={row.get('valid')} step program "
                f"{row['step_program_B_per_p']:.1f} B/p (temp "
                f"{row['forward']['temp_B_per_p']:.1f})"
                + (
                    f", device peak {row['device_peak_B_per_p']:.1f} B/p"
                    if row.get("device_peak_B_per_p")
                    else ""
                )
                + (
                    f", grad temp {row['grad_temp_B_per_p']:.1f} B/p"
                    if row.get("grad_temp_B_per_p")
                    else ""
                ),
                flush=True,
            )

    dev = _mem()
    limit = args.card_bytes or (dev["bytes_limit"] if dev else None)
    valid = [r for r in rows if r.get("valid")]
    capacity = {
        "step_program": _largest_n(
            valid, "step_program_bytes", limit, int(args.fit_min_n)
        ),
        # a process's peak is a high-water mark: after the first N it includes
        # the earlier rows' programs, so only first-in-process rows are fitted
        "device_peak": _largest_n(
            [r for r in valid if not r.get("peak_includes_earlier_rows")],
            "device_peak_bytes",
            limit,
            int(args.fit_min_n),
        ),
        "largest_measured_n": max((r["n"] for r in valid), default=None),
    }
    for k in ("step_program", "device_peak"):
        c = capacity[k]
        if c.get("largest_n_extrapolated"):
            print(
                f"{k}: {c['bytes_per_particle']:.1f} B/p + {c['fixed_bytes'] / 2**30:.2f}"
                f" GiB -> largest N on {limit / 2**30:.1f} GiB ~ "
                f"{c['largest_n_extrapolated']:.3e} (extrapolated)"
            )

    config = {
        "n": ns,
        "theta": float(GX.GALAXY.theta),
        "order": int(GX.GALAXY.max_order),
        "basis": "real",
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": "float32",
        "leaf_size": int(GX.GALAXY.leaf_size),
        "preset": "galaxy",
        "galaxy": galaxy,
        "gradient_solver_off_preset": {"retain_far_pairs": True},
        "distribution": args.distribution,
        "softening": float(args.softening),
        "G": float(args.g),
        "grad": bool(args.grad),
        "card_bytes": limit,
        "fit_min_n": int(args.fit_min_n),
        "launch_env": env,
        "measured_from": "lower().compile().memory_analysis() of the jitted step "
        "and of jax.grad through differentiable_step; device memory_stats() on GPU",
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": rows, "capacity": capacity},
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

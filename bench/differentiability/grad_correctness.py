"""Figure 13 data -- autodiff against finite differences, and against the exact gradient.

Two different questions, kept in two different blocks of the JSON.

1. Is the autodiff gradient the derivative of the function the FMM computes?
   (``data.records``, the figure.) On the fused lane (``preset="galaxy"``: cell
   leaves of 64, p6, ferrers3, the real basis; ``grad_bench_lib.solver``) the
   gradient of ``L = sum(w . acc)`` w.r.t. positions and masses is compared with
   central finite differences, swept over the MAC's tightness: ``theta`` for the
   geometric ``"dehnen"`` criterion (``--thetas``), ``adaptive_eps`` for the
   preset's ``"dehnen_error"`` (``--eps``).

   The topology is PINNED within the stencil. The MAC decision, the tree and the
   lists are piecewise constant in the positions, so across a switch a finite
   difference legitimately disagrees with any derivative. The ``frozen`` arm
   therefore differentiates one function of ``(x, m)`` on ONE topology, the
   prepare's: ``differentiable_accelerations(topology, x, m, plan)``
   (``FusedForceProvider``), and finite-differences the SAME function; autodiff
   and FD see the same map by construction. The ``full_pipeline`` arm
   finite-differences the production step itself (rebuild at ``x +- h`` and
   evaluate) and compares with the same autodiff gradient, but only on samples
   whose rebuilt topology (tree permutation, near CSR, far list) is identical at
   ``+h``, ``-h`` and the centre; the others are counted in
   ``full_pipeline_switched`` and excluded (they measure topology sensitivity,
   not autodiff).

   Precision: ``--dtype float64`` (the default) is what the figure uses. In
   fp32 the comparison bottoms out near 4e-4 (measured 2026-11-21 after
   3b8bbd8f): FD round-off, not a gradient error. ``--dtype float32`` is allowed
   and recorded, with a larger default step.

2. How far is the FMM's gradient from the exact one? (``data.oracle``.) The
   autodiff gradient (frozen topology, which is the step's own) against the
   autodiff gradient of an exact fp64 direct sum with the solver's pair law
   (``grad_bench_lib.direct_loss_grad``), per expansion order (``--orders``) at
   the preset's MAC, next to the FORCE's error against the A4 reference
   (``jaccpot.verification.reference_force``) of the same configuration: the
   gradient cannot reasonably be tighter than the force it differentiates. This
   is the approximation's accuracy, not an autodiff check.

Masses are differentiated under ``dehnen_error``: its f_b only sets the walk's
thresholds and is stopped; no zero-mass particles are involved, so the
mass-dependent criterion needs no geometric fallback here.

Validity: every point records the topology's capacity verdict and the gradient
lists' symmetry verdict (``grad_ok``); a failed one carries ``error`` and is not
plotted. ``far_pairs`` is recorded so an empty far field (only the near field's
gradient checked) is visible, not assumed away.

CPU. The reverse runs only on the gates' route (S9 interpret set,
``jax.default_backend`` reporting ``"gpu"``; ``grad_bench_lib``): every kernel
interpreted, p6 programs compiling for minutes. The numbers are the same
arithmetic, so a CPU smoke's agreement figures are meaningful at its N; its
times are not.

Usage
-----
CPU smoke::

    JAX_PLATFORMS=cpu python -m bench.differentiability.grad_correctness \\
        --n 512 --thetas 0.6 --eps 1e-4 --order 2 --orders 2,4 \\
        --fd-samples 3 --full-samples 2 \\
        --gpu-select none --json-out /tmp/fig13_smoke.json

Paper run::

    python -m bench.differentiability.grad_correctness
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.differentiability import grad_bench_lib as L  # noqa: E402
from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "differentiability/grad_correctness.json"


def _floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # N=4096 with cell leaves of 64 keeps the far field populated at every MAC
    # point; `far_pairs` is recorded per point to show it.
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--distribution", default="plummer")
    p.add_argument("--softening", type=float, default=1e-2)
    p.add_argument(
        "--thetas",
        default="0.4,0.6,0.8",
        help="geometric 'dehnen' MAC points ('' skips the arm)",
    )
    p.add_argument(
        "--eps",
        default="1e-3,1e-4,1e-5",
        help="'dehnen_error' adaptive_eps points (the preset's MAC; '' skips)",
    )
    p.add_argument("--wrt", default="positions,masses")
    p.add_argument(
        "--fd-samples",
        type=int,
        default=12,
        help="fixed-seed coordinates finite-differenced per (MAC point, wrt)",
    )
    p.add_argument(
        "--fd-eps",
        type=float,
        default=None,
        help="central-difference step (default 1e-6 in fp64, 1e-3 in fp32)",
    )
    p.add_argument(
        "--full-samples",
        type=int,
        default=4,
        help="of those, how many the full-pipeline arm also checks (0: skip it)",
    )
    p.add_argument(
        "--order",
        type=int,
        default=None,
        help="expansion order of the FD sweep (default: the preset's 6; lower "
        "only for CPU smokes, whose interpret-mode p6 reverse compiles for tens "
        "of minutes; recorded per record and in off_preset)",
    )
    p.add_argument(
        "--orders",
        default="2,4,6,8",
        help="expansion orders of the direct-sum oracle ('' skips it)",
    )
    p.add_argument(
        "--oracle-targets",
        type=int,
        default=4096,
        help="targets of the oracle's loss (all N if N is smaller)",
    )
    runmeta.add_common_args(p)
    args = p.parse_args()
    args.theta_list = _floats(args.thetas)
    args.eps_list = _floats(args.eps)
    args.order_list = [int(v) for v in _floats(args.orders)]
    args.wrt_list = [v.strip() for v in args.wrt.split(",") if v.strip()]
    if set(args.wrt_list) - {"positions", "masses"}:
        p.error("--wrt takes positions and/or masses")
    if args.fd_eps is None:
        args.fd_eps = 1e-6 if args.dtype == "float64" else 1e-3
    return args


def _far_and_verdict(prov: Any, topo: Any) -> tuple[Optional[int], bool, bool]:
    """``(far pairs, capacity ok, gradient lists ok)`` of the eager topology."""
    import jax

    plan, sym = jax.jit(prov.grad_plan)(topo)
    far = getattr(plan, "far_pair_active_count", None)
    # == overflow_ok(topology.overflow_bits()) (FusedForceProvider.capacity_ok)
    cap_ok = bool(prov.capacity_ok(topo))
    return (None if far is None else int(far)), cap_ok, bool(sym)


def _mac_point(args, x, m, w, seed, mac: str, param: float) -> list[dict]:
    """The frozen and full-pipeline FD checks at one MAC point."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    kw = {"theta": param} if mac == "dehnen" else {"adaptive_eps": param}
    fmm, scfg = L.solver(
        softening=args.softening,
        dtype=args.dtype,
        mac_type="dehnen" if mac == "dehnen" else None,
        order=args.order,
        **kw,
    )
    t0 = time.perf_counter()
    prov, topo, aux0 = L.provider(fmm, x, m, order=args.order)
    far, cap_ok, sym_ok = _far_and_verdict(prov, topo)
    prep_s = time.perf_counter() - t0

    def pinned(t, x, m, w):
        plan, _ = prov.grad_plan(t)
        return jnp.sum(w * prov.differentiable_accelerations(t, x, m, plan=plan))

    # one program for the gradient and every FD value (a value-only program would
    # be a second compile; at these N the extra reverse costs nothing)
    vg = jax.jit(jax.value_and_grad(pinned, argnums=(1, 2)))
    t0 = time.perf_counter()
    _, (gx, gm) = vg(topo, x, m, w)
    ad = {"positions": np.asarray(gx, np.float64), "masses": np.asarray(gm, np.float64)}
    grad_s = time.perf_counter() - t0
    finite = all(np.all(np.isfinite(a)) for a in ad.values())

    full = None
    if args.full_samples > 0:
        f_b = aux0[0]

        def full_fn(x, m, w):
            t = prov.rebuild(x, m, force_scale=f_b, keep_far_list=True)
            prep = t.prepared
            nl, fp = prep.neighbor_list, prep.compact_far_pairs
            ids = [
                prep.tree.particle_indices,
                nl.offsets,
                nl.neighbors,
                fp.sources,
                fp.targets,
                fp.far_pair_count,
            ]
            return jnp.sum(w * prov.accelerations(t)), ids

        full = jax.jit(full_fn)
        centre_ids = [np.asarray(a) for a in full(x, m, w)[1]]

    def value(xx, mm) -> float:
        return float(vg(topo, xx, mm, w)[0])

    records = []
    for wrt in args.wrt_list:
        base = np.asarray(x if wrt == "positions" else m, np.float64)
        rng = np.random.default_rng(seed ^ (1 if wrt == "positions" else 2))
        idx = rng.choice(base.size, size=min(args.fd_samples, base.size), replace=False)
        h = float(args.fd_eps)

        def bumped(k: int, s: float) -> Any:
            b = base.copy().ravel()
            b[k] += s
            return jnp.asarray(b.reshape(base.shape), x.dtype)

        def at(k: int, s: float, fn) -> Any:
            return fn(bumped(k, s), m) if wrt == "positions" else fn(x, bumped(k, s))

        fd = np.array([(at(k, h, value) - at(k, -h, value)) / (2 * h) for k in idx])
        ad_sub = ad[wrt].ravel()[idx]
        denom = float(np.linalg.norm(fd)) or 1.0
        # one random direction over the whole array as well (all coordinates)
        v = rng.normal(size=base.shape)
        vj = jnp.asarray(v, x.dtype)
        if wrt == "positions":
            dplus, dminus = value(x + h * vj, m), value(x - h * vj, m)
        else:
            dplus, dminus = value(x, m + h * vj), value(x, m - h * vj)
        fd_dir = (dplus - dminus) / (2 * h)
        ad_dir = float(np.sum(ad[wrt] * v))
        rec: dict[str, Any] = {
            "basis": "real",
            "lane": "fused",
            "mac_type": mac,
            "mac_param": float(param),
            "theta": float(scfg["theta"]),
            "adaptive_eps": scfg["adaptive_eps"],
            "order": int(scfg["max_order"]),
            "wrt": wrt,
            "seed": int(seed),
            "far_pairs": far,
            "capacity_ok": cap_ok,
            "grad_ok": sym_ok,
            "grad_finite": bool(finite),
            "fd_samples": int(len(idx)),
            "fd_eps": h,
            "frozen_rel_l2": float(np.linalg.norm(ad_sub - fd) / denom),
            "frozen_worst_rel": float(
                np.max(np.abs(ad_sub - fd) / np.maximum(np.abs(fd), denom / len(fd)))
            ),
            "frozen_directional_rel": abs(ad_dir - fd_dir) / (abs(fd_dir) + 1e-300),
            "ad_norm": float(np.linalg.norm(ad_sub)),
            "fd_norm": float(np.linalg.norm(fd)),
            "prepare_s": prep_s,
            "first_grad_s": grad_s,
        }
        if full is not None:
            few = idx[: min(args.full_samples, len(idx))]
            vals, switched = [], 0
            for k in few:
                lp, ip = full(*(at(k, h, lambda a, b: (a, b))), w)
                lm, im = full(*(at(k, -h, lambda a, b: (a, b))), w)
                same = all(
                    np.array_equal(np.asarray(a), c) for a, c in zip(ip, centre_ids)
                ) and all(
                    np.array_equal(np.asarray(a), c) for a, c in zip(im, centre_ids)
                )
                if same:
                    vals.append((k, (float(lp) - float(lm)) / (2 * h)))
                else:
                    switched += 1
            rec["full_pipeline_samples"] = int(len(few))
            rec["full_pipeline_switched"] = int(switched)
            rec["fd_eps_full"] = h
            if vals:
                ks = np.array([k for k, _ in vals])
                fdf = np.array([f for _, f in vals])
                adf = ad[wrt].ravel()[ks]
                rec["full_pipeline_rel_l2"] = float(
                    np.linalg.norm(adf - fdf) / (np.linalg.norm(fdf) or 1.0)
                )
            else:
                rec["full_pipeline_rel_l2"] = None
        if not (cap_ok and sym_ok and finite):
            rec["invalid"] = True
            rec["error"] = "capacity or gradient verdict failed: not a measurement"
        records.append(rec)
        print(
            f"{mac:>12s} {param:<8.2g} {wrt:>9s} frozen rel_l2 {rec['frozen_rel_l2']:.3e} "
            f"worst {rec['frozen_worst_rel']:.3e} dir {rec['frozen_directional_rel']:.3e} "
            f"full {rec.get('full_pipeline_rel_l2')} "
            f"(switched {rec.get('full_pipeline_switched')}) far={far}",
            flush=True,
        )
    return records


def _oracle(args, pos, mass, x, m, w_np, seed) -> list[dict]:
    """The FMM gradient against the exact fp64 direct sum's, by order."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.verification import force_law_of, reference_force, select_targets

    n = len(pos)
    targets = select_targets(n, min(args.oracle_targets, n), seed=seed)
    w_t = np.zeros_like(w_np)
    w_t[targets] = w_np[targets]
    wj = jnp.asarray(w_t, x.dtype)
    rows, direct, ref = [], None, None
    for p in args.order_list:
        rec: dict[str, Any] = {"order": int(p), "targets": int(len(targets))}
        try:
            fmm, scfg = L.solver(softening=args.softening, dtype=args.dtype, order=p)
            law = force_law_of(fmm)
            if direct is None:
                direct = L.direct_loss_grad(pos, mass, w_t, law, targets=targets)
                ref = reference_force(
                    pos, mass, np.arange(n, dtype=np.int64), targets, law
                )
                law_check = L.rel_l2(direct["acc"], ref.acceleration)
            prov, topo, _ = L.provider(fmm, x, m, order=p)
            far, cap_ok, sym_ok = _far_and_verdict(prov, topo)

            def pinned(t, x, m, w):
                plan, _ = prov.grad_plan(t)
                acc = prov.differentiable_accelerations(t, x, m, plan=plan)
                return jnp.sum(w * acc), acc

            (_, acc), (gx, gm) = jax.jit(
                jax.value_and_grad(pinned, argnums=(1, 2), has_aux=True)
            )(topo, x, m, wj)
            rec.update(
                mac_type=scfg["mac_type"],
                adaptive_eps=scfg["adaptive_eps"],
                theta=scfg["theta"],
                far_pairs=far,
                capacity_ok=cap_ok,
                grad_ok=sym_ok,
                grad_x_rel_l2=L.rel_l2(gx, direct["grad_x"]),
                grad_m_rel_l2=L.rel_l2(gm, direct["grad_m"]),
                force_rel_l2=L.rel_l2(np.asarray(acc)[targets], ref.acceleration),
                direct_vs_reference_force=law_check,
                law=law.as_dict(),
            )
            if not (cap_ok and sym_ok):
                rec["invalid"] = True
                rec["error"] = "capacity or gradient verdict failed"
            print(
                f"oracle p={p}: grad x {rec['grad_x_rel_l2']:.3e} m "
                f"{rec['grad_m_rel_l2']:.3e}  force {rec['force_rel_l2']:.3e} "
                f"(direct-sum law vs A4 reference {law_check:.1e})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - recorded
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"oracle p={p}: FAILED {rec['error']}", flush=True)
        rows.append(rec)
    return rows


def main() -> int:
    """Run the FD sweep and the oracle; write the artifact."""
    args = _parse_args()
    runmeta.select_gpu(args.gpu_select)
    env_set = L.prepare_env()
    import jax

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    L.patch_cpu_lane()
    import jax.numpy as jnp

    from bench.validation import _harness as H

    seed = runmeta.seed_sequence(args.seed, args.distribution, str(args.n))
    pos, mass = H.make_distribution(args.distribution, int(args.n), seed)
    dt = L.jnp_dtype(args.dtype)
    x, m = jnp.asarray(pos, dt), jnp.asarray(mass, dt)
    w_np = L.weights(args.n, seed)
    w = jnp.asarray(w_np, dt)

    records: list = []
    points = [("dehnen", t) for t in args.theta_list]
    points += [("dehnen_error", e) for e in args.eps_list]
    for mac, param in points:
        try:
            records += _mac_point(args, x, m, w, seed, mac, param)
        except Exception as exc:  # noqa: BLE001 - recorded, not plotted
            msg = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"{mac} {param}: FAILED {msg}", flush=True)
            records.append({"mac_type": mac, "mac_param": param, "error": msg})
    oracle = _oracle(args, pos, mass, x, m, w_np, seed) if args.order_list else []

    from bench.scaling import _galaxy

    config = {
        "n": int(args.n),
        "theta": args.theta_list,
        "adaptive_eps": args.eps_list,
        "order": int(args.order or _galaxy.GALAXY.max_order),
        "oracle_orders": args.order_list,
        "basis": ["real"],
        "seed": int(args.seed),
        "device": runmeta.device_label(),
        "precision": args.dtype,
        "preset": "galaxy",
        "lane": "fused",
        "leaf_size": int(_galaxy.GALAXY.leaf_size),
        "softening_kernel": str(_galaxy.GALAXY.softening_kernel),
        "distribution": args.distribution,
        "softening": float(args.softening),
        "wrt": args.wrt_list,
        "fd_samples": int(args.fd_samples),
        "fd_eps": float(args.fd_eps),
        "full_samples": int(args.full_samples),
        "oracle_targets": int(min(args.oracle_targets, args.n)),
        "loss": "sum(w . acc), w fixed-seed standard normal (grad_bench_lib.weights)",
        "cpu_route_env": env_set or None,
        "note": (
            "records: autodiff vs central FD on one pinned topology (frozen_*), "
            "and FD of the production step (rebuild included) on samples whose "
            "topology is unchanged at +-h (full_pipeline_*; switched samples "
            "excluded and counted). oracle: the FMM's gradient vs the exact fp64 "
            "direct sum's, by order, next to the force's error vs the A4 reference. "
            "Rows with `error` are not plotted."
        ),
    }
    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config=config,
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records, "oracle": oracle},
    )
    print(f"\nwrote {out}")
    bad = [r for r in records + oracle if "error" in r]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

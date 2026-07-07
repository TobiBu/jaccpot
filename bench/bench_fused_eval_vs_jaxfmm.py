"""Apples-to-apples eval benchmark: jaccpot FUSED eval-only vs jaxfmm.

Unlike ``bench_jaxfmm_paper_compare.py`` (which times jaccpot's non-fused
``prepare_state``/``evaluate_prepared_state`` path), this benchmark times the
*strict fused* static-radix eval via ``FastMultipoleMethod.strict_fused_prepared_eval_fn``
(the optimized kernels, minus refresh/velocity-Verlet), so the comparison
reflects the production lane's evaluate cost.

It also splits jaccpot's eval into near vs far (via
``JACCPOT_LARGE_N_NEARFIELD_DIAG_MODE``) and reports accuracy vs direct summation.

Findings on an RTX 2080 Ti (200k, p=4, theta=0.6, leaf/N_max=256, fp32):
  - jaccpot fused FORCE eval ~0.28s; of which far-field ~0.006s and near-field
    (direct P2P) ~0.27s -- the near-field is ~98% of the eval cost.
  - jaxfmm POTENTIAL eval ~0.05s. Apples-to-apples potential-to-potential,
    jaccpot is ~2.4x (not 5x); ~half the headline gap is that jaccpot computes
    forces (3-vector, ~2.3x costlier) while jaxfmm computes potential.
  - jaccpot's FMM far-field is already excellent; the residual near-field gap is
    a GPU-utilization gap that a pure-JAX dense-block P2P does NOT close (it
    matches the current ~0.24s force near-field regardless of batch size).

The fused-lane env flags must be set BEFORE constructing the solver.
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import time


def _set_fused_env(n_profile: str, far_pair_cap: str, neighbor_cap: str) -> None:
    defaults = {
        "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
        "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
        "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": n_profile,
        "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
        "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
        "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": far_pair_cap,
        "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
        "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": neighbor_cap,
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", default="200000", help="comma-separated particle counts")
    p.add_argument("--p", type=int, default=4, help="expansion order")
    p.add_argument("--theta", type=float, default=0.6)
    p.add_argument("--s", type=int, default=3, help="jaxfmm split parameter")
    p.add_argument("--leaf", type=int, default=256)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--acc-targets", type=int, default=256)
    p.add_argument("--far-pair-cap", default="131072")
    p.add_argument("--neighbor-cap", default="2097152")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    _set_fused_env(",".join(str(n) for n in ns), args.far_pair_cap, args.neighbor_cap)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaccpot import FastMultipoleMethod

    try:
        from jaxfmm.fmm import eval_potential
        from jaxfmm.hierarchy import gen_hierarchy
        have_jaxfmm = True
    except Exception:
        have_jaxfmm = False

    def block(x):
        return jax.tree_util.tree_map(
            lambda v: v.block_until_ready() if hasattr(v, "block_until_ready") else v, x
        )

    def timeit(fn):
        for _ in range(args.warmup):
            block(fn())
        s = []
        for _ in range(args.repeats):
            t = time.perf_counter()
            block(fn())
            s.append(time.perf_counter() - t)
        return min(s), statistics.mean(s)

    def relerr(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        num = np.linalg.norm(a - b, axis=-1) if a.ndim > 1 else np.abs(a - b)
        den = np.linalg.norm(b, axis=-1) if b.ndim > 1 else np.abs(b)
        return float(np.median(num / np.maximum(den, 1e-30)))

    def direct_accel(pts, mass, tgt, soft):
        p = pts[tgt]
        d = pts[None, :, :] - p[:, None, :]
        r2raw = jnp.sum(d * d, axis=-1)
        r2 = r2raw + soft * soft
        inv3 = jnp.where(r2raw > 0, r2 ** (-1.5), 0.0)
        return jnp.sum(d * (inv3 * mass[None, :])[..., None], axis=1)

    rows = []
    key = jax.random.PRNGKey(0)
    for n in ns:
        key, k = jax.random.split(key)
        kp, kc = jax.random.split(k)
        pts = jax.random.uniform(kp, (n, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        mass = jax.random.uniform(kc, (n,), minval=0.1, maxval=1.0, dtype=jnp.float32)
        tgt = jax.random.randint(kc, (min(args.acc_targets, n),), 0, n)
        row = {"n": n, "p": args.p, "theta": args.theta, "leaf": args.leaf}

        if have_jaxfmm:
            try:
                hier = gen_hierarchy(pts, p=args.p, theta=args.theta, s=args.s, N_max=args.leaf)
                fn = lambda: eval_potential(mass, **hier)
                mn, me = timeit(fn)
                row["jaxfmm_pot_min_s"], row["jaxfmm_pot_mean_s"] = mn, me
            except Exception as e:
                row["jaxfmm_status"] = f"{type(e).__name__}: {e}"[:160]

        try:
            solver = FastMultipoleMethod(
                preset="large_n_gpu", runtime_path="large_n",
                expansion_basis="solidfmm", complex_rotation="solidfmm",
                theta=args.theta, nearfield_mode="bucketed", nearfield_edge_chunk_size=64,
                grouped_interactions=False, working_dtype=jnp.float32,
                tree_build_mode="static_radix", fixed_order=args.p,
            )
            prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
                positions=pts, masses=mass, leaf_size=args.leaf, max_order=args.p, theta=args.theta,
            )
            mn, me = timeit(lambda: eval_fn(prepared))
            acc = np.asarray(eval_fn(prepared))
            diag = solver.get_runtime_diagnostics()
            soft = float(getattr(solver._impl, "softening", 1e-3))
            row["jaccpot_force_min_s"], row["jaccpot_force_mean_s"] = mn, me
            row["jaccpot_relerr_acc"] = relerr(acc[np.asarray(tgt)], direct_accel(pts, mass, tgt, soft))
            row["jaccpot_fused_active"] = bool(diag.get("strict_fused_mode_active"))
            row["jaccpot_fallback"] = int(diag.get("strict_fused_fallback_count", -1))
        except Exception as e:
            row["jaccpot_status"] = f"{type(e).__name__}: {e}"[:160]

        print(json.dumps(row), flush=True)
        rows.append(row)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print("WROTE", args.output, flush=True)


if __name__ == "__main__":
    main()

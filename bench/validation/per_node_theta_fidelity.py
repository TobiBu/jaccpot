"""Fidelity of the per-node effective-theta MAC against the exact criterion.

``mac_type="dehnen_theta"`` folds Dehnen eq (16a)/(16b) into one opening angle per
node so the traversal's scalar-theta test carries it and no ``pair_policy`` is
needed -- which is what unblocks the production fast lanes. The inversion of
eq (15) is exact, but one approximation is unavoidable: yggdrax's ``mac_extents``
is a single per-node array indexed by both sources and targets, so a node's extent
cannot differ by role, while eq (16a) pairs the *source's* mass and power against
the *sink's* force scale. ``dehnen_theta`` therefore pairs each node's own power
with its own force scale.

That is not guaranteed conservative, so this script measures what it costs:

- **accept-mask agreement** against the exact criterion: how much of the exact
  mask is reproduced, and how many pairs are accepted that the exact criterion
  rejects (the dangerous direction -- extra acceptance is extra error);
- **force error** against a direct sum, on Dehnen's own delta-a/f measure, so the
  comparison is at matched accuracy rather than matched eps.

Run bulge+halo as well as Plummer. The role conflation is worst where the
acceleration dynamic range is largest, which is precisely where the criterion wins
biggest (bulge+halo p99 5.9-85x against Plummer's 1.3-2.2x), so Plummer alone
would understate the risk.

Usage::

    eval $(.venv/bin/autocvd -l -q)
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \\
        python -m bench.validation.per_node_theta_fidelity \\
            --n 4096 --order 8 --distribution plummer,bulge_halo \\
            --eps 1e-4,3e-5,1e-5,3e-6 --json-out results/validation/theta_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import replace
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaccpot.config import FMMAdvancedConfig  # noqa: E402
from jaccpot.solver import FastMultipoleMethod  # noqa: E402

from .mac_error_distribution import (  # noqa: E402
    chunked_direct_accelerations,
    chunked_force_scale,
    make_distribution,
    per_particle_dehnen_scaled_error,
)


def _advanced(mac_type: str, cfg_runtime) -> FMMAdvancedConfig:
    cfg = FMMAdvancedConfig()
    return replace(
        cfg,
        mac_type=mac_type,
        runtime=replace(
            cfg.runtime,
            retain_traversal_result=True,
            retain_interactions=True,
            **cfg_runtime,
        ),
    )


def _accept_mask(state) -> set[tuple[int, int]]:
    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    keep = (sources >= 0) & (targets >= 0)
    return set(zip(sources[keep].tolist(), targets[keep].tolist()))


def _pair_work(state, order: int) -> int:
    """Hardware-independent far-field work proxy: far_pairs * (p+1)^2."""

    sources = np.asarray(state.interactions.sources)
    targets = np.asarray(state.interactions.targets)
    far = int(((sources >= 0) & (targets >= 0)).sum())
    return far * (order + 1) ** 2


def run_one(
    *, mac_type: str, eps: float, positions, masses, leaf_size, order, softening, G
):
    fmm = FastMultipoleMethod(
        preset="fast",
        basis="real",
        # theta does not gate acceptance in either mode: eq (16a) supplies its own
        # guard, and dehnen_theta's global theta cancels out of the rescaled test.
        theta=1.0,
        G=G,
        softening=softening,
        adaptive_eps=eps,
        dehnen_geometry_mode="com",
        advanced=_advanced(mac_type, {}),
    )
    state = fmm.prepare_state(positions, masses, leaf_size=leaf_size, max_order=order)
    acc = np.asarray(fmm.evaluate_prepared_state(state))
    return state, acc, fmm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--leaf-size", type=int, default=16)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--distribution", default="plummer,bulge_halo")
    ap.add_argument("--eps", default="1e-4,3e-5,1e-5,3e-6")
    ap.add_argument("--softening", type=float, default=1e-6)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    records: list[dict[str, Any]] = []
    print(
        f"{'dist':>11} {'eps':>8} {'far_exact':>10} {'far_theta':>10} "
        f"{'recall':>7} {'extra':>7} {'p99_exact':>10} {'p99_theta':>10} "
        f"{'p9999_ex':>10} {'p9999_th':>10} {'work x':>7}"
    )
    for dist in (d.strip() for d in args.distribution.split(",") if d.strip()):
        pos_np, mass_np = make_distribution(dist, args.n, args.seed)
        positions = jnp.asarray(pos_np, dtype=jnp.float64)
        masses = jnp.asarray(mass_np, dtype=jnp.float64)
        reference = chunked_direct_accelerations(
            positions, masses, softening=args.softening, G=args.G
        )
        force_scale = chunked_force_scale(
            positions, masses, softening=args.softening, G=args.G
        )
        jax.block_until_ready((reference, force_scale))

        for eps in (float(v) for v in args.eps.split(",")):
            out = {}
            for mac_type in ("dehnen_error", "dehnen_theta"):
                state, acc, _ = run_one(
                    mac_type=mac_type,
                    eps=eps,
                    positions=positions,
                    masses=masses,
                    leaf_size=args.leaf_size,
                    order=args.order,
                    softening=args.softening,
                    G=args.G,
                )
                err = per_particle_dehnen_scaled_error(acc, reference, force_scale)
                out[mac_type] = {
                    "mask": _accept_mask(state),
                    "work": _pair_work(state, args.order),
                    "p99": float(np.percentile(err, 99)),
                    "p9999": float(np.percentile(err, 99.99)),
                    "rms": float(np.sqrt(np.mean(err**2))),
                }
            exact, theta = out["dehnen_error"], out["dehnen_theta"]
            m_exact, m_theta = exact["mask"], theta["mask"]
            # recall: fraction of the exact mask reproduced. extra: pairs accepted
            # that the exact criterion rejects, relative to the exact mask size.
            recall = len(m_exact & m_theta) / max(len(m_exact), 1)
            extra = len(m_theta - m_exact) / max(len(m_exact), 1)
            rec = {
                "distribution": dist,
                "n": args.n,
                "order": args.order,
                "eps": eps,
                "far_exact": len(m_exact),
                "far_theta": len(m_theta),
                "recall": recall,
                "extra_frac": extra,
                "work_ratio": theta["work"] / max(exact["work"], 1),
            }
            for label, src in (("exact", exact), ("theta", theta)):
                for key in ("p99", "p9999", "rms"):
                    rec[f"{key}_{label}"] = src[key]
            records.append(rec)
            print(
                f"{dist:>11} {eps:>8.1e} {len(m_exact):>10d} {len(m_theta):>10d} "
                f"{recall:>7.3f} {extra:>7.3f} {exact['p99']:>10.2e} "
                f"{theta['p99']:>10.2e} {exact['p9999']:>10.2e} "
                f"{theta['p9999']:>10.2e} {rec['work_ratio']:>7.2f}",
                flush=True,
            )

    if args.json_out:
        out_path = pathlib.Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"config": vars(args), "records": records}, indent=2)
        )
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

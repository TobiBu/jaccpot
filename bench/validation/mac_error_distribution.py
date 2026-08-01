"""Fixed-theta vs Dehnen mass-dependent MAC: force-error distribution vs cost.

Dehnen (2014, arXiv:1405.2255) section 5.3 claims that replacing the geometric
opening criterion with the error-controlled criterion of eq (16a) gives a
"remarkable" reduction in the *large-error tail* at comparable *median* error,
while avoiding needlessly-accurate interactions. That is a claim about the shape
of the per-particle error distribution, not about rel-L2 and not about raw speed,
so a rel-L2 table structurally cannot support or refute it.

This script sweeps both criteria over their respective accuracy knobs, records
the full per-particle relative force-error distribution and hardware-independent
cost proxies for each, and matches the two arms at equal 90th-percentile error.

Arms
----
``fixed``   ``mac_type="dehnen"``, sweep theta. The baseline.
``mass``    ``mac_type="dehnen_error"``, sweep eps. eq (16a) verbatim.

Matching on the 90th percentile rather than the median is deliberate: when the
far field is shallow, most particles are pure near-field and the median error
saturates at machine precision, which makes median-matching degenerate.

Usage
-----
Small CPU smoke run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.validation.mac_error_distribution \\
        --n 4096 --leaf-size 16 --order 4 --distribution uniform --json-out /tmp/smoke.json

Full GPU run (use autocvd to pick a free device)::

    autocvd -- python -m bench.validation.mac_error_distribution \\
        --n 32768,131072 --leaf-size 32 --order 4,8 \\
        --distribution uniform,plummer,two_component,mass_spectrum \\
        --json-out results/validation/mac_error_distribution.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# Distributions, the direct-sum reference, Dehnen's force scale, and the
# per-particle error metrics live in `_harness` so that the paper's convergence
# figures (01/02) and this comparison measure error the same way. Extracting them
# was a pure move -- no numerics changed.
from bench.validation._harness import (  # noqa: E402
    QUANTILES,
    chunked_direct_accelerations,
    chunked_force_scale,
    error_summary,
    log_interp_at,
    make_distribution,
    per_particle_dehnen_scaled_error,
    per_particle_relative_error,
    per_particle_scaled_error,
)
from jaccpot.config import FMMAdvancedConfig  # noqa: E402
from jaccpot.runtime._adaptive_policy import (  # noqa: E402
    compute_node_force_scale_from_sorted_magnitudes,
)
from jaccpot.solver import FastMultipoleMethod  # noqa: E402

# ---------------------------------------------------------------------------
# one measurement
# ---------------------------------------------------------------------------


def _advanced(mac_type: str) -> FMMAdvancedConfig:
    cfg = FMMAdvancedConfig()
    return replace(
        cfg,
        mac_type=mac_type,
        # Both arms retain the traversal result so both pay the identical loss of
        # the streamed fast lane. Without this the mass arm would be charged for
        # a lane fallback the geometric arm avoids, and the cost comparison would
        # measure plumbing rather than the criterion.
        runtime=replace(
            cfg.runtime, retain_traversal_result=True, retain_interactions=True
        ),
    )


def measure(
    *,
    arm: str,
    knob: float,
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    force_scale: jnp.ndarray,
    leaf_size: int,
    order: int,
    geometry_mode: str,
    theta_max: Optional[float],
    softening: float,
    G: float,
) -> dict[str, Any]:
    """Run one (arm, knob) configuration and return its record."""

    if arm == "fixed":
        kwargs: dict[str, Any] = dict(theta=float(knob), advanced=_advanced("dehnen"))
    else:
        kwargs = dict(
            # theta does not gate acceptance in paper mode -- eq (16a) supplies
            # its own `theta < 1` convergence guard -- so it is pinned at 1.0 and
            # eps is the accuracy knob.
            theta=1.0,
            adaptive_eps=float(knob),
            dehnen_geometry_mode=geometry_mode,
            advanced=_advanced("dehnen_error"),
        )
        if theta_max is not None:
            kwargs["mac_theta_max"] = float(theta_max)
    kwargs["G"] = G
    kwargs["softening"] = softening

    fmm = FastMultipoleMethod(**kwargs)

    t0 = time.perf_counter()
    state = fmm.prepare_state(positions, masses, leaf_size=leaf_size, max_order=order)
    if arm == "mass_16b":
        # eq (16b) is eq (16a) with `min_b f_b` on the right-hand side instead of
        # `min_b |a_b|`; the criterion, the traversal and the error estimator are
        # untouched. So the whole of (16b) is a different force scale, and this arm
        # supplies the *exact* f_b -- an O(N^2) sum no production path would run --
        # to measure the ceiling before anyone builds an estimator for it.
        #
        # The first prepare_state above exists only to learn the tree; the node
        # count and particle ordering are MAC-independent, so re-preparing with the
        # injected scale reuses the same topology.
        f_b_sorted = jnp.asarray(force_scale)[state.tree.particle_indices]
        f_b_nodes = compute_node_force_scale_from_sorted_magnitudes(
            tree=state.tree,
            magnitudes_sorted=f_b_sorted,
            reduction="min",
        )
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=order,
            force_scale_nodes=f_b_nodes,
        )
    prepare_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = fmm.evaluate_prepared_state(state, return_potential=False)
    jax.block_until_ready(acc)
    evaluate_s = time.perf_counter() - t0

    errors = per_particle_relative_error(acc, reference)
    scaled_errors = per_particle_scaled_error(acc, reference)
    dehnen_errors = per_particle_dehnen_scaled_error(acc, reference, force_scale)

    node_ranges = np.asarray(state.tree.node_ranges)
    interactions = state.interactions
    far_pairs = 0
    far_work = 0
    if interactions is not None:
        src = np.asarray(interactions.sources)
        tgt = np.asarray(interactions.targets)
        keep = (src >= 0) & (tgt >= 0)
        far_pairs = int(keep.sum())
        far_work = far_pairs * (order + 1) ** 2

    nb = state.neighbor_list
    nb_counts = np.asarray(nb.counts)
    nb_offsets = np.asarray(nb.offsets)
    nb_neighbors = np.asarray(nb.neighbors)
    nb_leaves = np.asarray(nb.leaf_indices)
    near_pairs = 0
    near_work = 0
    for slot, leaf in enumerate(nb_leaves.tolist()):
        lo, hi = int(node_ranges[leaf, 0]), int(node_ranges[leaf, 1])
        n_t = max(hi - lo + 1, 0)
        start = int(nb_offsets[slot])
        for k in range(int(nb_counts[slot])):
            source_leaf = int(nb_neighbors[start + k])
            if source_leaf < 0:
                continue
            slo, shi = int(node_ranges[source_leaf, 0]), int(
                node_ranges[source_leaf, 1]
            )
            near_pairs += 1
            near_work += n_t * max(shi - slo + 1, 0)
        near_work += n_t * n_t  # self block

    record = {
        "arm": arm,
        "knob": float(knob),
        "far_pairs": far_pairs,
        "near_pairs": near_pairs,
        "far_work": int(far_work),
        "near_work": int(near_work),
        "pair_work": int(far_work + near_work),
        "prepare_s": prepare_s,
        "evaluate_s": evaluate_s,
        "rel_l2": float(
            np.linalg.norm(np.asarray(acc) - np.asarray(reference))
            / np.linalg.norm(np.asarray(reference))
        ),
    }
    record.update(error_summary(errors))
    record.update(error_summary(scaled_errors, prefix="scaled_"))
    record.update(error_summary(dehnen_errors, prefix="dehnen_"))
    return record


# ---------------------------------------------------------------------------
# matched-accuracy comparison
# ---------------------------------------------------------------------------


def compare_arms(
    fixed: list[dict[str, Any]],
    mass: list[dict[str, Any]],
    *,
    metric: str = "scaled_",
) -> list[dict[str, Any]]:
    """Compare the arms at matched 90th-percentile error.

    ``metric`` selects the error family: ``""`` for Dehnen's per-particle
    relative error, ``"scaled_"`` for the globally-normalised one (the default,
    because it stays meaningful where the true acceleration vanishes).
    """

    out = []
    p90 = f"{metric}p90"
    lo = max(
        min(r[p90] for r in fixed if r[p90] > 0),
        min(r[p90] for r in mass if r[p90] > 0),
    )
    hi = min(
        max(r[p90] for r in fixed),
        max(r[p90] for r in mass),
    )
    if not (hi > lo):
        return out
    for target in np.exp(np.linspace(np.log(lo), np.log(hi), 5)):
        row: dict[str, Any] = {"matched_p90": float(target)}
        ok = True
        for label, records in (("fixed", fixed), ("mass", mass)):
            for name in ("p99", "max", "median", "pair_work", "far_pairs"):
                field = f"{metric}{name}" if name in ("p99", "max", "median") else name
                val = log_interp_at(
                    records, target_p90=float(target), field=field, p90_key=p90
                )
                if val is None:
                    ok = False
                row[f"{label}_{name}"] = val
        if not ok:
            continue
        row["p99_ratio"] = (
            row["fixed_p99"] / row["mass_p99"] if row["mass_p99"] else None
        )
        row["max_ratio"] = (
            row["fixed_max"] / row["mass_max"] if row["mass_max"] else None
        )
        row["pair_work_ratio"] = (
            row["fixed_pair_work"] / row["mass_pair_work"]
            if row["mass_pair_work"]
            else None
        )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _ints(text: str) -> list[int]:
    return [int(v) for v in str(text).split(",") if v.strip()]


def _git_meta() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False
            ).stdout.strip()
        except OSError:  # pragma: no cover
            return ""

    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", default="4096")
    ap.add_argument("--leaf-size", type=int, default=16)
    ap.add_argument("--order", default="4")
    ap.add_argument("--distribution", default="uniform")
    ap.add_argument("--theta", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80")
    ap.add_argument("--eps", default="1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6")
    ap.add_argument("--geometry-mode", default="com")
    ap.add_argument(
        "--arm",
        default="fixed,mass",
        help=(
            "comma-separated arms. 'fixed' = geometric MAC swept over theta; "
            "'mass' = Dehnen eq (16a) swept over eps; 'mass_16b' = eq (16b), the "
            "same criterion with exact O(N^2) f_b as the force scale."
        ),
    )
    ap.add_argument("--theta-max", type=float, default=None)
    ap.add_argument("--softening", type=float, default=1e-3)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--metric",
        choices=("relative", "scaled", "dehnen"),
        default="dehnen",
        help=(
            "error family used for matching: 'relative' is Dehnen's per-particle "
            "relative error (valid for near-homogeneous systems); 'scaled' "
            "normalises by the global rms |a| and stays meaningful where the "
            "true acceleration vanishes (clustered systems)"
        ),
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    known_arms = ("fixed", "mass", "mass_16b")
    arms = tuple(a.strip() for a in str(args.arm).split(",") if a.strip())
    unknown = [a for a in arms if a not in known_arms]
    if unknown:
        ap.error(f"unknown --arm value(s) {unknown}; choose from {list(known_arms)}")
    if "fixed" not in arms:
        # compare_arms measures every mass arm against the geometric baseline, so
        # dropping it would silently produce an empty comparison table.
        ap.error("--arm must include 'fixed'; it is the comparison baseline")

    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for dist in str(args.distribution).split(","):
        dist = dist.strip()
        if not dist:
            continue
        for n in _ints(args.n):
            pos_np, mass_np = make_distribution(dist, n, args.seed)
            positions = jnp.asarray(pos_np, dtype=jnp.float64)
            masses = jnp.asarray(mass_np, dtype=jnp.float64)
            reference = chunked_direct_accelerations(
                positions, masses, softening=args.softening, G=args.G
            )
            jax.block_until_ready(reference)
            force_scale = chunked_force_scale(
                positions, masses, softening=args.softening, G=args.G
            )
            jax.block_until_ready(force_scale)

            for order in _ints(args.order):
                by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
                sweeps = tuple(
                    (arm, _floats(args.theta) if arm == "fixed" else _floats(args.eps))
                    for arm in arms
                )
                for arm, knobs in sweeps:
                    for knob in knobs:
                        rec = measure(
                            arm=arm,
                            knob=knob,
                            positions=positions,
                            masses=masses,
                            reference=reference,
                            force_scale=force_scale,
                            leaf_size=args.leaf_size,
                            order=order,
                            geometry_mode=args.geometry_mode,
                            theta_max=args.theta_max,
                            softening=args.softening,
                            G=args.G,
                        )
                        rec.update(
                            {
                                "distribution": dist,
                                "n": n,
                                "order": order,
                                "leaf_size": args.leaf_size,
                            }
                        )
                        by_arm[arm].append(rec)
                        records.append(rec)
                        print(
                            f"{dist:>14s} N={n:<7d} p={order} {arm:>9s} "
                            f"knob={knob:<8.3g} far={rec['far_pairs']:<7d} "
                            f"rel(med/p90/p99/max)="
                            f"{rec['median']:.1e}/{rec['p90']:.1e}/"
                            f"{rec['p99']:.1e}/{rec['max']:.1e}  "
                            f"dehnen(rms/p90/p99/p9999)="
                            f"{rec['dehnen_rms']:.1e}/{rec['dehnen_p90']:.1e}/"
                            f"{rec['dehnen_p99']:.1e}/{rec['dehnen_p9999']:.1e}",
                            flush=True,
                        )
                metric_prefix = {
                    "relative": "",
                    "scaled": "scaled_",
                    "dehnen": "dehnen_",
                }[args.metric]
                for mass_arm in (a for a in arms if a != "fixed"):
                    for row in compare_arms(
                        by_arm["fixed"], by_arm[mass_arm], metric=metric_prefix
                    ):
                        row.update(
                            {
                                "distribution": dist,
                                "n": n,
                                "order": order,
                                "mass_arm": mass_arm,
                            }
                        )
                        comparisons.append(row)

    print("\n=== matched at equal p90 (ratio > 1 favours the mass MAC) ===")
    print(
        f"{'dist':>14s} {'N':>7s} {'p':>2s} {'arm':>9s} {'p90':>9s} "
        f"{'p99 x':>7s} {'max x':>7s} {'work x':>7s}"
    )
    for row in comparisons:
        print(
            f"{row['distribution']:>14s} {row['n']:>7d} {row['order']:>2d} "
            f"{row.get('mass_arm', 'mass'):>9s} "
            f"{row['matched_p90']:.3e} {row['p99_ratio'] or float('nan'):7.2f} "
            f"{row['max_ratio'] or float('nan'):7.2f} "
            f"{row['pair_work_ratio'] or float('nan'):7.2f}"
        )

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {**_git_meta(), "args": vars(args)},
                    "records": records,
                    "comparisons": comparisons,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

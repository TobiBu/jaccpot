"""Audit reverse-pass residual memory of the differentiable FMM -- by TRACING only.

Reverse-mode memory, not forward memory, is what bounds the achievable particle
count for gradients: the forward streams its chunked scans, while reverse mode
retains one residual per scan iteration. Historically that was measured by
compiling and OOM-ing on a GPU, which costs 7-10 minutes per data point at
N>=16384 and needs an uncontended card.

``jax._src.ad_checkpoint.saved_residuals`` enumerates every retained buffer from
``make_jaxpr(linearize(f))`` -- no compilation, no GPU, seconds on CPU. And because
the per-kernel coefficients below are (by construction) independent of N, a cheap
run at N=4096 *predicts* the budget at 200k or 1M. So: measure memory by tracing,
and spend a GPU compile only once you expect to fit.

Usage
-----
    JAX_PLATFORMS=cpu python bench/audit_reverse_residuals.py --n 4096 --leaf 64

    # normalised coefficients only, machine readable
    JAX_PLATFORMS=cpu python bench/audit_reverse_residuals.py --json out.json

    # predict a budget at scale from the measured coefficients
    JAX_PLATFORMS=cpu python bench/audit_reverse_residuals.py --predict 200000,1048576

Interpreting the coefficients
-----------------------------
``B/pair`` (M2L), ``B/(level*node)`` (M2M, L2L), ``B/(edge*W)`` (near-field pair
blocks), ``B/(leaf*W^2)`` (near-field SELF blocks) and ``B/particle`` (P2M, L2P)
are the quantities that stay fixed as N grows. Multiply them by the topology
counts at the target N to get the budget. ``--predict`` does exactly that.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax._src.ad_checkpoint import saved_residuals

from jaccpot import FastMultipoleMethod
from jaccpot.config import FMMAdvancedConfig, NearFieldConfig

# Map a residual's originating function to a cost model. Order matters: the first
# matching pattern wins, so put the specific patterns first.
_ATTRIBUTION = (
    ("m2l", re.compile(r"_accumulate_m2l|_apply_m2l|_m2l_chunk|m2l_complex|m2l_real")),
    ("m2m", re.compile(r"_aggregate_m2m|m2m_complex|translate_children")),
    ("l2l", re.compile(r"_propagate_(solidfmm|real)_locals|_l2l_")),
    ("l2p", re.compile(r"_evaluate_local_expansions|evaluate_local_real")),
    ("p2m", re.compile(r"_p2m_leaves|p2m_complex")),
    (
        "nearfield",
        re.compile(r"_compute_leaf_p2p|_pair_accel|_radix_fast_lane|_self_contrib"),
    ),
    ("tree", re.compile(r"tree_moments|compute_tree_geometry|_where")),
)


def _attribute(src: str) -> str:
    for name, pattern in _ATTRIBUTION:
        if pattern.search(src):
            return name
    return "other"


def _fn_name(src: str) -> str:
    match = re.search(r"jitted function '([^']+)'", src)
    if match:
        return match.group(1)
    match = re.search(r"output of ([a-z_]+) from", src)
    return match.group(1) if match else src.split(" from ")[0][:40]


def galaxy(n, dtype, seed=3):
    """Clustered flattened disk -- the repo's galaxy-like generator."""
    rng = np.random.default_rng(seed)
    r = rng.gamma(shape=2.0, scale=1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    z = 0.1 * rng.standard_normal(n)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    mass = (np.abs(rng.standard_normal(n)) + 0.5) / n
    return jnp.asarray(pos, dtype), jnp.asarray(mass, dtype)


def _nbytes(aval) -> int:
    count = int(np.prod(aval.shape)) if aval.shape else 1
    return count * aval.dtype.itemsize


def audit(*, n, leaf, order, theta, dtype, softening, wrt, precompute_scatter=True):
    positions, masses = galaxy(n, dtype)
    # ``precompute_scatter`` matters a lot for the near-field reverse residual and
    # the canonical large-N production config sets it FALSE. The precomputed
    # schedule stacks three int32 (chunk*W) index arrays per chunk, where the plain
    # scatter needs only flat indices plus a mask -- so measuring with the default
    # True overstates the near field for the production configuration.
    advanced = FMMAdvancedConfig(
        nearfield=NearFieldConfig(precompute_scatter_schedules=bool(precompute_scatter))
    )
    fmm = FastMultipoleMethod(
        basis="complex",
        use_pallas=False,
        theta=theta,
        G=1.0,
        softening=softening,
        working_dtype=dtype,
        advanced=advanced,
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)

    inter = state.interactions
    far_pairs = int(jnp.sum(inter.counts)) if inter is not None else 0
    edges = int(state.neighbor_list.neighbors.shape[0])
    leaves = int(state.neighbor_list.counts.shape[0])
    nodes = int(state.tree.parent.shape[0])
    width = int(state.max_leaf_size)
    levels = getattr(state.tree, "node_level", None)
    depth = int(jnp.max(levels)) + 1 if levels is not None else -1

    topology = {
        "n": n,
        "leaf": leaf,
        "order": order,
        "theta": theta,
        "dtype": np.dtype(jnp.dtype(dtype)).name,
        "far_pairs": far_pairs,
        "near_edges": edges,
        "leaves": leaves,
        "nodes": nodes,
        "max_leaf_size": width,
        "depth": depth,
    }

    if wrt == "masses":

        def loss(m):
            return jnp.sum(fmm.differentiable_accelerations(state, positions, m) ** 2)

        residuals = saved_residuals(loss, masses)
    else:

        def loss(p):
            return jnp.sum(fmm.differentiable_accelerations(state, p, masses) ** 2)

        residuals = saved_residuals(loss, positions)

    per_group: dict[str, int] = defaultdict(int)
    per_fn: dict[tuple[str, str], int] = defaultdict(int)
    biggest: list[tuple[int, str, str]] = []
    total = 0
    for aval, src in residuals:
        nbytes = _nbytes(aval)
        total += nbytes
        group = _attribute(src)
        per_group[group] += nbytes
        per_fn[(group, _fn_name(src))] += nbytes
        biggest.append((nbytes, aval.str_short(), group))
    biggest.sort(reverse=True)

    # Normalisers: the quantity each group's residual is expected to scale with.
    denominators = {
        "m2l": ("B/pair", far_pairs),
        "m2m": ("B/(level*node)", depth * nodes),
        "l2l": ("B/(level*node)", depth * nodes),
        "nearfield": ("B/(edge*W)", edges * width),
        "p2m": ("B/particle", n),
        "l2p": ("B/particle", n),
        "tree": ("B/particle", n),
        "other": ("B/particle", n),
    }
    coefficients = {}
    for group, nbytes in per_group.items():
        unit, denominator = denominators.get(group, ("B/particle", n))
        coefficients[group] = {
            "bytes": nbytes,
            "unit": unit,
            "denominator": denominator,
            "coefficient": (nbytes / denominator) if denominator else None,
        }

    return {
        "topology": topology,
        "total_bytes": total,
        "bytes_per_particle": total / n,
        "groups": coefficients,
        "per_function": {
            f"{g}:{f}": b for (g, f), b in sorted(per_fn.items(), key=lambda kv: -kv[1])
        },
        "largest_residuals": [
            {"bytes": b, "aval": a, "group": g} for b, a, g in biggest[:12]
        ],
        "num_residuals": len(residuals),
    }


def _report(result):
    topology = result["topology"]
    print("=" * 78)
    print(
        f"N={topology['n']} leaf={topology['leaf']} W={topology['max_leaf_size']} "
        f"order={topology['order']} theta={topology['theta']} "
        f"dtype={topology['dtype']}"
    )
    print(
        f"  nodes={topology['nodes']} depth={topology['depth']} "
        f"leaves={topology['leaves']} far_pairs={topology['far_pairs']} "
        f"near_edges={topology['near_edges']}"
    )
    if topology["far_pairs"] == 0:
        print(
            "  !! far_pairs == 0: the M2L coefficient is UNMEASURED at this config."
            "\n     Use a smaller --leaf (e.g. 4) or larger --theta to open the far field."
        )
    print(
        f"\nTOTAL {result['total_bytes'] / 1e6:.2f} MB "
        f"({result['bytes_per_particle']:.0f} B/particle, "
        f"{result['num_residuals']} residuals)\n"
    )
    print(f"  {'group':<12} {'MB':>10}  {'coefficient':>14}  unit")
    for group, info in sorted(result["groups"].items(), key=lambda kv: -kv[1]["bytes"]):
        coefficient = info["coefficient"]
        shown = f"{coefficient:,.1f}" if coefficient is not None else "n/a"
        print(
            f"  {group:<12} {info['bytes'] / 1e6:>10.3f}  {shown:>14}  {info['unit']}"
        )
    print("\n  largest individual residuals:")
    for entry in result["largest_residuals"]:
        print(
            f"    {entry['bytes'] / 1e6:>9.3f} MB  {entry['group']:<10} {entry['aval']}"
        )
    print("\n  top functions:")
    for key, nbytes in list(result["per_function"].items())[:10]:
        print(f"    {nbytes / 1e6:>9.3f} MB  {key}")


def _predict(result, targets, *, leaf, order):
    """Extrapolate the measured coefficients to larger N.

    Uses the structural relations that hold at fixed (leaf, order, theta) on a
    self-similar distribution: pairs/N, edges/leaf and nodes/leaf are ratios, and
    depth grows like 2*log2(leaves).
    """
    topology = result["topology"]
    n0 = topology["n"]
    width = topology["max_leaf_size"]
    leaves0 = max(topology["leaves"], 1)
    pairs_per_n = topology["far_pairs"] / n0
    edges_per_leaf = topology["near_edges"] / leaves0
    print("\n" + "=" * 78)
    print("PREDICTED reverse residual (from the measured coefficients above)")
    print(
        f"  scaling: pairs/N={pairs_per_n:.2f}  edges/leaf={edges_per_leaf:.1f}  W={width}"
    )
    print(f"\n  {'N':>10} {'residual GB':>12}   dominant terms")
    for target in targets:
        leaves = max(target // max(width, 1), 1)
        nodes = 2 * leaves - 1
        depth = max(int(2 * np.log2(max(leaves, 2))), 1)
        counts = {
            "m2l": pairs_per_n * target,
            "m2m": depth * nodes,
            "l2l": depth * nodes,
            "nearfield": edges_per_leaf * leaves * width,
            "p2m": target,
            "l2p": target,
            "tree": target,
            "other": target,
        }
        parts = {}
        total = 0.0
        for group, info in result["groups"].items():
            coefficient = info["coefficient"]
            if coefficient is None:
                continue
            nbytes = coefficient * counts.get(group, target)
            parts[group] = nbytes
            total += nbytes
        ranked = sorted(parts.items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{g} {b / 1e9:.1f}" for g, b in ranked)
        print(f"  {target:>10} {total / 1e9:>12.1f}   {detail}")
    print("\n  NOTE: residual only -- add the forward working set and reverse")
    print("        transients for a peak estimate. Verify on GPU before trusting.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--leaf", type=int, default=64)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--softening", type=float, default=1e-2)
    parser.add_argument("--wrt", choices=["positions", "masses"], default="positions")
    parser.add_argument(
        "--no-precompute-scatter",
        action="store_true",
        help="match the canonical large-N config (precompute_scatter_schedules=False)",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="",
        help="comma-separated N values to extrapolate to",
    )
    parser.add_argument("--json", type=str, default="")
    args = parser.parse_args()

    dtype = jnp.float32 if args.dtype == "float32" else jnp.float64
    result = audit(
        n=args.n,
        leaf=args.leaf,
        order=args.order,
        theta=args.theta,
        dtype=dtype,
        softening=args.softening,
        wrt=args.wrt,
    )
    _report(result)
    if args.predict:
        _predict(
            result,
            [int(x) for x in args.predict.split(",")],
            leaf=args.leaf,
            order=args.order,
        )
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

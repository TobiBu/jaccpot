"""Quantify the padded-rectangle waste in the differentiable near field.

The leaf-major prepacked payload is a **rectangle** padded to the global maximum
neighbour count. Measured previously: some leaf neighbours every other leaf at
every N and every geometry tried, so that one leaf sets the slot width for all of
them. Fill was 45% at N=200000 and 14.5% at N=1000000, and the reverse pass --
which pays padded cost, having no per-leaf early exit -- grew 38x for a 5x
increase in N, tracking padding rather than valid work.

This script measures the padding structure directly from a prepared state, cheaply
and without running a reverse pass, so the ceiling on a CSR-sources rewrite can be
read off before committing to one:

* fill = valid slots / padded slots -- the ideal speedup from removing padding;
* the occupancy histogram, which is what determines whether *tiering* (the
  cheaper, already-implemented approximation to CSR) can capture most of it;
* the tier plan the shipped heuristic would actually choose, and its predicted
  slot-visit reduction.

Run (selects a free GPU via autocvd, per the repo policy)::

    python bench/audit_nearfield_padding.py --n 50000 200000
    python bench/audit_nearfield_padding.py --n 1000000 --geometry disc
"""

from __future__ import annotations

import argparse
import sys

from autocvd import autocvd

autocvd(num_gpus=1, least_used=True)

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from jaccpot import FastMultipoleMethod
from jaccpot.nearfield.near_field import build_leafpair_reverse_tiers
from jaccpot.runtime._nearfield_fastlane import (
    leaf_major_nearfield_payload_cached,
    nearfield_topology_arrays,
)


def make_system(n: int, geometry: str, seed: int = 0):
    """Build a particle distribution. Geometry matters: neighbour counts differ
    by ~3x between a uniform cube and a clustered disc at the same N."""
    rng = np.random.default_rng(seed)
    if geometry == "cube":
        positions = rng.uniform(-1.0, 1.0, size=(n, 3))
    elif geometry == "plummer":
        radius = 1.0 / np.sqrt(rng.uniform(size=n) ** (-2.0 / 3.0) - 1.0)
        direction = rng.normal(size=(n, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        positions = radius[:, None] * direction
    elif geometry == "disc":
        radius = rng.exponential(0.35, size=n)
        angle = rng.uniform(0.0, 2.0 * np.pi, size=n)
        positions = np.stack(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                rng.normal(0.0, 0.05, size=n),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    masses = rng.uniform(0.5, 1.5, size=n)
    return jnp.asarray(positions, jnp.float32), jnp.asarray(masses, jnp.float32)


def audit(n: int, geometry: str, theta: float, leaf: int, order: int) -> dict:
    positions, masses = make_system(n, geometry)
    fmm = FastMultipoleMethod(basis="real", theta=theta, softening=1e-3)
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)

    payload = leaf_major_nearfield_payload_cached(
        num_particles=int(positions.shape[0]),
        max_leaf_size=int(state.max_leaf_size),
        **nearfield_topology_arrays(
            state.tree, state.neighbor_list, state.nearfield_interop
        ),
    )
    valid = np.asarray(payload.source_leaf_valid_mask)
    leaves, blocks, block = valid.shape
    slots = blocks * block
    counts = valid.reshape(leaves, slots).sum(axis=1)
    width = int(state.max_leaf_size)

    padded_slots = leaves * slots
    valid_slots = int(counts.sum())
    fill = valid_slots / max(padded_slots, 1)

    # Tier plan the shipped heuristic would choose, and what it would save.
    tiers = build_leafpair_reverse_tiers(valid, slot_tile=8, min_gain=1.0)
    if tiers is None:
        tier_reduction, tier_count = 1.0, 0
    else:
        tiered = sum(len(members) * int(w) for members, w in tiers)
        tier_reduction = padded_slots / max(tiered, 1)
        tier_count = len(tiers)

    return {
        "n": n,
        "geometry": geometry,
        "leaves": leaves,
        "slots": slots,
        "padded_slots": padded_slots,
        "valid_slots": valid_slots,
        "fill": fill,
        "max_nbrs": int(counts.max()),
        "median_nbrs": float(np.median(counts)),
        "mean_nbrs": float(counts.mean()),
        "max_is_leaves_minus_1": int(counts.max()) == leaves - 1,
        # Particle-pair work: what the reverse actually executes vs what it must.
        "padded_pair_work": float(padded_slots) * width * width,
        "valid_pair_work": float(valid_slots) * width * width,
        "csr_ceiling": 1.0 / max(fill, 1e-12),
        "tier_reduction": tier_reduction,
        "tier_count": tier_count,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[20000, 50000])
    parser.add_argument(
        "--geometry",
        nargs="+",
        default=["cube", "disc"],
        choices=["cube", "plummer", "disc"],
    )
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--leaf", type=int, default=64)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args(argv)

    rows = []
    for geometry in args.geometry:
        for n in args.n:
            print(f"[audit] n={n} geometry={geometry} ...", flush=True)
            rows.append(audit(n, geometry, args.theta, args.leaf, args.order))

    header = (
        f"{'N':>9} {'geom':>8} {'leaves':>7} {'slots':>7} {'fill':>7} "
        f"{'maxnbr':>7} {'mednbr':>7} {'max=L-1':>8} {'CSR ceil':>9} "
        f"{'tiers':>6} {'tier x':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:>9} {r['geometry']:>8} {r['leaves']:>7} {r['slots']:>7} "
            f"{r['fill']*100:>6.1f}% {r['max_nbrs']:>7} {r['median_nbrs']:>7.0f} "
            f"{str(r['max_is_leaves_minus_1']):>8} {r['csr_ceiling']:>8.2f}x "
            f"{r['tier_count']:>6} {r['tier_reduction']:>6.2f}x"
        )

    print(
        "\nfill      = valid / padded slots; the reverse pays PADDED cost.\n"
        "CSR ceil  = ideal speedup from CSR sources (1/fill), an upper bound:\n"
        "            it ignores the ragged-access throughput CSR would give up.\n"
        "tier x    = predicted slot-visit reduction from the shipped tier\n"
        "            heuristic, evaluated here with min_gain=1.0 so it always\n"
        "            reports; in production it declines below 3.0x.\n"
        "The gap between 'tier x' and 'CSR ceil' is what a CSR rewrite would buy\n"
        "OVER the cheap approximation that already exists."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

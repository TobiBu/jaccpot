"""Throughput, scaling and momentum benchmark for the mutual (momentum-conserving) FMM.

Reports, per problem size:

* **interaction-list growth** -- pairs per particle for the mutual traversal *and*
  for jaccpot's own production traversal on the same tree. The mutual restructure
  must not change the asymptotics; the production numbers are the honest
  reference for what the tree and MAC already cost.
* **force throughput** -- particles/second for one full mutual evaluation, and
  the per-boundary cost of the fused block-step path.
* **traversal count** -- the fused-boundary primitive's actual win over a
  per-level interface, counted rather than asserted.
* **momentum residual** -- the property the whole path exists for, confirmed to
  be flat in N rather than degrading.

Usage::

    python -m bench.bench_mutual_fmm --sizes 4096 16384 65536 --theta 0.7 --order 4
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _system(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(0.0, 1.0, (n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, n), dtype=jnp.float64)
    return positions, masses


def _time(fn, *args, repeats: int = 3, **kwargs) -> tuple[float, Any]:
    """Return the best wall-clock time of ``repeats`` runs, plus the result.

    The first call is discarded: it pays JAX tracing and compilation, which is
    amortised away in any real run and would otherwise dominate the small sizes.
    """
    result = jax.block_until_ready(fn(*args, **kwargs))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = jax.block_until_ready(fn(*args, **kwargs))
        best = min(best, time.perf_counter() - start)
    return best, result


def _production_pair_counts(solver, positions, masses, *, leaf_size, order, theta):
    """Undirected near/far pair counts from jaccpot's own production traversal."""
    prepared = solver.prepare_state(
        positions, masses, leaf_size=leaf_size, max_order=order, theta=theta
    )
    neighbors = int(np.asarray(prepared.neighbor_list.neighbors).shape[0])
    far = (
        0
        if prepared.interactions is None
        else int(np.asarray(prepared.interactions.targets).shape[0])
    )
    return prepared, near_undirected(neighbors), far // 2


def near_undirected(directed: int) -> int:
    """Production near lists are directed; halve for comparison with canonical pairs."""
    return directed // 2


def run(sizes, *, theta, order, leaf_size, softening, k_max, repeats):
    from jaccpot import FastMultipoleMethod
    from jaccpot.mutual import (
        boundary_level_weights,
        build_mutual_state,
        build_mutual_topology_from_tree,
        mutual_accelerations,
        mutual_weighted_accelerations,
        n_sub,
    )

    solver = FastMultipoleMethod(preset="balanced", basis="real")
    header = (
        f"{'N':>8} {'leaves':>7} {'mutual/N':>9} {'prod/N':>8} {'topo s':>8} "
        f"{'force s':>9} {'part/s':>11} {'boundary s':>11} {'|sum m a|':>11}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        positions, masses = _system(n)
        prepared, near_prod, far_prod = _production_pair_counts(
            solver, positions, masses, leaf_size=leaf_size, order=order, theta=theta
        )
        start = time.perf_counter()
        topology = build_mutual_topology_from_tree(
            prepared.tree,
            np.asarray(prepared.positions_sorted),
            np.asarray(prepared.masses_sorted),
            theta=theta,
            order=order,
        )
        topo_seconds = time.perf_counter() - start
        state = build_mutual_state(topology, softening=softening)

        mutual_pairs = topology.num_far_pairs + topology.num_near_pairs
        force_seconds, accelerations = _time(
            lambda p: mutual_accelerations(state, p, masses),
            positions,
            repeats=repeats,
        )

        rung = jnp.asarray(
            np.random.default_rng(1).integers(0, k_max + 1, n), dtype=jnp.int32
        )
        weights = boundary_level_weights(1, k_max, 1.0e-3, dtype=jnp.float64)
        boundary_seconds, _ = _time(
            lambda p: mutual_weighted_accelerations(
                state, p, masses, rung=rung, level_weights=weights
            ),
            positions,
            repeats=repeats,
        )

        terms = masses[:, None] * accelerations
        residual = float(
            jnp.linalg.norm(jnp.sum(terms, axis=0))
            / (jnp.linalg.norm(jnp.sum(jnp.abs(terms), axis=0)) + 1e-300)
        )
        print(
            f"{n:8d} {topology.num_leaves:7d} {mutual_pairs / n:9.2f} "
            f"{(near_prod + far_prod) / n:8.2f} {topo_seconds:8.2f} "
            f"{force_seconds:9.3f} {n / force_seconds:11.0f} "
            f"{boundary_seconds:11.3f} {residual:11.2e}"
        )

    per_level_walks = sum(
        len(range(_floor(s, k_max), k_max + 1)) for s in range(n_sub(k_max) + 1)
    )
    print(
        f"\nfused-boundary primitive at k_max={k_max}: "
        f"{n_sub(k_max) + 1} traversals per base step vs {per_level_walks} "
        f"for a per-level interface ({per_level_walks / (n_sub(k_max) + 1):.1f}x fewer)"
    )


def _floor(s: int, k_max: int) -> int:
    from jaccpot.mutual import active_level_floor

    return active_level_floor(s, k_max)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--softening", type=float, default=1.0e-3)
    parser.add_argument("--k-max", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    run(
        args.sizes,
        theta=args.theta,
        order=args.order,
        leaf_size=args.leaf_size,
        softening=args.softening,
        k_max=args.k_max,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()

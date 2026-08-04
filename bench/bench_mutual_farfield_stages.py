"""Where the mutual far field actually spends its time.

The Phase-5 plan assumed the M2L translation was the far-field hotspot and that
fusing the rotation ``vmap``s into the Pallas kernel would be "the cheapest real
win". The backend A/B says otherwise -- the fused and z-core lanes are within 1%
of each other, which they cannot be if the rotations dominated. This script
splits the far field into its four stages so the assumption is checked rather
than argued about:

* **upward**  -- P2M into leaves + the M2M cascade;
* **M2L**     -- the dual translation itself (the only stage a Pallas M2L
  kernel can touch);
* **L2L**     -- the downward re-centring cascade;
* **L2P**     -- evaluating leaf local expansions at their particles.

Both cascades are Python loops over tree *levels*, so they emit a kernel per
level and are launch-bound at small N -- which is the hypothesis this measures.

Usage::

    python -m bench.bench_mutual_farfield_stages --sizes 10000 100000
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


def _time(fn, *args, repeats: int = 5) -> float:
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - start)
    return best


def run(sizes, *, theta: float, order: int, leaf_size: int, repeats: int) -> None:
    from jaccpot.mutual import build_mutual_state, build_mutual_topology
    from jaccpot.mutual.farfield import (
        _dual_m2l,
        _l2p_forces,
        _push_locals_down,
        mutual_upward_sweep,
    )

    print(f"device: {jax.devices()[0]}   theta={theta} order={order} leaf={leaf_size}")
    for n in sizes:
        rng = np.random.default_rng(0)
        positions = jnp.asarray(rng.normal(0.0, 1.0, (n, 3)))
        masses = jnp.asarray(rng.uniform(0.5, 1.5, n))
        topology, _ = build_mutual_topology(
            positions, masses, theta=theta, order=order, leaf_size=leaf_size
        )
        state = build_mutual_state(topology, softening=1.0e-3)
        tree = state.tree
        fwd = state.forward_permutation
        pos, mass = positions[fwd], masses[fwd]

        upward = jax.jit(lambda p: mutual_upward_sweep(p, mass, tree))
        _, centers, multipoles = jax.block_until_ready(upward(pos))

        m2l = jax.jit(
            lambda c, m: _dual_m2l(c, m, tree, None, use_pallas=False, interpret=False)
        )
        locals_ = jax.block_until_ready(m2l(centers, multipoles))

        l2l = jax.jit(lambda loc, c: _push_locals_down(loc, c, tree))
        pushed = jax.block_until_ready(l2l(locals_, centers))

        l2p = jax.jit(lambda p, loc, c: _l2p_forces(p, mass, c, loc, tree))

        t_up = _time(upward, pos, repeats=repeats)
        t_m2l = _time(m2l, centers, multipoles, repeats=repeats)
        t_l2l = _time(l2l, locals_, centers, repeats=repeats)
        t_l2p = _time(l2p, pos, pushed, centers, repeats=repeats)
        total = t_up + t_m2l + t_l2l + t_l2p

        depth = len(tree.level_nodes)
        print(
            f"\nN = {n}  leaves={topology.num_leaves} far_pairs={topology.num_far_pairs}"
            f"  tree levels={depth}"
        )
        for label, seconds in (
            ("upward (P2M+M2M)", t_up),
            ("M2L (dual)", t_m2l),
            ("L2L", t_l2l),
            ("L2P", t_l2p),
        ):
            print(
                f"  {label:<18} {seconds * 1e3:8.2f} ms   {100 * seconds / total:5.1f}%"
            )
        print(f"  {'stage sum':<18} {total * 1e3:8.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--theta", type=float, default=0.7)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--leaf-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    run(
        args.sizes,
        theta=args.theta,
        order=args.order,
        leaf_size=args.leaf_size,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()

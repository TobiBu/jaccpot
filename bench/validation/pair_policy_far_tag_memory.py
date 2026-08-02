"""What supplying a ``pair_policy`` actually costs the traversal in memory.

Step 3' -- carrying the Dehnen criterion into the fast lanes -- was gated on this
measurement. The concern on record was that ``store_far_tags = pair_policy is not
None`` allocates an extra per-pair tag buffer, and that it would land in the
*minimum-memory streamed* regime the large-N lane exists for, at 1M particles
where the reverse peak is already 11.07 GB of a 40 GB card.

Two builds have to be distinguished, because they allocate tags differently:

``monolithic`` (``build_interactions_and_neighbors``, ``_dual_tree_walk_impl``)
    Gates the tag buffer on ``store_far_tags`` and shapes it
    ``(total_nodes, max_interactions_per_node)`` -- identical to the far *index*
    buffer. So a pair policy doubles that allocation, and the cost is set by the
    capacity, not by the pair count actually found.

``split`` (``build_interactions_and_neighbors_split``, count pass then
``_dual_tree_walk_compact_fill_impl``)
    Allocates ``far_tags`` as ``jnp.full((total_far_pairs,), -1)`` whenever
    ``collect_far`` -- with **no** ``store_far_tags`` gate at all. The tag array is
    already there whether or not a policy is supplied, and it is sized by the
    exact pair count rather than by a capacity. The count pass allocates no far
    buffers at all, so it is unaffected either way.

Run it to confirm both, rather than trusting the reading::

    eval $(autocvd -l -q)
    XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=$PWD JAX_ENABLE_X64=0 \\
      python -m bench.validation.pair_policy_far_tag_memory --n 1000000 --leaf-size 256

``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is required: with preallocation on, the
allocator grabs the pool up front and ``peak_bytes_in_use`` reports the pool, not
the traversal.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import subprocess
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from yggdrax.geometry import compute_tree_geometry  # noqa: E402
from yggdrax.interactions import (  # noqa: E402
    DualTreeTraversalConfig,
    build_interactions_and_neighbors,
    build_interactions_and_neighbors_split,
)
from yggdrax.tree import Tree  # noqa: E402

MIB = 1024.0 * 1024.0


def _plummer(rng: np.random.Generator, n: int) -> np.ndarray:
    u = rng.uniform(size=n)
    radius = u ** (1.0 / 3.0) / np.sqrt(np.maximum(1.0 - u ** (2.0 / 3.0), 1e-12))
    radius = np.minimum(radius, 20.0)
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack(
        [radius * sin_t * np.cos(phi), radius * sin_t * np.sin(phi), radius * cos_t],
        axis=1,
    )


def _peak_bytes() -> Optional[int]:
    """Device peak allocation since process start.

    Process-cumulative and monotonic: there is no reliable way to reset it
    mid-process, so measuring two arms in one process reports the running maximum
    for both and the delta is always zero. Measured that way first, and it looked
    like a clean null result -- which is why each (build, arm) is now run in its
    own subprocess.
    """

    device = jax.local_devices()[0]
    stats = getattr(device, "memory_stats", None)
    if stats is None:
        return None
    payload = stats() or {}
    return payload.get("peak_bytes_in_use")


def _mirror_policy():
    """A pair policy that reproduces yggdrax's own decision, action for action.

    The point is to isolate the *cost of supplying a policy* from the cost of the
    different accept mask a real criterion would produce. A policy that decided
    anything differently would change the pair count, and the memory delta would
    then be measuring acceptance rather than the tag buffer. Mirrors
    ``_default_pair_actions`` exactly, but returns real tags instead of the
    all -1 placeholder, since emitting tags is the behaviour under test.
    """

    action_accept, action_near, action_refine = 0, 1, 2

    def policy(
        policy_state,  # noqa: ANN001 - yggdrax-owned protocol
        *,
        valid_pairs,
        mac_ok,
        different_nodes,
        target_leaf,
        source_leaf,
        **_ignored,
    ):
        should_near = (
            valid_pairs & (~mac_ok) & target_leaf & source_leaf & different_nodes
        )
        actions = jnp.full(valid_pairs.shape, action_refine, dtype=jnp.int32)
        actions = jnp.where(mac_ok, jnp.int32(action_accept), actions)
        actions = jnp.where(should_near, jnp.int32(action_near), actions)
        tags = jnp.where(mac_ok, jnp.int32(0), jnp.int32(-1))
        return actions, tags

    return policy


def measure_one(
    *, n: int, leaf_size: int, theta: float, seed: int, build: str, arm: str
) -> dict[str, Any]:
    """Measure a single (build, arm) in this process, and report its peak."""

    rng = np.random.default_rng(seed)
    positions = jnp.asarray(_plummer(rng, n), dtype=jnp.float32)
    masses = jnp.ones((n,), dtype=jnp.float32)
    tree = Tree.from_particles(
        positions,
        masses,
        leaf_size=leaf_size,
        tree_type="radix",
        target_leaf_particles=leaf_size,
        refine_local=False,
    )
    order = np.asarray(tree.particle_indices)
    geometry = compute_tree_geometry(tree, positions[order])
    jax.block_until_ready(geometry.center)

    total_nodes = int(tree.parent.shape[0])
    num_leaves = total_nodes - int(tree.num_internal_nodes)
    cfg = DualTreeTraversalConfig(
        max_pair_queue=1 << 17,
        process_block=512,
        max_interactions_per_node=8192,
        max_neighbors_per_leaf=4096,
    )
    out: dict[str, Any] = {
        "n": n,
        "leaf_size": leaf_size,
        "theta": theta,
        "total_nodes": total_nodes,
        "num_leaves": num_leaves,
        "max_interactions_per_node": cfg.max_interactions_per_node,
        # The analytic size of the buffer under discussion, exact for the
        # monolithic build: one int32 per (node, interaction slot).
        "monolithic_far_tag_buffer_mib": (
            total_nodes * cfg.max_interactions_per_node * 4 / MIB
        ),
    }

    kwargs: dict[str, Any] = dict(
        theta=theta,
        mac_type="dehnen",
        traversal_config=cfg,
        pair_policy=None if arm == "no_policy" else _mirror_policy(),
    )
    if build == "split":
        result = build_interactions_and_neighbors_split(tree, geometry, **kwargs)
    else:
        result = build_interactions_and_neighbors(tree, geometry, **kwargs)
    interactions = result[0] if isinstance(result, tuple) else result
    jax.block_until_ready(interactions.sources)
    peak = _peak_bytes()
    src = np.asarray(interactions.sources)
    tgt = np.asarray(interactions.targets)
    out.update(
        {
            "build": build,
            "arm": arm,
            "far_pairs": int(((src >= 0) & (tgt >= 0)).sum()),
            "peak_mib": None if peak is None else peak / MIB,
        }
    )
    del result, interactions
    gc.collect()
    return out


ARMS = (
    ("split", "no_policy"),
    ("split", "with_policy"),
    ("monolithic", "no_policy"),
    ("monolithic", "with_policy"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--leaf-size", type=int, default=256)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--single",
        default=None,
        help=(
            "internal: measure one 'build:arm' and print its JSON. The driver "
            "re-invokes itself with this per arm, because peak_bytes_in_use is "
            "process-cumulative and cannot be reset between arms."
        ),
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.single:
        build, arm = args.single.split(":", 1)
        print(
            json.dumps(
                measure_one(
                    n=args.n,
                    leaf_size=args.leaf_size,
                    theta=args.theta,
                    seed=args.seed,
                    build=build,
                    arm=arm,
                )
            )
        )
        return 0

    if jax.default_backend() != "gpu":
        print(
            "WARNING: not on a GPU backend; peak_bytes_in_use is a device statistic "
            "and the numbers below will not describe the lane under discussion.",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for build, arm in ARMS:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "bench.validation.pair_policy_far_tag_memory",
                "--n",
                str(args.n),
                "--leaf-size",
                str(args.leaf_size),
                "--theta",
                str(args.theta),
                "--seed",
                str(args.seed),
                "--single",
                f"{build}:{arm}",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=False,
        )
        payload = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("{"):
                payload = json.loads(line)
                break
        if payload is None:
            print(f"FAILED {build}:{arm}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            return 1
        rows.append(payload)
        print(
            f"  measured {build}:{arm}  peak={payload['peak_mib']:.1f} MiB", flush=True
        )

    head = rows[0]
    print(
        f"\nN={head['n']} leaf={head['leaf_size']} theta={head['theta']}  "
        f"nodes={head['total_nodes']} leaves={head['num_leaves']}  "
        f"capacity={head['max_interactions_per_node']}"
    )
    print(
        "monolithic far-tag buffer, analytic: "
        f"{head['monolithic_far_tag_buffer_mib']:.1f} MiB "
        "(= the far index buffer, so a policy doubles that one allocation)"
    )
    print(
        f"\n{'build':>11s} {'arm':>12s} {'far pairs':>11s} {'peak MiB':>10s} "
        f"{'policy delta':>13s}"
    )
    by_key = {(r["build"], r["arm"]): r for r in rows}
    for build in ("split", "monolithic"):
        base = by_key.get((build, "no_policy"))
        for arm in ("no_policy", "with_policy"):
            row = by_key.get((build, arm))
            if row is None:
                continue
            delta = ""
            if arm == "with_policy" and base and row["peak_mib"] and base["peak_mib"]:
                delta = f"{row['peak_mib'] - base['peak_mib']:+13.1f}"
            print(
                f"{build:>11s} {arm:>12s} {row['far_pairs']:>11d} "
                f"{row['peak_mib']:10.1f} {delta:>13s}"
            )
        if base and by_key.get((build, "with_policy")):
            if base["far_pairs"] != by_key[(build, "with_policy")]["far_pairs"]:
                print(
                    f"    WARNING: {build} changed its far-pair count between arms, "
                    "so the delta measures acceptance, not the tag buffer."
                )

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

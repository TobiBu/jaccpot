"""Warm-call ``prepare_state`` cost of the Dehnen mass MAC's force-scale prepass.

Dehnen's mass-dependent MAC (arXiv:1405.2255 eq (16a)) thresholds on
``eps * min_b |a_b|`` over the target cell, so the traversal cannot start until a
per-node force scale exists. ``mac_force_scale_mode="paper"`` obtains it from a
full extra low-order FMM on every ``prepare_state``; ``"paper_cached"`` runs that
prepass once and then reuses the cached scale, which §5.4 licenses explicitly
("only very slightly worse" than the exact ``a_b``).

This measures what that costs, as a ratio against the plain geometric MAC.

Two things this harness exists to get right, both of which produced wrong numbers
when done by hand:

- **Warm-call medians only.** At N=16384/p=8 a cold ``prepare_state`` is ~155 s
  against ~34 s warm; roughly 120 s of the cold call is JAX compilation. Comparing
  two cold calls across two processes reported a 1.29x overhead where the true
  steady-state figure was 3.5x. Every number here comes from timed calls that
  follow at least one discarded warm-up call.
- **Forced materialisation.** ``prepare_state`` returns device arrays, so a timer
  around the call alone measures dispatch. Each call is followed by
  ``block_until_ready`` on the returned pytree.

Usage::

    eval $(.venv/bin/autocvd -l -q)
    XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \\
        .venv/bin/python -m bench.validation.force_scale_prepare_cost \\
            --n 16384 --leaf-size 16 --order 8 --repeats 5 \\
            --json-out results/validation/force_scale_prepare_cost.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot import (
    FastMultipoleMethod,
    FMMAdvancedConfig,
    RuntimePolicyConfig,
)


@dataclass
class ArmResult:
    """Timings for one solver configuration."""

    name: str
    force_scale_mode: Optional[str]
    cold_seconds: float
    warm_seconds: list[float] = field(default_factory=list)
    prepass_calls: int = 0
    node_count: int = 0
    far_pairs: int = 0

    @property
    def warm_median(self) -> float:
        return statistics.median(self.warm_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "force_scale_mode": self.force_scale_mode,
            "cold_seconds": self.cold_seconds,
            "warm_seconds": self.warm_seconds,
            "warm_median_seconds": self.warm_median,
            "warm_min_seconds": min(self.warm_seconds),
            "prepass_calls": self.prepass_calls,
            "node_count": self.node_count,
            "far_pairs": self.far_pairs,
        }


def plummer(n: int, *, seed: int, dtype) -> tuple[jax.Array, jax.Array]:
    """Equal-mass untruncated Plummer sphere -- Dehnen's own test distribution."""

    key_r, key_dir = jax.random.split(jax.random.PRNGKey(seed))
    # Inverse-CDF sampling of the Plummer profile: r = a * u^(1/3) / sqrt(1 - u^(2/3)).
    u = jax.random.uniform(key_r, (n,), dtype=dtype, minval=1e-9, maxval=1.0 - 1e-9)
    radius = (u ** (1.0 / 3.0)) / jnp.sqrt(1.0 - u ** (2.0 / 3.0))
    direction = jax.random.normal(key_dir, (n, 3), dtype=dtype)
    direction = direction / jnp.linalg.norm(direction, axis=1, keepdims=True)
    positions = direction * radius[:, None]
    masses = jnp.full((n,), 1.0 / float(n), dtype=dtype)
    return positions, masses


def _traversal_runtime(args: argparse.Namespace) -> RuntimePolicyConfig:
    return RuntimePolicyConfig(
        traversal_config=DualTreeTraversalConfig(
            max_pair_queue=int(args.max_pair_queue),
            process_block=512,
            max_interactions_per_node=int(args.max_interactions_per_node),
            max_neighbors_per_leaf=int(args.max_neighbors_per_leaf),
        )
    )


def _build_solver(
    args: argparse.Namespace, *, mass_mac: bool, force_scale_mode: Optional[str]
) -> FastMultipoleMethod:
    kwargs: dict[str, Any] = {}
    if force_scale_mode is not None:
        kwargs["mac_force_scale_mode"] = force_scale_mode
    if mass_mac:
        # theta does not gate acceptance in paper mode; eps is the accuracy knob.
        kwargs["adaptive_eps"] = float(args.eps)
    return FastMultipoleMethod(
        preset=args.preset,
        basis=args.basis,
        theta=float(args.theta),
        softening=float(args.softening),
        precision=args.precision,
        advanced=FMMAdvancedConfig(
            mac_type="dehnen_error" if mass_mac else "dehnen",
            runtime=_traversal_runtime(args),
        ),
        **kwargs,
    )


def _instrument_prepass(fmm: FastMultipoleMethod) -> Callable[[], int]:
    """Count force-scale prepasses on this solver; returns a reader."""

    impl = fmm._impl
    counter = [0]
    paper = type(impl)._compute_force_scale_paper_prepass_from_tree_artifacts

    def counting_paper(self, **kwargs):
        counter[0] += 1
        return paper(self, **kwargs)

    # Bound to the instance so sibling arms in the same process stay independent.
    impl._compute_force_scale_paper_prepass_from_tree_artifacts = (
        lambda **kwargs: counting_paper(impl, **kwargs)
    )
    return lambda: counter[0]


def _far_pair_count(state: Any) -> int:
    """Number of accepted M2L pairs -- a sanity check that the arm does far-field.

    eq (16a) is very conservative at small N (the threshold is set by the
    *least*-accelerated particle in the target cell), so an arm can silently
    degenerate to all-near-field. A zero here invalidates the comparison.
    """

    interactions = getattr(state, "interactions", None)
    if interactions is None:
        return 0
    # Pair slots are padded with -1; count only the live ones, as
    # mac_error_distribution.py does.
    sources = np.asarray(interactions.sources)
    targets = np.asarray(interactions.targets)
    return int(((sources >= 0) & (targets >= 0)).sum())


def _time_prepare(
    fmm: FastMultipoleMethod,
    positions: jax.Array,
    masses: jax.Array,
    args: argparse.Namespace,
) -> tuple[float, Any]:
    t0 = time.perf_counter()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=int(args.leaf_size),
        max_order=int(args.order),
    )
    jax.block_until_ready(jax.tree_util.tree_leaves(state))
    return time.perf_counter() - t0, state


def run_arm(
    args: argparse.Namespace,
    *,
    name: str,
    mass_mac: bool,
    force_scale_mode: Optional[str],
    positions: jax.Array,
    masses: jax.Array,
    jitter: jax.Array,
) -> ArmResult:
    fmm = _build_solver(args, mass_mac=mass_mac, force_scale_mode=force_scale_mode)
    read_prepasses = _instrument_prepass(fmm)

    cold_seconds, state = _time_prepare(fmm, positions, masses, args)
    result = ArmResult(
        name=name,
        force_scale_mode=(
            None if not mass_mac else str(fmm._impl.mac_force_scale_mode)
        ),
        cold_seconds=cold_seconds,
        node_count=int(state.tree.parent.shape[0]),
        far_pairs=_far_pair_count(state),
    )

    # One more discarded call: the cold call also populates the topology/locals
    # caches, so the first warm call is not yet representative of steady state.
    _time_prepare(fmm, positions, masses, args)

    for step in range(int(args.repeats)):
        # A real simulation never re-prepares identical positions. Drifting them
        # keeps the cached force scale genuinely stale, which is the regime
        # 'paper_cached' claims is good enough.
        moved = positions + float(args.jitter) * (step + 1) * jitter
        elapsed, _ = _time_prepare(fmm, moved, masses, args)
        result.warm_seconds.append(elapsed)

    result.prepass_calls = read_prepasses()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=16384)
    parser.add_argument("--leaf-size", type=int, default=16)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.6)
    parser.add_argument("--eps", type=float, default=2.0e-7)
    parser.add_argument("--softening", type=float, default=1.0e-6)
    parser.add_argument("--preset", type=str, default="fast")
    parser.add_argument("--basis", type=str, default="real")
    parser.add_argument("--precision", type=str, default="fp64")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--jitter",
        type=float,
        default=1.0e-4,
        help="per-step position drift, so cached force scales are actually stale",
    )
    parser.add_argument("--max-pair-queue", type=int, default=1 << 21)
    parser.add_argument("--max-interactions-per-node", type=int, default=1 << 17)
    parser.add_argument("--max-neighbors-per-leaf", type=int, default=1 << 17)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    dtype = jnp.float64 if args.precision == "fp64" else jnp.float32
    positions, masses = plummer(int(args.n), seed=int(args.seed), dtype=dtype)
    jitter = jax.random.normal(
        jax.random.PRNGKey(int(args.seed) + 1), positions.shape, dtype=dtype
    )

    arms = [
        ("geometric", False, None),
        ("mass_paper", True, "paper"),
        ("mass_paper_cached", True, "paper_cached"),
    ]
    results: list[ArmResult] = []
    for name, mass_mac, mode in arms:
        result = run_arm(
            args,
            name=name,
            mass_mac=mass_mac,
            force_scale_mode=mode,
            positions=positions,
            masses=masses,
            jitter=jitter,
        )
        results.append(result)
        print(
            f"{result.name:<20} cold {result.cold_seconds:8.2f}s  "
            f"warm median {result.warm_median:8.2f}s  "
            f"warm {[round(v, 2) for v in result.warm_seconds]}  "
            f"prepasses {result.prepass_calls}  far_pairs {result.far_pairs}",
            flush=True,
        )

    baseline = next(r for r in results if r.name == "geometric")
    print()
    print(f"{'arm':<20} {'warm median':>12} {'ratio vs geometric':>20}")
    for result in results:
        ratio = result.warm_median / baseline.warm_median
        print(f"{result.name:<20} {result.warm_median:11.3f}s {ratio:19.2f}x")

    payload = {
        "config": vars(args),
        "environment": {
            "jax_version": jax.__version__,
            "platform": platform.platform(),
            "devices": [str(d) for d in jax.devices()],
            "x64_enabled": bool(jax.config.jax_enable_x64),
        },
        "arms": [r.to_dict() for r in results],
        "ratios_vs_geometric": {
            r.name: r.warm_median / baseline.warm_median for r in results
        },
    }
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

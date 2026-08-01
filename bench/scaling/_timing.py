"""Shared timing methodology for the scaling benchmarks (figures 04-07).

Extracted verbatim from ``bench/bench_jaxfmm_paper_compare.py``, which was built
for the jaxFMM head-to-head and defines the timing protocol the paper reports:
warm up, then take the **minimum** over repeats with the result blocked on
device. The minimum rather than the mean because on a shared GPU the mean
measures other users' jobs as much as ours; the spread is recorded alongside it
so a noisy measurement is visible rather than hidden.

Both that script and ``bench/scaling/*.py`` import from here, so the paper's
numbers and the engineering benchmark's cannot diverge. Pure move -- no timing
semantics changed in the extraction.

Imports jax at module scope: import this only after the device has been pinned.
"""

from __future__ import annotations

import pathlib
import statistics
import sys
import time
from typing import Any, Callable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

__all__ = [
    "block_until_ready",
    "distribution",
    "fit_log_log_exponent",
    "n_values",
    "time_min_repeat",
]


def block_until_ready(value: Any) -> Any:
    """Block on every array in a pytree, so timings exclude nothing."""

    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


def time_min_repeat(
    fn: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[float, float, float]:
    """Return ``(min, mean, population stdev)`` seconds over ``repeats`` calls."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        block_until_ready(fn())

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        block_until_ready(out)
        end = time.perf_counter()
        samples.append(end - start)

    return (
        float(min(samples)),
        float(statistics.mean(samples)),
        float(statistics.pstdev(samples)),
    )


def n_values(min_exp: int, max_exp: int, steps: int) -> list[int]:
    """Return ``steps`` log-even particle counts between ``2**min_exp`` and ``2**max_exp``.

    Duplicates are collapsed, so a fine ``steps`` at a narrow exponent range does
    not silently measure the same N twice.
    """

    if steps <= 1:
        return [int(round(2 ** float(min_exp)))]
    xs = jnp.linspace(float(min_exp), float(max_exp), steps)
    vals = [int(round(2 ** float(x))) for x in xs]
    out: list[int] = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def distribution(
    key: jax.Array, n: int, name: str, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array]:
    """Return ``(points, charges)`` for the named distribution.

    These are the jaxFMM-comparison distributions, kept distinct from
    ``bench/validation/_harness.make_distribution``: the validation figures need
    a numpy-seeded astrophysical sampler (Plummer, bulge+halo), whereas the
    timing ladder needs a jax-keyed one that costs nothing to generate at
    N = 2**25. Timing does not depend on the sampler, accuracy does.
    """

    k1, k2 = jax.random.split(key)
    if name == "uniform_cube":
        pts = jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype)
    elif name == "sphere_surface":
        raw = jax.random.normal(k1, (n, 3), dtype=dtype)
        norm = jnp.linalg.norm(raw, axis=1, keepdims=True)
        pts = raw / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    elif name == "normal":
        pts = jax.random.normal(k1, (n, 3), dtype=dtype)
    else:
        raise ValueError(f"unknown distribution: {name}")
    charges = jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0, dtype=dtype)
    return pts, charges


def fit_log_log_exponent(
    ns: list[int], times: list[float], *, min_n: int = 0
) -> dict[str, float]:
    """Least-squares fit of ``log t = alpha * log N + c``; returns alpha and R^2.

    ``min_n`` drops the small-N end, where a GPU measures launch overhead rather
    than the algorithm and would bias the exponent downwards. The number of
    points actually used is returned so the fit is never quoted without knowing
    what it was fitted over.
    """

    import numpy as np

    pairs = [
        (n, t)
        for n, t in zip(ns, times)
        if n >= min_n and t is not None and t > 0 and np.isfinite(t)
    ]
    if len(pairs) < 2:
        return {
            "exponent": float("nan"),
            "r_squared": float("nan"),
            "n_points": len(pairs),
        }

    xs = np.log(np.asarray([p[0] for p in pairs], dtype=float))
    ys = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n_points": len(pairs),
        "fit_min_n": int(min(p[0] for p in pairs)),
        "fit_max_n": int(max(p[0] for p in pairs)),
    }

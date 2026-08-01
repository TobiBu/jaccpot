"""Shared differentiability benchmarking helpers, used by
autodiff_overhead.py and grad_correctness.py in this directory.

Nothing under bench/ or examples/ at the repo root currently measures
forward-vs-backward overhead or checks autodiff gradients against finite
differences -- this is new.
"""

from __future__ import annotations

import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def time_forward_and_backward(
    forward_fn: Callable[..., jnp.ndarray],
    *args,
    n_repeats: int = 5,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[float, float]:
    """Return (mean forward-only time, mean forward+backward time), seconds.

    forward_fn should be the paper's canonical single-device force-eval
    entry point (jaccpot/runtime/fmm_evaluate.py) for the single-device
    curve, or the distributed driver for the multi-GPU variant.
    """
    loss_fn = loss_fn or (lambda out: jnp.sum(out**2))
    fwd = jax.jit(forward_fn)
    fwd_and_bwd = jax.jit(jax.grad(lambda *a: loss_fn(forward_fn(*a))))

    jax.block_until_ready(fwd(*args))
    jax.block_until_ready(fwd_and_bwd(*args))

    fwd_times, bwd_times = [], []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fwd(*args))
        fwd_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        jax.block_until_ready(fwd_and_bwd(*args))
        bwd_times.append(time.perf_counter() - t0)

    return float(np.mean(fwd_times)), float(np.mean(bwd_times))


def finite_difference_check(
    forward_fn: Callable[..., jnp.ndarray],
    x: jnp.ndarray,
    *other_args,
    eps: float = 1e-5,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Central-difference gradient of loss_fn(forward_fn(x, *other_args)) w.r.t. x.

    NOTE: loops one coordinate at a time (2 forward evals each) -- at real
    particle counts only check a random subsample of positions/components,
    not the full array.
    """
    loss_fn = loss_fn or (lambda out: jnp.sum(out**2))
    x = np.asarray(x, dtype=np.float64)
    grad_fd = np.zeros_like(x)
    flat, grad_flat = x.ravel(), grad_fd.ravel()
    for i in range(flat.size):
        flat[i] += eps
        f_plus = float(loss_fn(forward_fn(x, *other_args)))
        flat[i] -= 2 * eps
        f_minus = float(loss_fn(forward_fn(x, *other_args)))
        flat[i] += eps
        grad_flat[i] = (f_plus - f_minus) / (2 * eps)
    return jnp.asarray(grad_fd)

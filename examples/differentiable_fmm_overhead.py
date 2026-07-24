"""Autodiff-overhead benchmark for the differentiable FMM force.

Measures the forward vs forward+backward wall-clock ratio of
``FastMultipoleMethod.differentiable_accelerations`` as a function of N, on a
single GPU. Because the FMM translation cascade (M2M/M2L/L2L + rotations) is
linear, its reverse pass is a transpose and the expected overhead is a small
constant factor over the forward.

Run (selects a free GPU via autocvd, per the repo policy):

    python examples/differentiable_fmm_overhead.py
"""

from __future__ import annotations

import time

from autocvd import autocvd

autocvd(num_gpus=1, least_used=True)

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".5")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np

from jaccpot import FastMultipoleMethod


def _bench(n, *, basis="complex", theta=0.5, order=4, leaf=32, reps=5, seed=0):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    fmm = FastMultipoleMethod(basis=basis, use_pallas=False, theta=theta, G=1.0, softening=1e-2)
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)

    fwd = jax.jit(lambda p, m: fmm.differentiable_accelerations(state, p, m))

    def scalar(p, m):
        return jnp.sum(fmm.differentiable_accelerations(state, p, m) ** 2)

    vg = jax.jit(jax.value_and_grad(scalar, argnums=(0, 1)))

    # warm up (compile)
    jax.block_until_ready(fwd(positions, masses))
    v, g = vg(positions, masses)
    jax.block_until_ready((v, g))

    def timeit(fn):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            best = min(best, time.perf_counter() - t0)
        return best

    t_fwd = timeit(lambda: fwd(positions, masses))
    t_vg = timeit(lambda: vg(positions, masses))
    return t_fwd, t_vg


def main():
    print(f"{'N':>8} {'fwd (ms)':>12} {'fwd+bwd (ms)':>14} {'ratio':>8}")
    for n in (256, 1024, 4096, 16384):
        t_fwd, t_vg = _bench(n)
        print(f"{n:>8} {1e3 * t_fwd:>12.2f} {1e3 * t_vg:>14.2f} {t_vg / t_fwd:>8.2f}")


if __name__ == "__main__":
    main()

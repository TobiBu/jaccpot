"""Autodiff-overhead benchmark for the differentiable FMM force.

Measures the forward vs forward+backward wall-clock ratio of
``FastMultipoleMethod.differentiable_accelerations`` as a function of N, on a
single GPU. Because the FMM translation cascade (M2M/M2L/L2L + rotations) is
linear, its reverse pass is a transpose and the expected overhead is a small
constant factor over the forward.

Exact gradients via bare ``jax.grad``/``jax.vjp`` work at every N. Wrapping the
*entire* call (which re-runs the upward/downward sweeps) in an outer ``jax.jit``
is supported at moderate N; at large N it hits host-side ops in the prepared-
state sweeps (see ``docs/differentiable_fmm_design.md`` "jit limitation"), so
this benchmark times the jitted path where available and otherwise reports the
bare (eager-dispatch) timing.

The M2L path is a toggle: by default the differentiable path uses the pure-JAX
M2L, but setting ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1`` (Ampere+ GPU) opts
into the differentiable fused-Pallas M2L fast lane (the fused kernels carry a
``custom_vjp``; see ``docs/differentiable_fmm_design.md`` PR-2). Run the script
once per mode to A/B them -- the flag is read at trace time, so use separate
process invocations to avoid a stale compilation cache:

    python examples/differentiable_fmm_overhead.py                              # pure-JAX M2L
    JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1 python examples/differentiable_fmm_overhead.py  # fused Pallas

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
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=False, theta=theta, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)

    def scalar(p, m):
        return jnp.sum(fmm.differentiable_accelerations(state, p, m) ** 2)

    fwd_eager = lambda p, m: fmm.differentiable_accelerations(state, p, m)
    vg_eager = jax.value_and_grad(scalar, argnums=(0, 1))

    # Prefer the jitted path; fall back to bare eager dispatch when the whole
    # call is not jit-traceable at this N (host ops in the prepared sweeps).
    mode = "jit"
    try:
        fwd = jax.jit(fwd_eager)
        vg = jax.jit(vg_eager)
        jax.block_until_ready(fwd(positions, masses))
        jax.block_until_ready(vg(positions, masses))
    except Exception:
        mode = "eager"
        fwd, vg = fwd_eager, vg_eager
        jax.block_until_ready(fwd(positions, masses))
        jax.block_until_ready(vg(positions, masses))

    def timeit(fn):
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn())
            best = min(best, time.perf_counter() - t0)
        return best

    t_fwd = timeit(lambda: fwd(positions, masses))
    t_vg = timeit(lambda: vg(positions, masses))
    return t_fwd, t_vg, mode


def _fused_m2l_flag() -> bool:
    return os.environ.get(
        "JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def main():
    # Report whether the fused-Pallas M2L fast lane is requested AND actually
    # engaged on this hardware (the gates require an Ampere+ GPU; they fall back
    # to the pure-JAX M2L otherwise).
    from jaccpot.runtime.kernels.core import (
        _fused_complex_m2l_pallas_active,
        _real_m2l_pallas_active,
    )

    requested = _fused_m2l_flag()
    engaged = bool(_fused_complex_m2l_pallas_active() and _real_m2l_pallas_active())
    m2l = "fused-Pallas" if (requested and engaged) else "pure-JAX"
    if requested and not engaged:
        m2l = "pure-JAX (fused-Pallas requested but unsupported here)"
    print(f"M2L path: {m2l}", flush=True)
    print(
        f"{'N':>8} {'basis':>8} {'mode':>6} {'fwd (ms)':>12} {'fwd+bwd (ms)':>14} {'ratio':>8}",
        flush=True,
    )
    for basis in ("complex", "real"):
        for n in (256, 1024, 4096):
            try:
                t_fwd, t_vg, mode = _bench(n, basis=basis)
                print(
                    f"{n:>8} {basis:>8} {mode:>6} {1e3 * t_fwd:>12.2f} "
                    f"{1e3 * t_vg:>14.2f} {t_vg / t_fwd:>8.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 -- report and continue (e.g. OOM)
                print(f"{n:>8} {basis:>8} skipped: {type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()

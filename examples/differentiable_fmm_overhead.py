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

Two paths are toggles, both read at TRACE time -- A/B them with separate process
invocations so a stale compilation cache cannot leak across modes. Both are set
here through ``GradConfig`` (the supported interface); the ``JACCPOT_*``
environment variables remain equivalent fallbacks.

* **M2L.** ``--fused-m2l`` sets ``GradConfig(fused_m2l_pallas=True)``, opting into
  the differentiable fused-Pallas M2L on an Ampere+ GPU (the fused kernels carry
  a ``custom_vjp``). Default is the pure-JAX M2L.
* **Near field.** ``--near-field {auto,bucketed,fast_lane}`` sets
  ``GradConfig(nearfield_lane=...)``. ``fast_lane`` re-expresses the near field
  leaf-major and routes it through the radix fast lane and its analytic O(N)
  leaf-pair reverse. Same edge set, same force -- a different traversal. Pair it
  with ``--use-pallas`` to run the fused-Pallas leaf-major kernel rather than the
  pure-JAX leaf-major fallback. The default ``auto`` picks bucketed below
  N=100000 and the fast lane at or above it.

    python examples/differentiable_fmm_overhead.py                                    # defaults
    python examples/differentiable_fmm_overhead.py --fused-m2l                        # fused Pallas M2L
    python examples/differentiable_fmm_overhead.py --near-field fast_lane             # leaf-major near field
    python examples/differentiable_fmm_overhead.py --near-field fast_lane --use-pallas

Run (selects a free GPU via autocvd, per the repo policy):

    python examples/differentiable_fmm_overhead.py
"""

from __future__ import annotations

import sys
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

from jaccpot import FastMultipoleMethod, GradConfig


def _bench(
    n,
    *,
    basis="complex",
    theta=0.5,
    order=4,
    leaf=32,
    reps=5,
    seed=0,
    use_pallas=False,
    grad_config=None,
):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(n, 3)), dtype=jnp.float64)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(n,)), dtype=jnp.float64)
    fmm = FastMultipoleMethod(
        basis=basis, use_pallas=use_pallas, theta=theta, G=1.0, softening=1e-2
    )
    state = fmm.prepare_state(positions, masses, max_order=order, leaf_size=leaf)

    def accel(p, m):
        return fmm.differentiable_accelerations(state, p, m, grad_config=grad_config)

    def scalar(p, m):
        return jnp.sum(accel(p, m) ** 2)

    fwd_eager = accel
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


def _env_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def main():
    use_pallas = "--use-pallas" in sys.argv
    fused_m2l = "--fused-m2l" in sys.argv or _env_on(
        "JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS"
    )
    lane = "auto"
    if "--near-field" in sys.argv:
        lane = sys.argv[sys.argv.index("--near-field") + 1]
    elif _env_on("JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE"):
        lane = "fast_lane"
    grad_config = GradConfig(nearfield_lane=lane, fused_m2l_pallas=fused_m2l)

    # Report whether the fused-Pallas M2L fast lane is requested AND actually
    # engaged on this hardware (the gates require an Ampere+ GPU; they fall back
    # to the pure-JAX M2L otherwise). The gates are context-locals, so ask them
    # inside the same override scope the benchmark will run under.
    from jaccpot.runtime.grad_options import grad_option_overrides, resolve_grad_options
    from jaccpot.runtime.kernels.core import (
        _fused_complex_m2l_pallas_active,
        _real_m2l_pallas_active,
    )

    options = resolve_grad_options(
        grad_config, num_particles=0, supports_fast_lane=True
    )
    with grad_option_overrides(options):
        engaged = bool(_fused_complex_m2l_pallas_active() and _real_m2l_pallas_active())
    m2l = "fused-Pallas" if (fused_m2l and engaged) else "pure-JAX"
    if fused_m2l and not engaged:
        m2l = "pure-JAX (fused-Pallas requested but unsupported here)"
    print(f"M2L path: {m2l}", flush=True)

    if lane == "fast_lane":
        near = "leaf-major fast lane " + (
            "(fused Pallas)" if use_pallas else "(pure-JAX)"
        )
    else:
        near = "bucketed edge list"
        if use_pallas:
            near += " (--use-pallas ignored: it only steers the fast lane)"
    print(f"Near-field path: {near}", flush=True)

    print(
        f"{'N':>8} {'basis':>8} {'mode':>6} {'fwd (ms)':>12} {'fwd+bwd (ms)':>14} {'ratio':>8}",
        flush=True,
    )
    for basis in ("complex", "real"):
        for n in (256, 1024, 4096):
            try:
                t_fwd, t_vg, mode = _bench(
                    n, basis=basis, use_pallas=use_pallas, grad_config=grad_config
                )
                print(
                    f"{n:>8} {basis:>8} {mode:>6} {1e3 * t_fwd:>12.2f} "
                    f"{1e3 * t_vg:>14.2f} {t_vg / t_fwd:>8.2f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 -- report and continue (e.g. OOM)
                print(f"{n:>8} {basis:>8} skipped: {type(exc).__name__}", flush=True)


if __name__ == "__main__":
    main()

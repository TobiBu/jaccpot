"""Is a sub-unity forward/backward ratio launch-bound? Cost analysis says so.

fig12 reports complex-basis ratios of 0.78-1.07 at N=4096..16384 -- a
forward-plus-backward apparently cheaper than its own forward, which is
impossible. docs/fig12_autodiff_overhead_blocked.md rules out three causes: the
jit boundary, near-field lane selection, and the loss-vs-force denominator.

This measures the one thing that document does not have: XLA's own FLOP and byte
counts for the two compiled graphs, alongside the achieved throughput.

Kept as a script rather than a one-off because the question recurs -- section 7's
fig16 hit the same signature at leaf 64 -- and because the answer needs no quiet
machine: a flop count is a property of the compiled graph, not of the run.
Usage::

    python -m bench.differentiability.latency_bound_probe If the forward's wall-clock is dominated by
kernel launch and dispatch rather than arithmetic, then the wall-clock ratio
reports which graph XLA fused into fewer kernels and NOT what differentiation
costs -- and the flop ratio, which cannot go below one, stays sane exactly where
the wall-clock ratio goes impossible.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import pathlib as _pathlib

REPO_ROOT = _pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import jax
import jax.numpy as jnp
import numpy as np

from jaccpot import FastMultipoleMethod

print("device:", jax.devices()[0], flush=True)

ORDER, THETA, LEAF, SOFT = 4, 0.5, 32, 1e-2


def make(n, basis, seed=0):
    rng = np.random.default_rng(seed)
    # Plummer, as fig12 uses.
    r = 1.0 / np.sqrt(rng.uniform(0.02, 1.0, n) ** (-2.0 / 3.0) - 1.0)
    ct = rng.uniform(-1, 1, n)
    st = np.sqrt(np.maximum(1 - ct**2, 0))
    ph = rng.uniform(0, 2 * np.pi, n)
    pos = (r[:, None] * np.stack([st * np.cos(ph), st * np.sin(ph), ct], 1)).astype(
        np.float64
    )
    mass = np.full(n, 1.0 / n)
    fmm = FastMultipoleMethod(preset="fast", basis=basis, softening=SOFT, theta=THETA)
    state = fmm.prepare_state(
        jnp.asarray(pos),
        jnp.asarray(mass),
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    return fmm, state, jnp.asarray(pos), jnp.asarray(mass)


def analyse(fn, arg):
    c = fn.lower(*arg).compile()
    ca = c.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    return float(ca.get("flops", float("nan"))), float(
        ca.get("bytes accessed", float("nan"))
    )


def timeit(fn, arg, reps=10, warm=3):
    for _ in range(warm):
        jax.block_until_ready(fn(*arg))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*arg))
        ts.append(time.perf_counter() - t0)
    return min(ts)


print(
    f"\n{'basis':>9} {'N':>7} {'fwd ms':>9} {'grad ms':>9} {'wall ratio':>11} "
    f"{'flop ratio':>11} {'fwd Gflop':>10} {'fwd Gflop/s':>12}"
)
for basis in ("real", "complex"):
    for n in (1024, 4096, 16384):
        fmm, state, pos, mass = make(n, basis)
        fwd = jax.jit(lambda p, m: fmm.differentiable_accelerations(state, p, m))
        grd = jax.jit(
            jax.grad(
                lambda p, m: jnp.sum(fmm.differentiable_accelerations(state, p, m) ** 2)
            )
        )
        try:
            ff, fb = analyse(fwd, (pos, mass))
            gf, gb = analyse(grd, (pos, mass))
            tf = timeit(fwd, (pos, mass))
            tg = timeit(grd, (pos, mass))
            print(
                f"{basis:>9} {n:>7} {tf*1e3:>9.2f} {tg*1e3:>9.2f} {tg/tf:>11.2f} "
                f"{gf/ff:>11.2f} {ff/1e9:>10.4f} {ff/tf/1e9:>12.4f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"{basis:>9} {n:>7}  FAILED {type(exc).__name__}: {str(exc)[:70]}",
                flush=True,
            )

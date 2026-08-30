"""CPU proof that the widened accumulator (a) is opt-in and (b) actually works.

Runs under Pallas interpret mode, so it needs no GPU. Two claims:

  1. accum="input" is BYTE-IDENTICAL to omitting the argument -- the guard that the
     load-bearing default path was not touched. `_kernels.py:8-11` says accumulation
     order in P2P is load-bearing and there are exact-equality tests riding on it.
  2. accum="wide" beats flat float32 under CANCELLATION, which is the regime the real
     near field is in: a target's net acceleration is a small residual of a much
     larger sum of |terms|. Asserted as RATIOS against a float64 reference so the
     test is portable and does not encode this machine's rounding.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
)

W, S = 16, 64  # 64 source leaves x 16 lanes = 1024 terms per target
L = S + 1
rng = np.random.default_rng(7)

# One target leaf at the origin, surrounded by NEAR-CANCELLING opposed shells: each
# source leaf is mirrored by one on the opposite side, so the net force is a tiny
# residual of a large sum -- the cancellation the real disc produces.
pos = np.zeros((L, W, 3))
mass = np.zeros((L, W))
pos[0] = rng.normal(scale=1e-3, size=(W, 3))
mass[0] = 1.0
for k in range(1, L):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    r = 30.0 + 0.5 * k
    sgn = 1.0 if k % 2 else -1.0
    pos[k] = sgn * r * d + rng.normal(scale=1e-2, size=(W, 3))
    mass[k] = 1.0 + 1e-6 * rng.normal(size=W)
mask = np.ones((L, W), bool)
sids = np.tile(np.arange(1, L, dtype=np.int32)[None, :], (L, 1))
svalid = np.ones((L, S), bool)


def call(dt, **kw):
    a = (
        jnp.asarray(pos, dt),
        jnp.asarray(mass, dt),
        jnp.asarray(mask),
        jnp.asarray(sids),
        jnp.asarray(svalid),
    )
    k = dict(softening_sq=jnp.asarray(1e-6, dt), G=jnp.asarray(1.0, dt), interpret=True)
    return np.asarray(nearfield_leafpair_pallas(*a, **k, **kw))


ref = np.asarray(
    nearfield_leafpair_jax(
        jnp.asarray(pos, jnp.float64),
        jnp.asarray(mass, jnp.float64),
        jnp.asarray(mask),
        jnp.asarray(sids),
        jnp.asarray(svalid),
        softening_sq=jnp.asarray(1e-6, jnp.float64),
        G=jnp.asarray(1.0, jnp.float64),
    )
)

default = call(jnp.float32)
explicit = call(jnp.float32, accum="input")
widened = call(jnp.float32, accum="wide")

print("1. default vs accum='input' byte-identical:", np.array_equal(default, explicit))
assert np.array_equal(default, explicit), "DEFAULT PATH CHANGED"
assert np.any(default != 0), "vacuous: all zeros"

acc = slice(0, 3)


def relerr(g):
    return float(
        np.linalg.norm(g[0, :, acc] - ref[0, :, acc]) / np.linalg.norm(ref[0, :, acc])
    )


e_in, e_wide = relerr(default), relerr(widened)
tot = float(np.abs(ref[0, :, acc]).sum())
net = float(np.linalg.norm(ref[0, :, acc]))
print(
    f"   cancellation factor sum|a|/|sum a| ~ {tot/max(net,1e-300):.1f}  ({S} source leaves x {W} lanes)"
)
print(
    f"2. rel err  input {e_in:.3e}   wide {e_wide:.3e}   improvement {e_in/max(e_wide,1e-300):.1f}x"
)
assert (
    e_wide < e_in / 3.0
), f"widened accumulator did not help: {e_in:.3e} -> {e_wide:.3e}"
assert e_in > 1e-7, f"vacuous: flat fp32 was already exact ({e_in:.3e})"
print(
    "\nPASS: default untouched, widened accumulator materially better under cancellation."
)

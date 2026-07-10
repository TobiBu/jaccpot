"""WS C harness: isolated parity + speed bench for the leaf-pair near-field kernel.

The live fused-lane near-field kernel `nearfield_leafpair_t32_s2048_w256` is the
critical-path compute stage (a 233 ms/step swing vs pure-JAX). It vectorizes over
`bt` targets but loads sources one-at-a-time from an HBM gather table in a
`fori_loop`, with num_warps=max(1,bt//32)=1 and num_stages=1 -- low occupancy,
HBM-latency-bound.

This harness lets us iterate the kernel safely:
  * parity  -- small sizes, pallas vs the dense pure-JAX reference (<=1e-4 f32)
  * speed   -- realistic sizes (L=782, W=256, S=2048), sweep num_warps x subtile

Run (autocvd picks the GPU):
  PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=<gpu> \
      micromamba run -n odisseo python bench/nearfield_leafpair_microbench.py
"""
from __future__ import annotations
import os, time, itertools

import numpy as np
import jax
import jax.numpy as jnp

from jaccpot.pallas.nearfield_fused_leaf import (
    nearfield_leafpair_jax,
    nearfield_leafpair_pallas,
    pallas_nearfield_fused_supported,
)


def _make_inputs(L, W, S, valid_frac, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(L, W, 3)).astype(np.float32)
    mass = rng.uniform(0.5, 1.5, size=(L, W)).astype(np.float32)
    mask = np.ones((L, W), dtype=bool)
    # each target leaf has S neighbour source-leaf slots; a fraction are valid
    sids = rng.integers(0, L, size=(L, S)).astype(np.int32)
    svalid = rng.uniform(size=(L, S)) < valid_frac
    return (
        jnp.asarray(pos), jnp.asarray(mass), jnp.asarray(mask),
        jnp.asarray(sids), jnp.asarray(svalid),
    )


def parity():
    print("=== parity (small, pallas vs dense JAX reference) ===")
    L, W, S = 8, 16, 6
    pos, mass, mask, sids, svalid = _make_inputs(L, W, S, valid_frac=0.7, seed=1)
    soft = jnp.asarray(1e-4, jnp.float32)
    G = jnp.asarray(1.0, jnp.float32)
    ref = np.asarray(nearfield_leafpair_jax(pos, mass, mask, sids, svalid, softening_sq=soft, G=G))
    for bt in (16, 8):
        got = np.asarray(nearfield_leafpair_pallas(
            pos, mass, mask, sids, svalid, softening_sq=soft, G=G,
            target_subtile=bt))
        err = float(np.max(np.abs(got - ref)))
        rel = err / (float(np.max(np.abs(ref))) + 1e-30)
        ok = "OK" if rel <= 1e-5 else "FAIL"  # relative: accels have large magnitude
        print(f"  bt={bt:3d}  max_abs_err={err:.3e}  rel={rel:.3e}  [{ok}]")


def _time(fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn().block_until_ready()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3  # ms


def speed():
    L, W, S = 782, 256, 2048
    valid_frac = 256.0 / S  # ~256 valid source leaves/leaf (~real avg)
    print(f"\n=== speed (L={L} W={W} S={S} valid~{valid_frac:.2f}) ===")
    pos, mass, mask, sids, svalid = _make_inputs(L, W, S, valid_frac, seed=2)
    soft = jnp.asarray(1e-4, jnp.float32)
    G = jnp.asarray(1.0, jnp.float32)

    base = None
    print(f"  {'bt':>4} {'warps':>6} {'stages':>7} {'ms':>9} {'vs base':>9}")
    for bt, nw, ns in itertools.product((32, 64, 128), (None, 2, 4, 8), (1, 2)):
        try:
            fn = lambda bt=bt, nw=nw, ns=ns: nearfield_leafpair_pallas(
                pos, mass, mask, sids, svalid, softening_sq=soft, G=G,
                target_subtile=bt, num_warps=nw, num_stages=ns)
            ms = _time(fn)
            if bt == 32 and nw is None and ns == 1:
                base = ms
            spd = f"{base/ms:.2f}x" if base else "-"
            print(f"  {bt:>4} {str(nw):>6} {ns:>7} {ms:>9.3f} {spd:>9}")
        except Exception as e:
            print(f"  {bt:>4} {str(nw):>6} {ns:>7}  FAIL {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    cc = getattr(jax.devices()[0], "compute_capability", "n/a")
    print(f"device cc={cc}  pallas_supported={pallas_nearfield_fused_supported()}")
    parity()
    speed()

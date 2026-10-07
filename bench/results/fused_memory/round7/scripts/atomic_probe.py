"""Probe relaxed atomics on the GPU: correctness (int32/float32 add, float max) and speed."""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from jaccpot.pallas._compat import pallas_backend_kwargs
from jaccpot.pallas._relaxed_atomic import (
    RELAXED_ATOMICS,
    relaxed_atomic_add,
    relaxed_atomic_max,
)

print("RELAXED_ATOMICS", RELAXED_ATOMICS, flush=True)
B = 256


def full(shape):
    return pl.BlockSpec(shape, lambda *_: (0,) * len(shape))


def adder(fn, dtype, n, rows):
    def k(idx_ref, c_in, c_ref, old_ref):
        del c_in
        i = pl.program_id(0) * B + jnp.arange(B)
        r = idx_ref[i]
        old_ref[i] = fn(c_ref, (r,), jnp.ones((B,), dtype))

    return jax.jit(
        lambda idx, c: pl.pallas_call(
            k,
            grid=(n // B,),
            in_specs=[full((n,)), full((rows,))],
            out_specs=[full((rows,)), full((n,))],
            out_shape=[
                jax.ShapeDtypeStruct((rows,), dtype),
                jax.ShapeDtypeStruct((n,), dtype),
            ],
            input_output_aliases={1: 0},
            **pallas_backend_kwargs("triton"),
        )(idx, c)
    )


def maxer(fn, n, rows):
    def k(idx_ref, v_ref, c_in, c_ref):
        del c_in
        i = pl.program_id(0) * B + jnp.arange(B)
        fn(c_ref, (idx_ref[i],), v_ref[i])

    return jax.jit(
        lambda idx, v, c: pl.pallas_call(
            k,
            grid=(n // B,),
            in_specs=[full((n,)), full((n,)), full((rows,))],
            out_specs=full((rows,)),
            out_shape=jax.ShapeDtypeStruct((rows,), jnp.float32),
            input_output_aliases={2: 0},
            **pallas_backend_kwargs("triton"),
        )(idx, v, c)
    )


rng = np.random.default_rng(0)
n, rows = 1 << 16, 1000
idx = jnp.asarray(rng.integers(0, rows, n), jnp.int32)
want = np.bincount(np.asarray(idx), minlength=rows)
for name, fn in (("acq_rel", plgpu.atomic_add), ("relaxed", relaxed_atomic_add)):
    for dt in (jnp.float32, jnp.int32):
        c, old = adder(fn, dt, n, rows)(idx, jnp.zeros((rows,), dt))
        c, old = np.asarray(c), np.asarray(old)
        # old values per row must be a permutation of 0..count-1
        okp = all(
            sorted(old[np.asarray(idx) == r].astype(int)) == list(range(want[r]))
            for r in range(0, rows, 97)
        )
        print(
            f"add {name:8s} {np.dtype(dt).name:8s}: counts {'OK' if np.array_equal(c.astype(int), want) else 'WRONG'} old-values {'OK' if okp else 'WRONG'}",
            flush=True,
        )
v = jnp.asarray(rng.random(n), jnp.float32)
wmax = np.zeros(rows, np.float32)
np.maximum.at(wmax, np.asarray(idx), np.asarray(v))
for name, fn in (("acq_rel", plgpu.atomic_max), ("relaxed", relaxed_atomic_max)):
    c = np.asarray(maxer(fn, n, rows)(idx, v, jnp.zeros((rows,), jnp.float32)))
    print(f"max {name:8s}: {'OK' if np.array_equal(c, wmax) else 'WRONG'}", flush=True)
# speed: 2^25 float atomics over 2^21 rows (random), as in the placement
n, rows = 1 << 25, 1 << 21
idx = jnp.asarray(rng.integers(0, rows, n), jnp.int32)
for name, fn in (("acq_rel", plgpu.atomic_add), ("relaxed", relaxed_atomic_add)):
    f = adder(fn, jnp.float32, n, rows)
    z = jnp.zeros((rows,), jnp.float32)
    jax.block_until_ready(f(idx, z))
    ts = []
    for _ in range(5):
        z = jnp.zeros((rows,), jnp.float32)
        jax.block_until_ready(z)
        t0 = time.perf_counter()
        jax.block_until_ready(f(idx, z))
        ts.append(1e3 * (time.perf_counter() - t0))
    print(
        f"speed add {name}: {min(ts):.3f} ms for {n} atomics ({n / min(ts) / 1e6:.1f} G/s)",
        flush=True,
    )

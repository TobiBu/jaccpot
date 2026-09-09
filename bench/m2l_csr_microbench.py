"""Go/no-go microbench for the CSR M2L Pallas kernel (plan "small leaves", gate G2a).

Synthetic far-pair lists of 1M and 6M directed pairs over ``n`` nodes (the leaf
64 / leaf 32 counts at N=200k), orders 4 and 6, fp32:

* ``m2l_real_csr_pallas`` (one program per target, rotations on chip), jitted;
* the pure-JAX chunked lane as production runs it -- ``m2l_rot_scale_real_batch``
  over 4096-pair chunks in a ``lax.scan`` with ``_chunk_segment_scatter_add`` --
  with ``JACCPOT_M2L_DEGREE_BATCHED`` off and on (the plan's 2.0 candidate, and
  the honest pure-JAX baseline).

Reports ns per directed pair (min of the timed calls) and the fp32 rel-L2 of
the kernel against the pure-JAX lane. Gate: <= 15 ns/pair at p=4 and parity
<= 1e-5 rel is "go" for wiring; the pure-JAX lane sits at ~130-375 ns/pair.

    CUDA_VISIBLE_DEVICES=<idle> python bench/m2l_csr_microbench.py [--pairs 1000000 6000000] [--orders 4 6]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", default=[1_000_000, 6_000_000])
    ap.add_argument("--orders", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--nodes", type=int, default=12_500)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from jax import lax

    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.operators.real_harmonics import sh_size
    from jaccpot.pallas.m2l_real_csr import m2l_real_csr_pallas
    from jaccpot.runtime.kernels._m2l import _chunk_segment_scatter_add

    dev = jax.devices()[0]
    print(f"device {dev} cc={getattr(dev, 'compute_capability', '?')}", flush=True)
    rng = np.random.default_rng(0)
    n = int(args.nodes)
    centers = jnp.asarray(rng.uniform(-1, 1, (n, 3)).astype(np.float32))
    results = []

    def timed(fn, *a):
        out = jax.block_until_ready(fn(*a))
        ts = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = jax.block_until_ready(fn(*a))
            ts.append(time.perf_counter() - t0)
        return out, min(ts), float(np.median(ts))

    for order in args.orders:
        C = sh_size(order)
        mult = jnp.asarray(rng.standard_normal((n, C)).astype(np.float32))
        for P in args.pairs:
            # random well-separated pairs: reject |delta| < 0.3 (the MAC would too)
            src = rng.integers(0, n, P).astype(np.int32)
            tgt = rng.integers(0, n, P).astype(np.int32)
            cn = np.asarray(centers)
            for _ in range(50):  # resample until every pair is well separated
                bad = np.linalg.norm(cn[tgt] - cn[src], axis=1) < 0.3
                if not bad.any():
                    break
                src = np.where(bad, rng.integers(0, n, P), src).astype(np.int32)
            assert not bad.any()
            src_j, tgt_j = jnp.asarray(src), jnp.asarray(tgt)
            counts = np.bincount(tgt, minlength=n)

            csr = jax.jit(
                lambda m, c, s, t: m2l_real_csr_pallas(m, c, s, t, order=order)
            )
            out_k, t_k, med_k = timed(csr, mult, centers, src_j, tgt_j)

            chunk = int(args.chunk)
            n_chunks = -(-P // chunk)
            pad = n_chunks * chunk - P
            src_p = jnp.pad(src_j, (0, pad), constant_values=0)
            tgt_p = jnp.pad(tgt_j, (0, pad), constant_values=0)

            def pure(m, c, s, t, P=P, chunk=chunk, n_chunks=n_chunks):
                def body(acc, i):
                    idx = i * chunk + jnp.arange(chunk, dtype=jnp.int32)
                    valid = idx < P
                    sc = s[idx]
                    tc = t[idx]
                    contrib = m2l_rot_scale_real_batch(
                        m[sc], c[tc] - c[sc], order=order
                    )
                    return (
                        _chunk_segment_scatter_add(
                            acc, contrib, tc, valid, chunk_size=chunk
                        ),
                        None,
                    )

                acc, _ = lax.scan(
                    body,
                    jnp.zeros((n, C), jnp.float32),
                    jnp.arange(n_chunks, dtype=jnp.int32),
                )
                return acc

            rows = {}
            for db in ("0", "1"):
                os.environ["JACCPOT_M2L_DEGREE_BATCHED"] = db
                # the knob is read at trace time and the cascade's inner jits are
                # cached across variants -- without this the second variant reuses
                # the first's trace (measured: bit-identical output and time)
                jax.clear_caches()
                fn = jax.jit(pure)
                out_p, t_p, med_p = timed(fn, mult, centers, src_p, tgt_p)
                rows[db] = (out_p, t_p, med_p)
                del fn
            ref = np.asarray(rows["0"][0], np.float64)
            assert np.all(np.isfinite(ref)), "pure-JAX reference has non-finite rows"
            assert np.all(
                np.isfinite(np.asarray(out_k))
            ), "CSR kernel produced non-finite rows"
            rel = float(
                np.linalg.norm(np.asarray(out_k, np.float64) - ref)
                / np.linalg.norm(ref)
            )
            rel_db = float(
                np.linalg.norm(np.asarray(rows["1"][0], np.float64) - ref)
                / np.linalg.norm(ref)
            )
            row = dict(
                order=order,
                pairs=P,
                nodes=n,
                longest_row=int(counts.max()),
                csr_ns_per_pair=1e9 * t_k / P,
                csr_ms=1e3 * t_k,
                csr_median_ms=1e3 * med_k,
                pure_ns_per_pair=1e9 * rows["0"][1] / P,
                pure_ms=1e3 * rows["0"][1],
                pure_db_ns_per_pair=1e9 * rows["1"][1] / P,
                pure_db_ms=1e3 * rows["1"][1],
                rel_l2_csr_vs_pure=rel,
                rel_l2_db_vs_pure=rel_db,
                chunk=chunk,
            )
            results.append(row)
            print(
                f"p={order} pairs={P/1e6:.0f}M longest_row={counts.max()}: CSR {row['csr_ns_per_pair']:.1f} ns/pair "
                f"({row['csr_ms']:.1f} ms) | pure-JAX {row['pure_ns_per_pair']:.1f} ns/pair | degree-batched "
                f"{row['pure_db_ns_per_pair']:.1f} ns/pair | rel-L2 csr {rel:.2e}, db {rel_db:.2e}",
                flush=True,
            )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()

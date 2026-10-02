"""What does one cross-domain exchange round cost, and which XLA implementation is fastest?

The fused multi-GPU lane moves its cross field through four ``ragged_all_to_all`` rounds per
force (far payload, far CSR, near payload, near CSR). At 1M particles per card on two
PCIe A100s those rounds take 16.3 + 5.1 + 5.1 + 28.6 ms -- an effective ~2 GB/s. This
bench times the SAME helper the cross hook calls
(``yggdrax.distributed.comm.ragged_all_to_all_exchange``) over the row widths and row
counts the lane uses, so the XLA implementation can be chosen from numbers:

* payload rows of 56 floats (far: (p+1)^2 = 49 coefficients + 7) and 312 floats
  (near: a 64-slot particle tile + geometry + multipole), CSR rows of 2 x int32;
* live rows L, with the receive capacity at 2L, plus one run at 8L to see whether the
  cost follows the live bytes or the capacity (the helper clears the whole receive
  buffer on every call);
* references at the same bytes: a dense ``lax.all_to_all``, an ``all_gather`` and a raw
  device-to-device copy (``jax.device_put``).

The XLA implementation is chosen by the CALLER through ``XLA_FLAGS`` -- one process per
setting, since the flags are read once when the backend starts:

    --xla_gpu_ragged_all_to_all_mode=COLLECTIVES_{PEER,PRIVATE,SYMMETRIC}_MEMORY
    --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false
    --xla_gpu_unsupported_enable_ragged_all_to_all_decomposer=true

    CUDA_VISIBLE_DEVICES=1,2 BENCH_LABEL=default BENCH_JSON=out.json python bench/multigpu_exchange_bench.py
"""

from __future__ import annotations

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from yggdrax.distributed.comm import ragged_all_to_all_exchange

AXIS = "gpus"
LABEL = os.environ.get("BENCH_LABEL", "default")
REPS = int(os.environ.get("BENCH_REPS", "15"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "3"))

devices = jax.devices()
NDEV = len(devices)
assert NDEV >= 2, "needs two devices"
mesh = jax.sharding.Mesh(np.asarray(devices), (AXIS,))
row_spec = NamedSharding(mesh, P(AXIS))


def _time(fn, *args) -> dict:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    s = np.sort(np.asarray(samples)) * 1e3
    return {"min_ms": float(s[0]), "median_ms": float(np.median(s)), "out": out}


def ragged_case(live: int, width: int, cap_factor: int, dtype) -> dict:
    """Every device sends ``live`` rows to every OTHER device (the cross hook masks its own)."""
    cap_in = live * (NDEV - 1)
    cap_out = cap_factor * live * (NDEV - 1)
    rng = np.random.default_rng(0)
    host = rng.standard_normal((NDEV * cap_in, width)).astype(np.float32)
    if dtype == jnp.int32:
        host = (host * 1000).astype(np.int32)
    x = jax.device_put(jnp.asarray(host, dtype), row_spec)

    def body(xb):
        me = jax.lax.axis_index(AXIS)
        sizes = jnp.where(jnp.arange(NDEV) == me, 0, live).astype(jnp.int32)
        out, recv, _off = ragged_all_to_all_exchange(
            xb, sizes, output_capacity=cap_out, axis_name=AXIS, method="native"
        )
        return out, recv[None]

    fn = jax.jit(
        shard_map(
            body,
            mesh=mesh,
            in_specs=(P(AXIS),),
            out_specs=(P(AXIS), P(AXIS)),
            check_vma=False,
        )
    )
    r = _time(fn, x)
    out, recv = r.pop("out")
    # correctness: device 1 receives device 0's block first (device 0 sends rows
    # [0, live) of its buffer to device 1, since it skips itself)
    got = np.asarray(out[cap_out : cap_out + 4])
    want = host[0:4]
    ok = bool(np.array_equal(got, want.astype(np.asarray(got).dtype)))
    nbytes = live * width * jnp.dtype(dtype).itemsize  # sent per device per peer
    r.update(
        kind="ragged",
        live_rows=live,
        width=width,
        dtype=str(jnp.dtype(dtype)),
        cap_factor=cap_factor,
        bytes_per_peer=int(nbytes),
        correct=ok,
        recv_rows=int(np.asarray(recv)[1, 0]),
        gbps=float(nbytes / (r["min_ms"] * 1e-3) / 1e9),
    )
    return r


def dense_case(live: int, width: int) -> dict:
    host = (
        np.random.default_rng(1)
        .standard_normal((NDEV * NDEV * live, width))
        .astype(np.float32)
    )
    x = jax.device_put(jnp.asarray(host).reshape(NDEV * NDEV, live, width), row_spec)

    def body(xb):  # [NDEV, live, width] per device -> one block to each device
        return jax.lax.all_to_all(xb, AXIS, 0, 0, tiled=False)

    fn = jax.jit(
        shard_map(
            body, mesh=mesh, in_specs=(P(AXIS),), out_specs=P(AXIS), check_vma=False
        )
    )
    r = _time(fn, x)
    r.pop("out")
    nbytes = live * width * 4
    r.update(
        kind="dense_all_to_all",
        live_rows=live,
        width=width,
        bytes_per_peer=int(nbytes),
        gbps=float(nbytes / (r["min_ms"] * 1e-3) / 1e9),
    )
    return r


def gather_case(live: int, width: int) -> dict:
    host = (
        np.random.default_rng(2)
        .standard_normal((NDEV * live, width))
        .astype(np.float32)
    )
    x = jax.device_put(jnp.asarray(host), row_spec)

    def body(xb):
        return jax.lax.all_gather(xb, AXIS, tiled=True)

    fn = jax.jit(
        shard_map(
            body, mesh=mesh, in_specs=(P(AXIS),), out_specs=P(AXIS), check_vma=False
        )
    )
    r = _time(fn, x)
    r.pop("out")
    nbytes = live * width * 4
    r.update(
        kind="all_gather",
        live_rows=live,
        width=width,
        bytes_per_peer=int(nbytes),
        gbps=float(nbytes / (r["min_ms"] * 1e-3) / 1e9),
    )
    return r


def copy_case(live: int, width: int) -> dict:
    a = jax.device_put(jnp.ones((live, width), jnp.float32), devices[0])
    jax.block_until_ready(a)
    r = _time(lambda v: jax.device_put(v, devices[1]), a)
    r.pop("out")
    nbytes = live * width * 4
    r.update(
        kind="device_put_copy",
        live_rows=live,
        width=width,
        bytes_per_peer=int(nbytes),
        gbps=float(nbytes / (r["min_ms"] * 1e-3) / 1e9),
    )
    return r


rows = []
print(
    f"[{LABEL}] devices {[d.id for d in devices]}  XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}",
    flush=True,
)
cases = [
    # far payload rows (56 floats) and near payload rows (312 floats)
    *[("ragged", L, 56, 2, jnp.float32) for L in (1 << 14, 1 << 16, 1 << 18)],
    *[("ragged", L, 312, 2, jnp.float32) for L in (1 << 12, 1 << 14, 1 << 16)],
    # CSR rows: 2 x int32
    *[("ragged", L, 2, 2, jnp.int32) for L in (1 << 18, 1 << 20, 1 << 22)],
    # does the cost follow the capacity? same live rows, 4x the receive buffer
    ("ragged", 1 << 16, 56, 8, jnp.float32),
    ("ragged", 1 << 20, 2, 8, jnp.int32),
]
for kind, L, w, cf, dt in cases:
    try:
        r = ragged_case(L, w, cf, dt)
    except Exception as exc:  # noqa: BLE001 -- a mode may refuse a shape; record it
        r = dict(
            kind="ragged", live_rows=L, width=w, cap_factor=cf, error=repr(exc)[:300]
        )
    r["label"] = LABEL
    rows.append(r)
    if "error" in r:
        print(
            f"  ragged L={L:>8} w={w:>3} cap x{cf}: ERROR {r['error'][:160]}",
            flush=True,
        )
    else:
        print(
            f"  ragged L={L:>8} w={w:>3} {r['dtype']:>7} cap x{cf}: {r['min_ms']:8.3f} ms "
            f"(median {r['median_ms']:8.3f})  {r['bytes_per_peer'] / 1e6:8.2f} MB  {r['gbps']:6.2f} GB/s  "
            f"correct={r['correct']}",
            flush=True,
        )
if os.environ.get("BENCH_REFERENCES", "1") == "1":
    for L, w in ((1 << 16, 56), (1 << 16, 312), (1 << 20, 2)):
        for fn in (dense_case, gather_case, copy_case):
            r = fn(L, w)
            r["label"] = LABEL
            rows.append(r)
            print(
                f"  {r['kind']:<16} L={L:>8} w={w:>3}: {r['min_ms']:8.3f} ms  "
                f"{r['bytes_per_peer'] / 1e6:8.2f} MB  {r['gbps']:6.2f} GB/s",
                flush=True,
            )
if os.environ.get("BENCH_JSON"):
    json.dump(rows, open(os.environ["BENCH_JSON"], "w"), indent=1)

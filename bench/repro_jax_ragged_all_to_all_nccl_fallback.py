"""Upstream repro: jax 0.11.2's NCCL ragged_all_to_all moves no data, then deadlocks.

No jaccpot, no yggdrax: ``jit(shard_map(jax.lax.ragged_all_to_all))`` on two GPUs of one
process. Every device sends ``--rows`` rows of ``--width`` float32 to each device (itself
included). ``--mode`` picks XLA's implementation (XLA reads its flags once, so the script
sets ``XLA_FLAGS`` before importing jax; anything already in ``XLA_FLAGS`` is kept):

* ``default``: XLA's one-shot peer-memory kernel.
* ``nccl``: the NCCL send/recv fallback,
  ``--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`` plus, from jax
  0.11 on, ``--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true`` (without
  it XLA refuses the fallback with INVALID_ARGUMENT).
* ``nccl-nobarrier`` / ``nccl-barrier``: ``nccl`` plus
  ``--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`` / ``=true``.

Measured 2026-10-12, 2x A100-PCIE-40GB driven by one process (GPUs 3+5, peer access on),
width 56, 10 calls each:

| jax | mode | rows 1, 64 (<= 14 KB) | rows 512 ... 16384 (115 KB ... 3.7 MB) |
| --- | --- | --- | --- |
| 0.11.2 | default | | 10/10 (16384) |
| 0.11.2 | nccl (barrier on: the 0.11 default) | 10/10 | call 0: only the self blocks arrive, the cross-device half keeps the fill value; call 1 never returns |
| 0.11.2 | nccl-nobarrier | 10/10 | 10/10 |
| 0.10.2 | nccl (barrier off: the 0.10 default) | | 10/10 (16384) |
| 0.10.2 | nccl-barrier | | the same failure as 0.11.2 nccl (16384) |

So the defect is the NCCL fallback combined with ``use_barrier_with_nccl``, and it is
older than 0.11: jax 0.11's XLA (openxla 91888df6) turned that flag on by default, jax
0.10.2's (5a9e73cb) left it off. The NCCL library is not the variable (0.10.2 with NCCL
2.32.3, 0.11.2's, is clean). With the flag on, XLA puts the ragged output in collective
memory (``S(1)`` in the optimized HLO) and copies it out at the end; NCCL's debug log shows
matching send/recv counts on both ranks, receiving into that buffer. A hung call spins both
GPUs at 100 %, so each call runs under a timeout and the process exits on the first one
that does not return.

    CUDA_VISIBLE_DEVICES=0,1 python bench/repro_jax_ragged_all_to_all_nccl_fallback.py \\
        --mode nccl --rows 16384 --width 56
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

_ONE_SHOT_OFF = "--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false"
# jax 0.11's XLA refuses the fallback unless it is allowed; 0.10's does not know the flag
_ALLOW = "--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true"
_BARRIER = "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl="


def _jaxlib_at_least(major: int, minor: int) -> bool:
    from importlib.metadata import version

    parts = version("jaxlib").split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


_NCCL = f"{_ONE_SHOT_OFF} {_ALLOW}" if _jaxlib_at_least(0, 11) else _ONE_SHOT_OFF
MODES = {
    "default": "",
    "nccl": _NCCL,
    "nccl-nobarrier": f"{_NCCL} {_BARRIER}false",
    "nccl-barrier": f"{_NCCL} {_BARRIER}true",
}

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("--mode", choices=sorted(MODES), default="nccl")
ap.add_argument("--rows", type=int, default=16384, help="rows per (sender, receiver)")
ap.add_argument("--width", type=int, default=56, help="float32 per row")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--timeout", type=float, default=60.0, help="seconds per call")
args = ap.parse_args()
os.environ["XLA_FLAGS"] = (
    f"{os.environ.get('XLA_FLAGS', '')} {MODES[args.mode]}".strip()
)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

NDEV, R, W, FILL = 2, args.rows, args.width, -1.0
if len(jax.devices()) < NDEV:
    raise SystemExit(f"needs >= {NDEV} GPUs in one process, found {jax.devices()}")
mesh = Mesh(np.array(jax.devices()[:NDEV]), ("gpus",))
rows = NamedSharding(mesh, P("gpus"))

# Device s holds 2R rows; rows [t R, (t+1) R) go to device t and land there at offset s R.
host = np.repeat(np.arange(NDEV * NDEV * R, dtype=np.float32)[:, None], W, axis=1)
expected = np.concatenate(
    [
        host[s * NDEV * R + t * R : s * NDEV * R + (t + 1) * R]
        for t in range(NDEV)
        for s in range(NDEV)
    ]
)
meta = {
    "sizes": np.full((NDEV, NDEV), R, np.int32),
    "in_off": np.tile(np.arange(NDEV, dtype=np.int32) * R, (NDEV, 1)),
    "out_off": np.repeat(np.arange(NDEV, dtype=np.int32)[:, None] * R, NDEV, axis=1),
    "recv": np.full((NDEV, NDEV), R, np.int32),
}


def body(x, sizes, in_off, out_off, recv):
    out = jnp.full((NDEV * R, W), FILL, x.dtype)
    return jax.lax.ragged_all_to_all(
        x, out, in_off[0], sizes[0], out_off[0], recv[0], axis_name="gpus"
    )


run = jax.jit(
    shard_map(
        body, mesh=mesh, in_specs=(P("gpus"),) * 5, out_specs=P("gpus"), check_vma=False
    )
)
x = jax.device_put(jnp.asarray(host), rows)
m = [
    jax.device_put(jnp.asarray(meta[k]), rows)
    for k in ("sizes", "in_off", "out_off", "recv")
]

print(f"jax {jax.__version__}, devices {jax.devices()[:NDEV]}", flush=True)
print(f"XLA_FLAGS={os.environ['XLA_FLAGS']!r}", flush=True)
print(
    f"rows {R} x width {W} float32 per message ({R * W * 4 / 1e6:.3f} MB)", flush=True
)
bad = 0
for i in range(args.iters):
    box: list = []
    t = threading.Thread(target=lambda: box.append(np.asarray(run(x, *m))), daemon=True)
    t.start()
    t.join(args.timeout)
    if not box:
        print(
            f"call {i}: HUNG (no result after {args.timeout:.0f} s); exiting",
            flush=True,
        )
        os._exit(2)  # a collective that never returns cannot be cancelled
    got = box[0]
    right = int(np.sum(np.all(got == expected, axis=1)))
    fill = int(np.sum(np.all(got == FILL, axis=1)))
    ok = right == got.shape[0]
    bad += not ok
    if not ok or i < 2:
        print(
            f"call {i}: {right}/{got.shape[0]} rows right, {fill} still fill",
            flush=True,
        )
print(f"{args.iters - bad}/{args.iters} correct", flush=True)
sys.exit(1 if bad else 0)

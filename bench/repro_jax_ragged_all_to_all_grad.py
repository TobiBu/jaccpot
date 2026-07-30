"""Upstream-bug reproducer: jit + ragged_all_to_all + grad breaks later forwards.

Pure JAX -- no jaccpot, no yggdrax -- so it attributes the defect unambiguously.
This is the artifact to attach to a jax-ml/jax issue; see
``docs/differentiable_fmm_distributed_audit.md`` for how it was found and what it
cost us (the multi-GPU FMM gradient path has to avoid
``jax.lax.ragged_all_to_all`` because of it).

The bug: take a ``jax.jit``-wrapped ``shard_map`` containing one
``ragged_all_to_all``, evaluate it, execute a gradient of it, then evaluate it
again -- the second evaluation returns the output buffer's *fill value*, i.e. the
exchange no longer writes its output. The damage persists for the process.

Observed on jax/jaxlib 0.9.0.1, 2xA100-PCIE-40GB, CUDA::

    $ python bench/repro_jax_ragged_all_to_all_grad.py --jit
    before=[1. 2. 5. 6. 3. 4. 7. 8.]  after=[ 1.  2.  5.  6. -1. -1. -1. -1.]  -> CORRUPT

    $ python bench/repro_jax_ragged_all_to_all_grad.py          # no --jit
    before=[1. 2. 5. 6. 3. 4. 7. 8.]  after=[1. 2. 5. 6. 3. 4. 7. 8.]  -> CLEAN

*Which* device's rows are lost varies between runs (either half of ``after`` may
be the fill value, and sometimes both); that a jitted run corrupts at all does
not vary.

``jax.jit`` is necessary to trigger it; the bare ``shard_map`` is clean. Not
sensitive to whether the output buffer is a compile-time constant --
``--variant data_dependent`` ties it to a traced input and corrupts too.

**Run ONE case per process.** Each invocation tests a single configuration on
purpose: when several are exercised in one process the later ones perturb the
earlier ones' behaviour (a case that corrupts in a fresh process can report CLEAN
if another jit+grad sequence ran first). In fresh processes the result is
deterministic -- 6/6 CORRUPT for both variants under ``--jit``.

Run on >= 2 GPUs::

    CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o) \
        python bench/repro_jax_ragged_all_to_all_grad.py --jit
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

try:
    from jax import shard_map
except ImportError:  # pragma: no cover - older JAX
    from jax.experimental.shard_map import shard_map

NDEV = 2
CAP = 4
FILL = -1.0


def build(*, jitted: bool, variant: str = "constant"):
    """A shard_map doing one ragged all-to-all, optionally wrapped in jax.jit."""
    mesh = Mesh(np.array(jax.devices()[:NDEV]), ("gpus",))
    send_sizes = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))
    input_offsets = jnp.asarray(np.array([[0, 2], [0, 2]], np.int32))
    output_offsets = jnp.asarray(np.array([[0, 0], [2, 2]], np.int32))
    recv_sizes = jnp.asarray(np.array([[2, 2], [2, 2]], np.int32))

    def body(x, sizes, in_off, out_off, rec):
        out = jnp.full((CAP,), FILL, x.dtype)
        if variant == "data_dependent":
            # same values, but not a compile-time constant
            out = out + jnp.zeros_like(x[:CAP]) * x[:CAP]
        return jax.lax.ragged_all_to_all(
            x, out, in_off[0], sizes[0], out_off[0], rec[0], axis_name="gpus"
        )

    def run(x):
        return shard_map(
            body,
            mesh=mesh,
            in_specs=(P("gpus"),) * 5,
            out_specs=P("gpus"),
            check_vma=False,
        )(x, send_sizes, input_offsets, output_offsets, recv_sizes)

    return jax.jit(run) if jitted else run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jit", action="store_true", help="wrap the shard_map in jax.jit (triggers it)"
    )
    parser.add_argument(
        "--variant",
        choices=("constant", "data_dependent"),
        default="constant",
        help="how the ragged output buffer is built",
    )
    args = parser.parse_args()

    if len(jax.devices()) < NDEV:
        raise SystemExit(f"needs >= {NDEV} devices, found {len(jax.devices())}")

    run = build(jitted=args.jit, variant=args.variant)
    x = jnp.asarray(np.arange(1.0, NDEV * CAP + 1.0))

    before = np.asarray(run(x))
    grad = jax.grad(lambda v: jnp.sum(run(v) ** 2))(x)
    jax.block_until_ready(grad)
    after = np.asarray(run(x))

    corrupt = not np.array_equal(before, after)
    print(f"jit={args.jit}  variant={args.variant}")
    print(
        f"before={before}  after={after}  -> {'CORRUPT' if corrupt else 'CLEAN'}"
    )
    raise SystemExit(1 if corrupt else 0)


if __name__ == "__main__":
    main()

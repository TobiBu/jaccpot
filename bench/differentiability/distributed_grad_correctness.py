"""Distributed (multi-GPU) gradient correctness -- the data behind section 5's
differentiability paragraph.

This is the multi-device counterpart of ``grad_correctness.py``. It measures the
same three things that ``tests/distributed/test_distributed_grad_correctness.py``
asserts, and writes them to JSON so the manuscript can quote a number instead of
a passing test: a test tells you a bound held, not what the value was.

Why a separate script rather than parsing the test: the test's numbers only
appear inside assertion messages, which are not emitted when the assertion
passes. Nothing is currently recorded.

What is measured, per arm
-------------------------
``forward_bit_identity``
    ``differentiable=True`` must not perturb the forward force at all. Reported
    as an exact-equality boolean plus the rel-L2 drift, so a regression says how
    far it moved rather than only that it moved.

``fd_vs_ad``
    Directional central differences against ``jax.grad``, for positions and for
    masses separately -- they exercise different seams, positions through the
    geometry and masses only through the monopole weights.

``oracle``
    ``grad(FMM)`` against ``grad(exact direct sum)``, which is the strongest
    check available because the direct sum has no tree, no acceptance criterion
    and no approximation in it. Reported next to the *forward* force error of the
    same configuration, because that is the quantity the gradient error should be
    read against: a gradient cannot reasonably be tighter than the force it
    differentiates.

``forward_survives_gradient``
    Executing a gradient must not change a later forward. This exists because it
    once did: an upstream ragged-collective bug silently disabled the halo
    exchange for the rest of the process, and the forward then reproduced the
    *local-only* sum to machine precision while looking healthy. The resolved
    halo-exchange implementation is recorded in the config for exactly this
    reason -- the number means something different on ``native`` than on ``buf``.

Scope
-----
Correctness only. This script deliberately reports **no timing**: the reverse
pass costs a single-digit multiple of the forward, but the forward on a shared
host swings ~50% run to run, so a ratio measured here would be noise with a
decimal point. Overhead vs device count belongs in its own run on a quiet box.

Usage
-----
Paper run (claims its whole mesh in one selection)::

    python -m bench.differentiability.distributed_grad_correctness --ndev 2

CPU smoke -- exercises the plumbing only; the distributed path needs >= 2
devices, so this will refuse rather than pretend::

    JAX_PLATFORMS=cpu python -m bench.differentiability.distributed_grad_correctness \\
        --ndev 1 --gpu-select none --json-out /tmp/smoke.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.jaccpot_paper.common import jsonio, runmeta  # noqa: E402

DEFAULT_OUT = "differentiability/distributed_grad_correctness.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    runmeta.add_common_args(p)
    p.add_argument("--ndev", type=int, default=2, help="device count (mesh size)")
    p.add_argument(
        "--per", type=int, default=32, help="particles per device (default: 32)"
    )
    p.add_argument("--order", type=int, default=3)
    p.add_argument("--theta", type=float, default=0.4)
    p.add_argument("--leaf-size", type=int, default=8)
    p.add_argument("--basis", default="real", choices=("real", "solidfmm"))
    p.add_argument("--mac-type", default="dehnen")
    p.add_argument(
        "--nearfield-backend",
        default="baseline",
        choices=("auto", "pallas", "baseline"),
        help=(
            "baseline by default so the reported number is the portable one; "
            "'pallas' routes the near field through the fused kernel and its "
            "analytic reverse"
        ),
    )
    p.add_argument(
        "--halo-exchange",
        default="auto",
        choices=("auto", "buf", "native"),
        help="grad-path halo exchange; 'auto' version-gates the ragged one",
    )
    p.add_argument(
        "--fd-step",
        type=float,
        default=1e-5,
        help=(
            "central-difference step. Must stay inside the fixed-topology "
            "regime: large enough to clear the float64 noise floor, small "
            "enough that no particle crosses a cell or MAC boundary"
        ),
    )
    return p.parse_args()


def _rel_l2(a: Any, b: Any) -> float:
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def _clusters(ndev: int, per: int, seed: int) -> tuple[Any, Any]:
    """One spatially separated cluster per Morton domain.

    Separated deliberately: if the domains overlap, the cross-domain path is
    barely engaged and the measurement mostly reports the local sweep, which is
    already covered single-device.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0],
            [6.0, 6.0, 0.0],
            [6.0, 0.0, 6.0],
            [0.0, 6.0, 6.0],
            [6.0, 6.0, 6.0],
        ],
        dtype=np.float64,
    )
    if ndev > centers.shape[0]:
        raise SystemExit(
            f"--ndev {ndev} exceeds the {centers.shape[0]} cluster centres"
        )
    positions = np.concatenate(
        [centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    )
    return positions, rng.uniform(0.5, 2.0, size=(per * ndev,))


def main() -> int:
    """Measure distributed gradient correctness and write the artifact.

    Returns
    -------
    int
        Process exit status; 0 on success.
    """

    args = _parse_args()
    if int(args.ndev) < 2:
        raise SystemExit(
            f"--ndev {args.ndev}: distributed gradients need >= 2 devices. This "
            "script will not report a single-device number under a distributed "
            "label -- run bench/differentiability/grad_correctness.py for that."
        )
    # Both must precede the first jax import, and select_gpu claims the whole
    # mesh at once rather than one card at a time.
    runmeta.select_gpu(args.gpu_select, num_gpus=int(args.ndev))
    runmeta.enable_x64(args.dtype)

    import dataclasses

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402
    from yggdrax.distributed import device_count, make_mesh  # noqa: E402

    from jaccpot.distributed import DistributedFMMConfig  # noqa: E402
    from jaccpot.distributed.fmm import (  # noqa: E402
        make_force_evaluator,
        partition_for_devices,
        resolve_grad_halo_exchange,
    )

    ndev = int(args.ndev)
    if device_count() < ndev:
        raise SystemExit(
            f"need {ndev} devices, JAX sees {device_count()}. Not falling back: a "
            "distributed measurement on a different device count is a different "
            "measurement."
        )

    config = dataclasses.replace(
        DistributedFMMConfig(),
        order=int(args.order),
        theta=float(args.theta),
        leaf_size=int(args.leaf_size),
        basis=args.basis,
        mac_type=args.mac_type,
        nearfield_backend=args.nearfield_backend,
        local_walk="dual_tree",
    )

    positions, masses = _clusters(ndev, int(args.per), int(args.seed))
    part = partition_for_devices(positions, masses, ndev, leaf_size=config.leaf_size)
    mesh = make_mesh(ndev)
    pos = jnp.asarray(part["pos_flat"])
    mass = jnp.asarray(part["mass_flat"])
    gid = jnp.asarray(part["gid_flat"])
    counts = jnp.asarray(part["counts"])
    args_t = (pos, mass, gid, counts)

    def build(differentiable: bool) -> Any:
        return make_force_evaluator(
            config,
            ndev,
            part["cap"],
            mesh,
            jit=True,
            differentiable=differentiable,
            halo_exchange=args.halo_exchange,
        )

    forward = build(False)
    grad_path = build(True)

    def loss(p: Any, m: Any) -> Any:
        return jnp.sum(grad_path(p, m, gid, counts)[0] ** 2)

    records: list[dict[str, Any]] = []

    # 1. differentiable=True must not move the forward.
    shipped = np.asarray(forward(*args_t)[0])
    _fwd, _fwd_gid, _ = grad_path(*args_t)
    grad_fwd = np.asarray(_fwd)
    # Kept for the oracle below: the accelerations are in the per-device TREE order,
    # which the INPUT gid does not name once a device is padded. See
    # docs/distributed_padding_force_defect.md.
    fwd_gid = np.asarray(_fwd_gid).reshape(-1)
    records.append(
        {
            "check": "forward_bit_identity",
            "bit_identical": bool(np.array_equal(shipped, grad_fwd)),
            "rel_l2_drift": _rel_l2(grad_fwd, shipped),
        }
    )

    # 2. Directional finite differences vs jax.grad, per differentiation target.
    step = float(args.fd_step)
    for wrt, argnum in (("positions", 0), ("masses", 1)):
        base = pos if argnum == 0 else mass
        grad = np.asarray(jax.grad(loss, argnums=argnum)(pos, mass))
        direction = np.asarray(
            jax.random.normal(
                jax.random.PRNGKey(int(args.seed)), base.shape, base.dtype
            )
        )
        analytic = float(np.sum(grad * direction))
        d = jnp.asarray(direction)
        if argnum == 0:
            plus, minus = float(loss(pos + step * d, mass)), float(
                loss(pos - step * d, mass)
            )
        else:
            plus, minus = float(loss(pos, mass + step * d)), float(
                loss(pos, mass - step * d)
            )
        numeric = (plus - minus) / (2.0 * step)
        records.append(
            {
                "check": "fd_vs_ad",
                "wrt": wrt,
                "fd": numeric,
                "ad": analytic,
                "rel_err": abs(numeric - analytic) / (abs(numeric) + 1e-300),
                "fd_step": step,
                "grad_norm": float(np.linalg.norm(grad)),
                "all_finite": bool(np.all(np.isfinite(grad))),
            }
        )

    # 3. grad(FMM) vs grad(direct sum), reported against the forward force error.
    gid_np = np.asarray(gid).reshape(-1)
    valid = gid_np >= 0
    order_map = gid_np[valid].astype(int)
    g_pos, g_mass = jax.grad(loss, argnums=(0, 1))(pos, mass)
    positions0, masses0 = jnp.asarray(positions), jnp.asarray(masses)

    def direct_accel(p: Any, m: Any) -> Any:
        delta = p[:, None, :] - p[None, :, :]
        dist_sq = jnp.sum(delta * delta, -1) + config.softening**2
        eye = jnp.eye(p.shape[0], dtype=p.dtype)
        return -config.G * jnp.einsum(
            "ij,ijk->ik", m[None, :] * dist_sq**-1.5 * (1.0 - eye), delta
        )

    dg_pos, dg_mass = jax.grad(
        lambda p, m: jnp.sum(direct_accel(p, m) ** 2), argnums=(0, 1)
    )(positions0, masses0)

    # The force comes back in the tree order -> the RETURNED gid maps it.
    fwd_valid = fwd_gid >= 0
    scattered = np.zeros((positions0.shape[0], 3))
    scattered[fwd_gid[fwd_valid].astype(int)] = grad_fwd[fwd_valid]
    force_err = _rel_l2(scattered, np.asarray(direct_accel(positions0, masses0)))

    # The cotangents were taken w.r.t. the padded input arrays, so they stay in the
    # input layout and keep the input map. Two arrays, two orders.
    fmm_gp = np.zeros_like(np.asarray(dg_pos))
    fmm_gm = np.zeros_like(np.asarray(dg_mass))
    fmm_gp[order_map] = np.asarray(g_pos)[valid]
    fmm_gm[order_map] = np.asarray(g_mass)[valid]
    records.append(
        {
            "check": "oracle",
            "grad_rel_l2_positions": _rel_l2(fmm_gp, np.asarray(dg_pos)),
            "grad_rel_l2_masses": _rel_l2(fmm_gm, np.asarray(dg_mass)),
            # The reference the gradient errors should be read against.
            "forward_force_rel_l2": force_err,
        }
    )

    # 4. A gradient must not poison a later forward.
    after_fwd = np.asarray(forward(*args_t)[0])
    after_grad = np.asarray(grad_path(*args_t)[0])
    records.append(
        {
            "check": "forward_survives_gradient",
            "forward_bit_identical": bool(np.array_equal(shipped, after_fwd)),
            "grad_path_bit_identical": bool(np.array_equal(grad_fwd, after_grad)),
            "forward_rel_l2_drift": _rel_l2(after_fwd, shipped),
            "grad_path_rel_l2_drift": _rel_l2(after_grad, grad_fwd),
        }
    )

    out = jsonio.write_result(
        args.json_out or DEFAULT_OUT,
        config={
            "n": int(part["n"]),
            "theta": float(config.theta),
            "order": int(config.order),
            "basis": config.basis,
            "seed": int(args.seed),
            "device": runmeta.device_label(),
            "precision": args.dtype,
            # Distributed-specific axes. Without ndev and the resolved halo
            # exchange the row does not identify the run: the same numbers mean
            # different things on 2 devices vs 4, and on native vs buf.
            "ndev": ndev,
            "per_device_n": int(args.per),
            "leaf_size": int(config.leaf_size),
            "mac_type": config.mac_type,
            "nearfield_backend": config.nearfield_backend,
            "halo_exchange_requested": args.halo_exchange,
            "halo_exchange_resolved": resolve_grad_halo_exchange(args.halo_exchange),
            "fd_step": step,
        },
        meta=runmeta.run_meta({"argv": sys.argv[1:]}),
        data={"records": records},
    )
    print(f"wrote {out}")
    for r in records:
        print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

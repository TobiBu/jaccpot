"""Scenarios for the A4 reference force's multi-process gate (``_multiprocess_harness``).

``reference(n)``: the same particles laid out in an ``n``-dependent row order on the
``n``-device global mesh (one process with ``n`` devices, or ``n`` processes with
one), and the reference force at a fixed target set; plus the multi-GPU runner
(``DirectMeshForce``, migrating) scored by :func:`score_mesh_result` without a
gather. Every array the caller compares is replicated, so process 0 saves it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.distributed import multiprocess as mp

AXIS = "gpus"
N = 512
K = 64


def particles() -> tuple:
    """``[N, 3]`` positions, ``[N]`` masses (a clumpy Gaussian, unequal masses)."""
    rng = np.random.default_rng(11)
    x = rng.normal(size=(N, 3))
    x[:32] *= 0.02
    return x, rng.uniform(0.5, 1.5, N) / N


def layout(ndev: int) -> tuple:
    """Rows by device: an ``ndev``-dependent shuffle, ``cap`` rows each, padding last."""
    x, m = particles()
    rng = np.random.default_rng(100 + ndev)
    perm = rng.permutation(N)
    cap = -(-N // ndev) + 7
    xs = np.zeros((ndev * cap, 3))
    ms = np.zeros(ndev * cap)
    ids = -np.ones(ndev * cap, np.int32)
    for d, chunk in enumerate(np.array_split(perm, ndev)):
        lo = d * cap
        xs[lo : lo + len(chunk)] = x[chunk]
        ms[lo : lo + len(chunk)] = m[chunk]
        ids[lo : lo + len(chunk)] = chunk
        xs[lo + len(chunk) : lo + cap] = x[chunk[0]]
    return xs, ms, ids


def _global(mesh, host: np.ndarray):
    """``host`` as a ``P(AXIS)`` global array: each process supplies its shards."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    sharding = NamedSharding(mesh, P(AXIS))
    return jax.make_array_from_callback(host.shape, sharding, lambda idx: host[idx])


def reference(n: int) -> dict:
    """The reference on the ``n``-device mesh, and the runner scored on it."""
    from jaccpot import FastMultipoleMethod
    from jaccpot.distributed.segment import DirectMeshForce
    from jaccpot.production import MeshRunConfig, ProductionRunner, SegmentConfig
    from jaccpot.verification import (
        ForceLaw,
        reference_force,
        score_mesh_result,
        select_targets,
    )

    mesh = mp.global_mesh(AXIS)
    xs, ms, ids = layout(n)
    tids = select_targets(N, K, seed=2)
    out: dict = {}
    for law in (ForceLaw(1.0, 0.03, "ferrers3"), ForceLaw(1.0, 0.03, "plummer")):
        ref = reference_force(
            _global(mesh, xs),
            _global(mesh, ms),
            _global(mesh, ids),
            tids,
            law,
            mesh=mesh,
        )
        out[f"{law.kernel}_acc"] = ref.acceleration
        out[f"{law.kernel}_pot"] = ref.potential
        out[f"{law.kernel}_pos"] = ref.positions
        out[f"{law.kernel}_sources"] = ref.num_sources

    # the runner's multi-GPU mode, scored at its final state without a gather
    g, soft = 1e-3, 0.05
    x, m = particles()
    vel = 0.3 * np.random.default_rng(5).normal(size=(N, 3))
    state = np.stack([x, vel], axis=1)
    runner = ProductionRunner(
        FastMultipoleMethod(preset="large_n_gpu", G=g, softening=soft),
        segment=SegmentConfig(steps=2),
        mesh=MeshRunConfig(
            mesh,
            cap=int(np.ceil(1.6 * N / n)),
            repartition_every=1,
            num_samples=64,
            migrate=True,
            force=DirectMeshForce(G=g, softening=soft, axis_name=AXIS),
        ),
    )
    if mp.is_multiprocess():
        lo, hi = mp.local_id_range(N, mesh)
        res = runner.run(
            mp.ProcessLocalParticles(state[lo:hi], m[lo:hi], N),
            None,
            dt=0.02,
            num_steps=4,
        )
    else:
        res = runner.run(jnp.asarray(state), jnp.asarray(m), dt=0.02, num_steps=4)
    score = score_mesh_result(res, runner.solver, target_ids=tids)
    out["runner_score"] = {k: v for k, v in score.items() if k != "law"}
    return out

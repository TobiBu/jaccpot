"""Regression test for the reusable distributed FMM driver.

Proves that ``jaccpot.distributed.distributed_fmm_accelerations`` reproduces the
result of the validated in-test assembly in
``test_distributed_solidfmm_far_shardmap.py``: on the same spatially-separated
cluster IC (one cluster per Morton domain, so the cross-domain far-field is
genuinely engaged), the reassembled per-particle accelerations match a direct
N-body sum to within 1%.

    CUDA_VISIBLE_DEVICES=$(autocvd -n 3 -l -o) \
        pytest tests/test_distributed_fmm_driver.py -o addopts="" -q
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.distributed import device_count, make_mesh

from jaccpot.distributed import (
    DistributedFMMConfig,
    distributed_fmm_accelerations,
)

pytestmark = pytest.mark.skipif(
    device_count() < 2, reason="distributed FMM needs >= 2 devices"
)


def _direct(all_pos, all_mass, G, soft):
    diff = all_pos[:, None, :] - all_pos[None, :, :]
    d2 = (diff**2).sum(-1) + soft**2
    inv = d2 ** (-1.5)
    return -G * (all_mass[None, :, None] * diff * inv[..., None]).sum(axis=1)


def _separated_clusters(ndev, per, seed=4):
    """ndev spatially separated clusters (one per Morton domain)."""
    rng = np.random.default_rng(seed)
    cluster_centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    pts = np.concatenate(
        [cluster_centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return pts, mass


def test_driver_matches_direct():
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    pts, mass = _separated_clusters(ndev, per)

    # Defaults are the converged fast-lane config (real basis + dehnen MAC,
    # order=3, theta=0.4, theta_cross=0.1, leaf=8, soft=0.02).
    config = DistributedFMMConfig()

    result = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=False
    )

    direct = np.asarray(
        _direct(jnp.asarray(pts), jnp.asarray(mass), config.G, config.softening)
    )
    err = float(
        np.linalg.norm(result.accelerations - direct) / (np.linalg.norm(direct) + 1e-30)
    )
    print(f"driver FULL aggL2 vs direct = {err:.6f}  (ndev={ndev})")
    print("per-device diagnostics:")
    for k, v in result.diagnostics.items():
        print(f"  {k}: {np.asarray(v)}")

    assert not result.overflow, "traversal buffers overflowed -- grow the caps"
    assert err < 1e-2, f"driver aggL2 err {err:.6f} exceeds 1%"


def test_driver_real_basis_matches_direct():
    """Real (Dehnen) far-field basis matches direct N-body as well as solidfmm.

    Isolates the basis change: same IC and same bh MAC as the solidfmm baseline,
    only ``basis="real"``. Confirms the per-device far field converged onto the
    single-GPU fast-lane real path is correct.
    """
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    pts, mass = _separated_clusters(ndev, per)
    config = DistributedFMMConfig(basis="real", mac_type="bh")

    result = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=False
    )
    direct = np.asarray(
        _direct(jnp.asarray(pts), jnp.asarray(mass), config.G, config.softening)
    )
    err = float(
        np.linalg.norm(result.accelerations - direct) / (np.linalg.norm(direct) + 1e-30)
    )
    print(f"driver REAL-basis FULL aggL2 vs direct = {err:.6f}  (ndev={ndev})")
    assert not result.overflow, "traversal buffers overflowed -- grow the caps"
    assert err < 1e-2, f"real-basis aggL2 err {err:.6f} exceeds 1%"


def test_driver_solidfmm_matches_direct():
    """Legacy solidfmm(complex)/bh path still matches direct after the real default.

    Keeps the pre-convergence force path covered regardless of the (now real+dehnen)
    default, so a future change to the real path can't silently break solidfmm.
    """
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    pts, mass = _separated_clusters(ndev, per)
    config = DistributedFMMConfig(basis="solidfmm", mac_type="bh")

    result = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=False
    )
    direct = np.asarray(
        _direct(jnp.asarray(pts), jnp.asarray(mass), config.G, config.softening)
    )
    err = float(
        np.linalg.norm(result.accelerations - direct) / (np.linalg.norm(direct) + 1e-30)
    )
    print(f"driver SOLIDFMM FULL aggL2 vs direct = {err:.6f}  (ndev={ndev})")
    assert not result.overflow, "traversal buffers overflowed -- grow the caps"
    assert err < 1e-2, f"solidfmm aggL2 err {err:.6f} exceeds 1%"


def test_driver_jit_matches_eager():
    """The jitted (steady-state timing) path must match the eager path."""
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    pts, mass = _separated_clusters(ndev, per)
    config = DistributedFMMConfig()

    eager = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=False
    )
    jitted = distributed_fmm_accelerations(
        pts, mass, config=config, mesh=mesh, jit=True
    )

    diff = float(
        np.linalg.norm(eager.accelerations - jitted.accelerations)
        / (np.linalg.norm(eager.accelerations) + 1e-30)
    )
    print(f"jit-vs-eager aggL2 = {diff:.3e}")
    assert diff < 1e-6, f"jit path diverged from eager: {diff:.3e}"


def test_driver_auto_scale_caps():
    """``auto_scale_caps`` grows the traversal buffers on overflow until the
    walk fits, without changing the forces.

    The cross-domain LET grows with device count, so the fixed default caps can
    overflow at higher ``ndev``. Deliberately tiny caps force the overflow here;
    the retry must clear it and still match direct N-body.
    """
    ndev = min(4, device_count())
    mesh = make_mesh(ndev)
    per = 64
    pts, mass = _separated_clusters(ndev, per)
    tiny = dataclasses.replace(
        DistributedFMMConfig(),
        nearfield_backend="baseline",  # pallas is GPU-only; CI runs on CPU
        max_pair_queue=64,
        cross_max_pair_queue=64,
        max_interactions_per_node=16,
        cross_max_interactions_per_node=16,
        max_neighbors_per_leaf=16,
        cross_max_neighbors_per_leaf=16,
    )

    r0 = distributed_fmm_accelerations(pts, mass, config=tiny, mesh=mesh, jit=False)
    assert r0.overflow, "tiny caps should overflow without auto_scale_caps"

    r1 = distributed_fmm_accelerations(
        pts,
        mass,
        config=tiny,
        mesh=mesh,
        jit=False,
        auto_scale_caps=True,
        cap_scale_factor=2.0,
        max_cap_retries=6,
    )
    assert not r1.overflow, "auto_scale_caps should clear the overflow"
    assert r1.diagnostics["cap_retries"] >= 1

    direct = np.asarray(
        _direct(jnp.asarray(pts), jnp.asarray(mass), tiny.G, tiny.softening)
    )
    err = float(
        np.linalg.norm(r1.accelerations - direct) / (np.linalg.norm(direct) + 1e-30)
    )
    print(
        f"auto_scale_caps aggL2 vs direct = {err:.6f}  retries={r1.diagnostics['cap_retries']}"
    )
    assert err < 5e-2, f"forces wrong after cap retry: {err:.6f}"

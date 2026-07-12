"""Physical-accuracy gate for the O(N) octree FMM: forces vs direct N-body.

Runs the standalone uniform-octree U/V FMM (yggdrax octree + jaccpot FMM operators +
octree-node-space local evaluation + U-list near P2P) and asserts it matches direct
softened N-body to expansion-order truncation error. Parity is defined vs DIRECT, not
vs any other tree code.

This gate also guards the octree FMM kernel fixes (deep-tree level batching +
collision-free P2M/M2M scatters in runtime/_octree_fmm): it exercises trees deeper than
2 levels (L=3, L=4) on the natural node layout (root at index 0, no reserved sentinel),
and checks order-4 -> order-6 convergence.

CPU / float64 host-driven reference (small N); the on-device large-N path is a later
step.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("yggdrax")

from jaccpot.experimental.octree_fmm_uvwx import octree_fmm_accelerations


def _points(n, seed, dist):
    rng = np.random.default_rng(seed)
    if dist == "clustered":
        r = rng.uniform(size=n) ** (1.0 / 3.0)
        r = r / (1.0 - 0.85 * r)
        d = rng.standard_normal((n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pos = r[:, None] * d
    else:
        pos = rng.uniform(-1.0, 1.0, size=(n, 3))
    mass = np.abs(rng.standard_normal(n)) + 0.5
    return pos, mass


def _direct(pos, mass, G, soft):
    d = pos[:, None, :] - pos[None, :, :]
    d2 = np.sum(d * d, axis=-1) + soft**2
    np.fill_diagonal(d2, np.inf)
    return -G * np.sum(mass[None, :, None] * d * (d2**-1.5)[..., None], axis=1)


def _rel_errors(n, L, dist, order, seed=7, G=1.0, soft=1e-2):
    pos, mass = _points(n, seed, dist)
    acc = np.asarray(
        octree_fmm_accelerations(pos, mass, depth=L, order=order, G=G, softening=soft)
    )
    direct = _direct(pos, mass, G, soft)
    return np.linalg.norm(acc - direct, axis=1) / (
        np.linalg.norm(direct, axis=1) + 1e-12
    )


@pytest.mark.parametrize(
    ("n", "L", "dist"),
    [
        (2000, 3, "uniform"),
        (4000, 3, "clustered"),
        (6000, 4, "clustered"),
    ],
)
def test_octree_fmm_matches_direct(n, L, dist):
    rel = _rel_errors(n, L, dist, order=4)
    assert np.median(rel) < 2e-2, f"median rel err {np.median(rel):.3e}"
    assert np.percentile(rel, 90) < 5e-2, f"p90 rel err {np.percentile(rel, 90):.3e}"


def test_octree_fmm_converges_with_order():
    rel4 = _rel_errors(2000, 3, "uniform", order=4)
    rel6 = _rel_errors(2000, 3, "uniform", order=6)
    assert np.median(rel6) < np.median(rel4), (
        f"order 6 ({np.median(rel6):.3e}) not better than order 4 "
        f"({np.median(rel4):.3e})"
    )
    assert np.median(rel6) < 1e-3


def test_device_matches_host_build():
    """The on-device static build gives the same forces as the host-numpy reference."""
    pos, mass = _points(3000, 7, "clustered")
    acc_dev = np.asarray(
        octree_fmm_accelerations(pos, mass, depth=3, order=4, device=True)
    )
    acc_host = np.asarray(
        octree_fmm_accelerations(pos, mass, depth=3, order=4, device=False)
    )
    rel = np.linalg.norm(acc_dev - acc_host, axis=1) / (
        np.linalg.norm(acc_host, axis=1) + 1e-12
    )
    assert np.max(rel) < 1e-10, f"device vs host max rel diff {np.max(rel):.2e}"


def test_sparse_device_matches_direct():
    """The SPARSE-occupied static build (feasible on clustered data at the depth needed
    to bound occupancy) matches direct N-body through the fused-jit device path."""
    pos, mass = _points(4000, 7, "clustered")
    acc = np.asarray(
        octree_fmm_accelerations(
            pos,
            mass,
            depth=4,
            order=4,
            device=True,
            sparse=True,
            node_capacity=2000,
            leaf_capacity=1500,
            max_leaf_capacity=512,
        )
    )
    direct = _direct(pos, mass, 1.0, 1e-2)
    rel = np.linalg.norm(acc - direct, axis=1) / (
        np.linalg.norm(direct, axis=1) + 1e-12
    )
    assert np.median(rel) < 2e-2, f"median rel err {np.median(rel):.3e}"
    assert np.percentile(rel, 90) < 5e-2, f"p90 rel err {np.percentile(rel, 90):.3e}"


def test_device_build_does_not_recompile_across_positions():
    """Fixed (N, depth, order, max_leaf_capacity): changing only positions must NOT
    trigger recompilation -- the whole point of the static-shape device build (reuse
    across time-integration steps)."""
    p1, m1 = _points(2000, 1, "uniform")
    p2, m2 = _points(2000, 2, "clustered")  # same N, different positions
    kw = dict(depth=3, order=4, max_leaf_capacity=64, device=True)
    octree_fmm_accelerations(p1, m1, **kw).block_until_ready()  # warm / compile once
    with jax.log_compiles(True):
        with caplog_records() as records:
            octree_fmm_accelerations(p2, m2, **kw).block_until_ready()
    compiles = [m for m in records if "compil" in m.lower()]
    assert not compiles, f"unexpected recompilation:\n" + "\n".join(compiles[:5])


class caplog_records:
    """Capture jax logging records (compilation messages) for the duration of a block."""

    def __enter__(self):
        import logging

        self._records: list[str] = []
        self._handler = logging.Handler()
        self._handler.emit = lambda r: self._records.append(r.getMessage())
        self._logger = logging.getLogger("jax")
        self._prev_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self._records

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._prev_level)
        return False

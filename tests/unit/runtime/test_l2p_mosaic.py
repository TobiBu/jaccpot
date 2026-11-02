"""The Mosaic GPU port of the particle-major L2P forward (spike S1b) equals Triton's.

``jaccpot.pallas.l2p_real_mosaic`` traces the same Cartesian recurrence as the Triton
kernel (:mod:`jaccpot.pallas.l2p_real`); only the gather differs. Its ``"select"``
gather is pure Pallas and runs in Mosaic GPU's CPU interpret mode, so it is tested
here the way the Triton kernels are: against the Triton kernel's interpret mode, bit
for bit (both select or gather the stored values, then run the same arithmetic), on
leaves of 1-8 particles, empty padding leaves, dead rows past the leaves' particles,
fewer rows than one warpgroup, and through the ``JACCPOT_L2P_KERNEL`` dispatch and
its custom_vjp (whose reverse is Triton's). The ``"inline"`` gather has no interpret
mode (``inline_mgpu``); it is checked on a GPU by ``bench/s1b_mosaic_l2p.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime.kernels._evaluate import _evaluate_local_expansions_particle_major
from tests.unit.runtime.test_l2p_particle_major import _case

pytest.importorskip("jax.experimental.pallas.mosaic_gpu")

_ORDER = 4


def _leaf_args(seed: int, n_live: int, n_rows: int, pad: int) -> tuple:
    local, positions, leaf_nodes, ranges = _case(seed, n_live, n_rows, pad)
    lr = ranges[leaf_nodes]
    counts = jnp.maximum(lr[:, 1] - lr[:, 0] + 1, 0)
    return local, positions, leaf_nodes, ranges, lr[:, 0], counts


@pytest.mark.parametrize(
    "n_live, n_rows, pad",
    # one block; dead rows + padding leaves; fewer rows than a warpgroup; an
    # overlapping last block
    [(128, 128, 0), (187, 200, 2), (97, 97, 1), (300, 333, 4)],
)
def test_mosaic_select_equals_triton_bitwise(n_live, n_rows, pad):
    from jaccpot.pallas.l2p_real import l2p_real_particles_pallas
    from jaccpot.pallas.l2p_real_mosaic import l2p_real_particles_mosaic

    local, positions, leaf_nodes, _, start, counts = _leaf_args(7, n_live, n_rows, pad)
    args = (local.coefficients, local.centers, positions, leaf_nodes, start, counts)
    want = np.asarray(l2p_real_particles_pallas(*args, order=_ORDER, interpret=True))
    got = np.asarray(
        l2p_real_particles_mosaic(*args, order=_ORDER, gather="select", interpret=True)
    )
    assert got.shape == (n_rows, 3)
    assert np.array_equal(got, want)
    assert not np.any(got[n_live:])
    assert np.any(got[:n_live])


def test_mosaic_dispatch_and_reverse_equal_triton():
    """``kernel="mosaic_interpret"`` through the particle-major entry point, forward
    and ``jax.grad`` (the custom_vjp's reverse is the Triton one), equals
    ``kernel="interpret"`` bit for bit."""
    local, positions, leaf_nodes, ranges, _, _ = _leaf_args(3, 187, 200, 2)
    kw = dict(leaf_nodes=leaf_nodes, node_ranges=ranges, order=_ORDER, chunk=64)
    w = jnp.asarray(np.random.default_rng(0).normal(size=(200, 3)), jnp.float32)

    def run(kernel: str) -> tuple:
        def loss(coef, cent, x):
            data = type(local)(order=_ORDER, centers=cent, coefficients=coef)
            g = _evaluate_local_expansions_particle_major(data, x, kernel=kernel, **kw)
            return jnp.sum(w * g), g

        grads, g = jax.grad(loss, argnums=(0, 1, 2), has_aux=True)(
            local.coefficients, local.centers, positions
        )
        return np.asarray(g), [np.asarray(a) for a in grads]

    g_want, grads_want = run("interpret")
    g_got, grads_got = run("mosaic_interpret")
    assert np.array_equal(g_got, g_want)
    for a, b in zip(grads_got, grads_want):
        assert np.array_equal(a, b)


def test_mosaic_inline_gather_refuses_interpret():
    from jaccpot.pallas.l2p_real_mosaic import l2p_real_particles_mosaic

    local, positions, leaf_nodes, _, start, counts = _leaf_args(1, 97, 97, 0)
    with pytest.raises(ValueError, match="inline"):
        l2p_real_particles_mosaic(
            local.coefficients,
            local.centers,
            positions,
            leaf_nodes,
            start,
            counts,
            order=_ORDER,
            gather="inline",
            interpret=True,
        )

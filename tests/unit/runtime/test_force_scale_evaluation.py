"""The fused lane's evaluation returns eq (16b)'s force scale alongside the force.

``evaluate_large_n_state(..., return_force_scale=True)`` returns ``(acc, f_b)``.
Here the prepared state carries no far half (``force_scale_far_sorted`` is None
until the prepare stores one), so ``f_b`` is the near half: the near-field kernel's
force-scale lane, ``sum G m / (r^2 + eps^2)`` over the state's own near pairs (the
own leaf and the neighbour leaves, self excluded), in input order. Pinned against
a numpy sum over the prepared state's neighbour list, for the Plummer and a
compact kernel, with the accelerations unchanged by the flag. Runs the CSR lane
in Pallas interpret mode on CPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state
from tests.characterization.test_fmm_golden import _make_inputs
from tests.characterization.test_lane_goldens import LEAF, ORDER, SOFTENING

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the numpy reference runs in float64"
)

N = 512


_FLAGS = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    # the prepacked payload the CSR lane reads, as at large N (a small N picks the
    # materialised one, which the CSR lane cannot read)
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _solver(kernel):
    from jaccpot import FastMultipoleMethod

    return FastMultipoleMethod(
        preset="large_n_gpu",
        G=1.0,
        softening=SOFTENING,
        use_pallas=True,
        softening_kernel=kernel,
    )


def _reference(prepared, softening):
    tree = prepared.tree
    ps = np.asarray(prepared.positions_sorted, np.float64)
    ms = np.asarray(prepared.masses_sorted, np.float64)
    ranges = np.asarray(tree.node_ranges)
    nl = prepared.neighbor_list
    leaves = np.asarray(nl.leaf_indices)
    offsets, counts = np.asarray(nl.offsets), np.asarray(nl.counts)
    nbrs = np.asarray(nl.neighbors)
    fb = np.zeros(ps.shape[0])
    for row, leaf in enumerate(leaves):
        lo, hi = ranges[leaf]
        if hi < lo:
            continue
        sources = [leaf] + [
            int(x) for x in nbrs[offsets[row] : offsets[row] + counts[row]]
        ]
        src = np.concatenate(
            [np.arange(ranges[s, 0], ranges[s, 1] + 1) for s in sources]
        )
        for t in range(lo, hi + 1):
            other = src[src != t]
            d2 = np.sum((ps[other] - ps[t]) ** 2, axis=1)
            fb[t] = np.sum(ms[other] / (d2 + softening**2))
    perm = np.asarray(tree.particle_indices)
    out = np.zeros_like(fb)
    out[perm] = fb
    return out


@pytest.mark.parametrize("kernel", ["plummer", "ferrers3"])
def test_evaluation_returns_the_near_force_scale(monkeypatch, kernel):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in _FLAGS.items():
        monkeypatch.setenv(key, value)
    positions, masses = _make_inputs("clustered", N)
    solver = _solver(kernel)
    prepared = solver.prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=LEAF,
        max_order=ORDER,
    )
    assert prepared.neighbor_list is not None
    assert int(prepared.radix_fast_payload.source_particle_ids.size) == 0  # prepacked
    assert getattr(prepared, "force_scale_far_sorted", None) is None
    common = dict(
        target_indices=None, return_potential=False, max_acc_derivative_order=0
    )
    acc_fb, fb = evaluate_large_n_state(
        solver._impl, prepared, return_force_scale=True, **common
    )
    acc = evaluate_large_n_state(solver._impl, prepared, **common)
    np.testing.assert_allclose(np.asarray(acc_fb), np.asarray(acc), rtol=0, atol=0)
    want = _reference(prepared, SOFTENING)
    assert (want > 0).mean() > 0.9, "vacuous: most particles must have near pairs"
    np.testing.assert_allclose(np.asarray(fb), want, rtol=2e-5, atol=0.0)

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


_STRICT_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
}


def _dehnen_solver(kernel, eps):
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=0.8,
        G=1.0,
        softening=SOFTENING,
        softening_kernel=kernel,
        working_dtype=jnp.float32,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
        adaptive_eps=eps,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
    )


def _sorted_scales(prepared, softening):
    """The true eq (16b) sums per sorted particle: (all pairs, near pairs only)."""
    ps = np.asarray(prepared.positions_sorted, np.float64)
    ms = np.asarray(prepared.masses_sorted, np.float64)
    d2 = np.sum((ps[:, None, :] - ps[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    total = np.sum(ms[None, :] / (d2 + softening**2), axis=1)
    near = _reference(prepared, softening)[np.asarray(prepared.tree.particle_indices)]
    return total, near


def test_dehnen_error_prepare_stores_the_far_force_scale(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in {**_STRICT_FUSED_ENV, **_FLAGS}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", str(N))
    positions, masses = _make_inputs("clustered", N)
    solver = _dehnen_solver("ferrers3", 1e-3)
    prepared = solver._impl.prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=LEAF,
        max_order=ORDER,
        fused_device_mode=True,
    )
    far = prepared.force_scale_far_sorted
    assert far is not None, "the criterion prepare must store eq (16b)'s far half"
    far = np.asarray(far, np.float64)
    total, near = _sorted_scales(prepared, SOFTENING)
    true_far = total - near
    live = true_far > 0
    assert live.mean() > 0.9, "vacuous: no far field"
    # monopoles at the source COM, the sink pushed to its far edge: a LOWER bound
    # on the true far sum (an over-large scale would loosen the criterion), and a
    # usefully tight one
    ratio = far[live] / true_far[live]
    assert ratio.max() <= 1.0 + 1e-4, float(ratio.max())
    assert np.median(ratio) > 0.5, float(np.median(ratio))
    # the evaluation's f_b is the near half plus this far half, so it lower-bounds
    # the full sum the same way
    common = dict(
        target_indices=None, return_potential=False, max_acc_derivative_order=0
    )
    _, fb = evaluate_large_n_state(
        solver._impl, prepared, return_force_scale=True, sorted_output=True, **common
    )
    np.testing.assert_allclose(np.asarray(fb, np.float64), near + far, rtol=1e-4)
    assert np.all(np.asarray(fb, np.float64) <= total * (1.0 + 1e-4))


def test_the_geometric_prepare_stores_no_force_scale(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key, value in _FLAGS.items():
        monkeypatch.setenv(key, value)
    positions, masses = _make_inputs("clustered", N)
    prepared = _solver("ferrers3").prepare_state(
        jnp.asarray(positions, jnp.float32),
        jnp.asarray(masses, jnp.float32),
        leaf_size=LEAF,
        max_order=ORDER,
    )
    assert prepared.force_scale_far_sorted is None

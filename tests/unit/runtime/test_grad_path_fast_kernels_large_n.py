"""The production (large-N) gradient path on the fast Pallas kernels, end to end, on CPU.

Plan ``fast-gradients-for-the-sub10ms-fmm``. The sub-10 ms forward at N = 2x10^5
runs five Pallas kernels: the leaf P2M, the per-level M2M and L2L cascades, the
pair-per-lane M2L over the far-pair CSR and the CSR row-chunk near field. Each now
carries a ``custom_vjp`` whose reverse is a Pallas kernel of its own. This test
opens every one of them on CPU through its interpret flag, differentiates the
production ``large_n_gpu`` path (the same gates as
``test_large_n_grad_reverse_path.py``: backend gate patched, prepacked near-field
layout forced, far pairs retained) and checks the gradient against the direct-sum
oracle -- and COUNTS the five reverse kernels, because a path that quietly fell back
to a pure-JAX lane would pass the accuracy check while measuring nothing.

Found while writing it (Phase 1 of the plan, GPU): the M2L CSR lanes kernel had no
gradient guard at all, so on an Ampere card ``jax.grad`` still died in
``program_id`` after the Phase 0 fix -- the CPU tests never reached it because the
CSR M2L does not lower there. ``JACCPOT_M2L_CSR_INTERPRET=1`` is what reaches it.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import FastMultipoleMethod
from jaccpot.autodiff import direct_sum_gravitational_acceleration
from jaccpot.pallas import (
    cascade_real_level,
    m2l_real_csr_lanes,
    nearfield_leafpair_csr,
    p2m_real_leaf,
)
from jaccpot.runtime._large_n_types import LargeNPreparedState

_N = 2048
_LEAF = 16
_ORDER = 4
_G = 1.0
_SOFTENING = 1e-2
_RTOL = 3e-3  # test_large_n_grad_reverse_path's: 5x the worst measured fp32 error there

_FLAGS = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",  # the prepacked layout, as at N=200k
}


@pytest.fixture(scope="module", autouse=True)
def _every_pallas_lane_on_cpu() -> Iterator[None]:
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        jax, "default_backend", lambda: "gpu"
    )  # profile selection, not hardware
    for k, v in _FLAGS.items():
        patcher.setenv(k, v)
    yield
    patcher.undo()


def _problem(seed: int = 11):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.normal(size=(_N, 3)), dtype=jnp.float32)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=(_N,)), dtype=jnp.float32)
    weight = jnp.asarray(rng.normal(size=(_N, 3)), dtype=jnp.float32)
    return positions, masses, weight


@pytest.fixture(scope="module")
def prepared() -> tuple[Any, Any, Any, Any, Any]:
    positions, masses, weight = _problem()
    fmm = FastMultipoleMethod(
        preset="large_n_gpu",
        G=_G,
        softening=_SOFTENING,
        retain_far_pairs_for_grad=True,
        use_pallas=True,
    )
    state = fmm.prepare_state(positions, masses, leaf_size=_LEAF, max_order=_ORDER)
    assert isinstance(state, LargeNPreparedState)
    assert (
        int(getattr(state.radix_fast_payload, "source_particle_ids").size) == 0
    )  # prepacked
    assert state.compact_far_pairs is not None
    return fmm, state, positions, masses, weight


def _rel(got, want):
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.linalg.norm(g - w) / (np.linalg.norm(w) + 1e-30))


@pytest.fixture
def reverse_counters(monkeypatch):
    counts = {k: 0 for k in ("p2m", "m2m", "l2l", "m2l", "near")}

    def wrap(module, name, key):
        real = getattr(module, name)

        def counted(*a, **kw):
            counts[key] += 1
            return real(*a, **kw)

        monkeypatch.setattr(module, name, counted)

    wrap(p2m_real_leaf, "p2m_real_leaves_reverse_pallas", "p2m")
    wrap(cascade_real_level, "m2m_real_levels_reverse_pallas", "m2m")
    wrap(cascade_real_level, "l2l_real_levels_reverse_pallas", "l2l")
    wrap(m2l_real_csr_lanes, "m2l_real_csr_lanes_reverse_pallas", "m2l")
    wrap(nearfield_leafpair_csr, "nearfield_leafpair_csr_reverse_pallas", "near")
    return counts


@pytest.fixture
def forward_counters(monkeypatch):
    counts = {k: 0 for k in ("m2l_lanes", "near_csr")}
    for module, name, key in (
        (m2l_real_csr_lanes, "m2l_real_csr_lanes_pallas", "m2l_lanes"),
        (nearfield_leafpair_csr, "nearfield_leafpair_csr_pallas", "near_csr"),
    ):
        real = getattr(module, name)

        def counted(*a, _real=real, _key=key, **kw):
            counts[_key] += 1
            return _real(*a, **kw)

        monkeypatch.setattr(module, name, counted)
    return counts


def test_the_far_field_is_load_bearing(prepared):
    _, state, *_ = prepared
    assert int(np.asarray(state.compact_far_pairs.far_pair_count)) > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_grad_through_all_five_reverse_kernels_matches_the_direct_sum(
    prepared, reverse_counters, forward_counters
):
    fmm, state, positions, masses, weight = prepared

    def loss(p, m):
        return jnp.sum(weight * fmm.differentiable_accelerations(state, p, m))

    def oracle(p, m):
        return jnp.sum(
            weight
            * direct_sum_gravitational_acceleration(p, m, G=_G, softening=_SOFTENING)
        )

    forward = np.asarray(fmm.differentiable_accelerations(state, positions, masses))
    assert (
        forward_counters["m2l_lanes"] >= 1 and forward_counters["near_csr"] >= 1
    ), forward_counters
    assert reverse_counters == {k: 0 for k in reverse_counters}
    got_p, got_m = jax.grad(loss, argnums=(0, 1))(positions, masses)
    assert all(v == 1 for v in reverse_counters.values()), reverse_counters
    want_p, want_m = jax.grad(oracle, argnums=(0, 1))(positions, masses)
    assert np.all(np.isfinite(np.asarray(got_p))) and np.all(
        np.isfinite(np.asarray(got_m))
    )
    rel_p, rel_m = _rel(got_p, want_p), _rel(got_m, want_m)
    assert rel_p < _RTOL, f"d/dpos rel-L2 {rel_p:.3e}"
    assert rel_m < _RTOL, f"d/dmass rel-L2 {rel_m:.3e}"
    ref_fwd = np.asarray(
        direct_sum_gravitational_acceleration(
            positions, masses, G=_G, softening=_SOFTENING
        )
    )
    assert _rel(forward, ref_fwd) < 5e-3

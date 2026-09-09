"""The opt-in CSR M2L lane is (a) actually taken and (b) force-neutral.

``JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1`` routes the flat real-basis M2L of
``_solidfmm_downward_accumulate_from_multipoles`` to
:func:`jaccpot.pallas.m2l_real_csr.m2l_real_csr_pallas`. On CPU the kernel runs
in interpret mode (``JACCPOT_M2L_CSR_INTERPRET=1``). A lane that is silently not
reached would pass any parity check, so the kernel entry is counted.
"""

from __future__ import annotations

import numpy as np
import pytest

import jaccpot.pallas.m2l_real_csr as csr_mod
from jaccpot.runtime.kernels._downward_prep import _m2l_csr_pallas_active


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", raising=False)
    assert _m2l_csr_pallas_active() is False


def test_flag_on_cpu_needs_interpret(monkeypatch):
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.delenv("JACCPOT_M2L_CSR_INTERPRET", raising=False)
    import jax

    if jax.default_backend() != "gpu":
        assert _m2l_csr_pallas_active() is False
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    assert _m2l_csr_pallas_active() is True


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def test_csr_lane_is_taken_and_matches_the_chunked_lane(monkeypatch):
    import jax.numpy as jnp

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    pos, mass = _plummer(3000)

    def solve():
        # the default (non-strict) runtime: CPU-friendly, real basis, same downward
        solver = FastMultipoleMethod(
            basis="real", theta=0.6,
            G=1.0, softening=1e-3, working_dtype=jnp.float32,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(mode="static_radix", leaf_target=32),
                farfield=FarFieldConfig(mode="auto"), nearfield=NearFieldConfig(mode="auto"),
                mac_type="dehnen"),
            fixed_order=3)
        acc = solver.compute_accelerations(
            jnp.asarray(pos), jnp.asarray(mass), leaf_size=32, max_order=3, theta=0.6
        )
        return np.asarray(acc, np.float64)

    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", raising=False)
    a_ref = solve()

    calls = {"n": 0}
    real_kernel = csr_mod.m2l_real_csr_pallas

    def counting(*a, **k):
        calls["n"] += 1
        return real_kernel(*a, **k)

    monkeypatch.setattr(csr_mod, "m2l_real_csr_pallas", counting)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")
    a_csr = solve()
    assert calls["n"] >= 1, "the CSR lane was never entered"
    rel = np.linalg.norm(a_csr - a_ref) / np.linalg.norm(a_ref)
    assert np.all(np.isfinite(a_csr))
    assert rel < 2e-5, rel

"""The flat-walk lane is the solver's default walk, ``=0`` restores the dual walk,
and the two are force-neutral against each other.

Pattern of ``test_m2l_csr_lane_wiring.py``: a lane that is silently not reached
would pass any parity check, so the builder entry is counted. The strict fused
lane that owns this seam exists only above the large-N threshold on a GPU
(``LargeNPreparedState``), so this runs in the ordinary GPU suite at N=70k with
the env of ``tests/integration/test_strict_run_v2_refresh_capacity.py``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jaccpot.runtime._interaction_cache as ic

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the strict fused large-N lane is GPU-only"
)
_N = 70_000

_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "auto",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "4194304",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "1048576",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


def _plummer(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def _force(monkeypatch, n, leaf, flat):
    import jax
    import jax.numpy as jnp

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    for k, v in _ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", str(n))
    if flat is None:
        monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK", raising=False)
    else:
        monkeypatch.setenv(
            "JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK", "1" if flat else "0"
        )
    pos, mass = _plummer(n)
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=0.6,
        G=1.0,
        softening=1e-3,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=leaf),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=3,
    )
    prepared, ev = solver.strict_fused_prepared_eval_fn(
        positions=jnp.asarray(pos),
        masses=jnp.asarray(mass),
        leaf_size=leaf,
        max_order=3,
        theta=0.6,
    )
    acc = np.asarray(jax.block_until_ready(ev(prepared)), np.float64)
    caps = dict(getattr(solver._impl, "_strict_fused_validated_caps", None) or {})
    return acc, caps


def test_default_on_and_flag_zero_takes_the_dual_walk(monkeypatch):
    calls = {"n": 0}
    real = ic._build_flat_walk_artifacts_strict_streamed

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(ic, "_build_flat_walk_artifacts_strict_streamed", counting)
    a_dual, caps_dual = _force(monkeypatch, _N, 64, flat=False)
    assert calls["n"] == 0
    assert not caps_dual.get("flat_walk")
    a_flat, caps_flat = _force(monkeypatch, _N, 64, flat=None)
    assert calls["n"] >= 1, "the flat-walk lane was never entered by default"
    assert caps_flat.get("flat_walk") is True
    assert caps_flat.get("peak_wavefront", 0) > 0
    assert np.all(np.isfinite(a_flat))
    rel = np.linalg.norm(a_flat - a_dual) / np.linalg.norm(a_dual)
    # same lists as sets; only the fp32 summation order differs
    assert rel < 2e-5, rel


def test_both_walk_flags_set_is_refused(monkeypatch):
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK", "1")
    with pytest.raises(RuntimeError, match="pick one walk"):
        _force(monkeypatch, _N, 64, flat=True)


def test_treecode_flag_alone_wins_over_the_defaulted_flat_walk(monkeypatch):
    calls = {"flat": 0, "treecode": 0}
    real_flat = ic._build_flat_walk_artifacts_strict_streamed
    real_tree = ic._build_treecode_artifacts_strict_streamed

    def count_flat(*a, **k):
        calls["flat"] += 1
        return real_flat(*a, **k)

    def count_tree(*a, **k):
        calls["treecode"] += 1
        return real_tree(*a, **k)

    monkeypatch.setattr(ic, "_build_flat_walk_artifacts_strict_streamed", count_flat)
    monkeypatch.setattr(ic, "_build_treecode_artifacts_strict_streamed", count_tree)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK", "1")
    acc, _ = _force(monkeypatch, _N, 64, flat=None)
    assert calls["flat"] == 0 and calls["treecode"] >= 1, calls
    assert np.all(np.isfinite(acc))

"""Regression guard for the strict fused static-radix hot-path default.

The device-only fused hot path enables the streamed fast-lane
(``_prepare_state_dual_and_downward_strict_streamed_fast``), which is ~10x
faster than the host-routed fallback for the large-N strict fused lane (200k
particles: ~1224 -> ~119 ms/step on an A100) with bit-identical energy /
angular-momentum conservation. It must stay ON by default; a silent regression
of that default would quietly cost ~10x. The env var remains as an opt-out.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

pytest.importorskip("yggdrax")
from jaccpot import FastMultipoleMethod


def _build(**overrides):
    kwargs = dict(
        preset="large_n_gpu",
        tree_build_mode="static_radix",
        working_dtype=jnp.float32,
        fixed_order=4,
    )
    kwargs.update(overrides)
    return FastMultipoleMethod(**kwargs)


def test_strict_fused_device_only_defaults_on(monkeypatch):
    """With no env override, the device-only fused hot path is ON by default."""
    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY", raising=False)
    solver = _build()
    assert solver._impl._strict_fused_device_only is True


def test_strict_fused_device_only_env_opt_out(monkeypatch):
    """The env var still lets a user opt back into the host-routed fallback."""
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY", "0")
    solver = _build()
    assert solver._impl._strict_fused_device_only is False

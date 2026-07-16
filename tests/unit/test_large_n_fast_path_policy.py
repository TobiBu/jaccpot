"""Lock-policy tests for the large-N radix fast path."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from jaccpot.runtime._large_n_nearfield import resolve_large_n_execution_config
from jaccpot.runtime.fmm import FastMultipoleMethod


def _make_large_n_fmm():
    return FastMultipoleMethod(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        tree_type="radix",
        nearfield_mode="bucketed",
        grouped_interactions=False,
        working_dtype=jnp.float32,
    )


def test_large_n_fast_lane_defaults_on(monkeypatch):
    monkeypatch.delenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", raising=False)

    cfg = resolve_large_n_execution_config(_make_large_n_fmm(), num_particles=2048)
    assert bool(cfg.radix_fast_lane)
    assert str(cfg.nearfield_mode) == "bucketed"
    assert bool(cfg.retain_leaf_groups)
    assert bool(cfg.precompute_scatter) is False
    assert int(cfg.target_owned_block_size) == 32
    assert bool(cfg.speed_prepared_layout)


def test_large_n_fast_lane_legacy_opt_out_env_is_noop(monkeypatch):
    monkeypatch.setenv("JACCPOT_LARGE_N_RADIX_FAST_LANE", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "0")

    cfg = resolve_large_n_execution_config(_make_large_n_fmm(), num_particles=2048)
    assert bool(cfg.radix_fast_lane)
    assert int(cfg.target_owned_block_size) == 32


def test_large_n_accel_eval_requires_fast_lane_state(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_LARGE_N_TARGET_BLOCK_SIZE", "8")
    monkeypatch.setenv("JACCPOT_LARGE_N_SPEED_PREPARED_LAYOUT", "1")

    key = jax.random.PRNGKey(7)
    pos_key, mass_key = jax.random.split(key)
    positions = jax.random.uniform(
        pos_key,
        (1024, 3),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    masses = jax.random.uniform(
        mass_key,
        (1024,),
        minval=0.1,
        maxval=1.1,
        dtype=jnp.float32,
    )

    fmm = _make_large_n_fmm()
    state = fmm.prepare_state(
        positions,
        masses,
        leaf_size=256,
        max_order=4,
    )
    state_no_fast = replace(state, radix_fast_lane=False, radix_fast_payload=None)
    assert bool(getattr(state, "radix_fast_lane", False))
    assert getattr(state.neighbor_list, "neighbor_leaf_positions", None) is None

    with pytest.raises(RuntimeError, match="requires radix fast-lane state"):
        _ = fmm.evaluate_prepared_state(state_no_fast)

    acc = fmm.evaluate_prepared_state(state)
    assert tuple(acc.shape) == (1024, 3)
    state_bad = replace(state, nearfield_mode="baseline")
    with pytest.raises(RuntimeError, match="nearfield_mode='bucketed'"):
        _ = fmm.evaluate_prepared_state(state_bad)

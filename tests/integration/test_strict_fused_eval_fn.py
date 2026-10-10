"""``strict_fused_prepared_eval_fn`` -- audit item **F33**.

Lines 1123-1171 of ``fmm_strict_run.py``, unexercised on either backend. The
method exists for benchmarking: it builds a fused-lane prepared state eagerly and
returns a jitted *eval-only* closure, so the evaluate cost of the strict fused
static-radix lane can be compared like-for-like against functional FMM eval APIs.

A benchmarking entry point is exactly the kind of code that rots unnoticed --
nothing in the suite calls it, and a benchmark that fails to build simply looks
like a slow day. Its two guards are tested because they are the difference
between "this configuration cannot be measured" and a misleading number. (The
multi-GPU fused lane builds its shard plans through it too, via
``capacity_plan.measure_shard_plan``.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig, TreeConfig
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime._large_n_types import LargeNPreparedState

N_PARTICLES = 512
LEAF_SIZE = 64
MAX_ORDER = 2


@pytest.fixture
def strict_env(monkeypatch):
    """Make the large-N production profile reachable on CPU."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")


def _engine():
    return FMMEngine(
        preset="large_n_gpu",
        runtime_path="large_n",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        theta=0.6,
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=MAX_ORDER,
    )


def _particles(seed: int = 9):
    key = jax.random.PRNGKey(seed)
    key_pos, key_mass = jax.random.split(key)
    positions = jax.random.uniform(
        key_pos, (N_PARTICLES, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jax.random.uniform(
        key_mass, (N_PARTICLES,), minval=0.1, maxval=1.1, dtype=jnp.float32
    )
    return positions, masses


def test_requires_the_large_n_production_profile():
    """The first guard, and the only one a default engine reaches."""
    fmm = FMMEngine(theta=0.6, working_dtype=jnp.float32, fixed_order=MAX_ORDER)
    positions, masses = _particles()
    with pytest.raises(RuntimeError, match="large_n_gpu production profile"):
        fmm.strict_fused_prepared_eval_fn(
            positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )


@pytest.mark.slow
def test_requires_the_particle_count_in_the_profile_set(strict_env, monkeypatch):
    """The second guard: right profile, a particle count the profile set excludes.

    It names the variable and both remedies in its message rather than failing
    obscurely later, which is the behaviour worth pinning -- a benchmark that
    cannot run should say which setting would let it. The fused-mode switch this
    guard also named is gone: ``JACCPOT_STATIC_STRICT_FUSED_MODE=off`` raises
    before it (2026-10 cleanup, X6), checked last here.
    """
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", "999999")
    fmm = _engine()
    positions, masses = _particles()
    with pytest.raises(RuntimeError) as excinfo:
        fmm.strict_fused_prepared_eval_fn(
            positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )
    message = str(excinfo.value)
    assert "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET" in message
    assert f"N={N_PARTICLES}" in message
    assert "leave it empty to allow all N" in message
    assert not fmm._strict_fused_mode_active

    monkeypatch.delenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "off")
    with pytest.raises(ValueError, match=r"JACCPOT_STATIC_STRICT_FUSED_MODE=.*X6"):
        _engine().strict_fused_prepared_eval_fn(
            positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
        )


@pytest.mark.slow
def test_returns_a_prepared_state_and_an_eval_closure(strict_env, monkeypatch):
    """The happy path: the closure must evaluate the state it was built for.

    The returned callable is the thing being benchmarked, so it is not enough
    that it exists -- it is checked against ``evaluate_prepared_state`` on the
    same state, which is what a benchmark comparing it to another library would
    implicitly be claiming.
    """
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_MODE", "on")  # accepted, inert
    fmm = _engine()
    positions, masses = _particles()

    prepared, eval_fn = fmm.strict_fused_prepared_eval_fn(
        positions=positions, masses=masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER
    )

    assert isinstance(prepared, LargeNPreparedState)
    assert callable(eval_fn)
    assert fmm._strict_fused_mode_active, "the method must leave fused mode active"

    from_closure = np.asarray(eval_fn(prepared))
    from_method = np.asarray(fmm.evaluate_prepared_state(prepared))
    assert from_closure.shape == (N_PARTICLES, 3)
    relative = np.linalg.norm(from_closure - from_method) / np.linalg.norm(from_method)
    assert (
        relative <= 1e-6
    ), f"the eval closure disagrees with evaluate_prepared_state by {relative:.3e}"

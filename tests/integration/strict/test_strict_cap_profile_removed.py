"""No prepare reads or writes the strict cap-profile file any more (cleanup X4).

Until the 2026-10 cleanup (X4) the strict lanes kept a JSON catalogue of traversal
caps at ``JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH`` (by default
``/tmp/jaccpot_static_strict_caps.json``, one shared file on a shared box). A
strict prepare read it and widened its ``max_pair_queue`` / replaced its
``process_block`` from whatever an earlier run -- another session, another user
-- had left there, and every prepare that retried a traversal wrote it back, the
general lane included. So a run's traversal config depended on the file's
history. Both directions are gone; these tests pin that.

The file is watched through ``open`` itself, and an attempt raises: the old
reader and writer each swallowed every exception, so a guard that only raised
would have passed against them. The recorded attempt is what fails the test, and
the raise keeps a regression from writing to the real ``/tmp``.

Both tests fail on the parent commit of X4 (db02d19): the strict prepare opens
the profile to read it, and the general prepare opens it to record its retries.
"""

from __future__ import annotations

import builtins
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.interactions import DualTreeTraversalConfig

from jaccpot.config import (
    FarFieldConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.runtime._fmm_impl import FMMEngine

_ENV = "JACCPOT_STATIC_STRICT_CAP_PROFILE_PATH"


def _guard_profile_opens(monkeypatch):
    """Record, and refuse, every ``open`` of a cap-profile file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Restores ``builtins.open`` at teardown.

    Returns
    -------
    list[str]
        The paths an ``open`` was attempted on; empty is the pass condition.
    """
    attempts: list[str] = []
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        name = file if isinstance(file, int) else os.fspath(file)
        if isinstance(name, (str, bytes)) and "strict_caps" in str(name):
            attempts.append(str(name))
            raise PermissionError(f"test guard: {name} must not be opened")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    return attempts


def test_a_strict_prepare_does_not_read_a_cap_profile(monkeypatch, tmp_path):
    """A recorded profile for exactly this (leaf, N) is left unread.

    The configuration is the strict static-radix lane the removed
    ``test_strict_exact_cap_profile_match_fail_fast`` drove, with the profile it
    wrote: on db02d19 this prepare opened the file to apply the profile.
    """
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    profile = tmp_path / "strict_caps.json"
    payload = json.dumps(
        {
            "version": 2,
            "active_context_key": "tree_mode=static_radix|leaf=128|n=1024",
            "profiles": {
                "tree_mode=static_radix|leaf=128|n=1024": {
                    "max_pair_queue": 16384,
                    "pair_process_block": 1024,
                }
            },
        }
    )
    profile.write_text(payload, encoding="utf-8")
    monkeypatch.setenv(_ENV, str(profile))
    attempts = _guard_profile_opens(monkeypatch)

    key = jax.random.PRNGKey(20260513)
    positions = jax.random.uniform(
        key, (1024, 3), minval=-1.0, maxval=1.0, dtype=jnp.float32
    )
    masses = jnp.ones((1024,), dtype=jnp.float32)
    fmm = FMMEngine(
        preset="large_n_gpu",
        expansion_basis="solidfmm",
        farfield=FarFieldConfig(rotation="solidfmm"),
        theta=0.6,
        nearfield=NearFieldConfig(mode="bucketed", edge_chunk_size=64),
        working_dtype=jnp.float32,
        tree=TreeConfig(mode="static_radix"),
        fixed_order=2,
        softening_kernel="plummer",
    )
    state = fmm.prepare_state(positions, masses, leaf_size=128, max_order=2)
    acc = np.asarray(fmm.evaluate_prepared_state(state))

    assert np.all(np.isfinite(acc))
    assert fmm.get_runtime_diagnostics()["refresh_strict_mode_active_count"] >= 1, (
        "the strict lane did not run, so this test would not have reached the "
        "profile reader it guards"
    )
    assert attempts == []
    assert profile.read_text(encoding="utf-8") == payload


# Naming a full DualTreeTraversalConfig warns that it replaces all four capacities;
# that is the point here, not the subject.
@pytest.mark.filterwarnings("ignore:.*traversal_config.*:UserWarning")
def test_a_retrying_prepare_does_not_write_a_cap_profile(monkeypatch, tmp_path):
    """Traversal retries on the general lane leave no file behind.

    The traversal caps are undersized so the build must retry; on db02d19 the
    prepare then recorded the grown caps to the profile path. The env var is
    unset, so a write would have gone to the shared ``/tmp`` default.
    """
    monkeypatch.delenv(_ENV, raising=False)
    attempts = _guard_profile_opens(monkeypatch)

    rng = np.random.default_rng(0)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (512, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, 512))
    fmm = FMMEngine(
        theta=0.6,
        expansion_basis="solidfmm",
        softening_kernel="plummer",
        runtime_policy=RuntimePolicyConfig(
            traversal_config=DualTreeTraversalConfig(
                max_pair_queue=64,
                process_block=16,
                max_interactions_per_node=8,
                max_neighbors_per_leaf=8,
            )
        ),
    )
    fmm.prepare_state(positions, masses, leaf_size=16, max_order=3)

    assert len(fmm.recent_retry_events) > 0, (
        "the traversal did not retry, so this test would not have reached the "
        "profile writer it guards"
    )
    assert attempts == []
    assert not any("strict_caps" in p.name for p in tmp_path.iterdir())

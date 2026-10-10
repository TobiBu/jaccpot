"""The strict lane's switches after the 2026-10 cleanup (X6).

X6 removed the strict lane's superseded paths (``docs/cleanup_2026-10.md``). Their
switches split two ways, and both halves are pinned here:

* **Removed values raise.** A value that selected a deleted path now raises a
  ``ValueError`` naming the variable, at every door into the fused lane:
  ``strict_run_v2``, ``strict_fused_prepared_eval_fn`` and a fused-device refresh
  (the multi-GPU lane's door). Running the surviving path under a switch that
  asked for another one would measure something other than what the caller set.
* **Accepted values change nothing.** Odisseo's env block, the bench harness and
  the GPU pins set the surviving values, and some read a variable for their own
  purposes (Odisseo gates its lane on ``JACCPOT_STATIC_STRICT_GPU_MODE``). Those
  are accepted, and the fused scan they run is bitwise the one an unset
  environment runs.

The lane runs on CPU with ``jax.default_backend`` reporting ``"gpu"``, as in
``tests/characterization/test_lane_goldens.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import FarFieldConfig, TreeConfig
from jaccpot.runtime._fmm_impl import FMMEngine

N_PARTICLES = 512
LEAF_SIZE = 16
MAX_ORDER = 2

#: ``(variable, value)`` pairs whose path X6 deleted.
REMOVED = [
    ("JACCPOT_STATIC_STRICT_FUSED_MODE", "off"),
    ("JACCPOT_STATIC_STRICT_FUSED_MODE", "0"),
    ("JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY", "0"),
    ("JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING", "0"),
    ("JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE", "1"),
    ("JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH", "1"),
    ("JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD", "0"),
]

#: Every strict-lane variable X6 touched, cleared before each test so the
#: process environment cannot decide what a case measures.
SWITCHES = (
    "JACCPOT_STATIC_STRICT_FUSED_MODE",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY",
    "JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK",
    "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE",
    "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING",
    "JACCPOT_STATIC_STRICT_GPU_MODE",
    "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE",
    "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH",
    "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
    "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS",
)

#: Accepted settings, each run against the unset environment. ``harness`` is what
#: Odisseo's env block and ``compare_force.FAST_LANE_ENV`` set, and every other
#: switch at its default, named; ``inverted`` sets the other value of each switch
#: that is now ignored, and other spellings of the surviving values. Before X6
#: ``GPU_MODE=off`` closed the strict lane, and the fused scan refused to run.
ACCEPTED = {
    "harness": {
        "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
        "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
        "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
        "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
        "JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING": "1",
        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE": "0",
        "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH": "0",
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD": "1",
        "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS": "1",
        "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE": "auto",
        "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING": "1",
    },
    "inverted": {
        "JACCPOT_STATIC_STRICT_GPU_MODE": "off",
        "JACCPOT_STATIC_STRICT_FUSED_MODE": "yes",
        "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "true",
        "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "0",
        "JACCPOT_STATIC_STRICT_FUSED_DISABLE_HOT_TIMING": "on",
        "JACCPOT_STATIC_STRICT_FUSED_ALLOW_UNSAFE_COMPACT_PAIR_REUSE": "off",
        "JACCPOT_STATIC_STRICT_FUSED_NODE_INTERACTIONS_SAFE_PATH": "no",
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD": "true",
        "JACCPOT_STATIC_STRICT_FUSED_REUSE_COMPACT_PAIRS": "0",
        "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_MODE": "on",
        "JACCPOT_LARGE_N_REFRESH_DUAL_PLANNER_STEADY_NO_SUBSTAGE_TIMING": "0",
    },
}


@pytest.fixture
def strict_env(monkeypatch):
    """Open the large-N production profile on CPU, with every X6 switch unset."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


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
        softening_kernel="plummer",
    )


def _particles(seed: int = 5):
    rng = np.random.default_rng(seed)
    positions = jnp.asarray(rng.uniform(-1.0, 1.0, (N_PARTICLES, 3)), jnp.float32)
    velocities = jnp.asarray(0.05 * rng.normal(size=(N_PARTICLES, 3)), jnp.float32)
    masses = jnp.asarray(rng.uniform(0.5, 1.5, N_PARTICLES), jnp.float32)
    return positions, velocities, masses


def _enter(engine, entry: str, positions, velocities, masses):
    if entry == "strict_run_v2":
        return engine.strict_run_v2(
            state=jnp.stack([positions, velocities], axis=1),
            masses=masses,
            dt=1e-3,
            num_steps=1,
            refresh_every=1,
            leaf_size=LEAF_SIZE,
            max_order=MAX_ORDER,
        )
    if entry == "strict_fused_prepared_eval_fn":
        return engine.strict_fused_prepared_eval_fn(
            positions=positions,
            masses=masses,
            leaf_size=LEAF_SIZE,
            max_order=MAX_ORDER,
        )
    # the multi-GPU lane's door: `distributed.fused.fused_force_step` refreshes the
    # state directly. The check comes first, so no state is needed to reach it.
    return engine._refresh_large_n_same_topology(
        None,
        positions,
        masses,
        bounds=None,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        theta=None,
        fused_device_mode=True,
    )


@pytest.mark.parametrize(
    "entry", ["strict_run_v2", "strict_fused_prepared_eval_fn", "fused_refresh"]
)
@pytest.mark.parametrize("name, value", REMOVED)
def test_a_removed_value_raises_at_every_fused_entry(
    strict_env, monkeypatch, name, value, entry
):
    """Before any device work, and naming the variable and the phase."""
    monkeypatch.setenv(name, value)
    engine = _engine()
    with pytest.raises(ValueError, match=rf"{name}=.*removed.*\(X6\)"):
        _enter(engine, entry, *_particles())


def test_the_strict_lane_is_the_production_profile_whatever_gpu_mode_says(
    strict_env, monkeypatch
):
    """``JACCPOT_STATIC_STRICT_GPU_MODE`` no longer opens or closes the strict plan.

    ``off`` used to close it on the production profile, and ``on`` to open it on
    any engine. The plan counter shows which one a prepare resolved.
    """
    positions, _, masses = _particles()
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "off")
    strict = _engine()
    strict.prepare_state(positions, masses, leaf_size=LEAF_SIZE, max_order=MAX_ORDER)
    assert strict._refresh_strict_mode_active_count == 1

    monkeypatch.setenv("JACCPOT_STATIC_STRICT_GPU_MODE", "on")
    general = FMMEngine(theta=0.6, fixed_order=MAX_ORDER, softening_kernel="plummer")
    general.prepare_state(
        jnp.asarray(positions, jnp.float64),
        jnp.asarray(masses, jnp.float64),
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
    )
    assert general._refresh_strict_mode_active_count == 0


def _fused_run(engine, positions, velocities, masses):
    state, prepared, history = engine.strict_run_v2(
        state=jnp.stack([positions, velocities], axis=1),
        masses=masses,
        dt=1e-3,
        num_steps=2,
        refresh_every=1,
        leaf_size=LEAF_SIZE,
        max_order=MAX_ORDER,
        return_history=True,
        return_prepared_state=True,
    )
    accel = engine.evaluate_prepared_state(prepared)
    return np.asarray(state), np.asarray(history), np.asarray(accel)


@pytest.mark.slow
@pytest.mark.parametrize("setting", sorted(ACCEPTED))
def test_accepted_values_run_the_same_fused_scan(strict_env, monkeypatch, setting):
    """Bitwise: the state, its history and the force on the returned state."""
    particles = _particles()
    reference = _fused_run(_engine(), *particles)
    for name, value in ACCEPTED[setting].items():
        monkeypatch.setenv(name, value)
    engine = _engine()
    got = _fused_run(engine, *particles)
    assert engine.get_runtime_diagnostics()["strict_fused_mode_active"] is True
    for label, want, have in zip(("state", "history", "accel"), reference, got):
        np.testing.assert_array_equal(have, want, err_msg=f"{setting}: {label}")

"""Golden outputs for the production lanes ``test_fmm_golden.py`` does not reach.

``test_fmm_golden.py`` drives the general solver path under ``preset="accurate"``.
The lanes production actually runs -- the large-N fast lane, the fused
``strict_run_v2`` scan, and the block-step mutual FMM -- had no golden: their tests
check properties (momentum, direct-sum error, A-vs-A) that a changed but still
plausible answer passes. The cleanup of 2026-10 (``docs/cleanup_2026-10.md``)
deletes code around all three, so each gets a committed snapshot here, recomputed
on every run, with a direct-sum anchor so a regenerated golden cannot snapshot
garbage.

**Float32 lanes, and their tolerance.** The large-N fast lane requires
``working_dtype=float32``, so its goldens cannot use ``test_fmm_golden``'s 1e-12
gate across machines: XLA:CPU code generation differs with the host's instruction
set. Measured on the 512-particle cases below (AMD EPYC 7452, AVX2): forcing
``--xla_cpu_max_isa=AVX2`` reproduces the golden bitwise, and forcing ``SSE4_2``
moves the forces by rel-L2 1.0e-7 and the 3-step state by 4.8e-7. The gate is
rel-L2 <= 1e-5 -- 20-100x that spread, and 100x below the 1e-3 truncation error
any change to the algorithm (pair set, order, centres) moves these outputs by.
The block-step lane runs in float64 and keeps the 1e-12 gate.

The large-N lanes run on CPU here through the pure-JAX fallbacks, with
``jax.default_backend`` reporting ``"gpu"`` (the switch the lane's production
profile reads), the way ``tests/integration/test_fmm.py`` drives them.

Regenerate deliberately with ``JACCPOT_REGEN_GOLDEN=1`` and commit the ``.npz``.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.characterization.test_fmm_golden import (
    G_CONST,
    SOFTENING,
    _direct_sum_accelerations,
    _make_inputs,
)

GOLDEN_DIR = Path(__file__).parent / "golden_lanes"
REGEN = os.environ.get("JACCPOT_REGEN_GOLDEN") == "1"

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
FLOAT32_GATE_REL_L2 = 1.0e-5
FLOAT64_GATE = dict(rtol=1.0e-12, atol=1.0e-12)
# FMM vs direct sum. Observed: large-N 3.9e-4, block-step 5.0e-4.
ANCHOR_REL_L2 = 1.0e-2

# The fused lane's switches as Odisseo and the benches set them before the cleanup
# made them the library default (D2). The goldens below run with every one of them,
# and the near-field sizing knobs, UNSET -- they pin the defaults, and were recorded
# with this env set: the defaults reproduce them. Tests that pin the lane explicitly
# still import the dict.
_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET": str(N),
}

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="golden characterization requires float64 (JAX_ENABLE_X64=1)",
)


def _rel_l2(got: np.ndarray, want: np.ndarray) -> float:
    got = np.asarray(got, np.float64)
    want = np.asarray(want, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _velocities() -> np.ndarray:
    return 0.1 * np.random.default_rng(1).normal(size=(N, 3))


def _check_golden(name: str, values: dict[str, np.ndarray], *, float32: bool):
    path = GOLDEN_DIR / f"{name}.npz"
    if REGEN or not path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **values)
        if not REGEN:
            pytest.skip(f"generated missing golden {path.name} (commit it)")
        return
    golden = np.load(path)
    for key, got in values.items():
        if float32:
            err = _rel_l2(got, golden[key])
            assert err <= FLOAT32_GATE_REL_L2, (
                f"{name}/{key}: rel-L2 {err:.3e} from the committed golden "
                f"(gate {FLOAT32_GATE_REL_L2:g})"
            )
        else:
            np.testing.assert_allclose(
                got, golden[key], **FLOAT64_GATE, err_msg=f"{name}/{key} drifted"
            )


def _large_n_solver():
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
        theta=THETA,
        G=G_CONST,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=LEAF),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
        softening_kernel="plummer",
    )


@pytest.fixture
def _large_n_lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in (
        *_FUSED_ENV,
        "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.usefixtures("_large_n_lane")
def test_large_n_lane_golden():
    positions, masses = _make_inputs("clustered", N)
    solver = _large_n_solver()
    p = jnp.asarray(positions, jnp.float32)
    m = jnp.asarray(masses, jnp.float32)
    prepared = solver.prepare_state(p, m, leaf_size=LEAF, max_order=ORDER)
    accel = np.asarray(solver.evaluate_prepared_state(prepared))

    err = _rel_l2(accel, _direct_sum_accelerations(positions, masses))
    assert err < ANCHOR_REL_L2, f"large-N lane vs direct sum: {err:.3e}"
    _check_golden("large_n_real_clu_n512_p4", {"accel": accel}, float32=True)


def _direct_velocity_verlet(state: np.ndarray, masses, *, dt: float, steps: int):
    """The fused lane's kick-drift-kick with direct-sum forces, in float64."""
    pos, vel = state[:, 0].copy(), state[:, 1].copy()
    acc = _direct_sum_accelerations(pos, masses)
    for _ in range(steps):
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        new = _direct_sum_accelerations(pos, masses)
        vel = vel + 0.5 * (acc + new) * dt
        acc = new
    return np.stack([pos, vel], axis=1)


@pytest.mark.usefixtures("_large_n_lane")
def test_strict_fused_lane_golden():
    positions, masses = _make_inputs("clustered", N)
    state0 = np.stack([positions, _velocities()], axis=1)
    solver = _large_n_solver()
    out, _, _ = solver.strict_run_v2(
        state=jnp.asarray(state0, jnp.float32),
        masses=jnp.asarray(masses, jnp.float32),
        dt=1e-3,
        num_steps=3,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    diagnostics = solver.get_runtime_diagnostics()
    assert diagnostics["strict_fused_mode_active"] is True
    assert diagnostics["strict_fused_fallback_count"] == 0
    out = np.asarray(out)

    # Anchor on the displacement: the state itself is dominated by where the
    # particles started.
    ref = _direct_velocity_verlet(state0, masses, dt=1e-3, steps=3)
    err = _rel_l2(out - state0, ref - state0)
    assert err < ANCHOR_REL_L2, f"fused lane vs direct-sum Verlet: {err:.3e}"
    _check_golden("strict_fused_real_clu_n512_p4_3steps", {"state": out}, float32=True)


def test_blockstep_lane_golden():
    from jaccpot import BlockStepFMM

    positions, masses = _make_inputs("clustered", N)
    p = jnp.asarray(positions)
    m = jnp.asarray(masses)
    force = BlockStepFMM(
        softening=SOFTENING,
        k_max=2,
        theta=THETA,
        max_order=ORDER,
        leaf_size=LEAF,
        basis="real",
        softening_kernel="plummer",
    )
    force.prepare(p, m)
    accel = np.asarray(force.total_accelerations(p, m))
    err = _rel_l2(accel, _direct_sum_accelerations(positions, masses))
    assert err < ANCHOR_REL_L2, f"block-step lane vs direct sum: {err:.3e}"

    rung = jnp.asarray(np.arange(N) % 3, jnp.int32)
    x1, v1, _ = force.advance_base_step(
        p, jnp.asarray(_velocities()), m, rung=rung, dt_max=1e-3
    )
    # The mutual lane's defining property: momentum is conserved to round-off.
    p0 = np.sum(masses[:, None] * _velocities(), axis=0)
    p1 = np.sum(masses[:, None] * np.asarray(v1), axis=0)
    assert np.linalg.norm(p1 - p0) <= 1e-12 * np.sum(masses)
    _check_golden(
        "blockstep_real_clu_n512_p4",
        {"accel": accel, "positions": np.asarray(x1), "velocities": np.asarray(v1)},
        float32=False,
    )

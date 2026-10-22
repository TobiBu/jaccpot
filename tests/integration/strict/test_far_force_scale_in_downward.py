"""``mac_type='dehnen_error'``'s far force scale from the downward sweep (S3b), on CPU.

Each fused step takes eq (16a)'s thresholds from ``f_b``, whose far half is every
far pair's monopole term summed into its target node and pushed down the tree.
Until S3b a separate pass computed it after the walk
(``_force_scale_levels.far_force_scale_sorted``: a chunked scatter over the far
list and two scatters per level, ~390 ms of the 25M disc's 695 ms step). Where the
downward sweep runs the CSR M2L lanes kernel and the Pallas L2L cascade, those
launches now compute it on the side. Here, at N = 512 with both kernels in Pallas
interpret mode, a 2-step particle-carry run:

* takes the new route: the M2L launches carry the force-scale lane, in the eager
  prepare and in the traced refresh, and the separate pass does not run;
* gives the old route's states bit for bit (the far term differs from the
  separate pass's only in its summation order, which moves no list entry here)
  and its carried force scale to fp32 round-off.

CPU, the CSR near field in interpret mode (the force scale needs it), the
production leaf partition (cells), with ``jax.default_backend`` reporting
``"gpu"`` as ``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 512
LEAF, ORDER, THETA = 16, 2, 0.6
SOFTENING, DT, STEPS = 1e-2, 1e-3, 2

_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    # the downward sweep's kernels, as on an A100
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
}
_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
    "JACCPOT_STRICT_REFRESH_DETAIL_DIAG_MODE",
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
    "JACCPOT_CASCADE_PALLAS",
)


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)
    for key, val in _ENV.items():
        monkeypatch.setenv(key, val)


def _solver():
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float32,
        use_pallas=True,
        adaptive_eps=1e-5,
        adaptive_error_model="dehnen_paper",
        mac_force_scale_mode="paper_fb",
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen_error",
        ),
        fixed_order=ORDER,
    )


def _state():
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    state = jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32)
    return state, jnp.asarray(masses, jnp.float32)


def _run(solver, state, masses):
    out, handle, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=DT,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        return_prepared_state=True,
        carry="particles",
    )
    return np.asarray(jax.block_until_ready(out)), np.asarray(handle.force_scale)


def test_the_downward_sweep_computes_the_far_force_scale(monkeypatch):
    import jaccpot.pallas.m2l_real_csr_lanes as lanes
    import jaccpot.runtime.fmm_prepare as fmm_prepare

    calls = {"lane": 0, "separate": 0}
    lane_fn = lanes.m2l_real_csr_lanes_fb_pallas_cvjp
    separate_fn = fmm_prepare.far_force_scale_sorted

    def lane_spy(*args, **kwargs):
        calls["lane"] += 1
        return lane_fn(*args, **kwargs)

    def separate_spy(**kwargs):
        calls["separate"] += 1
        return separate_fn(**kwargs)

    monkeypatch.setattr(lanes, "m2l_real_csr_lanes_fb_pallas_cvjp", lane_spy)
    monkeypatch.setattr(fmm_prepare, "far_force_scale_sorted", separate_spy)
    state, masses = _state()
    new_state, new_scale = _run(_solver(), state, masses)
    # the eager prepare's refresh and the scan's traced one; the separate pass
    # runs only for the first step's seed (a geometric walk, no downward sweep)
    assert calls["lane"] >= 2, calls
    assert calls["separate"] == 1, calls

    # the old route: the same kernels for the locals, the separate pass for f_b
    monkeypatch.setattr(
        fmm_prepare, "_far_force_scale_in_downward_available", lambda *a, **k: False
    )
    calls.update(lane=0, separate=0)
    old_state, old_scale = _run(_solver(), state, masses)
    assert calls["lane"] == 0 and calls["separate"] >= 3, calls

    differ = int(np.sum(new_state != old_state))
    assert differ == 0, (
        f"{differ} of {new_state.size} state elements differ, max "
        f"{float(np.max(np.abs(new_state - old_state))):.3e}"
    )
    assert np.all(old_scale > 0)
    np.testing.assert_allclose(new_scale, old_scale, rtol=2e-6, atol=0.0)

"""``mac_type='dehnen_error'`` inside the fused scan: eq (16a) per step, f_b carried.

The strict fused lane evaluates Dehnen's criterion inside its flat walk on every
traced refresh, with each node's threshold ``eps * min_b f_b`` from the force
scale the PREVIOUS step's evaluation returned (near half from the near-field
kernel's lane, far half from the prepare's far pairs). Pinned on a GPU:

* both carries (state and particles) run a traced rollout with the criterion and
  agree with each other;
* the scale the particle carry hands back equals a fresh evaluation's at the
  returned positions (to the stale-by-one-step lists' tolerance);
* the criterion is live in the traced steps: at a tight ``eps`` the rollout's final
  force is closer to the direct sum than the geometric MAC's at the same theta;
* a handle without a force scale cannot continue a criterion run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.integration.test_strict_run_v2_donation import (
    _FUSED_ENV,
    _LEAF,
    _N,
    _ORDER,
    _THETA,
    _state,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
    ),
]

_SOFT = 2e-3


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
        "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
        "JACCPOT_STATIC_STRICT_FUSED_FRESH_COMPACT_PAIR_REBUILD",
        "JACCPOT_STRICT_CARRY",
        "JACCPOT_STRICT_CARRY_ORDER",
    ):
        monkeypatch.delenv(key, raising=False)


def _solver(mac_type="dehnen_error", eps=1e-5):
    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    extra = (
        dict(
            adaptive_eps=eps,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac_type == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=_THETA,
        G=1.0,
        softening=_SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=_LEAF,
                leaf_partition="cells",
                leaf_capacity=2048,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac_type,
        ),
        fixed_order=_ORDER,
        **extra,
    )


def _run(solver, state, masses, *, steps, carry, prepared=None):
    return solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=1e-3,
        num_steps=steps,
        refresh_every=1,
        leaf_size=_LEAF,
        max_order=_ORDER,
        theta=_THETA,
        prepared_state=prepared,
        return_prepared_state=True,
        carry=carry,
    )


def _direct(pos, masses, idx):
    pos = np.asarray(pos, np.float64)
    m = np.asarray(masses, np.float64)
    d = pos[idx][:, None, :] - pos[None, :, :]
    r2 = np.sum(d * d, axis=-1)
    r2[np.arange(idx.size), idx] = np.inf
    from jaccpot.softening import pair_factors, softening_params_np

    params = softening_params_np("ferrers3", _SOFT)
    g = pair_factors(r2, params, "ferrers3", xp=np)[0]
    g = np.where(np.isfinite(r2), g, 0.0)
    return -np.einsum("ij,ijk->ik", g * m[None, :], d)


def _force_error(solver, state_out, masses):
    prepared = solver.prepare_state(
        state_out[:, 0, :], masses, leaf_size=_LEAF, max_order=_ORDER
    )
    acc = np.asarray(solver.evaluate_prepared_state(prepared), np.float64)
    idx = np.random.default_rng(1).choice(_N, 512, replace=False)
    ref = _direct(state_out[:, 0, :], masses, idx)
    return float(np.linalg.norm(acc[idx] - ref) / np.linalg.norm(ref))


def test_both_carries_run_the_criterion_and_agree():
    state, masses = _state()
    solver = _solver()
    out_state, _, _ = _run(solver, state, masses, steps=3, carry="state")
    out_part, handle, _ = _run(_solver(), state, masses, steps=3, carry="particles")
    assert bool(jnp.all(jnp.isfinite(out_state))) and bool(
        jnp.all(jnp.isfinite(out_part))
    )
    scale = np.asarray(handle.force_scale)
    assert scale.shape == (_N,) and np.all(np.isfinite(scale)) and np.all(scale > 0)
    np.testing.assert_allclose(
        np.asarray(out_part), np.asarray(out_state), rtol=0, atol=2e-5
    )


def test_the_carried_scale_is_the_evaluations_own():
    from jaccpot.runtime._large_n_pipeline import evaluate_large_n_state

    state, masses = _state()
    solver = _solver()
    out, handle, _ = _run(solver, state, masses, steps=2, carry="particles")
    prepared = solver._impl.prepare_state(
        out[:, 0, :],
        masses,
        leaf_size=_LEAF,
        max_order=_ORDER,
        fused_device_mode=True,
    )
    _, fresh = evaluate_large_n_state(
        solver._impl,
        prepared,
        target_indices=None,
        return_potential=False,
        max_acc_derivative_order=0,
        return_force_scale=True,
    )
    carried, fresh = np.asarray(handle.force_scale), np.asarray(fresh)
    # the same positions; only the far lists differ (their thresholds were one
    # step older), which moves the monopole lower bound a little
    rel = np.abs(carried - fresh) / fresh
    assert np.median(rel) < 0.05, float(np.median(rel))


def test_the_criterion_is_live_in_the_traced_steps():
    state, masses = _state()
    geo = _solver(mac_type="dehnen")
    de = _solver(eps=1e-5)
    out_geo, _, _ = _run(geo, state, masses, steps=2, carry="particles")
    out_de, _, _ = _run(de, state, masses, steps=2, carry="particles")
    err_geo = _force_error(geo, out_geo, masses)
    err_de = _force_error(de, out_de, masses)
    assert err_de < 0.5 * err_geo, (err_de, err_geo)


def test_a_handle_without_a_scale_cannot_continue_a_criterion_run():
    state, masses = _state()
    out, handle, _ = _run(
        _solver(mac_type="dehnen"), state, masses, steps=1, carry="particles"
    )
    assert handle.force_scale is None
    with pytest.raises(RuntimeError, match="force scale"):
        _run(_solver(), out, masses, steps=1, carry="particles", prepared=handle)

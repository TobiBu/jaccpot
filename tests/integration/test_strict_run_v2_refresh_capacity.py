"""The fused ``strict_run_v2`` refresh must not truncate its interaction lists.

Regression test for the 2026-09-06 finding: on ``large_n_gpu``/static_radix at
N=200k, leaf 256, theta 0.6 the eager prepare built exact lists (rows up to
781, via yggdrax's retry ladder) while the traced refresh inside the compiled
velocity-Verlet scan ran the same walk with the preset capacities (256
neighbours per leaf, queue 65536), could not read the overflow flags under
``jit``, and silently kept 15 % of the near field.  Every step after the first
then carried a wrong force -- ~60 % at theta 0.6, 6 % at theta 1.0 -- while
``fallback_count`` stayed 0 and no overflow diagnostic fired.  An energy check
between two lanes with the same bug, or any single-step probe, cannot see it.

The check recovers the force the scan actually applied from the trajectory --
from rest ``a0 = 2 (x1 - x0) / dt^2`` and ``a_k = (x_{k+1} - 2 x_k + x_{k-1})
/ dt^2`` -- and compares it with an eager prepare+evaluate AT THE SAME
POSITIONS ``x_k``.  Comparing against the force at ``x0`` instead is wrong:
this sample has unresolved close pairs whose accelerations dominate the L2 norm
and move far in one step, so the true field changes by order unity between
steps (that false alarm cost half a day).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.slow]

_FUSED_ENV = {
    "JACCPOT_STATIC_STRICT_GPU_MODE": "on",
    "JACCPOT_STATIC_STRICT_FUSED_MODE": "on",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS": "1",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE": "4",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "64",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "2097152",
    "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH": "0",
    "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY": "1",
    "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK": "1",
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS": "1",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP": "131072",
    "JACCPOT_LARGE_N_COMPILED_STATE_MODE": "on",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED": "1",
}


def _plummer(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
    mu = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    st = np.sqrt(1.0 - mu * mu)
    pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return pos.astype(np.float32), np.full(n, 1.0 / n, np.float32)


def _rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="fused strict lane is GPU-only"
)
@pytest.mark.parametrize("flat_walk", ["0", "1"], ids=["dual_walk", "flat_walk"])
def test_strict_run_v2_refresh_keeps_full_neighbor_lists(monkeypatch, flat_walk):
    n = 200_000
    leaf, order, theta = 256, 3, 0.6
    for key, val in _FUSED_ENV.items():
        monkeypatch.setenv(key, val)
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET", str(n))
    # The flat-emission walk (plan "tree walk", 2026-09-10) has no per-leaf row
    # cap: its neighbour list is a CSR of width JACCPOT_LARGE_N_NEIGHBOR_EDGE_
    # PROFILE_FIXED_CAP, and every overflow saturates the far-pair count into
    # the same guard. Same force, same lists as sets, different fp32 order.
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK", flat_walk)

    from jaccpot import (
        FarFieldConfig,
        FastMultipoleMethod,
        FMMAdvancedConfig,
        NearFieldConfig,
        TreeConfig,
    )

    pos, mass = _plummer(n)
    solver = FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=theta,
        G=1.0,
        softening=1e-7,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(mode="static_radix", leaf_target=leaf),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=order,
    )
    P = jnp.asarray(pos)
    M = jnp.asarray(mass)

    # eager reference: the eval-only seam builds the exact lists
    prepared_eager, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=P, masses=M, leaf_size=leaf, max_order=order, theta=theta
    )
    a_eager = np.asarray(jax.block_until_ready(eval_fn(prepared_eager)))
    eager_rows = np.asarray(prepared_eager.neighbor_list.counts)
    assert (
        int(eager_rows.max()) > 256
    ), "this test needs rows longer than the preset cap to be meaningful"

    def eager_force_at(x):
        p, ev = solver.strict_fused_prepared_eval_fn(
            positions=jnp.asarray(x, jnp.float32),
            masses=M,
            leaf_size=leaf,
            max_order=order,
            theta=theta,
        )
        return np.asarray(jax.block_until_ready(ev(p)), np.float64)

    dt = 1e-2
    state0 = jnp.stack([P, jnp.zeros((n, 3), jnp.float32)], axis=1)
    final, prepared_out, history = solver.strict_run_v2(
        state=state0,
        masses=M,
        dt=dt,
        num_steps=2,
        refresh_every=1,
        leaf_size=leaf,
        max_order=order,
        theta=theta,
        prepared_state=None,
        return_prepared_state=True,
        return_history=True,
    )
    hist = np.asarray(jax.block_until_ready(history), np.float64)
    xs = [pos.astype(np.float64)] + [hist[k, :, 0, :] for k in range(hist.shape[0])]
    a0_scan = 2.0 * (xs[1] - xs[0]) / dt**2  # force at x0 (fresh prepare)
    a1_scan = (xs[2] - 2.0 * xs[1] + xs[0]) / dt**2  # force at x1 (traced refresh)

    err0 = _rel_l2(a0_scan, a_eager)
    err1 = _rel_l2(a1_scan, eager_force_at(xs[1]))
    # float32 positions limit the recovery to a few 1e-3; the truncated refresh
    # gave ~0.6 here (rows capped at 128 against 781).
    assert err0 < 1e-2, err0
    assert err1 < 1e-2, (err0, err1)

    # and the refreshed state's lists are full-length, not a capped subset
    refreshed_rows = np.asarray(prepared_out.neighbor_list.counts)
    assert int(refreshed_rows.max()) > 256, int(refreshed_rows.max())
    assert abs(int(refreshed_rows.max()) - int(eager_rows.max())) <= 8, (
        int(refreshed_rows.max()),
        int(eager_rows.max()),
    )
    caps = solver._impl._strict_fused_traced_caps
    if flat_walk == "1":
        assert caps["flat_walk"] is True
        assert caps["near_edge_capacity"] > int(refreshed_rows.sum())
        validated = solver._impl._strict_fused_validated_caps
        assert validated["peak_wavefront"] > 0
        assert caps["queue_capacity"] >= validated["peak_wavefront"]
    else:
        assert caps["max_neighbors_per_leaf_used"] > int(refreshed_rows.max())

"""The fused multi-GPU lane's segment runner on two forced CPU devices (S11, gates A1-A3).

The CPU end-to-end pattern of ``test_fused_multigpu_cpu_e2e.py`` (the large-N profile
reached on CPU with ``jax.default_backend`` reporting ``"gpu"``, the all-gather
``"buf"`` exchange for XLA:CPU's missing ``ragged_all_to_all``, the Pallas kernels
interpreted), at its toy size: N = 2048 uniform cube, leaf 64, p4, theta 0.8,
eps 1e-2 (ferrers3), two devices.

* A1: the template route's force (caps-only prepares, ONE shape template) equals the
  stacked route's (``setup_fused_force``: full eager prepares, stacked states)
  BITWISE with the same caps.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_runner.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import (  # noqa: E402
    decompose,
    make_reference_direct_force,
    setup_fused_force,
)
from jaccpot.distributed.template_force import setup_template_force  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
]

AXIS = "gpus"
NDEV = 2
N = 2048
LEAF = 64
LEAF_CAP = 64
ORDER = 4
THETA = 0.8
SOFT = 1e-2
CAP = int(N / NDEV * 1.3)


@pytest.fixture(scope="module")
def cpu_fused_lane():
    """The CPU switches of the end-to-end test, for the whole module."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "default_backend", lambda: "gpu")
        resolve = ygg_comm.resolve_ragged_method
        mp.setattr(
            ygg_comm,
            "resolve_ragged_method",
            lambda method="auto": "buf" if method == "auto" else resolve(method),
        )
        mp.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
        mp.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
        mp.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
        mp.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")
        yield


def _solver(leaf_capacity=LEAF_CAP):
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFT,
        working_dtype=jnp.float32,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=leaf_capacity,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )


def _mesh(ndev=NDEV):
    return jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))


def _ic(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(N, 3))).astype(np.float32)
    mass = np.full(N, 1.0 / N, np.float32)
    return pos, vel, mass


def _cross_caps():
    return CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF)


def _live_rows(count, cap=CAP):
    count = np.asarray(count)
    return np.concatenate(
        [np.arange(d * cap, d * cap + int(count[d])) for d in range(count.shape[0])]
    )


def _assert_bitwise(got, want, what):
    got, want = np.asarray(got), np.asarray(want)
    differ = int(np.sum(got != want))
    assert differ == 0, (
        f"{what}: {differ} of {got.size} elements differ, max "
        f"{float(np.nanmax(np.abs(got.astype(np.float64) - want))):.3e}"
    )


def test_a1_the_template_force_is_the_stacked_force_bitwise(cpu_fused_lane):
    """GATE A1: same caps, same force, bit for bit; nothing reads the template."""
    mesh = _mesh()
    pos, vel, mass = _ic()
    state, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    solver = _solver()
    stacked, plan_s, walk_s = setup_fused_force(
        solver,
        mesh,
        state,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=_cross_caps(),
    )
    acc_s, flag_s = stacked(state.positions, state.masses, state.count, state.ids)
    assert not bool(np.asarray(flag_s))
    # the template route on the same solver: its prepares start from the walk caps
    # the stacked setup validated, so the two compile the same caps
    template, plan_t, walk_t = setup_template_force(
        solver,
        mesh,
        state,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=_cross_caps(),
    )
    assert plan_t == plan_s
    assert walk_t == walk_s
    acc_t, flag_t = template(state.positions, state.masses, state.count, state.ids)
    assert not bool(np.asarray(flag_t))
    rows = _live_rows(state.count)
    got = np.asarray(acc_t)[rows]
    assert np.all(np.isfinite(got)), "the template route read a template value"
    _assert_bitwise(got, np.asarray(acc_s)[rows], "template vs stacked force")
    # and it is the force: the single-device fused lane's own error on these
    # particles is 3.578e-3 (test_fused_multigpu_cpu_e2e.py)
    ref = make_reference_direct_force(mesh, G=1.0, softening=SOFT)
    want = np.asarray(ref(state.positions, state.masses, state.count, state.ids)[0])
    want = want[rows].astype(np.float64)
    err = float(np.linalg.norm(got - want) / np.linalg.norm(want))
    assert err < 5e-3, err

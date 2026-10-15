"""The fused multi-GPU lane end to end on two forced CPU devices, flat M2L (review G8).

Until 2026-10 nothing ran this lane's force off a GPU: the unit tier drives
:class:`~jaccpot.distributed.rollout.FusedRollout` with a direct-sum stand-in, and
the two-card gates need A100s. So the lane's non-CSR M2L -- what runs on a
pre-Ampere card, with ``JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0``, or on a CPU -- was
never handed the cross-domain far pairs, and it could not take them: the merged list
has more source rows than targets (a jaxtyping ``nodes`` mismatch) and no live
prefix (``start_idx < None``).

This arm runs the real thing at toy size: ``decompose`` -> ``setup_fused_force``
(eager prepares, merged plans, the cross hook with its buffer exchange) ->
``FusedRollout`` with a repartition, and scores the force against the
partition-independent direct sum. On CPU it needs three switches, none of which
changes what is computed: the large-N profile is gated on a GPU backend (as in
``tests/integration/strict``), XLA:CPU has no ``ragged_all_to_all`` lowering (the
all-gather ``"buf"`` exchange gives the same rows), and the Pallas kernels run
interpreted.

Measured 2026-10-15 at these settings (N=2048 uniform cube, leaf 64, p4, theta 0.8,
eps 1e-2 ferrers3): rel-L2 vs the direct sum 3.643e-3 with the flat M2L and with the
CSR lane interpreted (the two forces differ by 8.3e-8), and 3.578e-3 for the
single-device fused lane on the same particles. 329 cross pairs per device reach
the M2L. 191 s alone on a loaded 64-core host (nearly all of it compilation),
2.9 GB peak RSS.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_fused_multigpu_cpu_e2e.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

import jaccpot.runtime.kernels._downward_prep as downward_prep  # noqa: E402
from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import CrossCapacities  # noqa: E402
from jaccpot.distributed.rollout import (  # noqa: E402
    FusedRollout,
    RolloutConfig,
    decompose,
    make_reference_direct_force,
    setup_fused_force,
)

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


def _rel_error(force, ref, state):
    acc, flag = force(state.positions, state.masses, state.count, state.ids)
    assert not bool(np.asarray(flag)), "a capacity flag fired"
    want = np.asarray(ref(state.positions, state.masses, state.count, state.ids)[0])
    count = np.asarray(state.count)
    rows = np.concatenate(
        [np.arange(d * CAP, d * CAP + int(count[d])) for d in range(NDEV)]
    )
    got, want = np.asarray(acc, np.float64)[rows], np.asarray(want, np.float64)[rows]
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


@pytest.fixture
def cpu_fused_lane(monkeypatch):
    """The three CPU switches (module docstring) plus the flat M2L, explicitly."""
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    resolve = ygg_comm.resolve_ragged_method
    monkeypatch.setattr(
        ygg_comm,
        "resolve_ragged_method",
        lambda method="auto": "buf" if method == "auto" else resolve(method),
    )
    monkeypatch.setenv("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "0")
    monkeypatch.setenv("JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP", "65536")
    monkeypatch.setenv("JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF", "32")


@pytest.fixture
def flat_m2l_cross_pairs(monkeypatch):
    """Per device, the cross-domain pairs each flat M2L call received.

    A call whose multipole table is longer than its local accumulator carries the
    imported sources behind the local nodes; its pairs with a source past the local
    rows are the cross pairs. Counted on device (a callback), at trace time the
    lists are tracers.
    """
    seen: list[int] = []

    def counting(real):
        def call(locals_coeffs, multipoles, centers, src, tgt, active, **kw):
            n_local = int(locals_coeffs.shape[0])
            if int(multipoles.shape[0]) > n_local:
                cross = jnp.sum((src >= n_local) & (tgt >= 0))
                jax.debug.callback(lambda c: seen.append(int(c)), cross)
            return real(locals_coeffs, multipoles, centers, src, tgt, active, **kw)

        return call

    for name in ("_accumulate_m2l_fullbatch", "_accumulate_m2l_chunked_scan"):
        monkeypatch.setattr(downward_prep, name, counting(getattr(downward_prep, name)))
    return seen


def test_two_device_fused_force_through_the_flat_m2l(
    cpu_fused_lane, flat_m2l_cross_pairs
):
    rng = np.random.default_rng(0)
    pos = rng.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
    vel = (0.1 * rng.normal(size=(N, 3))).astype(np.float32)
    mass = np.full(N, 1.0 / N, np.float32)
    solver = FastMultipoleMethod(
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
                leaf_capacity=LEAF_CAP,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type="dehnen",
        ),
        fixed_order=ORDER,
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:NDEV]), (AXIS,))
    state, _ = decompose(mesh, pos, vel, mass, cap=CAP, num_samples=64)
    force, _, _ = setup_fused_force(
        solver,
        mesh,
        state,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        cross_caps=CrossCapacities(max_cells=LEAF_CAP, leaf_width=LEAF),
    )
    ref = make_reference_direct_force(mesh, G=1.0, softening=SOFT)

    err = _rel_error(force, ref, state)
    # both devices import far sources and the flat M2L takes them: without that this
    # test would pass with the cross far field never evaluated
    assert len(flat_m2l_cross_pairs) == NDEV, flat_m2l_cross_pairs
    assert min(flat_m2l_cross_pairs) > 0, flat_m2l_cross_pairs
    # the single-device fused lane's own error on these particles is 3.578e-3; a
    # cross far field that is lost, doubled or misplaced is far outside this
    assert err < 5e-3, err

    roll = FusedRollout(
        mesh,
        state,
        RolloutConfig(
            cap=CAP, dt=1e-3, repartition_every=2, num_samples=64, axis_name=AXIS
        ),
        force,
    )
    roll.start()
    reports = roll.run(2)
    assert [r.repartitioned for r in reports] == [False, True]
    assert _rel_error(force, ref, roll.state) < 5e-3
    assert roll.gather()["positions"].shape == (N, 3)

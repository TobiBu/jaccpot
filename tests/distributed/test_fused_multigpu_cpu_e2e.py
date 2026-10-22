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

The second arm (S11 C) runs production's walk and M2L, interpreted: the Pallas walk
hands the M2L its far list in CSR order and the cross pairs are merged into its rows
(``merge_into_csr``) instead of being appended and the whole list sorted again. Its
force is bitwise the concatenate-and-sort route's (patched back in as the control)
and its program has no sort over the far list. 128 s alone on the same host (two
force programs compiled), 3.1 GB peak RSS.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
        pytest tests/distributed/test_fused_multigpu_cpu_e2e.py -q
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

import jaccpot.distributed.fused as fused  # noqa: E402
import jaccpot.pallas.m2l_real_csr_lanes as csr_lanes  # noqa: E402
import jaccpot.runtime.fmm_sweeps as fmm_sweeps  # noqa: E402
import jaccpot.runtime.kernels._downward_prep as downward_prep  # noqa: E402
import jaccpot.runtime.kernels._l2l as l2l  # noqa: E402
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
    return _score(acc, ref, state)


def _score(acc, ref, state):
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


def _setup():
    """The toy problem on a two-device mesh: ``(mesh, state, force, direct sum)``."""
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
    return mesh, state, force, make_reference_direct_force(mesh, G=1.0, softening=SOFT)


def test_two_device_fused_force_through_the_flat_m2l(
    cpu_fused_lane, flat_m2l_cross_pairs
):
    mesh, state, force, ref = _setup()

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


# ---- the CSR lane: the cross pairs merged into the local list's rows (S11 C) ----


@pytest.fixture
def cpu_csr_lane(cpu_fused_lane, monkeypatch):
    """The same switches with production's walk and M2L, interpreted.

    The Pallas walk (the GPU default) hands the M2L its far list in CSR order
    (``TargetSortedFarPairs``); the flat walk's list is a plain COO list, which
    the cross pairs are still concatenated onto.
    """
    monkeypatch.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", "1")
    monkeypatch.setenv("JACCPOT_M2L_CSR_INTERPRET", "1")


@pytest.fixture
def force_programs(monkeypatch):
    """The HLO of every call of the fused force, lowered from the call's arguments.

    ``jax.jit`` is wrapped only while ``make_fused_force_evaluator`` builds the
    force (its one ``jax.jit``), so this is the program the call then runs.
    """
    texts: list[str] = []
    real_make = fused.make_fused_force_evaluator

    def make(*args, **kwargs):
        real_jit = jax.jit

        def jit(fn, *jit_args, **jit_kwargs):
            jitted = real_jit(fn, *jit_args, **jit_kwargs)

            def call(*call_args):
                texts.append(jitted.lower(*call_args).as_text(dialect="hlo"))
                return jitted(*call_args)

            return call

        with monkeypatch.context() as patch:
            patch.setattr(jax, "jit", jit)
            return real_make(*args, **kwargs)

    monkeypatch.setattr(fused, "make_fused_force_evaluator", make)
    return texts


@pytest.fixture
def csr_m2l_inputs(monkeypatch):
    """What the cross lane's downward sweep and its CSR M2L lanes kernel got.

    ``widths``: ``(local far width, cross width, local list target-sorted)`` per
    sweep with cross pairs; ``lists``: ``(presorted, list width)`` per kernel call
    with imported sources behind the local ones; ``cross``: its live cross pairs
    per device and call (a callback: the lists are tracers at trace time).
    """
    seen: dict = {"widths": [], "lists": [], "cross": []}
    real_sweep = fmm_sweeps._prepare_solidfmm_downward_sweep

    def sweep(*args, **kwargs):
        if kwargs.get("cross_src") is not None:
            seen["widths"].append(
                (
                    int(kwargs["far_pairs_coo"].sources.shape[0]),
                    int(kwargs["cross_src"].shape[0]),
                    isinstance(
                        kwargs["far_pairs_coo"], downward_prep._TargetSortedFarPairCOO
                    ),
                )
            )
        return real_sweep(*args, **kwargs)

    real_lanes = csr_lanes.m2l_real_csr_lanes_pallas_cvjp

    def lanes(multipoles, centers, src, tgt, *rest):
        n_targets, presorted = rest[-3], rest[-2]
        if n_targets is not None and int(multipoles.shape[0]) > int(n_targets):
            seen["lists"].append((bool(presorted), int(src.shape[0])))
            # the merged list is -1 past its live prefix, the concatenated one at
            # its padding and holes: either way the imported sources are the cross
            # pairs
            cross = jnp.sum(src >= n_targets)
            jax.debug.callback(lambda c: seen["cross"].append(int(c)), cross)
        return real_lanes(multipoles, centers, src, tgt, *rest)

    monkeypatch.setattr(fmm_sweeps, "_prepare_solidfmm_downward_sweep", sweep)
    monkeypatch.setattr(csr_lanes, "m2l_real_csr_lanes_pallas_cvjp", lanes)
    return seen


def _sort_shapes(hlo: str) -> list[str]:
    """The shape of every ``sort`` in an HLO module, e.g. ``"s32[4352]"``.

    HLO prints an instruction's result type before its opcode and its operands by
    name (``sort.11 = (s32[10]{0}, s32[10]{0}) sort(k.1, v.1), ...``): the first
    shape of the result is the sorted length.
    """
    return re.findall(r"= \(?([a-z0-9]+\[[0-9,]*\])[^=]*? sort\(", hlo)


def test_two_device_force_merges_the_cross_pairs_into_the_csr(
    cpu_csr_lane, force_programs, csr_m2l_inputs, monkeypatch
):
    """The CSR lane's force: bitwise the concatenate-and-sort route's, without its sort.

    The local walk's far list arrives in CSR order and the cross pairs are slotted
    into its rows (``merge_into_csr``) in the order the stable sort of the
    concatenated list gives them, so the lanes kernel sums every row in the same
    order. The control is the route itself, still the library's for a list that is
    not target-sorted: patched in, the same force sorts the whole far list again.
    """
    mesh, state, force, ref = _setup()
    args = (state.positions, state.masses, state.count, state.ids)

    merged, flag = force(*args)
    assert not bool(np.asarray(flag)), "a capacity flag fired"
    ((width, cross_width, local_sorted),) = set(csr_m2l_inputs["widths"])
    assert local_sorted, "the local far list must arrive target-sorted"
    assert csr_m2l_inputs["lists"] == [(True, width + cross_width)]
    # both devices import far sources and the CSR lane takes them
    cross_merged = list(csr_m2l_inputs["cross"])
    assert len(cross_merged) == NDEV and min(cross_merged) > 0, cross_merged
    # the single-device fused lane's own error on these particles is 3.578e-3
    assert _score(merged, ref, state) < 5e-3

    # the concatenate-and-sort route: the merge's branch reads the CSR switch
    # through `_l2l`, the M2L through `_downward_prep`
    monkeypatch.setattr(l2l, "_m2l_csr_pallas_active", lambda: False)
    jax.clear_caches()
    concatenated, flag = force(*args)
    assert not bool(np.asarray(flag)), "a capacity flag fired"
    assert csr_m2l_inputs["lists"][-1] == (False, width + cross_width)
    assert csr_m2l_inputs["cross"][-NDEV:] == cross_merged

    # exact, not close: the kernel sums the same pairs in the same order per row
    count = np.asarray(state.count)
    rows = np.concatenate(
        [np.arange(d * CAP, d * CAP + int(count[d])) for d in range(NDEV)]
    )
    merged, concatenated = np.asarray(merged)[rows], np.asarray(concatenated)[rows]
    assert np.all(np.isfinite(merged))
    assert np.array_equal(merged, concatenated)

    new_program, old_program = force_programs[0], force_programs[-1]
    whole_list = f"s32[{width + cross_width}]"
    # the control: the old route's sort over the whole far list is visible here
    assert whole_list in _sort_shapes(old_program), _sort_shapes(old_program)
    # and gone from the merged route, which sorts the cross pairs alone
    assert whole_list not in _sort_shapes(new_program), _sort_shapes(new_program)
    assert f"s32[{cross_width}]" in _sort_shapes(new_program)

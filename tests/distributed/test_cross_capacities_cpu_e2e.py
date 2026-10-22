"""The cross capacities sized in the library, end to end on forced CPU devices (S11 B).

Before this the fused multi-GPU lane's cross capacities came from formulas in the
benches (``bench/multigpu_c4_cross_force_probe.py``: per-leaf factors x headroom,
rounded to powers of two with floors). ``setup_fused_force(cross_caps="auto")`` sizes
them in the library instead: a first guess from the shards' leaf counts, one force at
the start positions that measures what every capacity needed (the cross NEEDS vector),
and each capacity sized from its need. Three gates, on the CPU end-to-end configuration
of ``test_fused_multigpu_cpu_e2e.py`` (same switches, same particles):

(a) the force with the library's capacities equals the force with the bench formula's,
    bitwise, both flag-clean: capacities only change buffer widths. With ONE exception,
    found here and pinned around: the flat M2L (CPU, pre-Ampere cards,
    ``JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0``) runs one fullbatch segment sum when the
    merged pair list -- local far list plus the cross far list's CAPACITY -- fits its
    chunk (``FarFieldConfig.m2l_chunk_size``, default 4096) and a chunked scan when it
    does not (``_downward_prep._solidfmm_downward_accumulate_from_multipoles``), and
    the two round differently: 2347 of 6144 components moved by 1 ulp between
    ``export_far_cap`` 2048 and 4096 here, while 1024 = 2048 and 4096 = 8192 = 2^18
    bitwise. ``export_far_cap`` is the only width that moved anything. So this module
    runs the M2L in 512-pair chunks, below both lists: the regime of any real list;
(b) the needs vector equals the counts the probes read off the hook's record, on every
    device, and its mesh reduction is their maximum;
(c) an undersized capacity raises the flag, the needs it reports are what the clean
    run needed (they survive the overflow), and the re-planned capacities are the
    library's, so the re-run is the force of (a).

Plus the rollout on the library's capacities, flag-clean through a repartition.

Measured 2026-10-22 (N = 2048 uniform cube, leaf 64, p4, theta 0.8, two devices): the
library's capacities and the bench formula's give identical forces; ~2 min, nearly all
compilation (five programs). With four forced devices the (b) and (c) gates run at
``ndev = 4`` too:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_cross_capacities_cpu_e2e.py -q
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yggdrax.distributed.comm as ygg_comm

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

import jaccpot.distributed.fused as fused  # noqa: E402
from jaccpot import FastMultipoleMethod  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    TreeConfig,
)
from jaccpot.distributed.cross import (  # noqa: E402
    CROSS_NEEDS_FIELDS,
    CrossCapacities,
    make_cross_hook,
)
from jaccpot.distributed.rollout import (  # noqa: E402
    FusedRollout,
    RolloutConfig,
    decompose,
    setup_fused_force,
)

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
]

AXIS = "gpus"
N = 2048
LEAF = 64
LEAF_CAP = 64
ORDER = 4
THETA = 0.8
SOFT = 1e-2
# below the merged M2L list of both arms of gate (a), so both take the chunked scan
# (module docstring); 4096, the default, sits between them
M2L_CHUNK = 512
CAPS = CROSS_NEEDS_FIELDS[:10]
# what the probes read off the hook's record (`cross_record_keys`), and the hook's own
# needs entries beside them
RECORD_KEYS = (
    "summary_cells",
    "summary_nodes",
    "export_far",
    "export_near",
    "export_walk_peak",
    "send_nodes",
    "near_send_nodes",
    "recv_nodes",
    "near_recv_nodes",
    "send_particles",
    "near_particles",
) + tuple(f"need_{k}" for k in CROSS_NEEDS_FIELDS)


def _bench_formula_caps(shard_leaves: int, ndev: int) -> CrossCapacities:
    """The two-card probe's capacities (``multigpu_c4_cross_force_probe.py``,
    two-sided, defaults), with the e2e's summary (no size bound)."""

    def auto(per_leaf, floor_bits, headroom=2.5):
        need = headroom * per_leaf * shard_leaves * max(1, ndev - 1)
        return 1 << max(int(floor_bits), int(math.ceil(math.log2(max(need, 1.0)))))

    recv_near_csr = auto(7.5, 18)
    return CrossCapacities(
        max_leaves_per_cell=1,
        max_cells=LEAF_CAP,
        export_far_cap=auto(16, 18),
        export_near_cap=auto(10, 18),
        send_node_cap=auto(1.25, 15),
        send_csr_cap=auto(16, 18),
        recv_node_cap=auto(1.25, 15),
        recv_csr_cap=auto(16, 18),
        recv_near_csr_cap=recv_near_csr,
        send_particle_cap=auto(20, 18, headroom=2.0),
        recv_particle_cap=auto(20, 18, headroom=2.0),
        export_walk_queue=auto(3, 16, headroom=1.5),
        walk_queue=max(auto(13, 18, headroom=1.5), recv_near_csr),
        recv_far_cap=auto(9.5, 18, headroom=2.0),
        recv_near_cap=auto(4.5, 17, headroom=2.0),
        leaf_width=LEAF,
    )


class _Lane:
    """One mesh's library-sized force, and evaluators on its prepared states."""

    def __init__(self, ndev: int):
        rng = np.random.default_rng(0)
        pos = rng.uniform(-1.0, 1.0, (N, 3)).astype(np.float32)
        vel = (0.1 * rng.normal(size=(N, 3))).astype(np.float32)
        mass = np.full(N, 1.0 / N, np.float32)
        self.ndev = ndev
        self.cap = int(N / ndev * 1.3)
        self.solver = FastMultipoleMethod(
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
                farfield=FarFieldConfig(m2l_chunk_size=M2L_CHUNK),
                mac_type="dehnen",
            ),
            fixed_order=ORDER,
        )
        self.mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:ndev]), (AXIS,))
        self.state, _ = decompose(
            self.mesh, pos, vel, mass, cap=self.cap, num_samples=64
        )
        # the prepared states and plan the library's force was built on, so the
        # other capacities below are evaluated on exactly the same template
        seen: dict = {}
        real = fused.make_fused_force_evaluator

        def spy(solver, stacked, **kw):
            seen.update(stacked=stacked, plan=kw["plan"])
            return real(solver, stacked, **kw)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(fused, "make_fused_force_evaluator", spy)
            self.force, _, _ = setup_fused_force(
                self.solver,
                self.mesh,
                self.state,
                leaf_size=LEAF,
                max_order=ORDER,
                theta=THETA,
                cross_caps="auto",
            )
        self._real = real
        self._seen = seen
        self._programs: dict = {}

    def run(self, caps: CrossCapacities, *, record: bool) -> dict:
        """One force with ``caps``, the needs reduced over the mesh, and (with
        ``record``) the per-device record; each program is built once."""
        key = (repr(caps), record)
        if key not in self._programs:
            sink, rec = {}, ({} if record else None)
            hook = make_cross_hook(
                ndev=self.ndev,
                theta=THETA,
                caps=caps,
                near_sink=sink,
                record=rec,
                axis_name=AXIS,
            )
            self._programs[key] = self._real(
                self.solver,
                self._seen["stacked"],
                mesh=self.mesh,
                plan=self._seen["plan"],
                leaf_size=LEAF,
                max_order=ORDER,
                theta=THETA,
                axis_name=AXIS,
                cross_hook=hook,
                cross_near_sink=sink,
                cross_record=rec,
                cross_record_keys=RECORD_KEYS if record else (),
                cross_needs=True,
            )
        out = self._programs[key](
            self.state.positions, self.state.masses, self.state.count
        )
        result = dict(
            acc=np.asarray(out[0]),
            flag=bool(np.asarray(out[1])),
            needs=dict(zip(CROSS_NEEDS_FIELDS, np.asarray(out[2]).tolist())),
        )
        if record:
            result["record"] = {
                k: np.asarray(v).reshape(-1) for k, v in zip(RECORD_KEYS, out[3])
            }
        return result

    def live_rows(self) -> np.ndarray:
        count = np.asarray(self.state.count)
        return np.concatenate(
            [
                np.arange(d * self.cap, d * self.cap + int(count[d]))
                for d in range(self.ndev)
            ]
        )


@pytest.fixture(scope="module", params=[2, 4], ids=["ndev2", "ndev4"])
def lane(request):
    """The CPU switches of ``test_fused_multigpu_cpu_e2e.py`` and one lane per mesh."""
    ndev = request.param
    if jax.device_count() < ndev:
        pytest.skip(f"needs {ndev} forced devices")
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
        yield _Lane(ndev)


def test_the_library_capacities_give_the_bench_formulas_force_bitwise(lane):
    """Gate (a). The library sized every capacity from a measurement; the bench
    formula from per-leaf factors and power-of-two floors. Both fit, so the force is
    the same to the bit."""
    if lane.ndev != 2:
        pytest.skip("the force gate runs at two devices (CI's mesh)")
    lib_caps = lane.force.cross_caps
    measured = dict(zip(CROSS_NEEDS_FIELDS, lane.force.cross_needs.tolist()))
    bench_caps = _bench_formula_caps(measured["max_cells"], lane.ndev)
    lib, bench = lane.run(lib_caps, record=True), lane.run(bench_caps, record=True)
    assert not lib["flag"] and not bench["flag"]
    rows = lane.live_rows()
    np.testing.assert_array_equal(lib["acc"][rows], bench["acc"][rows])
    # not vacuous: the two programs ran different buffers (the library's is narrower
    # on every exchange list) and the cross field reached the force
    for name in CAPS[2:9]:
        assert getattr(lib_caps, name) < getattr(bench_caps, name), name
    assert measured["export_far_cap"] > 0 and measured["send_particle_cap"] > 0
    # the uninstrumented library force (no record, no needs) is the same too
    acc, flag = lane.force(
        lane.state.positions, lane.state.masses, lane.state.count, lane.state.ids
    )
    assert not bool(np.asarray(flag))
    np.testing.assert_array_equal(np.asarray(acc)[rows], lib["acc"][rows])


def test_the_needs_are_the_counts_the_probes_read(lane):
    """Gate (b), per device: every entry against the record's own count of the same
    buffer, and the mesh reduction against the per-device maximum."""
    out = lane.run(lane.force.cross_caps, record=True)
    assert not out["flag"]
    rec = out["record"]
    pairs = {
        "max_cells": rec["summary_cells"],
        "max_summary_nodes": rec["summary_nodes"],
        "export_far_cap": rec["export_far"],
        "export_near_cap": rec["export_near"],
        "export_walk_queue": rec["export_walk_peak"],
        "send_node_cap": np.maximum(rec["send_nodes"], rec["near_send_nodes"]),
        "recv_node_cap": np.maximum(rec["recv_nodes"], rec["near_recv_nodes"]),
        "send_particle_cap": rec["send_particles"],
        "recv_particle_cap": rec["near_particles"],
    }
    for name, counted in pairs.items():
        np.testing.assert_array_equal(rec[f"need_{name}"], counted, err_msg=name)
        assert out["needs"][name] == int(np.max(counted)), name
        assert np.all(counted > 0), f"vacuous: {name} is zero on a device"
    width = rec["need_leaf_width"]
    assert np.all((width > 0) & (width <= LEAF)), width
    for bit in CROSS_NEEDS_FIELDS[10:]:
        assert out["needs"][bit] == 0, bit
    # the needs do not depend on the capacities while nothing overflows: the
    # library's first guess measured the same vector
    np.testing.assert_array_equal(
        np.asarray([out["needs"][k] for k in CROSS_NEEDS_FIELDS]),
        lane.force.cross_needs,
    )


def test_an_undersized_capacity_fires_and_replans_to_the_library_force(lane):
    """Gate (c). Two receive-side buffers halved: the flag fires, the needs still read
    what the clean run needed, and the re-plan from them is the library's capacities
    -- whose force gate (a) pins."""
    lib_caps = lane.force.cross_caps
    clean = dict(zip(CROSS_NEEDS_FIELDS, lane.force.cross_needs.tolist()))
    small = lib_caps._with(
        recv_node_cap=clean["recv_node_cap"] // 2,
        send_particle_cap=clean["send_particle_cap"] // 2,
    )
    out = lane.run(small, record=False)
    assert out["flag"], "an undersized cross buffer did not raise the mesh flag"
    assert out["needs"]["overflow"] == 1
    for name in CAPS:
        assert out["needs"][name] == clean[name], name
    for bit in ("walk_lower_bound", "pairs_lower_bound", "mismatch"):
        assert out["needs"][bit] == 0, bit
    replanned = small.replanned(
        np.asarray([out["needs"][k] for k in CROSS_NEEDS_FIELDS]),
        leaf_capacity=LEAF_CAP,
    )
    assert vars(replanned) == vars(lib_caps)
    fixed = lane.run(replanned, record=True)
    assert not fixed["flag"]


def test_the_rollout_runs_flag_clean_on_the_library_capacities(lane):
    if lane.ndev != 2:
        pytest.skip("the rollout runs at two devices (CI's mesh)")
    roll = FusedRollout(
        lane.mesh,
        lane.state,
        RolloutConfig(
            cap=lane.cap, dt=1e-3, repartition_every=2, num_samples=64, axis_name=AXIS
        ),
        lane.force,
    )
    roll.start()
    reports = roll.run(2)
    assert [r.repartitioned for r in reports] == [False, True]
    assert not any(r.overflow for r in reports)

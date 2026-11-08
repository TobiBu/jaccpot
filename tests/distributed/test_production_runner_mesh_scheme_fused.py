"""The mesh segment runner with a scheme on the FUSED force, both MACs, two forced CPU devices.

I2's gate on the fused multi-GPU lane (``test_production_runner_mesh_scheme.py``
gates the runner's machinery with the direct force): the segment runner with a
rowwise KDK scheme (``nornax.schemes.LeapfrogKDK``'s expressions, test-local) is
the built-in runner BITWISE by id -- positions, velocities, the carried total
acceleration and, under ``mac_type='dehnen_error'``, ``f_b`` -- through

* the geometric MAC with cross capacities too small for the first segment (its
  flag fires, the segment is re-planned and re-run) and a repartition before
  every segment;
* the geometric MAC with migration to fixed cuts and an undersized migration
  buffer (``OVERFLOW_MIGRATION``, a re-sized retry);
* ``mac_type='dehnen_error'`` with migration and ``f_b`` on the carry, chained
  segments.

The CPU end-to-end pattern of ``test_mesh_segment_dehnen_error.py`` (its switches,
which the geometric solver runs under too: both arms of every comparison share
them). Minutes of compile per case (marker ``distributed_segment``).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_production_runner_mesh_scheme_fused.py -q
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.segment import MeshSegmentRunner  # noqa: E402
from tests.distributed import test_mesh_segment_dehnen_error as crit  # noqa: E402
from tests.distributed import test_mesh_segment_runner as geo  # noqa: E402
from tests.distributed.test_mesh_segment_dehnen_error import (  # noqa: E402,F401
    cpu_fused_lane,
)
from tests.distributed.test_production_runner_mesh_scheme import KDK  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]


def _same(a, b, names, what):
    ga, gb = a.gather(), b.gather()
    for name in names:
        got, want = np.asarray(ga[name]), np.asarray(gb[name])
        differ = int(np.sum(got != want))
        assert differ == 0, f"{what}: {name}: {differ} of {got.size} elements differ"


def _geometric(scheme, *, migrate, speedup=1.0, **kwargs):
    mesh = geo._mesh()
    pos, vel, mass = geo._ic()
    return MeshSegmentRunner.fused(
        geo._solver(),
        mesh,
        geo._mesh_state(mesh, pos, vel * np.float32(speedup), mass),
        leaf_size=geo.LEAF,
        max_order=geo.ORDER,
        theta=geo.THETA,
        repartition_every=1,
        num_samples=64,
        migrate=migrate,
        scheme=scheme,
        **kwargs,
    )


def test_geometric_mac_with_a_cross_retry(cpu_fused_lane):
    """3 segments of 2 steps, cross capacities too small for the first (one retry),
    a repartition before each after the first: scheme == built-in, bitwise."""
    runs = []
    for scheme in (None, KDK()):
        runner = _geometric(
            scheme,
            migrate=False,
            cross_caps=geo._cross_caps()._with(recv_node_cap=64, send_particle_cap=512),
        )
        reports = [runner.run(2, dt=geo.DT) for _ in range(3)]
        assert [r.retries for r in reports] == [1, 0, 0], reports
        runs.append(runner)
    _same(runs[1], runs[0], ("positions", "velocities", "acceleration"), "cross retry")


def test_geometric_mac_with_migration_and_a_buffer_retry(cpu_fused_lane):
    """30x the speeds, ``migrate=True`` with ``migration_cap=1``: the first segment's
    migration is declined, the buffer re-sized, the segment re-run; 2 segments.
    Scheme == built-in, bitwise (rows migrated in both)."""
    runs = []
    for scheme in (None, KDK()):
        runner = _geometric(scheme, migrate=True, speedup=30.0, migration_cap=1)
        reports = [runner.run(2, dt=geo.DT) for _ in range(2)]
        assert reports[0].retries >= 1 and all(r.bits == 0 for r in reports), reports
        assert all(r.migrated > 0 for r in reports), reports
        runs.append((runner, [r.retries for r in reports]))
    assert runs[0][1] == runs[1][1]
    _same(
        runs[1][0], runs[0][0], ("positions", "velocities", "acceleration"), "migrate"
    )


def test_dehnen_error_with_migration_and_the_force_scale(cpu_fused_lane, monkeypatch):
    """``mac_type='dehnen_error'``, ``migrate=True``, 2 + 2 steps chained, ``f_b``
    on the carry: scheme == built-in, bitwise including ``f_b``."""
    # as test_b_segments_chain_with_the_force_scale_carried: no far-cap retry here
    monkeypatch.setenv("JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP", "512")
    mesh = crit._mesh()
    pos, vel, mass = crit._ic(seed=3)
    runs = []
    for scheme in (None, KDK()):
        runner = MeshSegmentRunner.fused(
            crit._solver(),
            mesh,
            crit._mesh_state(mesh, pos, vel, mass),
            leaf_size=crit.LEAF,
            max_order=crit.ORDER,
            theta=crit.THETA,
            repartition_every=1,
            num_samples=64,
            migrate=True,
            scheme=scheme,
        )
        reports = [runner.run(2, dt=crit.DT) for _ in range(2)]
        runs.append((runner, [(r.retries, r.bits) for r in reports]))
    assert runs[0][1] == runs[1][1]
    names = ("positions", "velocities", "acceleration", "force_scale")
    _same(runs[1][0], runs[0][0], names, "dehnen_error")
    assert np.all(np.asarray(runs[1][0].gather()["force_scale"]) > 0)

"""Migration to fixed cuts on the fused lane's force, two forced CPU devices.

The CPU end-to-end pattern of ``test_mesh_segment_runner.py`` (its solver, toy size
and switches): N = 2048 uniform cube, leaf 64, p4, theta 0.8, at 30x its speeds so
boundary particles cross the cuts every step (the drift that makes intruder leaves,
``jaccpot.distributed.migration``).

* A migrating run (``migrate=True``, a repartition before every segment) is
  flag-clean on the library's capacities, every force on exact Morton domains in the
  force's box, accurate against the direct sum; and it needs at most the cross
  volume a run that does not migrate needs.
* An undersized migration buffer fires ``OVERFLOW_MIGRATION``; the segment is
  re-run with the buffer re-sized from the exact need, and the result equals a
  fresh run started with the final buffer (and the same cross capacities),
  bitwise by id.

Minutes of compilation per case (marker ``distributed_segment``).

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/distributed/test_mesh_segment_migration_fused.py -q
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytest.importorskip("yggdrax.distributed.partition", reason="needs sfc_repartition")

from jaccpot.distributed.cross import CROSS_NEEDS_FIELDS  # noqa: E402
from jaccpot.distributed.migration import OVERFLOW_MIGRATION  # noqa: E402
from jaccpot.distributed.rollout import make_reference_direct_force  # noqa: E402
from tests.distributed import test_mesh_segment_runner as a  # noqa: E402
from tests.distributed.test_mesh_segment_migration import _exact_domains  # noqa: E402
from tests.distributed.test_mesh_segment_runner import cpu_fused_lane  # noqa: E402,F401

pytestmark = [
    pytest.mark.skipif(
        jax.device_count() < 2, reason="the fused multi-GPU lane needs >= 2 devices"
    ),
    pytest.mark.slow,
    pytest.mark.distributed_segment,
]

STEPS = 2
SPEEDUP = 30.0


def _fast_ic():
    pos, vel, mass = a._ic()
    return pos, vel * np.float32(SPEEDUP), mass


def _runner(solver, migrate, **kwargs):
    from jaccpot.distributed.segment import MeshSegmentRunner

    mesh = a._mesh()
    pos, vel, mass = _fast_ic()
    return MeshSegmentRunner.fused(
        solver,
        mesh,
        a._mesh_state(mesh, pos, vel, mass),
        leaf_size=a.LEAF,
        max_order=a.ORDER,
        theta=a.THETA,
        repartition_every=1,
        num_samples=64,
        migrate=migrate,
        **kwargs,
    )


def _cross_need(report, name):
    return int(report.needs[-len(CROSS_NEEDS_FIELDS) :][CROSS_NEEDS_FIELDS.index(name)])


def test_a_migrating_fused_run_is_clean_exact_and_accurate(cpu_fused_lane):
    """Two segments of two steps, a repartition between them: no flag, no retry,
    rows migrated in both segments, exact Morton domains at the end, and the
    carried acceleration the direct sum's within the lane's error at this size
    (the single-device lane's own: 3.578e-3, ``test_mesh_segment_runner.py``).
    Its cross near pairs and shipped particles are at most a non-migrating run's
    (same solver, same capacities)."""
    solver = a._solver()
    moving = _runner(solver, True)
    reports = [moving.run(STEPS, dt=a.DT) for _ in range(2)]
    for r in reports:
        assert r.bits == 0 and r.retries == 0, r
        assert r.migrated > 0, r
    _exact_domains(moving)
    s = moving.state
    ref = make_reference_direct_force(moving.mesh, G=1.0, softening=a.SOFT)
    want = np.asarray(ref(s.state[:, 0], s.masses, s.count, s.ids)[0])
    rows = a._live_rows(s.count)
    got = np.asarray(s.acceleration)[rows]
    err = float(np.linalg.norm(got - want[rows]) / np.linalg.norm(want[rows]))
    assert err < 5e-3, err

    still = _runner(solver, False, cross_caps=moving.force.cross_caps)
    still_reports = [still.run(STEPS, dt=a.DT) for _ in range(2)]
    for name in ("export_near_cap", "send_particle_cap"):
        moved = max(_cross_need(r, name) for r in reports)
        kept = max(_cross_need(r, name) for r in still_reports)
        assert moved <= kept, (name, moved, kept)


def test_an_undersized_migration_buffer_is_replanned_and_equals_a_fresh_run(
    cpu_fused_lane,
):
    """``migration_cap=1``: the first migration that moves two rows from a device
    is declined and fails its step (``OVERFLOW_MIGRATION``); the runner re-sizes
    the buffer from the exact need and re-runs the segment from its start. It
    equals a run started with the final buffer and the same cross capacities,
    which needs no retry, bitwise by id."""
    solver = a._solver()
    retried = _runner(solver, True, migration_cap=1)
    report = retried.run(STEPS, dt=a.DT)
    assert report.retries >= 1 and report.bits == 0, report
    assert all(f["bits"] & OVERFLOW_MIGRATION for f in report.failures), report
    assert retried.migration_cap >= report.migration_need > 1

    fresh = _runner(
        solver,
        True,
        migration_cap=retried.migration_cap,
        cross_caps=retried.force.cross_caps,
    )
    assert fresh.run(STEPS, dt=a.DT).retries == 0
    a._gathered_equal(retried, fresh)

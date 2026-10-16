"""The particle carry's caps-only prepare (S3 session 2, the design's 1f / stage 10).

``strict_run_v2(carry="particles")`` reads four things of its eager prepare: the
template's shapes, the engine's trace-time state, the initial verdict and (under
``mac_type='dehnen_error'``) the eq (16b) seed with the eager permutation; its
start force comes from its start program. The caps-only prepare
(``FusedForceProvider.prepare(caps_only=True)``) builds the tree and the upward pass
and runs the eager walk's ladder, and places no list: its template is the shapes of
a rebuild of the eager tree under ``jax.eval_shape``. It is the particle carry's
default for the bootstrap, the segment retry and the runner's restore
(``JACCPOT_STRICT_CAPS_ONLY_PREPARE=0`` restores the full prepare). Gates, on CPU
at N = 512, each against the full prepare on a fresh solver:

* R8: the template key, the provider key, the verdict and the trace-time state
  (validated walk caps, level registry, depth bound, the near-field stage's static
  caps, the f_b seed) are the full prepare's -- the pure-JAX near field, the CSR
  near field (Pallas interpret mode) and ``dehnen_error``;
* the bootstrap places no list and runs no downward pass eagerly, and the run is
  the full prepare's, bitwise (state, self-gravity, handle keys);
* a segment retry re-prepares caps-only, and the run is the full re-prepare's;
* the runner's restore prepares caps-only and still matches its digests.

``jax.default_backend`` reports ``"gpu"``, as
``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np
import pytest

from jaccpot.runtime import _interaction_cache, _level_shapes
from jaccpot.runtime.fmm_caches import _contains_tracer
from jaccpot.runtime.fmm_prepare import PrepareMixin
from jaccpot.runtime.fused_provider import FusedForceProvider
from tests.integration.strict import test_strict_runners_on_provider as r6

_SWITCH = "JACCPOT_STRICT_CAPS_ONLY_PREPARE"
#: what an eager prepare leaves on the engine that a traced refresh is sized from
_TRACE_TIME_STATE = (
    "_strict_fused_validated_caps",
    "_cells_upward_num_levels",
    "_tree_leaf_capacity",
    "_large_n_fused_static_target_blocks_resolved_cap",
    "_large_n_fused_payload_static_target_block_preflight",
    "_large_n_neighbor_edges_profile_cap",
    "_large_n_overflow_profile_cap",
)


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in (*r6._ENV_UNSET, _SWITCH):
        monkeypatch.delenv(key, raising=False)
    # the level registry is process-global: each prepare here starts from none
    monkeypatch.setattr(_level_shapes, "_WIDTHS", {})
    monkeypatch.setattr(_level_shapes, "_LEVELS", {})


def _lane_solver(lane, monkeypatch):
    if lane == "purejax":
        return r6._solver()
    for key, val in r6._CSR_ON_CPU.items():
        monkeypatch.setenv(key, val)
    if lane == "csr":
        return r6._solver(use_pallas=True)
    return r6._dehnen_error_solver()


def _digest(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


@pytest.fixture()
def eager_builds(monkeypatch):
    """Counts the EAGER list placements and downward passes (a trace's are free)."""
    calls = {"lists": 0, "downward": 0}
    lists = _interaction_cache._flat_walk_lists
    downward = PrepareMixin._prepare_downward_with_artifacts

    def spy_lists(*args, **kwargs):
        calls["lists"] += not _contains_tracer(args)
        return lists(*args, **kwargs)

    def spy_downward(self, **kwargs):
        calls["downward"] += not _contains_tracer(kwargs.get("upward"))
        return downward(self, **kwargs)

    monkeypatch.setattr(_interaction_cache, "_flat_walk_lists", spy_lists)
    monkeypatch.setattr(PrepareMixin, "_prepare_downward_with_artifacts", spy_downward)
    return calls


def _prepare(lane, monkeypatch, *, caps_only):
    _level_shapes._WIDTHS.clear()
    _level_shapes._LEVELS.clear()
    blobs, _, masses = r6._systems()
    solver = _lane_solver(lane, monkeypatch)
    prov, topology = FusedForceProvider.prepare(
        solver,
        blobs,
        masses,
        leaf_size=r6.LEAF,
        max_order=r6.ORDER,
        theta=r6.THETA,
        **({"caps_only": True} if caps_only else {}),
    )
    engine = solver._impl
    state = {name: getattr(engine, name, None) for name in _TRACE_TIME_STATE}
    state["registry"] = (dict(_level_shapes._WIDTHS), dict(_level_shapes._LEVELS))
    seed = getattr(engine, "_last_force_scale_particles", None)
    return dict(
        prov=prov,
        prepared=topology.prepared,
        ok=bool(np.asarray(prov.capacity_ok(topology))),
        state=state,
        seed=None if seed is None else np.asarray(seed),
        permutation=np.asarray(topology.prepared.tree.particle_indices),
    )


@pytest.mark.parametrize("lane", ["purejax", "csr", "dehnen_error"])
def test_the_caps_only_prepare_is_the_full_prepares_template_and_state(
    lane, monkeypatch, eager_builds
):
    """R8 on CPU: everything the particle carry reads of its prepare, equal."""
    from jaccpot.runtime.strict_carry import PreparedCaps

    full = _prepare(lane, monkeypatch, caps_only=False)
    assert eager_builds == {"lists": 1 + (lane == "dehnen_error"), "downward": 1}
    caps = _prepare(lane, monkeypatch, caps_only=True)
    # the dehnen_error seed walks and places its own lists, in both prepares
    assert eager_builds == {"lists": 2 * (lane == "dehnen_error") + 1, "downward": 1}

    assert isinstance(caps["prepared"], PreparedCaps)
    assert caps["prov"].template.key() == full["prov"].template.key()
    assert _digest(caps["prov"].key()) == _digest(full["prov"].key())
    assert caps["ok"] and caps["ok"] == full["ok"]
    assert caps["state"] == full["state"]
    np.testing.assert_array_equal(caps["permutation"], full["permutation"])
    if lane == "dehnen_error":
        np.testing.assert_array_equal(caps["seed"], full["seed"])


def _bootstrap_run(lane, monkeypatch, *, caps_only, steps=r6.STEPS):
    monkeypatch.setenv(_SWITCH, "1" if caps_only else "0")
    _level_shapes._WIDTHS.clear()
    _level_shapes._LEVELS.clear()
    _, state, masses = r6._systems()
    out, handle, _ = r6._run(
        _lane_solver(lane, monkeypatch), state, masses, carry="particles", steps=steps
    )
    return out, handle


def _same_run(got, want):
    (out, handle), (out_ref, handle_ref) = got, want
    np.testing.assert_array_equal(np.asarray(out), np.asarray(out_ref))
    np.testing.assert_array_equal(
        np.asarray(handle.self_acceleration), np.asarray(handle_ref.self_acceleration)
    )
    assert handle.template.key() == handle_ref.template.key()
    assert _digest(handle.provider.key()) == _digest(handle_ref.provider.key())
    if handle_ref.force_scale is not None:
        np.testing.assert_array_equal(
            np.asarray(handle.force_scale), np.asarray(handle_ref.force_scale)
        )


@pytest.mark.parametrize("lane", ["purejax", "dehnen_error"])
def test_the_particle_carry_bootstraps_without_placing_a_list(
    lane, monkeypatch, eager_builds
):
    """The first call's eager work stops at the walk's caps; its run is the full
    prepare's, bit for bit (the start program evaluated the force either way)."""
    full = _bootstrap_run(lane, monkeypatch, caps_only=False)
    assert eager_builds["lists"] >= 1 and eager_builds["downward"] == 1
    before = dict(eager_builds)
    caps = _bootstrap_run(lane, monkeypatch, caps_only=True)
    # only the dehnen_error seed's own walk places lists
    assert eager_builds["downward"] == before["downward"]
    assert eager_builds["lists"] - before["lists"] == int(lane == "dehnen_error")
    _same_run(caps, full)


def test_a_segment_retry_re_prepares_caps_only(monkeypatch, eager_builds):
    """Caps sized on the blobs, the state stepped on a Plummer sphere: the walk
    overflows, the segment is re-planned and re-prepared at its start -- with no
    list placed -- and the run is the full re-prepare's."""
    blobs, state, masses = r6._systems()
    runs = {}
    for caps_only in (False, True):
        monkeypatch.setenv(_SWITCH, "1" if caps_only else "0")
        _level_shapes._WIDTHS.clear()
        _level_shapes._LEVELS.clear()
        solver = r6._solver()
        small = r6._caps_from(solver, blobs, masses)
        before = dict(eager_builds)
        out, handle, _ = r6._run(
            solver, state, masses, carry="particles", prepared=small
        )
        diag = solver.get_runtime_diagnostics()
        assert diag["strict_fused_last_fallback_reason"] == "capacity_segment_retry"
        runs[caps_only] = (out, handle)
        built = eager_builds["lists"] - before["lists"]
        assert built == (0 if caps_only else 1), built
    _same_run(runs[True], runs[False])


def test_the_restore_prepares_caps_only_and_keeps_its_digests(
    tmp_path, monkeypatch, eager_builds
):
    """A restart file restored on a fresh solver (in this process): no list, the
    digests of the run's compile keys matched, and the continuation is the run's."""
    from jaccpot.production import GalaxySettings, ProductionRunner, SegmentConfig

    settings = GalaxySettings(leaf_size=r6.LEAF, max_order=r6.ORDER, theta=r6.THETA)
    _, state, masses = r6._systems()
    path = str(tmp_path / "restart.npz")

    def runner():
        return ProductionRunner(
            r6._solver(), settings=settings, segment=SegmentConfig(steps=2)
        )

    straight = runner().run(state, masses, dt=r6.DT, num_steps=4)
    first = runner()
    first.run(state, masses, dt=r6.DT, num_steps=2)
    first.save_restart(path)
    before = dict(eager_builds)
    resumed = runner().run(dt=r6.DT, num_steps=4, restart_from=path)
    assert eager_builds == before
    assert resumed.restored["bitwise"], resumed.restored
    np.testing.assert_array_equal(np.asarray(resumed.state), np.asarray(straight.state))

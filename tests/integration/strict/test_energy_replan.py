"""#488: ``ProductionRunner``'s energy diagnostic never goes silently blind (A2).

The cause: a tree re-plan that widens only the level loops' batch (the level
registry, read when a program TRACES and in no provider key) clears the engine's
compiled runners -- but not the runner's energy programs, so the energy program
cached before kept the old width and reported ``energy_ok=False`` at every later
edge whose widest level outgrew it, while the force segments ran with ``retries
0``. The merger at 4.7e7 was blind from t 6.15 to the end, after a width-only
re-plan at t 3.04.

On CPU at N = 512 (``test_strict_runners_on_provider.py``'s cusp drifting into
four blobs, ``test_production_runner.py``'s four blobs), ``jax.default_backend``
reporting ``"gpu"``:

* the energy program's key holds the level registry: after a width-only re-plan
  the energy fits at every edge, and equals a fresh runner's bitwise;
* a rebuild that does not fit the plan is re-planned for the energy only (the
  engine's trace-time state, registry and compiled runners put back), its value
  bitwise the full plan's; with the segment retry off it reports ``ok=False`` with
  the overflow bits, and the row warns (never a silent ``None``).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.production import (
    GalaxySettings,
    ProductionRunner,
    SegmentConfig,
    _diagnostics_row,
    _energy_fields,
)
from tests.integration.strict import test_production_runner as tpr
from tests.integration.strict import test_strict_runners_on_provider as r6
from tests.integration.strict.test_strict_runners_on_provider import (  # noqa: F401
    _fresh_level_registry,
)

SETTINGS = GalaxySettings(
    leaf_size=r6.LEAF,
    max_order=r6.ORDER,
    theta=r6.THETA,
    mac_type="dehnen",
    adaptive_eps=None,
)


@pytest.fixture(scope="module", autouse=True)
def _lane():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        for key in sorted(set(r6._ENV_UNSET) | set(tpr._ENV_UNSET)):
            patch.delenv(key, raising=False)
        yield


def _energy_segments(steps: int = 1) -> SegmentConfig:
    return SegmentConfig(
        steps=steps,
        traced_params=False,
        diagnostics_every=1,
        diagnostics_every_energy=1,
        diagnostics=("energy",),
    )


def _cusp_drift():
    """The cusp a tenth of the way to the four blobs per two steps: step 1 needs a
    wider level batch than the cusp's (18 -> 33), step 3 more leaves (128 ->
    1024)."""
    blobs, _, _ = r6._systems()
    cusp = r6._cusp()
    state, masses = r6._drifting(cusp, cusp + 0.1 * (np.asarray(blobs) - cusp))
    return state, jnp.full_like(masses, 1.0 / r6.N)


def test_a_width_replan_does_not_blind_the_energy(monkeypatch, _fresh_level_registry):
    """The runner's energy program compiled on the cusp at rest (level batch 18),
    then a run whose first segment re-plans the width only (18 -> 33, the leaf
    capacity stays 128): every energy row fits, no energy re-plan is needed (the
    registry is in the key), and the energies are a fresh runner's, bitwise."""
    for key, val in (r6._CSR_ON_CPU | r6._WALK_CAPS["named"]).items():
        monkeypatch.setenv(key, val)
    state, masses = _cusp_drift()
    runner = ProductionRunner(
        r6._solver(leaf_capacity=128, use_pallas=True),
        settings=SETTINGS,
        segment=_energy_segments(),
    )
    warm = runner.run(state.at[:, 1].set(0.0), masses, dt=r6.DT, num_steps=1)
    assert warm.edges[-1].diagnostics["energy_ok"] is True
    shape = (2 * 128 - 1, 128 - 1)
    narrow = _fresh_level_registry._WIDTHS[shape]

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no blind row
        res = runner.run(state, masses, dt=r6.DT, num_steps=2)
    assert [e.retries for e in res.edges] == [0, 1, 0]
    assert [e.plan["leaf_capacity"] for e in res.edges[1:]] == [128, 128]
    assert _fresh_level_registry._WIDTHS[shape] > narrow  # the width-only re-plan
    rows = [e.diagnostics for e in res.edges[1:]]
    assert all(r["energy_ok"] is True and r["energy_replans"] == 0 for r in rows)

    # a fresh runner on a fresh registry: the same energies, bitwise
    monkeypatch.setattr(_fresh_level_registry, "_WIDTHS", {})
    monkeypatch.setattr(_fresh_level_registry, "_LEVELS", {})
    fresh = ProductionRunner(
        r6._solver(leaf_capacity=128, use_pallas=True),
        settings=SETTINGS,
        segment=_energy_segments(),
    ).run(state, masses, dt=r6.DT, num_steps=2)
    assert [e.diagnostics["energy"] for e in fresh.edges[1:]] == [
        r["energy"] for r in rows
    ]


def test_an_overflowing_rebuild_is_replanned_for_the_energy_alone(
    monkeypatch, _fresh_level_registry
):
    """A registry width too small for the state (4): the energy program traces with
    it and overflows; the energy-only re-plan (inside ``_trace_state_kept``) gives
    the full plan's ``W`` bitwise and leaves the engine as it was -- the registry
    at 4, the same compiled-runner cache. The grown program is used again at the
    next call. With the segment retry off nothing is re-planned: ``ok=False``, the
    tree's overflow bit, ``nan`` energies."""
    state, masses = tpr._system()
    runner = tpr._runner(_energy_segments(steps=2))
    res = runner.run(state, masses, dt=tpr.DT, num_steps=2)
    full = runner.potential_energy(res.state, res.masses, res.handle)
    assert full["ok"] and full["replans"] == 0

    engine = runner.engine
    for shape in list(_fresh_level_registry._WIDTHS):
        _fresh_level_registry._WIDTHS[shape] = 4
    registry = dict(_fresh_level_registry._WIDTHS)
    cache = engine._strict_fused_jit_function_cache
    caps = engine._strict_fused_validated_caps
    grown = runner.potential_energy(res.state, res.masses, res.handle)
    assert grown["ok"] and grown["replans"] == 1 and grown["bits"] is None
    assert grown["self"] == full["self"]  # the capacity changes no number
    assert dict(_fresh_level_registry._WIDTHS) == registry
    assert engine._strict_fused_jit_function_cache is cache
    assert engine._strict_fused_validated_caps is caps
    assert runner.potential_energy(res.state, res.masses, res.handle) == grown

    monkeypatch.setenv("JACCPOT_STRICT_SEGMENT_RETRY", "0")
    runner._energy_grown.clear()
    blind = runner.potential_energy(res.state, res.masses, res.handle)
    assert not blind["ok"] and blind["replans"] == 0
    assert blind["bits"] is not None and blind["bits"] & 0x8  # OVERFLOW_TREE
    assert np.isnan(blind["self"])


def test_a_blind_row_says_so():
    """A row whose potential did not fit: ``energy_ok`` False, the energy fields
    ``None`` -- and a ``RuntimeWarning`` with the time, the bits and the re-plans
    tried; a fitted one carries its re-plan count."""
    base = {"energy": -1.0}
    fitted = _energy_fields(0.5, {"self": -1.0, "external": 0.0, "ok": True}, base)
    assert fitted["energy_ok"] and fitted["energy_replans"] == 0
    blind_potential = {
        "self": float("nan"),
        "external": float("nan"),
        "ok": False,
        "replans": 2,
        "bits": 0x8,
    }
    moments = {
        "mass": 1.0,
        "momentum": np.zeros(3),
        "momentum_scale": 1.0,
        "angular_momentum": np.zeros(3),
        "mass_moment": np.zeros(3),
        "kinetic": 0.5,
    }
    baselines = {
        "mass": 1.0,
        "momentum": [0.0] * 3,
        "momentum_scale": 1.0,
        "angular_momentum": [0.0, 0.0, 1.0],
        "com": [0.0] * 3,
        "time": 0.0,
        "kinetic": 0.5,
        "energy": -1.0,
    }
    with pytest.warns(RuntimeWarning, match=r"t=1\.5 is UNKNOWN.*0x8.*2 energy"):
        row = _diagnostics_row(
            moments, baselines, 1.5, ("energy",), True, blind_potential
        )
    assert row["energy_ok"] is False and row["energy"] is None
    assert row["energy_replans"] == 2

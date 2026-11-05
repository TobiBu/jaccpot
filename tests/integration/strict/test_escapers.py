"""The escaper prototype (A2): far particles leave the tree with an exact ledger.

A bound Plummer sphere (N = 1024, clipped at r = 3) and 8 particles just inside
``r_escape = 5``, flying out at six times the escape speed, on CPU (the CSR near
field in interpret mode, cell leaves, the geometric MAC), 6 segments of 10 steps
at dt 1e-3. At the first edge past ``r_escape`` the fliers leave the tree (the rows
are permuted, ``n_b`` -- traced -- drops by 8) and move on the monopole coupling
to the bound system's centre of mass; the bound rows feel their uniform pull.

* the split changes no program: one scan and one edge program;
* the ledger: ``E - sum(jumps)`` stays at the run's own energy noise (the FMM and
  fp32 floor, ~1e-5 here with and without the split: measured 1.19e-5 against
  8.7e-6 for the same run without a split over 200 steps), and each jump (a
  transfer edge's change of representation, the far field's multipole error at
  the fliers) is small;
* the momentum is conserved to the FMM's own accuracy (the coupling's pair
  forces are equal and opposite);
* the final state comes back in the caller's row order.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.escapers import EscaperPolicy, EscaperRunner, escaper_energy
from tests.integration.strict import test_production_runner as tpr
from tests.integration.strict import test_strict_runners_on_provider as r6

N, K = 1024, 8
R_ESCAPE = 5.0
DT, STEPS, SEGMENTS = 1e-3, 10, 6


@pytest.fixture(scope="module", autouse=True)
def _lane():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(jax, "default_backend", lambda: "gpu")
        for key in sorted(set(r6._ENV_UNSET) | set(tpr._ENV_UNSET)):
            patch.delenv(key, raising=False)
        for key, val in r6._CSR_ON_CPU.items():
            patch.setenv(key, val)
        yield


def _system(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 3.0)
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pos = r[:, None] * d
    pos -= pos.mean(axis=0)  # the bound centre at the origin
    vel = 0.3 * rng.normal(size=(N, 3))
    de = rng.normal(size=(K, 3))
    de /= np.linalg.norm(de, axis=1, keepdims=True)
    re = rng.uniform(4.8, 4.9, K)
    ve = 6.0 * np.sqrt(2.0 / re)
    pos = np.concatenate([pos, re[:, None] * de])
    vel = np.concatenate([vel, ve[:, None] * de])
    masses = np.full(N + K, 1.0 / (N + K))
    vel -= np.sum(masses[:, None] * vel, axis=0) / masses.sum()
    return (
        jnp.asarray(np.stack([pos, vel], axis=1), jnp.float32),
        jnp.asarray(masses, jnp.float32),
    )


@pytest.fixture(scope="module")
def run():
    state, masses = _system()
    runner = EscaperRunner(
        r6._solver(leaf_capacity=1024, use_pallas=True),
        EscaperPolicy(r_escape=R_ESCAPE),
        leaf_size=r6.LEAF,
        max_order=r6.ORDER,
        steps=STEPS,
    )
    res = runner.run(state, masses, dt=DT, num_steps=STEPS * SEGMENTS)
    return state, masses, runner, res


def test_the_fliers_leave_the_tree_and_compile_nothing(run):
    _, _, runner, res = run
    counts = [e["n_escapers"] for e in res.edges]
    assert counts[0] == 0 and counts[-1] == K and counts == sorted(counts)
    assert all(e["energy_ok"] for e in res.edges)
    assert res.n_bound == N
    assert sorted(res.order[N:].tolist()) == list(range(N, N + K))
    # one scan program and one edge program: the split is traced, not compiled
    assert len(runner._programs) == 2


def test_the_ledger_conserves_the_energy_and_the_momentum(run):
    state, masses, _, res = run
    e0 = res.edges[0]["energy"]
    counts = [e["n_escapers"] for e in res.edges]
    for i, e in enumerate(res.edges):
        moved = i > 0 and counts[i] != counts[i - 1]
        assert (e["jump"] != 0.0) == moved
        assert abs(e["jump"]) / abs(e0) < 1e-4
        assert abs(e["energy_conserved"] - e0) / abs(e0) < 3e-5
    scale = float(
        np.sum(np.asarray(masses) * np.linalg.norm(np.asarray(state)[:, 1], axis=1))
    )
    p0 = np.asarray(res.edges[0]["momentum"])
    for e in res.edges:
        assert np.linalg.norm(np.asarray(e["momentum"]) - p0) / scale < 1e-6
    # the escapers carry their kinetic energy and the coupling
    last = res.edges[-1]
    assert last["kinetic_escapers"] > 0.0 and last["w_couple"] < 0.0


def test_the_state_comes_back_in_the_callers_order(run):
    state, masses, _, res = run
    assert np.array_equal(np.asarray(res.masses), np.asarray(masses))
    # the fliers are still the last K caller rows, now beyond r_escape
    r = np.linalg.norm(np.asarray(res.state)[:, 0], axis=1)
    assert np.all(r[N:] > R_ESCAPE) and np.all(r[:N] < R_ESCAPE)


def test_the_ledger_energies_of_a_two_body_split():
    """``escaper_energy`` by hand: one tree row and one escaper."""
    state = np.array([[[0.0, 0, 0], [0.0, 1, 0]], [[2.0, 0, 0], [0.0, 0, 3]]])
    masses = np.array([2.0, 1.0])
    row = escaper_energy(state, masses, 1, np.zeros(2), G=1.0)
    assert row["kinetic"] == pytest.approx(0.5 * 2 * 1 + 0.5 * 1 * 9)
    assert row["kinetic_escapers"] == pytest.approx(4.5)
    assert row["w_couple"] == pytest.approx(-1.0 * 2.0 * 1.0 / 2.0)
    assert row["w_tree"] == 0.0
    assert row["momentum"] == pytest.approx([0.0, 2.0, 3.0])

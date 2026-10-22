"""`bench/c1_criterion_calibration.py`: the two invariants the C1 rows rest on.

A HOLD run with a single rung is the shared-step KDK (to round-off: the two are different
jitted programs, bitwise on an A100 on 2026-10-22, 2 of 384 elements 1-2 ulp apart on
CPU); and HOLD's pair kicks balance -- the pair impulse and the angular momentum stay at
round-off, with and without the mid-step refinement. The first failure this guards against was real: a
target tile straddling the end of a rung subset kicked unsynchronised particles
(pair impulse 7e-4 of ``sum m |v|``). Synthetic 128-particle IC, CPU, Pallas interpret
mode.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

c1 = pytest.importorskip("bench.c1_criterion_calibration")

_N = 128


@pytest.fixture(scope="module")
def ic_path(tmp_path_factory):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(_N, 3)) * np.array([0.3, 0.3, 0.05])
    r = np.hypot(x[:, 0], x[:, 1])
    v = 2.5 * np.stack([-x[:, 1] / r, x[:, 0] / r, np.zeros(_N)], 1)
    v += 0.5 * rng.normal(size=(_N, 3))
    path = tmp_path_factory.mktemp("c1") / "ic.npz"
    np.savez(
        path,
        state0=np.stack([x, v], 1).astype(np.float32),
        mass=np.full(_N, 7.2 / _N, np.float32),
        component=(np.arange(_N) % 5 == 0).astype(np.int8),
        halo_mass_code=100.0,
        halo_rs_code=2.0,
        rdisk_code=0.24,
        bulge_mass_code=1.2,
        bulge_scale_code=0.08,
        disk_mass_code=6.0,
    )
    return str(path)


def _run(ic, out, *args, t_end="0.02"):
    argv = [
        "run",
        str(out),
        "--ic",
        ic,
        "--n",
        str(_N),
        "--t-end",
        t_end,
        "--n-out",
        "2",
    ]
    c1.main([*argv, *args])
    with open(str(out)[:-4] + ".json") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("C1_ALLOW_CPU", "1")


def test_single_rung_hold_is_the_shared_kdk(ic_path, tmp_path):
    hold = _run(
        ic_path,
        tmp_path / "hold.npz",
        "--scheme",
        "hold",
        "--kmax",
        "0",
        "--eta",
        "0.2",
    )
    dt = hold["config"]["dt_fine_target"]
    _run(ic_path, tmp_path / "shared.npz", "--scheme", "shared", "--dt", repr(dt))
    a, b = np.load(tmp_path / "hold.npz"), np.load(tmp_path / "shared.npz")
    assert hold["config"]["n_base"] > 1
    np.testing.assert_allclose(a["x"], b["x"], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a["v"], b["v"], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("refine", [False, True])
def test_hold_pair_kicks_balance(ic_path, tmp_path, refine):
    hold = ["--scheme", "hold", "--criterion", "dyn", "--kmax", "3", "--eta", "0.1"]
    out = _run(
        ic_path,
        tmp_path / "hold.npz",
        *hold,
        *(["--refine"] if refine else []),
        t_end="0.05",
    )
    hists = np.asarray(out["costs"]["rung_hist_per_base_step"])
    assert np.all(np.count_nonzero(hists, axis=1) >= 3)  # several rungs occupied
    # 21 particles move to a finer rung mid-step with --refine (2026-10-22)
    assert (out["costs"]["refinements"] > 0) == refine
    ser = out["series"]
    scale = ser[0]["sum_m_abs_v"]
    l0 = np.asarray(ser[0]["L"])
    for row in ser[1:]:
        # fp32 tile sums: ~1e-8 for these heavy particles; the straddling tile gave 7e-4
        assert np.linalg.norm(row["pair_impulse"]) < 1e-6 * scale
        # fp32 pair directions against fp64 positions: ~1e-9; the straddling tile 1e-4
        assert np.linalg.norm(np.asarray(row["L"]) - l0) < 1e-7 * np.linalg.norm(l0)

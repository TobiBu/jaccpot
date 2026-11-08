"""The A4 accuracy suite's tooling (``bench/accuracy``, ``bench/slurm/a4``): no solver runs.

* the sphere sampler: Plummer's Eddington DF equals the analytic one, the IC is
  near virial, a per-component prefix keeps each component's mass;
* the summaries on synthetic rows: the buckling time, the merger's pericentres;
* every ladder rung's command line parses with the suite's own parser, rung names
  are unique, the node tier keeps ``--devices 4``; the smoke rewrite shrinks N;
* the sbatch scripts pass ``bash -n`` and ``submit.sh --dry-run`` prints one array
  per ladder and tier.
"""

from __future__ import annotations

import os
import shlex
import subprocess

import numpy as np
import pytest

from bench.accuracy import analysis, ics, ladders
from bench.accuracy.suite import build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A4 = os.path.join(ROOT, "bench", "slurm", "a4")


def test_plummer_df_is_eddingtons_and_the_ic_is_virial() -> None:
    rho, mass, _ = ics.PROFILES["plummer"]()
    _r, _psi, energies, f = ics._tables(rho, mass)
    sel = (energies > 0.05) & (energies < 0.9)
    analytic = 24 * np.sqrt(2) / (7 * np.pi**3) * energies[sel] ** 3.5
    assert np.max(np.abs(f[sel] / analytic - 1)) < 2e-3
    ic = ics.sample_sphere("plummer", 4000, seed=3)
    x, v, m = ic["state0"][:, 0], ic["state0"][:, 1], ic["mass"]
    kin = 0.5 * np.sum(m * np.sum(v * v, 1))
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    pot = -0.5 * np.sum(m[:, None] * m[None] / d)
    assert abs(2 * kin / abs(pot) - 1) < 0.08
    assert abs(np.median(np.linalg.norm(x, axis=1)) - 1.305) < 0.08  # r50 = 1.305 a


@pytest.mark.parametrize("profile", ["hernquist", "nfw"])
def test_other_spheres_sample(profile: str) -> None:
    ic = ics.sample_sphere(profile, 2000, seed=1)
    assert ic["state0"].shape == (2000, 2, 3) and np.isfinite(ic["state0"]).all()
    assert abs(np.sum(ic["mass"]) - 1.0) < 1e-12


def test_subsample_keeps_component_masses() -> None:
    rng = np.random.default_rng(0)
    comp = np.repeat(np.arange(3, dtype=np.int8), [100, 50, 350])
    ic = dict(
        state0=rng.normal(size=(500, 2, 3)),
        mass=np.where(comp == 2, 2.0, 1.0) / 500.0,
        component=comp,
        component_names=np.array(["disc", "bulge", "halo"]),
        meta={},
    )
    sub = ics.subsample(ic, 100)
    for c in range(3):
        np.testing.assert_allclose(
            sub["mass"][sub["component"] == c].sum(), ic["mass"][comp == c].sum()
        )
    assert np.array_equal(
        sub["state0"][:20], ic["state0"][:20]
    )  # a prefix of each block
    assert sub["meta"]["subsample_fraction"] == 0.2


def test_buckling_time_and_merger_extrema_on_synthetic_rows() -> None:
    t = np.linspace(0.0, 20.0, 2001)
    rows = []
    for tk in t:
        z = 0.045 + 0.02 / (
            1 + np.exp(-(tk - 18.0) / 0.3)
        )  # thickening centred at t = 18
        a2 = np.full(analysis.BAR_NR, 0.5 * min(1.0, tk / 5.0))
        rows.append(
            dict(
                time=float(tk),
                bar=dict(
                    A2_R=a2.tolist(),
                    phase2_R=[0.1] * analysis.BAR_NR,
                    mass_R=[1.0] * analysis.BAR_NR,
                    zrms_R=[z] * analysis.BAR_NR,
                ),
            )
        )
    s = analysis.bar_summary(rows)
    assert abs(s["bar.t_buckle_myr"] - 18.0 * analysis.MYR) < 0.05 * analysis.MYR
    assert abs(s["bar.t_bar_myr"] - 2.0 * analysis.MYR) < 0.05 * analysis.MYR
    tt = np.linspace(0, 6, 601)
    sep = 0.5 + np.abs(np.sin(tt * np.pi / 4.0)) * 8.0  # minima at t = 0, 4
    centres = [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [s_ / 10.0, 0, 0], [0, 0, 0]]
        for s_ in sep
    ]
    mrows = [dict(time=float(a), centres=c) for a, c in zip(tt, centres)]
    m = analysis.merger_summary(mrows, reference={"sep_t": tt, "sep_kpc": sep})
    assert (
        abs(m["merger.peri1_t"] - 4.0) < 0.02
        and abs(m["merger.peri1_kpc"] - 0.5) < 0.05
    )
    assert m["merger.sep_max_dev_kpc"] < 1e-12


@pytest.mark.parametrize("tier", ["horeka-1card", "horeka-node"])
def test_every_ladder_rung_parses(tier: str, monkeypatch) -> None:
    monkeypatch.delenv("A4_SMOKE", raising=False)
    parser = build_parser()
    for name in ladders.ladder_names(tier):
        jobs = ladders.ladder_jobs(name, tier, "/r", "/ic")
        outs = set()
        for job in jobs:
            argv = shlex.split(job)
            assert argv[:4] == ["$A4_PY", "-m", "bench.accuracy.suite", "run"]
            a = parser.parse_args(argv[3:])
            assert a.tier == tier and a.out.startswith(f"/r/{name}/")
            assert (a.devices == 4) == (tier == "horeka-node")
            outs.add(a.out)
        assert len(outs) == len(jobs), name
        _p, _ic, rungs, ref = ladders.ladder_spec(name, tier)
        assert ref is None or ref in [r for r, _ in rungs]
    monkeypatch.setenv("A4_SMOKE", "1")
    a = parser.parse_args(shlex.split(ladders.ladder_jobs("disc_n", tier, "/r")[0])[3:])
    assert a.n == 512 and a.t_end == 0.008


def test_sbatch_scripts_parse_and_dry_run(tmp_path) -> None:
    for name in sorted(os.listdir(A4)):
        if name.endswith((".sh", ".sbatch")):
            subprocess.run(["bash", "-n", os.path.join(A4, name)], check=True)
    env = dict(os.environ, H0_ROOT=str(tmp_path), A4_RUN=str(tmp_path / "run"))
    out = subprocess.run(
        ["bash", os.path.join(A4, "submit.sh"), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln for ln in out.stderr.splitlines() if ln.startswith("sbatch")]
    n_arrays = len(ladders.ladder_names("horeka-1card")) + len(
        ladders.ladder_names("horeka-node")
    )
    assert len(lines) == n_arrays + 2  # + ics + summary
    assert sum("--array=" in ln for ln in lines) == n_arrays

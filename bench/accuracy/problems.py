"""The standard problems: their defaults, analyses, metrics and tolerances.

Every problem names
* its IC source (``sphere:<profile>`` sampled here, or an IC file from Odisseo's
  ``tools/a4_galaxy_ics.py``);
* the run defaults (dt, duration, softening, cadences) per tier;
* the snapshot analysis and the summary (:mod:`bench.accuracy.analysis`);
* the comparable config keys and the tolerances a ``check`` holds a new run to.

Tolerances come in two layers. **Regression** tolerances (every tier): a re-run of the
same configuration must reproduce the baseline's numbers within run-to-run noise
(ratios for errors, relative bands for physical quantities). **Targets** (the HoreKa
tiers): plan U4's absolute ceilings -- force rel-L2 median (``force.p50``) <= 1e-4 for
``dehnen_error``, |dE/E| <= 1e-4 and |dL/L| <= 1e-5 over 1 Gyr in the disc, momentum
at fp32 round-off (``momentum.max_dP`` <= 1e-8, the M1 measurement's scale x 30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from jaccpot.verification import Tolerance

#: the disc annuli of the heating curve (code units: 0.5-1, 1-2, 2-3 Rd at Rd = 0.24)
DISC_ANNULI = ((0.12, 0.24), (0.24, 0.48), (0.48, 0.72))

#: config keys every problem's check requires equal
COMMON_COMPARABLE = (
    "num_particles",
    "ic_seed",
    "dt",
    "t_end",
    "mac",
    "theta",
    "tol",
    "precision",
    "softening",
    "kernel",
    "targets",
    "target_seed",
)


@dataclass(frozen=True)
class Problem:
    """One standard problem.

    Attributes
    ----------
    name : str
        Registry key.
    ic : str
        ``sphere:<profile>`` or ``file`` (``--ic`` required).
    units : str
        ``nbody`` (G = M = 1, scale radius 1) or ``galaxy`` (10 kpc, 1e10 Msun).
    defaults : dict
        Run defaults: ``n``, ``dt``, ``t_end`` (``None``: from ``n_tdyn``), ``n_tdyn``,
        ``softening``, ``steps``, ``analyse_every``, ``score_every``,
        ``energy_every``; per tier overrides under ``tiers``.
    snapshot : str
        ``sphere``, ``disc``, ``bar`` or ``centres``.
    tolerances : Callable
        ``(tier, mac) -> {metric: Tolerance}``.
    extra_comparable : tuple
        Problem-specific comparable keys.
    """

    name: str
    ic: str
    units: str
    defaults: dict
    snapshot: str
    tolerances: Callable
    extra_comparable: tuple = field(default_factory=tuple)


def _force_tol(tier: str, mac: str) -> dict:
    out = {
        "force.rel_l2_start": Tolerance(
            "ratio", 1.25, "force error at the first edge: <= 1.25x the baseline"
        ),
        "force.rel_l2_end": Tolerance(
            "ratio", 1.5, "force error at the end (an evolved state): <= 1.5x"
        ),
        "force.p99_end": Tolerance("ratio", 1.5, "the error tail at the end: <= 1.5x"),
    }
    if tier != "local" and mac == "dehnen_error":
        out["force.p50_end"] = Tolerance(
            "ceiling", 1e-4, "plan U4: force rel-L2 median <= 1e-4"
        )
    return out


def _conservation_tol(tier: str, disc: bool) -> dict:
    out = {
        "energy.max_abs_dE": Tolerance(
            "ratio", 2.0, "energy drift: <= 2x the baseline's"
        ),
        "momentum.max_dP": Tolerance(
            "ratio", 10.0, "momentum drift is a round-off random walk: <= 10x"
        ),
    }
    if disc:
        out["angmom.max_dL"] = Tolerance("ratio", 2.0, "angular-momentum drift: <= 2x")
    if tier != "local":
        out["momentum.max_dP"] = Tolerance(
            "ceiling", 1e-8, "plan U4: momentum at round-off (fp32 forces)"
        )
        if disc:
            out["energy.max_abs_dE"] = Tolerance(
                "ceiling", 1e-4, "plan U4: |dE/E| <= 1e-4 over 1 Gyr"
            )
            out["angmom.max_dL"] = Tolerance(
                "ceiling", 1e-5, "plan U4: |dL/L| <= 1e-5 over 1 Gyr"
            )
    return out


def _sphere_tol(tier: str, mac: str) -> dict:
    out = {
        "lagr.max_drift_r10": Tolerance(
            "abs", 0.05, "inner Lagrangian radius drift within 5 points"
        ),
        "lagr.max_drift_r50": Tolerance(
            "abs", 0.03, "half-mass radius drift within 3 points"
        ),
        "lagr.max_drift_r90": Tolerance(
            "abs", 0.05, "outer Lagrangian radius drift within 5 points"
        ),
        "sigma.max_drift": Tolerance("abs", 0.05, "dispersion drift within 5 points"),
    }
    out.update(_force_tol(tier, mac))
    out.update(_conservation_tol(tier, disc=False))
    return out


def _heating_tol(tier: str, mac: str) -> dict:
    out = {}
    for k in range(len(DISC_ANNULI)):
        out[f"heat.sigz2_growth_{k}"] = Tolerance(
            "abs", 0.05, "sigma_z^2 growth within 5 points"
        )
        out[f"heat.sigR2_growth_{k}"] = Tolerance(
            "abs", 0.05, "sigma_R^2 growth within 5 points"
        )
    out.update(_force_tol(tier, mac))
    out.update(_conservation_tol(tier, disc=True))
    return out


def _bar_tol(tier: str, mac: str) -> dict:
    out = {
        "bar.a2max_peak": Tolerance(
            "abs", 0.05, "peak bar strength (converged across N: 0.73-0.76)"
        ),
        "bar.t_peak_myr": Tolerance(
            "abs", 150.0, "bar peak time (realisation scatter ~300 Myr at 1e7)"
        ),
    }
    if tier != "local":
        out["bar.t_buckle_myr"] = Tolerance(
            "abs", 150.0, "buckling time (round-off scatter ~15 Myr)"
        )
        out["bar.omega_p_end"] = Tolerance("abs", 2.0, "end pattern speed, km/s/kpc")
    out.update(_force_tol(tier, mac))
    out.update(_conservation_tol(tier, disc=True))
    if tier != "local":  # 3 Gyr, not 1: the U4 ceilings are per Gyr
        out["energy.max_abs_dE"] = Tolerance("ceiling", 3e-4, "plan U4 scaled to 3 Gyr")
        out["angmom.max_dL"] = Tolerance("ceiling", 3e-5, "plan U4 scaled to 3 Gyr")
    return out


def _merger_tol(tier: str, mac: str) -> dict:
    out = {
        "merger.peri1_t": Tolerance(
            "abs", 0.02, "first pericentre time (code units; dt ladder spread 0.001)"
        ),
        "merger.peri1_kpc": Tolerance("abs", 0.3, "first pericentre distance, kpc"),
    }
    out.update(_force_tol(tier, mac))
    out.update(_conservation_tol(tier, disc=False))
    return out


PROBLEMS: dict = {
    "plummer": Problem(
        "plummer",
        "sphere:plummer",
        "nbody",
        dict(
            n=8192,
            n_tdyn=100.0,
            t_end=None,
            dt_per_tdyn=1.0 / 64,
            dt_eta=0.05,
            softening=0.02,
            steps=50,
            analyse_every=4,
            score_every=0,
            energy_every=1,
            tiers={
                "local": dict(n=8192),
                "horeka-1card": dict(n=1_000_000),
                "horeka-node": dict(n=4_000_000),
            },
        ),
        "sphere",
        _sphere_tol,
    ),
    "hernquist": Problem(
        "hernquist",
        "sphere:hernquist",
        "nbody",
        dict(
            n=8192,
            n_tdyn=100.0,
            t_end=None,
            dt_per_tdyn=1.0 / 64,
            dt_eta=0.05,
            softening=0.05,
            steps=50,
            analyse_every=4,
            score_every=0,
            energy_every=1,
            tiers={
                "local": dict(n=8192),
                "horeka-1card": dict(n=1_000_000),
                "horeka-node": dict(n=4_000_000),
            },
        ),
        "sphere",
        _sphere_tol,
    ),
    "nfw": Problem(
        "nfw",
        "sphere:nfw",
        "nbody",
        dict(
            n=8192,
            n_tdyn=100.0,
            t_end=None,
            dt_per_tdyn=1.0 / 64,
            dt_eta=0.05,
            softening=0.05,
            steps=50,
            analyse_every=4,
            score_every=0,
            energy_every=1,
            tiers={
                "local": dict(n=8192),
                "horeka-1card": dict(n=1_000_000),
                "horeka-node": dict(n=4_000_000),
            },
        ),
        "sphere",
        _sphere_tol,
    ),
    "disc_heating": Problem(
        "disc_heating",
        "file",
        "galaxy",
        dict(
            n=0,
            dt=1e-3,
            t_end=6.707,
            softening=1.5e-3,
            steps=50,
            analyse_every=2,
            score_every=0,
            energy_every=2,
            tiers={
                "local": dict(n=100_000, t_end=0.5),
                "horeka-1card": dict(n=25_000_000),
                "horeka-node": dict(n=100_000_000),
            },
        ),
        "disc",
        _heating_tol,
    ),
    "bar": Problem(
        "bar",
        "file",
        "galaxy",
        dict(
            n=0,
            dt=1e-3,
            t_end=20.0,
            softening=1.5e-3,
            steps=10,
            analyse_every=1,
            score_every=0,
            energy_every=5,
            tiers={
                "local": dict(n=200_000, t_end=2.0),
                "horeka-1card": dict(n=40_000_000),
                "horeka-node": dict(n=100_000_000),
            },
        ),
        "bar",
        _bar_tol,
    ),
    "merger": Problem(
        "merger",
        "file",
        "galaxy",
        dict(
            n=0,
            dt=1e-3,
            t_end=10.0,
            softening=1.5e-3,
            steps=10,
            analyse_every=1,
            score_every=0,
            energy_every=5,
            tiers={
                "local": dict(n=200_000, t_end=1.6),
                "horeka-1card": dict(n=50_000_000),
                "horeka-node": dict(n=100_000_000),
            },
        ),
        "centres",
        _merger_tol,
    ),
}


def resolve(problem: str, tier: str, overrides: dict) -> dict:
    """The run defaults of ``problem`` at ``tier`` with the caller's overrides.

    Parameters
    ----------
    problem : str
        Registry key.
    tier : str
        ``local``, ``horeka-1card``, ``horeka-node``.
    overrides : dict
        Non-``None`` command-line values win.

    Returns
    -------
    dict
        The resolved run parameters.
    """
    p = PROBLEMS[problem]
    out = {k: v for k, v in p.defaults.items() if k != "tiers"}
    out.update(p.defaults.get("tiers", {}).get(tier, {}))
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def tolerances(problem: str, tier: str, mac: str, metrics: dict) -> dict:
    """The problem's tolerances restricted to metrics the run produced.

    Parameters
    ----------
    problem : str
        Registry key.
    tier : str
        Tier.
    mac : str
        The MAC.
    metrics : dict
        The run's metrics.

    Returns
    -------
    dict
        ``metric -> Tolerance``.
    """
    tol = PROBLEMS[problem].tolerances(tier, mac)
    return {k: v for k, v in tol.items() if metrics.get(k) is not None}


def comparable(problem: str) -> tuple:
    """Config keys a check requires equal.

    Parameters
    ----------
    problem : str
        Registry key.

    Returns
    -------
    tuple
        Keys.
    """
    return COMMON_COMPARABLE + ("problem_ic",) + PROBLEMS[problem].extra_comparable


def reference_or_none(path: Optional[str]) -> Optional[dict]:
    """A reference result's separation series (merger), or ``None``.

    Parameters
    ----------
    path : Optional[str]
        A result JSON.

    Returns
    -------
    Optional[dict]
        ``sep_t``, ``sep_kpc``.
    """
    if not path:
        return None
    from jaccpot.verification import load_result

    series = load_result(path)["series"]
    return {"sep_t": series["sep_t"], "sep_kpc": series["sep_kpc"]}

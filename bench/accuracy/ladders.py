"""The convergence ladders: N, dt, MAC (theta / ``dehnen_error`` tolerance) and precision.

``ladder_jobs(name, tier, root, ic_dir)`` returns one ``suite.py run`` command per rung
(``bench/slurm/a4/`` runs them as SLURM array tasks: ``suite.py ladder NAME --index
$SLURM_ARRAY_TASK_ID``). Every rung writes ``<root>/<ladder>/<rung>/result.json``;
``suite.py summarize`` tabulates a ladder and, for the merger, scores every rung's
orbit against the ladder's converged reference rung.

Galaxy ICs are generated once at the ladder's largest N by Odisseo's
``tools/a4_galaxy_ics.py`` (``bench/slurm/a4/ics.sbatch``): ``<ic_dir>/galaxy.npz``,
``bar.npz``, ``merger.npz``; a rung takes a per-component prefix (``--n``), so the rungs
are nested subsamples of one realisation.

``A4_SMOKE=1`` (the local dry run of gate 4) shrinks every rung to a few hundred
particles and a few steps, so each job's exact command line runs end to end on a CPU
(the node tier's ``--devices 4`` on four forced host devices).

The ``precision`` ladders carry an ``anchored`` rung: A3's anchored state, a slot until
A3 lands (``suite.py``'s ``PRECISIONS``); the job then exits 3 and the array task is
reported as skipped.
"""

from __future__ import annotations

import os

#: N values per tier for the N ladders (1 card: what fits one A100-40GB with headroom;
#: node: the runner's multi-GPU mode on 4 cards)
_N = {
    "sphere": {
        "horeka-1card": [100_000, 1_000_000, 10_000_000],
        "horeka-node": [10_000_000, 40_000_000],
    },
    "disc_heating": {
        "horeka-1card": [1_000_000, 4_000_000, 16_000_000, 64_000_000],
        "horeka-node": [64_000_000, 200_000_000],
    },
    "bar": {
        "horeka-1card": [2_500_000, 10_000_000, 40_000_000],
        "horeka-node": [100_000_000],
    },
    "merger": {
        "horeka-1card": [12_500_000, 25_000_000, 50_000_000],
        "horeka-node": [100_000_000, 200_000_000],
    },
}
#: the fixed N of the dt / MAC / precision ladders per tier
_FIXED_N = {
    "sphere": {"horeka-1card": 1_000_000, "horeka-node": 10_000_000},
    "disc_heating": {"horeka-1card": 16_000_000, "horeka-node": 64_000_000},
    "merger": {"horeka-1card": 25_000_000, "horeka-node": 100_000_000},
}
_MAC_RUNGS = [
    ("theta0.5", ["--mac", "dehnen", "--theta", "0.5"]),
    ("theta0.7", ["--mac", "dehnen", "--theta", "0.7"]),
    ("theta0.8", ["--mac", "dehnen", "--theta", "0.8"]),
    ("tol1e-3", ["--mac", "dehnen_error", "--tol", "1e-3"]),
    ("tol1e-4", ["--mac", "dehnen_error", "--tol", "1e-4"]),
    ("tol3e-5", ["--mac", "dehnen_error", "--tol", "3e-5"]),
]
_PRECISION_RUNGS = [
    (p, ["--precision", p]) for p in ("fp32", "fp64state", "fp64", "anchored")
]


def _fmt_n(n: int) -> str:
    """``n2.5e6`` for 2_500_000 (three significant digits, no ``+`` or leading zeros)."""
    mant, exp = f"{n:.3g}".split("e") if "e" in f"{n:.3g}" else (f"{n:.3g}", "0")
    return f"n{mant}e{int(exp)}"


def _ladders(tier: str) -> dict:
    """``name -> (problem, ic file or None, [(rung, args)], reference rung or None)``."""
    out: dict = {}
    for prob in ("plummer", "hernquist", "nfw"):
        out[f"{prob}_n"] = (
            prob,
            None,
            [(_fmt_n(n), ["--n", str(n)]) for n in _N["sphere"][tier]],
            None,
        )
    fixed = _FIXED_N["sphere"][tier]
    out["plummer_dt"] = (
        "plummer",
        None,
        [
            (f"dtt{d}", ["--n", str(fixed), "--dt-per-tdyn", str(1.0 / d)])
            for d in (32, 64, 128)
        ],
        "dtt128",
    )
    out["plummer_mac"] = (
        "plummer",
        None,
        [(r, ["--n", str(fixed)] + a) for r, a in _MAC_RUNGS],
        None,
    )
    out["plummer_precision"] = (
        "plummer",
        None,
        [(r, ["--n", str(fixed)] + a) for r, a in _PRECISION_RUNGS],
        None,
    )
    out["disc_n"] = (
        "disc_heating",
        "galaxy.npz",
        [(_fmt_n(n), ["--n", str(n)]) for n in _N["disc_heating"][tier]],
        None,
    )
    fixed = _FIXED_N["disc_heating"][tier]
    out["disc_dt"] = (
        "disc_heating",
        "galaxy.npz",
        [(f"dt{d}", ["--n", str(fixed), "--dt", d]) for d in ("2e-3", "1e-3", "5e-4")],
        "dt5e-4",
    )
    out["disc_mac"] = (
        "disc_heating",
        "galaxy.npz",
        [(r, ["--n", str(fixed)] + a) for r, a in _MAC_RUNGS],
        None,
    )
    out["disc_precision"] = (
        "disc_heating",
        "galaxy.npz",
        [(r, ["--n", str(fixed)] + a) for r, a in _PRECISION_RUNGS],
        None,
    )
    # the bar: geometric theta 0.7 was the 2026-10-31 ladder's MAC; dehnen_error beside it
    out["bar_n"] = (
        "bar",
        "bar.npz",
        [
            (_fmt_n(n) + "_t07", ["--n", str(n), "--mac", "dehnen", "--theta", "0.7"])
            for n in _N["bar"][tier]
        ]
        + [(_fmt_n(n) + "_de", ["--n", str(n)]) for n in _N["bar"][tier][:2]],
        None,
    )
    out["merger_n"] = (
        "merger",
        "merger.npz",
        [(_fmt_n(n), ["--n", str(n)]) for n in _N["merger"][tier]],
        None,
    )
    fixed = _FIXED_N["merger"][tier]
    out["merger_dt"] = (
        "merger",
        "merger.npz",
        [(f"dt{d}", ["--n", str(fixed), "--dt", d]) for d in ("2e-3", "1e-3", "5e-4")],
        "dt5e-4",
    )
    out["merger_mac"] = (
        "merger",
        "merger.npz",
        [
            (r, ["--n", str(fixed)] + a)
            for r, a in _MAC_RUNGS
            if r in ("theta0.7", "theta0.8", "tol1e-4")
        ],
        None,
    )
    if tier == "horeka-node":
        # the node tier: N, the MAC and precision on the disc, the bar, the merger's N and dt
        keep = (
            "plummer_n",
            "disc_n",
            "disc_mac",
            "disc_precision",
            "bar_n",
            "merger_n",
            "merger_dt",
        )
        out = {k: v for k, v in out.items() if k in keep}
    return out


#: smoke overrides (A4_SMOKE=1): every rung tiny, so its command runs on a CPU
_SMOKE = {
    "nbody": [
        "--n",
        "384",
        "--n-tdyn",
        "0.5",
        "--targets",
        "64",
        "--steps",
        "4",
        "--energy-every",
        "1",
        "--analyse-every",
        "1",
    ],
    "galaxy": [
        "--n",
        "512",
        "--t-end",
        "0.008",
        "--targets",
        "64",
        "--steps",
        "4",
        "--energy-every",
        "1",
        "--analyse-every",
        "1",
    ],
}


def ladder_names(tier: str = "horeka-1card") -> list:
    """The ladders of a tier.

    Parameters
    ----------
    tier : str
        ``horeka-1card`` or ``horeka-node``.

    Returns
    -------
    list
        Names.
    """
    return sorted(_ladders(tier))


def ladder_spec(name: str, tier: str) -> tuple:
    """``(problem, ic file or None, rungs, reference rung or None)``.

    Parameters
    ----------
    name : str
        Ladder.
    tier : str
        Tier.

    Returns
    -------
    tuple
        The ladder's definition.
    """
    return _ladders(tier)[name]


def ladder_jobs(name: str, tier: str, root: str, ic_dir: str = "$A4_IC") -> list:
    """One ``suite.py run`` command line per rung.

    Parameters
    ----------
    name : str
        Ladder (``ladder_names``).
    tier : str
        ``horeka-1card`` or ``horeka-node``.
    root : str
        Output root (``<root>/<name>/<rung>``).
    ic_dir : str
        Where the galaxy ICs are.

    Returns
    -------
    list
        Shell command lines (``$A4_PY`` is the interpreter).
    """
    from bench.accuracy.problems import PROBLEMS

    problem, ic, rungs, _ref = _ladders(tier)[name]
    smoke = os.environ.get("A4_SMOKE") == "1"
    jobs = []
    for rung, args in rungs:
        cmd = [
            "$A4_PY",
            "-m",
            "bench.accuracy.suite",
            "run",
            problem,
            "--tier",
            tier,
            "--out",
            f"{root}/{name}/{rung}",
        ]
        if ic:
            cmd += ["--ic", f"{ic_dir}/{ic}"]
        if tier == "horeka-node":
            cmd += ["--devices", "4"]
        cmd += args
        if smoke:
            cmd += _SMOKE[PROBLEMS[problem].units]
        jobs.append(" ".join(cmd))
    return jobs

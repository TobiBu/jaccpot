"""The accuracy verification suite's command line (multi-node plan A4).

Subcommands::

    ic PROFILE --n N --seed S --out FILE          an equilibrium sphere (plummer, hernquist, nfw)
    run PROBLEM --out DIR [--tier T] [options]    one standard problem end to end -> DIR/result.json
    record RESULT BASELINE                        store a result as the baseline
    check RESULT BASELINE                         compare (exit 1 on FAIL)
    analyse-existing bar|merger DIR[:DIR...] ...  summaries of the 2026-10-31 runs' outputs
    ladder NAME [--tier T] [--list]               a convergence ladder's job commands

``run`` writes ``DIR/probe.jsonl`` (one row per edge: time, the runner's diagnostics,
the snapshot analysis when due, the reference-force score when due),
``DIR/config.json`` and ``DIR/result.json`` (the :mod:`jaccpot.verification.baseline`
schema). The force is scored against :func:`jaccpot.verification.reference_force` at
``--targets`` particle ids (seeded, the same on every device count) at the first edge
with a force, every ``--score-every`` segments and the last edge.

Precision axis (``--precision``): ``fp32`` (fp32 state and force, production),
``fp64state`` (fp64 state, fp32 force: plan U2 (b)), ``fp64`` (fp64 state and force:
the lane's validation mode), ``anchored`` (A3's fp64 anchor per leaf + fp32 offsets;
:data:`PRECISIONS` maps it to the runner setting once A3 lands).

``--perturb dt=4|theta=0.95|tol=1e-2`` changes the run WITHOUT changing the recorded
comparable config (it is recorded under ``perturbation``): a stand-in for a code
regression, to show that ``check`` catches it (gate 3).

docs/accuracy_suite.md has the schema, the tolerances and the HoreKa tiers.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import socket
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.accuracy import analysis, problems  # noqa: E402
from bench.accuracy.ics import load_ic, sample_sphere, save_ic, subsample  # noqa: E402

#: ``--precision`` -> (state dtype, ``GalaxySettings.precision``, extra settings)
PRECISIONS: dict = {
    "fp32": ("float32", "fp32", {}),
    "fp64state": ("float64", "fp32", {}),
    "fp64": ("float64", "fp64", {}),
    # A3 (wave 5): the fp64 anchor per leaf + fp32 offsets. Fill in its runner
    # setting here when it lands (e.g. {"position_anchors": True}); until then the
    # ladders' "anchored" rung is refused with this message.
    "anchored": None,
}


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _git_head() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------------
# Runner construction.
# ----------------------------------------------------------------------


def build_runner(
    cfg: dict, run: dict, mesh: Any = None, callbacks: tuple = ()
) -> tuple:
    """The solver and :class:`~jaccpot.production.ProductionRunner` for a config.

    Parameters
    ----------
    cfg : dict
        The recorded config (``mac``, ``theta``, ``tol``, ``precision``, ...).
    run : dict
        The values actually run (``cfg`` with a perturbation applied).
    mesh : Any
        A 1-D mesh for the multi-GPU mode, or ``None`` (one card).
    callbacks : tuple
        Edge callbacks.

    Returns
    -------
    tuple
        ``(solver, runner, state_dtype)``.

    Raises
    ------
    NotImplementedError
        For the ``anchored`` precision before A3 lands.
    """
    import jax

    from jaccpot import FastMultipoleMethod
    from jaccpot.production import (
        GALAXY_SETTINGS,
        GalaxySettings,
        MeshRunConfig,
        ProductionRunner,
        SegmentConfig,
    )

    prec = PRECISIONS[run["precision"]]
    if prec is None:
        raise NotImplementedError(
            "precision 'anchored' is A3's anchored state: set its runner setting in "
            "bench/accuracy/suite.py PRECISIONS once A3 lands"
        )
    state_dtype, working, extra = prec
    on_cpu = jax.default_backend() == "cpu"
    base = FastMultipoleMethod(preset="galaxy").advanced
    if on_cpu:  # the fused lane's CPU route (FUP, opt-in)
        base = dataclasses.replace(
            base, runtime=dataclasses.replace(base.runtime, cpu_fused_lane=True)
        )
    common = dict(
        preset="galaxy",
        G=1.0,
        softening=float(run["softening"]),
        softening_kernel=run["kernel"],
        precision=working,
    )
    level = run["cell_min_level"]
    if run["mac"] == "dehnen_error":
        if on_cpu:
            common["use_pallas"] = True
        solver = FastMultipoleMethod(
            adaptive_eps=float(run["tol"]), advanced=base, **common
        )
        settings = dataclasses.replace(
            GALAXY_SETTINGS,
            adaptive_eps=float(run["tol"]),
            precision=working,
            softening_kernel=run["kernel"],
            cell_min_level=level,
            **extra,
        )
    else:
        solver = FastMultipoleMethod(
            theta=float(run["theta"]),
            advanced=dataclasses.replace(base, mac_type="dehnen"),
            **common,
        )
        settings = GalaxySettings(
            mac_type="dehnen",
            adaptive_eps=None,
            theta=float(run["theta"]),
            precision=working,
            softening_kernel=run["kernel"],
            cell_min_level=level,
            **extra,
        )
    segment = SegmentConfig(
        steps=int(run["steps"]),
        diagnostics_every=1,
        diagnostics_every_energy=int(run["energy_every"]),
        donate_input=True,
    )
    runner = ProductionRunner(
        solver,
        settings=settings,
        segment=segment,
        callbacks=tuple(callbacks),
        mesh=None if mesh is None else MeshRunConfig(mesh),
    )
    return solver, runner, state_dtype


# ----------------------------------------------------------------------
# The edge callback.
# ----------------------------------------------------------------------


class EdgeProbe:
    """At every edge: diagnostics; when due: the snapshot analysis and the force score.

    Parameters
    ----------
    out : str
        Run directory (``probe.jsonl``).
    problem : problems.Problem
        The problem.
    run : dict
        Resolved run values.
    component : np.ndarray
        ``[N]`` component labels (input order).
    names : list
        Component names.
    law : Any
        :class:`~jaccpot.verification.ForceLaw`.
    target_ids : np.ndarray
        The reference's targets.
    num_steps : int
        The run's steps (the last edge).
    mesh : Any
        The runner's mesh, or ``None``.
    """

    def __init__(
        self,
        out: str,
        problem: Any,
        run: dict,
        component: np.ndarray,
        names: list,
        law: Any,
        target_ids: np.ndarray,
        num_steps: int,
        mesh: Any,
    ) -> None:
        self.out, self.problem, self.run = out, problem, run
        self.component, self.names = component, [str(n) for n in names]
        self.law, self.tids, self.num_steps, self.mesh = (
            law,
            target_ids,
            num_steps,
            mesh,
        )
        self.rows: list = []
        self.scores: list = []
        self.disc = np.array(["disc" in n for n in self.names])[component]
        self.t_last = time.time()

    def _score(self, edge: Any) -> Optional[dict]:
        import jax.numpy as jnp

        from jaccpot.verification import gather_by_id, reference_force, score_force

        if self.mesh is None:
            acc = edge.self_acceleration
            if acc is None:
                return None
            n = int(edge.state.shape[0])
            ids = jnp.arange(n, dtype=jnp.int32)
            ref = reference_force(
                edge.state[:, 0], edge.masses, ids, self.tids, self.law
            )
            prod, _ = gather_by_id(acc, ids, self.tids)
        else:
            lay = edge.layout
            if lay is None or lay.acceleration is None:
                return None
            ref = reference_force(
                lay.state[:, 0],
                lay.masses,
                lay.ids,
                self.tids,
                self.law,
                mesh=self.mesh,
            )
            prod, _ = gather_by_id(lay.acceleration, lay.ids, self.tids, mesh=self.mesh)
        return score_force(prod, ref)

    def _host(self, edge: Any) -> tuple:
        if self.mesh is None:
            return np.asarray(edge.state_host()), np.asarray(edge.masses)
        got = edge.gather(("state", "masses"))
        return got["state"], got["masses"]

    def __call__(self, edge: Any) -> bool:
        t0 = time.time()
        seg = int(edge.segment)
        last = int(edge.step) >= self.num_steps
        row: dict = dict(
            step=int(edge.step),
            time=float(edge.time),
            segment=seg,
            retries=int(edge.retries),
            seg_wall_s=float(edge.wall_s),
            diagnostics=edge.diagnostics,
        )
        every = int(self.run["analyse_every"])
        if seg == 0 or last or (every and seg % every == 0):
            state, mass = self._host(edge)
            kind = self.problem.snapshot
            if kind == "sphere":
                row["sphere"] = analysis.sphere_snapshot(state, mass)
            elif kind == "disc":
                row["disc"] = analysis.disc_snapshot(
                    state, mass, self.disc, problems.DISC_ANNULI
                )
            elif kind == "bar":
                row["bar"] = analysis.bar_snapshot(state, mass, self.disc)
            else:
                row["centres"] = analysis.centres_snapshot(state, mass, self.component)[
                    "centres"
                ]
        score_every = int(self.run["score_every"])
        if seg == 1 or last or (score_every and seg % score_every == 0):
            score = self._score(edge)
            if score is not None:
                row["accuracy"] = score
                self.scores.append((float(edge.time), score))
        row["probe_s"] = time.time() - t0
        self.rows.append(row)
        with open(os.path.join(self.out, "probe.jsonl"), "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        d = edge.diagnostics or {}
        _log(
            f"edge {seg} step {edge.step} t {edge.time:.4f} wall {edge.wall_s:.1f}s "
            f"dE {d.get('energy_drift')} dL {d.get('angular_momentum_drift')} "
            f"dP {d.get('momentum_drift')}"
            + (f" rel_l2 {row['accuracy']['rel_l2']:.3e}" if "accuracy" in row else "")
        )
        return os.path.exists(os.path.join(self.out, "STOP"))


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------


def _ic_for(problem: Any, run: dict, ic_path: Optional[str], out: str) -> dict:
    if problem.ic.startswith("sphere:"):
        profile = problem.ic.split(":", 1)[1]
        if ic_path:
            ic = load_ic(ic_path)
        else:
            ic = sample_sphere(profile, int(run["n"]), seed=int(run["ic_seed"]))
            ic["meta"] = json.loads(ic["meta"])
        return subsample(ic, int(run["n"]))
    if not ic_path:
        raise SystemExit(
            f"problem {problem.name!r} needs --ic (Odisseo tools/a4_galaxy_ics.py)"
        )
    return subsample(load_ic(ic_path), int(run["n"]))


def _perturbed(cfg: dict, spec: Optional[str]) -> dict:
    run = dict(cfg)
    if not spec:
        return run
    key, val = spec.split("=", 1)
    if key == "dt":
        run["dt"] = cfg["dt"] * float(val)
    elif key in ("theta", "tol"):
        run[key] = float(val)
        if key == "theta":
            run["mac"] = "dehnen"
    elif key == "mac":
        run["mac"] = val
    else:
        raise SystemExit(f"--perturb {spec!r}: one of dt=F, theta=F, tol=F, mac=M")
    return run


def cmd_run(a: argparse.Namespace) -> int:
    """``run``: one standard problem end to end.

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaccpot.verification import (
        ForceLaw,
        make_result,
        save_result,
        select_targets,
    )

    problem = problems.PROBLEMS[a.problem]
    os.makedirs(a.out, exist_ok=True)
    for name in ("probe.jsonl",):
        p = os.path.join(a.out, name)
        if os.path.exists(p):
            os.remove(p)
    over = dict(
        n=a.n,
        dt=a.dt,
        t_end=a.t_end,
        n_tdyn=a.n_tdyn,
        dt_per_tdyn=a.dt_per_tdyn,
        softening=a.softening,
        steps=a.steps,
        analyse_every=a.analyse_every,
        score_every=a.score_every,
        energy_every=a.energy_every,
    )
    run = problems.resolve(a.problem, a.tier, over)
    ic = _ic_for(problem, dict(run, ic_seed=a.ic_seed), a.ic, a.out)
    state0, mass = ic["state0"], ic["mass"]
    n = int(state0.shape[0])
    if problem.units == "nbody":
        x = state0[:, 0]
        r = np.linalg.norm(x - np.sum(mass[:, None] * x, 0) / mass.sum(), axis=1)
        order = np.argsort(r)
        r50 = float(np.interp(0.5, np.cumsum(mass[order]) / mass.sum(), r[order]))
        t_dyn = math.sqrt(r50**3 / float(mass.sum()))
        if a.dt is None:
            # t_dyn / 64, capped by the softening scale's own time: dt_eta sqrt(eps / a(eps))
            # with a(eps) = G M(<eps) / eps^2 of the profile (the Hernquist cusp needs it:
            # at t_dyn / 64 alone its energy drifted 1.9e-2 over 100 t_dyn at N = 8192)
            from bench.accuracy.ics import PROFILES

            eps = float(run["softening"])
            prof_mass = PROFILES[problem.ic.split(":", 1)[1]]()[1]
            a_eps = float(prof_mass(np.array([eps]))[0]) / eps**2
            run["dt"] = min(
                float(run["dt_per_tdyn"]) * t_dyn,
                float(run["dt_eta"]) * math.sqrt(eps / a_eps),
            )
        if a.t_end is None:
            run["t_end"] = float(run["n_tdyn"]) * t_dyn
    else:
        t_dyn = None
    steps = int(run["steps"])
    # whole multiples of the energy cadence, so the last edge carries the energy
    block = steps * max(1, int(run["energy_every"]))
    num_steps = max(
        block, int(math.ceil(float(run["t_end"]) / float(run["dt"]) / block)) * block
    )
    level = a.cell_min_level
    if level is None:
        level = "auto" if n < 1_000_000 else 8
    elif level != "auto":
        level = int(level)
    cfg = dict(
        problem=a.problem,
        problem_ic=problem.ic,
        tier=a.tier,
        num_particles=n,
        ic=os.path.abspath(a.ic) if a.ic else None,
        ic_seed=int(a.ic_seed),
        ic_meta=ic["meta"],
        dt=float(run["dt"]),
        t_end=float(run["t_end"]),
        num_steps=num_steps,
        t_dyn=t_dyn,
        mac=a.mac,
        theta=float(a.theta) if a.mac == "dehnen" else None,
        tol=float(a.tol) if a.mac == "dehnen_error" else None,
        precision=a.precision,
        softening=float(run["softening"]),
        kernel=a.kernel,
        cell_min_level=level,
        steps=steps,
        analyse_every=int(run["analyse_every"]),
        score_every=int(run["score_every"]),
        energy_every=int(run["energy_every"]),
        targets=int(min(a.targets, n)),
        target_seed=int(a.target_seed),
        devices=int(a.devices),
        perturbation=a.perturb,
    )
    exec_cfg = _perturbed(cfg, a.perturb)
    if exec_cfg["mac"] == "dehnen" and exec_cfg.get("theta") is None:
        exec_cfg["theta"] = 0.8
    if exec_cfg["mac"] == "dehnen_error" and exec_cfg.get("tol") is None:
        exec_cfg["tol"] = 1e-4
    # a perturbed dt still covers t_end
    num_steps = max(
        block,
        int(math.ceil(float(run["t_end"]) / float(exec_cfg["dt"]) / block)) * block,
    )
    exec_cfg["num_steps"] = num_steps
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(dict(cfg, executed=exec_cfg), fh, indent=1, default=str)

    mesh = None
    if a.devices > 1:
        from jax.sharding import Mesh

        mesh = Mesh(np.asarray(jax.local_devices()[: a.devices]), ("gpus",))
    law = ForceLaw(G=1.0, softening=float(run["softening"]), kernel=a.kernel)
    tids = select_targets(n, cfg["targets"], seed=cfg["target_seed"])
    probe = EdgeProbe(
        a.out,
        problem,
        run,
        ic["component"],
        list(ic["component_names"]),
        law,
        tids,
        num_steps,
        mesh,
    )
    if exec_cfg["precision"] == "anchored" and PRECISIONS["anchored"] is None:
        _log("precision 'anchored' (A3) is not available yet: rung skipped")
        return 3
    solver, runner, state_dtype = build_runner(cfg, exec_cfg, mesh, callbacks=(probe,))
    if a.dry_run:
        _log(
            f"dry run: {a.problem} N={n:,} num_steps={num_steps} config OK, runner built"
        )
        return 0
    import jax.numpy as jnp

    st = jnp.asarray(np.asarray(state0, state_dtype))
    ms = jnp.asarray(np.asarray(mass, state_dtype))
    _log(
        f"{a.problem} N={n:,} dt={exec_cfg['dt']:.4g} steps={num_steps} mac={exec_cfg['mac']} "
        f"prec={a.precision} devices={jax.devices()[: max(1, a.devices)]}"
    )
    t0 = time.time()
    res = runner.run(st, ms, dt=float(exec_cfg["dt"]), num_steps=num_steps)
    wall = time.time() - t0
    _log(f"done step {res.step} t {res.time:.4f} wall {wall:.0f} s")

    rows = probe.rows
    diag = [{"time": r["time"], "diagnostics": r["diagnostics"]} for r in rows]
    metrics = analysis.conservation_metrics(diag)
    series: dict = {
        "t": [r["time"] for r in rows],
        "energy_drift": [(r["diagnostics"] or {}).get("energy_drift") for r in rows],
        "angular_momentum_drift": [
            (r["diagnostics"] or {}).get("angular_momentum_drift") for r in rows
        ],
        "momentum_drift": [
            (r["diagnostics"] or {}).get("momentum_drift") for r in rows
        ],
        "force": [
            [t, s["rel_l2"], s["p50"], s["p99"], s["max"]] for t, s in probe.scores
        ],
    }
    kind = problem.snapshot
    if kind == "sphere":
        metrics.update(analysis.sphere_summary(rows))
        snaps = [r for r in rows if r.get("sphere")]
        series["lagr"] = [[r["time"]] + r["sphere"]["lagr"] for r in snaps]
    elif kind == "disc":
        metrics.update(analysis.heating_summary(rows, problems.DISC_ANNULI))
        snaps = [r for r in rows if r.get("disc")]
        series["sigz2"] = [[r["time"]] + r["disc"]["sigz2"] for r in snaps]
        series["sigR2"] = [[r["time"]] + r["disc"]["sigR2"] for r in snaps]
    elif kind == "bar":
        metrics.update(analysis.bar_summary(rows))
        snaps = [r for r in rows if r.get("bar")]
        series["A2"] = [[r["time"], r["bar"]["A2"]] for r in snaps]
    else:
        pair = _bulge_pair(list(ic["component_names"]))
        ref = problems.reference_or_none(a.reference)
        metrics.update(analysis.merger_summary(rows, ref, pair))
        t, sep = analysis.separation_series(rows, pair)
        series["sep_t"], series["sep_kpc"] = t.tolist(), sep.tolist()
    if probe.scores:
        first, lastscore = probe.scores[0][1], probe.scores[-1][1]
        for tag, s in (("start", first), ("end", lastscore)):
            for k in ("rel_l2", "p50", "p99", "max"):
                metrics[f"force.{k}_{tag}"] = s[k]
    tol = problems.tolerances(a.problem, a.tier, a.mac, metrics)
    prov = dict(
        commit=_git_head(),
        host=socket.gethostname(),
        platform=jax.default_backend(),
        devices=[str(d) for d in jax.devices()[: max(1, a.devices)]],
        processes=jax.process_count(),
        jax=jax.__version__,
        date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        wall_s=wall,
        segments=int(res.segment),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        xla_flags=os.environ.get("XLA_FLAGS"),
    )
    result = make_result(
        a.problem,
        cfg,
        metrics,
        tier=a.tier,
        comparable=problems.comparable(a.problem),
        tolerances=tol,
        series=series,
        provenance=prov,
    )
    path = save_result(result, os.path.join(a.out, "result.json"))
    _log(f"result -> {path}")
    for k, v in sorted(metrics.items()):
        print(f"  {k:32s} {v}")
    return 0


def _bulge_pair(names: list) -> tuple:
    bulges = [i for i, nm in enumerate(names) if "bulge" in str(nm)]
    if len(bulges) >= 2:
        return bulges[0], bulges[1]
    discs = [i for i, nm in enumerate(names) if "disc" in str(nm)]
    return (discs[0], discs[1]) if len(discs) >= 2 else (0, 1)


# ----------------------------------------------------------------------
# record / check / ic / analyse-existing / ladder
# ----------------------------------------------------------------------


def cmd_record(a: argparse.Namespace) -> int:
    """``record``: store a result as the baseline (refuses to overwrite without --force).

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    from jaccpot.verification import load_result, save_result

    if os.path.exists(a.baseline) and not a.force:
        print(
            f"{a.baseline} exists: pass --force to replace the baseline",
            file=sys.stderr,
        )
        return 2
    res = load_result(a.result)
    res["provenance"]["recorded_from"] = os.path.abspath(a.result)
    save_result(res, a.baseline)
    print(f"baseline {res['problem']} ({res['tier']}) -> {a.baseline}")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    """``check``: compare a result against a baseline.

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        0 on PASS, 1 on FAIL.
    """
    from jaccpot.verification import check_result, load_result

    out = check_result(load_result(a.baseline), load_result(a.result))
    print(out.table())
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(
                dict(
                    passed=out.passed,
                    rows=out.rows,
                    mismatched_config={
                        k: list(v) for k, v in out.mismatched_config.items()
                    },
                    missing=out.missing,
                ),
                fh,
                indent=1,
            )
    return 0 if out.passed else 1


def cmd_ic(a: argparse.Namespace) -> int:
    """``ic``: write an equilibrium sphere.

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    save_ic(a.out, sample_sphere(a.profile, a.n, seed=a.seed))
    print(a.out)
    return 0


def cmd_analyse_existing(a: argparse.Namespace) -> int:
    """``analyse-existing``: summaries of existing runs (``probe.jsonl`` + ``diag.jsonl``).

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    out = {}
    for spec in a.runs:
        label, _, dirs = spec.rpartition("=") if "=" in spec else (spec, "", spec)
        out[label] = analysis.analyse_existing(a.kind, dirs.split(":"), a.reference)
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(text)
    return 0


def cmd_ladder(a: argparse.Namespace) -> int:
    """``ladder``: print a ladder's ``run`` commands, one per line.

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    from bench.accuracy.ladders import ladder_jobs

    jobs = ladder_jobs(a.name, a.tier, a.root, ic_dir=a.ic_dir)
    if a.index is not None:
        print(jobs[a.index])
    else:
        for j in jobs:
            print(j)
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    """``summarize``: one ladder's rungs in a table (and the merger orbit vs the reference).

    Parameters
    ----------
    a : argparse.Namespace
        Arguments.

    Returns
    -------
    int
        Exit code.
    """
    from bench.accuracy.ladders import ladder_names, ladder_spec
    from jaccpot.verification import load_result

    names = ladder_names(a.tier) if a.name == "all" else [a.name]
    table: dict = {}
    for name in names:
        problem, _ic, rungs, ref_rung = ladder_spec(name, a.tier)
        rows = {}
        ref = None
        if ref_rung:
            path = os.path.join(a.root, name, ref_rung, "result.json")
            if os.path.exists(path):
                ref = load_result(path)
        for rung, _args in rungs:
            path = os.path.join(a.root, name, rung, "result.json")
            if not os.path.exists(path):
                rows[rung] = None
                continue
            res = load_result(path)
            m = dict(res["metrics"])
            if (
                ref is not None
                and "sep_t" in res["series"]
                and "sep_t" in ref["series"]
            ):
                t, sep = np.asarray(res["series"]["sep_t"]), np.asarray(
                    res["series"]["sep_kpc"]
                )
                rt, rs = np.asarray(ref["series"]["sep_t"]), np.asarray(
                    ref["series"]["sep_kpc"]
                )
                common = (t >= rt[0]) & (t <= rt[-1])
                if common.any():
                    m["merger.sep_max_dev_kpc_vs_ref"] = float(
                        np.max(np.abs(sep[common] - np.interp(t[common], rt, rs)))
                    )
            if ref is not None:
                for k in (
                    "force.rel_l2_end",
                    "energy.max_abs_dE",
                    "heat.sigz2_growth_1",
                    "lagr.max_drift_r50",
                    "merger.peri1_t",
                ):
                    if m.get(k) is not None and ref["metrics"].get(k) is not None:
                        m[f"{k}_minus_ref"] = m[k] - ref["metrics"][k]
            m["wall_s"] = res["provenance"].get("wall_s")
            rows[rung] = m
        table[name] = {"problem": problem, "reference": ref_rung, "rungs": rows}
    text = json.dumps(table, indent=1, default=str)
    if a.json:
        with open(a.json, "w") as fh:
            fh.write(text)
    keys = a.keys.split(",")
    for name, entry in table.items():
        print(f"## {name} ({entry['problem']}; reference {entry['reference']})")
        print("| rung | " + " | ".join(keys) + " |")
        print("|" + " --- |" * (len(keys) + 1))
        for rung, m in entry["rungs"].items():
            vals = (
                ["missing"] * len(keys)
                if m is None
                else [
                    (
                        "-"
                        if m.get(k) is None
                        else (f"{m[k]:.4g}" if isinstance(m[k], float) else str(m[k]))
                    )
                    for k in keys
                ]
            )
            print(f"| {rung} | " + " | ".join(vals) + " |")
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser (every subcommand).

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one standard problem end to end")
    r.add_argument("problem", choices=sorted(problems.PROBLEMS))
    r.add_argument("--out", required=True)
    r.add_argument(
        "--tier", default="local", choices=("local", "horeka-1card", "horeka-node")
    )
    r.add_argument(
        "--ic", default=None, help="IC file (galaxy problems; optional for spheres)"
    )
    r.add_argument("--ic-seed", type=int, default=1)
    r.add_argument(
        "--n",
        type=int,
        default=None,
        help="particles (a per-component prefix of the IC)",
    )
    r.add_argument("--dt", type=float, default=None)
    r.add_argument("--t-end", type=float, default=None)
    r.add_argument(
        "--n-tdyn", type=float, default=None, help="spheres: duration in t_dyn"
    )
    r.add_argument(
        "--dt-per-tdyn",
        type=float,
        default=None,
        help="spheres: dt in t_dyn (default 1/64)",
    )
    r.add_argument("--softening", type=float, default=None)
    r.add_argument(
        "--kernel", default="ferrers3", choices=("ferrers3", "wendland_c2", "plummer")
    )
    r.add_argument("--mac", default="dehnen_error", choices=("dehnen_error", "dehnen"))
    r.add_argument("--theta", type=float, default=0.8)
    r.add_argument(
        "--tol", type=float, default=1e-4, help="dehnen_error's adaptive_eps"
    )
    r.add_argument("--precision", default="fp32", choices=sorted(PRECISIONS))
    r.add_argument(
        "--cell-min-level",
        default=None,
        help="'auto' or an int (default: auto below 1e6)",
    )
    r.add_argument(
        "--steps", type=int, default=None, help="steps per segment (one edge each)"
    )
    r.add_argument("--analyse-every", type=int, default=None)
    r.add_argument("--score-every", type=int, default=None)
    r.add_argument("--energy-every", type=int, default=None)
    r.add_argument("--targets", type=int, default=2048)
    r.add_argument("--target-seed", type=int, default=0)
    r.add_argument(
        "--devices",
        type=int,
        default=1,
        help="> 1: the runner's multi-GPU mode on this host",
    )
    r.add_argument(
        "--perturb",
        default=None,
        help="dt=F | theta=F | tol=F | mac=M (not recorded as config)",
    )
    r.add_argument(
        "--reference", default=None, help="merger: a converged run's result.json"
    )
    r.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the config and build the runner; no steps",
    )
    r.set_defaults(fn=cmd_run)

    rec = sub.add_parser("record", help="store a result as the baseline")
    rec.add_argument("result")
    rec.add_argument("baseline")
    rec.add_argument("--force", action="store_true")
    rec.set_defaults(fn=cmd_record)

    chk = sub.add_parser(
        "check", help="compare a result against a baseline (exit 1 on FAIL)"
    )
    chk.add_argument("result")
    chk.add_argument("baseline")
    chk.add_argument("--json", default=None)
    chk.set_defaults(fn=cmd_check)

    icp = sub.add_parser("ic", help="an equilibrium sphere IC")
    icp.add_argument("profile", choices=("plummer", "hernquist", "nfw"))
    icp.add_argument("--n", type=int, required=True)
    icp.add_argument("--seed", type=int, default=1)
    icp.add_argument("--out", required=True)
    icp.set_defaults(fn=cmd_ic)

    ae = sub.add_parser(
        "analyse-existing", help="summaries of existing bar/merger runs"
    )
    ae.add_argument("kind", choices=("bar", "merger"))
    ae.add_argument("runs", nargs="+", help="[LABEL=]DIR[:DIR...]")
    ae.add_argument("--reference", default=None, help="merger: reference DIR[:DIR...]")
    ae.add_argument("--json", default=None)
    ae.set_defaults(fn=cmd_analyse_existing)

    lad = sub.add_parser("ladder", help="a convergence ladder's run commands")
    lad.add_argument("name")
    lad.add_argument("--tier", default="horeka-1card")
    lad.add_argument("--root", default="$A4_OUT")
    lad.add_argument("--ic-dir", default="$A4_IC")
    lad.add_argument("--index", type=int, default=None)
    lad.set_defaults(fn=cmd_ladder)

    sm = sub.add_parser("summarize", help="tabulate a ladder's results (or 'all')")
    sm.add_argument("name")
    sm.add_argument("--tier", default="horeka-1card")
    sm.add_argument("--root", required=True)
    sm.add_argument("--json", default=None)
    sm.add_argument(
        "--keys",
        default="force.rel_l2_end,force.p50_end,force.p99_end,energy.max_abs_dE,angmom.max_dL,"
        "momentum.max_dP,wall_s",
    )
    sm.set_defaults(fn=cmd_summarize)
    return p


def main(argv: Optional[list] = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : Optional[list]
        Arguments (default ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code.
    """
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())

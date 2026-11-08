"""Per-snapshot analyses and per-run summaries of the accuracy suite's standard problems.

Snapshots are host arrays (``state [N, 2, 3]``, ``mass [N]``, ``component [N]``); a
snapshot analysis returns a JSON-able dict (one row of the run's ``probe.jsonl``),
and a summary turns the rows (and the runner's diagnostics) into flat metrics.

* spheres: Lagrangian radii and the velocity dispersion between them;
* discs: sigma_z^2 and sigma_R^2 in annuli (the heating curve);
* the bar: ``run_bar.py``'s probe (2026-10-31): A2(R), m=2 phase, z_rms in 0.4 kpc
  annuli about the disc's centre of mass, and ``bar_summary.py``'s summary plus the
  buckling time (the time of the steepest rise of the inner disc's thickness, a
  1.0-wide sliding linear fit) -- so it reads the 2026-10-31 runs' ``probe.jsonl``
  unchanged (``suite.py analyse-existing``);
* the merger: per-component centres (shrinking spheres), the bulge separation, its
  pericentres and apocentres; against a converged reference run: the separation's
  largest deviation.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

#: galaxy code units (10 kpc, 1e10 Msun, G = 1)
KPC = 10.0
MYR = 149.1
KMS = 65.58
#: the bar probe's annuli (run_bar.py)
BAR_DR = 0.04
BAR_NR = 30
#: Lagrangian mass fractions
FRACTIONS = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)


# ----------------------------------------------------------------------
# Snapshot analyses.
# ----------------------------------------------------------------------


def _com(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.sum(m[:, None] * x, axis=0) / np.sum(m)


def shrinking_centre(
    x: np.ndarray, m: np.ndarray, frac: float = 0.05, shrink: float = 0.8
) -> np.ndarray:
    """Power's shrinking-sphere centre.

    Parameters
    ----------
    x : np.ndarray
        ``[n, 3]``.
    m : np.ndarray
        ``[n]``.
    frac : float
        Stop when fewer than this fraction (or 100 particles) remain.
    shrink : float
        Radius factor per iteration.

    Returns
    -------
    np.ndarray
        ``[3]``.
    """
    c = _com(x, m)
    r = np.sqrt(np.max(np.sum((x - c) ** 2, axis=1)))
    sel = np.ones(x.shape[0], bool)
    stop = max(100, int(frac * x.shape[0]))
    while sel.sum() > stop:
        d2 = np.sum((x - c) ** 2, axis=1)
        r *= shrink
        new = d2 < r * r
        if new.sum() < 10:
            break
        sel = new
        c = _com(x[sel], m[sel])
    return c


def sphere_snapshot(state: np.ndarray, mass: np.ndarray) -> dict:
    """Lagrangian radii and sigma between them (about the centre of mass).

    Parameters
    ----------
    state : np.ndarray
        ``[N, 2, 3]``.
    mass : np.ndarray
        ``[N]``.

    Returns
    -------
    dict
        ``lagr`` (radii at :data:`FRACTIONS`), ``sigma`` (1-D dispersion in the
        shells between consecutive fractions, the innermost from 0).
    """
    x = np.asarray(state[:, 0], np.float64)
    v = np.asarray(state[:, 1], np.float64)
    m = np.asarray(mass, np.float64)
    c, vc = _com(x, m), _com(v, m)
    r = np.linalg.norm(x - c, axis=1)
    order = np.argsort(r)
    cm = np.cumsum(m[order]) / np.sum(m)
    radii = np.interp(FRACTIONS, cm, r[order])
    edges = np.concatenate([[0.0], radii])
    sig = []
    dv2 = np.sum((v - vc) ** 2, axis=1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (r >= lo) & (r < hi)
        sig.append(
            float(np.sqrt(np.sum(m[s] * dv2[s]) / max(np.sum(m[s]), 1e-300) / 3.0))
        )
    return {"lagr": radii.tolist(), "sigma": sig}


def disc_snapshot(
    state: np.ndarray, mass: np.ndarray, sel: np.ndarray, annuli: tuple
) -> dict:
    """sigma_z^2, sigma_R^2, z_rms and the mass in annuli of the selected (disc) rows.

    Parameters
    ----------
    state : np.ndarray
        ``[N, 2, 3]``.
    mass : np.ndarray
        ``[N]``.
    sel : np.ndarray
        Boolean disc rows.
    annuli : tuple
        ``(R_lo, R_hi)`` pairs (code units).

    Returns
    -------
    dict
        Lists per annulus: ``sigz2``, ``sigR2``, ``zrms``, ``mass``.
    """
    x = np.asarray(state[sel, 0], np.float64)
    v = np.asarray(state[sel, 1], np.float64)
    m = np.asarray(mass[sel], np.float64)
    c = shrinking_centre(x, m)
    near = np.linalg.norm(x - c, axis=1) < 0.5
    vc = _com(v[near], m[near]) if near.sum() > 10 else _com(v, m)
    x, v = x - c, v - vc
    R = np.hypot(x[:, 0], x[:, 1])
    cphi, sphi = x[:, 0] / np.maximum(R, 1e-12), x[:, 1] / np.maximum(R, 1e-12)
    vR = v[:, 0] * cphi + v[:, 1] * sphi
    out: dict = {"sigz2": [], "sigR2": [], "zrms": [], "mass": []}
    for lo, hi in annuli:
        s = (R >= lo) & (R < hi)
        w = m[s]
        mw = max(np.sum(w), 1e-300)
        vz = v[s, 2] - np.sum(w * v[s, 2]) / mw
        vr = vR[s] - np.sum(w * vR[s]) / mw
        out["sigz2"].append(float(np.sum(w * vz * vz) / mw))
        out["sigR2"].append(float(np.sum(w * vr * vr) / mw))
        out["zrms"].append(float(np.sqrt(np.sum(w * x[s, 2] ** 2) / mw)))
        out["mass"].append(float(np.sum(w)))
    return out


def bar_snapshot(state: np.ndarray, mass: np.ndarray, sel: np.ndarray) -> dict:
    """``run_bar.py``'s bar probe on the host (disc centre of mass, 0.4 kpc annuli).

    Parameters
    ----------
    state : np.ndarray
        ``[N, 2, 3]``.
    mass : np.ndarray
        ``[N]``.
    sel : np.ndarray
        Boolean disc rows.

    Returns
    -------
    dict
        ``A2``, ``phase2``, ``A2_R``, ``phase2_R``, ``buckle_R``, ``zrms_R``,
        ``vphi_R``, ``sigR_R``, ``mass_R`` (the 2026-10-31 probe's keys).
    """
    s = np.asarray(state[sel], np.float64)
    m = np.asarray(mass[sel], np.float64)
    cen, vcen = _com(s[:, 0], m), _com(s[:, 1], m)
    x, v = s[:, 0] - cen, s[:, 1] - vcen
    r = np.hypot(x[:, 0], x[:, 1])
    cphi, sphi = x[:, 0] / np.maximum(r, 1e-12), x[:, 1] / np.maximum(r, 1e-12)
    c2, s2 = cphi * cphi - sphi * sphi, 2 * cphi * sphi
    vr = v[:, 0] * cphi + v[:, 1] * sphi
    vphi = -v[:, 0] * sphi + v[:, 1] * cphi
    z = x[:, 2]
    k = np.floor(r / BAR_DR).astype(np.int64)
    k = np.where(k < BAR_NR, k, BAR_NR)
    cols = np.stack(
        [m, m * c2, m * s2, m * z * c2, m * z * s2, m * z * z, m * vphi, m * vr * vr], 1
    )
    b = np.zeros((BAR_NR + 1, 8))
    np.add.at(b, k, cols)
    b = b[:BAR_NR]
    mm = np.maximum(b[:, 0], 1e-300)
    inner = slice(0, int(round(0.6 / BAR_DR)))
    a2c, a2s = b[inner, 1].sum(), b[inner, 2].sum()
    return dict(
        A2=float(np.hypot(a2c, a2s) / b[inner, 0].sum()),
        phase2=float(0.5 * np.arctan2(a2s, a2c)),
        A2_R=(np.hypot(b[:, 1], b[:, 2]) / mm).tolist(),
        phase2_R=(0.5 * np.arctan2(b[:, 2], b[:, 1])).tolist(),
        buckle_R=(np.hypot(b[:, 3], b[:, 4]) / mm).tolist(),
        zrms_R=np.sqrt(b[:, 5] / mm).tolist(),
        vphi_R=(b[:, 6] / mm).tolist(),
        sigR_R=np.sqrt(b[:, 7] / mm).tolist(),
        mass_R=b[:, 0].tolist(),
    )


def centres_snapshot(
    state: np.ndarray, mass: np.ndarray, component: np.ndarray
) -> dict:
    """Shrinking-sphere centre of every component.

    Parameters
    ----------
    state : np.ndarray
        ``[N, 2, 3]``.
    mass : np.ndarray
        ``[N]``.
    component : np.ndarray
        ``[N]`` labels.

    Returns
    -------
    dict
        ``centres``: ``[n_components, 3]``.
    """
    x = np.asarray(state[:, 0], np.float64)
    m = np.asarray(mass, np.float64)
    out = []
    for c in range(int(component.max()) + 1):
        s = component == c
        out.append(shrinking_centre(x[s], m[s]).tolist() if s.any() else [np.nan] * 3)
    return {"centres": out}


# ----------------------------------------------------------------------
# Summaries.
# ----------------------------------------------------------------------


def sliding_slope(t: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    """``dy/dt`` by a linear fit in a sliding window of ``width``.

    Parameters
    ----------
    t : np.ndarray
        Times.
    y : np.ndarray
        Values.
    width : float
        Window (time units).

    Returns
    -------
    np.ndarray
        The slope (NaN where the window holds <= 3 points).
    """
    out = np.full(t.shape, np.nan)
    for k in range(t.shape[0]):
        s = np.abs(t - t[k]) <= width / 2
        if s.sum() > 3:
            out[k] = np.polyfit(t[s], y[s], 1)[0]
    return out


def conservation_metrics(diag: list) -> dict:
    """Max drifts from the runner's diagnostics rows (``SegmentEdge.diagnostics``).

    Parameters
    ----------
    diag : list
        ``{"time", "diagnostics"}`` rows.

    Returns
    -------
    dict
        ``energy.max_abs_dE``, ``energy.end_dE``, ``angmom.max_dL``,
        ``momentum.max_dP`` (``None`` where not reported).
    """

    def col(key: str) -> np.ndarray:
        vals = [(d.get("diagnostics") or {}).get(key) for d in diag]
        return np.array([np.nan if v is None else float(v) for v in vals])

    de, dl, dp = (
        col("energy_drift"),
        col("angular_momentum_drift"),
        col("momentum_drift"),
    )

    def mx(a: np.ndarray) -> Optional[float]:
        a = a[np.isfinite(a)]
        return float(np.max(np.abs(a))) if a.size else None

    end = de[np.isfinite(de)]
    return {
        "energy.max_abs_dE": mx(de),
        "energy.end_dE": float(end[-1]) if end.size else None,
        "angmom.max_dL": mx(dl),
        "momentum.max_dP": mx(dp),
    }


def sphere_summary(rows: list) -> dict:
    """Lagrangian-radius and dispersion drift of an equilibrium sphere.

    Parameters
    ----------
    rows : list
        Probe rows with ``sphere`` entries.

    Returns
    -------
    dict
        ``lagr.max_drift_r10/r50/r90`` (max ``|r(t)/r(0) - 1|``) and
        ``sigma.max_drift`` (shells 10-90 %).
    """
    rows = [r for r in rows if r.get("sphere")]
    lagr = np.array([r["sphere"]["lagr"] for r in rows])
    sig = np.array([r["sphere"]["sigma"] for r in rows])
    rel = np.abs(lagr / lagr[0] - 1.0)
    out = {}
    for name, f in (("r10", 0.1), ("r50", 0.5), ("r90", 0.9)):
        out[f"lagr.max_drift_{name}"] = float(np.max(rel[:, FRACTIONS.index(f)]))
    bulk = slice(FRACTIONS.index(0.1) + 1, FRACTIONS.index(0.9) + 1)
    out["sigma.max_drift"] = float(np.max(np.abs(sig[:, bulk] / sig[0, bulk] - 1.0)))
    return out


def heating_summary(rows: list, annuli: tuple) -> dict:
    """The heating curve: fractional sigma^2 growth and its rate per annulus.

    Parameters
    ----------
    rows : list
        Probe rows with ``disc`` entries.
    annuli : tuple
        The annuli (for names only).

    Returns
    -------
    dict
        Per annulus ``k``: ``heat.sigz2_growth_k`` (``sigma_z^2(end)/sigma_z^2(0) -
        1``), ``heat.sigz2_rate_k`` (least-squares ``d sigma_z^2/dt / sigma_z^2(0)``,
        per code time), the same for ``sigR2``; ``disc.zrms_growth_k``.
    """
    rows = [r for r in rows if r.get("disc")]
    t = np.array([r["time"] for r in rows])
    out = {}
    for key, name in (("sigz2", "sigz2"), ("sigR2", "sigR2")):
        s = np.array([r["disc"][key] for r in rows])
        for k in range(len(annuli)):
            out[f"heat.{name}_growth_{k}"] = float(s[-1, k] / s[0, k] - 1.0)
            rate = np.polyfit(t, s[:, k], 1)[0] if t.size > 1 else np.nan
            out[f"heat.{name}_rate_{k}"] = float(rate / s[0, k])
    z = np.array([r["disc"]["zrms"] for r in rows])
    for k in range(len(annuli)):
        out[f"disc.zrms_growth_{k}"] = float(z[-1, k] / z[0, k] - 1.0)
    return out


def bar_summary(rows: list, diag: Optional[list] = None) -> dict:
    """``bar_summary.py``'s numbers plus the buckling time.

    Parameters
    ----------
    rows : list
        Probe rows with ``bar`` entries (``time`` in code units).
    diag : Optional[list]
        Diagnostics rows (for ``dE`` / ``dL`` at the end).

    Returns
    -------
    dict
        ``bar.t_bar_myr`` (first ``max A2(R<5 kpc) > 0.2``), ``bar.t_peak_myr``,
        ``bar.a2max_peak``, ``bar.a2max_end``, ``bar.length_end_kpc``,
        ``bar.omega_p_end`` (km/s/kpc), ``bar.t_buckle_myr`` (max of the 1.0-wide
        sliding slope of the R < 2 kpc z_rms), ``bar.z_in_end_kpc``,
        ``bar.t_end_myr``; ``None`` where undefined.
    """
    rows = [r for r in rows if r.get("bar")]
    radii = (np.arange(BAR_NR) + 0.5) * BAR_DR * KPC
    t = np.array([r["time"] for r in rows])
    a2r = np.array([r["bar"]["A2_R"] for r in rows])
    ph = np.array([r["bar"]["phase2_R"] for r in rows])
    m = np.array([r["bar"]["mass_R"] for r in rows])
    zr = np.array([r["bar"]["zrms_R"] for r in rows]) * KPC
    inner = (radii > 1.0) & (radii < 3.0)
    w = m[:, inner] * a2r[:, inner]
    phase = 0.5 * np.unwrap(np.angle((w * np.exp(2j * ph[:, inner])).sum(1)))
    a2max = a2r[:, radii < 5.0].max(1)
    blen = []
    for k in range(t.shape[0]):
        d = np.abs(((ph[k] - phase[k]) + np.pi / 2) % np.pi - np.pi / 2)
        ok = (d < np.radians(10)) & (a2r[k] > 0.5 * a2max[k])
        j0 = int(np.argmax(np.where(radii < 5.0, a2r[k], -1.0)))
        j = j0
        while j < BAR_NR and ok[j]:
            j += 1
        blen.append(radii[j - 1] if j > j0 else 0.0)
    omega = sliding_slope(t, phase, 0.5) * KMS / KPC
    strong = a2max > 0.2
    z_in = zr[:, radii < 2.0].mean(1)
    slope = sliding_slope(t, z_in, 1.0)
    buckle = float(t[np.nanargmax(slope)] * MYR) if np.isfinite(slope).any() else None
    out = {
        "bar.t_end_myr": float(t[-1] * MYR),
        "bar.t_bar_myr": (
            float(t[int(np.argmax(strong))] * MYR) if strong.any() else None
        ),
        "bar.a2max_peak": float(a2max.max()),
        "bar.t_peak_myr": float(t[int(np.argmax(a2max))] * MYR),
        "bar.a2max_end": float(a2max[-1]),
        "bar.length_end_kpc": float(blen[-1]),
        "bar.omega_p_end": (
            float(np.nanmean(omega[-20:])) if strong[-20:].any() else None
        ),
        "bar.t_buckle_myr": buckle,
        "bar.z_in_end_kpc": float(z_in[-1]),
    }
    if diag:
        out.update(conservation_metrics(diag))
    return out


def _extrema(t: np.ndarray, d: np.ndarray, kind: str) -> list:
    """Local minima/maxima of ``d(t)``, refined by a parabola through 3 points.

    Parameters
    ----------
    t : np.ndarray
        Times.
    d : np.ndarray
        Values.
    kind : str
        ``"min"`` or ``"max"``.

    Returns
    -------
    list
        ``(t, d)`` pairs.
    """
    s = 1.0 if kind == "min" else -1.0
    y = s * d
    idx = np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1
    out = []
    for k in idx:
        t3, y3 = t[k - 1 : k + 2], y[k - 1 : k + 2]
        a, b, c = np.polyfit(t3 - t3[1], y3, 2)
        if a > 0:
            tv = -b / (2 * a)
            if abs(tv) <= (t3[2] - t3[0]) / 2:
                out.append((float(t3[1] + tv), float(s * (c - b * b / (4 * a)))))
                continue
        out.append((float(t[k]), float(d[k])))
    return out


def separation_series(rows: list, pair: tuple = (1, 4)) -> tuple:
    """``(t, |c_b - c_a|)`` in kpc from probe rows with ``centres``.

    Parameters
    ----------
    rows : list
        Probe rows.
    pair : tuple
        Component indices (the 2026-10-31 merger: 1 = A bulge, 4 = B bulge).

    Returns
    -------
    tuple
        Times (code units) and separations (kpc).
    """
    rows = [r for r in rows if r.get("centres")]
    t = np.array([r["time"] for r in rows])
    c = np.array([r["centres"] for r in rows])
    return t, np.linalg.norm(c[:, pair[1]] - c[:, pair[0]], axis=1) * KPC


def merger_summary(
    rows: list, reference: Optional[dict] = None, pair: tuple = (1, 4)
) -> dict:
    """Pericentres / apocentres of the bulge separation; deviation from a reference.

    Parameters
    ----------
    rows : list
        Probe rows with ``centres``.
    reference : Optional[dict]
        A converged run's ``series`` (``sep_t``, ``sep_kpc``), e.g. a result JSON's.
    pair : tuple
        The two components whose centres are compared.

    Returns
    -------
    dict
        ``merger.peri1_t``, ``merger.peri1_kpc`` (code time, kpc), ``apo1_*``,
        ``peri2_*`` when reached; with a reference: ``merger.sep_max_dev_kpc`` (largest
        ``|sep - sep_ref|`` over the common times, the reference interpolated) and
        ``merger.peri1_dt`` (``peri1_t - ref``).
    """
    t, sep = separation_series(rows, pair)
    mins, maxs = _extrema(t, sep, "min"), _extrema(t, sep, "max")
    out: dict = {"merger.t_end": float(t[-1]), "merger.sep_end_kpc": float(sep[-1])}
    for name, lst, k in (("peri1", mins, 0), ("apo1", maxs, 0), ("peri2", mins, 1)):
        out[f"merger.{name}_t"] = lst[k][0] if len(lst) > k else None
        out[f"merger.{name}_kpc"] = lst[k][1] if len(lst) > k else None
    if reference is not None:
        rt, rs = np.asarray(reference["sep_t"]), np.asarray(reference["sep_kpc"])
        common = (t >= rt[0]) & (t <= rt[-1])
        out["merger.sep_max_dev_kpc"] = (
            float(np.max(np.abs(sep[common] - np.interp(t[common], rt, rs))))
            if common.any()
            else None
        )
        rmins = _extrema(rt, rs, "min")
        out["merger.peri1_dt"] = mins[0][0] - rmins[0][0] if (mins and rmins) else None
    return out


# ----------------------------------------------------------------------
# The 2026-10-31 runs' outputs.
# ----------------------------------------------------------------------


def read_jsonl(path: str) -> list:
    """Rows of a ``.jsonl`` file (missing file: empty).

    Parameters
    ----------
    path : str
        File.

    Returns
    -------
    list
        Parsed rows.
    """
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def analyse_existing(
    kind: str, run_dirs: list, reference_dir: Optional[str] = None
) -> dict:
    """Summaries of existing run directories (``probe.jsonl`` + ``diag.jsonl``).

    Parameters
    ----------
    kind : str
        ``bar`` or ``merger``.
    run_dirs : list
        Run directories (several are concatenated in time order: a run continued
        in another directory, e.g. ``s1_main`` + ``s1_post``).
    reference_dir : Optional[str]
        ``merger``: the converged reference's directories, ``:``-separated.

    Returns
    -------
    dict
        The summary metrics.
    """
    rows, diag = [], []
    for d in run_dirs:
        rows += read_jsonl(os.path.join(d, "probe.jsonl"))
        diag += read_jsonl(os.path.join(d, "diag.jsonl"))
    rows.sort(key=lambda r: r.get("time", 0.0))
    if kind == "bar":
        return bar_summary(rows, diag)
    ref = None
    if reference_dir:
        rrows = []
        for d in reference_dir.split(":"):
            rrows += read_jsonl(os.path.join(d, "probe.jsonl"))
        rrows.sort(key=lambda r: r.get("time", 0.0))
        rt, rs = separation_series(rrows)
        ref = {"sep_t": rt, "sep_kpc": rs}
    out = merger_summary(rows, ref)
    out.update(conservation_metrics(diag))
    return out

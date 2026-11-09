"""Initial conditions for the accuracy suite.

* Equilibrium spheres (Plummer, Hernquist, truncated NFW): positions from the
  enclosed mass, isotropic velocities from the Eddington distribution function of the
  sphere's own potential (numerical inversion, numpy only). Units ``G = M = 1`` and
  scale radius 1 (the NFW's mass is the truncated halo's).
* Galaxies (disc + bulge + halo, the bar-unstable disc, the merger) come from agama in
  Odisseo's ``tools/a4_galaxy_ics.py`` (the odisseo env); they share this module's
  file format.

**File format** (``.npz``): ``state0`` ``(N, 2, 3)``, ``mass`` ``(N,)``, ``component``
``(N,) int8``, ``component_names`` (str array), ``meta`` (a JSON string: the
generator's parameters, units, seed). Every component block is shuffled, so a prefix
of each block is a fair subsample: :func:`subsample` takes the same fraction of every
component (nested ladders: a smaller rung is a prefix of the larger one), with the
masses scaled up so each component keeps its total mass.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np

#: per profile: (density(r), enclosed mass(r), potential(r)); scale radius 1, M = 1
#: (the NFW's ``M`` is its mass inside the exponential cutoff)


def _plummer() -> tuple:
    def rho(r: np.ndarray) -> np.ndarray:
        return 3.0 / (4.0 * np.pi) * (1.0 + r * r) ** -2.5

    def mass(r: np.ndarray) -> np.ndarray:
        return r**3 * (1.0 + r * r) ** -1.5

    return rho, mass, 100.0


def _hernquist() -> tuple:
    def rho(r: np.ndarray) -> np.ndarray:
        return 1.0 / (2.0 * np.pi) / (r * (1.0 + r) ** 3)

    def mass(r: np.ndarray) -> np.ndarray:
        return r * r / (1.0 + r) ** 2

    return rho, mass, 200.0


def _nfw(rcut: float = 10.0) -> tuple:
    """NFW with agama's cutoff ``exp(-(r/rcut)^2)``, normalised to unit total mass."""
    grid = np.logspace(-6, np.log10(6.0 * rcut), 20001)

    def raw(r: np.ndarray) -> np.ndarray:
        return 1.0 / (r * (1.0 + r) ** 2) * np.exp(-((r / rcut) ** 2))

    shell = 4.0 * np.pi * grid**2 * raw(grid)
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(grid))]
    )
    cum += 4.0 * np.pi * 0.5 * grid[0] ** 2  # the inner r^-1 cusp's mass inside grid[0]
    total = cum[-1]

    def rho(r: np.ndarray) -> np.ndarray:
        return raw(r) / total

    def mass(r: np.ndarray) -> np.ndarray:
        return np.interp(r, grid, cum / total, right=1.0)

    return rho, mass, 3.0 * rcut


PROFILES: dict = {"plummer": _plummer, "hernquist": _hernquist, "nfw": _nfw}


def _tables(rho: Callable, mass: Callable) -> tuple:
    """Radius grid, ``Psi = -Phi`` and the Eddington DF on a binding-energy grid."""
    r = np.logspace(-6, 6, 6001)
    m = mass(r)
    # Psi(r) = M(r)/r + int_r^inf 4 pi r' rho dr'  (G = 1)
    integrand = 4.0 * np.pi * r * rho(r)
    tail = np.concatenate(
        [
            np.cumsum((0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))[::-1])[
                ::-1
            ],
            [0.0],
        ]
    )
    psi = m / r + tail + 1.0 / r[-1] * (1.0 - m[-1])  # outside the grid: a point mass
    # Eddington: f(E) = 1/(sqrt(8) pi^2) d/dE int_0^E drho/dPsi dPsi / sqrt(E - Psi)
    order = np.argsort(psi)
    psi_s, rho_s = psi[order], rho(r)[order]
    drho = np.gradient(rho_s, psi_s)
    energies = psi_s[:: max(1, len(psi_s) // 1500)]
    u = np.linspace(0.0, 1.0, 401)
    g = np.empty_like(energies)
    for k, e in enumerate(energies):
        # Psi = E (1 - u^2): dPsi / sqrt(E - Psi) = 2 sqrt(E) du
        p = e * (1.0 - u * u)
        vals = np.interp(p, psi_s, drho, left=drho[0])
        g[k] = 2.0 * np.sqrt(e) * np.trapezoid(vals, u)
    f = np.gradient(g, energies) / (np.sqrt(8.0) * np.pi**2)
    return r, psi, energies, np.maximum(f, 0.0)


def sample_sphere(profile: str, num: int, seed: int = 0, **kwargs: float) -> dict:
    """An isotropic equilibrium sphere.

    Parameters
    ----------
    profile : str
        ``plummer``, ``hernquist`` or ``nfw`` (``rcut`` keyword, default 10).
    num : int
        Particles.
    seed : int
        numpy generator seed.
    **kwargs : float
        Profile parameters.

    Returns
    -------
    dict
        The IC file's arrays and ``meta``.
    """
    rho, mass, rmax = PROFILES[profile](**kwargs)
    rng = np.random.default_rng(seed)
    r_grid, psi_grid, energies, f = _tables(rho, mass)
    # radii by inverting M(r) up to rmax (mass beyond rmax dropped; masses renormalised)
    mmax = float(mass(np.array([rmax]))[0])
    grid = np.logspace(-5, np.log10(rmax), 20001)
    radii = np.interp(rng.uniform(0.0, mmax, num), mass(grid), grid)
    dirs = rng.normal(size=(num, 3))
    pos = radii[:, None] * dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    psi = np.interp(np.log(radii), np.log(r_grid), psi_grid)
    # speed: pdf ~ f(Psi - v^2/2) v^2 on [0, sqrt(2 Psi)]; rejection on q = v / v_esc
    qs = np.linspace(0.0, 1.0, 129)
    speed = np.empty(num)
    for lo in range(0, num, 1 << 20):
        hi = min(num, lo + (1 << 20))
        ps = psi[lo:hi]
        bound = np.max(
            np.interp(ps[:, None] * (1.0 - qs[None] ** 2), energies, f) * qs[None] ** 2,
            axis=1,
        )
        bound = 1.2 * bound + 1e-300
        out = np.full(hi - lo, -1.0)
        todo = np.arange(hi - lo)
        while todo.size:
            q = rng.uniform(0.0, 1.0, todo.size)
            val = np.interp(ps[todo] * (1.0 - q * q), energies, f) * q * q
            ok = rng.uniform(0.0, 1.0, todo.size) * bound[todo] < val
            out[todo[ok]] = q[ok]
            todo = todo[~ok]
        speed[lo:hi] = out * np.sqrt(2.0 * ps)
    vdir = rng.normal(size=(num, 3))
    vel = speed[:, None] * vdir / np.linalg.norm(vdir, axis=1, keepdims=True)
    m = np.full(num, 1.0 / num)
    pos -= np.sum(m[:, None] * pos, axis=0)
    vel -= np.sum(m[:, None] * vel, axis=0)
    meta = dict(
        profile=profile,
        seed=int(seed),
        rmax=float(rmax),
        mass_inside_rmax=mmax,
        units="G = 1, total mass 1, scale radius 1",
        **{k: float(v) for k, v in kwargs.items()},
    )
    return dict(
        state0=np.stack([pos, vel], axis=1),
        mass=m,
        component=np.zeros(num, np.int8),
        component_names=np.array([profile]),
        meta=json.dumps(meta),
    )


def save_ic(path: str, ic: dict) -> str:
    """Write an IC dict (:func:`sample_sphere`'s) as ``.npz``.

    Parameters
    ----------
    path : str
        Target.
    ic : dict
        Arrays and ``meta``.

    Returns
    -------
    str
        The path.
    """
    np.savez(path, **ic)
    return path


def load_ic(path: str) -> dict:
    """Read an IC file (this module's format; also the merger's ``ic/`` directory).

    Parameters
    ----------
    path : str
        ``.npz`` file, or a directory with ``state0.npy``, ``mass.npy``,
        ``meta.json`` (the 2026-10-31 merger's layout: blocks by galaxy/component).

    Returns
    -------
    dict
        ``state0``, ``mass``, ``component``, ``component_names``, ``meta`` (dict).
    """
    import os

    if os.path.isdir(path):
        meta = json.load(open(os.path.join(path, "meta.json")))
        state = np.load(os.path.join(path, "state0.npy"), mmap_mode="r")
        mass = np.load(os.path.join(path, "mass.npy"), mmap_mode="r")
        comp = np.concatenate(
            [np.full(b["n"], k, np.int8) for k, b in enumerate(meta["blocks"])]
        )
        names = np.array([f"{b['galaxy']}_{b['component']}" for b in meta["blocks"]])
        return dict(
            state0=state, mass=mass, component=comp, component_names=names, meta=meta
        )
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out.get("meta", "{}")))
    if "component" not in out:
        out["component"] = np.zeros(out["mass"].shape[0], np.int8)
        out["component_names"] = np.array(["all"])
    return out


def subsample(ic: dict, num: int) -> dict:
    """The same prefix fraction of every component block, masses scaled to keep totals.

    Parameters
    ----------
    ic : dict
        From :func:`load_ic`.
    num : int
        Target N (``<= 0`` or ``>= N``: everything).

    Returns
    -------
    dict
        The subsample (fp64 arrays), ``meta`` with ``subsample_fraction``.
    """
    comp = np.asarray(ic["component"])
    n_all = comp.shape[0]
    if num <= 0 or num >= n_all:
        idx = np.arange(n_all)
        frac = 1.0
    else:
        frac = num / n_all
        idx = []
        for c in np.unique(comp):
            rows = np.flatnonzero(comp == c)
            keep = max(1, int(round(frac * rows.size)))
            idx.append(rows[:keep])
        idx = np.concatenate(idx)
    state = np.asarray(ic["state0"][idx], np.float64)
    mass = np.asarray(ic["mass"][idx], np.float64)
    for c in np.unique(comp):
        sel = comp[idx] == c
        full = float(np.sum(np.asarray(ic["mass"], np.float64)[comp == c]))
        mass[sel] *= full / float(np.sum(mass[sel]))
    meta = dict(ic["meta"])
    meta["subsample_fraction"] = frac
    return dict(
        state0=state,
        mass=mass,
        component=comp[idx],
        component_names=ic["component_names"],
        meta=meta,
    )

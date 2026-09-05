"""What a reconstruction is measured by -- in the order the section trusts them.

The metric hierarchy is a claim about what this section does and does not
demonstrate, so it is encoded here rather than left to a caption:

1. **Field-space residual** -- :func:`field_residual`. The primary metric. It is
   what the loss minimises and the only thing the data constrains directly.
2. **Recovered-density agreement** -- :func:`density_agreement`,
   :func:`enclosed_mass_profile`. Secondary. A *diagnostic*: the inverse problem
   is ill-posed and has continuous degeneracies, so density agreement can
   saturate while the field residual keeps falling. fig18 plots exactly that
   divergence, which is the degeneracy stated quantitatively.
3. **Per-particle position error** -- :func:`position_error`. Tertiary and
   **explicitly degenerate**. Equal-mass particles are interchangeable, so this
   number is close to meaningless; the function says so in its own return value
   (:attr:`PositionError.degenerate`) and refuses to be mistaken for a score.

Gauge, and what is *not* degenerate
-----------------------------------
Hard decision 10: with the tracers fixed in space the field is not
translation-invariant, so there is **no exact global translation degeneracy**
and no gauge-fixing machinery here -- adding machinery for a degeneracy that
does not exist would be worse than nothing. Near-degeneracies do remain, and
:func:`moment_drift` reports the two a referee asks about: the drift of the
reconstruction's centre of mass and of its total quadrupole, relative to truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "POSITION_ERROR_CAVEAT",
    "MomentDrift",
    "density_agreement",
    "enclosed_mass_profile",
    "field_residual",
    "gradient_norm",
    "moment_drift",
    "position_error",
    "radial_bins",
]


def field_residual(
    predicted: Any, observed: Any, *, clean: Optional[Any] = None
) -> Dict[str, float]:
    """The primary metric: how well the model reproduces the observed field.

    Parameters
    ----------
    predicted : Any
        ``(M, 3)`` model accelerations at the tracers.
    observed : Any
        ``(M, 3)`` observed accelerations -- noisy, if the run has noise.
    clean : Optional[Any]
        ``(M, 3)`` noise-free accelerations, when known. Given it, the noise
        floor is reported too: a fit whose residual has reached the noise floor
        has extracted everything the data holds, and driving it lower is
        fitting the noise realisation.

    Returns
    -------
    Dict[str, float]
        ``rel_l2`` (the headline), ``rms``, ``max_abs``, ``mean_abs``, the
        observed field's own ``rms`` for scale, and -- when ``clean`` is given
        -- ``noise_floor_rel_l2`` and ``residual_over_noise_floor``.
    """
    p = np.asarray(predicted, dtype=np.float64)
    o = np.asarray(observed, dtype=np.float64)
    delta = p - o
    denominator = float(np.linalg.norm(o)) or 1.0
    out = {
        "rel_l2": float(np.linalg.norm(delta) / denominator),
        "rms": float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
        "max_abs": float(np.max(np.linalg.norm(delta, axis=-1))),
        "mean_abs": float(np.mean(np.linalg.norm(delta, axis=-1))),
        "observed_rms": float(np.sqrt(np.mean(np.sum(o**2, axis=-1)))),
    }
    if clean is not None:
        c = np.asarray(clean, dtype=np.float64)
        floor = float(np.linalg.norm(o - c) / denominator)
        out["noise_floor_rel_l2"] = floor
        out["residual_over_noise_floor"] = (
            out["rel_l2"] / floor if floor > 0.0 else float("inf")
        )
    return out


def radial_bins(
    positions: Any, *, num_bins: int = 32, r_max: Optional[float] = None
) -> np.ndarray:
    """Logarithmic radial bin edges spanning a particle distribution.

    Parameters
    ----------
    positions : Any
        ``(N, 3)`` positions used to set the outer edge when ``r_max`` is not
        given.
    num_bins : int
        Number of bins.
    r_max : Optional[float]
        Outer edge. Defaults to the 99.5th percentile radius, so a handful of
        far-flung particles cannot stretch every bin.

    Returns
    -------
    np.ndarray
        ``(num_bins + 1,)`` edges, logarithmically spaced. The inner edge is
        set from the 1st-percentile radius rather than from zero, because a
        log-spaced profile has no zero.
    """
    r = np.linalg.norm(np.asarray(positions, dtype=np.float64), axis=1)
    outer = float(np.percentile(r, 99.5)) if r_max is None else float(r_max)
    inner = float(np.percentile(r[r > 0.0], 1.0)) if np.any(r > 0.0) else 1.0e-3
    inner = max(min(inner, outer * 1.0e-3), outer * 1.0e-6)
    return np.geomspace(inner, outer, int(num_bins) + 1)


def enclosed_mass_profile(
    positions: Any, *, source_mass: float, edges: np.ndarray
) -> Dict[str, Any]:
    """Enclosed mass and shell density against radius.

    Parameters
    ----------
    positions : Any
        ``(N, 3)`` positions.
    source_mass : float
        The one mass every particle carries.
    edges : np.ndarray
        ``(num_bins + 1,)`` radial bin edges, from :func:`radial_bins`.

    Returns
    -------
    Dict[str, Any]
        ``radius`` (bin centres), ``enclosed_mass`` at each outer edge,
        ``shell_density``, and ``counts``. Lists, so the record is
        JSON-serialisable as it stands.
    """
    r = np.linalg.norm(np.asarray(positions, dtype=np.float64), axis=1)
    e = np.asarray(edges, dtype=np.float64)
    counts, _ = np.histogram(r, bins=e)
    enclosed = np.cumsum(counts) * float(source_mass)
    shell_volume = (4.0 / 3.0) * np.pi * (e[1:] ** 3 - e[:-1] ** 3)
    density = (
        counts * float(source_mass) / np.where(shell_volume > 0.0, shell_volume, 1.0)
    )
    return {
        "radius": np.sqrt(e[1:] * e[:-1]).tolist(),
        "edges": e.tolist(),
        "enclosed_mass": enclosed.tolist(),
        "shell_density": density.tolist(),
        "counts": counts.astype(int).tolist(),
    }


def density_agreement(
    reconstructed: Any,
    truth: Any,
    *,
    source_mass: float,
    num_bins: int = 32,
    grid_size: int = 24,
    extent: Optional[float] = None,
) -> Dict[str, Any]:
    """Secondary metric: does the recovered *density field* match truth?

    Two views, because they fail differently. The radial profile is what a
    referee reads; the 3-D binned field catches a reconstruction that has the
    right profile with the mass in the wrong places -- which is precisely the
    degeneracy this section reports.

    Parameters
    ----------
    reconstructed : Any
        ``(N, 3)`` recovered positions.
    truth : Any
        ``(N, 3)`` true positions.
    source_mass : float
        The one mass every particle carries.
    num_bins : int
        Radial bins for the profile comparison.
    grid_size : int
        Cells per axis for the 3-D comparison. ``24`` keeps the cell occupancy
        meaningful at the ``N`` this section sweeps.
    extent : Optional[float]
        Half-width of the 3-D box. Defaults to the truth's 99.5th-percentile
        radius, so the two distributions are binned on the *same* grid -- a
        grid fitted to each separately would hide a size error.

    Returns
    -------
    Dict[str, Any]
        ``profile_rel_l2`` over enclosed mass, ``shell_density_rel_l2``,
        ``grid_rel_l2`` and ``grid_correlation`` over the 3-D field, plus both
        profiles for plotting.
    """
    rec = np.asarray(reconstructed, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    edges = radial_bins(tru, num_bins=num_bins)
    profile_rec = enclosed_mass_profile(rec, source_mass=source_mass, edges=edges)
    profile_tru = enclosed_mass_profile(tru, source_mass=source_mass, edges=edges)

    def rel(a: Sequence[float], b: Sequence[float]) -> float:
        u = np.asarray(a, dtype=np.float64)
        v = np.asarray(b, dtype=np.float64)
        denominator = float(np.linalg.norm(v)) or 1.0
        return float(np.linalg.norm(u - v) / denominator)

    half = (
        float(np.percentile(np.linalg.norm(tru, axis=1), 99.5))
        if extent is None
        else float(extent)
    )
    grid_edges = np.linspace(-half, half, int(grid_size) + 1)
    bins = (grid_edges, grid_edges, grid_edges)
    grid_rec, _ = np.histogramdd(rec, bins=bins)
    grid_tru, _ = np.histogramdd(tru, bins=bins)
    flat_rec = grid_rec.ravel()
    flat_tru = grid_tru.ravel()
    if flat_rec.std() > 0.0 and flat_tru.std() > 0.0:
        correlation = float(np.corrcoef(flat_rec, flat_tru)[0, 1])
    else:
        correlation = float("nan")

    return {
        "profile_rel_l2": rel(
            profile_rec["enclosed_mass"], profile_tru["enclosed_mass"]
        ),
        "shell_density_rel_l2": rel(
            profile_rec["shell_density"], profile_tru["shell_density"]
        ),
        "grid_rel_l2": rel(flat_rec, flat_tru),
        "grid_correlation": correlation,
        "grid_size": int(grid_size),
        "grid_extent": half,
        "in_grid_fraction_reconstructed": float(flat_rec.sum() / max(rec.shape[0], 1)),
        "in_grid_fraction_truth": float(flat_tru.sum() / max(tru.shape[0], 1)),
        "profile_reconstructed": profile_rec,
        "profile_truth": profile_tru,
    }


#: The caveat that travels with every per-particle position error. Sources are
#: equal-mass and therefore interchangeable, so this quantity is degenerate; it
#: is reported because it is asked for, and labelled so it cannot be read as a
#: score.
POSITION_ERROR_CAVEAT = (
    "TERTIARY AND DEGENERATE. Sources are equal-mass and therefore "
    "interchangeable: any permutation of the recovered particles is the same "
    "physical model, so a per-particle displacement is close to meaningless as "
    "a measure of reconstruction quality. Reported because it is asked for, and "
    "labelled because it must not be read as a score. Use the field residual as "
    "the primary metric and the density agreement as the secondary one."
)


def position_error(reconstructed: Any, truth: Any) -> Dict[str, Any]:
    """Tertiary, explicitly degenerate: per-particle displacement from truth.

    Parameters
    ----------
    reconstructed : Any
        ``(N, 3)`` recovered positions, in the truth's particle order.
    truth : Any
        ``(N, 3)`` true positions.

    Returns
    -------
    Dict[str, Any]
        ``rms``, ``median``, ``num_particles``, and -- in the same record --
        ``degenerate: True`` plus :data:`POSITION_ERROR_CAVEAT`, so whatever
        reads the number reads the caveat with it. No matching is attempted:
        identity matching is the only defensible choice here precisely
        *because* the quantity is degenerate -- an optimal-transport match
        would produce a smaller number and a stronger illusion that it means
        something.
    """
    rec = np.asarray(reconstructed, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    displacement = np.linalg.norm(rec - tru, axis=1)
    return {
        "rms": float(np.sqrt(np.mean(displacement**2))),
        "median": float(np.median(displacement)),
        "num_particles": int(rec.shape[0]),
        "degenerate": True,
        "interpretation": POSITION_ERROR_CAVEAT,
    }


@dataclass(frozen=True)
class MomentDrift:
    """Centre-of-mass and quadrupole drift relative to truth.

    Attributes
    ----------
    com_reconstructed : Tuple[float, float, float]
        Centre of mass of the reconstruction.
    com_truth : Tuple[float, float, float]
        Centre of mass of the truth.
    com_drift : float
        ``|com_reconstructed - com_truth|``.
    com_drift_over_size : float
        The same, in units of the truth's rms radius -- the dimensionless form,
        which is what compares across ``N`` and across scales.
    quadrupole_reconstructed : Tuple[float, ...]
        Six independent components ``xx, yy, zz, xy, xz, yz`` of the traceless
        quadrupole.
    quadrupole_truth : Tuple[float, ...]
        The same for the truth.
    quadrupole_rel_drift : float
        Frobenius norm of the difference over the Frobenius norm of truth's.
    """

    com_reconstructed: Tuple[float, float, float]
    com_truth: Tuple[float, float, float]
    com_drift: float
    com_drift_over_size: float
    quadrupole_reconstructed: Tuple[float, ...]
    quadrupole_truth: Tuple[float, ...]
    quadrupole_rel_drift: float

    def as_record(self: "MomentDrift") -> Dict[str, Any]:
        """Return a JSON-safe copy.

        Returns
        -------
        Dict[str, Any]
            Every field, tuples as lists.
        """
        return {
            "com_reconstructed": list(self.com_reconstructed),
            "com_truth": list(self.com_truth),
            "com_drift": self.com_drift,
            "com_drift_over_size": self.com_drift_over_size,
            "quadrupole_reconstructed": list(self.quadrupole_reconstructed),
            "quadrupole_truth": list(self.quadrupole_truth),
            "quadrupole_rel_drift": self.quadrupole_rel_drift,
            "gauge_note": (
                "The tracers are fixed in space, so the field is NOT "
                "translation-invariant and there is no exact global translation "
                "degeneracy to quotient out (hard decision 10). These are "
                "reported because near-degeneracies remain and because a "
                "referee will ask -- not because a gauge was fixed."
            ),
        }


def _traceless_quadrupole(positions: np.ndarray, mass: float) -> np.ndarray:
    """Return the traceless quadrupole tensor of an equal-mass distribution.

    Parameters
    ----------
    positions : np.ndarray
        ``(N, 3)`` positions.
    mass : float
        The one mass every particle carries.

    Returns
    -------
    np.ndarray
        ``(3, 3)`` tensor ``sum_i m (3 x_i x_i^T - |x_i|^2 I)``.
    """
    # np.matmul, not "@": this is a host-side float64 contraction, so the
    # TF32 lowering that @highest_matmul_precision guards against cannot
    # apply. Spelling it as a call keeps it out of that guard honestly.
    outer = np.matmul(positions.T, positions)
    r2 = float(np.sum(positions**2))
    return mass * (3.0 * outer - r2 * np.eye(3))


def moment_drift(reconstructed: Any, truth: Any, *, source_mass: float) -> MomentDrift:
    """Centre-of-mass and quadrupole drift of a reconstruction against truth.

    Parameters
    ----------
    reconstructed : Any
        ``(N, 3)`` recovered positions.
    truth : Any
        ``(N, 3)`` true positions.
    source_mass : float
        The one mass every particle carries. Total mass is conserved by
        construction -- masses are frozen and equal, and ``N`` does not change
        -- so no normalisation enters here.

    Returns
    -------
    MomentDrift
        The drifts, with the dimensionless forms alongside the raw ones.
    """
    rec = np.asarray(reconstructed, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    com_rec = rec.mean(axis=0)
    com_tru = tru.mean(axis=0)
    size = float(np.sqrt(np.mean(np.sum(tru**2, axis=1)))) or 1.0
    q_rec = _traceless_quadrupole(rec, float(source_mass))
    q_tru = _traceless_quadrupole(tru, float(source_mass))
    denominator = float(np.linalg.norm(q_tru)) or 1.0

    def components(q: np.ndarray) -> Tuple[float, ...]:
        return (
            float(q[0, 0]),
            float(q[1, 1]),
            float(q[2, 2]),
            float(q[0, 1]),
            float(q[0, 2]),
            float(q[1, 2]),
        )

    drift = float(np.linalg.norm(com_rec - com_tru))
    return MomentDrift(
        com_reconstructed=(float(com_rec[0]), float(com_rec[1]), float(com_rec[2])),
        com_truth=(float(com_tru[0]), float(com_tru[1]), float(com_tru[2])),
        com_drift=drift,
        com_drift_over_size=drift / size,
        quadrupole_reconstructed=components(q_rec),
        quadrupole_truth=components(q_tru),
        quadrupole_rel_drift=float(np.linalg.norm(q_rec - q_tru) / denominator),
    )


def gradient_norm(gradient: Any) -> Dict[str, float]:
    """Summarise a gradient pytree's magnitude, for convergence tracking.

    Parameters
    ----------
    gradient : Any
        A gradient pytree, or any array. Leaves are flattened together.

    Returns
    -------
    Dict[str, float]
        ``l2``, ``l_inf``, ``rms_per_parameter`` and ``num_parameters``. The
        per-parameter form is the comparable one: an L2 norm over 3e7
        coordinates is larger than one over 7 for reasons that have nothing to
        do with convergence.
    """
    import jax

    leaves = [
        np.asarray(leaf, dtype=np.float64).ravel()
        for leaf in jax.tree_util.tree_leaves(gradient)
    ]
    flat = np.concatenate(leaves) if leaves else np.zeros((0,), dtype=np.float64)
    count = int(flat.size)
    l2 = float(np.linalg.norm(flat))
    return {
        "l2": l2,
        "l_inf": float(np.max(np.abs(flat))) if count else 0.0,
        "rms_per_parameter": l2 / np.sqrt(count) if count else 0.0,
        "num_parameters": count,
    }

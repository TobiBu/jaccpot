"""Shared measurement harness for the validation benchmarks (figures 01-03).

Extracted verbatim from ``mac_error_distribution.py`` so that the force-error
figures and the MAC comparison share one reference oracle, one set of
distributions, and one set of error metrics. The alternative -- each paper script
carrying its own copy -- is how two figures in the same paper end up disagreeing
about what "relative force error" means.

This is a pure move: no numerics changed in the extraction, and
``mac_error_distribution.py`` now imports from here rather than defining its own.

Imports jax at module scope, so a GPU bench script must import this *after* it
has called ``autocvd`` (see ``examples.jaccpot_paper.common.runmeta.select_gpu``).
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "QUANTILES",
    "chunked_direct_accelerations",
    "chunked_force_scale",
    "error_summary",
    "log_interp_at",
    "make_distribution",
    "per_particle_dehnen_scaled_error",
    "per_particle_relative_error",
    "per_particle_scaled_error",
    "rel_l2",
    "worst_component_error",
]

QUANTILES = (0.5, 0.9, 0.99, 0.999)


# ---------------------------------------------------------------------------
# distributions
# ---------------------------------------------------------------------------


def _plummer(rng: np.random.Generator, n: int, scale: float = 1.0) -> np.ndarray:
    """Sample positions from a Plummer sphere by inverse transform."""

    u = rng.uniform(size=n)
    radius = (
        scale * u ** (1.0 / 3.0) / np.sqrt(np.maximum(1.0 - u ** (2.0 / 3.0), 1e-12))
    )
    radius = np.minimum(radius, 20.0 * scale)
    cos_t = rng.uniform(-1.0, 1.0, size=n)
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack(
        [radius * sin_t * np.cos(phi), radius * sin_t * np.sin(phi), radius * cos_t],
        axis=1,
    )


def make_distribution(name: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, masses) for the named distribution."""

    rng = np.random.default_rng(seed)
    if name == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n, 3)), rng.uniform(0.5, 1.5, size=n)
    if name == "plummer":
        return _plummer(rng, n), np.ones(n)
    if name == "bulge_halo":
        # Co-centred heavy compact bulge + light extended halo. This is the
        # discriminating case for eq (16a): the acceleration spans orders of
        # magnitude between centre and outskirts, so the *absolute* accuracy a
        # halo particle needs is far coarser than a bulge particle's -- exactly
        # the freedom `eps * min_b |a_b|` exploits and a fixed opening angle
        # cannot see.
        #
        # Deliberately co-centred rather than two separated clumps: separated
        # clumps have a force null between them where |a| -> 0, and per-particle
        # *relative* error diverges there for any MAC, which swamps the
        # comparison with an artifact of the metric rather than a property of
        # the criterion.
        n_bulge = n // 2
        n_halo = n - n_bulge
        pos = np.concatenate([_plummer(rng, n_bulge, 0.2), _plummer(rng, n_halo, 2.0)])
        mass = np.concatenate(
            [np.full(n_bulge, 0.9 / n_bulge), np.full(n_halo, 0.1 / n_halo)]
        )
        return pos, mass
    if name == "disc":
        # A thin exponential-ish disc. Every distribution above is spheroidal, and
        # track B's leaf/order tables are all measured on a disc -- which is exactly
        # the discrepancy `docs/plan_2026-08_B_nearfield.md` flags against the MAC
        # study's "leaf 1024 is the production configuration": that conclusion was
        # reached on Plummer and bulge+halo. Reading the two on a common footing needs
        # the disc to live beside them.
        #
        # `main` added this case to make_distribution in mac_error_distribution.py at
        # the same time this branch was extracting that function into this module, so
        # neither side of the merge carries both. Ported by hand. It is load bearing:
        # main's own bench/validation/order_leaf_accuracy_sweep.py DEFAULTS to
        # --distribution disc and would raise "unknown distribution" without it.
        radius, thickness = 10.0, 0.2
        r = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
        th = rng.uniform(0.0, 2.0 * np.pi, n)
        pos = np.stack(
            [r * np.cos(th), r * np.sin(th), rng.normal(scale=thickness, size=n)],
            axis=1,
        )
        return pos, rng.uniform(0.5, 1.5, size=n)
    if name == "mass_spectrum":
        # Uniform positions, three decades of mass. Isolates mass-dependence
        # from spatial clustering.
        return (
            rng.uniform(-1.0, 1.0, size=(n, 3)),
            10.0 ** rng.uniform(0.0, 3.0, size=n),
        )
    raise ValueError(f"unknown distribution {name!r}")


# ---------------------------------------------------------------------------
# reference and error metrics
# ---------------------------------------------------------------------------


def chunked_direct_accelerations(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    *,
    softening: float,
    G: float,
    block: int = 512,
    targets: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Exact direct-sum accelerations, chunked over targets.

    The dense O(N^2) formulation the notebooks use needs 6.4 GB at N=16384 in
    float64; this streams over target blocks instead so the reference survives
    the N values the claim has to be checked at.

    ``targets`` restricts evaluation to a subset of target indices while still
    summing over *all* sources, which is what makes N=1e6 tractable: 1e4 targets
    against 1e6 sources is 1e10 pairs rather than 1e12. Returns one row per entry
    of ``targets``, in that order.
    """

    pos = jnp.asarray(positions)
    mass = jnp.asarray(masses)
    n = int(pos.shape[0])
    eps_sq = jnp.asarray(softening * softening, dtype=pos.dtype)
    target_idx = (
        jnp.arange(n, dtype=jnp.int32)
        if targets is None
        else jnp.asarray(targets, dtype=jnp.int32)
    )
    num_targets = int(target_idx.shape[0])

    def block_acc(start: jnp.ndarray) -> jnp.ndarray:
        slot = start + jnp.arange(block)
        safe_slot = jnp.clip(slot, 0, num_targets - 1)
        safe = target_idx[safe_slot]
        target_pos = pos[safe]
        delta = pos[None, :, :] - target_pos[:, None, :]
        dist_sq = jnp.sum(delta * delta, axis=2) + eps_sq
        inv = jnp.where(
            jnp.arange(n)[None, :] == safe[:, None],
            jnp.asarray(0.0, dtype=pos.dtype),
            dist_sq ** (-1.5),
        )
        return jnp.asarray(G, dtype=pos.dtype) * jnp.einsum(
            "ij,j,ijk->ik", inv, mass, delta
        )

    starts = jnp.arange(0, num_targets, block)
    out = jax.lax.map(block_acc, starts)
    return out.reshape(-1, 3)[:num_targets]


def per_particle_relative_error(
    estimate: jnp.ndarray, reference: jnp.ndarray
) -> np.ndarray:
    """Dehnen's own error measure: |a_fmm - a_ref| / |a_ref|, per particle."""

    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    num = np.linalg.norm(est - ref, axis=1)
    den = np.linalg.norm(ref, axis=1)
    return num / np.maximum(den, np.finfo(np.float64).tiny)


def chunked_force_scale(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    *,
    softening: float,
    G: float,
    block: int = 512,
    targets: Optional[np.ndarray] = None,
) -> jnp.ndarray:
    """Dehnen's per-particle force scale ``f_b = sum_{a!=b} G m_a / |x_a - x_b|^2``.

    ``targets`` restricts evaluation to a subset of target indices while summing
    over all sources, matching :func:`chunked_direct_accelerations`. Note that the
    ``mass_16b`` arm needs ``f_b`` for **every** particle, not just the measured
    subset, because it feeds the node reduction -- so that arm is unavailable under
    subsampling, and the driver rejects the combination rather than silently
    injecting a partial scale.

    This is the sum of pairwise force *magnitudes*, i.e. the acceleration a
    particle would feel if none of its interactions cancelled. Unlike ``|a_b|`` it
    never vanishes, which is why Dehnen (2014) section 3.1 defines the "scaled
    error" ``delta_a / f`` on it and equation (16b) puts ``min_b f_b`` on the
    criterion's right-hand side.

    It is *not* a local quantity, despite the single largest term being the
    nearest neighbour. In 3D the particle count in a shell grows like ``r^2 rho``
    while each contribution falls like ``1/r^2``, so every logarithmic shell
    contributes comparably and the sum converges slowly. Measured at N=4096,
    the 16 largest contributors capture a median 13% of ``f_b`` on Plummer (18%
    uniform, 7% bulge+halo) and even the largest 256 capture only 41%. Any
    O(N) estimator therefore needs the far-field monopole term -- a near-field
    sum alone is wrong by nearly an order of magnitude, not by a few percent.
    """

    pos = jnp.asarray(positions)
    mass = jnp.asarray(masses)
    n = int(pos.shape[0])
    eps_sq = jnp.asarray(softening * softening, dtype=pos.dtype)
    target_idx = (
        jnp.arange(n, dtype=jnp.int32)
        if targets is None
        else jnp.asarray(targets, dtype=jnp.int32)
    )
    num_targets = int(target_idx.shape[0])

    def block_scale(start: jnp.ndarray) -> jnp.ndarray:
        slot = start + jnp.arange(block)
        safe_slot = jnp.clip(slot, 0, num_targets - 1)
        safe = target_idx[safe_slot]
        target_pos = pos[safe]
        delta = pos[None, :, :] - target_pos[:, None, :]
        dist_sq = jnp.sum(delta * delta, axis=2) + eps_sq
        contrib = jnp.where(
            jnp.arange(n)[None, :] == safe[:, None],
            jnp.asarray(0.0, dtype=pos.dtype),
            mass[None, :] / dist_sq,
        )
        return jnp.asarray(G, dtype=pos.dtype) * jnp.sum(contrib, axis=1)

    starts = jnp.arange(0, num_targets, block)
    return jax.lax.map(block_scale, starts).reshape(-1)[:num_targets]


def per_particle_dehnen_scaled_error(
    estimate: jnp.ndarray, reference: jnp.ndarray, force_scale: jnp.ndarray
) -> np.ndarray:
    """Dehnen's scaled error ``|a_fmm - a_ref| / f_b`` (section 3.1)."""

    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    f = np.asarray(force_scale, dtype=np.float64)
    return np.linalg.norm(est - ref, axis=1) / np.maximum(f, np.finfo(np.float64).tiny)


def per_particle_scaled_error(
    estimate: jnp.ndarray, reference: jnp.ndarray
) -> np.ndarray:
    """|a_fmm - a_ref| / rms(|a_ref|), normalised by a single global scale.

    Dehnen's per-particle relative error is the right measure for a
    near-homogeneous system, but it is undefined wherever the true acceleration
    vanishes -- the centre of a centrally-concentrated system, or the force null
    between two clumps. There a vanishing *absolute* error still produces an
    unbounded *relative* one, so the upper tail of the relative-error
    distribution reports a property of the metric rather than of the MAC, and it
    does so identically for every criterion. Normalising by one global scale
    keeps the tail meaningful for clustered systems.
    """

    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    scale = float(np.sqrt(np.mean(np.sum(ref * ref, axis=1))))
    return np.linalg.norm(est - ref, axis=1) / max(scale, np.finfo(np.float64).tiny)


def error_summary(errors: np.ndarray, prefix: str = "") -> dict[str, float]:
    qs = np.quantile(errors, QUANTILES)
    return {
        # Dehnen reports the rms and the 99.99 percentile, noting the rms is
        # always about ten times smaller than the latter.
        f"{prefix}rms": float(np.sqrt(np.mean(errors * errors))),
        f"{prefix}p9999": float(np.quantile(errors, 0.9999)),
        f"{prefix}median": float(qs[0]),
        f"{prefix}p90": float(qs[1]),
        f"{prefix}p99": float(qs[2]),
        f"{prefix}p999": float(qs[3]),
        f"{prefix}max": float(errors.max()),
        f"{prefix}mean": float(errors.mean()),
    }


# ---------------------------------------------------------------------------
# aggregate error metrics used by the convergence figures (01 / 02)
# ---------------------------------------------------------------------------


def rel_l2(estimate: Any, reference: Any) -> float:
    """Relative L2 error over the whole acceleration field.

    The headline convergence metric: a single number per configuration, dominated
    by the particles with the largest absolute error rather than by the ones with
    the smallest reference force. Reported alongside
    :func:`worst_component_error` because an L2 norm alone hides a tail.
    """

    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(est - ref) / max(denom, np.finfo(np.float64).tiny))


def worst_component_error(estimate: Any, reference: Any) -> float:
    """Largest single-component absolute error, normalised by the rms |a_ref|.

    The companion to rel-L2: convergence plots that show only an L2 norm cannot
    distinguish "uniformly a bit worse" from "one particle catastrophically
    wrong", and for an integrator it is the latter that matters.
    """

    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    scale = float(np.sqrt(np.mean(np.sum(ref * ref, axis=1))))
    return float(np.abs(est - ref).max() / max(scale, np.finfo(np.float64).tiny))


# ---------------------------------------------------------------------------
# matched-accuracy interpolation
# ---------------------------------------------------------------------------


def log_interp_at(
    records: list[dict[str, Any]],
    *,
    target_p90: float,
    field: str,
    p90_key: str = "p90",
) -> Optional[float]:
    """Log-interpolate ``field`` at the knob where p90 == ``target_p90``."""

    usable = [r for r in records if r[p90_key] > 0 and r[field] > 0]
    if len(usable) < 2:
        return None
    usable.sort(key=lambda r: r[p90_key])
    p90 = np.log(np.asarray([r[p90_key] for r in usable]))
    vals = np.log(np.asarray([float(r[field]) for r in usable]))
    target = np.log(target_p90)
    if target < p90[0] or target > p90[-1]:
        return None
    return float(np.exp(np.interp(target, p90, vals)))

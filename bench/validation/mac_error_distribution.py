"""Fixed-theta vs Dehnen mass-dependent MAC: force-error distribution vs cost.

Dehnen (2014, arXiv:1405.2255) section 5.3 claims that replacing the geometric
opening criterion with the error-controlled criterion of eq (16a) gives a
"remarkable" reduction in the *large-error tail* at comparable *median* error,
while avoiding needlessly-accurate interactions. That is a claim about the shape
of the per-particle error distribution, not about rel-L2 and not about raw speed,
so a rel-L2 table structurally cannot support or refute it.

This script sweeps both criteria over their respective accuracy knobs, records
the full per-particle relative force-error distribution and hardware-independent
cost proxies for each, and matches the two arms at equal 90th-percentile error.

Arms
----
``fixed``   ``mac_type="dehnen"``, sweep theta. The baseline.
``mass``    ``mac_type="dehnen_error"``, sweep eps. eq (16a) verbatim.

Matching on the 90th percentile rather than the median is deliberate: when the
far field is shallow, most particles are pure near-field and the median error
saturates at machine precision, which makes median-matching degenerate.

Usage
-----
Small CPU smoke run::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m bench.validation.mac_error_distribution \\
        --n 4096 --leaf-size 16 --order 4 --distribution uniform --json-out /tmp/smoke.json

Full GPU run (use autocvd to pick a free device)::

    autocvd -- python -m bench.validation.mac_error_distribution \\
        --n 32768,131072 --leaf-size 32 --order 4,8 \\
        --distribution uniform,plummer,two_component,mass_spectrum \\
        --json-out bench/results/validation/mac_error_distribution.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from dataclasses import replace
from typing import Any, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaccpot.config import FMMAdvancedConfig  # noqa: E402
from jaccpot.runtime._adaptive_policy import (  # noqa: E402
    compute_node_force_scale_from_sorted_magnitudes,
)
from jaccpot.solver import FastMultipoleMethod  # noqa: E402

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
) -> jnp.ndarray:
    """Exact direct-sum accelerations, chunked over targets.

    The dense O(N^2) formulation the notebooks use needs 6.4 GB at N=16384 in
    float64; this streams over target blocks instead so the reference survives
    the N values the claim has to be checked at.
    """

    pos = jnp.asarray(positions)
    mass = jnp.asarray(masses)
    n = int(pos.shape[0])
    eps_sq = jnp.asarray(softening * softening, dtype=pos.dtype)

    def block_acc(start: jnp.ndarray) -> jnp.ndarray:
        idx = start + jnp.arange(block)
        safe = jnp.clip(idx, 0, n - 1)
        targets = pos[safe]
        delta = pos[None, :, :] - targets[:, None, :]
        dist_sq = jnp.sum(delta * delta, axis=2) + eps_sq
        inv = jnp.where(
            jnp.arange(n)[None, :] == safe[:, None],
            jnp.asarray(0.0, dtype=pos.dtype),
            dist_sq ** (-1.5),
        )
        return jnp.asarray(G, dtype=pos.dtype) * jnp.einsum(
            "ij,j,ijk->ik", inv, mass, delta
        )

    starts = jnp.arange(0, n, block)
    out = jax.lax.map(block_acc, starts)
    return out.reshape(-1, 3)[:n]


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
) -> jnp.ndarray:
    """Dehnen's per-particle force scale ``f_b = sum_{a!=b} G m_a / |x_a - x_b|^2``.

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

    def block_scale(start: jnp.ndarray) -> jnp.ndarray:
        idx = start + jnp.arange(block)
        safe = jnp.clip(idx, 0, n - 1)
        targets = pos[safe]
        delta = pos[None, :, :] - targets[:, None, :]
        dist_sq = jnp.sum(delta * delta, axis=2) + eps_sq
        contrib = jnp.where(
            jnp.arange(n)[None, :] == safe[:, None],
            jnp.asarray(0.0, dtype=pos.dtype),
            mass[None, :] / dist_sq,
        )
        return jnp.asarray(G, dtype=pos.dtype) * jnp.sum(contrib, axis=1)

    starts = jnp.arange(0, n, block)
    return jax.lax.map(block_scale, starts).reshape(-1)[:n]


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
# one measurement
# ---------------------------------------------------------------------------


def _advanced(mac_type: str) -> FMMAdvancedConfig:
    cfg = FMMAdvancedConfig()
    return replace(
        cfg,
        mac_type=mac_type,
        # Both arms retain the traversal result so both pay the identical loss of
        # the streamed fast lane. Without this the mass arm would be charged for
        # a lane fallback the geometric arm avoids, and the cost comparison would
        # measure plumbing rather than the criterion.
        runtime=replace(
            cfg.runtime, retain_traversal_result=True, retain_interactions=True
        ),
    )


def measure(
    *,
    arm: str,
    knob: float,
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    reference: jnp.ndarray,
    force_scale: jnp.ndarray,
    leaf_size: int,
    order: int,
    geometry_mode: str,
    theta_max: Optional[float],
    softening: float,
    G: float,
) -> dict[str, Any]:
    """Run one (arm, knob) configuration and return its record."""

    if arm == "fixed":
        kwargs: dict[str, Any] = dict(theta=float(knob), advanced=_advanced("dehnen"))
    else:
        kwargs = dict(
            # theta does not gate acceptance in paper mode -- eq (16a) supplies
            # its own `theta < 1` convergence guard -- so it is pinned at 1.0 and
            # eps is the accuracy knob.
            theta=1.0,
            adaptive_eps=float(knob),
            dehnen_geometry_mode=geometry_mode,
            advanced=_advanced("dehnen_error"),
        )
        if theta_max is not None:
            kwargs["mac_theta_max"] = float(theta_max)
    kwargs["G"] = G
    kwargs["softening"] = softening

    fmm = FastMultipoleMethod(**kwargs)

    t0 = time.perf_counter()
    state = fmm.prepare_state(positions, masses, leaf_size=leaf_size, max_order=order)
    if arm == "mass_16b":
        # eq (16b) is eq (16a) with `min_b f_b` on the right-hand side instead of
        # `min_b |a_b|`; the criterion, the traversal and the error estimator are
        # untouched. So the whole of (16b) is a different force scale, and this arm
        # supplies the *exact* f_b -- an O(N^2) sum no production path would run --
        # to measure the ceiling before anyone builds an estimator for it.
        #
        # The first prepare_state above exists only to learn the tree; the node
        # count and particle ordering are MAC-independent, so re-preparing with the
        # injected scale reuses the same topology.
        f_b_sorted = jnp.asarray(force_scale)[state.tree.particle_indices]
        f_b_nodes = compute_node_force_scale_from_sorted_magnitudes(
            tree=state.tree,
            magnitudes_sorted=f_b_sorted,
            reduction="min",
        )
        state = fmm.prepare_state(
            positions,
            masses,
            leaf_size=leaf_size,
            max_order=order,
            force_scale_nodes=f_b_nodes,
        )
    prepare_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    acc = fmm.evaluate_prepared_state(state, return_potential=False)
    jax.block_until_ready(acc)
    evaluate_s = time.perf_counter() - t0

    errors = per_particle_relative_error(acc, reference)
    scaled_errors = per_particle_scaled_error(acc, reference)
    dehnen_errors = per_particle_dehnen_scaled_error(acc, reference, force_scale)

    node_ranges = np.asarray(state.tree.node_ranges)
    interactions = state.interactions
    far_pairs = 0
    far_work = 0
    if interactions is not None:
        src = np.asarray(interactions.sources)
        tgt = np.asarray(interactions.targets)
        keep = (src >= 0) & (tgt >= 0)
        far_pairs = int(keep.sum())
        far_work = far_pairs * (order + 1) ** 2

    nb = state.neighbor_list
    nb_counts = np.asarray(nb.counts)
    nb_offsets = np.asarray(nb.offsets)
    nb_neighbors = np.asarray(nb.neighbors)
    nb_leaves = np.asarray(nb.leaf_indices)
    near_pairs = 0
    near_work = 0
    for slot, leaf in enumerate(nb_leaves.tolist()):
        lo, hi = int(node_ranges[leaf, 0]), int(node_ranges[leaf, 1])
        n_t = max(hi - lo + 1, 0)
        start = int(nb_offsets[slot])
        for k in range(int(nb_counts[slot])):
            source_leaf = int(nb_neighbors[start + k])
            if source_leaf < 0:
                continue
            slo, shi = int(node_ranges[source_leaf, 0]), int(
                node_ranges[source_leaf, 1]
            )
            near_pairs += 1
            near_work += n_t * max(shi - slo + 1, 0)
        near_work += n_t * n_t  # self block

    # Converged traversal capacities. The dual-tree build retries with growing
    # caps when the queue or per-node interaction list overflows, and *each retry
    # recompiles* -- at eps=2e-7 that was 6 retries and minutes of wall time per
    # config, which is what made the tight-eps sweep look like a hang. Recording
    # the capacities the retries converged to lets a later run pass them up front
    # via DualTreeTraversalConfig and skip the whole cycle.
    #
    # Do NOT round these up "for safety": the traversal allocates buffers of size
    # num_nodes * interaction_capacity, so an oversized cap OOMs. Measured:
    # interaction_capacity=1<<18 at N=16384 tried to allocate 4 GiB on top of 32
    # and died, while the converged value ran fine.
    retry_events = tuple(getattr(fmm._impl, "recent_retry_events", ()) or ())
    final_caps = None
    if retry_events:
        last = retry_events[-1]
        final_caps = {
            "attempts": len(retry_events),
            "queue_capacity": int(last.queue_capacity),
            "interaction_capacity": int(last.interaction_capacity),
            "status": str(last.status),
        }

    record = {
        "arm": arm,
        "knob": float(knob),
        "retry_final_caps": final_caps,
        "far_pairs": far_pairs,
        "near_pairs": near_pairs,
        "far_work": int(far_work),
        "near_work": int(near_work),
        "pair_work": int(far_work + near_work),
        "prepare_s": prepare_s,
        "evaluate_s": evaluate_s,
        "rel_l2": float(
            np.linalg.norm(np.asarray(acc) - np.asarray(reference))
            / np.linalg.norm(np.asarray(reference))
        ),
    }
    record.update(error_summary(errors))
    record.update(error_summary(scaled_errors, prefix="scaled_"))
    record.update(error_summary(dehnen_errors, prefix="dehnen_"))
    return record


# ---------------------------------------------------------------------------
# matched-accuracy comparison
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


def compare_arms(
    fixed: list[dict[str, Any]],
    mass: list[dict[str, Any]],
    *,
    metric: str = "scaled_",
    match_on: str = "p90",
) -> list[dict[str, Any]]:
    """Compare the arms at a matched error statistic.

    ``metric`` selects the error family: ``""`` for Dehnen's per-particle
    relative error, ``"scaled_"`` for the globally-normalised one (the default,
    because it stays meaningful where the true acceleration vanishes).

    ``match_on`` selects *which* statistic is equalised, and this matters more than
    it looks. Dehnen section 5.3 states the claim as a reduced large-error tail at
    comparable **median**, so ``"median"`` is the apples-to-apples comparison.
    ``"p90"`` was the original default because at N=4096 the far field is shallow
    enough that most particles are pure near-field and the median saturates at
    machine precision, which makes median-matching degenerate. That reasoning stops
    applying once the far field is deep: at N=1e5 the median runs 3e-7..4e-5.

    Be aware the two disagree substantially, because p90 is itself partly a tail
    statistic -- equalising it asks the mass MAC to give up exactly the property it
    is good at. Measured at N=1e5/p=8: matched-p90 gives Plummer p99 ratios of
    0.91-1.14, while matched-median on the same data gives rms 2.0x and p99.99 1.7x
    at 38% less work. Report the statistic you matched on, and prefer showing the
    whole distribution over any single ratio.
    """

    out = []
    p90 = f"{metric}{match_on}"
    lo = max(
        min(r[p90] for r in fixed if r[p90] > 0),
        min(r[p90] for r in mass if r[p90] > 0),
    )
    hi = min(
        max(r[p90] for r in fixed),
        max(r[p90] for r in mass),
    )
    if not (hi > lo):
        return out
    for target in np.exp(np.linspace(np.log(lo), np.log(hi), 5)):
        row: dict[str, Any] = {"matched_p90": float(target)}
        ok = True
        for label, records in (("fixed", fixed), ("mass", mass)):
            for name in ("p99", "max", "median", "pair_work", "far_pairs"):
                field = f"{metric}{name}" if name in ("p99", "max", "median") else name
                val = log_interp_at(
                    records, target_p90=float(target), field=field, p90_key=p90
                )
                if val is None:
                    ok = False
                row[f"{label}_{name}"] = val
        if not ok:
            continue
        row["p99_ratio"] = (
            row["fixed_p99"] / row["mass_p99"] if row["mass_p99"] else None
        )
        row["max_ratio"] = (
            row["fixed_max"] / row["mass_max"] if row["mass_max"] else None
        )
        row["pair_work_ratio"] = (
            row["fixed_pair_work"] / row["mass_pair_work"]
            if row["mass_pair_work"]
            else None
        )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _floats(text: str) -> list[float]:
    return [float(v) for v in str(text).split(",") if v.strip()]


def _ints(text: str) -> list[int]:
    return [int(v) for v in str(text).split(",") if v.strip()]


def _git_meta() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=False
            ).stdout.strip()
        except OSError:  # pragma: no cover
            return ""

    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", default="4096")
    ap.add_argument(
        "--leaf-size",
        type=int,
        default=256,
        help=(
            "particles per leaf. 256 is the production setting (it is what the 1M "
            "large-N runs use). Small leaves are actively bad at tight eps: more "
            "nodes means longer per-node interaction lists, which is what drives "
            "the retry-recompile cycle -- 6 retries at leaf 16 / eps=2e-7."
        ),
    )
    ap.add_argument("--order", default="4")
    ap.add_argument("--distribution", default="uniform")
    ap.add_argument("--theta", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70,0.80")
    ap.add_argument("--eps", default="1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6")
    ap.add_argument("--geometry-mode", default="com")
    ap.add_argument(
        "--arm",
        default="fixed,mass",
        help=(
            "comma-separated arms. 'fixed' = geometric MAC swept over theta; "
            "'mass' = Dehnen eq (16a) swept over eps; 'mass_16b' = eq (16b), the "
            "same criterion with exact O(N^2) f_b as the force scale."
        ),
    )
    ap.add_argument("--theta-max", type=float, default=None)
    ap.add_argument("--softening", type=float, default=1e-3)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--metric",
        choices=("relative", "scaled", "dehnen"),
        default="dehnen",
        help=(
            "error family used for matching: 'relative' is Dehnen's per-particle "
            "relative error (valid for near-homogeneous systems); 'scaled' "
            "normalises by the global rms |a| and stays meaningful where the "
            "true acceleration vanishes (clustered systems)"
        ),
    )
    ap.add_argument(
        "--match-on",
        default="p90",
        choices=("p90", "median"),
        help=(
            "which error statistic the two arms are equalised on. 'median' is "
            "Dehnen section 5.3's own comparison (reduced tail at comparable "
            "median); 'p90' is the historical default, safer when the far field is "
            "shallow enough that the median saturates. They disagree substantially "
            "-- see compare_arms."
        ),
    )
    ap.add_argument(
        "--reference-block",
        type=int,
        default=512,
        help=(
            "target-block size for the O(N^2) reference. Peak scratch is roughly "
            "block * N * 3 * 8 bytes, so 512 needs ~3.6 GB at N=1e5 -- drop it to 64 "
            "there. Keep ALL targets rather than subsampling: p99.99 at N=1e5 is only "
            "~10 particles, and subsampling targets would leave the tail unmeasurable."
        ),
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    known_arms = ("fixed", "mass", "mass_16b")
    arms = tuple(a.strip() for a in str(args.arm).split(",") if a.strip())
    unknown = [a for a in arms if a not in known_arms]
    if unknown:
        ap.error(f"unknown --arm value(s) {unknown}; choose from {list(known_arms)}")
    if "fixed" not in arms:
        # compare_arms measures every mass arm against the geometric baseline, so
        # dropping it would silently produce an empty comparison table.
        ap.error("--arm must include 'fixed'; it is the comparison baseline")

    # Guard the leaf-size / N interaction. With too few leaves the tree has no far
    # field to speak of and the MAC comparison is vacuous -- measured at N=16384 /
    # leaf 256 (64 leaves): the criterion accepted ZERO far pairs at eps=2e-7 and
    # the run degenerated to all-to-all direct summation (near_total = 64*63). It
    # looked fast, and it measured nothing. leaf 256 is the right production value
    # but wants N >= ~1e5 (390 leaves) to be meaningful.
    _min_leaves = 128
    for _n in _ints(args.n):
        _leaves = _n // max(int(args.leaf_size), 1)
        if _leaves < _min_leaves:
            print(
                f"WARNING: N={_n} with leaf_size={args.leaf_size} gives only "
                f"~{_leaves} leaves. The far field will be trivial or empty and the "
                f"arm comparison meaningless. Use N >= {_min_leaves * args.leaf_size} "
                f"at this leaf size, or a smaller leaf size.",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for dist in str(args.distribution).split(","):
        dist = dist.strip()
        if not dist:
            continue
        for n in _ints(args.n):
            pos_np, mass_np = make_distribution(dist, n, args.seed)
            positions = jnp.asarray(pos_np, dtype=jnp.float64)
            masses = jnp.asarray(mass_np, dtype=jnp.float64)
            reference = chunked_direct_accelerations(
                positions,
                masses,
                softening=args.softening,
                G=args.G,
                block=int(args.reference_block),
            )
            jax.block_until_ready(reference)
            force_scale = chunked_force_scale(
                positions,
                masses,
                softening=args.softening,
                G=args.G,
                block=int(args.reference_block),
            )
            jax.block_until_ready(force_scale)

            for order in _ints(args.order):
                by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
                sweeps = tuple(
                    (arm, _floats(args.theta) if arm == "fixed" else _floats(args.eps))
                    for arm in arms
                )
                for arm, knobs in sweeps:
                    for knob in knobs:
                        rec = measure(
                            arm=arm,
                            knob=knob,
                            positions=positions,
                            masses=masses,
                            reference=reference,
                            force_scale=force_scale,
                            leaf_size=args.leaf_size,
                            order=order,
                            geometry_mode=args.geometry_mode,
                            theta_max=args.theta_max,
                            softening=args.softening,
                            G=args.G,
                        )
                        rec.update(
                            {
                                "distribution": dist,
                                "n": n,
                                "order": order,
                                "leaf_size": args.leaf_size,
                            }
                        )
                        by_arm[arm].append(rec)
                        records.append(rec)
                        print(
                            f"{dist:>14s} N={n:<7d} p={order} {arm:>9s} "
                            f"knob={knob:<8.3g} far={rec['far_pairs']:<7d} "
                            f"rel(med/p90/p99/max)="
                            f"{rec['median']:.1e}/{rec['p90']:.1e}/"
                            f"{rec['p99']:.1e}/{rec['max']:.1e}  "
                            f"dehnen(rms/p90/p99/p9999)="
                            f"{rec['dehnen_rms']:.1e}/{rec['dehnen_p90']:.1e}/"
                            f"{rec['dehnen_p99']:.1e}/{rec['dehnen_p9999']:.1e}",
                            flush=True,
                        )
                metric_prefix = {
                    "relative": "",
                    "scaled": "scaled_",
                    "dehnen": "dehnen_",
                }[args.metric]
                for mass_arm in (a for a in arms if a != "fixed"):
                    for row in compare_arms(
                        by_arm["fixed"],
                        by_arm[mass_arm],
                        metric=metric_prefix,
                        match_on=str(args.match_on),
                    ):
                        row.update(
                            {
                                "distribution": dist,
                                "n": n,
                                "order": order,
                                "mass_arm": mass_arm,
                            }
                        )
                        comparisons.append(row)

    print(
        f"\n=== matched at equal {args.match_on} "
        "(ratio > 1 favours the mass MAC) ==="
    )
    print(
        f"{'dist':>14s} {'N':>7s} {'p':>2s} {'arm':>9s} {'matched':>9s} "
        f"{'p99 x':>7s} {'max x':>7s} {'work x':>7s}"
    )
    for row in comparisons:
        print(
            f"{row['distribution']:>14s} {row['n']:>7d} {row['order']:>2d} "
            f"{row.get('mass_arm', 'mass'):>9s} "
            f"{row['matched_p90']:.3e} {row['p99_ratio'] or float('nan'):7.2f} "
            f"{row['max_ratio'] or float('nan'):7.2f} "
            f"{row['pair_work_ratio'] or float('nan'):7.2f}"
        )

    capped = [r for r in records if r.get("retry_final_caps")]
    if capped:
        worst_q = max(r["retry_final_caps"]["queue_capacity"] for r in capped)
        worst_i = max(r["retry_final_caps"]["interaction_capacity"] for r in capped)
        worst_a = max(r["retry_final_caps"]["attempts"] for r in capped)
        print(
            f"\n=== traversal retries fired on {len(capped)}/{len(records)} configs "
            f"(max {worst_a} attempts) ===\n"
            "Pass these up front to skip the retry-recompile cycle next time:\n"
            "    DualTreeTraversalConfig(\n"
            f"        max_pair_queue={worst_q},\n"
            f"        max_interactions_per_node={worst_i},\n"
            "    )\n"
            "Do not round up -- buffers are num_nodes * interaction_capacity, and "
            "an oversized cap OOMs."
        )

    if args.json_out:
        out = pathlib.Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "meta": {**_git_meta(), "args": vars(args)},
                    "records": records,
                    "comparisons": comparisons,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

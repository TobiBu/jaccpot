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
``fixed``          ``mac_type="dehnen"``, sweep theta. The baseline.
``mass``           ``mac_type="dehnen_error"``, sweep eps. eq (16a) verbatim.
``mass_16b``       eq (16b) with an *exact* O(N^2) ``f_b`` injected. A ceiling, not
                   a production path.
``mass_16b_est``   eq (16b) with the O(N) ``f_b`` estimator
                   (``mac_force_scale_mode="paper_fb"``) -- what production would
                   actually run. Compare it against ``mass_16b`` to see how much of
                   the ceiling survives; each record carries its own
                   ``fb_fidelity`` against the exact sum.

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
from yggdrax.interactions import DualTreeTraversalConfig  # noqa: E402

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
# one measurement
# ---------------------------------------------------------------------------


def _advanced(
    mac_type: str,
    *,
    max_pair_queue: Optional[int] = None,
    max_interactions_per_node: Optional[int] = None,
    runtime_lane: str = "generic",
) -> FMMAdvancedConfig:
    cfg = FMMAdvancedConfig()
    if runtime_lane == "large_n":
        # Trap 10 recorded that every measurement in the status document ran the
        # generic path, because the large-N lane needs `expansion_basis="solidfmm"`
        # *and* `preset="large_n_gpu"` *and* a radix tree, and the bench asked for
        # none of them. It also used to refuse the criterion outright. Both are
        # fixed, so the lane is now selectable -- and `--runtime-lane large_n`
        # asserts it actually engaged rather than trusting it (a silent fallback
        # would reproduce the generic-lane numbers and read as success).
        cfg = replace(cfg, tree=replace(cfg.tree, tree_type="radix"))
    runtime = replace(
        cfg.runtime,
        # Both arms retain the traversal result so both pay the identical loss of
        # the streamed fast lane. Without this the mass arm would be charged for
        # a lane fallback the geometric arm avoids, and the cost comparison would
        # measure plumbing rather than the criterion.
        retain_traversal_result=True,
        retain_interactions=True,
    )
    if max_pair_queue is not None and max_interactions_per_node is not None:
        # Pre-sized traversal buffers, to skip the retry-recompile cycle. Each
        # overflow retry recompiles, which at tight eps was 6 attempts and minutes
        # per config -- indistinguishable from a hang. Feed back the caps a previous
        # run reported as converged, and do NOT round them up: buffers are
        # num_nodes * interaction_capacity, so an oversized cap OOMs (1<<18 at
        # N=16384 tried to allocate 4 GiB on top of 32 and died).
        base = cfg.runtime.traversal_config
        fields: dict[str, Any] = {
            "max_pair_queue": int(max_pair_queue),
            "max_interactions_per_node": int(max_interactions_per_node),
        }
        runtime = replace(
            runtime,
            traversal_config=(
                replace(base, **fields)
                if base is not None
                # process_block is a scheduling knob rather than a capacity, so it
                # is not something the caller should have to pin to pin the caps.
                else DualTreeTraversalConfig(process_block=512, **fields)
            ),
        )
    return replace(cfg, mac_type=mac_type, runtime=runtime)


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
    max_pair_queue: Optional[int] = None,
    max_interactions_per_node: Optional[int] = None,
    reference_targets: Optional[np.ndarray] = None,
    runtime_lane: str = "generic",
) -> dict[str, Any]:
    """Run one (arm, knob) configuration and return its record.

    ``reference_targets`` is the subset of particle indices the reference covers;
    when set, every error statistic is computed over that subset only. The FMM
    itself still runs on all N particles -- only the O(N^2) comparison is
    subsampled.
    """

    caps = dict(
        max_pair_queue=max_pair_queue,
        max_interactions_per_node=max_interactions_per_node,
        runtime_lane=runtime_lane,
    )
    if arm == "fixed":
        kwargs: dict[str, Any] = dict(
            theta=float(knob), advanced=_advanced("dehnen", **caps)
        )
    else:
        kwargs = dict(
            # theta does not gate acceptance in paper mode -- eq (16a) supplies
            # its own `theta < 1` convergence guard -- so it is pinned at 1.0 and
            # eps is the accuracy knob.
            #
            # It does still gate the *prepass* traversal underneath the criterion,
            # which is a separate thing entirely and is why the runtime resolves the
            # prepass angle from `mac_force_scale_prepass_theta` instead of from
            # this. Pinning theta=1.0 here used to hand the prepass an opening angle
            # of 1.0 as a side effect.
            theta=1.0,
            adaptive_eps=float(knob),
            dehnen_geometry_mode=geometry_mode,
            advanced=_advanced("dehnen_error", **caps),
        )
        if arm == "mass_16b_est":
            kwargs["mac_force_scale_mode"] = "paper_fb"
        if theta_max is not None:
            kwargs["mac_theta_max"] = float(theta_max)
    kwargs["G"] = G
    kwargs["softening"] = softening
    if runtime_lane == "large_n":
        kwargs["preset"] = "large_n_gpu"
        kwargs["expansion_basis"] = "solidfmm"

    fmm = FastMultipoleMethod(**kwargs)

    # `_lane_probe` is imported here rather than at module scope: it imports nothing
    # from this module, but this module is what `far_pair_census` imports, and keeping
    # the edge one-directional avoids a cycle if that ever reverses.
    from bench.validation._lane_probe import DualBuildProbe

    t0 = time.perf_counter()
    with DualBuildProbe() as dual_probe:
        state = fmm.prepare_state(
            positions, masses, leaf_size=leaf_size, max_order=order
        )
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
        with DualBuildProbe() as dual_probe:
            state = fmm.prepare_state(
                positions,
                masses,
                leaf_size=leaf_size,
                max_order=order,
                force_scale_nodes=f_b_nodes,
            )
    prepare_s = time.perf_counter() - t0

    declined_reason = fmm.get_runtime_diagnostics().get("large_n_path_declined_reason")
    if runtime_lane == "large_n":
        # Assert rather than hope. A silent fallback to the generic lane would
        # reproduce the old numbers exactly and look like a successful large-N run.
        if declined_reason is not None:
            raise RuntimeError(
                f"--runtime-lane large_n was requested but the lane declined: "
                f"{declined_reason!r} (arm={arm}, knob={knob})"
            )
        if type(state).__name__ != "LargeNPreparedState":
            raise RuntimeError(
                "--runtime-lane large_n was requested but prepare_state returned "
                f"{type(state).__name__}, not LargeNPreparedState "
                f"(arm={arm}, knob={knob})"
            )
    if arm != "fixed":
        # The criterion has to be *installed*, not merely requested: a build with no
        # pair policy runs the geometric MAC, and a policy state built from a None
        # force scale compares against eps*1. Both are cheaper and silent.
        dual_probe.check_criterion_was_applied(context=f"arm={arm} knob={knob:g}")

    t0 = time.perf_counter()
    acc = fmm.evaluate_prepared_state(state, return_potential=False)
    jax.block_until_ready(acc)
    evaluate_s = time.perf_counter() - t0

    # Under subsampling the reference and force scale cover only the measured
    # targets, so the FMM result has to be restricted to the same rows -- in the
    # same order -- before any error is taken.
    acc_measured = acc if reference_targets is None else acc[reference_targets]
    errors = per_particle_relative_error(acc_measured, reference)
    scaled_errors = per_particle_scaled_error(acc_measured, reference)
    dehnen_errors = per_particle_dehnen_scaled_error(
        acc_measured, reference, force_scale
    )

    node_ranges = np.asarray(state.tree.node_ranges)
    # The accept mask comes from the dual-build probe, not from `state.interactions`.
    # On the large-N lane the production contract pins `retain_interactions=False`, so
    # reading the state there reports **zero** far pairs -- which then trips every
    # `--min-far-pairs` guard and reads as "this configuration measures nothing" when
    # what is actually missing is the measurement plumbing. See `_lane_probe`.
    far_pairs = int(dual_probe.final.get("far_pairs", -1))
    if far_pairs < 0:
        interactions = state.interactions
        far_pairs = 0
        if interactions is not None:
            src = np.asarray(interactions.sources)
            tgt = np.asarray(interactions.targets)
            far_pairs = int(((src >= 0) & (tgt >= 0)).sum())
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

    # How much of the exact-f_b ceiling the O(N) estimator actually retains. The
    # eq (16b) gain was measured with an exact O(N^2) f_b, which is a ceiling and
    # not a prediction, so an estimator arm that does not report its own fidelity
    # cannot distinguish "the criterion is worse than hoped" from "the estimator
    # is". Recorded per config because it depends on the prepass traversal, and so
    # on the tree.
    fb_fidelity = None
    estimated_fb = getattr(fmm._impl, "_last_force_scale_particles", None)
    # Fidelity needs the exact f_b for every particle to line up with the
    # estimator's per-particle output; under subsampling `force_scale` covers only
    # the measured targets, so scoring it would silently compare mismatched rows.
    if estimated_fb is not None and reference_targets is None:
        est = np.asarray(estimated_fb, dtype=np.float64)
        exact = np.asarray(force_scale, dtype=np.float64)[
            np.asarray(state.tree.particle_indices)
        ]
        ratio = est / np.maximum(exact, np.finfo(np.float64).tiny)
        fb_fidelity = {
            "median": float(np.median(ratio)),
            "p01": float(np.quantile(ratio, 0.01)),
            "p99": float(np.quantile(ratio, 0.99)),
            "min": float(ratio.min()),
            "max": float(ratio.max()),
            "frac_above_one": float((ratio > 1.0 + 1e-9).mean()),
        }

    record = {
        "arm": arm,
        "knob": float(knob),
        "runtime_lane": runtime_lane,
        "prepared_state_type": type(state).__name__,
        "large_n_path_declined_reason": declined_reason,
        "fb_fidelity": fb_fidelity,
        "retry_final_caps": final_caps,
        "far_pairs": far_pairs,
        "near_pairs": near_pairs,
        "far_work": int(far_work),
        "near_work": int(near_work),
        "pair_work": int(far_work + near_work),
        "prepare_s": prepare_s,
        "evaluate_s": evaluate_s,
        # Over the measured targets only when subsampling -- so under subsampling
        # this is a rel-L2 of the sample, not of the system.
        "rel_l2": float(
            np.linalg.norm(np.asarray(acc_measured) - np.asarray(reference))
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
    min_far_pairs: int = 0,
    max_p9999: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Compare the arms at a matched error statistic.

    ``max_p9999`` drops configs whose 99.99th-percentile error exceeds it. An
    absolute error at or above 1 is a *diverged* expansion, not a coarse point on
    the same accuracy curve, so interpolating through it is meaningless (trap 3).
    Like ``min_far_pairs`` this stops being optional once leaf_size varies: at
    N=1e5/leaf 256 the fixed arm reaches p99.99 = 1.0e+01 at theta=0.78 and
    2.9e+02 at theta=0.90.


    ``min_far_pairs`` drops configs with fewer than that many accepted far pairs
    from the interpolation basis entirely. A config that accepts (almost) no far
    field is pure near-field direct summation: its error sits at machine precision,
    which *widens* the arm's apparent range and lets the matched target be
    log-interpolated straight across the far-field switch-on. That is trap 8, and it
    has produced p99 ratios of 34.8 and 2.9e7 out of pure round-off. It is not
    hypothetical at large leaf sizes -- measured at N=1e5, theta=0.30 accepts 63 866
    far pairs at leaf 64, **48** at leaf 128 and **0** at leaf 256 -- so any sweep
    that varies leaf_size needs this on. Left at 0 by default so existing runs are
    unchanged.

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
    if min_far_pairs > 0:
        dropped_fixed = [r for r in fixed if r.get("far_pairs", 0) < min_far_pairs]
        dropped_mass = [r for r in mass if r.get("far_pairs", 0) < min_far_pairs]
        fixed = [r for r in fixed if r.get("far_pairs", 0) >= min_far_pairs]
        mass = [r for r in mass if r.get("far_pairs", 0) >= min_far_pairs]
        if dropped_fixed or dropped_mass:
            # Announced, not silent: a dropped row is a knob value that measured
            # nothing, and knowing which ones went is how you tell "the grid is too
            # coarse here" from "the grid is fine and one endpoint was degenerate".
            print(
                f"    compare_arms: dropped {len(dropped_fixed)} fixed + "
                f"{len(dropped_mass)} mass config(s) with < {min_far_pairs} far "
                "pairs (all-near-field, error at machine precision)"
                + (
                    "  fixed knobs: "
                    + ",".join(f"{r['knob']:g}" for r in dropped_fixed)
                    if dropped_fixed
                    else ""
                ),
                flush=True,
            )
    if max_p9999 is not None:
        key = f"{metric}p9999"
        diverged_fixed = [r for r in fixed if r.get(key, 0.0) > max_p9999]
        diverged_mass = [r for r in mass if r.get(key, 0.0) > max_p9999]
        fixed = [r for r in fixed if r.get(key, 0.0) <= max_p9999]
        mass = [r for r in mass if r.get(key, 0.0) <= max_p9999]
        if diverged_fixed or diverged_mass:
            # Trap 3: an absolute error approaching or exceeding 1 is not a large
            # truncation error, it is a diverged expansion -- the multipole series
            # evaluated outside its region of convergence. Such a row is not a
            # coarser point on the same accuracy curve, so interpolating through it
            # is meaningless. Measured at N=1e5/leaf 256: the fixed arm reaches
            # p99.99 = 1.0e+01 at theta=0.78 and 2.9e+02 at theta=0.90.
            print(
                f"    compare_arms: dropped {len(diverged_fixed)} fixed + "
                f"{len(diverged_mass)} mass config(s) with {key} > {max_p9999:g} "
                "(diverged expansion, not truncation -- trap 3)"
                + (
                    "  fixed knobs: "
                    + ",".join(f"{r['knob']:g}" for r in diverged_fixed)
                    if diverged_fixed
                    else ""
                ),
                flush=True,
            )
    if len(fixed) < 2 or len(mass) < 2:
        print(
            f"    compare_arms: only {len(fixed)} fixed / {len(mass)} mass config(s) "
            "survived the guards -- no comparison. Widen the knob grid for this "
            "configuration rather than relaxing the guards.",
            flush=True,
        )
        return out
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
    # rms and p99.99 are the two statistics Dehnen quotes ("the rms error is always
    # ten times smaller"), so they are reported as ratios alongside p99/max rather
    # than left for whoever post-processes the JSON to rediscover.
    error_fields = ("rms", "p9999", "p99", "max", "median")
    for index, target in enumerate(np.exp(np.linspace(np.log(lo), np.log(hi), 5))):
        row: dict[str, Any] = {"matched_p90": float(target), "match_index": index}
        ok = True
        for label, records in (("fixed", fixed), ("mass", mass)):
            for name in error_fields + ("pair_work", "far_pairs"):
                field = f"{metric}{name}" if name in error_fields else name
                val = log_interp_at(
                    records, target_p90=float(target), field=field, p90_key=p90
                )
                if val is None:
                    ok = False
                row[f"{label}_{name}"] = val
        if not ok:
            continue
        for name in error_fields:
            mass_val = row[f"mass_{name}"]
            row[f"{name}_ratio"] = row[f"fixed_{name}"] / mass_val if mass_val else None
        row["pair_work_ratio"] = (
            row["fixed_pair_work"] / row["mass_pair_work"]
            if row["mass_pair_work"]
            else None
        )
        out.append(row)
    return out


def aggregate_over_seeds(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-seed matched rows into a median and a min/max spread.

    Joined on ``match_index`` -- the position in the five log-spaced matched
    targets -- rather than on the target value, because each seed's usable range
    differs slightly so the absolute targets never coincide.

    The spread is the point of this: the N-scaling trend was previously read off
    one seed at each end of the ladder, where a decaying advantage and a flat one
    are indistinguishable.
    """

    keys = ("rms_ratio", "p9999_ratio", "p99_ratio", "max_ratio", "pair_work_ratio")
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in comparisons:
        key = (
            row.get("distribution"),
            row.get("n"),
            row.get("order"),
            row.get("mass_arm"),
            row.get("match_index"),
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for (dist, n, order, arm, index), rows in sorted(
        grouped.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0, kv[0][4] or 0)
    ):
        agg: dict[str, Any] = {
            "distribution": dist,
            "n": n,
            "order": order,
            "mass_arm": arm,
            "match_index": index,
            "seeds": sorted({r.get("seed") for r in rows if r.get("seed") is not None}),
            "n_seeds": len(rows),
            "matched_p90_median": float(np.median([r["matched_p90"] for r in rows])),
        }
        for key in keys:
            vals = [r[key] for r in rows if r.get(key)]
            if not vals:
                agg[f"{key}_median"] = None
                agg[f"{key}_min"] = None
                agg[f"{key}_max"] = None
                continue
            agg[f"{key}_median"] = float(np.median(vals))
            agg[f"{key}_min"] = float(np.min(vals))
            agg[f"{key}_max"] = float(np.max(vals))
        out.append(agg)
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
    ap.add_argument(
        "--seed",
        default="0",
        help=(
            "comma-separated seeds. More than one turns on the cross-seed "
            "aggregate: matched rows are joined on their position in the matched "
            "ladder and reported as median [min, max]. Use >= 3 for any claim "
            "about a trend -- the N-scaling question was open for a session "
            "because it rested on one seed at each end with no spread."
        ),
    )
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
    ap.add_argument(
        "--max-p9999",
        type=float,
        default=None,
        help=(
            "drop configs whose 99.99th-percentile error exceeds this from the "
            "matched comparison. An absolute error at or above 1 is a diverged "
            "expansion rather than a coarse point on the same curve (trap 3), so "
            "interpolating through it is meaningless. Use 1.0. Like "
            "--min-far-pairs, required once leaf_size varies: at N=1e5/leaf 256 the "
            "fixed arm reaches p99.99 = 10 at theta=0.78 and 290 at theta=0.90."
        ),
    )
    ap.add_argument(
        "--min-far-pairs",
        type=int,
        default=0,
        help=(
            "drop configs accepting fewer than this many far pairs from the matched "
            "comparison. A config with (almost) no far field is direct summation: "
            "its error is machine precision, which widens the arm's apparent range "
            "and lets the matched target interpolate across the far-field switch-on "
            "(trap 8, which has produced ratios of 34.8 and 2.9e7 out of round-off). "
            "REQUIRED for any sweep that varies leaf_size -- at N=1e5, theta=0.30 "
            "accepts 63866 far pairs at leaf 64, 48 at leaf 128 and 0 at leaf 256."
        ),
    )
    ap.add_argument(
        "--reference-subsample",
        type=int,
        default=None,
        help=(
            "evaluate the O(N^2) reference for only this many randomly chosen "
            "targets (against ALL sources), which is what makes N=1e6 tractable: "
            "1e4 targets x 1e6 sources is 1e10 pairs, not 1e12. The FMM still runs "
            "on every particle; only the comparison is subsampled. Costs tail "
            "resolution -- p99.99 of K targets is K/1e4 particles, so K=1e4 leaves "
            "the headline statistic resting on ONE particle and K>=1e5 is needed to "
            "quote it. Omit below N~3e5, where keeping all targets is affordable."
        ),
    )
    ap.add_argument(
        "--max-pair-queue",
        type=int,
        default=None,
        help=(
            "pre-size the traversal pair queue, skipping the retry-recompile cycle. "
            "Use the value a previous run reported as converged; do not round up."
        ),
    )
    ap.add_argument(
        "--max-interactions-per-node",
        type=int,
        default=None,
        help=(
            "pre-size the per-node interaction list. Buffers are num_nodes * this, "
            "so an oversized value OOMs -- pass the converged value verbatim."
        ),
    )
    ap.add_argument(
        "--runtime-lane",
        choices=("generic", "large_n"),
        default="generic",
        help=(
            "Which runtime lane to measure on. 'large_n' selects "
            "preset='large_n_gpu' + expansion_basis='solidfmm' + a radix tree, and "
            "ASSERTS the lane engaged (get_runtime_diagnostics()"
            "['large_n_path_declined_reason'] is None). Trap 10: every number in "
            "the status document before this flag existed ran the generic path, "
            "because the bench asked for none of those three things. A silent "
            "fallback would reproduce the generic-lane numbers and read as success."
        ),
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    known_arms = ("fixed", "mass", "mass_16b", "mass_16b_est")
    arms = tuple(a.strip() for a in str(args.arm).split(",") if a.strip())
    unknown = [a for a in arms if a not in known_arms]
    if unknown:
        ap.error(f"unknown --arm value(s) {unknown}; choose from {list(known_arms)}")
    if "fixed" not in arms:
        # compare_arms measures every mass arm against the geometric baseline, so
        # dropping it would silently produce an empty comparison table.
        ap.error("--arm must include 'fixed'; it is the comparison baseline")

    if args.reference_subsample is not None:
        if int(args.reference_subsample) < 1:
            ap.error("--reference-subsample must be >= 1")
        if "mass_16b" in arms:
            # The exact-f_b arm injects a per-node force scale reduced from f_b for
            # *every* particle. A subsampled f_b would reduce to a scale built from
            # a tenth of a percent of the system and the arm would still run, just
            # measuring something else entirely.
            ap.error(
                "--arm mass_16b needs the exact f_b for every particle (it feeds "
                "the node reduction), which --reference-subsample does not compute. "
                "Use --arm mass_16b_est, whose estimator is O(N) and needs no "
                "reference at all."
            )
        smallest = min(_ints(args.n))
        if int(args.reference_subsample) >= smallest:
            print(
                f"NOTE: --reference-subsample {args.reference_subsample} >= N="
                f"{smallest}; that tier will use all targets and pay full O(N^2).",
                flush=True,
            )
        elif int(args.reference_subsample) < 100_000:
            print(
                f"WARNING: --reference-subsample {args.reference_subsample} puts "
                f"p99.99 at {int(args.reference_subsample) / 10_000:.1f} particles. "
                "Dehnen's headline statistic is the 99.99th percentile; below "
                "~1e5 targets it is not resolved and only rms/p99 are quotable.",
                flush=True,
            )

    if (args.max_pair_queue is None) != (args.max_interactions_per_node is None):
        # Pinning one and leaving the other to grow still pays the recompile per
        # retry, so the run would look pinned while behaving exactly as before.
        ap.error(
            "--max-pair-queue and --max-interactions-per-node must be given "
            "together; pinning only one leaves the other in the retry cycle"
        )

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

    metric_prefix = {"relative": "", "scaled": "scaled_", "dehnen": "dehnen_"}[
        args.metric
    ]

    def run_case(dist: str, n: int, seed: int) -> None:
        """Sweep every arm over one (distribution, N, seed) realisation."""

        pos_np, mass_np = make_distribution(dist, n, seed)
        positions = jnp.asarray(pos_np, dtype=jnp.float64)
        masses = jnp.asarray(mass_np, dtype=jnp.float64)

        reference_targets = None
        if args.reference_subsample is not None and int(args.reference_subsample) < n:
            # Derived from the realisation seed so the target set is reproducible
            # and differs between seeds, rather than fixing the same targets for
            # every realisation (which would correlate the seeds it exists to
            # average over).
            reference_targets = np.sort(
                np.random.default_rng(1_000_003 + seed).choice(
                    n, size=int(args.reference_subsample), replace=False
                )
            ).astype(np.int32)

        reference = chunked_direct_accelerations(
            positions,
            masses,
            softening=args.softening,
            G=args.G,
            block=int(args.reference_block),
            targets=reference_targets,
        )
        jax.block_until_ready(reference)
        force_scale = chunked_force_scale(
            positions,
            masses,
            softening=args.softening,
            G=args.G,
            block=int(args.reference_block),
            targets=reference_targets,
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
                        max_pair_queue=args.max_pair_queue,
                        max_interactions_per_node=args.max_interactions_per_node,
                        reference_targets=reference_targets,
                        runtime_lane=args.runtime_lane,
                    )
                    rec.update(
                        {
                            "distribution": dist,
                            "n": n,
                            "seed": seed,
                            "order": order,
                            "leaf_size": args.leaf_size,
                            "measured_targets": (
                                n
                                if reference_targets is None
                                else int(reference_targets.shape[0])
                            ),
                        }
                    )
                    by_arm[arm].append(rec)
                    records.append(rec)
                    fid = rec.get("fb_fidelity")
                    fid_text = (
                        f"  fb_est/exact med={fid['median']:.3f} min={fid['min']:.3f}"
                        if fid
                        else ""
                    )
                    print(
                        f"{dist:>14s} N={n:<7d} s={seed} p={order} {arm:>12s} "
                        f"knob={knob:<8.3g} far={rec['far_pairs']:<7d} "
                        f"rel(med/p90/p99/max)="
                        f"{rec['median']:.1e}/{rec['p90']:.1e}/"
                        f"{rec['p99']:.1e}/{rec['max']:.1e}  "
                        f"dehnen(rms/p90/p99/p9999)="
                        f"{rec['dehnen_rms']:.1e}/{rec['dehnen_p90']:.1e}/"
                        f"{rec['dehnen_p99']:.1e}/{rec['dehnen_p9999']:.1e}"
                        f"{fid_text}",
                        flush=True,
                    )
            for mass_arm in (a for a in arms if a != "fixed"):
                for row in compare_arms(
                    by_arm["fixed"],
                    by_arm[mass_arm],
                    metric=metric_prefix,
                    match_on=str(args.match_on),
                    min_far_pairs=int(args.min_far_pairs),
                    max_p9999=args.max_p9999,
                ):
                    row.update(
                        {
                            "distribution": dist,
                            "n": n,
                            "seed": seed,
                            "order": order,
                            "mass_arm": mass_arm,
                        }
                    )
                    comparisons.append(row)

    seeds = _ints(args.seed)
    for dist_name in str(args.distribution).split(","):
        dist_name = dist_name.strip()
        if not dist_name:
            continue
        for n_val in _ints(args.n):
            for seed_val in seeds:
                run_case(dist_name, n_val, seed_val)

    def _num(value: Optional[float]) -> float:
        return float("nan") if not value else float(value)

    print(
        f"\n=== matched at equal {args.match_on} "
        "(ratio > 1 favours the mass MAC) ==="
    )
    # rms and p99.99 lead: they are what Dehnen quotes, and the honest headline is
    # the tail, not p99. Reporting p99 alone understated the effect by more than an
    # order of magnitude.
    print(
        f"{'dist':>12s} {'N':>7s} {'s':>2s} {'p':>2s} {'arm':>12s} {'matched':>9s} "
        f"{'rms x':>7s} {'p9999 x':>8s} {'p99 x':>7s} {'max x':>7s} {'work x':>7s}"
    )
    for row in comparisons:
        print(
            f"{row['distribution']:>12s} {row['n']:>7d} {row.get('seed', 0):>2d} "
            f"{row['order']:>2d} {row.get('mass_arm', 'mass'):>12s} "
            f"{row['matched_p90']:.3e} {_num(row.get('rms_ratio')):7.2f} "
            f"{_num(row.get('p9999_ratio')):8.2f} {_num(row.get('p99_ratio')):7.2f} "
            f"{_num(row['max_ratio']):7.2f} {_num(row['pair_work_ratio']):7.2f}"
        )

    aggregates: list[dict[str, Any]] = []
    if len(seeds) > 1:
        aggregates = aggregate_over_seeds(comparisons)
        print(
            f"\n=== across {len(seeds)} seeds: median [min, max] "
            f"(matched at equal {args.match_on}) ==="
        )
        print(
            f"{'dist':>12s} {'N':>7s} {'p':>2s} {'arm':>12s} {'#':>2s} "
            f"{'matched':>9s} {'rms x':>21s} {'p9999 x':>21s} {'work x':>21s}"
        )
        for agg in aggregates:

            def band(key: str, agg: dict[str, Any] = agg) -> str:
                med = agg.get(f"{key}_median")
                if med is None:
                    return f"{'n/a':>21s}"
                return (
                    f"{med:7.2f} [{agg[f'{key}_min']:6.2f},"
                    f"{agg[f'{key}_max']:6.2f}]"
                )

            print(
                f"{agg['distribution']:>12s} {agg['n']:>7d} {agg['order']:>2d} "
                f"{agg['mass_arm']:>12s} {agg['n_seeds']:>2d} "
                f"{agg['matched_p90_median']:.3e} {band('rms_ratio')} "
                f"{band('p9999_ratio')} {band('pair_work_ratio')}"
            )

    fidelity = [r for r in records if r.get("fb_fidelity")]
    if fidelity:
        meds = [r["fb_fidelity"]["median"] for r in fidelity]
        mins = [r["fb_fidelity"]["min"] for r in fidelity]
        above = max(r["fb_fidelity"]["frac_above_one"] for r in fidelity)
        print(
            f"\n=== O(N) f_b estimator vs exact O(N^2) f_b, over "
            f"{len(fidelity)} configs ===\n"
            f"    ratio median: {min(meds):.4f} .. {max(meds):.4f}   "
            f"worst single particle: {min(mins):.4f}\n"
            f"    fraction above 1 (i.e. not a lower bound): {above:.2e}  "
            "-- must be 0 while mac_force_scale_fb_inflation >= 1"
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
                    "seed_aggregates": aggregates,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

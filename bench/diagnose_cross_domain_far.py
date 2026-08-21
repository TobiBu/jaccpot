"""Why the distributed cross-domain FAR field is wrong, measured from first inputs.

WHY THIS EXISTS. ``tests/distributed/`` is gated on ``device_count() >= 2``, so it
skipped in CPU CI and in every local run until it was first executed on two cards on
2026-08-21 (audit row F34: `distributed/fmm.py` 19% -> 81% coverage). Four of the five
failures it produced are one defect in the cross-domain far field, and this script is
the evidence trail for it -- see ``docs/distributed_cross_domain_far_diagnosis.md``.

It merges two diagnostics: the omission baseline from #190 (*an approximation
cannot be worse than omission*, with its domain masking corrected -- see
:func:`omission_baseline`) and the single-device isolation that localises the
error to the MAC extents rather than to the M2L.

THE DEFECT. ``build_coarse_frontier`` reduces every remote leaf to a single point (its
centre of mass), ``build_remote_coarse_tree`` builds the coarse tree over those points,
and its geometry -- hence the MAC extent the cross walk tests -- therefore bounds the
**centres of mass**, not the particles behind them. A coarse *leaf* gets an extent of
~0 no matter how large the remote leaf it stands for actually is. The cross-domain MAC
consequently accepts pairs whose true separation is smaller than the source's own
radius, and the M2L is evaluated inside the region it is expanding. The error then
**exceeds the term being approximated**, which is why dropping the cross far field
entirely is more accurate than computing it.

Measured, ndev=2, N=128, order 3, dehnen, leaf 8 (the failing default):

    worst accepted far pair, true (r_src + r_tgt) / d   1.193
    the same ratio as the MAC computed it               0.104     <- 11x understated
    largest "leaf" radius the coarse tree sees as 0     5.047

CORRECTING THE EXTENTS IS WHAT FIXES IT, not the order, the basis or the rotation:
inflating the coarse extents to bound the true remote leaves takes the cross-field
error from 6.4e-01 to 8e-06 at theta_cross=1.0 while still accepting 10 far pairs.
``--inflate`` reruns any configuration with that correction applied, which is how that
number is produced. The correction itself belongs in yggdrax (``CoarseFrontier`` has to
carry the leaf radius), so nothing here is shipped in the force path.

NO GPU NEEDED. The failure is geometric, not hardware-dependent: two forced CPU
devices reproduce the two-A100 aggL2 to six digits (10.090412 vs 10.090450, 0.260355,
0.018223, 0.000003). That is the default here, so this runs anywhere in ~2 min.

Run::

    python bench/diagnose_cross_domain_far.py                 # everything, 2 CPU devices
    python bench/diagnose_cross_domain_far.py --skip-driver   # geometry only, seconds
    python bench/diagnose_cross_domain_far.py --inflate       # with the correction
    python bench/diagnose_cross_domain_far.py --gpu           # claim 2 cards instead
"""

from __future__ import annotations

import argparse
import os
import sys

_PARSER = argparse.ArgumentParser(description=__doc__)
_PARSER.add_argument(
    "--gpu",
    action="store_true",
    help="claim two GPUs with autocvd instead of forcing two CPU devices",
)
_PARSER.add_argument(
    "--skip-driver",
    action="store_true",
    help="skip the theta_cross sweep through the real driver (the slow part)",
)
_PARSER.add_argument(
    "--inflate",
    action="store_true",
    help="inflate the coarse MAC extents to bound the true remote leaves",
)
_PARSER.add_argument(
    "--theta-cross",
    type=float,
    nargs="+",
    default=[1e6, 1.0, 0.1, 0.01],
    help="theta_cross values to sweep (default: 1e6 1.0 0.1 0.01)",
)
_PARSER.add_argument("--per", type=int, default=64, help="particles per device")
_ARGS = _PARSER.parse_args()

# Device selection must precede `import jax` (org rule; ARCHITECTURE.md 7).
if _ARGS.gpu:
    from autocvd import autocvd

    autocvd(num_gpus=2, least_used=True)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".30")
else:
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
    ).strip()

import dataclasses  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from yggdrax.distributed import device_count, make_mesh  # noqa: E402
from yggdrax.distributed.cross_walk import dual_tree_walk_cross_impl  # noqa: E402
from yggdrax.distributed.let import build_coarse_frontier  # noqa: E402
from yggdrax.dtypes import INDEX_DTYPE  # noqa: E402
from yggdrax.geometry import compute_tree_geometry  # noqa: E402
from yggdrax.tree import (  # noqa: E402
    Tree,
    get_level_offsets,
    get_node_levels,
    get_nodes_by_level,
)
from yggdrax.tree_moments import compute_tree_mass_moments  # noqa: E402

from jaccpot.distributed import (  # noqa: E402
    DistributedFMMConfig,
    distributed_fmm_accelerations,
)
from jaccpot.distributed.fmm import partition_for_devices  # noqa: E402
from jaccpot.downward.local_expansions import LocalExpansionData  # noqa: E402
from jaccpot.operators.real_harmonics import sh_size  # noqa: E402
from jaccpot.runtime.kernels.core import (  # noqa: E402
    _apply_real_m2l,
    _evaluate_local_expansions_for_particles,
    _propagate_solidfmm_locals_by_level,
)
from jaccpot.upward.real_tree_expansions import (  # noqa: E402
    aggregate_m2m_real_by_level,
    prepare_real_upward_sweep,
)
from jaccpot.upward.tree_geometry import compute_tree_geometry_compiled  # noqa: E402

ORDER = 3
LEAF = 8
G = 1.0
SOFTENING = 0.02
MAC = "dehnen"
ROTATION = "solidfmm"


def direct_sum(targets, sources, source_masses):
    """Exact acceleration on ``targets`` from ``(sources, source_masses)``.

    Parameters
    ----------
    targets : Array
        Target positions ``[T, 3]``.
    sources : Array
        Source positions ``[S, 3]``.
    source_masses : Array
        Source masses ``[S]``.

    Returns
    -------
    Array
        Accelerations ``[T, 3]`` with the shared ``G`` and ``SOFTENING``.
    """
    diff = targets[:, None, :] - sources[None, :, :]
    d2 = (diff**2).sum(-1) + SOFTENING**2
    inv = d2 ** (-1.5)
    return -G * (source_masses[None, :, None] * diff * inv[..., None]).sum(axis=1)


def separated_clusters(ndev, per, seed=4):
    """The failing tests' IC: ``ndev`` uniform balls of radius 0.5, spaced 6 apart.

    Parameters
    ----------
    ndev : int
        Number of clusters (one per device, as the tests intend).
    per : int
        Particles per cluster.
    seed : int
        RNG seed; 4 is the value the failing tests use.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions ``[ndev * per, 3]`` and masses ``[ndev * per]``, float32.
    """
    rng = np.random.default_rng(seed)
    cluster_centers = np.array(
        [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[:ndev]
    pts = np.concatenate(
        [cluster_centers[d] + rng.uniform(-0.5, 0.5, (per, 3)) for d in range(ndev)]
    ).astype(np.float32)
    mass = rng.uniform(0.5, 2.0, size=(per * ndev,)).astype(np.float32)
    return pts, mass


def omission_baseline(positions, masses, domain, softening=SOFTENING, newton_g=G):
    """Relative error from dropping the cross-domain term entirely.

    From the version of this script that landed in #190, which introduced the idea:
    accelerations are linear in the pair contributions, so masking same-domain pairs
    out of a direct sum gives the cross-domain term exactly, and *an approximation
    cannot be worse than omission*.

    CORRECTED HERE. That version took each particle's domain to be ``i // per`` --
    the INPUT order -- but ``partition_for_devices`` Morton-sorts before splitting,
    so on the separated-cluster IC that masks by *cluster* instead of by *domain*.
    Measured at ndev=2: 0.008814 by input order against **0.426192** by the real
    assignment, 48x apart, because the Morton domains interpenetrate (see
    :func:`report_domain_geometry`). ``domain`` is therefore passed in from the
    partitioner rather than assumed.

    READ THE RESULT CAREFULLY. This bounds the whole cross-domain term, while the
    far path handles only the part the MAC accepts -- the rest goes through the
    (exact) halo import. So an aggL2 comfortably under this baseline does NOT clear
    the far field: at the old theta_cross=0.1 the driver's 0.018223 is 0.043x this
    baseline, yet the far term was off by more than 100% of its own direct sum.
    :func:`isolate_cross_far` is what settles that, by comparing the far term
    against the direct sum over exactly the pairs it approximates.

    Parameters
    ----------
    positions : np.ndarray
        Positions ``[N, 3]`` in input order.
    masses : np.ndarray
        Masses ``[N]``.
    domain : np.ndarray
        Domain index per particle, in the same order as ``positions``.
    softening : float
        Plummer softening length.
    newton_g : float
        Gravitational constant.

    Returns
    -------
    float
        ``||a_cross|| / ||a_full||``, in float64.
    """
    pos = np.asarray(positions, np.float64)
    mass = np.asarray(masses, np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    inv = ((diff**2).sum(-1) + float(softening) ** 2) ** (-1.5)
    same = (np.asarray(domain)[:, None] == np.asarray(domain)[None, :]).astype(
        np.float64
    )
    weighted = mass[None, :] * inv
    a_full = -float(newton_g) * (weighted[..., None] * diff).sum(axis=1)
    a_self = -float(newton_g) * ((weighted * same)[..., None] * diff).sum(axis=1)
    return float(np.linalg.norm(a_full - a_self) / (np.linalg.norm(a_full) + 1e-30))


def domains_of(positions, masses, ndev, leaf_size=LEAF):
    """Which Morton domain each particle actually lands in.

    Parameters
    ----------
    positions : np.ndarray
        Positions ``[N, 3]`` in input order.
    masses : np.ndarray
        Masses ``[N]``.
    ndev : int
        Device count.
    leaf_size : int
        Leaf size the partitioner sizes its capacity from.

    Returns
    -------
    np.ndarray
        Domain index per particle, in input order.
    """
    part = partition_for_devices(positions, masses, ndev, leaf_size=leaf_size)
    gid = part["gid_flat"].reshape(ndev, part["cap"])
    domain = np.full(positions.shape[0], -1, np.int64)
    for d in range(ndev):
        live = gid[d][gid[d] >= 0]
        domain[live] = d
    assert (domain >= 0).all(), "a particle was not assigned to any domain"
    return domain


def report_domain_geometry(per, ndev):
    """Print which true cluster each Morton domain actually receives.

    The failing tests' docstrings claim "one cluster per Morton domain". That holds at
    ndev=4 and fails at ndev=2 and 3: the global box is 7x1x1, so the leading bits of
    the Morton code are not the axis that separates the clusters, and the median split
    cuts across them. Interpenetrating domains are what expose the extent defect.

    Parameters
    ----------
    per : int
        Particles per cluster.
    ndev : int
        Device count to report for.

    Returns
    -------
    None
        Prints a table.
    """
    print(f"\n=== Morton domains vs true clusters (per={per}) ===")
    for nd in sorted({2, 3, 4} | {ndev}):
        pts, mass = separated_clusters(nd, per)
        part = partition_for_devices(pts, mass, nd, leaf_size=LEAF)
        gid = part["gid_flat"].reshape(nd, part["cap"])
        rows = []
        for d in range(nd):
            g = gid[d][gid[d] >= 0]
            rows.append(np.bincount(g // per, minlength=nd))
        clean = all(int((r > 0).sum()) == 1 for r in rows)
        print(
            f"  ndev={nd}: "
            + "  ".join(f"dev{d}={r.tolist()}" for d, r in enumerate(rows))
            + ("   <- one cluster per domain" if clean else "   <- INTERPENETRATING")
        )


def driver_theta_cross_sweep(ndev, per, thetas):
    """Sweep ``theta_cross`` through the real driver and report aggL2 vs direct.

    Caps are grown for every run so a buffer overflow can never be mistaken for a
    numerical result. ``theta_cross -> 0`` sends every cross pair through the halo
    import and P2P (no expansion, so exact); ``theta_cross -> inf`` sends every cross
    pair through the coarse M2L.

    Parameters
    ----------
    ndev : int
        Device count.
    per : int
        Particles per device.
    thetas : list[float]
        ``theta_cross`` values to sweep.

    Returns
    -------
    None
        Prints one row per value.
    """
    mesh = make_mesh(ndev)
    pts, mass = separated_clusters(ndev, per)
    base = DistributedFMMConfig()
    direct = np.asarray(
        direct_sum(jnp.asarray(pts), jnp.asarray(pts), jnp.asarray(mass))
    )
    norm = np.linalg.norm(direct)
    roomy = dict(
        max_interactions_per_node=4096,
        max_neighbors_per_leaf=4096,
        max_pair_queue=131072,
        cross_max_interactions_per_node=4096,
        cross_max_neighbors_per_leaf=4096,
        cross_max_pair_queue=131072,
    )
    baseline = omission_baseline(pts, mass, domains_of(pts, mass, ndev))
    print(f"\n=== driver theta_cross sweep (ndev={ndev}, N={ndev * per}) ===")
    print(
        f"omission baseline ||a_cross||/||a_full|| = {baseline:.6f}  "
        "(masked by the REAL Morton domain; see omission_baseline)"
    )
    print(
        f"{'theta_cross':>12} {'aggL2':>12} {'/baseline':>10} {'overflow':>9}"
        "  cross far / near"
    )
    for theta_cross in thetas:
        cfg = dataclasses.replace(base, theta_cross=float(theta_cross), **roomy)
        res = distributed_fmm_accelerations(pts, mass, config=cfg, mesh=mesh, jit=False)
        err = float(np.linalg.norm(res.accelerations - direct) / (norm + 1e-30))
        far = np.asarray(res.diagnostics["cross_far_pairs"]).sum()
        near = np.asarray(res.diagnostics["cross_near_pairs"]).sum()
        ratio = err / baseline if baseline else float("inf")
        print(
            f"{theta_cross:>12g} {err:>12.6f} {ratio:>9.3f}x "
            f"{str(bool(res.overflow)):>9}  {far:.0f} / {near:.0f}",
            flush=True,
        )
    print(
        "  a row above 1.00x is unambiguously a bug (worse than dropping the term);\n"
        "  a row below it is NOT a clean bill of health -- the far path handles only\n"
        "  the accepted fraction of that term. The isolation below is the real test."
    )


class _DomainView:
    """One device's local tree, upward sweep, geometry and coarse frontier."""

    def __init__(self, positions, masses, bounds):
        self.tree = Tree.from_particles(
            jnp.asarray(positions),
            jnp.asarray(masses),
            tree_type="radix",
            bounds=bounds,
            return_reordered=True,
            leaf_size=LEAF,
        )
        self.lp = self.tree.positions_sorted
        self.lm = self.tree.masses_sorted
        self.up = prepare_real_upward_sweep(
            self.tree, self.lp, self.lm, max_order=ORDER, max_leaf_size=LEAF
        )
        self.geometry = compute_tree_geometry_compiled(
            self.tree, self.lp, max_leaf_size=LEAF
        )
        moments = compute_tree_mass_moments(self.tree, self.lp, self.lm)
        self.frontier = build_coarse_frontier(
            self.tree, moments.mass, moments.center_of_mass
        )


def isolate_cross_far(per, thetas, interpenetrating, inflate, target_device=0):
    """Measure the cross-domain far field alone, against the exact cross direct sum.

    Replays ``distributed/fmm.py``'s cross-far sequence for ndev=2 without
    ``shard_map``: at ndev=2 the coarse tree a domain builds is exactly the coarse tree
    over the *other* domain's frontier, so no collective is needed to reproduce it.
    This is what shows the M2L, the coarse seeding, the M2M, the L2L and the L2P to be
    correct, and localises the error to the MAC extents. The walk here is the real
    ``dual_tree_walk_cross_impl``, and its pair counts reproduce the driver's
    diagnostics exactly (12 far / 51 near per device at theta_cross=0.1).

    Parameters
    ----------
    per : int
        Particles per device.
    thetas : list[float]
        ``theta_cross`` values to sweep.
    interpenetrating : bool
        Take the two domains from the real Morton partition (True, what the driver
        does at ndev=2) or hand-assign one cluster per device (False, what the tests'
        docstrings assume).
    inflate : bool
        Correct the coarse MAC extents to bound the true remote leaves.
    target_device : int
        Which of the two domains is the target (the other is the remote source). Both
        are needed to account for the driver's aggregate aggL2; see
        :func:`predict_driver_aggl2`.

    Returns
    -------
    dict[float, float]
        ``theta_cross`` -> ABSOLUTE L2 error of this domain's cross-domain field. The
        printed ``relerr`` column is the same quantity divided by ``||exact cross||``.
    """
    pts, mass = separated_clusters(2, per)
    part = partition_for_devices(pts, mass, 2, leaf_size=LEAF)
    bounds = part["bounds"]
    cap = part["cap"]
    if interpenetrating:
        pos_d = part["pos_flat"].reshape(2, cap, 3)
        mass_d = part["mass_flat"].reshape(2, cap)
    else:
        pos_d = pts.reshape(2, per, 3)
        mass_d = mass.reshape(2, per)

    order = [target_device, 1 - target_device]
    target, source = (_DomainView(pos_d[d], mass_d[d], bounds) for d in order)
    centers = target.up.multipoles.centers
    total_nodes = int(np.asarray(target.tree.node_ranges).shape[0])
    num_internal = int(target.tree.left_child.shape[0])
    node_levels = get_node_levels(target.tree)
    left_child = jnp.asarray(target.tree.left_child, INDEX_DTYPE)
    right_child = jnp.asarray(target.tree.right_child, INDEX_DTYPE)
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    node_ranges = jnp.asarray(target.tree.node_ranges, INDEX_DTYPE)
    ranges = np.asarray(node_ranges)

    # --- build_remote_coarse_tree, ndev=2: keep = (domain != me) = the other domain ---
    frontier = source.frontier
    coarse_tree = Tree.from_particles(
        frontier.com,
        frontier.mass,
        tree_type="radix",
        bounds=bounds,
        return_reordered=True,
        leaf_size=1,
    )
    coarse_geometry = compute_tree_geometry(
        coarse_tree, coarse_tree.positions_sorted, max_leaf_size=1
    )
    perm = jnp.asarray(coarse_tree.particle_indices, INDEX_DTYPE)
    tag_range = np.asarray(frontier.node_range[perm])
    tag_node_id = np.asarray(frontier.node_id[perm])
    coarse_up = prepare_real_upward_sweep(
        coarse_tree,
        coarse_tree.positions_sorted,
        coarse_tree.masses_sorted,
        max_order=ORDER,
        max_leaf_size=1,
    )
    c_centers = coarse_up.multipoles.centers
    c_total = c_centers.shape[0]
    c_internal = int(coarse_tree.left_child.shape[0])
    c_ranges = np.asarray(coarse_tree.node_ranges)
    c_leaves = jnp.arange(c_internal, c_total, dtype=INDEX_DTYPE)

    # Seed the coarse leaves with the remote leaves' own multipoles, then M2M up.
    start = jnp.asarray(c_ranges[np.asarray(c_leaves), 0], INDEX_DTYPE)
    node_id = jnp.asarray(tag_node_id)[start]
    present = node_id >= 0
    seeded = source.up.multipoles.packed[jnp.where(present, node_id, 0)]
    seeded = jnp.where(present[:, None], seeded, 0.0)
    seed = (
        jnp.zeros((c_total, sh_size(ORDER)), dtype=target.lp.dtype)
        .at[c_leaves]
        .set(seeded)
    )
    coarse_packed = aggregate_m2m_real_by_level(
        seed,
        c_centers,
        jnp.asarray(coarse_tree.left_child, INDEX_DTYPE),
        jnp.asarray(coarse_tree.right_child, INDEX_DTYPE),
        jnp.asarray(get_nodes_by_level(coarse_tree), INDEX_DTYPE),
        jnp.asarray(get_level_offsets(coarse_tree), INDEX_DTYPE),
        order=ORDER,
        num_internal=c_internal,
        num_levels=int(get_level_offsets(coarse_tree).shape[0] - 1),
        level_batch_width=max(c_internal, 1),
    )

    src_pos = np.asarray(source.lp)
    src_mass = np.asarray(source.lm)
    tgt_pos = np.asarray(target.lp)
    coarse_pos = np.asarray(coarse_tree.positions_sorted)

    # True radius of each coarse "particle": the remote leaf it stands for, about the
    # centre of mass the frontier reduced it to. This is the quantity the coarse
    # geometry does not know about.
    true_radius = np.zeros(coarse_pos.shape[0])
    for p in range(coarse_pos.shape[0]):
        lo, hi = tag_range[p]
        if hi >= lo:
            true_radius[p] = np.linalg.norm(
                src_pos[lo : hi + 1] - coarse_pos[p], axis=1
            ).max()

    if inflate:
        pad = np.zeros(c_total)
        for node in range(c_total):
            lo, hi = c_ranges[node]
            if hi >= lo:
                pad[node] = true_radius[lo : hi + 1].max()
        coarse_geometry = coarse_geometry._replace(
            radius=jnp.asarray(
                np.asarray(coarse_geometry.radius) + pad, coarse_geometry.radius.dtype
            ),
            max_extent=jnp.asarray(
                np.asarray(coarse_geometry.max_extent) + pad,
                coarse_geometry.max_extent.dtype,
            ),
        )
    believed_radius = np.asarray(coarse_geometry.radius)

    reference = np.asarray(direct_sum(target.lp, source.lp, source.lm))
    ref_norm = np.linalg.norm(reference)
    local = np.asarray(direct_sum(target.lp, target.lp, target.lm))
    label = "INTERPENETRATING" if interpenetrating else "one cluster per domain"
    print(
        f"\n=== cross-domain far field in isolation: {label}"
        f"{', extents CORRECTED' if inflate else ''}"
        f" -- domain {target_device} <- {1 - target_device} ==="
    )
    print(
        f"  ||exact cross|| = {ref_norm:.4f}   ||exact local|| = "
        f"{np.linalg.norm(local):.4f}   cross/local = {ref_norm / np.linalg.norm(local):.6f}"
    )
    print(
        f"  true radius of a coarse 'point' source: min {true_radius.min():.4f}  "
        f"median {np.median(true_radius):.4f}  max {true_radius.max():.4f}"
    )

    def evaluate_locals(coeffs):
        """L2L cascade then L2P, exactly as the driver's ``_l2p`` does."""
        coeffs = _propagate_solidfmm_locals_by_level(
            coeffs,
            centers,
            left_child,
            right_child,
            node_levels,
            order=ORDER,
            rotation=ROTATION,
            total_nodes=total_nodes,
            basis_mode="real",
            num_levels=None,
        )
        expansion = LocalExpansionData(
            order=ORDER, centers=centers, coefficients=coeffs
        )
        return (
            -G
            * _evaluate_local_expansions_for_particles(
                expansion,
                target.lp,
                leaf_nodes=leaf_nodes,
                node_ranges=node_ranges,
                max_leaf_size=LEAF,
                order=ORDER,
                expansion_basis="solidfmm",
                return_potential=False,
            )[0]
        )

    def source_particles(coarse_node):
        """Remote particle rows behind a coarse node (what the halo would import)."""
        lo, hi = c_ranges[int(coarse_node)]
        rows = []
        for p in range(lo, hi + 1):
            a, b = tag_range[p]
            rows.extend(range(a, b + 1))
        return np.asarray(rows, dtype=np.int64)

    header = (
        f"  {'theta_cross':>11} {'far':>4} {'near':>5} {'||far||':>11} "
        f"{'||near||':>10} {'relerr':>10} {'true r/d':>9} {'believed':>9} {'overlap':>8}"
    )
    print(header)
    absolute_errors = {}
    for theta_cross in thetas:
        walk = dual_tree_walk_cross_impl(
            target.tree,
            target.geometry,
            coarse_tree,
            coarse_geometry,
            theta_cross,
            mac_type=MAC,
            max_interactions_per_node=512,
            max_neighbors_per_leaf=128,
            max_pair_queue=1 << 15,
        )
        targets = np.asarray(walk.interaction_targets)
        sources = np.asarray(walk.interaction_sources)
        live = targets >= 0
        far_tgt, far_src = targets[live], sources[live]

        if far_tgt.size:
            deltas = (
                centers[jnp.asarray(far_tgt, INDEX_DTYPE)]
                - c_centers[jnp.asarray(far_src, INDEX_DTYPE)]
            )
            contribs = _apply_real_m2l(
                coarse_packed[jnp.asarray(far_src, INDEX_DTYPE)],
                deltas,
                order=ORDER,
                m2l_impl="rot_scale",
            )
            far = np.asarray(
                evaluate_locals(
                    jax.ops.segment_sum(
                        contribs, jnp.asarray(far_tgt, INDEX_DTYPE), total_nodes
                    )
                )
            )
        else:
            far = np.zeros_like(reference)

        counts = np.asarray(walk.neighbor_counts)
        neighbors = np.asarray(walk.neighbor_indices)
        leaf_ids = np.asarray(walk.leaf_indices)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        near = np.zeros_like(reference)
        near_pairs = 0
        for row in range(counts.shape[0]):
            lo, hi = ranges[int(leaf_ids[row])]
            rows = np.arange(lo, hi + 1)
            for edge in range(offsets[row], offsets[row + 1]):
                coarse_node = int(neighbors[edge])
                if coarse_node < 0:
                    continue
                near_pairs += 1
                src_rows = source_particles(coarse_node)
                near[rows] += np.asarray(
                    direct_sum(
                        jnp.asarray(tgt_pos[rows]),
                        jnp.asarray(src_pos[src_rows]),
                        jnp.asarray(src_mass[src_rows]),
                    )
                )

        worst_true = 0.0
        worst_believed = 0.0
        overlapping = 0
        for s, t in zip(far_src, far_tgt):
            lo, hi = ranges[int(t)]
            src_rows = source_particles(s)
            c_src = np.asarray(c_centers[int(s)])
            c_tgt = np.asarray(centers[int(t)])
            distance = np.linalg.norm(c_tgt - c_src)
            r_src = np.linalg.norm(src_pos[src_rows] - c_src, axis=1).max()
            r_tgt = np.linalg.norm(tgt_pos[lo : hi + 1] - c_tgt, axis=1).max()
            worst_true = max(worst_true, (r_src + r_tgt) / distance)
            worst_believed = max(
                worst_believed, (believed_radius[int(s)] + r_tgt) / distance
            )
            overlapping += int(r_src + r_tgt > distance)

        absolute = float(np.linalg.norm(far + near - reference))
        absolute_errors[float(theta_cross)] = absolute
        print(
            f"  {theta_cross:>11g} {far_tgt.size:>4d} {near_pairs:>5d} "
            f"{np.linalg.norm(far):>11.4f} {np.linalg.norm(near):>10.4f} "
            f"{absolute / ref_norm:>10.6f} {worst_true:>9.3f} "
            f"{worst_believed:>9.3f} {overlapping:>4d}/{far_tgt.size:<3d}",
            flush=True,
        )
    return absolute_errors


def predict_driver_aggl2(per, thetas, per_domain_errors):
    """Predict the driver's aggregate aggL2 from the two isolated cross-field errors.

    The driver's assertion is ``||a_fmm - a_direct|| / ||a_direct||`` over all N. If the
    cross-domain far field is the *only* thing wrong, that aggregate is exactly the two
    domains' absolute cross-field errors in quadrature over ``||a_direct||`` -- so
    matching it is the check that nothing else contributes.

    Parameters
    ----------
    per : int
        Particles per device.
    thetas : list[float]
        ``theta_cross`` values that were swept.
    per_domain_errors : list[dict[float, float]]
        One ``theta_cross -> absolute error`` mapping per target domain.

    Returns
    -------
    None
        Prints one row per ``theta_cross``.
    """
    pts, mass = separated_clusters(2, per)
    norm = float(
        np.linalg.norm(
            np.asarray(
                direct_sum(jnp.asarray(pts), jnp.asarray(pts), jnp.asarray(mass))
            )
        )
    )
    print("\n=== does the isolated cross error account for the driver's aggL2? ===")
    print(f"  ||direct|| over all {2 * per} particles = {norm:.4f}")
    print(
        f"  {'theta_cross':>11} {'|err| dom 0':>12} {'|err| dom 1':>12} {'predicted aggL2':>16}"
    )
    for theta_cross in thetas:
        key = float(theta_cross)
        e0 = per_domain_errors[0].get(key, float("nan"))
        e1 = per_domain_errors[1].get(key, float("nan"))
        predicted = float(np.sqrt(e0**2 + e1**2)) / norm
        print(
            f"  {theta_cross:>11g} {e0:>12.4f} {e1:>12.4f} {predicted:>16.6f}",
            flush=True,
        )
    print(
        "  compare with the driver sweep above: at theta_cross=0.1 both are 0.018223, "
        "so the cross far field is the whole error and nothing else contributes."
    )


def main():
    """Run the whole diagnosis and print it.

    Returns
    -------
    int
        Process exit status (always 0; this is a measurement, not a gate).
    """
    ndev = device_count()
    print(f"devices = {jax.devices()}")
    if ndev < 2 and not _ARGS.skip_driver:
        print("fewer than 2 devices: skipping the driver sweep", file=sys.stderr)
    report_domain_geometry(_ARGS.per, ndev)
    if ndev >= 2 and not _ARGS.skip_driver:
        driver_theta_cross_sweep(min(4, ndev), _ARGS.per, _ARGS.theta_cross)
    interleaved_errors = [
        isolate_cross_far(
            _ARGS.per, _ARGS.theta_cross, True, _ARGS.inflate, target_device=d
        )
        for d in (0, 1)
    ]
    predict_driver_aggl2(_ARGS.per, _ARGS.theta_cross, interleaved_errors)
    isolate_cross_far(_ARGS.per, _ARGS.theta_cross, False, _ARGS.inflate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

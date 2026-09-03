"""SolidFMM-style complex-harmonic tree expansion helpers.

This module implements the *upward sweep* (P2M + M2M) for the complex
solid-harmonic basis used by the solidfmm-style backend. It is intentionally
kept separate from the Dehnen real-basis implementation.
"""

from __future__ import annotations

import time
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable
from jax import lax
from jaxtyping import Array, jaxtyped
from yggdrax.dtypes import INDEX_DTYPE, as_index, complex_dtype_for_real
from yggdrax.geometry import TreeGeometry
from yggdrax.tree import Tree, get_level_offsets, get_nodes_by_level
from yggdrax.tree_moments import TreeMassMoments, compute_tree_mass_moments, compute_tree_mass_moments_jit

from jaccpot._env import env_flag
from jaccpot.operators.complex_harmonics import p2m_complex_batch
from jaccpot.operators.complex_ops import (
    enforce_conjugate_symmetry_batch,
    m2m_complex,
    regular_solid_harmonic_directional_derivative_order_batch,
)
from jaccpot.operators.real_harmonics import sh_size

from .tree_geometry import compute_tree_geometry_compiled

__all__ = [
    "SolidFMMComplexNodeMultipoleData",
    "SolidFMMComplexTreeUpwardData",
    "prepare_solidfmm_complex_source_motion_multipoles",
    "prepare_solidfmm_complex_upward_sweep",
]


class SolidFMMComplexNodeMultipoleData(NamedTuple):
    """Packed complex multipole coefficients and their metadata.

    Attributes
    ----------
    order : int
        Expansion order ``p``. A Python ``int``, static under ``jit``.
    centers : Array
        ``[num_nodes, 3]`` expansion centres, real-valued. Which centre these are
        depends on the sweep's ``center_mode`` -- centre of mass, AABB centre, or
        caller-supplied -- so they are not interchangeable across modes.
    packed : Array
        ``[num_nodes, (p+1)^2]`` complex solid-harmonic coefficients in the
        solidfmm convention. Complex, and in a different basis from
        :class:`~jaccpot.upward.real_tree_expansions.RealNodeMultipoleData.packed`
        despite the matching field name.
    source_motion_packed : Optional[Array]
        Time-differentiated multipoles, ``[num_nodes, (p+1)^2]``, or ``None``.
        ``None`` is the normal case: only the derivative/jerk paths in
        :mod:`jaccpot.runtime.fmm_derivatives` populate it, and consumers such as
        :mod:`jaccpot.runtime.kernels._l2l` branch on it being ``None`` rather
        than assuming it is present.
    """

    order: int
    centers: Array  # (num_nodes, 3)
    packed: Array  # (num_nodes, (p+1)^2)
    source_motion_packed: Optional[Array]  # (num_nodes, (p+1)^2) or None


class SolidFMMComplexTreeUpwardData(NamedTuple):
    """Container bundling data needed for the complex upward sweep.

    Carries two fields beyond
    :class:`~jaccpot.upward.real_tree_expansions.RealTreeUpwardData`, which
    exposes ``multipoles`` alone. Code that must accept either lane can only rely
    on ``multipoles``.

    Attributes
    ----------
    geometry : TreeGeometry
        Per-node box extents and radii, from
        :func:`~jaccpot.upward.tree_geometry.compute_tree_geometry_compiled`.
        Needed by the MAC, not by the expansion algebra.
    mass_moments : TreeMassMoments
        Per-node total mass and centre of mass. Retained because the downward
        sweep and the force-scale machinery re-read them; recomputing would be
        a second pass over the particles.
    multipoles : SolidFMMComplexNodeMultipoleData
        The packed complex coefficients and their centres.
    """

    geometry: TreeGeometry
    mass_moments: TreeMassMoments
    multipoles: SolidFMMComplexNodeMultipoleData


_CENTER_MODES = ("com", "aabb", "explicit")
_DEFAULT_LEAF_BATCH_SIZE = 2048


def _upward_diagnostics() -> bool:
    """Whether opt-in upward-sweep diagnostics are enabled.

    Read per call rather than captured at import, so setting
    ``JACCPOT_PREPARE_DIAGNOSTICS=1`` after importing jaccpot takes effect.

    Returns
    -------
    bool
        Whether ``JACCPOT_PREPARE_DIAGNOSTICS`` is set to a truthy value. A
        host-side Python ``bool``, so it is a trace-time constant and branching
        on it does not break ``jit`` -- but for the same reason, flipping the
        variable after a function has been traced will not change the compiled
        graph.
    """
    return env_flag("JACCPOT_PREPARE_DIAGNOSTICS", False)


def _upward_diag(message: str) -> None:
    if _upward_diagnostics():
        print(f"[jaccpot.upward] {message}", flush=True)


def _format_bytes(count: int) -> str:
    value = float(max(int(count), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TiB"


def _diag_upward_stage_estimates(
    *,
    num_particles: int,
    total_nodes: int,
    num_leaves: int,
    max_leaf_size: int,
    leaf_batch_size: int,
    coeffs: int,
    positions_dtype: jnp.dtype,
    masses_dtype: jnp.dtype,
) -> None:
    """Log the byte cost of each upward-sweep stage, when diagnostics are on.

    A no-op unless ``JACCPOT_UPWARD_DIAGNOSTICS`` is set. It exists because the
    upward sweep's peak is not the packed coefficient array (which is obvious) but
    the transient per-leaf-batch gather: ``leaf_batch_size * max_leaf_size *
    (p+1)^2`` complex coefficients live at once, so a batch width chosen for
    throughput can dominate the whole prepare. Printing the terms side by side is
    what makes that visible before a run OOMs.

    All arguments are host-side integers and dtypes -- nothing here is traced, and
    the function returns nothing.

    Parameters
    ----------
    num_particles : int
        Particle count; sizes the mass and weighted-position prefix sums.
    total_nodes : int
        Node count; sizes the per-node centre, mass and packed-coefficient arrays.
    num_leaves : int
        Leaf count; caps the effective batch width.
    max_leaf_size : int
        Padded leaf width; the inner extent of the per-leaf gather.
    leaf_batch_size : int
        Requested leaves per scan step, before capping at ``num_leaves``.
    coeffs : int
        Coefficients per node, ``(p+1)^2``.
    positions_dtype : jnp.dtype
        Position dtype, for the position and centre byte counts.
    masses_dtype : jnp.dtype
        Mass dtype; with ``positions_dtype`` it also determines the complex
        coefficient dtype.

    Returns
    -------
    None
        Nothing; the estimates are emitted through the diagnostics channel.
    """
    if not _upward_diagnostics():
        return

    pos_itemsize = np.dtype(positions_dtype).itemsize
    mass_itemsize = np.dtype(masses_dtype).itemsize
    complex_itemsize = np.dtype(
        complex_dtype_for_real(jnp.result_type(positions_dtype, masses_dtype))
    ).itemsize

    mass_prefix_bytes = (num_particles + 1) * mass_itemsize
    weighted_prefix_bytes = (num_particles + 1) * 3 * pos_itemsize
    total_mass_bytes = total_nodes * mass_itemsize
    center_bytes = total_nodes * 3 * pos_itemsize
    effective_batch = max(1, min(int(leaf_batch_size), int(num_leaves)))
    leaf_point_bytes = effective_batch * max_leaf_size * 3 * pos_itemsize
    leaf_mass_bytes = effective_batch * max_leaf_size * mass_itemsize
    leaf_contrib_bytes = effective_batch * max_leaf_size * coeffs * complex_itemsize
    packed_bytes = total_nodes * coeffs * complex_itemsize

    _upward_diag(
        "stage estimates "
        f"mass_prefix={_format_bytes(mass_prefix_bytes)} "
        f"weighted_prefix={_format_bytes(weighted_prefix_bytes)} "
        f"total_mass={_format_bytes(total_mass_bytes)} "
        f"centers={_format_bytes(center_bytes)} "
        f"p2m_leaf_points={_format_bytes(leaf_point_bytes)} "
        f"p2m_leaf_masses={_format_bytes(leaf_mass_bytes)} "
        f"p2m_leaf_contribs={_format_bytes(leaf_contrib_bytes)} "
        f"packed={_format_bytes(packed_bytes)}"
    )


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "max_leaf_size",
        "num_internal",
        "total_nodes",
        "leaf_batch_size",
    ),
)
def _p2m_leaves_complex(
    node_ranges: Array,
    positions_sorted: Array,
    masses_sorted: Array,
    centers: Array,
    *,
    order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
    leaf_batch_size: int,
) -> Array:
    """Leaf P2M for solidfmm-style complex SH coefficients.

    The complex twin of :func:`~jaccpot.upward.real_tree_expansions._p2m_leaves_real`
    and, by construction, the same physics in the other basis: batched over leaves
    with a ``lax.scan`` so the padded gather keeps a static shape, with over-run
    slots masked to zero position and zero mass so they contribute exactly ``0``.

    Differentiable in ``positions_sorted`` and ``masses_sorted``; the gather
    indices are integer topology.

    Parameters
    ----------
    node_ranges : Array
        Per-node inclusive ``[start, end]`` particle spans, ``[total_nodes, 2]``.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    centers : Array
        Per-node expansion centres ``[total_nodes, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    max_leaf_size : int
        Padded leaf width. Static under ``jit``.
    num_internal : int
        Number of internal nodes; leaves are ``[num_internal, total_nodes)``.
        Static under ``jit``.
    total_nodes : int
        Total node count, and the leading axis of the result. Static under ``jit``.
    leaf_batch_size : int
        Leaves per scan step. Static under ``jit``; a tuning knob that does not
        change the result, but see :func:`_diag_upward_stage_estimates` -- it is
        the term that sets the sweep's transient peak.

    Returns
    -------
    Array
        Packed complex multipole coefficients ``[total_nodes, (p+1)^2]`` in the
        complex dtype matching the inputs, zero on internal nodes.

    Raises
    ------
    ValueError
        If ``order`` is negative.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")

    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    dtype = complex_dtype_for_real(
        jnp.result_type(positions_sorted.dtype, masses_sorted.dtype)
    )
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

    # Leaves live in [num_internal, total_nodes)
    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return packed

    batch = max(1, int(leaf_batch_size))
    num_leaves = int(total_nodes - num_internal)
    steps = (num_leaves + batch - 1) // batch
    pad_amount = steps * batch - num_leaves
    leaf_nodes = jnp.pad(
        leaf_nodes,
        (0, pad_amount),
        mode="constant",
        constant_values=int(num_internal),
    )
    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    batch_offsets = jnp.arange(batch, dtype=INDEX_DTYPE)

    def leaf_accumulate(pos_i: Array, mass_i: Array, center_i: Array) -> Array:
        delta = pos_i - center_i
        return p2m_complex_batch(delta, mass_i, order=p)

    leaf_vm = jax.vmap(leaf_accumulate, in_axes=(0, 0, 0))

    def body(state: Array, step_idx: Array) -> tuple[Array, None]:
        start = step_idx * batch
        batch_nodes = lax.dynamic_slice_in_dim(leaf_nodes, start, batch, axis=0)
        remaining = num_leaves - start
        batch_len = jnp.minimum(batch, jnp.maximum(remaining, 0))
        valid_leaf = batch_offsets < batch_len
        safe_nodes = jnp.where(valid_leaf, batch_nodes, as_index(num_internal))

        ranges = jnp.asarray(node_ranges, dtype=INDEX_DTYPE)[safe_nodes]
        starts = ranges[:, 0]
        ends_inclusive = ranges[:, 1]
        counts = ends_inclusive - starts + 1
        particle_idx = starts[:, None] + idx[None, :]
        valid_particle = valid_leaf[:, None] & (idx[None, :] < counts[:, None])
        safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)

        pos = positions_sorted[safe_idx]
        pos = jnp.where(valid_particle[..., None], pos, 0.0)
        masses = masses_sorted[safe_idx]
        masses = jnp.where(valid_particle, masses, 0.0)

        contribs = leaf_vm(pos, masses, centers[safe_nodes])
        leaf_coeffs = jnp.sum(contribs, axis=1).astype(state.dtype)
        leaf_coeffs = enforce_conjugate_symmetry_batch(leaf_coeffs, order=p)
        current = state[safe_nodes]
        updates = jnp.where(valid_leaf[:, None], leaf_coeffs, current)
        return state.at[safe_nodes].set(updates), None

    packed, _ = lax.scan(
        body,
        packed,
        jnp.arange(steps, dtype=INDEX_DTYPE),
    )
    return packed


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "time_derivative_order",
        "max_leaf_size",
        "num_internal",
        "total_nodes",
    ),
)
def _p2m_leaves_complex_source_motion(
    node_ranges: Array,
    positions_sorted: Array,
    masses_sorted: Array,
    velocities_sorted: Array,
    centers: Array,
    *,
    order: int,
    time_derivative_order: int,
    max_leaf_size: int,
    num_internal: int,
    total_nodes: int,
) -> Array:
    """Leaf source-motion P2M: d/dt[m * R(delta)] for fixed expansion centers.

    The time derivative of the leaf multipole at a **frozen** expansion centre, so
    the only time dependence is the particle displacement: ``d/dt R(x - c) =
    (v . grad) R(x - c)``. Holding the centre fixed is what makes this a
    derivative of the field rather than of the tree, and it is why the caller must
    pass ``centers`` explicitly instead of letting the sweep recompute them.

    Differentiable in ``positions_sorted``, ``masses_sorted`` and
    ``velocities_sorted``.

    Parameters
    ----------
    node_ranges : Array
        Per-node inclusive ``[start, end]`` particle spans, ``[total_nodes, 2]``.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    velocities_sorted : Array
        Particle velocities ``[N, 3]`` in the same order.
    centers : Array
        Frozen per-node expansion centres ``[total_nodes, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    time_derivative_order : int
        How many time derivatives to carry (``1`` gives ``d/dt``). Static under
        ``jit``; must be positive.
    max_leaf_size : int
        Padded leaf width. Static under ``jit``.
    num_internal : int
        Number of internal nodes. Static under ``jit``.
    total_nodes : int
        Total node count, and the leading axis of the result. Static under ``jit``.

    Returns
    -------
    Array
        Packed complex source-motion multipoles ``[total_nodes, (p+1)^2]``, zero
        on internal nodes.

    Raises
    ------
    ValueError
        If ``order`` is negative, or ``time_derivative_order`` is not positive.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    td_order = int(time_derivative_order)
    if td_order <= 0:
        raise ValueError("time_derivative_order must be positive")

    num_internal = int(num_internal)
    total_nodes = int(total_nodes)
    coeffs = sh_size(p)

    dtype = complex_dtype_for_real(
        jnp.result_type(
            positions_sorted.dtype,
            masses_sorted.dtype,
            velocities_sorted.dtype,
        )
    )
    packed = jnp.zeros((total_nodes, coeffs), dtype=dtype)

    leaf_nodes = jnp.arange(num_internal, total_nodes, dtype=INDEX_DTYPE)
    if leaf_nodes.size == 0:
        return packed

    ranges = jnp.asarray(node_ranges, dtype=INDEX_DTYPE)[leaf_nodes]
    starts = ranges[:, 0]
    ends_inclusive = ranges[:, 1]
    counts = ends_inclusive - starts + 1

    idx = jnp.arange(int(max_leaf_size), dtype=INDEX_DTYPE)
    particle_idx = starts[:, None] + idx[None, :]
    valid = idx[None, :] < counts[:, None]
    safe_idx = jnp.clip(particle_idx, 0, positions_sorted.shape[0] - 1)

    pos = positions_sorted[safe_idx]
    pos = jnp.where(valid[..., None], pos, 0.0)
    masses = masses_sorted[safe_idx]
    masses = jnp.where(valid, masses, 0.0)
    vel = velocities_sorted[safe_idx]
    vel = jnp.where(valid[..., None], vel, 0.0)

    delta = pos - centers[leaf_nodes][:, None, :]
    part_deriv = regular_solid_harmonic_directional_derivative_order_batch(
        delta.reshape((-1, 3)),
        vel.reshape((-1, 3)),
        order=p,
        derivative_order=td_order,
    )
    part_deriv = part_deriv.reshape((delta.shape[0], delta.shape[1], coeffs))
    part_deriv = part_deriv.astype(packed.dtype)
    part_deriv = jnp.where(valid[..., None], part_deriv, 0)

    leaf_coeffs = jnp.sum(masses[..., None] * part_deriv, axis=1)
    leaf_coeffs = enforce_conjugate_symmetry_batch(leaf_coeffs, order=p)
    packed = packed.at[leaf_nodes].set(leaf_coeffs)
    return packed


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "num_internal",
        "num_levels",
        "level_batch_width",
        "rotation",
    ),
)
def _aggregate_m2m_complex_by_level(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    rotation: str,
) -> Array:
    """Upward aggregation by translating child multipoles level by level.

    Walks levels deepest-first, translating each internal node's two children's
    multipoles to the parent centre and summing them. Levels are processed in
    ``level_batch_width``-wide slots to keep shapes static; **read the comment
    below the signature before changing that width** -- ``dynamic_slice_in_dim``
    clamps an out-of-range start rather than erroring, and because the slot mask is
    positional, a clamped window writes the wrong nodes and leaves the level's own
    internal nodes at a zero expansion.

    Differentiable in ``packed`` and ``centers``; the child and level index arrays
    are integer topology.

    Parameters
    ----------
    packed : Array
        Packed complex multipoles ``[total_nodes, (p+1)^2]`` with the leaves
        already filled. Returned updated.
    centers : Array
        Per-node expansion centres ``[total_nodes, 3]``; the M2M displacement is
        ``child_centre - parent_centre``.
    left_child : Array
        Left-child node index per node, ``[total_nodes]``.
    right_child : Array
        Right-child node index per node, ``[total_nodes]``.
    nodes_by_level : Array
        Internal node indices grouped by level, ``[num_internal]``.
    level_offsets : Array
        Start offset of each level into ``nodes_by_level``, ``[num_levels + 1]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    num_internal : int
        Number of internal nodes. Static under ``jit``; ``<= 0`` returns ``packed``
        unchanged, since a leaf-only tree has no aggregation work.
    num_levels : int
        Number of levels to walk. Static under ``jit``.
    level_batch_width : int
        Internal nodes per slot within a level. Static under ``jit``; see the
        clamp warning above.
    rotation : str
        Which rotation decomposition the M2M translation uses (``"solidfmm"``).
        Static under ``jit``.

    Returns
    -------
    Array
        Packed complex multipoles ``[total_nodes, (p+1)^2]`` with the internal
        nodes filled.

    Raises
    ------
    ValueError
        If ``order`` is negative.
    """

    p = int(order)
    if p < 0:
        raise ValueError("order must be >= 0")
    if int(num_internal) <= 0:
        # Leaf-only trees have no child->parent aggregation work.
        return packed

    batch_width = int(max(level_batch_width, 1))
    level_offsets = jnp.asarray(level_offsets, dtype=INDEX_DTYPE)
    nodes_by_level = jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE)
    # `dynamic_slice_in_dim` clamps an out-of-range start rather than erroring, so
    # the window slides whenever `start + batch_width > len(nodes_by_level)`. The
    # slot mask is positional, so a slid window selects the *wrong* nodes: the
    # level's own internal nodes are never written (they keep a zero expansion)
    # and unrelated shallower nodes are clobbered.
    #
    # Keeping `batch_width == num_internal` does NOT avoid this, contrary to the
    # note in `prepare_solidfmm_complex_upward_sweep`: `total_nodes` is
    # `2*num_internal + 1`, so every level starting past `num_internal + 1` still
    # overruns -- i.e. the deepest levels of any tree. Measured at N=1024/leaf=16:
    # levels 7, 8 and 9 of 9 were corrupted and 23% of the system mass was missing
    # from the root monopole. Pad instead, so the widest window is always in range.
    nodes_by_level = jnp.concatenate(
        [
            nodes_by_level,
            jnp.full((batch_width,), -1, dtype=INDEX_DTYPE),
        ]
    )
    level_slot = jnp.arange(batch_width, dtype=INDEX_DTYPE)

    def _translate_one(coeffs: Array, delta: Array) -> Array:
        return m2m_complex(coeffs, delta, order=p, rotation=rotation).astype(
            packed.dtype
        )

    translate_children = jax.vmap(
        jax.vmap(_translate_one, in_axes=(0, 0)),
        in_axes=(0, 0),
    )

    def _m2m_level_apply(
        state_in: Array,
        node_centers_all: Array,
        safe_child_idx: Array,
        gather_nodes: Array,
        child_mask: Array,
    ) -> Array:
        """Gather one level's children, M2M-translate them, reduce onto the parents.

        Wrapped in ``jax.checkpoint`` below so reverse mode retains only these
        inputs. Un-rematerialized, the level loop keeps every level's rotate-to-z /
        rotate-from-z blocks *and* their bilinear construction intermediates:
        measured at **14.2 kB per (level x node)** by
        ``bench/audit_reverse_residuals.py``, and because ``level_batch_width`` is
        deliberately ``num_internal`` (a narrower width is unsafe, see below) that
        is ``depth x nodes``, not ``nodes`` -- 5.4 GB at N=1048576.

        ``node_centers_all`` is loop-invariant so ``scan``'s partial-eval hoists it
        out and counts it once; only the carry and three integer/bool index arrays
        are retained per level.

        The zero-displacement case (a single-child internal node shares its child's
        centre of mass) is guarded inside ``m2m_complex`` itself, which is enclosed
        here, so the recomputed ``deltas`` keep that protection.

        Parameters
        ----------
        state_in : Array
            Carried coefficients ``[num_nodes + 1, (p+1)^2]`` -- the extra row is
            the dead scatter target described below, so this is one row taller
            than ``packed``.
        node_centers_all : Array
            ``[num_nodes, 3]`` centres for every node. Passed as an argument
            rather than closed over, for the hoisting reason above.
        safe_child_idx : Array
            ``[batch_width, 2]`` child ids with the invalid entries already
            replaced by ``0``, so the gather is always in range. The mask, not
            this array, is what removes them.
        gather_nodes : Array
            ``[batch_width]`` parent node ids for this level.
        child_mask : Array
            ``[batch_width, 2]`` boolean, true where the child slot is real.
            Applied *after* the translation, so a masked-off slot still costs a
            full M2M -- correct but not free.

        Returns
        -------
        Array
            ``[batch_width, (p+1)^2]`` parent coefficients with conjugate
            symmetry re-enforced. This is the level's contribution only; the
            caller scatters it into the carry.
        """
        child_coeffs = state_in[safe_child_idx]
        child_centers = node_centers_all[safe_child_idx]
        parent_centers = node_centers_all[gather_nodes][:, None, :]
        deltas = child_centers - parent_centers
        translated = translate_children(child_coeffs, deltas)
        translated = translated * child_mask[..., None]
        node_coeffs = jnp.sum(translated, axis=1, dtype=translated.dtype)
        return enforce_conjugate_symmetry_batch(node_coeffs, order=p)

    _m2m_level = jax.checkpoint(_m2m_level_apply)

    # Scatter target for invalid / padding slots. All padding slots must route
    # to a dead row rather than a real node: with duplicate scatter indices XLA
    # takes the last write, so if padding collapsed onto a real node id (e.g. 0,
    # the root) it would clobber that node's genuine M2M contribution. We append
    # one throwaway row to the carried state and discard it at the end. Reads
    # (children / centers) still clamp to a valid in-range index.
    dead_row = as_index(packed.shape[0])
    packed_ext = jnp.concatenate(
        [packed, jnp.zeros((1,) + tuple(packed.shape[1:]), dtype=packed.dtype)],
        axis=0,
    )

    def level_body(level_rev_idx: Array, state: Array) -> Array:
        level_idx = as_index((num_levels - 2) - level_rev_idx)
        start = level_offsets[level_idx]
        end = level_offsets[level_idx + 1]
        count = end - start
        batch_nodes = lax.dynamic_slice_in_dim(
            nodes_by_level,
            start_index=start,
            slice_size=batch_width,
            axis=0,
        )
        valid = level_slot < count
        internal_valid = (
            valid
            & (batch_nodes >= as_index(0))
            & (batch_nodes < as_index(num_internal))
        )
        # Clamped index for gathers (children / centers); dead row for scatters.
        gather_nodes = jnp.where(internal_valid, batch_nodes, as_index(0))
        scatter_nodes = jnp.where(internal_valid, batch_nodes, dead_row)

        child_idx_pair = jnp.stack(
            [left_child[gather_nodes], right_child[gather_nodes]],
            axis=1,
        )
        child_mask = child_idx_pair >= 0
        safe_child_idx = jnp.where(child_mask, child_idx_pair, 0)

        # Gather + translate + reduce, rematerialized (see ``_m2m_level_apply``).
        node_coeffs = _m2m_level(
            state, centers, safe_child_idx, gather_nodes, child_mask
        )

        return state.at[scatter_nodes].set(node_coeffs)

    internal_level_count = max(int(num_levels) - 1, 0)
    result = lax.fori_loop(0, internal_level_count, level_body, packed_ext)
    return result[: packed.shape[0]]


@jaxtyped(typechecker=beartype)
def prepare_solidfmm_complex_upward_sweep(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    *,
    velocities_sorted: Optional[Array] = None,
    max_order: int = 2,
    center_mode: str = "com",
    explicit_centers: Optional[Array] = None,
    max_leaf_size: Optional[int] = None,
    leaf_batch_size: Optional[int] = None,
    rotation: str = "solidfmm",
    precomputed_geometry: Optional[TreeGeometry] = None,
    upward_timing_callback: Optional[Callable[[str, float], None]] = None,
    defer_geometry: bool = False,
    static_num_levels: Optional[int] = None,
) -> SolidFMMComplexTreeUpwardData:
    """Compute complex multipoles for every node (solidfmm basis).

    ``static_num_levels`` lets callers that know the tree topology concretely
    (e.g. at full-prepare or via a fixed-topology template) pass the *actual* tree
    depth. Radix trees pad ``level_offsets`` to the full Morton depth, so deriving
    the M2M level count from the array shape makes the level loop iterate many
    empty levels (each still paying a full-width vmapped translate). Passing the
    concrete depth collapses that waste while remaining recompile-free under the
    fixed-shape contract and bit-identical to the padded result. It must be a
    static int (it feeds an ``@jax.jit`` static arg) and is safe to omit, in which
    case the padded shape-derived depth is used (correct, just slower).

    Note: the M2M per-level batch width stays at ``num_internal``, and the level
    node list is *padded* by that width. The level loop's ``dynamic_slice_in_dim``
    clamps its start index rather than erroring, so any level with
    ``start + width > len`` silently shifts its window and corrupts the
    aggregation. This note previously claimed that keeping the width at
    ``num_internal`` was sufficient to avoid that; it is not -- ``total_nodes`` is
    ``2 * num_internal + 1``, so the deepest levels overrun regardless. The
    padding is what makes any width safe.

    Differentiable in ``positions_sorted``, ``masses_sorted`` and
    ``velocities_sorted``.

    Parameters
    ----------
    tree : Tree
        Radix tree supplying the topology and per-node particle spans.
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    velocities_sorted : Optional[Array]
        Particle velocities ``[N, 3]``. Only needed for the source-motion tower;
        ``None`` skips it.
    max_order : int
        Expansion order ``p``. Static under ``jit``; default ``2``.
    center_mode : str
        How node expansion centres are chosen -- ``"com"`` for the centre of mass.
        Ignored when ``explicit_centers`` is given.
    explicit_centers : Optional[Array]
        Per-node centres ``[num_nodes, 3]`` to use instead of deriving them.
    max_leaf_size : Optional[int]
        Padded leaf width; derived from the tree when ``None``.
    leaf_batch_size : Optional[int]
        Leaves per P2M scan step. Batching only, but it is the term that sets the
        sweep's transient memory peak -- see
        :func:`_diag_upward_stage_estimates`.
    rotation : str
        Rotation decomposition for the M2M cascade; default ``"solidfmm"``.
    precomputed_geometry : Optional[TreeGeometry]
        Node geometry to reuse rather than recompute.
    upward_timing_callback : Optional[Callable[[str, float], None]]
        Called with ``(stage_name, seconds)`` per stage. Host-side; must not be
        passed on a traced path.
    defer_geometry : bool
        Skip building the geometry here and leave it to the caller.
    static_num_levels : Optional[int]
        The tree's *actual* depth, as described above. Must be a concrete int; it
        feeds a ``jit`` static argument. Omitting it is always correct, just
        slower.

    Returns
    -------
    SolidFMMComplexTreeUpwardData
        Packed complex multipoles for every node, the resolved centres and
        geometry, and the source-motion multipoles when velocities were given.

    Raises
    ------
    ValueError
        If ``max_order`` is negative.
    """

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")
    profile_stages = upward_timing_callback is not None and env_flag(
        "JACCPOT_PROFILE_UPWARD_STAGES", False
    )

    def _record_stage(name: str, start: float, value) -> None:
        if not profile_stages or upward_timing_callback is None:
            return
        jax.block_until_ready(value)
        upward_timing_callback(name, float(time.perf_counter() - start))

    _upward_diag(
        "geometry start "
        f"particles={int(positions_sorted.shape[0])} max_order={p} rotation={rotation}"
    )
    stage_t0 = time.perf_counter()
    # Thread the known leaf cap into geometry so JIT does not pad leaf-bound
    # gathers out to ``num_particles`` for large radix trees.
    geometry = (
        precomputed_geometry
        if precomputed_geometry is not None
        else (
            None
            if bool(defer_geometry)
            else compute_tree_geometry_compiled(
                tree,
                positions_sorted,
                max_leaf_size=int(max_leaf_size) if max_leaf_size is not None else None,
            )
        )
    )
    _record_stage("geometry", stage_t0, geometry)
    _upward_diag("geometry done")
    stage_t0 = time.perf_counter()
    #mass_moments = compute_tree_mass_moments(
    #    tree,
    #    positions_sorted,
    #    masses_sorted,
    #)
    mass_moments = compute_tree_mass_moments_jit(
        tree,
        positions_sorted,
        masses_sorted,
    )
    _record_stage("mass_moments", stage_t0, mass_moments)
    _upward_diag("mass moments done")

    total_nodes = int(tree.parent.shape[0])
    mode = str(center_mode).strip().lower()
    if mode == "com":
        centers = mass_moments.center_of_mass
    elif mode == "aabb":
        centers = geometry.center
    elif mode == "explicit":
        if explicit_centers is None:
            raise ValueError(
                "explicit_centers must be provided for 'explicit'",
            )
        if explicit_centers.shape != (total_nodes, 3):
            raise ValueError("explicit_centers must have shape (num_nodes, 3)")
        centers = explicit_centers
    else:
        raise ValueError(f"Unknown center_mode '{center_mode}'")

    centers = jnp.asarray(centers, dtype=positions_sorted.dtype)

    if max_leaf_size is None:
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        leaf_ranges = jax.device_get(tree.node_ranges)[num_internal:]
        if leaf_ranges.shape[0] == 0:
            max_leaf_size = 0
        else:
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            max_leaf_size = int(jnp.max(counts))

    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    total_nodes = int(tree.parent.shape[0])
    num_leaves = max(total_nodes - num_internal, 0)
    level_offsets = get_level_offsets(tree)
    nodes_by_level = get_nodes_by_level(tree)
    if static_num_levels is not None:
        num_levels = max(int(static_num_levels), 1)
    else:
        num_levels = int(level_offsets.shape[0] - 1)
        if num_levels <= 0:
            num_levels = 1
    # Keep batching shape-derived so this path remains JIT-safe under traced tree
    # builds. Width stays at num_internal (see the docstring note: the M2M slice
    # clamps its start, so a narrower width is unsafe).
    level_batch_width = max(int(num_internal), 1)
    resolved_leaf_batch_size = (
        min(num_leaves, _DEFAULT_LEAF_BATCH_SIZE)
        if leaf_batch_size is None
        else int(leaf_batch_size)
    )
    _upward_diag(
        "batch sizing "
        f"total_nodes={total_nodes} num_internal={num_internal} num_leaves={num_leaves} "
        f"resolved_leaf_batch_size={resolved_leaf_batch_size} "
        f"num_levels={num_levels} level_batch_width={level_batch_width}"
    )
    _diag_upward_stage_estimates(
        num_particles=int(positions_sorted.shape[0]),
        total_nodes=total_nodes,
        num_leaves=num_leaves,
        max_leaf_size=int(max_leaf_size),
        leaf_batch_size=resolved_leaf_batch_size,
        coeffs=sh_size(p),
        positions_dtype=positions_sorted.dtype,
        masses_dtype=masses_sorted.dtype,
    )

    _upward_diag("p2m start")
    stage_t0 = time.perf_counter()
    packed = _p2m_leaves_complex(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        centers,
        order=p,
        max_leaf_size=int(max_leaf_size),
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_batch_size=resolved_leaf_batch_size,
    )
    _record_stage("p2m", stage_t0, packed)
    _upward_diag(f"p2m done packed_shape={tuple(int(v) for v in packed.shape)}")

    _upward_diag("m2m start")
    stage_t0 = time.perf_counter()
    packed = _aggregate_m2m_complex_by_level(
        packed,
        centers,
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE),
        jnp.asarray(level_offsets, dtype=INDEX_DTYPE),
        order=p,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        rotation=rotation,
    )
    _record_stage("m2m", stage_t0, packed)
    _upward_diag("m2m done")

    source_motion_packed: Optional[Array] = None
    if velocities_sorted is not None:
        stage_t0 = time.perf_counter()
        source_motion_packed = prepare_solidfmm_complex_source_motion_multipoles(
            tree,
            positions_sorted,
            masses_sorted,
            velocities_sorted,
            max_order=p,
            centers=centers,
            max_leaf_size=int(max_leaf_size),
            rotation=rotation,
        )
        _record_stage("source_motion", stage_t0, source_motion_packed)

    multipoles = SolidFMMComplexNodeMultipoleData(
        order=p,
        centers=centers,
        packed=packed,
        source_motion_packed=source_motion_packed,
    )

    return SolidFMMComplexTreeUpwardData(
        geometry=geometry,
        mass_moments=mass_moments,
        multipoles=multipoles,
    )


@jaxtyped(typechecker=beartype)
def prepare_solidfmm_complex_source_motion_multipoles(
    tree: Tree,
    positions_sorted: Array,
    masses_sorted: Array,
    velocities_sorted: Array,
    *,
    max_order: int,
    centers: Array,
    time_derivative_order: int = 1,
    max_leaf_size: Optional[int] = None,
    rotation: str = "solidfmm",
) -> Array:
    """Compute packed source-motion multipoles for fixed expansion centers.

    Public entry point for the source-motion tower: runs
    :func:`_p2m_leaves_complex_source_motion` over the leaves and aggregates it up
    the tree with the same M2M cascade as the plain multipoles. ``centers`` is
    required rather than derived, because the whole point is that the centres are
    **frozen** -- letting them move would fold the tree's own time dependence into
    a quantity that is meant to describe only the sources.

    Differentiable in ``positions_sorted``, ``masses_sorted`` and
    ``velocities_sorted``.

    Parameters
    ----------
    tree : Tree
        Radix tree supplying the topology (``parent``, children, ``node_ranges``).
    positions_sorted : Array
        Particle positions ``[N, 3]`` in Morton order.
    masses_sorted : Array
        Particle masses ``[N]`` in the same order.
    velocities_sorted : Array
        Particle velocities ``[N, 3]`` in the same order; must match
        ``positions_sorted``'s shape.
    max_order : int
        Expansion order ``p``. Static under ``jit``.
    centers : Array
        Frozen per-node expansion centres; must be exactly
        ``(tree.parent.shape[0], 3)``.
    time_derivative_order : int
        How many time derivatives to carry. Must be positive; default ``1``.
    max_leaf_size : Optional[int]
        Padded leaf width. Derived from the tree when ``None``.
    rotation : str
        Rotation decomposition for the M2M cascade; default ``"solidfmm"``.

    Returns
    -------
    Array
        Packed complex source-motion multipoles ``[num_nodes, (p+1)^2]``.

    Raises
    ------
    ValueError
        If ``max_order`` is negative, ``time_derivative_order`` is not positive,
        ``centers`` is not ``(num_nodes, 3)``, or ``velocities_sorted`` does not
        match ``positions_sorted``'s shape.
    """

    p = int(max_order)
    if p < 0:
        raise ValueError("max_order must be >= 0")
    td_order = int(time_derivative_order)
    if td_order <= 0:
        raise ValueError("time_derivative_order must be positive")
    centers_arr = jnp.asarray(centers, dtype=positions_sorted.dtype)
    if centers_arr.shape != (int(tree.parent.shape[0]), 3):
        raise ValueError("centers must have shape (num_nodes, 3)")
    vel_sorted_arr = jnp.asarray(velocities_sorted, dtype=positions_sorted.dtype)
    if vel_sorted_arr.shape != positions_sorted.shape:
        raise ValueError(
            "velocities_sorted must have shape "
            f"{tuple(positions_sorted.shape)}, got {tuple(vel_sorted_arr.shape)}"
        )

    if max_leaf_size is None:
        num_internal = int(jnp.asarray(tree.left_child).shape[0])
        leaf_ranges = jax.device_get(tree.node_ranges)[num_internal:]
        if leaf_ranges.shape[0] == 0:
            max_leaf_size = 0
        else:
            counts = leaf_ranges[:, 1] - leaf_ranges[:, 0] + 1
            max_leaf_size = int(jnp.max(counts))

    num_internal = int(jnp.asarray(tree.left_child).shape[0])
    total_nodes = int(tree.parent.shape[0])
    source_motion_packed_leaf = _p2m_leaves_complex_source_motion(
        jnp.asarray(tree.node_ranges, dtype=INDEX_DTYPE),
        positions_sorted,
        masses_sorted,
        vel_sorted_arr,
        centers_arr,
        order=p,
        time_derivative_order=td_order,
        max_leaf_size=int(max_leaf_size),
        num_internal=num_internal,
        total_nodes=total_nodes,
    )
    level_offsets = get_level_offsets(tree)
    nodes_by_level = get_nodes_by_level(tree)
    num_levels = int(level_offsets.shape[0] - 1)
    if num_levels <= 0:
        num_levels = 1
    level_batch_width = max(int(num_internal), 1)

    return _aggregate_m2m_complex_by_level(
        source_motion_packed_leaf,
        centers_arr,
        jnp.asarray(tree.left_child, dtype=INDEX_DTYPE),
        jnp.asarray(tree.right_child, dtype=INDEX_DTYPE),
        jnp.asarray(nodes_by_level, dtype=INDEX_DTYPE),
        jnp.asarray(level_offsets, dtype=INDEX_DTYPE),
        order=p,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        rotation=rotation,
    )

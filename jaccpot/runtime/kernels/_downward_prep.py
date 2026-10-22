"""Building the inputs the solidfmm downward sweep consumes.

The downward sweep is driven from four bundles -- the initial locals, the
interaction lists, the multipole inputs and the per-level child views -- and this
module builds them, plus the accumulate-from-multipoles step that turns them into
local expansions. ARCHITECTURE §5 documents ``kernels/`` as four families; this is
the first.

The bundles are ``NamedTuple``s on purpose: they cross a ``jax.jit`` boundary, so
**field order is load-bearing** (NUMERICS_AND_JAX §2 -- reordering silently
permutes buffers). Do not re-order them, and do not convert them to dataclasses
without checking donation.

Split out of ``core.py`` (Tier 1.6, A.9 seam 1); every function body is unchanged.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple, Optional, Union

import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
    build_well_separated_interactions,
)
from yggdrax.tree import Tree

from jaccpot.downward.local_expansions import (
    LocalExpansionData,
)
from jaccpot.operators.real_harmonics import (
    sh_size,
)
from jaccpot.upward.tree_expansions import TreeUpwardData

from .._force_scale_levels import FarForceScaleInputs
from ..dtypes import INDEX_DTYPE
from ._m2l import (
    _accumulate_m2l_chunked_scan,
    _accumulate_m2l_fullbatch,
)

__all__: list[str] = []


def _m2l_csr_pallas_active() -> bool:
    """Whether the flat real-basis M2L runs the CSR Pallas lane kernel.

    Default on (since 2026-09-10; ``JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0``
    restores the chunked pure-JAX lanes) and only where it can lower: an
    Ampere+ GPU, or ``JACCPOT_M2L_CSR_INTERPRET=1`` for CPU parity tests. Read at
    trace time through :mod:`jaccpot._env`, never at import.

    Returns
    -------
    bool
        True when
        :func:`jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`
        replaces the full-batch / chunked flat lanes for the real basis.
    """
    from jaccpot._env import env_flag

    if not env_flag("JACCPOT_STATIC_STRICT_FUSED_M2L_CSR", True):
        return False
    if env_flag("JACCPOT_M2L_CSR_INTERPRET", False):
        return True
    from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

    return pallas_m2l_real_csr_supported()


class _FarPairCOO(NamedTuple):
    """Compact COO-style far-pair representation for streamed M2L execution.

    Field order is load-bearing -- this crosses a ``jit`` boundary.

    Attributes
    ----------
    sources : Array
        Source node index of each far pair.
    targets : Array
        Target node index of each far pair, aligned with ``sources``.
    active_count : Optional[Array]
        Number of leading entries that are live, when the arrays are padded to a
        fixed capacity. ``None`` means every entry is live.
    """

    sources: Array
    targets: Array
    active_count: Optional[Array] = None


class _SolidFMMDownwardInit(NamedTuple):
    """Resolved local-buffer initialization for solidfmm downward prep.

    Field order is load-bearing -- this crosses a ``jit`` boundary.

    Attributes
    ----------
    centers : Array
        Per-node expansion centres, ``[total_nodes, 3]``.
    locals_coeffs : Array
        Initial local coefficients, ``[total_nodes, coeff_count]``; zeros unless
        the caller supplied ``initial_locals``.
    total_nodes : int
        Node count, i.e. the leading dimension of both arrays above.
    coeff_count : int
        Coefficients per node, ``sh_size(order)``.
    dtype : Any
        Working dtype of the local buffer: the centres' own real dtype.
    """

    centers: Array
    locals_coeffs: Array
    total_nodes: int
    coeff_count: int
    dtype: Any


class _TargetSortedFarPairCOO(_FarPairCOO):
    """:class:`_FarPairCOO` of a ``TargetSortedFarPairs`` list: ``targets`` are ROW OFFSETS.

    The list is in CSR order, so it travels as ``sources`` plus the row offsets
    (``total_nodes + 1``) in ``targets``, and the lanes M2L reads it as the CSR it
    is (``csr_by_target(presorted=True)``). Every other M2L route expands one
    target per entry first (``targets_from_csr_offsets``). A field-less subclass:
    the pytree node type carries the contract through ``jit``.
    """

    __slots__ = ()


class _SymmetricTargetSortedFarPairCOO(_TargetSortedFarPairCOO):
    """A :class:`_TargetSortedFarPairCOO` list that holds every directed pair both ways.

    The frozen list of a mutual walk, CHECKED symmetric by its maker (the
    differentiable seam's grad plan, ``runtime/_large_n_grad.py``): the CSR M2L
    lanes' reverse then reads the forward's CSR in both of its passes and sorts
    nothing (``m2l_real_csr_lanes_pallas_cvjp(symmetric=True)``). The forward is
    the target-sorted list's. Field-less, like its parent.
    """

    __slots__ = ()


def _far_pair_coo_from(
    compact_far_pairs: Any, sources: Array, targets: Array
) -> _FarPairCOO:
    """The COO of a compact far list, typed as target-sorted when the list is.

    Parameters
    ----------
    compact_far_pairs : Any
        The walk's far list (``CompactTaggedFarPairs`` or its target-sorted
        subclass).
    sources : Array
        Its ``sources`` (cast), unmodified.
    targets : Array
        Its ``targets`` (cast), unmodified: the row offsets of a
        ``TargetSortedFarPairs`` list.

    Returns
    -------
    _FarPairCOO
        ``_TargetSortedFarPairCOO`` for a ``TargetSortedFarPairs`` list, else
        ``_FarPairCOO``; ``active_count`` from its ``far_pair_count``.
    """
    from jaccpot.runtime._interaction_cache import TargetSortedFarPairs

    cls = (
        _TargetSortedFarPairCOO
        if isinstance(compact_far_pairs, TargetSortedFarPairs)
        else _FarPairCOO
    )
    return cls(
        sources=sources,
        targets=targets,
        active_count=getattr(compact_far_pairs, "far_pair_count", None),
    )


class _SolidFMMDownwardInteractionInputs(NamedTuple):
    """Resolved far-pair arrays for solidfmm downward prep.

    Field order is load-bearing -- this crosses a ``jit`` boundary.

    Attributes
    ----------
    interactions : NodeInteractionList
        The interaction list actually used, whether passed in, freshly traversed
        or the empty placeholder.
    src : Array
        Source node index of each far pair, cast to ``INDEX_DTYPE``.
    tgt : Array
        Target node index of each far pair, aligned with ``src``.
    pair_count : int
        Allocated pair capacity -- a Python int, so it can select a lane at trace
        time.
    active_pair_count : Array
        Live pair count as a traced value; may be below ``pair_count`` when the
        arrays are padded.
    """

    interactions: NodeInteractionList
    src: Array
    tgt: Array
    pair_count: int
    active_pair_count: Array


class _SolidFMMDownwardMultipoleInputs(NamedTuple):
    """Resolved multipole coefficient payloads for downward accumulation.

    Field order is load-bearing -- this crosses a ``jit`` boundary.

    Attributes
    ----------
    multip_packed : Array
        Packed multipole coefficients in the working dtype.
    multip_packed_kernel : Array
        The array handed to the M2L kernel. Currently the same object as
        ``multip_packed``; kept a separate field so a lane can feed the kernel a
        different layout without disturbing the payload above.
    """

    multip_packed: Array
    multip_packed_kernel: Array


class _SolidFMMDownwardChildInputs(NamedTuple):
    """Resolved child-index arrays for L2L propagation.

    Field order is load-bearing -- this crosses a ``jit`` boundary.

    Attributes
    ----------
    num_internal_nodes : int
        Internal-node count. Zero for a single-leaf tree, in which case both
        child arrays are ``None`` and L2L has nothing to do.
    left_child : Optional[Array]
        Left child of each internal node, truncated to ``num_internal_nodes``.
    right_child : Optional[Array]
        Right child of each internal node, same length.
    """

    num_internal_nodes: int
    left_child: Optional[Array]
    right_child: Optional[Array]


def _empty_interaction_storage_for_tree(
    tree: Tree,
    *,
    index_dtype: Any = INDEX_DTYPE,
) -> NodeInteractionList:
    """Construct a minimal zero-pair interaction list for a given tree.

    Used when a caller supplies neither an interaction list nor a COO far-pair
    set: the downward sweep still needs a correctly-shaped list to index, and
    this one has the right per-node arrays with no pairs in them.

    Parameters
    ----------
    tree : Tree
        Tree whose node count fixes the offset and count array lengths.
    index_dtype : Any
        Integer dtype for every index array.

    Returns
    -------
    NodeInteractionList
        A list with ``total_nodes`` worth of offsets and counts, and zero pairs.
    """

    total_nodes = int(jnp.asarray(tree.parent).shape[0])
    return NodeInteractionList(
        offsets=jnp.zeros((total_nodes + 1,), dtype=index_dtype),
        sources=jnp.zeros((0,), dtype=index_dtype),
        targets=jnp.zeros((0,), dtype=index_dtype),
        counts=jnp.zeros((total_nodes,), dtype=index_dtype),
        level_offsets=jnp.zeros((1,), dtype=index_dtype),
        target_levels=jnp.zeros((0,), dtype=index_dtype),
    )


def _prepare_solidfmm_downward_interaction_inputs(
    *,
    tree: Tree,
    upward: TreeUpwardData,
    theta: float,
    mac_type: MACType,
    interactions: Optional[NodeInteractionList],
    far_pairs_coo: Optional[_FarPairCOO],
    traversal_config: Optional[DualTreeTraversalConfig],
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]],
    dehnen_radius_scale: float,
    separation_floor: float = 0.0,
) -> _SolidFMMDownwardInteractionInputs:
    """Resolve interaction storage and far-pair arrays for downward prep.

    Three ways in, in priority order: an explicit ``far_pairs_coo`` wins and
    supplies the pair arrays directly; otherwise an explicit ``interactions``
    list is used; otherwise the dual-tree traversal is run here. A COO set passed
    alongside ``interactions=None`` still gets the empty placeholder list, since
    the sweep indexes both.

    Parameters
    ----------
    tree : Tree
        Tree being swept.
    upward : TreeUpwardData
        Upward-pass result; its ``geometry`` drives the MAC.
    theta : float
        MAC opening angle.
    mac_type : MACType
        Which acceptance criterion the traversal applies.
    interactions : Optional[NodeInteractionList]
        Pre-built interaction list, or ``None`` to traverse.
    far_pairs_coo : Optional[_FarPairCOO]
        Pre-built COO far pairs, which take precedence over both of the above.
    traversal_config : Optional[DualTreeTraversalConfig]
        Capacity and retry configuration for the traversal.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Callback invoked when the traversal retries with a larger capacity.
    dehnen_radius_scale : float
        Scale applied to node radii by the Dehnen criteria.
    separation_floor : float
        Minimum gap of an accepted far pair (length units); ``0`` for none.

    Returns
    -------
    _SolidFMMDownwardInteractionInputs
        Resolved list plus the source/target arrays and their counts.
    """

    resolved_interactions = interactions
    if resolved_interactions is None and far_pairs_coo is None:
        resolved_interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            dehnen_radius_scale=dehnen_radius_scale,
            separation_floor=separation_floor,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )
    if resolved_interactions is None:
        resolved_interactions = _empty_interaction_storage_for_tree(tree)

    if far_pairs_coo is not None:
        src = jnp.asarray(far_pairs_coo.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(far_pairs_coo.targets, dtype=INDEX_DTYPE)
        active_pair_count = (
            jnp.asarray(far_pairs_coo.active_count, dtype=INDEX_DTYPE)
            if far_pairs_coo.active_count is not None
            else jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
        )
    else:
        src = jnp.asarray(resolved_interactions.sources, dtype=INDEX_DTYPE)
        tgt = jnp.asarray(resolved_interactions.targets, dtype=INDEX_DTYPE)
        active_pair_count = jnp.asarray(src.shape[0], dtype=INDEX_DTYPE)
    return _SolidFMMDownwardInteractionInputs(
        interactions=resolved_interactions,
        src=src,
        tgt=tgt,
        pair_count=int(src.shape[0]),
        active_pair_count=active_pair_count,
    )


def _prepare_solidfmm_downward_init(
    *,
    upward: TreeUpwardData,
    initial_locals: Optional[LocalExpansionData],
) -> _SolidFMMDownwardInit:
    """Resolve centers and local-expansion buffers for downward prep.

    Parameters
    ----------
    upward : TreeUpwardData
        Upward-pass result supplying the expansion order and centres.
    initial_locals : Optional[LocalExpansionData]
        Locals to accumulate into, or ``None`` to start from zeros. Supplying
        them is what lets a caller add a second contribution to an existing
        field rather than recomputing it. Without them the local buffer takes
        the centres' own dtype.

    Returns
    -------
    _SolidFMMDownwardInit
        Centres, local buffer, and the sizes and dtype derived from them.

    Raises
    ------
    ValueError
        If ``initial_locals`` does not have shape ``[total_nodes, sh_size(order)]``.
    """

    p = int(upward.multipoles.order)
    centers = jnp.asarray(upward.multipoles.centers)
    total_nodes = int(centers.shape[0])
    coeff_count = sh_size(p)
    dtype = centers.dtype
    if initial_locals is not None:
        locals_coeffs = jnp.asarray(initial_locals.coefficients)
        if locals_coeffs.shape != (total_nodes, coeff_count):
            raise ValueError("initial_locals must match solidfmm layout")
    else:
        locals_coeffs = jnp.zeros((total_nodes, coeff_count), dtype=dtype)
    return _SolidFMMDownwardInit(
        centers=centers,
        locals_coeffs=locals_coeffs,
        total_nodes=total_nodes,
        coeff_count=coeff_count,
        dtype=dtype,
    )


def _prepare_solidfmm_downward_multipole_inputs(
    *,
    upward: TreeUpwardData,
    dtype: Any,
) -> _SolidFMMDownwardMultipoleInputs:
    """Resolve multipole coefficient payloads for downward accumulation.

    There is deliberately **no** basis conversion: the native real upward sweep
    is the single source of real multipoles, so complex-packed input is a wiring
    regression and is rejected rather than silently converted. See the inline
    comment for the contract this enforces.

    Parameters
    ----------
    upward : TreeUpwardData
        Upward-pass result supplying the packed multipoles.
    dtype : Any
        Working dtype to cast the payloads to.

    Returns
    -------
    _SolidFMMDownwardMultipoleInputs
        Cast multipoles and the kernel-facing array.

    Raises
    ------
    TypeError
        If the multipoles are complex-packed -- the removed complex-to-real
        conversion is not reinstated silently.
    """

    # The native real upward sweep (prepare_real_upward_sweep) produces packed REAL
    # (Dehnen no-sqrt2) coefficients directly -- there is NO complex intermediate
    # and NO complex<->real conversion. Pass them straight through to the real
    # M2L/L2L/L2P operators. Hard-error if complex-packed multipoles ever reach
    # here (a wiring regression) rather than silently reintroducing a basis
    # conversion.
    packed_raw = jnp.asarray(upward.multipoles.packed)
    if jnp.iscomplexobj(packed_raw):
        raise TypeError(
            "the downward sweep expects REAL multipole coefficients from the "
            "native real upward sweep (prepare_real_upward_sweep), but received "
            "complex-packed multipoles. The complex->real conversion has been "
            "removed from the real path; check prepare_upward_sweep wiring."
        )
    multip_packed = packed_raw.astype(dtype)
    multip_packed_kernel = multip_packed
    return _SolidFMMDownwardMultipoleInputs(
        multip_packed=multip_packed,
        multip_packed_kernel=multip_packed_kernel,
    )


def _prepare_solidfmm_downward_child_inputs(
    tree: Tree,
) -> _SolidFMMDownwardChildInputs:
    """Resolve child-index arrays for L2L propagation.

    Parameters
    ----------
    tree : Tree
        Tree whose child arrays are truncated to the internal-node count.

    Returns
    -------
    _SolidFMMDownwardChildInputs
        Child indices, or a bundle with ``num_internal_nodes == 0`` and both
        arrays ``None`` for a tree with no internal nodes.
    """

    num_internal_nodes = int(jnp.asarray(tree.left_child).shape[0])
    if num_internal_nodes <= 0:
        return _SolidFMMDownwardChildInputs(
            num_internal_nodes=0,
            left_child=None,
            right_child=None,
        )
    return _SolidFMMDownwardChildInputs(
        num_internal_nodes=num_internal_nodes,
        left_child=jnp.asarray(tree.left_child[:num_internal_nodes], dtype=INDEX_DTYPE),
        right_child=jnp.asarray(
            tree.right_child[:num_internal_nodes], dtype=INDEX_DTYPE
        ),
    )


def _compact_live_pairs(src: Array, tgt: Array) -> tuple[Array, Array, Array]:
    """Move the live entries of a pair list to its front, in their order.

    A local far list with the cross-domain pairs concatenated behind it
    (:func:`jaccpot.runtime.kernels._l2l._prepare_solidfmm_downward_sweep`) is
    ``-1``-padded in the middle and has holes in its cross half, so no count marks
    a live prefix. One prefix sum gives each live entry its slot.

    Parameters
    ----------
    src : Array
        Source node index per pair; ``-1`` marks a dead entry.
    tgt : Array
        Target node index per pair, same convention.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(src, tgt, live_count)``: the same length, the live entries (both
        indices ``>= 0``) first in their original order and ``-1`` behind them,
        and the number of live entries.
    """
    src = jnp.asarray(src)
    tgt = jnp.asarray(tgt)
    live = (src >= 0) & (tgt >= 0)
    slot = jnp.cumsum(live.astype(INDEX_DTYPE)) - 1
    # dead entries aim one past the end and are dropped
    slot = jnp.where(live, slot, src.shape[0])
    src_live = jnp.full_like(src, -1).at[slot].set(src, mode="drop")
    tgt_live = jnp.full_like(tgt, -1).at[slot].set(tgt, mode="drop")
    return src_live, tgt_live, jnp.sum(live, dtype=INDEX_DTYPE)


def _solidfmm_downward_accumulate_from_multipoles(
    initial_locals_coeffs: Array,
    multipoles_coeffs: Array,
    *,
    n_targets: Optional[int] = None,
    centers: Array,
    src: Array,
    tgt: Array,
    pair_count: int,
    active_pair_count: Optional[Array],
    order: int,
    total_nodes: int,
    chunk_size: int,
    m2l_impl: str = "rot_scale",
    targets_sorted: bool = False,
    force_scale: Optional[FarForceScaleInputs] = None,
    symmetric_pairs: bool = False,
) -> Union[Array, tuple[Array, Array]]:
    """Run one M2L accumulation pass over the flat pair list (real basis).

    The complex basis, and the conjugate-symmetry enforcement its coefficients
    needed after the pass, went in the 2026-10 cleanup (C).

    Two lanes, selected without any traced branching: ``pair_count <=
    chunk_size`` picks full-batch over a chunked scan. Both compute the same
    operator; they differ in how the pair list is blocked. A third lane for the
    real basis, the default on Ampere+ GPUs (:func:`_m2l_csr_pallas_active`),
    hands the whole pair list to the CSR Pallas lane kernel, which owns one
    target row per program (one pair per lane) and so needs neither the chunked
    scan nor its per-chunk scatter. (The grouped and class-major lanes went in
    the 2026-10 cleanup, X3; the per-target and tiled CSR kernels in X5.)

    Parameters
    ----------
    initial_locals_coeffs : Array
        Local coefficients to accumulate into, ``[total_nodes, sh_size(order)]``.
    multipoles_coeffs : Array
        Packed source multipoles, same node ordering; with cross-domain sources
        concatenated behind the local nodes it has more rows than
        ``initial_locals_coeffs`` (every target is still local).
    n_targets : Optional[int]
        Number of leading nodes that receive M2L contributions. ``None`` means
        all. Set to the local node count when cross-domain sources sit behind the
        local ones, so the M2L output stays local-only and adds to the local
        expansions.
    centers : Array
        Per-node expansion centres.
    src : Array
        Source node index of each far pair.
    tgt : Array
        Target node index of each far pair.
    pair_count : int
        Allocated pair capacity. Static: it selects full-batch versus chunked.
    active_pair_count : Optional[Array]
        Live pair count, traced; entries beyond it contribute nothing. ``None``:
        the list has no live prefix (a local list with cross-domain pairs behind
        its padding) and every entry with both indices ``>= 0`` is live.
    order : int
        Expansion order.
    total_nodes : int
        Node count, i.e. the scatter extent (the local nodes).
    chunk_size : int
        Pairs per chunk, and the threshold that selects the chunked lane.
    m2l_impl : str
        M2L implementation selector for the flat lanes.
    targets_sorted : bool
        ``src`` / ``tgt`` are a ``TargetSortedFarPairs`` list as walked: ``tgt``
        holds the row offsets. The CSR lanes kernel reads it as is; every other
        route expands one target per entry first. Static.
    force_scale : Optional[FarForceScaleInputs]
        Also sum ``mac_type='dehnen_error'``'s far force-scale term per target in
        the CSR lanes kernel
        (:func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_fb_pallas_cvjp`).
        Only that lane computes it: the caller checks
        :func:`~jaccpot.runtime.kernels._l2l._far_force_scale_in_downward_available`
        first.
    symmetric_pairs : bool
        With ``targets_sorted``: the list holds every directed pair both ways
        (:class:`_SymmetricTargetSortedFarPairCOO`), so the CSR lanes kernel's
        reverse sorts nothing. The forward is unchanged. Static.

    Returns
    -------
    Union[Array, tuple[Array, Array]]
        Updated local coefficients; with ``force_scale``, ``(locals, own far
        term [n_targets or total_nodes])``.

    Raises
    ------
    ValueError
        ``force_scale`` off the CSR lanes kernel.
    """

    csr_lane = _m2l_csr_pallas_active()
    if force_scale is not None and not csr_lane:
        raise ValueError(
            "the far force scale is summed in the CSR M2L lanes kernel only; "
            "check _far_force_scale_in_downward_available() first"
        )
    if targets_sorted and not csr_lane:
        # only the CSR lane kernel reads a CSR as (sources, row offsets)
        from jaccpot.pallas.m2l_real_csr import targets_from_csr_offsets

        tgt = targets_from_csr_offsets(tgt, int(jnp.asarray(src).shape[0]))
        targets_sorted = False
    if active_pair_count is None and not csr_lane:
        # The CSR build drops -1 entries wherever they sit; the flat lanes read
        # `active_pair_count` as a live PREFIX (and the chunked scan skips every
        # chunk past it), so give them one.
        src, tgt, active_pair_count = _compact_live_pairs(src, tgt)

    if csr_lane:
        from jaccpot._env import env_flag
        from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas_cvjp

        interpret = env_flag("JACCPOT_M2L_CSR_INTERPRET", False)
        if force_scale is not None:
            from jaccpot.pallas.m2l_real_csr_lanes import (
                m2l_real_csr_lanes_fb_pallas_cvjp,
            )

            # the same launch with one more output: eq (16b)'s far term per row
            m2l_inc, own_far = m2l_real_csr_lanes_fb_pallas_cvjp(
                multipoles_coeffs,
                centers,
                src,
                tgt,
                active_pair_count,
                force_scale.node_mass,
                force_scale.node_radii,
                force_scale.softening_sq,
                force_scale.node_centers,
                order,
                int(os.environ.get("JACCPOT_M2L_CSR_LANES", "32")),
                interpret,
                "triton",
                int(os.environ.get("JACCPOT_M2L_CSR_WARPS", "1")),
                n_targets,
                bool(targets_sorted),
                float(force_scale.gravitational_constant),
            )
            return initial_locals_coeffs + m2l_inc, own_far
        # the custom_vjp seam: the forward is the same launch, and the reverse
        # runs the transposed (by-source) lane kernel
        m2l_inc = m2l_real_csr_lanes_pallas_cvjp(
            multipoles_coeffs,
            centers,
            src,
            tgt,
            active_pair_count,
            order,
            int(os.environ.get("JACCPOT_M2L_CSR_LANES", "32")),
            interpret,
            "triton",
            int(os.environ.get("JACCPOT_M2L_CSR_WARPS", "1")),
            n_targets,
            bool(targets_sorted),
            bool(targets_sorted) and bool(symmetric_pairs),
        )
        locals_updated = initial_locals_coeffs + m2l_inc
    elif pair_count <= chunk_size:
        locals_updated = _accumulate_m2l_fullbatch(
            initial_locals_coeffs,
            multipoles_coeffs,
            centers,
            src,
            tgt,
            active_pair_count,
            order=order,
            m2l_impl=m2l_impl,
            total_nodes=total_nodes,
        )
    else:
        locals_updated = _accumulate_m2l_chunked_scan(
            initial_locals_coeffs,
            multipoles_coeffs,
            centers,
            src,
            tgt,
            active_pair_count,
            order=order,
            m2l_impl=m2l_impl,
            total_nodes=total_nodes,
            chunk_size=chunk_size,
        )

    return locals_updated

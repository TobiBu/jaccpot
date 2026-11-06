"""Forward-mode (``ad_mode="forward"``) siblings of the large-N gradient lane's seams.

Every fast kernel of the large-N gradient lane is a ``jax.custom_vjp`` (its reverse a
hand-written Pallas kernel), and JAX refuses ``jax.jvp`` of a ``custom_vjp``. Forward
mode is a second dispatch beside the untouched reverse one: under
:func:`~jaccpot.runtime.grad_options.forward_mode_enabled` each call site routes its
seam to the ``jax.custom_jvp`` sibling defined here, whose rule computes

* the **primal** through the seam itself, i.e. with the production primal function --
  the very function the ``custom_vjp``'s forward calls -- so the primal stays bitwise
  and ``jax.linearize`` can split it off; and
* the **tangent** with a *tangent op* (``jaccpot/pallas/_tangent.py``): the seam's
  Pallas tangent kernel where one exists, else the pure-``jnp`` reference.

The seams (COMMON-FWD's binding signature table) and their tangent ops:

======  =================================================  ===============================
seam    sibling here                                       tangents (in argument order)
======  =================================================  ===============================
P2M     :func:`p2m_real_leaves_forward`                    positions, masses, leaf centres
M2M     :func:`m2m_real_levels_forward`                    packed multipoles, centres
L2L     :func:`l2l_real_levels_forward`                    locals, centres
L2P     :func:`l2p_real_particles_forward`                 coefficients, centres, positions
M2L     :func:`m2l_real_csr_lanes_forward`                 multipoles, centres
near    :func:`nearfield_leafpair_csr_sorted_direct_forward`  positions, masses, eps^2, G
======  =================================================  ===============================

A tangent op is called ``op(*primals, *tangents, **params, live=...)``: the primals
are the ``custom_vjp`` seam's array arguments in its argument order (topology
included), the tangents carry a leading ``K`` axis (``K = 1`` here; ``jax.jacfwd``
batches the rule, and a Pallas op's batching rule folds that into ``K``), the params
are the seam's static arguments by name, and ``live`` says which tangents are not
known zeros. :func:`resolve_tangent` picks the op by name, per
:func:`~jaccpot.runtime.grad_options.forward_tangent_kernels`:

1. ``"auto"``: the Pallas op ``<name>`` in the kernel module if it exists, else
   ``<name>_jnp`` there, else this module's reference;
2. ``"pallas"``: the Pallas op, or an error at trace time;
3. ``"jnp"``: this module's reference. (Not the kernel track's ``<name>_jnp``: the
   cascades' ``_jnp`` differentiate the J1 rotations, whose edge derivative is zero
   at a zero-length edge where the true one -- the Pallas kernels', the reference's
   ladder -- is not; the kernel tracks' ``_jnp`` are their parity references.)

THE REFERENCES (``_reference_*``) are ``jax.vmap`` over ``K`` of ``jax.jvp`` of a
pure-``jnp`` twin of the seam's primal, at test scale: the jnp lane's own level
functions where it has one (P2M ``_p2m_leaves_real``, M2M
``aggregate_m2m_real_by_level``, the L2L level loop with ``_l2l_real_batch_kernel``,
the M2L rot-scale operator with its transverse-degeneracy ``custom_jvp``) and, where
the jnp lane's own is not differentiable enough, the kernel's lane body on jnp
gathers (L2P ``l2p_real_grad_cartesian``) or a gather + ``segment_sum`` twin (the
direct CSR near field). Each twin's PRIMAL equals the production kernel to
round-off on the same inputs (``tests/unit/runtime/test_forward_seams.py``, Pallas in
interpret mode); they are slow (no kernel), and the parity oracle of the Pallas
tangent kernels.

Also here: :func:`tangent_gate` (the forward sibling of the reverse step's
``_gradient_gate``: the tangent NaN when the step's capacity verdict failed) and
:func:`keep_value_forward` (``production_grad._keep_value``'s sibling).

First order only: a jvp of a forward-mode tangent reaches a Pallas tangent op's
refusal (or, on the references, differentiates them again: not a supported path).
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot._jax_compat import Tracer
from jaccpot.pallas._tangent import drop_tangent_axis, with_tangent_axis

__all__ = [
    "JNP_TWINS",
    "SEAM_TANGENT_OPS",
    "jnp_stage",
    "keep_value_forward",
    "l2l_real_levels_forward",
    "l2p_real_particles_forward",
    "m2l_real_csr_lanes_forward",
    "m2m_real_levels_forward",
    "nearfield_leafpair_csr_sorted_direct_forward",
    "p2m_real_leaves_forward",
    "primal_exact",
    "resolve_tangent",
    "resolve_tangent_out",
    "tangent_gate",
]

#: seam -> (kernel module, Pallas tangent op name); ``<name>_jnp`` is the kernel
#: track's jnp reference of the same signature (COMMON-FWD)
SEAM_TANGENT_OPS: dict[str, tuple[str, str]] = {
    "p2m": ("jaccpot.pallas.p2m_real_leaf", "p2m_real_leaves_pallas_tangent"),
    "m2m": ("jaccpot.pallas.cascade_real_level", "m2m_real_levels_pallas_tangent"),
    "l2l": ("jaccpot.pallas.cascade_real_level", "l2l_real_levels_pallas_tangent"),
    "l2p": ("jaccpot.pallas.l2p_real", "l2p_real_particles_pallas_tangent"),
    "m2l": ("jaccpot.pallas.m2l_real_csr_lanes", "m2l_real_csr_lanes_pallas_tangent"),
    "near": (
        "jaccpot.pallas.nearfield_leafpair_csr_direct_vjp",
        "nearfield_leafpair_csr_sorted_direct_pallas_tangent",
    ),
}


# --------------------------------------------------------------------------
# Tangent resolution.
# --------------------------------------------------------------------------


def resolve_tangent(seam: str, choice: Optional[str] = None) -> tuple[Callable, str]:
    """The tangent op a seam's forward rule calls, and where it came from.

    Parameters
    ----------
    seam : str
        A key of :data:`SEAM_TANGENT_OPS`.
    choice : Optional[str]
        ``"auto"``, ``"pallas"`` or ``"jnp"``; ``None`` reads
        :func:`~jaccpot.runtime.grad_options.forward_tangent_kernels`.

    Returns
    -------
    tuple[Callable, str]
        ``(op, source)``, ``source`` one of ``"pallas"`` (the kernel module's
        tangent op), ``"jnp"`` (its ``<name>_jnp``, an ``"auto"`` fallback only)
        or ``"reference"`` (this module's; always for ``"jnp"``).

    Raises
    ------
    NotImplementedError
        For ``"pallas"`` when the seam has no Pallas tangent op yet.
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    kind = forward_tangent_kernels() if choice is None else str(choice)
    module_name, op_name = SEAM_TANGENT_OPS[seam]
    module = importlib.import_module(module_name)
    if kind in ("auto", "pallas"):
        op = getattr(module, op_name, None)
        if op is not None:
            return op, "pallas"
        if kind == "pallas":
            raise NotImplementedError(
                f"forward_tangent_kernels='pallas': {module_name} has no "
                f"{op_name} yet; use 'auto' or 'jnp' (the jnp reference)"
            )
    if kind == "auto":
        jnp_op = getattr(module, op_name + "_jnp", None)
        if jnp_op is not None:
            return jnp_op, "jnp"
    return _REFERENCES[seam], "reference"


def resolve_tangent_out(seam: str) -> Optional[Callable]:
    """The seam's Pallas ``<op>_out`` sibling, or ``None`` if its module has none.

    The ``_out`` op takes the seam's primal output as one more primal after the
    seam's arrays (the M2M / L2L tangent cascades read the forward's table instead
    of recomputing it). Only used when :func:`resolve_tangent` chose ``"pallas"``.

    Parameters
    ----------
    seam : str
        A key of :data:`SEAM_TANGENT_OPS`.

    Returns
    -------
    Optional[Callable]
        The op, or ``None``.
    """
    module_name, op_name = SEAM_TANGENT_OPS[seam]
    return getattr(importlib.import_module(module_name), op_name + "_out", None)


def _tangents(
    kernels: str,
    seam: str,
    primals: Sequence[Any],
    tangents: Sequence[Any],
    diff: tuple[int, ...],
    output: Any = None,
    **params: Any,
) -> tuple[Array, ...]:
    """One rule's output tangents: the resolved op at ``K = 1``.

    Parameters
    ----------
    kernels : str
        The tangent-kernel choice the primal was traced with
        (:func:`~jaccpot.runtime.grad_options.forward_tangent_kernels` then: the
        rule may run after the enclosing scope has exited, e.g. under
        ``jax.linearize`` of a jitted function).
    seam : str
        The seam.
    primals : Sequence[Any]
        The seam's array arguments (topology included), argument order.
    tangents : Sequence[Any]
        The rule's tangents, same order (topology's are float0 / symbolic zeros).
    diff : tuple[int, ...]
        The differentiable arguments' positions: the op's tangents, in order.
    output : Any
        The seam's primal output, or ``None``. Given, and the Pallas op resolved,
        its ``<op>_out`` sibling (if the kernel module has one) is called with the
        output as an extra primal after the seam's arrays: the M2M / L2L tangent
        cascades read it instead of recomputing the forward.
    **params : Any
        The seam's static arguments by name.

    Returns
    -------
    tuple[Array, ...]
        One tangent per primal output.
    """
    op, source = resolve_tangent(seam, kernels)
    tk, live = with_tangent_axis(
        [tangents[i] for i in diff], [primals[i] for i in diff]
    )
    if output is not None and source == "pallas":
        out_op = resolve_tangent_out(seam)
        if out_op is not None:
            return drop_tangent_axis(out_op(*primals, output, *tk, live=live, **params))
    return drop_tangent_axis(op(*primals, *tk, live=live, **params))


def _jvp_reference(
    twin: Callable[..., Any], num_primals: int, diff: tuple[int, ...]
) -> Callable[..., tuple[Array, ...]]:
    """A tangent op from a pure-``jnp`` twin: ``vmap`` over ``K`` of its ``jvp``.

    Parameters
    ----------
    twin : Callable[..., Any]
        ``twin(*primals, **params)`` -> an array or a tuple of arrays.
    num_primals : int
        Operands that are primals.
    diff : tuple[int, ...]
        The primals that have a tangent, in the tangents' order.

    Returns
    -------
    Callable[..., tuple[Array, ...]]
        ``ref(*primals, *tangents, live=..., **params)`` -> the output tangents,
        each with the leading ``K`` axis.
    """

    def ref(*operands: Any, live: Any = None, **params: Any) -> tuple[Array, ...]:
        del live  # a dead tangent is a zero array: the jvp is exact either way
        primals = operands[:num_primals]
        tangents = operands[num_primals:]

        def of_diff(*values: Any) -> tuple[Array, ...]:
            args = list(primals)
            for i, value in zip(diff, values):
                args[i] = value
            out = twin(*args, **params)
            return out if isinstance(out, tuple) else (out,)

        def one(*t: Any) -> tuple[Array, ...]:
            return jax.jvp(of_diff, tuple(primals[i] for i in diff), t)[1]

        return tuple(jax.vmap(one)(*tangents))

    return ref


# --------------------------------------------------------------------------
# The jnp twins.
# --------------------------------------------------------------------------


def _p2m_twin(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    *,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    **unused: Any,
) -> Array:
    """P2M as the jnp lane computes it (``_p2m_leaves_real``), leaf rows only.

    Parameters
    ----------
    positions_sorted : Array
        As the seam's argument of that name.
    masses_sorted : Array
        As the seam's argument of that name.
    leaf_centers : Array
        As the seam's argument of that name.
    leaf_ranges : Array
        As the seam's argument of that name.
    order : int
        As the seam's argument of that name.
    num_internal : int
        As the seam's argument of that name.
    total_nodes : int
        As the seam's argument of that name.
    leaf_width : int
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    from jaccpot.upward.real_tree_expansions import _p2m_leaves_real

    ni = int(num_internal)
    ranges = jnp.concatenate(
        [jnp.zeros((ni, 2), leaf_ranges.dtype), jnp.asarray(leaf_ranges)], axis=0
    )
    centers = jnp.concatenate(
        [jnp.zeros((ni, 3), leaf_centers.dtype), jnp.asarray(leaf_centers)], axis=0
    )
    num_leaves = int(total_nodes) - ni
    return _p2m_leaves_real(
        ranges,
        positions_sorted,
        masses_sorted,
        centers,
        order=int(order),
        max_leaf_size=int(leaf_width),
        num_internal=ni,
        total_nodes=int(total_nodes),
        leaf_batch_size=max(1, min(num_leaves, 64)),
    )


def _ladder_translation(which: str, order: int) -> Callable[[Array, Array], Array]:
    """One M2M / L2L edge translation whose tangent in the edge is the ladder's.

    ``w = T(delta) v`` by the jnp operator (:func:`~jaccpot.operators.
    real_translations.m2m_real` / ``l2l_real``, rotate -> z-shift -> rotate back).
    Its derivative in ``delta`` is NOT the rotations' (zero at ``delta = 0``, a
    single child on its parent's centre, where the true one is not): translating
    and differentiating commute, so ``dw / d delta_a = A_a w`` (M2M) and
    ``A_a^T w`` (L2L) with ``A_a`` the ladder of the solid harmonics
    (:func:`~jaccpot.operators.real_harmonic_derivatives.
    real_harmonic_lowering_matrix`) -- exact everywhere, the identity the Pallas
    reverse kernels use (``cascade_real_lanes._ladder_lanes``).

    Parameters
    ----------
    which : str
        ``"m2m"`` or ``"l2l"``.
    order : int
        Expansion order.

    Returns
    -------
    Callable[[Array, Array], Array]
        ``translate(v [C], delta [3]) -> [C]``.
    """
    from jaccpot.operators._precision import highest_matmul_precision
    from jaccpot.operators.real_harmonic_derivatives import (
        real_harmonic_lowering_matrix,
    )
    from jaccpot.operators.real_translations import l2l_real, m2m_real

    p = int(order)
    op = m2m_real if which == "m2m" else l2l_real
    ladder = [real_harmonic_lowering_matrix(p, a) for a in range(3)]
    if which == "l2l":
        ladder = [a.T for a in ladder]

    @jax.custom_jvp
    def translate(v: Array, delta: Array) -> Array:
        return op(v, delta, order=p)

    @highest_matmul_precision
    def rule(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
        v, delta = primals
        tv, td = tangents
        w = translate(v, delta)
        tw = op(tv, delta, order=p)  # linear in the coefficients
        for a in range(3):
            tw = tw + td[a] * (jnp.asarray(ladder[a], w.dtype) @ w)
        return w, tw.astype(w.dtype)

    translate.defjvp(rule)
    return translate


def _m2m_twin(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_internal: int,
    num_levels: int,
    **unused: Any,
) -> Array:
    """M2M bottom-up by level, every internal node set to its two children's sum.

    The lanes kernel's order (deepest level first), with the jnp lane's
    translation (``m2m_real``, edge ``child - parent``) and the ladder tangent
    (:func:`_ladder_translation`).

    Parameters
    ----------
    packed : Array
        As the seam's argument of that name.
    centers : Array
        As the seam's argument of that name.
    left_child : Array
        As the seam's argument of that name.
    right_child : Array
        As the seam's argument of that name.
    parent : Array
        As the seam's argument of that name.
    nodes_by_level : Array
        As the seam's argument of that name.
    level_offsets : Array
        As the seam's argument of that name.
    order : int
        As the seam's argument of that name.
    num_internal : int
        As the seam's argument of that name.
    num_levels : int
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    del parent
    translate = jax.vmap(_ladder_translation("m2m", order))
    total = int(packed.shape[0])
    ni = int(num_internal)
    idx = jnp.asarray(left_child).dtype
    width = max(1, ni)
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    slot = jnp.arange(width, dtype=idx)
    levels = int(num_levels)

    def level(i: Any, state: Array) -> Array:
        lv = levels - 1 - i
        start = offs[lv]
        count = offs[lv + 1] - start
        nodes = jax.lax.dynamic_slice_in_dim(nbl, start, width)
        live = (slot < count) & (nodes >= 0) & (nodes < ni)
        node = jnp.where(live, nodes, 0)
        out = jnp.zeros((width, state.shape[1]), state.dtype)
        for child_of in (left_child, right_child):
            child = jnp.asarray(child_of)[node]
            ok = live & (child >= 0)
            child = jnp.where(ok, child, 0)
            moved = translate(state[child], centers[child] - centers[node])
            out = out + jnp.where(ok[:, None], moved.astype(state.dtype), 0)
        # an internal row is SET to its children's sum, as the kernel stores it
        rows = jnp.where(live, node, total)
        return state.at[rows].set(out, mode="drop")

    return jax.lax.fori_loop(0, levels, level, packed)


def _l2l_twin(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    *,
    order: int,
    num_levels: int,
    **unused: Any,
) -> Array:
    """L2L top-down by level: every node pulls its parent's cascaded locals.

    The lanes kernel's order (one node per lane, ``level_offsets``' levels from
    the root down), with the jnp lane's translation (``l2l_real``, edge
    ``parent - child``) and the ladder tangent (:func:`_ladder_translation`).

    Parameters
    ----------
    coeffs_local : Array
        As the seam's argument of that name.
    centers : Array
        As the seam's argument of that name.
    parent : Array
        As the seam's argument of that name.
    left_child : Array
        As the seam's argument of that name.
    right_child : Array
        As the seam's argument of that name.
    nodes_by_level : Array
        As the seam's argument of that name.
    level_offsets : Array
        As the seam's argument of that name.
    order : int
        As the seam's argument of that name.
    num_levels : int
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    del left_child, right_child
    translate = jax.vmap(_ladder_translation("l2l", order))
    total = int(coeffs_local.shape[0])
    width = total
    idx = parent.dtype
    nbl = jnp.concatenate(
        [jnp.asarray(nodes_by_level, idx), jnp.full((width,), -1, idx)]
    )
    offs = jnp.asarray(level_offsets, idx)
    slot = jnp.arange(width, dtype=idx)

    def level(lv: Any, state: Array) -> Array:
        start = offs[lv]
        count = offs[lv + 1] - start
        nodes = jax.lax.dynamic_slice_in_dim(nbl, start, width)
        live = (slot < count) & (nodes >= 0) & (nodes < total)
        node = jnp.where(live, nodes, 0)
        par = parent[node]
        live = live & (par >= 0)
        par = jnp.where(live, par, 0)
        moved = translate(state[par], centers[par] - centers[node])
        moved = jnp.where(live[:, None], moved.astype(state.dtype), 0)
        rows = jnp.where(live, node, total)
        return state + jax.ops.segment_sum(moved, rows, total + 1)[:total]

    return jax.lax.fori_loop(0, int(num_levels), level, coeffs_local)


def _m2l_twin(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    *,
    order: int,
    n_targets: Optional[int],
    presorted: bool,
    has_active_count: bool,
    **unused: Any,
) -> Array:
    """M2L per pair with the rot-scale operator (its on-axis ``custom_jvp``).

    Parameters
    ----------
    multipoles : Array
        As the seam's argument of that name.
    centers : Array
        As the seam's argument of that name.
    sources : Array
        As the seam's argument of that name.
    targets : Array
        As the seam's argument of that name.
    active_pair_count : Array
        As the seam's argument of that name.
    order : int
        As the seam's argument of that name.
    n_targets : Optional[int]
        As the seam's argument of that name.
    presorted : bool
        As the seam's argument of that name.
    has_active_count : bool
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    from jaccpot.operators.m2l_real_rot_scale import m2l_rot_scale_real_batch
    from jaccpot.pallas.m2l_real_csr import targets_from_csr_offsets

    src = jnp.asarray(sources)
    tgt = jnp.asarray(targets)
    if presorted:
        tgt = targets_from_csr_offsets(tgt, int(src.shape[0]))
    valid = (src >= 0) & (tgt >= 0)
    if has_active_count:
        count = jnp.reshape(jnp.asarray(active_pair_count), (-1,))[0]
        valid = valid & (jnp.arange(src.shape[0], dtype=count.dtype) < count)
    rows = int(multipoles.shape[0]) if n_targets is None else int(n_targets)
    s = jnp.where(valid, src, 0)
    t = jnp.where(valid, tgt, 0)
    # padded pairs get a unit displacement (no singular rotation in the tangent);
    # their increments are zeroed below
    deltas = jnp.where(
        valid[:, None], centers[t] - centers[s], jnp.asarray([0.0, 0.0, 1.0])
    ).astype(centers.dtype)
    inc = m2l_rot_scale_real_batch(multipoles[s], deltas, order=int(order))
    inc = jnp.where(valid[:, None], inc.astype(multipoles.dtype), 0)
    return jax.ops.segment_sum(inc, jnp.where(valid, t, rows), rows + 1)[:rows]


def _l2p_twin(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    *,
    order: int,
    **unused: Any,
) -> Array:
    """L2P per particle: the kernel's Cartesian recurrence on jnp gathers.

    :func:`~jaccpot.pallas.l2p_real.l2p_real_grad_cartesian`, the lane body of the
    Pallas forward. (``evaluate_local_real``'s ``value_and_grad`` is the same
    forward to round-off, but its spherical coordinates give a wrong second
    derivative at ``delta = 0``, a single-particle leaf's: 4.7e-3 off the reverse
    kernel's adjoint at N = 512.)

    Parameters
    ----------
    coefficients : Array
        As the seam's argument of that name.
    centers : Array
        As the seam's argument of that name.
    positions_sorted : Array
        As the seam's argument of that name.
    leaf_nodes : Array
        As the seam's argument of that name.
    leaf_start : Array
        As the seam's argument of that name.
    leaf_count : Array
        As the seam's argument of that name.
    order : int
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    from jaccpot.pallas.l2p_real import l2p_real_grad_cartesian

    del leaf_start
    n = int(positions_sorted.shape[0])
    num_leaves = int(leaf_nodes.shape[0])
    counts = jnp.maximum(jnp.asarray(leaf_count), 0)
    leaf_of = jnp.repeat(
        jnp.arange(num_leaves, dtype=counts.dtype), counts, total_repeat_length=n
    )
    live = jnp.arange(n) < jnp.sum(counts)
    node = jnp.asarray(leaf_nodes)[leaf_of]
    delta = centers[node] - positions_sorted
    coeff = jnp.transpose(coefficients[node])
    grads = l2p_real_grad_cartesian(
        coeff, delta[:, 0], delta[:, 1], delta[:, 2], order=int(order)
    )
    grads = jnp.stack(grads, axis=-1)
    return jnp.where(live[:, None], grads.astype(positions_sorted.dtype), 0)


def _near_direct_twin(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    softening_sq: Array,
    G: Array,
    *,
    static: Any,
    **unused: Any,
) -> Array:
    """The direct CSR near field by gather + ``segment_sum`` (test scale).

    Every target leaf against its own leaf (the diagonal excluded) and each leaf of
    its CSR row, ``W x W`` slots per entry, as the kernel's lane body:
    ``a_i = -G sum_j m_j g(|x_i - x_j|^2) (x_i - x_j)``.

    Parameters
    ----------
    positions_sorted : Array
        As the seam's argument of that name.
    masses_sorted : Array
        As the seam's argument of that name.
    leaf_start : Array
        As the seam's argument of that name.
    leaf_count : Array
        As the seam's argument of that name.
    neighbors : Array
        As the seam's argument of that name.
    offsets : Array
        As the seam's argument of that name.
    counts : Array
        As the seam's argument of that name.
    softening_sq : Array
        As the seam's argument of that name.
    G : Array
        As the seam's argument of that name.
    static : Any
        As the seam's argument of that name.
    **unused : Any
        The seam's other static arguments (unused).

    Returns
    -------
    Array
        The seam's primal output.
    """
    from jaccpot.softening import masked_pair_factors, softening_params_from_sq

    pos = jnp.asarray(positions_sorted)
    mass = jnp.asarray(masses_sorted)
    dtype = pos.dtype
    n = int(pos.shape[0])
    w = int(static.leaf_width)
    start = jnp.asarray(leaf_start)
    count = jnp.asarray(leaf_count)
    nbr = jnp.asarray(neighbors)
    off = jnp.asarray(offsets)
    cnt = jnp.asarray(counts)
    num_rows = int(cnt.shape[0])
    num_leaves = int(count.shape[0])
    # the row of every CSR entry (offsets are row starts; entries past a row's
    # count are never read)
    entry = jnp.arange(nbr.shape[0], dtype=off.dtype)
    row = jnp.searchsorted(off[:num_rows], entry, side="right").astype(off.dtype) - 1
    row_safe = jnp.clip(row, 0, num_rows - 1)
    entry_live = (row >= 0) & (entry < off[row_safe] + cnt[row_safe])
    tl = jnp.concatenate([jnp.arange(num_leaves, dtype=off.dtype), row_safe])
    sl = jnp.concatenate(
        [jnp.arange(num_leaves, dtype=off.dtype), jnp.clip(nbr, 0, num_leaves - 1)]
    )
    live = jnp.concatenate([jnp.ones((num_leaves,), bool), entry_live])
    is_self = jnp.concatenate(
        [jnp.ones((num_leaves,), bool), jnp.zeros(nbr.shape, bool)]
    )
    params = softening_params_from_sq(
        static.softening_kernel, jnp.asarray(softening_sq, dtype), dtype
    )
    g_const = jnp.asarray(G, dtype)
    slots = jnp.arange(w)

    def one(e: tuple) -> Array:
        t_leaf, s_leaf, e_live, e_self = e
        ti = start[t_leaf] + slots
        si = start[s_leaf] + slots
        t_ok = e_live & (slots < count[t_leaf])
        s_ok = slots < count[s_leaf]
        xi = pos[jnp.clip(ti, 0, n - 1)]
        xj = pos[jnp.clip(si, 0, n - 1)]
        mj = mass[jnp.clip(si, 0, n - 1)]
        d = xi[:, None, :] - xj[None, :, :]
        r2 = jnp.sum(d * d, axis=-1)
        active = t_ok[:, None] & s_ok[None, :]
        active = active & ~(e_self & (slots[:, None] == slots[None, :]))
        g, _, _ = masked_pair_factors(r2, active, params, static.softening_kernel)
        scale = -g_const * g * mj[None, :]
        return jnp.sum(scale[..., None] * d, axis=1)

    parts = jax.lax.map(one, (tl, sl, live, is_self), batch_size=64)
    t_rows = start[tl][:, None] + slots[None, :]
    t_ok = live[:, None] & (slots[None, :] < count[tl][:, None])
    rows = jnp.where(t_ok, t_rows, n).reshape(-1)
    acc = jax.ops.segment_sum(parts.reshape(-1, 3), rows, n + 1)[:n]
    return acc.astype(dtype)


_REFERENCES: dict[str, Callable[..., tuple[Array, ...]]] = {
    "p2m": _jvp_reference(_p2m_twin, 4, (0, 1, 2)),
    "m2m": _jvp_reference(_m2m_twin, 7, (0, 1)),
    "l2l": _jvp_reference(_l2l_twin, 7, (0, 1)),
    "l2p": _jvp_reference(_l2p_twin, 6, (0, 1, 2)),
    "m2l": _jvp_reference(_m2l_twin, 5, (0, 1)),
    "near": _jvp_reference(_near_direct_twin, 9, (0, 1, 7, 8)),
}

#: the twins, by seam: their primals are the parity oracle of the references
JNP_TWINS: dict[str, Callable[..., Any]] = {
    "p2m": _p2m_twin,
    "m2m": _m2m_twin,
    "l2l": _l2l_twin,
    "l2p": _l2p_twin,
    "m2l": _m2l_twin,
    "near": _near_direct_twin,
}


# --------------------------------------------------------------------------
# The seams.
# --------------------------------------------------------------------------


@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11))
def _p2m_cjvp(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool,
    backend: str,
    num_warps: Optional[int],
    kernels: str,
) -> Array:
    from jaccpot.pallas.p2m_real_leaf import p2m_real_leaves_pallas

    return p2m_real_leaves_pallas(
        positions_sorted,
        masses_sorted,
        leaf_centers,
        leaf_ranges,
        order=order,
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_width=leaf_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def p2m_real_leaves_forward(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_centers: Array,
    leaf_ranges: Array,
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool,
    backend: str,
    num_warps: Optional[int],
) -> Array:
    """Forward-mode :func:`~jaccpot.pallas.p2m_real_leaf.p2m_real_leaves_pallas_cvjp`.

    The same arguments; the primal is the production launch
    (:func:`~jaccpot.pallas.p2m_real_leaf.p2m_real_leaves_pallas`), the tangent the
    ``p2m`` tangent op (:func:`resolve_tangent`).

    Parameters
    ----------
    positions_sorted : Array
        ``[n, 3]`` positions in tree order. Differentiable.
    masses_sorted : Array
        ``[n]`` masses. Differentiable.
    leaf_centers : Array
        ``[L, 3]`` leaf expansion centres. Differentiable.
    leaf_ranges : Array
        ``[L, 2]`` inclusive particle ranges (topology).
    order : int
        Expansion order. ``nondiff_argnums``.
    num_internal : int
        Internal node count. ``nondiff_argnums``.
    total_nodes : int
        Node count. ``nondiff_argnums``.
    leaf_width : int
        Leaf capacity. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : Optional[int]
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[total_nodes, C]`` packed multipoles, internal rows zero.
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    return _p2m_cjvp(
        positions_sorted,
        masses_sorted,
        leaf_centers,
        leaf_ranges,
        order,
        num_internal,
        total_nodes,
        leaf_width,
        interpret,
        backend,
        num_warps,
        forward_tangent_kernels(),
    )


def _p2m_jvp(
    order: int,
    num_internal: int,
    total_nodes: int,
    leaf_width: int,
    interpret: bool,
    backend: str,
    num_warps: Optional[int],
    kernels: str,
    primals: tuple,
    tangents: tuple,
) -> tuple[Array, Array]:
    statics = (order, num_internal, total_nodes, leaf_width, interpret, backend)
    out: Any = _p2m_cjvp(*primals, *statics, num_warps, kernels)
    (t,) = _tangents(
        kernels,
        "p2m",
        primals,
        tangents,
        (0, 1, 2),
        order=order,
        num_internal=num_internal,
        total_nodes=total_nodes,
        leaf_width=leaf_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, t.astype(out.dtype)


_p2m_cjvp.defjvp(_p2m_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_jvp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def _m2m_cjvp(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
) -> Array:
    from jaccpot.pallas.cascade_real_level import _m2m_forward

    out: Any = _m2m_forward(
        packed,
        centers,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out


def m2m_real_levels_forward(
    packed: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    parent: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Forward-mode :func:`~jaccpot.pallas.cascade_real_level.m2m_real_levels_pallas_cvjp`.

    Parameters
    ----------
    packed : Array
        ``[nodes, C]`` packed multipoles with the leaves filled. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    left_child : Array
        ``[internal]`` left children.
    right_child : Array
        ``[internal]`` right children.
    parent : Array
        ``[nodes]`` parent per node.
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    order : int
        Expansion order. ``nondiff_argnums``.
    num_internal : int
        Internal node count. ``nondiff_argnums``.
    num_levels : int
        Levels covered. ``nondiff_argnums``.
    level_batch_width : int
        Programs per level. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` packed multipoles with the internal nodes filled.
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    return _m2m_cjvp(
        packed,
        centers,
        left_child,
        right_child,
        parent,
        nodes_by_level,
        level_offsets,
        order,
        num_internal,
        num_levels,
        level_batch_width,
        interpret,
        backend,
        num_warps,
        forward_tangent_kernels(),
    )


def _m2m_jvp(
    order: int,
    num_internal: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
    primals: tuple,
    tangents: tuple,
) -> tuple[Array, Array]:
    statics = (order, num_internal, num_levels, level_batch_width, interpret, backend)
    out: Any = _m2m_cjvp(*primals, *statics, num_warps, kernels)
    (t,) = _tangents(
        kernels,
        "m2m",
        primals,
        tangents,
        (0, 1),
        output=out,
        order=order,
        num_internal=num_internal,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, t.astype(out.dtype)


_m2m_cjvp.defjvp(_m2m_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_jvp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _l2l_cjvp(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
) -> Array:
    from jaccpot.pallas.cascade_real_level import _l2l_forward

    out: Any = _l2l_forward(
        coeffs_local,
        centers,
        parent,
        nodes_by_level,
        level_offsets,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out


def l2l_real_levels_forward(
    coeffs_local: Array,
    centers: Array,
    parent: Array,
    left_child: Array,
    right_child: Array,
    nodes_by_level: Array,
    level_offsets: Array,
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Forward-mode :func:`~jaccpot.pallas.cascade_real_level.l2l_real_levels_pallas_cvjp`.

    Parameters
    ----------
    coeffs_local : Array
        ``[nodes, C]`` packed locals after M2L. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    parent : Array
        ``[nodes]`` parent per node.
    left_child : Array
        ``[internal]`` left children.
    right_child : Array
        ``[internal]`` right children.
    nodes_by_level : Array
        Level-major node ids.
    level_offsets : Array
        ``[levels + 1]`` starts.
    order : int
        Expansion order. ``nondiff_argnums``.
    num_levels : int
        Levels present. ``nondiff_argnums``.
    level_batch_width : int
        Programs per level. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[nodes, C]`` fully cascaded locals.
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    return _l2l_cjvp(
        coeffs_local,
        centers,
        parent,
        left_child,
        right_child,
        nodes_by_level,
        level_offsets,
        order,
        num_levels,
        level_batch_width,
        interpret,
        backend,
        num_warps,
        forward_tangent_kernels(),
    )


def _l2l_jvp(
    order: int,
    num_levels: int,
    level_batch_width: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
    primals: tuple,
    tangents: tuple,
) -> tuple[Array, Array]:
    statics = (order, num_levels, level_batch_width, interpret, backend, num_warps)
    out: Any = _l2l_cjvp(*primals, *statics, kernels)
    (t,) = _tangents(
        kernels,
        "l2l",
        primals,
        tangents,
        (0, 1),
        output=out,
        order=order,
        num_levels=num_levels,
        level_batch_width=level_batch_width,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, t.astype(out.dtype)


_l2l_cjvp.defjvp(_l2l_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_jvp, nondiff_argnums=(6, 7, 8, 9, 10, 11))
def _l2p_cjvp(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    order: int,
    block: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
) -> Array:
    from jaccpot.pallas.l2p_real import l2p_real_particles_pallas

    return l2p_real_particles_pallas(
        coefficients,
        centers,
        positions_sorted,
        leaf_nodes,
        leaf_start,
        leaf_count,
        order=order,
        block=block,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )


def l2p_real_particles_forward(
    coefficients: Array,
    centers: Array,
    positions_sorted: Array,
    leaf_nodes: Array,
    leaf_start: Array,
    leaf_count: Array,
    order: int,
    block: int,
    interpret: bool,
    backend: str,
    num_warps: int,
) -> Array:
    """Forward-mode :func:`~jaccpot.pallas.l2p_real.l2p_real_particles_pallas_cvjp`.

    Parameters
    ----------
    coefficients : Array
        ``[nodes, (p+1)^2]`` real local coefficients. Differentiable.
    centers : Array
        ``[nodes, 3]`` expansion centres. Differentiable.
    positions_sorted : Array
        ``[N, 3]`` positions in tree order. Differentiable.
    leaf_nodes : Array
        ``[L]`` node id of each leaf (topology).
    leaf_start : Array
        ``[L]`` first particle of each leaf (topology).
    leaf_count : Array
        ``[L]`` particles of each leaf (topology).
    order : int
        Expansion order. ``nondiff_argnums``.
    block : int
        Particles per program. ``nondiff_argnums``.
    interpret : bool
        Pallas interpret mode. ``nondiff_argnums``.
    backend : str
        Pallas GPU lowering. ``nondiff_argnums``.
    num_warps : int
        Warps per program. ``nondiff_argnums``.

    Returns
    -------
    Array
        ``[N, 3]`` gradient with respect to ``delta``.
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    return _l2p_cjvp(
        coefficients,
        centers,
        positions_sorted,
        leaf_nodes,
        leaf_start,
        leaf_count,
        order,
        block,
        interpret,
        backend,
        num_warps,
        forward_tangent_kernels(),
    )


def _l2p_jvp(
    order: int,
    block: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    kernels: str,
    primals: tuple,
    tangents: tuple,
) -> tuple[Array, Array]:
    out: Any = _l2p_cjvp(*primals, order, block, interpret, backend, num_warps, kernels)
    (t,) = _tangents(
        kernels,
        "l2p",
        primals,
        tangents,
        (0, 1, 2),
        order=order,
        block=block,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
    )
    return out, t.astype(out.dtype)


_l2p_cjvp.defjvp(_l2p_jvp, symbolic_zeros=True)


@functools.partial(jax.custom_jvp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
def _m2l_lanes_cjvp(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Array,
    has_active_count: bool,
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_targets: Optional[int],
    presorted: bool,
    symmetric: bool,
    kernels: str,
) -> Array:
    from jaccpot.pallas.m2l_real_csr_lanes import m2l_real_csr_lanes_pallas

    del symmetric  # the reverse's; the forward ignores it
    out: Any = m2l_real_csr_lanes_pallas(
        multipoles,
        centers,
        sources,
        targets,
        order=order,
        active_pair_count=active_pair_count if has_active_count else None,
        n_targets=n_targets,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        presorted=presorted,
    )
    return out


def _m2l_jvp(
    has_active_count: bool,
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_targets: Optional[int],
    presorted: bool,
    symmetric: bool,
    kernels: str,
    primals: tuple,
    tangents: tuple,
) -> tuple[Array, Array]:
    statics = (has_active_count, order, k_lanes, interpret, backend, num_warps)
    out: Any = _m2l_lanes_cjvp(
        *primals, *statics, n_targets, presorted, symmetric, kernels
    )
    (t,) = _tangents(
        kernels,
        "m2l",
        primals,
        tangents,
        (0, 1),
        order=order,
        k_lanes=k_lanes,
        interpret=interpret,
        backend=backend,
        num_warps=num_warps,
        n_targets=n_targets,
        presorted=presorted,
        symmetric=symmetric,
        has_active_count=has_active_count,
    )
    return out, t.astype(out.dtype)


_m2l_lanes_cjvp.defjvp(_m2l_jvp, symbolic_zeros=True)


def m2l_real_csr_lanes_forward(
    multipoles: Array,
    centers: Array,
    sources: Array,
    targets: Array,
    active_pair_count: Optional[Array],
    order: int,
    k_lanes: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    n_targets: Optional[int] = None,
    presorted: bool = False,
    symmetric: bool = False,
) -> Array:
    """Forward-mode :func:`~jaccpot.pallas.m2l_real_csr_lanes.m2l_real_csr_lanes_pallas_cvjp`.

    The same arguments. A primitive cannot take ``None``, so a missing
    ``active_pair_count`` travels as a dummy ``int32[1]`` with the static
    ``has_active_count=False`` (COMMON-FWD).

    Parameters
    ----------
    multipoles : Array
        ``[n, C]`` real multipoles. Differentiable.
    centers : Array
        ``[n, 3]`` expansion centres. Differentiable.
    sources : Array
        ``[P]`` source ids (topology).
    targets : Array
        ``[P]`` target ids, or the row offsets with ``presorted`` (topology).
    active_pair_count : Optional[Array]
        Live prefix length, or ``None``.
    order : int
        Expansion order. Static.
    k_lanes : int
        Pairs per iteration. Static.
    interpret : bool
        Pallas interpret mode. Static.
    backend : str
        Pallas GPU lowering. Static.
    num_warps : int
        Warps per program. Static.
    n_targets : Optional[int]
        As the forward. Static.
    presorted : bool
        ``targets`` are CSR row offsets. Static.
    symmetric : bool
        The reverse's sort-free switch; the forward and its tangent ignore it.
        Static.

    Returns
    -------
    Array
        ``[n_targets, C]`` local increments (``[n, C]`` when ``n_targets`` is None).
    """
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    has_count = active_pair_count is not None
    count = jnp.asarray(active_pair_count) if has_count else jnp.zeros((1,), jnp.int32)
    return _m2l_lanes_cjvp(
        multipoles,
        centers,
        sources,
        targets,
        count,
        bool(has_count),
        int(order),
        int(k_lanes),
        bool(interpret),
        str(backend),
        int(num_warps),
        None if n_targets is None else int(n_targets),
        bool(presorted),
        bool(symmetric),
        forward_tangent_kernels(),
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(9, 10))
def _direct_cjvp(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    softening_sq: Array,
    G: Array,
    static: Any,
    kernels: str,
) -> Any:
    from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import _direct_forward

    return _direct_forward(
        positions_sorted,
        masses_sorted,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        softening_sq,
        G,
        static,
    )


def _direct_jvp(
    static: Any, kernels: str, primals: tuple, tangents: tuple
) -> tuple[Any, Any]:
    out: Any = _direct_cjvp(*primals, static, kernels)
    acc = out[0] if static.with_force_scale else out
    # a literal G or softening reaches the op as an array
    operands = (*primals[:7], jnp.asarray(primals[7]), jnp.asarray(primals[8]))
    (t,) = _tangents(kernels, "near", operands, tangents, (0, 1, 7, 8), static=static)
    t = t.astype(acc.dtype)
    if static.with_force_scale:
        # the force-scale lane sets the next walk's thresholds: no tangent
        return out, (t, jnp.zeros_like(out[1]))
    return out, t


_direct_cjvp.defjvp(_direct_jvp, symbolic_zeros=True)


def nearfield_leafpair_csr_sorted_direct_forward(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    *,
    leaf_width: int,
    softening_sq: Array,
    G: Array,
    chunk: int,
    num_warps: int | None = None,
    num_stages: int = 1,
    target_subtile: int | None = None,
    interpret: bool = False,
    with_potential: bool = False,
    with_force_scale: bool = False,
    row_limit: int | None = None,
    source_tile: int | None = None,
    source_flags: str | None = None,
    softening_kernel: Optional[str] = None,
    tile_skip: bool | None = None,
) -> tuple[Array, Array | None]:
    """Forward-mode ``nearfield_leafpair_csr_sorted_direct_pallas_cvjp``.

    The same arguments and return value
    (:func:`~jaccpot.pallas.nearfield_leafpair_csr_direct_vjp.
    nearfield_leafpair_csr_sorted_direct_pallas_cvjp`); the primal is the production
    call, the tangent the ``near`` tangent op (positions, masses, softening squared
    and ``G``). The force-scale lane has a zero tangent.

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)`` positions in tree order. Differentiable.
    masses_sorted : Array
        ``(N,)`` masses. Differentiable.
    leaf_start : Array
        ``(L,)`` first particle of each leaf (topology).
    leaf_count : Array
        ``(L,)`` particles per leaf (topology).
    neighbors : Array
        Flat neighbour array of leaf indices (topology).
    offsets : Array
        CSR row starts (topology).
    counts : Array
        ``(L,)`` row lengths (topology).
    leaf_width : int
        As the forward. Static.
    softening_sq : Array
        Scalar squared softening. Differentiable.
    G : Array
        Scalar gravitational constant. Differentiable.
    chunk : int
        As the forward. Static.
    num_warps : int | None
        As the forward. Static.
    num_stages : int
        As the forward. Static.
    target_subtile : int | None
        As the forward. Static.
    interpret : bool
        Pallas interpret mode. Static.
    with_potential : bool
        Refused, as in reverse mode. Static.
    with_force_scale : bool
        Also return the force-scale lane (zero tangent). Static.
    row_limit : int | None
        As the forward. Static.
    source_tile : int | None
        As the forward. Static.
    source_flags : str | None
        As the forward. Static.
    softening_kernel : Optional[str]
        The pair kernel. Static.
    tile_skip : bool | None
        As the forward. Static.

    Returns
    -------
    tuple[Array, Array | None]
        ``(N, 3)`` accelerations and the ``(N,)`` force scale with
        ``with_force_scale``, else ``None``.

    Raises
    ------
    ValueError
        If ``with_potential`` is set.
    """
    from jaccpot.pallas.nearfield_leafpair_csr_direct_vjp import _DirectStatic
    from jaccpot.runtime.grad_options import forward_tangent_kernels

    if with_potential:
        raise ValueError(
            "nearfield_leafpair_csr_sorted_direct_forward differentiates the "
            "accelerations only; the potential lane has no tangent rule"
        )
    static = _DirectStatic(
        leaf_width=int(leaf_width),
        chunk=int(chunk),
        num_warps=num_warps,
        num_stages=int(num_stages),
        target_subtile=target_subtile,
        interpret=bool(interpret),
        with_force_scale=bool(with_force_scale),
        row_limit=row_limit,
        source_tile=source_tile,
        source_flags=source_flags,
        softening_kernel=softening_kernel,
        tile_skip=tile_skip,
    )
    out: Any = _direct_cjvp(
        positions_sorted,
        masses_sorted,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        softening_sq,
        G,
        static,
        forward_tangent_kernels(),
    )
    if static.with_force_scale:
        return out
    return out, None


# --------------------------------------------------------------------------
# The jnp lane's stages: JAX's own jvp, the primal kept.
# --------------------------------------------------------------------------


def primal_exact(fn: Callable[..., Any], *args: Any, **static: Any) -> Any:
    """``fn(*args, **static)`` whose jvp primal is ``fn``'s own forward, bitwise.

    The jnp lane (no Pallas: ``_p2m_leaves_real``, ``aggregate_m2m_real_by_level``,
    the M2L accumulators, the L2L level loop, the particle-major L2P) needs no
    tangent kernel -- JAX differentiates it natively -- but JAX's jvp rules
    evaluate some primals differently from the plain forward (the L2P's
    ``value_and_grad`` differentiated again, the operators' own ``custom_jvp``
    rules): ulps off the production forward (measured: 807 of 1536 far
    accelerations, at most 1.8e-15, N = 512, float64, CPU). Under this wrapper the
    primal is ``fn`` itself and the tangent ``jax.jvp(fn)``'s, so the forward-mode
    program's forward is the production one bit for bit.

    ``fn`` must not close over traced values (a ``custom_jvp`` cannot): pass them
    as arguments. Array leaves of the keyword arguments travel as arguments too.

    Parameters
    ----------
    fn : Callable[..., Any]
        The stage.
    *args : Any
        Its positional arguments (pytrees of arrays; integer ones get no tangent).
    **static : Any
        Its keyword arguments: Python values stay static, array leaves are
        passed through as arguments.

    Returns
    -------
    Any
        ``fn(*args, **static)``.
    """
    leaves, treedef = jax.tree_util.tree_flatten(static)
    is_array = [isinstance(leaf, (jax.Array, Tracer)) for leaf in leaves]
    arrays = [leaf for leaf, a in zip(leaves, is_array) if a]
    num_args = len(args)

    def stage(*xs: Any) -> Any:
        it = iter(xs[num_args:])
        kw = [next(it) if a else leaf for leaf, a in zip(leaves, is_array)]
        return fn(*xs[:num_args], **jax.tree_util.tree_unflatten(treedef, kw))

    @jax.custom_jvp
    def exact(*xs: Any) -> Any:
        return stage(*xs)

    def rule(primals: tuple, tangents: tuple) -> tuple[Any, Any]:
        return exact(*primals), jax.jvp(stage, primals, tangents)[1]

    exact.defjvp(rule)
    return exact(*args, *arrays)


def jnp_stage(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)``; under ``ad_mode="forward"`` via :func:`primal_exact`.

    The call sites of the jnp lane's stages use it: in reverse mode it is the
    plain call (the same program).

    Parameters
    ----------
    fn : Callable[..., Any]
        A jnp-lane stage.
    *args : Any
        Its positional arguments.
    **kwargs : Any
        Its keyword arguments.

    Returns
    -------
    Any
        ``fn(*args, **kwargs)``.
    """
    from jaccpot.runtime.grad_options import forward_mode_enabled

    if forward_mode_enabled():
        return primal_exact(fn, *args, **kwargs)
    return fn(*args, **kwargs)


# --------------------------------------------------------------------------
# The step's gate and the chain's restart boundary.
# --------------------------------------------------------------------------


@jax.custom_jvp
def tangent_gate(acc: Array, ok: Array) -> Array:
    """``acc`` itself; its tangent becomes NaN when ``ok`` is ``False``.

    The forward-mode sibling of the reverse step's ``_gradient_gate``: on a step
    whose capacities overflowed (``overflow_ok(bits)`` false: a truncated list,
    caps from a configuration with fewer pairs) the derivative is NaN, never a
    plausible wrong number. Forward mode needs no symmetric lists (its tangent
    kernels are target-driven), so the symmetry verdict is not part of ``ok``.

    Parameters
    ----------
    acc : Array
        The accelerations.
    ok : Array
        Boolean scalar.

    Returns
    -------
    Array
        ``acc``.
    """
    del ok
    return acc


def _tangent_gate_jvp(primals: tuple, tangents: tuple) -> tuple[Array, Array]:
    acc, ok = primals
    t_acc = tangents[0]
    out = tangent_gate(acc, ok)
    return out, jnp.where(ok, t_acc, jnp.asarray(jnp.nan, t_acc.dtype))


tangent_gate.defjvp(_tangent_gate_jvp)


@jax.custom_jvp
def keep_value_forward(value: Any, recomputed: Any) -> Any:
    """``value``, with the tangent of ``recomputed``: ``_keep_value`` in forward mode.

    A restart's start acceleration: the file's VALUE (so the forward stays the
    runner's), the TANGENT of the force recomputed at the restart positions
    (``production_grad._keep_value``'s reverse, as a jvp).

    Parameters
    ----------
    value : Any
        What the function returns.
    recomputed : Any
        Same shape and dtype; whose tangent the output carries.

    Returns
    -------
    Any
        ``value``.
    """
    del recomputed
    return value


def _keep_value_jvp(primals: tuple, tangents: tuple) -> tuple[Any, Any]:
    value, recomputed = primals
    return keep_value_forward(value, recomputed), tangents[1]


keep_value_forward.defjvp(_keep_value_jvp)

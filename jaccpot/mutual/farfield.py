"""Mutual (dual) far field on the Dehnen real spherical-harmonic basis.

The production downward sweep is one-directional: a node's local expansion
accumulates the fields of the sources it accepted, and the reciprocal effect is
picked up separately when the traversal reaches the other node. Here each
well-separated pair ``(A, B)`` appears **once** in the frozen list and a single
pass adds B's multipole field to A's local expansion *and* A's to B's -- Dehnen's
mutual M2L.

Why this conserves momentum exactly
-----------------------------------
Both forces are gradients of the *same* truncated mutual interaction energy
``W_AB = <M_A, T(R) M_B>``: the total force on A is ``-dW/dx_A`` and on B is
``-dW/dx_B = +dW/dx_A``. Because a single M2L evaluation supplies both sides from
one set of translation coefficients, ``F_A + F_B`` cancels *algebraically*, not
just to the truncation error -- measured at ~4e-16 relative, flat in ``theta``
and in the expansion order (``tests/integration/test_mutual_fmm.py``). The
remaining cascades preserve it because they are exact re-expansions: M2M loses
nothing mapping child degrees ``<= n`` into a parent degree ``n``, and L2L
re-centres a degree-``p`` polynomial into another degree-``p`` polynomial.

That is the whole reason to prefer this over a COM correction
(``a_i -= sum_j m_j a_j / sum_j m_j``): the correction also zeroes the momentum
sum, but by smearing a uniform nonlocal offset over every particle instead of
delivering each pair's back-reaction to the partner that actually caused it. It
does not survive a per-level decomposition, so it cannot support a block step.

Every stage is pure JAX over host-frozen index arrays, with static (Python) loop
bounds over tree levels, so the whole sweep transposes cleanly under
``jax.grad``.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from jaccpot.operators.dtypes import squared_radius_floor

# The two rotate helpers are package-private but imported deliberately: this
# module needs the *same* world<->z rotations the production rot-scale M2L uses,
# with only the z-translation core swapped (see `_m2l_batch`). Re-deriving them
# from the public rotation-block builders would duplicate logic that must not
# drift from the kernel it is checked against.
from jaccpot.operators.m2l_real_rot_scale import (
    _rotate_local_from_z_single,
    _rotate_multipole_to_z_single,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.operators.real_harmonics import (
    evaluate_local_real_with_grad,
    l2l_real,
    m2m_real,
    p2m_real_direct,
    sh_size,
    translate_along_z_m2l_real,
)

__all__ = [
    "MutualTreeArrays",
    "dense_level_schedule",
    "mutual_far_field_forces",
    "mutual_upward_sweep",
    "snap_capacity",
]

_M2L_BATCH_BUDGET = 1 << 16

# Peak bytes one chunk's *padded* rotation-block tensors may reach on the fused
# Pallas lane. The fused kernel takes the world<->z rotations as explicit
# ``(pairs, Bp, mdp, mdp)`` operands, so a chunk costs far more memory per pair
# than the three-stage sandwich (which only ever materialises ``(pairs, C)``
# coefficient vectors). At order 4 in float64 that is ~16 KB per pair per block
# against ~0.2 KB, so reusing the 65536-pair count would ask for gigabytes.
_M2L_FUSED_BLOCK_BUDGET_BYTES = 128 << 20


def _m2l_lane(use_pallas: bool, interpret: bool) -> str:
    """Which M2L implementation to dispatch: ``"jax"``, ``"fused"`` or ``"zcore"``.

    Selected by ``JACCPOT_MUTUAL_M2L``; ``"auto"`` (the default) resolves to:

    * ``"fused"`` under ``interpret`` -- interpret mode exists to execute the
      real kernel logic on CPU, so the parity and differentiability tests must
      keep reaching it. Routing interpret to pure JAX would make them vacuous;
    * ``"jax"`` on real hardware, **because the Pallas M2L is measurably slower
      here**. On an A100 at N=10^4 the whole far field costs 17.1 ms in pure JAX
      against 20.0 ms fused and 20.3 ms z-core -- a ~0.85x regression that both
      Pallas shapes share.

    That the two shapes are indistinguishable is the informative part. Fusing the
    rotations away changes nothing, so the rotation ``vmap``s were never what
    dominated the far field. What the fused kernel *does* change is its operand
    traffic: it takes the world<->z rotations as explicit ``(pairs, Bp, mdp,
    mdp)`` arrays, 32 KB per pair at order 4 in float64 against the sandwich's
    0.39 KB of coefficient vectors -- an 82x amplification. The sandwich builds
    the same blocks inside a fused XLA kernel and never spills them to HBM.

    The production real-M2L path does not contradict this: it faces the same
    trade and simply has different pair statistics. Here every directed pair has
    its own ``delta``, so no rotation block is ever reused.

    Both Pallas lanes stay wired, differentiable and covered by the interpret
    tests; set ``JACCPOT_MUTUAL_M2L=fused`` (or ``zcore``) to force one on
    hardware and reproduce the A/B.

    Parameters
    ----------
    use_pallas : bool
        Whether the caller asked for a Pallas lane at all. ``False`` short-cuts
        to ``"jax"`` without consulting the environment.
    interpret : bool
        Whether the Pallas kernels would run in interpret mode.

    Returns
    -------
    str
        One of ``"jax"``, ``"fused"`` or ``"zcore"``.

    Raises
    ------
    ValueError
        If ``JACCPOT_MUTUAL_M2L`` is set to something other than ``"auto"``,
        ``"jax"``, ``"fused"`` or ``"zcore"``.
    """
    if not use_pallas:
        return "jax"
    choice = os.environ.get("JACCPOT_MUTUAL_M2L", "auto").strip().lower()
    if choice not in {"auto", "jax", "fused", "zcore"}:
        raise ValueError(
            "JACCPOT_MUTUAL_M2L must be one of 'auto', 'jax', 'fused', 'zcore'; "
            f"got {choice!r}"
        )
    if choice == "auto":
        choice = "fused" if interpret else "jax"
    if choice == "jax":
        return "jax"
    if interpret:
        return choice
    # The fused gate is `pallas_m2l_real_fused_supported` (Ampere sm_80+), not the
    # z-core's `pallas_m2l_real_supported` (any gpu/tpu): the latter would route a
    # pre-Ampere GPU into a Triton lowering that fails at runtime.
    from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported

    return choice if pallas_m2l_real_fused_supported() else "jax"


def _fused_m2l_chunk(order: int, itemsize: int) -> int:
    """Pairs per scan step that keep the fused lane's block tensors in budget.

    The fused kernel takes its rotations as explicit ``(pairs, Bp, mdp, mdp)``
    operands, so its per-pair cost is set by the block tables, not by the
    coefficient vectors the three-stage sandwich moves.

    Parameters
    ----------
    order : int
        Expansion order; fixes the block-table shapes.
    itemsize : int
        Bytes per element of the working dtype.

    Returns
    -------
    int
        Chunk size, at least ``1``.
    """
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_tables

    tables = m2l_real_fused_tables(int(order))
    per_pair = max(1, int(tables["Bp"]) * int(tables["mdp"]) ** 2 * int(itemsize))
    return max(1, _M2L_FUSED_BLOCK_BUDGET_BYTES // per_pair)


class MutualTreeArrays(NamedTuple):
    """Device-resident frozen topology consumed by the far-field sweep.

    The level schedule is a **dense** ``(depth_cap, width_cap)`` block rather
    than a tuple of ragged per-level arrays. That is a shape decision, not a
    cosmetic one: a tuple's arity is the tree depth, so it is part of the pytree
    *structure*, and a rebuild that changes the depth by one changes the
    structure and forces a retrace. Padded to capacities, the whole topology is
    keyed on capacities alone and one compiled program serves every rebuild.

    ``far_source`` and ``far_target`` carry each canonical pair **twice**, once
    per direction, so one kernel invocation produces both halves under a single
    rounding regime -- the structural reason momentum cancels exactly.

    Attributes
    ----------
    num_nodes : int
        Total node count; sets the shape of every per-node accumulator.
    order : int
        Expansion order.
    leaf_nodes : Array
        Node index of each leaf, in leaf order.
    leaf_particles : Array
        ``(num_leaves, leaf_capacity)`` particle indices, Morton-sorted order.
    leaf_particle_valid : Array
        Boolean mask of the same shape marking occupied slots in the padding.
    level_nodes : Array
        ``(depth_cap, width_cap)`` node indices, row ``d`` holding the nodes at
        depth ``d + 1``, shallowest first.
    level_parents : Array
        Parent of each entry of ``level_nodes``, same shape and order.
    level_valid : Array
        Boolean mask over the level schedule. Padded slots hold index 0 and every
        consumer masks their contribution to the identity, so a padded slot is a
        no-op rather than a write to node 0 -- and a whole padded *row*, i.e. a
        level beyond the real tree depth, is a no-op iteration.
    far_source : Array
        Source node of each directed far entry.
    far_target : Array
        Target node of each directed far entry.
    far_valid : Array
        Boolean mask over the directed far entries.
    """

    num_nodes: int
    order: int
    leaf_nodes: Array
    leaf_particles: Array
    leaf_particle_valid: Array
    level_nodes: Array
    level_parents: Array
    level_valid: Array
    far_source: Array
    far_target: Array
    far_valid: Array


# Capacities are snapped onto a ladder rather than used raw, for the reason
# `jaccpot/runtime/_large_n_env.py` gives for its own tunables: each distinct
# capacity is a distinct compiled shape, so a free-running integer recompiles
# per rebuild -- which is exactly the cost this padding exists to remove.
# A 1 / 1.5 / 2 x powers-of-two ladder rather than powers of two alone. The
# rungs matter because the kernels do work proportional to the *capacity*, not
# the occupancy: the M2L chunks its directed pair list at a fixed budget, so a
# cap 2x the real pair count means twice the M2L chunks. Pure powers of two
# overshoot by up to 2x; adding the 1.5x rungs halves that to 1.33x worst case.
# Within a single run only one rung is ever used, so a finer ladder costs
# nothing -- it only means two runs at different N may not share a compile.
CAPACITY_LADDER: Tuple[int, ...] = tuple(
    sorted(
        {m * (1 << k) for k in range(3, 26) for m in (2, 3)}
        | {1 << k for k in range(3, 27)}
    )
)


def snap_capacity(
    value: int, *, relative: float = 0.10, absolute: int = 256, minimum: int = 8
) -> int:
    """Smallest ladder entry at or above ``value * (1 + relative) + absolute``.

    The headroom is deliberately **additive plus relative**, not a single
    multiplicative factor, because measured drift does not look like one factor.
    Over a 12 base-step Hernquist rollout at N = 4096 the near-pair count moved
    8040 -> 8118 and the far-pair count 10 -> 74: 1.0% against 640% in relative
    terms, but **+78 against +64 in absolute terms**. The absolute floor is what
    covers a small list whose relative swing is enormous; the relative term is
    what covers a large list where a fixed floor would be nothing.

    A single multiplicative factor sized for the far list (4x) was measured to
    resolve a cap of 262144 for 32830 real far pairs at N = 20 000 -- 8x the
    occupancy, and therefore 8x the M2L chunks. This form gives 49152 there.

    Parameters
    ----------
    value : int
        Observed occupancy to size the capacity from.
    relative : float
        Fractional headroom. Default ``0.10``.
    absolute : int
        Additive headroom, in the same units as ``value``. Default ``256``.
    minimum : int
        Floor on the result. Default ``8``.

    Returns
    -------
    int
        The smallest ladder entry at or above the padded value.

    Raises
    ------
    ValueError
        If the padded value exceeds the largest ladder entry. Widen
        :data:`CAPACITY_LADDER` deliberately rather than falling back to an
        unsnapped value.
    """
    want = max(
        int(minimum),
        int(max(0, value) * (1.0 + float(relative))) + int(absolute),
    )
    for entry in CAPACITY_LADDER:
        if entry >= want:
            return int(entry)
    raise ValueError(
        f"capacity {value} (+{relative:.0%}/+{absolute}) exceeds the largest "
        f"ladder entry {CAPACITY_LADDER[-1]}; widen CAPACITY_LADDER deliberately "
        "rather than falling back to an unsnapped value"
    )


def dense_level_schedule(
    level_nodes: Tuple[Array, ...],
    level_parents: Tuple[Array, ...],
    *,
    depth_cap: Optional[int] = None,
    width_cap: Optional[int] = None,
    index_dtype: Any = jnp.int32,
) -> Tuple[Array, Array, Array, int, int]:
    """Pack ragged per-level index arrays into one ``(depth, width)`` block.

    Returns ``(nodes, parents, valid, depth_cap, width_cap)``. Levels stay
    shallowest-first; padded slots hold index 0 with ``valid`` false, so every
    consumer masks them to its own identity.

    A depth beyond the real tree is an entirely-invalid row, which every cascade
    treats as a no-op -- so ``depth_cap`` may exceed the tree depth freely, and
    that headroom is what lets one program survive a rebuild that deepens the
    tree.

    Parameters
    ----------
    level_nodes : Tuple[Array, ...]
        Ragged per-level node indices, shallowest first.
    level_parents : Tuple[Array, ...]
        Parent of each entry of ``level_nodes``.
    depth_cap : Optional[int]
        Rows to emit; ``None`` sizes it from the schedule itself.
    width_cap : Optional[int]
        Slots per row; ``None`` sizes it from the schedule itself.
    index_dtype : Any
        Integer dtype of the packed index arrays.

    Returns
    -------
    Tuple[Array, Array, Array, int, int]
        ``(nodes, parents, valid, depth_used, width_used)``: the first three are
        ``(depth_cap, width_cap)``, the last two report the occupancy that was
        packed.

    Raises
    ------
    ValueError
        If the tree is deeper than ``depth_cap`` or a level wider than
        ``width_cap``. Padding a level schedule too small silently drops nodes from
        the M2M/L2L cascade, which is a force error and nothing else, so this is
        raised rather than masked.
    """
    depths = len(level_nodes)
    widths = [int(level.shape[0]) for level in level_nodes] or [0]
    # `None` means EXACTLY this schedule, which is what the docstring promises and
    # what the unpadded path needs. It used to mean `snap_capacity(...)`, and that
    # was wrong in a way worth spelling out, because it was invisible in every
    # correctness test: `snap_capacity`'s defaults are sized for the PAIR lists,
    # whose occupancies run to the thousands, so its `absolute=256` floor was being
    # added to a depth and a width of a few tens.
    #
    #   n = 256, leaf 8   depth 6, width 32, 63 nodes  ->  384 x 384 = 147,456 slots
    #
    # a 2340x inflation, and the cascades are a `lax.scan` over the rows, so that is
    # 384 iterations over 384 slots for a 63-node tree -- differentiated, since the
    # reverse pass keeps per-iteration residuals. Measured on the integration
    # shard, the same 173 tests, cold JAX cache, CI's own `-n 2 --dist loadgroup`:
    #
    #   merge-base library      11.87 GiB
    #   with the snapped default 37.91 GiB   <- OOM-killed the 16 GB CI runner
    #   sized exactly (this)    12.01 GiB
    #
    # 171 passed / 2 skipped in all three, so nothing about it was visible as a
    # failure -- only as a runner that stopped reporting.
    #
    # The padded path is unaffected: `caps` comes from `resolve_mutual_capacities`,
    # which snaps depth and width with headroom scaled to THEM (`absolute=8/4` and
    # `32/16`), not with the pair-list floor.
    d_cap = int(depth_cap) if depth_cap is not None else depths
    w_cap = int(width_cap) if width_cap is not None else max(widths)
    if depths > d_cap:
        raise ValueError(f"tree depth {depths} exceeds depth_cap {d_cap}")
    if max(widths) > w_cap:
        raise ValueError(f"widest level {max(widths)} exceeds width_cap {w_cap}")

    nodes = np.zeros((d_cap, w_cap), dtype=np.int64)
    parents = np.zeros((d_cap, w_cap), dtype=np.int64)
    valid = np.zeros((d_cap, w_cap), dtype=bool)
    for d, (lvl_nodes, lvl_parents) in enumerate(zip(level_nodes, level_parents)):
        n = int(np.asarray(lvl_nodes).shape[0])
        nodes[d, :n] = np.asarray(lvl_nodes)
        parents[d, :n] = np.asarray(lvl_parents)
        valid[d, :n] = True
    return (
        jnp.asarray(nodes, dtype=index_dtype),
        jnp.asarray(parents, dtype=index_dtype),
        jnp.asarray(valid),
        d_cap,
        w_cap,
    )


def _scan_levels(
    body: Any, carry: Any, tree: MutualTreeArrays, *, deepest_first: bool
) -> Any:
    """Run ``body`` once per tree level, in cascade order.

    The cascades are inherently sequential -- a level must be complete before the
    next reads it -- so this is a ``lax.scan``, not a ``vmap``. Replacing the
    Python loop it grew from costs nothing at runtime (the same number of
    kernels) and buys two things: the traced graph no longer grows with tree
    depth, and depth stops being part of the program's identity.

    Parameters
    ----------
    body : Any
        ``body(carry, nodes, parents, valid) -> carry``, applied once per level.
    carry : Any
        Value threaded through the levels.
    tree : MutualTreeArrays
        Supplies the dense ``(depth_cap, width_cap)`` level schedule.
    deepest_first : bool
        Reverse the scan, as the upward (M2M) cascade needs.

    Returns
    -------
    Any
        The final carry.
    """

    def step(acc: Any, level: Any) -> Any:
        nodes, parents, valid = level
        return body(acc, nodes, parents, valid), None

    out, _ = lax.scan(
        step,
        carry,
        (tree.level_nodes, tree.level_parents, tree.level_valid),
        reverse=bool(deepest_first),
    )
    return out


def _safe_translate(
    coeffs: Array,
    deltas: Array,
    translate: object,
    *,
    order: int,
) -> Array:
    """Apply a rotation-based translation, guarding the zero-displacement case.

    The rotate-scale operators normalise ``delta`` to build the alignment
    rotation, so a zero displacement is a ``0/0`` -- NaN in the forward for some
    orders and, worse, a NaN cotangent even where the forward survives. A zero
    displacement is a genuine configuration (a parent whose centre of mass
    coincides with its only massive child), and the correct answer there is the
    identity, so substitute a dummy axis and select the untranslated coefficients
    back. Both branches are evaluated, which is what keeps the reverse finite.

    Parameters
    ----------
    coeffs : Array
        ``(pairs, sh_size(order))`` coefficients to translate.
    deltas : Array
        ``(pairs, 3)`` displacements, in whatever convention the caller's
        ``translate`` expects.
    translate : object
        A rotate-scale translation operator, called as
        ``translate(coeffs_row, delta_row, order=order)``.
    order : int
        Expansion order.

    Returns
    -------
    Array
        Translated coefficients, with the degenerate rows passed through
        unchanged.
    """
    d2 = jnp.sum(deltas * deltas, axis=-1)
    # Anything at or below the operators' own squared-radius floor is degenerate
    # for them, not just an exact zero, so test against the same floor they use.
    live = d2 > jnp.asarray(squared_radius_floor(deltas.dtype), deltas.dtype)
    axis = jnp.zeros_like(deltas).at[:, 0].set(jnp.ones_like(d2))
    safe = jnp.where(live[:, None], deltas, axis)
    out = jax.vmap(lambda c, d: translate(c, d, order=int(order)))(coeffs, safe)
    return jnp.where(live[:, None], out, coeffs)


def _m2l_batch(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    use_pallas: bool,
    interpret: bool,
) -> Array:
    """Batched rotate + z-translate + rotate-back real M2L, differentiable either way.

    Three lanes, all computing the same operator as
    :func:`~jaccpot.operators.m2l_real_rot_scale.m2l_rot_scale_real_batch`:

    * **fused Pallas** (``use_pallas`` and Ampere+, or ``interpret``) -- one
      launch per chunk does rotate -> z-translate -> rotate-back on chip via
      :func:`~jaccpot.pallas.m2l_real_fused.m2l_real_fused_pallas_cvjp`. The
      rotations are handed over as explicit per-degree blocks, built by the *same*
      ``real_rotation_*`` ops the single-pair helpers use, so the two lanes are
      the same arithmetic in a different order.
    * **z-core Pallas** (``JACCPOT_MUTUAL_FUSED_M2L=0``) -- the original
      three-stage sandwich, kept as the A/B reference for the fused lane.
    * **pure JAX** -- the fallback and the correctness/AD oracle.

    Both Pallas lanes go through a ``custom_vjp`` wrapper rather than the bare
    kernel, and that distinction is load-bearing, not cosmetic. ``pallas_call``
    has no JVP/transpose rule, and jaccpot's own ``m2l_core_z_real`` helper calls
    the *bare* kernel, so routing through it would yield a forward that cannot be
    differentiated -- and on CPU the Pallas path is simply unsupported and
    silently falls back, so the failure would only appear on an actual GPU.

    ``interpret`` runs the Pallas kernel in interpret mode, which works on CPU and
    is how the parity tests exercise the kernel logic without a GPU.

    Parameters
    ----------
    multipoles : Array
        ``(pairs, sh_size(order))`` source multipoles, one row per directed pair.
    deltas : Array
        ``(pairs, 3)`` target centre minus source centre.
    order : int
        Expansion order.
    use_pallas : bool
        Request a Pallas lane; see :func:`_m2l_lane` for how it resolves.
    interpret : bool
        Run the Pallas kernel in interpret mode.

    Returns
    -------
    Array
        ``(pairs, sh_size(order))`` local coefficients at the target centres.
    """
    p = int(order)
    # NaN-safe radius: `norm` has a 0/0 reverse gradient at delta == 0. The
    # double-`where` keeps the cotangent finite there; the forward is unchanged.
    r2 = jnp.sum(deltas * deltas, axis=1)
    positive = r2 > 0
    radii = jnp.where(positive, jnp.sqrt(jnp.where(positive, r2, 1.0)), 0.0)
    floored = jnp.maximum(radii, jnp.asarray(1.0e-30, dtype=radii.dtype))

    lane = _m2l_lane(use_pallas, interpret)

    if lane == "fused":
        from jaccpot.pallas.m2l_real_fused import m2l_real_fused_pallas_cvjp

        dtype = multipoles.dtype
        blocks_to_z = real_rotation_blocks_to_z_multipole_batch(
            deltas, order=p, dtype=dtype
        )
        blocks_from_z = real_rotation_blocks_from_z_local_batch(
            deltas, order=p, dtype=dtype
        )
        return m2l_real_fused_pallas_cvjp(
            multipoles,
            blocks_to_z,
            blocks_from_z,
            floored,
            p,
            bool(interpret),
            "triton",
        )

    rotated = jax.vmap(lambda m, d: _rotate_multipole_to_z_single(m, d, order=p))(
        multipoles, deltas
    )

    if lane == "zcore":
        from jaccpot.pallas.m2l_core_z_real import m2l_core_z_real_pallas_cvjp

        locals_z = m2l_core_z_real_pallas_cvjp(
            rotated, floored, p, bool(interpret), "triton"
        )
    else:
        locals_z = jax.vmap(lambda m, rr: translate_along_z_m2l_real(m, rr, order=p))(
            rotated, floored
        )

    return jax.vmap(lambda l, d: _rotate_local_from_z_single(l, d, order=p))(
        locals_z, deltas
    )


def mutual_upward_sweep(
    positions: Array,
    masses: Array,
    tree: MutualTreeArrays,
) -> Tuple[Array, Array, Array]:
    """Return ``(node_mass, node_center, multipoles)`` for the live inputs.

    Centres are centres of mass, recomputed from the live positions/masses on
    every call -- only the *discrete* structure is frozen. Leaves are reduced
    from their padded particle blocks and internal nodes are then accumulated
    level by level from the deepest upward, so every node's children are complete
    before it is read.

    Parameters
    ----------
    positions : Array
        ``(n, 3)`` positions in Morton-sorted order.
    masses : Array
        ``(n,)`` masses in Morton-sorted order.
    tree : MutualTreeArrays
        Frozen device-resident topology.

    Returns
    -------
    Tuple[Array, Array, Array]
        ``(node_mass, node_center, multipoles)`` -- ``(num_nodes,)`` masses,
        ``(num_nodes, 3)`` centres of mass and ``(num_nodes, sh_size(order))``
        multipole coefficients.
    """
    p = int(tree.order)
    dtype = positions.dtype
    num_nodes = int(tree.num_nodes)

    particles = tree.leaf_particles
    valid = tree.leaf_particle_valid
    x = positions[particles]
    m = jnp.where(valid, masses[particles], jnp.zeros_like(masses[particles]))

    node_mass = (
        jnp.zeros((num_nodes,), dtype=dtype).at[tree.leaf_nodes].add(jnp.sum(m, axis=1))
    )
    node_moment = (
        jnp.zeros((num_nodes, 3), dtype=dtype)
        .at[tree.leaf_nodes]
        .add(jnp.sum(m[..., None] * x, axis=1))
    )

    # Deepest level first: `level_nodes[d]` sits at depth d+1, its parents at d.
    def _accumulate(acc, nodes, parents, valid):
        mass, moment = acc
        child = jnp.where(valid, nodes, 0)
        parent = jnp.where(valid, parents, 0)
        # Padding slots gather node 0 and contribute an additive zero, so they
        # land on node 0 as a no-op instead of double-counting it.
        mass = mass.at[parent].add(jnp.where(valid, mass[child], 0.0))
        moment = moment.at[parent].add(jnp.where(valid[:, None], moment[child], 0.0))
        return mass, moment

    node_mass, node_moment = _scan_levels(
        _accumulate, (node_mass, node_moment), tree, deepest_first=True
    )

    massive = node_mass > 0
    safe_mass = jnp.where(massive, node_mass, jnp.ones_like(node_mass))
    centers = jnp.where(massive[:, None], node_moment / safe_mass[:, None], 0.0)

    # P2M into the leaves, then M2M upward on the same level schedule.
    leaf_centers = centers[tree.leaf_nodes]
    deltas = x - leaf_centers[:, None, :]
    per_particle = jax.vmap(
        jax.vmap(lambda d, mm: p2m_real_direct(d, mm, order=p), in_axes=(0, 0)),
        in_axes=(0, 0),
    )(deltas, m)
    leaf_multipoles = jnp.sum(jnp.where(valid[..., None], per_particle, 0.0), axis=1)
    multipoles = (
        jnp.zeros((num_nodes, sh_size(p)), dtype=dtype)
        .at[tree.leaf_nodes]
        .add(leaf_multipoles.astype(dtype))
    )

    def _m2m(acc, nodes, parents, valid):
        child = jnp.where(valid, nodes, 0)
        parent = jnp.where(valid, parents, 0)
        # M2M convention: delta = child centre - parent centre.
        translated = _safe_translate(
            acc[child], centers[child] - centers[parent], m2m_real, order=p
        )
        translated = jnp.where(valid[:, None], translated, 0.0)
        return acc.at[parent].add(translated.astype(dtype))

    multipoles = _scan_levels(_m2m, multipoles, tree, deepest_first=True)

    return node_mass, centers, multipoles


def _dual_m2l(
    centers: Array,
    multipoles: Array,
    tree: MutualTreeArrays,
    far_weights: Optional[Array],
    *,
    use_pallas: bool,
    interpret: bool = False,
) -> Array:
    """Accumulate both directions of every well-separated pair into node locals.

    ``far_source``/``far_target`` already carry each canonical pair twice, once
    per direction, built from a single canonical list. Batching the two
    directions together is not just convenient: it guarantees the same kernel,
    the same order and the same rounding mode apply to both halves of a pair.

    Parameters
    ----------
    centers : Array
        ``(num_nodes, 3)`` centres of mass from the upward sweep.
    multipoles : Array
        ``(num_nodes, sh_size(order))`` multipole coefficients.
    tree : MutualTreeArrays
        Frozen device-resident topology supplying the directed pair list.
    far_weights : Optional[Array]
        Per-directed-entry weight, or ``None`` for unit weights. Both directions
        of a pair must carry the same weight or the cancellation breaks.
    use_pallas : bool
        Request a Pallas lane; see :func:`_m2l_lane` for how it resolves.
    interpret : bool
        Run the Pallas kernel in interpret mode.

    Returns
    -------
    Array
        ``(num_nodes, sh_size(order))`` accumulated local coefficients.
    """
    p = int(tree.order)
    dtype = centers.dtype
    num_nodes = int(tree.num_nodes)
    locals_ = jnp.zeros((num_nodes, sh_size(p)), dtype=dtype)

    n_pairs = int(tree.far_source.shape[0])
    if n_pairs == 0:
        return locals_

    budget = _M2L_BATCH_BUDGET
    if _m2l_lane(use_pallas, interpret) == "fused":
        budget = min(budget, _fused_m2l_chunk(p, dtype.itemsize))
    chunk = min(max(1, budget), n_pairs)
    steps = (n_pairs + chunk - 1) // chunk
    pad = steps * chunk - n_pairs
    source = tree.far_source
    target = tree.far_target
    valid = tree.far_valid
    weights = (
        jnp.ones((n_pairs,), dtype=dtype)
        if far_weights is None
        else jnp.asarray(far_weights, dtype=dtype)
    )
    if pad:
        source = jnp.concatenate([source, jnp.zeros((pad,), dtype=source.dtype)])
        target = jnp.concatenate([target, jnp.zeros((pad,), dtype=target.dtype)])
        valid = jnp.concatenate([valid, jnp.zeros((pad,), dtype=bool)])
        weights = jnp.concatenate([weights, jnp.zeros((pad,), dtype=dtype)])

    def body(acc: Array, idx: Array) -> tuple[Array, None]:
        start = idx * chunk
        src = lax.dynamic_slice_in_dim(source, start, chunk)
        tgt = lax.dynamic_slice_in_dim(target, start, chunk)
        live = lax.dynamic_slice_in_dim(valid, start, chunk)
        w = lax.dynamic_slice_in_dim(weights, start, chunk)
        # M2L convention: delta = target centre - source centre.
        delta = centers[tgt] - centers[src]
        # Padding slots carry src == tgt == 0, so their delta is exactly zero and
        # `_m2l_batch` floors the radius to 1e-30. The z-core then evaluates
        # r**-(p+1), which at that floor is 5.6e151 at order 4 and 2.0e213 at
        # order 6: finite in float64, but far past **inf in float32**. The
        # trailing `* where(live, w, 0)` below then turns inf * 0 into NaN rather
        # than dropping the slot, poisoning its target node -- which the L2L
        # cascade proceeds to broadcast across the tree.
        #
        # Substitute a unit axis *before* the reciprocal, the same double-`where`
        # discipline the kernels use; the zero weight still drops the slot.
        #
        # Two things kept this hidden: it needs float32 (float64 only overflows
        # from order 10, where 30*(p+1) passes 308), and it needs the directed
        # pair list not to divide the chunk -- which takes > 65536 directed far
        # pairs, i.e. roughly N > 10^4.
        safe_axis = jnp.zeros_like(delta).at[:, 2].set(jnp.ones_like(delta[:, 2]))
        delta = jnp.where(live[:, None], delta, safe_axis)
        contrib = _m2l_batch(
            multipoles[src],
            delta,
            order=p,
            use_pallas=use_pallas,
            interpret=interpret,
        )
        contrib = contrib * jnp.where(live, w, 0.0)[:, None].astype(contrib.dtype)
        return acc.at[tgt].add(contrib.astype(acc.dtype)), None

    locals_, _ = lax.scan(body, locals_, jnp.arange(steps))
    return locals_


def _push_locals_down(locals_: Array, centers: Array, tree: MutualTreeArrays) -> Array:
    """L2L cascade, shallowest level first so parents are complete when read.

    Parameters
    ----------
    locals_ : Array
        ``(num_nodes, sh_size(order))`` local coefficients from the M2L stage.
    centers : Array
        ``(num_nodes, 3)`` centres of mass from the upward sweep.
    tree : MutualTreeArrays
        Frozen device-resident topology supplying the level schedule.

    Returns
    -------
    Array
        The same array with each node's ancestors' expansions folded in.
    """
    p = int(tree.order)

    def _l2l(acc, nodes, parents, valid):
        child = jnp.where(valid, nodes, 0)
        parent = jnp.where(valid, parents, 0)
        # L2L convention: delta = parent centre - child centre.
        translated = _safe_translate(
            acc[parent], centers[parent] - centers[child], l2l_real, order=p
        )
        translated = jnp.where(valid[:, None], translated, 0.0)
        return acc.at[child].add(translated.astype(acc.dtype))

    return _scan_levels(_l2l, locals_, tree, deepest_first=False)


def _l2p_forces(
    positions: Array,
    masses: Array,
    centers: Array,
    locals_: Array,
    tree: MutualTreeArrays,
) -> Array:
    """Evaluate leaf local expansions at their particles and return forces.

    Forces, not accelerations: the mass factor is applied here and divided out
    once at the very end, because the quantity that cancels structurally between
    the two halves of a pair is the force.

    Parameters
    ----------
    positions : Array
        ``(n, 3)`` positions in Morton-sorted order.
    masses : Array
        ``(n,)`` masses in Morton-sorted order.
    centers : Array
        ``(num_nodes, 3)`` centres of mass from the upward sweep.
    locals_ : Array
        ``(num_nodes, sh_size(order))`` local coefficients, post-L2L.
    tree : MutualTreeArrays
        Frozen device-resident topology.

    Returns
    -------
    Array
        ``(n, 3)`` far-field forces, zero in the padded slots.
    """
    p = int(tree.order)
    particles = tree.leaf_particles
    valid = tree.leaf_particle_valid
    x = positions[particles]
    m = jnp.where(valid, masses[particles], jnp.zeros_like(masses[particles]))
    leaf_locals = locals_[tree.leaf_nodes]
    # L2P convention: delta = expansion centre - evaluation point, and the
    # returned gradient is d(phi)/d(delta); the acceleration is its negation.
    #
    # A degenerate offset needs no guard here. `evaluate_local_real` used to
    # branch the azimuth to a constant at rho == 0, which is forward-safe but has
    # no x/y derivative, so the transverse gradient vanished -- hit every time by
    # a leaf holding a single particle, since the particle IS its leaf's centre of
    # mass. That is fixed at the source: the operator now divides by the floored
    # rho unconditionally, so the degree-1 limit falls out of the algebra. This
    # module carried a nudge-the-offset workaround until then.
    offsets = centers[tree.leaf_nodes][:, None, :] - x
    grads = jax.vmap(
        lambda coeffs, offs: jax.vmap(
            lambda o: evaluate_local_real_with_grad(coeffs, o, order=p)[0]
        )(offs),
        in_axes=(0, 0),
    )(leaf_locals, offsets)
    contrib = jnp.where(valid[..., None], -grads * m[..., None], 0.0)
    return jnp.zeros_like(positions).at[particles].add(contrib.astype(positions.dtype))


def mutual_far_field_forces(
    positions: Array,
    masses: Array,
    tree: MutualTreeArrays,
    *,
    G: float = 1.0,
    far_weights: Optional[Array] = None,
    use_pallas: bool = False,
    interpret: bool = False,
    return_multipoles: bool = False,
) -> Array | Tuple[Array, Array, Array]:
    """Return the mutual far-field **force** on every particle.

    ``far_weights`` scales each directed pair entry. Because the two directions of
    a pair carry the *same* weight, the scaling commutes with the ``F_A + F_B``
    cancellation and momentum stays exact for any weighting -- which is what lets
    one traversal serve a whole block-step boundary.

    Parameters
    ----------
    positions : Array
        ``(n, 3)`` positions in Morton-sorted order.
    masses : Array
        ``(n,)`` masses in Morton-sorted order.
    tree : MutualTreeArrays
        Frozen device-resident topology.
    G : float
        Gravitational constant. Default ``1.0``.
    far_weights : Optional[Array]
        Per-directed-entry weight, or ``None`` for unit weights.
    use_pallas : bool
        Request a Pallas lane; see :func:`_m2l_lane` for how it resolves.
    interpret : bool
        Run the Pallas kernel in interpret mode.
    return_multipoles : bool
        Also return the upward sweep's node masses and centres.

    Returns
    -------
    Array | Tuple[Array, Array, Array]
        ``(n, 3)`` forces, or ``(forces, node_mass, centers)`` when
        ``return_multipoles`` is set.
    """
    node_mass, centers, multipoles = mutual_upward_sweep(positions, masses, tree)
    locals_ = _dual_m2l(
        centers,
        multipoles,
        tree,
        far_weights,
        use_pallas=use_pallas,
        interpret=interpret,
    )
    locals_ = _push_locals_down(locals_, centers, tree)
    forces = _l2p_forces(positions, masses, centers, locals_, tree) * jnp.asarray(
        G, dtype=positions.dtype
    )
    if return_multipoles:
        return forces, node_mass, centers
    return forces

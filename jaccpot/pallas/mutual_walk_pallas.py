"""The mutual dual-tree walk with ONE Pallas launch per wavefront round (plan sub-10ms, Phase 2, W1).

``yggdrax.interactions.dual_tree_walk_mutual`` runs each round of the flat
wavefront walk as ~40 XLA kernels (masked MAC over the queue, three prefix-sum
appends, the refinement expansion, a compaction of the next queue). On the
40-level radix tree over Morton-cell leaves at N = 2x10^5 that is 71 rounds
and 11.5 ms -- the launch floor, since the queue is already sized to the peak
wavefront (85.6k pairs).

Here a round is one ``pallas_call``: a program owns a block of ``B`` queue
slots, gathers both nodes' centres, radii and children, evaluates the MAC
(``bh``/``dehnen``: ``(r_a + r_b)^2 <= theta^2 d^2``), and emits

* accepted pairs to the far list, leaf-leaf rejects to the near list,
* the refinement of every other reject (split the larger node; both when
  equal or ``a == b``) -- up to four child pairs -- to the next queue,

each through a per-lane ``atomic_add`` on a counter that hands out slots, so
the lists come out as SETS in a nondeterministic order (the consumers sort
them by target and source, which makes the fused lane deterministic again).
A slot past the buffer's capacity is dropped and recorded in a flag, exactly
as the flat walk's saturation contract requires. Dead nodes (empty padding of
a capacity-padded leaf partition) are masked by ``node_active``.

The pair convention follows the flat walk: canonical pairs ``(lo, hi)``, the
root paired with itself first, near pairs only between two different leaves.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot._env import env_int
from jaccpot.pallas._compat import KernelRef, pallas_backend_kwargs
from jaccpot.pallas.m2l_real_csr import pallas_m2l_real_csr_supported

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = ["PallasWalkResult", "mutual_walk_pallas", "pallas_mutual_walk_supported"]


def pallas_mutual_walk_supported() -> bool:
    """True where the Triton lowering runs (sm_80+).

    Returns
    -------
    bool
        Whether the kernel can run natively here.
    """
    return pallas_m2l_real_csr_supported()


class PallasWalkResult(NamedTuple):
    """Far and near pair lists of the walk, ``-1``-padded to their capacities.

    Attributes
    ----------
    far_a : Array
        Lower node of each canonical far pair, live prefix ``far_count``.
    far_b : Array
        Upper node of each canonical far pair.
    far_count : Array
        Number of far pairs in the list, at most the capacity.
    near_a : Array
        Lower leaf of each canonical near pair, live prefix ``near_count``.
    near_b : Array
        Upper leaf of each canonical near pair.
    near_count : Array
        Number of near pairs emitted.
    far_overflow : Array
        The far list hit its capacity, so the lists are incomplete.
    near_overflow : Array
        The near list hit its capacity.
    queue_overflow : Array
        The wavefront queue hit its capacity.
    peak_wavefront : Array
        Largest queue occupancy seen.
    rounds : Array
        Rounds executed.
    far_needed : Array
        Far pairs the walk found, stored or not. A far or near overflow does not
        stop the walk, so this is the capacity the list needed -- exact unless
        ``queue_overflow`` (pairs never classified), when it is a lower bound.
    near_needed : Array
        The same for the near list.
    """

    far_a: Array
    far_b: Array
    far_count: Array
    near_a: Array
    near_b: Array
    near_count: Array
    far_overflow: Array
    near_overflow: Array
    queue_overflow: Array
    peak_wavefront: Array
    rounds: Array
    far_needed: Array
    near_needed: Array


# counter slots
#: counter slots: far, near, next-queue, and one overflow flag EACH (an OR of
#: bits under atomic_max would let a later flag erase an earlier one)
_C_FAR, _C_NEAR, _C_NEXT, _C_OVF_FAR, _C_OVF_NEAR, _C_OVF_Q = 0, 1, 2, 3, 4, 5
_NUM_COUNTERS = 6


def _round_kernel(
    qa_ref: KernelRef,
    qb_ref: KernelRef,
    size_ref: KernelRef,
    left_ref: KernelRef,
    right_ref: KernelRef,
    cent_ref: KernelRef,
    rad_ref: KernelRef,
    active_ref: KernelRef,
    theta_ref: KernelRef,
    far_a_ref: KernelRef,
    far_b_ref: KernelRef,
    near_a_ref: KernelRef,
    near_b_ref: KernelRef,
    next_a_ref: KernelRef,
    next_b_ref: KernelRef,
    counters_ref: KernelRef,
    far_a_out: KernelRef,
    far_b_out: KernelRef,
    near_a_out: KernelRef,
    near_b_out: KernelRef,
    next_a_out: KernelRef,
    next_b_out: KernelRef,
    counters_out: KernelRef,
    *,
    block: int,
    far_cap: int,
    near_cap: int,
    queue_cap: int,
) -> None:
    """One block of the wavefront: MAC, emit, refine.

    Parameters
    ----------
    qa_ref : KernelRef
        Current queue's first node per pair ``[Q]`` (``-1`` past ``size``).
    qb_ref : KernelRef
        Current queue's second node per pair ``[Q]``.
    size_ref : KernelRef
        Live pair count ``[1]``.
    left_ref : KernelRef
        Left child per node ``[nodes]``, ``-1`` for leaves.
    right_ref : KernelRef
        Right child per node ``[nodes]``, ``-1`` for leaves.
    cent_ref : KernelRef
        Centres ``[nodes, 4]`` (padded).
    rad_ref : KernelRef
        MAC radii ``[nodes]``.
    active_ref : KernelRef
        Node activity ``[nodes]`` (``int32`` 0/1).
    theta_ref : KernelRef
        ``theta^2`` ``[1]``.
    far_a_ref : KernelRef
        Far list, first node ``[far_cap]``; aliased to ``far_a_out``.
    far_b_ref : KernelRef
        Far list, second node ``[far_cap]``.
    near_a_ref : KernelRef
        Near list, first leaf ``[near_cap]``.
    near_b_ref : KernelRef
        Near list, second leaf ``[near_cap]``.
    next_a_ref : KernelRef
        Next queue, first node ``[queue_cap]``.
    next_b_ref : KernelRef
        Next queue, second node ``[queue_cap]``.
    counters_ref : KernelRef
        Counters ``[6]``: far, near, next, and one overflow flag each for
        far, near and the queue.
    far_a_out : KernelRef
        Output alias of ``far_a_ref``, written in place.
    far_b_out : KernelRef
        Output alias of ``far_b_ref``.
    near_a_out : KernelRef
        Output alias of ``near_a_ref``.
    near_b_out : KernelRef
        Output alias of ``near_b_ref``.
    next_a_out : KernelRef
        Output alias of ``next_a_ref``.
    next_b_out : KernelRef
        Output alias of ``next_b_ref``.
    counters_out : KernelRef
        Output alias of ``counters_ref``; the atomics target this ref.
    block : int
        Pairs per program. Static.
    far_cap : int
        Far list capacity. Static.
    near_cap : int
        Near list capacity. Static.
    queue_cap : int
        Next-queue capacity. Static.

    Returns
    -------
    None
        Writes into the aliased outputs.
    """
    del (
        far_a_ref,
        far_b_ref,
        near_a_ref,
        near_b_ref,
        next_a_ref,
        next_b_ref,
        counters_ref,
    )
    pid = pl.program_id(0)
    size = size_ref[0]
    # Grid-stride over the live blocks: the grid is capped (``num_programs``), so
    # a round over a small queue no longer launches the full queue capacity's
    # programs only for most of them to exit (66 rounds x 131k programs per
    # refresh at 8e6). Which program emits a pair changes its slot, not the set:
    # the deterministic lists sort it away.
    num_programs = pl.num_programs(0)
    live_blocks = (size + (block - 1)) // block
    trips = jnp.maximum(live_blocks - pid + (num_programs - 1), 0) // num_programs

    def _one(it, carry):
        _round_block(
            qa_ref,
            qb_ref,
            size,
            left_ref,
            right_ref,
            cent_ref,
            rad_ref,
            active_ref,
            theta_ref,
            far_a_out,
            far_b_out,
            near_a_out,
            near_b_out,
            next_a_out,
            next_b_out,
            counters_out,
            pid=pid + it * num_programs,
            block=block,
            far_cap=far_cap,
            near_cap=near_cap,
            queue_cap=queue_cap,
        )
        return carry

    lax.fori_loop(0, trips, _one, 0)


def _round_block(
    qa_ref: KernelRef,
    qb_ref: KernelRef,
    size: Array,
    left_ref: KernelRef,
    right_ref: KernelRef,
    cent_ref: KernelRef,
    rad_ref: KernelRef,
    active_ref: KernelRef,
    theta_ref: KernelRef,
    far_a_out: KernelRef,
    far_b_out: KernelRef,
    near_a_out: KernelRef,
    near_b_out: KernelRef,
    next_a_out: KernelRef,
    next_b_out: KernelRef,
    counters_out: KernelRef,
    *,
    pid: Array,
    block: int,
    far_cap: int,
    near_cap: int,
    queue_cap: int,
) -> None:
    """Body of one non-empty block (see :func:`_round_kernel`).

    Parameters
    ----------
    qa_ref : KernelRef
        Current queue's first node per pair ``[Q]``.
    qb_ref : KernelRef
        Current queue's second node per pair ``[Q]``.
    size : Array
        Live pair count, already loaded from the size ref.
    left_ref : KernelRef
        Left child per node ``[nodes]``, ``-1`` for leaves.
    right_ref : KernelRef
        Right child per node ``[nodes]``.
    cent_ref : KernelRef
        Centres ``[nodes, 4]`` (padded).
    rad_ref : KernelRef
        MAC radii ``[nodes]``.
    active_ref : KernelRef
        Node activity ``[nodes]`` (``int32`` 0/1).
    theta_ref : KernelRef
        ``theta^2`` ``[1]``.
    far_a_out : KernelRef
        Far list, first node; written in place.
    far_b_out : KernelRef
        Far list, second node.
    near_a_out : KernelRef
        Near list, first leaf.
    near_b_out : KernelRef
        Near list, second leaf.
    next_a_out : KernelRef
        Next queue, first node.
    next_b_out : KernelRef
        Next queue, second node.
    counters_out : KernelRef
        Counters ``[6]``; the atomics target this ref.
    pid : Array
        This program's index in the grid.
    block : int
        Pairs per program. Static.
    far_cap : int
        Far list capacity. Static.
    near_cap : int
        Near list capacity. Static.
    queue_cap : int
        Next-queue capacity. Static.

    Returns
    -------
    None
        Writes into the aliased outputs.
    """
    lane = (pid * block + lax.broadcasted_iota(jnp.int32, (block,), 0)).astype(
        jnp.int32
    )
    in_range = lane < size
    lane_safe = jnp.where(in_range, lane, jnp.zeros_like(lane))
    a = qa_ref[lane_safe]
    b = qb_ref[lane_safe]
    live = in_range & (a >= 0) & (b >= 0)
    a_s = jnp.where(live, a, jnp.zeros_like(a))
    b_s = jnp.where(live, b, jnp.zeros_like(b))
    live = live & (active_ref[a_s] > 0) & (active_ref[b_s] > 0)
    la = left_ref[a_s]
    lb = left_ref[b_s]
    ra = right_ref[a_s]
    rb = right_ref[b_s]
    rad_a = rad_ref[a_s]
    rad_b = rad_ref[b_s]
    dx = cent_ref[b_s, 0] - cent_ref[a_s, 0]
    dy = cent_ref[b_s, 1] - cent_ref[a_s, 1]
    dz = cent_ref[b_s, 2] - cent_ref[a_s, 2]
    d2 = dx * dx + dy * dy + dz * dz
    same = a_s == b_s
    rsum = rad_a + rad_b
    theta_sq = theta_ref[0]
    accept = live & (~same) & (d2 > 0.0) & (rsum * rsum <= theta_sq * d2)
    a_leaf = la < 0
    b_leaf = lb < 0
    both_leaf = a_leaf & b_leaf
    is_near = live & (~accept) & both_leaf & (~same)
    refine = live & (~accept) & (~both_leaf)
    split_a = refine & (~a_leaf) & (same | b_leaf | (rad_a >= rad_b))
    split_b = refine & (~b_leaf) & (same | a_leaf | (rad_b > rad_a))
    both = split_a & split_b
    only_a = split_a & (~split_b)
    only_b = split_b & (~split_a)
    zero = jnp.zeros_like(lane)
    one = jnp.ones_like(lane)
    neg1 = jnp.full_like(lane, -1)

    def emit(
        mask: Array,
        va: Array,
        vb: Array,
        out_a: KernelRef,
        out_b: KernelRef,
        counter: int,
        cap: int,
    ) -> Array:
        # One atomic per program: the block claims sum(mask) slots and hands them
        # out by an in-block exclusive prefix sum (no per-lane counter contention).
        inc = jnp.where(mask, one, zero).astype(jnp.int32)
        total = jnp.sum(inc).astype(jnp.int32)
        base = plgpu.atomic_add(counters_out, (jnp.asarray(counter, jnp.int32),), total)
        slot = (base + jnp.cumsum(inc) - inc).astype(jnp.int32)
        ok = mask & (slot < cap)
        over = mask & (slot >= cap)
        lo = jnp.minimum(va, vb)
        hi = jnp.maximum(va, vb)
        # Masked-out lanes point PAST the buffer (unique, out of range): the
        # Triton store never touches them and the interpreter drops them, whereas
        # a shared in-range dummy index collides with a real write to that slot.
        where = jnp.where(ok, slot, (cap + lane).astype(jnp.int32))
        plgpu.store(out_a.at[where], lo.astype(jnp.int32), mask=ok)
        plgpu.store(out_b.at[where], hi.astype(jnp.int32), mask=ok)
        return over

    over_far = emit(accept, a_s, b_s, far_a_out, far_b_out, _C_FAR, far_cap)
    over_near = emit(is_near, a_s, b_s, near_a_out, near_b_out, _C_NEAR, near_cap)
    # refinement: up to four child pairs per lane (the flat walk's five cases)
    # case split both & same:   (la,la) (la,ra) (ra,ra)
    # case split both & cross:  (la,lb) (la,rb) (ra,lb) (ra,rb)
    # case split a only:        (la,b) (ra,b)
    # case split b only:        (a,lb) (a,rb)
    c0a = jnp.where(both & same, la, jnp.where(both, la, jnp.where(only_a, la, a_s)))
    c0b = jnp.where(both & same, la, jnp.where(both, lb, jnp.where(only_a, b_s, lb)))
    c1a = jnp.where(both & same, la, jnp.where(both, la, jnp.where(only_a, ra, a_s)))
    c1b = jnp.where(both & same, ra, jnp.where(both, rb, jnp.where(only_a, b_s, rb)))
    c2a = jnp.where(both & same, ra, jnp.where(both, ra, neg1))
    c2b = jnp.where(both & same, ra, jnp.where(both, lb, neg1))
    c3a = jnp.where(both & (~same), ra, neg1)
    c3b = jnp.where(both & (~same), rb, neg1)
    over_q = jnp.zeros_like(live)
    for ca, cb in ((c0a, c0b), (c1a, c1b), (c2a, c2b), (c3a, c3b)):
        m = refine & (ca >= 0) & (cb >= 0)
        over_q = over_q | emit(m, ca, cb, next_a_out, next_b_out, _C_NEXT, queue_cap)
    for slot_id, over in (
        (_C_OVF_FAR, over_far),
        (_C_OVF_NEAR, over_near),
        (_C_OVF_Q, over_q),
    ):
        plgpu.atomic_max(
            counters_out,
            (jnp.asarray(slot_id, jnp.int32),),
            jnp.max(over.astype(jnp.int32)),
        )


def _full(arr: Array) -> "pl.BlockSpec":
    shp = tuple(arr.shape)
    return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))


def mutual_walk_pallas(
    left_child_full: Array,
    right_child_full: Array,
    centers: Array,
    radii: Array,
    theta: float,
    root: Array,
    *,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
    node_active: Optional[Array] = None,
    block: int = 64,
    interpret: bool = False,
    backend: str = "triton",
    num_warps: int = 2,
    max_rounds: int = 256,
    rounds_per_check: int = 16,
    seed_a: Optional[Array] = None,
    seed_b: Optional[Array] = None,
    seed_count: Optional[Array] = None,
) -> PallasWalkResult:
    """Run the mutual walk, one Pallas launch per round.

    Same contract as ``dual_tree_walk_mutual`` (``mac_type="dehnen"`` / ``"bh"``
    semantics: non-strict ``(r_a + r_b)^2 <= theta^2 d^2``), lists returned as
    sets in atomic emission order.

    Parameters
    ----------
    left_child_full : Array
        ``[nodes]`` left children, ``-1`` for leaves.
    right_child_full : Array
        ``[nodes]`` right children, ``-1`` for leaves.
    centers : Array
        ``[nodes, 3]`` MAC centres.
    radii : Array
        ``[nodes]`` MAC radii.
    theta : float
        Opening angle.
    root : Array
        Root node id (scalar).
    max_pair_queue : int
        Queue capacity (also the block-padded grid). Static.
    far_cap : int
        Far list capacity. Static.
    near_cap : int
        Near list capacity. Static.
    node_active : Optional[Array]
        ``[nodes]`` bool; ``None`` = all active.
    block : int
        Pairs per program (power of two). Static.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    max_rounds : int
        Safety bound on the round loop.
    rounds_per_check : int
        Rounds launched per ``while_loop`` iteration: every iteration costs a
        device-to-host copy of the predicate AND a copy of every carried buffer
        (XLA preserves the loop state around the aliased pallas_call: 19 D2D
        memcpys per iteration at N=2e5), so many rounds run between checks; an
        empty round is one early-exiting launch plus a few scalar ops.
    seed_a : Optional[Array]
        ``[K]`` first nodes of the starting pairs in place of ``(root, root)``,
        ``-1`` for dead slots (the walk filters them). ``K <= max_pair_queue``. This
        is what makes the kernel a ONE-SIDED walk: a node with no children (an
        imported row, a summary cell) is never split, so only the other side refines
        -- the same contract as ``dual_tree_walk_mutual``'s seeds, used by the
        multi-GPU export and receiver walks. Pairs come out as ``(min, max)``, so a
        block indexed above the local nodes always lands in ``*_b``.
    seed_b : Optional[Array]
        ``[K]`` second nodes of the starting pairs; given together with ``seed_a``.
    seed_count : Optional[Array]
        Live prefix of the seed (a width, not a filter: dead slots past it are
        never read). ``None`` uses ``K``.

    Returns
    -------
    PallasWalkResult
        The lists, counts and flags. ``queue_overflow`` is also raised when the
        walk stops at ``max_rounds`` with pairs still queued -- those pairs were
        never classified, so the lists are incomplete. A far or near overflow does
        not stop the walk: ``far_needed`` / ``near_needed`` then hold the capacity
        the list needed.

    Raises
    ------
    ValueError
        If the seed is longer than ``max_pair_queue`` or only one half is given.
    """
    if (seed_a is None) != (seed_b is None):
        raise ValueError("seed_a and seed_b go together")
    if seed_a is not None:
        K = int(jnp.asarray(seed_a).shape[0])
        if K > int(max_pair_queue):
            raise ValueError(
                f"seed of {K} pairs exceeds max_pair_queue={int(max_pair_queue)}"
            )
    # One jit around the whole walk, so an EAGER call creates the list and queue
    # buffers inside the program and the loop updates them in place. Called op by
    # op, the initial buffers were arguments of the while loop and stayed alive
    # beside its outputs: twice the walk's memory at the peak. Under a trace the
    # jit simply inlines.
    return _mutual_walk_jit(
        left_child_full,
        right_child_full,
        centers,
        radii,
        root,
        node_active,
        seed_a,
        seed_b,
        seed_count,
        theta=float(theta),
        max_pair_queue=int(max_pair_queue),
        far_cap=int(far_cap),
        near_cap=int(near_cap),
        block=int(block),
        interpret=bool(interpret),
        backend=str(backend),
        num_warps=int(num_warps),
        max_rounds=int(max_rounds),
        rounds_per_check=int(rounds_per_check),
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "theta",
        "max_pair_queue",
        "far_cap",
        "near_cap",
        "block",
        "interpret",
        "backend",
        "num_warps",
        "max_rounds",
        "rounds_per_check",
    ),
)
@jax.named_scope("fmm_walk")
def _mutual_walk_jit(
    left_child_full: Array,
    right_child_full: Array,
    centers: Array,
    radii: Array,
    root: Array,
    node_active: Optional[Array],
    seed_a: Optional[Array],
    seed_b: Optional[Array],
    seed_count: Optional[Array],
    *,
    theta: float,
    max_pair_queue: int,
    far_cap: int,
    near_cap: int,
    block: int,
    interpret: bool,
    backend: str,
    num_warps: int,
    max_rounds: int,
    rounds_per_check: int,
) -> PallasWalkResult:
    """The body of :func:`mutual_walk_pallas` (validated arguments, static sizes).

    Parameters
    ----------
    left_child_full : Array
        ``[nodes]`` left children.
    right_child_full : Array
        ``[nodes]`` right children.
    centers : Array
        ``[nodes, 3]`` MAC centres.
    radii : Array
        ``[nodes]`` MAC radii.
    root : Array
        Root node id.
    node_active : Optional[Array]
        ``[nodes]`` liveness mask.
    seed_a : Optional[Array]
        Seed pairs, first nodes.
    seed_b : Optional[Array]
        Seed pairs, second nodes.
    seed_count : Optional[Array]
        Live seed prefix.
    theta : float
        Opening angle.
    max_pair_queue : int
        Queue capacity.
    far_cap : int
        Far list capacity.
    near_cap : int
        Near list capacity.
    block : int
        Pairs per program.
    interpret : bool
        Pallas interpret mode.
    backend : str
        Pallas GPU lowering.
    num_warps : int
        Warps per program.
    max_rounds : int
        Safety bound on the round loop.
    rounds_per_check : int
        Rounds per ``while_loop`` iteration.

    Returns
    -------
    PallasWalkResult
        As :func:`mutual_walk_pallas`.
    """
    idx = jnp.int32
    nodes = int(left_child_full.shape[0])
    left = jnp.asarray(left_child_full, idx)
    right = jnp.asarray(right_child_full, idx)
    dtype = centers.dtype
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    rad = jnp.asarray(radii, dtype)
    active = (
        jnp.ones((nodes,), idx)
        if node_active is None
        else jnp.asarray(node_active).astype(idx)
    )
    theta_sq = jnp.asarray([float(theta) ** 2], dtype)
    Q = int(max_pair_queue)
    blk = int(block)
    # programs per round: enough to fill the card, the kernel strides over the rest
    max_programs = env_int("JACCPOT_WALK_MAX_PROGRAMS", 2048, minimum=1)
    grid = min((Q + blk - 1) // blk, int(max_programs))
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(
            num_warps=int(num_warps)
        )
    kernel = functools.partial(
        _round_kernel,
        block=blk,
        far_cap=int(far_cap),
        near_cap=int(near_cap),
        queue_cap=Q,
    )

    def one_round(
        qa: Array,
        qb: Array,
        size: Array,
        far_a: Array,
        far_b: Array,
        near_a: Array,
        near_b: Array,
        next_a: Array,
        next_b: Array,
        counters: Array,
    ) -> tuple[Array, ...]:  # pallas_call returns a tuple, not a list
        operands = [
            qa,
            qb,
            size,
            left,
            right,
            cent,
            rad,
            active,
            theta_sq,
            far_a,
            far_b,
            near_a,
            near_b,
            next_a,
            next_b,
            counters,
        ]
        outs = [far_a, far_b, near_a, near_b, next_a, next_b, counters]
        n_in = len(operands)
        alias = {n_in - 7 + k: k for k in range(7)}
        return pl.pallas_call(
            kernel,
            grid=(grid,),
            in_specs=[_full(o) for o in operands],
            out_specs=[_full(o) for o in outs],
            out_shape=[jax.ShapeDtypeStruct(o.shape, o.dtype) for o in outs],
            input_output_aliases=alias,
            interpret=bool(interpret),
            name=f"mutual_walk_round_b{blk}",
            **backend_kwargs,
        )(*operands)

    if seed_a is None:
        qa0 = jnp.full((Q,), -1, idx).at[0].set(jnp.asarray(root, idx))
        qb0 = qa0
        size0 = jnp.asarray(1, idx)
    else:
        assert seed_b is not None  # checked above; for the type checker
        K = int(jnp.asarray(seed_a).shape[0])
        qa0 = jnp.full((Q,), -1, idx).at[:K].set(jnp.asarray(seed_a, idx))
        qb0 = jnp.full((Q,), -1, idx).at[:K].set(jnp.asarray(seed_b, idx))
        size0 = (
            jnp.asarray(K, idx)
            if seed_count is None
            else jnp.minimum(jnp.asarray(seed_count, idx), jnp.asarray(K, idx))
        )
    init = (
        qa0,
        qb0,
        # the spare queue pair the next round writes into (double buffering)
        jnp.zeros((Q,), idx),
        jnp.zeros((Q,), idx),
        size0,
        jnp.full((int(far_cap),), -1, idx),
        jnp.full((int(far_cap),), -1, idx),
        jnp.full((int(near_cap),), -1, idx),
        jnp.full((int(near_cap),), -1, idx),
        jnp.zeros((_NUM_COUNTERS,), idx),  # far, near, next, overflow far/near/queue
        size0,  # peak
        jnp.asarray(0, idx),  # rounds
    )

    def cond(state: tuple[Array, ...]) -> Array:
        _qa, _qb, _sa, _sb, size, *_rest, counters, _peak, rounds = state
        # Only a QUEUE overflow stops the walk: pairs dropped from the queue are
        # never classified. A full far or near list drops what does not fit and
        # keeps counting, so the walk reports the capacity it needed and an eager
        # caller can size the list in one retry instead of a doubling ladder.
        return (size > 0) & (counters[_C_OVF_Q] == 0) & (rounds < int(max_rounds))

    def one_step(state: tuple[Array, ...]) -> tuple[Array, ...]:
        (
            qa,
            qb,
            spare_a,
            spare_b,
            size,
            far_a,
            far_b,
            near_a,
            near_b,
            counters,
            peak,
            rounds,
        ) = state
        # The next queue goes into the spare pair (the kernel reads only lanes <
        # size, so neither needs a fill) and this round's queue becomes the next
        # spare. A fresh ``jnp.empty`` per round cost a full-queue copy per round
        # (XLA merged the two identical broadcasts into one buffer, and an aliased
        # output cannot share it): 33 MB x ~2.4 per round at 8e6, 6.5 ms per step.
        # After an even number of rounds every carry slot is back in its own
        # buffer, so the while body needs no copy either.
        counters = counters.at[_C_NEXT].set(0)
        far_a, far_b, near_a, near_b, next_a, next_b, counters = one_round(
            qa, qb, size[None], far_a, far_b, near_a, near_b, spare_a, spare_b, counters
        )
        new_size = counters[_C_NEXT]
        peak = jnp.maximum(peak, new_size)
        # an empty round leaves everything untouched, so the count stays honest
        rounds = rounds + jnp.where(size > 0, 1, 0).astype(idx)
        return (
            next_a,
            next_b,
            qa,
            qb,
            jnp.minimum(new_size, Q),
            far_a,
            far_b,
            near_a,
            near_b,
            counters,
            peak,
            rounds,
        )

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        for _ in range(max(int(rounds_per_check), 1)):
            state = one_step(state)
        return state

    (
        qa,
        qb,
        _sa,
        _sb,
        size,
        far_a,
        far_b,
        near_a,
        near_b,
        counters,
        peak,
        rounds,
    ) = lax.while_loop(cond, body, init)
    # Stopping at max_rounds with pairs still queued leaves them unclassified. A
    # queue overflow also stops the loop with pairs queued and is flagged already.
    unfinished = (size > 0) & (counters[_C_OVF_Q] == 0)
    return PallasWalkResult(
        far_a=far_a,
        far_b=far_b,
        far_count=jnp.minimum(counters[_C_FAR], int(far_cap)),
        near_a=near_a,
        near_b=near_b,
        near_count=jnp.minimum(counters[_C_NEAR], int(near_cap)),
        far_overflow=counters[_C_OVF_FAR] > 0,
        near_overflow=counters[_C_OVF_NEAR] > 0,
        queue_overflow=(counters[_C_OVF_Q] > 0) | unfinished,
        peak_wavefront=peak,
        rounds=rounds,
        far_needed=counters[_C_FAR],
        near_needed=counters[_C_NEAR],
    )

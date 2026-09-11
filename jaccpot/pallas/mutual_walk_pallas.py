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
    far_a, far_b : Array
        Canonical far pairs (``a < b``), live prefix ``far_count``.
    far_count : Array
        Number of far pairs emitted (may exceed the capacity when ``far_overflow``).
    near_a, near_b : Array
        Canonical leaf-leaf near pairs, live prefix ``near_count``.
    near_count : Array
        Number of near pairs emitted.
    far_overflow, near_overflow, queue_overflow : Array
        Capacity flags; any of them means the lists are incomplete.
    peak_wavefront : Array
        Largest queue occupancy seen.
    rounds : Array
        Rounds executed.
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
    qa_ref, qb_ref : KernelRef
        Current queue ``[Q]`` (canonical pairs, ``-1`` past ``size``).
    size_ref : KernelRef
        Live pair count ``[1]``.
    left_ref, right_ref : KernelRef
        Children per node ``[nodes]``, ``-1`` for leaves.
    cent_ref : KernelRef
        Centres ``[nodes, 4]`` (padded).
    rad_ref : KernelRef
        MAC radii ``[nodes]``.
    active_ref : KernelRef
        Node activity ``[nodes]`` (``int32`` 0/1).
    theta_ref : KernelRef
        ``theta^2`` ``[1]``.
    far_a_ref .. counters_ref : KernelRef
        The list buffers and the counters ``[6]`` (far, near, next, overflow
        far/near/queue); aliased to the ``*_out`` refs, written in place.
    block : int
        Pairs per program. Static.
    far_cap, near_cap, queue_cap : int
        Buffer capacities. Static.

    Returns
    -------
    None
        Writes into the aliased outputs.
    """
    del far_a_ref, far_b_ref, near_a_ref, near_b_ref, next_a_ref, next_b_ref, counters_ref
    pid = pl.program_id(0)
    size = size_ref[0]

    @pl.when(pid * block < size)
    def _block():
        _round_block(
            qa_ref, qb_ref, size, left_ref, right_ref, cent_ref, rad_ref, active_ref, theta_ref,
            far_a_out, far_b_out, near_a_out, near_b_out, next_a_out, next_b_out, counters_out,
            pid=pid, block=block, far_cap=far_cap, near_cap=near_cap, queue_cap=queue_cap,
        )


def _round_block(
    qa_ref, qb_ref, size, left_ref, right_ref, cent_ref, rad_ref, active_ref, theta_ref,
    far_a_out, far_b_out, near_a_out, near_b_out, next_a_out, next_b_out, counters_out,
    *, pid, block, far_cap, near_cap, queue_cap,
):
    """Body of one non-empty block (see :func:`_round_kernel`)."""
    lane = (pid * block + lax.broadcasted_iota(jnp.int32, (block,), 0)).astype(jnp.int32)
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

    def emit(mask, va, vb, out_a, out_b, counter, cap):
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
    for slot_id, over in ((_C_OVF_FAR, over_far), (_C_OVF_NEAR, over_near), (_C_OVF_Q, over_q)):
        plgpu.atomic_max(
            counters_out, (jnp.asarray(slot_id, jnp.int32),), jnp.max(over).astype(jnp.int32)
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
    rounds_per_check: int = 4,
) -> PallasWalkResult:
    """Run the mutual walk, one Pallas launch per round.

    Same contract as ``dual_tree_walk_mutual`` (``mac_type="dehnen"`` / ``"bh"``
    semantics: non-strict ``(r_a + r_b)^2 <= theta^2 d^2``), lists returned as
    sets in atomic emission order.

    Parameters
    ----------
    left_child_full, right_child_full : Array
        ``[nodes]`` children, ``-1`` for leaves.
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
    far_cap, near_cap : int
        List capacities. Static.
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
        device-to-host copy of the loop predicate, so several rounds run
        between checks (an empty round is one early-exiting launch).

    Returns
    -------
    PallasWalkResult
        The lists, counts and flags.
    """
    idx = jnp.int32
    nodes = int(left_child_full.shape[0])
    left = jnp.asarray(left_child_full, idx)
    right = jnp.asarray(right_child_full, idx)
    dtype = centers.dtype
    cent = jnp.pad(jnp.asarray(centers, dtype), ((0, 0), (0, 1)))
    rad = jnp.asarray(radii, dtype)
    active = (
        jnp.ones((nodes,), idx) if node_active is None else jnp.asarray(node_active).astype(idx)
    )
    theta_sq = jnp.asarray([float(theta) ** 2], dtype)
    Q = int(max_pair_queue)
    blk = int(block)
    grid = (Q + blk - 1) // blk
    backend_kwargs = pallas_backend_kwargs(backend, interpret)
    if "compiler_params" in backend_kwargs:
        backend_kwargs["compiler_params"] = type(backend_kwargs["compiler_params"])(num_warps=int(num_warps))
    kernel = functools.partial(_round_kernel, block=blk, far_cap=int(far_cap), near_cap=int(near_cap), queue_cap=Q)

    def one_round(qa, qb, size, far_a, far_b, near_a, near_b, next_a, next_b, counters):
        operands = [qa, qb, size, left, right, cent, rad, active, theta_sq,
                    far_a, far_b, near_a, near_b, next_a, next_b, counters]
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

    qa0 = jnp.full((Q,), -1, idx).at[0].set(jnp.asarray(root, idx))
    qb0 = qa0
    init = (
        qa0, qb0, jnp.asarray(1, idx),
        jnp.full((int(far_cap),), -1, idx), jnp.full((int(far_cap),), -1, idx),
        jnp.full((int(near_cap),), -1, idx), jnp.full((int(near_cap),), -1, idx),
        jnp.zeros((_NUM_COUNTERS,), idx),  # far, near, next, overflow far/near/queue
        jnp.asarray(1, idx),   # peak
        jnp.asarray(0, idx),   # rounds
    )

    def cond(state):
        _qa, _qb, size, *_rest, counters, _peak, rounds = state
        no_overflow = (counters[_C_OVF_FAR] + counters[_C_OVF_NEAR] + counters[_C_OVF_Q]) == 0
        return (size > 0) & no_overflow & (rounds < int(max_rounds))

    def one_step(state):
        qa, qb, size, far_a, far_b, near_a, near_b, counters, peak, rounds = state
        # the kernel reads only lanes < size, so the next queue needs no fill
        next_a = jnp.empty((Q,), idx)
        next_b = jnp.empty((Q,), idx)
        counters = counters.at[_C_NEXT].set(0)
        far_a, far_b, near_a, near_b, next_a, next_b, counters = one_round(
            qa, qb, size[None], far_a, far_b, near_a, near_b, next_a, next_b, counters
        )
        new_size = counters[_C_NEXT]
        peak = jnp.maximum(peak, new_size)
        # an empty round leaves everything untouched, so the count stays honest
        rounds = rounds + jnp.where(size > 0, 1, 0).astype(idx)
        return (next_a, next_b, jnp.minimum(new_size, Q), far_a, far_b, near_a, near_b, counters, peak, rounds)

    def body(state):
        for _ in range(max(int(rounds_per_check), 1)):
            state = one_step(state)
        return state

    qa, qb, size, far_a, far_b, near_a, near_b, counters, peak, rounds = lax.while_loop(cond, body, init)
    return PallasWalkResult(
        far_a=far_a, far_b=far_b, far_count=jnp.minimum(counters[_C_FAR], int(far_cap)),
        near_a=near_a, near_b=near_b, near_count=jnp.minimum(counters[_C_NEAR], int(near_cap)),
        far_overflow=counters[_C_OVF_FAR] > 0, near_overflow=counters[_C_OVF_NEAR] > 0,
        queue_overflow=counters[_C_OVF_Q] > 0,
        peak_wavefront=peak, rounds=rounds,
    )

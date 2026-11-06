"""Reverse of the direct CSR near field: gradients on the production layout.

The fused lane's near field is
:func:`~jaccpot.pallas.nearfield_leafpair_csr.nearfield_leafpair_csr_sorted_direct_pallas`
(the "direct" layout): one program per leaf reads its CSR row's source leaves as
ranges of the Morton-sorted particles and stores straight into particle order. It
had no reverse rule, so the differentiable lane ran the TABLE layout instead
(:func:`~jaccpot.pallas.nearfield_leafpair_csr.nearfield_leafpair_csr_pallas_cvjp`):
a different forward from production (per-chunk partials, another summation order),
gathered ``(L, W)`` leaf tables that pad cell leaves of ~10-20 particles into 64
slots, and ``(chunks, W, 8)`` reverse partials (plan S8, D6 part 1).

:func:`nearfield_leafpair_csr_sorted_direct_pallas_cvjp` is the production call under
a ``custom_vjp``: its forward IS that call with the same operands (the same lowered
program; the force-scale lane of ``dehnen_error`` included), and its reverse,
:func:`nearfield_leafpair_csr_sorted_direct_reverse_pallas`, runs on the same layout.

THE REVERSE. The per-pair algebra is the table reverse's (``nearfield/grad.py``'s
tidal-tensor contraction with every softening kernel's ``g``, ``(1/r) dg/dr`` and
``dg/d(eps^2)``). The near list is SYMMETRIC -- one canonical walk emits ``(A, B)`` and
``(B, A)`` -- so sink ``p``'s program, walking ITS row, sees every pair ``p`` is part of
from both ends, and computes ``p``'s source-side cotangent (``p`` as a source of ``q``,
which needs ``q``'s output cotangent) as well as its sink-side one. Every cotangent of a
particle is therefore written by the one program that owns it: no atomics, no
by-source scatter, deterministic sums, and no second pass over a transposed list (the
M2L lanes need one because their far list is one-sided). With ``d = x_p - x_q``,
``c`` the acceleration cotangents and ``g``, ``g' = (1/r) dg/dr``, ``g_e = dg/d(eps^2)``
the kernel's factors at ``|d|^2``, per pair::

    x_p_bar += G (m_p g c_q - m_q g c_p - g' (m_q (c_p . d) - m_p (c_q . d)) d)
    m_p_bar += G g (c_q . d)
    eps2_bar += -G m_q (c_p . d) g_e

The program accumulates the sums ``sum m_q g``, ``sum g c_q``, ``sum g' (...) d``,
``sum g (c_q . d)`` and ``sum m_q (c_p . d) g_e`` over its sources (nine ``(Bt, Bs)``
accumulators, summed over the sources once at the end, as the forward's ``a`` flag
does) and applies ``G``, ``m_p`` and ``c_p`` once per target. ``G_bar`` needs no
lane of its own: the acceleration is linear in the masses, so ``G G_bar = sum_p m_p
m_p_bar`` (Euler), i.e. ``G_bar = sum_p m_p sum_q g (c_q . d)`` -- exact for ``G = 0``
too, and the same pair terms the explicit ``-sum m_q g (c_p . d)`` adds up, relabelled
by the symmetry.

MEMORY. The reverse reads the forward's ``(N + Wpad, 4)`` particle table (rebuilt,
not saved) and a ``(N + Wpad, 4)`` table of the cotangents, and writes five ``(N,)``
lanes (x, y, z, the mass sum, the softening sum): no ``(L, W)`` tables, no per-chunk
partials, nothing ``N x 16`` (the S8 gate). ``jax.grad`` of ``sum(acc . w)`` through
this rule alone on a captured 8e6 disc step (A100, float32,
``bench/near_direct_reverse_gate.py --memory``): 45.4 B/p of temporaries (0.34 GiB),
the largest array the ``(N + Wpad, 4)`` table; the table layout's rule on the same step
needs 1380 B/p (10.28 GiB).

TIME. On the same step the gradient takes 57.6 ms against the production forward's
26.8-30.0 ms (the table layout's: 274.5 ms); on the 2e5 disc 3.18 ms against 1.81; on
the galaxy preset's 25M disc+bulge step (``dehnen_error``, the forward with its
force-scale lane) 202.9 ms against 106.1, with the same 45.4 B/p of temporaries.
The reverse does about twice the forward's arithmetic per pair (the tidal contraction,
the softening derivative) and reads seven floats per source to its four.

LONG ROWS. As in the forward, a row runs at most ``row_limit`` entries per program
(one 101,870-entry outlier row serialised a whole launch, 2026-10-05). The rest of a
longer row runs in further launches of the same kernel, one per piece index, each
adding its piece into the particle-order lanes it reads (every particle is one lane of
one program per launch, so the read-add-store is race free). A ``fori_loop`` with a
traced trip count runs them; it runs zero times unless some row is that long.

WHAT IS NOT DIFFERENTIATED. The force-scale lane (``with_force_scale``) is returned by
the forward and treated as a constant: it only sets the next refresh's eq (16a)
thresholds, i.e. discrete MAC decisions, so its cotangent is dropped (the far lanes'
``_fb`` twins drop theirs the same way). The potential lane has no rule: the fast-lane
potential serves the energy diagnostic, which nothing differentiates, so
``with_potential`` is refused rather than given a silently zero gradient. ``accum="wide"``
does not exist on the direct layout (the caller routes it to the table kernel), so
it has no reverse here either. The tile skip (``tile_skip``, opt-in, off by default) is
forwarded to the forward only: the reverse evaluates the ``"ferrers3"`` polynomial on
every pair, which beyond the support gives the Newtonian factors anyway.

The module sits beside, not inside, ``nearfield_leafpair_csr.py`` so that file stays
byte for byte unchanged: the GPU program texts embed the kernels' source lines, and the
production forward must not move by one character.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot._env import env_int, env_text
from jaccpot._searchsorted import searchsorted_method
from jaccpot.pallas._compat import KernelRef
from jaccpot.pallas._tangent import make_tangent_op
from jaccpot.pallas.nearfield_fused_leaf import _resolve_accum_dtype, _resolve_subtile
from jaccpot.pallas.nearfield_leafpair_csr import (
    DIRECT_ROW_LIMIT,
    _rest_pieces_in_fixed_order,
    leafpair_extra_chunk_capacity,
    nearfield_leafpair_csr_sorted_direct_pallas,
)
from jaccpot.softening import (
    masked_pair_factors,
    pair_softening_sq_derivative,
    resolve_softening_kernel,
    softening_params,
    softening_params_from_sq,
)

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

__all__ = [
    "REVERSE_SOURCE_TILE",
    "REVERSE_TARGET_SUBTILE",
    "TANGENT_SOURCE_TILE",
    "TANGENT_TARGET_SUBTILE",
    "nearfield_leafpair_csr_sorted_direct_pallas_cvjp",
    "nearfield_leafpair_csr_sorted_direct_pallas_tangent",
    "nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp",
    "nearfield_leafpair_csr_sorted_direct_reverse_pallas",
]

#: The reverse kernel's default tile: ``Bt = 8`` targets against ``Bs = 8`` sources, one
#: warp. Half the forward's target tile (16, ``DIRECT_TILED_TARGET_SUBTILE``): the
#: reverse carries nine ``(Bt, Bs)`` accumulators to the forward's four. A100,
#: 2026-10-22, reverse alone against the production forward on captured steps
#: (``bench/near_direct_reverse_gate.py``, free card): clipped Plummer 8e6 (cells of
#: 10.5 particles, ferrers3 at softening 1e-7) 54.7 ms against 62.7 at (16, 8) and
#: 61.3 at (16, 8) with 2 warps, forward 25.5; the 8e6 disc (16.4 per cell, ferrers3
#: at 1.5e-3) 57.6 ms against 66.0 (16, 8), 60.8 (4, 8), 68.2 (8, 16), 62.4 (8, 8)
#: with 2 warps, forward 30.0; at 2e5 likewise the fastest or within 2 %.
REVERSE_TARGET_SUBTILE = 8
REVERSE_SOURCE_TILE = 8

#: Particle-order lanes the reverse writes: the position cotangent's x, y, z, then
#: ``sum_q g (c_q . d)`` (the mass cotangent over ``G``) and ``sum_q m_q (c_p . d)
#: dg/d(eps^2)`` (the softening cotangent over ``-G``), per particle.
_REV_LANES = 5


def _nearfield_direct_rev_kernel(
    pm_ref: KernelRef,
    cot_ref: KernelRef,
    leaf_start_ref: KernelRef,
    leaf_count_ref: KernelRef,
    neighbors_ref: KernelRef,
    row_start_ref: KernelRef,
    row_count_ref: KernelRef,
    row_first_ref: KernelRef,
    softening_ref: KernelRef,
    g_ref: KernelRef,
    *lane_refs: KernelRef,
    subtile: int,
    source_tile: int,
    softening_kernel: str,
) -> None:
    """One target subtile of one leaf: its row's piece (and self term) in reverse.

    The loop skeleton is the forward's with source tiles and the ``r`` flag
    (:func:`~jaccpot.pallas.nearfield_leafpair_csr._nearfield_leafpair_csr_sorted_kernel`):
    the row's entries whose particle ranges touch are merged into one range, and a
    range is swept in ``(Bt, Bs)`` pair tiles. The pair body is the reverse's.

    Parameters
    ----------
    pm_ref : KernelRef
        ``(N + pad, 4)`` sorted x, y, z, mass; rows past ``N`` are zero.
    cot_ref : KernelRef
        ``(N + pad, 4)`` acceleration cotangent in sorted order (lanes 0:3; lane 3
        and rows past ``N`` zero).
    leaf_start_ref : KernelRef
        ``(L,)`` first particle of each leaf.
    leaf_count_ref : KernelRef
        ``(L,)`` particles per leaf.
    neighbors_ref : KernelRef
        ``(E,)`` flat neighbour array of LEAF indices.
    row_start_ref : KernelRef
        ``(L,)`` first entry of this launch's piece of each row.
    row_count_ref : KernelRef
        ``(L,)`` entries in the piece (``0``: none).
    row_first_ref : KernelRef
        ``(L,)`` ``1`` where the self term rides (the first launch).
    softening_ref : KernelRef
        ``(2,)`` softening parameters
        (:func:`jaccpot.softening.softening_params_from_sq`).
    g_ref : KernelRef
        ``(1,)`` gravitational constant.
    *lane_refs : KernelRef
        :data:`_REV_LANES` aliased ``(N,)`` inputs (the lanes so far), then the
        same number of outputs: each live target lane is read from the first and
        stored, plus this piece's sums, into the second.
    subtile : int
        ``Bt``. Static.
    source_tile : int
        ``Bs``, a power of two. Static.
    softening_kernel : str
        The pair kernel, static (:mod:`jaccpot.softening`).

    Returns
    -------
    None
        The result is the write to the output lanes.
    """
    lanes_in, lanes_out = lane_refs[:_REV_LANES], lane_refs[_REV_LANES:]
    tl = pl.program_id(0)
    sub = pl.program_id(1)
    bt = int(subtile)
    bs = int(source_tile)
    tcount = leaf_count_ref[tl]
    tstart = leaf_start_ref[tl]
    off = sub * bt

    @pl.when(off < tcount)
    def _work():
        rows = pl.ds(tstart + off, bt)
        tx = pm_ref[rows, 0]
        ty = pm_ref[rows, 1]
        tz = pm_ref[rows, 2]
        tm = pm_ref[rows, 3]
        cx = cot_ref[rows, 0]
        cy = cot_ref[rows, 1]
        cz = cot_ref[rows, 2]
        lane = lax.broadcasted_iota(jnp.int32, (bt,), 0)
        lane_idx = off + lane
        tvalid = lane_idx < tcount
        soft = softening_ref[0]
        soft_params = (soft, softening_ref[1])
        s_lane = lax.broadcasted_iota(jnp.int32, (bs,), 0)
        acc0 = tuple(jnp.zeros((bt, bs), tx.dtype) for _ in range(9))

        def _range_pass(sstart, scount, acc, exclude_lane=None):
            # Bs sources per iteration, read as vectors. A tile may run past the
            # range (into the next leaf's particles or the tables' zero padding,
            # never past it: they hold Wpad >= Bs extra rows); those sources are
            # masked like the target lanes past the count.
            def _block(b, acc):
                j0 = b * bs
                srows = pl.ds(sstart + j0, bs)
                sx = pm_ref[srows, 0][None, :]
                sy = pm_ref[srows, 1][None, :]
                sz = pm_ref[srows, 2][None, :]
                sm = pm_ref[srows, 3][None, :]
                qx = cot_ref[srows, 0][None, :]
                qy = cot_ref[srows, 1][None, :]
                qz = cot_ref[srows, 2][None, :]
                j = (j0 + s_lane)[None, :]
                dx = tx[:, None] - sx
                dy = ty[:, None] - sy
                dz = tz[:, None] - sz
                r_sq = dx * dx + dy * dy + dz * dz
                # a pair at zero softened distance (coincident particles, no
                # softening) adds nothing, as in the forward's lean pair body; the
                # self pass drops the diagonal explicitly: its own and source-side
                # terms cancel only in exact arithmetic
                active = tvalid[:, None] & (j < scount) & ((r_sq > 0.0) | (soft > 0.0))
                if exclude_lane is not None:
                    active = active & (exclude_lane[:, None] != j)
                g, _, dgr = masked_pair_factors(
                    r_sq, active, soft_params, softening_kernel, derivative=True
                )
                if softening_kernel == "plummer":
                    # d/d(eps^2) (r^2 + eps^2)^(-3/2) = (1/2) (1/r) dg/dr
                    dge = 0.5 * dgr
                else:
                    dge = jnp.where(
                        active,
                        pair_softening_sq_derivative(
                            jnp.where(active, r_sq, 1.0), soft_params, softening_kernel
                        ),
                        0.0,
                    )
                cd = cx[:, None] * dx + cy[:, None] * dy + cz[:, None] * dz
                qd = qx * dx + qy * dy + qz * dz
                mcd = sm * cd
                tidal = dgr * (mcd - tm[:, None] * qd)
                terms = (
                    sm * g,  # m_q g: the sink's own term, times -G c_p
                    g * qx,  # g c_q: p as a source of q, times G m_p
                    g * qy,
                    g * qz,
                    tidal * dx,  # the tidal tensor's d d^T part, times -G
                    tidal * dy,
                    tidal * dz,
                    g * qd,  # the mass cotangent over G
                    mcd * dge,  # the softening cotangent over -G
                )
                return tuple(a + t for a, t in zip(acc, terms))

            n_blocks = (scount + (bs - 1)) // bs
            return lax.fori_loop(0, n_blocks, _block, acc)

        # Runs: a row lists leaves in Morton order, and neighbouring leaves are often
        # adjacent particle ranges. Consecutive entries whose ranges touch are one
        # range; the next entry's range is loaded one iteration ahead (the forward's
        # ``r`` flag, op for op in its control flow).
        start = row_start_ref[tl]
        cnt = row_count_ref[tl]
        last_entry = neighbors_ref.shape[0] - 1

        def _range_of(s):
            sid = neighbors_ref[jnp.clip(start + s, 0, last_entry)]
            return leaf_start_ref[sid], leaf_count_ref[sid]

        def _slot_body(s, carry):
            acc, rs, rl, cur = carry
            ss, sc = cur
            nxt = _range_of(jnp.minimum(s + 1, cnt - 1))
            joins = ss == rs + rl
            acc = lax.cond(joins, lambda a: a, lambda a: _range_pass(rs, rl, a), acc)
            rs = jnp.where(joins, rs, ss)
            rl = jnp.where(joins, rl + sc, sc)
            return acc, rs, rl, nxt

        zero_i = jnp.zeros((), tcount.dtype)
        acc, rs, rl, _ = lax.fori_loop(
            0, cnt, _slot_body, (acc0, zero_i, zero_i, _range_of(0))
        )
        acc = _range_pass(rs, rl, acc)

        def _self_pass(acc):
            return _range_pass(tstart, tcount, acc, exclude_lane=lane_idx)

        acc = lax.cond(row_first_ref[tl] != 0, _self_pass, lambda a: a, acc)

        s_mg, s_gx, s_gy, s_gz, s_tx, s_ty, s_tz, s_qd, s_se = (
            jnp.sum(a, axis=1) for a in acc
        )
        g_value = g_ref[0]
        values = (
            g_value * (tm * s_gx - cx * s_mg - s_tx),
            g_value * (tm * s_gy - cy * s_mg - s_ty),
            g_value * (tm * s_gz - cz * s_mg - s_tz),
            s_qd,
            s_se,
        )
        n_rows = lanes_out[0].shape[0]
        # masked lanes point PAST the lanes (unique, out of range), as the forward's
        # stores do: Triton never touches them and the interpreter drops them
        where = jnp.where(tvalid, tstart + lane_idx, n_rows + lane)
        for k in range(_REV_LANES):
            prev = plgpu.load(lanes_in[k].at[where], mask=tvalid, other=0.0)
            plgpu.store(lanes_out[k].at[where], prev + values[k], mask=tvalid)


def _direct_rev_call(
    pm: Array,
    cot: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    row_start: Array,
    row_count: Array,
    row_first: Array,
    softening: Array,
    g: Array,
    lanes: list,
    *,
    bt: int,
    bs: int,
    n_sub: int,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    softening_kernel: str,
    name: str,
) -> list:
    """One launch of :func:`_nearfield_direct_rev_kernel` over every leaf.

    Parameters
    ----------
    pm : Array
        ``(N + pad, 4)`` sorted x, y, z, mass.
    cot : Array
        ``(N + pad, 4)`` sorted acceleration cotangent.
    leaf_start : Array
        ``(L,)`` int32 first particle per leaf.
    leaf_count : Array
        ``(L,)`` int32 particles per leaf.
    neighbors : Array
        ``(E,)`` neighbour leaf indices.
    row_start : Array
        ``(L,)`` first entry of each row's piece.
    row_count : Array
        ``(L,)`` entries of each row's piece.
    row_first : Array
        ``(L,)`` ``1`` where the self term rides.
    softening : Array
        ``(2,)`` softening parameters.
    g : Array
        ``(1,)`` gravitational constant.
    lanes : list
        The :data:`_REV_LANES` ``(N,)`` lanes so far, aliased to the outputs.
    bt : int
        Target subtile.
    bs : int
        Source tile.
    n_sub : int
        Subtiles per leaf.
    num_warps : int
        Triton warps per program.
    num_stages : int
        Triton pipeline stages.
    interpret : bool
        Pallas interpret mode.
    softening_kernel : str
        The pair kernel, static.
    name : str
        Kernel name.

    Returns
    -------
    list
        The lanes plus this launch's sums.
    """
    num_leaves = int(leaf_start.shape[0])

    def _kernel(*refs):
        return _nearfield_direct_rev_kernel(
            *refs, subtile=bt, source_tile=bs, softening_kernel=softening_kernel
        )

    def _full(arr: Array) -> Any:
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    operands = [
        pm,
        cot,
        leaf_start,
        leaf_count,
        neighbors,
        row_start,
        row_count,
        row_first,
        softening,
        g,
    ]
    in_specs = [_full(o) for o in operands[:8]] + [
        pl.BlockSpec((2,), lambda t, sub: (0,)),
        pl.BlockSpec((1,), lambda t, sub: (0,)),
    ]
    in_specs += [_full(lane) for lane in lanes]
    out = pl.pallas_call(
        _kernel,
        out_shape=[jax.ShapeDtypeStruct(lane.shape, lane.dtype) for lane in lanes],
        in_specs=in_specs,
        out_specs=[_full(lane) for lane in lanes],
        grid=(num_leaves, n_sub),
        input_output_aliases={len(operands) + k: k for k in range(len(lanes))},
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=name + ("" if softening_kernel == "plummer" else "_" + softening_kernel),
    )(*operands, *lanes)
    return list(out)


def nearfield_leafpair_csr_sorted_direct_reverse_pallas(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    cotangent: Array,
    *,
    leaf_width: int,
    softening_sq: Array,
    G: Array,
    chunk: int,
    row_limit: int | None = None,
    softening_kernel: Optional[str] = None,
    interpret: bool = False,
    target_subtile: int | None = None,
    source_tile: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 1,
) -> tuple[Array, Array, Array, Array]:
    """Adjoint of the direct CSR near field's accelerations, on the sorted layout.

    The reverse of :func:`~jaccpot.pallas.nearfield_leafpair_csr.
    nearfield_leafpair_csr_sorted_direct_pallas`'s ``(N, 3)`` acceleration output:
    one program per (leaf, target subtile) walks the leaf's row and self term, as the
    forward does, and writes the cotangents of its own particles in particle order
    (see the module docstring for the algebra, the symmetry it relies on and why
    nothing is scattered). Every softening kernel, with its own ``(1/r) dg/dr`` and
    ``dg/d(eps^2)``.

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)`` particle positions in tree (Morton) order.
    masses_sorted : Array
        ``(N,)`` masses in the same order.
    leaf_start : Array
        ``(L,)`` first particle of each leaf.
    leaf_count : Array
        ``(L,)`` particles per leaf (at most ``leaf_width``).
    neighbors : Array
        Flat neighbour array of LEAF indices. Must be SYMMETRIC (every leaf lists
        every leaf that lists it), as the walk's near list is: the source-side
        cotangents are computed from the other end of each pair. A one-sided list
        gives a wrong gradient that nothing here can detect.
    offsets : Array
        CSR row starts into ``neighbors`` (``L`` or ``L + 1`` entries).
    counts : Array
        ``(L,)`` row lengths.
    cotangent : Array
        ``(N, 3)`` cotangent of the accelerations, sorted order.
    leaf_width : int
        ``W``, the most particles a leaf can hold. Static.
    softening_sq : Array
        Scalar squared (Plummer-equivalent) softening, as the forward's.
    G : Array
        Scalar gravitational constant.
    chunk : int
        The forward's ``chunk``: a row's pieces hold ``max(row_limit, chunk)``
        entries, as in the forward. Static.
    row_limit : int | None
        Entries of a row one program runs. ``None``:
        ``JACCPOT_NEARFIELD_DIRECT_ROW_LIMIT``, default
        :data:`~jaccpot.pallas.nearfield_leafpair_csr.DIRECT_ROW_LIMIT` (the
        forward's reading). Static.
    softening_kernel : Optional[str]
        The pair kernel (:mod:`jaccpot.softening`); ``None`` gives the default.
    interpret : bool
        Pallas interpret mode.
    target_subtile : int | None
        Targets per program; ``None``: :data:`REVERSE_TARGET_SUBTILE`, clamped to a
        power of two at most ``W``. Static.
    source_tile : int | None
        Sources per pair tile, a power of two; ``None``: :data:`REVERSE_SOURCE_TILE`.
        Static.
    num_warps : int | None
        Triton warps per program; ``None``: one per 32 target lanes, at least 1.
    num_stages : int
        Triton pipeline stages.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        ``(positions_bar (N, 3), masses_bar (N,), softening_sq_bar (), G_bar ())``
        in the input dtype; zero on rows no leaf holds.

    Raises
    ------
    RuntimeError
        If Pallas or its Triton backend could not be imported.
    ValueError
        If ``source_tile`` is not a power of two, or ``cotangent`` is not ``(N, 3)``.
    """
    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    positions_sorted = jnp.asarray(positions_sorted)
    dtype = positions_sorted.dtype
    masses_sorted = jnp.asarray(masses_sorted, dtype)
    cot = jnp.asarray(cotangent, dtype)
    n = int(positions_sorted.shape[0])
    num_leaves = int(leaf_start.shape[0])
    leaf_width = int(leaf_width)
    if tuple(cot.shape) != (n, 3):
        raise ValueError(f"cotangent must be ({n}, 3), got {tuple(cot.shape)}")
    if num_leaves == 0 or leaf_width == 0 or n == 0:
        zero = jnp.zeros((), dtype)
        return jnp.zeros((n, 3), dtype), jnp.zeros((n,), dtype), zero, zero
    bs = REVERSE_SOURCE_TILE if source_tile is None else int(source_tile)
    if bs < 1 or bs & (bs - 1):
        raise ValueError(f"source_tile must be a power of two, got {bs}")
    bt = _resolve_subtile(
        REVERSE_TARGET_SUBTILE if target_subtile is None else target_subtile,
        leaf_width,
    )
    width_pad = ((leaf_width + bt - 1) // bt) * bt
    if num_warps is None:
        num_warps = max(1, bt // 32)
    kernel_name = resolve_softening_kernel(softening_kernel)

    # The forward's particle table (_sorted_inputs) and its twin for the cotangent:
    # a target tile of the last leaf reads up to Wpad rows past N, a source tile up
    # to Bs; both stay inside the zero padding.
    pad = max(width_pad, bs)
    pm = jnp.concatenate([positions_sorted, masses_sorted[:, None]], axis=1)
    pm = jnp.pad(pm, ((0, pad), (0, 0)))
    cot_table = jnp.pad(cot, ((0, pad), (0, 1)))
    idx = jnp.asarray(counts).dtype
    counts_i = jnp.asarray(counts, idx)
    row_start = jnp.asarray(offsets, idx)[:num_leaves]
    neighbors = jnp.asarray(neighbors, idx)
    if int(neighbors.shape[0]) == 0:
        # the kernel prefetches entry 0 of every row, read or not
        neighbors = jnp.zeros((1,), idx)
    start_i = jnp.minimum(jnp.asarray(leaf_start, jnp.int32), n)
    count_i = jnp.asarray(leaf_count, jnp.int32)
    softening = softening_params_from_sq(kernel_name, softening_sq, dtype)
    g_arr = jnp.asarray([G], dtype=dtype)

    if row_limit is None:
        row_limit = env_int(
            "JACCPOT_NEARFIELD_DIRECT_ROW_LIMIT", DIRECT_ROW_LIMIT, minimum=1
        )
    piece = max(int(row_limit), max(1, int(chunk)))
    piece_i = jnp.asarray(piece, idx)
    launch = functools.partial(
        _direct_rev_call,
        pm,
        cot_table,
        start_i,
        count_i,
        neighbors,
        bt=bt,
        bs=bs,
        n_sub=width_pad // bt,
        num_warps=int(num_warps),
        num_stages=int(num_stages),
        interpret=interpret,
        softening_kernel=kernel_name,
        name=f"near_rev_direct_t{bt}_s{bs}_w{leaf_width}_c{piece}",
    )

    # 1. each row up to `piece` entries + the self term
    lanes = launch(
        row_start,
        jnp.minimum(counts_i, piece_i),
        jnp.ones((num_leaves,), idx),
        softening,
        g_arr,
        [jnp.zeros((n,), dtype) for _ in range(_REV_LANES)],
    )

    # 2. piece k >= 1 of every row, added in place; no launch unless a row is longer
    def _piece(k, lanes_in):
        lo = k * piece_i
        return launch(
            row_start + lo,
            jnp.clip(counts_i - lo, 0, piece_i),
            jnp.zeros((num_leaves,), idx),
            softening,
            g_arr,
            lanes_in,
        )

    n_pieces = (jnp.max(counts_i) + piece_i - 1) // piece_i
    lanes = lax.fori_loop(jnp.asarray(1, idx), n_pieces, _piece, lanes)

    g_scalar = jnp.asarray(G, dtype)
    pos_bar = jnp.stack(lanes[:3], axis=1)
    mass_bar = g_scalar * lanes[3]
    # Euler on the mass-linear acceleration: G G_bar = sum_p m_p m_p_bar
    g_bar = jnp.sum(masses_sorted * lanes[3])
    soft_bar = -g_scalar * jnp.sum(lanes[4])
    return pos_bar, mass_bar, soft_bar, g_bar


class _DirectStatic(NamedTuple):
    """The forward's static options, one hashable ``nondiff_argnums`` operand.

    Each field is the keyword of the same name of
    :func:`~jaccpot.pallas.nearfield_leafpair_csr.nearfield_leafpair_csr_sorted_direct_pallas`,
    passed through unchanged (``None`` keeps the forward's own resolution, from the
    environment or its defaults, so the forward resolves exactly as production does).

    Attributes
    ----------
    leaf_width : int
        ``W``.
    chunk : int
        Entries per chunk.
    num_warps : Optional[int]
        Triton warps per program.
    num_stages : int
        Triton pipeline stages.
    target_subtile : Optional[int]
        Targets per program.
    interpret : bool
        Pallas interpret mode (the reverse's too).
    with_force_scale : bool
        Return the force-scale lane.
    row_limit : Optional[int]
        Entries of a row its own program runs (the reverse's too).
    source_tile : Optional[int]
        Sources per vector tile.
    source_flags : Optional[str]
        The source tiles' options.
    softening_kernel : Optional[str]
        The pair kernel (the reverse's too).
    tile_skip : Optional[bool]
        The ferrers3 tile skip.
    """

    leaf_width: int
    chunk: int
    num_warps: Optional[int]
    num_stages: int
    target_subtile: Optional[int]
    interpret: bool
    with_force_scale: bool
    row_limit: Optional[int]
    source_tile: Optional[int]
    source_flags: Optional[str]
    softening_kernel: Optional[str]
    tile_skip: Optional[bool]


def _direct_forward(
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
):
    acc, fourth = nearfield_leafpair_csr_sorted_direct_pallas(
        positions_sorted,
        masses_sorted,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        leaf_width=static.leaf_width,
        softening_sq=softening_sq,
        G=G,
        chunk=static.chunk,
        num_warps=static.num_warps,
        num_stages=static.num_stages,
        target_subtile=static.target_subtile,
        interpret=static.interpret,
        with_force_scale=static.with_force_scale,
        row_limit=static.row_limit,
        source_tile=static.source_tile,
        source_flags=static.source_flags,
        softening_kernel=static.softening_kernel,
        tile_skip=static.tile_skip,
    )
    return (acc, fourth) if static.with_force_scale else acc


@functools.partial(jax.custom_vjp, nondiff_argnums=(9,))
def _direct_cvjp(
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
):
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


def _direct_cvjp_fwd(
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
):
    out = _direct_forward(
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
    # the operands only: the particle table is rebuilt in the reverse rather than
    # kept alive from the forward
    return out, (
        positions_sorted,
        masses_sorted,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        jnp.asarray(softening_sq),
        jnp.asarray(G),
    )


def _direct_cvjp_bwd(static, residual, cotangent):
    (
        positions_sorted,
        masses_sorted,
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        softening_sq,
        G,
    ) = residual
    # the force-scale lane is bookkeeping (it only sets the next walk's thresholds):
    # its cotangent is dropped
    acc_bar = cotangent[0] if static.with_force_scale else cotangent
    # a literal softening or G (the solver's own, folded at trace time) comes back
    # from the residuals as a numpy scalar under jit
    softening_sq, G = jnp.asarray(softening_sq), jnp.asarray(G)
    pos_bar, mass_bar, soft_bar, g_bar = (
        nearfield_leafpair_csr_sorted_direct_reverse_pallas(
            positions_sorted,
            masses_sorted,
            leaf_start,
            leaf_count,
            neighbors,
            offsets,
            counts,
            acc_bar,
            leaf_width=static.leaf_width,
            softening_sq=softening_sq,
            G=G,
            chunk=static.chunk,
            row_limit=static.row_limit,
            softening_kernel=static.softening_kernel,
            interpret=static.interpret,
        )
    )
    return (
        pos_bar.astype(positions_sorted.dtype),
        mass_bar.astype(masses_sorted.dtype),
        None,
        None,
        None,
        None,
        None,
        jnp.reshape(soft_bar.astype(softening_sq.dtype), jnp.shape(softening_sq)),
        jnp.reshape(g_bar.astype(G.dtype), jnp.shape(G)),
    )


_direct_cvjp.defvjp(_direct_cvjp_fwd, _direct_cvjp_bwd)


def nearfield_leafpair_csr_sorted_direct_pallas_cvjp(
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
    """Differentiable direct CSR near field: the production forward, its own reverse.

    The forward is :func:`~jaccpot.pallas.nearfield_leafpair_csr.
    nearfield_leafpair_csr_sorted_direct_pallas` called with exactly these arguments,
    so it computes the production bits and, for the same (traced) operands, lowers to
    the same program. Under ``jax.grad`` / ``jax.vjp`` the reverse is
    :func:`nearfield_leafpair_csr_sorted_direct_reverse_pallas` on the same layout.

    WHAT IS DIFFERENTIATED: the ``(N, 3)`` accelerations, with respect to
    ``positions_sorted`` (each particle as a sink and as a source), ``masses_sorted``
    (as sources), ``softening_sq`` and ``G``. The leaf ranges and the CSR are topology
    (integer, no cotangent). The force-scale lane (``with_force_scale``, the near half
    of ``dehnen_error``'s ``f_b``) is computed by the forward as in production but is a
    constant to the reverse: its cotangent is dropped, since it only sets the next
    refresh's thresholds (see the module docstring).

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)`` particle positions in tree (Morton) order. Differentiable.
    masses_sorted : Array
        ``(N,)`` masses in the same order. Differentiable.
    leaf_start : Array
        ``(L,)`` first particle of each leaf (topology).
    leaf_count : Array
        ``(L,)`` particles per leaf (topology).
    neighbors : Array
        Flat SYMMETRIC neighbour array of leaf indices (topology): the reverse takes
        each source-side cotangent from the other end of the pair.
    offsets : Array
        CSR row starts (topology).
    counts : Array
        ``(L,)`` row lengths (topology).
    leaf_width : int
        As the forward. Static.
    softening_sq : Array
        Scalar squared softening. Differentiable; pass an array (a Python float is
        accepted, but then the program is not the production call's text: there the
        literal folds at trace time).
    G : Array
        Scalar gravitational constant. Differentiable; the same remark.
    chunk : int
        As the forward. Static.
    num_warps : int | None
        The forward's. Static.
    num_stages : int
        The forward's. Static.
    target_subtile : int | None
        The forward's. Static.
    interpret : bool
        Pallas interpret mode, forward and reverse. Static.
    with_potential : bool
        Refused: the potential lane has no reverse rule (the fast-lane potential is
        an energy diagnostic, which nothing differentiates). Static.
    with_force_scale : bool
        Also return the force-scale lane, as the forward (not differentiated).
        Static.
    row_limit : int | None
        As the forward, for the forward and the reverse. Static.
    source_tile : int | None
        The forward's. Static.
    source_flags : str | None
        The forward's. Static.
    softening_kernel : Optional[str]
        The pair kernel, forward and reverse. Static.
    tile_skip : bool | None
        The forward's (the reverse evaluates every pair's polynomial). Static.

    Returns
    -------
    tuple[Array, Array | None]
        ``(N, 3)`` accelerations and the ``(N,)`` force scale with
        ``with_force_scale``, else ``None`` -- the forward's return value.

    Raises
    ------
    ValueError
        If ``with_potential`` is set.
    """
    if with_potential:
        raise ValueError(
            "nearfield_leafpair_csr_sorted_direct_pallas_cvjp differentiates the "
            "accelerations only; the potential lane has no reverse rule (call "
            "nearfield_leafpair_csr_sorted_direct_pallas for the potential)"
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
    out = _direct_cvjp(
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
    if static.with_force_scale:
        return out
    return out, None


# ------------------------------------------------------------------ tangent
# Forward mode (``ad_mode="forward"``, plan forward-mode Phase 1, track C). The
# tangent of a sink's acceleration is a sum over the same sources as the primal, so
# the tangent kernel is target-driven like the forward: one program per (leaf,
# target subtile) over the leaf's row and self term, the forward's merged source
# runs, the forward's row pieces (the first ``piece`` entries of every row stored in
# particle order, the rest of a longer row as per-piece partials summed in one fixed
# order and added once, under the same ``lax.cond``). No symmetric list is needed:
# the tangent has no source-side half.
#
# THE PAIR TANGENT, analytic (not ``jax.jvp`` of the forward's pair body). With ``d =
# x_p - x_q``, ``g``, ``g' = (1/r) dg/dr`` and ``g_e = dg/d(eps^2)`` the kernel's
# factors at ``|d|^2`` (:func:`jaccpot.softening.masked_pair_factors` and
# :func:`jaccpot.softening.pair_softening_sq_derivative`, the reverse's), the
# acceleration ``a_p = -G sum_q m_q g d`` has the tangent
#
#     da_p = -G sum_q [ m_q g dd + m_q g' (d . dd) d + dm_q g d ]
#            - dG sum_q m_q g d - G deps2 sum_q m_q g_e d,       dd = dx_p - dx_q
#
# The relative displacement ``dd`` is formed PER PAIR, so a rigid motion of a pair
# cancels before anything is summed (the tidal-tensor split ``T_p dx_p - sum T_pq
# dx_q`` would add the large self-tidal term and subtract it again: float32 loses the
# difference). The ``dG`` and ``deps2`` sums are shared by the ``K`` tangents. Why not
# ``jax.jvp`` of the forward's lean pair body: that body sets ``1/r = 1`` on pairs at
# zero separation (the diagonal, and coincident particles under softening), so its
# derivative there is ``m dd`` instead of ``m eps^-3 dd``; the analytic form uses the
# kernel's true factors on every live pair (``r > 0`` or ``eps > 0``), as the reverse
# does, and zero on the diagonal and on unsoftened coincident pairs.
#
# ACCUMULATION. As the forward (``accum="input"``): ``(Bt, Bs)`` partials in the input
# dtype over the whole program, summed over the sources once at the end. The direct
# FORWARD has no wide accumulator (``JACCPOT_NEARFIELD_ACCUM=wide`` routes the near
# field to the table layout, whose seam is not this one), so the tangent's
# ``accum="wide"`` is an opt-in of its own (``JACCPOT_NEARFIELD_TANGENT_ACCUM=wide`` or
# the op's ``accum`` parameter): a float32 partial per merged source run, float64
# across runs and pieces, one downcast at the end -- the table kernel's two-level
# scheme (memory ``nearfield-accumulator-fix``).

#: The tangent kernel's target subtile (the reverse's: ``K`` tangents carry ``3 K``
#: accumulators, more than the forward's four).
TANGENT_TARGET_SUBTILE = 8
#: The tangent kernel's source tile.
TANGENT_SOURCE_TILE = 8

_TANGENT_ACCUM_MODES = ("input", "wide")


def _nearfield_direct_tan_kernel(
    pm_ref: KernelRef,
    tpos_ref: KernelRef,
    tmass_ref: KernelRef,
    leaf_start_ref: KernelRef,
    leaf_count_ref: KernelRef,
    neighbors_ref: KernelRef,
    chunk_leaf_ref: KernelRef,
    chunk_start_ref: KernelRef,
    chunk_count_ref: KernelRef,
    chunk_first_ref: KernelRef,
    softening_ref: KernelRef,
    g_ref: KernelRef,
    tsoft_ref: KernelRef,
    tg_ref: KernelRef,
    *out_refs: KernelRef,
    subtile: int,
    source_tile: int,
    softening_kernel: str,
    k_tangents: int,
    tpp: int,
    live: tuple[bool, bool, bool, bool],
    direct: bool,
    wide_dtype: Any,
) -> None:
    """One target subtile of one row piece: its acceleration tangents of one group.

    The loop skeleton is the forward's (and the reverse's) with source tiles and the
    ``r`` flag; the pair body is the analytic tangent (see the section comment).
    Grid axis 2 is the tangent group (:mod:`jaccpot.pallas._tangent_grid`): the
    program carries tangents ``g tpp .. g tpp + tpp - 1``.

    Parameters
    ----------
    pm_ref : KernelRef
        ``(N + pad, 4)`` sorted x, y, z, mass; rows past ``N`` are zero.
    tpos_ref : KernelRef
        ``(K, N, 3)`` position tangents as given (no padded staging copy: the
        kernel reads them at row indices clamped to ``N - 1``; the clamped lanes
        are inactive pairs or masked targets).
    tmass_ref : KernelRef
        ``(K, N)`` mass tangents, read the same way.
    leaf_start_ref : KernelRef
        ``(L,)`` first particle of each leaf.
    leaf_count_ref : KernelRef
        ``(L,)`` particles per leaf.
    neighbors_ref : KernelRef
        ``(E,)`` flat neighbour array of LEAF indices.
    chunk_leaf_ref : KernelRef
        ``(C,)`` target leaf per program row, ``-1`` = padding.
    chunk_start_ref : KernelRef
        ``(C,)`` first entry of the piece in ``neighbors``.
    chunk_count_ref : KernelRef
        ``(C,)`` entries of the piece.
    chunk_first_ref : KernelRef
        ``(C,)`` ``1`` where the self term rides.
    softening_ref : KernelRef
        ``(2,)`` softening parameters.
    g_ref : KernelRef
        ``(1,)`` gravitational constant.
    tsoft_ref : KernelRef
        ``(K,)`` tangents of ``softening_sq``.
    tg_ref : KernelRef
        ``(K,)`` tangents of ``G``.
    *out_refs : KernelRef
        Direct: the aliased ``(K, N, 3)`` input (unused) then the ``(K, N, 3)``
        output, the op's result layout (no transpose afterwards), each live
        target lane stored once. Blocked: one ``(1, Bt, 3 tpp)`` block of the
        per-piece partials (this group's columns).
    subtile : int
        ``Bt``. Static.
    source_tile : int
        ``Bs``, a power of two. Static.
    softening_kernel : str
        The pair kernel. Static.
    k_tangents : int
        ``K``, the tangents of the launch. Static.
    tpp : int
        Tangents per program; those past ``K`` are clamped on load and not stored
        (direct) or land in padding columns (blocked). Static.
    live : tuple[bool, bool, bool, bool]
        Which of the position, mass, softening and ``G`` tangents are not known
        zero. Static.
    direct : bool
        Store into particle order (else the per-piece block). Static.
    wide_dtype : Any
        ``None``: accumulate in the input dtype over the program; else the dtype
        across source runs (a partial in the input dtype per run). Static.

    Returns
    -------
    None
        The result is the write to ``out_refs``.
    """
    assert pl is not None and plgpu is not None
    live_pos, live_mass, live_soft, live_g = live
    # (1/r) dg/dr: the tidal term, and Plummer's d/d(eps^2) (half of it)
    want_dgr = live_pos or (live_soft and softening_kernel == "plummer")
    n_lanes = 3 * tpp
    c = pl.program_id(0)
    sub = pl.program_id(1)
    group = pl.program_id(2)
    tails = int(k_tangents) % int(tpp) != 0
    js = [group * tpp + jj for jj in range(tpp)]
    # a partial last group reads tangent K - 1 in its spare slots
    jl = [jnp.minimum(j, k_tangents - 1) for j in js] if tails else js
    bt = int(subtile)
    bs = int(source_tile)
    tl = chunk_leaf_ref[c]
    is_live = tl >= 0
    tl_safe = jnp.maximum(tl, 0)
    off = sub * bt
    tcount = leaf_count_ref[tl_safe]
    tstart = leaf_start_ref[tl_safe]
    out_block = None if direct else out_refs[0]
    out_dtype = out_refs[-1].dtype
    if out_block is not None:
        zeros_bt = jnp.zeros((bt,), out_dtype)
        for comp in range(n_lanes):
            out_block[0, :, comp] = zeros_bt

    @pl.when(is_live & (off < tcount))
    def _work():
        assert pl is not None and plgpu is not None
        rows = pl.ds(tstart + off, bt)
        tx = pm_ref[rows, 0]
        ty = pm_ref[rows, 1]
        tz = pm_ref[rows, 2]
        dtype = tx.dtype
        lane = lax.broadcasted_iota(jnp.int32, (bt,), 0)
        lane_idx = off + lane
        tvalid = lane_idx < tcount
        soft = softening_ref[0]
        soft_params = (soft, softening_ref[1])
        s_lane = lax.broadcasted_iota(jnp.int32, (bs,), 0)
        # the targets' position tangents, once per program (rows past N clamped:
        # those lanes are masked targets); a dead tangent is never read
        ttx: list = []
        if live_pos:
            trow = jnp.minimum(tstart + lane_idx, tpos_ref.shape[1] - 1)
            ttx = [tuple(tpos_ref[j, trow, a] for a in range(3)) for j in jl]
        n_acc = n_lanes + (3 if live_g else 0) + (3 if live_soft else 0)
        wide = wide_dtype is not None
        if wide:
            acc0 = tuple(jnp.zeros((bt,), wide_dtype) for _ in range(n_acc))
        else:
            acc0 = tuple(jnp.zeros((bt, bs), dtype) for _ in range(n_acc))

        def _range_pass(sstart, scount, acc, exclude_lane=None):
            def _block(b, inner):
                assert pl is not None
                j0 = b * bs
                srows = pl.ds(sstart + j0, bs)
                sx = pm_ref[srows, 0][None, :]
                sy = pm_ref[srows, 1][None, :]
                sz = pm_ref[srows, 2][None, :]
                sm = pm_ref[srows, 3][None, :]
                # the tangents' source rows, clamped (past N: inactive pairs)
                srow = sstart + j0 + s_lane
                j = (j0 + s_lane)[None, :]
                dx = tx[:, None] - sx
                dy = ty[:, None] - sy
                dz = tz[:, None] - sz
                r_sq = dx * dx + dy * dy + dz * dz
                # a pair at zero softened distance (the diagonal, coincident
                # particles without softening) adds nothing, as in the reverse
                active = tvalid[:, None] & (j < scount) & ((r_sq > 0.0) | (soft > 0.0))
                if exclude_lane is not None:
                    active = active & (exclude_lane[:, None] != j)
                g, _, dgr = masked_pair_factors(
                    r_sq, active, soft_params, softening_kernel, derivative=want_dgr
                )
                mg = sm * g
                terms = []
                for jj, jt in enumerate(jl):
                    u = [jnp.zeros_like(dx)] * 3
                    if live_pos:
                        prow = jnp.minimum(srow, tpos_ref.shape[1] - 1)
                        tsx = tpos_ref[jt, prow, 0][None, :]
                        tsy = tpos_ref[jt, prow, 1][None, :]
                        tsz = tpos_ref[jt, prow, 2][None, :]
                        ddx = ttx[jj][0][:, None] - tsx
                        ddy = ttx[jj][1][:, None] - tsy
                        ddz = ttx[jj][2][:, None] - tsz
                        tidal = sm * dgr * (dx * ddx + dy * ddy + dz * ddz)
                        u = [mg * ddx + tidal * dx, mg * ddy + tidal * dy]
                        u.append(mg * ddz + tidal * dz)
                    if live_mass:
                        mrow = jnp.minimum(srow, tmass_ref.shape[1] - 1)
                        tmg = tmass_ref[jt, mrow][None, :] * g
                        u = [u[0] + tmg * dx, u[1] + tmg * dy, u[2] + tmg * dz]
                    terms.extend(u)
                if live_g:
                    terms.extend([mg * dx, mg * dy, mg * dz])
                if live_soft:
                    if softening_kernel == "plummer":
                        # d/d(eps^2) (r^2 + eps^2)^(-3/2) = (1/2) (1/r) dg/dr
                        assert dgr is not None
                        dge = 0.5 * dgr
                    else:
                        dge = jnp.where(
                            active,
                            pair_softening_sq_derivative(
                                jnp.where(active, r_sq, 1.0),
                                soft_params,
                                softening_kernel,
                            ),
                            0.0,
                        )
                    me = sm * dge
                    terms.extend([me * dx, me * dy, me * dz])
                return tuple(a + t for a, t in zip(inner, terms))

            n_blocks = (scount + (bs - 1)) // bs
            if not wide:
                return lax.fori_loop(0, n_blocks, _block, acc)
            part = tuple(jnp.zeros((bt, bs), dtype) for _ in range(n_acc))
            part = lax.fori_loop(0, n_blocks, _block, part)
            return tuple(
                a + jnp.sum(q, axis=1).astype(wide_dtype) for a, q in zip(acc, part)
            )

        # the forward's runs: consecutive entries whose ranges touch are one range,
        # the next entry's range loaded one iteration ahead
        start = chunk_start_ref[c]
        cnt = chunk_count_ref[c]
        last_entry = neighbors_ref.shape[0] - 1

        def _range_of(s):
            sid = neighbors_ref[jnp.clip(start + s, 0, last_entry)]
            return leaf_start_ref[sid], leaf_count_ref[sid]

        def _slot_body(s, carry):
            acc, rs, rl, cur = carry
            ss, sc = cur
            nxt = _range_of(jnp.minimum(s + 1, cnt - 1))
            joins = ss == rs + rl
            acc = lax.cond(joins, lambda a: a, lambda a: _range_pass(rs, rl, a), acc)
            rs = jnp.where(joins, rs, ss)
            rl = jnp.where(joins, rl + sc, sc)
            return acc, rs, rl, nxt

        zero_i = jnp.zeros((), tcount.dtype)
        acc, rs, rl, _ = lax.fori_loop(
            0, cnt, _slot_body, (acc0, zero_i, zero_i, _range_of(0))
        )
        acc = _range_pass(rs, rl, acc)

        def _self_pass(acc):
            return _range_pass(tstart, tcount, acc, exclude_lane=lane_idx)

        acc = lax.cond(chunk_first_ref[c] != 0, _self_pass, lambda a: a, acc)
        sums = acc if wide else tuple(jnp.sum(a, axis=1) for a in acc)
        g_value = g_ref[0].astype(sums[0].dtype)
        shared = sums[n_lanes:]
        s0 = shared[:3] if live_g else None
        e0 = shared[-3:] if live_soft else None
        values = []
        for jj, jt in enumerate(jl):
            for a in range(3):
                v = (
                    -g_value * sums[3 * jj + a]
                    if (live_pos or live_mass)
                    else jnp.zeros_like(sums[0])
                )
                if s0 is not None:
                    v = v - tg_ref[jt].astype(v.dtype) * s0[a]
                if e0 is not None:
                    v = v - g_value * tsoft_ref[jt].astype(v.dtype) * e0[a]
                values.append(v.astype(out_dtype))
        if out_block is not None:
            zero_out = jnp.zeros((bt,), out_dtype)
            for comp in range(n_lanes):
                out_block[0, :, comp] = jnp.where(tvalid, values[comp], zero_out)
            return
        out = out_refs[-1]
        n_rows = out.shape[1]
        # masked lanes point PAST the outputs, as the forward's stores do
        where = jnp.where(tvalid, tstart + lane_idx, n_rows + lane)
        for jj in range(tpp):
            mask = (tvalid & (js[jj] < k_tangents)) if tails else tvalid
            for a in range(3):
                plgpu.store(out.at[jl[jj], where, a], values[3 * jj + a], mask=mask)


def _direct_tan_call(
    pm: Array,
    tpos: Array,
    tmass: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    chunk_leaf: Array,
    chunk_start: Array,
    chunk_count: Array,
    chunk_first: Array,
    softening: Array,
    g: Array,
    tsoft: Array,
    tg: Array,
    *,
    bt: int,
    bs: int,
    n_sub: int,
    direct: bool,
    num_rows: int,
    out_dtype: Any,
    num_warps: int,
    num_stages: int,
    interpret: bool,
    softening_kernel: str,
    k_tangents: int,
    tpp: int,
    live: tuple[bool, bool, bool, bool],
    wide_dtype: Any,
    name: str,
) -> Any:
    """One launch of :func:`_nearfield_direct_tan_kernel` over ``chunk_leaf``.

    Parameters
    ----------
    pm : Array
        ``(N + pad, 4)`` sorted x, y, z, mass.
    tpos : Array
        ``(K, N, 3)`` position tangents.
    tmass : Array
        ``(K, N)`` mass tangents.
    leaf_start : Array
        ``(L,)`` int32 first particle per leaf.
    leaf_count : Array
        ``(L,)`` int32 particles per leaf.
    neighbors : Array
        ``(E,)`` neighbour leaf indices.
    chunk_leaf : Array
        ``(C,)`` target leaf per program row, ``-1`` = padding.
    chunk_start : Array
        ``(C,)`` first entry of each piece.
    chunk_count : Array
        ``(C,)`` entries of each piece.
    chunk_first : Array
        ``(C,)`` ``1`` where the self term rides.
    softening : Array
        ``(2,)`` softening parameters.
    g : Array
        ``(1,)`` gravitational constant.
    tsoft : Array
        ``(K,)`` softening tangents.
    tg : Array
        ``(K,)`` ``G`` tangents.
    bt : int
        Target subtile.
    bs : int
        Source tile.
    n_sub : int
        Subtiles per leaf.
    direct : bool
        Particle-order outputs (else per-piece partials).
    num_rows : int
        ``N`` (direct).
    out_dtype : Any
        Output dtype (the wide dtype with the wide accumulator).
    num_warps : int
        Triton warps per program.
    num_stages : int
        Triton pipeline stages.
    interpret : bool
        Pallas interpret mode.
    softening_kernel : str
        The pair kernel.
    k_tangents : int
        ``K``.
    tpp : int
        Tangents per program (grid axis 2 over ``ceil(K / tpp)`` groups).
    live : tuple[bool, bool, bool, bool]
        Tangent liveness (positions, masses, softening, ``G``).
    wide_dtype : Any
        Wide accumulator dtype or ``None``.
    name : str
        Kernel name.

    Returns
    -------
    Any
        Direct: the ``(K, N, 3)`` tangents. Blocked: ``(C, n_sub Bt, 3 G tpp)``
        partials (``G`` groups; columns past ``3 K`` are padding).
    """
    from jaccpot.pallas._tangent_grid import late_zeros, tangent_groups

    assert pl is not None and plgpu is not None
    groups = tangent_groups(int(k_tangents), int(tpp))
    capacity = int(chunk_leaf.shape[0])

    def _kernel(*refs):
        return _nearfield_direct_tan_kernel(
            *refs,
            subtile=bt,
            source_tile=bs,
            softening_kernel=softening_kernel,
            k_tangents=int(k_tangents),
            tpp=int(tpp),
            live=live,
            direct=direct,
            wide_dtype=wide_dtype,
        )

    def _full(arr: Array) -> Any:
        assert pl is not None
        shp = tuple(arr.shape)
        return pl.BlockSpec(shp, (lambda *_: (0,) * len(shp)))

    operands = [
        pm,
        tpos,
        tmass,
        leaf_start,
        leaf_count,
        neighbors,
        chunk_leaf,
        chunk_start,
        chunk_count,
        chunk_first,
        softening,
        g,
        tsoft,
        tg,
    ]
    in_specs = [_full(o) for o in operands]
    extra: dict[str, Any] = {}
    if direct:
        # rows no leaf holds stay zero: a table made after this call's inputs
        # exist, never hoisted and merged with another force's (late_zeros)
        zeros = late_zeros(
            (int(k_tangents), int(num_rows), 3), out_dtype, pm, interpret=interpret
        )
        in_specs.append(_full(zeros))
        out_specs: Any = _full(zeros)
        out_shape: Any = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)
        extra["input_output_aliases"] = {len(operands): 0}
        operands.append(zeros)
    else:
        width = 3 * int(tpp)
        out_specs = pl.BlockSpec((1, bt, width), lambda c, sub, g: (c, sub, g))
        out_shape = jax.ShapeDtypeStruct(
            (capacity, n_sub * bt, groups * width), out_dtype
        )
    return pl.pallas_call(
        _kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=(capacity, n_sub, groups),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=name + ("" if softening_kernel == "plummer" else "_" + softening_kernel),
        **extra,
    )(*operands)


def _resolve_tangent_accum(accum: Optional[str]) -> str:
    """The tangent's accumulator mode: the argument, else the environment.

    Parameters
    ----------
    accum : Optional[str]
        ``"input"``, ``"wide"`` or ``None`` (``JACCPOT_NEARFIELD_TANGENT_ACCUM``,
        default ``"input"``, the forward's).

    Returns
    -------
    str
        The mode.

    Raises
    ------
    ValueError
        For another mode.
    """
    mode = (
        env_text("JACCPOT_NEARFIELD_TANGENT_ACCUM", "input")
        if accum is None
        else str(accum)
    )
    if mode not in _TANGENT_ACCUM_MODES:
        raise ValueError(f"accum must be one of {_TANGENT_ACCUM_MODES}, got {mode!r}")
    return mode


def _scalar_tangent(t: Array, k: int, dtype: Any) -> Array:
    """A scalar input's ``K`` tangents as a ``(K,)`` array.

    Parameters
    ----------
    t : Array
        ``(K,)`` plus the scalar's shape (``()`` or ``(1,)``).
    k : int
        ``K``.
    dtype : Any
        Working dtype.

    Returns
    -------
    Array
        ``(K,)``.

    Raises
    ------
    ValueError
        If ``t`` is not one value per tangent.
    """
    t = jnp.asarray(t, dtype)
    if k != 1 and int(t.size) == 1 and t.ndim >= 1 and int(t.shape[0]) == 1:
        # a dead tangent's (1, ...) placeholder (``dead_placeholders``)
        return jnp.zeros((k,), dtype)
    if int(t.size) != k:
        raise ValueError(f"a scalar's tangent must hold K={k} values, got {t.shape}")
    return t.reshape((k,))


def _near_tangent_impl(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    softening_sq: Array,
    G: Array,
    t_positions: Array,
    t_masses: Array,
    t_softening_sq: Array,
    t_G: Array,
    *,
    static: _DirectStatic,
    live: tuple[bool, ...] = (True, True, True, True),
    accum: Optional[str] = None,
    tangents_per_program: Optional[int] = None,
) -> list:
    """The Pallas tangent kernels for any ``K``.

    See :data:`nearfield_leafpair_csr_sorted_direct_pallas_tangent`.

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)``.
    masses_sorted : Array
        ``(N,)``.
    leaf_start : Array
        ``(L,)``.
    leaf_count : Array
        ``(L,)``.
    neighbors : Array
        ``(E,)`` leaf indices.
    offsets : Array
        Row starts (``L`` or ``L + 1``).
    counts : Array
        ``(L,)`` row lengths.
    softening_sq : Array
        Scalar.
    G : Array
        Scalar.
    t_positions : Array
        ``(K, N, 3)``.
    t_masses : Array
        ``(K, N)``.
    t_softening_sq : Array
        ``(K,)`` (plus the scalar's shape).
    t_G : Array
        ``(K,)`` (plus the scalar's shape).
    static : _DirectStatic
        The forward's options.
    live : tuple[bool, ...]
        Liveness of the four tangents.
    accum : Optional[str]
        ``"input"`` / ``"wide"`` / ``None`` (the environment).
    tangents_per_program : Optional[int]
        Tangents per program (``None``: ``JACCPOT_TANGENT_TPP_NEAR``; see
        :func:`~jaccpot.pallas._tangent_grid.tangents_per_program`). Static.

    Returns
    -------
    list
        ``[t_accelerations (K, N, 3)]``.

    Raises
    ------
    RuntimeError
        If Pallas or its Triton backend could not be imported.
    ValueError
        On tangent shapes that do not match the primals.
    """
    from jaccpot.pallas._tangent_grid import tangent_k
    from jaccpot.pallas._tangent_grid import tangents_per_program as _tpp

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")
    positions_sorted = jnp.asarray(positions_sorted)
    dtype = positions_sorted.dtype
    n = int(positions_sorted.shape[0])
    num_leaves = int(leaf_start.shape[0])
    leaf_width = int(static.leaf_width)
    # a dead tangent may be its (1, ...) placeholder: never read (``live``)
    k_tan = tangent_k(t_positions, t_masses, t_softening_sq, t_G)
    if (bool(live[0]) and tuple(t_positions.shape) != (k_tan, n, 3)) or (
        bool(live[1]) and tuple(t_masses.shape) != (k_tan, n)
    ):
        raise ValueError(
            f"tangents must be ({k_tan}, {n}, 3) and ({k_tan}, {n}), got "
            f"{tuple(t_positions.shape)} and {tuple(t_masses.shape)}"
        )
    tsoft = _scalar_tangent(t_softening_sq, k_tan, dtype)
    tg = _scalar_tangent(t_G, k_tan, dtype)
    live4 = tuple(bool(x) for x in live)
    if len(live4) != 4:
        raise ValueError(f"live must hold four flags, got {live}")
    kernel_name = resolve_softening_kernel(static.softening_kernel)
    if num_leaves == 0 or leaf_width == 0 or n == 0 or not any(live4):
        return [jnp.zeros((k_tan, n, 3), dtype)]
    wide_dtype = _resolve_accum_dtype(_resolve_tangent_accum(accum), dtype)
    out_dtype = dtype if wide_dtype is None else wide_dtype
    bs = TANGENT_SOURCE_TILE
    bt = _resolve_subtile(TANGENT_TARGET_SUBTILE, leaf_width)
    width_pad = ((leaf_width + bt - 1) // bt) * bt
    num_warps = max(1, bt // 32)
    pad = max(width_pad, bs)
    pm = jnp.concatenate(
        [positions_sorted, jnp.asarray(masses_sorted, dtype)[:, None]], axis=1
    )
    pm = jnp.pad(pm, ((0, pad), (0, 0)))
    # the tangents as given: no concatenated, padded staging table
    tpos = jnp.asarray(t_positions, dtype)
    tmass = jnp.asarray(t_masses, dtype)
    idx = jnp.asarray(counts).dtype
    counts_i = jnp.asarray(counts, idx)
    row_start = jnp.asarray(offsets, idx)[:num_leaves]
    neighbors = jnp.asarray(neighbors, idx)
    num_entries = int(neighbors.shape[0])
    if num_entries == 0:
        # the kernel prefetches entry 0 of every row, read or not
        neighbors = jnp.zeros((1,), idx)
    start_i = jnp.minimum(jnp.asarray(leaf_start, jnp.int32), n)
    count_i = jnp.asarray(leaf_count, jnp.int32)
    softening = softening_params_from_sq(kernel_name, softening_sq, dtype)
    g_arr = jnp.asarray([G], dtype=dtype)
    row_limit = static.row_limit
    if row_limit is None:
        row_limit = env_int(
            "JACCPOT_NEARFIELD_DIRECT_ROW_LIMIT", DIRECT_ROW_LIMIT, minimum=1
        )
    piece = max(int(row_limit), max(1, int(static.chunk)))
    piece_i = jnp.asarray(piece, idx)
    tpp = _tpp("NEAR", k_tan, tangents_per_program)
    live_tag = "".join(t for t, on in zip("xmeg", live4) if on)
    tag = f"t{bt}_s{bs}_w{leaf_width}_c{piece}_K{k_tan}_tpp{tpp}_{live_tag}" + (
        "_wide" if wide_dtype is not None else ""
    )
    launch = functools.partial(
        _direct_tan_call,
        pm,
        tpos,
        tmass,
        start_i,
        count_i,
        neighbors,
        bt=bt,
        bs=bs,
        n_sub=width_pad // bt,
        out_dtype=out_dtype,
        num_warps=num_warps,
        num_stages=int(static.num_stages),
        interpret=bool(static.interpret),
        softening_kernel=kernel_name,
        k_tangents=k_tan,
        tpp=tpp,
        live=live4,
        wide_dtype=wide_dtype,
    )
    scalars = (softening, g_arr, tsoft, tg)
    # 1. each row up to `piece` entries + the self term, in particle order, into
    # the (K, N, 3) result itself
    out = launch(
        jnp.arange(num_leaves, dtype=idx),
        row_start,
        jnp.minimum(counts_i, piece_i),
        jnp.ones((num_leaves,), idx),
        *scalars,
        direct=True,
        num_rows=n,
        name=f"near_tan_direct_first_{tag}",
    )
    # 2. the rows' remaining pieces: partials, summed in one fixed order per row and
    # added once (the forward's `_rest`, op for op)
    capacity = leafpair_extra_chunk_capacity(num_entries, piece)

    def _rest(out_in: Array) -> Array:
        per_leaf = jnp.maximum((counts_i + piece_i - 1) // piece_i - 1, 0)
        ends = jnp.cumsum(per_leaf, dtype=idx)
        first = ends - per_leaf
        c = jnp.arange(capacity, dtype=idx)
        leaf = jnp.searchsorted(
            ends, c, side="right", method=searchsorted_method()
        ).astype(idx)
        valid = leaf < jnp.asarray(num_leaves, idx)
        leaf_safe = jnp.minimum(leaf, jnp.asarray(num_leaves - 1, idx))
        k = c - first[leaf_safe] + 1
        partials = launch(
            jnp.where(valid, leaf_safe, jnp.asarray(-1, idx)),
            jnp.where(valid, row_start[leaf_safe] + k * piece_i, 0),
            jnp.where(
                valid, jnp.clip(counts_i[leaf_safe] - k * piece_i, 0, piece_i), 0
            ),
            jnp.zeros((capacity,), idx),
            *scalars,
            direct=False,
            num_rows=0,
            name=f"near_tan_direct_rest_{tag}",
        )
        lane = jnp.arange(width_pad, dtype=jnp.int32)
        tcount = jnp.where(valid, count_i[leaf_safe], 0)
        rows = start_i[leaf_safe][:, None] + lane[None, :]
        rows = jnp.where(lane[None, :] < tcount[:, None], rows, n)
        partials, rows = _rest_pieces_in_fixed_order(
            partials[..., : 3 * k_tan], rows, jnp.where(valid, leaf_safe, -1), n
        )
        cap, wpad = int(partials.shape[0]), int(partials.shape[1])
        upd = jnp.transpose(partials.reshape(cap, wpad, k_tan, 3), (2, 0, 1, 3))
        return out_in.at[:, rows, :].add(upd, mode="drop")

    if capacity > 0:
        out = lax.cond(jnp.max(counts_i) > piece_i, _rest, lambda o: o, out)
    return [out.astype(dtype)]


def _near_tangent_abstract(
    positions_sorted: Any, *operands: Any, **params: Any
) -> list:
    """Closed-form output shape of the near-field tangent op.

    Parameters
    ----------
    positions_sorted : Any
        ``(N, 3)`` shape/dtype.
    *operands : Any
        The other operands; the ninth is ``t_positions`` ``(K, N, 3)``.
    **params : Any
        Unused.

    Returns
    -------
    list
        ``[ShapeDtypeStruct((K, N, 3))]``.
    """
    del params
    k_tan = max(int(t.shape[0]) for t in operands[8:12])
    return [
        jax.ShapeDtypeStruct(
            (k_tan, int(positions_sorted.shape[0]), 3), positions_sorted.dtype
        )
    ]


#: Forward-mode tangent of the direct near field's seam (:func:`_direct_cvjp`; the
#: seam table of the forward-mode plan): ``op(positions_sorted, masses_sorted,
#: leaf_start, leaf_count, neighbors, offsets, counts, softening_sq, G,
#: t_positions (K, N, 3), t_masses (K, N), t_softening_sq (K,), t_G (K,), *,
#: static, live) -> (t_accelerations (K, N, 3),)``. ``static`` is the seam's
#: :class:`_DirectStatic`; the force-scale lane has no tangent. Optional ``accum``
#: (``"input"`` / ``"wide"``; default ``JACCPOT_NEARFIELD_TANGENT_ACCUM`` or
#: ``"input"``) and ``tangents_per_program`` (default ``JACCPOT_TANGENT_TPP_NEAR``);
#: any ``K`` (the kernels' tangent-group grid axis masks a partial last group).
#: Every softening kernel. The reference is
#: :func:`nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp`.
def _near_tangent_op_options() -> dict:
    from jaccpot.pallas._tangent_grid import tangent_op_options

    return tangent_op_options()


nearfield_leafpair_csr_sorted_direct_pallas_tangent = make_tangent_op(
    "nearfield_leafpair_csr_sorted_direct",
    _near_tangent_impl,
    num_primals=9,
    abstract_eval=_near_tangent_abstract,
    **_near_tangent_op_options(),
)


def _near_tangent_twin(
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    *,
    num_particles: int,
    static: _DirectStatic,
) -> Callable[[Array, Array, Array, Array], Array]:
    """Pure-JAX twin of the direct near field: ``(x, m, eps^2, G) -> (N, 3)``.

    Every ``(target leaf, source leaf)`` of the CSR (each row's live entries) and
    every leaf's self pair (diagonal excluded), as dense ``(W, W)`` particle blocks,
    the kernel's factors (:func:`masked_pair_factors`, double-``where`` guarded so the
    jvp is finite on masked pairs) and a ``segment_sum`` onto the sinks. Small inputs
    only: it materialises ``(E + L) W^2`` pairs.

    Parameters
    ----------
    leaf_start : Array
        ``(L,)``.
    leaf_count : Array
        ``(L,)``.
    neighbors : Array
        ``(E,)`` leaf indices.
    offsets : Array
        Row starts (``L`` or ``L + 1``).
    counts : Array
        ``(L,)``.
    num_particles : int
        ``N``.
    static : _DirectStatic
        The forward's options (``leaf_width``, ``softening_kernel``).

    Returns
    -------
    Callable[[Array, Array, Array, Array], Array]
        ``f(positions_sorted, masses_sorted, softening_sq, G)``.
    """
    n = int(num_particles)
    num_leaves = int(leaf_start.shape[0])
    width = int(static.leaf_width)
    kernel_name = resolve_softening_kernel(static.softening_kernel)
    idx = jnp.int32
    start = jnp.minimum(jnp.asarray(leaf_start, idx), n)
    count = jnp.asarray(leaf_count, idx)
    nbr = jnp.asarray(neighbors, idx)
    num_entries = int(nbr.shape[0])
    row_off = jnp.asarray(offsets, idx)[:num_leaves]
    row_cnt = jnp.asarray(counts, idx)[:num_leaves]
    entry = jnp.arange(num_entries, dtype=idx)
    row = jnp.searchsorted(row_off, entry, side="right").astype(idx) - 1
    row_safe = jnp.clip(row, 0, max(num_leaves - 1, 0))
    entry_live = (row >= 0) & (entry < row_off[row_safe] + row_cnt[row_safe])
    tl = jnp.concatenate([row_safe, jnp.arange(num_leaves, dtype=idx)])
    sl = jnp.concatenate(
        [jnp.clip(nbr, 0, max(num_leaves - 1, 0)), jnp.arange(num_leaves, dtype=idx)]
    )
    pair_live = jnp.concatenate([entry_live, jnp.ones((num_leaves,), bool)])
    is_self = jnp.concatenate(
        [jnp.zeros((num_entries,), bool), jnp.ones((num_leaves,), bool)]
    )
    slot = jnp.arange(width, dtype=idx)
    ti = start[tl][:, None] + slot[None, :]
    si = start[sl][:, None] + slot[None, :]
    tmask = slot[None, :] < count[tl][:, None]
    smask = slot[None, :] < count[sl][:, None]
    base = (
        pair_live[:, None, None]
        & tmask[:, :, None]
        & smask[:, None, :]
        & ~(is_self[:, None, None] & (slot[:, None] == slot[None, :])[None])
    )
    ti_c = jnp.clip(ti, 0, max(n - 1, 0))
    si_c = jnp.clip(si, 0, max(n - 1, 0))
    sinks = jnp.broadcast_to(ti_c[:, :, None], base.shape).reshape(-1)

    def acc(pos: Array, mass: Array, soft: Array, g: Array) -> Array:
        dtype = pos.dtype
        if kernel_name == "plummer":
            params = softening_params_from_sq(kernel_name, soft, dtype)
        else:
            # softening_params_from_sq's sqrt with a guarded derivative at eps = 0
            # (the same value; there the kernel's dg/d(eps^2) is zero too)
            sq = jnp.asarray(soft, dtype)
            positive = sq > 0.0
            eps = jnp.where(positive, jnp.sqrt(jnp.where(positive, sq, 1.0)), 0.0)
            params = softening_params(kernel_name, eps, dtype)
        d = pos[ti_c][:, :, None, :] - pos[si_c][:, None, :, :]
        r_sq = jnp.sum(d * d, axis=-1)
        active = base & ((r_sq > 0.0) | (params[0] > 0.0))
        gk, _, _ = masked_pair_factors(r_sq, active, params, kernel_name)
        contrib = (-g * mass[si_c][:, None, :] * gk)[..., None] * d
        contrib = jnp.where(active[..., None], contrib, 0.0)
        return jax.ops.segment_sum(contrib.reshape(-1, 3), sinks, num_segments=n)

    return acc


def nearfield_leafpair_csr_sorted_direct_pallas_tangent_jnp(
    positions_sorted: Array,
    masses_sorted: Array,
    leaf_start: Array,
    leaf_count: Array,
    neighbors: Array,
    offsets: Array,
    counts: Array,
    softening_sq: Array,
    G: Array,
    t_positions: Array,
    t_masses: Array,
    t_softening_sq: Array,
    t_G: Array,
    *,
    static: _DirectStatic,
    live: tuple[bool, ...] = (True, True, True, True),
    accum: Optional[str] = None,
) -> tuple[Array]:
    """Pure-JAX reference of :data:`nearfield_leafpair_csr_sorted_direct_pallas_tangent`.

    ``jax.vmap`` over ``K`` of ``jax.jvp`` of :func:`_near_tangent_twin`, whose
    primal is the direct forward's to round-off (another summation order). The
    parity reference of the tangent kernel and track A's fallback; it materialises
    ``(E + L) W^2`` pairs per tangent, so it is for tests and modest sizes. A
    compact kernel at ``softening_sq == 0`` (Newton) gets a zero ``d/d(eps^2)``,
    as in the kernel (the true one is singular there: ``h ~ sqrt(eps^2)``).

    Parameters
    ----------
    positions_sorted : Array
        ``(N, 3)``.
    masses_sorted : Array
        ``(N,)``.
    leaf_start : Array
        ``(L,)``.
    leaf_count : Array
        ``(L,)``.
    neighbors : Array
        ``(E,)``.
    offsets : Array
        Row starts.
    counts : Array
        ``(L,)``.
    softening_sq : Array
        Scalar.
    G : Array
        Scalar.
    t_positions : Array
        ``(K, N, 3)``.
    t_masses : Array
        ``(K, N)``.
    t_softening_sq : Array
        ``(K,)`` plus the scalar's shape.
    t_G : Array
        ``(K,)`` plus the scalar's shape.
    static : _DirectStatic
        The forward's options.
    live : tuple[bool, ...]
        Unused (zero tangents give zero).
    accum : Optional[str]
        Unused.

    Returns
    -------
    tuple[Array]
        ``(t_accelerations (K, N, 3),)``.
    """
    del live, accum
    pos = jnp.asarray(positions_sorted)
    dtype = pos.dtype
    n = int(pos.shape[0])
    k_tan = int(t_positions.shape[0])
    if int(leaf_start.shape[0]) == 0 or int(static.leaf_width) == 0 or n == 0:
        return (jnp.zeros((k_tan, n, 3), dtype),)
    twin = _near_tangent_twin(
        leaf_start,
        leaf_count,
        neighbors,
        offsets,
        counts,
        num_particles=n,
        static=static,
    )
    mass = jnp.asarray(masses_sorted, dtype)
    soft = jnp.asarray(softening_sq, dtype)
    g = jnp.asarray(G, dtype)
    tsoft = _scalar_tangent(t_softening_sq, k_tan, dtype)
    tg = _scalar_tangent(t_G, k_tan, dtype)

    def one(tx: Array, tm: Array, ts: Array, tgk: Array) -> Array:
        return jax.jvp(
            twin,
            (pos, mass, soft, g),
            (
                tx.astype(dtype),
                tm.astype(dtype),
                jnp.broadcast_to(ts, soft.shape),
                jnp.broadcast_to(tgk, g.shape),
            ),
        )[1]

    return (jax.vmap(one)(t_positions, t_masses, tsoft, tg),)

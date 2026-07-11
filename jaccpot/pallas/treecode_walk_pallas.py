"""Pallas per-leaf treecode walk (device-resident, one program per target leaf).

The pure-JAX walk (:mod:`jaccpot.experimental.treecode_walk`) is a vmapped
``fori_loop``: device-resident (unlike the yggdrax dual-tree ``while_loop`` launch
storm that was the original wall; see ``benchmark_a100/WALK_SPEC.md``), but every
lane shares one trip count, so it must run the pessimistic ``max_iters ==
total_nodes`` bound even though a typical leaf's descent visits only a few percent
of the nodes. That makes it O(num_leaves × total_nodes): ~205 ms/walk at 200k on
an A100, larger than the whole 121 ms/step baseline.

This module ports the walk to a single Pallas kernel: ``grid=(num_leaves,)``, one
program per target leaf, each descending the source tree with a PRIVATE stack (à la
Bonsai's ``dev_approximate_gravity``). The structural win a per-program kernel has
that a vmap cannot: a ``while_loop`` that exits the moment THIS leaf's stack drains,
so each descent costs only its own length. Measured 40–140× faster than the pure-JAX
walk (5.1 ms at 200k), bit-identical output.

Design forced by the Triton backend (empirically probed on A100 / sm_80):

* register-array dynamic indexing does NOT lower on Pallas GPU (``dynamic_slice`` /
  ``scatter`` are unimplemented), so the walk CANNOT carry the stack / far / near
  buffers as loop-carried arrays the way the pure-JAX reference does;
* dynamic scalar ``pl.load`` / ``pl.store`` on *refs* DO lower, and store→load
  ordering on a per-program ref survives ACROSS loop iterations (the push/pop stack).

So every dynamically-indexed buffer is a ref: the shared per-node tables
(centres, extents, children) are full-gather input refs (indexed by node id, like
``nearfield_fused_leaf``'s source table); the private stack is a per-program output
block used as HBM scratch (discarded by the wrapper); the far / near lists and
counts are output refs written with guarded dynamic stores.

Because the kernel replays the pure-JAX walk exactly — same LIFO stack order (push
left then right, pop highest index first), same append-in-pop-order — the emitted
``far_nodes`` / ``near_leaves`` arrays are ELEMENT-WISE identical to
:func:`jaccpot.experimental.treecode_walk.treecode_leaf_walk`, not merely
set-equal. That is the parity gate (``interpret=True`` on CPU + on-device A100).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from jaccpot.experimental.treecode_walk import TreecodeLeafLists

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
except Exception:  # pragma: no cover - import is environment-dependent
    pl = None
    plgpu = None

_SENTINEL = -1


def _pow2_ceil(x: int) -> int:
    """Smallest power of two >= x (>= 1). Triton needs pow2 full-tensor widths."""
    x = int(x)
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def pallas_treecode_walk_supported() -> bool:
    """Return whether the active accelerator can run the treecode-walk kernel."""

    if pl is None or plgpu is None:
        return False
    if jax.default_backend() != "gpu":
        return False
    try:
        device = jax.devices()[0]
    except Exception:  # pragma: no cover - backend discovery is environment-dependent
        return False

    compute_capability = getattr(device, "compute_capability", None)
    if compute_capability is None:
        return False
    return float(compute_capability) >= 8.0


def _treecode_walk_kernel(
    cx_ref,  # (N,)  node centre x
    cy_ref,  # (N,)  node centre y
    cz_ref,  # (N,)  node centre z
    ext_ref,  # (N,)  MAC extents
    lc_ref,  # (N,)  left child (-1 at leaves)
    rc_ref,  # (N,)  right child (-1 at leaves)
    leaf_ref,  # (1,) this program's target leaf node id
    theta_ref,  # (1,) theta^2
    root_ref,  # (1,) root node id
    far_ref,  # (1, max_far_pad) out: accepted far source node ids
    near_ref,  # (1, max_near_pad) out: near source leaf node ids
    fc_ref,  # (1,) out: far count
    nc_ref,  # (1,) out: near count
    ovf_ref,  # (1,) out: overflow flag (this leaf)
    stack_ref,  # (1, max_stack) scratch: private descent stack (discarded)
    *,
    num_internal: int,
    max_far: int,
    max_near: int,
    max_far_pad: int,
    max_near_pad: int,
    max_stack: int,
    max_iters: int,
):
    """Treecode descent for ONE target leaf (mirrors ``_single_leaf_walk``)."""

    leaf_node = leaf_ref[0]
    ctx = cx_ref[leaf_node]
    cty = cy_ref[leaf_node]
    ctz = cz_ref[leaf_node]
    ext_t = ext_ref[leaf_node]
    theta_sq = theta_ref[0]

    # Sentinel-fill the (pow2-padded) output rows; only [:max_far]/[:max_near]
    # are returned. Stack needs no init (only slots below sp are ever read).
    far_ref[0, :] = jnp.full((max_far_pad,), _SENTINEL, dtype=far_ref.dtype)
    near_ref[0, :] = jnp.full((max_near_pad,), _SENTINEL, dtype=near_ref.dtype)
    stack_ref[0, 0] = root_ref[0].astype(stack_ref.dtype)

    def cond(carry):
        # Per-leaf early exit: stop as soon as this leaf's stack drains. A vmapped
        # fori_loop cannot do this (all lanes share a trip count); the whole point
        # of one-program-per-leaf is that each descent runs only its own length.
        # ``it < max_iters`` is a defensive cap (each node is popped at most once).
        sp, fc, nc, ovf, it = carry
        return jnp.logical_and(sp > 0, it < max_iters)

    def body(carry):
        sp, fc, nc, ovf, it = carry
        active = sp > 0  # always True given cond, kept for parity with the reference
        sp_pop = jnp.where(active, sp - 1, 0)
        s = stack_ref[0, sp_pop]  # pop

        csx = cx_ref[s]
        csy = cy_ref[s]
        csz = cz_ref[s]
        ext_s = ext_ref[s]
        dx = ctx - csx
        dy = cty - csy
        dz = ctz - csz
        dist_sq = dx * dx + dy * dy + dz * dz
        rsum = ext_t + ext_s
        well_sep = jnp.logical_and((rsum * rsum) <= (theta_sq * dist_sq), dist_sq > 0.0)
        is_leaf = s >= num_internal

        do_far = jnp.logical_and(active, well_sep)
        do_near = jnp.logical_and(
            active, jnp.logical_and(is_leaf, jnp.logical_not(well_sep))
        )
        do_refine = jnp.logical_and(
            active, jnp.logical_and(jnp.logical_not(is_leaf), jnp.logical_not(well_sep))
        )

        # Guarded dynamic stores == pure-JAX masked scatter with mode="drop":
        # write only when in capacity; the count still advances on overflow.
        @pl.when(jnp.logical_and(do_far, fc < max_far))
        def _emit_far():
            far_ref[0, fc] = s

        @pl.when(jnp.logical_and(do_near, nc < max_near))
        def _emit_near():
            near_ref[0, nc] = s

        lc = lc_ref[s]
        rc = rc_ref[s]

        @pl.when(jnp.logical_and(do_refine, sp_pop < max_stack))
        def _push_left():
            stack_ref[0, sp_pop] = lc

        @pl.when(jnp.logical_and(do_refine, sp_pop + 1 < max_stack))
        def _push_right():
            stack_ref[0, sp_pop + 1] = rc

        ovf = jnp.logical_or(
            ovf,
            jnp.logical_or(
                jnp.logical_and(do_far, fc >= max_far),
                jnp.logical_or(
                    jnp.logical_and(do_near, nc >= max_near),
                    jnp.logical_and(do_refine, sp_pop + 2 > max_stack),
                ),
            ),
        )
        fc = fc + do_far.astype(jnp.int32)
        nc = nc + do_near.astype(jnp.int32)
        sp = jnp.where(active, jnp.where(do_refine, sp_pop + 2, sp_pop), sp)
        return sp, fc, nc, ovf, it + 1

    sp0 = jnp.int32(1)
    _, fc, nc, ovf, _ = lax.while_loop(
        cond,
        body,
        (sp0, jnp.int32(0), jnp.int32(0), jnp.bool_(False), jnp.int32(0)),
    )
    fc_ref[0] = fc
    nc_ref[0] = nc
    ovf_ref[0] = ovf


def treecode_leaf_walk_pallas(
    leaf_nodes: Array,
    centers: Array,
    mac_extents: Array,
    left_child_full: Array,
    right_child_full: Array,
    theta_sq: Array,
    root_idx: Array,
    *,
    num_internal: int,
    max_far: int,
    max_near: int,
    max_stack: int,
    max_iters: int,
    num_warps: int = 1,
    num_stages: int = 1,
    interpret: bool = False,
) -> TreecodeLeafLists:
    """Per-leaf treecode walk with Pallas.

    Drop-in device-resident replacement for
    :func:`jaccpot.experimental.treecode_walk.treecode_leaf_walk` — same
    signature and (bit-identical) :class:`TreecodeLeafLists` output. See that
    function for the walk contract; this one runs the whole descent in a single
    kernel launch (``grid=(num_leaves,)``, one program per target leaf).

    The private descent stack is realised as a per-program output block used as
    HBM scratch (the Triton backend has no shared-memory scratch); it is written
    and read but not returned.
    """

    if pl is None or plgpu is None:
        raise RuntimeError("jax.experimental.pallas is not available")

    centers = jnp.asarray(centers)
    dtype = centers.dtype
    mac_extents = jnp.asarray(mac_extents, dtype=dtype)
    left_child_full = jnp.asarray(left_child_full)
    idx_dtype = left_child_full.dtype
    right_child_full = jnp.asarray(right_child_full, dtype=idx_dtype)
    leaf_nodes = jnp.asarray(leaf_nodes, dtype=idx_dtype)
    theta_arr = jnp.asarray([theta_sq], dtype=dtype)
    root_arr = jnp.asarray([root_idx], dtype=idx_dtype)

    if centers.ndim != 2 or centers.shape[-1] != 3:
        raise ValueError("centers must have shape (N, 3)")

    num_leaves = int(leaf_nodes.shape[0])
    max_far = int(max_far)
    max_near = int(max_near)
    max_stack = int(max_stack)

    if num_leaves == 0:
        empty = jnp.zeros((0, max_far), dtype=idx_dtype)
        return TreecodeLeafLists(
            far_nodes=empty,
            near_leaves=jnp.zeros((0, max_near), dtype=idx_dtype),
            far_count=jnp.zeros((0,), dtype=jnp.int32),
            near_count=jnp.zeros((0,), dtype=jnp.int32),
            overflow=jnp.bool_(False),
        )

    # 1-D centre columns: scalar dynamic loads only, no small-2D block load.
    cx = centers[:, 0]
    cy = centers[:, 1]
    cz = centers[:, 2]
    total_nodes = int(centers.shape[0])

    # Pad far/near output widths to a power of two (Triton full-tensor init).
    max_far_pad = _pow2_ceil(max_far)
    max_near_pad = _pow2_ceil(max_near)

    def _kernel(*refs):
        return _treecode_walk_kernel(
            *refs,
            num_internal=int(num_internal),
            max_far=max_far,
            max_near=max_near,
            max_far_pad=max_far_pad,
            max_near_pad=max_near_pad,
            max_stack=max_stack,
            max_iters=int(max_iters),
        )

    # Shared per-node tables: full-gather refs (whole array visible to every
    # program, indexed dynamically by node id). Per-node/per-leaf scalars: (1,).
    table = lambda n: pl.BlockSpec((n,), lambda leaf: (0,))
    scalar = pl.BlockSpec((1,), lambda leaf: (0,))

    kernel = pl.pallas_call(
        _kernel,
        out_shape=[
            jax.ShapeDtypeStruct((num_leaves, max_far_pad), idx_dtype),
            jax.ShapeDtypeStruct((num_leaves, max_near_pad), idx_dtype),
            jax.ShapeDtypeStruct((num_leaves,), jnp.int32),
            jax.ShapeDtypeStruct((num_leaves,), jnp.int32),
            jax.ShapeDtypeStruct((num_leaves,), jnp.bool_),
            jax.ShapeDtypeStruct((num_leaves, max_stack), idx_dtype),  # scratch
        ],
        in_specs=[
            table(total_nodes),  # cx
            table(total_nodes),  # cy
            table(total_nodes),  # cz
            table(total_nodes),  # ext
            table(total_nodes),  # lc
            table(total_nodes),  # rc
            pl.BlockSpec((1,), lambda leaf: (leaf,)),  # leaf node id
            scalar,  # theta^2
            scalar,  # root idx
        ],
        out_specs=[
            pl.BlockSpec((1, max_far_pad), lambda leaf: (leaf, 0)),
            pl.BlockSpec((1, max_near_pad), lambda leaf: (leaf, 0)),
            pl.BlockSpec((1,), lambda leaf: (leaf,)),
            pl.BlockSpec((1,), lambda leaf: (leaf,)),
            pl.BlockSpec((1,), lambda leaf: (leaf,)),
            pl.BlockSpec((1, max_stack), lambda leaf: (leaf, 0)),
        ],
        grid=(num_leaves,),
        compiler_params=plgpu.CompilerParams(
            num_warps=int(num_warps), num_stages=int(num_stages)
        ),
        interpret=bool(interpret),
        name=f"treecode_walk_l{num_leaves}_f{max_far}_n{max_near}",
    )

    far, near, fc, nc, ovf, _stack = kernel(
        cx,
        cy,
        cz,
        mac_extents,
        left_child_full,
        right_child_full,
        leaf_nodes,
        theta_arr,
        root_arr,
    )
    if max_far_pad != max_far:
        far = far[:, :max_far]
    if max_near_pad != max_near:
        near = near[:, :max_near]
    return TreecodeLeafLists(
        far_nodes=far,
        near_leaves=near,
        far_count=fc,
        near_count=nc,
        overflow=jnp.any(ovf),
    )


def treecode_leaf_walk_backend(*, prefer_pallas: bool = True) -> str:
    """Describe which backend a dual-path walk would use (see step 3 wiring)."""

    if prefer_pallas and pallas_treecode_walk_supported():
        return "pallas"
    return "jax"


__all__ = [
    "pallas_treecode_walk_supported",
    "treecode_leaf_walk_backend",
    "treecode_leaf_walk_pallas",
]

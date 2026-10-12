"""The M2L apply-and-accumulate seam: every batching, both gates.

THIS MODULE IS ONE UNIT ON PURPOSE. ``_apply_real_m2l`` consults its
``*_pallas_active`` gate and routes to either the reference batch kernel or the
fused Pallas twin. ARCHITECTURE §10's invariant is that **every discriminator on
that path is a static argument** -- ``order``, ``m2l_impl`` -- so the whole
dispatch compiles to one branch-free jaxpr per configuration. Splitting *within*
it would break the argument that the invariant holds, which is why the audit says
this seam moves whole. The complex basis's branch (``_apply_complex_m2l``) went
with that basis in the 2026-10 cleanup (C), and the ``basis_mode`` / ``rotation``
discriminators that selected it in Z-b.

EQUIVALENCE THAT MUST HOLD (NUMERICS_AND_JAX §1, asserted in
``tests/unit/operators/test_m2l_real_fused_pallas.py``):
``_m2l_real_batch_kernel_fused_pallas`` == ``_m2l_real_batch_kernel`` (rot-scale
reference). The Pallas kernel is an execution accelerator, not different
mathematics.

THE TWO ACCUMULATORS are two batchings of the same sum over a flat pair list:
full-batch and chunked scan. They must agree to reassociation only. There used to
be four: the class-grouped and class-major batchings went with the grouped far
field in the 2026-10 cleanup (X3). One of them was G.11, a 60x accuracy gap
caused by ``pair_grouped`` gathering rotations with class ids from the wrong
ordering -- exactly the kind of defect that hides in a family of near-duplicates.

Split out of ``core.py`` (Tier 1.6, A.9 seam 2); every function body is unchanged.
"""

from __future__ import annotations

import functools
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Bool, Float, Int, jaxtyped

from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    make_m2l_real_fused_carry_axis_derivative,
)
from jaccpot.runtime.grad_options import fused_m2l_pallas_enabled

# The fused-M2L transverse-tangent carrier, built ONCE and cached.
#
# Cached rather than module-level for two independent reasons. The pure-JAX twin
# the carrier needs lives in `jaccpot.pallas.*`, and importing that at module scope
# would drag `jax.experimental.pallas` onto the `import jaccpot` path, which it is
# deliberately not on -- every pallas import in this module is function-local for
# that reason. And they are `custom_jvp` objects, so building one per call would
# create a fresh primitive per call and retrace every time; `lru_cache` keeps the
# identity stable.
#
# `operators/` supplies the derivative rule, this module supplies the twin: that is
# audit G.3's inversion, which is why the factories exist at all.


@functools.lru_cache(maxsize=1)
def _real_fused_carry_axis_derivative() -> Any:
    """The real fused M2L's tangent carrier, built once.

    Returns
    -------
    Any
        ``carry(out, coeffs, delta, blocks_to_z, blocks_from_z, radii, order=p)``.
    """
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_jax

    return make_m2l_real_fused_carry_axis_derivative(m2l_real_fused_jax)


from ..dtypes import INDEX_DTYPE

__all__: list[str] = []


# WHY ONLY ONE AXIS IS ANNOTATED, AND WHY IT IS NOT A WHOLE-MODULE PASS.
#
# `bench/annotation_pilot.py` re-recorded on 2026-09-03 put this module last on rate --
# 23 silent acceptances of 286 perturbations, 8%, on 18 measured functions -- which is the
# August verdict confirmed: it is mostly validated already and converting every bare
# parameter would be effort spent where nothing is wrong. Most of the 23 sat in the
# grouped / class-major accumulators, which went in the 2026-10 cleanup (X3); of the
# survivors, `_chunk_segment_scatter_add` and `_m2l_chunk_contributions` took 2 each.
#
# `nodes` and `sh` tie the three arrays every batching takes, so the annotation is the same
# on the full-batch and chunked accumulators, which is what makes them comparable.
# Evidence: ~50 recorded calls across the (then four) accumulators shared the leading
# axis of `locals_coeffs`, `multip_packed` and `centers` at EIGHT distinct extents (5, 8,
# 13, 63, 127, 255, 511, 1023), and the two packed arrays shared the trailing axis at five
# (4, 9, 16, 25, 81).
#
# `Float` on the packed pair: it was `Inexact` while `basis_mode="complex"` passed
# complex coefficients, until the complex basis went in the 2026-10 cleanup (C, Z-b).
#
# The decorators go INSIDE `jax.jit` on the jitted accumulators, per the house order, so
# beartype runs once per trace rather than per call on the production M2L path.
@jaxtyped(typechecker=beartype)
def _chunk_segment_scatter_add(
    local_accum: Float[Array, "nodes sh"],
    contribs: Float[Array, "chunkflat sh"],
    tgt_chunk: Int[Array, "chunkflat"],
    valid: Bool[Array, "chunkflat"],
    *,
    chunk_size: int,
) -> Array:
    """Reduce one fixed-width chunk by target index and scatter-add into locals.

    Sorts the chunk by target so that contributions to the same target become a
    contiguous segment, reduces within segments with a segmented prefix scan,
    then scatters the one total per segment. Invalid slots are given the
    maximum index so they sort to the end and fall outside the scatter.

    The sort makes the summation order a deterministic function of the target
    indices rather than of the pair order, which is what keeps the two
    accumulators agreeing to reassociation.

    Why a segmented ``associative_scan`` and an out-of-bounds sink, not
    ``segment_sum`` and index 0 (measured 2026-09-07, N=200k Plummer, A100,
    ``strict_run_v2``): this function ran once per 4096-pair chunk, and at leaf
    64 (992k far pairs, 243 chunks per step) its scatter fusions cost 169 ms of
    a 410 ms step -- 686 us per launch for a 4096 x 25 reduction. Both scatters
    in the old body were pathological for XLA's atomic-add lowering: the
    ``segment_sum`` sent every pair of a target to the SAME group row (up to a
    few hundred duplicates per address, serialised), and the final
    ``.at[safe_targets].add`` sent every slot that was not a segment head --
    ~3800 of 4096 -- to node 0 with a zero value, another serialised address.
    The segmented scan reduces within segments with no scatter at all, and the
    non-head slots now carry an index one past the end of ``local_accum``,
    which ``mode="drop"`` discards without a write. Each in-bounds index is
    then unique within the chunk (one segment head per target), so the final
    scatter needs no atomics either.

    Parameters
    ----------
    local_accum : Float[Array, 'nodes sh']
        Local coefficient accumulator to add into. `nodes` is bound by this parameter
        alone -- nothing else in the signature carries it, and the recording shows it
        varying over 7, 15, 31, 127, 255, 511 and 1023 against an unchanged
        `contribs` -- so it is deliberately NOT cross-checked against anything.
    contribs : Float[Array, 'chunkflat sh']
        Per-pair M2L contributions for this chunk. Shares `sh` with `local_accum`,
        which is the coefficient count the two are added along.
    tgt_chunk : Int[Array, 'chunkflat']
        Target node index per pair in the chunk.
    valid : Bool[Array, 'chunkflat']
        Validity mask; the tail chunk is padded.
    chunk_size : int
        Fixed chunk width. Static -- it is what makes every chunk the same shape.

    Returns
    -------
    Array
        ``local_accum`` with this chunk's contributions added.
    """
    masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
    sort_idx = jnp.argsort(masked_targets)
    sorted_keys = masked_targets[sort_idx]
    tgt_sorted = tgt_chunk[sort_idx]
    contribs_sorted = contribs[sort_idx]
    valid_sorted = valid[sort_idx]
    contribs_sorted = jnp.where(valid_sorted[:, None], contribs_sorted, 0)

    boundary = sorted_keys[1:] != sorted_keys[:-1]
    new_group = jnp.concatenate((jnp.ones((1,), dtype=bool), boundary), axis=0)
    is_last = jnp.concatenate((boundary, jnp.ones((1,), dtype=bool)), axis=0)

    def _segmented_add(a: tuple[Array, Array], b: tuple[Array, Array]):
        # Prefix sums that restart at every segment head: the right operand
        # replaces the running sum when it starts a segment, else it adds.
        va, fa = a
        vb, fb = b
        return jnp.where(fb[..., None], vb, va + vb), fa | fb

    segment_prefix, _ = jax.lax.associative_scan(
        _segmented_add, (contribs_sorted, new_group), axis=0
    )
    take = is_last & valid_sorted
    sink = jnp.asarray(local_accum.shape[0], dtype=INDEX_DTYPE)  # out of bounds
    rows_tgt = jnp.where(take, tgt_sorted, sink)
    rows_val = jnp.where(take[:, None], segment_prefix, 0)
    return local_accum.at[rows_tgt].add(
        rows_val, mode="drop", indices_are_sorted=True, unique_indices=True
    )


@partial(jax.jit, static_argnames=("order", "m2l_impl"))
def _m2l_real_batch_kernel(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Vectorized real-basis M2L translation kernel.

    The rot-scale reference in the Dehnen real basis, and the definition its
    fused Pallas twin is asserted equal to.

    Parameters
    ----------
    multipoles : Array
        Real multipole coefficients, one row per pair.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    m2l_impl : str
        Must be ``"rot_scale"``; the real basis has no other implementation.
        Static.

    Returns
    -------
    Array
        Real local contributions, aligned with ``multipoles``.

    Raises
    ------
    ValueError
        If ``m2l_impl`` is anything but ``"rot_scale"``. Checked here rather than
        at the caller so the constraint holds for every route in.
    """
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    return m2l_rot_scale_real_batch(multipoles, deltas, order=order)


def _real_m2l_pallas_active() -> bool:
    """Whether to route the real-basis M2L z-core through the Pallas kernel.

    Gated by ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`` and the sm_80+ support
    check for the FUSED real kernel this routes to (falls back to the pure-JAX
    rot-scale otherwise). Trace-time; the flag does not change within a compiled
    run.

    Uses :func:`pallas_m2l_real_fused_supported` (the gate for the kernel actually
    dispatched, :func:`_m2l_real_batch_kernel_fused_pallas`), which requires
    Ampere+ (sm_80). The z-core kernel's
    ``pallas_m2l_real_supported`` (removed with that kernel in the 2026-10
    cleanup, X5) used previously only checked gpu/tpu, so it would route to
    Pallas on a pre-Ampere GPU where the Triton lowering fails.

    Returns
    -------
    bool
        ``True`` when the flag is set and an Ampere+ GPU is available.
    """
    if not fused_m2l_pallas_enabled():
        return False
    try:
        from jaccpot.pallas.m2l_real_fused import pallas_m2l_real_fused_supported

        return bool(pallas_m2l_real_fused_supported())
    except Exception:
        return False


def _m2l_real_batch_kernel_fused_pallas(
    multipoles: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: str,
) -> Array:
    """Real-basis M2L via the FULLY-fused Pallas kernel (rotate+z-translate+rotate
    in one launch). Builds the real rotation blocks + radii from deltas.

    Notes
    -----
    **This lane's transverse gradient near ``rho == 0`` is covered in two pieces rather
    than one, and neither is optional.** It could not take the ``custom_jvp`` the pure-JAX
    lanes did: ``m2l_real_fused_pallas_cvjp`` is a ``custom_vjp``, and JAX refuses
    forward-mode through one ("can't apply forward-mode autodiff (jvp) to a custom_vjp
    function"), so a rule that differentiates the operator cannot wrap it. Instead:

    * :data:`~jaccpot.operators.m2l_real_rot_scale.m2l_real_fused_align_deltas` runs on
      ``deltas`` **before** the radius and both block stacks are built, so
      everything the kernel sees comes from a displacement whose unusable transverse
      tangent has already been removed;
    * :func:`~jaccpot.operators.m2l_real_rot_scale.make_m2l_real_fused_carry_axis_derivative`
      runs on the output and adds the analytic term back, computing the one operator
      application it needs with the pure-JAX twin the kernel's own ``custom_vjp`` already
      uses as its correctness reference.

    Neither differentiates the kernel, and both primals return their input unchanged with
    no arithmetic performed on it, so the forward pass is untouched -- not even in the sign
    of zero. Drop either piece and the gradient is wrong: without the withdrawal the
    analytic term lands on top of the polar route's contribution, and without the carrier
    this lane's on-axis ``d/dx`` and ``d/dy`` come back as exactly zero, measured **1.98**
    away from the pure-JAX lane.

    Asserted by
    ``tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient``,
    which runs the kernel's reference lowering on CPU (``interpret=True``; agreement
    2.7e-15) and the real Triton kernel where the hardware allows. What still wants a GPU
    is the ``interpret=False`` half of that test, the fully-fused reverse kernel under
    ``JACCPOT_FUSED_M2L_VJP=1``, and a ``bench/audit_reverse_residuals.py`` re-run --
    nothing here changes what the ``custom_vjp`` saves, but the linearised block
    construction around it now carries one extra select.

    Parameters
    ----------
    multipoles : Array
        Real multipole coefficients, one row per pair.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    m2l_impl : str
        Must be ``"rot_scale"``, as in the reference kernel. Static.

    Returns
    -------
    Array
        Real local contributions, equal to :func:`_m2l_real_batch_kernel`'s
        output -- this is an execution accelerator, not different mathematics.

    Raises
    ------
    ValueError
        If ``m2l_impl`` is anything but ``"rot_scale"``.
    """
    mode = str(m2l_impl).strip().lower()
    if mode != "rot_scale":
        raise ValueError("real-basis m2l_impl must be 'rot_scale'")
    from jaccpot.operators.m2l_real_rot_scale import (
        m2l_real_fused_align_deltas,
        real_rotation_blocks_from_z_local_batch,
        real_rotation_blocks_to_z_multipole_batch,
    )
    from jaccpot.pallas.m2l_real_fused import m2l_real_fused_pallas_cvjp

    # Everything the kernel sees is built from a displacement whose unusable transverse
    # tangent has already been withdrawn, so the radius and the two block stacks all
    # agree on where the band is; the carrier below then puts the analytic term back.
    # Splitting it this way is what lets a custom_vjp kernel sit in the middle -- JAX
    # cannot forward-differentiate one, so the usual single decorator does not apply.
    aligned = m2l_real_fused_align_deltas(deltas)
    r = jnp.linalg.norm(aligned, axis=1)
    bto = real_rotation_blocks_to_z_multipole_batch(
        aligned, order=order, dtype=multipoles.dtype
    )
    bfr = real_rotation_blocks_from_z_local_batch(
        aligned, order=order, dtype=multipoles.dtype
    )
    # custom_vjp wrapper (forward == raw kernel) so this fused path is also
    # differentiable; see the complex counterpart above.
    out = m2l_real_fused_pallas_cvjp(multipoles, bto, bfr, r, order, False, "triton")
    return _real_fused_carry_axis_derivative()(
        out, multipoles, deltas, bto, bfr, r, order=order
    )


def _apply_real_m2l(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    m2l_impl: Optional[str],
) -> Array:
    """Real-basis batched M2L: fully-fused Pallas kernel when enabled, else pure-JAX.

    When the fused-M2L Pallas flag is active, route through the single-launch fused
    kernel (rotate -> z-translate -> rotate-back on-chip), collapsing the per-pair
    JAX rotation launches. Otherwise the pure-JAX rot-scale path.

    Parameters
    ----------
    src_mult : Array
        Real multipole coefficients, one row per pair.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    m2l_impl : Optional[str]
        Real M2L implementation. ``Optional`` because the accumulators declare
        it so and pass it through unresolved; both kernels below then require
        ``"rot_scale"`` and raise otherwise, so ``None`` reaches a ValueError
        rather than a default.

    Returns
    -------
    Array
        Real local contributions. The two routes are numerically equivalent --
        the gate selects execution, not mathematics.
    """
    if _real_m2l_pallas_active():
        return _m2l_real_batch_kernel_fused_pallas(
            src_mult, deltas, order=order, m2l_impl=m2l_impl
        )
    return _m2l_real_batch_kernel(src_mult, deltas, order=order, m2l_impl=m2l_impl)


@jaxtyped(typechecker=beartype)
def _m2l_chunk_contributions(
    multip_packed: Float[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src_idx: Array,
    tgt_idx: Array,
    valid: Array,
    *,
    order: int,
    m2l_impl: Optional[str],
    out_dtype: Any,
) -> Array:
    """Gather the multipoles/centre displacements for one pair batch, apply the M2L.

    Deliberately takes the loop-invariant arrays plus **index vectors**, not
    pre-gathered values, so that a caller can wrap it in ``jax.checkpoint`` and
    have reverse mode retain only these inputs. ``lax.scan``'s partial-eval hoists
    scan-invariant residuals out of the loop, so ``multip_packed``/``centers`` are
    counted **once** rather than once per chunk, leaving only two integer index
    vectors and a mask stacked per chunk.

    That matters a lot. Un-rematerialized, the retained residual is the
    rotate-to-z / rotate-from-z blocks *and* their bilinear construction
    intermediates (``D = B_U @ Dz_beta @ B_U @ Dz_alpha`` is bilinear, so the
    partial products are residuals too, for both directions and both the padded
    and per-degree forms). Measured with ``bench/audit_reverse_residuals.py``:
    **28.7 kB per pair** (fp32, order 4) versus ~34 B per pair once
    rematerialized -- i.e. ~28.7 GB at N=200000, which is what made the reverse
    pass OOM there.

    The double-``where`` delta guard MUST stay inside this function. Remat re-runs
    exactly what is enclosed here, so hoisting the guard out would let the
    *recomputed* ``deltas`` collapse to zero on padded lanes and reintroduce the
    singular-radius NaN cotangent the guard exists to prevent.

    Parameters
    ----------
    multip_packed : Float[Array, 'nodes sh']
        Packed multipole coefficients for every node. Loop-invariant, so hoisted
        out of the scan and counted once.
    centers : Float[Array, 'nodes 3']
        Node centres. Loop-invariant on the same terms; the pair displacement is
        formed here rather than passed in, which is what keeps the guard inside
        the rematerialized region.
    src_idx : Array
        Source node index per pair in this chunk.
    tgt_idx : Array
        Target node index per pair in this chunk.
    valid : Array
        Validity mask over the chunk. Padded lanes collapse to
        ``src_idx == tgt_idx == 0`` and are zeroed by the double ``where``.
    order : int
        Expansion order ``p``. Static.
    m2l_impl : Optional[str]
        Real M2L implementation.
    out_dtype : Any
        Dtype to cast the contributions to before accumulation.

    Returns
    -------
    Array
        Per-pair M2L contributions for the chunk, zero on invalid lanes.
    """
    src_mult = multip_packed[src_idx]
    deltas = centers[tgt_idx] - centers[src_idx]
    # Invalid/padded pairs collapse to src_idx == tgt_idx == 0, giving a zero
    # displacement whose M2L rotate-to-z ``norm(delta)`` has a 0/0 (NaN)
    # reverse-mode cotangent. Substitute a nonzero delta BEFORE the apply; the
    # contribution is masked to 0 by the caller, so the forward is unchanged.
    deltas = jnp.where(valid[:, None], deltas, jnp.ones_like(deltas))
    return _apply_real_m2l(src_mult, deltas, order=order, m2l_impl=m2l_impl).astype(
        out_dtype
    )


@partial(
    jax.jit,
    static_argnames=("order", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_m2l_fullbatch(
    locals_coeffs: Float[Array, "nodes sh"],
    multip_packed: Float[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    total_nodes: int,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions in one full interaction batch.

    Every discriminator is a ``static_argname``, so XLA specialises the jit per
    configuration.

    One of the two accumulators (module docstring): it takes raw source/target
    indices and applies the whole list in one batch.

    Parameters
    ----------
    locals_coeffs : Float[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Float[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres.
    src : Array
        Source node index per pair. Negative entries are treated as padding.
    tgt : Array
        Target node index per pair, same convention.
    active_pair_count : Array
        How many leading entries are live. Traced, not static -- the arrays are
        allocated to a fixed capacity and this says how much of it is real.
    order : int
        Expansion order ``p``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    m2l_impl : Optional[str]
        Real M2L implementation.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    idx = jnp.arange(src.shape[0], dtype=INDEX_DTYPE)
    valid = (idx < active_pair_count) & (src >= 0) & (tgt >= 0)
    safe_src = jnp.where(valid, src, 0)
    safe_tgt = jnp.where(valid, tgt, 0)
    # Shares the gather + double-where + apply with the chunked scan below via
    # ``_m2l_chunk_contributions`` so the two paths cannot drift apart (the same
    # reasoning as ``_pair_accel_pair_terms`` in the near field). NOT wrapped in
    # ``jax.checkpoint`` here: fullbatch only runs when the whole list fits in one
    # chunk (``pair_count <= chunk_size``, the caller's switch), where the retained
    # blocks are small, so remat would buy nothing and would perturb the small-N
    # forward schedule.
    contribs = _m2l_chunk_contributions(
        multip_packed,
        centers,
        safe_src,
        safe_tgt,
        valid,
        order=order,
        m2l_impl=m2l_impl,
        out_dtype=locals_coeffs.dtype,
    )
    contribs = jnp.where(valid[:, None], contribs, 0)
    return locals_coeffs + jax.ops.segment_sum(contribs, safe_tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "m2l_impl",
        "total_nodes",
        "chunk_size",
    ),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_m2l_chunked_scan(
    locals_coeffs: Float[Array, "nodes sh"],
    multip_packed: Float[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions with chunked scan reduction.

    One of the two accumulators (module docstring): the bounded-memory form. Its scan body is rematerialized, which is what keeps the
    reverse pass from retaining the rotation blocks per chunk -- see the comment
    on the ``jax.checkpoint`` below for the measured figures.

    Parameters
    ----------
    locals_coeffs : Float[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Float[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres.
    src : Array
        Source node index per pair; negative entries are padding.
    tgt : Array
        Target node index per pair, same convention.
    active_pair_count : Array
        How many leading entries are live. Traced; chunks entirely past it are
        skipped by a ``lax.cond`` rather than masked.
    order : int
        Expansion order ``p``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Pairs per scan step; sets peak memory. Static.
    m2l_impl : Optional[str]
        Real M2L implementation.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    pair_count = src.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    # Rematerialize the M2L apply. Reverse mode retains one residual set per scan
    # iteration, and un-rematerialized that residual is the rotation blocks plus
    # their bilinear construction intermediates -- 28.7 kB per pair (fp32, p=4),
    # i.e. ~28.7 GB at N=200000, which is what made the reverse pass OOM there.
    # Checkpointing drops it to ~34 B per pair: only two integer index vectors and
    # a mask are stacked, while the scan-invariant multipoles/centres are hoisted
    # out and counted once (see ``_m2l_chunk_contributions``).
    #
    # The wrapper sits OUTSIDE ``_apply_real_m2l``, which also fixes the fused-Pallas
    # M2L lane: that kernel's ``custom_vjp`` saves the blocks as its residual, and
    # being inside the recomputed region means that residual is discarded too.
    # Statics are captured by closure rather than passed through
    # ``jax.checkpoint``, which flattens (args, kwargs) and would trace them.
    def _m2l_chunk_apply(multip, cent, src_idx, tgt_idx, valid_mask):
        return _m2l_chunk_contributions(
            multip,
            cent,
            src_idx,
            tgt_idx,
            valid_mask,
            order=order,
            m2l_impl=m2l_impl,
            out_dtype=locals_coeffs.dtype,
        )

    _m2l_chunk = jax.checkpoint(_m2l_chunk_apply)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        def active_chunk(accum: Array) -> Array:
            offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
            idx = start_idx + offset
            valid = idx < pair_count
            safe_idx = jnp.where(valid, idx, 0)
            src_chunk_raw = src[safe_idx]
            tgt_chunk_raw = tgt[safe_idx]
            valid = (
                valid
                & (idx < active_pair_count)
                & (src_chunk_raw >= 0)
                & (tgt_chunk_raw >= 0)
            )
            src_chunk = jnp.where(valid, src_chunk_raw, 0)
            tgt_chunk = jnp.where(valid, tgt_chunk_raw, 0)
            # Gather + double-where guard + M2L apply, rematerialized (see above).
            contribs = _m2l_chunk(multip_packed, centers, src_chunk, tgt_chunk, valid)
            return _chunk_segment_scatter_add(
                accum,
                contribs,
                tgt_chunk,
                valid,
                chunk_size=chunk_size,
            )

        local_accum = jax.lax.cond(
            start_idx < active_pair_count,
            active_chunk,
            lambda accum: accum,
            local_accum,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum

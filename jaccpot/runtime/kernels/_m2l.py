"""The M2L apply-and-accumulate seam: every basis, every batching, both gates.

THIS MODULE IS ONE UNIT ON PURPOSE. ``_apply_m2l`` dispatches on ``basis_mode``
to ``_apply_real_m2l`` / ``_apply_complex_m2l``, each of which consults its
``*_pallas_active`` gate and routes to either the reference batch kernel or the
fused Pallas twin. ARCHITECTURE §10's invariant is that **every discriminator on
that path is a static argument** -- ``order``, ``rotation``, ``m2l_impl``,
``basis_mode`` -- so the whole dispatch compiles to one branch-free jaxpr per
configuration. Splitting *within* it would break the argument that the invariant
holds, which is why the audit says this seam moves whole.

EQUIVALENCES THAT MUST HOLD (NUMERICS_AND_JAX §1, asserted in
``tests/unit/operators/test_m2l_{real,complex}_fused_pallas.py``):

* ``_m2l_real_batch_kernel_fused_pallas`` == ``_m2l_real_batch_kernel`` (rot-scale
  reference), and
* ``_m2l_complex_batch_kernel_fused_pallas`` == ``_m2l_complex_batch_kernel``
  (solidfmm reference).

The Pallas kernels are execution accelerators, not different mathematics.

THE FOUR ACCUMULATORS are four batchings of the same sum: full-batch, chunked
scan, class-grouped and class-major. They must agree to reassociation only --
G.11 was a 60x accuracy gap between two of them, caused by ``pair_grouped``
gathering rotations with class ids from the wrong ordering, and it is exactly the
kind of defect that hides in a family of near-duplicates. Do not "unify" them
without re-running the grouped-mode goldens.

Split out of ``core.py`` (Tier 1.6, A.9 seam 2); every function body is unchanged.
"""

from __future__ import annotations

import functools
from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from jaxtyping import Array, Float, Inexact, Int, jaxtyped
from yggdrax.grouped_interactions import (
    GroupedInteractionBuffers,
)

from jaccpot.operators.complex_ops import (
    complex_rotation_blocks_from_z_solidfmm_batch,
    complex_rotation_blocks_to_z_solidfmm_batch,
    m2l_complex_fused_align_deltas,
    m2l_complex_reference_batch,
    m2l_complex_reference_batch_cached_blocks,
    make_m2l_complex_fused_carry_axis_derivative,
)
from jaccpot.operators.m2l_real_rot_scale import (
    m2l_rot_scale_real_batch,
    m2l_rot_scale_real_batch_cached_blocks,
    make_m2l_real_fused_carry_axis_derivative,
    real_rotation_blocks_from_z_local_batch,
    real_rotation_blocks_to_z_multipole_batch,
)
from jaccpot.runtime.grad_options import fused_m2l_pallas_enabled

# The two fused-M2L transverse-tangent carriers, built ONCE each and cached.
#
# Cached rather than module-level for two independent reasons. The pure-JAX twins
# each carrier needs live in `jaccpot.pallas.*`, and importing that at module scope
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


@functools.lru_cache(maxsize=1)
def _complex_fused_carry_axis_derivative() -> Any:
    """The complex fused M2L's tangent carrier, built once.

    Returns
    -------
    Any
        ``carry(out, coeffs, delta, blocks_to_z, blocks_from_z, radii, order=p)``.
    """
    from jaccpot.pallas.m2l_complex_fused import m2l_complex_fused_jax

    return make_m2l_complex_fused_carry_axis_derivative(m2l_complex_fused_jax)


from ..dtypes import INDEX_DTYPE
from ..fmm_caches import (
    _M2L_FULLBATCH_MAX_PAIRS,
    _grouped_operator_cache_get,
    _grouped_operator_cache_key,
    _grouped_operator_cache_put,
    _grouped_segment_cache_get,
    _grouped_segment_cache_key,
    _grouped_segment_cache_put,
)

__all__: list[str] = []


@partial(jax.jit, static_argnames=("order", "rotation"))
def _m2l_complex_batch_kernel(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Vectorized complex-basis M2L kernel for one interaction batch.

    The solidfmm reference: rotate to z, translate along z, rotate back. This is
    the definition the fused Pallas twin is asserted equal to, so it is the one
    to change if the mathematics ever must.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]``, one row per pair.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    rotation : str
        Rotation convention; ``"solidfmm"``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``, aligned with ``src_mult``.
    """
    return m2l_complex_reference_batch(
        src_mult,
        deltas,
        order=order,
        rotation=rotation,
    )


@partial(jax.jit, static_argnames=("order",))
def _m2l_complex_batch_cached_kernel(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
) -> Array:
    """Vectorized complex M2L kernel using precomputed rotation blocks.

    Same translation as :func:`_m2l_complex_batch_kernel`, with the rotation
    blocks supplied rather than rebuilt per pair. That is the entire point of the
    grouped path: pairs sharing a displacement class share their blocks.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]``.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    blocks_to_z : Array
        Per-pair multipole world-to-z rotation blocks.
    blocks_from_z : Array
        Per-pair local z-to-world rotation blocks.
    order : int
        Expansion order ``p``. Static under ``jit``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.
    """
    return m2l_complex_reference_batch_cached_blocks(
        src_mult,
        deltas,
        blocks_to_z,
        blocks_from_z,
        order=order,
    )


def _m2l_cached_kernel_dispatch(
    src_mult: Array,
    deltas: Array,
    blocks_to_z: Array,
    blocks_from_z: Array,
    *,
    order: int,
    basis_mode: str,
) -> Array:
    """Apply precomputed rotation blocks in the complex or real basis.

    ``basis_mode`` is a Python string (static under jit), so this branches at
    trace time. The real branch uses the Dehnen no-sqrt2 cached kernel; the
    complex branch is unchanged.

    The blocks must have been built for the same basis this dispatches on --
    neither kernel checks. See :func:`_accumulate_solidfmm_m2l_grouped` for what
    a mismatch costs.

    Parameters
    ----------
    src_mult : Array
        Multipole coefficients in whichever basis ``basis_mode`` names.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    blocks_to_z : Array
        Multipole world-to-z rotation blocks, in the same basis.
    blocks_from_z : Array
        Local z-to-world rotation blocks, in the same basis.
    order : int
        Expansion order ``p``. Static.
    basis_mode : str
        ``"real"`` or anything else for complex. Static.

    Returns
    -------
    Array
        Local contributions, packed as the basis requires.
    """
    if str(basis_mode).strip().lower() == "real":
        return m2l_rot_scale_real_batch_cached_blocks(
            src_mult, deltas, blocks_to_z, blocks_from_z, order=order
        )
    return _m2l_complex_batch_cached_kernel(
        src_mult, deltas, blocks_to_z, blocks_from_z, order=order
    )


# THE TWO AXES THIS MODULE NEEDED, AND WHY IT IS NOT A WHOLE-MODULE PASS.
#
# `bench/annotation_pilot.py` re-recorded on 2026-09-03 puts this module last on rate --
# 23 silent acceptances of 286 perturbations, 8%, on 18 measured functions -- which is the
# August verdict confirmed: it is mostly validated already and converting all 92 bare
# parameters would be effort spent where nothing is wrong. But the 23 are NOT spread
# evenly, and where they sit is the argument for the two axes below.
#
#     _rotation_blocks_for_grouped_classes           5   class_keys accepted ALL FOUR
#                                                        perturbations; class_deltas the
#                                                        misalignment against it
#     _accumulate_solidfmm_m2l_grouped_class_major   4   locals_coeffs, all four
#     _pair_class_ids_from_offsets                   3
#     _accumulate_solidfmm_m2l_grouped_chunked_scan  2
#     _chunk_segment_scatter_add                     2
#     _m2l_chunk_contributions                       2
#
# `classes` is G.11 expressed as a shape. The module docstring records G.11 as a 60x
# accuracy gap between two of the four accumulators, caused by `pair_grouped` gathering
# rotations with class ids from the WRONG ORDERING. Here `num_classes` is read from
# `class_deltas` while `class_keys` is the cache identity, so a `class_keys` of a different
# length keys the rotation cache on a different class count than the blocks it returns --
# and nothing downstream can tell. Eight recorded calls, class axis shared in every one, at
# three distinct extents: 7, 774 and 2487. The trailing 5 is a literal because the key width
# was 5 in all eight, across three problem sizes and two orders; if yggdrax ever changes the
# key layout this should fail loudly rather than mis-key a cache silently.
#
# `nodes` and `sh` tie the three arrays every batching takes. This is the family the module
# docstring says not to unify -- four batchings of one sum -- so the annotation is the same
# on all of them, which is what makes them comparable. Evidence is the strongest in the
# module: ~50 recorded calls across the family share the leading axis of `locals_coeffs`,
# `multip_packed` and `centers` at EIGHT distinct extents (5, 8, 13, 63, 127, 255, 511,
# 1023), and the two packed arrays share the trailing axis at five (4, 9, 16, 25, 81).
#
# `Inexact` and not `Float` on the packed pair: `basis_mode="complex"` is a live lane and
# passes complex coefficients. Narrowing to `Float` is the mistake #293 made one module
# over, where a real-basis-only recording cost 27 CI failures.
#
# `class_offsets` in `_pair_class_ids_from_offsets` stays rank-only. It is
# `(num_classes + 1,)` and `classes` is bound by nothing else in that signature -- and even
# if it were, the parameter precedes anything that could bind it, which jaxtyping cannot
# evaluate. Rank alone still closes the extra-leading-axis case, which is what the pilot
# found; the length case is left open and named here rather than annotated wrongly.
#
# The decorators go INSIDE `jax.jit` on the jitted accumulators, per the house order, so
# beartype runs once per trace rather than per call on the production M2L path.
@jaxtyped(typechecker=beartype)
def _rotation_blocks_for_grouped_classes(
    *,
    order: int,
    rotation: str,
    class_keys: Int[Array, "classes 5"],
    class_deltas: Float[Array, "classes 3"],
    dtype: jnp.dtype,
    basis_mode: str = "complex",
) -> tuple[Array, Array]:
    """Resolve rotation blocks for all grouped classes with cache reuse.

    For ``basis_mode == "real"`` the Dehnen no-sqrt2 real rotation blocks are
    built (multipole world->z and local z->world) and the ``rotation`` argument
    is ignored (the real path has a single rotation construction).

    Parameters
    ----------
    order : int
        Expansion order ``p``; sets the block shape ``(p+1, 2p+1, 2p+1)``.
    rotation : str
        Rotation convention for the complex path. Ignored when ``basis_mode`` is
        ``"real"``.
    class_keys : Int[Array, 'classes 5']
        Displacement-class keys, used as the cache identity.
    class_deltas : Float[Array, 'classes 3']
        One representative displacement per class ``[C, 3]``. Its length is the
        class count.
    dtype : jnp.dtype
        Dtype to build the blocks in; take it from the multipoles so the cached
        kernel does not have to promote.
    basis_mode : str
        ``"real"`` or ``"complex"``. Decides which rotation construction is used,
        and therefore which cached kernel the blocks may be fed to.

    Returns
    -------
    tuple[Array, Array]
        ``(blocks_to_classes, blocks_from_classes)``, indexed by class id --
        multipole world-to-z first, local z-to-world second.

    Raises
    ------
    ValueError
        If the rotation convention is not one this path can build blocks for.
    """
    num_classes = int(class_deltas.shape[0])
    max_m = 2 * int(order) + 1
    empty_shape = (0, int(order) + 1, max_m, max_m)
    if num_classes == 0:
        empty = jnp.zeros(empty_shape, dtype=dtype)
        return empty, empty

    real_basis = str(basis_mode).strip().lower() == "real"
    cache_key = _grouped_operator_cache_key(
        order=order,
        rotation=("real" if real_basis else rotation),
        dtype=dtype,
        class_keys=class_keys,
        class_deltas=class_deltas,
    )
    if cache_key is not None:
        cached = _grouped_operator_cache_get(cache_key)
        if cached is not None:
            return cached

    deltas = jnp.asarray(class_deltas)
    if real_basis:
        blocks_to = real_rotation_blocks_to_z_multipole_batch(
            deltas, order=order, dtype=dtype
        )
        blocks_from = real_rotation_blocks_from_z_local_batch(
            deltas, order=order, dtype=dtype
        )
        if cache_key is not None:
            _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
        return blocks_to, blocks_from

    if rotation == "solidfmm":
        blocks_to = complex_rotation_blocks_to_z_solidfmm_batch(
            deltas,
            order=order,
            basis="multipole",
            dtype=dtype,
        )
        blocks_from = complex_rotation_blocks_from_z_solidfmm_batch(
            deltas,
            order=order,
            basis="local",
            dtype=dtype,
        )
    else:
        raise ValueError(
            "grouped operator cache currently supports rotation='solidfmm'"
        )
    if cache_key is not None:
        _grouped_operator_cache_put(cache_key, (blocks_to, blocks_from))
    return blocks_to, blocks_from


def _chunk_segment_scatter_add(
    local_accum: Array,
    contribs: Array,
    tgt_chunk: Array,
    valid: Array,
    *,
    chunk_size: int,
) -> Array:
    """Reduce one fixed-width chunk by target index and scatter-add into locals.

    Sorts the chunk by target so that contributions to the same target become a
    contiguous segment, reduces within segments, then scatters once. Invalid
    slots are given the maximum index so they sort to the end and fall outside
    the scatter.

    The sort makes the summation order a deterministic function of the target
    indices rather than of the pair order, which is what keeps the four
    accumulators agreeing to reassociation.

    Parameters
    ----------
    local_accum : Array
        Local coefficient accumulator to add into.
    contribs : Array
        Per-pair M2L contributions for this chunk.
    tgt_chunk : Array
        Target node index per pair in the chunk.
    valid : Array
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
    new_group = jnp.concatenate(
        (
            jnp.asarray([True], dtype=bool),
            sorted_keys[1:] != sorted_keys[:-1],
        ),
        axis=0,
    )
    group_ids = jnp.cumsum(new_group.astype(INDEX_DTYPE)) - jnp.asarray(
        1,
        dtype=INDEX_DTYPE,
    )
    reduced = jax.ops.segment_sum(contribs_sorted, group_ids, chunk_size)

    unique_targets = jnp.zeros((chunk_size,), dtype=INDEX_DTYPE)
    unique_targets = unique_targets.at[group_ids].set(tgt_sorted)
    unique_valid = jnp.zeros((chunk_size,), dtype=bool)
    unique_valid = unique_valid.at[group_ids].set(valid_sorted)
    safe_targets = jnp.where(unique_valid, unique_targets, 0)
    reduced = jnp.where(unique_valid[:, None], reduced, 0)
    return local_accum.at[safe_targets].add(reduced)


@jaxtyped(typechecker=beartype)
def _pair_class_ids_from_offsets(
    class_offsets: Int[Array, "_"], pair_indices: Int[Array, "_"]
) -> Array:
    """Class id of each pair, addressed in the class-sorted pair order.

    ``class_sources`` / ``class_targets`` are stored sorted by class, so the
    CSR offsets alone say which class a pair index belongs to: pair ``i`` is in
    class ``c`` when ``class_offsets[c] <= i < class_offsets[c + 1]``.

    This is deliberately not ``GroupedInteractionBuffers.class_ids``, which
    yggdrax stores in the *original* (unsorted) pair order -- see
    :func:`_accumulate_solidfmm_m2l_grouped`.

    Parameters
    ----------
    class_offsets : Int[Array, '_']
        CSR class boundaries, shape ``(num_classes + 1,)``, strictly increasing.
    pair_indices : Int[Array, '_']
        Indices into the class-sorted pair arrays.

    Returns
    -------
    Array
        Class id per entry of ``pair_indices``.
    """
    return jnp.searchsorted(class_offsets[1:], pair_indices, side="right").astype(
        INDEX_DTYPE
    )


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_solidfmm_m2l_grouped_chunked_scan(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src_sorted: Array,
    tgt_sorted: Array,
    class_offsets: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions via chunked scan.

    One of the four accumulators (module docstring). This is the grouped path's
    bounded-memory form: pairs are already class-sorted, and the scan walks them
    in fixed-width chunks so peak memory is set by ``chunk_size`` rather than by
    the pair count. Its full-batch twin is
    :func:`_accumulate_solidfmm_m2l_grouped_fullbatch`; they must agree to
    reassociation only.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator. Donated -- do not use the argument after
        the call.
    multip_packed : Inexact[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres; the pair displacement is the difference of two rows.
    src_sorted : Array
        Source node index per pair, in class-sorted order.
    tgt_sorted : Array
        Target node index per pair, in the same order.
    class_offsets : Array
        CSR class boundaries, ``(num_classes + 1,)``. The pair's class comes from
        these, not from ``class_ids`` -- see
        :func:`_pair_class_ids_from_offsets`.
    blocks_to_classes : Array
        Per-class multipole world-to-z rotation blocks.
    blocks_from_classes : Array
        Per-class local z-to-world rotation blocks.
    order : int
        Expansion order ``p``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Pairs per scan step. Static.
    basis_mode : str
        ``"real"`` or ``"complex"``. Static, and must match the basis the blocks
        were built in.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    pair_count = src_sorted.shape[0]
    starts = jnp.arange(0, pair_count, chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, start_idx: Array) -> tuple[Array, None]:
        offset = jnp.arange(chunk_size, dtype=INDEX_DTYPE)
        idx = start_idx + offset
        valid = idx < pair_count
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        cls_chunk = _pair_class_ids_from_offsets(class_offsets, safe_idx)
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]
        blocks_to = blocks_to_classes[cls_chunk]
        blocks_from = blocks_from_classes[cls_chunk]

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        local_accum = _chunk_segment_scatter_add(
            local_accum,
            contribs,
            tgt_chunk,
            valid,
            chunk_size=chunk_size,
        )
        return local_accum, None

    local_accum, _ = jax.lax.scan(body, locals_coeffs, starts)
    return local_accum


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "basis_mode"),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_solidfmm_m2l_grouped_fullbatch(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src_sorted: Array,
    tgt_sorted: Array,
    class_offsets: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate grouped solidfmm M2L contributions in one full batch.

    One of the four accumulators (module docstring). The grouped path's unchunked
    form: every pair is translated at once, so there is no scan and no per-chunk
    scatter, at the cost of holding all contributions live.
    :func:`_accumulate_solidfmm_m2l_grouped` picks between this and the chunked
    twin on pair count.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres.
    src_sorted : Array
        Source node index per pair, class-sorted.
    tgt_sorted : Array
        Target node index per pair, same order.
    class_offsets : Array
        CSR class boundaries, ``(num_classes + 1,)``.
    blocks_to_classes : Array
        Per-class multipole world-to-z rotation blocks.
    blocks_from_classes : Array
        Per-class local z-to-world rotation blocks.
    order : int
        Expansion order ``p``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    basis_mode : str
        ``"real"`` or ``"complex"``. Static, and must match the blocks.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    src_mult = multip_packed[src_sorted]
    deltas = centers[tgt_sorted] - centers[src_sorted]
    class_ids_sorted = _pair_class_ids_from_offsets(
        class_offsets,
        jnp.arange(src_sorted.shape[0], dtype=INDEX_DTYPE),
    )
    blocks_to = blocks_to_classes[class_ids_sorted]
    blocks_from = blocks_from_classes[class_ids_sorted]
    contribs = _m2l_cached_kernel_dispatch(
        src_mult,
        deltas,
        blocks_to,
        blocks_from,
        order=order,
        basis_mode=basis_mode,
    ).astype(locals_coeffs.dtype)
    return locals_coeffs + jax.ops.segment_sum(contribs, tgt_sorted, total_nodes)


def _build_grouped_class_segments(
    grouped: GroupedInteractionBuffers,
    *,
    chunk_size: int,
) -> tuple[Array, Array, Array]:
    """Build compact class-major segment metadata for chunked execution.

    Cuts the class-sorted pair list into segments no wider than ``chunk_size``,
    each belonging to exactly one class. That single-class property is what lets
    the class-major accumulators gather one rotation block per segment rather
    than per pair.

    Cached on the class layout and chunk size, since the result depends on
    neither the multipoles nor the centres and a refresh at fixed topology can
    reuse it.

    Parameters
    ----------
    grouped : GroupedInteractionBuffers
        Grouped pair buffers; supplies the class offsets and targets.
    chunk_size : int
        Maximum segment width. A class wider than this is split across several
        segments.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(segment_starts, segment_lengths, segment_class_ids)``, one entry per
        segment.
    """
    cache_key = _grouped_segment_cache_key(
        class_offsets=grouped.class_offsets,
        class_targets=grouped.class_targets,
        chunk_size=int(chunk_size),
    )
    if cache_key is not None:
        cached = _grouped_segment_cache_get(cache_key)
        if cached is not None:
            return cached

    class_offsets = np.asarray(jax.device_get(grouped.class_offsets), dtype=np.int64)
    if class_offsets.size <= 1:
        empty = jnp.zeros((0,), dtype=INDEX_DTYPE)
        result = (empty, empty, empty)
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    starts: list[int] = []
    lengths: list[int] = []
    class_ids: list[int] = []
    for class_idx in range(class_offsets.shape[0] - 1):
        start = int(class_offsets[class_idx])
        end = int(class_offsets[class_idx + 1])
        while start < end:
            seg_len = min(int(chunk_size), end - start)
            starts.append(start)
            lengths.append(seg_len)
            class_ids.append(class_idx)
            start += seg_len

    if len(starts) == 0:
        result = (
            jnp.asarray(starts, dtype=INDEX_DTYPE),
            jnp.asarray(lengths, dtype=INDEX_DTYPE),
            jnp.asarray(class_ids, dtype=INDEX_DTYPE),
        )
        if cache_key is not None:
            _grouped_segment_cache_put(cache_key, result)
        return result

    result = (
        jnp.asarray(starts, dtype=INDEX_DTYPE),
        jnp.asarray(lengths, dtype=INDEX_DTYPE),
        jnp.asarray(class_ids, dtype=INDEX_DTYPE),
    )
    if cache_key is not None:
        _grouped_segment_cache_put(cache_key, result)
    return result


@partial(
    jax.jit,
    static_argnames=("order", "total_nodes", "chunk_size", "basis_mode"),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_solidfmm_m2l_class_major_chunked_scan(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src_sorted: Array,
    tgt_sorted: Array,
    segment_starts: Array,
    segment_lengths: Array,
    segment_class_ids: Array,
    blocks_to_classes: Array,
    blocks_from_classes: Array,
    *,
    order: int,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Accumulate class-major grouped M2L contributions via chunked scan.

    One of the four accumulators (module docstring). Where the grouped chunked
    scan walks fixed-width chunks that may straddle classes, this walks the
    segment table from :func:`_build_grouped_class_segments`, so every step has
    exactly one class and reads exactly one pair of rotation blocks.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres.
    src_sorted : Array
        Source node index per pair, class-sorted.
    tgt_sorted : Array
        Target node index per pair, same order.
    segment_starts : Array
        First pair index of each segment.
    segment_lengths : Array
        Pair count of each segment; at most ``chunk_size``.
    segment_class_ids : Array
        The one class id each segment belongs to.
    blocks_to_classes : Array
        Per-class multipole world-to-z rotation blocks.
    blocks_from_classes : Array
        Per-class local z-to-world rotation blocks.
    order : int
        Expansion order ``p``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Segment width bound; sets the padded per-step shape. Static.
    basis_mode : str
        ``"real"`` or ``"complex"``. Static, and must match the blocks.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    num_segments = segment_starts.shape[0]
    if num_segments == 0:
        return locals_coeffs

    offsets = jnp.arange(chunk_size, dtype=INDEX_DTYPE)

    def body(local_accum: Array, seg_idx: Array) -> tuple[Array, None]:
        start = segment_starts[seg_idx]
        seg_len = segment_lengths[seg_idx]
        cls = segment_class_ids[seg_idx]
        idx = start + offsets
        valid = offsets < seg_len
        safe_idx = jnp.where(valid, idx, 0)

        src_chunk = src_sorted[safe_idx]
        tgt_chunk = tgt_sorted[safe_idx]
        src_mult = multip_packed[src_chunk]
        deltas = centers[tgt_chunk] - centers[src_chunk]

        block_to = blocks_to_classes[cls]
        block_from = blocks_from_classes[cls]
        blocks_to = jnp.broadcast_to(block_to, (chunk_size,) + block_to.shape)
        blocks_from = jnp.broadcast_to(block_from, (chunk_size,) + block_from.shape)

        contribs = _m2l_cached_kernel_dispatch(
            src_mult,
            deltas,
            blocks_to,
            blocks_from,
            order=order,
            basis_mode=basis_mode,
        ).astype(locals_coeffs.dtype)
        contribs = jnp.where(valid[:, None], contribs, 0)
        masked_targets = jnp.where(valid, tgt_chunk, jnp.iinfo(INDEX_DTYPE).max)
        sort_idx = jnp.argsort(masked_targets)
        tgt_sorted_chunk = tgt_chunk[sort_idx]
        contribs_sorted = contribs[sort_idx]
        valid_sorted = valid[sort_idx]
        prev_targets = jnp.concatenate(
            [
                jnp.asarray([-1], dtype=INDEX_DTYPE),
                tgt_sorted_chunk[:-1],
            ]
        )
        group_starts = valid_sorted & (
            jnp.logical_not(jnp.roll(valid_sorted, 1))
            | (tgt_sorted_chunk != prev_targets)
        )
        group_ids = jnp.cumsum(group_starts.astype(INDEX_DTYPE)) - 1
        safe_group_ids = jnp.where(valid_sorted, group_ids, 0)
        reduced = jax.ops.segment_sum(contribs_sorted, safe_group_ids, chunk_size)
        unique_targets = jnp.where(valid_sorted, tgt_sorted_chunk, 0)
        return local_accum.at[unique_targets].add(reduced), None

    local_accum, _ = jax.lax.scan(
        body,
        locals_coeffs,
        jnp.arange(num_segments, dtype=INDEX_DTYPE),
    )
    return local_accum


@jaxtyped(typechecker=beartype)
def _accumulate_solidfmm_m2l_grouped_class_major(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    grouped: GroupedInteractionBuffers,
    grouped_segment_starts: Optional[Array],
    grouped_segment_lengths: Optional[Array],
    grouped_segment_class_ids: Optional[Array],
    grouped_segment_sort_permutation: Optional[Array],
    grouped_segment_group_ids: Optional[Array],
    grouped_segment_unique_targets: Optional[Array],
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Class-major grouped accumulation without per-pair operator gathers.

    One of the four accumulators (module docstring), and the entry point to the
    class-major pair: it builds or accepts the segment table, then hands off to
    :func:`_accumulate_solidfmm_m2l_class_major_chunked_scan`. Non-solidfmm
    rotations fall back to :func:`_accumulate_m2l_fullbatch`, which takes the
    sparse per-pair path.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
        Packed multipole coefficients for every node.
    centers : Float[Array, 'nodes 3']
        Node centres.
    grouped : GroupedInteractionBuffers
        Grouped pair buffers: class-sorted sources and targets, class offsets,
        keys and representative displacements.
    grouped_segment_starts : Optional[Array]
        Precomputed segment starts; ``None`` builds the table here.
    grouped_segment_lengths : Optional[Array]
        Precomputed segment lengths.
    grouped_segment_class_ids : Optional[Array]
        Precomputed per-segment class ids.
    grouped_segment_sort_permutation : Optional[Array]
        Accepted and immediately discarded. Kept in the signature so callers can
        pass a whole precomputed schedule without knowing which parts this
        accumulator happens to use.
    grouped_segment_group_ids : Optional[Array]
        Accepted and discarded, as above.
    grouped_segment_unique_targets : Optional[Array]
        Accepted and discarded, as above.
    order : int
        Expansion order ``p``. Static.
    rotation : str
        Rotation convention. Anything but ``"solidfmm"`` takes the sparse
        fallback.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Segment width bound. Static.
    basis_mode : str
        ``"real"`` or ``"complex"``. Static.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """
    del (
        grouped_segment_sort_permutation,
        grouped_segment_group_ids,
        grouped_segment_unique_targets,
    )

    if rotation not in ("solidfmm",):
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            basis_mode=basis_mode,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=jnp.asarray(grouped.class_keys, dtype=jnp.int32),
        class_deltas=jnp.asarray(grouped.class_displacements),
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    if (
        grouped_segment_starts is None
        or grouped_segment_lengths is None
        or grouped_segment_class_ids is None
    ):
        (
            segment_starts,
            segment_lengths,
            segment_class_ids,
        ) = _build_grouped_class_segments(
            grouped,
            chunk_size=int(chunk_size),
        )
    else:
        segment_starts = jnp.asarray(grouped_segment_starts, dtype=INDEX_DTYPE)
        segment_lengths = jnp.asarray(grouped_segment_lengths, dtype=INDEX_DTYPE)
        segment_class_ids = jnp.asarray(grouped_segment_class_ids, dtype=INDEX_DTYPE)
    return _accumulate_solidfmm_m2l_class_major_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        jnp.asarray(grouped.class_sources, dtype=INDEX_DTYPE),
        jnp.asarray(grouped.class_targets, dtype=INDEX_DTYPE),
        segment_starts,
        segment_lengths,
        segment_class_ids,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
    )


@jaxtyped(typechecker=beartype)
def _accumulate_solidfmm_m2l_grouped(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    grouped: GroupedInteractionBuffers,
    *,
    order: int,
    rotation: str,
    total_nodes: int,
    chunk_size: int,
    basis_mode: str = "complex",
) -> Array:
    """Grouped M2L accumulation using cached class blocks and pair chunking.

    The per-pair class id is derived from ``grouped.class_offsets`` rather than
    read from ``grouped.class_ids``: yggdrax stores ``class_sources`` /
    ``class_targets`` sorted by class but applies the inverse permutation to
    ``class_ids``, so the two are not co-indexed and gathering blocks with
    ``class_ids`` hands most pairs another class's rotation. This is the same
    class assignment the class-major scan reads out of its segment table.

    Builds the per-class rotation blocks once, then picks full-batch or chunked
    scan on pair count. Both calls must be handed the same ``basis_mode`` as the
    block construction: omitting it is silent, because both kernels default to
    ``"complex"`` and will consume real-dtype blocks happily. The result is wrong
    rather than an error -- 3.8e-01 relative between the two branches at order 4,
    pinned by ``tests/unit/runtime/test_grouped_m2l_basis_mode.py``.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
        Packed multipole coefficients for every node. Its dtype is what the
        rotation blocks are built in.
    centers : Float[Array, 'nodes 3']
        Node centres.
    grouped : GroupedInteractionBuffers
        Grouped pair buffers.
    order : int
        Expansion order ``p``. Static.
    rotation : str
        Rotation convention. Anything but ``"solidfmm"`` takes the sparse
        per-pair fallback rather than the grouped path.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Pairs per scan step, and half of the full-batch cutoff -- the other half
        being ``_M2L_FULLBATCH_MAX_PAIRS``. Static.
    basis_mode : str
        ``"real"`` or ``"complex"``. Static; see above for the cost of losing it.

    Returns
    -------
    Array
        The accumulated local coefficients.
    """

    if rotation not in ("solidfmm",):
        # Keep existing sparse path semantics for other conventions.
        src = grouped.class_sources
        tgt = grouped.class_targets
        return _accumulate_m2l_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src,
            tgt,
            jnp.asarray(src.shape[0], dtype=INDEX_DTYPE),
            order=order,
            basis_mode=basis_mode,
            rotation=rotation,
            total_nodes=total_nodes,
        )

    src_sorted = grouped.class_sources
    tgt_sorted = grouped.class_targets
    class_offsets = jnp.asarray(grouped.class_offsets, dtype=INDEX_DTYPE)
    class_keys = jnp.asarray(grouped.class_keys, dtype=jnp.int32)
    class_deltas = jnp.asarray(grouped.class_displacements)

    blocks_to_classes, blocks_from_classes = _rotation_blocks_for_grouped_classes(
        order=order,
        rotation=rotation,
        class_keys=class_keys,
        class_deltas=class_deltas,
        dtype=multip_packed.dtype,
        basis_mode=basis_mode,
    )
    # The branch below is a pure batching choice: both accumulators must be handed
    # the same ``basis_mode``, or the rotation blocks built above (real when
    # ``basis_mode == "real"``) get applied by the other basis' cached kernel.
    # Omitting it on either call is silent -- both kernels default to "complex" and
    # happily consume real-dtype blocks, so the result is wrong rather than an error:
    # measured 3.8e-01 relative between the two branches at order 4. Pinned by
    # ``tests/unit/runtime/test_grouped_m2l_basis_mode.py``.
    if int(src_sorted.shape[0]) <= min(int(chunk_size), _M2L_FULLBATCH_MAX_PAIRS):
        return _accumulate_solidfmm_m2l_grouped_fullbatch(
            locals_coeffs,
            multip_packed,
            centers,
            src_sorted,
            tgt_sorted,
            class_offsets,
            blocks_to_classes,
            blocks_from_classes,
            order=order,
            total_nodes=total_nodes,
            basis_mode=basis_mode,
        )
    return _accumulate_solidfmm_m2l_grouped_chunked_scan(
        locals_coeffs,
        multip_packed,
        centers,
        src_sorted,
        tgt_sorted,
        class_offsets,
        blocks_to_classes,
        blocks_from_classes,
        order=order,
        total_nodes=total_nodes,
        chunk_size=int(chunk_size),
        basis_mode=basis_mode,
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
    Ampere+ (sm_80) -- matching the complex gate. The z-core
    ``pallas_m2l_real_supported`` used previously only checks gpu/tpu, so it would
    route to Pallas on a pre-Ampere GPU where the Triton lowering fails.

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
        Real M2L implementation. ``Optional`` because :func:`_apply_m2l` declares
        it so and passes it through unresolved; both kernels below then require
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


def _fused_complex_m2l_pallas_active() -> bool:
    """Whether to route the complex-basis M2L through the fused Pallas kernel.

    Gated by ``JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS`` and the sm_80+ support
    check; falls back to the solidfmm reference batch on unsupported hardware.
    Evaluated at trace time; the flag does not change within a compiled run.

    Returns
    -------
    bool
        ``True`` when the flag is set and an Ampere+ GPU is available.
    """
    if not fused_m2l_pallas_enabled():
        return False
    try:
        from jaccpot.pallas.m2l_complex_fused import (
            pallas_m2l_complex_fused_supported,
        )

        return bool(pallas_m2l_complex_fused_supported())
    except Exception:
        return False


def _m2l_complex_batch_kernel_fused_pallas(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
) -> Array:
    """Complex-basis M2L via the fully-fused Pallas kernel.

    Adapter over solidfmm: solidfmm is the sole rotation strategy, and it already
    materialises the block-diagonal rotate-to-z / rotate-from-z matrices the fused
    kernel consumes (``complex_rotation_blocks_*_z_solidfmm_batch``, padded to
    ``[N, p+1, 2p+1, 2p+1]``). This builds those blocks plus the pair radii and
    hands them to the kernel, which keeps the rotate -> z-translate -> rotate-back
    intermediates on-chip. Numerically equivalent to ``_m2l_complex_batch_kernel``
    (the solidfmm reference); the kernel is purely an execution accelerator.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]`` for each pair.
    deltas : Array
        Target-minus-source center displacements ``[N, 3]``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.

    Notes
    -----
    **This lane's transverse gradient near ``rho == 0`` is covered in two pieces rather
    than one, and neither is optional** -- the same shape as the real fused lane, and for
    the same reason: ``m2l_complex_fused_pallas_cvjp`` is a ``custom_vjp``, and JAX
    refuses forward-mode through one, so a rule that differentiates the operator cannot
    wrap it. Instead
    :data:`~jaccpot.operators.complex_ops.m2l_complex_fused_align_deltas` runs on
    ``deltas`` **before** the radius and both block stacks are built, and
    :func:`~jaccpot.operators.complex_ops.make_m2l_complex_fused_carry_axis_derivative` runs on
    the output and adds the analytic term back.

    The withdrawal alone was already in force here, because
    ``_complex_rotation_blocks_{to,from}_z_solidfmm_padded`` carry
    ``without_unresolvable_transverse_jvp`` for the cached-blocks lane's sake -- and half
    the pair is worse than neither half. Without the carrier this lane's on-axis ``d/dx``
    and ``d/dy`` came back exactly zero, measured **5.1e-01** from the pure-JAX reference
    batch, where before any of the G.10 work the two agreed to 6.7e-16. Asserted by
    ``tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative``,
    which differentiates *this function* with respect to ``deltas`` --
    ``test_m2l_complex_fused_pallas_custom_vjp_matches_twin`` cannot see it, because it
    differentiates the kernel's four inputs and never the displacement.
    """
    from jaccpot.pallas.m2l_complex_fused import m2l_complex_fused_pallas_cvjp

    # Everything the kernel sees is built from a displacement whose unusable transverse
    # tangent has already been withdrawn, so the radius and both block stacks agree on
    # where the band is; the carrier below then puts the analytic term back.
    aligned = m2l_complex_fused_align_deltas(deltas)
    r = jnp.sqrt(jnp.sum(aligned * aligned, axis=-1))
    blocks_to_z = complex_rotation_blocks_to_z_solidfmm_batch(
        aligned,
        order=order,
        basis="multipole",
        dtype=src_mult.dtype,
    )
    blocks_from_z = complex_rotation_blocks_from_z_solidfmm_batch(
        aligned,
        order=order,
        basis="local",
        dtype=src_mult.dtype,
    )
    # Route through the custom_vjp wrapper (not the raw kernel): the forward is
    # byte-identical to m2l_complex_fused_pallas, but the wrapper carries the
    # reverse rule (autodiff of the pure-jnp twin) so this fused path is also
    # differentiable -- required for FMMEngine.differentiable_accelerations
    # to run the fast lane. interpret=False, backend="triton" (the runtime always
    # runs the real Pallas GPU kernel here).
    out = m2l_complex_fused_pallas_cvjp(
        src_mult, blocks_to_z, blocks_from_z, r, order, False, "triton"
    )
    return _complex_fused_carry_axis_derivative()(
        out, src_mult, deltas, blocks_to_z, blocks_from_z, r, order=order
    )


def _apply_complex_m2l(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    rotation: str,
) -> Array:
    """Complex-basis batched M2L: fused Pallas kernel when enabled, else solidfmm.

    When the fused-M2L Pallas flag is active (and the GPU is Ampere+), route
    through the single-launch fused kernel fed by solidfmm rotation blocks.
    Otherwise use the default solidfmm rotate/z-translate/rotate-back reference
    batch. Both paths are numerically equivalent.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]`` for each pair.
    deltas : Array
        Target-minus-source center displacements ``[N, 3]``.
    order : int
        Expansion order ``p``.
    rotation : str
        Rotation strategy; must be ``"solidfmm"``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.
    """
    if _fused_complex_m2l_pallas_active():
        return _m2l_complex_batch_kernel_fused_pallas(src_mult, deltas, order=order)
    return _m2l_complex_batch_kernel(src_mult, deltas, order=order, rotation=rotation)


def _apply_m2l(
    src_mult: Array,
    deltas: Array,
    *,
    order: int,
    basis_mode: str,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Basis-dispatched batched M2L apply seam.

    ``basis_mode`` is a static discriminator, so XLA specialises each branch to
    the exact HLO of the corresponding single-basis kernel. Real basis routes
    through :func:`_apply_real_m2l` (``m2l_impl``); solidfmm/complex through
    :func:`_apply_complex_m2l` (``rotation``).

    Parameters
    ----------
    src_mult : Array
        Multipole coefficients in whichever basis ``basis_mode`` names, one row
        per pair.
    deltas : Array
        Target-minus-source centre displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static.
    basis_mode : str
        ``"real"`` selects the real branch; anything else the complex one.
        Static.
    rotation : Optional[str]
        Rotation convention. Read by the complex branch, ignored by the real one.
    m2l_impl : Optional[str]
        Real M2L implementation. Read by the real branch, ignored by the complex
        one.

    Returns
    -------
    Array
        Local contributions, packed as the selected basis requires.
    """
    if str(basis_mode).strip().lower() == "real":
        return _apply_real_m2l(src_mult, deltas, order=order, m2l_impl=m2l_impl)
    return _apply_complex_m2l(src_mult, deltas, order=order, rotation=rotation)


@jaxtyped(typechecker=beartype)
def _m2l_chunk_contributions(
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src_idx: Array,
    tgt_idx: Array,
    valid: Array,
    *,
    order: int,
    basis_mode: str,
    rotation: Optional[str],
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
    multip_packed : Inexact[Array, 'nodes sh']
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
    basis_mode : str
        ``"real"`` or ``"complex"``. Static.
    rotation : Optional[str]
        Rotation convention; complex branch only.
    m2l_impl : Optional[str]
        Real M2L implementation; real branch only.
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
    return _apply_m2l(
        src_mult,
        deltas,
        order=order,
        basis_mode=basis_mode,
        rotation=rotation,
        m2l_impl=m2l_impl,
    ).astype(out_dtype)


@partial(
    jax.jit,
    static_argnames=("order", "basis_mode", "rotation", "m2l_impl", "total_nodes"),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_m2l_fullbatch(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    basis_mode: str,
    total_nodes: int,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions in one full interaction batch (both bases).

    Unifies the former ``_accumulate_{solidfmm,real}_m2l_fullbatch`` behind the
    static ``basis_mode`` seam. Numerics-preserving: every discriminator is a
    ``static_argname`` so XLA specialises the merged jit per basis to the exact
    HLO each single-basis kernel produced.

    One of the four accumulators (module docstring), and the sparse per-pair one:
    unlike the grouped pair it takes raw source/target indices with no class
    structure, so it is also where non-solidfmm rotations end up.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
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
    basis_mode : str
        ``"real"`` or ``"complex"``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    rotation : Optional[str]
        Rotation convention; used by the complex branch only.
    m2l_impl : Optional[str]
        Real M2L implementation; used by the real branch only.

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
    # ``jax.checkpoint`` here: fullbatch only runs at pair_count <=
    # _M2L_FULLBATCH_MAX_PAIRS, where the retained blocks are tens of MB, so remat
    # would buy nothing and would perturb the small-N forward schedule.
    contribs = _m2l_chunk_contributions(
        multip_packed,
        centers,
        safe_src,
        safe_tgt,
        valid,
        order=order,
        basis_mode=basis_mode,
        rotation=rotation,
        m2l_impl=m2l_impl,
        out_dtype=locals_coeffs.dtype,
    )
    contribs = jnp.where(valid[:, None], contribs, 0)
    return locals_coeffs + jax.ops.segment_sum(contribs, safe_tgt, total_nodes)


@partial(
    jax.jit,
    static_argnames=(
        "order",
        "basis_mode",
        "rotation",
        "m2l_impl",
        "total_nodes",
        "chunk_size",
    ),
    donate_argnums=(0,),
)
@jaxtyped(typechecker=beartype)
def _accumulate_m2l_chunked_scan(
    locals_coeffs: Inexact[Array, "nodes sh"],
    multip_packed: Inexact[Array, "nodes sh"],
    centers: Float[Array, "nodes 3"],
    src: Array,
    tgt: Array,
    active_pair_count: Array,
    *,
    order: int,
    basis_mode: str,
    total_nodes: int,
    chunk_size: int,
    rotation: Optional[str] = None,
    m2l_impl: Optional[str] = None,
) -> Array:
    """Accumulate M2L contributions with chunked scan reduction (both bases).

    Unifies the former ``_accumulate_{solidfmm,real}_m2l_chunked_scan`` behind
    the static ``basis_mode`` seam; numerics-preserving (identical HLO per
    basis, single shared ``lax.scan`` body).

    One of the four accumulators (module docstring): the sparse path's
    bounded-memory form. Its scan body is rematerialized, which is what keeps the
    reverse pass from retaining the rotation blocks per chunk -- see the comment
    on the ``jax.checkpoint`` below for the measured figures.

    Parameters
    ----------
    locals_coeffs : Inexact[Array, 'nodes sh']
        Local coefficient accumulator.
    multip_packed : Inexact[Array, 'nodes sh']
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
    basis_mode : str
        ``"real"`` or ``"complex"``. Static.
    total_nodes : int
        Node count, sizing the accumulator. Static.
    chunk_size : int
        Pairs per scan step; sets peak memory. Static.
    rotation : Optional[str]
        Rotation convention; complex branch only.
    m2l_impl : Optional[str]
        Real M2L implementation; real branch only.

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
    # The wrapper sits OUTSIDE ``_apply_m2l``, which also fixes the fused-Pallas
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
            basis_mode=basis_mode,
            rotation=rotation,
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

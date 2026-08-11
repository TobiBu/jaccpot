"""``basis_mode`` survives the grouped-M2L batching branch.

:func:`_accumulate_solidfmm_m2l_grouped` picks between two accumulators purely on pair
count -- ``_accumulate_solidfmm_m2l_grouped_fullbatch`` when the batch fits in one
``segment_sum``, ``_accumulate_solidfmm_m2l_grouped_chunked_scan`` otherwise. That is a
batching decision, not a numerical one, so the two branches must agree in every basis.

WHY THIS FILE EXISTS. The fullbatch call site once omitted ``basis_mode``, so that
kernel fell back to its ``"complex"`` default while ``_rotation_blocks_for_grouped_classes``
had already built *real* (Dehnen no-sqrt2) rotation blocks. Nothing raises: the complex
cached kernel accepts real-dtype blocks, so it applies the wrong algebra and returns
wrong numbers. Measured on the fixture below at ``order=4``: in the real basis the
fullbatch branch came out ``3.8e-01`` relative off the chunked one, against a
reassociation-only gap of ``8e-17``.

The public API cannot reach the broken combination today -- ``basis="real"`` with
``grouped_interactions=True`` raises in ``prepare_upward_sweep`` before any M2L runs,
because grouped classification needs AABB expansion centres while the native real-basis
upward sweep accepts ``center_mode="com"`` only. So these are unit tests against the
accumulators directly; no end-to-end path would catch a regression here.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from yggdrax.grouped_interactions import GroupedInteractionBuffers

from jaccpot.operators.real_harmonics import sh_size
from jaccpot.runtime.kernels.core import (
    _accumulate_solidfmm_m2l_grouped,
    _accumulate_solidfmm_m2l_grouped_chunked_scan,
    _accumulate_solidfmm_m2l_grouped_fullbatch,
    _rotation_blocks_for_grouped_classes,
)

#: Expansion order used throughout. High enough that the real and complex cached
#: kernels cannot coincide by accident, low enough to compile quickly.
_ORDER = 4

# Geometry: two rows of four unit-spaced nodes, separated by 3 units transversally, so
# every source-target displacement is (j - i, 3, 0) with |delta| >= 3. That keeps the
# translations comfortably inside the M2L regime -- random centres paired with random
# class displacements instead produce O(1e+150) coefficients, which measures nothing.
_ROW_LENGTH = 4
_ROW_SEPARATION = 3.0
_TOTAL_NODES = 2 * _ROW_LENGTH
_NUM_PAIRS = _ROW_LENGTH * _ROW_LENGTH
#: Displacement classes are the distinct ``j - i`` offsets along the row.
_NUM_CLASSES = 2 * _ROW_LENGTH - 1

#: ``chunk_size`` values that force each side of the branch in
#: :func:`_accumulate_solidfmm_m2l_grouped`; pinned by
#: :func:`test_grouped_branch_selection_is_actually_exercised`.
_FULLBATCH_CHUNK = _NUM_PAIRS + 4
_CHUNKED_CHUNK = _NUM_PAIRS // 2

# Both branches sum exactly the same per-pair contributions and differ only in the
# reduction (one global ``segment_sum`` versus per-chunk reduction then scatter-add).
# The gap is therefore pure float64 reassociation over at most _NUM_PAIRS terms:
# measured 8e-17 (real) and 6e-17 (complex), so 1e-12 leaves four decades of headroom
# while still sitting ten decades below the 3.8e-01 gap the missing basis_mode caused.
_REASSOCIATION_RTOL = 1e-12


def _coefficient_dtype(basis_mode):
    """Packed-coefficient dtype the runtime uses for this basis."""
    return jnp.complex128 if basis_mode == "complex" else jnp.float64


def _fixture(basis_mode):
    """Build a small grouped far-field set: buffers, centres, packed multipoles."""
    rng = np.random.default_rng(20260809)

    sources = np.arange(_ROW_LENGTH)
    targets = _ROW_LENGTH + np.arange(_ROW_LENGTH)
    centers = np.zeros((_TOTAL_NODES, 3))
    centers[sources, 0] = sources
    centers[targets, 0] = np.arange(_ROW_LENGTH)
    centers[targets, 1] = _ROW_SEPARATION

    # Class-major ordering: pairs sorted by displacement class, as the grouped
    # interaction builder emits them.
    pair_src, pair_tgt = (
        np.repeat(sources, _ROW_LENGTH),
        np.tile(targets, _ROW_LENGTH),
    )
    offsets = pair_tgt - _ROW_LENGTH - pair_src
    class_ids = offsets + (_ROW_LENGTH - 1)
    order_by_class = np.argsort(class_ids, kind="stable")
    pair_src, pair_tgt, class_ids = (
        pair_src[order_by_class],
        pair_tgt[order_by_class],
        class_ids[order_by_class],
    )

    class_offsets_along_row = np.arange(_NUM_CLASSES) - (_ROW_LENGTH - 1)
    class_displacements = np.zeros((_NUM_CLASSES, 3))
    class_displacements[:, 0] = class_offsets_along_row
    class_displacements[:, 1] = _ROW_SEPARATION
    # Keys must agree with the displacements: the grouped operator cache is keyed on
    # both, so an inconsistent key would alias two different classes' rotation blocks.
    class_keys = np.zeros((_NUM_CLASSES, 5), dtype=np.int32)
    class_keys[:, 0] = class_offsets_along_row
    class_keys[:, 1] = int(_ROW_SEPARATION)

    grouped = GroupedInteractionBuffers(
        class_keys=jnp.asarray(class_keys),
        class_displacements=jnp.asarray(class_displacements),
        class_offsets=jnp.asarray(
            np.concatenate(
                [[0], np.cumsum(np.bincount(class_ids, minlength=_NUM_CLASSES))]
            ),
            dtype=jnp.int32,
        ),
        class_sources=jnp.asarray(pair_src, dtype=jnp.int32),
        class_targets=jnp.asarray(pair_tgt, dtype=jnp.int32),
        class_ids=jnp.asarray(class_ids, dtype=jnp.int32),
        level_offsets=jnp.zeros((1,), dtype=jnp.int32),
        level_nodes=jnp.zeros((1,), dtype=jnp.int32),
    )

    coeffs = rng.standard_normal((_TOTAL_NODES, sh_size(_ORDER)))
    if basis_mode == "complex":
        coeffs = coeffs + 1j * rng.standard_normal(coeffs.shape)
    multipoles = jnp.asarray(coeffs, dtype=_coefficient_dtype(basis_mode))
    return grouped, jnp.asarray(centers), multipoles


def _zero_locals(basis_mode):
    """Fresh zeroed local coefficients (the accumulators donate argument 0)."""
    return jnp.zeros(
        (_TOTAL_NODES, sh_size(_ORDER)), dtype=_coefficient_dtype(basis_mode)
    )


def _relative_gap(left, right):
    """Max absolute difference, normalised by the magnitude of ``right``."""
    scale = float(np.max(np.abs(right)))
    assert scale > 0.0, "fixture produced no far-field contribution to compare"
    return float(np.max(np.abs(left - right))) / scale


def _run_grouped(chunk_size, basis_mode):
    """Drive the grouped accumulator, letting ``chunk_size`` select the branch."""
    grouped, centers, multipoles = _fixture(basis_mode)
    return np.asarray(
        _accumulate_solidfmm_m2l_grouped(
            _zero_locals(basis_mode),
            multipoles,
            centers,
            grouped,
            order=_ORDER,
            rotation="solidfmm",
            total_nodes=_TOTAL_NODES,
            chunk_size=chunk_size,
            basis_mode=basis_mode,
        )
    )


@pytest.mark.parametrize("basis_mode", ["complex", "real"])
def test_grouped_branches_agree_in_both_bases(basis_mode):
    """Fullbatch and chunked-scan grouped M2L agree; the branch is batching only.

    The ``real`` case is the regression pin: the branches disagreed by 38% relative
    while the fullbatch call site dropped ``basis_mode``.
    """
    fullbatch = _run_grouped(_FULLBATCH_CHUNK, basis_mode)
    chunked = _run_grouped(_CHUNKED_CHUNK, basis_mode)
    assert _relative_gap(fullbatch, chunked) <= _REASSOCIATION_RTOL


def test_grouped_branch_selection_is_actually_exercised():
    """The two ``chunk_size`` values really do land on different branches.

    Without this, :func:`test_grouped_branches_agree_in_both_bases` would keep passing
    if the branch condition ever changed so that both calls took the same path -- the
    "the test passed but covered nothing" failure mode.
    """
    from jaccpot.runtime.fmm_caches import _M2L_FULLBATCH_MAX_PAIRS

    assert _NUM_PAIRS <= min(_FULLBATCH_CHUNK, _M2L_FULLBATCH_MAX_PAIRS)
    assert _NUM_PAIRS > min(_CHUNKED_CHUNK, _M2L_FULLBATCH_MAX_PAIRS)


def _run_accumulators_directly(basis_mode, *, kernel_basis_mode=None):
    """Run both accumulators on identical inputs.

    ``basis_mode`` selects the rotation blocks and the coefficient dtype;
    ``kernel_basis_mode`` (defaulting to the same value) is what the accumulators are
    told, so the defect -- real blocks interpreted by the complex kernel -- can be
    reproduced deliberately.
    """
    kernel_basis_mode = basis_mode if kernel_basis_mode is None else kernel_basis_mode
    grouped, centers, multipoles = _fixture(basis_mode)
    blocks_to, blocks_from = _rotation_blocks_for_grouped_classes(
        order=_ORDER,
        rotation="solidfmm",
        class_keys=grouped.class_keys,
        class_deltas=grouped.class_displacements,
        dtype=multipoles.dtype,
        basis_mode=basis_mode,
    )
    common = (
        multipoles,
        centers,
        grouped.class_sources,
        grouped.class_targets,
        grouped.class_ids,
        blocks_to,
        blocks_from,
    )
    fullbatch = _accumulate_solidfmm_m2l_grouped_fullbatch(
        _zero_locals(basis_mode),
        *common,
        order=_ORDER,
        total_nodes=_TOTAL_NODES,
        basis_mode=kernel_basis_mode,
    )
    chunked = _accumulate_solidfmm_m2l_grouped_chunked_scan(
        _zero_locals(basis_mode),
        *common,
        order=_ORDER,
        total_nodes=_TOTAL_NODES,
        chunk_size=_CHUNKED_CHUNK,
        basis_mode=kernel_basis_mode,
    )
    return np.asarray(fullbatch), np.asarray(chunked)


@pytest.mark.parametrize("basis_mode", ["complex", "real"])
def test_accumulators_agree_when_given_the_same_basis_mode(basis_mode):
    """The two accumulators are equivalent given matching ``basis_mode``.

    Complements the dispatch test above: this one says the kernels agree, that one says
    the dispatcher actually hands them the same basis.
    """
    fullbatch, chunked = _run_accumulators_directly(basis_mode)
    assert _relative_gap(fullbatch, chunked) <= _REASSOCIATION_RTOL


def test_complex_kernel_on_real_blocks_is_wrong_not_merely_different():
    """``basis_mode`` is load-bearing on the fullbatch accumulator, not decorative.

    This reconstructs the defect: real blocks and real coefficients, but the kernel told
    ``"complex"``. If the two cached kernels ever agreed on the same blocks, every
    assertion above would hold vacuously and the original defect would be invisible.
    Measured relative gap 3.8e-01; the threshold is set well below it so the test fails
    loudly rather than marginally. The dtype warnings JAX emits here (a complex
    result cast back to float64) are part of the defect's signature, not a problem with
    the test -- hence the local filter.
    """
    correct, _ = _run_accumulators_directly("real")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrong, _ = _run_accumulators_directly("real", kernel_basis_mode="complex")
    assert _relative_gap(wrong, correct) > 1e-02

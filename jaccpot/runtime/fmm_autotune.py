"""AutotuneMixin: fmm_autotune methods extracted from the FMMEngine
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.operators.real_harmonics import complex_to_dehnen_real_coeffs, sh_size
from jaccpot.upward.tree_expansions import TreeUpwardData

from .dtypes import INDEX_DTYPE, complex_dtype_for_real
from .fmm_caches import (
    _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES,
    _GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES,
    _GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS,
    _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES,
    _GPU_M2L_AUTOTUNE_PAIR_BINS,
    _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES,
    _GPU_M2L_AUTOTUNE_XL_CANDIDATES,
    _m2l_autotune_lookup,
    _m2l_autotune_store,
)
from .kernels.core import _accumulate_m2l_chunked_scan

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # The engine lives in `_fmm_impl`, which imports *these mixins* -- so this import
    # must stay under TYPE_CHECKING or it would form the cycle ARCHITECTURE section 8
    # forbids. Inheriting `_EngineBase` makes the mixin *be* the engine under a type
    # checker, so every `self.<engine attribute>` resolves; at runtime the alias is
    # `object`, leaving the MRO exactly as it was.
    #
    # `ad7b00c` added this import to all eleven mixins and annotated `self` in most of
    # them, but not this one's three methods, so the import sat unused until audit item
    # 0.14 removed it as dead. Inheritance is what the import earns now -- see E.2.
    from ._fmm_impl import FMMEngine

    _EngineBase = FMMEngine
else:  # pragma: no cover - annotations only, never an import at runtime
    _EngineBase = object

__all__ = [
    "AutotuneMixin",
]


class AutotuneMixin(_EngineBase):
    def _select_autotune_m2l_candidates(self, *, pair_count: int) -> tuple[int, ...]:
        """Return candidate chunk sizes for one pair-count regime.

        Pure table lookup against ``_GPU_M2L_AUTOTUNE_PAIR_BINS`` -- it times
        nothing. The timing happens in
        :meth:`_autotune_runtime_m2l_chunk_size`, which uses this to bound how
        many candidates it has to measure.

        Parameters
        ----------
        pair_count : int
            Number of far pairs the run will process. Host-side; only its bin
            matters, so nearby counts share a candidate list -- and share a cache
            entry downstream.

        Returns
        -------
        tuple[int, ...]
            Chunk sizes to try, smallest regime first. Never empty, so a caller
            always has something to time.
        """

        pairs = int(pair_count)
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[0]:
            return _GPU_M2L_AUTOTUNE_SMALL_CANDIDATES
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[1]:
            return _GPU_M2L_AUTOTUNE_MEDIUM_CANDIDATES
        if pairs < _GPU_M2L_AUTOTUNE_PAIR_BINS[2]:
            return _GPU_M2L_AUTOTUNE_LARGE_CANDIDATES
        return _GPU_M2L_AUTOTUNE_XL_CANDIDATES

    def _sample_and_remap_far_pairs_for_autotune(
        self,
        *,
        src: Array,
        tgt: Array,
        max_pairs: int = _GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS,
        max_nodes: int = _GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample far pairs and remap node ids to a compact contiguous range.

        Builds a small stand-in problem for the timing loop. Two reductions
        happen and both are lossy on purpose, because this feeds a benchmark
        rather than a result: the pair list is **strided**, not randomly sampled,
        and node ids are renumbered into a dense range so the sampled multipoles
        can be gathered into a compact array.

        Node admission stops at ``max_nodes``, and pairs whose endpoints did not
        make it in are **dropped** -- so the returned pair count can be well below
        ``max_pairs`` on a wide tree, and the sample is biased towards nodes seen
        early in the pair list. Acceptable for choosing a chunk size, not for
        anything whose value depends on which pairs were kept.

        Calls ``jax.device_get``, so this is a host sync and must not be reached
        from inside a jitted path.

        Parameters
        ----------
        src : Array
            Source node id per far pair, any shape -- flattened here.
        tgt : Array
            Target node id per far pair, flattened the same way. Paired with
            ``src`` positionally after the flatten, so the two must share a
            layout.
        max_pairs : int
            Upper bound on sampled pairs; defaults to
            ``_GPU_M2L_AUTOTUNE_MAX_SAMPLE_PAIRS``. Applied as a stride first,
            then a truncation.
        max_nodes : int
            Upper bound on distinct nodes admitted; defaults to
            ``_GPU_M2L_AUTOTUNE_MAX_SAMPLE_NODES``. This is what bounds the
            multipole array the benchmark allocates.

        Returns
        -------
        np.ndarray
            Sampled source ids remapped to ``[0, num_local)``, int32.
        np.ndarray
            Sampled target ids under the same remapping, int32.
        np.ndarray
            Local-to-global node id table, int64. Its length is the local node
            count, and it is the gather index for the multipoles.

            All three are empty when there are no pairs, which the caller treats
            as "do not autotune" rather than as an error.
        """

        src_np = np.asarray(jax.device_get(src), dtype=np.int64).reshape(-1)
        tgt_np = np.asarray(jax.device_get(tgt), dtype=np.int64).reshape(-1)
        pair_count = int(src_np.shape[0])
        if pair_count == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int64),
            )

        stride = max(1, pair_count // int(max_pairs))
        src_view = src_np[::stride]
        tgt_view = tgt_np[::stride]
        if src_view.shape[0] > int(max_pairs):
            src_view = src_view[: int(max_pairs)]
            tgt_view = tgt_view[: int(max_pairs)]

        node_to_local: dict[int, int] = {}
        local_to_global: list[int] = []
        src_local: list[int] = []
        tgt_local: list[int] = []
        max_nodes_int = int(max_nodes)
        for src_i, tgt_i in zip(src_view.tolist(), tgt_view.tolist()):
            src_g = int(src_i)
            tgt_g = int(tgt_i)
            src_local_id = node_to_local.get(src_g)
            if src_local_id is None:
                if len(local_to_global) >= max_nodes_int:
                    continue
                src_local_id = len(local_to_global)
                node_to_local[src_g] = src_local_id
                local_to_global.append(src_g)
            tgt_local_id = node_to_local.get(tgt_g)
            if tgt_local_id is None:
                if len(local_to_global) >= max_nodes_int:
                    continue
                tgt_local_id = len(local_to_global)
                node_to_local[tgt_g] = tgt_local_id
                local_to_global.append(tgt_g)
            src_local.append(src_local_id)
            tgt_local.append(tgt_local_id)

        return (
            np.asarray(src_local, dtype=np.int32),
            np.asarray(tgt_local, dtype=np.int32),
            np.asarray(local_to_global, dtype=np.int64),
        )

    def _autotune_runtime_m2l_chunk_size(
        self,
        *,
        upward: TreeUpwardData,
        src: Array,
        tgt: Array,
        order: int,
        pair_count: int,
    ) -> Optional[int]:
        """Auto-select M2L chunk size on GPU for streamed far-pair execution.

        Times :func:`_accumulate_m2l_chunked_scan` once per candidate on a
        sampled sub-problem and keeps the fastest. Each candidate is run twice
        and only the second run is timed, so the compile is not counted.

        Returns ``None`` -- meaning "no opinion, leave the chunk size alone" --
        whenever autotuning is off, the basis is not ``solidfmm``, the backend is
        not GPU, there are no pairs, or the sample came back empty. ``None`` is
        never a failure signal.

        The result is memoized in the process-wide autotune store under a key of
        (backend, basis mode, dtype, order, rotation, m2l impl, Pallas flag, pair
        **bin**). The bin, not the pair count -- so a second run at a nearby
        problem size reuses this measurement rather than re-timing.

        This is a wall-clock measurement, so it is only as good as the machine is
        quiet: on a shared or loaded GPU the winner is noise. It does not affect
        the numbers computed -- the accumulate result is discarded and the choice
        is made on ``perf_counter`` alone.

        Parameters
        ----------
        upward : TreeUpwardData
            Upward-sweep output. Read for the multipole centres, their dtype, and
            the packed coefficients the benchmark feeds the kernel.
        src : Array
            Source node id per far pair.
        tgt : Array
            Target node id per far pair.
        order : int
            Expansion order ``p``, part of the cache key.
        pair_count : int
            Total far pairs. Selects both the candidate list and the cache bin;
            a non-positive value short-circuits to ``None``.

        Returns
        -------
        Optional[int]
            The winning chunk size, or ``None`` to leave the caller's choice
            untouched. See above for every route to ``None``.
        """

        if (
            not bool(self.autotune_m2l_chunk)
            or self.expansion_basis != "solidfmm"
            or jax.default_backend() != "gpu"
            or int(pair_count) <= 0
        ):
            return None

        basis_mode = self._solidfmm_basis_mode()
        order_int = int(order)
        pair_count_int = int(pair_count)
        dtype_name = str(jnp.asarray(upward.multipoles.centers).dtype)
        pair_bin = 0
        for idx, upper in enumerate(_GPU_M2L_AUTOTUNE_PAIR_BINS):
            if pair_count_int < int(upper):
                pair_bin = idx
                break
        else:
            pair_bin = len(_GPU_M2L_AUTOTUNE_PAIR_BINS)
        key = (
            "gpu",
            str(basis_mode),
            dtype_name,
            order_int,
            str(self.complex_rotation),
            "" if self.m2l_impl is None else str(self.m2l_impl),
            int(bool(self.use_pallas)),
            int(pair_bin),
        )
        cached = _m2l_autotune_lookup(key)
        if cached is not None:
            return int(cached)

        candidates = self._select_autotune_m2l_candidates(pair_count=pair_count_int)
        (
            src_sample_np,
            tgt_sample_np,
            local_to_global_np,
        ) = self._sample_and_remap_far_pairs_for_autotune(src=src, tgt=tgt)
        if src_sample_np.size == 0 or local_to_global_np.size == 0:
            return None

        local_to_global = jnp.asarray(local_to_global_np, dtype=INDEX_DTYPE)
        src_sample = jnp.asarray(src_sample_np, dtype=INDEX_DTYPE)
        tgt_sample = jnp.asarray(tgt_sample_np, dtype=INDEX_DTYPE)
        centers = jnp.asarray(upward.multipoles.centers)[local_to_global]
        coeff_count = sh_size(order_int)
        if basis_mode == "complex":
            coeff_dtype = complex_dtype_for_real(centers.dtype)
            multip_all = jnp.asarray(upward.multipoles.packed).astype(coeff_dtype)
        else:
            coeff_dtype = centers.dtype
            # The Dehnen no-sqrt2 converter, NOT `basis.real_sh`'s: the kernel
            # being timed below is the real M2L, which consumes that basis. The
            # sqrt(2) tesseral converter was used here and is a different,
            # incompatible normalization (see the note on
            # `complex_to_dehnen_real_coeffs`, and the guard in
            # tests/unit/operators/test_real_sh_roundtrip.py).
            #
            # This does not change the chunk size this function returns: the
            # accumulate result is discarded (`_ = ...`) and the choice is made
            # on `perf_counter` alone, and both converters produce the same
            # shapes and dtypes, so the kernel does identical arithmetic either
            # way. The fix is that the benchmark now feeds the kernel the data it
            # actually runs on, instead of data that only looks like it.
            multip_all = complex_to_dehnen_real_coeffs(
                jnp.asarray(upward.multipoles.packed), order=order_int
            ).astype(coeff_dtype)
        multip = multip_all[local_to_global, :coeff_count]
        locals0 = jnp.zeros(
            (int(local_to_global_np.shape[0]), int(coeff_count)),
            dtype=coeff_dtype,
        )
        total_nodes = int(local_to_global_np.shape[0])
        best_chunk: Optional[int] = None
        best_time = math.inf

        for chunk in candidates:
            chunk_int = int(chunk)
            if chunk_int <= 0:
                continue
            try:
                if basis_mode == "complex":
                    _ = _accumulate_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        basis_mode="complex",
                        rotation=str(self.complex_rotation),
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                    t0 = time.perf_counter()
                    _ = _accumulate_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        basis_mode="complex",
                        rotation=str(self.complex_rotation),
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                else:
                    m2l_impl = (
                        "rot_scale" if self.m2l_impl is None else str(self.m2l_impl)
                    )
                    # Benchmark the real M2L kernel evaluation actually runs
                    # (merged _accumulate_m2l_chunked_scan -> _apply_real_m2l,
                    # pure-JAX or fused-Pallas). The old non-fused Pallas
                    # accumulate variant is not on the evaluation path.
                    _ = _accumulate_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        basis_mode="real",
                        m2l_impl=m2l_impl,
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                    t0 = time.perf_counter()
                    _ = _accumulate_m2l_chunked_scan(
                        locals0,
                        multip,
                        centers,
                        src_sample,
                        tgt_sample,
                        jnp.asarray(src_sample.shape[0], dtype=INDEX_DTYPE),
                        order=order_int,
                        basis_mode="real",
                        m2l_impl=m2l_impl,
                        total_nodes=total_nodes,
                        chunk_size=chunk_int,
                    ).block_until_ready()
                elapsed = float(time.perf_counter() - t0)
            except Exception:
                continue
            if elapsed < best_time:
                best_time = elapsed
                best_chunk = chunk_int

        if best_chunk is not None:
            _m2l_autotune_store(key, int(best_chunk))
        return best_chunk

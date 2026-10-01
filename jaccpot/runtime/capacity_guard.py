"""Traced capacity guard for the fused lane, shared by the single- and multi-GPU drivers.

Under ``jit`` the fused refresh cannot raise when a fixed capacity saturates: the
walk's far / near / queue overflow and the cell partition's ``leaf_capacity``
overflow are tracers. They are all folded into ONE place,
``compact_far_pairs.far_pair_count``, which ``_interaction_cache`` saturates to the
full length of the far-pair buffer on any overflow. A saturated count is therefore
the signal, and this module is the one function that reads it.

It used to live as a closure inside ``strict_run_v2`` only. The multi-GPU lane
(:mod:`jaccpot.distributed.fused`) instead read three attributes that
``LargeNPreparedState`` does not have, so its local overflow flag was the constant
``False`` -- every saturation on a mesh device went unreported.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["fused_state_capacity_ok"]


def fused_state_capacity_ok(
    prepared: Any,
    *,
    traced_caps: Optional[dict] = None,
    rectangle_guard_active: bool = False,
) -> Array:
    """Whether a refreshed fused-lane state fits every fixed capacity.

    Parameters
    ----------
    prepared : Any
        A refreshed ``LargeNPreparedState`` (traced or concrete).
    traced_caps : Optional[dict]
        Host constants recorded while the refresh traced
        (``engine._strict_fused_traced_caps``): ``max_neighbors_per_leaf_used`` and
        ``compact_far_pair_capacity``. ``None`` skips the checks that need them;
        the structural far-pair check below does not.
    rectangle_guard_active : bool
        Also check the rectangle near-field layout's per-target capacity. Only
        meaningful when the CSR near-field lane is off; the CSR lane never reads
        the rectangle.

    Returns
    -------
    Array
        Boolean scalar, ``True`` when nothing saturated.
    """
    offsets = jnp.asarray(prepared.neighbor_list.offsets)
    counts = offsets[1:] - offsets[:-1]
    ok = jnp.asarray(True)
    padded = getattr(prepared, "nearfield_target_block_source_leaf_ids_padded", None)
    if padded is not None and rectangle_guard_active:
        padded_arr = jnp.asarray(padded)
        if padded_arr.ndim == 3 and int(padded_arr.shape[1]) > 0:
            capacity = int(padded_arr.shape[1]) * int(padded_arr.shape[2])
            ok = ok & jnp.all(counts <= jnp.asarray(capacity, dtype=counts.dtype))
    far_pairs = getattr(prepared, "compact_far_pairs", None)
    far_count = getattr(far_pairs, "far_pair_count", None)
    if far_count is not None:
        # STRUCTURAL: an overflow saturates the count to the buffer's own static
        # length, so this needs no recorded cap. A list that fills its buffer
        # exactly with real pairs is flagged too, which is conservative.
        sources = getattr(far_pairs, "sources", None)
        if sources is not None and int(jnp.asarray(sources).shape[0]) > 0:
            ok = ok & (
                jnp.asarray(far_count) < jnp.asarray(int(jnp.asarray(sources).shape[0]))
            )
    if isinstance(traced_caps, dict):
        # The neighbour cap is the one that fails SILENTLY on the dual walk:
        # yggdrax's near-overflow flag is a tracer under jit and nothing reads it.
        nbr_cap = traced_caps.get("max_neighbors_per_leaf_used")
        if nbr_cap is not None and int(counts.shape[0]) > 0:
            ok = ok & (jnp.max(counts) < jnp.asarray(int(nbr_cap), counts.dtype))
        far_cap = traced_caps.get("compact_far_pair_capacity")
        if far_cap is not None and far_count is not None:
            ok = ok & (jnp.asarray(far_count) < jnp.asarray(int(far_cap)))
    return ok

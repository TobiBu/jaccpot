"""The galaxy's Morton domains for the direct-sum clusters on the mesh (HH4c).

Hybrid Hermite session 4 (``docs/production_2026-11.md`` ``## HH4c``; nornax
``docs/hybrid_hermite.md`` section 15). nornax's mesh clusters
(``nornax.hybrid_mesh``) need two things of the galaxy's domain decomposition, inside
``shard_map``: move the galaxy rows to their owners (with every per-row array that
travels with a row) and say which device owns a point -- a cluster belongs to the
device that owns its centre of mass. :class:`MortonPartition` is jaccpot's answer, the
ownership the mesh force builds its trees in:

* the rows move with ``yggdrax.distributed.partition.sfc_repartition`` in the
  force's box (:func:`~jaccpot.distributed.fused.global_mesh_bounds` with the
  force's box shape), as the runner's repartition does
  (:func:`~jaccpot.distributed.rollout._repartition_shard`): each device then owns
  an exact Morton range;
* the cuts are the new ownership's lowest codes per device
  (:func:`~jaccpot.distributed.migration.splitters_shard`'s rule), and a point's owner is ``searchsorted(cuts, code, 'right')`` of its
  Morton code in the same box (clipped into it), ``sfc_repartition``'s rule.

nornax imports this; jaccpot does not import nornax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.distributed.fused import global_mesh_bounds

__all__ = ["MortonPartition"]


def _fill(a: Array) -> Any:
    a = jnp.asarray(a)
    if jnp.issubdtype(a.dtype, jnp.integer):
        return -1
    return 0.0


@dataclass(frozen=True)
class MortonPartition:
    """Exact Morton ranges in the force's box (nornax's ``MeshPartition``).

    Attributes
    ----------
    cubic : Optional[bool]
        The force's box shape (``TreeConfig.cubic_bounds``; the galaxy preset's
        cube); ``None``: the environment's default, as
        :func:`~jaccpot.distributed.fused.cube_bounds`.
    num_samples : int
        Pivot samples per device of the repartition.
    """

    cubic: Optional[bool] = None
    num_samples: int = 256

    def repartition(
        self, x: Array, m: Array, n_live: Array, payload: dict, *, axis_name: str
    ) -> tuple:
        """The galaxy rows to their owners (inside ``shard_map``).

        Parameters
        ----------
        x : Array
            ``(rows, 3)`` positions (the first ``n_live`` live).
        m : Array
            ``(rows,)`` masses.
        n_live : Array
            Live rows (scalar).
        payload : dict
            ``name -> (rows, ...)`` arrays that travel with their row (moved
            bitwise; integers filled with ``-1``, floats with ``0`` past the live
            rows).
        axis_name : str
            The mesh axis.

        Returns
        -------
        tuple
            ``(x, m, n_live, payload, cuts, ok)``: the new rows (live first,
            Morton-sorted; free rows at the origin with zero mass), the ``uint64``
            cuts of the new ownership and ``ok`` (False when the repartition was
            declined -- some device would overflow its rows -- and every device
            kept its own rows).
        """
        from yggdrax.distributed.partition import sfc_repartition
        from yggdrax.morton import morton_encode_impl

        from jaccpot.distributed.migration import _CODE_SENTINEL

        rows = int(x.shape[0])
        ndev = int(jax.lax.all_gather(jnp.zeros((), jnp.int32), axis_name).shape[0])
        live_n = jnp.asarray(n_live, jnp.int32)
        bounds = global_mesh_bounds(
            x, num_valid=live_n, axis_name=axis_name, cubic=self.cubic
        )
        res = sfc_repartition(
            x,
            m,
            live_n,
            ndev,
            output_capacity=rows,
            bounds=bounds,
            num_samples=int(self.num_samples),
            axis_name=axis_name,
            payload=dict(payload),
            payload_fill={k: _fill(v) for k, v in payload.items()},
        )
        count = jnp.asarray(res.live_count, jnp.int32)
        live = jnp.arange(rows) < count

        def tidy(a: Array, fill: Any) -> Array:
            keep = live
            while keep.ndim < a.ndim:
                keep = keep[..., None]
            return jnp.where(keep, a, jnp.asarray(fill, a.dtype))

        x_new = tidy(res.positions, 0.0)
        m_new = tidy(res.masses, 0.0)
        moved = {k: tidy(v, _fill(v)) for k, v in res.payload.items()}
        # the cuts of the new ownership (migration.splitters_shard's rule, in
        # the box the rows were routed in): each device's lowest live code, an
        # empty device taking the next device's cut
        codes = morton_encode_impl(x_new, bounds)
        lowest = jnp.min(jnp.where(live, codes, _CODE_SENTINEL))
        per_device = jax.lax.all_gather(lowest, axis_name, tiled=False)
        cuts = jax.lax.cummin(per_device, reverse=True)[1:]
        return x_new, m_new, count, moved, (cuts, bounds), ~res.declined

    def owners(self, points: Array, cuts: Any, *, axis_name: str) -> Array:
        """The device owning each point under ``cuts`` (``int32``).

        Parameters
        ----------
        points : Array
            ``(P, 3)``.
        cuts : Any
            :meth:`repartition`'s ``(splitters, box)``.
        axis_name : str
            Unused (the cuts are replicated).

        Returns
        -------
        Array
            ``(P,)`` ``int32``.
        """
        from yggdrax.morton import morton_encode_impl

        del axis_name
        splitters, bounds = cuts
        lo, hi = bounds
        p = jnp.clip(jnp.asarray(points, lo.dtype), lo, hi)
        codes = morton_encode_impl(p, bounds)
        return jnp.searchsorted(splitters, codes, side="right").astype(jnp.int32)

"""jaccpot's mesh forces as nornax ``ShardedForceModel`` s (multi-node plan I2).

nornax owns the integration schemes and the protocol a force evaluated inside
``shard_map`` speaks (``nornax.sharded.ShardedForceModel``): one device's rows,
a live-row mask, particle ids and a per-device carry, the other devices reached
through collectives. :class:`ShardedMeshForce` puts a jaccpot mesh force --
:class:`~jaccpot.distributed.template_force.FusedMeshForce` (the fused
multi-GPU lane) or :class:`~jaccpot.distributed.segment.DirectMeshForce` --
behind that protocol, structurally: jaccpot does not import nornax.

The production runner does not go through this adapter: its mesh segments call
the force directly (their programs are unchanged by I2), and take nornax's
scheme through :class:`jaccpot.scheme.SplitStepScheme`. The adapter is what
lets nornax's own drivers and its conformance kit
(``nornax.conformance.check_sharded_force_model``) run jaccpot's force.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

import jax.numpy as jnp
from jaxtyping import Array

__all__ = ["ShardedMeshForce"]


class ShardedMeshForce:
    """A jaccpot mesh force behind nornax's ``ShardedForceModel`` protocol.

    Parameters
    ----------
    force : Any
        A prepared mesh force: ``shard(positions, masses, live, ids, [force_scale])
        -> (acceleration, bits, needs, [f_b])`` inside ``shard_map``, with
        ``context()``, ``before_trace()``, ``needs_size``, ``axis_name`` and,
        for :meth:`shard_potential`, ``potential_shard``. Prepare it at the
        system first (``force.prepare(positions, masses, counts)``): the plan
        binds the force to its mesh and capacities.

    Notes
    -----
    The live rows must be a PREFIX of each device's block (jaccpot's layouts
    are; ``nornax.sharded.decompose_rows`` makes them so): the force takes
    their count.

    The carry is per device: ``bits`` (the OR of every evaluation's overflow
    bits since the first; nonzero means a capacity was exceeded and the force
    is not to be trusted: re-prepare and run again), ``needs`` (the max of the
    walk needs) and, under ``mac_type='dehnen_error'``, ``force_scale`` (eq
    16b's ``f_b``, the next evaluation's thresholds). Under the criterion the
    first carry must be given (:meth:`initial_carry`); otherwise ``None`` starts
    one.
    """

    def __init__(self, force: Any) -> None:
        self.force = force
        #: the mesh axis the force's collectives run over
        self.axis_name: str = str(force.axis_name)
        #: mac_type='dehnen_error': f_b rides the carry
        self.scale_carry = bool(getattr(force, "scale_carry", False))

    @contextlib.contextmanager
    def context(self) -> Iterator[None]:
        """The force's plan installed, its trace record reset (enter around a program).

        Yields
        ------
        None
            Inside the force's context.
        """
        with self.force.context():
            self.force.before_trace()
            yield

    def initial_carry(self, ndev: Optional[int] = None) -> dict:
        """The first carry, one entry per device (leading ``ndev`` axis).

        Parameters
        ----------
        ndev : Optional[int]
            The mesh's device count (``None``: the force's mesh, else every
            device).

        Returns
        -------
        dict
            ``bits`` ``(ndev,)`` and ``needs`` ``(ndev, needs_size)`` zero, and
            under the criterion ``force_scale`` ``(ndev, cap)``: the last
            prepare's seed.

        Raises
        ------
        ValueError
            Under the criterion, before the force was prepared.
        """
        ndev = _mesh_size(self.force) if ndev is None else int(ndev)
        carry: dict = {
            "bits": jnp.zeros((ndev,), jnp.uint32),
            "needs": jnp.zeros((ndev, int(self.force.needs_size)), jnp.int32),
        }
        if self.scale_carry:
            seed = getattr(self.force, "force_scale_seed", None)
            if seed is None:
                raise ValueError("the force has no f_b seed: prepare it first")
            carry["force_scale"] = jnp.reshape(seed, (ndev, -1))
        return carry

    def _carry(self, carry: Optional[dict]) -> dict:
        if carry is not None:
            return carry
        if self.scale_carry:
            raise ValueError(
                "mac_type='dehnen_error': the first carry holds f_b; pass "
                "ShardedMeshForce.initial_carry()"
            )
        return {
            "bits": jnp.uint32(0),
            "needs": jnp.zeros((int(self.force.needs_size),), jnp.int32),
        }

    def shard_accelerations(
        self,
        positions: Array,
        masses: Array,
        ids: Array,
        live_rows: Array,
        *,
        carry: Optional[dict] = None,
        args: Any = None,
    ) -> tuple[Array, dict]:
        """The self-gravity on this device's rows (inside ``shard_map``).

        Parameters
        ----------
        positions : Array
            ``(cap, 3)``.
        masses : Array
            ``(cap,)``; zero on padding rows.
        ids : Array
            ``(cap,)`` particle ids.
        live_rows : Array
            ``(cap,)`` boolean, a prefix.
        carry : Optional[dict]
            The previous evaluation's (see the class notes).
        args : Any
            Unused (the force is static in ``G`` and the softening).

        Returns
        -------
        tuple[Array, dict]
            ``(acceleration, carry)``; the acceleration zero on padding rows.
        """
        del args
        carry = self._carry(carry)
        live = jnp.sum(live_rows.astype(jnp.int32))
        out = self.force.shard(
            positions,
            masses,
            live,
            ids,
            **({"force_scale": carry["force_scale"]} if self.scale_carry else {}),
        )
        acc, bits, needs = out[:3]
        new = {
            "bits": carry["bits"] | bits,
            "needs": jnp.maximum(carry["needs"], needs),
        }
        if self.scale_carry:
            new["force_scale"] = out[3]
        acc = jnp.where(live_rows[:, None], acc, 0).astype(positions.dtype)
        return acc, new

    def shard_potential(
        self,
        positions: Array,
        masses: Array,
        ids: Array,
        live_rows: Array,
        *,
        carry: Optional[dict] = None,
        args: Any = None,
    ) -> Array:
        """The self-gravity potential per unit mass on this device's rows.

        Parameters
        ----------
        positions : Array
            ``(cap, 3)``.
        masses : Array
            ``(cap,)``.
        ids : Array
            ``(cap,)`` particle ids.
        live_rows : Array
            ``(cap,)`` boolean, a prefix.
        carry : Optional[dict]
            Under the criterion its ``force_scale`` sets the lists' thresholds.
        args : Any
            Unused.

        Returns
        -------
        Array
            ``(cap,)``, zero on padding rows.
        """
        del args
        carry = self._carry(carry)
        live = jnp.sum(live_rows.astype(jnp.int32))
        phi = self.force.potential_shard(
            positions,
            masses,
            live,
            ids,
            **({"force_scale": carry["force_scale"]} if self.scale_carry else {}),
        )[0]
        return jnp.where(live_rows, phi, 0).astype(positions.dtype)


def _mesh_size(force: Any) -> int:
    """The number of devices of the force's mesh."""
    mesh = getattr(force, "mesh", None)
    if mesh is not None:
        return int(mesh.devices.size)
    import jax

    return int(jax.device_count())

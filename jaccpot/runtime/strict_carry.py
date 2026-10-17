"""``strict_run_v2(carry="particles")``: the fused scan carries the particles only.

The fused lane's scan carried the whole prepared state -- tree, upward and
downward payloads, lists: 1.4 GiB at 8x10^6 particles, plus the far list then
held outside the scan to re-attach to the returned state (0.66 GiB; no longer
re-attached since 2026-10-15, it was the start positions' list) -- although the
in-scan refresh rebuilds every one of them from the positions. Measured by
jax's dead-code elimination on the scan body (``tests/integration/strict/
test_strict_run_v2.py::TestParticleCarry``): with the default fresh far-pair rebuild,
a step reads NONE of the carried state's 51 leaves, only their shapes, dtypes
and the static fields.

So the carry can be ``(state, acceleration, self acceleration, ok, walk needs)``
and the prepared state a SHAPE TEMPLATE, materialised inside each step as
broadcasts that XLA removes again (nothing reads them; floating leaves are NaN,
so a future refresh that did read one would poison the trajectory instead of
silently using zeros). Between calls the template travels in a
:class:`StrictParticleCarry` handle with the self-gravity at the returned
positions, which is exactly the next call's initial force -- the eager
``prepare_state`` and the extra force evaluation at the start of each call are
gone too.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "PreparedCaps",
    "PreparedShapeTemplate",
    "StrictParticleCarry",
    "external_field_key",
    "host_copy",
    "materialize_template",
    "shape_template",
]


@dataclasses.dataclass(frozen=True)
class PreparedShapeTemplate:
    """A prepared state's pytree structure and leaf shapes, without its buffers.

    Attributes
    ----------
    treedef : Any
        The state's ``PyTreeDef`` (static fields included).
    leaves : tuple
        Per leaf ``jax.ShapeDtypeStruct`` for arrays, else the leaf itself.
    """

    treedef: Any
    leaves: tuple

    def key(self) -> tuple:
        """Hashable identity for compile caches: structure, shapes, dtypes.

        Returns
        -------
        tuple
            ``(treedef, ((shape, dtype) | leaf, ...))``.
        """
        return (
            self.treedef,
            tuple(
                (
                    (tuple(x.shape), str(x.dtype))
                    if isinstance(x, jax.ShapeDtypeStruct)
                    else ("leaf", repr(x))
                )
                for x in self.leaves
            ),
        )


@dataclasses.dataclass(frozen=True)
class PreparedCaps:
    """What the caps-only prepare leaves in place of a prepared state.

    The particle carry reads four things from the eager prepare: the template's
    SHAPES, the engine's trace-time state (validated walk caps, level registry,
    depth bound), the initial capacity verdict and, under
    ``mac_type='dehnen_error'``, the eq (16b) seed with the eager permutation. Its
    start force comes from the start program. So the caps-only prepare
    (``PrepareMixin.prepare_state_caps_only``) builds the tree and the upward pass
    and runs the eager flat walk's ladder, and stops: no list, no M2L or L2L, no
    near-field payload, no far f_b. The template is derived from a rebuild of
    :attr:`stub` under ``jax.eval_shape``
    (:meth:`~jaccpot.runtime.fused_provider.FusedForceProvider.prepare` with
    ``caps_only=True``).

    Attributes
    ----------
    stub : Any
        A ``LargeNPreparedState`` holding the eager tree and the state's static
        fields, nothing else: what a rebuild reads of a prepared state. Its tree's
        ``particle_indices`` are the eager permutation the f_b seed is scattered
        through.
    walk : Any
        The eager walk's ``_FlatWalkCaps``: the inputs of the verdict.
    capacity_ok : Optional[bool]
        The verdict the full prepare's state would get
        (``capacity_guard.fused_state_capacity_ok``), set with the template;
        ``None`` before.
    shapes : Any
        The rebuild's output under ``jax.eval_shape`` (a ``LargeNPreparedState``
        of ``jax.ShapeDtypeStruct``) the template was taken from; ``None`` before.
    """

    stub: Any
    walk: Any
    capacity_ok: Optional[bool] = None
    shapes: Any = None

    @property
    def tree(self) -> Any:
        """The eager tree (the full prepare's ``prepared.tree``).

        Returns
        -------
        Any
            The stub's tree.
        """
        return self.stub.tree


@dataclasses.dataclass(frozen=True)
class StrictParticleCarry:
    """What ``strict_run_v2(carry="particles")`` returns in place of a prepared state.

    Pass it back as ``prepared_state`` together with the state that call
    returned. It is accepted by nothing else: a full prepared state for other
    APIs comes from ``prepare_state`` on the returned positions.

    Attributes
    ----------
    template : PreparedShapeTemplate
        Shapes of the fused lane's prepared state (far list detached).
    self_acceleration : Array
        ``[N, 3]`` self-gravity at the returned state's positions -- the next
        call's initial force.
    num_particles : int
        ``N``.
    force_scale : Optional[Array]
        ``mac_type='dehnen_error'``: eq (16b)'s ``f_b`` per particle (input order)
        at the returned state -- the next call's first thresholds. ``None``
        otherwise.
    provider : Any
        The :class:`~jaccpot.runtime.fused_provider.FusedForceProvider` the call
        stepped with (its ``template`` is ``template``): the force at the returned
        positions for a caller that drives the steps itself. The next
        ``strict_run_v2`` call builds its own from ``template`` and its arguments.
    force_key : Optional[tuple]
        ``(G, eps2)`` as host floats: the force parameters ``self_acceleration``
        was computed with (``None``: the solver's own). A call with others
        evaluates its initial force again instead of taking this one.
    acceleration : Optional[Array]
        ``strict_run_v2(carry_total_acceleration=True)`` only: the TOTAL
        acceleration ``[N, 3]`` at the returned state as the scan's last step
        computed it -- the self-gravity plus the external field (without a field,
        ``self_acceleration`` itself). The next call with the same field,
        parameters and force parameters starts its scan from it, and consumes it.
        ``None`` otherwise.
    external_key : Optional[tuple]
        :func:`external_field_key` of the field and parameters ``acceleration``
        was computed with (``None``: no field).
    """

    template: PreparedShapeTemplate
    self_acceleration: Array
    num_particles: int
    force_scale: Optional[Array] = None
    provider: Any = None
    force_key: Optional[tuple] = None
    acceleration: Optional[Array] = None
    external_key: Optional[tuple] = None


def external_field_key(
    external_acceleration_fn: Optional[Callable[..., Array]], external_params: Any
) -> Optional[tuple]:
    """What a handle's total acceleration was computed with: the field and its values.

    ``(id(field), sha256 of the parameters)``: the function's identity, as the
    compile caches key it, and the parameters' structure and VALUES (traced arrays
    are fetched; they are small), so a call with other parameter values never
    starts from a total computed with the old ones.

    Parameters
    ----------
    external_acceleration_fn : Optional[Callable[..., Array]]
        The external field, or ``None``.
    external_params : Any
        Its parameters (any pytree), or ``None`` for a one-argument field.

    Returns
    -------
    Optional[tuple]
        ``None`` without a field.
    """
    if external_acceleration_fn is None:
        return None
    leaves, treedef = jax.tree_util.tree_flatten(external_params)
    digest = hashlib.sha256(repr(treedef).encode())
    for leaf in leaves:
        if isinstance(leaf, (jax.Array, np.ndarray, np.generic)):
            value = np.asarray(jax.device_get(leaf))
            digest.update(f"{value.dtype}{value.shape}".encode())
            digest.update(value.tobytes(order="C"))
        else:
            digest.update(repr(leaf).encode())
    return (id(external_acceleration_fn), digest.hexdigest())


def shape_template(prepared: Any) -> PreparedShapeTemplate:
    """The shape template of ``prepared``; holds no device buffer.

    Parameters
    ----------
    prepared : Any
        A prepared-state pytree.

    Returns
    -------
    PreparedShapeTemplate
        Its structure and leaf shapes.
    """
    leaves, treedef = jax.tree_util.tree_flatten(prepared)
    return PreparedShapeTemplate(
        treedef=treedef,
        leaves=tuple(
            (
                jax.ShapeDtypeStruct(tuple(x.shape), x.dtype)
                if isinstance(x, (jax.Array, np.ndarray))
                else x
            )
            for x in leaves
        ),
    )


def materialize_template(template: PreparedShapeTemplate) -> Any:
    """A prepared state of the template's shapes made of broadcasts (inside a trace).

    Floating leaves are NaN, integer and boolean leaves zero: a consumer that read
    a value it should not would turn the trajectory to NaN rather than pass.

    Parameters
    ----------
    template : PreparedShapeTemplate
        From :func:`shape_template`.

    Returns
    -------
    Any
        The prepared-state pytree.
    """

    def _fill(x: Any) -> Any:
        if not isinstance(x, jax.ShapeDtypeStruct):
            return x
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.full(x.shape, jnp.nan, x.dtype)
        return jnp.zeros(x.shape, x.dtype)

    return jax.tree_util.tree_unflatten(
        template.treedef, [_fill(x) for x in template.leaves]
    )


def host_copy(state: Array) -> Any:
    """A copy of a device state off the device, for a state about to be donated.

    ``strict_run_v2(donate_state=True)`` hands its start state to the scan, and a
    capacity retry re-runs the segment from this copy; the runner keeps step 0's
    state this way for its energy when ``donate_input`` consumes it. On a GPU the
    copy is an asynchronous ``device_put`` into pinned host memory: the call
    returns at once, and a program that consumes ``state`` waits for the DMA (PjRt
    orders a donation after the buffer's pending reads). 2.4 GB at 1e8 particles
    on an A100: 0.10 s per copy (the run's first 1.4 s, the pinned allocation),
    against 0.97-1.11 s for the synchronous ``device_get`` into pageable NumPy
    memory it replaces -- the larger part of the runner's edge at 1e8. Elsewhere
    that NumPy copy. ``jax.device_put(copy, sharding)`` restores either, bitwise.

    Parameters
    ----------
    state : Array
        The state, about to be donated.

    Returns
    -------
    Any
        A ``pinned_host`` array (GPU) or a NumPy array.
    """
    devices = list(state.devices())
    pinned = all(
        d.platform == "gpu"
        and any(m.kind == "pinned_host" for m in d.addressable_memories())
        for d in devices
    )
    if pinned:
        return jax.device_put(state, state.sharding.with_memory_kind("pinned_host"))
    return np.asarray(jax.device_get(state))

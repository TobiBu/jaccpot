"""``strict_run_v2(carry="particles")``: the fused scan carries the particles only.

The fused lane's scan carried the whole prepared state -- tree, upward and
downward payloads, lists: 1.4 GiB at 8x10^6 particles, plus the far list held
outside the scan to re-attach to the returned state (0.66 GiB) -- although the
in-scan refresh rebuilds every one of them from the positions. Measured by
jax's dead-code elimination on the scan body (``tests/integration/
test_strict_run_v2_particle_carry.py``): with the default fresh far-pair rebuild,
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
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

__all__ = [
    "PreparedShapeTemplate",
    "StrictParticleCarry",
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
    """

    template: PreparedShapeTemplate
    self_acceleration: Array
    num_particles: int


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

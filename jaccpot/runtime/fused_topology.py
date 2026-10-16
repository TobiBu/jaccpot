"""``FusedTopology`` and the force parameters of the fused lane.

What :class:`~jaccpot.runtime.fused_provider.FusedForceProvider` hands from one half
of a step to the other, and what the force is parameterised by:

:class:`FusedTopology`
    a step's topology: the refreshed prepared state (tree, local expansions after
    L2L, CSR near list, near-field payload) plus what the rebuild's walk and tree
    reported about their fixed capacities -- tracers under a trace, moved off the
    engine's side channels. :meth:`FusedTopology.overflow_bits` gives one bit per
    capacity (``OVERFLOW_*``), :func:`overflow_ok` the verdict.
:class:`ForceParams`, :class:`StepParams`
    pytrees whose Python floats are STATIC (aux data: compile-time literals,
    bitwise the closure constants they replace) and whose arrays are traced.

Containers and their arithmetic only: nothing here builds a tree or evaluates a
force, so a caller can carry and inspect them without the engine.
"""

from __future__ import annotations

import dataclasses
from numbers import Real
from typing import Any, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .capacity_guard import WALK_NEEDS_FIELDS

__all__ = [
    "OVERFLOW_CROSS",
    "OVERFLOW_FAIL_MASK",
    "OVERFLOW_FAR_CAP",
    "OVERFLOW_LEVEL_PLAN",
    "OVERFLOW_NEIGHBOR_ROW",
    "OVERFLOW_PARAMS",
    "OVERFLOW_RECTANGLE",
    "OVERFLOW_SATURATED",
    "OVERFLOW_TREE",
    "OVERFLOW_WALK_FAR",
    "OVERFLOW_WALK_NEAR",
    "OVERFLOW_WALK_QUEUE",
    "ForceParams",
    "FusedTopology",
    "StepParams",
    "overflow_ok",
]

#: Bits of :meth:`FusedTopology.overflow_bits`. Bits 0-2 say WHAT overflowed in the
#: walk; the flat walk also saturates the fresh far list on each of them (bit 4), so
#: they inform a re-plan and never decide a verdict on their own. Bits 3-10 are the
#: verdict: any of them set means the step's force is wrong.
OVERFLOW_WALK_FAR = 1 << 0
OVERFLOW_WALK_NEAR = 1 << 1
OVERFLOW_WALK_QUEUE = 1 << 2
#: the cell-leaf rebuild: a leaf over capacity, a tree deeper than the level-loop
#: bound, or a level wider than the level loops' static batch. A verdict bit: the
#: traced refresh does not fold it into the walk's saturation (the prepare does),
#: so bit 4 does not imply it -- a capacity overflow of the cell leaves left the
#: scan's verdict ``True`` with the force 55 % off (rel-L2, N = 512, CSR lane)
OVERFLOW_TREE = 1 << 3
#: a far list this rebuild built is saturated (the flat walk's overflow signal)
OVERFLOW_SATURATED = 1 << 4
#: a near-field row longer than the traced per-leaf cap (the dual walk's lane)
OVERFLOW_NEIGHBOR_ROW = 1 << 5
#: the returned state's far count at the traced compact far-pair cap
OVERFLOW_FAR_CAP = 1 << 6
#: the rectangle near-field layout's per-target capacity (CSR lane off)
OVERFLOW_RECTANGLE = 1 << 7
#: the caller's (multi-GPU) level plan; passed in through ``extra``
OVERFLOW_LEVEL_PLAN = 1 << 8
#: the caller's (multi-GPU) cross-domain capacities; passed in through ``extra``
OVERFLOW_CROSS = 1 << 9
#: traced force parameters outside what the topology was built for (C4)
OVERFLOW_PARAMS = 1 << 10
#: the verdict bits: ``ok = (bits & OVERFLOW_FAIL_MASK) == 0``
OVERFLOW_FAIL_MASK = 0x7F8


def overflow_ok(bits: Array) -> Array:
    """The capacity verdict of :meth:`FusedTopology.overflow_bits`.

    Parameters
    ----------
    bits : Array
        ``uint32`` overflow bits.

    Returns
    -------
    Array
        Boolean scalar, ``True`` when no verdict bit (3-10) is set.
    """
    return (jnp.asarray(bits, jnp.uint32) & jnp.uint32(OVERFLOW_FAIL_MASK)) == 0


# --------------------------------------------------------------------------
# Force parameters: Python floats static, arrays traced.
# --------------------------------------------------------------------------

#: placeholder in a parameter pytree's aux data for a field that is a CHILD (traced)
_TRACED = "<traced>"


def _is_static(value: Any) -> bool:
    """Whether ``value`` is a Python number: aux data, a compile-time literal.

    Parameters
    ----------
    value : Any
        A parameter value.

    Returns
    -------
    bool
        ``True`` for a real number (not a bool, not an array).
    """
    return isinstance(value, Real) and not isinstance(value, bool)


def _split_static(values: tuple) -> tuple[list, tuple]:
    """Python numbers to aux, everything else to children.

    Parameters
    ----------
    values : tuple
        Field values, in order.

    Returns
    -------
    tuple[list, tuple]
        ``(children, aux)``; aux holds each static value, ``_TRACED`` where a
        child goes.
    """
    children = [v for v in values if not _is_static(v)]
    aux = tuple(v if _is_static(v) else _TRACED for v in values)
    return children, aux


def _join_static(aux: tuple, children: Any) -> list:
    """Inverse of :func:`_split_static`.

    Parameters
    ----------
    aux : tuple
        Static values and ``_TRACED`` markers.
    children : Any
        The traced values, in order.

    Returns
    -------
    list
        Field values, in order.
    """
    it = iter(children)
    return [next(it) if a == _TRACED else a for a in aux]


def _construct(cls: type, values: list) -> Any:
    """A frozen dataclass from field values, bypassing ``__init__``.

    ``tree_unflatten`` receives whatever jax puts in the leaves (abstract values,
    ``None``, sentinels), which the field annotations do not describe; the runtime
    type checker must not see them.

    Parameters
    ----------
    cls : type
        The dataclass.
    values : list
        Its field values, in declaration order.

    Returns
    -------
    Any
        The instance.
    """
    obj = object.__new__(cls)
    for field, value in zip(dataclasses.fields(cls), values):
        object.__setattr__(obj, field.name, value)
    return obj


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ForceParams:
    """The physical parameters of the self-gravity: ``G`` and the softening squared.

    A Python float is STATIC (aux data: a compile-time literal, exactly the closure
    constant the lane traces today); an array is a traced child. The softening is
    carried squared, as the kernels use it.

    Attributes
    ----------
    G : Union[float, Array]
        Gravitational constant.
    eps2 : Union[float, Array]
        Softening length squared.
    """

    G: Union[float, Array]
    eps2: Union[float, Array]

    def tree_flatten(self) -> tuple[list, tuple]:
        """Children (traced fields) and aux (static fields).

        Returns
        -------
        tuple[list, tuple]
            ``(children, aux)``.
        """
        return _split_static((self.G, self.eps2))

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: Any) -> "ForceParams":
        """Rebuild from :meth:`tree_flatten`'s output.

        Parameters
        ----------
        aux : tuple
            Static fields, ``_TRACED`` where a child goes.
        children : Any
            The traced fields, in order.

        Returns
        -------
        ForceParams
            The parameters.
        """
        return _construct(cls, _join_static(aux, children))

    def is_static(self) -> bool:
        """Whether every field is a Python number (compile-time literals).

        Returns
        -------
        bool
            ``True`` when nothing is traced.
        """
        return _is_static(self.G) and _is_static(self.eps2)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class StepParams:
    """Everything a step of the fused scan is parameterised by.

    Attributes
    ----------
    force : ForceParams
        Self-gravity parameters.
    dt : Union[float, Array]
        Timestep; static as a Python float, traced as an array.
    external : Any
        Parameters of the external field (any pytree; ``None`` for none). Its
        Python-number leaves are static like ``dt``.
    """

    force: ForceParams
    dt: Union[float, Array]
    external: Any = None

    def tree_flatten(self) -> tuple[list, tuple]:
        """Children (traced fields) and aux (static fields).

        Returns
        -------
        tuple[list, tuple]
            ``(children, aux)``: the force parameters, the traced ``dt`` and the
            external field's traced leaves; the static values and the external
            field's structure.
        """
        ext_leaves, ext_def = jax.tree_util.tree_flatten(self.external)
        children, aux = _split_static((self.dt, *ext_leaves))
        return [self.force, *children], (ext_def, aux)

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: Any) -> "StepParams":
        """Rebuild from :meth:`tree_flatten`'s output.

        Parameters
        ----------
        aux : tuple
            ``(external treedef, static values)``.
        children : Any
            The force parameters, then the traced values in order.

        Returns
        -------
        StepParams
            The parameters.
        """
        ext_def, static = aux
        children = list(children)
        dt, *ext_leaves = _join_static(static, children[1:])
        external = jax.tree_util.tree_unflatten(ext_def, ext_leaves)
        return _construct(cls, [children[0], dt, external])


# --------------------------------------------------------------------------
# The topology a rebuild returns, and its overflow bits.
# --------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class FusedTopology:
    """What :meth:`FusedForceProvider.rebuild` returns: a step's topology.

    Attributes
    ----------
    prepared : Any
        The refreshed ``LargeNPreparedState``: tree, local expansions after L2L,
        CSR near list, near-field payload. Its far list is the template's
        placeholder (``None`` on the fused lane): the rebuild built a fresh one,
        used it for M2L and kept none.
    refresh_ok : Any
        Boolean scalar: the far list THIS rebuild built did not saturate (the flat
        walk saturates it on any walk or tree overflow). ``None`` when no rebuild
        made the topology.
    walk_needs : Any
        ``int32[7]`` (``capacity_guard.WALK_NEEDS_FIELDS``): what the traced walk
        needed. ``None`` when the walk reported none (an eager or dual walk).
    tree_overflow : Any
        Boolean scalar: the cell-leaf rebuild overflowed its leaf capacity, depth
        bound or level width. ``None`` on bucket leaves or eagerly.
    fresh : bool
        Made by a rebuild in this trace, so ``refresh_ok`` belongs to it. Static.
    caps : tuple
        ``(max_neighbors_per_leaf_used, compact_far_pair_capacity,
        rectangle_guard_active)``: the traced caps the rebuild recorded and whether
        the rectangle layout is checked. Static.
    """

    prepared: Any
    refresh_ok: Any = None
    walk_needs: Any = None
    tree_overflow: Any = None
    fresh: bool = False
    caps: tuple = (None, None, False)

    def tree_flatten(self) -> tuple[tuple, tuple]:
        """Children (the state and the verdict tracers) and aux (static fields).

        Returns
        -------
        tuple[tuple, tuple]
            ``(children, aux)``.
        """
        return (
            (self.prepared, self.refresh_ok, self.walk_needs, self.tree_overflow),
            (bool(self.fresh), tuple(self.caps)),
        )

    @classmethod
    def tree_unflatten(cls, aux: tuple, children: Any) -> "FusedTopology":
        """Rebuild from :meth:`tree_flatten`'s output.

        Parameters
        ----------
        aux : tuple
            ``(fresh, caps)``.
        children : Any
            ``(prepared, refresh_ok, walk_needs, tree_overflow)``.

        Returns
        -------
        FusedTopology
            The topology.
        """
        return _construct(cls, [*children, *aux])

    @classmethod
    def from_prepared(cls, prepared: Any) -> "FusedTopology":
        """A topology from an existing prepared state (an eager prepare's).

        Parameters
        ----------
        prepared : Any
            A ``LargeNPreparedState``.

        Returns
        -------
        FusedTopology
            No rebuild verdict: ``fresh=False``.
        """
        return cls(prepared=prepared)

    def needs(self) -> Array:
        """What the rebuild's walk needed, ``int32[7]`` (zeros when unreported).

        Returns
        -------
        Array
            ``capacity_guard.WALK_NEEDS_FIELDS`` in order.
        """
        if self.walk_needs is None:
            return jnp.zeros((len(WALK_NEEDS_FIELDS),), jnp.int32)
        return jnp.asarray(self.walk_needs, jnp.int32)

    def overflow_bits(self, extra: Union[int, Array] = 0) -> Array:
        """Every fixed capacity of this topology, one bit each (``OVERFLOW_*``).

        ``overflow_ok(bits)`` is the verdict (bits 3-10); bits 0-2 name what the
        walk overflowed. It is :meth:`FusedForceProvider.capacity_ok`'s verdict and
        the tree's (bit 3), which ``capacity_ok`` -- the strict scan's verdict,
        kept bitwise -- does not see on the cell-leaf rebuild.

        Parameters
        ----------
        extra : Union[int, Array]
            Bits the caller adds (``OVERFLOW_LEVEL_PLAN``, ``OVERFLOW_CROSS``).

        Returns
        -------
        Array
            ``uint32`` scalar.
        """

        def bit(flag: Any, value: int) -> Array:
            return jnp.where(jnp.asarray(flag, bool), jnp.uint32(value), jnp.uint32(0))

        prepared = self.prepared
        bits = jnp.asarray(extra, jnp.uint32)
        if self.walk_needs is not None:
            needs = self.needs()
            bits = bits | bit(needs[3] != 0, OVERFLOW_WALK_FAR)
            bits = bits | bit(needs[4] != 0, OVERFLOW_WALK_NEAR)
            bits = bits | bit(needs[5] != 0, OVERFLOW_WALK_QUEUE)
        if self.tree_overflow is not None:
            bits = bits | bit(self.tree_overflow, OVERFLOW_TREE)
        saturated = jnp.asarray(False)
        if self.fresh and self.refresh_ok is not None:
            saturated = saturated | ~jnp.asarray(self.refresh_ok, bool)
        far_pairs = getattr(prepared, "compact_far_pairs", None)
        far_count = getattr(far_pairs, "far_pair_count", None)
        sources = getattr(far_pairs, "sources", None)
        if far_count is not None and sources is not None:
            width = int(jnp.asarray(sources).shape[0])
            if width > 0:
                saturated = saturated | (jnp.asarray(far_count) >= width)
        bits = bits | bit(saturated, OVERFLOW_SATURATED)
        offsets = jnp.asarray(prepared.neighbor_list.offsets)
        counts = offsets[1:] - offsets[:-1]
        nbr_cap, far_cap, rectangle = self.caps
        if nbr_cap is not None and int(counts.shape[0]) > 0:
            over = jnp.max(counts) >= jnp.asarray(int(nbr_cap), counts.dtype)
            bits = bits | bit(over, OVERFLOW_NEIGHBOR_ROW)
        if far_cap is not None and far_count is not None:
            over = jnp.asarray(far_count) >= int(far_cap)
            bits = bits | bit(over, OVERFLOW_FAR_CAP)
        padded = getattr(
            prepared, "nearfield_target_block_source_leaf_ids_padded", None
        )
        if rectangle and padded is not None:
            padded_arr = jnp.asarray(padded)
            if padded_arr.ndim == 3 and int(padded_arr.shape[1]) > 0:
                capacity = int(padded_arr.shape[1]) * int(padded_arr.shape[2])
                over = jnp.any(counts > jnp.asarray(capacity, counts.dtype))
                bits = bits | bit(over, OVERFLOW_RECTANGLE)
        return bits

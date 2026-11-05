"""Forward mode with the tangent directions processed in groups (``JACCPOT_FORWARD_TANGENT_GROUP``).

Why this exists
---------------
``jax.jacfwd`` pushes all ``K`` directions through the program at once. Every
tangent op folds them into its own tangent axis (:mod:`jaccpot.pallas._tangent`), so
every tangent intermediate of one force evaluation is ``K``-stacked: at the M2L the
multipole-tangent table is read while the local-tangent table is written, two
``f32[K, nodes, (p + 1)^2]`` tables live at once -- 2 x 196 nu B/p per direction at
order 6 (nu = node slots per particle: 0.30 at 2e6, 0.23 at 8e6), the largest part
of forward mode's per-direction memory (F8; round 2 track M).

:func:`grouped_jvp` gives one force evaluation a ``jax.custom_jvp`` whose rule

1. **linearizes** the force once (``jax.linearize``): the primal runs ONCE, with the
   production kernels exactly as without grouping (each seam's rule computes its
   primal through the seam), and the tangent part becomes a linear jaxpr whose
   residuals are the primal intermediates the tangent ops read (the multipole and
   local tables, the sorted positions, ...);
2. **applies** that linear jaxpr through one primitive, ``jaccpot_tangent_grouped``,
   whose batching rule folds ``jax.jacfwd``'s vmapped direction axis into an
   explicit ``K`` (the pattern of :func:`~jaccpot.pallas._tangent.make_tangent_op`)
   and whose implementation loops over groups of ``g`` directions: each iteration
   is the linear jaxpr ``vmap``-ped over ``g`` directions, so every tangent op inside
   runs at ``K = g`` and only ``g`` directions' tangent intermediates are live. A
   partial last group runs once more after the loop at its own size (no padded
   copy of the tangents).

Unlike ``directions_per_pass`` (:mod:`jaccpot.runtime.forward_batching`), which
reruns the PRIMAL per chunk of directions, nothing is recomputed: the loop body is
tangent work only. What stays ``K``-stacked is what crosses the boundary: the
tangents of the force's inputs (positions, masses, ``G``, ``eps2``) and of its
output (the accelerations).

A ``lax.map`` inside a ``custom_jvp`` rule would not do this: under ``jax.jacfwd``
the rule itself is vmapped, so ``K`` is invisible to it. ``K`` is explicit only in a
primitive's batching rule -- hence the primitive.

Where it applies
----------------
``FMMEngine._differentiable_accelerations_large_n`` (the fused provider's
:meth:`~jaccpot.runtime.fused_provider.FusedForceProvider.differentiable_step` and
everything built on it: ``differentiable_rollout(..., ad_mode="forward")``, the
gradient plan's forward twins), when forward mode is on
(:func:`~jaccpot.runtime.grad_options.forward_mode_enabled`) and
``JACCPOT_FORWARD_TANGENT_GROUP`` is set to ``g >= 1``. Unset (or ``0``), the
program is the ungrouped one, byte for byte. The boundary is the whole force
evaluation (sorting gather, upward and downward sweeps, near field, inverse
permutation): every tangent intermediate of the force is then per group, not only
the far field's node tables.

Contract
--------
* The primal is the ungrouped program's (the same ops, computed once).
* The tangents equal the ungrouped program's to round-off (the same tangent ops
  at ``K = g`` instead of ``K``; the kernels' tangent groups are independent).
* ``jax.jvp`` (one direction) runs one group of one.
* First order only: a jvp of the grouped tangent reaches the primitive's refusal;
  ``jax.grad`` its transpose refusal.
* An outer ``vmap`` over the PRIMALS (systems, not directions) runs the members
  one by one, as the tangent ops do.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

# the public spelling (jax.extend.core has no eval_jaxpr); pyright misses the module
_eval_jaxpr = jax.core.eval_jaxpr  # pyright: ignore[reportAttributeAccessIssue]

Array = Any

#: Environment knob: process forward mode's tangent directions in groups of this
#: many (unset or ``0``: off, the ungrouped program).
ENV_FORWARD_TANGENT_GROUP = "JACCPOT_FORWARD_TANGENT_GROUP"

__all__ = [
    "ENV_FORWARD_TANGENT_GROUP",
    "forward_tangent_group",
    "grouped_jvp",
]


def forward_tangent_group() -> Optional[int]:
    """The tangent group size ``g`` forward mode runs with, or ``None`` (off).

    Read at trace time. Off unless forward mode is on
    (:func:`~jaccpot.runtime.grad_options.forward_mode_enabled`) and
    ``JACCPOT_FORWARD_TANGENT_GROUP`` holds an integer ``>= 1``.

    Returns
    -------
    Optional[int]
        ``g``, or ``None`` when grouping is off.

    Raises
    ------
    ValueError
        If the variable is set to something other than a non-negative integer.
    """
    from .grad_options import forward_mode_enabled

    raw = os.environ.get(ENV_FORWARD_TANGENT_GROUP)
    if raw is None or not str(raw).strip():
        return None
    try:
        group = int(str(raw).strip())
    except ValueError:
        group = -1
    if group < 0:
        raise ValueError(
            f"{ENV_FORWARD_TANGENT_GROUP} must be a non-negative integer (0: off), "
            f"got {raw!r}"
        )
    if group == 0 or not forward_mode_enabled():
        return None
    return group


def _differentiable(leaf: Any) -> bool:
    """Whether a pytree leaf is an inexact array (a differentiated input).

    Parameters
    ----------
    leaf : Any
        A leaf.

    Returns
    -------
    bool
        ``True`` for floating / complex arrays and tracers.
    """
    dtype = getattr(leaf, "dtype", None)
    return (
        dtype is not None
        and hasattr(leaf, "shape")
        and jnp.issubdtype(dtype, jnp.inexact)
    )


def grouped_jvp(fn: Callable[..., Any], args: Sequence[Any], *, group: int) -> Any:
    """``fn(*args)``, whose forward-mode tangent runs in groups of ``group`` directions.

    Parameters
    ----------
    fn : Callable[..., Any]
        A function of ``args`` (pytrees). Its inexact array leaves are the
        differentiated inputs; every other leaf is held fixed (closed over).
    args : Sequence[Any]
        The arguments.
    group : int
        Directions per group (``>= 1``).

    Returns
    -------
    Any
        ``fn(*args)``, with a ``custom_jvp`` (module docstring).

    Raises
    ------
    ValueError
        If ``group < 1``.
    """
    group = int(group)
    if group < 1:
        raise ValueError(f"the tangent group holds >= 1 directions, got {group}")
    leaves, tree = jax.tree_util.tree_flatten(tuple(args))
    diff_idx = [i for i, leaf in enumerate(leaves) if _differentiable(leaf)]

    def call(*dleaves: Array) -> Any:
        full = list(leaves)
        for i, leaf in zip(diff_idx, dleaves):
            full[i] = leaf
        return fn(*jax.tree_util.tree_unflatten(tree, full))

    if not diff_idx:
        return call()
    wrapped = jax.custom_jvp(call)
    wrapped.defjvp(functools.partial(_grouped_rule, call, group), symbolic_zeros=True)
    return wrapped(*[leaves[i] for i in diff_idx])


def _grouped_rule(
    call: Callable[..., Any], group: int, primals: tuple, tangents: tuple
) -> tuple[Any, Any]:
    """The ``custom_jvp`` rule: linearize once, apply the tangent map in groups.

    Parameters
    ----------
    call : Callable[..., Any]
        The function of the differentiated leaves.
    group : int
        Directions per group.
    primals : tuple
        The differentiated leaves.
    tangents : tuple
        Their tangents (:class:`SymbolicZero` for a known zero).

    Returns
    -------
    tuple[Any, Any]
        ``(primal output, tangent output)``.
    """
    live = [i for i, t in enumerate(tangents) if not isinstance(t, SymbolicZero)]
    if not live:
        out = call(*primals)
        return out, jax.tree_util.tree_map(_zero_tangent, out)

    def of_live(*xs: Array) -> Any:
        full = list(primals)
        for i, x in zip(live, xs):
            full[i] = x
        return call(*full)

    out, lin = jax.linearize(of_live, *[primals[i] for i in live])
    t_live = [tangents[i] for i in live]
    return out, _apply_grouped(lin, t_live, group)


def _zero_tangent(x: Array) -> Array:
    """A dense zero tangent of ``x`` (``float0`` for an integer output).

    Parameters
    ----------
    x : Array
        A primal output.

    Returns
    -------
    Array
        Its zero tangent.
    """
    if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        return jnp.zeros(jnp.shape(x), jnp.result_type(x))
    import numpy as np

    return np.zeros(jnp.shape(x), jax.dtypes.float0)


def _depends(jaxpr: Any) -> list[bool]:
    """Which outputs of ``jaxpr`` depend on its (non-const) inputs.

    Conservative: an equation with any dependent input makes all its outputs
    dependent.

    Parameters
    ----------
    jaxpr : Any
        An open jaxpr (``constvars`` are not inputs).

    Returns
    -------
    list[bool]
        One flag per output.
    """
    from jax.extend.core import Literal

    dep = {id(v) for v in jaxpr.invars}
    for eqn in jaxpr.eqns:
        if any(not isinstance(v, Literal) and id(v) in dep for v in eqn.invars):
            dep.update(id(v) for v in eqn.outvars)
    return [not isinstance(v, Literal) and id(v) in dep for v in jaxpr.outvars]


def _apply_grouped(lin: Callable[..., Any], tangents: list, group: int) -> Any:
    """``lin(*tangents)`` through :data:`grouped_tangent_p`.

    Parameters
    ----------
    lin : Callable[..., Any]
        The linear map (``jax.linearize``'s), closing over the residuals.
    tangents : list
        One direction's tangents (the rule's; ``jax.jacfwd`` vmaps them).
    group : int
        Directions per group.

    Returns
    -------
    Any
        The output tangent, with ``lin``'s output structure.
    """
    specs = [jax.ShapeDtypeStruct(jnp.shape(t), jnp.result_type(t)) for t in tangents]
    closed, out_shape = jax.make_jaxpr(lin, return_shape=True)(*specs)
    out_tree = jax.tree_util.tree_structure(out_shape)
    jaxpr, consts = closed.jaxpr, list(closed.consts)
    if jaxpr.effects:
        # an effectful tangent (a callback) is not looped over: ungrouped
        return lin(*tangents)
    dep = _depends(jaxpr)
    out_avals = [v.aval for v in jaxpr.outvars]
    ts = [jnp.asarray(t)[None] for t in tangents]
    outs = (
        grouped_tangent_p.bind(
            *consts,
            *ts,
            jaxpr=jaxpr,
            num_consts=len(consts),
            group=int(group),
            out_dep=tuple(dep),
        )
        if any(dep)
        else []
    )
    it = iter(outs)
    flat = [
        next(it)[0] if d else jnp.zeros(a.shape, a.dtype)  # pyright: ignore
        for d, a in zip(dep, out_avals)
    ]
    return jax.tree_util.tree_unflatten(out_tree, flat)


# --------------------------------------------------------------------------
# The primitive: K explicit, a loop over groups of directions.
# --------------------------------------------------------------------------

grouped_tangent_p = Primitive("jaccpot_tangent_grouped")
grouped_tangent_p.multiple_results = True


def _out_avals(jaxpr: Any, out_dep: Sequence[bool]) -> list[Any]:
    return [v.aval for v, d in zip(jaxpr.outvars, out_dep) if d]


def _grouped_impl(
    *operands: Array,
    jaxpr: Any,
    num_consts: int,
    group: int,
    out_dep: Sequence[bool],
) -> list[Array]:
    """The linear jaxpr over ``K`` directions, ``group`` at a time.

    Parameters
    ----------
    *operands : Array
        The residuals (``num_consts``), then the tangents (leading axis ``K``).
    jaxpr : Any
        The linear jaxpr (its ``constvars`` are the residuals).
    num_consts : int
        How many leading operands are residuals.
    group : int
        Directions per group.
    out_dep : Sequence[bool]
        Which of the jaxpr's outputs are returned (those depending on a tangent).

    Returns
    -------
    list[Array]
        The returned outputs, each with the leading ``K`` axis.
    """
    consts = list(operands[:num_consts])
    ts = list(operands[num_consts:])
    k = int(jnp.shape(ts[0])[0])

    def one(*t: Array) -> list[Array]:
        outs = _eval_jaxpr(jaxpr, consts, *t)
        return [o for o, d in zip(outs, out_dep) if d]

    batched = jax.vmap(one)
    g = max(1, min(int(group), k))
    full, rem = divmod(k, g)
    if full <= 1 and rem == 0:
        return list(batched(*ts))
    avals = _out_avals(jaxpr, out_dep)
    init = [jnp.zeros((k,) + tuple(a.shape), a.dtype) for a in avals]

    def body(i: Array, outs: list[Array]) -> list[Array]:
        start = i * g
        chunk = [jax.lax.dynamic_slice_in_dim(t, start, g, axis=0) for t in ts]
        res = batched(*chunk)
        return [
            jax.lax.dynamic_update_slice_in_dim(o, r.astype(o.dtype), start, axis=0)
            for o, r in zip(outs, res)
        ]

    outs = jax.lax.fori_loop(0, full, body, init) if full else init
    if rem:
        start = full * g
        res = batched(*[t[start:] for t in ts])
        outs = [
            jax.lax.dynamic_update_slice_in_dim(o, r.astype(o.dtype), start, axis=0)
            for o, r in zip(outs, res)
        ]
    return list(outs)


def _grouped_abstract(
    *avals: Any,
    jaxpr: Any,
    num_consts: int,
    group: int,
    out_dep: Sequence[bool],
) -> list[Any]:
    del group
    k = int(avals[num_consts].shape[0])
    return [a.update(shape=(k,) + tuple(a.shape)) for a in _out_avals(jaxpr, out_dep)]


def _grouped_batch(
    args: Sequence[Array],
    dims: Sequence[Optional[int]],
    *,
    jaxpr: Any,
    num_consts: int,
    group: int,
    out_dep: Sequence[bool],
) -> tuple[list[Array], list[Any]]:
    """Fold a vmapped direction axis into ``K`` (residuals unbatched).

    Batched residuals (a vmap over systems, not directions) run the members one by
    one, as :func:`~jaccpot.pallas._tangent.make_tangent_op`'s batching rule does.

    Parameters
    ----------
    args : Sequence[Array]
        The operands.
    dims : Sequence[Optional[int]]
        Their batch dimensions.
    jaxpr : Any
        The linear jaxpr.
    num_consts : int
        How many leading operands are residuals.
    group : int
        Directions per group.
    out_dep : Sequence[bool]
        The returned outputs.

    Returns
    -------
    tuple[list[Array], list[Any]]
        The batched outputs and their batch dimensions (0).
    """
    params = dict(jaxpr=jaxpr, num_consts=num_consts, group=group, out_dep=out_dep)
    size = next(int(jnp.shape(a)[d]) for a, d in zip(args, dims) if d is not None)
    if any(d is not None for d in dims[:num_consts]):
        mapped = [i for i, d in enumerate(dims) if d is not None]
        moved = [
            jnp.moveaxis(a, d, 0) if d is not None else a for a, d in zip(args, dims)
        ]

        def member(xs: Sequence[Array]) -> tuple[Array, ...]:
            full = list(moved)
            for i, x in zip(mapped, xs):
                full[i] = x
            return tuple(grouped_tangent_p.bind(*full, **params))

        outs = jax.lax.map(member, [moved[i] for i in mapped])
        return list(outs), [0] * len(outs)
    consts = list(args[:num_consts])
    folded = []
    k = 1
    for a, d in zip(args[num_consts:], dims[num_consts:]):
        t = (
            jnp.broadcast_to(a, (size,) + tuple(a.shape))
            if d is None
            else jnp.moveaxis(a, d, 0)
        )
        k = int(t.shape[1])
        folded.append(jnp.reshape(t, (size * k,) + tuple(t.shape[2:])))
    outs = grouped_tangent_p.bind(*consts, *folded, **params)
    return (
        [jnp.reshape(o, (size, k) + tuple(o.shape[1:])) for o in outs],
        [0] * len(outs),
    )


def _refuse_transpose(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "jaccpot_tangent_grouped is a forward-mode (ad_mode='forward') tangent "
        "map and has no transpose. Build reverse-mode gradients (jax.grad / "
        "jax.vjp) with ad_mode='reverse', the default."
    )


def _refuse_jvp(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "jaccpot_tangent_grouped: second-order forward mode (a jvp of a "
        "forward-mode tangent) is not supported; the tangent kernels are first "
        "order only."
    )


grouped_tangent_p.def_impl(_grouped_impl)
grouped_tangent_p.def_abstract_eval(_grouped_abstract)
mlir.register_lowering(
    grouped_tangent_p, mlir.lower_fun(_grouped_impl, multiple_results=True)
)
batching.primitive_batchers[grouped_tangent_p] = _grouped_batch
ad.primitive_transposes[grouped_tangent_p] = _refuse_transpose
ad.primitive_jvps[grouped_tangent_p] = _refuse_jvp

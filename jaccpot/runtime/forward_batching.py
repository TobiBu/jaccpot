"""Forward-mode Jacobians in chunks of directions, and the forward-mode memory model.

``jax.jacfwd`` pushes EVERY direction through the program at once: the tangent
kernels carry all ``K`` directions on their tangent axis, so the device memory is
``W(K) = a + b K`` per particle (the F8 fit: ``b`` ~ 120-195 B/p per direction).
At ``N = 1e8`` on an A100-40 one direction (``K = 1``) already peaks at 384 B/p
(35.8 GiB) -- a Jacobian over 7 parameters does not fit in one pass.

:func:`jacfwd_batched` is ``jax.jacfwd`` with ``directions_per_pass=k``: ONE jitted
program that runs a ``lax.map`` over chunks of ``k`` directions, each chunk a
``jax.jvp`` vmapped over its ``k`` tangent columns (so the tangent kernels see
``K = k``), the primal recomputed inside every chunk. The last chunk is padded with
zero directions and the padding sliced off. The memory is then ``W(k)``, not
``W(K)``; the price is the primal once per chunk (``ceil(K / k)`` primals instead of
one), plus one more primal AFTER the loop for the returned value (so the loop
carries nothing ``N``-sized: at 1e8 a carried final state would be ~4 GiB). An
``optimization_barrier`` orders that primal after the loop.

``directions_per_pass=None`` is ``jax.jacfwd`` itself (the same program).

:func:`apply_in_chunks` maps a function over a leading axis of directions in chunks
of ``k`` the same way (``GradientPlan.linearize``'s map applied to a matrix of
directions).

:func:`forward_memory_estimate` is the F8 model (``bench/forward_mode_memory.py``,
round 1 on an A100-40, galaxy preset, float32): the peak bytes of a forward-mode
program of ``k`` directions per pass at ``N``, and the largest ``k`` that fits a
card.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "F8_PEAK_BYTES_PER_PARTICLE",
    "apply_in_chunks",
    "forward_memory_estimate",
    "jacfwd_batched",
    "jacfwd_columns",
    "jacfwd_unravel",
]


def _check_directions(directions_per_pass: Optional[int]) -> Optional[int]:
    """``directions_per_pass`` validated: ``None`` or an int >= 1.

    Parameters
    ----------
    directions_per_pass : Optional[int]
        As :func:`jacfwd_batched`.

    Returns
    -------
    Optional[int]
        The value, an ``int``.

    Raises
    ------
    ValueError
        If it is not ``None`` and not a positive int.
    """
    if directions_per_pass is None:
        return None
    if isinstance(directions_per_pass, bool) or not isinstance(
        directions_per_pass, (int, np.integer)
    ):
        raise ValueError(
            f"directions_per_pass must be None or an int >= 1, got "
            f"{directions_per_pass!r}"
        )
    k = int(directions_per_pass)
    if k < 1:
        raise ValueError(f"directions_per_pass must be >= 1, got {k}")
    return k


def _basis_column(leaves: list, j: Any) -> list:
    """Direction ``j`` of the standard basis over ``leaves`` (zero if ``j >= D``).

    Parameters
    ----------
    leaves : list
        The differentiated inputs' leaves.
    j : Any
        The direction index (traced int).

    Returns
    -------
    list
        One tangent per leaf, the leaf's shape and dtype: one-hot at flat index
        ``j`` of the concatenated leaves, all zero for a padding index.
    """
    out = []
    offset = 0
    for leaf in leaves:
        size = int(np.size(leaf))
        hot = (jnp.arange(size) + offset) == j
        out.append(jnp.reshape(hot.astype(jnp.result_type(leaf)), jnp.shape(leaf)))
        offset += size
    return out


def jacfwd_columns(
    push: Callable[[Any], Any], x: Any, directions_per_pass: int
) -> tuple[Any, int]:
    """The Jacobian's columns, ``directions_per_pass`` per pass of ``push``.

    Parameters
    ----------
    push : Callable[[Any], Any]
        ``push(tangent) -> output tangent``: a ``jax.jvp`` of the function at ``x``
        along one direction (the tangent ``x``'s structure).
    x : Any
        The point (a pytree of float arrays).
    directions_per_pass : int
        Directions per pass ``k``.

    Returns
    -------
    tuple[Any, int]
        ``(columns, D)``: ``columns`` the output tangents' structure, each leaf
        ``[D, *out_shape]`` (direction ``j`` the flat index into ``x``'s
        concatenated leaves); ``D`` the number of directions.

    Raises
    ------
    TypeError
        If a leaf of ``x`` is not a real floating array (as ``jax.jacfwd``).
    """
    k = _check_directions(directions_per_pass)
    assert k is not None
    leaves, treedef = jax.tree_util.tree_flatten(x)
    for leaf in leaves:
        if not jnp.issubdtype(jnp.result_type(leaf), jnp.floating):
            raise TypeError(
                "jacfwd requires real-valued inputs, got dtype "
                f"{jnp.result_type(leaf)}"
            )
    total = int(sum(int(np.size(leaf)) for leaf in leaves))
    chunks = max(1, -(-total // k))

    def chunk(c: Any) -> Any:
        js = c * k + jnp.arange(k)
        tangents = jax.vmap(
            lambda j: jax.tree_util.tree_unflatten(treedef, _basis_column(leaves, j))
        )(js)
        return jax.vmap(push)(tangents)

    stacked = jax.lax.map(chunk, jnp.arange(chunks))
    columns = jax.tree_util.tree_map(
        lambda s: jnp.reshape(s, (chunks * k,) + tuple(s.shape[2:]))[:total], stacked
    )
    return columns, total


def jacfwd_unravel(x: Any, y: Any, columns: Any) -> Any:
    """``jax.jacfwd``'s Jacobian structure from :func:`jacfwd_columns`' columns.

    Parameters
    ----------
    x : Any
        The differentiated inputs.
    y : Any
        The function's output (its structure and dtypes).
    columns : Any
        ``y``'s structure, each leaf ``[D, *out_shape]``.

    Returns
    -------
    Any
        ``y``'s structure whose leaves are ``x``'s structure, each leaf
        ``[*out_shape, *in_shape]`` in the output leaf's dtype: what
        ``jax.jacfwd`` returns.
    """
    leaves, treedef = jax.tree_util.tree_flatten(x)
    sizes = [int(np.size(leaf)) for leaf in leaves]
    bounds = np.cumsum([0] + sizes)

    def one(out_leaf: Any, col: Any) -> Any:
        moved = jnp.moveaxis(col, 0, -1)
        out_shape = tuple(jnp.shape(out_leaf))
        dtype = jnp.result_type(out_leaf)
        parts = [
            jnp.reshape(
                moved[..., int(lo) : int(hi)], out_shape + tuple(jnp.shape(leaf))
            ).astype(dtype)
            for leaf, lo, hi in zip(leaves, bounds[:-1], bounds[1:])
        ]
        return jax.tree_util.tree_unflatten(treedef, parts)

    return jax.tree_util.tree_map(one, y, columns)


def jacfwd_batched(
    fun: Callable[..., Any],
    *,
    directions_per_pass: Optional[int] = None,
    has_aux: bool = False,
) -> Callable[[Any], Any]:
    """``jax.jacfwd`` of ``fun`` (first argument), ``directions_per_pass`` at a time.

    Use it on a forward-mode program (``differentiable_rollout(...,
    ad_mode="forward")`` or a function traced under ``ad_mode_override("forward")``)
    whose ``jax.jacfwd`` would not fit a card: the device memory is that of
    ``directions_per_pass`` tangents, not of all of them (module docstring). Call
    it inside ``jax.jit``.

    Parameters
    ----------
    fun : Callable[..., Any]
        ``fun(x) -> y`` (``(y, aux)`` with ``has_aux``); ``x`` a pytree of real
        float arrays.
    directions_per_pass : Optional[int]
        Directions (Jacobian columns) per pass; ``None``: ``jax.jacfwd`` itself,
        every direction in one pass.
    has_aux : bool
        As ``jax.jacfwd``.

    Returns
    -------
    Callable[[Any], Any]
        ``jac(x)``: ``jax.jacfwd(fun, has_aux=has_aux)(x)``'s result (to
        round-off: the tangent kernels may sum in another order at another
        ``K``). With ``has_aux`` the aux is the primal's, computed once more
        after the loop.

    Raises
    ------
    ValueError
        If ``directions_per_pass`` is not ``None`` or a positive int.
    """
    k = _check_directions(directions_per_pass)
    if k is None:
        return jax.jacfwd(fun, has_aux=has_aux)

    def jac(x: Any) -> Any:
        def push(t: Any) -> Any:
            if has_aux:
                return jax.jvp(fun, (x,), (t,), has_aux=True)[1]
            return jax.jvp(fun, (x,), (t,))[1]

        columns, _ = jacfwd_columns(push, x, k)
        # the returned value: a primal AFTER the loop (nothing N-sized carried)
        columns, x_after = jax.lax.optimization_barrier((columns, x))
        if has_aux:
            y, aux = fun(x_after)
            return jacfwd_unravel(x, y, columns), aux
        y = fun(x_after)
        return jacfwd_unravel(x, y, columns)

    return jac


def apply_in_chunks(
    fun: Callable[[Any], Any], directions: Any, directions_per_pass: Optional[int]
) -> Any:
    """``jax.vmap(fun)(directions)``, ``directions_per_pass`` directions at a time.

    Parameters
    ----------
    fun : Callable[[Any], Any]
        A function of one direction (a linear map, e.g.
        :meth:`~jaccpot.production_grad.GradientPlan.linearize`'s).
    directions : Any
        The directions, every leaf with a leading axis ``D``.
    directions_per_pass : Optional[int]
        ``k``: a ``lax.map`` over ``ceil(D / k)`` chunks, each ``vmap`` over
        ``k`` directions (the last padded with zeros, sliced off); ``None``: one
        ``vmap`` over all ``D``.

    Returns
    -------
    Any
        ``fun``'s outputs, every leaf with the leading axis ``D``.

    Raises
    ------
    ValueError
        If the leaves disagree on ``D`` or ``directions_per_pass`` is invalid.
    """
    k = _check_directions(directions_per_pass)
    leaves = jax.tree_util.tree_leaves(directions)
    ds = {int(jnp.shape(leaf)[0]) for leaf in leaves}
    if len(ds) != 1:
        raise ValueError(f"directions need one leading axis D, got {sorted(ds)}")
    total = ds.pop()
    if k is None or k >= total:
        return jax.vmap(fun)(directions)
    chunks = -(-total // k)
    pad = chunks * k - total

    def split(a: Any) -> Any:
        a = jnp.asarray(a)
        if pad:
            a = jnp.concatenate([a, jnp.zeros((pad,) + a.shape[1:], a.dtype)], axis=0)
        return jnp.reshape(a, (chunks, k) + a.shape[1:])

    out = jax.lax.map(jax.vmap(fun), jax.tree_util.tree_map(split, directions))
    return jax.tree_util.tree_map(
        lambda s: jnp.reshape(s, (chunks * k,) + s.shape[2:])[:total], out
    )


#: The F8 model's anchors (round 1, A100-40, galaxy preset at p = 6, float32,
#: ``dt = 1e-3``, the scanned rollout ``T = 16``): ``N -> (a, b)`` with the
#: allocator peak ``a + b k`` bytes per particle for ``k`` directions per pass. The
#: 2e6 / 8e6 / 2.5e7 rows are fits through ``K = 1`` and ``K = 8``; the 1e8 ``b``
#: is extrapolated (``b ~ N^-0.122`` from the three) and its ``a`` set so that
#: ``a + b`` is the measured ``K = 1`` peak (384.1 B/p).
F8_PEAK_BYTES_PER_PARTICLE: dict[int, tuple[float, float]] = {
    2_000_000: (369.0, 194.7),
    8_000_000: (307.8, 163.8),
    25_000_000: (266.1, 143.1),
    100_000_000: (263.2, 120.9),
}

#: an A100-40's allocator limit at ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.95``
#: (``bytes_limit`` of the F8 rows)
A100_40_BYTES_LIMIT = 40_286_289_920


def _interp_loglog(n: float, table: dict[int, tuple[float, float]], i: int) -> float:
    """Entry ``i`` of ``table`` at ``n``: log-log interpolation, end slopes beyond.

    Parameters
    ----------
    n : float
        Particle count.
    table : dict[int, tuple[float, float]]
        :data:`F8_PEAK_BYTES_PER_PARTICLE`.
    i : int
        0 (``a``) or 1 (``b``).

    Returns
    -------
    float
        The interpolated value.
    """
    xs = sorted(table)
    lx = [math.log(x) for x in xs]
    ly = [math.log(table[x][i]) for x in xs]
    t = math.log(max(float(n), 1.0))
    j = 0
    while j < len(xs) - 2 and t > lx[j + 1]:
        j += 1
    slope = (ly[j + 1] - ly[j]) / (lx[j + 1] - lx[j])
    return math.exp(ly[j] + slope * (t - lx[j]))


def forward_memory_estimate(
    n: int,
    k: int = 1,
    steps: int = 16,
    *,
    card_bytes: int = A100_40_BYTES_LIMIT,
) -> dict[str, Any]:
    """The F8 model's device-memory estimate of a forward-mode rollout program.

    ``peak(n, k) = n (a(n) + b(n) k)`` bytes for ``k`` directions per pass
    (:data:`F8_PEAK_BYTES_PER_PARTICLE`, log-log interpolated in ``n``): flat in
    the steps for a scanned rollout (``steps >= 2``). An ESTIMATE from one card
    type and one configuration (galaxy preset, p = 6, float32, the round-1
    kernels); measure before a production run (``bench/forward_mode_memory.py``).

    Parameters
    ----------
    n : int
        Particle count.
    k : int
        Directions per pass (``jvp``: 1; ``jacfwd``: the column count, or its
        ``directions_per_pass``).
    steps : int
        Rollout steps. ``1`` is not a scanned rollout: the model under-estimates
        it (the F8 ``T = 1`` rows were 1.4-1.8x the ``T = 16`` ones).
    card_bytes : int
        The allocator limit to fit (default an A100-40 at 0.95 preallocation).

    Returns
    -------
    dict[str, Any]
        ``bytes_per_particle``, ``peak_bytes``, ``fits`` (``peak_bytes <=
        card_bytes``), ``largest_k_that_fits`` (the largest ``k`` in 1..8 whose
        peak fits, ``0`` if none; a ``directions_per_pass`` to pass),
        ``a_bytes_per_particle``, ``b_bytes_per_particle_per_direction`` and
        ``steps_note``.

    Raises
    ------
    ValueError
        If ``n`` or ``k`` is not positive.
    """
    if int(n) < 1 or int(k) < 1:
        raise ValueError(f"n and k must be positive, got n={n}, k={k}")
    a = _interp_loglog(n, F8_PEAK_BYTES_PER_PARTICLE, 0)
    b = _interp_loglog(n, F8_PEAK_BYTES_PER_PARTICLE, 1)
    per = a + b * int(k)
    peak = int(math.ceil(per * int(n)))
    largest = 0
    for kk in range(1, 9):
        if (a + b * kk) * int(n) <= card_bytes:
            largest = kk
    return {
        "bytes_per_particle": per,
        "peak_bytes": peak,
        "fits": peak <= int(card_bytes),
        "largest_k_that_fits": largest,
        "a_bytes_per_particle": a,
        "b_bytes_per_particle_per_direction": b,
        "steps_note": (
            "scanned rollout: flat in steps"
            if int(steps) >= 2
            else "steps=1 is not scanned: expect 1.4-1.8x this"
        ),
    }

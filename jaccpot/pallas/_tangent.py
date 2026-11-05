"""Forward-mode (JVP) seams for the Pallas kernels: tangent ops as JAX primitives.

Why this exists
---------------
Every fast kernel on the gradient path is a ``jax.custom_vjp`` (its reverse is a
hand-written Pallas kernel), and JAX refuses ``jax.jvp`` of a ``custom_vjp``
outright. ``pallas_call``'s own JVP rule is no substitute: it re-traces the kernel
body without a grid context (so ``pl.program_id`` asserts), it refuses
``input_output_aliases``, scratch operands and atomics, and where it does work it
emits ONE kernel computing primal and tangent together -- which ``jax.linearize``,
``jax.jacfwd`` and ``jax.grad`` cannot split (measured on jax 0.9.0 and 0.11.2,
2026-11-03).

Forward mode is therefore a second dispatch (``ad_mode="forward"``) next to the
untouched reverse one. Each production seam gets a ``jax.custom_jvp`` sibling whose
rule computes

* the **primal** with the production kernel, from the primals alone -- so the
  primal stays bitwise and ``jax.linearize`` can split it off; and
* the **tangent** with a *tangent op* built here: a ``jax.extend`` primitive whose
  lowering is a Pallas tangent kernel.

The tangent op is a primitive rather than a plain function for three reasons:

1. **Batching.** ``jax.jacfwd`` is ``vmap`` of ``jvp``. The primitive's batching
   rule folds a vmapped tangent axis into the op's own leading tangent axis ``K``,
   so a Jacobian over a handful of parameters is ONE tangent launch per seam (the
   kernel shares its loads and primal intermediates across the ``K`` tangents)
   instead of one ``pallas_call`` batching rule's grid replica per column.
2. **A clear refusal under reverse mode.** ``jax.grad`` of a forward-mode program
   reaches the op's transpose rule, which says what to do instead of failing deep
   inside Pallas.
3. **An upgrade path.** Pointing the transpose rule at the existing reverse kernel
   would make the forward seams reverse-differentiable too, without touching the
   reverse dispatch.

Contract of a tangent op
------------------------
``op(*primals, *tangents, **params) -> tuple[Array, ...]``

* ``primals`` -- the first ``num_primals`` operands, exactly what the primal
  kernel reads, topology (integer arrays) included.
* ``tangents`` -- one operand per differentiable input, each with a leading
  tangent axis ``K`` (the same ``K`` for all).
* ``params`` -- static keyword parameters (hashable: ints, bools, strings,
  tuples), passed to the kernel implementation unchanged.
* returns -- the tangent outputs, each with the leading ``K`` axis.

The kernel implementation sees ``K`` only from :data:`TANGENT_K_BUCKETS` (padded
with zero tangents, chunked above the largest), so each seam compiles at most four
tangent kernels however many columns ``jacfwd`` asks for.

Contract of a forward seam (the ``custom_jvp`` rule)
----------------------------------------------------
::

    @jax.custom_jvp
    def seam(x, m):
        return primal_kernel(x, m)

    @seam.defjvp
    def _seam_jvp(primals, tangents):
        out = seam(*primals)                       # NOT primal_kernel(*primals)
        tk, live = with_tangent_axis(tangents, primals)
        (t,) = drop_tangent_axis(op(*primals, *tk, live=live))
        return out, t

The rule computes the primal **through the seam itself**. Under a nested
transformation (a jvp of a forward-mode program) the outer derivative then reaches
the seam again and the tangent op's own refusal, instead of ``pallas_call``'s
generic JVP rule asserting on ``program_id``. Measured on a toy kernel: jvp,
``jit(jvp)``, ``jacfwd`` over 11 columns (one ``K = 8`` and one ``K = 4`` launch),
``linearize`` over three directions, ``vmap`` over primals, the primal bitwise, and
the refusals of ``grad`` and of second order
(``tests/unit/operators/test_tangent_op.py``).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.custom_derivatives import SymbolicZero
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

Array = Any

#: Static tangent counts a tangent kernel is compiled for. A request for ``K``
#: tangents runs in chunks of :data:`MAX_TANGENT_K`, the last one padded with zero
#: tangents up to the next bucket.
TANGENT_K_BUCKETS: tuple[int, ...] = (1, 2, 4, 8)

#: The largest ``K`` one tangent-kernel launch carries.
MAX_TANGENT_K: int = TANGENT_K_BUCKETS[-1]

_REVERSE_HINT = (
    "Build reverse-mode gradients (jax.grad / jax.vjp) with ad_mode='reverse', "
    "the default: those programs use the custom_vjp kernels and their Pallas "
    "reverse kernels."
)


def tangent_k_bucket(k: int) -> int:
    """The compiled tangent count that carries ``k`` tangents (``1 <= k <= 8``).

    Parameters
    ----------
    k : int
        Tangents in one chunk.

    Returns
    -------
    int
        The smallest entry of :data:`TANGENT_K_BUCKETS` that is at least ``k``.

    Raises
    ------
    ValueError
        If ``k`` is not in ``[1, MAX_TANGENT_K]``.
    """
    k = int(k)
    for bucket in TANGENT_K_BUCKETS:
        if k <= bucket:
            if k < 1:
                break
            return bucket
    raise ValueError(f"a tangent chunk holds 1..{MAX_TANGENT_K} tangents, got {k}")


class TangentOp:
    """A forward-mode tangent op: a primitive lowering to a K-tangent kernel.

    Build one with :func:`make_tangent_op`; call it as
    ``op(*primals, *tangents, **params)``.

    Parameters
    ----------
    primitive : Primitive
        The registered primitive.
    num_primals : int
        How many leading operands are primals.
    """

    def __init__(self, primitive: Primitive, num_primals: int) -> None:
        self.primitive = primitive
        self.num_primals = int(num_primals)

    @property
    def name(self) -> str:
        """The primitive's name.

        Returns
        -------
        str
            ``jaccpot_tangent_<name>``.
        """
        return str(self.primitive.name)

    def __call__(self, *operands: Array, **params: Any) -> tuple[Array, ...]:
        """Bind the primitive.

        Parameters
        ----------
        *operands : Array
            ``num_primals`` primals, then the tangents (leading axis ``K``).
        **params : Any
            Static parameters of the kernel implementation.

        Returns
        -------
        tuple[Array, ...]
            The tangent outputs, each with the leading ``K`` axis.
        """
        _check_tangent_axis(operands[self.num_primals :], self.name)
        return tuple(self.primitive.bind(*operands, **params))


def _check_tangent_axis(tangents: Sequence[Array], name: str) -> int:
    if not tangents:
        raise ValueError(f"{name}: a tangent op needs at least one tangent operand")
    ks = {int(jnp.shape(t)[0]) if jnp.ndim(t) else -1 for t in tangents}
    if len(ks) != 1 or -1 in ks or 0 in ks:
        shapes = [tuple(jnp.shape(t)) for t in tangents]
        raise ValueError(
            f"{name}: every tangent operand needs the same leading tangent axis "
            f"K >= 1, got shapes {shapes}"
        )
    return ks.pop()


def _chunked(
    impl: Callable[..., Sequence[Array]], num_primals: int
) -> Callable[..., list[Array]]:
    """Run ``impl`` over the tangent axis in bucket-sized chunks.

    Parameters
    ----------
    impl : Callable[..., Sequence[Array]]
        The K-tangent kernel implementation (any ``K`` in the buckets).
    num_primals : int
        How many leading operands are primals.

    Returns
    -------
    Callable[..., list[Array]]
        ``f(*primals, *tangents, **params)`` for any ``K``.
    """

    @functools.wraps(impl)
    def run(*operands: Array, **params: Any) -> list[Array]:
        primals = operands[:num_primals]
        tangents = operands[num_primals:]
        k_total = int(tangents[0].shape[0])
        pieces: list[list[Array]] = []
        for start in range(0, k_total, MAX_TANGENT_K):
            k = min(MAX_TANGENT_K, k_total - start)
            kb = tangent_k_bucket(k)
            chunk = []
            for t in tangents:
                piece = t[start : start + k]
                if kb > k:
                    pad = jnp.zeros((kb - k,) + tuple(piece.shape[1:]), piece.dtype)
                    piece = jnp.concatenate([piece, pad], axis=0)
                chunk.append(piece)
            outs = impl(*primals, *chunk, **params)
            pieces.append([o[:k] for o in outs])
        if len(pieces) == 1:
            return list(pieces[0])
        return [jnp.concatenate(parts, axis=0) for parts in zip(*pieces)]

    return run


def make_tangent_op(
    name: str,
    impl: Callable[..., Sequence[Array]],
    *,
    num_primals: int,
    abstract_eval: Callable[..., Sequence[jax.ShapeDtypeStruct]] | None = None,
) -> TangentOp:
    """Register a tangent op whose lowering is ``impl``.

    Parameters
    ----------
    name : str
        Short name; the primitive is ``jaccpot_tangent_<name>``.
    impl : Callable[..., Sequence[Array]]
        ``impl(*primals, *tangents, **params)``: the tangent kernel. It is called
        with a ``K`` from :data:`TANGENT_K_BUCKETS` only, and returns the tangent
        outputs with that leading axis.
    num_primals : int
        How many leading operands are primals (topology included).
    abstract_eval : Callable[..., Sequence[jax.ShapeDtypeStruct]] | None
        ``abstract_eval(*operand_shape_dtypes, **params)`` -> the output shapes for
        the operands' ``K``. ``None``: ``jax.eval_shape`` of the chunked ``impl``
        (traces the kernel once more per bind; pass a closed form for big kernels).

    Returns
    -------
    TangentOp
        The callable op.
    """
    prim = Primitive(f"jaccpot_tangent_{name}")
    prim.multiple_results = True
    run = _chunked(impl, num_primals)

    prim.def_impl(run)

    def _abstract(*avals: Any, **params: Any) -> list[ShapedArray]:
        specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in avals]
        if abstract_eval is not None:
            outs = abstract_eval(*specs, **params)
        else:
            outs = jax.eval_shape(functools.partial(run, **params), *specs)
        return [ShapedArray(o.shape, o.dtype) for o in outs]

    prim.def_abstract_eval(_abstract)
    mlir.register_lowering(prim, mlir.lower_fun(run, multiple_results=True))

    def _batch(
        args: Sequence[Array], dims: Sequence[int | None], **params: Any
    ) -> tuple[list[Array], list[int]]:
        size = next(int(jnp.shape(a)[d]) for a, d in zip(args, dims) if d is not None)
        if any(d is not None for d in dims[:num_primals]):
            # Batched primals (a vmap over systems, not over tangents): the
            # primal kernel would be batched too. Run the members one by one.
            mapped = [i for i, d in enumerate(dims) if d is not None]
            moved = [
                jnp.moveaxis(a, d, 0) if d is not None else a
                for a, d in zip(args, dims)
            ]

            def one(xs: Sequence[Array]) -> tuple[Array, ...]:
                full = list(moved)
                for i, x in zip(mapped, xs):
                    full[i] = x
                return tuple(prim.bind(*full, **params))

            outs = jax.lax.map(one, [moved[i] for i in mapped])
            return list(outs), [0] * len(outs)
        primals = list(args[:num_primals])
        folded = []
        k = None
        for a, d in zip(args[num_primals:], dims[num_primals:]):
            t = (
                jnp.broadcast_to(a, (size,) + tuple(a.shape))
                if d is None
                else jnp.moveaxis(a, d, 0)
            )
            k = int(t.shape[1])
            folded.append(jnp.reshape(t, (size * k,) + tuple(t.shape[2:])))
        outs = prim.bind(*primals, *folded, **params)
        assert k is not None
        return (
            [jnp.reshape(o, (size, k) + tuple(o.shape[1:])) for o in outs],
            [0] * len(outs),
        )

    batching.primitive_batchers[prim] = _batch

    def _refuse_transpose(*_: Any, **__: Any) -> None:
        raise NotImplementedError(
            f"{prim.name} is a forward-mode (ad_mode='forward') tangent op and has "
            f"no transpose. {_REVERSE_HINT}"
        )

    def _refuse_jvp(*_: Any, **__: Any) -> None:
        raise NotImplementedError(
            f"{prim.name}: second-order forward mode (a jvp of a forward-mode "
            "tangent, e.g. jax.jacfwd(jax.jacfwd(f)) or jax.hessian) is not "
            "supported; the tangent kernels are first order only."
        )

    ad.primitive_transposes[prim] = _refuse_transpose
    ad.primitive_jvps[prim] = _refuse_jvp
    return TangentOp(prim, num_primals)


def instantiate_tangent(tangent: Any, primal: Array) -> tuple[Array, bool]:
    """A ``custom_jvp`` rule's tangent as a dense array, and whether it is live.

    Parameters
    ----------
    tangent : Any
        The tangent the rule received: an array, a :class:`SymbolicZero`
        (``symbolic_zeros=True``) or a ``float0`` zero (integer primals).
    primal : Array
        The matching primal.

    Returns
    -------
    tuple[Array, bool]
        ``(dense tangent in the primal's dtype, False if it was a zero)``.
    """
    if isinstance(tangent, SymbolicZero) or (
        hasattr(tangent, "dtype") and tangent.dtype == jax.dtypes.float0
    ):
        return jnp.zeros(jnp.shape(primal), jnp.result_type(primal)), False
    return tangent, True


def with_tangent_axis(
    tangents: Sequence[Any], primals: Sequence[Array]
) -> tuple[list[Array], tuple[bool, ...]]:
    """A rule's tangents as ``K = 1`` tangent-op operands.

    Inside a ``custom_jvp`` rule a tangent is one direction; ``jax.jacfwd`` batches
    the rule, and the op's batching rule turns that into ``K``. So the rule always
    binds ``K = 1``.

    Parameters
    ----------
    tangents : Sequence[Any]
        The rule's tangents for the differentiable inputs.
    primals : Sequence[Array]
        The matching primals.

    Returns
    -------
    tuple[list[Array], tuple[bool, ...]]
        ``(tangents with a leading axis of 1, which ones are live)``. The live
        mask is static and may be passed to the op as a parameter so a kernel can
        skip a zero input's terms.
    """
    dense = [instantiate_tangent(t, p) for t, p in zip(tangents, primals)]
    return [t[None] for t, _ in dense], tuple(live for _, live in dense)


def drop_tangent_axis(outputs: Sequence[Array]) -> tuple[Array, ...]:
    """The ``K = 1`` op outputs back as one direction each.

    Parameters
    ----------
    outputs : Sequence[Array]
        Tangent-op outputs with a leading axis of 1.

    Returns
    -------
    tuple[Array, ...]
        The outputs without the tangent axis.
    """
    return tuple(o[0] for o in outputs)


def linearize_lanes(
    body: Callable[..., Any], primals: Sequence[Array]
) -> tuple[Any, Callable[..., Any]]:
    """Inside a kernel: the body's primal once, and its linear map for the tangents.

    ``jax.linearize`` traced into the kernel: the primal intermediates are computed
    once, and every call of the returned map applies only the linear part -- the
    sharing that makes a ``K``-tangent kernel cheaper than ``K`` single-tangent
    launches.

    Parameters
    ----------
    body : Callable[..., Any]
        The per-lane body, pure ``jnp`` on the lane tiles.
    primals : Sequence[Array]
        Its primal inputs.

    Returns
    -------
    tuple[Any, Callable[..., Any]]
        ``(body(*primals), linear map: tangents -> output tangent)``.
    """
    out, lin = jax.linearize(body, *primals)
    return out, lin


__all__ = [
    "MAX_TANGENT_K",
    "TANGENT_K_BUCKETS",
    "TangentOp",
    "drop_tangent_axis",
    "instantiate_tangent",
    "linearize_lanes",
    "make_tangent_op",
    "tangent_k_bucket",
    "with_tangent_axis",
]

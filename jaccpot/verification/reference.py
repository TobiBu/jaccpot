"""The distributed reference force: an exact-order fp64 direct sum on a target subset.

The subsampled rel-L2 probes of the multi-GPU benches cannot be compared across
probe settings or device counts: the sample, and the order the reference is summed
in, change with them (a 41 % swing on one configuration). This module's reference
is the same number wherever it runs:

* **Targets by particle id.** :func:`select_targets` draws ``K`` ids with a seeded
  generator from ``0..N-1``: the set depends on ``(N, K, seed)`` only, never on the
  decomposition, the device count or the row order.
* **No float reduction anywhere.** Every source-target pair term (``m g d`` for
  the acceleration, ``-m psi`` for the potential) is computed in fp64 and rounded
  onto a fixed-point grid ``2^-s`` per target and component, where ``2^(85 - s)``
  bounds every term of that target and component (``s`` from the GLOBAL max term,
  a ``pmax``: max is exact in any order). The rounded integers (below ``2^85``)
  are split into three 30-bit limbs and summed as ``int64``. Integer addition is
  associative, so the sum over sources is the same bits in ANY order: one card,
  N cards, N processes, any row order, migrating or not. The limbs are carried
  once at the end on the host in Python integers and converted with one
  correctly-rounded ``float``.
* **Accuracy.** Each term is quantised to ``2^-86`` of the largest term, so the
  result is the exactly-rounded sum of the fp64 pair terms to within
  ``N * 2^-86 * max|term|``: far below an fp64 direct sum's own round-off. The
  terms themselves are the module's fp64 :func:`~jaccpot.softening.masked_pair_factors`
  (the production kernels' pair law, every kernel), so the reference equals
  ``jaccpot.runtime.reference.direct_sum`` to round-off (tested).
* **No gather of the particles.** Only the ``K`` targets' rows are gathered (an
  ``all_gather`` of ``[K, 4]`` per device with a found-count; every target must be
  found exactly once). The sources stay where they are: per device a scan over
  fixed-size chunks of its own rows, so the per-device program is the same on
  every device count (the chunk length depends on ``K`` only).

Use :func:`reference_force` on any flat (optionally ``P(axis)``-sharded) arrays,
:func:`score_force` to score a production force against it, and the runner
adapters :func:`score_single_result` / :func:`score_mesh_result` to score a
:class:`~jaccpot.production.ProductionRunner` result at its final state without
gathering it. The external field is not part of the reference: the single-card
adapter scores the self-gravity, the mesh adapter (whose carry is the TOTAL
acceleration) takes the field at the targets as a host callable.

Cost: two passes over all ``N * K`` pairs (the max, then the limbs) on the
devices that hold the sources, ~3x a plain fp64 direct sum. ``K = 4096`` at
``N = 1e8`` is ``4e11`` pair terms per pass; it is a verification step, not a
per-step diagnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaccpot.softening import (
    masked_pair_factors,
    resolve_softening_kernel,
    softening_params_np,
)

#: bits per limb; three limbs hold the scaled terms (below ``2^HEADROOM_BITS``)
LIMB_BITS = 30
#: every scaled term of a (target, component) is below ``2^HEADROOM_BITS``
HEADROOM_BITS = 85
#: pairs per chunk and device (``chunk = max(MIN_CHUNK, PAIRS_PER_CHUNK // K)``)
PAIRS_PER_CHUNK = 1 << 21
#: the smallest source chunk
MIN_CHUNK = 64
#: the reference's private mesh axis when the caller gives no mesh
_AXIS = "a4ref"
#: output columns: three acceleration components and the potential
_COLUMNS = 4

#: jitted programs by key (mesh devices, axis, shapes, kernel, chunk)
_PROGRAMS: dict = {}


@dataclass(frozen=True)
class ForceLaw:
    """The pair law the reference sums: ``a = G sum m g(r) (x_s - x_t)``.

    Attributes
    ----------
    G : float
        Gravitational constant.
    softening : float
        The Plummer-equivalent softening length (:mod:`jaccpot.softening`).
    kernel : str
        The softening kernel name.
    """

    G: float = 1.0
    softening: float = 0.0
    kernel: str = "ferrers3"

    def params(self) -> np.ndarray:
        """The ``(2,)`` fp64 pair-factor parameters (host-computed, device-independent).

        Returns
        -------
        np.ndarray
            :func:`~jaccpot.softening.softening_params_np`.
        """
        return softening_params_np(self.kernel, float(self.softening))

    def as_dict(self) -> dict:
        """JSON-able form.

        Returns
        -------
        dict
            ``G``, ``softening``, ``kernel``.
        """
        return {
            "G": float(self.G),
            "softening": float(self.softening),
            "kernel": self.kernel,
        }


def force_law_of(solver: Any, handle: Any = None) -> ForceLaw:
    """The pair law of a ``FastMultipoleMethod`` (and a run's handle).

    Parameters
    ----------
    solver : Any
        A ``FastMultipoleMethod``.
    handle : Any
        Optional run handle with a ``force_key`` ``(G, eps2)`` (the parameters a
        single-card run's self-gravity was computed with); it wins over the
        solver's when present.

    Returns
    -------
    ForceLaw
        ``G``, softening and kernel.
    """
    impl = getattr(solver, "_impl", solver)
    G = float(impl.G)
    eps = float(impl.softening)
    kernel = resolve_softening_kernel(getattr(impl, "softening_kernel", None))
    key = getattr(handle, "force_key", None) if handle is not None else None
    if key is not None:
        G = float(key[0])
        eps = math.sqrt(float(key[1]))
    return ForceLaw(G=G, softening=eps, kernel=kernel)


def select_targets(num_particles: int, num_targets: int, seed: int = 0) -> np.ndarray:
    """``K`` distinct particle ids, sorted: a function of ``(N, K, seed)`` only.

    Parameters
    ----------
    num_particles : int
        ``N``; ids are ``0..N-1``.
    num_targets : int
        ``K`` (clipped to ``N``).
    seed : int
        Generator seed.

    Returns
    -------
    np.ndarray
        ``int64[K]``, ascending.

    Raises
    ------
    ValueError
        If ``num_targets`` is not positive.
    """
    n, k = int(num_particles), min(int(num_targets), int(num_particles))
    if k <= 0:
        raise ValueError(f"num_targets must be positive, got {num_targets}")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=k, replace=False).astype(np.int64))


def _chunk_for(num_targets: int) -> int:
    """The source chunk length: a function of ``K`` only.

    Parameters
    ----------
    num_targets : int
        ``K``.

    Returns
    -------
    int
        Rows per scan step.
    """
    return max(MIN_CHUNK, PAIRS_PER_CHUNK // max(int(num_targets), 1))


# ----------------------------------------------------------------------
# Per-device pieces (run inside shard_map).
# ----------------------------------------------------------------------


def _pair_terms(
    tx: Any,
    tid: Any,
    sx: Any,
    sm: Any,
    sid: Any,
    params: Any,
    kernel: str,
    newton: bool,
) -> Any:
    """``[K, C, 4]`` fp64 pair terms: ``m g d`` (3) and ``-m psi``.

    Parameters
    ----------
    tx : Any
        ``[K, 3]`` target positions (fp64).
    tid : Any
        ``[K]`` target ids.
    sx : Any
        ``[C, 3]`` source positions (fp64).
    sm : Any
        ``[C]`` source masses (fp64).
    sid : Any
        ``[C]`` source ids (``< 0``: padding).
    params : Any
        ``(2,)`` pair-factor parameters.
    kernel : str
        Kernel name (static).
    newton : bool
        Zero softening: coincident distinct pairs are masked (their force is 0
        by symmetry, their Newtonian potential undefined).

    Returns
    -------
    Any
        ``[K, C, 4]``.
    """
    d = sx[None, :, :] - tx[:, None, :]
    r2 = jnp.sum(d * d, axis=-1)
    active = (sid[None, :] != tid[:, None]) & (sid[None, :] >= 0) & (sm[None, :] != 0.0)
    if newton:
        active = active & (r2 > 0.0)
    g, psi, _ = masked_pair_factors(r2, active, params, kernel, potential=True)
    mg = sm[None, :] * g
    acc = mg[:, :, None] * d
    pot = -(sm[None, :] * psi)
    return jnp.concatenate([acc, pot[:, :, None]], axis=-1)


def _chunked(x: Any, m: Any, ids: Any, chunk: int) -> tuple:
    """Pad the local rows to whole chunks and reshape ``[n_chunks, chunk, ...]``.

    Parameters
    ----------
    x : Any
        ``[R, 3]``.
    m : Any
        ``[R]``.
    ids : Any
        ``[R]``.
    chunk : int
        Chunk length.

    Returns
    -------
    tuple
        ``(x, m, ids)`` chunked; padding rows have id ``-1`` and mass 0.
    """
    rows = int(x.shape[0])
    n_chunks = max(1, -(-rows // chunk))
    pad = n_chunks * chunk - rows
    x = jnp.pad(x.astype(jnp.float64), ((0, pad), (0, 0)))
    m = jnp.pad(m.astype(jnp.float64), (0, pad))
    ids = jnp.pad(ids.astype(jnp.int64), (0, pad), constant_values=-1)
    return (
        x.reshape(n_chunks, chunk, 3),
        m.reshape(n_chunks, chunk),
        ids.reshape(n_chunks, chunk),
    )


def _limbs(t: Any, scale: Any) -> Any:
    """``int64[K, 4, 3]``: the chunk's terms on the grid, as summed 30-bit limbs.

    Parameters
    ----------
    t : Any
        ``[K, C, 4]`` terms.
    scale : Any
        ``[K, 4]`` powers of two (``2^s``): every ``|t| * scale < 2^85``.

    Returns
    -------
    Any
        Limb sums over the chunk, low limb first.
    """
    x = jnp.round(t * scale[:, None, :])  # exact: a power-of-two scale, then rint
    a = jnp.abs(x)
    b = float(1 << LIMB_BITS)
    d2 = jnp.floor(a * (1.0 / (b * b)))
    r = a - d2 * (b * b)  # exact (the bits of a below 2^60)
    d1 = jnp.floor(r * (1.0 / b))
    d0 = r - d1 * b
    neg = x < 0.0
    out = []
    for d in (d0, d1, d2):
        di = d.astype(jnp.int64)
        out.append(jnp.sum(jnp.where(neg, -di, di), axis=1))
    return jnp.stack(out, axis=-1)


def _program(kind: str, mesh: Any, axis: str, key: tuple, build: Callable) -> Any:
    """A cached ``jit(shard_map(...))``.

    Parameters
    ----------
    kind : str
        Program name.
    mesh : Any
        The mesh.
    axis : str
        Its axis.
    key : tuple
        Shapes and statics.
    build : Callable
        ``() -> jitted program``.

    Returns
    -------
    Any
        The program.
    """
    devs = tuple(int(d.id) for d in np.asarray(mesh.devices).reshape(-1))
    full = (kind, devs, axis) + key
    prog = _PROGRAMS.get(full)
    if prog is None:
        prog = build()
        _PROGRAMS[full] = prog
    return prog


def _replicated(mesh: Any, value: np.ndarray) -> Any:
    """``value`` as a replicated global array on ``mesh`` (the same on every process).

    Parameters
    ----------
    mesh : Any
        The mesh.
    value : np.ndarray
        Host value, identical on every process.

    Returns
    -------
    Any
        ``P()``-sharded array.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return jax.device_put(np.asarray(value), NamedSharding(mesh, P()))


def _host(arr: Any) -> np.ndarray:
    """A replicated or addressable result on this host.

    Parameters
    ----------
    arr : Any
        A jax array.

    Returns
    -------
    np.ndarray
        Its value.
    """
    from jaccpot.distributed.multiprocess import host_value

    return host_value(arr)


def _mesh_of(arrays: Sequence[Any], mesh: Any, axis_name: Optional[str]) -> tuple:
    """The mesh and axis to run on: the caller's, or a one-device mesh.

    Parameters
    ----------
    arrays : Sequence[Any]
        The source arrays (their device, when no mesh is given).
    mesh : Any
        Caller's 1-D mesh or ``None``.
    axis_name : Optional[str]
        Its axis (default: the mesh's only axis).

    Returns
    -------
    tuple
        ``(mesh, axis)``.
    """
    from jax.sharding import Mesh

    if mesh is not None:
        axis = axis_name if axis_name is not None else mesh.axis_names[0]
        return mesh, axis
    dev = None
    for a in arrays:
        if isinstance(a, jax.Array):
            dev = next(iter(a.devices()))
            break
    if dev is None:
        dev = jax.devices()[0]
    return Mesh(np.array([dev]), (_AXIS,)), _AXIS


def gather_by_id(
    values: Any,
    ids: Any,
    target_ids: np.ndarray,
    *,
    mesh: Any = None,
    axis_name: Optional[str] = None,
) -> tuple:
    """The rows of ``values`` whose id is in ``target_ids``, bit for bit.

    Each device finds its rows by a binary search of the sorted targets; the
    ``[K, D]`` candidates and found-counts are all-gathered and each target is
    taken from the device that holds it (a select, no float arithmetic).

    Parameters
    ----------
    values : Any
        ``[R, D]`` (or ``[R]``) rows, ``P(axis)``-sharded on ``mesh`` or on one
        device.
    ids : Any
        ``[R]`` global ids (``< 0``: padding), sharded like ``values``.
    target_ids : np.ndarray
        Sorted ids.
    mesh : Any
        1-D mesh of the sharding; ``None``: one device.
    axis_name : Optional[str]
        Its axis.

    Returns
    -------
    tuple
        ``(rows [K, D] as values' dtype on the host, found int64[K])``.
    """
    from jax.sharding import PartitionSpec as P

    mesh, axis = _mesh_of((values, ids), mesh, axis_name)
    tids = np.asarray(target_ids, np.int64)
    k = int(tids.shape[0])
    flat = values.ndim == 1
    vals = values[:, None] if flat else values
    width = int(vals.shape[1])
    dtype = vals.dtype

    def body(v: Any, i: Any, t: Any) -> tuple:
        i64 = i.astype(jnp.int64)
        pos = jnp.clip(jnp.searchsorted(t, i64), 0, k - 1)
        hit = (t[pos] == i64) & (i64 >= 0)
        slot = jnp.where(hit, pos, k)
        buf = jnp.zeros((k, width), v.dtype).at[slot].set(v, mode="drop")
        cnt = jnp.zeros((k,), jnp.int64).at[slot].add(1, mode="drop")
        bufs = jax.lax.all_gather(buf, axis)  # [ndev, K, D]
        cnts = jax.lax.all_gather(cnt, axis)  # [ndev, K]
        src = jnp.argmax(cnts > 0, axis=0)
        rows = jnp.take_along_axis(bufs, src[None, :, None], axis=0)[0]
        return rows, jnp.sum(cnts, axis=0)

    def build() -> Any:
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P(axis), P(axis), P()),
                out_specs=(P(), P()),
                check_vma=False,
            )
        )

    prog = _program(
        "gather", mesh, axis, (tuple(vals.shape), str(dtype), str(ids.dtype), k), build
    )
    rows, found = prog(vals, ids, _replicated(mesh, tids))
    rows = _host(rows)
    return (rows[:, 0] if flat else rows), _host(found).astype(np.int64)


def _max_pass(
    tx: np.ndarray,
    tids: np.ndarray,
    x: Any,
    m: Any,
    ids: Any,
    law: ForceLaw,
    mesh: Any,
    axis: str,
) -> np.ndarray:
    """``[K, 4]`` max ``|term|`` over every source (a ``pmax``: exact in any order).

    Parameters
    ----------
    tx : np.ndarray
        ``[K, 3]`` fp64 targets.
    tids : np.ndarray
        ``[K]`` ids.
    x : Any
        ``[R, 3]`` sources.
    m : Any
        ``[R]``.
    ids : Any
        ``[R]``.
    law : ForceLaw
        Pair law.
    mesh : Any
        Mesh.
    axis : str
        Axis.

    Returns
    -------
    np.ndarray
        ``[K, 4]`` fp64.
    """
    from jax.sharding import PartitionSpec as P

    k = int(tids.shape[0])
    chunk = _chunk_for(k)
    kernel = resolve_softening_kernel(law.kernel)
    newton = float(law.softening) == 0.0

    def body(xs: Any, ms: Any, is_: Any, t: Any, ti: Any, prm: Any) -> Any:
        cx, cm, ci = _chunked(xs, ms, is_, chunk)

        def step(carry: Any, c: tuple) -> tuple:
            terms = _pair_terms(t, ti, c[0], c[1], c[2], prm, kernel, newton)
            return jnp.maximum(carry, jnp.max(jnp.abs(terms), axis=1)), None

        local, _ = jax.lax.scan(
            step, jnp.zeros((k, _COLUMNS), jnp.float64), (cx, cm, ci)
        )
        return jax.lax.pmax(local, axis)

    def build() -> Any:
        spec = P(axis)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec, spec, spec, P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )

    key = (
        tuple(x.shape),
        str(x.dtype),
        str(m.dtype),
        str(ids.dtype),
        k,
        kernel,
        newton,
    )
    prog = _program("max", mesh, axis, key, build)
    out = prog(
        x,
        m,
        ids,
        _replicated(mesh, tx),
        _replicated(mesh, tids),
        _replicated(mesh, law.params()),
    )
    return _host(out).astype(np.float64)


def _limb_pass(
    tx: np.ndarray,
    tids: np.ndarray,
    scale: np.ndarray,
    x: Any,
    m: Any,
    ids: Any,
    law: ForceLaw,
    mesh: Any,
    axis: str,
) -> np.ndarray:
    """``int64[K, 4, 3]`` limb sums over every source (an integer ``psum``).

    Parameters
    ----------
    tx : np.ndarray
        ``[K, 3]``.
    tids : np.ndarray
        ``[K]``.
    scale : np.ndarray
        ``[K, 4]`` powers of two.
    x : Any
        ``[R, 3]``.
    m : Any
        ``[R]``.
    ids : Any
        ``[R]``.
    law : ForceLaw
        Pair law.
    mesh : Any
        Mesh.
    axis : str
        Axis.

    Returns
    -------
    np.ndarray
        Limb sums, low limb first.
    """
    from jax.sharding import PartitionSpec as P

    k = int(tids.shape[0])
    chunk = _chunk_for(k)
    kernel = resolve_softening_kernel(law.kernel)
    newton = float(law.softening) == 0.0

    def body(xs: Any, ms: Any, is_: Any, t: Any, ti: Any, sc: Any, prm: Any) -> Any:
        cx, cm, ci = _chunked(xs, ms, is_, chunk)

        def step(carry: Any, c: tuple) -> tuple:
            terms = _pair_terms(t, ti, c[0], c[1], c[2], prm, kernel, newton)
            return carry + _limbs(terms, sc), None

        local, _ = jax.lax.scan(
            step, jnp.zeros((k, _COLUMNS, 3), jnp.int64), (cx, cm, ci)
        )
        return jax.lax.psum(local, axis)  # integers: exact in any order

    def build() -> Any:
        spec = P(axis)
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(spec, spec, spec, P(), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            )
        )

    key = (
        tuple(x.shape),
        str(x.dtype),
        str(m.dtype),
        str(ids.dtype),
        k,
        kernel,
        newton,
    )
    prog = _program("limbs", mesh, axis, key, build)
    out = prog(
        x,
        m,
        ids,
        _replicated(mesh, tx),
        _replicated(mesh, tids),
        _replicated(mesh, scale),
        _replicated(mesh, law.params()),
    )
    return _host(out).astype(np.int64)


def _scale_of(max_term: np.ndarray) -> tuple:
    """Per (target, column): the grid exponent ``s`` and ``2^s``.

    Parameters
    ----------
    max_term : np.ndarray
        ``[K, 4]`` max ``|term|``.

    Returns
    -------
    tuple
        ``(s int64[K, 4], 2^s fp64[K, 4])`` with ``max_term * 2^s < 2^85``.

    Raises
    ------
    FloatingPointError
        On a non-finite term (a NaN position, or coincident particles without
        softening).
    """
    if not np.all(np.isfinite(max_term)):
        raise FloatingPointError(
            "reference force: a non-finite pair term (non-finite positions or masses?)"
        )
    _, e = np.frexp(max_term)  # max_term < 2^e (e = 0 for 0)
    s = np.clip(HEADROOM_BITS - e.astype(np.int64), -1000, 1000)
    return s, np.ldexp(np.ones_like(max_term), s)


def _limbs_to_float(limbs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``sum_l limb_l 2^(30 l) / 2^s`` exactly, rounded once to fp64.

    Parameters
    ----------
    limbs : np.ndarray
        ``int64[K, 4, 3]``.
    s : np.ndarray
        ``int64[K, 4]``.

    Returns
    -------
    np.ndarray
        ``[K, 4]`` fp64.
    """
    out = np.empty(s.shape, np.float64)
    for idx in np.ndindex(*s.shape):
        n = 0
        for limb in range(3):
            n += int(limbs[idx + (limb,)]) << (LIMB_BITS * limb)
        # exact rational n / 2^s, rounded once (Python's int/int true division
        # of a power of two is correctly rounded)
        e = int(s[idx])
        out[idx] = n / (1 << e) if e >= 0 else float(n * (1 << -e))
    return out


@dataclass(frozen=True)
class ReferenceForce:
    """The reference at a target set.

    Attributes
    ----------
    ids : np.ndarray
        ``int64[K]`` target ids (ascending).
    positions : np.ndarray
        ``[K, 3]`` fp64 target positions as found in the sources.
    acceleration : np.ndarray
        ``[K, 3]`` fp64 self-gravity (``G`` applied).
    potential : np.ndarray
        ``[K]`` fp64 self potential per unit mass (``G`` applied; the self pair
        excluded).
    law : ForceLaw
        The pair law.
    num_sources : int
        Rows with an id ``>= 0`` and a non-zero mass, summed over the groups
        (the integer check that every source was seen).
    """

    ids: np.ndarray
    positions: np.ndarray
    acceleration: np.ndarray
    potential: np.ndarray
    law: ForceLaw
    num_sources: int


def _count_sources(m: Any, ids: Any, mesh: Any, axis: str) -> int:
    """Live sources (``id >= 0``, ``m != 0``) over the mesh (an integer ``psum``).

    Parameters
    ----------
    m : Any
        ``[R]``.
    ids : Any
        ``[R]``.
    mesh : Any
        Mesh.
    axis : str
        Axis.

    Returns
    -------
    int
        The count.
    """
    from jax.sharding import PartitionSpec as P

    def body(ms: Any, is_: Any) -> Any:
        live = (is_ >= 0) & (ms != 0)
        return jax.lax.psum(jnp.sum(live.astype(jnp.int64)), axis)

    def build() -> Any:
        return jax.jit(
            jax.shard_map(
                body,
                mesh=mesh,
                in_specs=(P(axis), P(axis)),
                out_specs=P(),
                check_vma=False,
            )
        )

    prog = _program(
        "count", mesh, axis, (tuple(m.shape), str(m.dtype), str(ids.dtype)), build
    )
    return int(_host(prog(m, ids)))


def reference_force(
    positions: Any,
    masses: Any,
    ids: Any,
    target_ids: np.ndarray,
    law: ForceLaw,
    *,
    mesh: Any = None,
    axis_name: Optional[str] = None,
    extra_sources: Sequence[tuple] = (),
) -> ReferenceForce:
    """The exact-order fp64 direct sum at ``target_ids``.

    Parameters
    ----------
    positions : Any
        ``[R, 3]`` source positions (any float dtype; read as fp64), flat and
        ``P(axis)``-sharded on ``mesh``, or on one device / the host.
    masses : Any
        ``[R]``; zero rows are skipped.
    ids : Any
        ``[R]`` global ids, ``< 0`` on padding. Every target id must be present
        exactly once over all source groups.
    target_ids : np.ndarray
        Sorted target ids (:func:`select_targets`).
    law : ForceLaw
        The pair law.
    mesh : Any
        The sources' 1-D mesh; ``None``: one device.
    axis_name : Optional[str]
        Its axis (default: the mesh's only one).
    extra_sources : Sequence[tuple]
        More source groups ``(positions, masses, ids, mesh)`` (e.g. an escaper
        tail with its own sharding; ``mesh`` may be ``None``). Their limbs are
        added exactly; the grid comes from the max over every group.

    Returns
    -------
    ReferenceForce
        The reference.

    Raises
    ------
    ValueError
        If a target id is missing or present more than once.
    """
    tids = np.asarray(target_ids, np.int64)
    if tids.ndim != 1 or np.any(np.diff(tids) <= 0):
        raise ValueError("target_ids must be strictly ascending (select_targets)")
    groups = [(positions, masses, ids, mesh, axis_name)] + [
        (g[0], g[1], g[2], g[3] if len(g) > 3 else None, None) for g in extra_sources
    ]
    resolved = []
    for x, m, i, msh, ax in groups:
        msh, ax = _mesh_of((x, m, i), msh, ax)
        if not isinstance(x, jax.Array):
            x, m, i = jnp.asarray(x), jnp.asarray(m), jnp.asarray(i)
        resolved.append((x, m, i, msh, ax))

    # target positions, from whichever group holds each id
    tx = np.zeros((tids.shape[0], 3), np.float64)
    found = np.zeros(tids.shape[0], np.int64)
    for x, _m, i, msh, ax in resolved:
        rows, cnt = gather_by_id(x, i, tids, mesh=msh, axis_name=ax)
        tx = np.where((cnt > 0)[:, None], rows.astype(np.float64), tx)
        found += cnt
    if np.any(found != 1):
        bad = tids[found != 1][:8]
        raise ValueError(
            f"reference force: {int(np.sum(found != 1))} target ids not found exactly "
            f"once in the sources (e.g. {bad.tolist()})"
        )

    max_term = np.zeros((tids.shape[0], _COLUMNS), np.float64)
    for x, m, i, msh, ax in resolved:
        max_term = np.maximum(max_term, _max_pass(tx, tids, x, m, i, law, msh, ax))
    s, scale = _scale_of(max_term)
    total = np.zeros((tids.shape[0], _COLUMNS, 3), dtype=object)
    sources = 0
    for x, m, i, msh, ax in resolved:
        limbs = _limb_pass(tx, tids, scale, x, m, i, law, msh, ax)
        total = total + limbs.astype(object)
        sources += _count_sources(m, i, msh, ax)
    sums = _limbs_to_float(total, s)
    G = float(law.G)
    return ReferenceForce(
        ids=tids,
        positions=tx,
        acceleration=G * sums[:, :3],
        potential=G * sums[:, 3],
        law=law,
        num_sources=int(sources),
    )


# ----------------------------------------------------------------------
# Scoring.
# ----------------------------------------------------------------------


def score_force(
    acceleration: np.ndarray,
    reference: ReferenceForce,
    *,
    potential: Optional[np.ndarray] = None,
) -> dict:
    """Score a force at the reference's targets.

    ``rel_l2 = sqrt(sum |a - a_ref|^2 / sum |a_ref|^2)``; the per-particle
    relative error ``|a - a_ref| / |a_ref|`` gives ``p50`` / ``p90`` / ``p99`` /
    ``max`` / ``mean``.

    Parameters
    ----------
    acceleration : np.ndarray
        ``[K, 3]`` the scored force at the targets (target order).
    reference : ReferenceForce
        The reference.
    potential : Optional[np.ndarray]
        ``[K]`` a potential to score too.

    Returns
    -------
    dict
        JSON-able scores (``num_targets``, ``nonfinite`` included).
    """
    a = np.asarray(acceleration, np.float64)
    ref = reference.acceleration
    diff = a - ref
    num = np.linalg.norm(diff, axis=1)
    den = np.linalg.norm(ref, axis=1)
    finite = np.isfinite(num)
    rel = np.where(den > 0, num / np.where(den > 0, den, 1.0), num)
    out = {
        "num_targets": int(ref.shape[0]),
        "nonfinite": int(np.sum(~finite)),
        "rel_l2": float(np.sqrt(np.sum(num[finite] ** 2) / np.sum(den[finite] ** 2))),
        "p50": float(np.percentile(rel[finite], 50)),
        "p90": float(np.percentile(rel[finite], 90)),
        "p99": float(np.percentile(rel[finite], 99)),
        "max": float(np.max(rel[finite])),
        "mean": float(np.mean(rel[finite])),
    }
    if potential is not None:
        p = np.asarray(potential, np.float64)
        dp = np.abs(p - reference.potential)
        rp = dp / np.maximum(np.abs(reference.potential), np.finfo(np.float64).tiny)
        out["potential_rel_l2"] = float(
            np.sqrt(np.sum(dp**2) / np.sum(reference.potential**2))
        )
        out["potential_p99"] = float(np.percentile(rp, 99))
    return out


def score_single_result(
    result: Any,
    solver: Any,
    *,
    num_targets: int = 4096,
    seed: int = 0,
    target_ids: Optional[np.ndarray] = None,
) -> dict:
    """Score a single-card :class:`~jaccpot.production.RunResult` at its final state.

    The production force is ``result.handle.self_acceleration`` (the self-gravity
    the last step computed at ``result.state``); the reference uses the handle's
    force parameters.

    Parameters
    ----------
    result : Any
        A ``RunResult`` with ``mode == "single"`` and no escaper split.
    solver : Any
        The run's ``FastMultipoleMethod``.
    num_targets : int
        ``K`` when ``target_ids`` is not given.
    seed : int
        :func:`select_targets` seed.
    target_ids : Optional[np.ndarray]
        Explicit sorted targets.

    Returns
    -------
    dict
        :func:`score_force` plus ``law`` and ``seed``.

    Raises
    ------
    NotImplementedError
        For a mesh result or a run with escapers (their row order differs).
    ValueError
        If the handle holds no self-gravity.
    """
    if getattr(result, "mode", "single") != "single":
        raise NotImplementedError("score_single_result: use score_mesh_result")
    if getattr(result, "escapers", None) is not None:
        raise NotImplementedError(
            "score_single_result: runs with escapers are not supported"
        )
    handle = result.handle
    acc = handle.self_acceleration
    if acc is None:
        raise ValueError("the handle holds no self_acceleration (an escaper re-split?)")
    state = result.state
    n = int(state.shape[0])
    tids = (
        target_ids if target_ids is not None else select_targets(n, num_targets, seed)
    )
    ids = jnp.arange(n, dtype=jnp.int32)
    if isinstance(state, jax.Array):
        ids = jax.device_put(ids, next(iter(state.devices())))
    law = force_law_of(solver, handle)
    ref = reference_force(state[:, 0], result.masses, ids, tids, law)
    prod, _ = gather_by_id(acc, ids, tids)
    out = score_force(prod, ref)
    out["law"] = law.as_dict()
    out["seed"] = int(seed)
    return out


def score_mesh_result(
    result: Any,
    solver: Any,
    *,
    num_targets: int = 4096,
    seed: int = 0,
    target_ids: Optional[np.ndarray] = None,
    external_acceleration: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
    """Score a multi-GPU (mesh) ``RunResult`` at its final state, without a gather.

    Reads the segment runner's mesh-resident layout (``handle.state``: rows by
    device, ids per row) and its carried TOTAL acceleration. Works under several
    processes (``jax.distributed``): every array stays sharded; only ``K`` rows
    are gathered.

    Parameters
    ----------
    result : Any
        A ``RunResult`` with ``mode == "mesh"`` (no escaper tail).
    solver : Any
        The run's ``FastMultipoleMethod``.
    num_targets : int
        ``K`` when ``target_ids`` is not given.
    seed : int
        :func:`select_targets` seed.
    target_ids : Optional[np.ndarray]
        Explicit sorted targets.
    external_acceleration : Optional[Callable[[np.ndarray], np.ndarray]]
        The run's external field at ``[K, 3]`` fp64 positions (host), added to
        the reference (the carry is the total acceleration). ``None``: no field.

    Returns
    -------
    dict
        :func:`score_force` plus ``law``, ``seed`` and ``devices``.

    Raises
    ------
    NotImplementedError
        For a single-card result or a run with an escaper tail.
    ValueError
        If the mesh state holds no acceleration yet, or ``N`` is unknown and no
        ``target_ids`` are given.
    """
    if getattr(result, "mode", None) != "mesh":
        raise NotImplementedError("score_mesh_result: use score_single_result")
    handle = result.handle
    if getattr(handle, "tail_on", False):
        raise NotImplementedError(
            "score_mesh_result: runs with escapers are not supported"
        )
    st = handle.state
    if st.acceleration is None:
        raise ValueError("the mesh state holds no acceleration yet (no start force)")
    n = int(np.asarray(result.masses).shape[0]) if result.masses is not None else None
    if target_ids is None:
        if n is None:
            raise ValueError("pass target_ids (the run's N is unknown)")
        target_ids = select_targets(n, num_targets, seed)
    mesh, axis = handle.mesh, handle.axis_name
    law = force_law_of(solver)
    ref = reference_force(
        st.state[:, 0], st.masses, st.ids, target_ids, law, mesh=mesh, axis_name=axis
    )
    if external_acceleration is not None:
        ext = np.asarray(external_acceleration(ref.positions), np.float64)
        ref = ReferenceForce(
            ids=ref.ids,
            positions=ref.positions,
            acceleration=ref.acceleration + ext,
            potential=ref.potential,
            law=ref.law,
            num_sources=ref.num_sources,
        )
    prod, _ = gather_by_id(
        st.acceleration, st.ids, target_ids, mesh=mesh, axis_name=axis
    )
    out = score_force(prod, ref)
    out["law"] = law.as_dict()
    out["seed"] = int(seed)
    out["devices"] = int(np.asarray(mesh.devices).size)
    return out


__all__ = [
    "HEADROOM_BITS",
    "LIMB_BITS",
    "ForceLaw",
    "ReferenceForce",
    "force_law_of",
    "gather_by_id",
    "reference_force",
    "score_force",
    "score_mesh_result",
    "score_single_result",
    "select_targets",
]

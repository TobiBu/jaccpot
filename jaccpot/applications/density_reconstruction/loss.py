"""Data misfit and regularisers. Every term is toggleable, weighted, and reported.

Hard decision 9: regularisation is **required and reported, never hidden**. So
:func:`total_loss` returns every component unweighted alongside the total, the
weights travel into the results JSON, and a weight of ``0.0`` disables a term
while still reporting its value -- which is what makes the deliberately
unregularised run of fig17(d) an informative panel rather than a missing one.

Why the regularisers act inside leaves
--------------------------------------
Both regularisers are built on the **frozen leaf partition** of the prepared
state, not on a fresh neighbour search:

* It is the right scale. A leaf is a small spatial region holding ``S ~ 16``
  particles, so a within-leaf statistic is local: it suppresses leaf-scale
  clumping and voiding while permitting the genuine large-scale density
  contrast a galaxy has. A *global* spacing penalty would fight the physics.
* It is affordable. ``O(N S^2)`` with ``S = 16``, against the ``O(N^2)`` an
  exact all-pairs nearest neighbour would cost -- at ``N = 1e7`` that is the
  difference between the penalty being free and being impossible.
* It obeys the same contract as the force. The leaf *assignment* is part of the
  frozen topology and enters as a constant; only the coordinates are
  differentiated. So the regularisers are exactly as piecewise-smooth as the
  data term, and no more.

Two limitations, both measured rather than assumed
--------------------------------------------------
**The reach is one leaf, and the leaf is one iteration stale.** A "nearest
neighbour" here is the nearest neighbour *in the same leaf of the frozen
partition*, which is not always the true one. Measured at N=192: driving two
particles that the frozen partition placed in *different* leaves onto the same
point leaves :func:`softening_floor_penalty` at exactly 0.0 -- it does not see
them. What closes the gap is the rebuild: two coincident particles land in the
same leaf at the next rebuild, so at the default cadence ``k = 1`` the penalty
is never more than one iteration behind. At ``k > 1`` it is up to ``k``
iterations behind, which is one more thing the cadence panel of fig19 is
trading off. Widening the search to the near-field neighbour leaves would cost
another factor ~27 and is what makes it unaffordable at ``N = 1e7``.

**At exactly zero separation the gradient vanishes.** Measured on a same-leaf
pair at separations ``1.0, 0.5, 0.1, 0.0`` times the softening: the penalty
reads ``0, 2.6e-3, 8.4e-3, 1.04e-2`` and the gradient norm ``0, 1.47, 2.65,
0``. The final zero is not a bug -- it is the symmetric point, where the
direction to push is genuinely undefined -- and it is finite rather than NaN
because the ``sqrt`` is guarded on its argument. It is a measure-zero
configuration an optimiser does not land on exactly, and the penalty rises
monotonically on the approach to it, which is what matters.

These are penalties, not measurements;
:mod:`~jaccpot.applications.density_reconstruction.diagnostics` measures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "LEAF_CHUNK",
    "LeafBlocks",
    "Regularization",
    "data_misfit",
    "leaf_blocks_from_state",
    "regularization_terms",
    "softening_floor_penalty",
    "spacing_smoothness_penalty",
    "total_loss",
]


def data_misfit(
    predicted: ArrayLike,
    observed: ArrayLike,
    *,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Mean squared acceleration residual over tracers.

    Parameters
    ----------
    predicted : ArrayLike
        ``(M, 3)`` model accelerations at the tracers. NumPy or JAX.
    observed : ArrayLike
        ``(M, 3)`` observed accelerations. NumPy or JAX.
    scale : float
        Divides the residual before squaring, so the loss is O(1) whatever the
        field magnitude. Use a characteristic acceleration; ``1.0`` leaves the
        raw units.

    Returns
    -------
    jnp.ndarray
        Scalar. This is the **field-space residual**, the primary metric of
        the section.
    """
    residual = (jnp.asarray(predicted) - jnp.asarray(observed)) / scale
    return jnp.mean(jnp.sum(residual**2, axis=-1))


@dataclass(frozen=True)
class LeafBlocks:
    """The frozen leaf partition, as padded index blocks.

    Attributes
    ----------
    indices : jnp.ndarray
        ``(L, S)`` int32 particle ids per leaf, padded with 0 where invalid.
    valid : jnp.ndarray
        ``(L, S)`` bool mask; ``False`` at padding.
    leaf_size : int
        ``S``, the padded block width.
    num_leaves : int
        ``L``.
    """

    indices: jnp.ndarray
    valid: jnp.ndarray
    leaf_size: int
    num_leaves: int


def leaf_blocks_from_state(
    state: Any, *, num_sources: Optional[int] = None
) -> LeafBlocks:
    """Extract the frozen leaf partition as padded index blocks.

    Parameters
    ----------
    state : Any
        A prepared ``FMMPreparedState``. Its arrays are host-side or on device;
        either way the result is a constant, because the leaf assignment is
        part of the frozen topology and must not be differentiated.
    num_sources : Optional[int]
        When given, particles with id ``>= num_sources`` are masked out. This
        is how the zero-mass tracers are excluded: they are appended to the
        particle set and sit in leaves like anything else, but they are fixed
        inputs, so a penalty on their spacing would be a penalty on a constant.

    Returns
    -------
    LeafBlocks
        The padded blocks and their validity mask.
    """
    topo = state.tree.topology
    particle_indices = np.asarray(topo.particle_indices).astype(np.int64)
    node_ranges = np.asarray(topo.node_ranges).astype(np.int64)
    leaf_nodes = np.asarray(state.neighbor_list.leaf_indices).astype(np.int64)

    starts = node_ranges[leaf_nodes, 0]
    stops = node_ranges[leaf_nodes, 1] + 1
    sizes = np.maximum(stops - starts, 0)
    width = int(sizes.max()) if sizes.size else 0
    num_leaves = int(leaf_nodes.size)

    indices = np.zeros((num_leaves, width), dtype=np.int32)
    valid = np.zeros((num_leaves, width), dtype=bool)
    # Vectorised gather: one (L, S) grid of sorted slots, clipped and masked.
    if num_leaves and width:
        ramp = np.arange(width, dtype=np.int64)[None, :]
        slots = starts[:, None] + ramp
        inside = ramp < sizes[:, None]
        indices = particle_indices[np.clip(slots, 0, particle_indices.size - 1)].astype(
            np.int32
        )
        valid = inside
        if num_sources is not None:
            valid = valid & (indices < int(num_sources))
        indices = np.where(valid, indices, 0).astype(np.int32)

    return LeafBlocks(
        indices=jnp.asarray(indices),
        valid=jnp.asarray(valid),
        leaf_size=width,
        num_leaves=num_leaves,
    )


#: Leaves processed per chunk inside :func:`_within_leaf_nearest_distances`.
#: The pairwise block is ``(chunk, S, S, 3)``, so peak memory is bounded by this
#: rather than by ``N``. MEASURED: unchunked at N=1048576 and leaf 256 the
#: single ``f64[4112, 256, 256, 3]`` block is 6.5 GB and was the binding
#: constraint on the largest fit that would run -- and it was being paid by runs
#: whose regularisation weight is ZERO, since the terms are deliberately
#: evaluated so an unregularised run still reports what it is not paying for.
#: 128 leaves at S=256 is ~200 MB.
LEAF_CHUNK = 128


def _nearest_within_chunk(
    gathered: jnp.ndarray, valid: jnp.ndarray, leaf_size: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Nearest-neighbour squared distance within each leaf of one chunk.

    Parameters
    ----------
    gathered : jnp.ndarray
        ``(chunk, S, 3)`` positions of each leaf's members.
    valid : jnp.ndarray
        ``(chunk, S)`` validity mask.
    leaf_size : int
        ``S``, the padded block width.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        ``(chunk, S)`` nearest squared distances (``inf`` where no neighbour
        exists) and the ``(chunk, S)`` mask of entries that had one.
    """
    delta = gathered[:, :, None, :] - gathered[:, None, :, :]
    sq = jnp.sum(delta**2, axis=-1)

    pair_valid = valid[:, :, None] & valid[:, None, :]
    eye = jnp.eye(leaf_size, dtype=bool)[None, :, :]
    pair_valid = pair_valid & ~eye

    # Masked-out pairs go to +inf so the min ignores them.
    sq = jnp.where(pair_valid, sq, jnp.asarray(jnp.inf, dtype=sq.dtype))
    nearest_sq = jnp.min(sq, axis=-1)
    return nearest_sq, jnp.isfinite(nearest_sq) & valid


def _within_leaf_nearest_distances(
    positions: ArrayLike, blocks: LeafBlocks, *, leaf_chunk: int = LEAF_CHUNK
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Nearest-neighbour distance of each particle within its own leaf.

    Chunked over leaves, so peak memory is set by ``leaf_chunk`` and not by
    ``N``. ``jax.lax.map`` runs the chunks as a scan, which keeps one chunk's
    ``(chunk, S, S, 3)`` block live at a time; the reverse pass stores only the
    ``(chunk, S)`` outputs, so the gradient is bounded too. Semantics are
    identical to the unchunked form -- each leaf's neighbours are exactly its
    own members either way, since a leaf never spans a chunk boundary.

    Parameters
    ----------
    positions : ArrayLike
        ``(N, 3)`` source positions, differentiated.
    blocks : LeafBlocks
        The frozen leaf partition.
    leaf_chunk : int
        Leaves per chunk. Smaller trades speed for memory.

    Returns
    -------
    Tuple[jnp.ndarray, jnp.ndarray]
        ``(L, S)`` nearest-neighbour distances and the ``(L, S)`` mask of
        entries for which a neighbour existed at all. A leaf holding one valid
        particle contributes nothing: its "nearest neighbour" is itself.
    """
    x = jnp.asarray(positions, dtype=jnp.float64)
    gathered = x[blocks.indices]  # (L, S, 3)
    valid = blocks.valid
    num_leaves = int(blocks.num_leaves)
    size = int(blocks.leaf_size)

    if num_leaves == 0 or size == 0:
        empty = jnp.zeros((num_leaves, size), dtype=x.dtype)
        return empty + 1.0, jnp.zeros((num_leaves, size), dtype=bool)

    chunk = max(int(leaf_chunk), 1)
    if num_leaves <= chunk:
        nearest_sq, has_neighbour = _nearest_within_chunk(gathered, valid, size)
    else:
        # Pad the leaf axis up to a whole number of chunks. Padding leaves are
        # all-invalid, so they contribute no pair and no cotangent.
        chunks = -(-num_leaves // chunk)
        pad = chunks * chunk - num_leaves
        gathered_padded = jnp.pad(gathered, ((0, pad), (0, 0), (0, 0)))
        valid_padded = jnp.pad(valid, ((0, pad), (0, 0)), constant_values=False)
        shaped = gathered_padded.reshape(chunks, chunk, size, 3)
        mask = valid_padded.reshape(chunks, chunk, size)

        # jax.checkpoint is what actually bounds the memory. lax.map is a
        # scan, which keeps the FORWARD to one chunk -- but a scan's reverse
        # pass stores every iteration's residuals, so without remat the
        # backward pass materialises the pairwise block for all chunks at once.
        # Measured: at N=1048576, leaf 256, the unremated scan allocated
        # f64[33, 128, 256, 256, 3] = 6.6 GB, which is the whole saving undone
        # and was still the fig20 ceiling. With remat the scan carries only each
        # chunk's (chunk, S, 3) input and (chunk, S) output, and recomputes the
        # block on the way back -- the classic memory-for-flops trade, and the
        # flops here are cheap next to the FMM's own.
        nearest_sq, has_neighbour = jax.lax.map(
            jax.checkpoint(lambda pair: _nearest_within_chunk(pair[0], pair[1], size)),
            (shaped, mask),
        )
        nearest_sq = nearest_sq.reshape(chunks * chunk, size)[:num_leaves]
        has_neighbour = has_neighbour.reshape(chunks * chunk, size)[:num_leaves]

    # Guard the sqrt at zero: two coincident particles give an exactly-zero
    # squared distance, whose gradient is NaN. The guard is a floor on the
    # *argument*, not a clip on the result, so the penalty still pushes them
    # apart.
    nearest_sq = jnp.where(has_neighbour, nearest_sq, 1.0)
    return jnp.sqrt(nearest_sq + 1.0e-300), has_neighbour


def softening_floor_penalty(
    positions: ArrayLike,
    blocks: LeafBlocks,
    *,
    softening: float,
    floor_fraction: float = 1.0,
) -> jnp.ndarray:
    """Penalise pairs closer than a fraction of the softening length.

    Position gradients of the P2P term scale like ``1 / r^3``, so an optimiser
    handed a softening-free configuration will exploit it: two particles driven
    together produce an enormous gradient and a meaningless field. Hard
    decision 6 fixes the softening; this keeps the fit from living inside it.

    Parameters
    ----------
    positions : ArrayLike
        ``(N, 3)`` source positions.
    blocks : LeafBlocks
        The frozen leaf partition.
    softening : float
        The Plummer softening length in use.
    floor_fraction : float
        The floor is ``floor_fraction * softening``. One softening length by
        default.

    Returns
    -------
    jnp.ndarray
        Scalar, dimensionless: the mean squared *shortfall* below the floor,
        in units of the floor, over particles that have a neighbour. A
        one-sided hinge -- particles further apart than the floor contribute
        exactly zero, so this cannot fight the data term wherever it is not
        needed.

    Raises
    ------
    ValueError
        If the floor is not positive.
    """
    floor = float(floor_fraction) * float(softening)
    if floor <= 0.0:
        raise ValueError(
            f"softening floor must be positive, got floor_fraction="
            f"{floor_fraction!r} * softening={softening!r}"
        )
    distances, has_neighbour = _within_leaf_nearest_distances(positions, blocks)
    shortfall = jnp.maximum(floor - distances, 0.0) / floor
    count = jnp.maximum(jnp.sum(has_neighbour), 1)
    return jnp.sum(jnp.where(has_neighbour, shortfall**2, 0.0)) / count


def spacing_smoothness_penalty(
    positions: ArrayLike, blocks: LeafBlocks, *, floor: float = 1.0e-12
) -> jnp.ndarray:
    """Penalise leaf-scale clumping and voiding, not large-scale contrast.

    The variance of ``log`` nearest-neighbour spacing **within each leaf**,
    averaged over leaves. Scale-free by construction, so it does not prefer any
    particular density; and local, so a galaxy's real radial contrast costs
    nothing while a leaf that has partly collapsed and partly emptied costs a
    lot. That asymmetry is the point: the degeneracy this section reports is
    the optimiser's freedom to clump and void at fixed field, and this is the
    term that buys it back.

    Parameters
    ----------
    positions : ArrayLike
        ``(N, 3)`` source positions.
    blocks : LeafBlocks
        The frozen leaf partition.
    floor : float
        Added inside the logarithm so a coincident pair gives a large finite
        value rather than ``-inf``.

    Returns
    -------
    jnp.ndarray
        Scalar, dimensionless. Zero when every leaf has uniform spacing.
    """
    distances, has_neighbour = _within_leaf_nearest_distances(positions, blocks)
    weights = has_neighbour.astype(distances.dtype)
    log_d = jnp.log(distances + floor)
    per_leaf_count = jnp.sum(weights, axis=-1)
    # A leaf with fewer than two measured spacings has no variance to speak of.
    usable = per_leaf_count >= 2.0
    safe_count = jnp.maximum(per_leaf_count, 1.0)
    mean = jnp.sum(weights * log_d, axis=-1) / safe_count
    centred = (log_d - mean[:, None]) * weights
    variance = jnp.sum(centred**2, axis=-1) / safe_count
    return jnp.sum(jnp.where(usable, variance, 0.0)) / jnp.maximum(jnp.sum(usable), 1)


@dataclass(frozen=True)
class Regularization:
    """Weights for every regularisation term, recorded verbatim in the JSON.

    A weight of ``0.0`` disables a term but still reports its value, which is
    how the unregularised run stays comparable to the regularised ones.

    Attributes
    ----------
    softening_floor : float
        Weight on :func:`softening_floor_penalty`.
    spacing_smoothness : float
        Weight on :func:`spacing_smoothness_penalty`.
    floor_fraction : float
        Passed to :func:`softening_floor_penalty`; not a weight.
    """

    softening_floor: float = 1.0
    spacing_smoothness: float = 1.0e-2
    floor_fraction: float = 1.0

    @classmethod
    def none(cls: type["Regularization"]) -> "Regularization":
        """Return the deliberately unregularised setting.

        Returns
        -------
        Regularization
            All weights zero. Both terms are still *evaluated* and reported --
            fig17(d) needs to show what the degeneracy does, which means
            knowing the penalty values along a run that was not paying them.
        """
        return cls(softening_floor=0.0, spacing_smoothness=0.0)

    @property
    def enabled(self: "Regularization") -> bool:
        """Whether any term carries a non-zero weight.

        Returns
        -------
        bool
            False for the unregularised run.
        """
        return bool(self.softening_floor != 0.0 or self.spacing_smoothness != 0.0)

    def weights(self: "Regularization") -> Dict[str, float]:
        """Return the term weights, including the data term's implicit 1.0.

        Returns
        -------
        Dict[str, float]
            Term name to weight, ready for :func:`total_loss`.
        """
        return {
            "data": 1.0,
            "softening_floor": float(self.softening_floor),
            "spacing_smoothness": float(self.spacing_smoothness),
        }

    def as_record(self: "Regularization") -> Dict[str, Any]:
        """Return a JSON-safe copy for a results file.

        Returns
        -------
        Dict[str, Any]
            Every field.
        """
        return asdict(self)


def regularization_terms(
    positions: ArrayLike,
    blocks: LeafBlocks,
    *,
    softening: float,
    regularization: Regularization,
) -> Dict[str, jnp.ndarray]:
    """Evaluate every regulariser, whatever its weight.

    Parameters
    ----------
    positions : ArrayLike
        ``(N, 3)`` source positions.
    blocks : LeafBlocks
        The frozen leaf partition.
    softening : float
        The Plummer softening length in use.
    regularization : Regularization
        Weights and the floor fraction. Terms are evaluated even at weight
        zero, so an unregularised run still reports what it was not paying.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Term name to unweighted scalar value.
    """
    return {
        "softening_floor": softening_floor_penalty(
            positions,
            blocks,
            softening=softening,
            floor_fraction=regularization.floor_fraction,
        ),
        "spacing_smoothness": spacing_smoothness_penalty(positions, blocks),
    }


def total_loss(
    predicted: ArrayLike,
    observed: ArrayLike,
    *,
    weights: Dict[str, float],
    scale: float = 1.0,
    extra_terms: Optional[Dict[str, Any]] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Weighted sum of the data misfit and any supplied regulariser values.

    Parameters
    ----------
    predicted : ArrayLike
        ``(M, 3)`` model accelerations. NumPy or JAX.
    observed : ArrayLike
        ``(M, 3)`` observed accelerations. NumPy or JAX.
    weights : Dict[str, float]
        Term name to weight. ``"data"`` weights the misfit; other keys weight
        matching entries of ``extra_terms``. A weight of ``0.0`` disables a
        term but still reports it.
    scale : float
        Passed to :func:`data_misfit`.
    extra_terms : Optional[Dict[str, Any]]
        Already-evaluated regulariser scalars by name, from
        :func:`regularization_terms`.

    Returns
    -------
    Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]
        The total, and every component (unweighted) so the results JSON can
        report them alongside the weights.
    """
    components: Dict[str, jnp.ndarray] = {
        "data": data_misfit(predicted, observed, scale=scale)
    }
    for name, value in (extra_terms or {}).items():
        components[name] = jnp.asarray(value)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for name, value in components.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total, components

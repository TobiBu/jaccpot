"""Local expansion buffer helpers for the FMM downward pass."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Tuple
from jax import lax
from jaxtyping import Array, jaxtyped
from yggdrax.dense_interactions import DenseInteractionBuffers
from yggdrax.dtypes import INDEX_DTYPE, as_index
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
    build_well_separated_interactions,
)
from yggdrax.multipole_utils import (
    MAX_MULTIPOLE_ORDER,
    level_offset,
    multi_index_factorial,
    multi_index_tuples,
    multi_power,
    total_coefficients,
)
from yggdrax.tree import Tree
from yggdrax.tree_moments import (
    _hexadecapole_from_fourth,
    _octupole_from_third,
    _quadrupole_from_second,
    multipole_from_packed,
    tree_moments_from_raw,
)

from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData


class LocalExpansionData(NamedTuple):
    """Local expansion coefficients and metadata.

    Attributes
    ----------
    order : int
        Expansion order ``p`` these coefficients were built at.
    centers : Array
        Expansion centre per node, shape ``(num_nodes, 3)``.
    coefficients : Array
        Packed local coefficients per node, shape ``(num_nodes, ncoeff(p))``.
    """

    order: int
    centers: Array
    coefficients: Array


class TreeDownwardData(NamedTuple):
    """Bundle for interaction lists and resulting local expansions.

    Attributes
    ----------
    interactions : NodeInteractionList
        The accepted far-field node pairs this sweep consumed.
    locals : LocalExpansionData
        The local expansions the sweep produced.
    source_motion_locals : Optional[LocalExpansionData]
        Locals for the source-motion (time-derivative) tower, or ``None`` when no
        source motion was requested. ``None`` means "not requested"; an all-zero
        value means "requested, and the far field is empty" -- the two are
        deliberately distinguishable.
    """

    interactions: NodeInteractionList
    locals: LocalExpansionData
    source_motion_locals: Optional[LocalExpansionData] = None


_LEVEL_COMBOS: Dict[int, Tuple[Tuple[int, int, int], ...]] = {
    level: multi_index_tuples(level) for level in range(MAX_MULTIPOLE_ORDER + 1)
}


def _multipole_component_matrix(
    multipoles: NodeMultipoleData,
    *,
    coeff_count: int,
    dtype: Any,
) -> Array:
    """Return the packed multipole coefficients used by the M2L kernels.

    Parameters
    ----------
    multipoles : NodeMultipoleData
        Per-node multipole data from the upward sweep.
    coeff_count : int
        Number of packed coefficients per node.
    dtype : Any
        Working dtype for the allocated buffers.

    Returns
    -------
    Array
        The packed multipole coefficients the Cartesian translation consumes.
    """
    return jnp.asarray(multipoles.packed[:, :coeff_count], dtype=dtype)


_LEVEL_INDEX_LOOKUP: Dict[int, Dict[Tuple[int, int, int], int]] = {
    level: {combo: idx for idx, combo in enumerate(combos)}
    for level, combos in _LEVEL_COMBOS.items()
}

_COMBO_FACTORIAL: Dict[Tuple[int, int, int], int] = {
    combo: multi_index_factorial(combo)
    for combos in _LEVEL_COMBOS.values()
    for combo in combos
}


def _double_factorial(n: int) -> int:
    """Compute n!! for non-negative integers.

    Parameters
    ----------
    n : int
        Non-negative integer argument.

    Returns
    -------
    int
        ``n!!``, the product of every second integer down from ``n``.
    """
    if n <= 0:
        return 1
    result = 1
    value = n
    while value > 1:
        result *= value
        value -= 2
    return result


_LEVEL_DOUBLE_FACTORIAL = tuple(
    _double_factorial(2 * level - 1) for level in range(MAX_MULTIPOLE_ORDER + 1)
)


_MAX_M2L_DERIV_ORDER = 2 * MAX_MULTIPOLE_ORDER

ComponentPowers = Tuple[
    Array,
    Array,
    Array,
]


class _DerivativeInfo(NamedTuple):
    """Metadata for one mixed derivative polynomial.

    Attributes
    ----------
    level : int
        Total derivative order of this polynomial.
    terms : Tuple[Tuple[Tuple[int, int, int], int], ...]
        The polynomial as ``((x_exp, y_exp, z_exp), coefficient)`` pairs, sorted so
        the representation is canonical and hashable.
    """

    level: int
    terms: Tuple[Tuple[Tuple[int, int, int], int], ...]


def _scale_poly(
    poly: Dict[Tuple[int, int, int], int],
    scale: int,
) -> Dict[Tuple[int, int, int], int]:
    """Scale polynomial coefficients by an integer factor.

    Parameters
    ----------
    poly : Dict[Tuple[int, int, int], int]
        Sparse polynomial as ``{(x_exp, y_exp, z_exp): coefficient}``.
    scale : int
        Integer factor to multiply the polynomial by.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        The polynomial with every coefficient multiplied by ``scale``.
    """
    if scale == 0 or not poly:
        return {}
    return {exp: coeff * scale for exp, coeff in poly.items()}


def _add_poly(
    left: Dict[Tuple[int, int, int], int],
    right: Dict[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int], int]:
    """Add two sparse integer polynomials in 3D exponents.

    Parameters
    ----------
    left : Dict[Tuple[int, int, int], int]
        Left operand polynomial, same sparse form.
    right : Dict[Tuple[int, int, int], int]
        Right operand polynomial, same sparse form.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        The sum, with zero coefficients dropped so the form stays canonical.
    """
    if not left:
        return dict(right)
    if not right:
        return dict(left)
    result = dict(left)
    for exp, coeff in right.items():
        value = result.get(exp, 0) + coeff
        if value:
            result[exp] = value
        elif exp in result:
            del result[exp]
    return result


def _differentiate_poly(
    poly: Dict[Tuple[int, int, int], int],
    axis: int,
) -> Dict[Tuple[int, int, int], int]:
    """Differentiate a sparse polynomial with respect to one axis.

    Parameters
    ----------
    poly : Dict[Tuple[int, int, int], int]
        Sparse polynomial as ``{(x_exp, y_exp, z_exp): coefficient}``.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, 2 for z.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        The derivative with respect to the selected axis.
    """
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        power = exp[axis]
        if power == 0:
            continue
        new_exp = list(exp)
        new_exp[axis] -= 1
        # Spelled out rather than `tuple(new_exp)` so the length survives: these
        # dicts are keyed by (x, y, z) exponent triples, and `tuple(list)` erases
        # that to `tuple[int, ...]`, which then fails to match the declared key
        # type at every use (9 errors, audit E.4 bucket L).
        key: Tuple[int, int, int] = (new_exp[0], new_exp[1], new_exp[2])
        result[key] = result.get(key, 0) + coeff * power
    return result


def _mul_by_axis(
    poly: Dict[Tuple[int, int, int], int],
    axis: int,
) -> Dict[Tuple[int, int, int], int]:
    """Multiply a sparse polynomial by x/y/z for the selected axis.

    Parameters
    ----------
    poly : Dict[Tuple[int, int, int], int]
        Sparse polynomial as ``{(x_exp, y_exp, z_exp): coefficient}``.
    axis : int
        Cartesian axis index: 0 for x, 1 for y, 2 for z.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        The polynomial multiplied by ``x``, ``y`` or ``z``.
    """
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        new_exp = list(exp)
        new_exp[axis] += 1
        # Spelled out rather than `tuple(new_exp)` so the length survives: these
        # dicts are keyed by (x, y, z) exponent triples, and `tuple(list)` erases
        # that to `tuple[int, ...]`, which then fails to match the declared key
        # type at every use (9 errors, audit E.4 bucket L).
        key: Tuple[int, int, int] = (new_exp[0], new_exp[1], new_exp[2])
        result[key] = result.get(key, 0) + coeff
    return result


def _mul_by_r2(
    poly: Dict[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int], int]:
    """Multiply a sparse polynomial by r^2 = x^2 + y^2 + z^2.

    Parameters
    ----------
    poly : Dict[Tuple[int, int, int], int]
        Sparse polynomial as ``{(x_exp, y_exp, z_exp): coefficient}``.

    Returns
    -------
    Dict[Tuple[int, int, int], int]
        The polynomial multiplied by ``r**2``.
    """
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        for axis in range(3):
            new_exp = list(exp)
            new_exp[axis] += 2
            # Spelled out rather than `tuple(new_exp)` so the length survives: these
            # dicts are keyed by (x, y, z) exponent triples, and `tuple(list)` erases
            # that to `tuple[int, ...]`, which then fails to match the declared key
            # type at every use (9 errors, audit E.4 bucket L).
            key: Tuple[int, int, int] = (new_exp[0], new_exp[1], new_exp[2])
            result[key] = result.get(key, 0) + coeff
    return result


def _term_tuple(
    poly: Dict[Tuple[int, int, int], int],
) -> Tuple[Tuple[Tuple[int, int, int], int], ...]:
    """Canonicalize sparse polynomial dict into a sorted tuple.

    Parameters
    ----------
    poly : Dict[Tuple[int, int, int], int]
        Sparse polynomial as ``{(x_exp, y_exp, z_exp): coefficient}``.

    Returns
    -------
    Tuple[Tuple[Tuple[int, int, int], int], ...]
        The polynomial as a sorted tuple, so it is hashable and comparable.
    """
    return tuple(sorted(poly.items()))


def _generate_derivative_info() -> Tuple[
    Dict[Tuple[int, int, int], _DerivativeInfo],
    int,
]:
    """Generate derivative polynomial metadata up to the supported order.

    Returns
    -------
    Dict[Tuple[int, int, int], _DerivativeInfo]
        Metadata for every mixed derivative, keyed by its exponent triple.
    int
        The highest level the recurrence closed over, i.e. the supported order.

    Raises
    ------
    RuntimeError
        If the recurrence fails to close over the requested order range.
    """
    combos_by_level = {
        level: multi_index_tuples(level) for level in range(_MAX_M2L_DERIV_ORDER + 1)
    }

    polynomials: Dict[
        Tuple[int, int, int],
        Dict[Tuple[int, int, int], int],
    ] = {(0, 0, 0): {(0, 0, 0): 1}}

    for level in range(_MAX_M2L_DERIV_ORDER):
        for combo in combos_by_level[level]:
            base_poly = polynomials[combo]
            for axis in range(3):
                child = list(combo)
                child[axis] += 1
                child = tuple(child)
                if sum(child) != level + 1:
                    continue
                scaled = _scale_poly(
                    _mul_by_axis(base_poly, axis),
                    2 * level + 1,
                )
                derivative = _differentiate_poly(base_poly, axis)
                if derivative:
                    scaled = _add_poly(
                        scaled,
                        _scale_poly(_mul_by_r2(derivative), -1),
                    )
                scaled = {exp: coeff for exp, coeff in scaled.items() if coeff != 0}
                if child in polynomials:
                    if scaled != polynomials[child]:
                        raise RuntimeError(
                            "Inconsistent derivative polynomial construction",
                        )
                else:
                    polynomials[child] = scaled

    derivative_info: Dict[Tuple[int, int, int], _DerivativeInfo] = {}
    max_exponent = 0
    for combo, poly in polynomials.items():
        terms = _term_tuple(poly)
        derivative_info[combo] = _DerivativeInfo(level=sum(combo), terms=terms)
        for exp, _coeff in terms:
            max_exponent = max(max_exponent, exp[0], exp[1], exp[2])

    return derivative_info, max_exponent


_DERIVATIVE_INFO, _MAX_POLY_EXPONENT = _generate_derivative_info()


def _build_component_powers(
    delta: Array,
    max_exponent: int,
) -> ComponentPowers:
    """Precompute per-axis powers of displacement components.

    Parameters
    ----------
    delta : Array
        Displacement vector ``(3,)``, target centre minus source centre.
    max_exponent : int
        Highest power of each component to precompute.

    Returns
    -------
    ComponentPowers
        Per-axis powers of the displacement components, precomputed once.
    """
    dtype = delta.dtype
    powers = []
    for axis in range(3):
        values = [dtype.type(1.0)]
        base = delta[axis]
        current = dtype.type(1.0)
        for _ in range(1, max_exponent + 1):
            current = current * base
            values.append(current)
        powers.append(jnp.stack(values))
    return tuple(powers)  # type: ignore[return-value]


def _build_derivative_tables() -> (
    Tuple[Array, Array, Array, Array, Array, Dict[Tuple[int, int, int], int]]
):
    """Materialize derivative metadata tables for vectorized lookup.

    Returns
    -------
    Tuple[Array, Array, Array, Array, Array, Dict[Tuple[int, int, int], int]]
        ``(combos, levels, term_exp, term_coeff, term_mask, lookup)``: the exponent
        triple and level of each derivative-basis entry, its terms' exponents,
        coefficients and validity mask padded to the widest polynomial, and a dict
        mapping an exponent triple to its row index.

        The return annotation is added by this change, and it is the one non-docstring
        edit in the batch. It is not optional: with no annotation pydoclint demands a
        ``Returns`` section (DOC201) but rejects a section that names a type (DOC203)
        and cannot parse one that does not (DOC001), so the file cannot reach zero
        without it.
    """
    combos: List[Tuple[int, int, int]] = []
    levels: List[int] = []
    term_lists: List[List[Tuple[int, int, int]]] = []
    coeff_lists: List[List[int]] = []

    sorted_items = sorted(
        _DERIVATIVE_INFO.items(),
        key=lambda item: (item[1].level, item[0]),
    )
    for combo, info in sorted_items:
        combos.append(combo)
        levels.append(int(info.level))
        term_lists.append([exp for exp, _coeff in info.terms])
        coeff_lists.append([int(_coeff) for _exp, _coeff in info.terms])

    max_terms = max(len(terms) for terms in term_lists)

    term_exp = np.zeros((len(combos), max_terms, 3), dtype=np.int64)
    term_coeff = np.zeros((len(combos), max_terms), dtype=np.int64)
    term_mask = np.zeros((len(combos), max_terms), dtype=bool)

    for idx, terms in enumerate(term_lists):
        coeffs = coeff_lists[idx]
        for term_idx, exp in enumerate(terms):
            term_exp[idx, term_idx, :] = exp
            term_coeff[idx, term_idx] = coeffs[term_idx]
            term_mask[idx, term_idx] = True

    combos_arr = np.asarray(combos, dtype=np.int64)
    levels_arr = np.asarray(levels, dtype=np.int64)

    lookup = {tuple(combo): int(idx) for idx, combo in enumerate(combos)}

    return (
        jnp.asarray(combos_arr, dtype=INDEX_DTYPE),
        jnp.asarray(levels_arr, dtype=INDEX_DTYPE),
        jnp.asarray(term_exp, dtype=INDEX_DTYPE),
        jnp.asarray(term_coeff, dtype=INDEX_DTYPE),
        jnp.asarray(term_mask, dtype=jnp.bool_),
        lookup,
    )


(
    _DERIVATIVE_COMBOS,
    _DERIVATIVE_LEVELS,
    _DERIVATIVE_TERM_EXP,
    _DERIVATIVE_TERM_COEFF,
    _DERIVATIVE_TERM_MASK,
    _DERIVATIVE_LOOKUP,
) = _build_derivative_tables()


class _M2LStencil(NamedTuple):
    """Precomputed index/scale stencil used by Cartesian M2L.

    Attributes
    ----------
    gamma_indices : Array
        Index of the derivative-basis entry each output coefficient draws on.
    scales : Array
        Multiplicative factor paired with each gathered entry.
    component_sizes : Tuple[int, ...]
        Number of coefficients per level, so the flat stencil can be split back
        into levels without recomputing the triangular sizes.
    """

    gamma_indices: Array
    scales: Array
    component_sizes: Tuple[int, ...]


def _build_m2l_stencils() -> Tuple[_M2LStencil, ...]:
    """Build M2L stencils for every order up to MAX_MULTIPOLE_ORDER.

    Returns
    -------
    Tuple[_M2LStencil, ...]
        One stencil per order up to ``MAX_MULTIPOLE_ORDER``.
    """
    stencils: List[_M2LStencil] = []
    for order in range(MAX_MULTIPOLE_ORDER + 1):
        total_targets = total_coefficients(order)
        component_sizes = tuple(len(_LEVEL_COMBOS[level]) for level in range(order + 1))
        total_components = int(sum(component_sizes))

        gamma_indices = np.zeros(
            (total_targets, total_components),
            dtype=np.int64,
        )
        scales = np.zeros(
            (total_targets, total_components),
            dtype=np.float64,
        )

        for level in range(order + 1):
            combos_alpha = _LEVEL_COMBOS[level]
            target_base = level_offset(level)
            for alpha_idx, alpha in enumerate(combos_alpha):
                target_idx = target_base + alpha_idx
                component_offset = 0
                for m in range(order + 1):
                    combos_beta = _LEVEL_COMBOS[m]
                    sign = -1 if (m % 2) else 1
                    double_factorial = float(_LEVEL_DOUBLE_FACTORIAL[m])
                    for beta_idx, beta in enumerate(combos_beta):
                        gamma = (
                            alpha[0] + beta[0],
                            alpha[1] + beta[1],
                            alpha[2] + beta[2],
                        )
                        gamma_idx = _DERIVATIVE_LOOKUP[gamma]
                        combo_factorial = float(_COMBO_FACTORIAL[beta])
                        scale = sign / (combo_factorial * double_factorial)
                        col_idx = component_offset + beta_idx
                        gamma_indices[target_idx, col_idx] = gamma_idx
                        scales[target_idx, col_idx] = scale
                    component_offset += len(combos_beta)

        stencils.append(
            _M2LStencil(
                gamma_indices=jnp.asarray(gamma_indices, dtype=INDEX_DTYPE),
                scales=jnp.asarray(scales, dtype=jnp.float64),
                component_sizes=component_sizes,
            )
        )

    return tuple(stencils)


_M2L_STENCILS = _build_m2l_stencils()

# Empirically chosen batch size that balances fusion and peak memory for the
# chunked M2L accumulation. Adjust via the ``chunk_size`` argument when needed.
DEFAULT_M2L_CHUNK_SIZE = 8192  # 4096  # 2048  # 1024  # 512  # 256


def _evaluate_derivative_table(displacement: Array, max_level: int) -> Array:
    """Evaluate all derivative basis entries at one displacement.

    Parameters
    ----------
    displacement : Array
        Displacement the derivative basis is evaluated at.
    max_level : int
        Highest derivative level to tabulate.

    Returns
    -------
    Array
        Every derivative basis entry evaluated at one displacement.
    """
    dtype = displacement.dtype

    comp_powers = _build_component_powers(displacement, _MAX_POLY_EXPONENT)
    comp_x, comp_y, comp_z = comp_powers

    exp_x = _DERIVATIVE_TERM_EXP[..., 0]
    exp_y = _DERIVATIVE_TERM_EXP[..., 1]
    exp_z = _DERIVATIVE_TERM_EXP[..., 2]

    coeff = _DERIVATIVE_TERM_COEFF.astype(dtype)
    mask = _DERIVATIVE_TERM_MASK.astype(dtype)

    term_values = coeff * comp_x[exp_x] * comp_y[exp_y] * comp_z[exp_z]
    poly_vals = jnp.sum(term_values * mask, axis=1)

    levels = _DERIVATIVE_LEVELS
    sign = jnp.where(
        (levels % 2) == 0,
        jnp.ones_like(levels, dtype=dtype),
        -jnp.ones_like(levels, dtype=dtype),
    )

    r2 = jnp.dot(displacement, displacement, precision=lax.Precision.HIGHEST)
    eps = jnp.finfo(dtype).tiny
    inv_r = jnp.reciprocal(jnp.sqrt(jnp.maximum(r2, eps)))

    powers = (2 * levels) + 1
    inv_r_powers = jnp.power(inv_r, powers.astype(INDEX_DTYPE))

    values = sign * poly_vals * inv_r_powers

    valid = jnp.where(
        levels <= max_level,
        jnp.ones_like(levels, dtype=dtype),
        jnp.zeros_like(levels, dtype=dtype),
    )
    return values * valid


@partial(jax.jit, static_argnames=("order",))
def _translate_components_to_local(
    component_vec: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Translate one packed component vector into local coefficients.

    Parameters
    ----------
    component_vec : Array
        Packed multipole component vector for one node.
    delta : Array
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Local coefficients for one node.
    """
    batched = _translate_components_batch(
        component_vec[None, :],
        delta[None, :],
        order=order,
    )
    return jnp.squeeze(batched, axis=0)


@partial(jax.jit, static_argnames=("order",))
def _translate_components_batch(
    component_chunk: Array,
    delta_chunk: Array,
    *,
    order: int,
) -> Array:
    """Batch version of component-to-local translation.

    Parameters
    ----------
    component_chunk : Array
        Packed component vectors for one chunk of nodes.
    delta_chunk : Array
        Displacement vectors for one chunk, shape ``(chunk, 3)``.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        Local coefficients for a chunk of nodes.
    """
    zero_disp = jnp.all(delta_chunk == 0, axis=1)
    derivative_chunk = jax.vmap(
        lambda disp: _evaluate_derivative_table(disp, order * 2),
    )(delta_chunk)
    stencil = _M2L_STENCILS[order]
    dtype = component_chunk.dtype
    scales = stencil.scales.astype(dtype)
    gamma_flat = stencil.gamma_indices.reshape(-1)
    gathered = jnp.take(derivative_chunk, gamma_flat, axis=1)
    gathered = gathered.reshape(
        derivative_chunk.shape[0],
        stencil.gamma_indices.shape[0],
        stencil.gamma_indices.shape[1],
    )
    weighted = gathered * scales
    translated = jnp.einsum(
        "nqc,nc->nq",
        weighted,
        component_chunk,
        precision=lax.Precision.HIGHEST,
    )
    return jnp.where(zero_disp[:, None], 0, translated)


@partial(jax.jit, static_argnames=("order", "chunk_size"))
def _accumulate_level(
    coeffs: Array,
    component_matrix: Array,
    centers_target: Array,
    centers_source: Array,
    sources: Array,
    offsets: Array,
    counts: Array,
    *,
    order: int,
    chunk_size: int,
) -> Array:
    """Accumulate M2L contributions in fixed-size chunks per target node.

    The Cartesian-basis level accumulator. Interaction pairs arrive as a CSR-ish
    ``(offsets, counts)`` slice per target node and are consumed ``chunk_size`` at
    a time so the traced shape is fixed regardless of how many sources a target
    actually has; ``sources`` is padded by one full chunk for that reason, and the
    over-run slots are masked out rather than clamped into a real source.

    **Accumulation order is load-bearing** (NUMERICS_AND_JAX §1): the per-chunk
    scan-and-add is the order the goldens were taken in. ``chunk_size`` changes
    the batching, not the sum, but reassociating the adds inside a chunk would
    change the last digits.

    Parameters
    ----------
    coeffs : Array
        Local-expansion coefficients to accumulate into,
        ``[num_targets, num_components]``. Returned updated.
    component_matrix : Array
        Per-source-node packed multipole components,
        ``[num_source_nodes, num_components]``.
    centers_target : Array
        Target node centres ``[num_targets, 3]``.
    centers_source : Array
        Source node centres ``[num_source_nodes, 3]``.
    sources : Array
        Flat source-node index per interaction pair, ``[num_pairs]``.
    offsets : Array
        Start offset into ``sources`` for each target, ``[num_targets]``.
    counts : Array
        Number of sources for each target, ``[num_targets]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    chunk_size : int
        Pairs consumed per scan step, floored at 1. Static under ``jit``.

    Returns
    -------
    Array
        The updated local coefficients, same shape as ``coeffs``. Returned
        unchanged when there are no pairs.
    """
    pair_count = sources.shape[0]
    if pair_count == 0:
        return coeffs

    chunk = int(max(chunk_size, 1))
    padded_len = pair_count + chunk
    sources_padded = jnp.pad(
        sources,
        (0, padded_len - pair_count),
        mode="constant",
        constant_values=0,
    )

    num_targets = centers_target.shape[0]
    slot_idx = jnp.arange(chunk, dtype=INDEX_DTYPE)

    def target_body(target_idx: Array, coeff_state: Array) -> Array:
        start = offsets[target_idx]
        count = counts[target_idx]

        def accumulate_target(state: Array) -> Array:
            steps = (count + chunk - 1) // chunk
            target_center = centers_target[target_idx]

            def chunk_body(step_idx: Array, inner_state: Array) -> Array:
                chunk_start = start + step_idx * chunk
                remaining = count - step_idx * chunk
                chunk_len = jnp.minimum(chunk, jnp.maximum(remaining, 0))
                src_chunk = lax.dynamic_slice_in_dim(
                    sources_padded,
                    chunk_start,
                    chunk,
                    axis=0,
                )
                valid = slot_idx < chunk_len
                safe_src = jnp.where(valid, src_chunk, as_index(0))
                delta_slice = target_center - centers_source[safe_src]
                component_slice = component_matrix[safe_src]
                contrib = _translate_components_batch(
                    component_slice,
                    delta_slice,
                    order=order,
                )
                contrib = jnp.where(valid[:, None], contrib, 0.0)
                total = jnp.sum(contrib, axis=0)
                return inner_state.at[target_idx].add(total)

            return lax.fori_loop(0, steps, chunk_body, state)

        return lax.cond(
            count > 0,
            accumulate_target,
            lambda s: s,
            coeff_state,
        )

    return lax.fori_loop(0, num_targets, target_body, coeffs)


@partial(jax.jit, static_argnames=("order",))
def _accumulate_dense_m2l_impl(
    coeffs: Array,
    component_matrix: Array,
    node_indices: Array,
    sources: Array,
    mask: Array,
    centers_target: Array,
    centers_source: Array,
    *,
    order: int,
) -> Array:
    """Dense-buffer M2L accumulation kernel.

    The fixed-capacity alternative to :func:`_accumulate_level`: instead of a CSR
    slice per target, interactions arrive already laid out as
    ``[levels, slots_per_level, pairs_per_slot]`` with a validity ``mask``. That
    makes every shape static without any padding arithmetic at trace time, which
    is what the dense-interaction traversal path wants; the cost is that the
    buffers are sized for the worst case.

    Invalid slots are dropped through ``mask``, and the source index is clamped to
    ``component_matrix``'s last row so a masked slot cannot gather out of bounds.
    **Accumulation order is load-bearing** (NUMERICS_AND_JAX §1).

    Parameters
    ----------
    coeffs : Array
        Local-expansion coefficients to accumulate into,
        ``[num_nodes, num_components]``. Returned updated.
    component_matrix : Array
        Per-source-node packed multipole components,
        ``[num_source_nodes, num_components]``.
    node_indices : Array
        Target node index per slot, ``[levels, slots_per_level]``.
    sources : Array
        Source node indices, ``[levels, slots_per_level, pairs_per_slot]``.
    mask : Array
        Validity mask with the same shape as ``sources``; ``False`` slots
        contribute nothing.
    centers_target : Array
        Node centres used for the target side, ``[num_nodes, 3]``.
    centers_source : Array
        Node centres used for the source side, ``[num_source_nodes, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.

    Returns
    -------
    Array
        The updated local coefficients, same shape as ``coeffs``.
    """
    levels = int(node_indices.shape[0])
    slots_per_level = int(node_indices.shape[1])
    total_slots = levels * slots_per_level
    pairs_per_slot = int(sources.shape[2])

    nodes_flat = jnp.reshape(node_indices, (total_slots,))
    sources_flat = jnp.reshape(sources, (total_slots, pairs_per_slot))
    mask_flat = jnp.reshape(mask, (total_slots, pairs_per_slot))

    coeff_dtype = coeffs.dtype
    max_source_idx = component_matrix.shape[0] - 1

    def body(idx: Array, coeff_state: Array) -> Array:
        node = nodes_flat[idx]

        def accumulate_target(state_coeffs: Array) -> Array:
            mask_row = mask_flat[idx]
            has_pairs = jnp.any(mask_row)

            def compute(inner_state: Array) -> Array:
                safe_sources = jnp.clip(
                    sources_flat[idx],
                    as_index(0),
                    as_index(max_source_idx),
                )
                component_slice = component_matrix[safe_sources]
                target_center = centers_target[node]
                source_centers = centers_source[safe_sources]
                delta_slice = target_center - source_centers
                contrib = _translate_components_batch(
                    component_slice,
                    delta_slice,
                    order=order,
                )
                mask_vals = mask_row[:, None].astype(coeff_dtype)
                masked = contrib * mask_vals
                total = jnp.sum(masked, axis=0)
                return inner_state.at[node].add(total)

            return lax.cond(
                has_pairs,
                compute,
                lambda s: s,
                state_coeffs,
            )

        return lax.cond(
            node >= 0,
            accumulate_target,
            lambda s: s,
            coeff_state,
        )

    return lax.fori_loop(0, total_slots, body, coeffs)


@jaxtyped(typechecker=beartype)
def accumulate_dense_m2l_contributions(
    dense_buffers: DenseInteractionBuffers,
    multipoles: NodeMultipoleData,
    local_data: LocalExpansionData,
) -> LocalExpansionData:
    """Accumulate M2L contributions using dense level-major buffers.

    Parameters
    ----------
    dense_buffers : DenseInteractionBuffers
        Dense level-major interaction buffers.
    multipoles : NodeMultipoleData
        Per-node multipole data from the upward sweep.
    local_data : LocalExpansionData
        The local expansions being accumulated into.

    Returns
    -------
    LocalExpansionData
        The local expansions with the dense M2L contributions added.

    Raises
    ------
    ValueError
        If the dense buffers disagree with the interaction list or the order.
    """

    if multipoles.order != local_data.order:
        raise ValueError("multipole and local orders must match")

    centers = jnp.asarray(local_data.centers)
    coeffs = jnp.asarray(local_data.coefficients)
    order = int(local_data.order)

    if centers.shape != multipoles.centers.shape:
        raise ValueError("local centers must align with multipole centers")

    coeff_count = int(coeffs.shape[1])
    component_matrix = _multipole_component_matrix(
        multipoles,
        coeff_count=coeff_count,
        dtype=coeffs.dtype,
    )
    if component_matrix.shape[0] == 0:
        raise ValueError("component_matrix must contain at least one node")

    node_indices = jnp.asarray(
        dense_buffers.geometry.node_indices,
        dtype=INDEX_DTYPE,
    )
    sources = jnp.asarray(dense_buffers.m2l_sources, dtype=INDEX_DTYPE)
    mask = jnp.asarray(dense_buffers.m2l_mask, dtype=jnp.bool_)

    if sources.shape[:2] != node_indices.shape:
        raise ValueError("dense buffer layout does not match node indices")

    updated_coeffs = _accumulate_dense_m2l_impl(
        coeffs,
        component_matrix,
        node_indices,
        sources,
        mask,
        centers,
        jnp.asarray(multipoles.centers, dtype=coeffs.dtype),
        order=order,
    )

    return LocalExpansionData(
        order=order,
        centers=centers,
        coefficients=updated_coeffs,
    )


@partial(jax.jit, static_argnames=("order",))
def _translate_multipole_to_local_impl(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    mass: Array,
    dipole: Array,
    second: Array,
    third: Array,
    fourth: Array,
) -> Array:
    """Low-level Cartesian multipole-to-local translation implementation.

    Builds the raw component vector from the central moments and contracts it
    against the Cartesian translation operator for ``delta``. This is the
    implementation half of :func:`translate_multipole_to_local`, split out so the
    moment-recovery fallback in the public wrapper stays separate from the algebra.

    ZERO-DISPLACEMENT GUARD. A ``delta`` of exactly zero returns zeros rather than
    the translation, because a coincident source and target is not a far-field
    pair at all -- it arises only from a degenerate single-child node whose centre
    equals its parent's. Note this guard is forward-only in intent; it is the same
    ``where``-on-a-degenerate-branch shape that G.10 showed can zero a real
    gradient component, and it has not been audited for that here.

    Parameters
    ----------
    multipole : Array
        Packed Cartesian multipole coefficients for one source node.
    delta : Array
        Target-minus-source centre displacement ``[3]``.
    order : int
        Expansion order ``p``, at most ``MAX_MULTIPOLE_ORDER``. Static under
        ``jit``.
    mass : Array
        Monopole moment (scalar).
    dipole : Array
        Dipole moment ``[3]``.
    second : Array
        Second central moment, packed symmetric.
    third : Array
        Third central moment, packed symmetric.
    fourth : Array
        Fourth central moment, packed symmetric.

    Returns
    -------
    Array
        Local-expansion coefficients at ``delta``, in ``multipole``'s dtype; all
        zero when ``delta`` is exactly zero.
    """
    dtype = multipole.dtype
    component_vec = _build_component_vector(
        mass,
        dipole,
        second,
        third,
        fourth,
        order=order,
    ).astype(dtype)
    translated = _translate_components_to_local(
        component_vec,
        delta,
        order=order,
    )
    zero_disp = jnp.all(jnp.asarray(delta, dtype=dtype) == 0)
    return jnp.where(zero_disp, jnp.zeros_like(translated), translated)


def _pack_symmetric_tensor(tensor: Array, level: int) -> Array:
    """Pack a symmetric tensor level into triangular coefficient ordering.

    Parameters
    ----------
    tensor : Array
        Symmetric tensor level to pack into triangular form.
    level : int
        Tensor level (total derivative order).

    Returns
    -------
    Array
        The tensor level packed into triangular coefficient order.
    """
    combos = _LEVEL_COMBOS[level]
    if level == 0:
        return jnp.reshape(jnp.asarray(tensor, dtype=tensor.dtype), (1,))
    values = []
    for combo in combos:
        idx = (0,) * combo[0] + (1,) * combo[1] + (2,) * combo[2]
        values.append(tensor[idx])
    return jnp.stack(values)


@partial(jax.jit, static_argnames=("order",))
def _build_component_vector(
    mass: Array,
    dipole: Array,
    second: Array,
    third: Array,
    fourth: Array,
    *,
    order: int,
) -> Array:
    """Build packed multipole component vector from raw tensor moments.

    Parameters
    ----------
    mass : Array
        Monopole term.
    dipole : Array
        Dipole term.
    second : Array
        Second moment.
    third : Array
        Third moment.
    fourth : Array
        Fourth moment.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        The packed multipole component vector.

    Raises
    ------
    ValueError
        If the supplied moments do not match the requested order.
    """
    tensors = (mass, dipole, second, third, fourth)
    pieces: List[Array] = []
    for level in range(order + 1):
        packed = _pack_symmetric_tensor(tensors[level], level).reshape(-1)
        pieces.append(packed)
    if not pieces:
        raise ValueError("M2L component vector requires at least order 0")
    return jnp.concatenate(pieces, axis=0)


@jaxtyped(typechecker=beartype)
def translate_local_expansion(
    coefficients: Array,
    delta: Array,
    *,
    order: int,
) -> Array:
    """Shift a local expansion by ``delta`` using explicit binomial sums.

    Parameters
    ----------
    coefficients : Array
        Packed coefficients to wrap or reinterpret.
    delta : Array
        Displacement vector ``(3,)``, target centre minus source centre.
    order : int
        Expansion order ``p``.

    Returns
    -------
    Array
        The local expansion shifted by ``delta``.

    Raises
    ------
    NotImplementedError
        If the requested order exceeds the tabulated derivative metadata.
    ValueError
        If ``delta`` or the coefficient length is the wrong shape.
    """

    order_int = int(order)
    if order_int < 0:
        raise ValueError("order must be >= 0")
    if order_int > MAX_MULTIPOLE_ORDER:
        raise NotImplementedError("orders above 4 are not supported")

    dtype = coefficients.dtype
    delta_vec = jnp.asarray(delta, dtype=dtype)

    delta_powers = {
        level: jnp.asarray(
            [multi_power(delta_vec, combo) for combo in _LEVEL_COMBOS[level]],
            dtype=dtype,
        )
        for level in range(order_int + 1)
    }

    total = total_coefficients(order_int)
    result = jnp.zeros((total,), dtype=dtype)

    for level in range(order_int + 1):
        combos_alpha = _LEVEL_COMBOS[level]
        translated = []
        for alpha in combos_alpha:
            accum = jnp.array(0.0, dtype=dtype)
            for higher in range(level, order_int + 1):
                combos_beta = _LEVEL_COMBOS[higher]
                start_high = level_offset(higher)
                end_high = start_high + len(combos_beta)
                coeff_slice = coefficients[start_high:end_high]
                for idx_beta, beta in enumerate(combos_beta):
                    if beta[0] < alpha[0] or beta[1] < alpha[1] or beta[2] < alpha[2]:
                        continue
                    gamma = (
                        beta[0] - alpha[0],
                        beta[1] - alpha[1],
                        beta[2] - alpha[2],
                    )
                    gamma_level = gamma[0] + gamma[1] + gamma[2]
                    gamma_idx = _LEVEL_INDEX_LOOKUP[gamma_level][gamma]
                    gamma_factorial = jnp.asarray(
                        _COMBO_FACTORIAL[gamma],
                        dtype=dtype,
                    )
                    delta_term = delta_powers[gamma_level][gamma_idx]
                    accum = accum + (
                        coeff_slice[idx_beta] * delta_term / gamma_factorial
                    )
            translated.append(accum)

        start = level_offset(level)
        end = start + len(translated)
        result = result.at[start:end].set(jnp.stack(translated))

    return result


def _missing_raw_moment_message(name: str, order: int) -> str:
    """Return the error text for a raw moment the requested order needs.

    Parameters
    ----------
    name : str
        The omitted keyword argument, e.g. ``"raw_second"``.
    order : int
        The resolved expansion order that requires it.

    Returns
    -------
    str
        A message naming the argument, the order, and the way out.
    """
    return (
        f"translate_multipole_to_local was given `raw_mass`, which selects the "
        f"raw-moment path, but order={order} also needs `{name}`. Supply every "
        "raw moment up to the order, or omit `raw_mass` too and let the "
        "packed-coefficient fallback recover them."
    )


@jaxtyped(typechecker=beartype)
def translate_multipole_to_local(
    multipole: Array,
    delta: Array,
    *,
    order: int,
    raw_mass: Optional[Array] = None,
    raw_dipole: Optional[Array] = None,
    raw_second: Optional[Array] = None,
    raw_third: Optional[Array] = None,
    raw_fourth: Optional[Array] = None,
) -> Array:
    """Convert a multipole expansion into a local expansion at ``delta``.

    When raw central moments for the source multipole are available they can be
    provided via ``raw_mass``/``raw_dipole``/etc.  This avoids the need to
    recover these moments from the packed coefficients (which may store
    symmetric trace-free tensors).  Callers that only have raw packed moments
    may omit these arguments and rely on the slower fallback path.

    Cartesian basis only, and **the Cartesian basis is experimental**: its
    relative L2 force error is ~1.8e-1 independent of expansion order -- a
    divergent-series signature, not truncation -- so raising ``order`` does not
    improve it. Use the real or solidfmm basis for quantitative work.

    Parameters
    ----------
    multipole : Array
        Packed Cartesian multipole coefficients for one source node.
    delta : Array
        Target-minus-source centre displacement ``[3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.
    raw_mass : Optional[Array]
        Monopole moment, if the caller already has it; recovered from
        ``multipole`` when ``None``.
    raw_dipole : Optional[Array]
        Dipole moment ``[3]``, same convention.
    raw_second : Optional[Array]
        Second central moment, packed symmetric, same convention.
    raw_third : Optional[Array]
        Third central moment, same convention.
    raw_fourth : Optional[Array]
        Fourth central moment, same convention.

    Returns
    -------
    Array
        Local-expansion coefficients at ``delta``.

    Raises
    ------
    ValueError
        If ``order`` is negative, or if ``raw_mass`` is given without every other
        raw moment the order needs.
    NotImplementedError
        If ``order`` exceeds ``MAX_MULTIPOLE_ORDER`` (4).
    """

    order_int = int(order)
    if order_int < 0:
        raise ValueError("order must be >= 0")
    if order_int > MAX_MULTIPOLE_ORDER:
        raise NotImplementedError("orders above 4 are not supported")

    dtype = multipole.dtype
    displacement = jnp.asarray(delta, dtype=dtype)

    total = total_coefficients(order_int)
    packed = multipole[:total]
    raw_supplied = raw_mass is not None

    if raw_supplied:
        # `raw_mass` alone selects this branch, so each moment the order needs
        # must be checked before it is used. Without these, a caller that
        # supplies `raw_mass` and omits one reached yggdrax's
        # `_quadrupole_from_second` (or a sibling) with `None` and failed there --
        # `TypeError: trace requires ndarray or scalar arguments`, from a frame
        # naming neither the argument nor this function. Same values, same
        # failure-vs-success cases; only the message changes.
        #
        # Checked at each use rather than once up front so the narrowing is
        # visible: a list comprehension collecting the missing names reads better
        # but leaves every `raw_*` still `Optional` to a type checker.
        mass = jnp.asarray(raw_mass, dtype=dtype)
        if order_int >= 1:
            if raw_dipole is None:
                raise ValueError(_missing_raw_moment_message("raw_dipole", order_int))
            dipole = jnp.asarray(raw_dipole, dtype=dtype)
        else:
            dipole = jnp.zeros((3,), dtype=dtype)
        if order_int >= 2:
            if raw_second is None:
                raise ValueError(_missing_raw_moment_message("raw_second", order_int))
            second = jnp.asarray(
                _quadrupole_from_second(raw_second),
                dtype=dtype,
            )
        else:
            second = jnp.zeros((3, 3), dtype=dtype)
        if order_int >= 3:
            if raw_third is None:
                raise ValueError(_missing_raw_moment_message("raw_third", order_int))
            third = jnp.asarray(
                _octupole_from_third(raw_third),
                dtype=dtype,
            )
        else:
            third = jnp.zeros((3, 3, 3), dtype=dtype)
        if order_int >= 4:
            if raw_fourth is None:
                raise ValueError(_missing_raw_moment_message("raw_fourth", order_int))
            fourth = jnp.asarray(
                _hexadecapole_from_fourth(raw_fourth),
                dtype=dtype,
            )
        else:
            fourth = jnp.zeros((3, 3, 3, 3), dtype=dtype)
    else:
        dummy_center = jnp.zeros((1, 3), dtype=dtype)
        raw_moments = tree_moments_from_raw(
            packed[jnp.newaxis, :],
            dummy_center,
            order_int,
        )
        stf_moments = multipole_from_packed(
            packed[jnp.newaxis, :],
            dummy_center,
            order_int,
        )

        if order_int >= 2:
            start_lvl2 = level_offset(2)
            end_lvl2 = level_offset(3)
            level2 = packed[start_lvl2:end_lvl2]
            lookup2 = _LEVEL_INDEX_LOOKUP[2]
            trace_diag = (
                level2[lookup2[(2, 0, 0)]]
                + level2[lookup2[(0, 2, 0)]]
                + level2[lookup2[(0, 0, 2)]]
            )
            scale = jnp.max(jnp.abs(level2))
            tol = dtype.type(1e-10) * jnp.maximum(scale, dtype.type(1.0))
            use_raw = jnp.abs(trace_diag) > tol
        else:
            use_raw = jnp.asarray(False, dtype=jnp.bool_)

        mass = jnp.where(use_raw, raw_moments.mass[0], stf_moments.mass[0])
        if order_int >= 1:
            dipole = jnp.where(
                use_raw,
                raw_moments.dipole[0],
                stf_moments.dipole[0],
            )
        else:
            dipole = jnp.zeros((3,), dtype=dtype)
        if order_int >= 2:
            second = jnp.where(
                use_raw,
                raw_moments.quadrupole[0],
                stf_moments.quadrupole[0],
            )
        else:
            second = jnp.zeros((3, 3), dtype=dtype)
        if order_int >= 3:
            third = jnp.where(
                use_raw,
                raw_moments.octupole[0],
                stf_moments.octupole[0],
            )
        else:
            third = jnp.zeros((3, 3, 3), dtype=dtype)
        if order_int >= 4:
            fourth = jnp.where(
                use_raw,
                raw_moments.hexadecapole[0],
                stf_moments.hexadecapole[0],
            )
        else:
            fourth = jnp.zeros((3, 3, 3, 3), dtype=dtype)

    mass = jnp.asarray(mass, dtype=dtype)
    dipole = jnp.asarray(dipole, dtype=dtype)
    second = jnp.asarray(second, dtype=dtype)
    third = jnp.asarray(third, dtype=dtype)
    fourth = jnp.asarray(fourth, dtype=dtype)

    return _translate_multipole_to_local_impl(
        packed,
        displacement,
        order=order_int,
        mass=mass,
        dipole=dipole,
        second=second,
        third=third,
        fourth=fourth,
    )


@jaxtyped(typechecker=beartype)
def initialize_local_expansions(
    tree: Tree,
    centers: Array,
    *,
    max_order: int,
) -> LocalExpansionData:
    """Allocate zeroed local expansion buffers for every tree node.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being swept.
    centers : Array
        Expansion centre per node, shape ``(num_nodes, 3)``.
    max_order : int
        Maximum expansion order ``p`` to allocate for.

    Returns
    -------
    LocalExpansionData
        Zeroed local buffers for every tree node.

    Raises
    ------
    ValueError
        If the order is outside the supported range.
    """

    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be >= 0")
    # NOTE: The cartesian/STF local-expansion implementation is only
    # implemented up to MAX_MULTIPOLE_ORDER, but the spherical-harmonics
    # backend stores locals in a different basis/size and is handled by a
    # separate code path. We keep the allocator permissive so callers can
    # allocate larger buffers when they know what they are doing.

    centers_arr = jnp.asarray(centers)
    num_nodes = int(tree.parent.shape[0])
    if centers_arr.shape != (num_nodes, 3):
        raise ValueError("centers must have shape (num_nodes, 3)")

    coeff_count = total_coefficients(min(order, MAX_MULTIPOLE_ORDER))
    coeffs = jnp.zeros((num_nodes, coeff_count), dtype=centers_arr.dtype)

    return LocalExpansionData(
        order=order,
        centers=centers_arr,
        coefficients=coeffs,
    )


@jaxtyped(typechecker=beartype)
def accumulate_m2l_contributions(
    interactions: NodeInteractionList,
    multipoles: NodeMultipoleData,
    local_data: LocalExpansionData,
    chunk_size: int = DEFAULT_M2L_CHUNK_SIZE,
) -> LocalExpansionData:
    """Accumulate M2L contributions into the provided local expansions.

    The interaction list is ordered by tree level (see
    :class:`~yggdrax.interactions.NodeInteractionList.level_offsets`).
    That metadata lets us process interactions level by level while still
    limiting each JAX `vmap` call to ``chunk_size`` pairs for good fusion and
    peak memory characteristics. Override ``chunk_size`` when benchmarking
    different batching strategies; the value must stay positive.

    Parameters
    ----------
    interactions : NodeInteractionList
        Accepted far-field node pairs to accumulate.
    multipoles : NodeMultipoleData
        Per-node multipole data from the upward sweep.
    local_data : LocalExpansionData
        The local expansions being accumulated into.
    chunk_size : int
        Pairs processed per chunk, bounding peak memory.

    Returns
    -------
    LocalExpansionData
        The local expansions with the sparse M2L contributions added.

    Raises
    ------
    ValueError
        If the interaction list and local buffers disagree.
    """

    if int(multipoles.order) != int(local_data.order):
        raise ValueError("multipole and local orders must match")

    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("chunk_size must be positive")

    order = int(local_data.order)

    return _accumulate_m2l_contributions_impl(
        interactions,
        multipoles,
        local_data,
        chunk_size=chunk,
        order=order,
    )


@partial(jax.jit, static_argnames=("chunk_size", "order"))
def _accumulate_m2l_contributions_impl(
    interactions: NodeInteractionList,
    multipoles: NodeMultipoleData,
    local_data: LocalExpansionData,
    *,
    chunk_size: int,
    order: int,
) -> LocalExpansionData:
    """JIT core for sparse interaction-list M2L accumulation.

    Parameters
    ----------
    interactions : NodeInteractionList
        Accepted far-field node pairs to accumulate.
    multipoles : NodeMultipoleData
        Per-node multipole data from the upward sweep.
    local_data : LocalExpansionData
        The local expansions being accumulated into.
    chunk_size : int
        Pairs processed per chunk, bounding peak memory.
    order : int
        Expansion order ``p``.

    Returns
    -------
    LocalExpansionData
        As :func:`accumulate_m2l_contributions`, inside the jitted core.
    """
    centers_target = jnp.asarray(local_data.centers)
    centers_source = jnp.asarray(multipoles.centers)
    coeffs = jnp.asarray(local_data.coefficients)

    sources = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    pair_count = int(sources.shape[0])
    if pair_count == 0:
        return LocalExpansionData(
            order=order,
            centers=centers_target,
            coefficients=coeffs,
        )

    coeff_dtype = coeffs.dtype

    total_coeff = int(coeffs.shape[1])
    component_matrix = _multipole_component_matrix(
        multipoles,
        coeff_count=total_coeff,
        dtype=coeff_dtype,
    )

    coeffs_updated = _accumulate_level(
        coeffs,
        component_matrix,
        centers_target,
        centers_source,
        sources,
        jnp.asarray(interactions.offsets, dtype=INDEX_DTYPE),
        jnp.asarray(interactions.counts, dtype=INDEX_DTYPE),
        order=order,
        chunk_size=chunk_size,
    )

    return LocalExpansionData(
        order=order,
        centers=centers_target,
        coefficients=coeffs_updated,
    )


@jaxtyped(typechecker=beartype)
def run_downward_sweep(
    tree: Tree,
    multipoles: NodeMultipoleData,
    interactions: Optional[NodeInteractionList] = None,
    *,
    initial_locals: Optional[LocalExpansionData] = None,
    m2l_chunk_size: Optional[int] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
) -> LocalExpansionData:
    """Execute the full downward pass (M2L followed by L2L propagation).

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being swept.
    multipoles : NodeMultipoleData
        Per-node multipole data from the upward sweep.
    interactions : Optional[NodeInteractionList]
        Accepted far-field node pairs to accumulate.
    initial_locals : Optional[LocalExpansionData]
        Preallocated local buffers, or None to allocate them here.
    m2l_chunk_size : Optional[int]
        M2L chunk size, or None for the module default.
    dense_buffers : Optional[DenseInteractionBuffers]
        Dense level-major interaction buffers.

    Returns
    -------
    LocalExpansionData
        The local expansions after M2L accumulation and the L2L cascade.

    Raises
    ------
    ValueError
        If the inputs are mutually inconsistent.
    """

    order = int(multipoles.order)

    if dense_buffers is not None and interactions is None:
        interactions = dense_buffers.sparse_interactions

    if interactions is None:
        raise ValueError(
            "interactions must be provided when dense_buffers is None",
        )

    if initial_locals is None:
        locals_init = initialize_local_expansions(
            tree,
            multipoles.centers,
            max_order=order,
        )
    else:
        locals_init = initial_locals
        if int(locals_init.order) != order:
            raise ValueError(
                "initial_locals order must equal multipoles order",
            )
        if locals_init.centers.shape != multipoles.centers.shape:
            raise ValueError(
                "initial_locals centers must match multipole centers",
            )
        if locals_init.coefficients.shape != multipoles.packed.shape:
            raise ValueError(
                "initial_locals coefficients must match multipole coefficients"
            )

    if dense_buffers is not None:
        accumulated = accumulate_dense_m2l_contributions(
            dense_buffers,
            multipoles,
            locals_init,
        )
    else:
        chunk = (
            DEFAULT_M2L_CHUNK_SIZE if m2l_chunk_size is None else int(m2l_chunk_size)
        )
        if chunk <= 0:
            raise ValueError("m2l_chunk_size must be positive")

        accumulated = accumulate_m2l_contributions(
            interactions,
            multipoles,
            locals_init,
            chunk_size=chunk,
        )
    return propagate_local_expansions(tree, accumulated)


@jaxtyped(typechecker=beartype)
def prepare_downward_sweep(
    tree: Tree,
    upward: TreeUpwardData,
    *,
    theta: float = 0.5,
    mac_type: MACType = "bh",
    initial_locals: Optional[LocalExpansionData] = None,
    interactions: Optional[NodeInteractionList] = None,
    m2l_chunk_size: Optional[int] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
    retry_logger: Optional[Callable[[DualTreeRetryEvent], None]] = None,
    traversal_config: Optional[DualTreeTraversalConfig] = None,
    max_interactions_per_node: Optional[int] = None,
    max_pair_queue: Optional[int] = None,
    process_block: Optional[int] = None,
) -> TreeDownwardData:
    """Construct interactions and locals for the downward pass.

    The standalone downward driver: runs the dual-tree traversal to get the
    well-separated interaction list (unless one is supplied), then accumulates M2L
    into the local expansions. The runtime's own prepare path does not go through
    here -- it uses ``runtime/kernels`` directly -- so this is the reference entry
    point, used by tests and by callers driving the sweep themselves.

    THE FOUR CAPACITY ARGUMENTS ARE A PADDING CONTRACT, not tuning. Interaction
    counts are data-dependent and JAX needs static shapes, so the traversal writes
    into fixed-capacity buffers and *fails* when one overflows; ``retry_logger``
    observes those failures so a caller can grow the capacity and rebuild. Passing
    a full ``traversal_config`` replaces all four at once; passing the individual
    ``max_*`` / ``process_block`` arguments overrides them one at a time.

    Parameters
    ----------
    tree : Tree
        Radix tree supplying the topology.
    upward : TreeUpwardData
        Result of the upward sweep: the per-node multipoles and geometry this
        pass translates.
    theta : float
        Opening angle for the acceptance criterion; smaller is more accurate and
        slower. Default ``0.5``.
    mac_type : MACType
        Which multipole acceptance criterion to apply. Default ``"bh"``.
    initial_locals : Optional[LocalExpansionData]
        Local expansions to accumulate into; a fresh zero buffer when ``None``.
    interactions : Optional[NodeInteractionList]
        A precomputed interaction list. When ``None`` the traversal is run here.
    m2l_chunk_size : Optional[int]
        Pairs per accumulation chunk; see :func:`_accumulate_level`. Batching
        only, not an accuracy knob.
    dense_buffers : Optional[DenseInteractionBuffers]
        Fixed-capacity dense interaction buffers, selecting
        :func:`_accumulate_dense_m2l_impl` over the CSR path.
    retry_logger : Optional[Callable[[DualTreeRetryEvent], None]]
        Called with each traversal capacity-overflow event, so a caller can
        resize and retry rather than guess.
    traversal_config : Optional[DualTreeTraversalConfig]
        Replaces **all four** traversal capacities at once.
    max_interactions_per_node : Optional[int]
        Per-node far-interaction capacity, overriding the config's value.
    max_pair_queue : Optional[int]
        Node-pair queue capacity for the traversal.
    process_block : Optional[int]
        Node pairs processed per traversal step.

    Returns
    -------
    TreeDownwardData
        The interaction list actually used and the accumulated local expansions.
    """

    buffers = dense_buffers
    if buffers is not None and interactions is None:
        interactions = buffers.sparse_interactions

    if interactions is None:
        interactions = build_well_separated_interactions(
            tree,
            upward.geometry,
            theta=theta,
            mac_type=mac_type,
            max_interactions_per_node=max_interactions_per_node,
            max_pair_queue=max_pair_queue,
            process_block=process_block,
            traversal_config=traversal_config,
            retry_logger=retry_logger,
        )
    locals_data = run_downward_sweep(
        tree,
        upward.multipoles,
        interactions,
        initial_locals=initial_locals,
        m2l_chunk_size=m2l_chunk_size,
        dense_buffers=buffers,
    )
    return TreeDownwardData(interactions=interactions, locals=locals_data)


@partial(jax.jit, static_argnames=("order", "num_internal"))
def _propagate_local_expansions_impl(
    coeffs: Array,
    centers: Array,
    left_child: Array,
    right_child: Array,
    *,
    order: int,
    num_internal: int,
) -> Array:
    """JIT L2L propagation kernel over internal tree nodes.

    Parameters
    ----------
    coeffs : Array
        Packed local coefficients per node.
    centers : Array
        Expansion centre per node, shape ``(num_nodes, 3)``.
    left_child : Array
        Left child index per internal node.
    right_child : Array
        Right child index per internal node.
    order : int
        Expansion order ``p``.
    num_internal : int
        Number of internal (non-leaf) nodes.

    Returns
    -------
    Array
        Local coefficients after the parent-to-child cascade.
    """
    if num_internal == 0:
        return coeffs

    def add_child(
        state_coeffs: Array,
        parent_coeff: Array,
        node_idx: Array,
        child_idx: Array,
    ) -> Array:
        def true_branch(idx: Array) -> Array:
            delta = centers[idx] - centers[node_idx]
            translated = translate_local_expansion(
                parent_coeff,
                delta,
                order=order,
            )
            return state_coeffs.at[idx].add(translated)

        return lax.cond(
            child_idx >= 0,
            true_branch,
            lambda _: state_coeffs,
            child_idx,
        )

    def body(node_idx: Array, state_coeffs: Array) -> Array:
        parent_coeff = state_coeffs[node_idx]
        child_left = left_child[node_idx]
        child_right = right_child[node_idx]
        state_coeffs = add_child(
            state_coeffs,
            parent_coeff,
            node_idx,
            child_left,
        )
        state_coeffs = add_child(
            state_coeffs,
            parent_coeff,
            node_idx,
            child_right,
        )
        return state_coeffs

    return lax.fori_loop(0, num_internal, body, coeffs)


@jaxtyped(typechecker=beartype)
def propagate_local_expansions(
    tree: Tree,
    local_data: LocalExpansionData,
) -> LocalExpansionData:
    """Perform an L2L sweep to accumulate parent locals into children.

    Parameters
    ----------
    tree : Tree
        The tree whose nodes are being swept.
    local_data : LocalExpansionData
        The local expansions being accumulated into.

    Returns
    -------
    LocalExpansionData
        The local expansions with each parent's contribution added to its children.

    Raises
    ------
    NotImplementedError
        If the tree shape is not one the L2L cascade supports.
    """

    order = int(local_data.order)
    if order > MAX_MULTIPOLE_ORDER:
        raise NotImplementedError("orders above 4 are not supported")

    centers = jnp.asarray(local_data.centers)
    coeffs = jnp.asarray(local_data.coefficients)

    # Use the static array shape (not int(tree.num_internal_nodes)) so this is
    # safe to trace inside a transform such as jax.shard_map, where the tree's
    # scalar count fields are traced leaves.
    left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
    right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)
    num_internal = int(left_child.shape[0])

    if num_internal == 0:
        return LocalExpansionData(
            order=order,
            centers=centers,
            coefficients=coeffs,
        )

    updated = _propagate_local_expansions_impl(
        coeffs,
        centers,
        left_child,
        right_child,
        order=order,
        num_internal=num_internal,
    )

    return LocalExpansionData(
        order=order,
        centers=centers,
        coefficients=updated,
    )


__all__ = [
    "LocalExpansionData",
    "accumulate_dense_m2l_contributions",
    "accumulate_m2l_contributions",
    "initialize_local_expansions",
    "prepare_downward_sweep",
    "run_downward_sweep",
    "TreeDownwardData",
    "propagate_local_expansions",
    "translate_multipole_to_local",
    "translate_local_expansion",
]

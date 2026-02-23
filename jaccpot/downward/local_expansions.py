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
from yggdrax.multipole_utils import (
    MAX_MULTIPOLE_ORDER,
    level_offset,
    multi_index_factorial,
    multi_index_tuples,
    multi_power,
    total_coefficients,
)
from yggdrax.tree import RadixTree
from jaccpot.upward.tree_expansions import NodeMultipoleData, TreeUpwardData
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    NodeInteractionList,
    build_well_separated_interactions,
)
from yggdrax.tree_moments import (
    _hexadecapole_from_fourth,
    _octupole_from_third,
    _quadrupole_from_second,
    multipole_from_packed,
    tree_moments_from_raw,
)


class LocalExpansionData(NamedTuple):
    """Local expansion coefficients and metadata."""

    order: int
    centers: Array
    coefficients: Array


class TreeDownwardData(NamedTuple):
    """Bundle for interaction lists and resulting local expansions."""

    interactions: NodeInteractionList
    locals: LocalExpansionData


_LEVEL_COMBOS: Dict[int, Tuple[Tuple[int, int, int], ...]] = {
    level: multi_index_tuples(level) for level in range(MAX_MULTIPOLE_ORDER + 1)
}

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
    level: int
    terms: Tuple[Tuple[Tuple[int, int, int], int], ...]


def _scale_poly(
    poly: Dict[Tuple[int, int, int], int],
    scale: int,
) -> Dict[Tuple[int, int, int], int]:
    if scale == 0 or not poly:
        return {}
    return {exp: coeff * scale for exp, coeff in poly.items()}


def _add_poly(
    left: Dict[Tuple[int, int, int], int],
    right: Dict[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int], int]:
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
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        power = exp[axis]
        if power == 0:
            continue
        new_exp = list(exp)
        new_exp[axis] -= 1
        key = tuple(new_exp)
        result[key] = result.get(key, 0) + coeff * power
    return result


def _mul_by_axis(
    poly: Dict[Tuple[int, int, int], int],
    axis: int,
) -> Dict[Tuple[int, int, int], int]:
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        new_exp = list(exp)
        new_exp[axis] += 1
        key = tuple(new_exp)
        result[key] = result.get(key, 0) + coeff
    return result


def _mul_by_r2(
    poly: Dict[Tuple[int, int, int], int],
) -> Dict[Tuple[int, int, int], int]:
    if not poly:
        return {}
    result: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        for axis in range(3):
            new_exp = list(exp)
            new_exp[axis] += 2
            key = tuple(new_exp)
            result[key] = result.get(key, 0) + coeff
    return result


def _term_tuple(
    poly: Dict[Tuple[int, int, int], int],
) -> Tuple[Tuple[Tuple[int, int, int], int], ...]:
    return tuple(sorted(poly.items()))


def _generate_derivative_info() -> Tuple[
    Dict[Tuple[int, int, int], _DerivativeInfo],
    int,
]:
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


def _build_derivative_tables():
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
    gamma_indices: Array
    scales: Array
    component_sizes: Tuple[int, ...]


def _build_m2l_stencils() -> Tuple[_M2LStencil, ...]:
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

    r2 = jnp.dot(displacement, displacement)
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
    return jnp.einsum(
        "nqc,nc->nq",
        weighted,
        component_chunk,
        precision=lax.Precision.HIGHEST,
    )


@partial(jax.jit, static_argnames=("order", "chunk_size"))
def _accumulate_level(
    coeffs: Array,
    component_matrix: Array,
    centers_target: Array,
    centers_source: Array,
    sources: Array,
    targets: Array,
    *,
    order: int,
    chunk_size: int,
) -> Array:
    pair_count = sources.shape[0]
    if pair_count == 0:
        return coeffs

    chunk = int(max(chunk_size, 1))
    steps = (pair_count + chunk - 1) // chunk
    padded_len = steps * chunk
    pad_amount = padded_len - pair_count
    sources_padded = jnp.pad(
        sources,
        (0, pad_amount),
        mode="constant",
        constant_values=0,
    )
    targets_padded = jnp.pad(
        targets,
        (0, pad_amount),
        mode="constant",
        constant_values=0,
    )

    def body(idx: Array, carry: Array) -> Array:
        coeff_state = carry
        start = idx * chunk
        remaining = pair_count - start
        chunk_len = jnp.minimum(chunk, jnp.maximum(remaining, 0))
        src_chunk = lax.dynamic_slice_in_dim(
            sources_padded,
            start,
            chunk,
            axis=0,
        )
        tgt_chunk = lax.dynamic_slice_in_dim(
            targets_padded,
            start,
            chunk,
            axis=0,
        )
        valid = jnp.arange(chunk, dtype=INDEX_DTYPE) < chunk_len
        safe_src = jnp.where(valid, src_chunk, as_index(0))
        safe_tgt = jnp.where(valid, tgt_chunk, as_index(0))
        delta_slice = centers_target[safe_tgt] - centers_source[safe_src]
        component_slice = component_matrix[safe_src]
        contrib = _translate_components_batch(
            component_slice,
            delta_slice,
            order=order,
        )
        contrib = jnp.where(valid[:, None], contrib, 0.0)
        coeff_state = coeff_state.at[safe_tgt].add(contrib)
        return coeff_state

    return lax.fori_loop(0, steps, body, coeffs)


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
    """Accumulate M2L contributions using dense level-major buffers."""

    if multipoles.order != local_data.order:
        raise ValueError("multipole and local orders must match")

    centers = jnp.asarray(local_data.centers)
    coeffs = jnp.asarray(local_data.coefficients)
    order = int(local_data.order)

    if centers.shape != multipoles.centers.shape:
        raise ValueError("local centers must align with multipole centers")

    coeff_count = int(coeffs.shape[1])
    component_matrix = jnp.asarray(
        multipoles.component_matrix[:, :coeff_count],
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
    dtype = multipole.dtype
    component_vec = _build_component_vector(
        mass,
        dipole,
        second,
        third,
        fourth,
        order=order,
    ).astype(dtype)
    return _translate_components_to_local(
        component_vec,
        delta,
        order=order,
    )


def _pack_symmetric_tensor(tensor: Array, level: int) -> Array:
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
    """Shift a local expansion by ``delta`` using explicit binomial sums."""

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
        mass = jnp.asarray(raw_mass, dtype=dtype)
        dipole = (
            jnp.asarray(raw_dipole, dtype=dtype)
            if order_int >= 1
            else jnp.zeros((3,), dtype=dtype)
        )
        if order_int >= 2:
            second = jnp.asarray(
                _quadrupole_from_second(raw_second),
                dtype=dtype,
            )
        else:
            second = jnp.zeros((3, 3), dtype=dtype)
        if order_int >= 3:
            third = jnp.asarray(
                _octupole_from_third(raw_third),
                dtype=dtype,
            )
        else:
            third = jnp.zeros((3, 3, 3), dtype=dtype)
        if order_int >= 4:
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
    tree: RadixTree,
    centers: Array,
    *,
    max_order: int,
) -> LocalExpansionData:
    """Allocate zeroed local expansion buffers for every tree node."""

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
    centers_target = jnp.asarray(local_data.centers)
    centers_source = jnp.asarray(multipoles.centers)
    coeffs = jnp.asarray(local_data.coefficients)

    sources = jnp.asarray(interactions.sources, dtype=INDEX_DTYPE)
    targets = jnp.asarray(interactions.targets, dtype=INDEX_DTYPE)
    pair_count = int(sources.shape[0])
    if pair_count == 0:
        return LocalExpansionData(
            order=order,
            centers=centers_target,
            coefficients=coeffs,
        )

    coeff_dtype = coeffs.dtype

    total_coeff = int(coeffs.shape[1])
    component_matrix = jnp.asarray(
        multipoles.packed[:, :total_coeff],
        dtype=coeff_dtype,
    )

    coeffs_updated = _accumulate_level(
        coeffs,
        component_matrix,
        centers_target,
        centers_source,
        sources,
        targets,
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
    tree: RadixTree,
    multipoles: NodeMultipoleData,
    interactions: Optional[NodeInteractionList] = None,
    *,
    initial_locals: Optional[LocalExpansionData] = None,
    m2l_chunk_size: Optional[int] = None,
    dense_buffers: Optional[DenseInteractionBuffers] = None,
) -> LocalExpansionData:
    """Execute the full downward pass (M2L followed by L2L propagation)."""

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
    tree: RadixTree,
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
    """Construct interactions and locals for the downward pass."""

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
    tree: RadixTree,
    local_data: LocalExpansionData,
) -> LocalExpansionData:
    """Perform an L2L sweep to accumulate parent locals into children."""

    order = int(local_data.order)
    if order > MAX_MULTIPOLE_ORDER:
        raise NotImplementedError("orders above 4 are not supported")

    centers = jnp.asarray(local_data.centers)
    coeffs = jnp.asarray(local_data.coefficients)

    num_internal = int(tree.num_internal_nodes)
    left_child = jnp.asarray(tree.left_child, dtype=INDEX_DTYPE)
    right_child = jnp.asarray(tree.right_child, dtype=INDEX_DTYPE)

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

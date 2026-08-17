"""Reference helpers used by jaccpot runtime internals."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jax import lax
from jaxtyping import Array, jaxtyped


class MultipoleExpansion(NamedTuple):
    """Multipole coefficients around a shared expansion center.

    Cartesian *reduced* (trace-free) moments, not solid harmonics: the
    quadrupole is ``3 * sum(m r_i r_j) - delta_ij * sum(m r^2)`` and the higher
    orders carry the matching delta subtractions. This is the reference
    implementation, independent of the harmonic operators in
    :mod:`jaccpot.operators`.

    Attributes
    ----------
    monopole : jnp.ndarray
        Total mass, a 0-d array.
    dipole : jnp.ndarray
        First-order vector moment ``[3]``. When built by
        :func:`compute_expansion` the moments are taken about the centre of
        mass, so this is zero up to round-off rather than meaningfully nonzero.
    center : jnp.ndarray
        Expansion centre ``[3]`` (the centre of mass in
        :func:`compute_expansion`; zero when the total mass is zero).
    quadrupole : jnp.ndarray
        Second-order symmetric trace-free tensor ``[3, 3]``.
    octupole : jnp.ndarray
        Third-order symmetric trace-free tensor ``[3, 3, 3]``.
    hexadecapole : jnp.ndarray
        Fourth-order symmetric trace-free tensor ``[3, 3, 3, 3]``.

    Notes
    -----
    Moments above the requested ``order`` are returned as exact zeros of the
    right shape, so the tuple always has all six fields regardless of ``order``.
    """

    monopole: jnp.ndarray
    dipole: jnp.ndarray
    center: jnp.ndarray
    quadrupole: jnp.ndarray
    octupole: jnp.ndarray
    hexadecapole: jnp.ndarray


@partial(jax.jit, static_argnums=(2,))
@jaxtyped(typechecker=beartype)
def compute_expansion(
    positions: Array,
    masses: Array,
    order: int = 1,
) -> MultipoleExpansion:
    """Compute multipole expansion up to ``order`` around the center of mass.
    Parameters
    ----------
    positions : Array
        ``(n, 3)`` particle positions.
    masses : Array
        ``(n,)`` particle masses.
    order : int
        Highest moment to compute. Moments above this are left zero rather than
        omitted, so the returned container has a fixed layout.

    Returns
    -------
    MultipoleExpansion
        Moments about the centre of mass of the given particles.
    """

    total_mass = jnp.sum(masses)
    center_num = jnp.sum(positions * masses[:, None], axis=0)
    center = jax.lax.cond(
        jnp.abs(total_mass) > 0,
        lambda _: center_num / total_mass,
        lambda _: jnp.zeros((3,), dtype=positions.dtype),
        operand=None,
    )

    rel = positions - center
    eye3 = jnp.eye(3, dtype=positions.dtype)
    r2 = jnp.sum(rel * rel, axis=1)

    monopole = total_mass
    dipole = jnp.where(
        order >= 1,
        jnp.sum(masses[:, None] * rel, axis=0),
        jnp.zeros((3,), dtype=positions.dtype),
    )

    def quad_compute() -> Array:
        rr = jnp.einsum(
            "ni,nj,n->ij", rel, rel, masses, precision=lax.Precision.HIGHEST
        )
        r2_sum = jnp.einsum("n->", masses * r2, precision=lax.Precision.HIGHEST)
        return 3.0 * rr - eye3 * r2_sum

    quadrupole = jax.lax.cond(
        order >= 2,
        lambda _: quad_compute(),
        lambda _: jnp.zeros((3, 3), dtype=positions.dtype),
        operand=None,
    )

    def oct_compute() -> Array:
        t3 = jnp.einsum(
            "ni,nj,nk,n->ijk", rel, rel, rel, masses, precision=lax.Precision.HIGHEST
        )
        mr2 = masses * r2
        term_a = jnp.einsum(
            "ij,nk,n->ijk", eye3, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_b = jnp.einsum(
            "ik,nj,n->ijk", eye3, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_c = jnp.einsum(
            "jk,ni,n->ijk", eye3, rel, mr2, precision=lax.Precision.HIGHEST
        )
        return 5.0 * t3 - (term_a + term_b + term_c)

    octupole = jax.lax.cond(
        order >= 3,
        lambda _: oct_compute(),
        lambda _: jnp.zeros((3, 3, 3), dtype=positions.dtype),
        operand=None,
    )

    def hexa_compute() -> Array:
        t4 = jnp.einsum(
            "ni,nj,nk,nl,n->ijkl",
            rel,
            rel,
            rel,
            rel,
            masses,
            precision=lax.Precision.HIGHEST,
        )
        mr2 = masses * r2
        term_ij = jnp.einsum(
            "ij,nk,nl,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_ik = jnp.einsum(
            "ik,nj,nl,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_il = jnp.einsum(
            "il,nj,nk,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_jk = jnp.einsum(
            "jk,ni,nl,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_jl = jnp.einsum(
            "jl,ni,nk,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        term_kl = jnp.einsum(
            "kl,ni,nj,n->ijkl", eye3, rel, rel, mr2, precision=lax.Precision.HIGHEST
        )
        s_r4 = jnp.einsum("n->", masses * (r2 * r2), precision=lax.Precision.HIGHEST)
        delta_delta = (
            jnp.einsum("ij,kl->ijkl", eye3, eye3, precision=lax.Precision.HIGHEST)
            + jnp.einsum("ik,jl->ijkl", eye3, eye3, precision=lax.Precision.HIGHEST)
            + jnp.einsum("il,jk->ijkl", eye3, eye3, precision=lax.Precision.HIGHEST)
        )
        combined_terms = term_ij + term_ik + term_il + term_jk + term_jl + term_kl
        return 35.0 * t4 - 5.0 * combined_terms + s_r4 * delta_delta

    hexadecapole = jax.lax.cond(
        order >= 4,
        lambda _: hexa_compute(),
        lambda _: jnp.zeros((3, 3, 3, 3), dtype=positions.dtype),
        operand=None,
    )

    return MultipoleExpansion(
        monopole=monopole,
        dipole=dipole,
        center=center,
        quadrupole=quadrupole,
        octupole=octupole,
        hexadecapole=hexadecapole,
    )


@partial(jax.jit, static_argnums=(1,))
@jaxtyped(typechecker=beartype)
def evaluate_expansion(
    expansion: MultipoleExpansion,
    order: int = 1,
    eval_point: Optional[Array] = None,
    *,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
) -> Array:
    """Evaluate acceleration from ``expansion`` at one point.

    The potential is assembled from moments up to ``order`` and then
    differentiated with ``jax.grad`` to produce acceleration.

    Taking the gradient of the assembled potential, rather than coding an
    acceleration series, is what makes this a usable oracle: there is no second
    expression that could drift from the first.

    Parameters
    ----------
    expansion : MultipoleExpansion
        Moments to evaluate.
    order : int
        Highest moment to include. May be below the expansion's own order.
    eval_point : Optional[Array]
        Point of evaluation. ``None`` is rejected -- see ``Raises``.
    G : Union[float, Array]
        Gravitational constant.
    softening : Union[float, Array]
        Plummer softening length.

    Returns
    -------
    Array
        Acceleration at ``eval_point``.

    Raises
    ------
    ValueError
        If ``eval_point`` is ``None``, or ``order`` exceeds what the expansion
        carries.
    """

    if eval_point is None:
        raise ValueError("eval_point must be provided")

    def phi_at(x: Array) -> Array:
        r_vec = x - expansion.center
        r2 = jnp.dot(r_vec, r_vec, precision=lax.Precision.HIGHEST)
        r_soft = jnp.sqrt(r2 + softening * softening)

        phi = -G * (expansion.monopole / r_soft)

        if order >= 1:
            d_dot_r = jnp.dot(expansion.dipole, r_vec, precision=lax.Precision.HIGHEST)
            phi = phi + (-G) * d_dot_r / (r_soft**3)

        if order >= 2:
            q_rr = jnp.einsum(
                "ij,i,j->",
                expansion.quadrupole,
                r_vec,
                r_vec,
                precision=lax.Precision.HIGHEST,
            )
            phi = phi + (-G) * 0.5 * q_rr / (r_soft**5)

        if order >= 3:
            o_rrr = jnp.einsum(
                "ijk,i,j,k->",
                expansion.octupole,
                r_vec,
                r_vec,
                r_vec,
                precision=lax.Precision.HIGHEST,
            )
            phi = phi + (-G) * ((1.0 / 6.0) * o_rrr) / (r_soft**7)

        if order >= 4:
            h_rrrr = jnp.einsum(
                "ijkl,i,j,k,l->",
                expansion.hexadecapole,
                r_vec,
                r_vec,
                r_vec,
                r_vec,
                precision=lax.Precision.HIGHEST,
            )
            phi = phi + (-G) * ((1.0 / 24.0) * h_rrrr) / (r_soft**9)

        return phi

    grad_phi = jax.grad(phi_at)(eval_point)
    return -grad_phi


@partial(jax.jit, static_argnums=())
@jaxtyped(typechecker=beartype)
def direct_sum(
    positions: Array,
    masses: Array,
    eval_point: Array,
    *,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
) -> Array:
    """Compute gravitational acceleration via O(N) direct summation.
    O(N) for one evaluation point, so O(N^2) to evaluate at every particle. Exact
    -- this is the oracle the FMM paths are checked against.

    Parameters
    ----------
    positions : Array
        ``(n, 3)`` source positions.
    masses : Array
        ``(n,)`` source masses.
    eval_point : Array
        Single point at which to evaluate.
    G : Union[float, Array]
        Gravitational constant.
    softening : Union[float, Array]
        Plummer softening length.

    Returns
    -------
    Array
        Acceleration at ``eval_point``.
    """

    r_vec = eval_point - positions
    dist_sq = jnp.sum(r_vec**2, axis=1, keepdims=True)
    r = jnp.sqrt(dist_sq + softening**2)
    return -G * jnp.sum(masses[:, None] * r_vec / (r**3), axis=0)


@jax.jit
@jaxtyped(typechecker=beartype)
def compute_gravitational_potential(
    positions: Array,
    masses: Array,
    eval_points: Array,
    G: Union[float, Array] = 1.0,
    softening: Union[float, Array] = 0.0,
) -> Array:
    """Compute gravitational potential at ``eval_points`` with direct sums.
    Vectorised over ``eval_points``, unlike :func:`direct_sum`, which takes one
    point.

    Parameters
    ----------
    positions : Array
        ``(n, 3)`` source positions.
    masses : Array
        ``(n,)`` source masses.
    eval_points : Array
        ``(m, 3)`` points at which to evaluate.
    G : Union[float, Array]
        Gravitational constant.
    softening : Union[float, Array]
        Plummer softening length.

    Returns
    -------
    Array
        ``(m,)`` potentials.
    """

    def compute_potential(eval_point: Array) -> Array:
        r_vec = eval_point - positions
        r = jnp.sqrt(jnp.sum(r_vec**2, axis=1) + softening**2)
        return -G * jnp.sum(masses / (r + 1e-10))

    return jax.vmap(compute_potential)(eval_points)


__all__ = [
    "MultipoleExpansion",
    "compute_expansion",
    "evaluate_expansion",
    "direct_sum",
    "compute_gravitational_potential",
]

"""Reference helpers used by jaccpot runtime internals."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, jaxtyped


class MultipoleExpansion(NamedTuple):
    """Multipole coefficients around a shared expansion center.

    Attributes
    ----------
    monopole:
        Zeroth-order mass moment.
    dipole:
        First-order vector moment.
    center:
        Expansion center (center of mass in ``compute_expansion``).
    quadrupole:
        Second-order symmetric trace-free tensor.
    octupole:
        Third-order symmetric trace-free tensor.
    hexadecapole:
        Fourth-order symmetric trace-free tensor.
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
    """Compute multipole expansion up to ``order`` around the center of mass."""

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
        rr = jnp.einsum("ni,nj,n->ij", rel, rel, masses)
        r2_sum = jnp.einsum("n->", masses * r2)
        return 3.0 * rr - eye3 * r2_sum

    quadrupole = jax.lax.cond(
        order >= 2,
        lambda _: quad_compute(),
        lambda _: jnp.zeros((3, 3), dtype=positions.dtype),
        operand=None,
    )

    def oct_compute() -> Array:
        t3 = jnp.einsum("ni,nj,nk,n->ijk", rel, rel, rel, masses)
        mr2 = masses * r2
        term_a = jnp.einsum("ij,nk,n->ijk", eye3, rel, mr2)
        term_b = jnp.einsum("ik,nj,n->ijk", eye3, rel, mr2)
        term_c = jnp.einsum("jk,ni,n->ijk", eye3, rel, mr2)
        return 5.0 * t3 - (term_a + term_b + term_c)

    octupole = jax.lax.cond(
        order >= 3,
        lambda _: oct_compute(),
        lambda _: jnp.zeros((3, 3, 3), dtype=positions.dtype),
        operand=None,
    )

    def hexa_compute() -> Array:
        t4 = jnp.einsum("ni,nj,nk,nl,n->ijkl", rel, rel, rel, rel, masses)
        mr2 = masses * r2
        term_ij = jnp.einsum("ij,nk,nl,n->ijkl", eye3, rel, rel, mr2)
        term_ik = jnp.einsum("ik,nj,nl,n->ijkl", eye3, rel, rel, mr2)
        term_il = jnp.einsum("il,nj,nk,n->ijkl", eye3, rel, rel, mr2)
        term_jk = jnp.einsum("jk,ni,nl,n->ijkl", eye3, rel, rel, mr2)
        term_jl = jnp.einsum("jl,ni,nk,n->ijkl", eye3, rel, rel, mr2)
        term_kl = jnp.einsum("kl,ni,nj,n->ijkl", eye3, rel, rel, mr2)
        s_r4 = jnp.einsum("n->", masses * (r2 * r2))
        delta_delta = (
            jnp.einsum("ij,kl->ijkl", eye3, eye3)
            + jnp.einsum("ik,jl->ijkl", eye3, eye3)
            + jnp.einsum("il,jk->ijkl", eye3, eye3)
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
    """

    if eval_point is None:
        raise ValueError("eval_point must be provided")

    def phi_at(x: Array) -> Array:
        r_vec = x - expansion.center
        r2 = jnp.dot(r_vec, r_vec)
        r_soft = jnp.sqrt(r2 + softening * softening)

        phi = -G * (expansion.monopole / r_soft)

        if order >= 1:
            d_dot_r = jnp.dot(expansion.dipole, r_vec)
            phi = phi + (-G) * d_dot_r / (r_soft**3)

        if order >= 2:
            q_rr = jnp.einsum("ij,i,j->", expansion.quadrupole, r_vec, r_vec)
            phi = phi + (-G) * 0.5 * q_rr / (r_soft**5)

        if order >= 3:
            o_rrr = jnp.einsum("ijk,i,j,k->", expansion.octupole, r_vec, r_vec, r_vec)
            phi = phi + (-G) * ((1.0 / 6.0) * o_rrr) / (r_soft**7)

        if order >= 4:
            h_rrrr = jnp.einsum(
                "ijkl,i,j,k,l->",
                expansion.hexadecapole,
                r_vec,
                r_vec,
                r_vec,
                r_vec,
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
    """Compute gravitational acceleration via O(N) direct summation."""

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
    """Compute gravitational potential at ``eval_points`` with direct sums."""

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

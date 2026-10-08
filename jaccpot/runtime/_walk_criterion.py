"""Dehnen (2014) eq (16a) as a per-pair test inside the flat walks.

The general path evaluates the criterion through ``adaptive_pair_policy`` on the
yggdrax dual walk. The fused strict lane runs a FLAT walk -- the Pallas
``mutual_walk_pallas`` on the GPU, ``yggdrax.dual_tree_walk_mutual`` elsewhere --
which has no policy seam and must stay cheap per pair. So the criterion is split
in two:

* :func:`dehnen_walk_table` builds, once per walk, a ``[nodes, W]`` table holding
  everything eq (16a) reads about a node: ``G M``, the node's acceptance threshold
  ``eps * min_b f_b`` and the normalised multipole powers
  ``s_n = P_n / (M rho^n)``, ``n = 1..p`` (eq 12, with ``rho`` the walk's own
  exact centre-of-mass radius). ``W`` is a power of two (Triton's array shapes).
* :func:`dehnen_pair_accept` evaluates eq (15)/(16a) for one pair, symmetrised,
  from two gathered rows. It is elementwise only -- no gathers, no reductions --
  so the same function runs inside the Pallas kernel on a block of lanes and in
  the XLA walk on a whole round.

Eq (15) in the normalised form: with ``a = rho_A / r`` and ``b = rho_B / r``,

    E_{A->B} = 8 max(rho_A, rho_B) / (rho_A + rho_B)
               * sum_{n=0}^{p} C(p, n) s_n(A) a^n b^(p-n),        s_0 = 1,

which equals the paper's ``sum C(p,n) P_n rho_B^(p-n) / (M_A r^p)`` term by term.
``s_n`` lies in ``[0, 1]`` whenever ``rho`` bounds the node's particles about its
expansion centre (``P_n <= M rho^n`` by the triangle inequality), so the sum is
well scaled in float32 at any node size; the paper's form needs ``r^p`` and
``rho^n`` separately. eq (16a) then accepts when, in BOTH directions,

    G M_A E_{A->B} / r^2 < thr_B   and   rho_A + rho_B < theta_max r.

Symmetrised because the flat walk visits each unordered pair once and uses one
decision for both directions -- exactly what ``adaptive_pair_policy`` does on the
self walk. Like that policy (and unlike the geometric test) there is no ``d > 0``
guard: at ``r = 0`` the convergence clause already fails.

The radii are the walk's exact COM radii, not the general path's ``"com"`` policy
bound (the minimum of a centre-referenced bound and the box reach): both bound
every particle, the walk's is the tighter, so this criterion opens slightly fewer
pairs than the general path at the same ``eps``. The parity tests feed the policy
the walk's radii to compare the two as sets.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, NamedTuple, Sequence

import jax.numpy as jnp
from jaxtyping import Array

from jaccpot.runtime._adaptive_policy import dehnen_multipole_power_by_degree

__all__ = [
    "WALK_TABLE_GM",
    "WALK_TABLE_THRESHOLD",
    "DehnenWalkAccept",
    "dehnen_pair_accept",
    "dehnen_walk_table",
    "walk_table_width",
]

#: column of ``G M`` in the walk table
WALK_TABLE_GM = 0
#: column of the node's acceptance threshold ``eps * min_b f_b`` (a force)
WALK_TABLE_THRESHOLD = 1
#: column of ``s_1``; ``s_n`` sits at ``WALK_TABLE_POWER0 + n - 1``
WALK_TABLE_POWER0 = 2

#: the improvement factor of eq (15): ``8 max(rho) / (rho_A + rho_B)``
_IMPROVEMENT = 8.0
#: floor of every division, as in ``adaptive_pair_policy``
_TINY = 1e-24


def walk_table_width(order: int) -> int:
    """Columns of the walk table for expansion order ``order``.

    ``G M``, the threshold and ``s_1..s_p``, padded to a power of two.

    Parameters
    ----------
    order : int
        Expansion order ``p``.

    Returns
    -------
    int
        The table width, a power of two ``>= order + 2``.
    """
    need = int(order) + WALK_TABLE_POWER0
    return 1 << max(0, (need - 1).bit_length())


def dehnen_walk_table(
    *,
    multipole_packed: Array,
    mass: Array,
    radius: Array,
    threshold: Array,
    gravitational_constant: float,
    order: int,
) -> Array:
    """Per-node inputs of eq (16a) for the flat walk, ``[nodes, W]`` float.

    Parameters
    ----------
    multipole_packed : Array
        ``[nodes, (p+1)^2]`` packed multipoles about the walk centres (the COM).
    mass : Array
        ``[nodes]`` node masses.
    radius : Array
        ``[nodes]`` the walk's MAC radii about the same centres.
    threshold : Array
        ``[nodes]`` acceptance threshold ``eps * min_b f_b``, a force.
    gravitational_constant : float
        ``G``.
    order : int
        Expansion order ``p``; the multipoles must carry at least degree ``p``.

    Returns
    -------
    Array
        ``(G M, threshold, s_1, ..., s_p, 0...)`` per node, in the multipoles'
        real dtype.
    """
    p = int(order)
    power = dehnen_multipole_power_by_degree(multipole_packed=multipole_packed)
    dtype = power.dtype
    m = jnp.maximum(jnp.abs(jnp.asarray(mass, dtype)), jnp.asarray(_TINY, dtype))
    rho = jnp.asarray(radius, dtype)
    width = walk_table_width(p)
    cols = [
        jnp.asarray(gravitational_constant, dtype) * m,
        jnp.asarray(threshold, dtype),
    ]
    # s_n = P_n / (M rho^n), built as (P_n / M) / rho / rho ... so no rho^n is
    # ever formed; a zero radius (all mass at the centre) has P_n = 0, n >= 1.
    safe_rho = jnp.where(rho > 0.0, rho, jnp.ones_like(rho))
    for n in range(1, p + 1):
        s = power[:, n] / m
        for _ in range(n):
            s = s / safe_rho
        cols.append(jnp.where(rho > 0.0, s, jnp.zeros_like(s)))
    cols += [jnp.zeros_like(m)] * (width - len(cols))
    return jnp.stack(cols, axis=1)


def _powers(x: Array, p: int) -> list[Array]:
    out = [jnp.ones_like(x)]
    for _ in range(p):
        out.append(out[-1] * x)
    return out


def dehnen_pair_accept(
    *,
    row_a: Sequence[Array],
    row_b: Sequence[Array],
    radius_a: Array,
    radius_b: Array,
    dist_sq: Array,
    order: int,
    theta_max: Any,
) -> Array:
    """Eq (16a), symmetrised, for pairs given their two gathered table rows.

    Elementwise in every argument, so it runs on a block of lanes inside Pallas
    as well as on a whole round in XLA.

    Parameters
    ----------
    row_a : Sequence[Array]
        Node A's table columns ``0..p+1`` (see :func:`dehnen_walk_table`).
    row_b : Sequence[Array]
        Node B's table columns.
    radius_a : Array
        A's MAC radius.
    radius_b : Array
        B's MAC radius.
    dist_sq : Array
        Squared centre distance.
    order : int
        Expansion order ``p``. Static.
    theta_max : Any
        The convergence clause's bound (the paper's 1).

    Returns
    -------
    Array
        True where both directions pass eq (16a) and the expansion converges.
    """
    p = int(order)
    dtype = dist_sq.dtype
    tiny = jnp.asarray(_TINY, dtype)
    r = jnp.sqrt(dist_sq)
    rsum = radius_a + radius_b
    convergent = rsum < theta_max * r
    inv_r = 1.0 / jnp.maximum(r, tiny)
    a = radius_a * inv_r
    b = radius_b * inv_r
    pa = _powers(a, p)
    pb = _powers(b, p)
    e_ab = pb[p]  # n = 0: s_0 = 1
    e_ba = pa[p]
    for n in range(1, p + 1):
        c = float(math.comb(p, n))
        e_ab = e_ab + c * row_a[WALK_TABLE_POWER0 + n - 1] * pa[n] * pb[p - n]
        e_ba = e_ba + c * row_b[WALK_TABLE_POWER0 + n - 1] * pb[n] * pa[p - n]
    improvement = (
        _IMPROVEMENT * jnp.maximum(radius_a, radius_b) / jnp.maximum(rsum, tiny)
    )
    scale = improvement * inv_r * inv_r
    ok_ab = row_a[WALK_TABLE_GM] * e_ab * scale < row_b[WALK_TABLE_THRESHOLD]
    ok_ba = row_b[WALK_TABLE_GM] * e_ba * scale < row_a[WALK_TABLE_THRESHOLD]
    return convergent & ok_ab & ok_ba


@dataclasses.dataclass(frozen=True)
class DehnenWalkAccept:
    """``pair_accept`` callable for ``yggdrax.dual_tree_walk_mutual``: eq (16a).

    A frozen dataclass so the walk's static argument hashes by VALUE: two
    instances for the same order share one compiled walk. The traced data is
    ``{"table": [nodes, W], "theta_max": scalar}``.

    Attributes
    ----------
    order : int
        Expansion order ``p``; shapes the unrolled eq (15) sum.
    """

    order: int

    def __call__(
        self,
        data: Any,
        a: Array,
        b: Array,
        dist_sq: Array,
        radius_a: Array,
        radius_b: Array,
    ) -> Array:
        """Eq (16a) for the round's pairs.

        Parameters
        ----------
        data : Any
            ``{"table": [nodes, W], "theta_max": scalar}``.
        a : Array
            First node per pair.
        b : Array
            Second node per pair.
        dist_sq : Array
            Squared centre distance.
        radius_a : Array
            A's MAC radius.
        radius_b : Array
            B's MAC radius.

        Returns
        -------
        Array
            The acceptance mask.
        """
        table = jnp.asarray(data["table"], dist_sq.dtype)
        cols = WALK_TABLE_POWER0 + int(self.order)
        rows_a = table[a]
        rows_b = table[b]
        return dehnen_pair_accept(
            row_a=[rows_a[:, k] for k in range(cols)],
            row_b=[rows_b[:, k] for k in range(cols)],
            radius_a=radius_a,
            radius_b=radius_b,
            dist_sq=dist_sq,
            order=int(self.order),
            theta_max=jnp.asarray(data["theta_max"], dist_sq.dtype),
        )


class FlatWalkCriterion(NamedTuple):
    """What the fused lane hands its flat-walk builder to accept by eq (16a).

    The table is built INSIDE the builder from the walk's own MAC extents (for
    leaves the depth-padded proxy), so the criterion and the walk's refinement
    read the same radii. A Python-side carrier, never a ``jit`` argument: ``order``
    must stay a Python int.

    Attributes
    ----------
    multipole_packed : Array
        ``[nodes, (p+1)^2]`` real-basis multipoles about the walk centres (COM).
    mass : Array
        ``[nodes]`` node masses.
    threshold : Array
        ``[nodes]`` acceptance threshold ``eps * min_b f_b`` (a force).
    order : int
        Expansion order ``p``.
    theta_max : float
        eq (16a)'s convergence bound (the paper's 1).
    gravitational_constant : float
        ``G``.
    """

    multipole_packed: Array
    mass: Array
    threshold: Array
    order: int
    theta_max: float = 1.0
    gravitational_constant: float = 1.0

    def table(self, radius: Array) -> Array:
        """The walk table on the given MAC radii.

        Parameters
        ----------
        radius : Array
            ``[nodes]`` the walk's MAC extents.

        Returns
        -------
        Array
            ``[nodes, W]`` (:func:`dehnen_walk_table`), in the radii's dtype.
        """
        return jnp.asarray(
            dehnen_walk_table(
                multipole_packed=self.multipole_packed,
                mass=self.mass,
                radius=radius,
                threshold=self.threshold,
                gravitational_constant=float(self.gravitational_constant),
                order=int(self.order),
            ),
            jnp.asarray(radius).dtype,
        )

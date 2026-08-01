"""Momentum-conserving FMM force for a block-step individual-timestep KDK.

:class:`BlockStepFMM` exposes jaccpot's mutual FMM (:mod:`jaccpot.mutual`) in the
shape a block-power-of-two KDK leapfrog consumes. It **structurally** matches
nornax's ``MutualForceModel`` protocol -- ``level_accelerations(positions,
masses, *, rung, level, args=None) -> (N, 3)`` -- without importing nornax, so
the dependency graph stays acyclic (``Jaccpot -> Yggdrax``, ``Nornax``
standalone, ``ODISSEO -> Nornax + Jaccpot``). Nornax's own test suite stays
FMM-free; the cross-repo checks live here, guarded by ``importorskip``.

What the integrator needs, and why the FMM could not supply it before
--------------------------------------------------------------------
The scheme splits interactions by level ``k = max(rung_i, rung_j)`` and requires
each level's contribution to be applied *antisymmetrically*, so an inactive
coarse partner of an active fine interaction still receives its equal-and-
opposite kick. That makes ``sum_i m_i Delta v_i == 0`` per level the defining
correctness property. Jaccpot's production force is target-centric and evaluates
every pair twice independently, so its momentum error sits at the force accuracy
(~1e-3 .. 1e-5); the mutual path in :mod:`jaccpot.mutual` computes each pair once
and applies ``+f``/``-f``, taking the residual to round-off (~1e-17 relative,
flat in ``theta`` and expansion order).

Two entry points
----------------
:meth:`BlockStepFMM.boundary_kick` is the **production** call: one activity-gated
mutual traversal per sub-step boundary, returning the updated velocities. Over a
base step that is ``n_sub + 1`` traversals (9 for ``k_max = 3``) rather than the
``sum_s (active levels at s)`` -- about 19 -- that a per-level interface forces.
For an FMM each traversal is the dominant cost, so this is the difference between
keeping and losing the individual-timestep advantage.

:meth:`BlockStepFMM.level_accelerations` completes the ``MutualForceModel``
contract and is what a stock nornax integrator drives today. It is correct at any
N but pays one traversal per active level.

Topology lifetime
-----------------
The discrete topology is frozen, host-side, and severed from the gradient --
mirroring nornax's ``stop_gradient``-ed rung schedule. Call :meth:`prepare` (or
:meth:`refresh`) once per base step; every boundary within that step then reuses
the same tree, and ``jax.grad`` through the evaluations is an exact fixed-topology
gradient. Calling a force method under tracing without a prepared state raises
rather than attempting a host traversal on tracers.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from jaccpot.mutual.force import (
    MutualFMMState,
    active_level_floor,
    build_mutual_state,
    is_sync_boundary,
    level_weights_from_floor,
    mutual_level_accelerations,
    mutual_weighted_accelerations,
    n_sub,
)
from jaccpot.mutual.topology import build_mutual_topology_from_tree

__all__ = ["BlockStepFMM"]

_SUPPORTED_BASES = ("real",)
_SUPPORTED_BACKENDS = ("jax", "pallas")


class BlockStepFMM:
    """Rung-aware, momentum-conserving FMM force model.

    Parameters
    ----------
    theta :
        Multipole acceptance parameter of the mutual MAC
        ``theta * |c_B - c_A| > R_A + R_B``. Sets the force accuracy; it has no
        effect on momentum conservation, which is structural.
    max_order :
        Multipole expansion order ``p``.
    softening :
        Plummer softening ``1 / (r^2 + eps^2)^{3/2}``.
    G :
        Gravitational constant.
    k_max :
        Highest block-step rung. Levels run ``0 .. k_max``.
    basis :
        Only ``"real"`` (the Dehnen real spherical-harmonic submode) is
        supported. It is the production basis and roughly halves the flops and
        memory of the complex submode; the cartesian basis has no mutual
        operators here.
    backend :
        ``"jax"`` (default) runs the pure-JAX kernels, differentiable by generic
        autodiff. ``"pallas"`` additionally routes the real-basis z-axis M2L
        translation -- the far-field hotspot -- through jaccpot's Pallas kernel
        where the hardware supports it, falling back to pure JAX otherwise.
    leaf_size :
        Target particles per leaf for the tree build.
    pallas_interpret :
        Run the Pallas kernels in interpret mode. Works without a GPU, so it lets
        the Pallas path's *logic* be exercised on CPU; far too slow for real use.
    """

    def __init__(
        self,
        *,
        softening: float,
        k_max: int,
        theta: float = 0.6,
        max_order: int = 4,
        G: float = 1.0,
        basis: str = "real",
        backend: str = "jax",
        leaf_size: int = 32,
        near_chunk_size: Optional[int] = None,
        pallas_interpret: bool = False,
    ) -> None:
        basis = str(basis).lower()
        if basis not in _SUPPORTED_BASES:
            raise ValueError(
                f"BlockStepFMM supports basis={_SUPPORTED_BASES!r}; got {basis!r}. "
                "The mutual near-field and dual M2L are implemented on the Dehnen "
                "real spherical-harmonic submode only."
            )
        backend = str(backend).lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "BlockStepFMM supports backend="
                f"{_SUPPORTED_BACKENDS!r}; got {backend!r}"
            )
        if int(k_max) < 0:
            raise ValueError(f"k_max must be >= 0; got {k_max!r}")

        self.theta = float(theta)
        self.max_order = int(max_order)
        self.softening = float(softening)
        self.G = float(G)
        self.k_max = int(k_max)
        self.basis = basis
        self.backend = backend
        self.leaf_size = int(leaf_size)
        self.near_chunk_size = near_chunk_size
        self.pallas_interpret = bool(pallas_interpret)
        self._state: Optional[MutualFMMState] = None
        self._solver: Any = None

    # -- topology lifetime --------------------------------------------------

    @property
    def state(self: "BlockStepFMM") -> Optional[MutualFMMState]:
        """The prepared frozen state, or ``None`` before the first build."""
        return self._state

    def prepare(
        self: "BlockStepFMM", positions: Array, masses: Array
    ) -> MutualFMMState:
        """Build the frozen topology for ``positions``/``masses`` and cache it.

        Tree construction is a host operation and cannot be traced, so this must
        run on concrete arrays -- exactly the contract of
        :meth:`jaccpot.FastMultipoleMethod.prepare_state`. Call it once per base
        step; the rung schedule is reassigned at the same cadence, so the two
        discrete refreshes line up.
        """
        from jaccpot import FastMultipoleMethod

        if self._solver is None:
            self._solver = FastMultipoleMethod(preset="balanced", basis="real")
        prepared = self._solver.prepare_state(
            positions,
            masses,
            leaf_size=self.leaf_size,
            max_order=self.max_order,
            theta=self.theta,
        )
        topology = build_mutual_topology_from_tree(
            prepared.tree,
            np.asarray(prepared.positions_sorted),
            np.asarray(prepared.masses_sorted),
            theta=self.theta,
            order=self.max_order,
        )
        self._state = build_mutual_state(
            topology,
            softening=self.softening,
            G=self.G,
            use_pallas=(self.backend == "pallas"),
            near_chunk_size=self.near_chunk_size,
            pallas_interpret=self.pallas_interpret,
        )
        return self._state

    def refresh(
        self: "BlockStepFMM", positions: Array, masses: Array
    ) -> MutualFMMState:
        """Rebuild the frozen topology (alias of :meth:`prepare`)."""
        return self.prepare(positions, masses)

    def _require_state(self, positions: Array, masses: Array) -> MutualFMMState:
        """Return the cached state, building it if that is legal here."""
        state = self._state
        if state is not None and state.num_particles == int(
            jnp.asarray(positions).shape[0]
        ):
            return state
        if isinstance(jnp.asarray(positions), jax.core.Tracer):
            raise RuntimeError(
                "BlockStepFMM has no prepared topology and cannot build one from "
                "traced positions (the dual-tree traversal is a host operation). "
                "Call prepare(positions, masses) on concrete arrays once per base "
                "step, then differentiate/jit the force evaluation."
            )
        return self.prepare(positions, masses)

    def _validate_rung(self, rung: Array) -> Array:
        """Reject rungs outside ``[0, k_max]`` while they are still concrete.

        A rung above ``k_max`` has no kick weight, so the traversal would have to
        either invent one or drop the interaction -- both of which quietly
        integrate the wrong equations. Caught here it is an obvious
        configuration error; caught nowhere it surfaces as a NaN velocity many
        steps later.
        """
        rung = jnp.asarray(rung)
        if isinstance(rung, jax.core.Tracer):
            return rung
        lo, hi = int(jnp.min(rung)), int(jnp.max(rung))
        if lo < 0 or hi > self.k_max:
            raise ValueError(
                f"rung values must lie in [0, k_max={self.k_max}]; got "
                f"[{lo}, {hi}]. Build the model with k_max >= {max(hi, 0)} or "
                "clamp the rung assignment."
            )
        return rung

    # -- MutualForceModel contract -----------------------------------------

    def level_accelerations(
        self: "BlockStepFMM",
        positions: Array,
        masses: Array,
        *,
        rung: Array,
        level: int,
        args: object = None,
    ) -> Array:
        """Return the level-``k`` antisymmetric acceleration for every particle.

        Only interactions assigned to ``level`` contribute; they are applied
        antisymmetrically, so ``sum_i m_i a_i`` vanishes to round-off, and summing
        over ``k = 0 .. k_max`` reproduces the full acceleration.

        The near field applies the exact per-particle predicate
        ``max(rung_i, rung_j) == level``. The far field assigns each cell the rung
        of its most active particle and splits at cell granularity, so this is a
        different -- but equally valid -- partition from a direct-sum oracle's.
        See :mod:`jaccpot.mutual.force`.
        """
        del args
        state = self._require_state(positions, masses)
        rung = self._validate_rung(rung)
        if not 0 <= int(level) <= self.k_max:
            raise ValueError(
                f"level must lie in [0, k_max={self.k_max}]; got {level!r}"
            )
        return mutual_level_accelerations(
            state,
            positions,
            masses,
            rung=rung,
            level=int(level),
            k_max=self.k_max,
        )

    def total_accelerations(
        self: "BlockStepFMM",
        positions: Array,
        masses: Array,
        *,
        rung: Optional[Array] = None,
        args: object = None,
    ) -> Array:
        """Return the full acceleration in a single traversal.

        Equivalent to summing :meth:`level_accelerations` over every level, but
        at one traversal instead of ``k_max + 1``.
        """
        del args, rung
        state = self._require_state(positions, masses)
        return mutual_weighted_accelerations(state, positions, masses)

    # -- fused boundary primitive ------------------------------------------

    def boundary_kick(
        self: "BlockStepFMM",
        positions: Array,
        velocities: Array,
        masses: Array,
        *,
        rung: Array,
        active_floor: int,
        dt_max: float,
        half: float = 1.0,
        args: object = None,
    ) -> Array:
        """Apply one sub-step boundary's kick in a single mutual traversal.

        Every level at or above ``active_floor`` is kicked with
        ``half * dt_max / 2**k``. Rather than evaluating one force per active
        level and summing, the levels' weights are pushed *into* the traversal,
        so the tree is walked once. Momentum is untouched by the weighting: each
        weight is a single symmetric scalar per pair, multiplying ``+f`` and
        ``-f`` alike, so ``sum_i m_i Delta v_i == 0`` holds for the fused kick
        exactly as it does per level.

        Parameters
        ----------
        active_floor :
            Smallest level kicked at this boundary (nornax's
            ``active_level_floor(s, k_max)``).
        half :
            ``0.5`` at the base step's synchronized ends, ``1.0`` inside.

        Returns
        -------
        Array
            Updated velocities.
        """
        del args
        state = self._require_state(positions, masses)
        rung = self._validate_rung(rung)
        weights = level_weights_from_floor(
            int(active_floor),
            self.k_max,
            float(dt_max),
            half=float(half),
            dtype=jnp.asarray(positions).dtype,
        )
        delta_v = mutual_weighted_accelerations(
            state, positions, masses, rung=rung, level_weights=weights
        )
        return jnp.asarray(velocities) + delta_v

    def boundary_kick_at(
        self: "BlockStepFMM",
        positions: Array,
        velocities: Array,
        masses: Array,
        *,
        rung: Array,
        s: int,
        dt_max: float,
        args: object = None,
    ) -> Array:
        """:meth:`boundary_kick` addressed by sub-step boundary index ``s``.

        Derives ``active_floor`` and ``half`` from the standard block schedule,
        so a caller that knows only ``s`` and ``k_max`` needs nothing else.
        """
        return self.boundary_kick(
            positions,
            velocities,
            masses,
            rung=rung,
            active_floor=active_level_floor(int(s), self.k_max),
            dt_max=dt_max,
            half=0.5 if is_sync_boundary(int(s), self.k_max) else 1.0,
            args=args,
        )

    def advance_base_step(
        self: "BlockStepFMM",
        positions: Array,
        velocities: Array,
        masses: Array,
        *,
        rung: Array,
        dt_max: float,
    ) -> Tuple[Array, Array, Array]:
        """Run one full base step on the fused path, at one traversal per boundary.

        The recursively-symmetric palindrome of Farr & Bertschinger (2007): a kick
        at every boundary ``s = 0 .. n_sub`` (half at the synchronized ends, full
        inside) with a drift of ``dt_min`` between consecutive boundaries. The
        rung assignment is held fixed for the whole step, which is what makes the
        map symplectic and time-reversible.

        Returns ``(positions, velocities, acceleration)`` where the acceleration
        is the full field at the end-of-step positions, ready to seed the next
        base step's rung assignment.
        """
        state = self._require_state(positions, masses)
        rung = self._validate_rung(rung)
        dtype = jnp.asarray(positions).dtype
        steps = n_sub(self.k_max)
        dt_min = jnp.asarray(dt_max, dtype=dtype) / steps
        pos, vel = jnp.asarray(positions), jnp.asarray(velocities)

        for s in range(steps + 1):
            weights = level_weights_from_floor(
                active_level_floor(s, self.k_max),
                self.k_max,
                float(dt_max),
                half=0.5 if is_sync_boundary(s, self.k_max) else 1.0,
                dtype=dtype,
            )
            vel = vel + mutual_weighted_accelerations(
                state, pos, masses, rung=rung, level_weights=weights
            )
            if s < steps:
                pos = pos + dt_min * vel

        acc = mutual_weighted_accelerations(state, pos, masses)
        return pos, vel, acc

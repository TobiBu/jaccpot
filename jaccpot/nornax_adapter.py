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
from jax import lax
from jaxtyping import Array

from jaccpot.mutual.force import (
    MutualFMMState,
    active_level_floor,
    boundary_weight_table,
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
    softening : float
        Plummer softening ``1 / (r^2 + eps^2)^{3/2}``.
    k_max : int
        Highest block-step rung. Levels run ``0 .. k_max``.
    theta : float
        Multipole acceptance parameter of the mutual MAC
        ``theta * |c_B - c_A| > R_A + R_B``. Sets the force accuracy; it has no
        effect on momentum conservation, which is structural.
    max_order : int
        Multipole expansion order ``p``.
    G : float
        Gravitational constant.
    basis : str
        Only ``"real"`` (the Dehnen real spherical-harmonic submode) is
        supported. It is the production basis and roughly halves the flops and
        memory of the complex submode; the cartesian basis has no mutual
        operators here.
    backend : str
        ``"jax"`` (default) runs the pure-JAX kernels, differentiable by generic
        autodiff. ``"pallas"`` additionally routes the real-basis z-axis M2L
        translation -- the far-field hotspot -- through jaccpot's Pallas kernel
        where the hardware supports it, falling back to pure JAX otherwise.
    leaf_size : int
        Target particles per leaf for the tree build.
    near_chunk_size : Optional[int]
        Leaf pairs per near-field scan step; ``None`` derives it from the
        pair-tensor memory budget.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode. Works without a GPU, so it lets
        the Pallas path's *logic* be exercised on CPU; far too slow for real use.

    Raises
    ------
    ValueError
        If ``basis`` or ``backend`` is unsupported, or ``k_max`` is negative.
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

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        masses : Array
            ``(n,)`` particle masses.

        Returns
        -------
        MutualFMMState
            The freshly built state, also cached on ``self`` so the force methods
            can find it.
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
        """Rebuild the frozen topology (alias of :meth:`prepare`).

        Present because ``refresh`` is the name nornax's integrator calls at a base
        step boundary; there is no behavioural difference.

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        masses : Array
            ``(n,)`` particle masses.

        Returns
        -------
        MutualFMMState
            The rebuilt state.
        """
        return self.prepare(positions, masses)

    _NO_TOPOLOGY_UNDER_TRACE = (
        "BlockStepFMM has no prepared topology and cannot build one while tracing "
        "(the dual-tree traversal is a host operation). Call "
        "prepare(positions, masses) on concrete arrays once per base step, then "
        "differentiate/jit the force evaluation."
    )

    def _require_state(self, positions: Array, masses: Array) -> MutualFMMState:
        """Return the cached state, building it if that is legal here.

        Reuses the cached state only when the particle count still matches, and
        otherwise rebuilds -- legal on concrete arrays, impossible under a trace.

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        masses : Array
            ``(n,)`` particle masses.

        Returns
        -------
        MutualFMMState
            A state matching the current particle count.

        Raises
        ------
        RuntimeError
            If no usable state is cached and one cannot be built because the
            arrays are traced. The low-level concretization failure is translated
            into an actionable instruction: prepare on concrete arrays once per
            base step, then jit or differentiate the evaluation.
        """
        state = self._state
        if state is not None and state.num_particles == int(
            jnp.asarray(positions).shape[0]
        ):
            return state
        try:
            return self.prepare(positions, masses)
        except jax.errors.JAXTypeError as exc:
            # Building needs to read the positions on the host. Translate the
            # low-level concretization failure into the actionable instruction.
            raise RuntimeError(self._NO_TOPOLOGY_UNDER_TRACE) from exc

    def _validate_rung(self, rung: Array) -> Array:
        """Reject rungs outside ``[0, k_max]`` when the bound can be read.

        A rung above ``k_max`` has no kick weight, so the traversal would have to
        either invent one or drop the interaction -- both of which quietly
        integrate the wrong equations. Caught here it is an obvious configuration
        error; caught nowhere it surfaces as a NaN velocity many steps later.

        The check is attempted and skipped on failure, rather than gated on
        ``isinstance(rung, jax.core.Tracer)``. That test looks equivalent and is
        not: a **concrete** array closed over by a ``lax.cond``/``lax.scan`` branch
        is not a ``Tracer``, yet reducing it still yields a tracer inside the
        trace, so ``int(...)`` raises. nornax's per-level integrator path closes
        over exactly such a rung array, and the isinstance form let it through into
        a ``ConcretizationTypeError``. Attempting the read is the only test that
        actually asks the question "can a value be read here".

        Parameters
        ----------
        rung : Array
            ``(n,)`` per-particle rung assignment.

        Returns
        -------
        Array
            ``rung`` as an array, unchanged. Returned rather than validated in
            place so callers can use it inline.

        Raises
        ------
        ValueError
            If the bound is readable and any rung falls outside ``[0, k_max]``.
            When it cannot be read the check is skipped, so passing this is not
            proof the rungs are in range.
        """
        rung = jnp.asarray(rung)
        try:
            lo, hi = int(jnp.min(rung)), int(jnp.max(rung))
        except jax.errors.JAXTypeError:
            return rung
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

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        masses : Array
            ``(n,)`` particle masses.
        rung : Array
            ``(n,)`` per-particle rung assignment.
        level : int
            Interaction level to isolate. Must lie in ``[0, k_max]``.
        args : object
            Ignored; present because the ``MutualForceModel`` contract passes it.

        Returns
        -------
        Array
            ``(n, 3)`` accelerations from ``level`` alone.

        Raises
        ------
        ValueError
            If ``level`` is outside ``[0, k_max]``, or the rungs are out of range.
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

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        masses : Array
            ``(n,)`` particle masses.
        rung : Optional[Array]
            Ignored -- the unweighted total does not depend on the rung
            assignment. Accepted so the signature matches the level-aware methods.
        args : object
            Ignored; present for the ``MutualForceModel`` contract.

        Returns
        -------
        Array
            ``(n, 3)`` full accelerations.
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
        active_floor: Any = None,
        dt_max: Any = None,
        half: Any = 1.0,
        level_weights: Optional[Array] = None,
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
        positions : Array
            ``(N, 3)`` particle positions, in the caller's original order.
        velocities : Array
            ``(N, 3)`` velocities to kick.
        masses : Array
            ``(N,)`` particle masses.
        rung : Array
            ``(N,)`` per-particle block-step rung, in ``[0, k_max]``.
        active_floor : Any
            Smallest level kicked at this boundary (nornax's
            ``active_level_floor(s, k_max)``). May be a tracer.
        dt_max : Any
            Base-step timestep; level ``k`` is kicked with ``half * dt_max / 2**k``.
            May be a tracer.
        half : Any
            ``0.5`` at the base step's synchronized ends, ``1.0`` inside. May be
            a tracer.
        level_weights : Optional[Array]
            The ``(k_max + 1,)`` weight vector, supplied directly instead of being
            derived from ``active_floor``/``half``/``dt_max``. Takes precedence
            over all three, which are then ignored.

            This is the seam that lets an integrator drive the boundaries with
            ``lax.scan`` rather than unrolling them. With the weights derived
            from *static* ``active_floor``/``half``, a fused base step has to emit
            one traced boundary kick per boundary, so the traced graph grows like
            ``2**k_max`` even though the runtime cost is only ``n_sub + 1``
            traversals. Handing in a row of
            :func:`~jaccpot.mutual.force.boundary_weight_table` -- indexable with a
            traced boundary index -- collapses that to a single traced kick while
            keeping the runtime win.
        args : object
            Unused; present for the ``MutualForceModel`` protocol's signature.

        Returns
        -------
        Array
            Updated velocities.

        Raises
        ------
        ValueError
            If neither ``level_weights`` nor both ``active_floor`` and ``dt_max``
            are given, or ``level_weights`` has the wrong length for ``k_max``.
        """
        del args
        state = self._require_state(positions, masses)
        rung = self._validate_rung(rung)
        if level_weights is None:
            if dt_max is None or active_floor is None:
                raise ValueError(
                    "boundary_kick needs either level_weights, or both "
                    "active_floor and dt_max"
                )
            level_weights = level_weights_from_floor(
                active_floor,
                self.k_max,
                dt_max,
                half=half,
                dtype=jnp.asarray(positions).dtype,
            )
        else:
            level_weights = jnp.asarray(
                level_weights, dtype=jnp.asarray(positions).dtype
            )
            if int(level_weights.shape[-1]) != self.k_max + 1:
                raise ValueError(
                    f"level_weights must have {self.k_max + 1} entries for "
                    f"k_max={self.k_max}; got shape {tuple(level_weights.shape)}"
                )
        delta_v = mutual_weighted_accelerations(
            state, positions, masses, rung=rung, level_weights=level_weights
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

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` particle positions.
        velocities : Array
            ``(n, 3)`` particle velocities, the quantity being kicked.
        masses : Array
            ``(n,)`` particle masses.
        rung : Array
            ``(n,)`` per-particle rung assignment.
        s : int
            Sub-step boundary index, ``0 .. n_sub``. Must be concrete -- it drives
            Python-level schedule arithmetic.
        dt_max : float
            Base-step size, the time step of level ``0``.
        args : object
            Forwarded to :meth:`boundary_kick`, which ignores it.

        Returns
        -------
        Array
            ``(n, 3)`` velocities after the kick.
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
        scan_boundaries: bool = False,
    ) -> Tuple[Array, Array, Array]:
        """Run one full base step on the fused path, at one traversal per boundary.

        The recursively-symmetric palindrome of Farr & Bertschinger (2007): a kick
        at every boundary ``s = 0 .. n_sub`` (half at the synchronized ends, full
        inside) with a drift of ``dt_min`` between consecutive boundaries. The
        rung assignment is held fixed for the whole step, which is what makes the
        map symplectic and time-reversible.

        ``scan_boundaries`` walks the boundaries with a ``lax.scan`` over a
        precomputed weight table, so the *traced* graph holds **one** boundary kick
        regardless of ``k_max`` -- 10.35x fewer top-level jaxpr equations at
        ``k_max = 3`` -- while the runtime still performs ``n_sub + 1`` traversals.

        It is **off by default**, because that win costs peak memory rather than
        saving it. Jaccpot's inner kernels are individually jitted, so the unrolled
        Python loop reuses their cached executables, whereas the scan has to inline
        the whole force into one program and compile that. Measured over a 6
        base-step rollout at ``k_max = 2``, float64:

            scan   2.67 GB (N=512)   2.70 GB (N=256)
            unroll 2.08 GB (N=512)   1.92 GB (N=256)

        Note how little ``N`` moves either number: this is compile/executable
        memory, not data. Turning the scan on by default regressed the CI
        integration shard from passing to OOM-killing its workers.

        So: leave it off for ordinary eager stepping, and turn it on when trace
        size is what binds -- an outer ``jax.jit`` over the rollout, or a deep
        ``k_max`` where ``2**k_max`` unrolled kicks stop fitting. An integrator that
        wants both (small traces *and* per-boundary traversals) should drive
        :meth:`boundary_kick` with rows of
        :func:`~jaccpot.mutual.force.boundary_weight_table` from its own scan.

        Returns ``(positions, velocities, acceleration)`` where the acceleration
        is the full field at the end-of-step positions, ready to seed the next
        base step's rung assignment. It is a separate evaluation on purpose: a
        boundary kick returns *weighted* levels, and the unweighted total cannot be
        recovered from them.

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` positions at the start of the base step.
        velocities : Array
            ``(n, 3)`` velocities at the start of the base step.
        masses : Array
            ``(n,)`` particle masses.
        rung : Array
            ``(n,)`` per-particle rung assignment, held fixed for the whole step.
        dt_max : float
            Base-step size, the time step of level ``0``.
        scan_boundaries : bool
            Walk the boundaries with ``lax.scan`` rather than an unrolled Python
            loop. Off by default -- see the memory trade above.

        Returns
        -------
        Tuple[Array, Array, Array]
            ``(positions, velocities, acceleration)`` at the end of the step, the
            acceleration being the full unweighted field at the final positions.
        """
        state = self._require_state(positions, masses)
        rung = self._validate_rung(rung)
        dtype = jnp.asarray(positions).dtype
        steps = n_sub(self.k_max)
        dt_min = jnp.asarray(dt_max, dtype=dtype) / steps
        pos, vel = jnp.asarray(positions), jnp.asarray(velocities)

        if scan_boundaries:
            table = boundary_weight_table(self.k_max, dt_max, dtype=dtype)
            zero = jnp.asarray(0.0, dtype=dtype)

            def body(
                carry: Tuple[Array, Array], s: Array
            ) -> Tuple[Tuple[Array, Array], None]:
                position, velocity = carry
                velocity = velocity + mutual_weighted_accelerations(
                    state, position, masses, rung=rung, level_weights=table[s]
                )
                # The drift is a no-op after the final kick, expressed as a select
                # so every scan iteration has the same shape.
                position = position + jnp.where(s < steps, dt_min, zero) * velocity
                return (position, velocity), None

            (pos, vel), _ = lax.scan(
                body, (pos, vel), jnp.arange(steps + 1, dtype=jnp.int32)
            )
        else:
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

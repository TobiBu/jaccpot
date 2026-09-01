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

from types import SimpleNamespace
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array, Float, Int

from jaccpot.mutual.force import (
    MutualCapacities,
    MutualFMMState,
    active_level_floor,
    boundary_weight_table,
    build_mutual_state,
    is_sync_boundary,
    level_weights_from_floor,
    mutual_level_accelerations,
    mutual_weighted_accelerations,
    n_sub,
    resolve_mutual_capacities,
)
from jaccpot.mutual.topology import build_mutual_topology_from_tree

__all__ = ["BlockStepFMM", "DistributedBlockStepFMM"]

_SUPPORTED_BASES = ("real",)
_SUPPORTED_BACKENDS = ("jax", "pallas")


class BlockStepFMM:
    """Rung-aware, momentum-conserving FMM force model.

    **On the leaf_size default of 64.** It is measured, not conventional, and the
    measurement is worth recording because leaf size is the knob that decides
    which *stage* of a traversal dominates -- and therefore which optimisation is
    worth doing at all. Full mutual traversal, A100, fp64, theta 0.7, order 4,
    device topology, Hernquist (absolutes are contention-soft by ~1.4x; the shape
    reproduced across three problem sizes):

    ======  ==========  =========  =========  ===============  ==============
    leaf    N = 2e4     N = 1e5    N = 1e6    near, M2L @ 1e6  force err @ 1e5
    ======  ==========  =========  =========  ===============  ==============
    16      --           396.9 ms   5987 ms   17.8%, 68.7%     2.34e-3
    32       40.5 ms     237.8 ms   3846 ms   31.7%, 53.2%     2.28e-3
    **64**   28.8 ms    *193.1 ms* *3037 ms*  59.0%, 31.1%     2.10e-3
    128      28.3 ms     338.0 ms   4808 ms   80.4%, 10.8%     1.78e-3
    256     --           527.6 ms   8401 ms   94.7%,  2.5%     8.27e-4
    ======  ==========  =========  =========  ===============  ==============

    A U-curve whose minimum is 64 at every size measured (64 and 128 tie at
    N = 2e4). The previous default of 32 was **1.23-1.41x slower AND less
    accurate**, so moving to 64 costs nothing on either axis -- the accuracy
    gradient runs *with* leaf size, because a bigger leaf resolves more pairs by
    exact near-field summation instead of by a multipole approximation.

    Above 64 the curve buys accuracy with time at roughly a linear rate (leaf 256:
    2.54x better force error for 2.77x the time), so 128 and 256 are legitimate
    choices for an accuracy-led run -- they are on the Pareto frontier. 16 and 32
    are not: they are dominated on both axes.

    The share columns are the reason this note exists. M2L is 2.5% of a traversal
    at leaf 256 and 68.7% at leaf 16; the near field is 94.7% and 17.8%. Any claim
    that one stage "is the hotspot" is a claim about this knob.

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
        Target particles per leaf for the tree build. Default 64, which is the
        measured optimum for this lane rather than a convention -- see the note
        above :class:`BlockStepFMM` for the curve.
    near_chunk_size : Optional[int]
        Leaf pairs per near-field scan step; ``None`` derives it from the
        pair-tensor memory budget.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode. Works without a GPU, so it lets
        the Pallas path's *logic* be exercised on CPU; far too slow for real use.
    topology_backend : str
        ``"host"`` (default) builds the topology with the NumPy dual-tree
        traversal in :mod:`jaccpot.mutual.topology`, which cannot be traced.
        ``"device"`` builds it in JAX instead, so :meth:`rebuild_state` can live
        inside a ``jax.jit`` or a ``lax.scan``; it implies ``static_shapes``,
        because every device output shape comes from the capacities.
    static_shapes : bool
        Pad the pair lists and the level schedule to fixed capacities, resolved
        from the first :meth:`prepare` and then held (see :attr:`capacities`).
        Without it a prepared state's shapes track the particle distribution, so
        every rebuild is a distinct set of compile-time constants and a jitted
        force recompiles per base step -- measured ~200 s each at N = 20 000, with
        no warm-up. With it the shapes depend on the capacities alone and one
        program serves the run. The cost is doing a little work on padding: the
        pair lists carry their existing ``near_valid``/``far_valid`` masks, and a
        padded level row is an entirely-invalid no-op.
    caps : Optional[MutualCapacities]
        Use these capacities instead of resolving them from the first build.
        Implies ``static_shapes=True``. Pass a profile recorded from a previous
        run to make the *first* compile reusable too.
    validate_rung : bool
        Check that rungs lie in ``[0, k_max]``. The check is a device-to-host
        sync on concrete inputs, so a driver that has already established the
        range can turn it off; see :meth:`_validate_rung`.

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
        leaf_size: int = 64,
        near_chunk_size: Optional[int] = None,
        pallas_interpret: bool = False,
        topology_backend: str = "host",
        static_shapes: bool = False,
        caps: Optional[MutualCapacities] = None,
        validate_rung: bool = True,
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
        topology_backend = str(topology_backend).lower()
        if topology_backend not in ("host", "device"):
            raise ValueError(
                "topology_backend must be 'host' or 'device'; got "
                f"{topology_backend!r}"
            )
        self.topology_backend = topology_backend
        # A device topology has no unpadded form to fall back on: every output
        # shape comes from the capacities, so they are not optional there.
        self.static_shapes = (
            bool(static_shapes) or caps is not None or topology_backend == "device"
        )
        self.validate_rung = bool(validate_rung)
        self._caps: Optional[MutualCapacities] = caps
        self._state: Optional[MutualFMMState] = None
        self._solver: Any = None
        # Device backend only: the frozen static-radix template and the tree
        # arrays derived from it. Built once on the host; every later refresh is
        # traceable.
        self._template: Any = None
        self._tree_static: Optional[dict] = None

    # -- topology lifetime --------------------------------------------------

    @property
    def state(self: "BlockStepFMM") -> Optional[MutualFMMState]:
        """The prepared frozen state, or ``None`` before the first build."""
        return self._state

    @property
    def capacities(self: "BlockStepFMM") -> Optional[MutualCapacities]:
        """The frozen capacity profile, or ``None`` when shapes float.

        With ``static_shapes=True`` this is resolved from the *first* topology and
        then held: the pair lists and level schedule are padded to it on every
        subsequent build, so the prepared state's shapes stop depending on the
        particle distribution and one compiled program serves the whole run. It
        is the same resolve-eagerly-then-freeze discipline
        ``jaccpot/runtime/_large_n_pipeline.py`` uses for its own caps.
        """
        return self._caps

    def prepare(
        self: "BlockStepFMM", positions: Float[Array, "n 3"], masses: Float[Array, "n"]
    ) -> MutualFMMState:
        """Build the frozen topology for ``positions``/``masses`` and cache it.

        Tree construction is a host operation and cannot be traced, so this must
        run on concrete arrays -- exactly the contract of
        :meth:`jaccpot.FastMultipoleMethod.prepare_state`. Call it once per base
        step; the rung schedule is reassigned at the same cadence, so the two
        discrete refreshes line up.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.

        Returns
        -------
        MutualFMMState
            The freshly built state, also cached on ``self`` so the force methods
            can find it.

        Raises
        ------
        RuntimeError
            On the device backend, if the built topology overflowed its capacity
            profile. Overflow drops interactions while leaving momentum exactly
            conserved, so it is raised here rather than reported.
        """
        from jaccpot import FastMultipoleMethod

        if self.topology_backend == "device":
            # Freeze the template on the first call, then every refresh -- this
            # one included -- goes through the traceable device path.
            self.freeze_template(positions, masses)
            self._state = self.rebuild_state(positions, masses)
            # prepare() is the eager entry point, so this is the one place the
            # overflow flag can be turned into an exception. A driver stepping
            # through rebuild_state under trace must check it itself.
            try:
                overflowed = bool(self._state.topology_overflow)
            except jax.errors.JAXTypeError:  # pragma: no cover - traced caller
                overflowed = False
            if overflowed:
                from jaccpot.mutual.force import OVERFLOW_CAUSES

                bits = int(self._state.overflow_causes)
                blamed = [
                    name
                    for index, name in enumerate(OVERFLOW_CAUSES)
                    if bits & (1 << index)
                ]
                raise RuntimeError(
                    "the device topology overflowed its capacity profile: "
                    f"{', '.join(blamed) or 'unknown'} exceeded. Profile was "
                    f"far={self._caps.far}, near={self._caps.near}, "
                    f"depth={self._caps.depth}, width={self._caps.width}. "
                    "Interactions were dropped, and that is invisible in the "
                    "force -- a dropped canonical pair loses both halves, so "
                    "momentum stays exact -- hence the raise. Rebuild with larger "
                    "caps, or pass caps=None to resolve them from this "
                    "configuration."
                )
            return self._state

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
        if self.static_shapes and self._caps is None:
            # Only the host lane asks for drift headroom. It rebuilds an LBVH
            # tree every base step, and LBVH depth is volatile over a rollout;
            # the device lane's static-radix linkage is frozen, so freeze_template
            # overrides depth with the invariant measured value instead.
            self._caps = resolve_mutual_capacities(topology, drift_headroom=True)
        self._state = build_mutual_state(
            topology,
            softening=self.softening,
            G=self.G,
            use_pallas=(self.backend == "pallas"),
            near_chunk_size=self.near_chunk_size,
            pallas_interpret=self.pallas_interpret,
            caps=self._caps,
        )
        return self._state

    # -- device topology backend -------------------------------------------

    def freeze_template(
        self: "BlockStepFMM",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
    ) -> None:
        """Build the static-radix template and capacity profile. Host-side, once.

        Only the *data structure* is frozen here -- the parent/child links and the
        leaf bucket boundaries, which for a static-radix tree are
        ``arange(0, N, leaf_size)`` and so do not depend on the particle
        distribution at all. The spatial content is not frozen: every
        :meth:`rebuild_state` re-sorts the particles by Morton code and
        recomputes the centres of mass and radii from the live positions, so the
        MAC re-decides which pairs are far on every call. Only the *number* of
        such pairs is bounded, by the capacities.

        Idempotent: a second call is a no-op, because re-freezing would silently
        change the capacities out from under an already-compiled program.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Concrete positions to build the template from.
        masses : Float[Array, 'n']
            Concrete particle masses.
        """
        if self._template is not None:
            return
        import numpy as _np
        from yggdrax._tree_impl import rebuild_static_radix_tree_from_template
        from yggdrax.tree import Tree

        from jaccpot.mutual.device_topology import node_depths
        from jaccpot.mutual.farfield import snap_capacity

        tree = Tree.from_particles(
            positions,
            masses,
            tree_type="radix",
            build_mode="static_radix",
            leaf_size=self.leaf_size,
        )
        template = getattr(tree, "topology", tree)
        refreshed, sorted_positions, sorted_masses, inverse = (
            rebuild_static_radix_tree_from_template(
                positions, masses, template, return_reordered=True
            )
        )
        parent = _np.asarray(refreshed.parent)
        root = int(_np.flatnonzero(parent < 0)[0]) if (parent < 0).any() else 0
        if self._caps is None:
            # Resolve the capacities from a host build on this configuration --
            # the one place the host traversal is still used, and only once.
            # `SimpleNamespace` rather than a bare class with attributes bolted on
            # afterwards: the same duck-typed object, but the attributes exist by
            # construction instead of being five assignments a checker must reject
            # (audit E.4 bucket L).
            shim = SimpleNamespace(
                parent=refreshed.parent,
                left_child=refreshed.left_child,
                right_child=refreshed.right_child,
                node_ranges=refreshed.node_ranges,
                inverse_permutation=inverse,
            )
            self._caps = resolve_mutual_capacities(
                build_mutual_topology_from_tree(
                    shim,
                    np.asarray(sorted_positions),
                    np.asarray(sorted_masses),
                    theta=self.theta,
                    order=self.max_order,
                )
            )
        # The generous depth bound `resolve_mutual_capacities` applies exists for
        # LBVH, whose depth follows the Morton-code distribution and was measured
        # to swing 12 -> 34 over a Hernquist rollout. It does not apply here and
        # costs real work: `depth` is the cascade scan's iteration count, so a 4x
        # bound is 4x the M2M/L2L iterations.
        #
        # A static-radix template's node linkage is a fixed balanced bisection
        # over leaf-bucket *indices* and is frozen for the run -- `rebuild_state`
        # re-sorts the particles but never rewires the tree -- so the level
        # structure, and hence the depth, is literally invariant across rebuilds.
        # One row of slack is enough.
        measured_depth = 0
        probe_depth = node_depths(
            jnp.asarray(refreshed.parent), root, depth_cap=int(self._caps.depth) + 2
        )
        measured_depth = int(jnp.max(probe_depth)) + 1
        self._caps = self._caps._replace(
            depth=snap_capacity(measured_depth, relative=0.0, absolute=2)
        )

        # Resolve the wavefront capacity by trial. It bounds the widest
        # *intermediate* pair front, which no finished topology records, so
        # there is nothing to compute it from -- only something to test it
        # against. Doubling from a floor and stopping at the first front that
        # does not overflow mirrors yggdrax's own capacity-retry ladder, and is
        # the same resolve-eagerly-then-freeze discipline the rest of the
        # capacities use. Done once, here, on concrete arrays.
        if int(self._caps.queue) <= 0:
            self._caps = self._caps._replace(
                queue=self._resolve_queue_capacity(
                    refreshed, sorted_positions, sorted_masses, root
                )
            )
        self._template = template
        self._tree_static = {
            "parent": jnp.asarray(refreshed.parent),
            "left_child": jnp.asarray(refreshed.left_child),
            "right_child": jnp.asarray(refreshed.right_child),
            "root": jnp.asarray(root),
        }

    _QUEUE_FLOOR = 1 << 14
    _QUEUE_CEILING = 1 << 24
    # Rungs to skip by starting the ladder near the answer instead of at the
    # floor. Measured first-non-overflowing wavefront against the leaf count:
    #
    #   N=4e3   leaf 64        64 leaves   ->     16,384   (256 per leaf)
    #   N=65e3  leaf 256      256 leaves   ->     16,384   ( 64 per leaf)
    #   N=1e6   leaf 256    3,906 leaves   ->  1,048,576   (268 per leaf)
    #   N=1e6   leaf 64    15,625 leaves   ->  2,097,152   (134 per leaf)
    #
    # so the requirement runs 64-268 wavefront slots per leaf. Seeded at the
    # bottom of that range ON PURPOSE: the ladder has to converge on a TIGHT
    # capacity, because the queue sizes the slices every walk iteration processes,
    # so overshooting costs run time on every rebuild -- whereas undershooting
    # only costs one more probe here, once per run.
    _QUEUE_SEED_PER_LEAF = 64

    def _resolve_queue_capacity(
        self: "BlockStepFMM",
        refreshed: Any,
        sorted_positions: Array,
        sorted_masses: Array,
        root: int,
    ) -> int:
        """Smallest power-of-two wavefront this configuration traverses cleanly.

        Raises rather than returning an overflowing capacity: a truncated
        traversal is a wrong force that looks healthy from every other angle.

        Parameters
        ----------
        refreshed : Any
            The refreshed static-radix tree to probe against.
        sorted_positions : Array
            ``(N, 3)`` positions in tree order.
        sorted_masses : Array
            ``(N,)`` masses in tree order.
        root : int
            Index of the root node.

        Returns
        -------
        int
            A wavefront capacity that traverses this configuration without overflowing,
            with one doubling of headroom for the front widening as the system evolves.

        Raises
        ------
        RuntimeError
            If no capacity up to the ceiling traverses it. A larger ``leaf_size``
            shrinks the tree and hence the pair front.
        """
        from jaccpot.mutual.device_topology import build_mutual_state_device

        # Start near the answer rather than at the floor. This ladder is the bulk
        # of `freeze_template`: measured 31-65 s across a leaf/N sweep, almost all
        # of it climbing 2^14 -> 2^20/2^21/2^22, i.e. 6-8 full device topology
        # builds spent discovering a capacity the leaf count already predicts. It
        # also costs ~12 s on every device-lane test.
        num_leaves = -(-int(sorted_positions.shape[0]) // int(self.leaf_size))
        want = self._QUEUE_SEED_PER_LEAF * max(1, num_leaves)
        queue = max(self._QUEUE_FLOOR, 1 << max(0, (want - 1).bit_length()))
        queue = min(queue, self._QUEUE_CEILING)
        while queue <= self._QUEUE_CEILING:
            probe = build_mutual_state_device(
                sorted_positions,
                sorted_masses,
                parent=jnp.asarray(refreshed.parent),
                left_child=jnp.asarray(refreshed.left_child),
                right_child=jnp.asarray(refreshed.right_child),
                node_ranges=refreshed.node_ranges,
                inverse_permutation=jnp.arange(int(sorted_positions.shape[0])),
                root=jnp.asarray(root),
                theta=self.theta,
                order=self.max_order,
                leaf_size=self.leaf_size,
                caps=self._caps._replace(queue=queue),
                softening=self.softening,
                G=self.G,
                max_pair_queue=queue,
            )
            # Only the pair-queue cause is this resolver's business. Doubling the
            # wavefront cannot fix a starved far/near/level cap, so conflating the
            # causes here would exhaust the ladder and report "no workable
            # wavefront" for a profile whose actual problem is somewhere else --
            # a misleading error for a real misconfiguration. Those causes are
            # left to `prepare`, which names them.
            from jaccpot.mutual.force import OVERFLOW_CAUSES

            queue_bit = 1 << OVERFLOW_CAUSES.index("pair_queue")
            if not bool(int(probe.overflow_causes) & queue_bit):
                # One doubling of headroom: the front widens as the system
                # evolves, and a re-resolve mid-run would change the compiled
                # shape out from under an already-traced program.
                return min(queue * 2, self._QUEUE_CEILING)
            queue *= 2
        raise RuntimeError(
            f"could not find a wavefront capacity <= {self._QUEUE_CEILING} that "
            f"traverses this configuration (N={int(sorted_positions.shape[0])}, "
            f"leaf_size={self.leaf_size}, theta={self.theta}) without overflow. "
            "A larger leaf_size shrinks the tree and hence the pair front."
        )

    def weighted_accelerations(
        self: "BlockStepFMM",
        state: MutualFMMState,
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Optional[Int[Array, "n"]] = None,
        level_weights: Optional[Float[Array, "levels"]] = None,
    ) -> Array:
        """Evaluate ``sum_k level_weights[k] * a_k`` against an explicit state.

        The stateless counterpart of :meth:`boundary_kick` /
        :meth:`total_accelerations`, which read ``self._state``. A driver that
        rebuilds the topology inside a ``lax.scan`` has the state as a traced
        value in its carry, not on the instance, so it needs this form -- and
        having it here keeps ``jaccpot.mutual``'s internals out of the caller.

        With both ``rung`` and ``level_weights`` omitted this is the full
        acceleration.

        Parameters
        ----------
        state : MutualFMMState
            The prepared topology to evaluate against.
        positions : Float[Array, 'n 3']
            Positions, in the caller's original order.
        masses : Float[Array, 'n']
            Particle masses.
        rung : Optional[Int[Array, 'n']]
            Per-particle rung; ``None`` weights every pair equally.
        level_weights : Optional[Float[Array, 'levels']]
            Per-level weights, ``k_max + 1`` of them; ``None`` means all ones.

        Returns
        -------
        Array
            ``(N, 3)`` weighted acceleration.
        """
        return mutual_weighted_accelerations(
            state, positions, masses, rung=rung, level_weights=level_weights
        )

    def boundary_weights(
        self: "BlockStepFMM",
        active_floor: Any,
        dt_max: Any,
        half: Any = 1.0,
        *,
        dtype: Any = None,
    ) -> Array:
        """The ``(k_max + 1,)`` weight row for one sub-step boundary.

        Exposed so a caller driving :meth:`weighted_accelerations` directly does
        not have to re-derive the schedule, and cannot get it subtly wrong.

        Parameters
        ----------
        active_floor : Any
            Smallest level kicked at this boundary. May be a tracer.
        dt_max : Any
            Base-step timestep. May be a tracer.
        half : Any
            ``0.5`` at a synchronized end of the base step, ``1.0`` inside.
        dtype : Any
            Result dtype; ``None`` takes the model's working dtype.

        Returns
        -------
        Array
            ``(k_max + 1,)`` weight row.
        """
        return level_weights_from_floor(
            active_floor, self.k_max, dt_max, half=half, dtype=dtype
        )

    def rebuild_state(
        self: "BlockStepFMM",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
    ) -> MutualFMMState:
        """Build a complete mutual state on device. **Traceable.**

        This is the seam a fully-jitted rollout drives: Morton re-sort, node
        geometry, dual-tree traversal, level schedule and leaf blocks, all in
        JAX, all at capacities fixed by :meth:`freeze_template`. One compiled
        program serves every call, including calls where the accepted pair set
        changes -- measured: far pairs 404 -> 542 on a displaced system with the
        jit cache still at one entry.

        Unlike :meth:`prepare` it does **not** cache the result on the instance:
        under trace there is nothing meaningful to cache, and a driver carrying
        the state through a ``lax.scan`` needs it returned, not stashed.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            Positions, in the caller's original order.
        masses : Float[Array, 'n']
            Particle masses.

        Returns
        -------
        MutualFMMState
            A freshly built device state at the frozen template's capacities.

        Raises
        ------
        RuntimeError
            If :meth:`freeze_template` has not been called. The template and the
            capacity profile are host-built and cannot be derived under trace.
        """
        if self._template is None:
            raise RuntimeError(
                "call freeze_template(positions, masses) on concrete arrays once "
                "before rebuild_state; the template and capacities are host-built "
                "and cannot be derived under trace"
            )
        from yggdrax._tree_impl import rebuild_static_radix_tree_from_template

        from jaccpot.mutual.device_topology import build_mutual_state_device

        refreshed, sorted_positions, sorted_masses, inverse = (
            rebuild_static_radix_tree_from_template(
                positions, masses, self._template, return_reordered=True
            )
        )
        return build_mutual_state_device(
            sorted_positions,
            sorted_masses,
            node_ranges=refreshed.node_ranges,
            inverse_permutation=inverse,
            caps=self._caps,
            theta=self.theta,
            order=self.max_order,
            leaf_size=self.leaf_size,
            softening=self.softening,
            G=self.G,
            use_pallas=(self.backend == "pallas"),
            near_chunk_size=self.near_chunk_size,
            pallas_interpret=self.pallas_interpret,
            max_pair_queue=int(self._caps.queue) or (1 << 16),
            **self._tree_static,
        )

    def refresh(
        self: "BlockStepFMM", positions: Float[Array, "n 3"], masses: Float[Array, "n"]
    ) -> MutualFMMState:
        """Rebuild the frozen topology (alias of :meth:`prepare`).

        Present because ``refresh`` is the name nornax's integrator calls at a base
        step boundary; there is no behavioural difference.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
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
        ``isinstance(rung, Tracer)``. That test looks equivalent and is
        not: a **concrete** array closed over by a ``lax.cond``/``lax.scan`` branch
        is not a ``Tracer``, yet reducing it still yields a tracer inside the
        trace, so ``int(...)`` raises. nornax's per-level integrator path closes
        over exactly such a rung array, and the isinstance form let it through into
        a ``ConcretizationTypeError``. Attempting the read is the only test that
        actually asks the question "can a value be read here".

        On a concrete array the read is a **device-to-host sync**, paid on every
        ``boundary_kick`` -- ``2**k_max + 1`` times per base step. That is cheap
        next to a traversal and worth it by default, because the failure it
        catches is otherwise a NaN velocity many steps later. Set
        ``validate_rung=False`` once the caller has checked the range itself,
        which is what a driver stepping a fixed rung ladder can do.

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
        if not self.validate_rung:
            return rung
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
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
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
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
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
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Optional[Int[Array, "n"]] = None,
        args: object = None,
    ) -> Array:
        """Return the full acceleration in a single traversal.

        Equivalent to summing :meth:`level_accelerations` over every level, but
        at one traversal instead of ``k_max + 1``.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Optional[Int[Array, 'n']]
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
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        active_floor: Any = None,
        dt_max: Any = None,
        half: Any = 1.0,
        level_weights: Optional[Float[Array, "levels"]] = None,
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
        positions : Float[Array, 'n 3']
            ``(N, 3)`` particle positions, in the caller's original order.
        velocities : Float[Array, 'n 3']
            ``(N, 3)`` velocities to kick.
        masses : Float[Array, 'n']
            ``(N,)`` particle masses.
        rung : Int[Array, 'n']
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
        level_weights : Optional[Float[Array, 'levels']]
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
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        s: int,
        dt_max: float,
        args: object = None,
    ) -> Array:
        """:meth:`boundary_kick` addressed by sub-step boundary index ``s``.

        Derives ``active_floor`` and ``half`` from the standard block schedule,
        so a caller that knows only ``s`` and ``k_max`` needs nothing else.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        velocities : Float[Array, 'n 3']
            ``(n, 3)`` particle velocities, the quantity being kicked.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
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
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
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
        positions : Float[Array, 'n 3']
            ``(n, 3)`` positions at the start of the base step.
        velocities : Float[Array, 'n 3']
            ``(n, 3)`` velocities at the start of the base step.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
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


class DistributedBlockStepFMM:
    """:class:`BlockStepFMM` over a device mesh, on the distributed mutual lane.

    Same contract, same guarantees, one more thing frozen. It satisfies nornax's
    ``MutualForceModel`` structurally -- ``level_accelerations(positions, masses, *,
    rung, level, args=None)`` -- and :class:`~nornax.forces.FusedMutualForceModel` via
    :meth:`boundary_kick`, without importing nornax, exactly as the single-device class
    does.

    Momentum is conserved across the WHOLE mesh, not per device, and by construction
    rather than numerically: the intra-domain half evaluates each pair once and writes
    ``+f``/``-f`` from the same tensor, and the cross-domain half returns the ``-f`` to
    the domain that owns the other endpoint. The level weight rides both halves as a
    single symmetric scalar per pair, so ``sum_i m_i Delta v_i == 0`` holds per level
    for the fused kick as well.

    **What :meth:`prepare` freezes, and how that differs from the single-device lane.**
    On one device ``prepare`` builds the TOPOLOGY -- the accepted pair lists -- and
    every boundary in the base step reuses it. Here it freezes the *partition* (which
    particle belongs to which device), the padding layout, the tree bounds, the
    capacities, and the compiled program; the per-device tree is rebuilt inside that
    program on each call, from the positions handed in. Two consequences, both worth
    stating plainly rather than glossing:

    * The rebuild cadence is *finer*, not coarser: successive boundaries within a base
      step see trees built from their own positions. Nothing about legality changes,
      because each evaluation is internally self-consistent -- its levels partition its
      own pairs and each level's momentum cancels exactly. What is lost is only the
      bit-level identity of the topology across a base step.
    * ``prepare`` therefore does not have to be called once per base step. Call it when
      the *partition* should change -- the system's extent has moved materially, or the
      domains have gone out of balance -- not on a fixed cadence.

    The reason the compiled program is cached rather than rebuilt is not tidiness.
    ``shard_map`` wraps a fresh closure on every call, so ``jax.jit`` sees a fresh
    cache key and recompiles: :func:`~jaccpot.mutual.distributed.distributed_mutual_fmm`
    pays that on every invocation, which is fine for one force and ruinous for the
    ``n_sub + 1`` evaluations a base step asks for. Holding it is what makes a
    distributed base step cost traversals instead of compiles -- the same
    resolve-eagerly-then-freeze discipline
    :meth:`BlockStepFMM.freeze_template` applies on one device.

    **Eager, not traceable.** Every method returns concrete arrays: the driver
    assembles the result back into input order with NumPy, and the partition itself is
    host work. So there is no distributed counterpart to
    :meth:`BlockStepFMM.rebuild_state` or to ``advance_base_step(scan_boundaries=True)``
    -- a ``lax.scan`` over boundaries cannot cross this seam. A driver gets its
    per-boundary traversals from :meth:`boundary_kick`, which is the seam nornax uses
    anyway.

    Parameters
    ----------
    softening : float
        Plummer softening ``1 / (r^2 + eps^2)^{3/2}``, shared by both halves.
    k_max : int
        Highest block-step rung. Levels run ``0 .. k_max``.
    theta : float
        INTRA-domain opening angle. Sets the force accuracy within a domain; it has
        no effect on momentum conservation, which is structural.
    cross_theta : float
        CROSS-domain opening angle. ``0.0`` makes every cross pair exact -- nothing is
        accepted, so all of them refine to leaf-leaf and are summed particle by
        particle. Above zero, accepted cross pairs go through M2L against the remote
        leaf's multipole, which is what collapses the halo import from the whole remote
        system to a surface.
    max_order : int
        Multipole expansion order ``p``.
    G : float
        Gravitational constant.
    leaf_size : int
        Particles per leaf for each domain's own tree.
    backend : str
        ``"jax"`` or ``"pallas"``; see
        :class:`~jaccpot.mutual.distributed.DistributedMutualConfig`.
    pallas_interpret : bool
        Run the Pallas kernels in interpret mode, which works without a GPU.
    ndev : Optional[int]
        Device count when ``mesh`` is ``None``. ``None`` uses every visible device.
    mesh : Any
        Device mesh. ``None`` builds one over ``ndev`` devices.
    caps : Any
        Intra-domain capacities (a ``MutualCapacities``). ``None`` derives heuristics.
    partitioner : str
        ``"rcb"`` (default) or ``"morton"``.
    cross_caps : Optional[dict]
        Cross-domain capacity overrides, passed straight through to
        :class:`~jaccpot.mutual.distributed.DistributedMutualConfig`: any of
        ``near_cap``, ``max_pair_queue``, ``recv_capacity``, ``max_req_leaves``,
        ``max_recv_leaves``, ``far_cap``, ``far_recv_capacity``, ``coarse_depth_cap``.
    validate_rung : bool
        Check that rungs lie in ``[0, k_max]``. Here the check is free rather than a
        device sync -- the rung is laid out on the host anyway -- so leaving it on
        costs nothing.

    Attributes
    ----------
    traced_boundary_weights : bool
        Read by nornax's ``supports_traced_level_weights``, and ``True`` here, so it
        walks the boundaries with a ``lax.scan`` over a weight table rather than
        unrolling ``2**k_max`` of them. Declared explicitly rather than left to
        nornax's signature probe, because the case is stronger on this lane than on
        one device: each boundary kick is a whole distributed program -- tree build,
        cross walk, halo exchange, reverse halo -- and under an outer ``jit`` or
        ``lax.scan`` over base steps the cached executable gets inlined anyway, so
        unrolling would put ``2**k_max`` copies of it in one graph.
        :meth:`boundary_kick` honours a traced ``level_weights``: only its LENGTH is
        read anywhere on this lane.

    Raises
    ------
    ValueError
        If ``backend`` is unsupported, ``k_max`` is negative, or an unknown key
        appears in ``cross_caps``.
    """

    traced_boundary_weights: bool = True

    #: Cross-domain capacity names `cross_caps` may set.
    _CROSS_CAP_KEYS = (
        "near_cap",
        "max_pair_queue",
        "recv_capacity",
        "max_req_leaves",
        "max_recv_leaves",
        "far_cap",
        "far_recv_capacity",
        "coarse_depth_cap",
    )

    def __init__(
        self,
        *,
        softening: float,
        k_max: int,
        theta: float = 0.5,
        cross_theta: float = 0.0,
        max_order: int = 4,
        G: float = 1.0,
        leaf_size: int = 32,
        backend: str = "jax",
        pallas_interpret: bool = False,
        ndev: Optional[int] = None,
        mesh: Any = None,
        caps: Any = None,
        partitioner: str = "rcb",
        cross_caps: Optional[dict] = None,
        validate_rung: bool = True,
    ) -> None:
        backend = str(backend).lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                "DistributedBlockStepFMM supports backend="
                f"{_SUPPORTED_BACKENDS!r}; got {backend!r}"
            )
        if int(k_max) < 0:
            raise ValueError(f"k_max must be >= 0; got {k_max!r}")
        extra = set(cross_caps or ()) - set(self._CROSS_CAP_KEYS)
        if extra:
            raise ValueError(
                f"unknown cross_caps keys {sorted(extra)}; known names are "
                f"{list(self._CROSS_CAP_KEYS)}"
            )

        from jaccpot.mutual.distributed import DistributedMutualConfig

        self.k_max = int(k_max)
        self.softening = float(softening)
        self.theta = float(theta)
        self.cross_theta = float(cross_theta)
        self.max_order = int(max_order)
        self.G = float(G)
        self.leaf_size = int(leaf_size)
        self.backend = backend
        self.validate_rung = bool(validate_rung)
        self.mesh = mesh
        self.ndev = ndev
        self.config = DistributedMutualConfig(
            leaf_size=int(leaf_size),
            theta=float(theta),
            cross_theta=float(cross_theta),
            order=int(max_order),
            k_max=int(k_max) if bool(validate_rung) else None,
            softening=float(softening),
            g=float(G),
            caps=caps,
            backend=backend,
            pallas_interpret=bool(pallas_interpret),
            partitioner=str(partitioner),
            **(cross_caps or {}),
        )
        self._evaluator: Any = None

    # -- partition lifetime -------------------------------------------------

    @property
    def evaluator(self) -> Any:
        """The prepared evaluator, or ``None`` before the first :meth:`prepare`."""
        return self._evaluator

    def prepare(
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
    ) -> Any:
        """Partition across the mesh, build the mapped program, and cache both.

        Host work, and not traceable: the domain assignment is a NumPy sort. Call it
        before any force method, and again when the partition should change -- see the
        class docstring for why that is not once per base step.

        The program is *built* here and compiled on its FIRST evaluation, lazily and
        per weighting mode -- the unweighted force takes three operands and a weighted
        one five, so they are different graphs. So this call is cheap and the first
        force is not: measured on 2 forced CPU devices at N = 128, prepare < 0.1 s, the
        first evaluation 20.5 s, and each later one ~8 s. That ~8 s is the whole point
        of holding the program; without it every evaluation pays the 20.5 s.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` positions to partition on.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.

        Returns
        -------
        Any
            The :class:`~jaccpot.mutual.distributed.DistributedMutualEvaluator`, also
            cached on the instance.
        """
        from jaccpot.mutual.distributed import make_distributed_mutual_evaluator

        self._evaluator = make_distributed_mutual_evaluator(
            positions,
            masses,
            config=self.config,
            mesh=self.mesh,
            ndev=self.ndev,
        )
        return self._evaluator

    def _require_evaluator(self) -> Any:
        """Return the prepared evaluator, or say what is missing.

        Returns
        -------
        Any
            The cached evaluator.

        Raises
        ------
        RuntimeError
            If :meth:`prepare` has not been called. Raised rather than partitioning
            implicitly, because an implicit partition would silently recompile on
            every call -- the exact cost this class exists to avoid.
        """
        if self._evaluator is None:
            raise RuntimeError(
                "call prepare(positions, masses) before evaluating a force. It is not "
                "done implicitly: partitioning per call would recompile the mapped "
                "program every time, which is what this class exists to avoid"
            )
        return self._evaluator

    def _weighted(
        self,
        positions: Array,
        masses: Array,
        rung: Optional[Array],
        level_weights: Optional[Array],
    ) -> Array:
        """One evaluation, checked for overflow, returned as a jax array.

        Parameters
        ----------
        positions : Array
            ``(n, 3)`` positions.
        masses : Array
            ``(n,)`` masses.
        rung : Optional[Array]
            ``(n,)`` per-particle rung, or ``None``.
        level_weights : Optional[Array]
            ``(k_max + 1,)`` weights, or ``None`` for the full force.

        Returns
        -------
        Array
            ``(n, 3)`` accelerations in the caller's order.

        Raises
        ------
        RuntimeError
            If any capacity on any device overflowed. **Raised, not reported**: the
            single-device lane raises on ``topology_overflow`` for the same reason,
            which is that a dropped canonical pair drops both its halves, so momentum
            stays exact and no norm on the result reveals it.
        """
        got = self._require_evaluator()(
            positions, masses, rung=rung, level_weights=level_weights
        )
        # `bool` exactly when the evaluator managed to read the flag -- it attempts
        # the read itself and leaves a traced scalar alone. So this raises on every
        # eager call and is skipped under trace, where an exception is not available
        # and a branch on the value is not either. A traced driver therefore does NOT
        # get this check: it has to evaluate once eagerly first, which is what
        # `prepare` plus a warm-up call gives it.
        if isinstance(got.overflow, bool) and got.overflow:
            raise RuntimeError(
                "a distributed mutual capacity overflowed -- "
                f"cross={np.asarray(got.cross_overflow).tolist()} "
                f"local={np.asarray(got.local_overflow).tolist()} "
                f"causes={np.asarray(got.local_overflow_causes).tolist()}. "
                "The force is wrong in a "
                "way no norm reveals: a dropped pair loses both its halves, so "
                "momentum stays exact. Raise the capacities (cross_caps / caps) and "
                "call prepare again"
            )
        return jnp.asarray(got.accelerations)

    def _validate_rung(self, rung: Array) -> Array:
        """Reject rungs outside ``[0, k_max]``.

        Unconditional here, unlike :meth:`BlockStepFMM._validate_rung`, and cheaper:
        that one pays a device-to-host sync to read the bound, while this lane lays
        the rung out on the host anyway, so the reduction is free. The check itself
        happens inside the evaluator, which is where the layout is; this only enforces
        that a rung was supplied at all.

        Parameters
        ----------
        rung : Array
            ``(n,)`` per-particle rung assignment.

        Returns
        -------
        Array
            ``rung`` as an array, unchanged.
        """
        return jnp.asarray(rung)

    # -- MutualForceModel contract -----------------------------------------

    def level_accelerations(
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        level: int,
        args: object = None,
    ) -> Array:
        """Return the level-``k`` antisymmetric acceleration for every particle.

        A one-hot weight row at ``level``, so it costs ONE traversal of the whole mesh
        rather than a masked sweep. The near field applies the exact per-particle
        predicate ``max(rung_i, rung_j) == level`` on both the intra- and
        cross-domain halves; the far field splits at cell granularity on both, taking
        each cell the rung of its most active particle. Both are partitions, so
        summing over ``k = 0 .. k_max`` reproduces the full acceleration.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
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
            If ``level`` is outside ``[0, k_max]``.
        """
        del args
        if not 0 <= int(level) <= self.k_max:
            raise ValueError(
                f"level must lie in [0, k_max={self.k_max}]; got {level!r}"
            )
        weights = (
            jnp.zeros((self.k_max + 1,), dtype=jnp.asarray(positions).dtype)
            .at[int(level)]
            .set(1.0)
        )
        return self._weighted(positions, masses, self._validate_rung(rung), weights)

    def total_accelerations(
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Optional[Int[Array, "n"]] = None,
        args: object = None,
    ) -> Array:
        """Return the full acceleration in a single traversal.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Optional[Int[Array, 'n']]
            Ignored -- the unweighted total does not depend on the rung assignment.
            Accepted so the signature matches the level-aware methods.
        args : object
            Ignored; present for the ``MutualForceModel`` contract.

        Returns
        -------
        Array
            ``(n, 3)`` full accelerations.
        """
        del args, rung
        return self._weighted(positions, masses, None, None)

    # -- fused boundary primitive ------------------------------------------

    def boundary_weights(
        self: "DistributedBlockStepFMM",
        active_floor: Any,
        dt_max: Any,
        half: Any = 1.0,
        *,
        dtype: Any = None,
    ) -> Array:
        """The ``(k_max + 1,)`` weight row for one sub-step boundary.

        Parameters
        ----------
        active_floor : Any
            Smallest level kicked at this boundary.
        dt_max : Any
            Base-step timestep.
        half : Any
            ``0.5`` at a synchronized end of the base step, ``1.0`` inside.
        dtype : Any
            Result dtype; ``None`` takes the model's working dtype.

        Returns
        -------
        Array
            ``(k_max + 1,)`` weight row.
        """
        return level_weights_from_floor(
            active_floor, self.k_max, dt_max, half=half, dtype=dtype
        )

    def boundary_kick(
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        active_floor: Any = None,
        dt_max: Any = None,
        half: Any = 1.0,
        level_weights: Optional[Float[Array, "levels"]] = None,
        args: object = None,
    ) -> Array:
        """Apply one sub-step boundary's kick in a single traversal of the mesh.

        Every level at or above ``active_floor`` is kicked with
        ``half * dt_max / 2**k``, with the weights pushed *into* the traversal so the
        whole mesh is walked once. That is what keeps a base step at ``n_sub + 1``
        evaluations instead of ``sum_s (active levels at s)``; for a distributed FMM,
        where each evaluation is a tree build, a cross walk and a halo exchange, the
        difference is the individual-timestep advantage itself.

        Momentum is untouched by the weighting on either half. On the near half the
        weight is one symmetric scalar multiplying the tensor both sides read; on the
        cross-domain far half it is applied once, on the evaluating device, to both
        directions of the batched M2L before either crosses the wire.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(N, 3)`` particle positions.
        velocities : Float[Array, 'n 3']
            ``(N, 3)`` velocities to kick.
        masses : Float[Array, 'n']
            ``(N,)`` particle masses.
        rung : Int[Array, 'n']
            ``(N,)`` per-particle block-step rung, in ``[0, k_max]``.
        active_floor : Any
            Smallest level kicked at this boundary.
        dt_max : Any
            Base-step timestep; level ``k`` is kicked with ``half * dt_max / 2**k``.
        half : Any
            ``0.5`` at the base step's synchronized ends, ``1.0`` inside.
        level_weights : Optional[Float[Array, 'levels']]
            The ``(k_max + 1,)`` weight vector supplied directly. Takes precedence
            over ``active_floor``/``dt_max``/``half``, which are then ignored.
        args : object
            Unused; present for the ``MutualForceModel`` protocol's signature.

        Returns
        -------
        Array
            Updated velocities.

        Raises
        ------
        ValueError
            If neither ``level_weights`` nor both ``active_floor`` and ``dt_max`` are
            given, or ``level_weights`` has the wrong length for ``k_max``.
        """
        del args
        dtype = jnp.asarray(positions).dtype
        if level_weights is None:
            if dt_max is None or active_floor is None:
                raise ValueError(
                    "boundary_kick needs either level_weights, or both "
                    "active_floor and dt_max"
                )
            level_weights = level_weights_from_floor(
                active_floor, self.k_max, dt_max, half=half, dtype=dtype
            )
        else:
            level_weights = jnp.asarray(level_weights, dtype=dtype)
            if int(level_weights.shape[-1]) != self.k_max + 1:
                raise ValueError(
                    f"level_weights must have {self.k_max + 1} entries for "
                    f"k_max={self.k_max}; got shape {tuple(level_weights.shape)}"
                )
        delta_v = self._weighted(
            positions, masses, self._validate_rung(rung), level_weights
        )
        return jnp.asarray(velocities) + delta_v

    def boundary_kick_at(
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        s: int,
        dt_max: float,
        args: object = None,
    ) -> Array:
        """:meth:`boundary_kick` addressed by sub-step boundary index ``s``.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` particle positions.
        velocities : Float[Array, 'n 3']
            ``(n, 3)`` particle velocities, the quantity being kicked.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
            ``(n,)`` per-particle rung assignment.
        s : int
            Sub-step boundary index, ``0 .. n_sub``. Must be concrete.
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
        self: "DistributedBlockStepFMM",
        positions: Float[Array, "n 3"],
        velocities: Float[Array, "n 3"],
        masses: Float[Array, "n"],
        *,
        rung: Int[Array, "n"],
        dt_max: float,
    ) -> Tuple[Array, Array, Array]:
        """Run one full base step, at one traversal of the mesh per boundary.

        The recursively-symmetric palindrome of Farr & Bertschinger (2007): a kick at
        every boundary ``s = 0 .. n_sub`` (half at the synchronized ends, full inside)
        with a drift of ``dt_min`` between consecutive boundaries. The rung assignment
        is held fixed for the whole step, which is what makes the map symplectic and
        time-reversible.

        There is no ``scan_boundaries`` counterpart here: this lane's readout is a
        NumPy scatter back to input order, so the boundary loop cannot be traced. The
        loop is therefore always the unrolled Python one, which on this lane costs
        nothing extra -- the mapped program is compiled once by :meth:`prepare` and
        every boundary reuses it.

        Returns ``(positions, velocities, acceleration)`` with the acceleration the
        full field at the end-of-step positions, ready to seed the next base step's
        rung assignment. It is a separate evaluation on purpose: a boundary kick
        returns *weighted* levels, and the unweighted total cannot be recovered from
        them.

        Parameters
        ----------
        positions : Float[Array, 'n 3']
            ``(n, 3)`` positions at the start of the base step.
        velocities : Float[Array, 'n 3']
            ``(n, 3)`` velocities at the start of the base step.
        masses : Float[Array, 'n']
            ``(n,)`` particle masses.
        rung : Int[Array, 'n']
            ``(n,)`` per-particle rung assignment, held fixed for the whole step.
        dt_max : float
            Base-step size, the time step of level ``0``.

        Returns
        -------
        Tuple[Array, Array, Array]
            ``(positions, velocities, acceleration)`` at the end of the step.
        """
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
            vel = vel + self._weighted(pos, masses, rung, weights)
            if s < steps:
                pos = pos + dt_min * vel

        acc = self._weighted(pos, masses, None, None)
        return pos, vel, acc

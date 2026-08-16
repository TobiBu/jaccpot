"""PolicyMixin: fmm_policy methods extracted from the FastMultipoleMethod
god-class (Phase 2d mixin split). Methods are verbatim (self unchanged); the
engine class inherits this mixin. Sibling of _fmm_impl at runtime level.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterator, Optional

import jax.numpy as jnp
from beartype.typing import Callable
from jaxtyping import Array
from yggdrax.interactions import (
    DualTreeRetryEvent,
    DualTreeTraversalConfig,
    MACType,
    build_octree_native_far_pairs,
    build_octree_native_neighbor_lists,
)
from yggdrax.tree import Tree

from jaccpot.config import MACTypeInput
from jaccpot.upward.tree_expansions import TreeUpwardData

from ._adaptive_policy import (
    AdaptivePolicyState,
    build_adaptive_policy_state,
    compute_node_force_scale_from_sorted_acc,
    estimate_particle_force_scale,
    per_node_effective_theta,
    per_node_mac_radius,
    source_error_proxy_by_order_from_multipoles,
)
from ._octree_adapter import build_octree_execution_data_with_status
from .fmm_caches import _contains_tracer
from .fmm_state import (
    FMMPreparedState,
    _build_octree_downward_artifacts,
    _build_octree_upward_artifacts,
    _finalize_octree_downward_artifacts,
    _prepared_state_octree_upward_payload,
    _prepared_state_upward_payload,
    _PrepareStateTreeUpwardArtifacts,
)
from .kernels.core import _build_nearfield_interop_data

if TYPE_CHECKING:  # pragma: no cover - annotations only, no runtime import
    # The mixins annotate `self` as the engine they are mixed into, which lives in
    # `_fmm_impl` and imports *them* -- so this must stay under TYPE_CHECKING or it
    # would form the cycle ARCHITECTURE §8 forbids. Before this block the names were
    # dangling: `typing.get_type_hints` raised NameError on every mixin method, so the
    # annotations documented an intent no tool could check.
    from ._fmm_impl import FMMEngine


#: Opening angle the force-scale prepass traversal defaults to. See
#: :meth:`PolicyMixin._force_scale_prepass_theta` for the measurement behind it.
_DEFAULT_FORCE_SCALE_PREPASS_THETA = 0.5


def _far_pair_arrays_for_fb_prepass(
    *,
    interactions: object,
    compact_far_pairs: object,
) -> tuple[Optional[Array], Optional[Array]]:
    """Return flat (sources, targets) far-pair arrays for the eq (16b) estimator.

    The estimator reads the far list as one flat COO pair array with ``-1`` for
    inactive entries, which is exactly the node interaction list's layout -- and
    also the streamed lane's ``CompactTaggedFarPairs``, once its padding is
    handled. That matters because the streamed/large-N lane returns compact far
    pairs and *no* node interaction list, so requiring the latter shut eq (16b)
    out of the only lane that reaches 10^6.

    ``far_pair_count`` is honoured explicitly rather than trusted to be encoded
    as ``-1`` padding: yggdrax documents the arrays as "fixed-capacity padded"
    without specifying the fill value, and a fill of ``0`` would read as the pair
    ``(node 0, node 0)``. Node 0 is the root, so every spurious entry would add
    mass to ``own[root]`` and the downward accumulation would push it into
    *every* node -- inflating ``f_b``, loosening ``eps * s``, and making the
    solver faster and wronger with nothing to show it.

    Parameters
    ----------
    interactions : object
        Node interaction list, when the lane produced one. Preferred: its arrays
        already use the ``-1`` convention, so they pass through untouched.
    compact_far_pairs : object
        Streamed-lane compact far pairs, used when there is no interaction list.
        Its padding is rewritten to ``-1`` here.

    Returns
    -------
    tuple[Optional[Array], Optional[Array]]
        ``(sources, targets)`` as flat COO arrays with ``-1`` marking inactive
        entries, or ``(None, None)`` when neither input was supplied.
    """

    if interactions is not None:
        return interactions.sources, interactions.targets
    if compact_far_pairs is None:
        return None, None
    sources = jnp.asarray(compact_far_pairs.sources)
    targets = jnp.asarray(compact_far_pairs.targets)
    count = getattr(compact_far_pairs, "far_pair_count", None)
    if count is None:
        return sources, targets
    live = jnp.arange(sources.shape[0]) < jnp.asarray(count).reshape(())
    sentinel = jnp.asarray(-1, dtype=sources.dtype)
    return jnp.where(live, sources, sentinel), jnp.where(live, targets, sentinel)


class PolicyMixin:
    def _solidfmm_basis_mode(self: "FMMEngine") -> str:
        """Return active solidfmm coefficient family ('complex' or 'real').

        Reads the basis object's own name rather than ``expansion_basis``: the
        two agree in normal use, but this is the one the coefficient layout
        actually follows.

        Returns
        -------
        str
            ``"real"`` or ``"complex"``; anything unrecognized reads as complex.
        """
        basis_obj = self.basis_impl
        name = str(getattr(basis_obj, "name", "")).strip().lower()
        if name == "real":
            return "real"
        return "complex"

    def _compute_node_force_scale_from_sorted_acc(
        self: "FMMEngine",
        *,
        tree: Tree,
        accelerations_sorted: Array,
        reduction: str = "max",
    ) -> Array:
        """Estimate per-node force scales from sorted per-particle accelerations.

        Parameters
        ----------
        tree : Tree
            Built tree, supplying the per-node particle ranges.
        accelerations_sorted : Array
            Per-particle accelerations ``[N, 3]`` in Morton order.
        reduction : str
            ``"min"`` or ``"max"`` over each node's particles. Not a tuning
            knob -- eq (16a) wants ``min``; see
            :meth:`_force_scale_reduction_mode`.

        Returns
        -------
        Array
            One force scale per node.
        """

        return compute_node_force_scale_from_sorted_acc(
            tree=tree,
            accelerations_sorted=accelerations_sorted,
            reduction=reduction,
        )

    def _record_force_scale_from_evaluation(
        self: "FMMEngine",
        *,
        state: FMMPreparedState,
        evaluation: object,
        full_evaluation: bool,
    ) -> None:
        """Cache node force scales from a completed full-order evaluation.

        This is what makes ``mac_force_scale_mode='prev'`` mean anything: it is the
        only writer of ``_last_force_scale_nodes`` outside the prepass, so without
        it 'prev' silently fell back to a unit force scale on the non-paper path
        and to a prepass on every call on the paper path.

        The accelerations of the step just evaluated are a strictly better estimate
        of Dehnen's ``a_b`` than the low-order prepass is -- they *are* ``a_b``, one
        step stale -- which is the reuse §5.4 licenses. Skipped when tracing (an
        instance attribute must never capture a tracer), during a prepass (the
        caller reduces that result itself), and for target-subset evaluations,
        whose accelerations do not cover every node.

        Parameters
        ----------
        state : FMMPreparedState
            The state just evaluated; supplies the tree the reduction runs over.
        evaluation : object
            The evaluation result. A tuple's first element is taken as the
            accelerations, so this accepts every return shape the evaluate
            entry points produce.
        full_evaluation : bool
            Whether every particle was evaluated. A target subset does not cover
            every node, so its accelerations must not be cached.
        """

        if not full_evaluation or self._in_force_scale_prepass:
            return
        if not self._uses_paper_style_force_scale():
            return
        if self._uses_fb_force_scale():
            # eq (16b)'s scale is `min_b f_b`, not `min_b |a_b|`, and the two are
            # different quantities -- f_b is the cancellation-free *sum of pairwise
            # magnitudes*, so it is strictly larger and does not vanish. Recording
            # accelerations here would silently replace the f_b cache with a (16a)
            # scale after the first evaluation, which is exactly the back-door
            # failure `force_scale_nodes=` was added to remove: an injected f_b used
            # to survive exactly one prepare_state, so a prepare/evaluate loop
            # measured (16a) while believing it measured (16b). f_b depends only on
            # positions and masses, so there is nothing an evaluation can contribute
            # to it anyway.
            return
        acc_sorted = evaluation[0] if isinstance(evaluation, tuple) else evaluation
        if _contains_tracer((acc_sorted, state.tree)):
            return
        self._last_force_scale_nodes = self._compute_node_force_scale_from_sorted_acc(
            tree=state.tree,
            accelerations_sorted=jnp.asarray(acc_sorted),
            reduction=self._force_scale_reduction_mode(),
        )

    def _source_error_proxy_by_order_from_multipoles(
        self: "FMMEngine",
        *,
        multipole_packed: Array,
        p_gears: tuple[int, ...],
    ) -> Array:
        """Compute a conservative per-node residual proxy for each candidate order.

        Parameters
        ----------
        multipole_packed : Array
            Packed multipole coefficients per node.
        p_gears : tuple[int, ...]
            Candidate expansion orders to score. Must be non-empty.

        Returns
        -------
        Array
            Residual proxy per ``(node, gear)``. Conservative by construction --
            it bounds the truncation residual from above, so an order it accepts
            is safe and one it rejects may still have been fine.
        """

        return source_error_proxy_by_order_from_multipoles(
            multipole_packed=multipole_packed,
            p_gears=p_gears,
        )

    def _adaptive_error_model_code(self: "FMMEngine") -> int:
        """Return the integer policy code for the active adaptive error model.

        Returns
        -------
        int
            ``2`` for ``"dehnen_paper"``, ``1`` for ``"dehnen_degree"``, ``0``
            otherwise. The codes are what the traversal consumes, so they are an
            interface, not an implementation detail.
        """

        if self.adaptive_error_model == "dehnen_paper":
            return 2
        if self.adaptive_error_model == "dehnen_degree":
            return 1
        return 0

    def _uses_dehnen_error_policy(self: "FMMEngine") -> bool:
        """Return whether the solver evaluates the Dehnen error criterion at all.

        True for both ``dehnen_error`` (exact, evaluated pair-by-pair through a
        solver-owned ``pair_policy``) and ``dehnen_theta`` (the same criterion
        folded into one opening angle per node). The two share everything that
        feeds the criterion -- the min-reduced force scale, the mandatory
        ``adaptive_eps``, the low-order prepass, the ``"dehnen"`` base MAC and the
        paper error-model code. They diverge at exactly one place: whether a pair
        policy is installed. See :meth:`_uses_per_node_effective_theta`.

        Returns
        -------
        bool
            ``True`` for ``mac_type`` in ``("dehnen_error", "dehnen_theta")``.
            This set must stay in step with :meth:`_mac_type_for_traversal`.
        """

        return str(self.mac_type) in ("dehnen_error", "dehnen_theta")

    def _uses_per_node_effective_theta(self: "FMMEngine") -> bool:
        """Return whether the MAC is folded into a per-node opening angle.

        ``dehnen_theta`` evaluates the same Dehnen criterion as ``dehnen_error``,
        but collapses it into one opening angle per node and feeds that to the
        traversal as rescaled ``geometry.radius`` values, so it installs no
        ``pair_policy`` and none of the fast-lane vetoes trigger.

        **REFUTED -- do not use for production.** Measured against the exact
        criterion at N=4096/p=8 (`bench/validation/per_node_theta_fidelity.py`,
        `bench/results/validation/theta_fidelity_p8.json`): 12-9300x worse error at
        1.35-15x *more* interaction work, with a p99.99 of 2.3e+02 on bulge+halo.
        Retained only so the negative result stays reproducible; selecting it warns.

        The obstruction is structural, not a tuning failure. eq (16a) accepts when
        ``r**(p+2)`` exceeds a *product* of a source term and a sink term, while the
        lane test is a *sum* ``e_A + e_B <= theta_g r``. A sum cannot represent a
        product, so a per-node extent is either tight on average and unsound on the
        tails (this mode) or sound and empty (see
        :func:`~jaccpot.runtime._adaptive_policy.per_node_conservative_extent`,
        which recovers <=0.6% of the exact criterion's far pairs at its optimum).
        Carrying this criterion into the fast lanes needs pair-policy support in the
        lanes themselves.

        Returns
        -------
        bool
            ``True`` only for ``mac_type == "dehnen_theta"``. Its sibling
            ``dehnen_error`` shares the criterion but installs a pair policy
            instead.
        """

        return str(self.mac_type) == "dehnen_theta"

    def _uses_dehnen_paper_error_model(self: "FMMEngine") -> bool:
        """Return whether the active adaptive error model is the paper estimator.

        Returns
        -------
        bool
            ``True`` for ``adaptive_error_model == "dehnen_paper"`` -- the eq (15)
            estimator, which also flips the force-scale reduction to ``min``.
        """

        return self.adaptive_error_model == "dehnen_paper"

    def _uses_paper_style_traversal_policy(self: "FMMEngine") -> bool:
        """Return whether traversal should use the paper-style error policy.

        Returns
        -------
        bool
            ``True`` when either the paper error model or a Dehnen mass-dependent
            MAC is active -- the two reach the traversal the same way.
        """

        return self._uses_dehnen_paper_error_model() or self._uses_dehnen_error_policy()

    def _traversal_policy_error_model_code(self: "FMMEngine") -> int:
        """Return the policy error model code used during traversal.

        Returns
        -------
        int
            ``2`` whenever a Dehnen mass-dependent MAC is active, since that
            criterion needs the paper estimator regardless of
            ``adaptive_error_model``; otherwise
            :meth:`_adaptive_error_model_code`.
        """

        if self._uses_dehnen_error_policy():
            return 2
        return self._adaptive_error_model_code()

    def _force_scale_reduction_mode(self: "FMMEngine") -> str:
        """Return the node reduction mode used for adaptive force scales.

        Returns
        -------
        str
            ``"min"`` under the paper estimator, ``"max"`` otherwise. eq (16a)
            takes ``min_b |a_b|`` over the node, so the reduction is part of the
            criterion rather than a tuning choice.
        """

        return "min" if self._uses_dehnen_paper_error_model() else "max"

    def _uses_fb_force_scale(self: "FMMEngine") -> bool:
        """Return whether the force scale is eq (16b)'s ``f_b`` rather than ``|a_b|``.

        eq (16b) is eq (16a) with ``min_b f_b`` on the right-hand side instead of
        ``min_b |a_b|``. The criterion, the traversal and the eq (15) error
        estimator are untouched, so the whole of (16b) is a different per-node
        force scale -- which is why it needs no traversal work, only a different
        prepass.

        Returns
        -------
        bool
            ``True`` for ``mac_force_scale_mode`` in
            ``("paper_fb", "paper_fb_cached")``.
        """

        return str(self.mac_force_scale_mode) in ("paper_fb", "paper_fb_cached")

    def _force_scale_prepass_theta(self: "FMMEngine") -> float:
        """Return the opening angle the force-scale prepass traversal should use.

        The prepass runs a *geometric* traversal, so its own theta decides how much
        of the scale comes from the exact near field and how much from the monopole
        far-field approximation. That is easy to get wrong, because paper mode pins
        the solver's ``theta`` at 1.0 on the grounds that it does not gate
        acceptance -- true for the criterion, false for the prepass underneath it.

        Measured effect on the ``f_b`` estimate against the exact O(N^2) sum
        (Plummer, N=4096, ratio estimate/exact):

        ===== ========== =========
        theta median     minimum
        ===== ========== =========
        0.3   1.000      1.000
        0.5   0.997      0.911
        0.7   0.929      0.645
        1.0   0.736      0.334
        ===== ========== =========

        So the default is 0.5, near Dehnen §5.2's ``theta_crit ~ 0.46``, where the
        estimate is essentially exact. ``mac_force_scale_prepass_theta`` overrides
        it.

        Returns
        -------
        float
            The override when one is set, otherwise
            ``_DEFAULT_FORCE_SCALE_PREPASS_THETA``. Independent of the solver's
            own ``theta`` -- which is the point.
        """

        override = getattr(self, "mac_force_scale_prepass_theta", None)
        if override is not None:
            return float(override)
        return _DEFAULT_FORCE_SCALE_PREPASS_THETA

    def _uses_paper_style_force_scale(self: "FMMEngine") -> bool:
        """Return whether prepare_state needs paper-style force-scale handling.

        Returns
        -------
        bool
            ``True`` when adaptive order is on or a paper-style traversal policy
            is active -- either way a force scale has to be produced.
        """

        return self.adaptive_order or self._uses_paper_style_traversal_policy()

    @staticmethod
    def _mac_type_for_traversal(mac_type: MACTypeInput) -> MACType:
        """Map a caller-facing MAC type onto the one yggdrax's traversal accepts.

        ``"dehnen_error"`` and ``"dehnen_theta"`` are jaccpot-level policies -- the
        Dehnen (2014) §5 mass-dependent MAC -- layered on the geometric
        ``"dehnen"`` test. yggdrax has never heard of either: its ``MACType`` is
        ``Literal["bh", "engblom", "dehnen"]`` and its traversal raises
        ``ValueError("Unknown mac_type: ...")`` for anything else. So every path
        that reaches the traversal must come through here first.

        **Both** names map, and the set here must stay in step with
        :meth:`_uses_dehnen_error_policy`. They differ only in *how* the criterion
        is applied -- ``dehnen_error`` installs a solver-owned pair policy,
        ``dehnen_theta`` folds it into one opening angle per node (rescaled
        ``geometry.radius``) -- and neither changes the geometric test underneath,
        so both reduce to ``"dehnen"`` here. Mapping only one of them is not a
        narrower translation but a crash: yggdrax's runtime type check rejects the
        unmapped literal before the traversal starts.

        A ``staticmethod`` on purpose: it maps a *value*, so it also works for an
        explicitly-passed override rather than only for ``self.mac_type``. That is
        what :meth:`_base_mac_type` needs, and what
        ``fmm_sweeps.prepare_downward_sweep`` needs for its ``mac_type=`` argument.

        Parameters
        ----------
        mac_type : MACTypeInput
            What the caller asked for, including the two jaccpot-level policies.

        Returns
        -------
        MACType
            The geometric criterion to hand the traversal.
        """
        return (
            "dehnen" if str(mac_type) in ("dehnen_error", "dehnen_theta") else mac_type
        )

    def _base_mac_type(self: "FMMEngine") -> MACType:
        """Return the Yggdrax-facing geometric MAC for the active solver mode.

        Returns
        -------
        MACType
            ``self.mac_type`` mapped through :meth:`_mac_type_for_traversal`, so
            the jaccpot-level policies collapse to ``"dehnen"``.
        """

        return self._mac_type_for_traversal(self.mac_type)

    def _policy_orders_for_prepare_state(
        self: "FMMEngine", *, max_order: int
    ) -> tuple[int, ...]:
        """Return candidate orders used to build adaptive traversal policy state.

        The paper estimator without adaptive order runs at a single fixed order,
        so offering it the full gear set would build policy state for orders that
        can never be selected.

        Parameters
        ----------
        max_order : int
            The order this call will actually run at.

        Returns
        -------
        tuple[int, ...]
            ``(max_order,)`` for the paper estimator at fixed order, otherwise
            ``self.p_gears``.
        """

        if (not self.adaptive_order) and self._uses_dehnen_paper_error_model():
            return (int(max_order),)
        return self.p_gears

    def _build_adaptive_policy_state(
        self: "FMMEngine",
        *,
        upward: TreeUpwardData,
        tree: Tree,
        positions_sorted: Array,
        p_gears: tuple[int, ...],
        force_scale_nodes: Optional[Array],
        eps: Array,
        theta: Array,
        error_model_code: Array,
        dehnen_geometry_mode: str,
    ) -> AdaptivePolicyState:
        """Build the solver-owned adaptive policy state from upward data.

        The state the traversal consults per pair. "Solver-owned" is the point:
        yggdrax evaluates the geometric MAC, and everything mass-dependent about
        Dehnen §5 lives in this object rather than in the traversal.

        Parameters
        ----------
        upward : TreeUpwardData
            Upward-sweep result; supplies the multipoles the error proxy reads.
        tree : Tree
            Built tree.
        positions_sorted : Array
            Particle positions ``[N, 3]`` in Morton order.
        p_gears : tuple[int, ...]
            Candidate orders, from :meth:`_policy_orders_for_prepare_state`.
        force_scale_nodes : Optional[Array]
            Per-node force scale; ``None`` leaves the policy to fall back to a
            unit scale, which makes eq (16a) vacuous rather than wrong.
        eps : Array
            Relative force-accuracy target of eq (16a).
        theta : Array
            Opening angle entering the criterion.
        error_model_code : Array
            Which estimator to evaluate; see
            :meth:`_traversal_policy_error_model_code`.
        dehnen_geometry_mode : str
            How node centres and radii are measured.

        Returns
        -------
        AdaptivePolicyState
            The policy state, ready to hand to the traversal.
        """

        return build_adaptive_policy_state(
            gravitational_constant=float(self.G),
            mac_theta_max=float(getattr(self, "mac_theta_max", 1.0)),
            upward=upward,
            tree=tree,
            positions_sorted=positions_sorted,
            p_gears=p_gears,
            force_scale_nodes=force_scale_nodes,
            eps=eps,
            theta=theta,
            error_model_code=error_model_code,
            dehnen_geometry_mode=dehnen_geometry_mode,
        )

    def _apply_per_node_effective_theta(
        self: "FMMEngine",
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        force_scale_nodes: Optional[Array],
        max_order: int,
        theta_val: float,
    ) -> _PrepareStateTreeUpwardArtifacts:
        """Fold the Dehnen criterion into ``geometry.radius`` as per-node angles.

        Returns ``tree_artifacts`` with the upward geometry's ``radius`` replaced by
        ``rho_i * theta_val / theta_i``, which makes the traversal's own
        ``(e_t + e_s)**2 <= theta**2 d**2`` test algebraically equal to
        ``rho_t/theta_t + rho_s/theta_s <= d``. Everything downstream -- generic
        walk, split build, streamed build, treecode, Pallas -- then carries the
        criterion with no pair policy and no lane veto.

        ``theta_val`` cancels out of acceptance, so its value does not matter here;
        it is threaded through only because the traversal compares against it.

        Parameters
        ----------
        tree_artifacts : _PrepareStateTreeUpwardArtifacts
            Tree and upward artifacts whose geometry radii are rescaled.
        force_scale_nodes : Optional[Array]
            Per-node force scale. Required -- without it there is no criterion to
            fold, so this raises rather than silently returning the artifacts
            unchanged.
        max_order : int
            Expansion order entering the per-node angle.
        theta_val : float
            The angle the traversal will compare against. Cancels out of
            acceptance; see above.

        Returns
        -------
        _PrepareStateTreeUpwardArtifacts
            The artifacts with ``geometry.radius`` replaced by the rescaled
            radii. A new value -- the input is not mutated.

        Raises
        ------
        ValueError
            If ``force_scale_nodes`` is ``None``.
        """

        if force_scale_nodes is None:
            raise ValueError(
                "mac_type='dehnen_theta' requires a per-node force scale; "
                "prepare_state should have produced one via the paper prepass"
            )
        order = int(tree_artifacts.upward.multipoles.order)
        policy_state = self._build_adaptive_policy_state(
            upward=tree_artifacts.upward,
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            p_gears=(order,),
            force_scale_nodes=force_scale_nodes,
            eps=jnp.asarray(float(self.adaptive_eps)),
            theta=jnp.asarray(float(theta_val)),
            error_model_code=jnp.asarray(
                self._traversal_policy_error_model_code(), dtype=jnp.int32
            ),
            dehnen_geometry_mode=self.dehnen_geometry_mode,
        )
        theta_nodes = per_node_effective_theta(
            source_power=policy_state.source_dehnen_power,
            radius_bound=policy_state.source_radius_bound,
            force_scale=force_scale_nodes,
            masked_binomial=policy_state.dehnen_binomial_masked_by_order[0],
            exponent=policy_state.dehnen_exponent_by_order[0],
            order=order,
            eps=float(self.adaptive_eps),
            gravitational_constant=float(self.G),
            theta_max=float(getattr(self, "mac_theta_max", 1.0)),
        )
        scaled_radius = per_node_mac_radius(
            radius_bound=policy_state.source_radius_bound,
            theta_nodes=theta_nodes,
            theta_global=float(theta_val),
        )
        self._recent_effective_theta_nodes = theta_nodes
        upward = tree_artifacts.upward
        return tree_artifacts._replace(
            upward=upward._replace(
                geometry=upward.geometry._replace(radius=scaled_radius)
            )
        )

    @contextlib.contextmanager
    def _force_scale_prepass_scope(
        self: "FMMEngine", *, low_order: int
    ) -> Iterator[None]:
        """Run a force-scale prepass with solver state restored on the way out.

        A prepass is an inner solve on the same particles, so it necessarily
        overwrites solver attributes that the enclosing ``prepare_state`` still
        needs. Two kinds have to be restored:

        - the knobs the prepass deliberately overrides (a single low order, no
          adaptive order, the geometric MAC), and
        - the bookkeeping the inner solve updates as a side effect. That includes
          ``_topology_reuse_entry``: the non-paper prepass re-enters
          ``prepare_state``, whose reuse block increments ``reuse_count``, so
          without this every outer call consumed two slots of the ``rebuild_every``
          cadence and the tree was rebuilt twice as often as requested.

        Both prepass branches share this scope so they cannot drift apart again.

        Parameters
        ----------
        low_order : int
            The single order the prepass runs at. The prepass only needs a scale,
            not an accurate force, so this is deliberately far below the solver's
            own order.

        Yields
        ------
        None
            A bare scope. Everything it manages is solver attributes, restored on
            exit whether or not the body raised.
        """

        saved_p_gears = self.p_gears
        saved_adaptive_order = self.adaptive_order
        saved_adaptive_error_model = self.adaptive_error_model
        saved_mac_type = self.mac_type
        saved_recent_counts = self._recent_far_pairs_by_gear_counts
        saved_topology_reuse_entry = self._topology_reuse_entry
        saved_recent_topology_reused = bool(self._recent_topology_reused)
        self._in_force_scale_prepass = True
        try:
            self.p_gears = (int(low_order),)
            yield
        finally:
            self.p_gears = saved_p_gears
            self.adaptive_order = saved_adaptive_order
            self.adaptive_error_model = saved_adaptive_error_model
            self.mac_type = saved_mac_type
            self._recent_far_pairs_by_gear_counts = saved_recent_counts
            self._topology_reuse_entry = saved_topology_reuse_entry
            self._recent_topology_reused = saved_recent_topology_reused
            self._in_force_scale_prepass = False

    def _compute_force_scale_fb_prepass_from_tree_artifacts(
        self: "FMMEngine",
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        upward_center_mode: str,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
    ) -> Array:
        """Estimate eq (16b)'s ``f_b`` for every particle, in sorted order.

        Cheaper than the eq (16a) prepass, not just different: that one runs a
        whole low-order FMM *evaluation* (upward sweep, M2L, L2L, L2P) to get
        ``|a_b|``, whereas ``f_b`` is a sum of pairwise force magnitudes and needs
        only the traversal's pair partition -- near pairs get an exact scalar sum,
        far pairs a monopole. No expansions are built or applied at any order.

        The traversal runs with the geometric MAC at
        :meth:`_force_scale_prepass_theta` and with no pair policy, so it cannot
        recurse into the criterion it is computing the scale for.

        Parameters
        ----------
        tree_artifacts : _PrepareStateTreeUpwardArtifacts
            Tree and upward artifacts the prepass traverses.
        upward_center_mode : str
            Centre convention for the prepass upward data.
        runtime_traversal_config : Optional[DualTreeTraversalConfig]
            Traversal override for the prepass; ``None`` takes the default.
        runtime_m2l_chunk_size : Optional[int]
            M2L chunk size. Threaded through for shape consistency -- the prepass
            applies no expansions.
        runtime_l2l_chunk_size : Optional[int]
            L2L chunk size, likewise.
        grouped_interactions : bool
            Whether the prepass traversal groups interactions.
        farfield_mode : str
            Far-field mode for the prepass traversal.
        record_retry : Callable[[DualTreeRetryEvent], None]
            Sink for traversal retry events, so a retry inside the prepass is
            still visible in the outer call's diagnostics.
        refine_local_val : bool
            Local refinement setting for the prepass build.
        max_refine_levels_val : int
            Refinement level cap for the prepass build.
        aspect_threshold_val : float
            Aspect threshold for the prepass build.

        Returns
        -------
        Array
            ``f_b`` per particle ``[N]``, in Morton order. Strictly larger than
            ``|a_b|`` and never vanishing, being a cancellation-free sum of
            magnitudes -- which is why it must not be overwritten by a recorded
            acceleration; see :meth:`_record_force_scale_from_evaluation`.

        Raises
        ------
        RuntimeError
            If the prepass traversal produced neither an interaction list nor
            compact far pairs, leaving nothing for the estimator to read.
        """

        prepass_theta = self._force_scale_prepass_theta()
        with self._force_scale_prepass_scope(
            low_order=int(min(self.p_gears)) if self.p_gears else 0
        ):
            self.adaptive_order = False
            self.adaptive_error_model = "tail_proxy"
            self.mac_type = "dehnen"
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=tree_artifacts,
                force_scale_nodes=None,
                upward_center_mode=upward_center_mode,
                theta_val=prepass_theta,
                mac_type_val=self.mac_type,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=False,
                # The streamed lane builds compact far pairs and then discards them
                # unless something asks; this is that ask. Without it the prepass
                # gets the near list only, and a near-only `f_b` captures 53-66% of
                # the true value once the far field matters -- a silent under-estimate
                # rather than a visible failure.
                retain_compact_far_pairs=True,
            )

        interactions = dual_downward_artifacts.interactions
        neighbor_list = dual_downward_artifacts.neighbor_list
        compact_far_pairs = getattr(dual_downward_artifacts, "compact_far_pairs", None)
        far_sources, far_targets = _far_pair_arrays_for_fb_prepass(
            interactions=interactions,
            compact_far_pairs=compact_far_pairs,
        )
        if far_sources is None or neighbor_list is None:
            raise RuntimeError(
                "the f_b force-scale prepass needs both the far pair list and the "
                "near neighbour list; the traversal returned "
                f"interactions={interactions is not None}, "
                f"compact_far_pairs={compact_far_pairs is not None}, "
                f"neighbors={neighbor_list is not None}. Without both, the "
                "near/far partition is incomplete and f_b would be silently "
                "under-counted rather than approximated."
            )
        geometry = tree_artifacts.upward.geometry
        return estimate_particle_force_scale(
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            node_centers=geometry.center,
            node_radii=geometry.radius,
            interaction_sources=far_sources,
            interaction_targets=far_targets,
            neighbor_offsets=neighbor_list.offsets,
            neighbor_counts=neighbor_list.counts,
            neighbor_leaf_indices=neighbor_list.leaf_indices,
            neighbor_indices=neighbor_list.neighbors,
            max_leaf_size=int(tree_artifacts.leaf_cap),
            softening=float(self.softening),
            gravitational_constant=float(self.G),
            far_center_inflation=float(
                getattr(self, "mac_force_scale_fb_inflation", 1.0)
            ),
        )

    def _compute_force_scale_paper_prepass_from_tree_artifacts(
        self: "FMMEngine",
        *,
        tree_artifacts: _PrepareStateTreeUpwardArtifacts,
        low_order: int,
        theta_val: float,
        upward_center_mode: str,
        runtime_traversal_config: Optional[DualTreeTraversalConfig],
        runtime_m2l_chunk_size: Optional[int],
        runtime_l2l_chunk_size: Optional[int],
        grouped_interactions: bool,
        farfield_mode: str,
        record_retry: Callable[[DualTreeRetryEvent], None],
        refine_local_val: bool,
        max_refine_levels_val: int,
        aspect_threshold_val: float,
    ) -> Array:
        """Compute paper-mode force scales via a low-order prepass on the current tree.

        The eq (16a) counterpart of
        :meth:`_compute_force_scale_fb_prepass_from_tree_artifacts`, and the more
        expensive of the two: it runs a complete low-order FMM *evaluation* --
        upward, M2L, L2L, L2P -- because ``|a_b|`` is an acceleration and there is
        no cheaper way to get one. The f_b prepass needs only the pair partition.

        Parameters
        ----------
        tree_artifacts : _PrepareStateTreeUpwardArtifacts
            Tree and upward artifacts the prepass evaluates on.
        low_order : int
            Order for the prepass evaluation. Low on purpose -- the result is a
            scale, not a force.
        theta_val : float
            Opening angle for the prepass evaluation.
        upward_center_mode : str
            Centre convention for the prepass upward data.
        runtime_traversal_config : Optional[DualTreeTraversalConfig]
            Traversal override; ``None`` takes the default.
        runtime_m2l_chunk_size : Optional[int]
            M2L chunk size for the prepass evaluation.
        runtime_l2l_chunk_size : Optional[int]
            L2L chunk size for the prepass evaluation.
        grouped_interactions : bool
            Whether the prepass traversal groups interactions.
        farfield_mode : str
            Far-field mode for the prepass.
        record_retry : Callable[[DualTreeRetryEvent], None]
            Sink for traversal retry events raised inside the prepass.
        refine_local_val : bool
            Local refinement setting for the prepass build.
        max_refine_levels_val : int
            Refinement level cap for the prepass build.
        aspect_threshold_val : float
            Aspect threshold for the prepass build.

        Returns
        -------
        Array
            Per-node force scale, reduced with
            :meth:`_force_scale_reduction_mode` -- ``min`` under the paper
            estimator, which is what eq (16a) asks for.
        """

        low_upward = self.prepare_upward_sweep(
            tree_artifacts.tree,
            tree_artifacts.positions_sorted,
            tree_artifacts.masses_sorted,
            max_order=int(low_order),
            center_mode=upward_center_mode,
            max_leaf_size=tree_artifacts.leaf_cap,
        )
        low_locals_template = self._build_locals_template_for_prepare_state(
            tree=tree_artifacts.tree,
            upward=low_upward,
            max_order=int(low_order),
            pos_sorted=tree_artifacts.positions_sorted,
        )
        low_tree_artifacts = _PrepareStateTreeUpwardArtifacts(
            tree_mode=tree_artifacts.tree_mode,
            tree=tree_artifacts.tree,
            positions_sorted=tree_artifacts.positions_sorted,
            masses_sorted=tree_artifacts.masses_sorted,
            inverse_permutation=tree_artifacts.inverse_permutation,
            leaf_cap=tree_artifacts.leaf_cap,
            leaf_parameter=tree_artifacts.leaf_parameter,
            topology_key=tree_artifacts.topology_key,
            upward=low_upward,
            locals_template=low_locals_template,
        )

        with self._force_scale_prepass_scope(low_order=int(low_order)):
            self.adaptive_order = False
            self.adaptive_error_model = "tail_proxy"
            self.mac_type = "dehnen"
            dual_downward_artifacts = self._prepare_state_dual_and_downward(
                tree_artifacts=low_tree_artifacts,
                force_scale_nodes=None,
                upward_center_mode=upward_center_mode,
                theta_val=theta_val,
                mac_type_val=self.mac_type,
                dehnen_radius_scale=self.dehnen_radius_scale,
                runtime_traversal_config=runtime_traversal_config,
                runtime_m2l_chunk_size=runtime_m2l_chunk_size,
                runtime_l2l_chunk_size=runtime_l2l_chunk_size,
                grouped_interactions=grouped_interactions,
                farfield_mode=farfield_mode,
                record_retry=record_retry,
                refine_local_val=refine_local_val,
                max_refine_levels_val=max_refine_levels_val,
                aspect_threshold_val=aspect_threshold_val,
                allow_stateful_cache=False,
            )
            prepass_execution_backend = self._resolve_execution_backend()
            if prepass_execution_backend == "octree":
                prepass_octree, prepass_octree_native = (
                    build_octree_execution_data_with_status(low_tree_artifacts.tree)
                )
            else:
                prepass_octree, prepass_octree_native = None, False
            # See the main prepared-state path: only build native-octree interaction
            # lists when the octree view is non-degenerate; otherwise far/near come from
            # the consistent compat lists on the fallback (binary) tree.
            prepass_octree_native_neighbors = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_neighbors = build_octree_native_neighbor_lists(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_nearfield_interop = _build_nearfield_interop_data(
                low_tree_artifacts.tree,
                dual_downward_artifacts.neighbor_list,
                octree=prepass_octree,
                native_neighbors=prepass_octree_native_neighbors,
            )
            nearfield_artifacts = self._prepare_state_nearfield_artifacts(
                neighbor_list=dual_downward_artifacts.neighbor_list,
                nearfield_interop=prepass_nearfield_interop,
                leaf_cap=low_tree_artifacts.leaf_cap,
                num_particles=int(low_tree_artifacts.positions_sorted.shape[0]),
                cache_entry=dual_downward_artifacts.cache_entry,
                allow_stateful_cache=False,
            )
            prepass_octree_upward = _build_octree_upward_artifacts(
                octree=prepass_octree,
                positions_sorted=low_tree_artifacts.positions_sorted,
                masses_sorted=low_tree_artifacts.masses_sorted,
                expansion_basis=self.expansion_basis,
                max_order=int(low_order),
            )
            prepass_octree_native_far_pairs = None
            if (
                prepass_execution_backend == "octree"
                and prepass_octree is not None
                and prepass_octree_native
            ):
                prepass_octree_native_far_pairs = build_octree_native_far_pairs(
                    low_tree_artifacts.tree,
                    low_tree_artifacts.upward.geometry,
                    theta=theta_val,
                    mac_type=self.mac_type,
                    dehnen_radius_scale=self.dehnen_radius_scale,
                    max_pair_queue=self.max_pair_queue,
                    process_block=self.pair_process_block,
                    traversal_config=runtime_traversal_config,
                )
            prepass_octree_downward = _build_octree_downward_artifacts(
                octree=prepass_octree,
                octree_upward=prepass_octree_upward,
                interactions=dual_downward_artifacts.interactions,
                native_far_pairs=prepass_octree_native_far_pairs,
                execution_backend=prepass_execution_backend,
            )
            prepass_state = FMMPreparedState(
                tree=low_tree_artifacts.tree,
                upward=_prepared_state_upward_payload(
                    upward=low_tree_artifacts.upward,
                    memory_objective=self.memory_objective,
                ),
                downward=dual_downward_artifacts.downward,
                neighbor_list=dual_downward_artifacts.neighbor_list,
                max_leaf_size=low_tree_artifacts.leaf_cap,
                input_dtype=low_tree_artifacts.positions_sorted.dtype,
                working_dtype=low_tree_artifacts.positions_sorted.dtype,
                expansion_basis=self.expansion_basis,
                theta=theta_val,
                topology_key=low_tree_artifacts.topology_key,
                interactions=dual_downward_artifacts.interactions,
                dual_tree_result=dual_downward_artifacts.traversal_result,
                retry_events=tuple(),
                nearfield_interop=prepass_nearfield_interop,
                nearfield_target_leaf_ids=nearfield_artifacts.target_leaf_ids,
                nearfield_source_leaf_ids=nearfield_artifacts.source_leaf_ids,
                nearfield_valid_pairs=nearfield_artifacts.valid_pairs,
                nearfield_chunk_sort_indices=nearfield_artifacts.chunk_sort_indices,
                nearfield_chunk_group_ids=nearfield_artifacts.chunk_group_ids,
                nearfield_chunk_unique_indices=nearfield_artifacts.chunk_unique_indices,
                force_scale_nodes=None,
                execution_backend=prepass_execution_backend,
                octree=prepass_octree,
                octree_upward=_prepared_state_octree_upward_payload(
                    octree_upward=prepass_octree_upward,
                    memory_objective=self.memory_objective,
                ),
                octree_downward=_finalize_octree_downward_artifacts(
                    octree=prepass_octree,
                    octree_upward=prepass_octree_upward,
                    octree_downward=prepass_octree_downward,
                    expansion_basis=self.expansion_basis,
                    execution_backend=prepass_execution_backend,
                    m2l_chunk_size=runtime_m2l_chunk_size,
                ),
            )
            prepass_acc = self.evaluate_prepared_state(
                prepass_state,
                return_potential=False,
                jit_traversal=False,
            )

        sorted_idx = jnp.argsort(low_tree_artifacts.inverse_permutation)
        return jnp.asarray(prepass_acc)[sorted_idx]

"""The observation operator: source positions -> accelerations at tracers.

This module owns the two invariants section 7 rests on, and enforces both
rather than documenting them:

1. **Masses are frozen and equal, and are inputs, not parameters.** The
   differentiated pytree contains positions only. :func:`assert_masses_frozen_and_equal`
   checks a pytree carries no mass leaf, so a later refactor cannot silently
   free them.
2. **The MAC is not differentiated.** jaccpot's fixed-topology contract
   (``docs/differentiable_fmm.md``) holds the Morton permutation, node
   membership, the M2L interaction list, the near-field neighbour CSR and
   *every MAC accept/reject decision* constant under ``jax.grad``; only the
   numeric pipeline is differentiated. With free positions the accept/reject
   test depends on the parameters through geometry for every MAC variant, so
   this is not optional -- and there is no jaccpot option that differentiates
   it. The one route to a parameter-dependent criterion is a **mass-dependent**
   MAC (``dehnen_error`` / ``dehnen_theta``), whose acceptance is driven by
   node force scales that would read zero-mass tracers as massless sources and
   whose criterion is exactly what a frozen topology must not depend on. Those
   are refused here.

Tracers are realised as **zero-mass particles** appended to the source set. A
zero-mass particle exerts no force and receives the full acceleration from the
sources, so taking its rows out of a full evaluation gives the field at the
tracer positions. Verified against a sources-only direct sum to 1.6e-15 relative
L2 before this module was written.

Why the tracer rows are taken by a SLICE and not by ``target_indices``
----------------------------------------------------------------------
They are appended, so they occupy the contiguous block ``[N, N + M)`` and a
slice suffices. That is not a stylistic preference -- passing
``target_indices`` makes the operator unusable at the scale this section is
about. Measured at N=262144, M=4096, leaf 256, fp64 on an A100:

===================================  ==========  ==========
forward-only evaluation              eager       jitted
===================================  ==========  ==========
``target_indices`` gather            OUT OF      15.86 s
                                     MEMORY
full evaluation + contiguous slice   5.35 s      15.67 s
===================================  ==========  ==========

The ``target_indices`` path builds a dense ``f64[M, N + M, 3]`` all-pairs
displacement array and reduces it -- 26 GB at this N. Under ``jax.jit`` XLA
fuses the reduction into its producer and never allocates it, which is why the
jitted column is fine and why the timing figure was never affected. Eagerly it
must be materialised, and it is fatal. Since the fit driver runs the eager path
by default (:mod:`~jaccpot.applications.density_reconstruction.fit` explains
why), every fit and every forward-only diagnostic above N ~ 1e5 hit this.

The two forms agree to **1.3e-16 relative L2** -- float64 round-off from a
different reduction order, not a change of what is computed -- and the jitted
cost is the same to 1%, so the full evaluation of ``N + M`` particles instead of
``M`` targets is free in practice: the FMM's sweeps compute every particle's
field regardless.

:attr:`ForwardOperator.tracer_indices` is kept, because it is the honest
description of which rows are tracers and is useful to a caller, but it is no
longer on the evaluation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from jaccpot import FastMultipoleMethod

__all__ = [
    "ForwardOperator",
    "MASS_DEPENDENT_MAC_TYPES",
    "assert_masses_frozen_and_equal",
    "assert_no_differentiated_mac",
    "make_forward_operator",
]

#: MAC variants whose accept/reject decision depends on masses. Refused when
#: tracers are present: their criterion would see zero-mass tracers as sources
#: with no force scale, and a parameter-dependent criterion is precisely what the
#: fixed-topology contract must not carry.
MASS_DEPENDENT_MAC_TYPES: Tuple[str, ...] = ("dehnen_error", "dehnen_theta")

#: Leaf names that would indicate masses have been freed as parameters.
_MASS_LEAF_NAMES: Tuple[str, ...] = ("mass", "masses", "source_mass", "m")


def assert_masses_frozen_and_equal(
    parameters: Any,
    masses: Union[np.ndarray, jnp.ndarray],
    *,
    rtol: float = 0.0,
) -> None:
    """Enforce hard decision 1: masses are inputs, equal, and not parameters.

    Parameters
    ----------
    parameters : Any
        The pytree that will be differentiated. Its leaves are inspected by
        path name: any leaf whose final path component is a mass-like name is
        rejected, and any leaf with the same length as ``masses`` that equals
        it elementwise is rejected too, so a mass array smuggled in under
        another name is still caught.
    masses : Union[np.ndarray, jnp.ndarray]
        ``(N,)`` masses that will be passed as an *input*.
    rtol : float
        Relative tolerance for the equality check. ``0.0`` (default) means
        bit-equal, which is what a value assigned once as ``M / N`` gives.

    Raises
    ------
    ValueError
        If any mass differs from the first, or if a mass leaf is found in the
        parameter pytree.
    """
    m = np.asarray(masses, dtype=np.float64)
    if m.ndim != 1 or m.size == 0:
        raise ValueError(f"masses must be a non-empty 1-D array, got shape {m.shape}")
    if not np.allclose(m, m[0], rtol=rtol, atol=0.0):
        spread = float(m.max() - m.min())
        raise ValueError(
            "masses must be equal (hard decision 1): "
            f"spread {spread:.3e} around {m[0]:.6e}"
        )
    leaves_with_paths = jax.tree_util.tree_flatten_with_path(parameters)[0]
    for path, leaf in leaves_with_paths:
        names = [getattr(p, "name", getattr(p, "key", None)) for p in path]
        last = str(names[-1]).lower() if names else ""
        if last in _MASS_LEAF_NAMES:
            raise ValueError(
                f"parameter pytree contains a mass leaf at {jax.tree_util.keystr(path)}; "
                "masses are frozen inputs, not parameters (hard decision 1)"
            )
        arr = np.asarray(leaf)
        if arr.shape == m.shape and np.array_equal(arr, m):
            raise ValueError(
                f"parameter leaf at {jax.tree_util.keystr(path)} equals the mass "
                "array; masses must not be smuggled in as parameters"
            )


def assert_no_differentiated_mac(
    *,
    mac_type: Optional[str],
    has_tracers: bool,
    differentiate_mac: bool = False,
) -> None:
    """Enforce hard decision 5: the MAC is ``stop_gradient``-wrapped.

    Parameters
    ----------
    mac_type : Optional[str]
        The resolved multipole acceptance criterion, or ``None`` for the
        preset default.
    has_tracers : bool
        Whether zero-mass tracers are appended to the particle set.
    differentiate_mac : bool
        A caller asking for the acceptance test to be differentiated. There is
        no such mode in jaccpot; the flag exists so a configuration that
        *requests* it fails loudly here instead of being silently ignored.

    Raises
    ------
    ValueError
        If ``differentiate_mac`` is set, or if a mass-dependent MAC is used
        with tracers present.
    """
    if differentiate_mac:
        raise ValueError(
            "differentiate_mac=True is not supported and never will be here: "
            "jaccpot's fixed-topology contract holds every MAC accept/reject "
            "decision constant under jax.grad (docs/differentiable_fmm.md, 'The "
            "contract: fixed topology'). With free positions the criterion "
            "depends on the parameters through geometry for every MAC variant, "
            "so it must be stop_gradient-wrapped -- which the frozen topology "
            "does structurally."
        )
    if has_tracers and mac_type is not None and mac_type in MASS_DEPENDENT_MAC_TYPES:
        raise ValueError(
            f"mac_type={mac_type!r} is mass-dependent and cannot be used with "
            "zero-mass tracers: its acceptance is driven by node force scales "
            "that read a massless particle as a source with no scale, and a "
            "mass-dependent criterion is exactly what a frozen topology must not "
            "depend on. Use 'bh', 'engblom' or 'dehnen'."
        )


@dataclass(frozen=True)
class ForwardOperator:
    """Positions -> accelerations at fixed tracer positions.

    Build with :func:`make_forward_operator`. The operator appends the tracers
    as zero-mass particles, prepares the tree on the host (not traceable, by
    contract), and returns the tracer rows. :meth:`evaluate` rebuilds the
    topology from the current positions; :meth:`evaluate_at_topology` takes a
    prepared state so the same frozen function can be differentiated and
    finite-differenced.

    Attributes
    ----------
    fmm : FastMultipoleMethod
        The configured solver.
    tracer_positions : np.ndarray
        ``(M, 3)`` fixed observation points.
    source_mass : float
        The one mass every source carries.
    num_sources : int
        ``N``.
    softening : float
        Plummer softening, recorded because hard decision 6 says so.
    order : int
        Expansion order for ``prepare_state``.
    leaf_size : int
        Leaf size for ``prepare_state``.
    theta : Optional[float]
        Acceptance parameter, or ``None`` for the preset default.
    mac_type : Optional[str]
        The MAC in use, or ``None`` for the preset default.
    """

    fmm: FastMultipoleMethod
    tracer_positions: np.ndarray
    source_mass: float
    num_sources: int
    softening: float
    order: int
    leaf_size: int
    theta: Optional[float]
    mac_type: Optional[str]

    @property
    def num_tracers(self: "ForwardOperator") -> int:
        """Return ``M``.

        Returns
        -------
        int
            Tracer count.
        """
        return int(self.tracer_positions.shape[0])

    @property
    def tracer_indices(self: "ForwardOperator") -> jnp.ndarray:
        """Return the rows of the combined particle set that are tracers.

        Returns
        -------
        jnp.ndarray
            ``(M,)`` int32 indices ``[N, N + M)``.
        """
        return jnp.arange(
            self.num_sources, self.num_sources + self.num_tracers, dtype=jnp.int32
        )

    def masses(self: "ForwardOperator") -> jnp.ndarray:
        """Return the combined mass vector: equal source masses, zero tracers.

        Returns
        -------
        jnp.ndarray
            ``(N + M,)`` float64.
        """
        return jnp.concatenate(
            [
                jnp.full((self.num_sources,), self.source_mass, dtype=jnp.float64),
                jnp.zeros((self.num_tracers,), dtype=jnp.float64),
            ]
        )

    def combined_positions(
        self: "ForwardOperator", source_positions: ArrayLike
    ) -> jnp.ndarray:
        """Append the fixed tracers to the (differentiated) sources.

        Parameters
        ----------
        source_positions : ArrayLike
            ``(N, 3)`` source positions -- NumPy or JAX, concrete or traced;
            converted with ``jnp.asarray`` so both are accepted under the
            runtime typecheck hook.

        Returns
        -------
        jnp.ndarray
            ``(N + M, 3)``. The tracer block is a constant, so no cotangent
            reaches it.
        """
        return jnp.concatenate(
            [
                jnp.asarray(source_positions, dtype=jnp.float64),
                jnp.asarray(self.tracer_positions, dtype=jnp.float64),
            ],
            axis=0,
        )

    def prepare(self: "ForwardOperator", source_positions: Any) -> Any:
        """Build the frozen topology for the given source positions.

        Host-side and not traceable, by contract. Call it outside ``jax.grad``
        and pass the result to :meth:`evaluate_at_topology`.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        Any
            An ``FMMPreparedState`` for the combined source + tracer set.
        """
        return self.fmm.prepare_state(
            self.combined_positions(source_positions),
            self.masses(),
            leaf_size=self.leaf_size,
            max_order=self.order,
            theta=self.theta,
        )

    def evaluate_at_topology(
        self: "ForwardOperator", state: Any, source_positions: ArrayLike
    ) -> jnp.ndarray:
        """Accelerations at the tracers, at the frozen topology ``state``.

        This is the differentiable map. Within one call the gradient is the
        exact gradient of this fixed-topology, fixed-interaction-list forward
        map -- the contract section 7 states.

        Parameters
        ----------
        state : Any
            Prepared state from :meth:`prepare`.
        source_positions : ArrayLike
            ``(N, 3)`` source positions, differentiated. NumPy or JAX.

        Returns
        -------
        jnp.ndarray
            ``(M, 3)`` accelerations at the tracer positions.

        Notes
        -----
        Evaluates every particle and slices the contiguous tracer block rather
        than passing ``target_indices``. The module docstring has the
        measurement: the ``target_indices`` path materialises a dense
        ``f64[M, N + M, 3]`` array eagerly and runs out of memory at N=262144,
        while the slice does not, the two agree to 1.3e-16, and the jitted cost
        is identical.
        """
        accelerations = self.fmm.differentiable_accelerations(
            state,
            self.combined_positions(source_positions),
            self.masses(),
        )
        return accelerations[self.num_sources :]

    def evaluate(self: "ForwardOperator", source_positions: Any) -> jnp.ndarray:
        """Rebuild the topology from ``source_positions`` and evaluate.

        Parameters
        ----------
        source_positions : Any
            ``(N, 3)`` concrete positions.

        Returns
        -------
        jnp.ndarray
            ``(M, 3)`` accelerations at the tracers.
        """
        return self.evaluate_at_topology(
            self.prepare(source_positions), jnp.asarray(source_positions)
        )

    def leaf_blocks(self: "ForwardOperator", state: Any) -> Any:
        """Return the frozen leaf partition the regularisers are built on.

        Parameters
        ----------
        state : Any
            Prepared state from :meth:`prepare`.

        Returns
        -------
        Any
            A :class:`~jaccpot.applications.density_reconstruction.loss.LeafBlocks`.
            Exists so this operator and the distributed one are
            interchangeable to ``run_fit``: the distributed path has no leaf
            partition and returns ``None`` here instead.
        """
        from jaccpot.applications.density_reconstruction.loss import (
            leaf_blocks_from_state,
        )

        return leaf_blocks_from_state(state, num_sources=self.num_sources)

    def record(self: "ForwardOperator") -> Dict[str, Any]:
        """Return the operator configuration for a results JSON.

        Returns
        -------
        Dict[str, Any]
            ``N``, ``M``, softening, order, leaf size, theta and MAC.
        """
        return {
            "N": int(self.num_sources),
            "M": int(self.num_tracers),
            "softening": float(self.softening),
            "order": int(self.order),
            "leaf_size": int(self.leaf_size),
            "theta": None if self.theta is None else float(self.theta),
            "mac_type": self.mac_type,
            # Named so it can never be confused with the distributed force
            # evaluation, whose wall-clock is not comparable with this one's.
            "sharding_mode": "single_device_or_parameter_sharding",
        }


def make_forward_operator(
    *,
    tracer_positions: Union[np.ndarray, Sequence[Sequence[float]]],
    source_mass: float,
    num_sources: int,
    softening: float,
    order: int = 4,
    theta: Optional[float] = None,
    leaf_size: int = 16,
    mac_type: Optional[str] = None,
    preset: str = "accurate",
    basis: str = "solidfmm",
    differentiate_mac: bool = False,
) -> ForwardOperator:
    """Configure the observation operator and enforce its invariants.

    Parameters
    ----------
    tracer_positions : Union[np.ndarray, Sequence[Sequence[float]]]
        ``(M, 3)`` fixed observation points.
    source_mass : float
        The one mass every source carries.
    num_sources : int
        ``N``.
    softening : float
        Plummer softening length (hard decision 6: fixed and recorded).
    order : int
        Expansion order.
    theta : Optional[float]
        Acceptance parameter, or ``None`` for the preset default.
    leaf_size : int
        Leaf size.
    mac_type : Optional[str]
        Multipole acceptance criterion. Mass-dependent variants are refused.
    preset : str
        jaccpot preset; ``"accurate"`` by default.
    basis : str
        Expansion basis; ``"solidfmm"`` is the radix differentiable path.
    differentiate_mac : bool
        Must be ``False``; see :func:`assert_no_differentiated_mac`.

    Returns
    -------
    ForwardOperator
        The configured operator.

    Raises
    ------
    ValueError
        If ``tracer_positions`` is not ``(M, 3)``. The MAC checks raise from
        :func:`assert_no_differentiated_mac` and are documented there.
    """
    assert_no_differentiated_mac(
        mac_type=mac_type, has_tracers=True, differentiate_mac=differentiate_mac
    )
    tracers = np.asarray(tracer_positions, dtype=np.float64)
    if tracers.ndim != 2 or tracers.shape[1] != 3:
        raise ValueError(f"tracer_positions must be (M, 3), got {tracers.shape}")
    kwargs: Dict[str, Any] = dict(
        preset=preset, basis=basis, softening=float(softening)
    )
    if theta is not None:
        kwargs["theta"] = float(theta)
    if mac_type is not None:
        from jaccpot import FMMAdvancedConfig

        kwargs["advanced"] = FMMAdvancedConfig(mac_type=mac_type)
    fmm = FastMultipoleMethod(**kwargs)
    return ForwardOperator(
        fmm=fmm,
        tracer_positions=tracers,
        source_mass=float(source_mass),
        num_sources=int(num_sources),
        softening=float(softening),
        order=int(order),
        leaf_size=int(leaf_size),
        theta=theta,
        mac_type=mac_type,
    )

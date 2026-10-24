"""Galaxy-scale production on the fused lane: what ``preset="galaxy"`` runs, and the runner.

``FastMultipoleMethod(preset="galaxy")`` is the configuration of the fused lane's
production rows: ``bench/fused_memory_budget.py`` at its defaults (those of the L1-L3
rows, 8e6 and 1e8 particles on one A100 and the 25M disc + bulge) with a cubic Morton
box (``--cubic-bounds``, S3b) and, since 2026-10-21, Dehnen's error MAC
(``--mac-type dehnen_error --adaptive-eps 1e-4``), named so a caller no longer copies
it from the bench. It is built from ONE :class:`GalaxySettings` instance, so the
constructor half (the tree, the box, the MAC, the opening angle, the softening
kernel, the precision) and the run-time half (leaf size, expansion order, carry --
arguments of ``strict_run_v2``) cannot drift apart.

THE RUNNER. :class:`ProductionRunner` runs a long simulation as a host loop of
``SegmentConfig.steps``-step segments, each ONE ``strict_run_v2(carry="particles")``
call that continues from the previous call's
:class:`~jaccpot.runtime.strict_carry.StrictParticleCarry` handle. It is not a new
scan over ``FusedForceProvider.step``: the segments ARE the library's programs, so
"the runner equals one long call" is "chained calls equal one call" -- bitwise on
CPU, with the handle's total acceleration (``carry_total_acceleration``; without it
an external field is evaluated again outside the scan at every edge, 1.9e-6 on one
of 3072 components at N = 512) -- and every memory optimisation of the scan, its
segment retry and its re-plan come with it. Between segments, on the host: the
diagnostics (momentum, angular momentum, centre of mass and kinetic energy, fp64
reductions on device; at every ``diagnostics_every_energy``-th edge also the total
energy, from the fused lane's O(N) potential), callbacks (:class:`SnapshotWriter`, :class:`DiagnosticsLog`, or any callable; one
returning ``True`` stops the run after that edge) and restart files.

SEGMENT LENGTHS. ``num_steps`` is part of a scan's compile key, so every distinct
segment length compiles its own scan: a ragged last segment of 14 steps recompiled
for 52.8 s at 25M on an A100 (S3 probe). Choose ``SegmentConfig.steps`` to divide
``num_steps``; otherwise the run is ``num_steps // steps`` segments of ``steps`` and
one of the remainder, last, and the runner warns once (:func:`segment_lengths`). A
remainder of one step joins the last segment: a 1-step scan compiles to different
code from a longer one's loop, so it would not be bitwise one long call.

COMPILES. A fresh run compiles the eager prepare's programs, the start force and one
scan; with ``donate_input=False`` (the default: the caller's state survives) the
second segment compiles the scan once more, with the state donated, and every later
segment reuses it. ``donate_input=True`` gives every segment the same program.

RESTARTS. :meth:`ProductionRunner.save_restart` (or ``SegmentConfig.restart_every``)
writes one ``.npz``: the arrays a fresh process needs, and as JSON the counters, the
baselines, the plan the scan programs were traced with and the digests of their
compile keys. ``run(..., restart_from=path)`` restores that plan exactly and checks
the digests, so the continuation is bitwise the uninterrupted run (CPU, four
variants); a mismatch is reported, never silent.

LAUNCH ENVIRONMENT. On a GPU, a fresh process continues a run bitwise only if XLA
compiles the same code as the process that wrote the restart, and XLA's fusion
autotuner picks the emitter of some fusions by timing them on the card. Production
runs therefore start with ``--xla_gpu_experimental_enable_fusion_autotuner=false`` in
``XLA_FLAGS``; :meth:`ProductionRunner.run` warns once per process on a GPU when it is
missing. It cannot set the flag itself: XLA reads ``XLA_FLAGS`` once, when the backend
starts (docs/production.md, "Launch environment").

THE MULTI-GPU MODE. ``ProductionRunner(solver, ..., mesh=<a 1-D jax Mesh or a
:class:`MeshRunConfig`>)`` runs the same loop -- segments, edges, callbacks, the
diagnostics with the energy, restart files and their continuation, the segment
retry -- on the fused multi-GPU lane's segment runner
(:class:`~jaccpot.distributed.segment.MeshSegmentRunner`;
:mod:`jaccpot.distributed.mesh_runner` is that mode's backend). The API stays in
input (id) order: ``run(state [N, 2, 3], masses [N], ...)`` decomposes the particles
over the mesh, :attr:`RunResult.state` is the final state gathered by id, and
:meth:`MeshSegmentEdge.state_host` gathers an edge's state on demand. ``mesh=None``
(the default) is the single-card runner, unchanged (docs/production.md, "Multi-GPU
mode").

The module imports nothing heavy at import time (``jaccpot.solver`` imports it).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

import numpy as np

__all__ = [
    "DIAGNOSTICS",
    "GALAXY_SETTINGS",
    "RESTART_FORMAT",
    "DiagnosticsLog",
    "GalaxySettings",
    "MESH_CAP_HEADROOM",
    "MeshRunConfig",
    "MeshSegmentEdge",
    "ProductionRunner",
    "RunResult",
    "SegmentConfig",
    "SegmentEdge",
    "SnapshotWriter",
    "load_restart",
    "mesh_shard_capacity",
    "run",
    "segment_lengths",
]


@dataclass(frozen=True)
class GalaxySettings:
    """The galaxy production configuration of the fused lane.

    Frozen. The defaults are ``preset="galaxy"``; :data:`GALAXY_SETTINGS` is that
    instance.

    THE MAC (user decision, 2026-10-21). The preset accepts a far pair by Dehnen
    (2014) eq (16a) at ``adaptive_eps`` 1e-4 (``mac_type="dehnen_error"``); before, by
    the geometric Dehnen test at ``theta`` 0.8. On the 25M disc + bulge on one A100
    (S3b) its step takes 1.12x the geometric one (260.5 against 233.6 ms) for a 2.8x
    lower force error (rel-L2 1.49e-4 against 4.18e-4); the 1e8 clipped Plummer step
    831 ms at 279 B/particle (geometric 257). What it costs: about 22 B/particle
    more at 1e8 (a lower one-card ceiling); ``G`` and the softening cannot be traced
    -- eq (16a)'s thresholds are built with the solver's own, so another value means
    another solver and a compile (``dt`` and the external field's parameters stay
    traced); its 0.1-1 % error tail does not narrow like the bulk (Dehnen 2014);
    restart files carry f_b (+4 B/particle). The geometric MAC is ``mac_type=
    "dehnen", adaptive_eps=None`` (``docs/production.md``, "The MAC").

    Attributes
    ----------
    leaf_size : int
        Particles per cell leaf at most (``strict_run_v2(leaf_size=...)``; also the
        tree's ``leaf_target``). 64: the leaf-size U-curve's minimum on one A100.
    max_order : int
        Expansion order (``strict_run_v2(max_order=...)``, and the solver's
        ``fixed_order``). p6 since round 5: at ``cell_min_level`` 8 it halves the
        fp64 rel-L2 of p5 for ~5 % of the step.
    theta : float
        Opening angle of the geometric Dehnen MAC. Under ``"dehnen_error"`` it
        decides no pair: the flat walk accepts by eq (16a) and its convergence
        clause ``rho_A + rho_B < mac_theta_max r`` (``mac_theta_max`` 1.0), and the
        first step's f_b seed walks at 0.8 whatever ``theta`` is
        (``mac_force_scale_prepass_theta``). Theta 0.3, 0.5 and 0.8 gave the same
        lists and the same states bit for bit (CPU, N = 2048). It stays 0.8, the
        value the bench passes; it is still part of the compile keys.
    cell_min_level : int
        No leaf coarser than this Morton level (``TreeConfig.cell_min_level``).
    leaf_capacity : Union[int, Literal["auto"]]
        ``TreeConfig.leaf_capacity``; ``"auto"`` sizes it at the first eager
        prepare.
    leaf_headroom : float
        ``TreeConfig.leaf_headroom``: the factor over the live cell leaves.
    cubic_bounds : bool
        ``TreeConfig.cubic_bounds``: the Morton box is a cube on the particles'
        longest extent, not padded per axis, so every cell is a cube whatever the
        box's aspect (S3b). On the 25M disc + bulge, whose box is a cube to 0.4 %
        (the bulge reaches r = 20 on every axis), the two boxes' lists agree to
        1 % and their rel-L2 within 9 %, either way, at step 0 and along the
        production run (steps 250 and 500: the per-axis box stays within 1.2 % of
        a cube); on the same disc
        without its bulge (3:1, thin in z) the cube needs 63 % fewer far and 50 %
        fewer near pairs and halves the force error (rel-L2 1.02e-3 -> 5.3e-4).
        ``JACCPOT_CUBIC_BOUNDS=0`` turns it off for a run.
    softening_kernel : str
        Pair softening kernel (:mod:`jaccpot.softening`).
    mac_type : str
        Multipole acceptance criterion. ``"dehnen_error"`` (the default since
        2026-10-21): eq (16a) inside the flat walk, with the eq (15) estimator and
        eq (16b)'s f_b carried from step to step --
        ``bench/fused_memory_budget.py --mac-type dehnen_error``, which also passes
        ``adaptive_error_model="dehnen_paper"`` and ``mac_force_scale_mode=
        "paper_fb"``, as the preset does (the fused lane carries f_b whatever they
        are; the general path reads them).
        ``"dehnen"``: the geometric Dehnen MAC at ``theta``, 0.90x the step at 25M
        and 0.92x the peak at 1e8, ``G`` and the softening traceable, 2.8x the
        rel-L2.
    adaptive_eps : Optional[float]
        ``"dehnen_error"``'s relative force-accuracy target, 1e-4; ``None`` with the
        geometric MAC.
    carry : Literal["particles", "state"]
        ``strict_run_v2(carry=...)``: the particle carry (input order, a handle
        between calls) is the production one.
    precision : Literal["fp32", "fp64"]
        Working precision; fp64 is the lane's validation mode.
    """

    leaf_size: int = 64
    max_order: int = 6
    theta: float = 0.8
    cell_min_level: int = 8
    leaf_capacity: Union[int, Literal["auto"]] = "auto"
    leaf_headroom: float = 1.15
    cubic_bounds: bool = True
    softening_kernel: str = "ferrers3"
    mac_type: str = "dehnen_error"
    adaptive_eps: Optional[float] = 1e-4
    carry: Literal["particles", "state"] = "particles"
    precision: Literal["fp32", "fp64"] = "fp32"


#: ``preset="galaxy"``.
GALAXY_SETTINGS = GalaxySettings()

#: the diagnostics a :class:`SegmentConfig` may name
DIAGNOSTICS = ("momentum", "angular_momentum", "com", "kinetic", "energy")


@dataclass(frozen=True)
class SegmentConfig:
    """How :class:`ProductionRunner` cuts a run into ``strict_run_v2`` calls.

    Frozen.

    Attributes
    ----------
    steps : int
        Steps per segment (one ``strict_run_v2`` call). Choose it to divide the
        run's ``num_steps``: a remainder segment compiles its own scan
        (:func:`segment_lengths`).
    traced_params : bool
        Pass ``dt``, ``G`` and the softening squared (not under
        ``mac_type='dehnen_error'``, ``preset="galaxy"``'s MAC, which takes the
        solver's own: another ``G`` or softening is another solver, and compiles)
        and the external field's float parameters as ARRAYS: arguments of the
        compiled programs, so a run with other values compiles nothing new. The
        runner's default;
        ``strict_run_v2`` itself keeps literals by default (under ``ferrers3`` a
        traced softening is not bitwise the literal one: one float32 rounding,
        ``tests/integration/strict/test_traced_params.py``).
    donate_input : bool
        Let the first segment consume the caller's ``state`` (its buffer becomes
        the returned state). Segments 2 and later always donate: their input is the
        runner's own.
    diagnostics_every : int
        Moments (momentum, angular momentum, centre of mass, kinetic energy) at
        step 0, every ``diagnostics_every``-th segment edge and the last one; ``0``
        turns the diagnostics off.
    diagnostics : tuple
        Which to report, of :data:`DIAGNOSTICS`. ``"energy"``: the potential
        energies of the self-gravity and the external field and the total energy,
        at every ``diagnostics_every_energy``-th edge
        (:meth:`ProductionRunner.potential_energy`).
    diagnostics_every_energy : int
        Segment edges between energy diagnostics (the potential program costs about
        one force evaluation, 1.1x at 25M and 1e8); ``0`` turns the energy off.
    restart_every : int
        Write a restart file every ``restart_every`` segments (``0``: never).
    restart_path : Optional[str]
        Where; ``{step}`` and ``{segment}`` are formatted in, otherwise each restart
        replaces the last (written to a temporary file and renamed).
    """

    steps: int = 50
    traced_params: bool = True
    donate_input: bool = False
    diagnostics_every: int = 1
    diagnostics: tuple = DIAGNOSTICS
    diagnostics_every_energy: int = 10
    restart_every: int = 0
    restart_path: Optional[str] = None

    def __post_init__(self) -> None:
        if int(self.steps) < 1:
            raise ValueError(f"SegmentConfig.steps must be >= 1, got {self.steps}")
        for name in ("diagnostics_every", "diagnostics_every_energy", "restart_every"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"SegmentConfig.{name} must be >= 0")
        unknown = sorted(set(self.diagnostics) - set(DIAGNOSTICS))
        if unknown:
            raise ValueError(
                f"unknown diagnostics {unknown}; known: {list(DIAGNOSTICS)}"
            )
        if int(self.restart_every) > 0 and not self.restart_path:
            raise ValueError("SegmentConfig.restart_every needs a restart_path")


@dataclass(frozen=True)
class SegmentEdge:
    """What a callback sees between two segments.

    The arrays are the run's own and only valid DURING the callback: the next
    segment consumes the state (donation). Copy what you keep (:meth:`state_host`).
    The edges in :attr:`RunResult.edges` hold none of them.

    Attributes
    ----------
    step : int
        Steps done since the run's start (``t0``).
    time : float
        Simulation time at ``step``.
    segment : int
        Segments done (``0``: the initial state, before any step).
    retries : int
        Segment retries (capacity re-plans) the segment that ended here needed.
    wall_s : float
        Host wall time of that segment.
    state : Any
        ``[N, 2, 3]`` positions and velocities at ``step`` (``None`` once kept in
        :attr:`RunResult.edges`).
    masses : Any
        ``[N]``.
    self_acceleration : Any
        ``[N, 3]`` self-gravity at ``state`` (``None`` at step 0).
    diagnostics : Optional[dict]
        The moments and their drifts against step 0 and, when due, the energy
        (:class:`ProductionRunner`), or ``None`` on an edge without diagnostics.
    plan : dict
        The plan the segment ran with: leaf capacity, validated walk caps, level
        registry, depth bound, force key.
    """

    step: int
    time: float
    segment: int
    retries: int
    wall_s: float
    state: Any
    masses: Any
    self_acceleration: Any
    diagnostics: Optional[dict]
    plan: dict

    def state_host(self) -> np.ndarray:
        """The state as a host array (a copy).

        Returns
        -------
        np.ndarray
            ``[N, 2, 3]``.

        Raises
        ------
        RuntimeError
            Outside the callback, where the edge holds no state any more.
        """
        import jax

        if self.state is None:
            raise RuntimeError(
                "this edge holds no state: the next segment consumed it. Copy the "
                "state inside the callback (SnapshotWriter does)."
            )
        return np.asarray(jax.device_get(self.state))


#: why an edge kept in ``RunResult.edges`` has no arrays
_CONSUMED_EDGE = (
    "this edge holds no state: the next segment consumed it. Copy the state inside "
    "the callback (SnapshotWriter does)."
)


@dataclass(frozen=True)
class MeshSegmentEdge(SegmentEdge):
    """What a callback sees between two segments in the multi-GPU mode.

    The fields of :class:`SegmentEdge`, with the mesh layout's arrays: ``state``
    and ``masses`` are the flat ``(ndev * cap, 2, 3)`` and ``(ndev * cap,)`` arrays
    of :attr:`layout` (rows by device, padding rows included, sharded over the
    mesh), and ``self_acceleration`` is ``None`` -- the segments carry the TOTAL
    acceleration (``layout.acceleration``). Nothing is gathered at an edge unless
    something asks: :meth:`state_host` (what :class:`SnapshotWriter` calls) and
    :meth:`gather` copy to the host by global id, on demand. Valid DURING the
    callback only, as on one card: the next segment consumes the arrays.

    Attributes
    ----------
    layout : Any
        The :class:`~jaccpot.distributed.segment.MeshState` at the edge (``None``
        once kept in :attr:`RunResult.edges`).
    """

    layout: Any = None

    def state_host(self) -> np.ndarray:
        """The state by global id, as a host array (one gather of the state and ids).

        Returns
        -------
        np.ndarray
            ``[N, 2, 3]`` in input (id) order.

        Raises
        ------
        RuntimeError
            Outside the callback, where the edge holds no state any more.
        """
        if self.layout is None:
            raise RuntimeError(_CONSUMED_EDGE)
        return self.gather(("state",))["state"]

    def gather(self, names: Sequence = ("state", "masses", "acceleration")) -> dict:
        """The edge's arrays by global id on the host.

        Parameters
        ----------
        names : Sequence
            Of ``"state"``, ``"masses"``, ``"acceleration"`` (the total) and
            ``"force_scale"`` (``mac_type='dehnen_error'``); one not computed yet
            is left out.

        Returns
        -------
        dict
            ``name -> (N, ...)`` host arrays in id order, and ``owner``: the device
            that holds each id.

        Raises
        ------
        RuntimeError
            Outside the callback, where the edge holds no state any more.
        """
        from jaccpot.distributed.mesh_runner import gather_by_id

        if self.layout is None:
            raise RuntimeError(_CONSUMED_EDGE)
        return gather_by_id(self.layout, names)


@dataclass
class RunResult:
    """What :meth:`ProductionRunner.run` returns.

    Pass it as ``restart_from`` to continue the run in this process; that run
    consumes ``state`` and ``handle``.

    Attributes
    ----------
    state : Any
        ``[N, 2, 3]`` final state (device).
    masses : Any
        ``[N]``.
    handle : Any
        The last segment's :class:`~jaccpot.runtime.strict_carry.StrictParticleCarry`.
    step : int
        Steps done since ``t0``.
    time : float
        Simulation time at ``step``.
    dt : float
        Timestep.
    t0 : float
        The run's start time.
    segment : int
        Segments done since ``t0``.
    edges : list
        One :class:`SegmentEdge` per edge of THIS call (arrays dropped).
    baselines : Optional[dict]
        Step 0's moments (``None`` without diagnostics).
    stopped : bool
        Whether a callback stopped the run.
    restarts : list
        Restart files this call wrote.
    restored : Optional[dict]
        Where this call continued from: ``path``, ``step``, ``bitwise`` (the
        digests matched) and ``mismatches``; ``None`` for a fresh run.
    wall_s : float
        Host wall time of this call.
    time_anchor : tuple
        ``(time, step)`` edge times are counted from with ``dt``.
    state_owned : bool
        Whether ``state`` is the runner's own -- a continuation consumes it --,
        and not the caller's input (a fresh run that did no segment).
    mode : str
        ``"single"`` (one card) or ``"mesh"`` (the multi-GPU mode: ``state`` and
        ``masses`` are host arrays gathered by id, ``handle`` the
        :class:`~jaccpot.distributed.segment.MeshSegmentRunner`); a run continues
        only a result of its own mode.
    """

    state: Any
    masses: Any
    handle: Any
    step: int
    time: float
    dt: float
    t0: float
    segment: int
    edges: list
    baselines: Optional[dict]
    stopped: bool
    restarts: list
    restored: Optional[dict]
    wall_s: float
    time_anchor: tuple = (0.0, 0)
    state_owned: bool = True
    mode: str = "single"


def segment_lengths(num_steps: int, steps: int) -> tuple:
    """The runner's segment plan: ``num_steps // steps`` segments, then the rest.

    Every distinct length compiles its own scan (``num_steps`` is in the compile
    key), so a run whose ``num_steps`` is a multiple of ``steps`` compiles ONE
    scan; otherwise the remainder segment, last, compiles a second.

    A remainder of ONE step joins the last full segment instead (5 steps of 2 run as
    2 + 3, not 2 + 2 + 1). XLA inlines a scan of length one into straight-line code
    and fuses it differently from the loop body of every longer scan, so a 1-step
    segment does not round like the steps of one long call: 2 + 2 + 1 differed from
    5 on a CI runner (#419 on main, 2026-10-16) where 2 + 2 + 2 matched 6 bitwise.
    With ``steps=1`` every segment is a 1-step scan; such a run is the same
    trajectory to round-off, not bitwise one long call.

    Parameters
    ----------
    num_steps : int
        Steps to run; ``>= 0``.
    steps : int
        Segment length; ``>= 1``.

    Returns
    -------
    tuple
        Segment lengths, in order; empty for ``num_steps == 0``.
    """
    full, rest = divmod(int(num_steps), int(steps))
    if rest == 1 and full >= 1 and int(steps) > 1:
        return (int(steps),) * (full - 1) + (int(steps) + 1,)
    return (int(steps),) * full + ((rest,) if rest else ())


# --------------------------------------------------------------------------
# The multi-GPU mode's configuration.
# --------------------------------------------------------------------------

#: rows per device over an even split, by default (:func:`mesh_shard_capacity`)
MESH_CAP_HEADROOM = 1.15


def mesh_shard_capacity(
    num_particles: int, ndev: int, headroom: float = MESH_CAP_HEADROOM
) -> int:
    """The multi-GPU mode's default rows per device: an even split with headroom.

    ``ceil(ceil(N / ndev) * headroom)``. A device's live count is set by the
    decomposition and every repartition (``sfc_repartition``'s sample sort, whose
    balance overshoot is about ``2 ndev / num_samples`` of a shard: 6 % on 8 devices
    at 256 samples) and stays fixed within a segment, so the headroom only has to
    cover that; 1.15 is the S11 and S12 gates' (``bench/multigpu_segment_gate.py``).
    A repartition that would overflow a device is declined (the old ownership
    stands, the edge says so); a decomposition that would is an error.

    Parameters
    ----------
    num_particles : int
        N.
    ndev : int
        Devices of the mesh.
    headroom : float
        The factor over the even split, at least 1.

    Returns
    -------
    int
        Rows per device.

    Raises
    ------
    ValueError
        For a headroom below 1, or no particles or devices.
    """
    if float(headroom) < 1.0:
        raise ValueError(f"the shard headroom must be >= 1, got {headroom}")
    if int(num_particles) < 1 or int(ndev) < 1:
        raise ValueError(f"no particles ({num_particles}) or devices ({ndev})")
    return int(math.ceil(math.ceil(int(num_particles) / int(ndev)) * float(headroom)))


@dataclass(frozen=True)
class MeshRunConfig:
    """How :class:`ProductionRunner`'s multi-GPU mode lays a run out (``mesh=``).

    Frozen. ``ProductionRunner(solver, mesh=<a jax Mesh>)`` is ``mesh=
    MeshRunConfig(mesh=<that mesh>)``.

    Attributes
    ----------
    mesh : Any
        A 1-D ``jax.sharding.Mesh`` (``yggdrax.distributed.sharding.make_mesh``).
    cap : Optional[int]
        Rows per device (every device's shard has this many rows, the live ones
        first); ``None``: :func:`mesh_shard_capacity` of the run's N at
        ``cap_headroom``. A restart on the same mesh keeps the file's.
    cap_headroom : float
        The default ``cap``'s factor over an even split.
    repartition_every : int
        Repartition before the segment after every ``repartition_every``
        completed segments: ``1`` (the default, the S12 gate's cadence) before
        every segment after the first; ``0`` never after the initial
        decomposition. Between segments, in a program of its own.
    num_samples : int
        Pivot samples per device of the decomposition and the repartitions.
    cross_caps : Any
        The fused force's cross-domain capacities: ``"auto"`` (the default, sized
        in the library at the start state), a
        :class:`~jaccpot.distributed.cross.CrossCapacities`, or ``None`` (its
        defaults). A segment retry grows them either way.
    force : Any
        ``None`` (the default): the fused lane's
        :class:`~jaccpot.distributed.template_force.FusedMeshForce` on the
        runner's solver, built per run (``settings.leaf_size`` /
        ``settings.max_order``, the solver's opening angle). A force object of the
        :class:`~jaccpot.distributed.segment.MeshSegmentRunner` protocol
        (:class:`~jaccpot.distributed.segment.DirectMeshForce` in tests) is used
        as given.
    migrate : bool
        Migrate to fixed cuts after every drift (``MeshSegmentRunner(migrate=True)``,
        :mod:`jaccpot.distributed.migration`): every force on exact Morton domains in
        the force's box, the repartitions in that box choosing the cuts. Off by
        default (opt-in, S11 session 2). A restart on the same mesh keeps the
        file's cuts and buffer.
    migration_cap : Any
        ``migrate=True``: the migration buffer's rows per device, ``"auto"`` (the
        default: sized from one step's drift at the first segment) or an int.
    """

    mesh: Any
    cap: Optional[int] = None
    cap_headroom: float = MESH_CAP_HEADROOM
    repartition_every: int = 1
    num_samples: int = 256
    cross_caps: Any = "auto"
    force: Any = None
    migrate: bool = False
    migration_cap: Any = "auto"

    def __post_init__(self) -> None:
        if len(tuple(getattr(self.mesh, "axis_names", ()))) != 1:
            raise ValueError(
                "MeshRunConfig.mesh must be a 1-D jax.sharding.Mesh, got axes "
                f"{getattr(self.mesh, 'axis_names', None)!r}"
            )
        if self.cap is not None and int(self.cap) < 1:
            raise ValueError(f"MeshRunConfig.cap must be >= 1, got {self.cap}")
        if float(self.cap_headroom) < 1.0:
            raise ValueError("MeshRunConfig.cap_headroom must be >= 1")
        if int(self.repartition_every) < 0:
            raise ValueError("MeshRunConfig.repartition_every must be >= 0")
        if int(self.num_samples) < 1:
            raise ValueError("MeshRunConfig.num_samples must be >= 1")
        if self.migration_cap != "auto" and int(self.migration_cap) < 1:
            raise ValueError(
                f"MeshRunConfig.migration_cap must be 'auto' or >= 1, got "
                f"{self.migration_cap!r}"
            )


# --------------------------------------------------------------------------
# The launch environment: XLA's fusion autotuner.
# --------------------------------------------------------------------------

#: XLA's fusion autotuner times the native against the block-level (Triton) emitter
#: for some fusions, so on a busy card a fresh process can compile other code from the
#: same HLO. S3 session 2 (A100s shared with other jobs, a 2e6 disc with an NFW field):
#: 8 of 12 fresh-process resumes differed from the uninterrupted run in 436,234 of 12M
#: elements (max 4.4e-6); with the autotuner off all 8 were bitwise, at no measurable
#: step, peak or compile cost on L1-L3.
_FUSION_AUTOTUNER_FLAG = "--xla_gpu_experimental_enable_fusion_autotuner"
#: whether this process has warned that a GPU run leaves the autotuner on
_FUSION_AUTOTUNER_WARNED = False


def _fusion_autotuner_off(xla_flags: Optional[str]) -> bool:
    """Whether ``xla_flags`` turns XLA's fusion autotuner off.

    XLA parses ``XLA_FLAGS`` left to right, so the last setting of a flag wins; a
    boolean flag is ``false`` or ``0`` (any case), ``true`` or ``1``, and bare means
    ``true`` (any other value stops XLA at start-up).

    Parameters
    ----------
    xla_flags : Optional[str]
        The value of ``XLA_FLAGS``; ``None`` when it is unset.

    Returns
    -------
    bool
        ``True`` when the last setting is false.
    """
    off = False
    for token in (xla_flags or "").split():
        name, sep, value = token.partition("=")
        if name == _FUSION_AUTOTUNER_FLAG:
            off = bool(sep) and value.lower() in ("false", "0")
    return off


def _warn_fusion_autotuner(platform: str, xla_flags: Optional[str]) -> bool:
    """Warn, once per process, when a GPU run leaves XLA's fusion autotuner on.

    Parameters
    ----------
    platform : str
        The platform the run's state lives on (``"gpu"``, ``"cpu"``, ...).
    xla_flags : Optional[str]
        The value of ``XLA_FLAGS``; ``None`` when it is unset.

    Returns
    -------
    bool
        Whether this call warned.
    """
    global _FUSION_AUTOTUNER_WARNED
    if (
        platform != "gpu"
        or _FUSION_AUTOTUNER_WARNED
        or _fusion_autotuner_off(xla_flags)
    ):
        return False
    _FUSION_AUTOTUNER_WARNED = True
    warnings.warn(
        f"XLA_FLAGS does not set {_FUSION_AUTOTUNER_FLAG}=false: XLA's fusion "
        "autotuner picks some fusions' emitters by timing, so on a shared GPU a "
        "resumed run (or a rerun) may not be bitwise the uninterrupted one. Set the "
        "flag in the launch environment, before jax starts (docs/production.md, "
        '"Launch environment").',
        stacklevel=3,
    )
    return True


# --------------------------------------------------------------------------
# Diagnostics: moments of the particle state, fp64 on device.
# --------------------------------------------------------------------------

_MOMENTS_FN: list = []


def _moments(state: Any, masses: Any) -> dict:
    """Total mass, momentum, angular momentum, mass moment and kinetic energy.

    One small jitted reduction in float64 (the float32 state is converted inside
    the fused reductions, never materialised), fetched to the host.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]``.
    masses : Any
        ``[N]``.

    Returns
    -------
    dict
        Host float64 values: ``mass``, ``momentum`` ``[3]``, ``angular_momentum``
        ``[3]``, ``mass_moment`` ``[3]`` (``sum m x``), ``kinetic``,
        ``momentum_scale`` (``sum m |v|``).
    """
    import jax
    import jax.numpy as jnp

    if not _MOMENTS_FN:

        def moments(state_in: Any, masses_in: Any) -> dict:
            x = state_in[:, 0].astype(jnp.float64)
            v = state_in[:, 1].astype(jnp.float64)
            m = masses_in.astype(jnp.float64)
            return {
                "mass": jnp.sum(m),
                "momentum": jnp.sum(m[:, None] * v, axis=0),
                "angular_momentum": jnp.sum(m[:, None] * jnp.cross(x, v), axis=0),
                "mass_moment": jnp.sum(m[:, None] * x, axis=0),
                "kinetic": 0.5 * jnp.sum(m * jnp.sum(v * v, axis=1)),
                "momentum_scale": jnp.sum(m * jnp.sqrt(jnp.sum(v * v, axis=1))),
            }

        _MOMENTS_FN.append(jax.jit(moments))
    values = jax.device_get(_MOMENTS_FN[0](state, masses))
    return {k: np.asarray(v, np.float64) for k, v in values.items()}


def _baselines(moments: dict, time_now: float) -> dict:
    """Step 0's moments, the reference of every drift (JSON-able).

    Parameters
    ----------
    moments : dict
        :func:`_moments` at step 0.
    time_now : float
        The time of step 0.

    Returns
    -------
    dict
        ``time``, ``mass``, ``momentum``, ``angular_momentum``, ``com``,
        ``kinetic``, ``momentum_scale``, and ``energy`` / ``potential`` (``None``
        until step 0's potential is evaluated, after the first segment).
    """
    mass = float(moments["mass"])
    return {
        "time": float(time_now),
        "mass": mass,
        "momentum": moments["momentum"].tolist(),
        "angular_momentum": moments["angular_momentum"].tolist(),
        "com": (moments["mass_moment"] / mass).tolist(),
        "kinetic": float(moments["kinetic"]),
        "momentum_scale": float(moments["momentum_scale"]),
        "energy": None,
        "potential": None,
    }


def _energy_fields(kinetic: float, potential: Optional[dict], base: dict) -> dict:
    """The energy diagnostic's fields: ``W``, ``E = K + W`` and its drift.

    Parameters
    ----------
    kinetic : float
        ``K`` at the edge (fp64).
    potential : Optional[dict]
        :meth:`ProductionRunner.potential_energy` at the edge, or ``None`` when it
        cannot be evaluated yet (step 0, before the first segment's provider).
    base : dict
        :func:`_baselines` (``energy``: ``E0``, or ``None``).

    Returns
    -------
    dict
        ``potential_self``, ``potential_external``, ``potential``, ``energy``,
        ``energy_drift`` (``(E - E0) / |E0|``) and ``energy_ok`` (the potential's
        rebuild fitted its capacities); ``None`` where unknown or not fitted.
    """
    fields: dict = dict(
        potential_self=None,
        potential_external=None,
        potential=None,
        energy=None,
        energy_drift=None,
        energy_ok=None if potential is None else bool(potential["ok"]),
    )
    if potential is None or not fields["energy_ok"]:
        return fields
    w = potential["self"] + potential["external"]
    energy = float(kinetic) + w
    e0 = base.get("energy")
    fields.update(
        potential_self=potential["self"],
        potential_external=potential["external"],
        potential=w,
        energy=energy,
        energy_drift=(energy - e0) / abs(e0) if e0 else None,
    )
    return fields


def _diagnostics_row(
    moments: dict,
    base: dict,
    time_now: float,
    names: Sequence,
    energy_due: bool,
    potential: Optional[dict] = None,
) -> dict:
    """The diagnostics of one edge, against step 0's ``base``.

    Drifts: ``momentum_drift = |P - P0| / sum m |v|_0`` (a scale that is not ~0
    when P0 is); ``angular_momentum_drift = |L - L0| / |L0|``; ``com_drift = |COM
    - COM0 - P0 / M (t - t0)|``, the centre of mass against its free motion;
    ``kinetic_change = (K - K0) / K0``; and with the energy ``energy_drift = (E -
    E0) / |E0|`` (:func:`_energy_fields`). An external field changes P and L
    physically; the drifts then measure that, not an error.

    Parameters
    ----------
    moments : dict
        :func:`_moments` at the edge.
    base : dict
        :func:`_baselines`.
    time_now : float
        The edge's time.
    names : Sequence
        The diagnostics to report (:data:`DIAGNOSTICS`).
    energy_due : bool
        Whether the energy diagnostic is due at this edge.
    potential : Optional[dict]
        The edge's potential energies (:meth:`ProductionRunner.potential_energy`),
        when due and available.

    Returns
    -------
    dict
        Host floats and lists (JSON-able).
    """
    mass = float(moments["mass"])
    p0 = np.asarray(base["momentum"])
    row: dict = {"mass": mass}
    if "momentum" in names:
        p = moments["momentum"]
        scale = base["momentum_scale"]
        row["momentum"] = p.tolist()
        row["momentum_drift"] = (
            float(np.linalg.norm(p - p0) / scale) if scale > 0 else None
        )
    if "angular_momentum" in names:
        lv = moments["angular_momentum"]
        l0 = np.asarray(base["angular_momentum"])
        norm0 = float(np.linalg.norm(l0))
        row["angular_momentum"] = lv.tolist()
        row["angular_momentum_drift"] = (
            float(np.linalg.norm(lv - l0) / norm0) if norm0 > 0 else None
        )
    if "com" in names:
        com = moments["mass_moment"] / mass
        free = np.asarray(base["com"]) + p0 / base["mass"] * (
            float(time_now) - base["time"]
        )
        row["com"] = com.tolist()
        row["com_drift"] = float(np.linalg.norm(com - free))
    if "kinetic" in names:
        kin = float(moments["kinetic"])
        row["kinetic"] = kin
        row["kinetic_change"] = (
            (kin - base["kinetic"]) / base["kinetic"] if base["kinetic"] else None
        )
    if "energy" in names and energy_due:
        row.update(_energy_fields(float(moments["kinetic"]), potential, base))
    return row


def _plan_record(engine: Any, handle: Any) -> dict:
    """The trace-time plan a segment ran with (host values, JSON-able).

    Parameters
    ----------
    engine : Any
        The runtime engine.
    handle : Any
        The segment's :class:`~jaccpot.runtime.strict_carry.StrictParticleCarry`,
        or ``None`` before the first.

    Returns
    -------
    dict
        ``leaf_capacity``, ``walk_caps`` (the validated caps the traced walk is
        sized from), ``level_widths`` / ``level_counts`` (the level registry, as
        ``[total_nodes, num_internal, value]`` rows), ``upward_num_levels`` (the
        depth bound) and ``force_key``.
    """
    from jaccpot.runtime import _level_shapes

    caps = getattr(engine, "_strict_fused_validated_caps", None)
    force_key = getattr(handle, "force_key", None)
    return {
        "leaf_capacity": getattr(engine, "_tree_leaf_capacity", None),
        "walk_caps": None if caps is None else _jsonable(dict(caps)),
        "level_widths": [
            [int(a), int(b), int(w)] for (a, b), w in _level_shapes._WIDTHS.items()
        ],
        "level_counts": [
            [int(a), int(b), int(n)] for (a, b), n in _level_shapes._LEVELS.items()
        ],
        "upward_num_levels": getattr(engine, "_cells_upward_num_levels", None),
        "force_key": None if force_key is None else [float(v) for v in force_key],
    }


@contextlib.contextmanager
def _trace_state_kept(engine: Any) -> Iterator[None]:
    """Leave the engine's trace-time state as it was, whatever runs inside.

    Tracing a program that rebuilds a topology -- the energy diagnostic's -- runs
    the rebuild's trace-time side effects on the engine (the walk's traced caps and
    needs, the refresh side channels, the level stash) and on the level registry. A
    LATER scan trace reads some of them (the second segment's donated program, a
    remainder segment, a re-plan's), so a diagnostic must leave nothing behind:
    with and without it the run is the same, bitwise (S3 R4). Every attribute of
    the engine is put back, the objects and, for dicts, their contents; attributes
    the trace added are removed; the level registry is restored.

    Parameters
    ----------
    engine : Any
        The runtime engine.

    Yields
    ------
    None
        Inside the block.
    """
    from jaccpot.runtime import _level_shapes

    saved = dict(vars(engine))
    contents = {k: dict(v) for k, v in saved.items() if isinstance(v, dict)}
    registry = (dict(_level_shapes._WIDTHS), dict(_level_shapes._LEVELS))
    try:
        yield
    finally:
        for key in set(vars(engine)) - set(saved):
            delattr(engine, key)
        for key, value in saved.items():
            setattr(engine, key, value)
            if key in contents:
                value.clear()
                value.update(contents[key])
        for live, kept in zip((_level_shapes._WIDTHS, _level_shapes._LEVELS), registry):
            live.clear()
            live.update(kept)


def _jsonable(value: Any) -> Any:
    """``value`` with numpy scalars as Python numbers and tuples as lists.

    Parameters
    ----------
    value : Any
        Nested dicts, lists, tuples and scalars.

    Returns
    -------
    Any
        The same, JSON-able.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# --------------------------------------------------------------------------
# Built-in callbacks.
# --------------------------------------------------------------------------


class SnapshotWriter:
    """A callback that keeps the state every ``every`` segments.

    To ``directory`` as ``<prefix>_<step:08d>.npz`` (``state``, ``step``, ``time``;
    written to a temporary file and renamed), or in memory (``directory=None``:
    :attr:`snapshots`, host copies -- small runs only).

    Parameters
    ----------
    directory : Optional[str]
        Output directory (created), or ``None`` for memory.
    every : int
        Segments between snapshots.
    initial : bool
        Also the initial state (segment 0).
    prefix : str
        File name prefix.

    Raises
    ------
    ValueError
        If ``every`` is below 1.
    """

    def __init__(
        self,
        directory: Optional[str] = None,
        *,
        every: int = 1,
        initial: bool = True,
        prefix: str = "snapshot",
    ) -> None:
        if int(every) < 1:
            raise ValueError("SnapshotWriter.every must be >= 1")
        self.directory = directory
        self.every = int(every)
        self.initial = bool(initial)
        self.prefix = str(prefix)
        self.snapshots: list = []
        self.paths: list = []
        if directory is not None:
            os.makedirs(directory, exist_ok=True)

    def __call__(self, edge: SegmentEdge) -> None:
        """Keep ``edge``'s state if it is due.

        Parameters
        ----------
        edge : SegmentEdge
            The edge.
        """
        if edge.segment == 0 and not self.initial:
            return
        if edge.segment % self.every:
            return
        state = edge.state_host()
        if self.directory is None:
            self.snapshots.append(
                {"step": edge.step, "time": edge.time, "state": state}
            )
            return
        path = os.path.join(self.directory, f"{self.prefix}_{edge.step:08d}.npz")
        _atomic_savez(
            path,
            state=state,
            step=np.asarray(edge.step),
            time=np.asarray(edge.time, np.float64),
        )
        self.paths.append(path)


class DiagnosticsLog:
    """A callback that logs each edge's diagnostics, one JSON line per edge.

    Parameters
    ----------
    path : Optional[str]
        A ``.jsonl`` file appended to (flushed per line), or ``None`` to keep the
        rows in :attr:`rows` only (they are kept either way).
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.rows: list = []

    def __call__(self, edge: SegmentEdge) -> None:
        """Log ``edge``.

        Parameters
        ----------
        edge : SegmentEdge
            The edge.
        """
        caps = edge.plan.get("walk_caps") or {}
        row = {
            "step": edge.step,
            "time": edge.time,
            "segment": edge.segment,
            "retries": edge.retries,
            "wall_s": edge.wall_s,
            "diagnostics": edge.diagnostics,
            "leaf_capacity": edge.plan.get("leaf_capacity"),
            "far_capacity": caps.get("compact_far_pair_capacity"),
            "near_capacity": caps.get("near_edge_capacity"),
        }
        self.rows.append(row)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()


#: the restart file's ``meta["format"]``
RESTART_FORMAT = "jaccpot.production.restart/1"


def load_restart(path: Union[str, "os.PathLike"]) -> tuple:
    """A restart file's arrays and ``meta`` (:meth:`ProductionRunner.save_restart`).

    Parameters
    ----------
    path : Union[str, os.PathLike]
        The file.

    Returns
    -------
    tuple
        ``(arrays, meta)``: a dict of host arrays (``state``, ``masses``,
        ``self_acceleration``, and ``acceleration`` / ``force_scale`` where the
        run had them; a multi-GPU file, whose ``meta["mesh"]`` is set, holds the
        total ``acceleration`` and no ``self_acceleration``, plus the layout's
        ``mesh_ids`` and ``mesh_padding``) and the decoded JSON.

    Raises
    ------
    ValueError
        If the file is not a restart of this format.
    """
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    meta = json.loads(arrays.pop("meta", np.zeros(0, np.uint8)).tobytes() or b"{}")
    if meta.get("format") != RESTART_FORMAT:
        raise ValueError(
            f"{path} is not a {RESTART_FORMAT} file (format {meta.get('format')!r})"
        )
    return arrays, meta


def _template_shapes(template: Any) -> list:
    """A shape template's leaves as JSON: ``[shape, dtype]``, or ``None`` (static).

    Parameters
    ----------
    template : Any
        A :class:`~jaccpot.runtime.strict_carry.PreparedShapeTemplate`.

    Returns
    -------
    list
        One entry per leaf.
    """
    import jax

    return [
        ([list(x.shape), str(x.dtype)] if isinstance(x, jax.ShapeDtypeStruct) else None)
        for x in template.leaves
    ]


def _template_with_shapes(template: Any, shapes: Optional[list]) -> Any:
    """``template`` with a restart's leaf shapes, or ``None`` if they do not fit.

    The structure (treedef, static leaves, which leaves are arrays, dtypes) stays
    the template's and must match; only array shapes are taken from ``shapes``.

    Parameters
    ----------
    template : Any
        The restore prepare's template.
    shapes : Optional[list]
        :func:`_template_shapes` of the run's template (``None``: not recorded).

    Returns
    -------
    Any
        The template with the run's shapes, or ``None``.
    """
    import jax

    from jaccpot.runtime.strict_carry import PreparedShapeTemplate

    if shapes is None or len(shapes) != len(template.leaves):
        return None
    leaves = []
    for leaf, saved in zip(template.leaves, shapes):
        is_array = isinstance(leaf, jax.ShapeDtypeStruct)
        if is_array != (saved is not None):
            return None
        if not is_array:
            leaves.append(leaf)
            continue
        if str(leaf.dtype) != saved[1] or len(leaf.shape) != len(saved[0]):
            return None
        leaves.append(jax.ShapeDtypeStruct(tuple(int(v) for v in saved[0]), leaf.dtype))
    return PreparedShapeTemplate(treedef=template.treedef, leaves=tuple(leaves))


def _key_digest(key: Any) -> str:
    """sha256 of a compile key's repr (no object addresses in it: process-free).

    Parameters
    ----------
    key : Any
        ``PreparedShapeTemplate.key()`` or ``FusedForceProvider.key()``.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(repr(key).encode()).hexdigest()


def _qualname(fn: Any) -> Optional[str]:
    """``module.qualname`` of a function, the field's name across processes.

    Parameters
    ----------
    fn : Any
        A function, or ``None``.

    Returns
    -------
    Optional[str]
        The name, or ``None``.
    """
    if fn is None:
        return None
    module = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None) or type(fn).__qualname__
    return f"{module}.{name}"


def _versions() -> dict:
    """Versions of the stack and the git commit of each source checkout.

    Returns
    -------
    dict
        ``jax``, ``jaxlib``, ``numpy`` versions; ``jaccpot`` and ``yggdrax`` git
        HEADs (``None`` outside a checkout).
    """
    import importlib
    import importlib.metadata
    import subprocess
    from pathlib import Path

    out: dict = {}
    for name in ("jax", "jaxlib", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    for name in ("jaccpot", "yggdrax"):
        try:
            where = Path(importlib.import_module(name).__file__).resolve().parent
            head = subprocess.run(
                ["git", "-C", str(where), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            out[name] = head.stdout.strip() if head.returncode == 0 else None
        except (ImportError, OSError, subprocess.SubprocessError):
            out[name] = None
    return out


def _atomic_savez(path: str, **arrays: Any) -> None:
    """``np.savez`` to ``path`` through a temporary file and a rename.

    Parameters
    ----------
    path : str
        Output file.
    **arrays : Any
        The entries.
    """
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# The runner.
# --------------------------------------------------------------------------


class ProductionRunner:
    """A long fused-lane run: ``strict_run_v2(carry="particles")`` segments on a host loop.

    Each segment is one ``strict_run_v2`` call of ``segment.steps`` steps that
    continues from the previous call's handle, with ``carry_total_acceleration``,
    so the run computes what ONE call of all its steps computes, bit for bit
    (``tests/integration/strict/test_production_runner.py``). Between segments,
    at an EDGE: the diagnostics, a restart file when due, then the callbacks.

    * Donation: the first segment consumes the caller's state only with
      ``SegmentConfig.donate_input``; later segments always consume their input
      (the runner's own), so the scan writes the next state into the last one's
      buffer. ``strict_run_v2`` keeps a host copy of a donated start state for its
      segment retry (24 B per particle on the host, one device-to-host copy per
      segment).
    * Parameters: ``SegmentConfig.traced_params`` (the default) passes ``dt``,
      ``G``, the softening squared (not under ``dehnen_error``, the galaxy
      preset's MAC) and the field's float parameters as arrays.
    * Retry: a segment whose lists or tree overflow is re-planned and re-run by
      ``strict_run_v2`` itself, once (twice when the first re-plan grew the tree:
      the second for the walk only); a failure after that raises, naming the last
      restart.
    * Callbacks: ``Callable[[SegmentEdge], Optional[bool]]``, called at step 0 and
      after every segment; a truthy return stops the run after that edge.
      ``step_callback`` is ``strict_run_v2``'s in-scan hook, passed through: its
      step index counts from 0 in EACH segment.
    * Launch environment: on a GPU, :meth:`run` warns once per process when
      ``XLA_FLAGS`` leaves XLA's fusion autotuner on (see the module docstring).
    * Multi-GPU: with ``mesh`` the same loop runs on the fused multi-GPU lane's
      segment runner (:mod:`jaccpot.distributed.mesh_runner`; docs/production.md,
      "Multi-GPU mode").

    Parameters
    ----------
    solver : Any
        ``FastMultipoleMethod`` on the fused lane (``preset="galaxy"``, or any
        ``large_n_gpu`` solver on cell leaves).
    settings : GalaxySettings
        The run-time half read here: ``leaf_size``, ``max_order`` and ``carry``
        (``"particles"`` only).
    segment : SegmentConfig
        Segment length, parameters, donation, diagnostics, restarts.
    external_acceleration_fn : Optional[Callable[..., Any]]
        ``field(state)`` or, with ``external_params``, ``field(state,
        external_params)`` -> ``[N, 3]`` (``strict_run_v2``'s contract: the packed
        state, positions only read). Keep ONE function object for the run: its
        identity is part of the compile keys.
    external_potential_fn : Optional[Callable[..., Any]]
        The external field's potential per unit mass, for the energy diagnostic:
        ``potential(state)`` or ``potential(state, external_params)`` -> ``[N]``,
        the same contract and parameters as ``external_acceleration_fn`` (its
        gradient is minus that field). ``None``: no external potential energy
        (``W_ext = 0``; with a field, the energy then omits it).
    callbacks : Sequence[Callable[[SegmentEdge], Optional[bool]]]
        Host-side, at every edge, in order.
    step_callback : Optional[Callable[[Any, Any], None]]
        ``strict_run_v2``'s in-scan streaming hook.
    step_callback_stride : int
        Its stride.
    mesh : Any
        ``None`` (the default): one card. A 1-D ``jax.sharding.Mesh`` or a
        :class:`MeshRunConfig`: the multi-GPU mode. There the external field
        and potential see one device's ``(cap, 2, 3)`` rows (the same contract
        restricted to rows: a per-particle field works unchanged), ``dt`` and
        the field's parameters are program arguments, ``G`` and the softening
        the force's own, and there is no ``step_callback``.

    Raises
    ------
    ValueError
        If ``settings.carry`` is not ``"particles"``, or with ``mesh`` and a
        ``step_callback``.

    Notes
    -----
    THE ENERGY. ``E = K + W_self + W_ext``: ``K = 1/2 sum m v^2``, ``W_self = 1/2
    sum m phi`` with the fused lane's potential at the edge's positions
    (:meth:`potential_energy`: a rebuild there and ``FusedForceProvider.potential``,
    in a program of its own, compiled once per plan) and ``W_ext = sum m Phi_ext``,
    each summed in fp64 on device. Step 0's ``E0`` needs a provider, so it is
    evaluated at the first diagnostics edge after a segment, from step 0's state
    (kept: the caller's array, or a host copy when ``donate_input`` consumes it);
    the step-0 edge reports its energy fields as ``None``. The energy program's
    trace leaves the engine as it found it (:func:`_trace_state_kept`), so the
    trajectory is the one without diagnostics, bitwise.
    """

    def __init__(
        self,
        solver: Any,
        *,
        settings: GalaxySettings = GALAXY_SETTINGS,
        segment: SegmentConfig = SegmentConfig(),
        external_acceleration_fn: Optional[Callable[..., Any]] = None,
        external_potential_fn: Optional[Callable[..., Any]] = None,
        callbacks: Sequence[Callable[[SegmentEdge], Optional[bool]]] = (),
        step_callback: Optional[Callable[[Any, Any], None]] = None,
        step_callback_stride: int = 1,
        mesh: Any = None,
    ) -> None:
        if settings.carry != "particles":
            raise ValueError(
                "ProductionRunner chains strict_run_v2(carry='particles') calls "
                f"through their handle; settings.carry={settings.carry!r}"
            )
        self.solver = solver
        self.engine = getattr(solver, "_impl", solver)
        self.settings = settings
        self.segment = segment
        self.external_acceleration_fn = external_acceleration_fn
        self.external_potential_fn = external_potential_fn
        self.callbacks = tuple(callbacks)
        self.step_callback = step_callback
        self.step_callback_stride = int(step_callback_stride)
        #: (steps, remainder) plans already warned about
        self._warned: set = set()
        #: the run's position at its last edge (what save_restart writes)
        self._current: Optional[dict] = None
        self._last_restart: Optional[str] = None
        #: the energy programs, by plan (potential_energy)
        self._energy_programs: dict = {}
        #: the multi-GPU mode's configuration and backend (``None``: one card)
        self.mesh: Optional[MeshRunConfig] = None
        self._mesh_backend: Any = None
        if mesh is not None:
            from jaccpot.distributed.mesh_runner import MeshBackend

            if step_callback is not None:
                raise ValueError(
                    "the multi-GPU mode has no in-scan step_callback (the mesh "
                    "segment runner's scan streams nothing); use a callback at the "
                    "segment edges"
                )
            self.mesh = mesh if isinstance(mesh, MeshRunConfig) else MeshRunConfig(mesh)
            self._mesh_backend = MeshBackend(self)

    # ------------------------------------------------------------------
    # The loop.
    # ------------------------------------------------------------------

    def run(
        self,
        state: Any = None,
        masses: Any = None,
        *,
        dt: Union[float, Any],
        num_steps: int,
        t0: float = 0.0,
        force_params: Any = None,
        external_params: Any = None,
        restart_from: Any = None,
    ) -> RunResult:
        """Run to step ``num_steps`` (counted from ``t0``), segment by segment.

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]`` positions and velocities; ``None`` with ``restart_from``.
        masses : Any
            ``[N]``; ``None`` with ``restart_from``.
        dt : Union[float, Any]
            Timestep.
        num_steps : int
            The run's total length in steps from ``t0``: a continued run
            (``restart_from``) runs the steps still missing.
        t0 : float
            Start time (a continued run keeps its own).
        force_params : Any
            ``ForceParams(G, eps2)``; ``None`` takes the solver's.
        external_params : Any
            The external field's parameters (a pytree), or ``None`` for a
            one-argument field.
        restart_from : Any
            A restart file (:meth:`save_restart`) to continue from -- in a fresh
            process too --, a :class:`RunResult` of this runner to continue in
            memory (its state and handle are consumed), or ``None``.

        Returns
        -------
        RunResult
            The final state, handle, counters and the edges of this call. In the
            multi-GPU mode (``mesh=``; :mod:`jaccpot.distributed.mesh_runner`)
            ``state`` and ``masses`` are gathered by id on the host, ``handle`` is
            the ``MeshSegmentRunner`` and ``mode`` is ``"mesh"``.

        Raises
        ------
        ValueError
            Without ``state`` and ``masses`` for a fresh run, or for a restart file
            of another format.
        TypeError
            For a ``restart_from`` that is neither.
        RuntimeError
            When a segment fails after ``strict_run_v2``'s own re-plan; the message
            names the last restart file.
        """
        if self._mesh_backend is not None:
            return self._mesh_backend.run(
                state,
                masses,
                dt=dt,
                num_steps=num_steps,
                t0=t0,
                force_params=force_params,
                external_params=external_params,
                restart_from=restart_from,
            )
        import jax
        import jax.numpy as jnp

        from jaccpot._env import env_text
        from jaccpot.runtime.strict_carry import host_copy

        started = time.perf_counter()
        dt_value = float(np.asarray(dt))
        restored: Optional[dict] = None
        if restart_from is None:
            if state is None or masses is None:
                raise ValueError("a fresh run needs state and masses")
            state_now = jnp.asarray(state)
            masses_now = jnp.asarray(masses)
            handle = None
            step, segment_done = 0, 0
            t0_run = float(t0)
            anchor: tuple = (t0_run, 0)
            baselines: Optional[dict] = None
            # the caller's state: consumed by the first segment only on request
            owned = bool(self.segment.donate_input)
        elif isinstance(restart_from, RunResult):
            res = restart_from
            if res.mode != "single":
                raise ValueError(
                    f"a {res.mode!r} RunResult continues in its own mode: pass the "
                    "runner its mesh (ProductionRunner(..., mesh=...))"
                )
            state_now, masses_now, handle = res.state, res.masses, res.handle
            step, segment_done, t0_run = res.step, res.segment, res.t0
            # edge times count from the same anchor while dt is unchanged
            anchor = res.time_anchor if dt_value == res.dt else (res.time, res.step)
            baselines = res.baselines
            owned = bool(res.state_owned) or bool(self.segment.donate_input)
            restored = {"from": "RunResult", "step": step, "bitwise": True}
            res.state = res.handle = None  # consumed
        elif isinstance(restart_from, (str, os.PathLike)):
            arrays, meta = load_restart(restart_from)
            if meta.get("mesh") is not None:
                raise ValueError(
                    f"{restart_from} was written by the multi-GPU mode (on "
                    f"{meta['mesh'].get('ndev')} devices): continue it with "
                    "ProductionRunner(..., mesh=...)"
                )
            state_now = jnp.asarray(arrays["state"])
            masses_now = jnp.asarray(arrays["masses"])
            counters = meta["counters"]
            step, segment_done = int(counters["step"]), int(counters["segment"])
            t0_run = float(counters["t0"])
            anchor = (
                tuple(counters["anchor"])
                if dt_value == float(counters["dt"])
                else (float(counters["time"]), step)
            )
            baselines = meta["baselines"]
            owned = True  # the file's state, on device: the runner's own
            handle = None  # below, once the call's parameters are known
        else:
            raise TypeError(
                "restart_from must be a RunResult or a restart file, got "
                f"{type(restart_from).__name__}"
            )
        _warn_fusion_autotuner(
            next(iter(state_now.devices())).platform, env_text("XLA_FLAGS")
        )
        dt_arg, force_arg, external_arg = self._call_args(
            dt, force_params, external_params, state_now.dtype
        )
        if isinstance(restart_from, (str, os.PathLike)):
            handle, restored = self._restore(
                str(restart_from),
                arrays,
                meta,
                state_now,
                masses_now,
                dt_value=dt_value,
                force_arg=force_arg,
                external_arg=external_arg,
            )
            del arrays

        def time_at(step_now: int) -> float:
            return anchor[0] + (step_now - anchor[1]) * dt_value

        names = tuple(self.segment.diagnostics)
        every = int(self.segment.diagnostics_every)
        every_energy = int(self.segment.diagnostics_every_energy)
        energy_on = every > 0 and every_energy > 0 and "energy" in names
        edges: list = []
        restarts: list = []
        stopped = False
        # step 0's state, for its potential once a provider exists (the first
        # segment's): the caller's array, or a host copy when it is consumed
        start_state: Any = None

        def diagnostics_at(step_now: int, state_in: Any, due: bool) -> Optional[dict]:
            nonlocal baselines, start_state
            if every == 0 or not due:
                return None
            moments = _moments(state_in, masses_now)
            if baselines is None:
                baselines = dict(_baselines(moments, time_at(step_now)), step=step_now)
            energy_due = every_energy > 0 and segment_done % every_energy == 0
            potential = None
            if energy_on and handle is not None:
                if start_state is not None and baselines.get("energy") is None:
                    start = self.potential_energy(
                        jax.device_put(start_state, masses_now.sharding),
                        masses_now,
                        handle,
                        external_params=external_arg,
                    )
                    if start["ok"]:
                        w0 = start["self"] + start["external"]
                        baselines["potential"] = w0
                        baselines["energy"] = float(baselines["kinetic"]) + w0
                start_state = None
                if energy_due:
                    potential = self.potential_energy(
                        state_in, masses_now, handle, external_params=external_arg
                    )
            return _diagnostics_row(
                moments, baselines, time_at(step_now), names, energy_due, potential
            )

        def edge_at(retries: int, wall_s: float, due: bool) -> SegmentEdge:
            return SegmentEdge(
                step=step,
                time=time_at(step),
                segment=segment_done,
                retries=retries,
                wall_s=wall_s,
                state=state_now,
                masses=masses_now,
                self_acceleration=getattr(handle, "self_acceleration", None),
                diagnostics=diagnostics_at(step, state_now, due),
                plan=_plan_record(self.engine, handle),
            )

        def at_edge(edge: SegmentEdge) -> bool:
            # the position save_restart writes, the restart if due, the callbacks
            self._current = dict(
                state=state_now,
                masses=masses_now,
                handle=handle,
                step=step,
                time=edge.time,
                dt=dt_value,
                t0=t0_run,
                segment=segment_done,
                num_steps=int(num_steps),
                baselines=baselines,
                anchor=anchor,
                external_params=external_arg,
            )
            restart = self._restart_due(segment_done)
            if restart is not None:
                restarts.append(restart)
            stop = False
            for callback in self.callbacks:
                result = callback(edge)
                if result is not None and bool(result):
                    stop = True
            edges.append(
                dataclasses.replace(
                    edge, state=None, masses=None, self_acceleration=None
                )
            )
            return stop

        if restart_from is None:
            if energy_on:
                # a donated input: an asynchronous copy (pinned, on a GPU)
                start_state = host_copy(state_now) if owned else state_now
            stopped = at_edge(edge_at(0, 0.0, due=True))
        missing = max(0, int(num_steps) - step)
        lengths = () if stopped else segment_lengths(missing, self.segment.steps)
        self._warn_remainder(lengths)
        external = self.external_acceleration_fn
        ran = False
        for index, steps in enumerate(lengths):
            retries_before = int(
                getattr(self.engine, "_strict_fused_fallback_count", 0)
            )
            begun = time.perf_counter()
            try:
                state_now, handle, _ = self.solver.strict_run_v2(
                    state=state_now,
                    masses=masses_now,
                    dt=dt_arg,
                    num_steps=int(steps),
                    refresh_every=1,
                    leaf_size=int(self.settings.leaf_size),
                    max_order=int(self.settings.max_order),
                    prepared_state=handle,
                    return_prepared_state=True,
                    add_external=external is not None,
                    external_acceleration_fn=external,
                    external_params=external_arg,
                    step_callback=self.step_callback,
                    step_callback_stride=self.step_callback_stride,
                    carry="particles",
                    donate_state=owned or index > 0,
                    force_params=force_arg,
                    carry_total_acceleration=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"ProductionRunner: segment {segment_done + 1} (steps {step} to "
                    f"{step + int(steps)}) failed after strict_run_v2's own re-plan. "
                    f"Last restart: {self._last_restart or 'none written'}. {exc}"
                ) from exc
            jax.block_until_ready(state_now)
            wall_s = time.perf_counter() - begun
            ran = True
            step += int(steps)
            segment_done += 1
            last = index == len(lengths) - 1
            retries = (
                int(getattr(self.engine, "_strict_fused_fallback_count", 0))
                - retries_before
            )
            due = every > 0 and (segment_done % every == 0 or last)
            if at_edge(edge_at(retries, wall_s, due)):
                stopped = True
                break
        return RunResult(
            state=state_now,
            masses=masses_now,
            handle=handle,
            step=step,
            time=time_at(step),
            dt=dt_value,
            t0=t0_run,
            segment=segment_done,
            edges=edges,
            baselines=baselines,
            stopped=stopped,
            restarts=restarts,
            restored=restored,
            wall_s=time.perf_counter() - started,
            time_anchor=anchor,
            # without a segment the state is still the one this call was given
            state_owned=owned or ran,
        )

    # ------------------------------------------------------------------
    # The energy.
    # ------------------------------------------------------------------

    def potential_energy(
        self,
        state: Any,
        masses: Any,
        handle: Any,
        *,
        external_params: Any = None,
    ) -> dict:
        """The potential energies at ``state``: ``1/2 sum m phi`` and ``sum m Phi_ext``.

        One card. (On the mesh: ``RunResult.handle.potential_energy(...)``,
        :meth:`~jaccpot.distributed.segment.MeshSegmentRunner.potential_energy`.)

        One jitted program per plan (the handle's provider key, its force
        parameters, the external potential): the topology rebuilt at ``state``'s
        positions from the provider's template, ``FusedForceProvider.potential``
        on it with the force parameters the handle's self-gravity was computed
        with (as Python floats: the energy is a diagnostic, compiled once per run),
        the external potential, and both sums in fp64 on device. Its trace leaves
        the engine's trace-time state alone (:func:`_trace_state_kept`).

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]``; positions only are read.
        masses : Any
            ``[N]``.
        handle : Any
            A :class:`~jaccpot.runtime.strict_carry.StrictParticleCarry` of this
            run: its provider (template, plan), force parameters and, under
            ``mac_type='dehnen_error'``, its force scale for the rebuild's
            thresholds.
        external_params : Any
            The external potential's parameters, as the segments pass them.

        Returns
        -------
        dict
            Host values: ``self`` (fp64 ``W_self``), ``external`` (``W_ext``, ``0``
            without ``external_potential_fn``) and ``ok`` (the rebuild fitted its
            capacities; the energies are meaningless otherwise).
        """
        import jax
        import jax.numpy as jnp

        from jaccpot.runtime.fused_topology import ForceParams

        prov = handle.provider
        field = self.external_potential_fn
        force_key = handle.force_key
        params = (
            None
            if force_key is None
            else ForceParams(G=float(force_key[0]), eps2=float(force_key[1]))
        )
        key = (prov.key(), force_key, None if field is None else id(field))
        program = self._energy_programs.get(key)
        if program is None:

            def energy(
                state_in: Any, masses_in: Any, scale_in: Any, external_in: Any
            ) -> tuple:
                topology = prov.rebuild(state_in[:, 0], masses_in, force_scale=scale_in)
                phi = prov.potential(topology, params)
                m = jnp.asarray(masses_in).astype(jnp.float64)
                w_self = 0.5 * jnp.sum(m * phi.astype(jnp.float64))
                if field is None:
                    w_ext = jnp.zeros((), jnp.float64)
                else:
                    ext = (
                        field(state_in)
                        if external_in is None
                        else field(state_in, external_in)
                    )
                    w_ext = jnp.sum(m * jnp.asarray(ext).astype(jnp.float64))
                return w_self, w_ext, prov.capacity_ok(topology)

            program = jax.jit(energy)
            self._energy_programs[key] = program
        scale = handle.force_scale if prov.scale_carry else None
        # one placement for every call: step 0's state comes committed (a host copy
        # put back on the device), the segments' states do not, and jit compiles
        # one executable per commitment -- a second ~50 s compile at 25M and 1e8
        state = jax.device_put(state, masses.sharding)
        with _trace_state_kept(self.engine):
            out = program(state, masses, scale, external_params)
        w_self, w_ext, ok = jax.device_get(out)
        return {"self": float(w_self), "external": float(w_ext), "ok": bool(ok)}

    # ------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------

    def _call_args(
        self, dt: Any, force_params: Any, external_params: Any, state_dtype: Any
    ) -> tuple:
        """``(dt, force_params, external_params)`` as the segments pass them.

        With ``traced_params`` as arrays: ``dt`` in the state's dtype, ``G`` and
        ``eps2`` in the working dtype (as ``fused_memory_budget.py
        --traced-params``; not under ``dehnen_error``), the field's floats in the
        state's dtype. The same objects every segment, so one compile key.

        Parameters
        ----------
        dt : Any
            The timestep.
        force_params : Any
            The caller's ``ForceParams`` or ``None``.
        external_params : Any
            The caller's field parameters or ``None``.
        state_dtype : Any
            The state's dtype.

        Returns
        -------
        tuple
            ``(dt, force_params, external_params)``.
        """
        import jax
        import jax.numpy as jnp

        if not self.segment.traced_params:
            return dt, force_params, external_params
        from jaccpot.runtime.fused_topology import ForceParams

        dt_arg = jnp.asarray(dt, state_dtype)
        force_arg = force_params
        if not self.engine._flat_walk_criterion_active():
            own = force_params
            if own is None:
                own = ForceParams(
                    G=float(self.engine.G), eps2=float(self.engine.softening) ** 2
                )
            dtype = self.engine.working_dtype
            force_arg = ForceParams(
                G=jnp.asarray(own.G, dtype), eps2=jnp.asarray(own.eps2, dtype)
            )
        external_arg = (
            None
            if external_params is None
            else jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, state_dtype) if isinstance(x, float) else x,
                external_params,
            )
        )
        return dt_arg, force_arg, external_arg

    def _warn_remainder(self, lengths: tuple) -> None:
        """Warn once per plan whose last segment is shorter than the others.

        Parameters
        ----------
        lengths : tuple
            :func:`segment_lengths`.
        """
        if len(lengths) < 2 or lengths[-1] == lengths[0]:
            return
        plan = (lengths[0], lengths[-1])
        if plan in self._warned:
            return
        self._warned.add(plan)
        warnings.warn(
            f"the run's last segment is {lengths[-1]} steps, the others "
            f"{lengths[0]}: it compiles its own scan (num_steps is part of the "
            "compile key; 52.8 s at 25M on an A100). Choose SegmentConfig.steps to "
            "divide num_steps.",
            stacklevel=3,
        )

    def _restart_due(self, segment_done: int) -> Optional[str]:
        """Write the restart if one is due after ``segment_done`` segments.

        Parameters
        ----------
        segment_done : int
            Segments done.

        Returns
        -------
        Optional[str]
            The file written, or ``None``.
        """
        every = int(self.segment.restart_every)
        if every == 0 or segment_done == 0 or segment_done % every:
            return None
        assert self._current is not None and self.segment.restart_path
        path = str(self.segment.restart_path).format(
            step=self._current["step"], segment=segment_done
        )
        return self.save_restart(path)

    # ------------------------------------------------------------------
    # Restart files.
    # ------------------------------------------------------------------

    def save_restart(self, path: Union[str, "os.PathLike"]) -> str:
        """Write the run's position at its last edge to ``path`` (one ``.npz``).

        What a fresh process needs to continue the run bitwise: the state, the
        masses, the handle's self-gravity (and total acceleration with an external
        field, and f_b under ``mac_type='dehnen_error'``), the counters and step
        0's baselines, the plan the scan programs were traced with (leaf capacity,
        validated walk caps, level registry, depth bound, force key, the shape
        template's leaf shapes), the sha256 of the template's and the provider's
        compile keys, and a fingerprint of the configuration, versions,
        environment and device. Arrays are saved as
        ``np.savez`` entries, everything else as one JSON document ``meta`` (UTF-8
        bytes); the file is written to a temporary name and renamed, so a crash
        leaves the previous restart intact. Valid during a callback and after
        :meth:`run`.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Output file.

        Returns
        -------
        str
            ``path``.

        Raises
        ------
        RuntimeError
            Before the first segment (step 0 is the run's own input).
        """
        if self._mesh_backend is not None:
            return self._mesh_backend.save_restart(path)
        import jax

        current = self._current
        handle = None if current is None else current["handle"]
        if handle is None:
            raise RuntimeError(
                "save_restart needs a run that has done a segment: the state at "
                "step 0 is the run's input"
            )
        arrays = {
            "state": np.asarray(jax.device_get(current["state"])),
            "masses": np.asarray(jax.device_get(current["masses"])),
            "self_acceleration": np.asarray(jax.device_get(handle.self_acceleration)),
        }
        if handle.force_scale is not None:
            arrays["force_scale"] = np.asarray(jax.device_get(handle.force_scale))
        total = handle.acceleration
        total_is_self = total is handle.self_acceleration
        if total is not None and not total_is_self:
            arrays["acceleration"] = np.asarray(jax.device_get(total))
        field = self.external_acceleration_fn
        meta = {
            "format": RESTART_FORMAT,
            "counters": {
                "step": int(current["step"]),
                "time": float(current["time"]),
                "dt": float(current["dt"]),
                "t0": float(current["t0"]),
                "segment": int(current["segment"]),
                "num_steps": int(current["num_steps"]),
                "anchor": [float(current["anchor"][0]), int(current["anchor"][1])],
            },
            "baselines": current["baselines"],
            "plan": _plan_record(self.engine, handle),
            "digests": {
                "template_key": _key_digest(handle.template.key()),
                "provider_key": _key_digest(handle.provider.key()),
            },
            "template_shapes": _template_shapes(handle.template),
            "external": {
                "field": _qualname(field),
                "params_sha256": (
                    None if handle.external_key is None else handle.external_key[1]
                ),
                "total_is_self": bool(total_is_self),
            },
            "fingerprint": self._fingerprint(current["state"], current["masses"]),
        }
        # UTF-8 bytes (a numpy str array would hold 4 bytes per character)
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        _atomic_savez(str(path), **arrays)
        self._last_restart = str(path)
        return str(path)

    def _restore(
        self,
        path: str,
        arrays: dict,
        meta: dict,
        state: Any,
        masses: Any,
        *,
        dt_value: float,
        force_arg: Any,
        external_arg: Any,
    ) -> tuple:
        """The handle a restart file continues from, rebuilt in this process.

        The procedure: the leaf capacity and the walk caps of the run that wrote
        it, a prepare at the restart's positions (the template's structure, the
        trace-time engine state), the caps installed again, the level registry
        and depth bound set exactly and the template's leaf shapes taken from the
        file, then the digests of the template's and the provider's compile keys
        checked against the file's -- equal, the next segments trace the programs
        the run would have; not, the continuation is correct but not bitwise, and
        the runner warns and records why.

        Parameters
        ----------
        path : str
            The file (for messages).
        arrays : dict
            Its arrays.
        meta : dict
            Its ``meta``.
        state : Any
            ``[N, 2, 3]`` on device.
        masses : Any
            ``[N]`` on device.
        dt_value : float
            This run's timestep.
        force_arg : Any
            The force parameters the segments pass.
        external_arg : Any
            The field parameters the segments pass.

        Returns
        -------
        tuple
            ``(StrictParticleCarry, restored)``; ``restored`` holds ``from``,
            ``step``, ``bitwise``, ``mismatches`` and ``digests``.
        """
        import jax.numpy as jnp

        from jaccpot._env import env_flag
        from jaccpot.runtime import _level_shapes
        from jaccpot.runtime.capacity_plan import install_walk_caps
        from jaccpot.runtime.fused_provider import FusedForceProvider
        from jaccpot.runtime.strict_carry import StrictParticleCarry, external_field_key

        engine = self.engine
        n = int(state.shape[0])
        written = meta["fingerprint"]
        now = self._fingerprint(state, masses)
        mismatches = [
            key
            for key in sorted(set(written) | set(now))
            if written.get(key) != now.get(key)
        ]
        plan = meta["plan"]
        if plan["leaf_capacity"] is not None:
            engine._tree_leaf_capacity = int(plan["leaf_capacity"])
        install_walk_caps(engine, plan["walk_caps"])
        # caps-only (as the particle carry's own bootstrap): the template and the
        # trace-time state, no list and no downward pass -- the handle brings the
        # start force
        provider, topology = FusedForceProvider.prepare(
            self.solver,
            state[:, 0],
            masses,
            leaf_size=int(self.settings.leaf_size),
            max_order=int(self.settings.max_order),
            caps_only=env_flag("JACCPOT_STRICT_CAPS_ONLY_PREPARE", True),
        )
        del topology
        # what the prepare re-measured at these positions goes back to the plan
        # the run traced its programs with
        install_walk_caps(engine, plan["walk_caps"])
        for a, b, width in plan["level_widths"]:
            _level_shapes._WIDTHS[(int(a), int(b))] = int(width)
        for a, b, levels in plan["level_counts"]:
            _level_shapes._LEVELS[(int(a), int(b))] = int(levels)
        if plan["upward_num_levels"] is not None:
            engine._cells_upward_num_levels = int(plan["upward_num_levels"])
        # The template's shapes are the run's own, not the prepare's: the prepare
        # sizes an unnamed list from its count HERE (x1.5, in granules), and a
        # count that grew past a rung since the run's first prepare widens it
        # (N = 512, 4 steps: 2048 -> 4096 near edges) although the run's walk still
        # fits the validated width. The prepare's structure must be the run's.
        template = _template_with_shapes(
            provider.template_or_raise(), meta.get("template_shapes")
        )
        if template is not None:
            provider = FusedForceProvider(
                engine,
                leaf_size=int(self.settings.leaf_size),
                max_order=int(self.settings.max_order),
                runtime_overrides=provider.runtime_overrides,
                template=template,
            )
        digests = {
            "template_key": _key_digest(provider.template_or_raise().key()),
            "provider_key": _key_digest(provider.key()),
        }
        mismatches += [k for k, v in digests.items() if meta["digests"].get(k) != v]
        force_key = None if plan["force_key"] is None else tuple(plan["force_key"])
        resolved = (
            None
            if force_arg is None
            else FusedForceProvider(
                engine,
                leaf_size=int(self.settings.leaf_size),
                max_order=int(self.settings.max_order),
            ).resolve_params(force_arg)
        )
        call_key = (
            None
            if resolved is None
            else (float(np.asarray(resolved.G)), float(np.asarray(resolved.eps2)))
        )
        if call_key != force_key:
            mismatches.append("force_params")
        if dt_value != float(meta["counters"]["dt"]):
            mismatches.append("dt")
        self_acceleration = jnp.asarray(arrays["self_acceleration"])
        field = self.external_acceleration_fn
        external = meta["external"]
        external_key = external_field_key(field, external_arg)
        acceleration = None
        if external["total_is_self"] and field is None:
            acceleration = self_acceleration
        elif (
            "acceleration" in arrays
            and field is not None
            and external["field"] == _qualname(field)
            and external["params_sha256"] == external_key[1]
        ):
            acceleration = jnp.asarray(arrays["acceleration"])
        else:
            # the start total of another field (or of none): the segment evaluates
            # this one's again, eagerly
            mismatches.append("external_field")
            external_key = None
        handle = StrictParticleCarry(
            template=provider.template_or_raise(),
            self_acceleration=self_acceleration,
            num_particles=n,
            force_scale=(
                jnp.asarray(arrays["force_scale"]) if "force_scale" in arrays else None
            ),
            provider=provider,
            force_key=force_key,
            acceleration=acceleration,
            external_key=external_key if acceleration is not None else None,
        )
        if mismatches:
            warnings.warn(
                f"restart {path}: {mismatches} differ from the run that wrote it; "
                "the continuation is correct but not bitwise the uninterrupted run",
                stacklevel=3,
            )
        restored = {
            "from": path,
            "step": int(meta["counters"]["step"]),
            "bitwise": not mismatches,
            "mismatches": mismatches,
            "digests": digests,
        }
        return handle, restored

    def _fingerprint(self, state: Any, masses: Any) -> dict:
        """What decides a run's numbers besides its arrays (JSON-able).

        Parameters
        ----------
        state : Any
            ``[N, 2, 3]``.
        masses : Any
            ``[N]``.

        Returns
        -------
        dict
            N and dtypes, the solver's configuration, the settings, whether the
            parameters are traced, the ``JACCPOT_*`` / ``YGGDRAX_*`` environment,
            the XLA flags, the device, x64, and the versions (:func:`_versions`).
        """
        import jax
        import jax.numpy as jnp

        from jaccpot._env import env_snapshot, env_text

        engine = self.engine
        if getattr(self, "_static_fingerprint", None) is None:
            self._static_fingerprint = {"versions": _versions()}
        device = jax.devices()[0]
        return {
            "n": int(state.shape[0]),
            "dtypes": [str(state.dtype), str(masses.dtype)],
            "solver": _jsonable(
                {
                    "G": float(engine.G),
                    "softening": float(engine.softening),
                    "softening_kernel": str(getattr(engine, "softening_kernel", None)),
                    "theta": float(engine.theta),
                    "mac_type": str(getattr(engine, "mac_type", None)),
                    "adaptive_eps": getattr(engine, "adaptive_eps", None),
                    "working_dtype": jnp.dtype(engine.working_dtype).name,
                    "use_pallas": bool(getattr(engine, "use_pallas", False)),
                    "leaf_partition": str(
                        getattr(engine, "_tree_leaf_partition", None)
                    ),
                    "cell_min_level": int(getattr(engine, "_tree_cell_min_level", 0)),
                    "cubic_bounds": bool(getattr(engine, "_tree_cubic_bounds", False)),
                    "fixed_order": getattr(engine, "fixed_order", None),
                }
            ),
            "settings": _jsonable(dataclasses.asdict(self.settings)),
            "traced_params": bool(self.segment.traced_params),
            "env": env_snapshot(
                ("JACCPOT_", "YGGDRAX_"), exclude_suffixes=("_WORKTREE",)
            ),
            "xla_flags": env_text("XLA_FLAGS"),
            "device": [device.platform, str(device.device_kind)],
            "x64": bool(jax.config.jax_enable_x64),
            **self._static_fingerprint,
        }


def run(
    solver: Any,
    state: Any,
    masses: Any,
    *,
    dt: Union[float, Any],
    num_steps: int,
    settings: GalaxySettings = GALAXY_SETTINGS,
    segment: SegmentConfig = SegmentConfig(),
    external_acceleration_fn: Optional[Callable[..., Any]] = None,
    external_potential_fn: Optional[Callable[..., Any]] = None,
    callbacks: Sequence[Callable[[SegmentEdge], Optional[bool]]] = (),
    step_callback: Optional[Callable[[Any, Any], None]] = None,
    step_callback_stride: int = 1,
    t0: float = 0.0,
    force_params: Any = None,
    external_params: Any = None,
    mesh: Any = None,
) -> RunResult:
    """One-shot :class:`ProductionRunner` run.

    Parameters
    ----------
    solver : Any
        As :class:`ProductionRunner`.
    state : Any
        ``[N, 2, 3]``.
    masses : Any
        ``[N]``.
    dt : Union[float, Any]
        Timestep.
    num_steps : int
        Steps.
    settings : GalaxySettings
        As :class:`ProductionRunner`.
    segment : SegmentConfig
        As :class:`ProductionRunner`.
    external_acceleration_fn : Optional[Callable[..., Any]]
        As :class:`ProductionRunner`.
    external_potential_fn : Optional[Callable[..., Any]]
        As :class:`ProductionRunner`.
    callbacks : Sequence[Callable[[SegmentEdge], Optional[bool]]]
        As :class:`ProductionRunner`.
    step_callback : Optional[Callable[[Any, Any], None]]
        As :class:`ProductionRunner`.
    step_callback_stride : int
        As :class:`ProductionRunner`.
    t0 : float
        Start time.
    force_params : Any
        As :meth:`ProductionRunner.run`.
    external_params : Any
        As :meth:`ProductionRunner.run`.
    mesh : Any
        As :class:`ProductionRunner`: ``None`` (one card), a 1-D mesh or a
        :class:`MeshRunConfig` (the multi-GPU mode).

    Returns
    -------
    RunResult
        As :meth:`ProductionRunner.run`.
    """
    return ProductionRunner(
        solver,
        settings=settings,
        segment=segment,
        external_acceleration_fn=external_acceleration_fn,
        external_potential_fn=external_potential_fn,
        callbacks=callbacks,
        step_callback=step_callback,
        step_callback_stride=step_callback_stride,
        mesh=mesh,
    ).run(
        state,
        masses,
        dt=dt,
        num_steps=num_steps,
        t0=t0,
        force_params=force_params,
        external_params=external_params,
    )

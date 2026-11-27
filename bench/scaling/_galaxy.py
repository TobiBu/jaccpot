"""The paper's single-card operating point: ``preset="galaxy"`` on the fused lane (D3).

REBASE_PLAN D3 fixes two configurations for Jaccpot I. Section 3's validation runs
the general path in fp64 (``preset="accurate"``); sections 4-7 run what users run,
``FastMultipoleMethod(preset="galaxy")``: the fused lane, cell leaves of 64, p6,
``dehnen_error`` at ``adaptive_eps`` 1e-4, ferrers3, fp32
(:data:`jaccpot.production.GALAXY_SETTINGS`). Every single-card timing bench builds
its solver here, so the figures cannot drift onto different configurations.

What a step costs. The fused lane rebuilds its topology at every step (tree, P2M,
M2M, flat walk, M2L, L2L) and then evaluates (P2P, L2P, scatter):
:meth:`~jaccpot.runtime.fused_provider.FusedForceProvider.step`. That is what a
simulation pays per step and what :func:`fused_step` times; the eager
``prepare`` is setup, paid once per run (and per segment retry), and is timed
separately when a bench reports it at all.

Overrides. With no override the solver is the preset exactly (``advanced=None``),
which compiles the programs the production rows were measured with
(``bench/provider_hlo_identity.py``'s ``galaxy_*`` pair). An override (another MAC,
a fixed leaf capacity, the far pairs kept for the gradient, the CPU route) builds the
preset's own advanced config (``jaccpot.solver._galaxy_advanced``) and replaces only
that field, and :func:`describe` records which fields moved, so a figure made off
the preset says so in its JSON.

The CPU route. On a CPU backend the ``large_n_gpu`` profile declines the fused lane
unless ``RuntimePolicyConfig(cpu_fused_lane=True)``; then the lane runs with its
Pallas kernels interpreted. :func:`galaxy_solver` switches it on whenever the
backend is a CPU, so every bench has a CPU smoke that runs the code the GPU runs.
CPU timings are smoke only: interpreted kernels say nothing about a GPU.

Import only after the device has been pinned (``runmeta.select_gpu``).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from typing import Any, Optional

import jax
import jax.numpy as jnp

__all__ = [
    "GALAXY",
    "card_free",
    "card_load",
    "describe",
    "fused_step",
    "galaxy_solver",
    "on_cpu",
]


def _settings() -> Any:
    from jaccpot.production import GALAXY_SETTINGS

    return GALAXY_SETTINGS


#: ``jaccpot.production.GALAXY_SETTINGS``, the preset's single source of truth.
GALAXY = _settings()


def on_cpu() -> bool:
    """Whether the default backend is a CPU (the lane then needs its CPU route)."""
    return jax.default_backend() == "cpu"


def galaxy_solver(
    *,
    softening: float,
    G: float = 1.0,
    mac_type: Optional[str] = None,
    theta: Optional[float] = None,
    adaptive_eps: Optional[float] = None,
    leaf_capacity: Any = None,
    cell_min_level: Any = None,
    retain_far_pairs: bool = False,
    working_dtype: Any = None,
    cpu_route: Optional[bool] = None,
) -> Any:
    """Return ``FastMultipoleMethod(preset="galaxy")``, with only the named overrides.

    Parameters
    ----------
    softening : float
        Softening length (the preset's kernel, ferrers3, is kept).
    G : float
        Gravitational constant.
    mac_type : str, optional
        ``None`` keeps the preset's ``dehnen_error``. ``"dehnen"`` is the geometric
        test at ``theta`` (the preset's 0.8 if ``theta`` is ``None``).
    theta : float, optional
        Opening angle; ``None`` keeps the preset's.
    adaptive_eps : float, optional
        ``dehnen_error``'s tolerance; ``None`` keeps the preset's 1e-4.
    leaf_capacity : int or ``"auto"``, optional
        ``None`` keeps the preset's ``"auto"`` (sized at the first prepare).
    cell_min_level : int or ``"auto"``, optional
        ``None`` keeps the preset's floor. The preset's floor is set for production
        N: at small N it leaves about one particle per leaf, so nearly every pair is
        far and the near field (and, on a mesh, the cross-domain near term) goes
        unexercised. A small-N CORRECTNESS check should lower it; a timing figure
        should not.
    retain_far_pairs : bool
        Keep the far pairs on the prepared state (the gradient plan reads them).
    working_dtype : dtype, optional
        ``None`` keeps the preset's fp32.
    cpu_route : bool, optional
        ``None``: on whenever the backend is a CPU.

    Returns
    -------
    FastMultipoleMethod
        The solver.
    """
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FarFieldConfig
    from jaccpot.solver import _galaxy_advanced

    cpu = on_cpu() if cpu_route is None else bool(cpu_route)
    overrides = {
        "mac_type": mac_type,
        "leaf_capacity": leaf_capacity,
        "cell_min_level": cell_min_level,
        "retain_far_pairs": retain_far_pairs or None,
        "cpu_route": cpu or None,
    }
    advanced = None
    if any(v is not None for v in overrides.values()):
        advanced = _galaxy_advanced(GALAXY)
        if mac_type is not None:
            advanced = dataclasses.replace(advanced, mac_type=str(mac_type))
        if leaf_capacity is not None:
            advanced = dataclasses.replace(
                advanced,
                tree=dataclasses.replace(advanced.tree, leaf_capacity=leaf_capacity),
            )
        if cell_min_level is not None:
            advanced = dataclasses.replace(
                advanced,
                tree=dataclasses.replace(
                    advanced.tree,
                    cell_min_level=(
                        cell_min_level
                        if cell_min_level == "auto"
                        else int(cell_min_level)
                    ),
                ),
            )
        if retain_far_pairs:
            advanced = dataclasses.replace(
                advanced,
                farfield=dataclasses.replace(
                    advanced.farfield or FarFieldConfig(),
                    retain_far_pairs_for_grad=True,
                ),
            )
        if cpu:
            advanced = dataclasses.replace(
                advanced,
                runtime=dataclasses.replace(advanced.runtime, cpu_fused_lane=True),
            )
    kwargs: dict[str, Any] = {"softening": float(softening), "G": float(G)}
    if theta is not None:
        kwargs["theta"] = float(theta)
    if adaptive_eps is not None:
        kwargs["adaptive_eps"] = float(adaptive_eps)
    if working_dtype is not None:
        kwargs["working_dtype"] = working_dtype
    if cpu:
        # The card's near-field layout on the CPU route: the CSR lane, kernels
        # interpreted. Without it the CPU takes the pure-JAX near field, which has
        # no force-scale lane (dehnen_error cannot step) and no reverse (the
        # gradient plan refuses it); forward.fused_solver does the same (A3).
        kwargs["use_pallas"] = True
    return FastMultipoleMethod(preset="galaxy", advanced=advanced, **kwargs)


def describe(
    solver: Any,
    *,
    mac_type: Optional[str] = None,
    theta: Optional[float] = None,
    adaptive_eps: Optional[float] = None,
    leaf_capacity: Any = None,
    cell_min_level: Any = None,
    retain_far_pairs: bool = False,
) -> dict[str, Any]:
    """The configuration a result JSON records: the preset and what moved off it."""
    moved = {
        k: v
        for k, v in {
            "mac_type": mac_type,
            "theta": theta,
            "adaptive_eps": adaptive_eps,
            "leaf_capacity": leaf_capacity,
            "cell_min_level": cell_min_level,
            "retain_far_pairs": retain_far_pairs or None,
        }.items()
        if v is not None
    }
    return {
        "preset": "galaxy",
        "lane": "fused",
        "leaf_size": int(GALAXY.leaf_size),
        "max_order": int(GALAXY.max_order),
        "mac_type": str(solver.mac_type),
        "theta": float(theta if theta is not None else GALAXY.theta),
        "adaptive_eps": (
            float(
                getattr(getattr(solver, "_impl", None), "adaptive_eps", None)
                or GALAXY.adaptive_eps
            )
            if solver.mac_type == "dehnen_error"
            else None
        ),
        "softening_kernel": str(GALAXY.softening_kernel),
        "working_dtype": "float32",
        "cpu_route": on_cpu(),
        "off_preset": moved,
    }


class FusedStep:
    """One prepared provider and its jitted ``step``, ready to time.

    Attributes
    ----------
    provider : FusedForceProvider
        The provider the eager prepare returned.
    topology : FusedTopology
        The prepare's topology (the evaluation-only series times on it).
    aux : tuple
        ``(f_b, None, None)``: the step's first carry (f_b under ``dehnen_error``,
        else ``None``).
    """

    def __init__(self, provider: Any, topology: Any, aux: Any) -> None:
        self.provider = provider
        self.topology = topology
        self.aux = aux
        self._step = jax.jit(provider.step)
        self._eval = jax.jit(lambda t: provider.accelerations(t))
        self._potential = jax.jit(lambda t: provider.potential(t))

    def step(self, positions: Any, masses: Any) -> tuple[Any, Any]:
        """Rebuild + evaluate at ``positions``: ``(acc, (f_b, bits, needs))``."""
        return self._step(positions, masses, None, self.aux)

    def evaluate(self) -> Any:
        """Evaluation only, on the prepare's topology (no rebuild)."""
        return self._eval(self.topology)

    def potential(self) -> Any:
        """The potential per particle on the prepare's topology."""
        return self._potential(self.topology)

    def ok(self, aux_out: Any) -> bool:
        """The step's capacity verdict: ``False`` means the force is WRONG, not slow."""
        from jaccpot.runtime.fused_topology import overflow_ok

        return bool(overflow_ok(aux_out[1]))


def fused_step(
    solver: Any, positions: Any, masses: Any, *, differentiable: bool = False
) -> FusedStep:
    """Prepare a provider at ``positions`` and return its timed step.

    Parameters
    ----------
    solver : FastMultipoleMethod
        From :func:`galaxy_solver`.
    positions, masses : array
        The particles (the prepare sizes the leaf capacity from them).
    differentiable : bool
        Freeze the gradient plan as well (``differentiable_step``); the solver
        must keep its far pairs (``retain_far_pairs=True``).

    Returns
    -------
    FusedStep
        The provider, its topology and its first carry.
    """
    from jaccpot import FusedForceProvider

    kw: dict[str, Any] = {}
    if differentiable:
        kw["differentiable"] = True
    provider, topology = FusedForceProvider.prepare(
        solver,
        positions,
        masses,
        leaf_size=int(GALAXY.leaf_size),
        max_order=int(GALAXY.max_order),
        **kw,
    )
    f_b = None
    if solver.mac_type == "dehnen_error":
        f_b = _start_force_scale(solver, provider, topology)
    return FusedStep(provider, topology, (f_b, None, None))


def _start_force_scale(solver: Any, provider: Any, topology: Any) -> Any:
    """The first step's f_b, in input order, as ``strict_run_v2`` seeds it.

    Eq (16a)'s thresholds come from the previous evaluation's f_b. The eager
    prepare leaves eq (16b)'s estimate on the engine, in its tree's sorted order
    (``_fused_force_scale_seed``); scattered back to input order it is the seed
    ``fmm_strict_run._force_scale_seed`` hands the scan. Only if the prepare left
    none is one evaluation's f_b used (that needs the CSR near lane, so a GPU).
    """
    from jaccpot.runtime.fused_provider import _engine_of

    engine = _engine_of(solver)
    fb_sorted = getattr(engine, "_last_force_scale_particles", None)
    perm = jnp.asarray(topology.prepared.tree.particle_indices)
    if fb_sorted is not None and int(fb_sorted.shape[0]) == int(perm.shape[0]):
        fb_sorted = jnp.asarray(fb_sorted)
        return jnp.zeros_like(fb_sorted).at[perm].set(fb_sorted)
    return jax.jit(lambda t: provider.accelerations(t, force_scale=True))(topology)[1]


def card_load() -> Optional[dict[str, Any]]:
    """The visible cards' compute processes, and whether any is not this one.

    Timing figures must come from a free card (REBASE_PLAN, step B): call this
    before and after a timed series and AND :func:`card_free` of both into the
    row's ``valid``. Called after jax has allocated, this process is always
    listed, so it is told apart by pid. ``None`` off a GPU.

    Returns
    -------
    dict or None
        ``{"processes": [[pid, name, used_memory], ...], "own_pid": int,
        "contended": bool or None}``; ``contended`` is ``None`` when
        ``nvidia-smi`` could not be read (then the card is not known to be free).
    """
    if on_cpu():
        return None
    own = os.getpid()
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        cmd += ["-i", cvd]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"processes": [], "own_pid": own, "contended": None, "error": str(exc)}
    if out.returncode != 0:
        return {
            "processes": [],
            "own_pid": own,
            "contended": None,
            "error": out.stderr.strip()[:300],
        }
    procs = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0].isdigit():
            procs.append([int(parts[0])] + parts[1:])
    return {
        "processes": procs,
        "own_pid": own,
        "contended": any(p[0] != own for p in procs),
    }


def card_free(*loads: Optional[dict[str, Any]]) -> bool:
    """Whether every :func:`card_load` record says the card had no other process.

    ``True`` off a GPU (``None`` records: nothing to share). An unreadable
    record (``contended is None``) is not free.
    """
    return all(load is None or load.get("contended") is False for load in loads)

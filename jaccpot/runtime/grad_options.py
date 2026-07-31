"""Resolution of :class:`~jaccpot.config.GradConfig` into concrete grad-path options.

The differentiable FMM grew a set of ``JACCPOT_*`` switches, one of which
(the near-field lane) is the difference between a working galaxy-scale gradient
and a 30 GB OOM. Environment variables are a poor home for that: they are
process-global, invisible to ``help()``, and cannot differ between two solvers
in one program.

:class:`~jaccpot.config.GradConfig` is the replacement. This module turns one
into a :class:`ResolvedGradOptions` -- every value concrete, no ``None`` left --
under a single precedence rule:

    explicit ``GradConfig`` field  >  ``JACCPOT_*`` environment variable  >  measured default

so scripts that export the variables keep working unchanged, and a caller who
sets a field always wins over an inherited environment.

Resolution happens **once per call**, outside any trace, and the result is
captured as Python constants. Nothing here reads the environment again further
down the stack.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

from jaccpot._env import env_flag, env_float, env_int
from jaccpot.config import GradConfig

# Env-var names, kept in one place so the docs, the resolver and the tests all
# agree on the spelling.
ENV_NEARFIELD_FAST_LANE = "JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE"
ENV_FUSED_M2L_PALLAS = "JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS"
ENV_ANALYTIC_P2P_VJP = "JACCPOT_ANALYTIC_P2P_VJP"
ENV_ANALYTIC_L2P_VJP = "JACCPOT_ANALYTIC_L2P_VJP"
ENV_REV_TIERED = "JACCPOT_GRAD_REV_TIERED"
ENV_REV_TIERS = "JACCPOT_GRAD_REV_TIERS"
ENV_REV_TIER_MIN_GAIN = "JACCPOT_GRAD_REV_TIER_MIN_GAIN"
ENV_REV_SKIP_EMPTY_TILES = "JACCPOT_GRAD_REV_SKIP_EMPTY_TILES"
ENV_REV_LEAF_BATCH = "JACCPOT_GRAD_REV_LEAF_BATCH"
ENV_REV_BLOCK_TILE = "JACCPOT_GRAD_REV_BLOCK_TILE"


@dataclass(frozen=True)
class LeafPairReverseOptions:
    """Concrete tuning for the analytic leaf-pair reverse.

    Bundled rather than passed as six separate arguments because it has to
    travel four layers down (``differentiable_accelerations`` -> the eval seam
    -> ``evaluate_tree`` -> the fast lane) and ride through a ``custom_vjp`` in
    ``nondiff_argnums``, which requires it to be hashable.
    """

    tiered: bool = True
    max_tiers: int = 4
    tier_min_gain: float = 3.0
    skip_empty_tiles: bool = True
    leaf_batch: int = 8
    block_tile: int = 8


@dataclass(frozen=True)
class ResolvedGradOptions:
    """Fully-resolved grad-path execution options. No ``None`` values remain."""

    nearfield_lane: str  # "bucketed" | "fast_lane"
    nearfield_lane_was_auto: bool
    fused_m2l_pallas: bool
    analytic_p2p_vjp: bool
    analytic_l2p_vjp: bool
    reverse: LeafPairReverseOptions


def resolve_grad_options(
    config: Optional[GradConfig],
    *,
    num_particles: int,
    supports_fast_lane: bool,
) -> ResolvedGradOptions:
    """Resolve a :class:`~jaccpot.config.GradConfig` against the environment.

    Parameters
    ----------
    config : Optional[GradConfig]
        User-supplied options. ``None`` is treated as an all-defaults
        ``GradConfig()``, which is what makes the environment variables the
        effective interface for callers who never pass one.
    num_particles : int
        Particle count, used to resolve ``nearfield_lane="auto"``.
    supports_fast_lane : bool
        Whether this prepared state can actually take the leaf-major lane. The
        auto policy must not select a lane the state cannot run; an explicit
        request is still honoured (and fails loudly downstream) so a user who
        asks for it is told why rather than silently given something else.

    Returns
    -------
    ResolvedGradOptions
        Concrete options, safe to capture as trace constants.
    """
    cfg = GradConfig() if config is None else config

    lane = str(cfg.nearfield_lane).strip().lower()
    lane_was_auto = lane == "auto"
    if lane_was_auto:
        # The environment variable keeps its original meaning as the default for
        # "auto": if it is set, it decides. Otherwise fall back to the measured
        # N crossover -- the bucketed reverse OOMs at 200k, so defaulting large
        # runs to it would hand users a crash they have no way to anticipate.
        if env_flag(ENV_NEARFIELD_FAST_LANE, False):
            lane = "fast_lane"
        elif supports_fast_lane and int(num_particles) >= int(
            cfg.nearfield_fast_lane_min_particles
        ):
            lane = "fast_lane"
        else:
            lane = "bucketed"

    return ResolvedGradOptions(
        nearfield_lane=lane,
        nearfield_lane_was_auto=lane_was_auto,
        fused_m2l_pallas=(
            bool(cfg.fused_m2l_pallas)
            if cfg.fused_m2l_pallas is not None
            else env_flag(ENV_FUSED_M2L_PALLAS, False)
        ),
        analytic_p2p_vjp=(
            bool(cfg.analytic_p2p_vjp)
            if cfg.analytic_p2p_vjp is not None
            else env_flag(ENV_ANALYTIC_P2P_VJP, True)
        ),
        analytic_l2p_vjp=(
            bool(cfg.analytic_l2p_vjp)
            if cfg.analytic_l2p_vjp is not None
            else env_flag(ENV_ANALYTIC_L2P_VJP, True)
        ),
        reverse=LeafPairReverseOptions(
            tiered=env_flag(ENV_REV_TIERED, True),
            max_tiers=(
                int(cfg.reverse_tiers)
                if cfg.reverse_tiers is not None
                else env_int(ENV_REV_TIERS, 4)
            ),
            tier_min_gain=(
                float(cfg.reverse_tier_min_gain)
                if cfg.reverse_tier_min_gain is not None
                else env_float(ENV_REV_TIER_MIN_GAIN, 3.0)
            ),
            skip_empty_tiles=(
                bool(cfg.reverse_skip_empty_tiles)
                if cfg.reverse_skip_empty_tiles is not None
                else env_flag(ENV_REV_SKIP_EMPTY_TILES, True)
            ),
            leaf_batch=(
                int(cfg.reverse_leaf_batch)
                if cfg.reverse_leaf_batch is not None
                else env_int(ENV_REV_LEAF_BATCH, 8, minimum=1)
            ),
            block_tile=(
                int(cfg.reverse_block_tile)
                if cfg.reverse_block_tile is not None
                else env_int(ENV_REV_BLOCK_TILE, 8, minimum=1)
            ),
        ),
    )


# --------------------------------------------------------------------------
# Scoped overrides for gates that are read deep inside the kernels.
#
# Three switches (the fused-Pallas M2L and the two analytic VJP rules) are
# consulted at trace time by code several layers below the public entry point --
# inside the M2L dispatch and the L2P/P2P operators -- with no argument channel
# reaching them. Threading a config object down every one of those paths would
# touch the forward-only production code for no forward-only benefit.
#
# So the gates consult a context-local override first and fall back to the
# environment. ``differentiable_accelerations`` installs the override for the
# duration of its own call, which makes the corresponding ``GradConfig`` fields
# real rather than silently inert -- the failure mode this whole module exists
# to remove.
#
# ContextVar rather than mutating ``os.environ``: it is thread-safe, async-safe,
# and leaves no process-global state behind if the call raises.
#
# Caveat, shared with the environment variables these replace: all three are
# read at TRACE time. Changing one between two calls that hit the same compiled
# jaxpr does not retrace, so A/B measurement wants either separate processes or
# a cleared compile cache.
# --------------------------------------------------------------------------

_fused_m2l_pallas_override: ContextVar[Optional[bool]] = ContextVar(
    "jaccpot_fused_m2l_pallas_override", default=None
)
_analytic_p2p_vjp_override: ContextVar[Optional[bool]] = ContextVar(
    "jaccpot_analytic_p2p_vjp_override", default=None
)
_analytic_l2p_vjp_override: ContextVar[Optional[bool]] = ContextVar(
    "jaccpot_analytic_l2p_vjp_override", default=None
)


def fused_m2l_pallas_enabled() -> bool:
    """Whether the fused-Pallas M2L is requested (hardware support checked separately)."""
    override = _fused_m2l_pallas_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_FUSED_M2L_PALLAS, False)


def analytic_p2p_vjp_enabled() -> bool:
    """Whether the analytic near-field P2P reverse rule is enabled (default on)."""
    override = _analytic_p2p_vjp_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_ANALYTIC_P2P_VJP, True)


def analytic_l2p_vjp_enabled() -> bool:
    """Whether the analytic real-basis L2P reverse rule is enabled (default on)."""
    override = _analytic_l2p_vjp_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_ANALYTIC_L2P_VJP, True)


@contextlib.contextmanager
def grad_option_overrides(options: ResolvedGradOptions) -> Iterator[None]:
    """Install ``options`` as the context-local answer for the deep gates.

    Scoped to the ``with`` block and restored on exit, including on exception.
    """
    tokens = (
        _fused_m2l_pallas_override.set(options.fused_m2l_pallas),
        _analytic_p2p_vjp_override.set(options.analytic_p2p_vjp),
        _analytic_l2p_vjp_override.set(options.analytic_l2p_vjp),
    )
    try:
        yield
    finally:
        _fused_m2l_pallas_override.reset(tokens[0])
        _analytic_p2p_vjp_override.reset(tokens[1])
        _analytic_l2p_vjp_override.reset(tokens[2])


__all__ = [
    "LeafPairReverseOptions",
    "ResolvedGradOptions",
    "analytic_l2p_vjp_enabled",
    "analytic_p2p_vjp_enabled",
    "fused_m2l_pallas_enabled",
    "grad_option_overrides",
    "resolve_grad_options",
]

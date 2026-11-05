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
from typing import Any, Iterator, Optional

import jax

from jaccpot._env import env_choice, env_flag, env_float, env_int
from jaccpot.config import GradConfig

# Env-var names, kept in one place so the docs, the resolver and the tests all
# agree on the spelling.
ENV_NEARFIELD_FAST_LANE = "JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE"
ENV_FUSED_M2L_PALLAS = "JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS"
ENV_ANALYTIC_P2P_VJP = "JACCPOT_ANALYTIC_P2P_VJP"
ENV_ANALYTIC_L2P_VJP = "JACCPOT_ANALYTIC_L2P_VJP"
ENV_CASCADE_PALLAS = "JACCPOT_CASCADE_PALLAS"
ENV_REV_TIERED = "JACCPOT_GRAD_REV_TIERED"
ENV_REV_TIERS = "JACCPOT_GRAD_REV_TIERS"
ENV_REV_TIER_MIN_GAIN = "JACCPOT_GRAD_REV_TIER_MIN_GAIN"
ENV_REV_SKIP_EMPTY_TILES = "JACCPOT_GRAD_REV_SKIP_EMPTY_TILES"
ENV_REV_LEAF_BATCH = "JACCPOT_GRAD_REV_LEAF_BATCH"
ENV_REV_BLOCK_TILE = "JACCPOT_GRAD_REV_BLOCK_TILE"
ENV_AD_MODE = "JACCPOT_GRAD_AD_MODE"
ENV_FORWARD_TANGENT_KERNELS = "JACCPOT_GRAD_FORWARD_TANGENT_KERNELS"


@dataclass(frozen=True)
class LeafPairReverseOptions:
    """Concrete tuning for the analytic leaf-pair reverse.

    Bundled rather than passed as six separate arguments because it has to
    travel four layers down (``differentiable_accelerations`` -> the eval seam
    -> ``evaluate_tree`` -> the fast lane) and ride through a ``custom_vjp`` in
    ``nondiff_argnums``, which requires it to be hashable.

    Attributes
    ----------
    tiered : bool
        Use the occupancy-tiered reverse rather than a single pass.
    max_tiers : int
        Cap on the number of tiers.
    tier_min_gain : float
        Minimum predicted slot-visit reduction before tiering is worth its
        compile cost.
    skip_empty_tiles : bool
        Skip tiles with no live pairs.
    leaf_batch : int
        Target leaves per reverse batch.
    block_tile : int
        Slot-tile width the reverse scans with.
    """

    tiered: bool = True
    max_tiers: int = 4
    tier_min_gain: float = 3.0
    skip_empty_tiles: bool = True
    leaf_batch: int = 8
    block_tile: int = 8


@dataclass(frozen=True)
class ResolvedGradOptions:
    """Fully-resolved grad-path execution options. No ``None`` values remain.

    "Resolved" is the contract: every field is concrete, so the deep gates never
    have to re-derive a default or consult the environment. Produced once at the
    top of a grad call and installed with :func:`grad_option_overrides`.

    Attributes
    ----------
    nearfield_lane : str
        Which near-field reverse lane runs.
    nearfield_lane_was_auto : bool
        Whether ``nearfield_lane`` came from auto-selection rather than the caller.
        Kept because a diagnostic needs to distinguish "chose this" from "was told
        this".
    cascade_pallas : bool
        Whether the per-level Pallas M2M/L2L cascades and the leaf P2M are
        requested on the gradient path. They carry ``custom_vjp`` rules with
        reverse Pallas kernels of their own (:mod:`jaccpot.pallas.cascade_real_level`,
        :mod:`jaccpot.pallas.p2m_real_leaf`), so this is a speed knob, not a
        correctness one; hardware support is checked separately by
        :func:`jaccpot.runtime._level_shapes.pallas_cascades_enabled`.
    fused_m2l_pallas : bool
        Whether the fused Pallas M2L is requested; hardware support is checked
        separately.
    analytic_p2p_vjp : bool
        Use the analytic near-field P2P reverse rule.
    analytic_l2p_vjp : bool
        Use the analytic real-basis L2P reverse rule.
    reverse : LeafPairReverseOptions
        Tuning for the analytic leaf-pair reverse.
    ad_mode : str
        ``"reverse"`` (the custom_vjp kernels) or ``"forward"`` (their
        ``custom_jvp`` siblings, :mod:`jaccpot.runtime._forward_seams`).
    forward_tangent_kernels : str
        ``"auto"``, ``"pallas"`` or ``"jnp"``: the tangent kernels forward mode
        runs (ignored in reverse mode).
    """

    nearfield_lane: str  # "bucketed" | "fast_lane"
    nearfield_lane_was_auto: bool
    cascade_pallas: bool
    fused_m2l_pallas: bool
    analytic_p2p_vjp: bool
    analytic_l2p_vjp: bool
    reverse: LeafPairReverseOptions
    ad_mode: str = "reverse"
    forward_tangent_kernels: str = "auto"


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
        ad_mode=resolve_ad_mode(cfg.ad_mode),
        forward_tangent_kernels=resolve_forward_tangent_kernels(
            cfg.forward_tangent_kernels
        ),
        nearfield_lane=lane,
        nearfield_lane_was_auto=lane_was_auto,
        cascade_pallas=(
            bool(cfg.cascade_pallas)
            if cfg.cascade_pallas is not None
            else env_flag(ENV_CASCADE_PALLAS, True)
        ),
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


def resolve_ad_mode(ad_mode: Optional[str] = None) -> str:
    """The AD mode a gradient-path program is built for.

    Precedence: ``ad_mode`` itself, then the mode an enclosing
    :func:`grad_option_overrides` (or :func:`ad_mode_override`) installed, then
    ``JACCPOT_GRAD_AD_MODE``, then ``"reverse"``. An enclosing scope beats the
    environment so a nested resolution (``differentiable_accelerations`` resolves
    its own options inside a rollout's) keeps the caller's mode.

    Parameters
    ----------
    ad_mode : Optional[str]
        ``"reverse"``, ``"forward"`` or ``None`` (defer).

    Returns
    -------
    str
        ``"reverse"`` or ``"forward"``.

    Raises
    ------
    ValueError
        For any other explicit value.
    """
    if ad_mode is not None:
        mode = str(ad_mode).strip().lower()
        if mode not in ("reverse", "forward"):
            raise ValueError(f"ad_mode must be 'reverse' or 'forward', got {ad_mode!r}")
        return mode
    ambient = _ad_mode_override.get()
    if ambient is not None:
        return ambient
    return env_choice(ENV_AD_MODE, "reverse", ("reverse", "forward"))


def resolve_forward_tangent_kernels(choice: Optional[str] = None) -> str:
    """Which tangent kernels forward mode runs.

    Precedence as :func:`resolve_ad_mode`: ``choice``, an enclosing override,
    ``JACCPOT_GRAD_FORWARD_TANGENT_KERNELS``, ``"auto"``.

    Parameters
    ----------
    choice : Optional[str]
        ``"auto"``, ``"pallas"``, ``"jnp"`` or ``None`` (defer).

    Returns
    -------
    str
        ``"auto"``, ``"pallas"`` or ``"jnp"``.

    Raises
    ------
    ValueError
        For any other explicit value.
    """
    choices = ("auto", "pallas", "jnp")
    if choice is not None:
        text = str(choice).strip().lower()
        if text not in choices:
            raise ValueError(
                f"forward_tangent_kernels must be one of {choices}, got {choice!r}"
            )
        return text
    ambient = _forward_tangent_kernels_override.get()
    if ambient is not None:
        return ambient
    return env_choice(ENV_FORWARD_TANGENT_KERNELS, "auto", choices)


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
#: The gradient path's answer for the Pallas-cascade gate. History: Phase 6 of
#: the sub-10 ms plan made the per-level Pallas M2M/L2L cascades and the leaf
#: P2M default-ON while they had no autodiff rule, and ``pallas_call``'s generic
#: JVP rule dies on ``program_id`` (no grid context under that trace) -- which
#: broke ``jax.grad`` everywhere Pallas lowers, invisibly, because Pallas does
#: not lower on CPU and the whole gradient suite runs there. The first fix forced
#: ``False`` here; the kernels now carry ``custom_vjp`` rules with reverse Pallas
#: kernels, so the override carries the resolved ``GradConfig.cascade_pallas``.
_cascade_pallas_override: ContextVar[Optional[bool]] = ContextVar(
    "jaccpot_cascade_pallas_override", default=None
)
#: The AD mode (``"reverse"`` / ``"forward"``) the seams below dispatch on, and
#: forward mode's tangent-kernel choice. ``None``: no enclosing scope.
_ad_mode_override: ContextVar[Optional[str]] = ContextVar(
    "jaccpot_ad_mode_override", default=None
)
_forward_tangent_kernels_override: ContextVar[Optional[str]] = ContextVar(
    "jaccpot_forward_tangent_kernels_override", default=None
)
#: The forward mode's key in ``jax.jit``'s caches. The seams below sit inside
#: jitted helpers (``_evaluate_local_expansions_particle_major``,
#: ``_propagate_solidfmm_locals_by_level``, ...) whose trace cache keys on their
#: arguments only: a reverse-mode trace reused under forward mode would hand
#: ``jax.jvp`` the custom_vjp kernels (and a forward trace reused under reverse
#: mode would change the reverse program). Forward mode enters this jit-key
#: context with its tangent-kernel choice; reverse mode leaves it at its default
#: ``None``, so the reverse programs' keys (and texts) are what they were.
_FORWARD_JIT_KEY: Any = jax.make_user_context(None)


def cascade_pallas_override() -> Optional[bool]:
    """Context-local answer for the Pallas-cascade gate, if one is installed.

    Returns
    -------
    Optional[bool]
        The resolved ``GradConfig.cascade_pallas`` inside
        :func:`grad_option_overrides`, else ``None``, meaning "no opinion, read
        the environment".
    """
    return _cascade_pallas_override.get()


def fused_m2l_pallas_enabled() -> bool:
    """Whether the fused-Pallas M2L is requested (hardware support checked separately).

    Returns
    -------
    bool
        ``True`` when the fused lane is requested. Requested, not available: the
        Ampere+ gate is a separate check, so a ``True`` here can still fall back.
    """
    override = _fused_m2l_pallas_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_FUSED_M2L_PALLAS, False)


def analytic_p2p_vjp_enabled() -> bool:
    """Whether the analytic near-field P2P reverse rule is enabled (default on).

    Returns
    -------
    bool
        ``True`` when the analytic P2P reverse is enabled. On by default; always
        ``False`` in forward mode (:func:`forward_mode_enabled`).
    """
    if forward_mode_enabled():
        # a custom_vjp, which JAX refuses to jvp; its forward is plain autodiff's
        return False
    override = _analytic_p2p_vjp_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_ANALYTIC_P2P_VJP, True)


def analytic_l2p_vjp_enabled() -> bool:
    """Whether the analytic real-basis L2P reverse rule is enabled (default on).

    Returns
    -------
    bool
        ``True`` when the analytic L2P reverse is enabled. On by default; always
        ``False`` in forward mode (:func:`forward_mode_enabled`).
    """
    if forward_mode_enabled():
        # a custom_vjp, which JAX refuses to jvp; its forward is plain autodiff's
        # ``value_and_grad`` (the same bits)
        return False
    override = _analytic_l2p_vjp_override.get()
    if override is not None:
        return bool(override)
    return env_flag(ENV_ANALYTIC_L2P_VJP, True)


def forward_mode_enabled() -> bool:
    """Whether the seams trace their forward-mode (``custom_jvp``) siblings.

    Read at trace time by every production seam of the large-N gradient lane:
    ``True`` routes the seam to :mod:`jaccpot.runtime._forward_seams`, ``False``
    (the default) to the ``custom_vjp`` kernels, unchanged.

    Returns
    -------
    bool
        ``True`` inside a forward-mode :func:`grad_option_overrides` or
        :func:`ad_mode_override`, else whether ``JACCPOT_GRAD_AD_MODE`` is
        ``forward``.
    """
    return resolve_ad_mode(None) == "forward"


def forward_tangent_kernels() -> str:
    """Forward mode's tangent-kernel choice (``"auto"``, ``"pallas"``, ``"jnp"``).

    Returns
    -------
    str
        The enclosing scope's resolved choice, else the environment's, else
        ``"auto"``.
    """
    return resolve_forward_tangent_kernels(None)


@contextlib.contextmanager
def ad_mode_override(
    ad_mode: Optional[str], forward_tangent_kernels: Optional[str] = None
) -> Iterator[None]:
    """Trace the gradient path's seams for ``ad_mode`` within the block.

    The AD mode alone, without the other resolved options:
    :func:`~jaccpot.runtime.grad_rollout.differentiable_rollout` and
    :class:`~jaccpot.production_grad.GradientPlan` install it around the trace of
    a forward-mode program. Like the other gates it is read at TRACE time.

    Parameters
    ----------
    ad_mode : Optional[str]
        ``"reverse"`` or ``"forward"``; ``None`` keeps the enclosing mode.
    forward_tangent_kernels : Optional[str]
        ``"auto"``, ``"pallas"`` or ``"jnp"``; ``None`` keeps the enclosing one.

    Yields
    ------
    None
        The block runs with the mode installed.
    """
    mode = None if ad_mode is None else resolve_ad_mode(ad_mode)
    kernels = (
        None
        if forward_tangent_kernels is None
        else resolve_forward_tangent_kernels(forward_tangent_kernels)
    )
    tokens = []
    if mode is not None:
        tokens.append((_ad_mode_override, _ad_mode_override.set(mode)))
    if kernels is not None:
        tokens.append(
            (
                _forward_tangent_kernels_override,
                _forward_tangent_kernels_override.set(kernels),
            )
        )
    try:
        with _forward_jit_key():
            yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _forward_jit_key() -> Any:
    """The jit-cache context of the mode now installed (see ``_FORWARD_JIT_KEY``).

    Returns
    -------
    Any
        A context manager: ``("forward", <tangent kernels>)`` in forward mode,
        ``None`` (the default, no new cache entries) in reverse mode.
    """
    if resolve_ad_mode(None) == "forward":
        return _FORWARD_JIT_KEY(("forward", resolve_forward_tangent_kernels(None)))
    return _FORWARD_JIT_KEY(None)


@contextlib.contextmanager
def grad_option_overrides(options: ResolvedGradOptions) -> Iterator[None]:
    """Install ``options`` as the context-local answer for the deep gates.

    Scoped to the ``with`` block and restored on exit, including on exception.

    The gates are read far below the call that resolves them, so the value travels
    context-locally rather than through four layers of arguments.

    Parameters
    ----------
    options : ResolvedGradOptions
        Fully-resolved options to install for the duration of the block.

    Yields
    ------
    None
        Nothing; the value is delivered through the module-local gates, not through
        the ``as`` target.
    """
    tokens = (
        _fused_m2l_pallas_override.set(options.fused_m2l_pallas),
        _analytic_p2p_vjp_override.set(options.analytic_p2p_vjp),
        _analytic_l2p_vjp_override.set(options.analytic_l2p_vjp),
        _cascade_pallas_override.set(options.cascade_pallas),
        _ad_mode_override.set(options.ad_mode),
        _forward_tangent_kernels_override.set(options.forward_tangent_kernels),
    )
    try:
        with _forward_jit_key():
            yield
    finally:
        _fused_m2l_pallas_override.reset(tokens[0])
        _analytic_p2p_vjp_override.reset(tokens[1])
        _analytic_l2p_vjp_override.reset(tokens[2])
        _cascade_pallas_override.reset(tokens[3])
        _ad_mode_override.reset(tokens[4])
        _forward_tangent_kernels_override.reset(tokens[5])


__all__ = [
    "LeafPairReverseOptions",
    "ResolvedGradOptions",
    "ad_mode_override",
    "analytic_l2p_vjp_enabled",
    "analytic_p2p_vjp_enabled",
    "cascade_pallas_override",
    "forward_mode_enabled",
    "forward_tangent_kernels",
    "fused_m2l_pallas_enabled",
    "grad_option_overrides",
    "resolve_ad_mode",
    "resolve_forward_tangent_kernels",
    "resolve_grad_options",
]

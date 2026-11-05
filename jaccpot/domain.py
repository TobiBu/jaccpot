"""The IC guard: what the Morton box of a state will be, before the first force (A2).

A few far particles set the Morton box. When the box is much larger than the
system, a level-21 cell (``box / 2**21``) grows wide against the dense centre,
many leaves share one cell, and the radix tree gets deeper: past yggdrax's
``MAX_TREE_LEVELS`` (64) it cannot be represented at all. The 1e8 Sgr IC's
sampler tails (stellar halo to 30 Mpc, the Sgr Plummer to 5 Mpc) made a 44 Mpc
box, 22 pc cells and a tree 68 levels deep, which needed a hand clip.

:func:`domain_check` measures it on the state, cheaply (device reductions over
every particle; the median and the quantile radius on a strided subsample of at
most ``sample`` rows): the box, the level-21 cell, the farthest particle, the
99.9th-percentile radius, how much larger the box is than the bulk, and the
particles in the level-21 cell at the median centre -- from which a bound on the
tree's depth follows (a radix tree over 63-bit codes is at most 64 levels when
its cell codes are unique; ``L`` leaves sharing one cell add ``ceil(log2 L)``).
:func:`domain_guard` warns (``"warn"``) or refuses (``"error"``) when the box is
dominated by a few particles.

``ProductionRunner`` runs the guard before a run's first segment
(``SegmentConfig(domain_guard="warn")``, the default: it changes no program and
reads the state only). Whatever the guard says, a tree deeper than the bound
raises (``yggdrax.TreeDepthError`` from the eager prepare, ``RuntimeError`` from
the scan): never a silent truncation.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Optional

import numpy as np

__all__ = [
    "DomainWarning",
    "domain_check",
    "domain_guard",
    "format_domain_check",
]

#: Morton bits per axis (yggdrax's 63-bit codes)
_MORTON_LEVELS = 21
#: the radix tree's level tables (``yggdrax.MAX_TREE_LEVELS``)
_MAX_TREE_LEVELS = 64
#: ``dominance`` above which the guard warns: the box is this many times the
#: diameter of the sphere holding 99.9 % of the particles
DOMINANCE_WARN = 20.0


class DomainWarning(RuntimeWarning):
    """The IC guard's warning: the Morton box is dominated by a few particles."""


def _reductions(pos: Any, centre: Any, beyond2: Any, half: Any) -> tuple:
    """The guard's passes over every particle, as ONE jitted program.

    Parameters
    ----------
    pos : Any
        ``[N, 3]``.
    centre : Any
        ``[3]``.
    beyond2 : Any
        The squared radius past which a particle counts as beyond.
    half : Any
        Half a level-21 cell.

    Returns
    -------
    tuple
        ``(argmax r^2, max r^2, count beyond, count in the central cell)``.
    """
    import jax

    global _REDUCE
    if _REDUCE is None:
        import jax.numpy as jnp

        def reduce(p: Any, c: Any, b2: Any, h: Any) -> tuple:
            d = p - c
            r2 = jnp.sum(d * d, axis=1)
            return (
                jnp.argmax(r2),
                jnp.max(r2),
                jnp.sum(r2 > b2, dtype=jnp.int64),
                jnp.sum(jnp.all(jnp.abs(d) <= h, axis=1), dtype=jnp.int64),
            )

        _REDUCE = jax.jit(reduce)
    return _REDUCE(pos, centre, beyond2, half)


_REDUCE: Any = None


def domain_check(
    state: Any,
    masses: Any = None,
    *,
    leaf_size: int = 64,
    cubic_bounds: bool = True,
    sample: int = 1 << 20,
    quantile: float = 0.999,
) -> dict:
    """The Morton domain a state will be built on, and the depth it implies.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]`` packed state or ``[N, 3]`` positions (device or host).
    masses : Any
        Unused (kept for the runner's call); the centre is the median position.
    leaf_size : int
        Particles per cell leaf at most (``GalaxySettings.leaf_size``).
    cubic_bounds : bool
        The box is a cube on the longest extent (``TreeConfig.cubic_bounds``);
        otherwise the largest side is still the one that sets the level-21 cell.
    sample : int
        Rows of the strided subsample the median centre and the quantile radius
        are taken on.
    quantile : float
        The radius quantile (default the 99.9th percentile).

    Returns
    -------
    dict
        Host values: ``n``, ``box_lower`` / ``box_upper`` (the particles' AABB),
        ``box_extent`` (its largest side: the cube's side under ``cubic_bounds``),
        ``cell21``, ``centre`` (the subsample's median), ``r_max`` and
        ``farthest`` (index, position), ``quantile``, ``r_quantile``,
        ``dominance`` (``box_extent / (2 r_quantile)``), ``n_beyond`` (particles
        beyond ``10 r_quantile``), ``central_count`` (particles within half a
        level-21 cell of the centre, per axis), ``leaves_sharing`` (``ceil(
        central_count / leaf_size)``), ``depth_bound`` (``64 + ceil(log2
        leaves_sharing)``, an upper bound when the centre is dense down to the
        cell scale) and ``depth_overflow_possible`` (``depth_bound > 64``).

    Raises
    ------
    ValueError
        If there is no particle.
    """
    import jax
    import jax.numpy as jnp

    del masses, cubic_bounds  # the largest side decides the cell either way
    arr = jnp.asarray(state)
    pos = arr[:, 0, :] if arr.ndim == 3 else arr
    n = int(pos.shape[0])
    if n < 1:
        raise ValueError("domain_check needs at least one particle")
    lower = np.asarray(jax.device_get(jnp.min(pos, axis=0)), np.float64)
    upper = np.asarray(jax.device_get(jnp.max(pos, axis=0)), np.float64)
    extent = float(np.max(upper - lower))
    cell21 = extent / float(1 << _MORTON_LEVELS)
    stride = max(1, n // max(1, int(sample)))
    sub = np.asarray(jax.device_get(pos[::stride]), np.float64)
    centre = np.median(sub, axis=0)
    r_sub = np.sqrt(np.sum((sub - centre) ** 2, axis=1))
    r_q = float(np.quantile(r_sub, quantile))
    far, r2_max, n_beyond, central = jax.device_get(
        _reductions(
            pos,
            jnp.asarray(centre, pos.dtype),
            jnp.asarray((10.0 * r_q) ** 2, pos.dtype),
            jnp.asarray(0.5 * cell21, pos.dtype),
        )
    )
    far, n_beyond, central = int(far), int(n_beyond), int(central)
    r_max = float(np.sqrt(float(r2_max)))
    far_pos = np.asarray(jax.device_get(pos[far]), np.float64).tolist()
    sharing = max(1, math.ceil(central / max(1, int(leaf_size))))
    depth_bound = _MAX_TREE_LEVELS + math.ceil(math.log2(sharing))
    return {
        "n": n,
        "box_lower": lower.tolist(),
        "box_upper": upper.tolist(),
        "box_extent": extent,
        "cell21": cell21,
        "centre": centre.tolist(),
        "r_max": r_max,
        "farthest": {"index": far, "position": far_pos},
        "quantile": float(quantile),
        "r_quantile": r_q,
        "dominance": extent / (2.0 * r_q) if r_q > 0 else math.inf,
        "n_beyond": n_beyond,
        "central_count": central,
        "leaves_sharing": sharing,
        "depth_bound": depth_bound,
        "depth_overflow_possible": depth_bound > _MAX_TREE_LEVELS,
    }


def format_domain_check(report: dict) -> str:
    """:func:`domain_check`'s numbers in one sentence.

    Parameters
    ----------
    report : dict
        :func:`domain_check`.

    Returns
    -------
    str
        The box, the cell, the outliers and the depth bound.
    """
    far = report["farthest"]
    pos = ", ".join(f"{v:.6g}" for v in far["position"])
    q = 100.0 * float(report["quantile"])
    return (
        f"Morton box {report['box_extent']:.6g} across (level-21 cell "
        f"{report['cell21']:.6g}); farthest particle #{far['index']} at ({pos}), "
        f"{report['r_max']:.6g} from the median centre; {q:g} % of the particles "
        f"lie within {report['r_quantile']:.6g}, so the box is "
        f"{report['dominance']:.3g}x their diameter, and {report['n_beyond']} of "
        f"{report['n']} lie beyond 10x that radius; {report['central_count']} "
        f"particles share the central level-21 cell ({report['leaves_sharing']} "
        f"leaves): tree depth up to {report['depth_bound']} levels (the bound is "
        f"{_MAX_TREE_LEVELS})"
    )


def domain_guard(
    state: Any,
    *,
    mode: Literal["warn", "error", "off"] = "warn",
    leaf_size: int = 64,
    cubic_bounds: bool = True,
    dominance_warn: float = DOMINANCE_WARN,
    sample: int = 1 << 20,
) -> Optional[dict]:
    """Warn about, or refuse, a state whose Morton box a few particles dominate.

    Parameters
    ----------
    state : Any
        ``[N, 2, 3]`` or ``[N, 3]``.
    mode : Literal["warn", "error", "off"]
        ``"warn"`` (the runner's default): a :class:`DomainWarning` when the box is
        more than ``dominance_warn`` times the bulk's diameter (the depth bound is
        named then; it is an upper bound, and the exact depth is checked by the
        first prepare, which raises past 64). ``"error"``: ``ValueError``
        instead. ``"off"``: nothing is measured.
    leaf_size : int
        Particles per cell leaf at most.
    cubic_bounds : bool
        As in :func:`domain_check`.
    dominance_warn : float
        The ``dominance`` threshold.
    sample : int
        As in :func:`domain_check`.

    Returns
    -------
    Optional[dict]
        :func:`domain_check`, or ``None`` when ``mode="off"``.

    Raises
    ------
    ValueError
        Under ``mode="error"`` when the guard fires, or on an unknown mode.
    """
    if mode not in ("warn", "error", "off"):
        raise ValueError(f"domain_guard mode must be warn, error or off, got {mode!r}")
    if mode == "off":
        return None
    report = domain_check(
        state, leaf_size=leaf_size, cubic_bounds=cubic_bounds, sample=sample
    )
    if float(report["dominance"]) > float(dominance_warn):
        deep = bool(report["depth_overflow_possible"])
        message = (
            "domain guard: the Morton box is dominated by a few far particles"
            + (
                ", and the radix tree may be deeper than its 64 levels "
                "(yggdrax.MAX_TREE_LEVELS: a deeper tree raises at the first "
                "prepare)"
                if deep
                else ""
            )
            + ". "
            + format_domain_check(report)
            + ". Clip or remove the outliers, or run them on the escaper path "
            "(jaccpot.escapers.EscaperPolicy)."
        )
        if mode == "error":
            raise ValueError(message)
        warnings.warn(message, DomainWarning, stacklevel=3)
    return report

"""Persist and reuse ``MutualCapacities`` across runs and devices.

The mutual lane's capacities are resolved by measuring a built topology, and one of
them -- the wavefront ``queue`` -- cannot be derived from a finished topology at
all, only found by trial (see :class:`~jaccpot.mutual.force.MutualCapacities` and
``BlockStepFMM._resolve_queue_capacity``). That trial is the bulk of
``freeze_template``: seeding it from the leaf count brought N = 1e6 leaf 64 from
51.05 s to 16.77 s, and a recorded preset removes even the remaining 2-3 probes.

This is the mutual analogue of :mod:`jaccpot.distributed.cap_presets`, which does
the same job for the target-centric lane, and it deliberately differs in two ways.

**The key carries leaf_size, theta and order**, not just the particle count and
device count. Those are not incidental: the capacities move by 4x between leaf 64
and leaf 256 at fixed N, and theta moves the far/near split directly. The
target-centric table can omit them because its config pins them elsewhere; here they
are the knobs being tuned, so a table keyed without them would hand a leaf-256
profile to a leaf-64 run.

**Scaling is PER FIELD, not one linear rule.** The target-centric ``_scale_caps``
scales every cap linearly in the per-GPU particle count, which is right for pair
lists and wrong for tree depth. Measured (A100, theta 0.7, order 4, leaf 64):

    N       near        far         width    depth
    1e5       524,288     262,144    1,536      16
    1e6     6,291,456   3,145,728   24,576      24

Pair lists and the level width track N; **depth does not** -- it is a tree depth, so
it grows logarithmically. Scaling it linearly would inflate the dense
``(depth, width)`` level schedule that every M2M/L2L cascade scans, which is exactly
the waste the quarter-octave ladder was introduced to remove. Extrapolating that
decade the log rule gives 23 against the measured 24: one shallow, which is the side
to be wrong on.

Under-estimating is safe in a way over-estimating is not: a too-small cap raises a
loud overflow flag and costs a retry, while a too-large one silently costs time on
every rebuild for the whole run.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

__all__ = [
    "MUTUAL_CAP_FIELDS",
    "apply_caps",
    "caps_of",
    "load_presets",
    "lookup",
    "record",
    "save_presets",
]

# Every field that sets a static buffer shape in the mutual lane.
MUTUAL_CAP_FIELDS = ("near", "far", "depth", "width", "queue")

# How each cap responds to the per-GPU particle count. "linear" scales with N;
# "log" adds a margin per doubling, because a tree depth does not scale with N.
_SCALING = {
    "near": "linear",
    "far": "linear",
    "width": "linear",
    "queue": "linear",
    "depth": "log",
}

# Extra depth per doubling of N. Two per doubling lands at 23 where 1e5 -> 1e6 at
# leaf 64 measured 24, i.e. one shallow -- and that direction is deliberate rather
# than tuned away. A shallow guess raises a loud overflow flag and costs one retry;
# a deep one silently inflates the dense (depth, width) level schedule for the whole
# run. Two data points do not justify fitting a sharper law, so this stays a
# conservative rule of thumb rather than a formula pretending to be exact.
_DEPTH_PER_DOUBLING = 2


def caps_of(caps: Any) -> dict[str, Any]:
    """Extract the cap values from a ``MutualCapacities``.

    Parameters
    ----------
    caps : Any
        A ``MutualCapacities`` (or anything exposing the same attributes).

    Returns
    -------
    dict[str, Any]
        The cap fields as a plain dict, ready to serialise.
    """
    return {f: getattr(caps, f) for f in MUTUAL_CAP_FIELDS}


def apply_caps(caps: Any, values: dict[str, Any]) -> Any:
    """Return a copy of ``caps`` with the fields present in ``values`` applied.

    Parameters
    ----------
    caps : Any
        The ``MutualCapacities`` to update.
    values : dict[str, Any]
        Cap fields to override; absent fields keep their current value.

    Returns
    -------
    Any
        A new ``MutualCapacities``.
    """
    present = {
        f: int(values[f]) for f in MUTUAL_CAP_FIELDS if values.get(f) is not None
    }
    return caps._replace(**present)


def _key(per_gpu_n: int, ndev: int, leaf_size: int, theta: float, order: int) -> str:
    """Profile identity for the table.

    Parameters
    ----------
    per_gpu_n : int
        Particles per GPU.
    ndev : int
        Device count.
    leaf_size : int
        Tree leaf size.
    theta : float
        Opening angle, rounded so 0.7 and 0.7000001 share an entry.
    order : int
        Expansion order.

    Returns
    -------
    str
        The table key.
    """
    return (
        f"{int(per_gpu_n)}:{int(ndev)}:{int(leaf_size)}:"
        f"{round(float(theta), 4)}:{int(order)}"
    )


def load_presets(path: Optional[str]) -> dict:
    """Load the presets table from JSON, or an empty table if missing.

    Parameters
    ----------
    path : Optional[str]
        File to read; ``None`` or a missing file yields ``{}``.

    Returns
    -------
    dict
        The presets table.
    """
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_presets(path: str, presets: dict) -> None:
    """Write the presets table atomically.

    Parameters
    ----------
    path : str
        Destination file.
    presets : dict
        The table to write.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(presets, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _scale_caps(caps: dict[str, Any], num: int, den: int) -> dict[str, Any]:
    """Extrapolate caps from ``den`` particles per GPU to ``num``, per field.

    Parameters
    ----------
    caps : dict[str, Any]
        Caps measured at ``den`` particles per GPU.
    num : int
        Target particles per GPU.
    den : int
        Particles per GPU the caps were measured at.

    Returns
    -------
    dict[str, Any]
        Extrapolated caps. ``depth`` grows logarithmically, everything else
        linearly; see the module docstring for why that distinction matters.
    """
    out: dict[str, Any] = {}
    ratio = max(float(num), 1.0) / max(float(den), 1.0)
    doublings = max(0.0, math.log2(ratio))
    for f in MUTUAL_CAP_FIELDS:
        v = caps.get(f)
        if v is None:
            out[f] = None
        elif _SCALING[f] == "log":
            out[f] = int(v) + int(math.ceil(_DEPTH_PER_DOUBLING * doublings))
        else:
            out[f] = int((int(v) * int(num) + int(den) - 1) // int(den))
    return out


def lookup(
    presets: dict,
    per_gpu_n: int,
    ndev: int,
    leaf_size: int,
    theta: float,
    order: int,
) -> Optional[dict]:
    """Caps for this profile: exact, else nearest larger, else nearest smaller scaled.

    Only ever matches entries with the same ``(ndev, leaf_size, theta, order)`` --
    extrapolating across leaf size or theta would be guessing, since those change the
    far/near split rather than merely its size.

    Preference order, and why: an exact hit is used as-is; otherwise the nearest
    LARGER particle count at the same profile, which is a safe over-estimate; failing
    that the nearest smaller, scaled up. A scaled seed is a starting point, not an
    answer -- the resolver still verifies and grows it, so a slight under-estimate
    costs a retry rather than correctness.

    Parameters
    ----------
    presets : dict
        The table.
    per_gpu_n : int
        Particles per GPU wanted.
    ndev : int
        Device count.
    leaf_size : int
        Tree leaf size.
    theta : float
        Opening angle.
    order : int
        Expansion order.

    Returns
    -------
    Optional[dict]
        Caps, or None if the table holds nothing for this profile.
    """
    exact = _key(per_gpu_n, ndev, leaf_size, theta, order)
    if exact in presets:
        return presets[exact]["caps"]
    suffix = f":{int(ndev)}:{int(leaf_size)}:{round(float(theta), 4)}:{int(order)}"
    same = [
        (int(k.split(":")[0]), v["caps"])
        for k, v in presets.items()
        if k.endswith(suffix)
    ]
    if not same:
        return None
    larger = [(n, c) for n, c in same if n >= int(per_gpu_n)]
    if larger:
        return min(larger, key=lambda t: t[0])[1]
    n0, c0 = max(same, key=lambda t: t[0])
    return _scale_caps(c0, int(per_gpu_n), n0)


def record(
    presets: dict,
    per_gpu_n: int,
    ndev: int,
    leaf_size: int,
    theta: float,
    order: int,
    total_n: int,
    caps: dict[str, Any],
) -> dict:
    """Insert or update this profile's entry, in place.

    Parameters
    ----------
    presets : dict
        The table to update.
    per_gpu_n : int
        Particles per GPU.
    ndev : int
        Device count.
    leaf_size : int
        Tree leaf size.
    theta : float
        Opening angle.
    order : int
        Expansion order.
    total_n : int
        Total particle count, recorded for provenance.
    caps : dict[str, Any]
        The resolved caps.

    Returns
    -------
    dict
        The same table, updated.
    """
    presets[_key(per_gpu_n, ndev, leaf_size, theta, order)] = {
        "per_gpu_n": int(per_gpu_n),
        "ndev": int(ndev),
        "leaf_size": int(leaf_size),
        "theta": round(float(theta), 4),
        "order": int(order),
        "total_n": int(total_n),
        "caps": {
            f: (None if caps.get(f) is None else int(caps[f]))
            for f in MUTUAL_CAP_FIELDS
        },
    }
    return presets

"""Persistent capacity presets for the distributed FMM driver.

The ``auto_scale_caps`` retry loop DISCOVERS the right traversal-buffer capacities for
a given (per-GPU N, ndev, IC) but pays a ``shard_map`` recompile per retry. Persisting
the converged caps keyed by problem size lets a repeat run at a known size START from
them -- zero retries, a single compile. This is the cheap realisation of "size the caps
from a pre-count": ``auto_scale`` *is* the pre-count, and we cache what it found.

The overflow flags stay the safety net: if a preset undersizes (e.g. a denser IC at the
same N), ``auto_scale`` still grows the caps and the caller can refresh the preset with
the newly converged values. Caps are per-device, so the natural key is (per-GPU N, ndev);
total N is recorded too. A preset is a validated STARTING POINT that transfers within an
IC family (same morphology), not a guarantee across wildly different distributions.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

from .fmm import DistributedFMMConfig

#: MAC types that add nothing to a presets key: the three geometric literals the
#: caps rule was calibrated against. Anything else is a jaccpot-level policy whose
#: caps are derived differently, so it gets its own slot -- see :func:`_key`.
_GEOMETRIC_MAC_TYPES = frozenset({"bh", "engblom", "dehnen"})

__all__ = [
    "CAP_FIELDS",
    "apply_caps",
    "caps_of",
    "load_presets",
    "lookup",
    "record",
    "save_presets",
]

# The traversal-buffer capacities the retry loop grows -- everything that affects a
# static buffer shape. Order-independent; None means "driver right-sizes it".
CAP_FIELDS = (
    "max_interactions_per_node",
    "max_neighbors_per_leaf",
    "max_pair_queue",
    "cross_max_interactions_per_node",
    "cross_max_neighbors_per_leaf",
    "cross_max_pair_queue",
    "treecode_near_cap",
    "treecode_far_cap",
    "cross_far_cap",
)


def caps_of(config: DistributedFMMConfig) -> dict[str, Any]:
    """Extract the cap values from a config (e.g. the converged ``result.config``)."""
    return {f: getattr(config, f) for f in CAP_FIELDS}


def apply_caps(
    config: DistributedFMMConfig, caps: dict[str, Any]
) -> DistributedFMMConfig:
    """Return a copy of ``config`` with the cap fields present in ``caps`` applied."""
    return dataclasses.replace(config, **{f: caps[f] for f in CAP_FIELDS if f in caps})


def _key(
    per_gpu_n: int,
    ndev: int,
    theta: float = 0.0,
    leaf_size: int = 0,
    mac_type: str = "",
) -> str:
    """Build the presets key.

    ``theta`` and ``leaf_size`` are part of the key because the caps depend on both
    and a preset recorded under one is NOT valid under another. The wavefront queue
    requirement scales as ``theta ** -1.5`` (measured), so a preset taken at
    theta 0.7 is a 2.8x under-estimate at theta 0.4 -- and an under-sized queue
    truncates the walk SILENTLY, reading faster with only ``self_near_pairs`` as the
    witness. `leaf_size` sets ``num_leaves``, which every cap is derived from.

    ``mac_type`` is here for the same reason one step further out. Under
    ``"dehnen_error"`` the self walk accepts on a pair policy and NOT on theta, so
    its caps come from measured criterion coefficients rather than from theta
    and it is deliberately LARGER than the geometric one at the same theta. Without
    the criterion in the key those two runs share an entry: a geometric run at
    theta 0.8 records the smaller caps, the next criterion run reads them back, and
    the walk truncates silently -- exactly the failure the theta paragraph above
    describes, reached from a different direction. This is the interaction-cache-key
    hazard from ``docs/dehnen_mass_mac_status_and_plan.md`` in a second cache.

    Only a non-geometric ``mac_type`` appears in the key, so every key an existing
    presets file already holds is byte-identical to what it was.

    Legacy two-part keys (no theta, no leaf) are still read, so an existing presets
    file keeps working; they are simply never written any more.

    Parameters
    ----------
    per_gpu_n : int
        Particles per device.
    ndev : int
        Device count.
    theta : float
        Opening angle. 0.0 reproduces the legacy two-part key.
    leaf_size : int
        Leaf size. 0 reproduces the legacy two-part key.
    mac_type : str
        Acceptance criterion. Empty or a geometric literal adds nothing to the key.

    Returns
    -------
    str
        The table key.
    """

    if not theta and not leaf_size:
        return f"{int(per_gpu_n)}:{int(ndev)}"
    base = f"{int(per_gpu_n)}:{int(ndev)}:t{float(theta):g}:l{int(leaf_size)}"
    mac = str(mac_type).strip()
    if mac and mac not in _GEOMETRIC_MAC_TYPES:
        return f"{base}:m{mac}"
    return base


def load_presets(path: Optional[str]) -> dict:
    """Load the presets table from a JSON file (empty dict if missing/unset)."""
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_presets(path: str, presets: dict) -> None:
    """Atomically write the presets table to ``path``."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(presets, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _scale_caps(caps: dict[str, Any], num: int, den: int) -> dict[str, Any]:
    """Scale integer caps by num/den (ceil); None stays None."""
    return {
        f: (None if caps.get(f) is None else int((int(caps[f]) * num + den - 1) // den))
        for f in CAP_FIELDS
    }


def lookup(
    presets: dict,
    per_gpu_n: int,
    ndev: int,
    theta: float = 0.0,
    leaf_size: int = 0,
    mac_type: str = "",
) -> Optional[dict]:
    """Caps for (per_gpu_n, ndev): exact match, else the nearest LARGER per-GPU N at the
    same ndev (a safe over-estimate), else the nearest SMALLER preset SCALED UP by the
    per-GPU-N ratio. The scaled seed is only a starting point -- auto_scale refines it (and
    the caller can refresh the preset), so a slight under/over-estimate just costs a retry
    or a little memory, not correctness. Extrapolating from a nearby preset is what makes
    calibration cheap at an unmeasured N (few retries instead of the full doubling ladder
    from the small defaults). None if no preset at this ndev at all."""
    k = _key(per_gpu_n, ndev, theta, leaf_size, mac_type)
    if k in presets:
        return presets[k]["caps"]
    same = [
        (int(kk.split(":")[0]), v["caps"])
        for kk, v in presets.items()
        if kk.endswith(f":{int(ndev)}")
    ]
    if not same:
        return None
    larger = [(n, c) for n, c in same if n >= int(per_gpu_n)]
    if larger:
        return min(larger, key=lambda t: t[0])[1]
    n0, c0 = max(same, key=lambda t: t[0])  # largest smaller
    return _scale_caps(c0, int(per_gpu_n), n0)


def record(
    presets: dict,
    per_gpu_n: int,
    ndev: int,
    total_n: int,
    caps: dict[str, Any],
    theta: float = 0.0,
    leaf_size: int = 0,
    mac_type: str = "",
) -> dict:
    """Insert/update the (per_gpu_n, ndev) entry with the given caps (in place)."""
    presets[_key(per_gpu_n, ndev, theta, leaf_size, mac_type)] = {
        "per_gpu_n": int(per_gpu_n),
        "ndev": int(ndev),
        "total_n": int(total_n),
        "caps": {
            f: (None if caps.get(f) is None else int(caps[f])) for f in CAP_FIELDS
        },
    }
    return presets

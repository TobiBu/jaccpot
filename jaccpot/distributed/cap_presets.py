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
    return dataclasses.replace(
        config, **{f: caps[f] for f in CAP_FIELDS if f in caps}
    )


def _key(per_gpu_n: int, ndev: int) -> str:
    return f"{int(per_gpu_n)}:{int(ndev)}"


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


def lookup(presets: dict, per_gpu_n: int, ndev: int) -> Optional[dict]:
    """Caps for (per_gpu_n, ndev): exact match, else the nearest LARGER per-GPU N at the
    same ndev (a safe over-estimate). None if nothing usable."""
    k = _key(per_gpu_n, ndev)
    if k in presets:
        return presets[k]["caps"]
    larger = [
        (int(kk.split(":")[0]), v["caps"])
        for kk, v in presets.items()
        if kk.endswith(f":{int(ndev)}") and int(kk.split(":")[0]) >= int(per_gpu_n)
    ]
    if larger:
        return min(larger, key=lambda t: t[0])[1]
    return None


def record(
    presets: dict, per_gpu_n: int, ndev: int, total_n: int, caps: dict[str, Any]
) -> dict:
    """Insert/update the (per_gpu_n, ndev) entry with the given caps (in place)."""
    presets[_key(per_gpu_n, ndev)] = {
        "per_gpu_n": int(per_gpu_n),
        "ndev": int(ndev),
        "total_n": int(total_n),
        "caps": {
            f: (None if caps.get(f) is None else int(caps[f])) for f in CAP_FIELDS
        },
    }
    return presets

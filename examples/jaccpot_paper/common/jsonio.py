"""Read/write the paper's ``results/**/*.json`` artifacts.

Stdlib only, and no JAX at module scope: every GPU bench script must be able to
import this before it has selected a device.

The contract this module enforces is the one the manuscript depends on -- a
figure value must never be typed from memory, so every artifact carries the
configuration that produced it. :func:`write_result` refuses to write a record
whose ``config`` is missing any of :data:`REQUIRED_CONFIG_KEYS`, which is what
makes "traceable" a property of the pipeline rather than of whoever ran it.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterable, Mapping, Optional

__all__ = [
    "REQUIRED_CONFIG_KEYS",
    "RESULTS_ROOT",
    "read_result",
    "repo_root",
    "results_path",
    "write_result",
]

# `parents[4]` walks examples/jaccpot_paper/common/jsonio.py -> repo root.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS_ROOT = _REPO_ROOT / "results"

# The axes that make a number reproducible. A bench script that cannot fill one
# of these in does not yet know what it measured.
REQUIRED_CONFIG_KEYS: tuple[str, ...] = (
    "n",
    "theta",
    "order",
    "basis",
    "seed",
    "device",
    "precision",
)


def repo_root() -> pathlib.Path:
    """Return the repository root, independent of the caller's cwd."""

    return _REPO_ROOT


def results_path(*parts: str) -> pathlib.Path:
    """Return ``results/<parts...>``, e.g. ``results_path("validation", "x.json")``."""

    return RESULTS_ROOT.joinpath(*parts)


def _missing_config_keys(
    config: Mapping[str, Any], required: Iterable[str]
) -> list[str]:
    missing = []
    for key in required:
        if key not in config:
            missing.append(key)
        elif config[key] is None:
            # An explicit None is how a script says "this axis does not apply"
            # (e.g. `order` for a direct-sum-only run). That is a deliberate
            # statement and is allowed; simply omitting the key is not.
            continue
    return missing


def write_result(
    path: str | pathlib.Path,
    *,
    config: Mapping[str, Any],
    data: Any,
    meta: Optional[Mapping[str, Any]] = None,
    required_config_keys: Iterable[str] = REQUIRED_CONFIG_KEYS,
    indent: int = 2,
) -> pathlib.Path:
    """Write ``{"config": ..., "meta": ..., "data": ...}`` to ``path``.

    Idempotent: rerunning a bench script overwrites the same path rather than
    accumulating timestamped siblings, so a figure always has exactly one input.

    Raises
    ------
    ValueError
        If ``config`` omits any of ``required_config_keys``. Pass an explicit
        ``None`` for an axis that genuinely does not apply to the measurement.
    """

    missing = _missing_config_keys(config, required_config_keys)
    if missing:
        raise ValueError(
            "result config is missing required key(s) "
            f"{missing}; every artifact must record the config that produced it "
            "so the manuscript stays traceable. Pass an explicit None for an "
            "axis that does not apply to this measurement."
        )

    out = pathlib.Path(path)
    if not out.is_absolute():
        out = RESULTS_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": dict(config),
        "meta": dict(meta or {}),
        "data": data,
    }
    # Write via a sibling temp file so an interrupted run cannot leave a
    # half-written JSON that a notebook would then happily plot.
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, sort_keys=False) + "\n")
    tmp.replace(out)
    return out


def read_result(
    path: str | pathlib.Path,
    *,
    require_config_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Load an artifact written by :func:`write_result`.

    Notebooks call this and nothing else -- if a figure needs a number that is
    not in here, the fix is to add it to the bench script and rerun, never to
    compute it in the notebook.
    """

    src = pathlib.Path(path)
    if not src.is_absolute():
        src = RESULTS_ROOT / src
    if not src.exists():
        raise FileNotFoundError(
            f"{src} does not exist. Run the bench script that produces it first; "
            "figure notebooks never recompute."
        )
    payload = json.loads(src.read_text())

    for key in ("config", "meta", "data"):
        if key not in payload:
            raise ValueError(f"{src} is not a paper result artifact (no {key!r} key)")

    missing = _missing_config_keys(payload["config"], require_config_keys)
    if missing:
        raise ValueError(f"{src} config is missing required key(s) {missing}")
    return payload


def config_caption(config: Mapping[str, Any], keys: Iterable[str] = ()) -> str:
    """Render a compact ``k=v`` provenance string for a figure caption/annotation.

    Lets a figure state the config it was measured at without any risk of the
    annotation drifting from the data it sits next to.
    """

    selected = tuple(keys) or tuple(config.keys())
    parts = []
    for key in selected:
        if key in config and config[key] is not None:
            parts.append(f"{key}={config[key]}")
    return "  ".join(parts)

"""The accuracy suite's result schema and its ``record`` / ``check`` comparison.

A result is one JSON object::

    {
      "schema": "jaccpot.accuracy/1",
      "problem": "plummer",              # the standard problem
      "tier": "local",                   # local | horeka-1card | horeka-node | ...
      "config": {...},                   # everything that defines the run
      "comparable": ["num_particles", ...],  # config keys a check requires EQUAL
      "metrics": {"force.rel_l2": 1.2e-4, ...},  # flat name -> float
      "tolerances": {"force.rel_l2": {"kind": "ratio", "value": 1.5}, ...},
      "series": {...},                   # time series for plots (never checked)
      "provenance": {...}                # commit, host, devices, date (never checked)
    }

``record`` saves a result as the baseline. ``check`` compares a new result
against it: first the ``comparable`` config keys must be equal (a number is only a
comparison if both sides were measured the same way: the subsampled-probe lesson),
then every metric with a tolerance in the BASELINE must hold:

``"abs"``     ``|new - base| <= value``
``"rel"``     ``|new - base| <= value * |base|``
``"ratio"``   ``new <= value * base`` (an error may not grow by more than ``value``)
``"ceiling"`` ``new <= value`` (an absolute target, e.g. plan U4's ``1e-4``)
``"floor"``   ``new >= value``

A metric the baseline names but the new result lacks fails; non-finite values fail.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

#: the schema tag every result carries
SCHEMA_VERSION = "jaccpot.accuracy/1"

#: the tolerance kinds :func:`check_result` understands
TOLERANCE_KINDS = ("abs", "rel", "ratio", "ceiling", "floor")


@dataclass(frozen=True)
class Tolerance:
    """One metric's tolerance.

    Attributes
    ----------
    kind : str
        One of :data:`TOLERANCE_KINDS`.
    value : float
        Its parameter.
    why : str
        One line: why this tolerance (recorded in the baseline).
    """

    kind: str
    value: float
    why: str = ""

    def __post_init__(self) -> None:
        """Validate ``kind``.

        Raises
        ------
        ValueError
            On an unknown kind.
        """
        if self.kind not in TOLERANCE_KINDS:
            raise ValueError(f"tolerance kind {self.kind!r} not in {TOLERANCE_KINDS}")

    def as_dict(self) -> dict:
        """JSON-able form.

        Returns
        -------
        dict
            ``kind``, ``value``, ``why``.
        """
        return {"kind": self.kind, "value": float(self.value), "why": self.why}

    def holds(self, new: float, base: Optional[float]) -> bool:
        """Whether ``new`` passes against ``base``.

        Parameters
        ----------
        new : float
            The new value.
        base : Optional[float]
            The baseline's value (unused by ``ceiling`` / ``floor``).

        Returns
        -------
        bool
            The verdict (``False`` on a non-finite value).
        """
        if not math.isfinite(new):
            return False
        v = float(self.value)
        if self.kind in ("ceiling", "floor"):
            return new <= v if self.kind == "ceiling" else new >= v
        if base is None or not math.isfinite(base):
            return False
        if self.kind == "abs":
            return abs(new - base) <= v
        if self.kind == "rel":
            return abs(new - base) <= v * abs(base)
        return new <= v * base  # ratio


@dataclass
class CheckOutcome:
    """What :func:`check_result` found.

    Attributes
    ----------
    passed : bool
        Every comparable key equal and every tolerance held.
    rows : list
        One dict per checked metric: ``metric``, ``base``, ``new``, ``kind``,
        ``value``, ``ok``.
    mismatched_config : dict
        ``key -> (base, new)`` for the comparable keys that differ.
    missing : list
        Metrics with a tolerance that the new result lacks.
    """

    passed: bool
    rows: list = field(default_factory=list)
    mismatched_config: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    def table(self) -> str:
        """A plain-text table of the rows.

        Returns
        -------
        str
            One line per metric, then the config mismatches.
        """
        lines = [f"{'metric':<34} {'base':>12} {'new':>12} {'tolerance':>18}  ok"]
        for r in self.rows:
            base = "-" if r["base"] is None else f"{r['base']:.4e}"
            new = "-" if r["new"] is None else f"{r['new']:.4e}"
            tol = f"{r['kind']} {r['value']:.3g}"
            lines.append(
                f"{r['metric']:<34} {base:>12} {new:>12} {tol:>18}  "
                f"{'PASS' if r['ok'] else 'FAIL'}"
            )
        for k, (b, n) in self.mismatched_config.items():
            lines.append(
                f"config {k!r} differs: baseline {b!r} vs new {n!r} (not comparable)"
            )
        lines.append("CHECK " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _clean(value: Any) -> Any:
    """Make ``value`` JSON-able (numpy scalars/arrays to Python, NaN kept as str).

    Parameters
    ----------
    value : Any
        Any nested value.

    Returns
    -------
    Any
        The JSON-able value.
    """
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def make_result(
    problem: str,
    config: dict,
    metrics: dict,
    *,
    tier: str = "local",
    comparable: tuple = (),
    tolerances: Optional[dict] = None,
    series: Optional[dict] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Build a result object.

    Parameters
    ----------
    problem : str
        The standard problem's name.
    config : dict
        The run's defining configuration.
    metrics : dict
        Flat ``name -> float``.
    tier : str
        ``local``, ``horeka-1card``, ``horeka-node``, ...
    comparable : tuple
        Config keys a check requires equal.
    tolerances : Optional[dict]
        ``name -> Tolerance`` (or its dict form).
    series : Optional[dict]
        Time series (not checked).
    provenance : Optional[dict]
        Commit, host, devices, ... (not checked).

    Returns
    -------
    dict
        The result.

    Raises
    ------
    KeyError
        If a comparable key is not in ``config`` or a tolerance names no metric.
    """
    for k in comparable:
        if k not in config:
            raise KeyError(f"comparable key {k!r} is not in the config")
    tol = {}
    for name, t in (tolerances or {}).items():
        if name not in metrics:
            raise KeyError(f"tolerance for unknown metric {name!r}")
        tol[name] = t.as_dict() if isinstance(t, Tolerance) else dict(t)
    return {
        "schema": SCHEMA_VERSION,
        "problem": str(problem),
        "tier": str(tier),
        "config": _clean(config),
        "comparable": list(comparable),
        "metrics": {
            k: (float(v) if v is not None else None) for k, v in metrics.items()
        },
        "tolerances": tol,
        "series": _clean(series or {}),
        "provenance": _clean(provenance or {}),
    }


def save_result(result: dict, path: Union[str, "os.PathLike"]) -> str:
    """Write ``result`` as JSON (atomic rename).

    Parameters
    ----------
    result : dict
        From :func:`make_result`.
    path : Union[str, os.PathLike]
        Target file.

    Returns
    -------
    str
        The path.
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(_clean(result), fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_result(path: Union[str, "os.PathLike"]) -> dict:
    """Read a result and check its schema tag.

    Parameters
    ----------
    path : Union[str, os.PathLike]
        The file.

    Returns
    -------
    dict
        The result.

    Raises
    ------
    ValueError
        On a missing or unknown schema tag.
    """
    with open(os.fspath(path)) as fh:
        result = json.load(fh)
    if result.get("schema") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema {result.get('schema')!r}, expected {SCHEMA_VERSION!r}"
        )
    return result


def _as_float(value: Any) -> Optional[float]:
    """A metric value as a float (``None`` stays ``None``; ``"nan"`` strings parse).

    Parameters
    ----------
    value : Any
        Stored value.

    Returns
    -------
    Optional[float]
        The float.
    """
    if value is None:
        return None
    return float(value)


def check_result(baseline: dict, new: dict) -> CheckOutcome:
    """Compare ``new`` against ``baseline`` with the baseline's tolerances.

    Parameters
    ----------
    baseline : dict
        The recorded baseline.
    new : dict
        A fresh result of the same problem.

    Returns
    -------
    CheckOutcome
        The verdict and its rows.
    """
    out = CheckOutcome(passed=True)
    if baseline.get("problem") != new.get("problem"):
        out.mismatched_config["problem"] = (baseline.get("problem"), new.get("problem"))
    for k in baseline.get("comparable", []):
        b, n = baseline["config"].get(k), new.get("config", {}).get(k)
        if b != n:
            out.mismatched_config[k] = (b, n)
    for name, t in sorted(baseline.get("tolerances", {}).items()):
        tol = Tolerance(kind=t["kind"], value=float(t["value"]), why=t.get("why", ""))
        base = _as_float(baseline["metrics"].get(name))
        if name not in new.get("metrics", {}) or new["metrics"][name] is None:
            out.missing.append(name)
            out.rows.append(
                {
                    "metric": name,
                    "base": base,
                    "new": None,
                    "kind": tol.kind,
                    "value": tol.value,
                    "ok": False,
                }
            )
            continue
        val = _as_float(new["metrics"][name])
        ok = tol.holds(val, base)
        out.rows.append(
            {
                "metric": name,
                "base": base,
                "new": val,
                "kind": tol.kind,
                "value": tol.value,
                "ok": bool(ok),
            }
        )
    out.passed = not out.mismatched_config and all(r["ok"] for r in out.rows)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "TOLERANCE_KINDS",
    "CheckOutcome",
    "Tolerance",
    "check_result",
    "load_result",
    "make_result",
    "save_result",
]

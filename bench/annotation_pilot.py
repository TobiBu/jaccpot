"""Answer STYLE_GUIDE 4.1's question for code you cannot call by hand.

THE QUESTION, AND WHY IT NEEDS A TOOL
-------------------------------------
Section 4.1 says: shape-annotate an array parameter when **nothing else validates it**,
and leave alone what is already checked. That is a question about behaviour, so it is
answered by feeding a function a malformed input and seeing whether it complains.

For a public surface that is easy, and Phase 1 of the rollout did it by hand: 13 of 15
malformed inputs were already rejected in `downward/` + `upward/`, 20 of 23 in `runtime/`,
which is why those phases annotated 9 functions rather than 36. But Phase 2's targets are
internal kernels whose entry points take **9 to 13 array arguments each**. Nobody is
hand-building a valid call to
`_compute_leaf_p2p_prepared_large_n_pairs_target_blocks_tiled_impl`.

So: record the shapes of real calls during a test run, rebuild an equivalent call, and
perturb one array at a time.

WHY IT RECORDS SHAPES AND NOT VALUES
------------------------------------
The obvious design records the actual arguments and replays them. It does not work here.
`vmap` and `scan` bodies always see **tracers**, jit or no jit, and section 4.4 counts 122
such sites. Measured: with `JAX_DISABLE_JIT=1`, three of four `complex_ops` targets still
recorded nothing, because `JAX_DISABLE_JIT` does not disable `vmap` tracing.

A tracer carries no value but it does carry `.shape` and `.dtype`. Non-array arguments
(orders, flags, block sizes) stay concrete inside a trace. So this records shape+dtype for
arrays and values for scalars, and the replay synthesizes zeros. That is enough for the
question actually being asked -- "is a wrong shape rejected?" -- because rank and
broadcasting depend on shape alone.

THE CONTROL, WHICH IS THE PART THAT KEEPS IT HONEST
---------------------------------------------------
Synthesized zeros are not always a valid stand-in: a kernel may branch on values, or an
argument may be opaque and get defaulted wrong. So every function is replayed **unperturbed
first**. If that control raises, the function is reported INCONCLUSIVE and counted in
neither direction.

This matters more than it sounds. Counting an inconclusive function as "already validated"
would understate the remaining annotation work, and the entire point of the exercise is to
size that work honestly. A pilot that silently swallowed its own failures would produce the
most comfortable possible answer, which is the same failure mode
`bench/annotation_census.py` refuses when it declines to report an empty walk as zero.

Functions with an argument this cannot describe are reported UNREPLAYABLE, also uncounted.

WHAT IT DOES NOT MEASURE
------------------------
* **Value-dependent validation.** A check like `if mask.sum() != n: raise` is invisible
  here, because the replay passes zeros.
* **Whether a rejection is a GOOD rejection.** An opaque
  `TypeError: mul got incompatible shapes for broadcasting: (8, 3), (7, 1)` counts as
  rejected, and it is -- but it names neither parameter, and replacing it with a shape
  annotation is still an improvement. The tool measures whether anything complains, not
  how well.
* **Lanes that never ran.** Same limitation as `bench/annotation_capture.py`: a function
  reached only through one backend is recorded only as that backend called it. Read the
  recorded function count against the targeted count before trusting a per-module rate.

USAGE
-----
Record, during any part of the suite::

    PILOT_TARGETS='jaccpot.operators.complex_ops:evaluate_local_complex,complex_dot' \\
    PILOT_OUT=/tmp/pilot.pkl \\
      pytest -p bench.annotation_pilot tests/unit tests/integration

Then replay::

    python -m bench.annotation_pilot replay /tmp/pilot.pkl

The report ranks modules by the fraction of perturbations **silently accepted**, which is
section 4.1's predictor -- and which is not the same ordering as bare-parameter count.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import pickle
import sys
from collections import Counter
from functools import wraps
from pathlib import Path
from typing import Any

__all__ = [
    "PERTURBATIONS",
    "describe_argument",
    "main",
    "replay",
    "shape_perturbations",
]

_TARGETS_ENV = "PILOT_TARGETS"
_OUT_ENV = "PILOT_OUT"
_MAX_ENV = "PILOT_MAX_PER_FN"

_recorded: dict[str, list] = {}
_installed: list[tuple[Any, str, Any]] = []


def describe_argument(value: Any) -> tuple[str, Any, str]:
    """Describe one argument well enough to rebuild an equivalent for replay.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    tuple of (str, Any, str)
        ``("array", shape, dtype)`` for anything array-like -- shape and dtype
        only, because a tracer has both and no value. ``("scalar", value, type)``
        or ``("tuple", value, "tuple")`` for values that survive a trace.
        ``("opaque", None, type)`` otherwise, which makes the call unreplayable.
    """
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return ("array", tuple(int(d) for d in value.shape), str(value.dtype))
        except TypeError:
            return ("opaque", None, type(value).__name__)
    if isinstance(value, (int, float, bool, str, type(None))):
        return ("scalar", value, type(value).__name__)
    if isinstance(value, tuple) and all(
        isinstance(v, (int, float, bool, str, type(None))) for v in value
    ):
        return ("tuple", value, "tuple")
    return ("opaque", None, type(value).__name__)


def _wrap(function: Any, label: str, limit: int) -> Any:
    """Wrap a function so the first ``limit`` calls record their argument shapes.

    Parameters
    ----------
    function : Any
        The function to wrap.
    label : str
        ``module:name``.
    limit : int
        How many calls to record.

    Returns
    -------
    Any
        The wrapper.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(_recorded.get(label, [])) < limit:
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                snapshot = {
                    name: describe_argument(value)
                    for name, value in bound.arguments.items()
                }
                replayable = not any(k == "opaque" for k, _, _ in snapshot.values())
                _recorded.setdefault(label, []).append((snapshot, replayable))
            except TypeError:
                # A call that does not match the signature is about to raise on
                # its own. Record nothing and let the real error surface.
                pass
        return function(*args, **kwargs)

    return wrapper


def _binding_sites(defining_module: Any, name: str, original: Any) -> list:
    """Find every module-level name bound to this object.

    Same reasoning as ``bench/annotation_capture.py``: ``from X import Y`` binds a
    separate name and Python resolves the call against that one, so patching only
    the defining module observes nothing. Matched on identity, never on name.

    Parameters
    ----------
    defining_module : Any
        Module the target was imported from.
    name : str
        Its name there.
    original : Any
        The unwrapped object.

    Returns
    -------
    list
        ``(module, attribute)`` pairs, the defining module first.
    """
    sites = [(defining_module, name)]
    for module in list(sys.modules.values()):
        if module is None or module is defining_module:
            continue
        try:
            members = vars(module)
        except TypeError:
            continue
        for attribute, value in list(members.items()):
            if value is original:
                sites.append((module, attribute))
    return sites


def pytest_configure(config: Any) -> None:
    """Install the recorders at session start, if targets were requested.

    Parameters
    ----------
    config : Any
        The pytest config object.
    """
    spec = os.environ.get(_TARGETS_ENV)
    if not spec:
        return
    limit = int(os.environ.get(_MAX_ENV, "1"))
    missing: list[str] = []
    for group in spec.split(";"):
        if not group.strip():
            continue
        module_path, names = group.split(":", 1)
        module = importlib.import_module(module_path)
        for name in (n.strip() for n in names.split(",") if n.strip()):
            original = getattr(module, name, None)
            if original is None:
                missing.append(f"{module_path}:{name}")
                continue
            wrapper = _wrap(original, f"{module_path}:{name}", limit)
            for holder, attribute in _binding_sites(module, name, original):
                setattr(holder, attribute, wrapper)
                _installed.append((holder, attribute, original))
    print(f"\nannotation_pilot: instrumented {len(_installed)} binding sites")
    if missing:
        # Loud, because a typo'd target otherwise reads as "already validated".
        print(f"annotation_pilot: MISSING TARGETS {missing}")


def pytest_unconfigure(config: Any) -> None:
    """Remove the recorders and write the recording.

    Parameters
    ----------
    config : Any
        The pytest config object.
    """
    for holder, attribute, original in _installed:
        setattr(holder, attribute, original)
    if _recorded:
        out = Path(os.environ.get(_OUT_ENV, "annotation_pilot.pkl"))
        with out.open("wb") as handle:
            pickle.dump(_recorded, handle)
        print(f"\nannotation_pilot: recorded {len(_recorded)} functions -> {out}")


PERTURBATIONS = (
    "leading axis -1",
    "trailing axis -1",
    "extra leading axis",
    "flattened",
)


def shape_perturbations(shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Return the shape corruptions worth trying for one recorded shape.

    Each is a mistake a caller could plausibly make: an off-by-one length, a
    dropped spatial component, an unsqueezed batch axis, a flattened buffer.
    Degenerate cases are skipped -- shrinking an axis of extent 1 gives an empty
    array, which many kernels legitimately accept.

    Parameters
    ----------
    shape : tuple of int
        The recorded shape.

    Returns
    -------
    list of tuple
        ``(label, perturbed_shape)`` pairs.
    """
    out: list[tuple[str, tuple[int, ...]]] = []
    if shape and shape[0] > 1:
        out.append(("leading axis -1", (shape[0] - 1,) + shape[1:]))
    if len(shape) >= 2 and shape[-1] > 1:
        out.append(("trailing axis -1", shape[:-1] + (shape[-1] - 1,)))
    if len(shape) >= 1:
        out.append(("extra leading axis", (1,) + shape))
    if len(shape) >= 2:
        total = 1
        for dim in shape:
            total *= dim
        out.append(("flattened", (total,)))
    return out


def _build(kind: str, value: Any, meta: str) -> Any:
    """Rebuild one argument from its description.

    Parameters
    ----------
    kind : str
        ``"array"``, ``"scalar"`` or ``"tuple"``.
    value : Any
        Shape for an array, the value otherwise.
    meta : str
        Dtype string for an array.

    Returns
    -------
    Any
        A stand-in argument.
    """
    if kind != "array":
        return value
    import jax.numpy as jnp

    return jnp.zeros(value, dtype=jnp.dtype(meta))


def replay(recorded: dict[str, list]) -> tuple[str, Counter]:
    """Replay every recorded call with one array perturbed at a time.

    Parameters
    ----------
    recorded : dict
        As written by the pytest plugin.

    Returns
    -------
    tuple of (str, Counter)
        The report, and the tallies behind it.
    """
    lines: list[str] = []
    tally: Counter = Counter()
    per: dict[str, Counter] = {}

    def bump(label: str, key: str) -> None:
        per.setdefault(label.rsplit(":", 1)[0], Counter())[key] += 1

    for label in sorted(recorded):
        snapshot, replayable = recorded[label][0]
        module_path, function_name = label.rsplit(":", 1)
        function = getattr(importlib.import_module(module_path), function_name)

        if not replayable:
            opaque = [k for k, (kind, _, _) in snapshot.items() if kind == "opaque"]
            lines.append(
                f"\n{label}\n  UNREPLAYABLE -- opaque args: {', '.join(opaque)}"
            )
            tally["unreplayable"] += 1
            bump(label, "unreplayable")
            continue

        base = {name: _build(*desc) for name, desc in snapshot.items()}
        try:
            function(**base)
        except (
            Exception
        ) as error:  # noqa: BLE001 -- any failure invalidates the control
            lines.append(
                f"\n{label}\n  INCONCLUSIVE -- control failed: "
                f"{type(error).__name__}: {str(error).splitlines()[0][:70]}"
            )
            tally["inconclusive"] += 1
            bump(label, "inconclusive")
            continue

        arrays = [k for k, (kind, _, _) in snapshot.items() if kind == "array"]
        lines.append(f"\n{label}\n  control OK, {len(arrays)} array params")
        bump(label, "ok")
        for name in arrays:
            shape = snapshot[name][1]
            for plabel, new_shape in shape_perturbations(shape):
                arguments = dict(base)
                arguments[name] = _build("array", new_shape, snapshot[name][2])
                try:
                    function(**arguments)
                except Exception:  # noqa: BLE001 -- any complaint counts as rejection
                    tally["rejected"] += 1
                    bump(label, "rejected")
                else:
                    lines.append(
                        f"    >> ACCEPTED  {name}{list(shape)} -> "
                        f"{list(new_shape)}  ({plabel})"
                    )
                    tally["accepted"] += 1
                    bump(label, "accepted")

    total = tally["accepted"] + tally["rejected"]
    lines.append(
        f"\n  === {tally['rejected']}/{total} perturbations rejected; "
        f"{tally['accepted']} SILENTLY ACCEPTED ==="
    )
    lines.append(
        f"  functions: {tally['inconclusive']} inconclusive, "
        f"{tally['unreplayable']} unreplayable (neither is counted above)"
    )
    lines.append(
        f"\n{'module':<38} {'tested':>7} {'ACCEPTED':>9} {'rate':>6}  ok/inc/unrep"
    )

    def rate(counter: Counter) -> float:
        seen = counter["accepted"] + counter["rejected"]
        return counter["accepted"] / seen if seen else -1.0

    for module_path in sorted(per, key=lambda m: -rate(per[m])):
        counter = per[module_path]
        seen = counter["accepted"] + counter["rejected"]
        shown = f"{counter['accepted'] / seen:.0%}" if seen else "  -"
        lines.append(
            f"  {module_path:<36} {seen:>7} {counter['accepted']:>9} {shown:>6}  "
            f"{counter['ok']}/{counter['inconclusive']}/{counter['unreplayable']}"
        )
    return "\n".join(lines), tally


def main(argv: list[str] | None = None) -> int:
    """Replay a recording from the command line.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay_parser = subparsers.add_parser("replay", help="replay a recording")
    replay_parser.add_argument("path", type=Path, help="the pickle the plugin wrote")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"{args.path} does not exist -- was the recording run?", file=sys.stderr)
        return 1
    with args.path.open("rb") as handle:
        recorded = pickle.load(handle)
    report, _ = replay(recorded)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Derive array shapes by execution, and say how strong the evidence is.

WHY THIS EXISTS
---------------
STYLE_GUIDE section 4.2 says to derive a shape by running the code, never from the
docstring, and it has the measurements to justify saying so: ``NodeMultipoleData.packed``
documented ``sh_size(order)`` and is ``total_coefficients(order)``, wrong at every order
the package runs at. Three PRs have now done that derivation, each with a throwaway
instrumentation script. This is the reusable version.

But recording shapes is the easy half, and it is not the half that went wrong. The
``farleaves`` incident (section 4.3) is the warning: shapes derived from 64 captured calls
made ``leaves`` and ``farleaves`` look like the same axis, an annotation asserted they were,
and 7 tests broke the moment a decorator made it enforced. Every one of those 64 calls
was honest. The capture simply never entered the octree backend, where the two differ --
5 against 3.

So the rule section 4.3 states is **capture coverage bounds annotation validity**, and a
tool that reports shapes without reporting its own coverage invites exactly that mistake.
This one reports both, and refuses to let an axis equality look stronger than it is:

* **Distinct extents behind every equality.** An equality observed across 64 calls that
  all ran at one problem size is one observation, not 64. The summary prints the number
  of DISTINCT values an equality was seen at and flags fewer than two as unproven.
  **This is the weak guard, and it would NOT by itself have caught ``farleaves``**: those
  captures did span problem sizes, so the equality would have read as well-evidenced. It
  catches the narrower mistake of a capture taken at one size.
* **Which tests produced the calls.** A per-function list of the test files that reached
  it, so "the octree backend never ran" is visible in the output instead of being an
  assumption nobody wrote down. **This is the guard that matters for ``farleaves``**, and
  it is the one that needs a human: no amount of captured data reveals a lane that was
  never run. Read this list before annotating and ask which lane is missing from it.
* **Constant extents are named as constants.** An axis that never varied is reported as a
  literal, not as a name. ``3`` is the spatial dimension and always will be; a leading
  axis that happened to be 3 in every captured call is a trap.

USAGE
-----
Capture, by running any part of the suite with the plugin enabled::

    JACCPOT_CAPTURE_TARGETS='jaccpot.operators.multipole_utils:pack_tensor,unpack_tensor' \\
    JACCPOT_CAPTURE_OUT=/tmp/caps.jsonl \\
      pytest -p bench.annotation_capture tests/unit tests/integration

Targets are ``module:name`` or ``module:name1,name2`` and may be repeated with ``;``.
Then summarise::

    python -m bench.annotation_capture summarize /tmp/caps.jsonl

Run the capture over as many lanes as the parameter can reach, not just the fastest one.
For anything in the far field that means including a run that selects the octree backend;
for the distributed lanes it means forced host devices. The summary cannot tell you a lane
was missing -- only which ones it saw.

WHAT IT CANNOT SEE
------------------
**Methods.** The wrapper rebinds a MODULE attribute, so a function defined inside a class
cannot be targeted at all -- `runtime/fmm_sweeps.py`'s three diagnostic methods are the
first real case. Where such a method delegates to a module-level function, capture the
function and the shapes are the same; where it does not, this tool has nothing to say and
will report no observations, which must not be read as "never called". Supporting
`module:Class.method` is the obvious extension and is not done yet.

The wrapper replaces the module attribute, so it observes calls that resolve the name at
call time -- which is the normal case, including calls from inside the defining module.
A reference captured into a closure or a ``jit`` cache *before* the patch went on is not
observed. Under ``jit`` the recorded values are tracers, which is the useful case: the
shape is the abstract one, and that is the one an annotation is checked against.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

__all__ = [
    "capture_shapes",
    "main",
    "parse_targets",
    "summarize",
]

_TARGETS_ENV = "JACCPOT_CAPTURE_TARGETS"
_OUT_ENV = "JACCPOT_CAPTURE_OUT"


def parse_targets(spec: str) -> dict[str, list[str]]:
    """Parse a target specification into module to function names.

    Parameters
    ----------
    spec : str
        ``module:name1,name2;other.module:name`` -- semicolons separate modules,
        commas separate names within one module.

    Returns
    -------
    dict of str to list of str
        Module path to the function names to wrap.

    Raises
    ------
    ValueError
        If a group has no ``:`` separating module from names.
    """
    targets: dict[str, list[str]] = {}
    for group in spec.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"target {group!r} is not module:name")
        module, names = group.split(":", 1)
        targets.setdefault(module.strip(), []).extend(
            n.strip() for n in names.split(",") if n.strip()
        )
    return targets


def _describe(value: Any) -> dict[str, Any] | None:
    """Describe one argument, if it is array-like.

    Parameters
    ----------
    value : Any
        The argument value.

    Returns
    -------
    dict or None
        ``{"shape": [...], "dtype": str}`` for anything carrying a shape, else
        None. Python scalars and None return None: they have no axes, so they
        cannot inform a shape annotation.
    """
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(d) for d in shape]
    except TypeError:
        # A shape that is not a tuple of ints -- an object array, or something
        # merely exposing the attribute. Not ours to annotate.
        return None
    return {"shape": dims, "dtype": str(getattr(value, "dtype", "?"))}


def _current_test() -> str:
    """Return the pytest node id of the running test, if there is one.

    Returns
    -------
    str
        The node id with pytest's ``(setup)``/``(call)`` suffix removed, or
        ``"<no test>"`` outside a pytest run.
    """
    raw = os.environ.get("PYTEST_CURRENT_TEST", "")
    return raw.split(" (")[0] if raw else "<no test>"


def _wrap(function: Callable[..., Any], label: str, sink: Any) -> Callable[..., Any]:
    """Wrap one function so each call records the shape of its array arguments.

    Parameters
    ----------
    function : Callable
        The function to wrap.
    label : str
        ``module:name``, recorded with each observation.
    sink : Any
        An open text file; one JSON object is written per call.

    Returns
    -------
    Callable
        The wrapper, carrying the original's metadata.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError:
            # A call that does not match the signature is about to raise anyway.
            # Record nothing and let the real error surface unchanged -- swallowing
            # it here would turn a caller's bug into a missing observation.
            return function(*args, **kwargs)
        parameters = {
            name: description
            for name, value in bound.arguments.items()
            if (description := _describe(value)) is not None
        }
        record = {
            "function": label,
            "test": _current_test(),
            "parameters": parameters,
        }
        sink.write(json.dumps(record) + "\n")
        return function(*args, **kwargs)

    return wrapper


def _binding_sites(
    defining_module: Any, name: str, original: Any
) -> list[tuple[Any, str]]:
    """Find every module-level name currently bound to the target function.

    Patching only the defining module is not enough, and this is the failure mode
    that made the tool report "never called" for a function called on every step.
    ``from yggdrax.multipole_utils import multi_power`` binds a SEPARATE name in
    the importing module's globals, and Python resolves the call against that one.
    Rebinding the attribute on the defining module leaves the importer's reference
    untouched, so the wrapper never runs.

    So every loaded module is scanned for an attribute that IS this object. The
    identity test is what keeps it safe: a same-named function from a different
    module -- and `jaccpot.operators.multipole_utils` duplicates eight of
    yggdrax's names -- is a different object and is left alone.

    Parameters
    ----------
    defining_module : Any
        The module the target was imported from.
    name : str
        The function's name in that module.
    original : Any
        The unwrapped function object.

    Returns
    -------
    list of tuple
        ``(module, attribute_name)`` pairs to patch, the defining module first.
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


@contextmanager
def capture_shapes(targets: dict[str, list[str]], out_path: Path) -> Iterator[None]:
    """Wrap the target functions for the duration of the block.

    Parameters
    ----------
    targets : dict of str to list of str
        Module path to the function names to wrap.
    out_path : Path
        JSONL file to append observations to.

    Yields
    ------
    None
        The block runs with the wrappers installed; they are removed on exit even
        if it raises.
    """
    with ExitStack() as stack:
        sink = stack.enter_context(out_path.open("a", encoding="utf-8"))
        for module_path, names in targets.items():
            module = importlib.import_module(module_path)
            for name in names:
                original = getattr(module, name)
                wrapper = _wrap(original, f"{module_path}:{name}", sink)
                sites = _binding_sites(module, name, original)
                for holder, attribute in sites:
                    setattr(holder, attribute, wrapper)
                    stack.callback(setattr, holder, attribute, original)
                print(
                    f"annotation_capture: {module_path}:{name} patched at "
                    f"{len(sites)} binding site(s)"
                )
        yield


# --- pytest plugin -----------------------------------------------------------
#
# Enabled with `-p bench.annotation_capture`, configured by environment rather
# than by command-line options. The env route is what lets the same invocation be
# pasted into a CI job or a distributed run, where the pytest command line is
# already fixed by something else.


def pytest_configure(config: Any) -> None:
    """Install the wrappers at session start, if targets were requested.

    Parameters
    ----------
    config : Any
        The pytest config object; the ExitStack is stashed on it.
    """
    spec = os.environ.get(_TARGETS_ENV)
    if not spec:
        return
    out_path = Path(os.environ.get(_OUT_ENV, "annotation_capture.jsonl"))
    stack = ExitStack()
    stack.enter_context(capture_shapes(parse_targets(spec), out_path))
    config._annotation_capture_stack = stack
    print(f"\nannotation_capture: recording to {out_path}")


def pytest_unconfigure(config: Any) -> None:
    """Remove the wrappers at session end.

    Parameters
    ----------
    config : Any
        The pytest config object.
    """
    stack = getattr(config, "_annotation_capture_stack", None)
    if stack is not None:
        stack.close()


# --- summary -----------------------------------------------------------------


def _axis_key(parameter: str, position: int) -> str:
    """Name one axis of one parameter.

    Parameters
    ----------
    parameter : str
        The parameter name.
    position : int
        Zero-based axis index.

    Returns
    -------
    str
        ``"positions[0]"``.
    """
    return f"{parameter}[{position}]"


def summarize(records: list[dict[str, Any]]) -> str:
    """Render what the capture saw, and how strong its evidence is.

    Parameters
    ----------
    records : list of dict
        Observations as written by the plugin.

    Returns
    -------
    str
        The report.
    """
    by_function: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_function[record["function"]].append(record)

    lines: list[str] = []
    for function in sorted(by_function):
        calls = by_function[function]
        tests = sorted({c["test"] for c in calls})
        files = sorted({t.split("::")[0] for t in tests})
        lines.append(f"{function}")
        lines.append(f"  {len(calls)} calls, {len(tests)} tests, {len(files)} files")

        shapes: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        dtypes: dict[str, set[str]] = defaultdict(set)
        for call in calls:
            for parameter, description in call["parameters"].items():
                shapes[parameter].add(tuple(description["shape"]))
                dtypes[parameter].add(description["dtype"])

        for parameter in sorted(shapes):
            observed = sorted(shapes[parameter])
            ranks = {len(s) for s in observed}
            rank_note = "" if len(ranks) == 1 else "  RANK VARIES"
            shown = ", ".join(str(list(s)) for s in observed[:6])
            if len(observed) > 6:
                shown += f", ... ({len(observed)} distinct)"
            lines.append(f"    {parameter}: {shown}{rank_note}")
            lines.append(f"      dtypes: {', '.join(sorted(dtypes[parameter]))}")

        lines.extend(f"      {line}" for line in _equality_report(calls))
        lines.append("    tests reaching it:")
        lines.extend(f"      {f}" for f in files)
        lines.append("")

    if not lines:
        lines.append("no observations -- the targets were never called")
    return "\n".join(lines)


def _equality_report(calls: list[dict[str, Any]]) -> list[str]:
    """Report which axes were always equal, and at how many distinct values.

    An equality holding across every recorded call is worth nothing if every call
    ran at one problem size, so the count that matters is the number of DISTINCT
    extents it was seen at, not the number of calls.

    This is a weak guard and it is worth being clear which mistake it catches.
    It catches a capture taken at a single problem size. It does NOT catch
    ``farleaves``, whose captures did span sizes and whose equality was real in
    every lane that ran -- only the coverage list can raise that, and only for a
    reader who notices which lane is absent.

    Parameters
    ----------
    calls : list of dict
        The observations for one function.

    Returns
    -------
    list of str
        Report lines; empty when no two axes were ever both present.
    """
    per_call: list[dict[str, int]] = []
    for call in calls:
        axes: dict[str, int] = {}
        for parameter, description in call["parameters"].items():
            for position, extent in enumerate(description["shape"]):
                axes[_axis_key(parameter, position)] = extent
        per_call.append(axes)

    constant: dict[str, set[int]] = defaultdict(set)
    for axes in per_call:
        for key, extent in axes.items():
            constant[key].add(extent)

    # Only axes that VARY can pose an axis-naming question. Pairing the constant
    # ones reports `delta[0] == raw_dipole[0]` for every pair of 3s in the
    # signature -- 55 lines of "3 == 3" on one function, measured -- and buries
    # the one equality a reader has to adjudicate. Constants are already handled
    # above, as literals.
    equal_pairs: list[tuple[str, str, int]] = []
    keys = sorted(k for k, v in constant.items() if len(v) > 1)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            shared = [a for a in per_call if left in a and right in a]
            if not shared or any(a[left] != a[right] for a in shared):
                continue
            distinct = len({a[left] for a in shared})
            equal_pairs.append((left, right, distinct))

    lines: list[str] = []
    frozen = [k for k, v in sorted(constant.items()) if len(v) == 1]
    if frozen:
        lines.append(
            "CONSTANT axes (never varied -- name as a literal, not an axis): "
            + ", ".join(f"{k}={next(iter(constant[k]))}" for k in frozen)
        )
    for left, right, distinct in equal_pairs:
        verdict = "UNPROVEN" if distinct < 2 else f"at {distinct} distinct values"
        lines.append(f"equal: {left} == {right}  ({verdict})")
    if any(d < 2 for _, _, d in equal_pairs):
        lines.append(
            "UNPROVEN equalities held at a single extent. That is what the "
            "farleaves capture reported too -- widen the lanes before naming "
            "these axes the same."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Summarise a capture file from the command line.

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
    summary = subparsers.add_parser("summarize", help="report on a capture file")
    summary.add_argument("path", type=Path, help="the JSONL file the plugin wrote")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"{args.path} does not exist -- was the capture run?", file=sys.stderr)
        return 1
    records = [
        json.loads(line)
        for line in args.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(summarize(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())

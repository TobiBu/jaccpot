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

WHAT COUNTS AS DESCRIBABLE, AND WHY IT GREW
-------------------------------------------
Shape+dtype for arrays, values for scalars -- and, since the `pallas/nearfield_mutual.py`
pass, two more kinds, because three of that module's eight targets were UNREPLAYABLE for
reasons that had nothing to do with the code being hard to describe:

* **Sequences of arrays.** `_block_tile` and `_block_vjp_tiles` take their positions as
  `tuple[Array, Array, Array]`. The old rule described a tuple only when every element
  was a scalar, so these fell through to opaque -- yet each element has exactly the shape
  and dtype the replay needs. They are now described element by element, and the
  perturbation addresses arrays **by path**, so `a_xyz[0]` is corrupted individually.
  That last part is not a detail: rebuilding a container without perturbing what it holds
  would report "control OK, 0 array params", turning a gap the report NAMES into one it
  hides. See `test_arrays_inside_a_container_are_perturbed_and_not_merely_rebuilt`.
* **Dtype objects.** `_pad_inputs` takes the working `dtype`. It is not an array and not a
  scalar, and it rebuilds from its name.

Then two more since the `runtime/_adaptive_policy.py` pass, where 11 of 21 targets came back
UNREPLAYABLE on the same argument and left ~60 of that module's ~99 array parameters
unmeasured -- half a module reported as neither validated nor not:

* **Trees.** `yggdrax`'s tree is described by what `build_tree` NEEDS -- `num_particles`,
  `leaf_size` -- and rebuilt for real, not synthesized. Its 22 array leaves are mutually
  constrained (`node_ranges` partitions the particles, `parent`/`left_child` form the
  topology), so zeros are not a tree and every function taking one failed the control.
  `num_nodes` rides along as a CHECK and not an input, because the node count is
  data-dependent: the same `(n, leaf_size)` gave 31 nodes on one distribution and 37 on
  another. A rebuild that cannot reproduce the recorded count is REFUSED, which surfaces as
  INCONCLUSIVE rather than as a replay against a tree of the wrong size -- arrays recorded
  beside the tree were sized against the original, so that would quietly answer a different
  question.
* **NamedTuples.** `TreeUpwardData` was matched by the plain-tuple rule and rebuilt as a
  bare tuple, so the control died on `AttributeError: 'tuple' object has no attribute
  'multipoles'` -- which reads like the function's problem and was the tool's. The class is
  now remembered and restored, and `_leaves`/`_substitute` walk the fields, so an array
  inside one is perturbed by path (`upward[2][3]`) like any other. Without that last part it
  would report "control OK, 0 array params" -- the same invisible gap the container note
  above warns about.

Measured effect on `_adaptive_policy`: 11 UNREPLAYABLE and 0 measured became **11 measured,
0 unreplayable, 0 inconclusive** -- and 46 of 104 perturbations on that half are silently
accepted, 44%, against 41% on the half that was measurable all along.

All four widen what can be measured; none widens what counts as validated. A container
holding one un-describable element is still UNREPLAYABLE, checked recursively.

A THIRD THING THAT KEEPS IT HONEST: ONLY CALLS THAT RETURNED
-----------------------------------------------------------
A description is committed **after** the call returns, not before it. The suite is full of
calls that raise on purpose -- every shape-contract test is one -- and the description of a
deliberately malformed call is not a valid control. Recording it made the *better-annotated*
modules the harder ones to measure, which is exactly backwards.

Measured on `pallas/nearfield_mutual.py`: all four annotated entry points came back
INCONCLUSIVE, their controls failing the very contracts the module had just been given,
because the one recorded description per function had been taken from
`test_nearfield_mutual_shape_contracts.py` -- `ma` at `(1, 3, 4)` from the extra-leading-axis
case, and a `b` side one slot narrower than the `a` side from the mismatched-widths case. The
2026-08-30 pass measured those same functions fine, because the contracts and their negative
tests did not exist yet.

Relatedly, the replay now takes the first **replayable** recording rather than the first one,
which is what makes `PILOT_MAX_PER_FN` above 1 worth setting.

**An existing recording does not benefit.** The description is frozen in the pickle, so a
pass recorded before this change still reports its trees and NamedTuples as opaque --
verified, an old recording replays to the identical 33/81 and 11 unreplayable. Re-record to
pick the new kinds up.

WHAT IT DOES NOT MEASURE
------------------------
* **Value-dependent validation.** A check like `if mask.sum() != n: raise` is invisible
  here, because the replay passes zeros.
* **Whether a rejection is a GOOD rejection.** An opaque
  `TypeError: mul got incompatible shapes for broadcasting: (8, 3), (7, 1)` counts as
  rejected, and it is -- but it names neither parameter, and replacing it with a shape
  annotation is still an improvement. The tool measures whether anything complains, not
  how well.
* **Methods.** `bench/annotation_capture.py` gained `module:Class.method` support in the
  same change that added the two kinds above; this tool did NOT, and the asymmetry is
  principled rather than unfinished. Capture only OBSERVES a call, so patching the class
  is enough. The pilot has to MAKE one, which needs a `self` -- and `self` is an
  `FMMEngine`, precisely the kind of object `describe_argument` reports as opaque. So the
  `runtime/` mixins can have their shapes derived by capture but not their section 4.1
  question answered here. Anyone wanting that needs a real engine fixture, which is a
  different tool: it would have to build one, not describe one.
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

Under `pytest-xdist` each worker writes its own shard -- `/tmp/pilot.gw0.pkl` and friends --
and the replay merges every shard beside the path you name, so pass the same path either way.

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
from typing import Any, Iterator

import numpy as np

__all__ = [
    "PERTURBATIONS",
    "describe_argument",
    "is_replayable",
    "main",
    "replay",
    "shape_perturbations",
]

_TARGETS_ENV = "PILOT_TARGETS"
_OUT_ENV = "PILOT_OUT"
_MAX_ENV = "PILOT_MAX_PER_FN"

_recorded: dict[str, list] = {}
_installed: list[tuple[Any, str, Any]] = []


def _is_tree(value: Any) -> bool:
    """Is this a yggdrax tree, duck-typed rather than by class?

    Duck-typed on purpose: the value handed to `runtime/_adaptive_policy` is a
    `RadixTree`, the annotation says `Tree`, and `Tree.__getattr__` forwards to a
    topology -- so an `isinstance` check would have to name all three and would go stale
    when a fourth arrives. What the rebuild actually needs is these four attributes.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    bool
        True if the value exposes the particle count, leaf size, node count and node
        ranges a rebuild needs.
    """
    return all(
        hasattr(value, attribute)
        for attribute in ("num_particles", "leaf_size", "num_nodes", "node_ranges")
    )


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
        ``("container", [description, ...], "tuple" | "list")`` for a sequence
        holding anything else, described element by element. ``("dtype", name,
        type)`` for a dtype object. ``("opaque", None, type)`` otherwise, which
        makes the call unreplayable.

    Notes
    -----
    The branch ORDER is load-bearing, because ``numpy.dtype`` is far more
    permissive than it looks: ``np.dtype(None)`` is ``float64`` and
    ``np.dtype("float64")`` parses the string, so an ``Optional[Array]``
    argument recorded as ``None`` and any string argument would both come back
    as dtypes if the dtype branch ran first. Scalars are matched before it for
    that reason, and sequences before it too -- ``np.dtype((np.int32, 3))`` is
    also a valid dtype spec.
    """
    if _is_tree(value):
        # A tree is described by what `build_tree` needs to make an equivalent one, not
        # by its 22 array leaves. `num_particles` and `leaf_size` are static aux data on
        # the pytree, so they survive tracing as plain ints -- which is the only reason
        # this is possible at all. `num_nodes` rides along as a CHECK, not an input: the
        # replay refuses a rebuild whose node count differs, because the arrays recorded
        # beside the tree were sized against the original.
        return (
            "tree",
            {
                "num_particles": int(value.num_particles),
                "leaf_size": int(value.leaf_size),
                "num_nodes": int(value.num_nodes),
                "dtype": str(value.node_ranges.dtype),
            },
            type(value).__name__,
        )
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return ("array", tuple(int(d) for d in value.shape), str(value.dtype))
        except TypeError:
            # Both attributes present and the shape unreadable means this is not
            # an array at all: numpy's scalar TYPES (`np.float32`, the class) carry
            # `shape` and `dtype` as descriptors, and they are a legitimate dtype
            # spelling. Fall through rather than concluding opaque here.
            pass
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        # A NamedTuple, described element by element like any container but REMEMBERING
        # its class. Flattening it to a bare tuple is what made `build_adaptive_policy_state`
        # and `resolve_dehnen_geometry` INCONCLUSIVE: the control died on
        # `AttributeError: 'tuple' object has no attribute 'multipoles'`, which reads like
        # the function's problem and is the tool's.
        cls = type(value)
        return (
            "namedtuple",
            {
                "module": cls.__module__,
                "name": cls.__name__,
                "fields": [describe_argument(v) for v in value],
            },
            cls.__name__,
        )
    if isinstance(value, (int, float, bool, str, type(None))):
        return ("scalar", value, type(value).__name__)
    if isinstance(value, tuple) and all(
        isinstance(v, (int, float, bool, str, type(None))) for v in value
    ):
        return ("tuple", value, "tuple")
    if isinstance(value, (tuple, list)):
        kind = "tuple" if isinstance(value, tuple) else "list"
        return ("container", [describe_argument(v) for v in value], kind)
    try:
        return ("dtype", str(np.dtype(value)), type(value).__name__)
    except (TypeError, ValueError):
        return ("opaque", None, type(value).__name__)


def is_replayable(description: tuple[str, Any, str]) -> bool:
    """Say whether one described argument can be rebuilt, looking inside containers.

    The recursion is the point. A tuple holding one un-describable element is
    NOT replayable, and reporting it as replayable would not hide the problem --
    it would relabel it, because the control would then fail and the function
    would be reported INCONCLUSIVE instead of UNREPLAYABLE. Both are uncounted,
    so the tally survives either way, but the diagnosis in the report would name
    the wrong cause and send the reader looking for a value-dependent kernel
    that does not exist.

    Parameters
    ----------
    description : tuple of (str, Any, str)
        As returned by :func:`describe_argument`.

    Returns
    -------
    bool
        False if the description contains an opaque leaf at any depth.
    """
    kind, value, _ = description
    if kind == "opaque":
        return False
    if kind == "container":
        return all(is_replayable(element) for element in value)
    if kind == "namedtuple":
        return all(is_replayable(element) for element in value["fields"])
    return True


def _leaves(
    description: tuple[str, Any, str], wanted: str, prefix: tuple[int, ...] = ()
) -> Iterator[tuple[tuple[int, ...], Any, str]]:
    """Walk a description and yield every leaf of one kind, with its path.

    Parameters
    ----------
    description : tuple of (str, Any, str)
        As returned by :func:`describe_argument`.
    wanted : str
        The leaf kind to yield, ``"array"`` or ``"opaque"``.
    prefix : tuple of int
        Element indices walked so far.

    Yields
    ------
    tuple of (tuple of int, Any, str)
        ``(path, value, meta)`` -- path is empty for a description that is
        itself a leaf, and holds one index per container level otherwise.
    """
    kind, value, meta = description
    if kind == "container":
        for index, element in enumerate(value):
            yield from _leaves(element, wanted, prefix + (index,))
    elif kind == "namedtuple":
        for index, element in enumerate(value["fields"]):
            yield from _leaves(element, wanted, prefix + (index,))
    elif kind == wanted:
        yield prefix, value, meta


def _substitute(
    description: tuple[str, Any, str],
    path: tuple[int, ...],
    replacement: tuple[str, Any, str],
) -> tuple[str, Any, str]:
    """Return a copy of a description with the leaf at ``path`` replaced.

    Parameters
    ----------
    description : tuple of (str, Any, str)
        The description to copy.
    path : tuple of int
        Element indices, as produced by :func:`_leaves`. Empty replaces the
        whole description.
    replacement : tuple of (str, Any, str)
        The description to put there.

    Returns
    -------
    tuple of (str, Any, str)
        The copy. The original is not mutated.
    """
    if not path:
        return replacement
    kind, value, meta = description
    if kind == "namedtuple":
        fields = list(value["fields"])
        fields[path[0]] = _substitute(fields[path[0]], path[1:], replacement)
        return (kind, {**value, "fields": fields}, meta)
    elements = list(value)
    elements[path[0]] = _substitute(elements[path[0]], path[1:], replacement)
    return (kind, elements, meta)


def _format_path(name: str, path: tuple[int, ...]) -> str:
    """Render a parameter name and element path the way the source spells it.

    Parameters
    ----------
    name : str
        The parameter name.
    path : tuple of int
        Element indices into it.

    Returns
    -------
    str
        ``a_xyz`` or ``a_xyz[0]``.
    """
    return name + "".join(f"[{index}]" for index in path)


def _worker_output(out: Path, config: Any) -> Path:
    """Return a per-xdist-worker path, so workers do not overwrite each other.

    `_recorded` is per process. Every worker used to write the SAME `PILOT_OUT` on
    unconfigure, so the surviving file was whichever worker finished last and every other
    worker's functions were silently gone. Measured on `pallas/nearfield_mutual.py` at
    `-n 4`: 7 of 12 targeted functions in the file, the other 5 recorded by workers whose
    write was clobbered -- which reads exactly like "that lane never ran".

    Parameters
    ----------
    out : Path
        The path the user asked for.
    config : Any
        The pytest config; carries `workerinput` only inside an xdist worker.

    Returns
    -------
    Path
        `out` under a single process, else `out` with the worker id folded into the stem.
    """

    worker = getattr(config, "workerinput", {}).get("workerid")
    if not worker:
        return out
    return out.with_name(f"{out.stem}.{worker}{out.suffix}")


def _load_recordings(path: Path) -> dict[str, list[Any]]:
    """Load a recording, merging every per-worker shard beside it.

    Parameters
    ----------
    path : Path
        The path passed to `replay`, as given to `PILOT_OUT`.

    Returns
    -------
    dict
        Label to list of recordings, concatenated across shards in shard order.

    Raises
    ------
    FileNotFoundError
        If neither the path nor any shard beside it exists.
    """

    shards = sorted(path.parent.glob(f"{path.stem}.*{path.suffix}"))
    files = ([path] if path.exists() else []) + [s for s in shards if s != path]
    if not files:
        raise FileNotFoundError(path)

    merged: dict[str, list[Any]] = {}
    for file in files:
        with file.open("rb") as handle:
            for label, entries in pickle.load(handle).items():
                merged.setdefault(label, []).extend(entries)
    return merged


def _first_replayable(
    entries: list[tuple[dict[str, Any], bool]],
) -> tuple[dict[str, Any], bool]:
    """Return the first recording that can be replayed, else the first one.

    Only the first recording was ever used, which made ``PILOT_MAX_PER_FN`` above 1
    pointless: a function whose first observed call carried an opaque argument was
    UNREPLAYABLE even when a later, fully describable call had been recorded beside it.
    Falling back to ``entries[0]`` keeps the report identical when none is replayable,
    so the UNREPLAYABLE path still names the opaque arguments of a real call.

    Parameters
    ----------
    entries : list of (dict, bool)
        The recordings for one function, in the order they were observed.

    Returns
    -------
    tuple of (dict, bool)
        The chosen description and its replayable flag.
    """

    for entry in entries:
        if entry[1]:
            return entry
    return entries[0]


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
        snapshot: dict[str, Any] | None = None
        if len(_recorded.get(label, [])) < limit:
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                snapshot = {
                    name: describe_argument(value)
                    for name, value in bound.arguments.items()
                }
            except TypeError:
                # A call that does not match the signature is about to raise on
                # its own. Record nothing and let the real error surface.
                snapshot = None

        # Call FIRST, and commit the description only if the call returned. A call
        # that raises is not a valid control, and the suite is full of calls that
        # raise on purpose: a shape-contract test's deliberately malformed block
        # would otherwise be recorded as this function's one description and turn
        # every later replay of it INCONCLUSIVE. Measured on
        # `pallas/nearfield_mutual.py`, where four annotated entry points came back
        # inconclusive against descriptions taken from
        # `test_nearfield_mutual_shape_contracts.py`'s negative cases.
        result = function(*args, **kwargs)
        if snapshot is not None and len(_recorded.get(label, [])) < limit:
            replayable = all(is_replayable(d) for d in snapshot.values())
            _recorded.setdefault(label, []).append((snapshot, replayable))
        return result

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
        out = _worker_output(
            Path(os.environ.get(_OUT_ENV, "annotation_pilot.pkl")), config
        )
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


def _build_tree_stand_in(spec: dict[str, Any]) -> Any:
    """Rebuild a REAL tree matching a recorded spec, or refuse.

    A real build, not a synthesized stand-in, and that is the whole point. A tree's 22
    array leaves are mutually constrained -- `node_ranges` partitions the particles,
    `parent`/`left_child` form the topology -- so zeros are not a tree and every function
    taking one would have failed the control. Building over deterministic random
    positions gives arrays that are actually consistent.

    The node count is DATA-dependent, not a function of ``(n, leaf_size)`` alone: the same
    pair gave 31 nodes on one distribution and 37 on another during the
    `_adaptive_policy` pass. So the rebuild is checked against the recorded count and
    REFUSED on a mismatch, which the replay reports as INCONCLUSIVE. Refusing is the
    honest failure: replaying against a tree of the wrong size would compare arrays that
    were sized against the original and quietly answer the wrong question.

    Parameters
    ----------
    spec : dict
        ``num_particles``, ``leaf_size``, ``num_nodes`` and ``dtype``, as recorded by
        :func:`describe_argument`.

    Returns
    -------
    Any
        A freshly built tree with the recorded particle count, leaf size and node count.

    Raises
    ------
    RuntimeError
        If no seed reproduces the recorded node count, so the caller reports
        INCONCLUSIVE rather than replaying against a differently shaped tree.
    """
    import jax.numpy as jnp
    from yggdrax.tree import build_tree

    n = int(spec["num_particles"])
    leaf_size = int(spec["leaf_size"])
    want_nodes = int(spec["num_nodes"])
    dtype = jnp.dtype(spec.get("dtype", "float64"))
    float_dtype = jnp.float64 if dtype.itemsize >= 8 else jnp.float32

    # A few seeds, because the node count is data-dependent and a single uniform draw is
    # not guaranteed to reproduce a clustered original's. Deterministic, so a replay is
    # reproducible.
    for seed in range(8):
        rng = np.random.default_rng(seed)
        positions = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, 3)), float_dtype)
        masses = jnp.asarray(np.abs(rng.standard_normal(n)) + 0.5, float_dtype)
        tree, _, _, _ = build_tree(
            positions, masses, return_reordered=True, leaf_size=leaf_size
        )
        if int(tree.num_nodes) == want_nodes:
            return tree
    raise RuntimeError(
        f"no rebuild over 8 seeds reproduced num_nodes={want_nodes} at "
        f"num_particles={n}, leaf_size={leaf_size}; the node count is data-dependent "
        "and the recorded tree's distribution is not recoverable from the record"
    )


def _build(kind: str, value: Any, meta: str) -> Any:
    """Rebuild one argument from its description.

    Parameters
    ----------
    kind : str
        ``"array"``, ``"scalar"``, ``"tuple"``, ``"container"`` or ``"dtype"``.
    value : Any
        Shape for an array, the element descriptions for a container, the dtype
        name for a dtype, the value itself otherwise.
    meta : str
        Dtype string for an array, ``"tuple"`` or ``"list"`` for a container.

    Returns
    -------
    Any
        A stand-in argument.
    """
    import jax.numpy as jnp

    if kind == "tree":
        return _build_tree_stand_in(value)
    if kind == "namedtuple":
        cls = getattr(importlib.import_module(value["module"]), value["name"])
        return cls(*(_build(*element) for element in value["fields"]))
    if kind == "array":
        return jnp.zeros(value, dtype=jnp.dtype(meta))
    if kind == "container":
        rebuilt = [_build(*element) for element in value]
        return tuple(rebuilt) if meta == "tuple" else rebuilt
    if kind == "dtype":
        return jnp.dtype(value)
    return value


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
        snapshot, replayable = _first_replayable(recorded[label])
        module_path, function_name = label.rsplit(":", 1)
        function = getattr(importlib.import_module(module_path), function_name)

        if not replayable:
            opaque = [
                _format_path(name, path)
                for name, description in snapshot.items()
                for path, _, _ in _leaves(description, "opaque")
            ]
            lines.append(
                f"\n{label}\n  UNREPLAYABLE -- opaque args: {', '.join(opaque)}"
            )
            tally["unreplayable"] += 1
            bump(label, "unreplayable")
            continue

        # Inside the try, not before it. A rebuild can now FAIL -- `_build_tree_stand_in`
        # refuses a tree whose node count it cannot reproduce -- and a refusal is an
        # inconclusive replay, not a crashed report.
        try:
            base = {name: _build(*desc) for name, desc in snapshot.items()}
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

        # Addressed by PATH, not by parameter name, so the arrays inside a
        # container are perturbed too. Rebuilding a container without perturbing
        # what it holds would turn an UNREPLAYABLE function -- reported, and
        # uncounted -- into one that prints "control OK, 0 array params" and
        # contributes nothing, which converts a visible gap into an invisible one.
        arrays = [
            ((name,) + path, shape, dtype)
            for name, description in snapshot.items()
            for path, shape, dtype in _leaves(description, "array")
        ]
        lines.append(f"\n{label}\n  control OK, {len(arrays)} array params")
        bump(label, "ok")
        for (name, *rest), shape, dtype in arrays:
            path = tuple(rest)
            for plabel, new_shape in shape_perturbations(shape):
                arguments = dict(base)
                arguments[name] = _build(
                    *_substitute(snapshot[name], path, ("array", new_shape, dtype))
                )
                try:
                    function(**arguments)
                except Exception:  # noqa: BLE001 -- any complaint counts as rejection
                    tally["rejected"] += 1
                    bump(label, "rejected")
                else:
                    lines.append(
                        f"    >> ACCEPTED  {_format_path(name, path)}{list(shape)} -> "
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

    try:
        recorded = _load_recordings(args.path)
    except FileNotFoundError:
        print(f"{args.path} does not exist -- was the recording run?", file=sys.stderr)
        return 1
    report, _ = replay(recorded)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Memory budget of the fused single-card lane: where the bytes sit, per cap setting.

Step 0 of the memory-reduction plan (``fused-memory-reduction.md``). One
configuration per PROCESS -- the allocator's high-water mark is never reset, so a
second configuration in the same process would inherit the first one's peak.

``--mode budget`` (the default) records, for one ``(N, cap setting)``:

* the eager prepare, phase by phase: ``memory_stats()`` in-use and peak around
  the tree + upward pass, each walk attempt of the eager ladder, the far/near list
  build, the downward pass and the near-field artifacts, plus the 20 largest
  ``jax.live_arrays()`` and the 20 largest leaves of the prepared state by path;
* the compiled programs: ``lower().compile().memory_analysis()`` (argument,
  output, alias, temp) of the eval-only closure and of the ``strict_run_v2``
  runner, and, with ``--dump-dir``, XLA's buffer assignment of both;
* time per force (eval-only, min of ``--eval-repeats``) and per scan step (min of
  ``--reps`` warm ``--steps``-step calls).

The peak is a running maximum, so a phase's own peak is visible only when it sets
a new high-water mark; the table reports both numbers at every boundary.

``--mode drift`` measures how much the list counts move along a rollout: an eager
prepare (which records the far-pair count, near edges and peak wavefront) at each
of ``--drift-steps``, with ``strict_run_v2`` advancing the state in between. Two
ICs: ``plummer`` with equilibrium velocities (Aarseth, Henon & Wielen 1974) and
``disc``, a prefix of the shuffled 25M disc+bulge IC with its masses rescaled and
the analytic NFW halo it was set up in added as the external force.

Cap settings (``--caps``):

* ``bench`` -- the Odisseo harness's caps, ``pow2(200k fit x ceil(N / 200k))``
  for both lists (``compare_force.fast_lane_overrides_for_leaf``), as the N-max
  ladder ran;
* ``named`` -- ``--far-cap`` / ``--near-cap`` exactly (directed pairs, even);
* ``unnamed`` -- neither cap in the environment: the library sizes them.

Run on one card, one process per row::

    PYTHONPATH=/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/sitecustom_wt \\
      JACCPOT_WORKTREE=... YGGDRAX_WORKTREE=... CUDA_VISIBLE_DEVICES=3 \\
      python bench/fused_memory_budget.py --n 2000000 --caps bench --out row.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import json
import os
import sys
import time

# the Odisseo bench harness (codes/, common/); BENCH_DIR on any other machine
sys.path.insert(
    0,
    os.environ.get(
        "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
    ),
)

_DISC_IC = "/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz"
_CAP_VARS = (
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", default="budget", choices=("budget", "drift"))
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument(
        "--ic", default="plummer", choices=("plummer", "plummer_clipped", "disc")
    )
    ap.add_argument(
        "--rmax",
        type=float,
        default=20.0,
        help="plummer_clipped: radius cut in scale radii (a truncated inverse CDF, no "
        "outliers to stretch the per-axis Morton box)",
    )
    ap.add_argument(
        "--prealloc",
        type=float,
        default=0.0,
        help="preallocate this fraction of the card (jax's default allocator mode); 0 "
        "= grow on demand, which fragments after the eager prepare's peak",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--leaf", type=int, default=64)
    ap.add_argument("--theta", type=float, default=0.8)
    # p6 since round 5 (2026-10-05): at cell_min_level 8 it halves the fp64 rel-L2 of p5
    # for ~5 % of the step (clipped Plummer 2e6: 5.0e-4 at 34.0 ms against 9.2e-4 at
    # 32.4) and costs nothing on the unclipped draw or the disc; theta 0.7 bought less
    # accuracy for 30 % of the step. cell_min_level stays 8: at 6 an outskirt cell of
    # the unclipped draw neighbours 101,870 leaves (bench/results/fused_memory/round5)
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--cell-min-level", type=int, default=8)
    ap.add_argument(
        "--leaf-cap-factor",
        type=float,
        default=1.15,
        help="leaf capacity = this x the live cell leaves, in steps of 1024",
    )
    ap.add_argument("--caps", default="bench", choices=("bench", "named", "unnamed"))
    ap.add_argument(
        "--clean-env",
        action="store_true",
        help="apply none of the harness's JACCPOT_* / YGGDRAX_* environment (only "
        "--env), so the run sees what a caller who sets nothing sees; the solver "
        "configuration stays the bench's",
    )
    ap.add_argument(
        "--library-defaults",
        action="store_true",
        help="drop the harness's fused-lane switches that are library defaults since "
        "the 2026-10 cleanup (D2), so the run checks those defaults",
    )
    ap.add_argument("--far-cap", type=int, default=0)
    ap.add_argument("--near-cap", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3, help="scan steps per timed call")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--eval-repeats", type=int, default=7)
    ap.add_argument(
        "--accuracy-targets",
        type=int,
        default=0,
        help="score the eager force against an fp64 direct sum on this many random "
        "targets, LAST (after every FMM peak is recorded)",
    )
    ap.add_argument(
        "--no-donate",
        action="store_true",
        help="do not donate the carried state into strict_run_v2 (the A/B control; "
        "the bench always passes the returned state on, so donating is safe)",
    )
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument(
        "--donate-state",
        action="store_true",
        help="pass donate_state=True (carry='particles' only): the scan writes the "
        "returned state into the input state's buffer",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="go straight to strict_run_v2 (one prepare, then the scan: a production "
        "rollout's sequence; the eval section otherwise prepares a second time first)",
    )
    ap.add_argument("--no-analysis", action="store_true", help="skip memory_analysis()")
    ap.add_argument("--dump-dir", default=None, help="XLA dump (buffer assignment)")
    ap.add_argument(
        "--dump-re",
        default=".*(_eval|_compiled_runner).*",
        help="module regex for --dump-dir ('.*' dumps every jitted piece of the "
        "eager prepare too)",
    )
    ap.add_argument(
        "--trace-dir",
        default=None,
        help="profile one warm strict_run_v2 call here (analyse with "
        "bench/analyse_trace_by_stage.py --module-re '*_compiled_runner*')",
    )
    ap.add_argument(
        "--no-command-buffers",
        action="store_true",
        help="run without CUDA graphs, so a trace names every kernel",
    )
    ap.add_argument(
        "--save-forces",
        default=None,
        metavar="NPZ",
        help="save the first eval's force and the final scan state (bitwise A/B)",
    )
    ap.add_argument(
        "--use-pallas",
        default="auto",
        choices=("auto", "on", "off"),
        help="FastMultipoleMethod(use_pallas=...). off moves the NEAR FIELD to its "
        "pure-JAX rectangle / target-block route (the one pre-Ampere GPUs and "
        "ODISSEO_FMM_USE_PALLAS=0 take); the walk, M2L, cascade, COM-radii and P2M "
        "kernels keep their own sm_80 checks, so the fully Pallas-free route is the "
        "CPU one (tests/characterization/test_lane_goldens.py)",
    )
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--drift-steps", default="0,50,100")
    ap.add_argument("--softening", type=float, default=None)
    ap.add_argument(
        "--softening-kernel",
        default=None,
        choices=("ferrers3", "wendland_c2", "plummer"),
        help="pair softening kernel (jaccpot.softening); the reference uses the same",
    )
    ap.add_argument(
        "--mac-type",
        default="dehnen",
        choices=("dehnen", "dehnen_error"),
        help="multipole acceptance: the geometric Dehnen MAC, or Dehnen (2014) eq "
        "(16a) evaluated per pair inside the flat walk",
    )
    ap.add_argument(
        "--adaptive-eps",
        type=float,
        default=None,
        help="eq (16)'s relative force-accuracy target (mandatory with dehnen_error)",
    )
    ap.add_argument(
        "--mac-force-scale-mode",
        default="paper_fb",
        help="per-node force scale of eq (16): paper_fb = eq (16b)'s f_b prepass "
        "(the mesh lane's), paper = eq (16a)'s |a_b| prepass",
    )
    ap.add_argument(
        "--error-arrays",
        action="store_true",
        help="with --accuracy-targets: also score da/f (f = sum G m / r^2, fp64) and "
        "save every target's errors next to --out (<out>.err.npz)",
    )
    ap.add_argument("--env", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


ARGS = _args()

from codes.compare_force import (  # noqa: E402
    FAST_LANE_ENV_BY_LEAF,
    apply_fast_lane_env,
    fast_lane_overrides_for_leaf,
)

if ARGS.prealloc > 0:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(ARGS.prealloc)
    # jax 0.11.2's BFC allocator partitions the preallocated range "spatially" by
    # default, and once in ~120 preallocated runs it died freeing a buffer in the
    # eager prepare (`Check failed: central_gap_ == kInvalidChunkHandle ... spatial
    # partitioning expects one central gap`). Off, the 1e8 step and its peak are the
    # same (1147-1148 ms, 25.185 GiB both ways; round-5 record).
    if "xla_gpu_enable_allocator_spatial_partitioning" not in os.environ.get(
        "XLA_FLAGS", ""
    ):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + " --xla_gpu_enable_allocator_spatial_partitioning=false"
        ).strip()
else:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
# the record configuration's command buffers (the N-max ladder ran with these)
_CB = "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL --xla_gpu_graph_min_graph_size=2"
if ARGS.no_command_buffers:
    # explicitly EMPTY: an unset flag still builds command buffers
    _CB = "--xla_gpu_enable_command_buffer="
if "xla_gpu_enable_command_buffer" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _CB).strip()
if ARGS.dump_dir:
    os.makedirs(ARGS.dump_dir, exist_ok=True)
    os.environ["XLA_FLAGS"] += (
        f" --xla_dump_to={ARGS.dump_dir} --xla_dump_hlo_as_text"
        f" --xla_dump_hlo_module_re={ARGS.dump_re}"
    )

overrides = fast_lane_overrides_for_leaf(ARGS.leaf, ARGS.n)
extra = dict(kv.split("=", 1) for kv in ARGS.env)
overrides.update(extra)
if ARGS.clean_env:
    # Nothing from the harness and nothing inherited (the worktree pointers the
    # sitecustom hook reads at start-up excepted): the library's own defaults. The
    # 2026-10 CSR payload bug hid for months behind the harness's
    # RADIX_FAST_PAYLOAD_MAX_MB=0, which no production caller sets.
    for _k in [k for k in os.environ if k.startswith(("JACCPOT_", "YGGDRAX_"))]:
        if not _k.endswith("_WORKTREE"):
            os.environ.pop(_k)
    os.environ.update(extra)
else:
    apply_fast_lane_env(ARGS.n, overrides=overrides)
if ARGS.caps == "named":
    if ARGS.far_cap <= 0 or ARGS.near_cap <= 0:
        raise SystemExit("--caps named needs --far-cap and --near-cap")
    os.environ[_CAP_VARS[0]] = str(int(ARGS.far_cap))
    os.environ[_CAP_VARS[1]] = str(int(ARGS.near_cap))
elif ARGS.caps == "unnamed":
    for v in _CAP_VARS:
        os.environ.pop(v, None)
if ARGS.library_defaults:
    # What Odisseo's env block and the harness set that the library now defaults to
    # (D2). The caps, the payload budget and the index precision stay the harness's.
    for v in (
        "JACCPOT_STATIC_STRICT_GPU_MODE",
        "JACCPOT_STATIC_STRICT_FUSED_MODE",
        "JACCPOT_STATIC_STRICT_REQUIRE_EXACT_CAP_PROFILE_MATCH",
        "JACCPOT_STATIC_STRICT_FUSED_DEVICE_ONLY",
        "JACCPOT_STATIC_STRICT_FUSED_DISALLOW_HOST_SEGMENT_FALLBACK",
        "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
        "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
        "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
        "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
        "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
        "JACCPOT_LARGE_N_COMPILED_STATE_MODE",  # read nowhere; the harness sets it
    ):
        os.environ.pop(v, None)
_TRAV = dict(
    (FAST_LANE_ENV_BY_LEAF.get(ARGS.leaf) or {}).get("_traversal_overrides", {})
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

from common.gpu_guard import GpuMonitor  # noqa: E402
from common.ic import IC_GENERATORS  # noqa: E402
from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy  # noqa: E402
from yggdrax.morton import morton_encode  # noqa: E402

import jaccpot.pallas.mutual_walk_pallas as _mwp  # noqa: E402
import jaccpot.runtime._interaction_cache as _ic  # noqa: E402
from jaccpot import FastMultipoleMethod, TraversalOverrides  # noqa: E402
from jaccpot.config import (  # noqa: E402
    FarFieldConfig,
    FMMAdvancedConfig,
    NearFieldConfig,
    RuntimePolicyConfig,
    TreeConfig,
)
from jaccpot.runtime.fmm_prepare import PrepareMixin  # noqa: E402
from jaccpot.runtime.kernels._evaluate import _infer_bounds  # noqa: E402

GIB = float(1 << 30)
#: physical card(s) for the contention monitor
_PHYS = [
    int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()
][:1]


# --------------------------------------------------------------------------- memory


def _mem() -> dict:
    stats = jax.local_devices()[0].memory_stats() or {}
    return dict(
        in_use=int(stats.get("bytes_in_use", 0)),
        peak=int(stats.get("peak_bytes_in_use", 0)),
        limit=int(stats.get("bytes_limit", 0)),
        largest_free=int(stats.get("largest_free_block_bytes", 0)),
    )


EVENTS: list[dict] = []
_T0 = time.perf_counter()


def _traced(x) -> bool:
    return any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x)
    )


def _sync(x) -> None:
    for leaf in jax.tree_util.tree_leaves(x):
        fn = getattr(leaf, "block_until_ready", None)
        if fn is not None and not isinstance(leaf, jax.core.Tracer):
            try:
                fn()
            except Exception:  # noqa: BLE001 -- a deleted/donated buffer is fine here
                pass


def _event(label: str, **info) -> None:
    m = _mem()
    EVENTS.append(dict(label=label, t=time.perf_counter() - _T0, **m, **info))
    print(
        f"  [{EVENTS[-1]['t']:7.1f}s] {label:<34} in-use {m['in_use'] / GIB:6.2f} GiB"
        f"  peak {m['peak'] / GIB:6.2f} GiB  {info if info else ''}",
        flush=True,
    )


def _wrap(owner, name: str, label: str, info=None) -> None:
    orig = getattr(owner, name)

    @functools.wraps(orig)
    def wrapper(*a, **k):
        probe = (a, k)
        if _traced(probe):
            return orig(*a, **k)
        extra_info = info(a, k) if info is not None else {}
        _event(f"{label}:start", **extra_info)
        out = orig(*a, **k)
        _sync(out)
        _event(f"{label}:end")
        return out

    setattr(owner, name, wrapper)


def _walk_info(a, k) -> dict:
    return dict(
        queue=int(k.get("max_pair_queue", 0)),
        far_cap=int(k.get("far_cap", 0)),
        near_cap=int(k.get("near_cap", 0)),
    )


_wrap(_mwp, "mutual_walk_pallas", "walk", _walk_info)
_wrap(_ic, "_build_flat_walk_artifacts_strict_streamed", "walk+lists")
for _name, _label in (
    ("prepare_state", "prepare"),
    ("_prepare_state_tree_and_upward", "tree+upward"),
    ("_prepare_state_dual_and_downward_strict_streamed_fast", "dual+downward"),
    ("_prepare_downward_with_artifacts", "downward"),
    ("_prepare_state_nearfield_artifacts", "nearfield_artifacts"),
):
    _wrap(PrepareMixin, _name, _label)


def _live_arrays(top: int = 20) -> dict:
    arrs = sorted(jax.live_arrays(), key=lambda x: -int(x.nbytes))
    return dict(
        count=len(arrs),
        total_gib=sum(int(x.nbytes) for x in arrs) / GIB,
        top=[
            dict(shape=list(x.shape), dtype=str(x.dtype), gib=int(x.nbytes) / GIB)
            for x in arrs[:top]
        ],
    )


def _named_leaves(obj, path: str = "") -> list:
    """``(path, array)`` for every array in ``obj``, NamedTuple fields by name."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype") and hasattr(obj, "nbytes"):
        return [(path, obj)]
    out = []
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        for f in obj._fields:
            out += _named_leaves(getattr(obj, f), f"{path}.{f}")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            out += _named_leaves(getattr(obj, f.name, None), f"{path}.{f.name}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _named_leaves(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _named_leaves(v, f"{path}[{i}]")
    return out


def _pytree_breakdown(tree, top: int = 25) -> dict:
    rows = []
    by_field: dict[str, int] = {}
    seen: set[int] = set()
    for key, leaf in _named_leaves(tree):
        nbytes = int(leaf.nbytes)
        if not nbytes or id(leaf) in seen:
            continue
        seen.add(id(leaf))
        rows.append((nbytes, key, list(leaf.shape), str(leaf.dtype)))
        head = key.split(".")[1] if key.count(".") else key
        by_field[head] = by_field.get(head, 0) + nbytes
    rows.sort(key=lambda r: -r[0])
    return dict(
        total_gib=sum(r[0] for r in rows) / GIB,
        top=[dict(path=k, shape=s, dtype=d, gib=b / GIB) for b, k, s, d in rows[:top]],
        by_field_gib={
            k: v / GIB for k, v in sorted(by_field.items(), key=lambda kv: -kv[1])[:top]
        },
    )


def _analysis(compiled) -> dict:
    try:
        ma = compiled.memory_analysis()
    except Exception as exc:  # noqa: BLE001
        return dict(error=str(exc)[:300])
    out = {}
    for f in (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    ):
        v = getattr(ma, f, None)
        if v is not None:
            out[f.replace("_size_in_bytes", "_gib")] = int(v) / GIB
    return out


def _buffer_assignment(dump_dir: str, needle: str, top: int = 25) -> dict:
    """Largest allocations of the dumped module whose name contains ``needle``."""
    files = sorted(
        glob.glob(os.path.join(dump_dir, f"*{needle}*buffer-assignment.txt"))
    )
    if not files:
        return dict(error="no buffer-assignment dump")
    path = files[-1]
    allocs = []
    cur = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("allocation "):
                head, _, rest = line.partition(":")
                size = (
                    int(rest.split("size ")[1].split(",")[0]) if "size " in rest else 0
                )
                cur = dict(id=head, bytes=size, kind=rest.strip()[:160], values=[])
                allocs.append(cur)
            elif cur is not None and line.strip().startswith("value:"):
                txt = line.strip()[len("value:") :].strip()
                try:
                    vsize = int(txt.split("(size=")[1].split(",")[0])
                except (IndexError, ValueError):
                    vsize = 0
                cur["values"].append((vsize, txt[:200]))
            elif line.startswith("Total bytes used") or line.startswith("Used values"):
                cur = None
    allocs.sort(key=lambda a: -a["bytes"])
    return dict(
        file=os.path.basename(path),
        total_gib=sum(a["bytes"] for a in allocs) / GIB,
        top=[
            dict(
                id=a["id"],
                gib=a["bytes"] / GIB,
                kind=a["kind"],
                largest_values=[
                    v for _, v in sorted(a["values"], key=lambda v: -v[0])[:3]
                ],
            )
            for a in allocs[:top]
        ],
    )


# --------------------------------------------------------------------------- ICs


def _plummer_velocities(pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Isotropic equilibrium speeds for a G = M = a = 1 Plummer sphere (AHW 1974)."""
    r = np.linalg.norm(pos, axis=1)
    n = r.shape[0]
    q = np.empty(n)
    todo = np.arange(n)
    while todo.size:
        x = rng.uniform(0.0, 1.0, todo.size)
        y = rng.uniform(0.0, 0.1, todo.size)
        ok = y < x * x * (1.0 - x * x) ** 3.5
        q[todo[ok]] = x[ok]
        todo = todo[~ok]
    speed = q * np.sqrt(2.0) * (1.0 + r * r) ** -0.25
    mu = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    st = np.sqrt(1.0 - mu * mu)
    return (
        speed[:, None] * np.stack([st * np.cos(phi), st * np.sin(phi), mu], 1)
    ).astype(np.float32)


def _initial_conditions():
    """``(pos, vel, mass, softening, dt, external_fn)`` for ``ARGS.ic``."""
    if ARGS.ic == "plummer_clipped":
        # the Plummer inverse CDF on [0, X(rmax)], X(r) = r^3 / (r^2 + 1)^1.5: the same
        # sphere without the outliers that set the per-axis box at large N
        rng = np.random.default_rng(ARGS.seed)
        x_max = ARGS.rmax**3 / (ARGS.rmax**2 + 1.0) ** 1.5
        x = rng.uniform(0.0, x_max, size=ARGS.n)
        r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
        mu = rng.uniform(-1.0, 1.0, size=ARGS.n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=ARGS.n)
        st = np.sqrt(1.0 - mu * mu)
        pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
        pos = pos.astype(np.float32)
        mass = np.full(ARGS.n, 1.0 / ARGS.n, np.float32)
        vel = _plummer_velocities(np.asarray(pos, np.float64), np.random.default_rng(7))
        soft = 1e-7 if ARGS.softening is None else ARGS.softening
        dt = 1e-2 if ARGS.dt is None else ARGS.dt
        return pos, vel, mass, soft, dt, None
    if ARGS.ic == "plummer":
        pos, mass = IC_GENERATORS["plummer"](ARGS.n, seed=ARGS.seed)
        vel = _plummer_velocities(np.asarray(pos, np.float64), np.random.default_rng(7))
        soft = 1e-7 if ARGS.softening is None else ARGS.softening
        dt = 1e-2 if ARGS.dt is None else ARGS.dt
        return (
            np.asarray(pos, np.float32),
            vel,
            np.asarray(mass, np.float32),
            soft,
            dt,
            None,
        )
    ic = np.load(_DISC_IC)
    n_all = int(ic["state0"].shape[0])
    if ARGS.n > n_all:
        raise SystemExit(f"the disc IC has {n_all} particles")
    # the IC is shuffled, so a prefix is a random subsample; rescale the masses so the
    # subsample is the same galaxy
    state = np.asarray(ic["state0"][: ARGS.n], np.float32)
    mass = np.asarray(ic["mass"][: ARGS.n], np.float64) * (n_all / ARGS.n)
    g = float(ic["G_code"]) if "G_code" in ic.files else 1.0
    halo_m = float(ic["halo_mass_code"])
    halo_rs = float(ic["halo_rs_code"])
    rdisk = float(ic["rdisk_code"])
    soft = (
        float(0.5 * rdisk / np.sqrt(ARGS.n / 1e5))
        if ARGS.softening is None
        else ARGS.softening
    )
    dt = 5e-4 if ARGS.dt is None else ARGS.dt

    def nfw(state_in):
        # AGAMA's NFW convention, as tools/mesh_galaxy_run.py in Odisseo
        pos = state_in[:, 0, :]
        r2 = jnp.sum(pos * pos, axis=1)
        r = jnp.sqrt(jnp.maximum(r2, 1e-30))
        x = r / halo_rs
        big = jnp.log1p(x) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
            r2 * (halo_rs + r), 1e-30
        )
        ser = (1.0 / (2.0 * halo_rs**3)) * (1.0 - (4.0 / 3.0) * x)
        coeff = jnp.where(x < 1e-3, ser, big)
        return -(g * halo_m) * coeff[:, None] * pos

    return state[:, 0], state[:, 1], mass.astype(np.float32), soft, dt, nfw


# --------------------------------------------------------------------------- solver


def _leaf_capacity(pos: np.ndarray) -> tuple[int, int]:
    p = jnp.asarray(pos)
    codes = np.sort(np.asarray(morton_encode(p, _infer_bounds(p))).astype(np.uint64))
    live = int(
        adaptive_cell_leaf_partition_numpy(
            codes, leaf_size=ARGS.leaf, min_level=ARGS.cell_min_level or None
        )[0].size
    )
    cap = int(-(-int(np.ceil(ARGS.leaf_cap_factor * live)) // 1024) * 1024)
    return live, cap


def _solver(leaf_cap: int, soft: float) -> FastMultipoleMethod:
    return FastMultipoleMethod(
        preset="large_n_gpu",
        runtime_path="large_n",
        basis="real",
        theta=ARGS.theta,
        G=1.0,
        softening=soft,
        softening_kernel=ARGS.softening_kernel,
        working_dtype=jnp.float32,
        use_pallas={"auto": None, "on": True, "off": False}[ARGS.use_pallas],
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=ARGS.leaf,
                leaf_partition="cells",
                leaf_capacity=leaf_cap,
                cell_min_level=ARGS.cell_min_level or None,
            ),
            farfield=FarFieldConfig(mode="auto"),
            nearfield=NearFieldConfig(mode="auto"),
            runtime=(
                RuntimePolicyConfig(
                    traversal_config=TraversalOverrides(
                        **{k: int(v) for k, v in _TRAV.items()}
                    )
                )
                if _TRAV
                else RuntimePolicyConfig()
            ),
            mac_type=str(ARGS.mac_type),
        ),
        fixed_order=ARGS.order,
        **(
            dict(
                adaptive_eps=float(ARGS.adaptive_eps),
                adaptive_error_model="dehnen_paper",
                mac_force_scale_mode=str(ARGS.mac_force_scale_mode),
            )
            if ARGS.mac_type == "dehnen_error"
            else {}
        ),
    )


def _counts(solver) -> dict:
    v = dict(getattr(solver._impl, "_strict_fused_validated_caps", None) or {})
    keep = (
        "far_pair_count",
        "total_neighbors",
        "peak_wavefront",
        "max_neighbors_observed",
        "rounds",
        "compact_far_pair_capacity",
        "near_edge_capacity",
        "queue_capacity",
        "far_named",
        "near_edge_named",
        "grew",
    )
    return {k: v.get(k) for k in keep}


def _write(result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
    with open(ARGS.out, "w") as fh:
        json.dump(result, fh, indent=1, default=str)


# --------------------------------------------------------------------------- modes


def run_budget(result: dict) -> None:
    pos, vel, mass, soft, dt, ext = _initial_conditions()
    live, leaf_cap = _leaf_capacity(pos)
    result.update(live_leaves=live, leaf_capacity=leaf_cap, softening=soft, dt=dt)
    print(f"N={ARGS.n} live leaves {live} -> leaf capacity {leaf_cap}", flush=True)
    P = jnp.asarray(pos)
    M = jnp.asarray(mass)
    solver = _solver(leaf_cap, soft)
    _event("before_prepare")
    a_host = None
    saved = None
    if ARGS.skip_eval:
        result["prepare_events"] = []  # the scan's own prepare fills it afterwards
    if not ARGS.skip_eval:
        t0 = time.perf_counter()
        prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
            positions=P,
            masses=M,
            leaf_size=ARGS.leaf,
            max_order=ARGS.order,
            theta=ARGS.theta,
        )
        _sync(prepared)
        result["prepare_s"] = time.perf_counter() - t0
        _event("after_prepare")
        result["counts"] = _counts(solver)
        result["prepare_events"] = list(EVENTS)
        result["live_after_prepare"] = _live_arrays()
        result["prepared_breakdown"] = _pytree_breakdown(prepared)
        _write(result)

        a = jax.block_until_ready(eval_fn(prepared))
        _event("after_eval_compile")
        a_host = np.asarray(a, np.float64) if ARGS.accuracy_targets else None
        saved = {"force": np.asarray(a)} if ARGS.save_forces else None
        del a
        samples = []
        for _ in range(2):
            jax.block_until_ready(eval_fn(prepared))
        with GpuMonitor(_PHYS) as mon:
            for _ in range(ARGS.eval_repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(eval_fn(prepared))
                samples.append(time.perf_counter() - t0)
        result["eval_contention"] = mon.summary().as_dict()
        _event("after_eval_timing")
        result["eval_ms"] = dict(
            min=min(samples) * 1e3,
            median=float(np.median(samples)) * 1e3,
            samples=[s * 1e3 for s in samples],
        )
        print(f"eval-only min {result['eval_ms']['min']:.2f} ms", flush=True)
        if not ARGS.no_analysis:
            result["eval_memory_analysis"] = _analysis(
                eval_fn.lower(prepared).compile()
            )
            print(f"eval memory_analysis {result['eval_memory_analysis']}", flush=True)
        result["peak_after_eval_gib"] = _mem()["peak"] / GIB
        _write(result)
        del prepared, eval_fn

    if ARGS.no_scan:
        if saved is not None:
            np.savez(ARGS.save_forces, **saved)
        _accuracy(result, pos, mass, soft, a_host)
        return
    state0 = jnp.stack([P, jnp.asarray(vel)], axis=1)

    def run(state, prep, k):
        out = solver.strict_run_v2(
            state=state,
            masses=M,
            dt=dt,
            num_steps=k,
            refresh_every=1,
            leaf_size=ARGS.leaf,
            max_order=ARGS.order,
            theta=ARGS.theta,
            prepared_state=prep,
            return_prepared_state=True,
            add_external=ext is not None,
            external_acceleration_fn=ext,
            **({} if ARGS.no_donate else {"donate_prepared_state": True}),
            **({"donate_state": True} if ARGS.donate_state else {}),
        )
        jax.block_until_ready(out[0])
        return out

    t0 = time.perf_counter()
    state, prep, _ = run(state0, None, 1)
    state, prep, _ = run(state, prep, ARGS.steps)
    result["scan_compile_s"] = time.perf_counter() - t0
    _event("after_scan_compile")
    samples = []
    with GpuMonitor(_PHYS) as mon:
        for _ in range(ARGS.reps):
            t0 = time.perf_counter()
            state, prep, _ = run(state, prep, ARGS.steps)
            samples.append((time.perf_counter() - t0) / ARGS.steps)
    result["scan_contention"] = mon.summary().as_dict()
    _event("after_scan_timing")
    result["step_ms"] = dict(
        min=min(samples) * 1e3,
        median=float(np.median(samples)) * 1e3,
        samples=[s * 1e3 for s in samples],
    )
    result["peak_after_scan_gib"] = _mem()["peak"] / GIB
    result["scan_events"] = EVENTS[len(result["prepare_events"]) :]
    if ARGS.trace_dir:
        os.makedirs(ARGS.trace_dir, exist_ok=True)
        with jax.profiler.trace(ARGS.trace_dir):
            state, prep, _ = run(state, prep, ARGS.steps)
        result["trace"] = dict(dir=ARGS.trace_dir, steps=ARGS.steps)
    print(f"scan min {result['step_ms']['min']:.2f} ms/step", flush=True)
    if ARGS.skip_eval:
        # the scan's own prepare is the only one: its events are the prepare's
        result["counts"] = _counts(solver)
        result["prepare_events"] = [
            e for e in EVENTS if "prepare" in e["label"] or ":" in e["label"]
        ]
    _write(result)
    if saved is not None:
        np.savez(ARGS.save_forces, state=np.asarray(state), **saved)
    if not ARGS.no_analysis:
        cache = getattr(solver._impl, "_strict_fused_jit_function_cache", {}) or {}
        runner = next(
            (fn for key, fn in cache.items() if int(key[6]) == int(ARGS.steps)), None
        )
        if runner is not None:
            # lower with the carry the scan sees: the far list rides outside it
            # where the lane rebuilds it fresh (jaccpot perf/fused-carry)
            ride = getattr(
                solver._impl, "_strict_far_pairs_ride_outside_the_scan", None
            )
            if ride is not None and ride(prep):
                prep = dataclasses.replace(prep, compact_far_pairs=None)
            spec = jax.tree_util.tree_map(
                lambda x: (
                    jax.ShapeDtypeStruct(x.shape, x.dtype)
                    if hasattr(x, "shape") and hasattr(x, "dtype")
                    else x
                ),
                (prep, state, jnp.zeros_like(state[:, 0]), M),
            )
            try:
                result["runner_memory_analysis"] = _analysis(
                    runner.lower(*spec).compile()
                )
            except Exception as exc:  # noqa: BLE001 -- diagnostics must not end the row
                result["runner_memory_analysis"] = dict(error=str(exc)[:300])
            print(
                f"runner memory_analysis {result['runner_memory_analysis']}", flush=True
            )
    if ARGS.dump_dir:
        result["eval_buffer_assignment"] = _buffer_assignment(ARGS.dump_dir, "_eval")
        result["runner_buffer_assignment"] = _buffer_assignment(
            ARGS.dump_dir, "_compiled_runner"
        )
    _write(result)
    del state, prep
    _accuracy(result, pos, mass, soft, a_host)


def _direct_kernel_fp64(pos, mass, idx, soft: float, kernel: str):
    """fp64 direct sum with a compact softening kernel at the targets ``idx``.

    The harness's ``common.reference`` is Plummer only; this is the same chunked
    sum (self excluded by index) through :func:`jaccpot.softening.pair_factors`.
    """
    from jaccpot.softening import pair_factors, softening_params

    p = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)
    n = int(p.shape[0])
    chunk = 1 << 18
    npad = -(-n // chunk) * chunk
    sp = jnp.zeros((npad, 3), jnp.float64).at[:n].set(p).reshape(-1, chunk, 3)
    sm = jnp.zeros((npad,), jnp.float64).at[:n].set(m).reshape(-1, chunk)
    sid = jnp.arange(npad).reshape(-1, chunk)
    params = softening_params(kernel, soft, jnp.float64)

    @jax.jit
    def block(tp, tid):
        def body(c, acc):
            d = tp[:, None, :] - sp[c][None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            g = pair_factors(jnp.where(r2 > 0, r2, 1.0), params, kernel)[0]
            w = jnp.where(tid[:, None] == sid[c][None, :], 0.0, g * sm[c][None, :])
            return acc - jnp.einsum("ij,ijk->ik", w, d)

        return jax.lax.fori_loop(0, sp.shape[0], body, jnp.zeros_like(tp))

    out = []
    for b in range(0, len(idx), 256):
        ib = jnp.asarray(idx[b : b + 256])
        out.append(np.asarray(block(p[ib], ib)))
    return np.concatenate(out)


def _force_scale_reference(pos, mass, soft: float, idx, acc_cache: str) -> np.ndarray:
    """``f_b = sum_{a != b} G m_a / (|x_a - x_b|^2 + eps^2)`` at the targets, fp64.

    Dehnen (2014) eq (4)'s force scale (softened like the force), the
    denominator of the scaled error ``da/f`` his eq (16b) controls. Cached next to
    the acceleration reference.
    """
    cache = acc_cache.replace("direct_fp64_", "fscale_fp64_")
    if os.path.exists(cache):
        return np.load(cache)
    import jax

    @jax.jit
    def _block(tp, tg, sp, sm, sg):
        d = tp[:, None, :] - sp[None, :, :]
        r2 = jnp.sum(d * d, axis=-1) + soft**2
        w = jnp.where(tg[:, None] == sg[None, :], 0.0, sm[None, :] / r2)
        return jnp.sum(w, axis=1)

    p64 = jnp.asarray(np.asarray(pos, np.float64))
    m64 = jnp.asarray(np.asarray(mass, np.float64))
    gid = jnp.arange(p64.shape[0], dtype=jnp.int64)
    ti = jnp.asarray(np.asarray(idx, np.int64))
    block = max(1, (1 << 30) // max(int(p64.shape[0]) * 8, 1))
    out = np.empty(len(idx), np.float64)
    for a in range(0, len(idx), block):
        b = min(a + block, len(idx))
        out[a:b] = np.asarray(_block(p64[ti[a:b]], ti[a:b], p64, m64, gid))
    np.save(cache, out)
    return out


def _accuracy(result: dict, pos, mass, soft: float, a_host) -> None:
    """fp64 direct-sum score of the eager force (``--accuracy-targets``)."""
    if a_host is None:
        return
    from common.reference import direct_accelerations

    from jaccpot.softening import resolve_softening_kernel

    kernel = resolve_softening_kernel(ARGS.softening_kernel)

    result["peak_before_accuracy_gib"] = _mem()["peak"] / GIB
    idx = np.sort(
        np.random.default_rng(12345).choice(
            ARGS.n, ARGS.accuracy_targets, replace=False
        )
    )
    # the harness's reference cache (codes/jzfmm_force_eval.py writes and reads the
    # same file): one fp64 direct sum per (IC, N, softening, targets) for every code
    cache = os.path.join(
        os.environ.get(
            "BENCH_DIR", "/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu"
        ),
        "artifacts",
        "reference",
        f"direct_fp64_{ARGS.ic}{ARGS.n}_soft{soft:g}"
        f"{'' if kernel == 'plummer' else '_' + kernel}_ref{len(idx)}_seed12345.npy",
    )
    if os.path.exists(cache):
        ref = np.load(cache)
        result["accuracy_reference"] = cache
    elif kernel != "plummer":
        ref = _direct_kernel_fp64(pos, mass, idx, soft, kernel)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, ref)
        result["accuracy_reference"] = cache
    else:
        ref = direct_accelerations(
            np.asarray(pos, np.float64),
            np.asarray(mass, np.float64),
            G=1.0,
            softening=soft,
            target_indices=idx,
        )
        if ARGS.error_arrays:
            np.save(cache, ref)  # the distribution runs reuse 16k-target references
    got = a_host[idx]
    err = np.linalg.norm(got - ref, axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1), 1e-300
    )
    result["accuracy"] = dict(
        targets=int(len(idx)),
        rel_l2=float(np.linalg.norm(got - ref) / np.linalg.norm(ref)),
        median=float(np.median(err)),
        p90=float(np.percentile(err, 90)),
        p99=float(np.percentile(err, 99)),
        p999=float(np.percentile(err, 99.9)),
        max=float(err.max()),
    )
    if ARGS.error_arrays:
        f = _force_scale_reference(pos, mass, soft, idx, cache)
        err_f = np.linalg.norm(got - ref, axis=1) / np.maximum(f, 1e-300)
        result["accuracy"]["scaled"] = dict(
            median=float(np.median(err_f)),
            p90=float(np.percentile(err_f, 90)),
            p99=float(np.percentile(err_f, 99)),
            p999=float(np.percentile(err_f, 99.9)),
            max=float(err_f.max()),
        )
        np.savez(
            os.path.splitext(ARGS.out)[0] + ".err.npz",
            idx=idx,
            da_over_a=err,
            da_over_f=err_f,
            a_ref_norm=np.linalg.norm(ref, axis=1),
            f=f,
            r=np.linalg.norm(np.asarray(pos)[idx], axis=1),
        )
    print(
        f"accuracy vs fp64 direct ({len(idx)} targets): {result['accuracy']}",
        flush=True,
    )
    _write(result)


def run_drift(result: dict) -> None:
    pos, vel, mass, soft, dt, ext = _initial_conditions()
    live, leaf_cap = _leaf_capacity(pos)
    result.update(live_leaves=live, leaf_capacity=leaf_cap, softening=soft, dt=dt)
    print(
        f"N={ARGS.n} {ARGS.ic} live leaves {live} -> leaf capacity {leaf_cap}",
        flush=True,
    )
    M = jnp.asarray(mass)
    solver = _solver(leaf_cap, soft)
    marks = sorted(int(s) for s in ARGS.drift_steps.split(","))
    state = jnp.stack([jnp.asarray(pos), jnp.asarray(vel)], axis=1)
    rows = []
    done = 0
    for mark in marks:
        if mark > done:
            state, _, _ = solver.strict_run_v2(
                state=state,
                masses=M,
                dt=dt,
                num_steps=mark - done,
                refresh_every=1,
                leaf_size=ARGS.leaf,
                max_order=ARGS.order,
                theta=ARGS.theta,
                return_prepared_state=False,
                add_external=ext is not None,
                external_acceleration_fn=ext,
            )
            jax.block_until_ready(state)
            done = mark
        p_now = np.asarray(state[:, 0])
        live_now, _ = _leaf_capacity(p_now)
        prepared, _ = solver.strict_fused_prepared_eval_fn(
            positions=jnp.asarray(p_now),
            masses=M,
            leaf_size=ARGS.leaf,
            max_order=ARGS.order,
            theta=ARGS.theta,
        )
        del prepared
        lo, hi = p_now.min(0), p_now.max(0)
        ext_ = hi - lo
        row = dict(
            step=mark,
            live_leaves=live_now,
            aspect=float(ext_.max() / ext_.min()),
            **_counts(solver),
        )
        rows.append(row)
        print(
            f"step {mark:5d}: far {row['far_pair_count']} near {row['total_neighbors']} "
            f"wavefront {row['peak_wavefront']} live leaves {live_now} aspect {row['aspect']:.2f}",
            flush=True,
        )
        result["drift"] = rows
        _write(result)
    base = rows[0]
    result["drift_ratio"] = [
        {
            k: (r[k] / base[k] if base.get(k) else None)
            for k in (
                "far_pair_count",
                "total_neighbors",
                "peak_wavefront",
                "live_leaves",
            )
        }
        | {"step": r["step"]}
        for r in rows
    ]
    _write(result)


def main() -> int:
    result = dict(
        mode=ARGS.mode,
        n=ARGS.n,
        ic=ARGS.ic,
        seed=ARGS.seed,
        leaf=ARGS.leaf,
        theta=ARGS.theta,
        order=ARGS.order,
        cell_min_level=ARGS.cell_min_level,
        caps=ARGS.caps,
        donate=not ARGS.no_donate,
        cap_env={v: os.environ.get(v) for v in _CAP_VARS},
        xla_flags=os.environ.get("XLA_FLAGS"),
        mem_fraction=os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        rmax=ARGS.rmax if ARGS.ic == "plummer_clipped" else None,
        worktree=os.environ.get("JACCPOT_WORKTREE"),
        yggdrax_worktree=os.environ.get("YGGDRAX_WORKTREE"),
        device=str(jax.devices()[0]),
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    _event("start")
    if ARGS.mode == "budget":
        run_budget(result)
    else:
        run_drift(result)
    result["events"] = EVENTS
    result["final_peak_gib"] = _mem()["peak"] / GIB
    _write(result)
    print(f"wrote {ARGS.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

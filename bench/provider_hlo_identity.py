"""Program identity and bitwise goldens of the fused ``strict_run_v2`` lane (S2).

S2 moves the fused scan's in-step refresh and evaluation onto a
``FusedForceProvider`` (``~/.claude/plans/s2-provider-design.md``, stages C1-C3)
and must not change one number on any variant of the lane. This tool records, on
the frozen base, what each variant compiles and what it computes, so that every
refactor commit can be checked against it:

``record``
    runs every variant below in its own process on a FRESH solver, with
    ``jax.jit`` wrapped before jaccpot and yggdrax are imported. Every program
    jitted with ``jax.jit`` -- at import (the eager prepare's Morton codes, COM
    radii, walk, M2L, ...) or per call (the scan runner, the particle carry's
    start program, the state carry's eager initial force, the eval seam's closure)
    -- that runs at the top level is lowered with the call's own arguments just
    before it runs, and its ``lower(...).as_text(debug_info=False)`` is saved (on a
    GPU also the compiled HLO, without metadata), with the sha256 of each input and
    output leaf and of what the call returns (the final state, the history, the
    returned prepared state's leaves or the handle's force and force scale, the
    step callback's emissions). ``--save-arrays`` (default on CPU) keeps the
    arrays themselves in ``<variant>/arrays.npz``.
``compare A B [--control A-b]``
    fails on ANY difference: a program more, fewer or renamed, one character of a
    program's text, one bit of one array. With ``--control``, what differs between
    A and its A-vs-A control too is listed as ``noise`` and forgiven. On CPU there
    is none (every variant identical on f967805). On a GPU the eager walk
    ladder's raw outputs are not reproducible -- the pair order of the attempt that
    fits, the needs of the attempts that overflow (Pallas atomics) -- while
    ``_flat_walk_lists``' outputs and everything after them are bitwise.

**Why the capture wraps ``jax.jit`` rather than reading the engine's jit cache.**
The capture must still find the runners after C2/C3 have moved them. It finds a
program by how it RUNS -- a callable that ``jax.jit`` returned while the variant
was running, called outside any trace -- not by its name or where it is stored.
The lowering happens before the call, from the same arguments, so it is the
program that runs: the lowering and the call share jax's trace cache
(``_infer_params_cached``), the function is traced once, and its trace-time side
effects (the engine's traced caps, the level stash) happen once, in the order the
plain call has them. ``bench/tests/test_provider_hlo_identity.py`` pins that. A
strict call must end with a program that contains the scan (a
``stablehlo.while``), or ``record`` fails: a capture that found nothing must not
look like a capture that found no difference. While the engine still has
``_strict_fused_jit_function_cache``, every entry must be a captured program.

**What it cannot see.** The eager prepare's op-by-op work, and anything jitted
through jax's internals or a ``from jax import jit`` binding made before the wrapper
(none in jaccpot or yggdrax on f967805), are checked only through the outputs that
reach a captured program's inputs. A jitted function called inside another trace is
part of that program's text. A refactor that renames a runner function changes its
module name and fails ``compare``; the report says when the module name is the only
difference.

Variants (N = 512 on CPU, 2e5 on a GPU; a fresh process and solver each, 2 steps a
call). Each runs the near field on the CSR lane (production's; Pallas interpret mode
on CPU); ``<name>_purejax`` is the same with ``use_pallas=False`` (the pure-JAX
rectangle route the lane goldens and pre-Ampere cards take), for every variant but
the ``dehnen_error`` ones, whose force scale only the CSR lane accumulates. The
GPU default set is the CSR variants but the two ``retry`` ones (CPU only).

============================  ========================================================
state                         ``carry="state"`` (the library default)
particles                     ``carry="particles"``: start program + runner
tree                          particles with ``JACCPOT_STRICT_CARRY_ORDER=tree``
dehnen_error_state            ``mac_type="dehnen_error"``, state carry (f_b carried)
dehnen_error_particles        the same on the particle carry
dehnen_error_chain_state      two calls, the second from the returned prepared state
                              (``donate_prepared_state=True``)
dehnen_error_chain_particles  two calls, the second from the handle (``donate_state``)
external_state                an analytic NFW field as ``external_acceleration_fn``
external_particles            the same, particle carry
external_tree                 the same, tree order (the field sees tree-order rows)
history_state                 ``return_history=True``
history_particles             the same, particle carry
callback_state                a ``step_callback`` (stride 1), ``jax.debug.callback``
callback_particles            the same, particle carry
g_unit                        ``G = 4.3e-6``, the masses divided by G (state carry)
chain_state                   two calls: the second continues from the returned
                              prepared state, ``donate_prepared_state=True``
chain_particles               two calls: the second from the handle,
                              ``donate_state=True``
retry_state                   the segment retry: caps sized on the blobs, the state
                              stepped on a clipped Plummer sphere, so the walk caps
                              overflow, are re-planned and the segment re-runs (it
                              must, or the variant fails)
retry_particles               the same, particle carry (the start program overflows)
eval_seam                     ``strict_fused_prepared_eval_fn`` and one ``eval_fn`` call
galaxy_explicit               (S3, not in the default sets) the particle carry on
                              ``fused_memory_budget.py``'s explicit configuration at
                              its defaults (leaf 64, p6, theta 0.8, cell level 8, its
                              leaf-64 traversal overrides), capacity counted inside
                              the capture as the bench counts it
galaxy_preset                 the same run on ``preset="galaxy"`` (``leaf_capacity=
                              "auto"``, theta from the preset): ``compare --pair
                              galaxy_explicit:galaxy_preset`` must find nothing
============================  ========================================================

What the base pins down includes two behaviours of ``mac_type="dehnen_error"`` that
are not the same run on the two carries (CPU, N = 512, f967805): the state carry's
first step takes its thresholds from the eager prepare's f_b prepass, the particle
carry's from its start program's evaluation, so one 3-step call differs between the
carries (1,327 elements of the final state); and a state-carry call that continues
from a returned prepared state reseeds f_b from the engine's LAST eager prepass --
the previous call's start positions, in that tree's order, scattered through the
returned tree's permutation -- so 1 + 2 chained steps differ from 3 in one call
(1,349 elements, max 5.7e-5), where the particle carry's handle continues bitwise.
The geometric MAC is bitwise on both counts. A refactor that changes either moves
the ``dehnen_error`` variants; that is a decision, not a side effect. S2 made it for
the second (``c3fadd4``: a state-carry call continues from the scan's own f_b, and
``dehnen_error_chain_state``'s second call moves from the base).

CPU runs as the lane goldens do (``tests/characterization/test_lane_goldens.py``):
``jax.default_backend`` reports ``"gpu"``; the CSR variants add
``tests/unit/runtime/test_force_scale_evaluation.py``'s switches (interpret mode, the
prepacked payload at small N). Every process scrubs the ``JACCPOT_*`` /
``YGGDRAX_*`` variables (``*_WORKTREE`` kept) before the first import, so a record
sees the library's own defaults whatever the shell holds; the variants set their
own. CPU adds ``--xla_backend_extra_options=-vectorize-loops=false`` (the test
conftest's flag), GPU ``--xla_gpu_deterministic_ops=true`` and the bench's
command-buffer flags.

Usage::

    cd <worktree>  # S1_JACCPOT=<worktree> for the S2 wrapper
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python bench/provider_hlo_identity.py \\
        record --tag s2-base-cpu --jobs 4
    python bench/provider_hlo_identity.py compare s2-base-cpu c2-<sha>-cpu
    python bench/provider_hlo_identity.py record --tag s3-galaxy-cpu \
        --variants galaxy_explicit,galaxy_preset --jobs 2
    python bench/provider_hlo_identity.py compare s3-galaxy-cpu s3-galaxy-cpu \
        --pair galaxy_explicit:galaxy_preset

    export CUDA_VISIBLE_DEVICES=$(autocvd -n 1 -o -q -i 20)   # the org rule
    python bench/provider_hlo_identity.py record --device gpu --tag s2-base-gpu
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import difflib
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = "/export/scratch/tbuck/s2-goldens"

#: the deterministic, bench-like XLA flags per device
_CPU_FLAGS = ("--xla_backend_extra_options=-vectorize-loops=false",)
_GPU_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    # fused_memory_budget.py's record configuration
    "--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL",
    "--xla_gpu_graph_min_graph_size=2",
)
#: the CSR near-field lane on CPU (tests/unit/runtime/test_force_scale_evaluation.py)
_CSR_ON_CPU = (
    ("JACCPOT_NEARFIELD_PALLAS_INTERPRET", "1"),
    ("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "1"),
    # the prepacked payload the CSR lane reads, as at large N
    ("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", "0"),
)


# --------------------------------------------------------------------- capture


@dataclasses.dataclass
class Program:
    """One top-level run of a captured jitted callable."""

    seq: int
    module: str
    lowered: str
    compiled: Optional[str]
    inputs: dict
    outputs: dict

    @property
    def label(self) -> str:
        return f"{self.seq:02d}_{self.module}"


class Capture:
    """The programs one variant ran, in order, and the arrays they saw."""

    def __init__(self, *, compile_text: bool = False, keep_arrays: bool = True):
        self.compile_text = compile_text
        self.keep_arrays = keep_arrays
        self.programs: list[Program] = []
        self.arrays: dict[str, Any] = {}
        #: index into ``programs`` -> what that program is to the variant
        self.roles: dict[int, str] = {}

    def digest(self, prefix: str, tree: Any) -> dict:
        """sha256, shape and dtype of every array leaf of ``tree``, by key path."""
        return _digest(prefix, tree, self.arrays if self.keep_arrays else None)


_ACTIVE: list[Capture] = []


def _module_name(text: str) -> str:
    m = re.search(r"module @([A-Za-z0-9_.$-]+)", text)
    return m.group(1) if m else "unknown"


class _Recorded:
    """What the wrapped ``jax.jit`` returns: the jitted callable, recorded per run.

    Transparent otherwise (``lower``, ``trace``, ... are forwarded), so the library
    and its benches see a jitted function.
    """

    def __init__(self, jitted: Any):
        self._jitted = jitted
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._jitted, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from jax._src import core

        cap = _ACTIVE[-1] if _ACTIVE else None
        if cap is None or not core.trace_state_clean():
            # nested inside another trace: part of that program's text
            return self._jitted(*args, **kwargs)
        self.calls += 1
        lowered = self._jitted.lower(*args, **kwargs)
        text = lowered.as_text(debug_info=False)
        compiled = _compiled_text(lowered.compile()) if cap.compile_text else None
        seq = len(cap.programs)
        module = _module_name(text)
        prefix = f"prog:{seq:02d}_{module}"
        # inputs BEFORE the call: a donated argument is gone after it
        inputs = cap.digest(f"{prefix}:in", (args, kwargs))
        out = self._jitted(*args, **kwargs)
        outputs = cap.digest(f"{prefix}:out", out)
        cap.programs.append(Program(seq, module, text, compiled, inputs, outputs))
        return out


def _compiled_text(compiled: Any) -> str:
    """The optimised HLO without metadata.

    ``compiled.as_text()`` carries every op's source file and line, the calling
    frames included: a moved line anywhere on the stack changes it (the A-vs-A
    control did, by an edit to this file). Large constants are printed in full.
    """
    try:
        from jaxlib._hlo import HloPrintOptions  # private: jax 0.11.2's jaxlib
    except ImportError:  # pragma: no cover -- another jaxlib: the full text
        return compiled.as_text()
    opts = HloPrintOptions()
    opts.print_metadata = False
    opts.print_large_constants = True
    return "\n".join(
        m.to_string(opts) for m in compiled.runtime_executable().hlo_modules()
    )


@contextlib.contextmanager
def wrapped_jit():
    """``jax.jit`` returns :class:`_Recorded` callables while this is active."""
    import jax

    real = jax.jit

    def jit(fun: Any = None, /, **kwargs: Any) -> Any:
        if fun is None:
            return lambda f: jit(f, **kwargs)
        return _Recorded(real(fun, **kwargs))

    jax.jit = jit
    try:
        yield
    finally:
        jax.jit = real


@contextlib.contextmanager
def capturing(cap: Capture):
    """Record the top-level runs of every :class:`_Recorded` into ``cap``."""
    _ACTIVE.append(cap)
    try:
        yield cap
    finally:
        _ACTIVE.pop()


def _flat_with_paths(tree: Any) -> list[tuple[str, Any]]:
    """``(key path, leaf)`` for every leaf; unregistered dataclasses by field."""
    import jax

    out = []
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, leaf in flat:
        key = jax.tree_util.keystr(path)
        if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            for f in dataclasses.fields(leaf):
                for sub, val in _flat_with_paths(getattr(leaf, f.name)):
                    out.append((f"{key}.{f.name}{sub}", val))
        else:
            out.append((key, leaf))
    return out


def _digest(prefix: str, tree: Any, keep: Optional[dict]) -> dict:
    import jax
    import numpy as np

    rows = {}
    for key, leaf in _flat_with_paths(tree):
        name = f"{prefix}{key}"
        if isinstance(leaf, jax.ShapeDtypeStruct):
            rows[name] = {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
            continue
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            continue
        a = np.asarray(leaf)
        rows[name] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest(),
        }
        if keep is not None:
            keep[name] = a
    return rows


# --------------------------------------------------------------------- problem


@dataclasses.dataclass(frozen=True)
class Setup:
    """The lane's configuration on one device."""

    device: str
    n: int
    leaf: int
    order: int
    theta: float
    softening: float
    dt: float
    steps: int
    cell_min_level: int


def _setup(device: str, n: Optional[int]) -> Setup:
    if device == "cpu":
        # tests/characterization/test_lane_goldens.py's operating point, on the
        # production leaf partition (cells: the level bound is stashed)
        return Setup("cpu", n or 512, 16, 4, 0.6, 1e-2, 1e-3, 2, 0)
    # S1 / S6's (bench/fused_memory_budget.py's defaults)
    return Setup("gpu", n or 200_000, 64, 6, 0.8, 1e-7, 1e-2, 2, 8)


def _initial_state(setup: Setup):
    """``(state [N, 2, 3] f32, masses [N] f32)``, deterministic on any host."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = setup.n
    if setup.device == "cpu":
        # four tight blobs in [-1, 1]^3, as the lane goldens' clustered draw
        centres = rng.uniform(-0.8, 0.8, (4, 3))
        pos = centres[rng.integers(0, 4, n)] + 0.08 * rng.normal(size=(n, 3))
        pos = np.clip(pos, -1.0, 1.0)
        vel = 0.1 * rng.normal(size=(n, 3))
        mass = np.abs(rng.normal(size=n)) + 0.5
    else:
        # fused_memory_budget.py's clipped Plummer (rmax 20); the velocities need
        # not be in equilibrium for two steps
        x_max = 20.0**3 / (20.0**2 + 1.0) ** 1.5
        x = rng.uniform(0.0, x_max, n)
        r = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
        mu = rng.uniform(-1.0, 1.0, n)
        phi = rng.uniform(0.0, 2.0 * np.pi, n)
        st = np.sqrt(1.0 - mu * mu)
        pos = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
        vel = rng.normal(0.0, 0.3, (n, 3))
        mass = np.full(n, 1.0 / n)
    state = np.stack([pos, vel], axis=1).astype(np.float32)
    return state, mass.astype(np.float32)


def _retry_positions(setup: Setup, positions):
    """``(sparse, dense)``: walk caps sized on ``sparse`` overflow on ``dense``.

    The CPU blobs (N = 512: 1,140 far pairs, 1,168 near edges; caps 2,048 each)
    and a clipped Plummer sphere (2,504 and 3,656). CPU only (see
    ``_default_variants``); on a card ``tests/integration/strict``'s
    ``TestSegmentRetry`` runs the retry at 2e4.
    """
    import numpy as np

    if setup.device != "cpu":
        raise SystemExit("the retry variants run on CPU only")
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, setup.n)
    r = np.minimum(1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0), 20.0)
    mu = rng.uniform(-1.0, 1.0, setup.n)
    phi = rng.uniform(0.0, 2.0 * np.pi, setup.n)
    st = np.sqrt(1.0 - mu * mu)
    plummer = np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    return positions, plummer.astype(np.float32)


def _leaf_capacity(setup: Setup, pos) -> int:
    import jax.numpy as jnp
    import numpy as np
    from yggdrax._cell_partition import adaptive_cell_leaf_partition_numpy
    from yggdrax.morton import morton_encode

    from jaccpot.runtime.kernels._evaluate import _infer_bounds

    p = jnp.asarray(pos)
    codes = np.sort(np.asarray(morton_encode(p, _infer_bounds(p))).astype(np.uint64))
    live = int(
        adaptive_cell_leaf_partition_numpy(
            codes, leaf_size=setup.leaf, min_level=setup.cell_min_level
        )[0].size
    )
    # fused_memory_budget.py's --leaf-cap-factor 1.15, in steps of 1024
    return int(-(-int(np.ceil(1.15 * live)) // 1024) * 1024)


def _solver(setup: Setup, leaf_cap: int, v: "Variant"):
    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    extra = (
        dict(
            adaptive_eps=1e-5,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if v.mac_type == "dehnen_error"
        else {}
    )
    if not v.pallas:
        use_pallas: Optional[bool] = False
    else:
        # the library's choice on a card; the near field's interpret mode on CPU
        use_pallas = True if setup.device == "cpu" else None
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=setup.theta,
        G=v.G,
        softening=setup.softening,
        working_dtype=jnp.float32,
        use_pallas=use_pallas,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=setup.leaf,
                leaf_partition="cells",
                leaf_capacity=leaf_cap,
                cell_min_level=setup.cell_min_level or None,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=v.mac_type,
        ),
        fixed_order=setup.order,
        **extra,
    )


#: fused_memory_budget.py's defaults (its L1-L3 rows): leaf, order, theta, cell level
_BENCH_GALAXY = dict(leaf=64, order=6, theta=0.8, cell_min_level=8)
#: and the TraversalOverrides it passes at leaf 64 (the harness's
#: FAST_LANE_ENV_BY_LEAF[64]["_traversal_overrides"])
_BENCH_TRAVERSAL_LEAF64 = dict(
    max_neighbors_per_leaf=4096, max_interactions_per_node=16384
)


def _galaxy_solver(setup: Setup, v: "Variant", positions):
    """The galaxy pair: ``preset="galaxy"``, or the bench's explicit configuration.

    ``"explicit"`` is ``fused_memory_budget.py``'s ``_solver`` at its defaults, its
    leaf capacity counted as the bench counts it (the library's device count, since
    S3), inside the capture -- where ``"preset"``'s ``leaf_capacity="auto"`` counts
    it, in its first eager prepare. Both then run the same counting programs on the
    same codes, so the two records must agree program for program.
    """
    import warnings

    import jax.numpy as jnp

    from jaccpot import FastMultipoleMethod, TraversalOverrides
    from jaccpot.config import (
        FMMAdvancedConfig,
        NearFieldConfig,
        RuntimePolicyConfig,
        TreeConfig,
    )
    from jaccpot.runtime.capacity_plan import auto_leaf_capacity, live_cell_leaves
    from jaccpot.runtime.kernels._evaluate import _infer_bounds

    use_pallas = True if setup.device == "cpu" else None
    if v.galaxy == "preset":
        return FastMultipoleMethod(
            preset="galaxy", G=v.G, softening=setup.softening, use_pallas=use_pallas
        )
    p = jnp.asarray(positions)
    live = live_cell_leaves(
        p,
        leaf_size=setup.leaf,
        cell_min_level=setup.cell_min_level,
        bounds=_infer_bounds(p),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fixed_order, as the bench
        return FastMultipoleMethod(
            preset="large_n_gpu",
            basis="real",
            theta=setup.theta,
            G=v.G,
            softening=setup.softening,
            softening_kernel=None,
            working_dtype=jnp.float32,
            use_pallas=use_pallas,
            advanced=FMMAdvancedConfig(
                tree=TreeConfig(
                    mode="static_radix",
                    leaf_target=setup.leaf,
                    leaf_partition="cells",
                    leaf_capacity=auto_leaf_capacity(live, 1.15),
                    cell_min_level=setup.cell_min_level or None,
                ),
                nearfield=NearFieldConfig(mode="auto"),
                runtime=RuntimePolicyConfig(
                    traversal_config=TraversalOverrides(**_BENCH_TRAVERSAL_LEAF64)
                ),
                mac_type="dehnen",
            ),
            fixed_order=setup.order,
        )


#: the analytic halo of the external-field variants (fused_memory_budget.py's NFW
#: in AGAMA's convention), module level so its id is fixed for the process
_HALO_M, _HALO_RS = 10.0, 1.0


def nfw_halo(state_in):
    """The halo's acceleration at the packed state's positions ``state_in[:, 0]``."""
    import jax.numpy as jnp

    pos = state_in[:, 0, :]
    r2 = jnp.sum(pos * pos, axis=1)
    r = jnp.sqrt(jnp.maximum(r2, 1e-30))
    x = r / _HALO_RS
    big = jnp.log1p(x) / jnp.maximum(r2 * r, 1e-30) - 1.0 / jnp.maximum(
        r2 * (_HALO_RS + r), 1e-30
    )
    ser = (1.0 / (2.0 * _HALO_RS**3)) * (1.0 - (4.0 / 3.0) * x)
    coeff = jnp.where(x < 1e-3, ser, big)
    return -_HALO_M * coeff[:, None] * pos


# --------------------------------------------------------------------- variants


@dataclasses.dataclass(frozen=True)
class Call:
    """One ``strict_run_v2`` call of a variant."""

    carry: str = "state"
    chain: bool = False  # continue from the previous call's returned state/handle
    external: bool = False
    history: bool = False
    callback: bool = False
    donate_prepared_state: bool = False
    donate_state: bool = False


@dataclasses.dataclass(frozen=True)
class Variant:
    """A configuration of the lane: its calls, environment and solver."""

    calls: tuple = (Call(),)
    env: tuple = ()
    mac_type: str = "dehnen"
    G: float = 1.0
    seam: bool = False
    pallas: bool = True
    #: prepare on a uniform cube, then step the real state: a segment retry
    retry: bool = False
    #: the galaxy pair (S3 R7b): ``"explicit"`` (the bench's configuration) or
    #: ``"preset"`` (``preset="galaxy"``), compared with each other by ``--pair``
    galaxy: str = ""


_TREE_ORDER = (("JACCPOT_STRICT_CARRY_ORDER", "tree"),)
_PARTICLES = Call(carry="particles")
_BASE: dict[str, Variant] = {
    "state": Variant(),
    "particles": Variant(calls=(_PARTICLES,)),
    "tree": Variant(calls=(_PARTICLES,), env=_TREE_ORDER),
    "dehnen_error_state": Variant(mac_type="dehnen_error"),
    "dehnen_error_particles": Variant(calls=(_PARTICLES,), mac_type="dehnen_error"),
    "external_state": Variant(calls=(Call(external=True),)),
    "external_particles": Variant(calls=(Call(carry="particles", external=True),)),
    "external_tree": Variant(
        calls=(Call(carry="particles", external=True),), env=_TREE_ORDER
    ),
    "history_state": Variant(calls=(Call(history=True),)),
    "history_particles": Variant(calls=(Call(carry="particles", history=True),)),
    "callback_state": Variant(calls=(Call(callback=True),)),
    "callback_particles": Variant(calls=(Call(carry="particles", callback=True),)),
    "g_unit": Variant(G=4.3e-6),
    "dehnen_error_chain_state": Variant(
        calls=(Call(), Call(chain=True, donate_prepared_state=True)),
        mac_type="dehnen_error",
    ),
    "dehnen_error_chain_particles": Variant(
        calls=(_PARTICLES, Call(carry="particles", chain=True, donate_state=True)),
        mac_type="dehnen_error",
    ),
    "chain_state": Variant(
        calls=(Call(), Call(chain=True, donate_prepared_state=True)),
    ),
    "chain_particles": Variant(
        calls=(_PARTICLES, Call(carry="particles", chain=True, donate_state=True)),
    ),
    "retry_state": Variant(retry=True),
    "retry_particles": Variant(calls=(_PARTICLES,), retry=True),
    "eval_seam": Variant(seam=True),
}
VARIANTS: dict[str, Variant] = dict(_BASE)
VARIANTS.update(
    {
        f"{name}_purejax": dataclasses.replace(v, pallas=False)
        for name, v in _BASE.items()
        # the force scale needs the CSR lane's direct kernel
        if v.mac_type != "dehnen_error"
    }
)
#: S3 R7b: ``preset="galaxy"`` must compile and compute what the bench's explicit
#: configuration does (``compare TAG TAG --pair galaxy_explicit:galaxy_preset``).
#: Not in the default sets, which stay the S2 base's.
_GALAXY = {
    "galaxy_explicit": Variant(calls=(_PARTICLES,), galaxy="explicit"),
    "galaxy_preset": Variant(calls=(_PARTICLES,), galaxy="preset"),
}
VARIANTS.update(_GALAXY)


def _default_variants(device: str) -> list[str]:
    if device == "cpu":
        return [name for name in VARIANTS if name not in _GALAXY]
    # no retry pair at 2e5: a uniform cube cut into cells of level >= 8 has a leaf
    # per particle, and a second clipped Plummer draw has the first one's counts
    return [name for name, v in _BASE.items() if not v.retry]


@contextlib.contextmanager
def _env(pairs: tuple):
    old = {k: os.environ.get(k) for k, _ in pairs}
    os.environ.update(dict(pairs))
    try:
        yield
    finally:
        for k, val in old.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def _run_variant(name: str, setup: Setup, cap: Capture) -> dict:
    """Run one variant on a fresh solver; return its record (arrays in ``cap``)."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    v = VARIANTS[name]
    if v.galaxy:
        # the bench's operating point on this device's problem (CPU: the blobs)
        setup = dataclasses.replace(setup, **_BENCH_GALAXY)
    state_np, mass_np = _initial_state(setup)
    if v.G != 1.0:
        # the same system in units where G != 1: G m is unchanged
        mass_np = (mass_np.astype(np.float64) / v.G).astype(np.float32)
    # the galaxy pair counts its capacity inside the capture
    leaf_cap = 0 if v.galaxy else _leaf_capacity(setup, state_np[:, 0])
    run_leaf, run_order, run_theta = setup.leaf, setup.order, setup.theta
    if v.galaxy == "preset":
        from jaccpot.production import GALAXY_SETTINGS

        # the run-time half from the settings; theta from the preset's default
        run_leaf, run_order = GALAXY_SETTINGS.leaf_size, GALAXY_SETTINGS.max_order
        run_theta = None
    env = v.env + (_CSR_ON_CPU if v.pallas and setup.device == "cpu" else ())
    returns: dict = {}
    emitted: list = []
    warnings: list[str] = []

    def callback(step, state_now):
        def host(s, c):
            emitted.append((int(s), np.asarray(c)))

        jax.debug.callback(host, step, jnp.sum(state_now[:, 0], axis=0))

    t0 = time.perf_counter()
    with _env(env), capturing(cap):
        solver = (
            _galaxy_solver(setup, v, state_np[:, 0])
            if v.galaxy
            else _solver(setup, leaf_cap, v)
        )
        masses = jnp.asarray(mass_np)
        if v.seam:
            prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
                positions=jnp.asarray(state_np[:, 0]),
                masses=masses,
                leaf_size=setup.leaf,
                max_order=setup.order,
                theta=setup.theta,
            )
            returns.update(cap.digest("ret:prepared", prepared))
            first = len(cap.programs)
            returns.update(cap.digest("ret:force", eval_fn(prepared)))
            if len(cap.programs) != first + 1:
                raise RuntimeError(
                    f"{name}: eval_fn ran {len(cap.programs) - first} captured "
                    "programs, not one"
                )
            cap.roles[first] = "eval_fn"
        else:
            state = jnp.asarray(state_np)
            handle = None
            if v.retry:
                # tests/integration/strict's TestSegmentRetry: caps sized on a
                # configuration with fewer pairs, outgrown by the stepped one
                sparse, dense = _retry_positions(setup, state_np[:, 0])
                handle, _ = solver.strict_fused_prepared_eval_fn(
                    positions=jnp.asarray(sparse),
                    masses=masses,
                    leaf_size=setup.leaf,
                    max_order=setup.order,
                    theta=setup.theta,
                )
                state = state.at[:, 0].set(jnp.asarray(dense))
            for i, call in enumerate(v.calls):
                first = len(cap.programs)
                out, handle, history = solver.strict_run_v2(
                    state=state,
                    masses=masses,
                    dt=setup.dt,
                    num_steps=setup.steps,
                    refresh_every=1,
                    leaf_size=run_leaf,
                    max_order=run_order,
                    theta=run_theta,
                    prepared_state=handle if call.chain or v.retry else None,
                    return_prepared_state=True,
                    add_external=call.external,
                    external_acceleration_fn=nfw_halo if call.external else None,
                    return_history=call.history,
                    step_callback=callback if call.callback else None,
                    carry=call.carry,
                    donate_prepared_state=call.donate_prepared_state,
                    donate_state=call.donate_state,
                )
                jax.effects_barrier()
                # digest now: the next call may donate what this one returned
                returns.update(cap.digest(f"ret:call{i}:state", out))
                if history is not None:
                    returns.update(cap.digest(f"ret:call{i}:history", history))
                if hasattr(handle, "self_acceleration"):
                    # the handle's contract, by name: what the next call reads
                    returns.update(
                        cap.digest(
                            f"ret:call{i}:handle",
                            {
                                "self_acceleration": handle.self_acceleration,
                                "force_scale": getattr(handle, "force_scale", None),
                            },
                        )
                    )
                else:
                    returns.update(cap.digest(f"ret:call{i}:prepared", handle))
                state = out
                ran = cap.programs[first:]
                if not ran or "stablehlo.while" not in ran[-1].lowered:
                    raise RuntimeError(
                        f"{name} call {i}: the last captured program is not a scan "
                        f"({[p.label for p in ran]}): the capture missed the runner"
                    )
                cap.roles[len(cap.programs) - 1] = f"call{i}:runner"
            if emitted:
                # one emission per step; the host may see them out of order
                emitted.sort(key=lambda e: e[0])
                returns.update(
                    cap.digest(
                        "ret:callback",
                        {"steps": np.asarray([s for s, _ in emitted])}
                        | {f"step{s}": c for s, c in emitted},
                    )
                )
        diag = dict(solver.get_runtime_diagnostics() or {})
        fallback = str(diag.get("strict_fused_last_fallback_reason"))
        if v.retry and fallback != "capacity_segment_retry":
            raise RuntimeError(f"{name}: no segment retry ran (last: {fallback})")
        impl = getattr(solver, "_impl", solver)
        cache = getattr(impl, "_strict_fused_jit_function_cache", None)
        if cache is not None:
            loose = [
                repr(k[0]) for k, f in cache.items() if not isinstance(f, _Recorded)
            ]
            if loose:
                raise RuntimeError(f"{name}: jit-cache entries not captured: {loose}")
            unrun = [repr(k[0]) for k, f in cache.items() if f.calls == 0]
            if unrun:
                warnings.append(f"jit-cache entries never run: {unrun}")
    return {
        "variant": dataclasses.asdict(v),
        "env": dict(env),
        "seconds": time.perf_counter() - t0,
        "programs": program_rows(cap),
        "returns": returns,
        "callback_emissions": len(emitted),
        "fallback_count": diag.get("strict_fused_fallback_count"),
        "last_fallback_reason": fallback,
        "warnings": warnings,
    }


def program_rows(cap: Capture) -> list[dict]:
    """The record of each captured program: label, role, text hashes, array digests."""
    return [
        {
            "label": p.label,
            "role": cap.roles.get(i),
            "lowered_sha256": hashlib.sha256(p.lowered.encode()).hexdigest(),
            "lowered_bytes": len(p.lowered),
            "compiled_sha256": (
                None
                if p.compiled is None
                else hashlib.sha256(p.compiled.encode()).hexdigest()
            ),
            "inputs": p.inputs,
            "outputs": p.outputs,
        }
        for i, p in enumerate(cap.programs)
    ]


# --------------------------------------------------------------------- record


def _write_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def _read_text(path: Path) -> str:
    with gzip.open(path, "rt") as fh:
        return fh.read()


def write_variant(vdir: Path, rec: dict, cap: Capture) -> None:
    """``record.json``, the programs' texts and (if kept) ``arrays.npz``."""
    import numpy as np

    vdir.mkdir(parents=True, exist_ok=True)
    for p in cap.programs:
        _write_text(vdir / f"{p.label}.lowered.txt.gz", p.lowered)
        if p.compiled is not None:
            _write_text(vdir / f"{p.label}.compiled.txt.gz", p.compiled)
    if cap.keep_arrays:
        np.savez_compressed(vdir / "arrays.npz", **cap.arrays)
    (vdir / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


def _git(args: list[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _start_jax(device: str):
    """Import jax and check that it runs on ``device``."""
    import jax

    # jaccpot needs 64-bit types (README); as bench/dce_pins.py's cases
    jax.config.update("jax_enable_x64", True)
    platform = jax.devices()[0].platform
    if platform != device:
        raise SystemExit(
            f"--device {device} but jax runs on {platform} (book a card with "
            "autocvd for gpu; JAX_PLATFORMS=cpu for cpu)"
        )
    return jax


def _variant(args) -> int:
    """One variant, in this process (``record`` spawns one per variant)."""
    jax = _start_jax(args.device)
    # imported inside the wrapper, so their module-level jits are captured too
    early = [m for m in ("jaccpot", "yggdrax") if m in sys.modules]
    if early:
        raise RuntimeError(f"{early} imported before jax.jit was wrapped")
    setup = _setup(args.device, args.n)
    cap = Capture(compile_text=args.device == "gpu", keep_arrays=args.save_arrays)
    with wrapped_jit():
        import jaccpot  # noqa: F401

        if args.device == "cpu":
            # the large-N production profile reads it; as the lane goldens do
            jax.default_backend = lambda: "gpu"  # type: ignore[assignment]
        rec = _run_variant(args.name, setup, cap)
    write_variant(Path(args.out), rec, cap)
    print(
        f"{args.name}: {rec['seconds']:.1f} s, programs "
        f"{[p['label'] for p in rec['programs']]}",
        flush=True,
    )
    return 0


def _record(args) -> int:
    jax = _start_jax(args.device)
    import jaccpot

    setup = _setup(args.device, args.n)
    out_dir = Path(args.root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names = (
        args.variants.split(",") if args.variants else _default_variants(args.device)
    )
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {list(VARIANTS)}")
    keep = args.save_arrays if args.save_arrays is not None else args.device == "cpu"
    src = Path(jaccpot.__file__).resolve().parent
    manifest = {
        "tag": args.tag,
        "setup": dataclasses.asdict(setup),
        "jaccpot": str(src),
        "git_head": _git(["rev-parse", "HEAD"], src),
        "git_dirty": bool(_git(["status", "--porcelain", "--", "."], src)),
        "jax": jax.__version__,
        "device": str(jax.devices()[0]),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "jobs": args.jobs,
        "variants": {},
    }
    t0 = time.perf_counter()
    running: list[tuple[str, subprocess.Popen, Any]] = []
    todo = list(names)
    failed = []

    def reap(block: bool) -> None:
        for item in list(running):
            name, proc, log = item
            if block:
                proc.wait()
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            rec_path = out_dir / name / "record.json"
            if proc.returncode != 0 or not rec_path.exists():
                failed.append(name)
                manifest["variants"][name] = {"error": f"rc={proc.returncode}"}
                print(f"[{args.tag}] {name} FAILED (see {log.name})", flush=True)
                continue
            rec = json.loads(rec_path.read_text())
            manifest["variants"][name] = {
                "seconds": rec["seconds"],
                "programs": [p["label"] for p in rec["programs"]],
                "warnings": rec["warnings"],
            }
            print(
                f"[{args.tag}] {name}: {rec['seconds']:.1f} s, "
                f"{manifest['variants'][name]['programs']}",
                flush=True,
            )

    while todo or running:
        while todo and len(running) < max(1, args.jobs):
            name = todo.pop(0)
            log = open(out_dir / f"{name}.log", "w")
            cmd = [
                sys.executable,
                __file__,
                "variant",
                name,
                "--out",
                str(out_dir / name),
                "--device",
                args.device,
                "--save-arrays" if keep else "--no-save-arrays",
            ] + (["--n", str(args.n)] if args.n else [])
            running.append(
                (name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
            )
        time.sleep(1.0)
        reap(block=False)
    manifest["wall_seconds"] = time.perf_counter() - t0
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"[{args.tag}] wrote {out_dir} in {manifest['wall_seconds']:.0f} s; "
        f"failed: {failed or 'none'}"
    )
    return 1 if failed else 0


# --------------------------------------------------------------------- compare


def _text_diff(a: str, b: str, limit: int = 40) -> dict:
    la, lb = a.splitlines(), b.splitlines()
    only_module = (
        len(la) == len(lb)
        and sum(x != y for x, y in zip(la, lb)) == 1
        and la[0] != lb[0]
        and la[0].startswith("module @")
    )
    diff = list(difflib.unified_diff(la, lb, "ref", "new", n=1, lineterm=""))
    return {"only_module_name": only_module, "diff_head": diff[:limit]}


def _compare_arrays(ref: dict, new: dict, ref_npz, new_npz) -> dict:
    """The digests that differ, with element counts where both arrays were kept."""
    import numpy as np

    rows = {}
    for key in sorted(set(ref) | set(new)):
        r, n = ref.get(key), new.get(key)
        if r is None or n is None:
            rows[key] = {"missing_in": "ref" if r is None else "new"}
            continue
        if r == n:
            continue
        row: dict = {"ref": r, "new": n}
        if (
            ref_npz is not None
            and new_npz is not None
            and key in ref_npz.files
            and key in new_npz.files
            and r.get("shape") == n.get("shape")
        ):
            x, y = ref_npz[key], new_npz[key]
            differ = x != y
            if np.issubdtype(x.dtype, np.floating):
                differ &= ~(np.isnan(x) & np.isnan(y))
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            row.update(
                n_diff=int(np.count_nonzero(differ)),
                size=int(x.size),
                max_abs=float(np.nanmax(d)) if d.size else 0.0,
            )
        rows[key] = row
    return rows


def compare_variant(
    ref_dir: Path, new_dir: Path, control_dir: Optional[Path] = None
) -> dict:
    """``compare``'s verdict on one variant's two records.

    With ``control_dir`` (an A-vs-A control of ``ref_dir``), a text or array that
    differs between ref and control too is not run to run reproducible: it is
    listed under ``noise`` and does not fail the variant.
    """
    import numpy as np

    noise: set = set()
    if control_dir is not None and (control_dir / "record.json").exists():
        control = compare_variant(ref_dir, control_dir)
        noise = set(control.get("texts", {})) | set(control.get("arrays", {}))

    r = json.loads((ref_dir / "record.json").read_text())
    n = json.loads((new_dir / "record.json").read_text())
    entry: dict = {}
    rl = [p["label"] for p in r["programs"]]
    nl = [p["label"] for p in n["programs"]]
    if rl != nl:
        entry["programs"] = {"ref": rl, "new": nl}
    texts = {}
    for rp, np_ in zip(r["programs"], n["programs"]):
        for kind in ("lowered", "compiled"):
            rs, ns = rp.get(f"{kind}_sha256"), np_.get(f"{kind}_sha256")
            if rs == ns:
                continue
            if rs is None or ns is None:
                texts[f"{rp['label']}:{kind}"] = {"missing": True}
                continue
            texts[f"{rp['label']}:{kind}"] = _text_diff(
                _read_text(ref_dir / f"{rp['label']}.{kind}.txt.gz"),
                _read_text(new_dir / f"{np_['label']}.{kind}.txt.gz"),
            )
    if texts:
        entry["texts"] = texts
    digests: list[dict] = [{}, {}]
    for rec, into in zip((r, n), digests):
        for p in rec["programs"]:
            into.update(p["inputs"])
            into.update(p["outputs"])
        into.update(rec["returns"])
    rz, nz = ref_dir / "arrays.npz", new_dir / "arrays.npz"
    arrays = _compare_arrays(
        digests[0],
        digests[1],
        np.load(rz) if rz.exists() else None,
        np.load(nz) if nz.exists() else None,
    )
    if arrays:
        entry["arrays"] = arrays
    noisy = sorted(noise & (set(texts) | set(arrays)))
    if noisy:
        entry["noise"] = noisy
    entry["pass"] = not (
        entry.get("programs") or set(texts) - noise or set(arrays) - noise
    )
    entry["n_programs"] = len(nl)
    entry["n_arrays"] = len(digests[1])
    return entry


def _compare(args) -> int:
    root = Path(args.root)
    ref_dir, new_dir = root / args.ref, root / args.new
    control_dir = root / args.control if args.control else None
    ref_m = json.loads((ref_dir / "manifest.json").read_text())
    new_m = json.loads((new_dir / "manifest.json").read_text())
    names = (
        args.variants.split(",")
        if args.variants
        else sorted(set(ref_m["variants"]) | set(new_m["variants"]))
    )
    keys = ("git_head", "git_dirty", "jax", "device", "xla_flags")
    report: dict = {
        "ref": {k: ref_m.get(k) for k in keys},
        "new": {k: new_m.get(k) for k in keys},
        "variants": {},
    }
    # ``--pair A:B``: variant A of ref against variant B of new (the galaxy pair)
    pairs = [tuple(args.pair.split(":", 1))] if args.pair else [(n, n) for n in names]
    ok = True
    for a, b in pairs:
        name = a if a == b else f"{a}:{b}"
        rv, nv = ref_dir / a, new_dir / b
        missing = [
            t for t, p in (("ref", rv), ("new", nv)) if not (p / "record.json").exists()
        ]
        if missing:
            report["variants"][name] = {"pass": False, "missing_in": missing}
            ok = False
            continue
        entry = compare_variant(
            rv, nv, None if control_dir is None else control_dir / a
        )
        ok = ok and entry["pass"]
        report["variants"][name] = entry
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text)
    print(text)
    summary = {k: v["pass"] for k, v in report["variants"].items()}
    failed = [k for k, v in summary.items() if not v]
    print(
        f"{'PASS' if ok else 'FAIL'}: {len(summary) - len(failed)}/{len(summary)} "
        f"variants identical{'; differ: ' + ', '.join(failed) if failed else ''}"
    )
    return 0 if ok else 1


# --------------------------------------------------------------------- main


def _prepare_process_env(device: str) -> None:
    """Scrub the jaccpot/yggdrax environment and set the device's XLA flags.

    Must run before jax is imported: the flags are read when the backend starts.
    """
    for key in [k for k in os.environ if k.startswith(("JACCPOT_", "YGGDRAX_"))]:
        if not key.endswith("_WORKTREE"):
            os.environ.pop(key)
    flags = os.environ.get("XLA_FLAGS", "")
    for flag in _CPU_FLAGS if device == "cpu" else _GPU_FLAGS:
        if flag.split("=", 1)[0] not in flags:
            flags = f"{flags} {flag}".strip()
    os.environ["XLA_FLAGS"] = flags
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--tag", required=True)
    rec.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    rec.add_argument("--n", type=int, default=None, help="512 on cpu, 2e5 on gpu")
    rec.add_argument(
        "--variants",
        default=None,
        help="comma list (default: all on cpu, the CSR ones on gpu); "
        f"known: {', '.join(VARIANTS)}",
    )
    rec.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="variants run in parallel (one process each)",
    )
    rec.add_argument(
        "--save-arrays",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="keep every array next to its sha256 (default: on cpu only)",
    )
    rec.add_argument("--root", default=ROOT_DEFAULT)
    var = sub.add_parser("variant", help="(internal) one variant, in this process")
    var.add_argument("name", choices=sorted(VARIANTS))
    var.add_argument("--out", required=True)
    var.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    var.add_argument("--n", type=int, default=None)
    var.add_argument(
        "--save-arrays", default=True, action=argparse.BooleanOptionalAction
    )
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("ref")
    cmp_.add_argument("new")
    cmp_.add_argument("--variants", default=None)
    cmp_.add_argument(
        "--control",
        default=None,
        help="an A-vs-A control of ref: what differs there too is reported as "
        "noise, not a failure (on a GPU, the eager walk's raw lists)",
    )
    cmp_.add_argument(
        "--pair",
        default=None,
        metavar="A:B",
        help="compare variant A of ref with variant B of new instead (ref and new "
        "may be one tag): galaxy_explicit:galaxy_preset",
    )
    cmp_.add_argument("--root", default=ROOT_DEFAULT)
    cmp_.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return _compare(args)
    _prepare_process_env(args.device)
    return _record(args) if args.cmd == "record" else _variant(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Shared timing methodology for the scaling benchmarks (figures 03-07).

Extracted verbatim from ``bench/bench_jaxfmm_paper_compare.py``, which was built
for the jaxFMM head-to-head and defines the timing protocol the paper reports:
warm up, then take the **minimum** over repeats with the result blocked on
device. The minimum rather than the mean because on a shared GPU the mean
measures other users' jobs as much as ours; the spread is recorded alongside it
so a noisy measurement is visible rather than hidden.

Both that script and ``bench/scaling/*.py`` import from here, so the paper's
numbers and the engineering benchmark's cannot diverge.

Added for the fused-lane port (REBASE_PLAN A2):

* :func:`launch_env` -- the production launch environment (``docs/production.md``:
  XLA's fusion autotuner off; x64 enabled, which ``FastMultipoleMethod`` requires
  even for an fp32 force). It must run before jax starts, which is why jax is
  imported lazily here: importing this module does not start a backend.
* ``plummer_clipped`` in :func:`distribution` -- the production rows' IC (the
  Plummer inverse CDF cut at ``rmax`` scale radii, equal masses), drawn with a jax
  key so it costs nothing at N = 1.9e8.
* :func:`direct_reference` -- the fp64 direct sum on a sampled target set with the
  solver's own softening kernel (ferrers3 for ``preset="galaxy"``), the accuracy
  reference every fused-lane figure scores against; :func:`direct_sum_fn` -- the
  same pair law over every target in the working dtype, jitted: the timed O(N^2) arm.
"""

from __future__ import annotations

import os
import pathlib
import statistics
import sys
import time
from typing import Any, Callable, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__all__ = [
    "DISTRIBUTIONS",
    "aggregate_rel_l2",
    "block_until_ready",
    "direct_reference",
    "direct_sum_fn",
    "distribution",
    "fit_log_log_exponent",
    "launch_env",
    "n_values",
    "per_particle_errors",
    "sample_targets",
    "time_min_repeat",
    "warm_fused_step",
]

#: docs/production.md, "Start every production run with XLA's fusion autotuner off".
AUTOTUNER_OFF = "--xla_gpu_experimental_enable_fusion_autotuner=false"
#: Explicitly EMPTY: an unset flag still builds CUDA graphs, inside which a trace
#: names the graph instead of each kernel (bench/analyse_trace_by_stage.py).
COMMAND_BUFFERS_OFF = "--xla_gpu_enable_command_buffer="

#: The names :func:`distribution` accepts.
DISTRIBUTIONS = ("plummer_clipped", "uniform_cube", "normal", "sphere_surface")


def launch_env(*, command_buffers: bool = True) -> dict[str, Any]:
    """Set the production launch environment; call BEFORE jax is imported.

    Parameters
    ----------
    command_buffers : bool
        ``False`` switches XLA's CUDA graphs off (needed for a per-kernel trace;
        it changes the launch cost, so a figure timed so must say it).

    Returns
    -------
    dict
        What was set, for the result JSON's config.
    """
    if "jax" in sys.modules:
        print(
            "[launch_env] WARNING: jax was imported before the launch environment "
            "was set; XLA_FLAGS may not apply",
            flush=True,
        )
    # FastMultipoleMethod refuses to run without x64 (jaccpot.runtime.dtypes
    # .require_x64); it does not change the working precision.
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_experimental_enable_fusion_autotuner" not in flags:
        flags = f"{flags} {AUTOTUNER_OFF}"
    if not command_buffers and "xla_gpu_enable_command_buffer" not in flags:
        flags = f"{flags} {COMMAND_BUFFERS_OFF}"
    os.environ["XLA_FLAGS"] = flags.strip()
    return {
        "XLA_FLAGS": os.environ["XLA_FLAGS"],
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "command_buffers": "default" if command_buffers else "off",
    }


def block_until_ready(value: Any) -> Any:
    """Block on every array in a pytree, so timings exclude nothing."""
    import jax

    def _maybe_block(x: Any) -> Any:
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    return jax.tree_util.tree_map(_maybe_block, value)


def time_min_repeat(
    fn: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[float, float, float]:
    """Return ``(min, mean, population stdev)`` seconds over ``repeats`` calls."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        block_until_ready(fn())

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        block_until_ready(out)
        end = time.perf_counter()
        samples.append(end - start)

    return (
        float(min(samples)),
        float(statistics.mean(samples)),
        float(statistics.pstdev(samples)),
    )


def n_values(min_exp: int, max_exp: int, steps: int) -> list[int]:
    """Return ``steps`` log-even particle counts between ``2**min_exp`` and ``2**max_exp``.

    Duplicates are collapsed, so a fine ``steps`` at a narrow exponent range does
    not silently measure the same N twice.
    """
    import numpy as np

    if steps <= 1:
        return [int(round(2 ** float(min_exp)))]
    xs = np.linspace(float(min_exp), float(max_exp), steps)
    vals = [int(round(2 ** float(x))) for x in xs]
    out: list[int] = []
    seen = set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def distribution(
    key: Any, n: int, name: str, dtype: Any, *, rmax: float = 20.0
) -> tuple[Any, Any]:
    """Return ``(points, charges)`` for the named distribution.

    These are the jaxFMM-comparison distributions, kept distinct from
    ``bench/validation/_harness.make_distribution``: the validation figures need
    a numpy-seeded astrophysical sampler (Plummer, bulge+halo), whereas the
    timing ladder needs a jax-keyed one that costs nothing to generate at
    N = 2**25. Timing does not depend on the sampler, accuracy does.

    ``plummer_clipped`` is the production rows' IC (``bench/fused_memory_budget.py
    --ic plummer_clipped``): the Plummer inverse CDF on ``[0, X(rmax)]``,
    ``X(r) = r^3 / (r^2 + 1)^1.5``, isotropic, equal masses ``1/N`` -- the same
    sphere without the outliers that set the Morton box at large N. Drawn with a
    jax key, so it is that bench's law, not its sample.
    """
    import jax
    import jax.numpy as jnp

    k1, k2 = jax.random.split(key)
    if name == "uniform_cube":
        pts = jax.random.uniform(k1, (n, 3), minval=-1.0, maxval=1.0, dtype=dtype)
    elif name == "sphere_surface":
        raw = jax.random.normal(k1, (n, 3), dtype=dtype)
        norm = jnp.linalg.norm(raw, axis=1, keepdims=True)
        pts = raw / jnp.maximum(norm, jnp.asarray(1e-12, dtype=dtype))
    elif name == "normal":
        pts = jax.random.normal(k1, (n, 3), dtype=dtype)
    elif name == "plummer_clipped":
        ka, kb, kc = jax.random.split(k1, 3)
        x_max = float(rmax) ** 3 / (float(rmax) ** 2 + 1.0) ** 1.5
        # the floor keeps x^(-2/3) - 1 finite and positive
        x = jax.random.uniform(ka, (n,), minval=1e-12, maxval=x_max, dtype=jnp.float32)
        r = 1.0 / jnp.sqrt(x ** (-2.0 / 3.0) - 1.0)
        mu = jax.random.uniform(kb, (n,), minval=-1.0, maxval=1.0, dtype=jnp.float32)
        phi = jax.random.uniform(
            kc, (n,), minval=0.0, maxval=2.0 * jnp.pi, dtype=jnp.float32
        )
        st = jnp.sqrt(jnp.maximum(1.0 - mu * mu, 0.0))
        pts = jnp.stack(
            [r * st * jnp.cos(phi), r * st * jnp.sin(phi), r * mu], axis=1
        ).astype(dtype)
        return pts, jnp.full((n,), 1.0 / n, dtype=dtype)
    else:
        raise ValueError(f"unknown distribution: {name}")
    charges = jax.random.uniform(k2, (n,), minval=0.0, maxval=1.0, dtype=dtype)
    return pts, charges


def sample_targets(n: int, count: int, seed: int) -> Any:
    """Sorted, seeded target indices for an accuracy score (all ``n`` if ``count >= n``)."""
    import numpy as np

    if count <= 0 or count >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, int(count), replace=False)).astype(np.int64)


def direct_reference(
    positions: Any,
    masses: Any,
    targets: Any,
    *,
    softening: float,
    kernel: Optional[str],
    G: float = 1.0,
    potential: bool = False,
    source_chunk: int = 1 << 16,
    target_block: int = 256,
) -> Any:
    """fp64 direct sum at ``targets`` against every source, with ``kernel``'s softening.

    The reference the fused-lane figures score against. The pair law is
    :func:`jaccpot.softening.pair_factors` (the solver's own), self excluded by
    index, in float64 (needs ``JAX_ENABLE_X64``). Sources stream in chunks of
    ``source_chunk`` and targets in blocks of ``target_block``; every array is a jit
    ARGUMENT, so nothing is constant-folded into the program.

    Returns
    -------
    numpy.ndarray or tuple
        ``acc[T, 3]`` (float64), or ``(acc, pot[T])`` with ``potential=True``,
        ``pot = -G sum_{j != i} m_j psi(r_ij)`` (the fused lane's convention).
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from jaccpot.softening import pair_factors, softening_params

    if not jax.config.jax_enable_x64:
        raise RuntimeError("direct_reference needs JAX_ENABLE_X64=1 (fp64 reference)")
    pos = np.asarray(positions, np.float64)
    mass = np.asarray(masses, np.float64)
    tid_all = np.asarray(targets, np.int64)
    n = int(pos.shape[0])
    chunk = int(min(source_chunk, max(n, 1)))
    npad = -(-n // chunk) * chunk
    sp_h = np.zeros((npad, 3), np.float64)
    sp_h[:n] = pos
    sm_h = np.zeros((npad,), np.float64)
    sm_h[:n] = mass
    sp = jnp.asarray(sp_h.reshape(-1, chunk, 3))
    sm = jnp.asarray(sm_h.reshape(-1, chunk))
    sid = jnp.asarray(np.arange(npad, dtype=np.int64).reshape(-1, chunk))
    params = jnp.asarray(softening_params(kernel, float(softening), jnp.float64))

    @jax.jit
    def block(tp, tid, sp, sm, sid, params):
        def body(c, carry):
            acc, pot = carry
            d = tp[:, None, :] - sp[c][None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            g, psi, _ = pair_factors(
                jnp.where(r2 > 0, r2, 1.0), params, kernel, potential=True
            )
            live = (tid[:, None] != sid[c][None, :]) & (sm[c][None, :] != 0.0)
            w = jnp.where(live, g * sm[c][None, :], 0.0)
            wp = jnp.where(live, psi * sm[c][None, :], 0.0)
            return acc - jnp.einsum("ij,ijk->ik", w, d), pot - jnp.sum(wp, axis=1)

        zero = (jnp.zeros_like(tp), jnp.zeros(tp.shape[:1], tp.dtype))
        return jax.lax.fori_loop(0, sp.shape[0], body, zero)

    accs, pots = [], []
    pos_dev = jnp.asarray(pos)
    for b in range(0, len(tid_all), target_block):
        ib = tid_all[b : b + target_block]
        # pad the last block so every call has one shape (one compile)
        ib_pad = np.concatenate([ib, np.full(target_block - len(ib), ib[-1])])
        ib_dev = jnp.asarray(ib_pad)
        a, p = block(pos_dev[ib_dev], ib_dev, sp, sm, sid, params)
        accs.append(np.asarray(a)[: len(ib)])
        pots.append(np.asarray(p)[: len(ib)])
    acc = float(G) * np.concatenate(accs)
    pot = float(G) * np.concatenate(pots)
    return (acc, pot) if potential else acc


def direct_sum_fn(
    *, softening: float, kernel: Optional[str], G: float, n: int
) -> Callable[[Any, Any], Any]:
    """The timed O(N^2) arm: a jitted ``f(x, m) -> acc[N, 3]`` in ``x``'s dtype.

    Same pair law as :func:`direct_reference`, over every target, blocked over
    targets with ``lax.map`` so its working set is ``block x N``. Positions and
    masses are jit ARGUMENTS (XLA would otherwise constant-fold the whole sum).
    """
    import jax
    import jax.numpy as jnp

    from jaccpot.softening import pair_factors, softening_params

    block = int(max(1, min(256, (1 << 26) // max(int(n), 1))))
    num_blocks = -(-int(n) // block)

    @jax.jit
    def direct(x, m):
        params = softening_params(kernel, float(softening), x.dtype)
        idx = jnp.arange(x.shape[0])

        def one(b):
            t = jnp.clip(b * block + jnp.arange(block), 0, x.shape[0] - 1)
            d = x[t][:, None, :] - x[None, :, :]
            r2 = jnp.sum(d * d, axis=-1)
            g, _, _ = pair_factors(jnp.where(r2 > 0, r2, 1.0), params, kernel)
            w = jnp.where(t[:, None] == idx[None, :], 0.0, g * m[None, :])
            return -jnp.einsum("ij,ijk->ik", w, d)

        out = jax.lax.map(one, jnp.arange(num_blocks))
        return jnp.asarray(G, x.dtype) * out.reshape(-1, 3)[: x.shape[0]]

    return direct


def aggregate_rel_l2(estimate: Any, reference: Any) -> float:
    """``||est - ref|| / ||ref||`` over every component of every target (fp64)."""
    import numpy as np

    est = np.asarray(estimate, np.float64)
    ref = np.asarray(reference, np.float64)
    den = float(np.linalg.norm(ref))
    return float(np.linalg.norm(est - ref) / max(den, 1e-300))


def per_particle_errors(estimate: Any, reference: Any) -> dict[str, float]:
    """Percentiles of ``|a - a_ref| / |a_ref|`` over the targets (the error distribution)."""
    import numpy as np

    ref = np.asarray(reference, np.float64)
    est = np.asarray(estimate, np.float64).reshape(ref.shape[0], -1)
    ref = ref.reshape(ref.shape[0], -1)
    e = np.linalg.norm(est - ref, axis=1) / np.maximum(
        np.linalg.norm(ref, axis=1), 1e-300
    )
    return {
        "p50": float(np.percentile(e, 50)),
        "p90": float(np.percentile(e, 90)),
        "p99": float(np.percentile(e, 99)),
        "max": float(np.max(e)),
    }


def warm_fused_step(
    solver: Any, positions: Any, masses: Any, *, differentiable: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Prepare the fused lane, run one step, and leave the step in steady state.

    ``bench.scaling._galaxy.fused_step`` seeds the first step with the prepare's
    eq (16b) estimate of f_b. A simulation pays that once; every later step takes
    its thresholds from the previous step's evaluation. So this runs one step
    (which also compiles it) and carries ITS f_b into the step the benches time,
    as a scan would. Under the geometric MAC there is no f_b and nothing changes.

    Returns
    -------
    tuple
        ``(fused_step, info)``: info holds the eager prepare's wall time (setup,
        paid once per run; includes its compiles), the first step's (compile +
        run), the first step's capacity verdict and its overflow bits.
    """
    from bench.scaling import _galaxy

    t0 = time.perf_counter()
    fs = _galaxy.fused_step(solver, positions, masses, differentiable=differentiable)
    block_until_ready(fs.topology)
    t1 = time.perf_counter()
    acc, aux = fs.step(positions, masses)
    block_until_ready(acc)
    t2 = time.perf_counter()
    ok = fs.ok(aux)
    if aux[0] is not None:
        fs.aux = (aux[0], None, None)
    return fs, {
        "prepare_s": t1 - t0,
        "first_step_s": t2 - t1,
        "first_step_ok": bool(ok),
        "first_step_overflow_bits": int(aux[1]),
        "f_b_source": (
            "the previous step's evaluation (steady state)"
            if aux[0] is not None
            else "none (geometric MAC)"
        ),
    }


def fit_log_log_exponent(
    ns: list[int], times: list[float], *, min_n: int = 0
) -> dict[str, float]:
    """Least-squares fit of ``log t = alpha * log N + c``; returns alpha and R^2.

    ``min_n`` drops the small-N end, where a GPU measures launch overhead rather
    than the algorithm and would bias the exponent downwards. The number of
    points actually used is returned so the fit is never quoted without knowing
    what it was fitted over.
    """

    import numpy as np

    pairs = [
        (n, t)
        for n, t in zip(ns, times)
        if n >= min_n and t is not None and t > 0 and np.isfinite(t)
    ]
    if len(pairs) < 2:
        return {
            "exponent": float("nan"),
            "r_squared": float("nan"),
            "n_points": len(pairs),
        }

    xs = np.log(np.asarray([p[0] for p in pairs], dtype=float))
    ys = np.log(np.asarray([p[1] for p in pairs], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "exponent": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n_points": len(pairs),
        "fit_min_n": int(min(p[0] for p in pairs)),
        "fit_max_n": int(max(p[0] for p in pairs)),
    }

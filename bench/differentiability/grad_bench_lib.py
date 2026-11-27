"""Shared code for the gradient benches: figure 12, figure 13, the mesh rows.

What lives here and nothing else: the CPU route's environment, the solver at the
paper's operating point with the overrides these benches need, the single-card
differentiable provider, a timer (min and median of warm calls), the memory
records, and an fp64 direct-sum gradient oracle with the solver's own pair law.

No JAX at module scope: :func:`prepare_env` must run before the first ``import
jax`` (the interpret flags and ``XLA_FLAGS`` are read once, when the backend
starts).

The CPU route. The fused lane's REVERSE has one CPU route, the one every gradient
gate uses (``tests/integration/strict/test_fused_differentiable_step.py``,
``tests/distributed/test_mesh_grad_gates.py``): the S9 interpret set (every
Pallas kernel of the lane, and the GPU's walk, in interpret mode) with
``jax.default_backend`` reporting ``"gpu"`` so the large-N profile takes the lane
it takes on a card. ``_galaxy``'s ``cpu_fused_lane`` route is the forward's only:
its differentiable step traces the level loops with dynamic bounds, which have no
reverse (measured: ``ValueError`` from ``_propagate_solidfmm_locals_by_level``).
CPU numbers are smoke only: interpreted kernels say nothing about a GPU.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import numpy as np

#: The S9 interpret set (``tests/distributed/test_mesh_grad_gates.py``,
#: ``bench/multigpu_grad_gate.py``'s ``SMOKE_ENV``).
S9_ENV = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP": "65536",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF": "32",
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
S9_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_CROSS_WALK",
)
#: jax 0.11.2's XLA:CPU loop vectorizer stalls interpret-mode compiles for many
#: minutes (tests/conftest.py); CPU only.
VECTORIZER_OFF = "--xla_backend_extra_options=-vectorize-loops=false"


def cpu_requested() -> bool:
    """Whether ``JAX_PLATFORMS`` pins the CPU (readable before jax starts)."""
    return os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"


def prepare_env(*, host_devices: int = 0) -> dict:
    """The environment of a run, before jax starts; returns what it set.

    On a GPU nothing changes. On the CPU (``JAX_PLATFORMS=cpu``): the S9
    interpret set, the loop vectorizer off and, with ``host_devices >= 2``, that
    many forced host devices (the mesh's only CPU route).

    Parameters
    ----------
    host_devices : int
        Forced host devices on the CPU (``0``: leave XLA's default).

    Returns
    -------
    dict
        The variables set (recorded in the result's config).
    """
    if not cpu_requested():
        return {}
    for key in S9_UNSET:
        os.environ.pop(key, None)
    os.environ.update(S9_ENV)
    flags = os.environ.get("XLA_FLAGS", "")
    if "vectorize-loops" not in flags:
        flags = f"{flags} {VECTORIZER_OFF}"
    if host_devices >= 2 and "xla_force_host_platform_device_count" not in flags:
        flags = f"{flags} --xla_force_host_platform_device_count={int(host_devices)}"
    os.environ["XLA_FLAGS"] = flags.strip()
    return {**S9_ENV, "XLA_FLAGS": os.environ["XLA_FLAGS"]}


def patch_cpu_lane(*, mesh: bool = False) -> bool:
    """After ``import jax``, on the CPU: the gates' patches. Returns whether applied.

    ``jax.default_backend`` reports ``"gpu"`` (the large-N profile's gate) and, on
    a mesh, the ragged exchange resolves to ``buf`` (XLA:CPU has no
    ``ragged_all_to_all``).
    """
    if not cpu_requested():
        return False
    import jax

    jax.default_backend = lambda: "gpu"
    if mesh:
        import yggdrax.distributed.comm as ygg_comm

        resolve = ygg_comm.resolve_ragged_method
        ygg_comm.resolve_ragged_method = lambda method="auto": (
            "buf" if method == "auto" else resolve(method)
        )
    return True


def jnp_dtype(name: str) -> Any:
    """``"float32"`` / ``"float64"`` as a ``jnp`` dtype."""
    import jax.numpy as jnp

    return {"float32": jnp.float32, "float64": jnp.float64}[str(name)]


def solver(
    *,
    softening: float,
    dtype: str,
    mac_type: Optional[str] = None,
    theta: Optional[float] = None,
    adaptive_eps: Optional[float] = None,
    order: Optional[int] = None,
    leaf_capacity: Optional[int] = None,
    cell_min_level: Any = None,
) -> tuple[Any, dict]:
    """``preset="galaxy"`` with the far pairs kept, plus these benches' overrides.

    ``bench/scaling/_galaxy.galaxy_solver`` builds it; ``adaptive_eps``, the order
    and a fixed leaf capacity (LEAVES per tree, the mesh's; ``leaf_headroom``
    dropped with it, as ``bench/multigpu_grad_gate.py`` does) are not among its
    overrides, so a solver that moves one is rebuilt from that one's advanced
    config with only those fields replaced. ``cell_min_level`` (int or
    ``"auto"``; the preset pins 8, ~1 particle per leaf at small N) is
    ``_galaxy``'s own override.

    Returns
    -------
    tuple
        ``(solver, config)``: the config block for the JSON (``_galaxy.describe``,
        with the precision, the tolerance and the order actually used).
    """
    import dataclasses

    from bench.scaling import _galaxy

    base = _galaxy.galaxy_solver(
        softening=softening,
        mac_type=mac_type,
        theta=theta,
        retain_far_pairs=True,
        working_dtype=jnp_dtype(dtype),
        cell_min_level=cell_min_level,
        # on the CPU route the backend reports "gpu", so _galaxy would not set
        # use_pallas (the CSR near lane, the one with a force-scale lane and a
        # reverse); its cpu_fused_lane is inert under the patch
        cpu_route=True if cpu_requested() else None,
    )
    fmm = base
    if adaptive_eps is not None or order is not None or leaf_capacity is not None:
        from jaccpot import FastMultipoleMethod

        advanced = base.advanced
        if leaf_capacity is not None:
            advanced = dataclasses.replace(
                advanced,
                tree=dataclasses.replace(
                    advanced.tree,
                    leaf_capacity=int(leaf_capacity),
                    leaf_headroom=None,
                ),
            )
        if order is not None:
            advanced = dataclasses.replace(
                advanced,
                runtime=dataclasses.replace(advanced.runtime, fixed_order=int(order)),
            )
        kw: dict[str, Any] = {
            "softening": float(softening),
            "G": 1.0,
            "working_dtype": jnp_dtype(dtype),
        }
        if theta is not None:
            kw["theta"] = float(theta)
        if adaptive_eps is not None:
            kw["adaptive_eps"] = float(adaptive_eps)
        if cpu_requested():
            kw["use_pallas"] = True
        fmm = FastMultipoleMethod(preset="galaxy", advanced=advanced, **kw)
    cfg = _galaxy.describe(
        fmm,
        mac_type=mac_type,
        theta=theta,
        leaf_capacity=leaf_capacity,
        cell_min_level=cell_min_level,
        retain_far_pairs=True,
    )
    eps = getattr(getattr(fmm, "_impl", fmm), "adaptive_eps", None)
    cfg.update(
        working_dtype=str(dtype),
        max_order=int(order if order is not None else _galaxy.GALAXY.max_order),
        adaptive_eps=(
            float(eps if eps is not None else _galaxy.GALAXY.adaptive_eps)
            if str(fmm.mac_type) == "dehnen_error"
            else None
        ),
        softening=float(softening),
        # describe() reads on_cpu(), which the CPU route's patch makes False
        cpu_route=("s9_interpret" if cpu_requested() else False),
    )
    if adaptive_eps is not None:
        cfg["off_preset"]["adaptive_eps"] = float(adaptive_eps)
    if order is not None:
        cfg["off_preset"]["max_order"] = int(order)
    if dtype != "float32":
        cfg["off_preset"]["working_dtype"] = str(dtype)
    return fmm, cfg


def provider(fmm: Any, x: Any, m: Any, *, order: Optional[int] = None) -> tuple:
    """The single card's differentiable provider at ``x, m``.

    ``FusedForceProvider.prepare(..., differentiable=True)`` (the gradient plan's
    statics frozen on the eager topology, its far list kept) and the first step's
    carry ``(f_b, None, None)`` as ``_galaxy.fused_step`` seeds it.

    Returns
    -------
    tuple
        ``(provider, eager topology, aux0)``.
    """
    from bench.scaling import _galaxy
    from jaccpot import FusedForceProvider

    prov, topo = FusedForceProvider.prepare(
        fmm,
        x,
        m,
        leaf_size=int(_galaxy.GALAXY.leaf_size),
        max_order=int(order if order is not None else _galaxy.GALAXY.max_order),
        differentiable=True,
    )
    f_b = None
    if fmm.mac_type == "dehnen_error":
        f_b = _galaxy._start_force_scale(fmm, prov, topo)
    return prov, topo, (f_b, None, None)


def compile_timed(fn: Any, *args: Any) -> tuple[Any, float]:
    """``fn.lower(*args).compile()`` and its seconds."""
    t0 = time.perf_counter()
    compiled = fn.lower(*args).compile()
    return compiled, time.perf_counter() - t0


def timed(fn: Callable[..., Any], *args: Any, reps: int) -> dict:
    """Min and median seconds of ``reps`` warm calls (one untimed call first)."""
    import jax

    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(int(reps)):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return {"min_s": float(min(ts)), "median_s": float(np.median(ts)), "reps": reps}


def memory(compiled: Any, n: int) -> Optional[dict]:
    """``memory_analysis()`` of a compiled program, bytes per particle."""
    try:
        ma = compiled.memory_analysis()
    except Exception:  # noqa: BLE001 - not every backend reports it
        return None
    if ma is None:
        return None
    return {
        "argument_B_per_p": ma.argument_size_in_bytes / n,
        "output_B_per_p": ma.output_size_in_bytes / n,
        "temp_B_per_p": ma.temp_size_in_bytes / n,
    }


def device_memory() -> dict:
    """The first device's ``peak_bytes_in_use`` / ``bytes_in_use`` (``None`` on CPU).

    The peak is the process's high-water mark: it includes every earlier program.
    """
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {
        "peak_bytes": stats.get("peak_bytes_in_use"),
        "in_use_bytes": stats.get("bytes_in_use"),
    }


def differing(a: Any, b: Any) -> int:
    """Elements whose bits differ."""
    a, b = np.asarray(a), np.asarray(b)
    view = np.uint64 if a.dtype == np.float64 else np.uint32
    return int(np.sum(a.view(view) != b.view(view)))


def rel_l2(a: Any, b: Any) -> float:
    """``|a - b| / |b|`` in fp64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def weights(n: int, seed: int) -> np.ndarray:
    """The loss's fixed weights ``w``: ``L = sum(w . acc)`` (standard normal)."""
    return np.random.default_rng(int(seed) ^ 0x5EED).normal(size=(int(n), 3))


def direct_loss_grad(
    pos: np.ndarray,
    mass: np.ndarray,
    w: np.ndarray,
    law: Any,
    *,
    targets: Optional[np.ndarray] = None,
    chunk: int = 64,
) -> dict:
    """The exact fp64 direct sum's ``L = sum_t w_t . a_t`` and its gradient.

    Autodiff of a dense fp64 sum over all sources with the solver's pair law
    (``jaccpot.verification.force_law_of``: G, softening, kernel), chunked over
    the targets (all particles, or ``targets``). No tree and no acceptance test:
    the gradient of the force the FMM approximates. Also returns the target
    accelerations, so the caller can check the law against
    ``jaccpot.verification.reference_force`` (the A4 reference).

    Returns
    -------
    dict
        ``L``, ``grad_x`` ``[N, 3]``, ``grad_m`` ``[N]``, ``acc`` ``[K, 3]``.
    """
    import jax
    import jax.numpy as jnp

    from jaccpot.softening import masked_pair_factors, softening_params_from_sq

    kernel = law.kernel
    eps2 = float(law.softening) ** 2
    g = float(law.G)
    x = jnp.asarray(pos, jnp.float64)
    m = jnp.asarray(mass, jnp.float64)
    tid = np.arange(len(pos)) if targets is None else np.asarray(targets)

    @jax.jit
    def block(x, m, tj, wt):
        def loss(x, m):
            d = x[None, :, :] - x[tj][:, None, :]
            r2 = jnp.sum(d * d, -1)
            params = softening_params_from_sq(kernel, eps2, x.dtype)
            inv = masked_pair_factors(r2, r2 > 0, params, kernel)[0]
            a = g * jnp.einsum(
                "ij,ijk->ik", m[None, :] * inv, d, precision=jax.lax.Precision.HIGHEST
            )
            return jnp.sum(a * wt), a

        (val, a), grads = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)(x, m)
        return val, a, grads

    total, gx, gm = 0.0, np.zeros(pos.shape), np.zeros(mass.shape)
    accs = []
    for lo in range(0, len(tid), chunk):
        part = tid[lo : lo + chunk]
        tj = np.zeros(chunk, np.int64)
        wt = np.zeros((chunk, 3))
        tj[: len(part)] = part
        wt[: len(part)] = np.asarray(w)[part]
        val, a, (bx, bm) = block(x, m, jnp.asarray(tj), jnp.asarray(wt))
        total += float(val)
        gx += np.asarray(bx)
        gm += np.asarray(bm)
        accs.append(np.asarray(a)[: len(part)])
    return {"L": total, "grad_x": gx, "grad_m": gm, "acc": np.concatenate(accs)}

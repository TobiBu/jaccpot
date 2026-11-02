"""``differentiable_rollout`` over ``FusedForceProvider.differentiable_step`` (S9).

Track A's differentiable step inside track B's checkpointed velocity Verlet: the
session-1 gates of the plan (``s9-differentiable-rollout-segments.md``, section 5),
on CPU with every Pallas kernel and the GPU's walk in interpret mode, N = 512,
float64, order 2, 3 steps:

* G1 -- the forward-only rollout over ``differentiable_step`` IS
  ``strict_run_v2(carry="particles")``, bitwise, under every checkpoint mode
  (``"nested"`` in segments of 2: one segment and a remainder of 1), Plummer and
  ``mac_type='dehnen_error'`` (seeded with the runner's own f_b);
* G2 -- central differences of the whole rollout (every step's rebuild included)
  in the start positions, velocities and masses, ``G``, ``eps2`` and an external
  field's parameter, with each step's lists checked identical on the +-h runs
  (#316's rule: the differences then differentiate one map), at measured steps;
* G4 -- the three checkpoint modes' gradients agree to round-off, and each is
  bitwise against itself (a second call of the same executable; a second trace
  lowers to the same program text);
* G5 -- every step has far pairs, every reverse kernel of the production layout is
  in the gradient's trace, and zeroing the M2L reverse moves the gradient;
* G8 -- caps sized on a configuration with fewer pairs give a failed verdict, a
  refused gradient, and after the provider's re-plan a clean rerun.

``jax.default_backend`` reports ``"gpu"``, as ``tests/characterization/
test_lane_goldens.py`` does. Interpret-mode reverses compile for minutes, so the
lanes are module-scoped and every test is listed in ``tests/slow_tests.txt``.
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 512
LEAF, ORDER, THETA, SOFTENING = 16, 2, 0.6, 1e-2
DT, STEPS = 1e-3, 3
#: the external field's spring constant (a row-by-row harmonic pull)
K_EXT = 0.37
MODES = [(None, None), ("step", None), ("nested", 2)]

#: what the strict tests scrub (a shell's JACCPOT_* must not reach the lane)
_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_FLAT_WALK_CAP_HEADROOM",
    "JACCPOT_STATIC_STRICT_FUSED_WALK",
    "JACCPOT_STRICT_SEGMENT_RETRY",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
)
#: every Pallas kernel of the lane in interpret mode, the prepacked near layout,
#: the GPU's walk (``test_fused_differentiable_step.py``'s set)
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
    "JACCPOT_WALK_PALLAS_INTERPRET": "1",
}
#: the reverse kernels of the production layout (``test_fused_differentiable_step``)
_REVERSE_KERNELS = (
    ("jaccpot.pallas.p2m_real_leaf", "p2m_real_leaves_reverse_pallas", "p2m"),
    ("jaccpot.pallas.cascade_real_level", "m2m_real_levels_reverse_pallas", "m2m"),
    ("jaccpot.pallas.cascade_real_level", "l2l_real_levels_reverse_pallas", "l2l"),
    ("jaccpot.pallas.m2l_real_csr_lanes", "m2l_real_csr_lanes_reverse_pallas", "m2l"),
    ("jaccpot.pallas.l2p_real", "l2p_real_particles_reverse_pallas", "l2p"),
    (
        "jaccpot.pallas.nearfield_leafpair_csr_direct_vjp",
        "nearfield_leafpair_csr_sorted_direct_reverse_pallas",
        "near",
    ),
)


@pytest.fixture(scope="module", autouse=True)
def _lane_on_cpu() -> Iterator[None]:
    """The interpret flags, and the large-N profile's backend gate open.

    Yields
    ------
    None
        With the environment set for the module.
    """
    patcher = pytest.MonkeyPatch()
    for key in _ENV_UNSET:
        patcher.delenv(key, raising=False)
    for key, value in _ENV.items():
        patcher.setenv(key, value)
    patcher.setattr(jax, "default_backend", lambda: "gpu")
    yield
    patcher.undo()
    _LANES.clear()
    _GRADS.clear()
    _DIGESTS.clear()


def _solver(mac_type: str = "dehnen") -> Any:
    from jaccpot import FastMultipoleMethod
    from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig

    criterion = (
        dict(
            adaptive_eps=1e-5,
            adaptive_error_model="dehnen_paper",
            mac_force_scale_mode="paper_fb",
        )
        if mac_type == "dehnen_error"
        else {}
    )
    return FastMultipoleMethod(
        preset="large_n_gpu",
        basis="real",
        theta=THETA,
        G=1.0,
        softening=SOFTENING,
        working_dtype=jnp.float64,
        use_pallas=True,
        advanced=FMMAdvancedConfig(
            tree=TreeConfig(
                mode="static_radix",
                leaf_target=LEAF,
                leaf_partition="cells",
                leaf_capacity=1024,
            ),
            nearfield=NearFieldConfig(mode="auto"),
            mac_type=mac_type,
        ),
        fixed_order=ORDER,
        **criterion,
    )


def _blobs(seed: int = 0) -> np.ndarray:
    """Four tight blobs: the pair counts the caps of G8 are sized on."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    return np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )


def _system() -> tuple[Any, Any, Any, Any]:
    """The blobs with velocities, total mass about 1.

    With masses of order 1 each (a total of ~460, accelerations ~1e4 in the blobs)
    the particles move ~5e-3 per step and the third step's walk sits on a decision
    boundary: G scaled by 1 + 1e-9 already adds two far pairs there, so no FD step
    keeps the lists. At a total mass of ~1 every input's +-h runs below keep every
    step's lists (CPU, 2026-10-31).
    """
    pos = _blobs(0)
    rng = np.random.default_rng(1)
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = (np.abs(rng.normal(size=N)) + 0.5) / N
    weight = rng.normal(size=(N, 3))
    return tuple(jnp.asarray(a, jnp.float64) for a in (pos, vel, masses, weight))


def _harmonic(state_rows, params):
    return -params["k"] * state_rows[:, 0]


@dataclasses.dataclass
class Lane:
    """A configuration prepared at the start positions, with its first aux."""

    provider: Any
    aux0: Any
    positions: Any
    velocities: Any
    masses: Any
    weight: Any


_LANES: dict = {}


def _seed(solver: Any, topology: Any) -> Any:
    """``strict_run_v2``'s first f_b: the eager prepare's eq (16b) prepass, in
    input order."""
    fb = jnp.asarray(solver._impl._last_force_scale_particles, jnp.float64)
    perm = jnp.asarray(topology.prepared.tree.particle_indices)
    return jnp.zeros_like(fb).at[perm].set(fb)


def _lane(name: str) -> Lane:
    """The lane ``name`` (``plummer`` / ``dehnen_error``), prepared once."""
    if name in _LANES:
        return _LANES[name]
    from jaccpot import FusedForceProvider

    solver = _solver("dehnen_error" if name == "dehnen_error" else "dehnen")
    x, v, m, w = _system()
    prov, topology = FusedForceProvider.prepare(
        solver,
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        differentiable=True,
    )
    aux0 = (_seed(solver, topology), None, None) if prov.scale_carry else None
    _LANES[name] = Lane(prov, aux0, x, v, m, w)
    return _LANES[name]


def _rollout(lane, step_fn, x, v, m, *, checkpoint="step", inner=None, **kw):
    from jaccpot.runtime.grad_rollout import differentiable_rollout

    return differentiable_rollout(
        step_fn,
        x,
        v,
        m,
        DT,
        STEPS,
        aux=lane.aux0,
        checkpoint=checkpoint,
        inner_steps=inner,
        **kw,
    )


def _field_kw(lane: Lane, g: Any, eps2: Any, k: Any) -> dict:
    """The traced force parameters (not under ``dehnen_error``: it takes the
    solver's own) and the harmonic field with a traced constant."""
    from jaccpot import ForceParams

    params = None if lane.provider.scale_carry else ForceParams(G=g, eps2=eps2)
    return dict(params=params, external_fn=_harmonic, external_args={"k": k})


def _loss(lane: Lane, checkpoint: Any, inner: Any):
    """``(sum(w x_T) + sum(w v_T), RolloutResult)`` of the differentiable rollout."""

    def loss(x, v, m, g, eps2, k):
        out = _rollout(
            lane,
            lane.provider.differentiable_step,
            x,
            v,
            m,
            checkpoint=checkpoint,
            inner=inner,
            **_field_kw(lane, g, eps2, k),
        )
        value = jnp.sum(lane.weight * out.positions)
        return value + jnp.sum(lane.weight * out.velocities), out

    return loss


def _args(lane: Lane) -> tuple:
    return (
        lane.positions,
        lane.velocities,
        lane.masses,
        jnp.asarray(1.0),
        jnp.asarray(SOFTENING**2),
        jnp.asarray(K_EXT),
    )


@dataclasses.dataclass
class Gradient:
    """One checkpoint mode's value and gradient at the lane's start, computed once."""

    value: Any
    out: Any
    grads: tuple
    #: bytes that differ between two calls of the same executable (A-vs-A)
    repeat_differ: int
    #: sha256 of the lowered program's text (``debug_info=False``): the program a
    #: recompile compiles
    program: str


_GRADS: dict = {}

#: The tests that read Plummer's ``"step"`` gradient (G2, G4, G5) share a worker
#: under ``--dist loadgroup``, and a CI shard (``integration-strict-grad-modes`` in
#: ``.github/scripts/test_shards.py``), so it is compiled once (~3 min on CPU). The
#: file is ~45 min of compilation in one process, so CI runs it as two shards.
_PLUMMER_GRADIENT = pytest.mark.xdist_group("grad_rollout_plummer")


def _value_and_grad(lane: Lane, checkpoint: Any, inner: Any) -> Any:
    """A fresh ``jit(value_and_grad)`` of :func:`_loss`: a new trace on every call."""
    return jax.jit(
        jax.value_and_grad(
            _loss(lane, checkpoint, inner), argnums=tuple(range(6)), has_aux=True
        )
    )


def _drop_executables() -> None:
    """Free jax's in-process executables (a rollout gradient's is 3-6 GB on CPU).

    The module's results are kept as arrays, never as compiled programs, so a test
    that follows recompiles nothing it reads.
    """
    jax.clear_caches()
    gc.collect()


def _program_digest(lowered: Any) -> str:
    """sha256 of a lowered program's text without locations: what a compile reads."""
    return hashlib.sha256(lowered.as_text(debug_info=False).encode()).hexdigest()


def _gradient(name: str, checkpoint: Any = "step", inner: Any = None) -> Gradient:
    """The gradient of ``(name, checkpoint, inner)`` at the lane's start, once.

    Lowered, compiled and called twice: the second call is G4's A-vs-A on the
    same executable, and the lowered text is what G4 compares a second trace
    against. The executable is dropped before returning.
    """
    key = (name, checkpoint, inner)
    if key not in _GRADS:
        lane = _lane(name)
        args = _args(lane)
        lowered = _value_and_grad(lane, checkpoint, inner).lower(*args)
        compiled = lowered.compile()
        (value, out), grads = compiled(*args)
        _, again = compiled(*args)
        repeat = sum(_bitwise(a, b) for a, b in zip(again, grads))
        program = _program_digest(lowered)
        del lowered, compiled, again
        _drop_executables()
        _GRADS[key] = Gradient(value, out, grads, repeat, program)
    return _GRADS[key]


def _bitwise(got: Any, want: Any) -> int:
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    # flattened first: a 0-d array cannot change its dtype's itemsize in a view
    got, want = got.reshape(-1), want.reshape(-1)
    return int(np.sum(got.view(np.uint8) != want.view(np.uint8)))


# ---------------------------------------------------------------------------
# Each step's lists, recorded along a forward rollout (G2's identity check, G5).
# ---------------------------------------------------------------------------


def _list_digest(prepared: Any) -> Any:
    """Integer fingerprints of a topology's index arrays: the tree's permutation,
    the near CSR, the far list and its count."""
    nl, far = prepared.neighbor_list, prepared.compact_far_pairs
    arrays = (
        prepared.tree.particle_indices,
        nl.offsets,
        nl.neighbors,
        far.sources,
        far.targets,
    )
    digests = [
        jnp.sum(
            jnp.asarray(a, jnp.int64).ravel()
            * (jnp.arange(a.size, dtype=jnp.int64) % 1_000_003 + 1)
        )
        for a in arrays
    ]
    return jnp.stack([*digests, jnp.asarray(far.far_pair_count, jnp.int64)])


#: each lane's recorded lists along the forward rollout from its start (G2 records
#: them; G5 reads them instead of compiling the recording rollout again)
_DIGESTS: dict = {}


def _centre_digests(name: str) -> np.ndarray:
    """The lists every force of ``name``'s rollout from the start rebuilt."""
    if name not in _DIGESTS:
        lane = _lane(name)
        _, digests, bits = _recorded(lane)(*_args(lane))
        assert int(bits) == 0
        _DIGESTS[name] = np.asarray(digests)
    return _DIGESTS[name]


def _recorded(lane: Lane):
    """A jitted forward rollout over ``differentiable_step`` that also records, per
    force (the start's and each step's), the lists the step rebuilt."""
    prov = lane.provider

    def recording(x, m, params, aux):
        inner, digests, i = aux
        f_b = None if inner is None else inner[0]
        stop = jax.lax.stop_gradient
        prepared = prov.rebuild(
            stop(x), stop(m), force_scale=f_b, keep_far_list=True
        ).prepared
        acc, inner = prov.differentiable_step(x, m, params, inner)
        return acc, (inner, digests.at[i].set(_list_digest(prepared)), i + 1)

    @jax.jit
    def run(x, v, m, g, eps2, k):
        from jaccpot.runtime.grad_rollout import differentiable_rollout

        out = differentiable_rollout(
            recording,
            x,
            v,
            m,
            DT,
            STEPS,
            aux=(
                lane.aux0,
                jnp.zeros((STEPS + 1, 6), jnp.int64),
                jnp.asarray(0, jnp.int32),
            ),
            checkpoint=None,
            **_field_kw(lane, g, eps2, k),
        )
        value = jnp.sum(lane.weight * out.positions)
        value = value + jnp.sum(lane.weight * out.velocities)
        return value, out.aux[1], out.aux[0][1]

    return run


# ---------------------------------------------------------------------------
# G1: the forward is the runner's.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_the_forward_rollout_is_the_particle_carry(name):
    """The jitted forward over ``differentiable_step`` (masses an argument, the
    start force inside) against ``strict_run_v2(carry="particles")``, every mode."""
    from jaccpot.runtime.grad_rollout import check_rollout

    lane = _lane(name)
    solver = _solver("dehnen_error" if name == "dehnen_error" else "dehnen")
    state = jnp.stack([lane.positions, lane.velocities], axis=1)
    want, _, _ = solver.strict_run_v2(
        state=state,
        masses=lane.masses,
        dt=DT,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        carry="particles",
    )
    want = np.asarray(want)
    for checkpoint, inner in MODES:
        out = jax.jit(
            lambda x, v, m: _rollout(
                lane,
                lane.provider.differentiable_step,
                x,
                v,
                m,
                checkpoint=checkpoint,
                inner=inner,
            )
        )(lane.positions, lane.velocities, lane.masses)
        check_rollout(out, gradient=True)
        got = np.asarray(jnp.stack([out.positions, out.velocities], axis=1))
        differ = int(np.sum(got != want))
        assert differ == 0, (
            f"checkpoint={checkpoint!r}: {differ} of {got.size} elements differ, "
            f"max {float(np.max(np.abs(got - want))):.3e}"
        )


# ---------------------------------------------------------------------------
# G2: finite differences of the whole rollout, one map.
# ---------------------------------------------------------------------------

#: The step sizes, MEASURED on these directions (CPU, 2026-10-31): absolute for the
#: arrays (along unit-variance normal directions; the masses are ~2e-3), relative
#: to the value for the scalars. Every step below keeps every force's lists on both
#: +-h runs, and the relative error there is 4e-11 - 4.4e-10 (Plummer) and 8e-13 -
#: 2.3e-10 (``dehnen_error``). Larger steps cross walk decisions: x0 at >= 1e-6
#: rebuilt other lists under ``dehnen_error`` (error 1.9e-4, 2.2e-5 at 1e-5), m at
#: >= 1e-6 under both (9.6e-2 and 1.2e-4); v0 at 1e-3 under ``dehnen_error``. Smaller
#: ones reach round-off: G at 1e-8 relative 2.6e-5, k 1.7e-5.
_FD_STEPS = {
    "plummer": {"x0": 1e-7, "v0": 1e-4, "m": 1e-7, "G": 1e-3, "eps2": 1e-4, "k": 1e-3},
    "dehnen_error": {"x0": 1e-7, "v0": 1e-4, "m": 1e-7, "k": 1e-3},
}


@pytest.mark.parametrize(
    "name", [pytest.param("plummer", marks=_PLUMMER_GRADIENT), "dehnen_error"]
)
def test_the_rollout_gradient_matches_finite_differences(name):
    """Central differences of the 3-step rollout in float64, every rebuild in it.

    Each +-h run records the lists every force of the rollout rebuilt (the start
    force and the 3 steps) and they must equal the centre's, so the difference
    quotient differentiates the map the gradient differentiates (#316). The
    tolerance is 1e-7 relative (measured: at most 4.4e-10). Under ``dehnen_error`` f_b
    enters only through the walk's thresholds (discretely: a 1e-4 scaling of it
    leaves the step's acceleration bitwise unchanged, track A), so a +-h pair that
    crosses a walk decision -- even one that lands both runs on the same
    neighbouring topology -- shows as a plateau, not a missing derivative: the
    identity check is what keeps such a pair out.
    """
    lane = _lane(name)
    gradient = _gradient(name)
    grads = gradient.grads
    from jaccpot.runtime.grad_rollout import check_rollout

    check_rollout(gradient.out, gradient=True)
    run = _recorded(lane)
    centre = _args(lane)
    _, digests0, bits0 = run(*centre)
    assert int(bits0) == 0
    _DIGESTS.setdefault(name, np.asarray(digests0))
    names = ("x0", "v0", "m", "G", "eps2", "k")
    rng = np.random.default_rng(7)
    for i, input_name in enumerate(names):
        if input_name not in _FD_STEPS[name]:
            continue
        a = centre[i]
        if a.ndim:
            d = jnp.asarray(rng.normal(size=a.shape))
            h = _FD_STEPS[name][input_name]
        else:
            d = jnp.asarray(1.0)
            h = _FD_STEPS[name][input_name] * abs(float(a))
        values = []
        for sign in (1.0, -1.0):
            moved = list(centre)
            moved[i] = a + sign * h * d
            value, digests, _ = run(*moved)
            assert np.array_equal(np.asarray(digests), np.asarray(digests0)), (
                f"{input_name}: the {'+' if sign > 0 else '-'}h run rebuilt "
                "other lists"
            )
            values.append(float(value))
        fd = (values[0] - values[1]) / (2.0 * h)
        exact = float(jnp.sum(grads[i] * d))
        assert exact != 0.0, input_name
        assert abs(exact - fd) <= 1e-7 * abs(fd), (input_name, exact, fd)


# ---------------------------------------------------------------------------
# G4: the checkpoint modes, and A-vs-A.
# ---------------------------------------------------------------------------


@_PLUMMER_GRADIENT
def test_the_checkpoint_modes_agree_and_each_repeats_bitwise():
    """``None`` / ``"step"`` / ``"nested"``: gradients within 1e-12 of the largest
    component (round-off: a checkpoint regroups cotangent sums, ``"nested"`` sums
    the constants' per segment; ``tests/unit/runtime/test_grad_rollout.py``), the
    gradient verdict passes; each mode's gradient bitwise on a second call of the
    same executable, and a second trace lowers to the same program text (a recompute
    that rebuilt other lists would differ).

    The second trace is compared as program text, not compiled and run: the text is
    all a recompile reads, and under CI's persistent compilation cache the old
    compile-and-run was a cache hit on that text, the same executable. Measured on
    CPU (2026-11-01, loaded 64-core host): a second compile cost 100-270 s per mode
    without the cache, 22-73 s with it, and 3-6 GB of RSS either way; the trace
    alone 15-35 s.
    """
    from jaccpot.runtime.grad_rollout import rollout_ok

    lane = _lane("plummer")
    ref = None
    for checkpoint, inner in MODES:
        gradient = _gradient("plummer", checkpoint, inner)
        grads = gradient.grads
        assert rollout_ok(gradient.out, gradient=True)
        assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in grads)
        assert gradient.repeat_differ == 0, checkpoint
        retraced = _value_and_grad(lane, checkpoint, inner).lower(*_args(lane))
        assert _program_digest(retraced) == gradient.program, checkpoint
        del retraced
        _drop_executables()
        if ref is None:
            ref = grads
            continue
        for g, w in zip(grads, ref):
            g, w = np.asarray(g), np.asarray(w)
            scale = float(np.max(np.abs(w)))
            assert float(np.max(np.abs(g - w))) <= 1e-12 * scale, checkpoint


# ---------------------------------------------------------------------------
# G5: the far field is load-bearing.
# ---------------------------------------------------------------------------


@_PLUMMER_GRADIENT
def test_the_far_field_is_load_bearing_in_the_rollout_gradient():
    """Far pairs at every force of the rollout; every reverse kernel of the
    production layout traced in the rollout's gradient; and the M2L reverse
    zeroed, the gradient moves (S8's inertness rule: a gradient that skipped the
    far field would still be finite)."""
    lane = _lane("plummer")
    assert all(int(c) > 0 for c in _centre_digests("plummer")[:, -1])
    grads = _gradient("plummer").grads
    counts = {key: 0 for *_, key in _REVERSE_KERNELS}
    # the kernels are counted (and the M2L reverse zeroed) when they TRACE: an
    # inner jitted function an earlier test traced at these shapes would be reused
    # from jax's caches and never call them
    jax.clear_caches()
    patcher = pytest.MonkeyPatch()
    for module_name, fn_name, key in _REVERSE_KERNELS:
        module = importlib.import_module(module_name)
        real = getattr(module, fn_name)

        def counted(*args, _real=real, _key=key, **kwargs):
            counts[_key] += 1
            out = _real(*args, **kwargs)
            if _key == "m2l":
                out = jax.tree_util.tree_map(jnp.zeros_like, out)
            return out

        patcher.setattr(module, fn_name, counted)
    try:
        fn = jax.jit(
            jax.grad(
                lambda *a: _loss(lane, "step", None)(*a)[0], argnums=tuple(range(6))
            )
        )
        no_m2l = fn(*_args(lane))
        jax.block_until_ready(no_m2l)
    finally:
        patcher.undo()
        jax.clear_caches()  # no zeroed reverse outlives the test
    assert all(c >= 1 for c in counts.values()), counts
    for name, g, w in zip(("x0", "v0", "m"), no_m2l, grads):
        g, w = np.asarray(g), np.asarray(w)
        moved = float(np.max(np.abs(g - w))) / float(np.max(np.abs(w)))
        assert moved > 1e-6, (name, moved)


# ---------------------------------------------------------------------------
# G8: an overflow is an output, and the gradient is refused.
# ---------------------------------------------------------------------------


def test_undersized_caps_refuse_the_gradient_and_a_replan_runs_clean():
    """Caps sized on the four blobs, the rollout started on a clipped Plummer
    sphere (more pairs): the folded bits are nonzero, ``value_and_grad_checked``
    refuses the gradient, and after ``provider.replanned`` (from the rollout's
    needs, at its start) and ``with_grad_plan`` the rerun's verdict is clean and
    its gradient finite."""
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.grad_rollout import (
        RolloutVerdictError,
        check_rollout,
        rollout_ok,
        value_and_grad_checked,
    )

    solver = _solver()
    x, v, m, w = _system()
    small, _ = FusedForceProvider.prepare(
        solver,
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        differentiable=True,
    )
    rng = np.random.default_rng(1)
    u = rng.uniform(0.0, 1.0, N)
    r = np.minimum(1.0 / np.sqrt(u ** (-2.0 / 3.0) - 1.0), 20.0)
    mu, phi = rng.uniform(-1.0, 1.0, N), rng.uniform(0.0, 2.0 * np.pi, N)
    st = np.sqrt(1.0 - mu * mu)
    plummer = jnp.asarray(
        np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * mu], 1)
    )

    def loss_for(prov):
        def loss(x0, v0, m0):
            from jaccpot.runtime.grad_rollout import differentiable_rollout

            out = differentiable_rollout(
                prov.differentiable_step, x0, v0, m0, DT, STEPS
            )
            return jnp.sum(w * out.positions), out

        return loss

    run = value_and_grad_checked(loss_for(small), argnums=(0, 2))
    with pytest.raises(RolloutVerdictError, match="gradient verdict failed"):
        run(plummer, v, m)
    del run
    _drop_executables()  # each program here is GBs: one at a time
    _, out = jax.jit(loss_for(small))(plummer, v, m)
    bits, needs = int(out.aux[1]), np.asarray(out.aux[2])
    _drop_executables()
    assert bits != 0 and not rollout_ok(out, gradient=True)
    with pytest.raises(RolloutVerdictError):
        check_rollout(out, gradient=True)
    assert small.retry_mode(needs, 0, False) == "full"
    replanned = small.replanned(needs, plummer, m)
    assert replanned is not None
    prov, topology = replanned
    prov = prov.with_grad_plan(topology)
    (_, out), grads = value_and_grad_checked(loss_for(prov), argnums=(0, 2))(
        plummer, v, m
    )
    assert int(out.aux[1]) == 0
    assert all(bool(np.all(np.isfinite(np.asarray(g)))) for g in grads)

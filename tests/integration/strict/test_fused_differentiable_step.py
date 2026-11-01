"""``FusedForceProvider.differentiable_step``: the fused lane's step, differentiable (S9).

``step`` rebuilds the topology (tree, walk, P2M ... L2L) and evaluates on it; its
topology stage holds dynamic-bound while loops (the walk's rounds, the extents'
propagation, the level loops) that have no reverse, and its evaluation reads no
positions, so ``accelerations(rebuild(stop_gradient(x)))`` differentiates to exactly
zero. ``differentiable_step`` rebuilds on the inputs' stopped gradients, keeps the
walk's far list, freezes the gradient plan's static fields once on the provider
(``prepare(..., differentiable=True)``) and evaluates with S8's differentiable force
at the LIVE positions and masses. Gates, on CPU (every Pallas kernel in interpret
mode, the GPU's walk in interpret mode unless said otherwise: its far list is the
target-sorted, symmetric CSR), N = 512, float64, order 2:

* the forward is ``step``'s bitwise, accelerations AND aux (f_b, bits, needs):
  Plummer and ferrers3 softening, ``mac_type='dehnen_error'``, at several position
  sets, and a velocity-Verlet scan of three steps chained through the aux;
* the gradient is S8's (``differentiable_step_fn``) on the eager prepared topology at
  the same positions, bitwise -- on the GPU's walk and on the CPU's flat walk (whose
  far list is neither sorted nor symmetric: the sorting M2L reverse);
* central differences in float64 of the whole step (the rebuild included, its
  topology checked identical at +-h) in the positions and masses (directional), G and
  eps2: S8's tolerances, 1e-6 / 1e-6 / 1e-10 / 1e-6;
* the far field is load-bearing: far pairs in the traced list, and every reverse
  kernel of the production layout traced once (P2M, M2M, M2L, L2L, L2P, near);
* the linearized program holds no while loop (the topology stage is primal only);
* a list that is not symmetric sets ``OVERFLOW_GRAD_LISTS`` (outside ``overflow_ok``,
  inside ``grad_ok``) and makes the gradient NaN; the static fields are checked on
  every trace.

``jax.default_backend`` reports ``"gpu"``, as
``tests/characterization/test_lane_goldens.py`` does. Interpret-mode reverses compile
for tens of seconds each, so the fixtures are module-scoped and the tests are listed in
``tests/slow_tests.txt``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import ForceParams
from jaccpot.runtime.fused_topology import (
    OVERFLOW_GRAD_LISTS,
    grad_ok,
    overflow_ok,
)

N = 512
LEAF, ORDER, THETA, SOFTENING = 16, 2, 0.6, 1e-2
DT = 1e-3

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
    "JACCPOT_WALK_PALLAS_INTERPRET",
)
#: every Pallas kernel of the lane in interpret mode, the prepacked near layout
#: (``tests/unit/runtime/test_large_n_far_chain_grad.py``'s set)
_ENV = {
    "JACCPOT_CASCADE_PALLAS": "1",
    "JACCPOT_CASCADE_PALLAS_INTERPRET": "1",
    "JACCPOT_M2L_CSR_INTERPRET": "1",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_L2P_KERNEL": "interpret",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


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


def _solver(kernel: str = "plummer", mac_type: str = "dehnen") -> Any:
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
        softening_kernel=kernel,
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


def _system() -> tuple[Any, Any, Any, Any]:
    """Four tight blobs (provider_hlo_identity.py's CPU draw), float64.

    Returns
    -------
    tuple[Any, Any, Any, Any]
        Positions, velocities, masses and a weight for the loss.
    """
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    weight = rng.normal(size=(N, 3))
    return tuple(jnp.asarray(a, jnp.float64) for a in (pos, vel, masses, weight))


@dataclasses.dataclass
class Lane:
    """A configuration, prepared at the blobs, with its first step's aux."""

    fmm: Any
    provider: Any
    topology: Any
    aux0: Any
    positions: Any
    velocities: Any
    masses: Any
    weight: Any


_LANES: dict = {}

#: Two worker groups under ``--dist loadgroup``: the forward checks and the
#: gradient's, ~10 and ~9 min of compilation on CPU (2026-11-01); one group ran the
#: file in one worker (18.5 min). Each group prepares the lanes it reads (~30 s each).
_FORWARD = pytest.mark.xdist_group("fused_differentiable_step_forward")
_GRADIENT = pytest.mark.xdist_group("fused_differentiable_step_gradient")


@pytest.fixture(autouse=True)
def _walk(request, monkeypatch) -> None:
    """The walk of the test's lane, read when its programs TRACE.

    ``flat`` is the CPU's walk (yggdrax ``dual_tree_walk_mutual``: a far list in
    pair order); every other lane the GPU's (the Pallas walk in interpret mode: the
    target-sorted CSR, symmetric). A test without a ``name`` uses ``plummer``'s.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The test's parameters.
    monkeypatch : pytest.MonkeyPatch
        Undoes the switch.
    """
    callspec = getattr(request.node, "callspec", None)
    name = callspec.params.get("name", "plummer") if callspec else "plummer"
    if name == "flat":
        monkeypatch.delenv("JACCPOT_WALK_PALLAS_INTERPRET", raising=False)
    else:
        monkeypatch.setenv("JACCPOT_WALK_PALLAS_INTERPRET", "1")


def _lane(name: str) -> Lane:
    """The lane ``name``, prepared once per module.

    Parameters
    ----------
    name : str
        ``plummer`` / ``ferrers3`` (the softening kernel) / ``dehnen_error`` (the
        MAC; Plummer softening), each on the GPU's walk; ``flat``: ``plummer`` on
        the CPU's walk (:func:`_walk`).

    Returns
    -------
    Lane
        The configuration.
    """
    if name in _LANES:
        return _LANES[name]
    from jaccpot.runtime.fused_provider import FusedForceProvider

    kernel = "ferrers3" if name == "ferrers3" else "plummer"
    mac_type = "dehnen_error" if name == "dehnen_error" else "dehnen"
    fmm = _solver(kernel, mac_type)
    x, v, m, w = _system()
    provider, topology = FusedForceProvider.prepare(
        fmm,
        x,
        m,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        differentiable=True,
    )
    aux0 = None
    if provider.scale_carry:
        _, f_b = provider.accelerations(topology, force_scale=True)
        aux0 = (f_b, None, None)
    _LANES[name] = Lane(fmm, provider, topology, aux0, x, v, m, w)
    return _LANES[name]


@pytest.fixture(scope="module", autouse=True)
def _drop_lanes() -> Iterator[None]:
    """Release the module's solvers and gradients at its end.

    Yields
    ------
    None
        For the module's tests.
    """
    yield
    _LANES.clear()
    _GRADS.clear()


def _bitwise(got: Any, want: Any) -> int:
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape
    # 1-d: a 0-d array cannot be viewed as bytes
    got, want = got.reshape(-1), want.reshape(-1)
    return int(np.sum(got.view(np.uint8) != want.view(np.uint8)))


# ---------------------------------------------------------------------------
# The forward is ``step``'s.
# ---------------------------------------------------------------------------


@_FORWARD
@pytest.mark.parametrize("name", ["plummer", "ferrers3", "dehnen_error", "flat"])
def test_the_forward_is_steps_bitwise(name):
    """Accelerations and aux at the prepare's positions and two drifts from it.

    The masses are an argument of both programs (closed over, they are a constant
    that XLA compiles differently; ``test_strict_runners_on_provider.py``).
    """
    lane = _lane(name)
    prov = lane.provider
    step = jax.jit(lambda x, m, aux: prov.step(x, m, aux=aux))
    dstep = jax.jit(lambda x, m, aux: prov.differentiable_step(x, m, aux=aux))
    for shift in (0.0, 0.01, 0.03):
        x = lane.positions + shift * lane.velocities
        acc, (f_b, bits, needs) = step(x, lane.masses, lane.aux0)
        d_acc, (d_f_b, d_bits, d_needs) = dstep(x, lane.masses, lane.aux0)
        assert int(bits) == 0, f"shift {shift}: the caps do not fit ({int(bits)})"
        assert _bitwise(d_acc, acc) == 0, f"shift {shift}"
        assert int(d_bits) == int(bits) and _bitwise(d_needs, needs) == 0
        if prov.scale_carry:
            assert np.any(np.asarray(f_b)) and _bitwise(d_f_b, f_b) == 0
        else:
            assert f_b is None and d_f_b is None


@_FORWARD
@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_a_verlet_scan_through_the_aux_is_steps(name):
    """Three velocity-Verlet steps (the production association), the aux chained:
    the state, the force and the folded aux are ``step``'s bitwise."""
    lane = _lane(name)
    prov = lane.provider
    dt = jnp.asarray(DT, jnp.float64)

    def run(force):
        def verlet(carry, m):
            x, v, a, aux = carry
            x = x + v * dt + 0.5 * a * dt**2
            a_new, aux = force(x, m, aux=aux)
            return (x, v + 0.5 * (a + a_new) * dt, a_new, aux), None

        @jax.jit
        def go(x, v, m, aux):
            a0, aux = force(x, m, aux=aux)
            carry = (x, v, a0, aux)
            return jax.lax.scan(lambda c, _: verlet(c, m), carry, None, length=3)[0]

        return go(lane.positions, lane.velocities, lane.masses, lane.aux0)

    want = run(prov.step)
    got = run(prov.differentiable_step)
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        assert _bitwise(g, w) == 0
    assert int(want[3][1]) == 0


# ---------------------------------------------------------------------------
# The gradient.
# ---------------------------------------------------------------------------

#: the reverse kernels of the production layout, counted while the gradient traces
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

_G0 = 1.3


def _loss_fn(lane: Lane) -> Any:
    """``sum(w . acc)`` of one differentiable step, in (x, m, G, eps2).

    Parameters
    ----------
    lane : Lane
        The configuration.

    Returns
    -------
    Any
        The loss; ``G`` and ``eps2`` are ignored under ``dehnen_error`` (its
        parameters are the solver's own only).
    """
    prov = lane.provider

    def loss(x, m, g, eps2):
        params = None if prov.scale_carry else ForceParams(G=g, eps2=eps2)
        acc, _ = prov.differentiable_step(x, m, params, aux=lane.aux0)
        return jnp.sum(lane.weight * acc)

    return loss


_GRADS: dict = {}


def _gradient(name: str) -> tuple[Any, dict]:
    """The jitted gradient of :func:`_loss_fn` at the lane's start, traced once
    with the reverse kernels counted.

    Parameters
    ----------
    name : str
        The lane.

    Returns
    -------
    tuple[Any, dict]
        ``(dx, dm, dG, deps2)`` and the reverse-kernel counts of the trace.
    """
    if name in _GRADS:
        return _GRADS[name]
    import importlib

    lane = _lane(name)
    counts = {key: 0 for *_, key in _REVERSE_KERNELS}
    # counted when they TRACE: an inner jitted function another module's test
    # traced at these shapes in the same worker would come from jax's caches and
    # never call them (seen on s9/integration at -n 8: l2l and l2p counted 0)
    jax.clear_caches()
    patcher = pytest.MonkeyPatch()
    for module_name, fn_name, key in _REVERSE_KERNELS:
        module = importlib.import_module(module_name)
        real = getattr(module, fn_name)

        def counted(*args, _real=real, _key=key, **kwargs):
            counts[_key] += 1
            return _real(*args, **kwargs)

        patcher.setattr(module, fn_name, counted)
    try:
        grads = jax.jit(jax.grad(_loss_fn(lane), argnums=(0, 1, 2, 3)))(
            lane.positions,
            lane.masses,
            jnp.asarray(_G0),
            jnp.asarray(SOFTENING**2),
        )
        jax.block_until_ready(grads)
    finally:
        patcher.undo()
    _GRADS[name] = (grads, counts)
    return _GRADS[name]


@_GRADIENT
@pytest.mark.parametrize("name", ["plummer", "flat"])
def test_the_gradient_is_s8s_on_the_eager_topology(name):
    """At the prepare's positions the rebuilt topology is the eager one, so the
    gradient is ``differentiable_step_fn``'s (S8) on the eager prepared state --
    bitwise, in the positions, the masses, G and eps2."""
    lane = _lane(name)
    (dx, dm, dg, de), _ = _gradient(name)
    seam = lane.fmm.differentiable_step_fn(lane.topology.prepared)

    def loss(x, m, g, eps2):
        return jnp.sum(lane.weight * seam(x, m, ForceParams(G=g, eps2=eps2)))

    want = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3)))(
        lane.positions, lane.masses, jnp.asarray(_G0), jnp.asarray(SOFTENING**2)
    )
    for got, ref in zip((dx, dm, dg, de), want):
        assert np.all(np.isfinite(np.asarray(got)))
        assert _bitwise(got, ref) == 0


def _topology_indices(lane: Lane, x: Any, m: Any) -> list:
    """The integer arrays of the topology a step rebuilds at ``x``.

    Parameters
    ----------
    lane : Lane
        The configuration.
    x : Any
        Positions.
    m : Any
        Masses.

    Returns
    -------
    list
        The tree's permutation, the near CSR and the far list, on the host.
    """
    prov = lane.provider
    f_b = None if lane.aux0 is None else lane.aux0[0]

    @jax.jit
    def indices(x, m):
        prepared = prov.rebuild(x, m, force_scale=f_b, keep_far_list=True).prepared
        nl, far = prepared.neighbor_list, prepared.compact_far_pairs
        return [
            prepared.tree.particle_indices,
            nl.offsets,
            nl.neighbors,
            far.sources,
            far.targets,
            far.far_pair_count,
        ]

    return [np.asarray(a) for a in indices(x, m)]


@_GRADIENT
@pytest.mark.parametrize("name", ["plummer", "dehnen_error"])
def test_the_gradient_matches_finite_differences(name):
    """Central differences of the whole step in float64, rebuild included.

    Directional in the positions and the masses (each +-h run checked to rebuild the
    topology of the centre, so the differences differentiate one map, as #316's
    gates); G (linear: exact to round-off) and eps2 (smooth at fixed lists) where
    the MAC takes traced parameters. S8's tolerances.
    """
    lane = _lane(name)
    (dx, dm, dg, de), _ = _gradient(name)
    value = jax.jit(_loss_fn(lane))
    x0, m0 = lane.positions, lane.masses
    g0, e0 = jnp.asarray(_G0), jnp.asarray(SOFTENING**2)
    centre = _topology_indices(lane, x0, m0)
    rng = np.random.default_rng(7)
    v_x = jnp.asarray(rng.normal(size=(N, 3)))
    v_m = jnp.asarray(rng.normal(size=(N,)))
    # The step sizes are measured (2026-10-31, these directions, both lanes): along
    # v_x the relative error falls as h^2 (truncation: 4.9e-6 at 1e-5, 4.4e-7 at
    # 3e-6, 4.8e-8 at 1e-6) to a round-off floor of ~3e-9 at 1e-7, and h_x >= 1e-5
    # rebuilds another topology under dehnen_error (>= 3e-5 under dehnen); along
    # v_m it is flat (<= 1.3e-9 for 1e-6 <= h <= 1e-4), the lists unchanged.
    h_x, h_m = 1e-6, 1e-5

    def fd(dx_=0.0, dm_=0.0, dg_=0.0, de_=0.0):
        # the walk reads the positions and the masses (COM geometry): both
        # displaced runs must rebuild the centre's topology
        for sign in (1.0, -1.0):
            moved = _topology_indices(lane, x0 + sign * dx_, m0 + sign * dm_)
            assert all(np.array_equal(a, b) for a, b in zip(moved, centre))
        plus = value(x0 + dx_, m0 + dm_, g0 + dg_, e0 + de_)
        minus = value(x0 - dx_, m0 - dm_, g0 - dg_, e0 - de_)
        return (float(plus) - float(minus)) / 2.0

    fd_x, fd_m = fd(dx_=h_x * v_x) / h_x, fd(dm_=h_m * v_m) / h_m
    got_x, got_m = float(jnp.sum(dx * v_x)), float(jnp.sum(dm * v_m))
    assert abs(got_x - fd_x) <= 1e-6 * abs(fd_x), (got_x, fd_x)
    assert abs(got_m - fd_m) <= 1e-6 * abs(fd_m), (got_m, fd_m)
    if lane.provider.scale_carry:
        return
    # eps2 measured as h_x: truncation, 3.8e-6 at 1e-3 eps^2 (S8's step), 3.8e-8 at
    # 1e-4 eps^2, a floor of ~3e-9 at 1e-5 eps^2; G is linear (<= 2.6e-12 at 1e-4)
    h_g, h_e = 1e-4, 1e-4 * SOFTENING**2
    np.testing.assert_allclose(float(dg), fd(dg_=h_g) / h_g, rtol=1e-10)
    np.testing.assert_allclose(float(de), fd(de_=h_e) / h_e, rtol=1e-6)
    assert float(de) != 0.0


@_GRADIENT
def test_the_far_field_is_load_bearing():
    """Far pairs in the traced list, and every reverse kernel of the production
    layout in the gradient's trace -- the far chain's source side included (S8's
    inertness rule: a gradient that skipped the far field would still be finite)."""
    lane = _lane("plummer")
    _, counts = _gradient("plummer")
    far_count = _topology_indices(lane, lane.positions, lane.masses)[-1]
    assert int(far_count) > 0
    assert counts == {key: 1 for key in counts}, counts


@_GRADIENT
def test_the_linearized_program_holds_no_while_loop():
    """The topology stage (the walk's rounds, the extents, the level loops) is
    primal only: the linear program holds no while loop at all, while the primal
    one holds the walk's -- so the check is not vacuous."""
    from jax.extend import core as jcore

    lane = _lane("plummer")
    prov = lane.provider

    def whiles(jaxpr: Any) -> int:
        count = 0
        for eqn in jaxpr.eqns:
            count += eqn.primitive.name == "while"
            for param in eqn.params.values():
                for sub in param if isinstance(param, (list, tuple)) else [param]:
                    if isinstance(sub, jcore.ClosedJaxpr):
                        count += whiles(sub.jaxpr)
                    elif isinstance(sub, jcore.Jaxpr):
                        count += whiles(sub)
        return count

    def force(x, m):
        return prov.differentiable_step(x, m)[0]

    linear: dict = {}

    def primal_and_linear(x, m):
        out, f_jvp = jax.linearize(force, x, m)
        linear["jaxpr"] = jax.make_jaxpr(f_jvp)(x, m)
        return out

    whole = jax.make_jaxpr(primal_and_linear)(lane.positions, lane.masses)
    assert whiles(whole.jaxpr) > 0
    assert whiles(linear["jaxpr"].jaxpr) == 0


# ---------------------------------------------------------------------------
# The verdict and the static fields.
# ---------------------------------------------------------------------------


@_GRADIENT
def test_an_asymmetric_list_refuses_the_gradient(monkeypatch):
    """A near list whose reverse would read pairs it does not hold: the force is
    right (``overflow_ok``), the gradient's verdict fails (``grad_ok``,
    ``OVERFLOW_GRAD_LISTS``) and the gradient is NaN, never a plausible number."""
    from jaccpot.runtime import _large_n_grad

    lane = _lane("plummer")
    prov = lane.provider
    monkeypatch.setattr(
        _large_n_grad,
        "_near_list_symmetric_on_device",
        lambda neighbor_list: jnp.asarray(False),
    )

    def loss(x, m):
        acc, aux = prov.differentiable_step(x, m)
        return jnp.sum(lane.weight * acc), (acc, aux[1])

    (_, (acc, bits)), (dx, dm) = jax.jit(
        jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
    )(lane.positions, lane.masses)
    want, (_, step_bits, _) = jax.jit(prov.step)(lane.positions, lane.masses)
    assert _bitwise(acc, want) == 0
    assert int(bits) == int(step_bits) | OVERFLOW_GRAD_LISTS
    assert bool(overflow_ok(bits)) and not bool(grad_ok(bits))
    assert bool(jnp.all(jnp.isnan(dx))) and bool(jnp.all(jnp.isnan(dm)))


def test_the_device_symmetry_check_tells_a_list_that_is_not():
    """The on-device hash check gives the host check's verdicts (S8's cases)."""
    from jaccpot.runtime._large_n_grad import _csr_pairs_symmetric_on_device

    check = jax.jit(_csr_pairs_symmetric_on_device)
    # rows 0: [1], 1: [0, 2], 2: [1]; 9 is padding nobody reads
    entries = jnp.asarray([1, 9, 0, 2, 9, 1, 9, 9])
    offsets, counts = jnp.asarray([0, 2, 5, 6]), jnp.asarray([1, 2, 1])
    assert bool(check(entries, offsets, counts))
    assert bool(check(entries, offsets[:-1], counts))
    assert not bool(check(entries, offsets, jnp.asarray([2, 2, 1])))
    # a duplicated pair is not its reverse (a multiset test)
    assert not bool(
        check(jnp.asarray([1, 1, 0]), jnp.asarray([0, 2]), jnp.asarray([2, 1]))
    )


@_GRADIENT
def test_the_static_fields_are_checked_on_every_trace():
    """The frozen statics are template properties; a trace whose topology differs
    from them raises instead of differentiating another plan."""
    lane = _lane("plummer")
    prov = lane.provider
    wrong = type(prov)(
        prov.engine,
        leaf_size=prov.leaf_size,
        max_order=prov.max_order,
        theta=prov.theta,
        runtime_overrides=prov.runtime_overrides,
        template=prov.template,
        grad_statics=dataclasses.replace(
            prov.grad_statics, num_far_pairs=prov.grad_statics.num_far_pairs + 2
        ),
    )
    with pytest.raises(RuntimeError, match="grad-plan statics differ"):
        jax.eval_shape(wrong.differentiable_step, lane.positions, lane.masses)


@_GRADIENT
def test_the_api_refuses_what_it_cannot_differentiate():
    """No frozen plan, a caps-only prepare, a topology without its far list."""
    from jaccpot.runtime.fused_provider import FusedForceProvider

    lane = _lane("plummer")
    prov = lane.provider
    bare = FusedForceProvider.freeze_template(
        lane.fmm,
        lane.topology.prepared,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
    )
    assert bare.grad_statics is None
    with pytest.raises(ValueError, match="no gradient plan"):
        bare.differentiable_step(lane.positions, lane.masses)
    with pytest.raises(ValueError, match="caps_only"):
        FusedForceProvider.prepare(
            lane.fmm,
            lane.positions,
            lane.masses,
            leaf_size=LEAF,
            max_order=ORDER,
            theta=THETA,
            caps_only=True,
            differentiable=True,
        )
    without_list = dataclasses.replace(
        lane.topology,
        prepared=dataclasses.replace(lane.topology.prepared, compact_far_pairs=None),
    )
    with pytest.raises(ValueError, match="far list"):
        prov.with_grad_plan(without_list)

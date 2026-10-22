"""Traced force and step parameters of the fused lane (S2, C4), on CPU.

``strict_run_v2`` and the eval seam take ``G`` and the softening squared
(``force_params=ForceParams(G, eps2)``), the timestep and the external field's
parameters as Python numbers -- compile-time literals, the default, every program
as it was -- or as arrays: TRACED arguments of the compiled programs, cast at the
boundary, so one compile serves every value.

* a traced value computes what the literal computes (bitwise here; the GPU A/B is
  ``bench/fused_memory_budget.py --traced-params``);
* the force is linear in G to the last bit, and its derivatives in G and eps2 are
  the finite differences' (float64 force, the pure-JAX and the CSR near field);
* a traced dt and traced external parameters are the closure form's numbers;
* three calls with three different (dt, G, eps2, external) values trace once;
* a softening beyond the one the walk is built for sets ``OVERFLOW_PARAMS``
  (compact kernels: the walk's separation floor is static) and is refused;
* ``mac_type='dehnen_error'`` refuses anything but the solver's own G and softening.

CPU, N = 512, the production leaf partition (cells), with ``jax.default_backend``
reporting ``"gpu"`` as ``tests/characterization/test_lane_goldens.py`` does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

N = 512
LEAF, ORDER, THETA = 16, 4, 0.6
SOFTENING, DT, STEPS = 1e-2, 1e-3, 2
#: the external field's spring constant (a row-by-row harmonic pull)
K_EXT = 0.37

_ENV_UNSET = (
    "JACCPOT_STATIC_STRICT_FUSED_FLAT_COMPACT_FAR_PAIRS",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_IN_FUSED",
    "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET",
    "JACCPOT_LARGE_N_TARGET_BLOCK_SIZE",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS",
    "JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF",
    "JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP",
    "JACCPOT_LARGE_N_NEIGHBOR_EDGE_PROFILE_FIXED_CAP",
    "JACCPOT_STRICT_CARRY",
    "JACCPOT_STRICT_CARRY_ORDER",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR",
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB",
)
#: the CSR near-field lane (production's) on CPU: Pallas interpret mode
_CSR_ON_CPU = {
    "JACCPOT_NEARFIELD_PALLAS_INTERPRET": "1",
    "JACCPOT_NEARFIELD_LEAFPAIR_CSR": "1",
    "JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB": "0",
}


@pytest.fixture(autouse=True)
def _lane(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    for key in _ENV_UNSET:
        monkeypatch.delenv(key, raising=False)


def _solver(
    *,
    use_pallas: bool = False,
    dtype=jnp.float32,
    softening_kernel: str = "plummer",
    mac_type: str = "dehnen",
):
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
        softening_kernel=softening_kernel,
        working_dtype=dtype,
        use_pallas=use_pallas,
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


def _system(dtype=jnp.float32):
    """Four tight blobs (provider_hlo_identity.py's CPU draw), with velocities."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-0.8, 0.8, (4, 3))
    pos = np.clip(
        centres[rng.integers(0, 4, N)] + 0.08 * rng.normal(size=(N, 3)), -1, 1
    )
    vel = 0.1 * rng.normal(size=(N, 3))
    masses = np.abs(rng.normal(size=N)) + 0.5
    state = jnp.asarray(np.stack([pos, vel], axis=1), dtype)
    return state, jnp.asarray(masses, dtype)


def _run(solver, state, masses, *, carry="state", dt=DT, **kw):
    out, _, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=dt,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        carry=carry,
        **kw,
    )
    return np.asarray(out)


def _eval(solver, state, masses, **kw):
    prepared, eval_fn = solver.strict_fused_prepared_eval_fn(
        positions=state[:, 0],
        masses=masses,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        **kw,
    )
    return np.asarray(eval_fn(prepared))


def _traced_force(g=1.0, eps2=SOFTENING**2):
    from jaccpot import ForceParams

    return ForceParams(G=jnp.asarray(g), eps2=jnp.asarray(eps2))


@pytest.mark.parametrize("kernel", ["plummer", "ferrers3"])
@pytest.mark.parametrize("csr", [False, True], ids=["purejax", "csr"])
def test_traced_force_parameters_compute_the_literals_numbers(csr, kernel, monkeypatch):
    """The eval seam with ``G`` and ``eps2`` as float64 arrays (cast to the float32
    force at the boundary) against the default program, the pure-JAX and the CSR
    near field (interpret mode).

    Plummer: bitwise. ``ferrers3`` (the library default): not bitwise on CPU, one
    float32 rounding (CPU, 2026-10-16: 753 / 1536 elements pure-JAX, 732 CSR; at most
    1.35e-7 of the largest component). The lowered programs differ only in the
    constant becoming a parameter; XLA's algebraic simplifier then rewrites the
    kernel's ``s * s * s`` (``s = rsqrt(max(r^2, h^2))``) into ``rsqrt(x) / x`` only
    when it can prove ``x > 0`` -- which a literal ``h^2`` floor gives it and a
    parameter does not."""
    if csr:
        for key, val in _CSR_ON_CPU.items():
            monkeypatch.setenv(key, val)
    state, masses = _system()
    solver = _solver(use_pallas=csr, softening_kernel=kernel)
    ref = _eval(solver, state, masses)
    solver = _solver(use_pallas=csr, softening_kernel=kernel)
    got = _eval(solver, state, masses, force_params=_traced_force())
    if kernel == "plummer":
        assert np.array_equal(got, ref)
    else:
        scale = float(np.max(np.abs(ref)))
        assert float(np.max(np.abs(got - ref))) <= 4e-7 * scale


def test_the_force_is_linear_in_G_to_the_last_bit():
    """One compiled evaluation, two values of G: ``acc(2 G) == 2 acc(G)`` bitwise (a
    power of two scales every pair term and every partial sum exactly)."""
    from jaccpot import FusedForceProvider

    state, masses = _system()
    prov, topo = FusedForceProvider.prepare(
        _solver(), state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    acc = jax.jit(prov.accelerations)
    one = np.asarray(acc(topo, _traced_force(g=1.0)))
    two = np.asarray(acc(topo, _traced_force(g=2.0)))
    assert acc._cache_size() == 1
    assert np.array_equal(two, 2.0 * one)


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the float64 force lane needs x64"
)
@pytest.mark.parametrize("kernel", ["plummer", "ferrers3"])
@pytest.mark.parametrize("csr", [False, True], ids=["purejax", "csr"])
def test_the_derivatives_in_G_and_eps2_are_the_finite_differences(
    csr, kernel, monkeypatch
):
    """``jax.grad`` through the provider's evaluation (float64) in ``G`` and ``eps2``
    against central differences of the same evaluation: the pure-JAX near field, and
    the CSR lane's direct layout (production's, interpret mode), whose forward runs
    under its ``custom_vjp`` when the parameters are arrays (plan S8; before it,
    ``pallas_call`` had no differentiation rule there)."""
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_topology import ForceParams

    if csr:
        for key, val in _CSR_ON_CPU.items():
            monkeypatch.setenv(key, val)
    state, masses = _system(jnp.float64)
    solver = _solver(use_pallas=csr, dtype=jnp.float64, softening_kernel=kernel)
    prov, topo = FusedForceProvider.prepare(
        solver, state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    w = jnp.asarray(np.random.default_rng(3).normal(size=(N, 3)))

    @jax.jit
    def loss(g, eps2):
        acc = prov.accelerations(topo, ForceParams(G=g, eps2=eps2))
        return jnp.sum(w * acc)

    g0, e0 = jnp.asarray(1.3), jnp.asarray(SOFTENING**2)
    d_g, d_e = jax.grad(loss, argnums=(0, 1))(g0, e0)
    h_g, h_e = 1e-4, 1e-3 * SOFTENING**2
    fd_g = (loss(g0 + h_g, e0) - loss(g0 - h_g, e0)) / (2 * h_g)
    fd_e = (loss(g0, e0 + h_e) - loss(g0, e0 - h_e)) / (2 * h_e)
    # linear in G: the difference quotient is exact up to round-off
    np.testing.assert_allclose(float(d_g), float(fd_g), rtol=1e-10)
    # eps2: second-order truncation, (h / eps2)^2 ~ 1e-6
    np.testing.assert_allclose(float(d_e), float(fd_e), rtol=1e-6)
    assert float(d_e) != 0.0


def _harmonic(state_rows):
    """The closure form: the spring constant a literal."""
    return -K_EXT * state_rows[:, 0]


def _harmonic_params(state_rows, params):
    """The parameter form: the spring constant traced."""
    return -params["k"] * state_rows[:, 0]


@pytest.mark.parametrize(
    "carry, order",
    [("state", None), ("particles", None), ("particles", "tree")],
    ids=["state", "particles", "tree"],
)
def test_a_traced_dt_and_external_parameters_compute_the_closure_form(
    carry, order, monkeypatch
):
    """The scan with ``dt``, the external field's parameter and ``G`` / ``eps2`` as
    arrays against the literal ``dt``, the closure field and the solver's own
    force: bitwise on the state and particle carries.

    The tree-order carry: the positions bitwise, two velocity components apart
    (CPU, 2026-10-16: 2 / 3072 elements, 2.4e-7 and 6e-8 -- 1 and 8 ulps of
    their own values, the second a kick that nearly cancels) -- with any ONE traced
    argument (dt, the field's parameter or the Plummer force's), none with the same
    parameters as literals. The parameters' arithmetic is the same; the extra
    argument changes how XLA compiles the program, and with it the rounding of the
    acceleration sum of two particles (about an ulp of the force, below the
    positions' resolution after the drift)."""
    if order is not None:
        monkeypatch.setenv("JACCPOT_STRICT_CARRY_ORDER", order)
    state, masses = _system()
    ref = _run(
        _solver(),
        state,
        masses,
        carry=carry,
        add_external=True,
        external_acceleration_fn=_harmonic,
    )
    got = _run(
        _solver(),
        state,
        masses,
        carry=carry,
        dt=jnp.asarray(DT),
        add_external=True,
        external_acceleration_fn=_harmonic_params,
        external_params={"k": jnp.asarray(K_EXT, jnp.float32)},
        force_params=_traced_force(),
    )
    if order != "tree":
        assert np.array_equal(got, ref)
        return
    assert np.array_equal(got[:, 0], ref[:, 0])
    largest = np.float32(np.max(np.abs(ref[:, 1])))
    assert float(np.max(np.abs(got - ref))) <= float(np.spacing(largest))


@pytest.mark.parametrize("carry", ["state", "particles"])
def test_one_trace_serves_every_parameter_value(carry):
    """Three calls, three different (dt, G, eps2, external) values, one solver:
    one compile key per program, one executable each, and the scan body (where the
    external field is called with tracers) traced once."""
    state, masses = _system()
    solver = _solver()
    traced_calls = []

    def field(state_rows, params):
        if isinstance(state_rows, jax.core.Tracer):
            traced_calls.append(1)
        return _harmonic_params(state_rows, params)

    outs = []
    for i, (dt, g, eps2, k) in enumerate(
        [(1e-3, 1.0, 1e-4, 0.37), (2e-3, 0.5, 4e-4, 0.1), (5e-4, 2.0, 9e-5, 1.0)]
    ):
        outs.append(
            _run(
                solver,
                state,
                masses,
                carry=carry,
                dt=jnp.asarray(dt),
                add_external=True,
                external_acceleration_fn=field,
                external_params={"k": jnp.asarray(k, jnp.float32)},
                force_params=_traced_force(g=g, eps2=eps2),
            )
        )
    cache = solver._impl._strict_fused_jit_function_cache
    tags = sorted(k[0] for k in cache)
    assert tags == (
        ["strict_velocity_verlet"]
        if carry == "state"
        else ["strict_particle_start_force", "strict_velocity_verlet_particles"]
    )
    assert all(fn._cache_size() == 1 for fn in cache.values())
    assert len(traced_calls) == 1
    # the values did change the run
    assert not np.array_equal(outs[0], outs[1])
    assert not np.array_equal(outs[1], outs[2])


def test_a_softening_beyond_the_walks_sets_the_params_bit():
    """Compact kernels: the walk keeps pairs inside the support ``h = k eps`` of the
    solver's softening out of the far field; a larger traced ``eps2`` reaches pairs
    the far field took unsoftened, sets ``OVERFLOW_PARAMS`` and fails the verdict.
    ``strict_run_v2`` refuses it before any work. Plummer has no support."""
    from jaccpot import FusedForceProvider
    from jaccpot.runtime.fused_topology import OVERFLOW_PARAMS, ForceParams, overflow_ok

    state, masses = _system()
    solver = _solver(softening_kernel="ferrers3")
    prov, _ = FusedForceProvider.prepare(
        solver, state[:, 0], masses, leaf_size=LEAF, max_order=ORDER, theta=THETA
    )
    own = SOFTENING**2
    assert prov.walk_eps2() == pytest.approx(own, rel=1e-12)
    bits = jax.jit(prov.params_overflow)
    assert int(bits(_traced_force(eps2=own))) == 0
    assert int(bits(_traced_force(eps2=0.5 * own))) == 0
    assert int(bits(_traced_force(eps2=1.5 * own))) == OVERFLOW_PARAMS
    assert int(prov.params_overflow(ForceParams(G=1.0, eps2=1.5 * own))) == (
        OVERFLOW_PARAMS
    )
    # through a step: the verdict fails on it alone
    _, (_, step_bits, _) = jax.jit(prov.step)(
        state[:, 0], masses, _traced_force(eps2=1.5 * own)
    )
    assert int(step_bits) & OVERFLOW_PARAMS and not bool(overflow_ok(step_bits))
    with pytest.raises(ValueError, match="OVERFLOW_PARAMS"):
        _run(_solver(softening_kernel="ferrers3"), state, masses, **_kw(1.5 * own))
    # Plummer: no support, no floor, no bit
    plummer = FusedForceProvider(_solver(), leaf_size=LEAF, max_order=ORDER)
    assert plummer.walk_eps2() is None
    assert int(plummer.params_overflow(_traced_force(eps2=100 * own))) == 0


def _kw(eps2):
    return {"force_params": _traced_force(eps2=eps2)}


def test_dehnen_error_takes_the_solvers_own_parameters_only():
    """Eq (16a)'s thresholds and the far half of its force scale are built in the
    rebuild with the solver's G and softening: anything else is refused up front."""
    from jaccpot import ForceParams

    state, masses = _system()
    for params in (_traced_force(), ForceParams(G=2.0, eps2=SOFTENING**2)):
        with pytest.raises(ValueError, match="dehnen_error"):
            _run(
                _solver(use_pallas=True, mac_type="dehnen_error"),
                state,
                masses,
                force_params=params,
            )


def test_external_parameters_need_an_external_field():
    state, masses = _system()
    with pytest.raises(ValueError, match="external_params"):
        _run(_solver(), state, masses, external_params={"k": jnp.asarray(1.0)})


def test_a_handle_from_other_force_parameters_is_not_the_next_start_force():
    """The particle carry's handle holds the self-gravity at the returned positions
    under ITS call's G and softening; a call continuing with others evaluates the
    start force again, so the chain equals a call that starts there afresh."""
    state, masses = _system()
    solver = _solver()
    mid, handle, _ = solver.strict_run_v2(
        state=state,
        masses=masses,
        dt=DT,
        num_steps=STEPS,
        refresh_every=1,
        leaf_size=LEAF,
        max_order=ORDER,
        theta=THETA,
        carry="particles",
        force_params=_traced_force(g=1.0),
    )
    assert handle.force_key == (1.0, float(np.float32(SOFTENING**2)))
    two = _traced_force(g=2.0)
    chained = _run(
        solver, mid, masses, carry="particles", prepared_state=handle, force_params=two
    )
    fresh = _run(_solver(), mid, masses, carry="particles", force_params=two)
    assert np.array_equal(chained, fresh)

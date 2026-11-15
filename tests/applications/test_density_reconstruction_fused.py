"""Section 7 on the fused lane: the observation operator the paper now measures.

The fused lane (``FusedForceProvider``: static cell leaves, the flat walk, the S8
gradient kernels) is what sections 4-7 run. These tests check the reconstruction's
operator on it against the same oracles the radix operator answers to, on the CPU
fused route (``RuntimePolicyConfig(cpu_fused_lane=True)``), which runs the lane's
own programs with its pure-JAX kernels:

* the field at the tracers and the gradient of a weighted misfit match an exact
  dense sum over the same combined set, with the solver's own softening kernel;
* the zero-mass tracers are excluded from the leaf blocks the regularisers use,
  and a tracer-only cluster outside the sources (cells with no mass) is finite;
* a regularised fit with per-iteration rebuilds lowers the field residual, and
  the switch log reads the fused lists (fig19's rates exist and are fractions).

All are ``slow``: the fused programs compile for tens of seconds on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.applications.density_reconstruction import make_ground_truth
from jaccpot.applications.density_reconstruction.diagnostics import field_residual
from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
from jaccpot.applications.density_reconstruction.forward import (
    FusedForwardOperator,
    FusedPreparedTopology,
    make_forward_operator,
)
from jaccpot.applications.density_reconstruction.loss import Regularization
from jaccpot.applications.density_reconstruction.parameterize import (
    initial_positions,
    make_parameterization,
)
from jaccpot.applications.density_reconstruction.truth import TruthConfig

pytestmark = pytest.mark.slow

SOFTENING = 1.0e-2


def _truth(n, m, *, seed=0):
    config = TruthConfig(
        num_particles=n,
        num_tracers=m,
        seed=seed,
        softening=SOFTENING,
        generating_order=6,
        generating_theta=0.4,
        generating_leaf_size=16,
    )
    return config, make_ground_truth(config)


def _fused(truth, n, *, tracers=None, order=4, theta=0.6, leaf_size=32):
    return make_forward_operator(
        tracer_positions=truth.tracer_positions if tracers is None else tracers,
        source_mass=truth.source_mass,
        num_sources=n,
        softening=SOFTENING,
        order=order,
        theta=theta,
        leaf_size=leaf_size,
        lane="fused",
    )


def _direct_at_tracers(operator, sources):
    """The exact field of the sources at the tracers, with the solver's kernel."""
    from jaccpot.softening import masked_pair_factors, softening_params_from_sq

    engine = getattr(operator.fmm, "_impl", operator.fmm)
    kernel = engine.softening_kernel
    n = operator.num_sources
    x = operator.combined_positions(sources)
    d = x[None, :n, :] - x[n:, None, :]
    r2 = jnp.sum(d * d, -1)
    params = softening_params_from_sq(kernel, jnp.asarray(SOFTENING**2), x.dtype)
    inv = masked_pair_factors(r2, r2 > 0, params, kernel)[0]
    return jnp.einsum(
        "mn,mnk->mk",
        operator.source_mass * inv,
        d,
        precision=jax.lax.Precision.HIGHEST,
    )


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def test_fused_operator_matches_direct_sum_value_and_gradient():
    """Field and gradient at a frozen fused topology against the exact sum.

    Half the tracers sit in a tight cluster outside the source distribution, so
    the cell tree has leaves (and internal nodes) holding only zero-mass rows:
    their centre of mass is undefined, and they must neither poison the field
    with NaN nor take part in the gradient.
    """
    n, m = 768, 64
    _, truth = _truth(n, m // 2)
    far = np.max(np.abs(truth.source_positions)) * 1.2
    rng = np.random.default_rng(7)
    cluster = np.full(3, far) + 0.01 * rng.normal(size=(m - m // 2, 3))
    tracers = np.concatenate([truth.tracer_positions, cluster])
    # p6: on this extended disc + halo the truncation error is 7.8e-3 at p4 and
    # 1.3e-3 at p6 (theta 0.6), and falls with theta as it should; the
    # tracer-only cluster is exact at either order.
    operator = _fused(truth, n, tracers=tracers, order=6)
    assert isinstance(operator, FusedForwardOperator)

    sources = jnp.asarray(truth.source_positions)
    state = operator.prepare(truth.source_positions)
    assert isinstance(state, FusedPreparedTopology)
    got = np.asarray(operator.evaluate_at_topology(state, sources))
    want = np.asarray(_direct_at_tracers(operator, sources))
    assert np.all(np.isfinite(got)), "zero-mass cells gave a non-finite field"
    rel = _rel(got, want)
    rel_cluster = _rel(got[m // 2 :], want[m // 2 :])
    print(f"fused field vs direct: {rel:.3e} (tracer-only cluster {rel_cluster:.3e})")
    assert rel < 3e-3, rel
    # The cluster's cells hold no mass; it sees the sources only through the
    # far field and the near pairs, and must come out as accurate as the rest.
    assert rel_cluster < 3e-3, rel_cluster

    weights = jnp.asarray(rng.normal(size=(m, 3)))
    grad_fmm = jax.grad(
        lambda s: jnp.sum(operator.evaluate_at_topology(state, s) * weights)
    )(sources)
    grad_ref = jax.grad(lambda s: jnp.sum(_direct_at_tracers(operator, s) * weights))(
        sources
    )
    assert np.all(np.isfinite(np.asarray(grad_fmm)))
    grel = _rel(grad_fmm, grad_ref)
    print(f"fused gradient vs direct: {grel:.3e}")
    # Measured 2.6e-3: the gradient's far field is an expansion of the force's
    # derivative, one order less accurate than the field (the WP-G gates allow
    # the same headroom).
    assert grel < 5e-3, grel

    record = operator.record()
    assert record["lane"] == "fused" and record["mac_type"] == "dehnen"
    assert record["softening_kernel"] == "ferrers3"
    assert record["sharding_mode"] == "single_device_or_parameter_sharding"


def test_fused_leaf_blocks_cover_sources_only():
    """The regularisers' leaf blocks hold every source once and no tracer."""
    n, m = 512, 32
    _, truth = _truth(n, m)
    operator = _fused(truth, n)
    blocks = operator.leaf_blocks(operator.prepare(truth.source_positions))
    members = np.asarray(blocks.indices)[np.asarray(blocks.valid)]
    assert np.all(members < n), "a tracer entered the leaf blocks"
    assert sorted(members.tolist()) == list(range(n))


def test_fused_mass_dependent_mac_refused():
    """The fused lane's production MAC is mass-dependent and must be refused."""
    _, truth = _truth(64, 8)
    with pytest.raises(ValueError, match="mass-dependent"):
        make_forward_operator(
            tracer_positions=truth.tracer_positions,
            source_mass=truth.source_mass,
            num_sources=64,
            softening=SOFTENING,
            mac_type="dehnen_error",
            lane="fused",
        )


def test_fused_fit_lowers_field_residual_through_rebuilds():
    """A regularised fit on the fused lane, rebuilt every iteration.

    The rebuilt path is the subject, so ``rebuild_cadence = 1``, and switch
    tracking is on: the log reads the fused lane's padded lists, and its rates
    must come out as fractions.
    """
    n, m = 256, 96
    config, truth = _truth(n, m)
    operator = _fused(truth, n, leaf_size=16)
    parameterization = make_parameterization("positions", config=config)
    start = initial_positions(
        truth.source_positions, mode="perturbed_truth", seed=3, perturbation=0.05
    )
    before = field_residual(operator.evaluate(start), truth.observed_accelerations)
    result = run_fit(
        operator=operator,
        observed=truth.observed_accelerations,
        parameterization=parameterization,
        initial_params=parameterization.pack(start),
        config=FitConfig(
            num_iterations=6,
            learning_rate=2.0e-3,
            rebuild_cadence=1,
            regularization=Regularization(),
            history_every=1,
            near_set_sample=32,
        ),
    )
    after = field_residual(
        operator.evaluate(result.positions), truth.observed_accelerations
    )
    print(f"fused fit: rel_l2 {before['rel_l2']:.4e} -> {after['rel_l2']:.4e}")
    assert np.isfinite(result.final_loss)
    assert after["rel_l2"] < before["rel_l2"]
    assert result.final_loss < result.initial_loss
    assert result.config["lane"] == "fused"
    summary = result.switch_summary
    assert summary is not None and summary["extensive"]["observations"] == 6
    mean = summary["intensive"]["mean"]
    for key in ("slot_churn", "leaf_churn", "near_pair_churn", "near_set_churn"):
        assert mean[key] is None or 0.0 <= mean[key] <= 1.0, (key, mean[key])

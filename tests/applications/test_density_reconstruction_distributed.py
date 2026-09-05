"""The distributed observation operator, on a forced-CPU mesh.

Section 7's secondary claim needs the force evaluation itself distributed, not
just the parameter array -- parameter sharding was measured to move neither the
wall-clock nor the ceiling. So this path carries the claim, and it is checked
here against the strongest oracle available rather than only for absence of
crashes.

Two host devices via ``--xla_force_host_platform_device_count``, which is how
``tests/distributed`` exercises this machinery without GPUs, so the claim stays
testable in CI.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = textwrap.dedent("""
    import numpy as np
    import jax
    import jax.numpy as jnp

    from jaccpot.autodiff import direct_sum_gravitational_acceleration
    from jaccpot.applications.density_reconstruction import make_ground_truth
    from jaccpot.applications.density_reconstruction.distributed import (
        make_distributed_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import TruthConfig

    assert len(jax.devices()) == 2, f"expected 2 devices, got {len(jax.devices())}"

    N, M, SOFT = 256, 32, 1.0e-2
    config = TruthConfig(
        num_particles=N, num_tracers=M, seed=0, softening=SOFT,
        generating_order=6, generating_theta=0.4, generating_leaf_size=16,
    )
    truth = make_ground_truth(config)
    operator = make_distributed_forward_operator(
        tracer_positions=truth.tracer_positions,
        source_mass=truth.source_mass,
        num_sources=N,
        num_devices=2,
        softening=SOFT,
        order=4,
        theta=0.5,
        leaf_size=16,
    )
    assert operator.record()["sharding_mode"] == "distributed_force_evaluation"

    # 1. The operator is the field of the sources at the tracers -- checked
    #    against the exact O(N^2) sum over the same combined set, which is the
    #    same oracle test 1 uses for the radix operator. This is the check that
    #    the frozen readout permutation is the RIGHT permutation: reusing the
    #    input gid instead would compare each particle against a Morton
    #    neighbour's force, which is smooth, plausible and wrong by tens of
    #    percent (partition_for_devices' own docstring, with a pinned test).
    got = np.asarray(operator.evaluate(truth.source_positions))
    combined = np.concatenate([truth.source_positions, truth.tracer_positions])
    masses = np.concatenate([truth.masses, np.zeros(M)])
    reference = np.asarray(
        direct_sum_gravitational_acceleration(
            jnp.asarray(combined), jnp.asarray(masses), softening=SOFT
        )
    )[N:]
    rel = np.linalg.norm(got - reference) / np.linalg.norm(reference)
    assert got.shape == (M, 3), got.shape
    assert rel < 1e-5, f"distributed operator vs direct sum rel L2 {rel:.3e}"
    print(f"operator vs direct sum: {rel:.3e}")

    # 2. A fit through it converges.
    parameterization = make_parameterization("positions", config=config)
    start = initial_positions(
        truth.source_positions, mode="perturbed_truth", seed=3, perturbation=0.03
    )
    result = run_fit(
        operator=operator,
        observed=truth.observed_accelerations,
        parameterization=parameterization,
        initial_params=parameterization.pack(start),
        config=FitConfig(
            num_iterations=5,
            learning_rate=2.0e-3,
            rebuild_cadence=1,
            regularization=Regularization.none(),
            history_every=1,
            track_switches=False,
        ),
    )
    assert result.final_loss < result.initial_loss, (
        f"{result.initial_loss:.4e} -> {result.final_loss:.4e}"
    )
    assert np.isfinite(result.final_loss)
    print(f"fit: {result.initial_loss:.4e} -> {result.final_loss:.4e}")

    # 3. Asking for regularisation must RAISE, not silently drop it. fig17(d)
    #    exists to show what an unregularised fit does, so a fit that became
    #    unregularised by accident is the one result this section must not
    #    produce.
    try:
        run_fit(
            operator=operator,
            observed=truth.observed_accelerations,
            parameterization=parameterization,
            initial_params=parameterization.pack(start),
            config=FitConfig(num_iterations=1, regularization=Regularization()),
        )
    except ValueError as exc:
        assert "leaf partition" in str(exc), str(exc)
        print("regularisation refusal OK")
    else:
        raise AssertionError("a regularised distributed fit was allowed")

    print("DISTRIBUTED OK")
    """)


@pytest.mark.slow
def test_distributed_operator_matches_direct_sum_and_fits(tmp_path):
    """The distributed operator reproduces the direct sum and drives a fit."""
    script = tmp_path / "distributed_check.py"
    script.write_text(SCRIPT)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        env={
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            # See test_density_reconstruction_sharding for why this is required.
            "PYTHONPATH": REPO_ROOT,
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
    )
    assert completed.returncode == 0, (
        f"exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )
    assert "DISTRIBUTED OK" in completed.stdout, completed.stdout[-2000:]

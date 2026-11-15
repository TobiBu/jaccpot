"""The distributed observation operator on the fused mesh lane, on a forced-CPU mesh.

Section 7's secondary claim needs the force evaluation itself distributed, not
just the parameter array -- parameter sharding was measured to move neither the
wall-clock nor the ceiling. So this path carries the claim, and it is checked
here against the strongest oracles available rather than only for absence of
crashes: an exact direct sum with the solver's own softening kernel, and the
single-card fused operator in the same configuration.

Two host devices via ``--xla_force_host_platform_device_count``, with the S9
interpret set -- the mesh gradient's only CPU route, the one
``tests/distributed/test_mesh_grad_gates.py`` and ``bench/multigpu_grad_gate.py
--cpu-smoke`` use: the lane's Pallas kernels and the GPU's walk interpreted,
``jax.default_backend`` reporting ``"gpu"`` and the ragged exchange on its
``buf`` method. The interpret-mode programs compile for minutes, so the fit is one
jitted epoch.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT = textwrap.dedent("""
    import os

    # The S9 interpret set (tests/distributed/test_mesh_grad_gates.py).
    for key in (
        "JACCPOT_STATIC_STRICT_FUSED_M2L_CSR",
        "JACCPOT_STATIC_STRICT_FUSED_WALK",
        "JACCPOT_CROSS_WALK",
    ):
        os.environ.pop(key, None)
    os.environ.update({
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
    })

    import jax
    import yggdrax.distributed.comm as ygg_comm

    jax.default_backend = lambda: "gpu"
    _resolve = ygg_comm.resolve_ragged_method
    ygg_comm.resolve_ragged_method = (
        lambda method="auto": "buf" if method == "auto" else _resolve(method)
    )

    import numpy as np
    import jax.numpy as jnp

    from jaccpot.autodiff import direct_sum_gravitational_acceleration
    from jaccpot.applications.density_reconstruction import make_ground_truth
    from jaccpot.applications.density_reconstruction.distributed import (
        make_distributed_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.forward import (
        make_forward_operator,
    )
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import TruthConfig

    assert len(jax.devices()) == 2, f"expected 2 devices, got {len(jax.devices())}"

    N, M, SOFT = 256, 32, 1.0e-2
    ORDER, THETA, LEAF = 4, 0.5, 16
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
        order=ORDER,
        theta=THETA,
        leaf_size=LEAF,
    )
    record = operator.record()
    assert record["sharding_mode"] == "distributed_force_evaluation"
    assert record["lane"] == "fused_mesh" and record["mac_type"] == "dehnen"

    # 1. The operator is the field of the sources at the tracers -- checked
    #    against the exact O(N^2) sum over the same combined set with the
    #    solver's own softening kernel, and against the single-card fused
    #    operator at the same settings. The second is the check that the frozen
    #    readout is the RIGHT readout and that no cross-domain term was dropped:
    #    a wrong permutation or a missing cross term is an O(1) error, while the
    #    two lanes' truncation errors agree.
    got = np.asarray(operator.evaluate(truth.source_positions))
    combined = np.concatenate([truth.source_positions, truth.tracer_positions])
    masses = np.concatenate([truth.masses, np.zeros(M)])
    reference = np.asarray(
        direct_sum_gravitational_acceleration(
            jnp.asarray(combined), jnp.asarray(masses), softening=SOFT,
            softening_kernel=record["softening_kernel"],
        )
    )[N:]

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert got.shape == (M, 3), got.shape
    mesh_err = rel(got, reference)
    single = make_forward_operator(
        tracer_positions=truth.tracer_positions,
        source_mass=truth.source_mass,
        num_sources=N,
        softening=SOFT,
        order=ORDER,
        theta=THETA,
        leaf_size=LEAF,
        lane="fused",
    )
    single_got = np.asarray(single.evaluate(truth.source_positions))
    single_err = rel(single_got, reference)
    print(
        f"vs direct sum: mesh {mesh_err:.3e}, single card {single_err:.3e}; "
        f"mesh vs single card {rel(got, single_got):.3e}"
    )
    # FMM-class at p4 / theta 0.5 on this extended truth (measured 1.8e-3).
    assert mesh_err < 5e-3, mesh_err
    # One-sided on purpose: a dropped cross-domain term or a wrong readout makes
    # the mesh WORSE than the single card, by O(1). It may come out better --
    # measured 1.8e-3 against the single card's 6.4e-3 here -- because each
    # device cuts its own tree in its own smaller box, so at this N more of the
    # problem goes to the exact near field.
    assert mesh_err <= 1.5 * single_err, (mesh_err, single_err)

    # 2. A fit through it converges: one jitted epoch (each re-placement
    #    prepares a new mesh force, which compiles for minutes in interpret mode).
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
            rebuild_cadence=5,
            jit_within_epoch=True,
            regularization=Regularization.none(),
            history_every=1,
            track_switches=False,
        ),
    )
    assert result.final_loss < result.initial_loss, (
        f"{result.initial_loss:.4e} -> {result.final_loss:.4e}"
    )
    assert np.isfinite(result.final_loss)
    assert result.config["lane"] == "fused_mesh"
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

    # 4. The September path's options are refused, not silently remapped.
    for bad in ({"partitioner": "rcb"}, {"caps": {"max_pair_queue": 1 << 20}}):
        try:
            make_distributed_forward_operator(
                tracer_positions=truth.tracer_positions,
                source_mass=truth.source_mass,
                num_sources=N,
                num_devices=2,
                softening=SOFT,
                **bad,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")

    print("DISTRIBUTED OK")
    """)


@pytest.mark.slow
def test_distributed_operator_matches_direct_sum_and_fits(tmp_path):
    """The mesh operator reproduces the direct sum and the single card, and fits."""
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

"""The multi-device parameter-sharding path, exercised without a GPU.

This file exists because the path it tests was broken and nobody could have
noticed. ``run_fit`` only shards when handed more than one device, so every
single-device run -- which is every test and every CPU smoke -- skipped the
branch entirely. It called ``jax.sharding.PositionalSharding``, which does not
exist in jax 0.10.2, and the first execution was fig20 on four GPUs, which
failed at every device count with an ``AttributeError`` after the fit had
already been set up.

``--xla_force_host_platform_device_count`` gives JAX several *host* devices, so
the sharding logic runs for real on CPU. That makes the secondary claim of
section 7 -- that the same optimisation runs sharded, so parameter count is
bounded by aggregate device memory -- testable in CI rather than only on a
machine with eight free cards.
"""

import os
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Four forced host devices. Set before JAX initialises its backend, which is
#: why this runs in a subprocess rather than in the parent test process.
FORCED_DEVICES = 4

SCRIPT = textwrap.dedent("""
    import numpy as np
    import jax

    from jaccpot.applications.density_reconstruction import (
        make_forward_operator,
        make_ground_truth,
    )
    from jaccpot.applications.density_reconstruction.fit import FitConfig, run_fit
    from jaccpot.applications.density_reconstruction.loss import Regularization
    from jaccpot.applications.density_reconstruction.parameterize import (
        initial_positions,
        make_parameterization,
    )
    from jaccpot.applications.density_reconstruction.truth import TruthConfig

    devices = jax.devices()
    assert len(devices) == {devices}, f"expected {devices} devices, got {{len(devices)}}"

    config = TruthConfig(
        num_particles=128,
        num_tracers=32,
        seed=0,
        softening=1.0e-2,
        generating_order=4,
        generating_theta=0.5,
        generating_leaf_size=16,
    )
    truth = make_ground_truth(config)
    operator = make_forward_operator(
        tracer_positions=truth.tracer_positions,
        source_mass=truth.source_mass,
        num_sources=128,
        softening=1.0e-2,
        order=4,
        theta=0.5,
        leaf_size=16,
    )

    for kind in ("positions", "parametric"):
        parameterization = make_parameterization(kind, config=config)
        if kind == "positions":
            start = parameterization.pack(
                initial_positions(
                    truth.source_positions,
                    mode="perturbed_truth",
                    seed=3,
                    perturbation=0.02,
                )
            )
        else:
            start = parameterization.pack(parameterization.true_params(config))

        result = run_fit(
            operator=operator,
            observed=truth.observed_accelerations,
            parameterization=parameterization,
            initial_params=start,
            config=FitConfig(
                num_iterations=3,
                learning_rate=1.0e-3,
                rebuild_cadence=3,
                regularization=Regularization.none(),
                history_every=1,
                near_set_sample=0,
            ),
            devices=devices,
        )

        assert result.timing["num_devices"] == {devices}
        assert len(result.history) == 3
        assert np.isfinite(result.final_loss)
        assert result.positions.shape == (128, 3)

        # The point of the test: the leaves really are distributed. A (128, 3)
        # position array over 4 devices must be split on its leading axis; the
        # parametric model's 0-d scalars have no axis to split and must be
        # replicated instead of raising.
        leaves = jax.tree_util.tree_leaves(result.params)
        assert leaves, "no parameter leaves"
        for leaf in leaves:
            shards = leaf.sharding.num_devices
            assert shards == {devices}, (
                f"{{kind}} leaf {{leaf.shape}} sits on {{shards}} devices"
            )
            if leaf.ndim >= 1 and leaf.shape[0] % {devices} == 0:
                sizes = {{s.data.shape[0] for s in leaf.addressable_shards}}
                assert sizes == {{leaf.shape[0] // {devices}}}, (
                    f"{{kind}} leaf {{leaf.shape}} not split evenly: {{sizes}}"
                )
        print(f"{{kind}} OK")

    print("SHARDING OK")
    """).format(devices=FORCED_DEVICES)


@pytest.mark.slow
def test_parameter_sharding_on_forced_host_devices(tmp_path):
    """``run_fit`` shards its parameters across several devices and converges."""
    script = tmp_path / "shard_check.py"
    script.write_text(SCRIPT)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        env={
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={FORCED_DEVICES}",
            # Required: the script lives in tmp_path, so its directory -- not
            # the repo -- is sys.path[0], and the editable install resolves
            # `jaccpot` to whichever checkout it was installed from rather than
            # to this worktree. Without this the subprocess imports a jaccpot
            # that has no `applications` subpackage at all.
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
    assert "SHARDING OK" in completed.stdout, completed.stdout[-2000:]

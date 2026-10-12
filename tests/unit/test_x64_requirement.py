"""jaccpot requires ``jax_enable_x64`` and checks for it (README, 2026-10).

yggdrax switched x64 on when it was imported until 2026-10, so jaccpot programs ran
with it whether they asked or not. With x64 off, JAX narrows the 63-bit ``uint64``
Morton codes to 32 bits with at most a warning and the trees, and so the forces,
come out wrong with no error (rel-L2 0.19 on the float32 lanes). The public entry
points now refuse to run without it instead.
"""

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot import BlockStepFMM, DistributedBlockStepFMM, FastMultipoleMethod
from jaccpot.distributed import (
    DistributedFMMConfig,
    distributed_fmm_accelerations,
    make_force_evaluator,
    partition_for_devices,
)
from jaccpot.distributed.rollout import FusedRollout, RolloutConfig, ShardState
from jaccpot.runtime._fmm_impl import FMMEngine
from jaccpot.runtime.dtypes import require_x64


@pytest.fixture
def x64_off():
    """Run the test with x64 off, then restore the suite's setting."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _particles(n: int = 16):
    rng = np.random.default_rng(0)
    return (
        rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32),
        np.ones((n,), np.float32),
    )


def _fused_rollout():
    """``FusedRollout`` with arguments that pass the runtime type check (CI runs one)."""
    f32 = jnp.zeros((8, 3), jnp.float32)
    state = ShardState(
        positions=f32,
        velocities=f32,
        masses=jnp.ones((8,), jnp.float32),
        ids=jnp.arange(8, dtype=jnp.int32),
        count=jnp.asarray([8], jnp.int32),
        accel=f32,
    )
    return FusedRollout(None, state, RolloutConfig(cap=8, dt=1e-3), lambda *a: ())


_ENTRY_POINTS = {
    "FastMultipoleMethod": lambda: FastMultipoleMethod(),
    "FMMEngine": lambda: FMMEngine(),
    "BlockStepFMM": lambda: BlockStepFMM(softening=1e-2, k_max=2),
    "DistributedBlockStepFMM": lambda: DistributedBlockStepFMM(softening=1e-2, k_max=2),
    "FusedRollout": _fused_rollout,
    "partition_for_devices": lambda: partition_for_devices(
        *_particles(), 1, leaf_size=4
    ),
    "make_force_evaluator": lambda: make_force_evaluator(
        DistributedFMMConfig(), 1, 16, None
    ),
    "distributed_fmm_accelerations": lambda: distributed_fmm_accelerations(
        *_particles()
    ),
}


@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_entry_points_refuse_to_run_without_x64(x64_off, entry):
    """Each public entry point names the flag before it builds anything."""
    with pytest.raises(RuntimeError, match="jax_enable_x64"):
        _ENTRY_POINTS[entry]()


def test_the_message_says_x64_does_not_pick_the_precision(x64_off):
    with pytest.raises(RuntimeError, match="does not switch jaccpot to float64"):
        require_x64("FastMultipoleMethod")


def test_float32_inputs_stay_float32_with_x64_on():
    """x64 makes 64-bit types available; the input dtype still sets the precision."""
    assert jax.config.jax_enable_x64
    pos, mass = (jnp.asarray(x) for x in _particles(64))
    acc = FastMultipoleMethod(working_dtype=jnp.float32).compute_accelerations(
        pos, mass, leaf_size=8, max_order=2
    )
    assert acc.dtype == jnp.float32


def test_importing_jaccpot_leaves_jax_enable_x64_alone():
    """A fresh process keeps x64 off once yggdrax no longer turns it on."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    env["JAX_PLATFORMS"] = "cpu"
    code = (
        "import jax, yggdrax; a = jax.config.jax_enable_x64; "
        "import jaccpot; print(a, jax.config.jax_enable_x64)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    after_yggdrax, after_jaccpot = out.stdout.strip().splitlines()[-1].split()
    if after_yggdrax == "True":
        pytest.skip("this yggdrax still switches x64 on at import")
    assert after_jaccpot == "False"

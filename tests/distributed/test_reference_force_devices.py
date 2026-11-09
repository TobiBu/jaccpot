"""The A4 reference force is the same bits on 1, 2 and 4 devices (forced CPU devices).

The same particles in a device-count-dependent row order (shuffled, padded) on
meshes of 1, 2 and (with four devices) 4 devices: the reference at a fixed target
set is bitwise identical, for a compact kernel and Plummer. The multi-GPU runner
(``DirectMeshForce``, an fp64 direct sum) scored by ``score_mesh_result`` reads
~round-off. N processes x 1 device: ``test_multiprocess_reference.py``.

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        JAX_ENABLE_X64=1 pytest tests/distributed/test_reference_force_devices.py -q
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from jaccpot.verification import ForceLaw, reference_force, select_targets
from tests.distributed import _reference_scenarios as sc

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="the cross-device gate needs >= 2 devices"
)


def _on(ndev: int, law: ForceLaw):
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.asarray(jax.devices()[:ndev]), (sc.AXIS,))
    sh = NamedSharding(mesh, P(sc.AXIS))
    xs, ms, ids = sc.layout(ndev)
    put = [jax.device_put(a, sh) for a in (xs, ms, ids)]
    return reference_force(*put, select_targets(sc.N, sc.K, seed=2), law, mesh=mesh)


@pytest.mark.parametrize("kernel", ["ferrers3", "plummer"])
def test_the_reference_is_bitwise_on_any_device_count(kernel: str) -> None:
    law = ForceLaw(1.0, 0.03, kernel)
    counts = [c for c in (1, 2, 4) if c <= jax.device_count()]
    refs = [_on(c, law) for c in counts]
    for c, r in zip(counts[1:], refs[1:]):
        for name in ("acceleration", "potential", "positions"):
            got, want = getattr(r, name), getattr(refs[0], name)
            assert np.array_equal(got.view(np.int64), want.view(np.int64)), (c, name)
        assert r.num_sources == sc.N


def test_the_mesh_runner_scores_at_round_off_against_its_direct_force() -> None:
    out = sc.reference(2) if jax.device_count() == 2 else None
    if out is None:
        pytest.skip("the scenario's global mesh takes every device: run with exactly 2")
    s = out["runner_score"]
    assert s["num_targets"] == sc.K and s["nonfinite"] == 0 and s["devices"] == 2
    assert s["rel_l2"] < 1e-13 and s["max"] < 1e-12, s

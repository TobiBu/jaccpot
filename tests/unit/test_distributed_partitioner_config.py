"""``DistributedFMMConfig`` must be able to name the decomposition it wants.

#212 added an RCB partitioner alongside the Morton one and left the default
alone; #215 flipped the default to RCB, on the strength of a win in every
geometry measured. Neither touched ``DistributedFMMConfig``, and
``distributed_fmm_accelerations`` called ``partition_for_devices`` with no
argument -- so the FMM lane's decomposition moved with the flip while having no
knob to move it back, which sits badly against #215's own rationale that flipping
a default deserves a deliberate step *because* it moves every baseline.
``DistributedMutualConfig`` already carried the field; this is the other lane
catching up.

Device-free on purpose. The end-to-end statement lives in
``tests/distributed/test_distributed_fmm_partitioner.py``, but every file in that
tier skips below two devices, so on a single-device CI runner it is not a guard at
all -- and "the field is silently ignored" is precisely the failure these two
tests have to catch, because an ignored ``partitioner`` returns a perfectly good
force from the wrong decomposition and looks like success.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("yggdrax")

from jaccpot.distributed import DistributedFMMConfig  # noqa: E402
from jaccpot.distributed import fmm as _fmm  # noqa: E402


def test_both_distributed_lanes_default_to_the_shared_partitioner():
    """One decomposition default across the two lanes and the partitioner itself.

    Read out of the signature rather than restated as a literal, so that flipping
    the shared default again cannot leave a lane behind without failing here.
    """
    from jaccpot.mutual.distributed import DistributedMutualConfig

    shared = (
        inspect.signature(_fmm.partition_for_devices).parameters["partitioner"].default
    )
    assert DistributedFMMConfig().partitioner == shared
    assert DistributedMutualConfig().partitioner == shared


class _Stop(Exception):
    """Raised by the spy to end the call once the argument has been seen."""


@pytest.mark.parametrize("partitioner", ["morton", "rcb"])
def test_the_configured_partitioner_reaches_the_partitioner(monkeypatch, partitioner):
    """``config.partitioner`` is what ``partition_for_devices`` is called with.

    The driver is stopped at the partition call rather than run to completion:
    the decomposition is chosen there and nowhere else, and stopping keeps this a
    single-device test.
    """
    seen = {}

    def _spy(positions, masses, ndev, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(_fmm, "partition_for_devices", _spy)
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(32, 3)).astype(np.float32)
    mass = np.ones(32, dtype=np.float32)

    with pytest.raises(_Stop):
        _fmm.distributed_fmm_accelerations(
            pos,
            mass,
            config=DistributedFMMConfig(partitioner=partitioner),
            ndev=1,
        )
    assert seen["partitioner"] == partitioner

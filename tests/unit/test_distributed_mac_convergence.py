"""The distributed path's shipped basis/MAC must match what the paper claims.

Phase 0 of ``PROJECT_PLAN.md`` (in the ``jaccpot-paper-i`` repo, not this one)
asked which basis and MAC the distributed driver
runs, because the multi-GPU numbers in the manuscript describe either the
fast-lane path or a slower interim baseline, and the text has to say which. That
question is settled: items 5a-5d landed, and the default is the fast-lane
config, real basis + ``dehnen`` MAC, measured at 0.19% vs direct on 4 devices.
See ``docs/phase5_multigpu_pallas_foldin_plan.md``'s STATUS block and
``MULTIGPU_STATE.md``.

What is left is keeping it that way. The failure mode this guards is silent
drift: someone changes a default for a good local reason, every distributed test
still passes because they all pass an explicit config or simply track the new
default, and the manuscript goes on describing a configuration the code no
longer runs. Nothing else in the suite compares the shipped default against the
written claim.

Accuracy against direct summation is deliberately *not* re-tested here. It is
already covered, on real devices and in the right place, by
``tests/distributed/test_distributed_fmm_driver.py`` --
``test_driver_real_basis_matches_direct`` for the shipped default,
``test_driver_solidfmm_matches_direct`` for the legacy path, and
``test_driver_jit_matches_eager``. Those files skip below 2 devices, which is
correct for them and is why this file holds only the device-free part.
"""

from __future__ import annotations

import pytest

from jaccpot.distributed import DistributedFMMConfig

# The configuration the manuscript describes. Changing a value here is a
# deliberate act: it means the paper text needs changing too, and the message
# below says where.
DOCUMENTED_BASIS = "real"
DOCUMENTED_MAC = "dehnen"

_WHERE_TO_UPDATE = """
If this default changed on purpose, update all of these in the same change:
  - docs/phase5_multigpu_pallas_foldin_plan.md   (STATUS block)
  - MULTIGPU_STATE.md                            (section 1)
  - the paper repo's PROJECT_PLAN.md              (Phase 0)
  - the paper repo: sections/02_method.tex, and any accuracy figure quoting
    the distributed configuration
Otherwise this is drift, and the manuscript is now describing a configuration
the code does not run.
""".strip()


def test_distributed_config_matches_documented_status() -> None:
    """The shipped default basis/MAC is the one the paper describes.

    Runs without devices: it inspects the default configuration rather than
    evaluating a force, so it holds on CPU CI where the distributed suite skips.
    """

    config = DistributedFMMConfig()
    assert config.basis == DOCUMENTED_BASIS, (
        f"distributed default basis is {config.basis!r}, but the paper "
        f"describes {DOCUMENTED_BASIS!r}.\n{_WHERE_TO_UPDATE}"
    )
    assert config.mac_type == DOCUMENTED_MAC, (
        f"distributed default mac_type is {config.mac_type!r}, but the paper "
        f"describes {DOCUMENTED_MAC!r}.\n{_WHERE_TO_UPDATE}"
    )


@pytest.mark.skip(
    reason=(
        "Redundant: accuracy vs direct summation on the shipped default is "
        "covered by tests/distributed/test_distributed_fmm_driver.py::"
        "test_driver_real_basis_matches_direct, which runs on real devices. "
        "Kept as a placeholder rather than removed so the collected count is "
        "stable; it is a deletion candidate, not work waiting to be done."
    )
)
def test_distributed_matches_direct_within_tolerance() -> None:
    """Superseded. See the skip reason and this module's docstring."""

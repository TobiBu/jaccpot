"""``DistributedFMMConfig``'s traversal capacities must come from per-device N.

The distributed config shipped five bare constants -- ``process_block=64``,
``max_pair_queue=32768``, ``max_interactions_per_node=512``,
``max_neighbors_per_leaf=128``, ``leaf_size=8`` -- with no N dependence at all,
while the single-GPU lane had already been through this and cured it
(``runtime/fmm_constants._sub_million_minimum_memory_pair_queue``, whose
docstring carries the measurement: "N=131072 fits, N=262144 raises 'Pair queue
capacity exceeded'", a hard ceiling from a constant in the middle of the measured
range).

Two of those constants cannot be constants at all. Measured on a thin disc with
the dehnen MAC at theta=0.4, over per-device N from 2048 to 131072 and leaf sizes
8 to 256 (``bench/distributed_ceiling_sweep.py --sweep occupancy``):

    leaves    max far per node    max near per leaf
        32                   0                   31
       128                  21                  127
       512                 127                  511
      2048                 496                 2047

``max_near_per_leaf`` is ``num_leaves - 1`` at *every* point: on a radix tree over
a disc, at least one leaf has a bounding sphere wide enough to fail the MAC
against every other leaf, so the near list of the worst leaf is complete. A
constant ``128`` therefore truncates the moment a device holds more than 128
leaves -- 1024 particles at the shipped ``leaf_size=8``. ``max_far_per_node``
tracks ``num_leaves / 4`` across two decades and both leaf sizes.

These tests pin the derivation, not the numbers it happens to produce at one
size: N-monotonicity, the power-of-two ladder (each distinct capacity is a
distinct compiled shape, so a ladder bounds recompiles across an N sweep to
log2(N)), the floors that keep small problems where they were, and that an
explicitly-set capacity is never overridden.

Device-free: this is a sizing rule, and the end-to-end statement that a
default-constructed config runs 10^5 particles/device belongs to the distributed
tier, which skips below two devices.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("yggdrax")

from jaccpot.distributed.fmm import (  # noqa: E402
    DERIVED_CAP_FIELDS,
    DistributedFMMConfig,
)
from jaccpot.runtime.fmm_constants import (  # noqa: E402
    _sub_million_minimum_memory_pair_queue,
)

#: Per-device particle counts spanning the tier's tests to the track's target.
_SIZES = (256, 1024, 8192, 32768, 131072, 1_048_576)

#: Mesh size the self-walk numbers below were measured at. Only the CROSS caps
#: depend on it, and they depend on it strongly: at ndev 4 the ndev-blind sizing
#: overflowed ``cross_near`` at every rung with a 61 % force error.
_NDEV = 2


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def test_the_shipped_default_leaves_every_derived_capacity_unset():
    """The sentinel is the default, so nobody has to opt in to correct sizing.

    A capacity that is only right when the caller asks for it is a capacity that
    is usually wrong.
    """
    config = DistributedFMMConfig()
    for field in DERIVED_CAP_FIELDS:
        assert getattr(config, field) is None, (
            f"{field} still ships a bare constant; the derivation below cannot "
            "reach it"
        )


@pytest.mark.parametrize("per_device_n", _SIZES)
def test_resolution_fills_every_derived_capacity_with_a_positive_int(per_device_n):
    """After resolution nothing is left for a downstream ``int()`` to trip over."""
    resolved = DistributedFMMConfig().resolved_for(per_device_n, _NDEV)
    for field in DERIVED_CAP_FIELDS:
        value = getattr(resolved, field)
        assert isinstance(value, int) and value > 0, f"{field} resolved to {value!r}"


def test_the_neighbour_cap_covers_the_worst_leaf_at_every_size():
    """The measured requirement is ``num_leaves``, so the rule must reach it.

    This is the one that the shipped ``128`` failed: it is the difference between
    a near field and a truncated near field, and the truncation reads as a
    speedup.
    """
    for per_device_n in _SIZES:
        config = DistributedFMMConfig(leaf_size=64)
        resolved = config.resolved_for(per_device_n, _NDEV)
        num_leaves = max(1, -(-per_device_n // 64))
        assert resolved.max_neighbors_per_leaf >= num_leaves, (
            f"at {per_device_n}/device and leaf 64 the worst leaf needs "
            f"{num_leaves} neighbours but the cap resolved to "
            f"{resolved.max_neighbors_per_leaf}"
        )


def test_the_far_cap_covers_the_measured_quarter_of_the_leaf_count():
    """``max_far_per_node`` tracked ``num_leaves / 4``; keep headroom over it."""
    for per_device_n in _SIZES:
        resolved = DistributedFMMConfig(leaf_size=64).resolved_for(per_device_n, _NDEV)
        num_leaves = max(1, -(-per_device_n // 64))
        assert resolved.max_interactions_per_node >= num_leaves // 4, (
            f"at {per_device_n}/device the far list needs ~{num_leaves // 4} per "
            f"node but the cap resolved to {resolved.max_interactions_per_node}"
        )


def test_every_derived_capacity_is_monotone_in_per_device_n():
    """More particles per device may never buy a smaller buffer."""
    previous = None
    for per_device_n in _SIZES:
        resolved = DistributedFMMConfig().resolved_for(per_device_n, _NDEV)
        current = {f: getattr(resolved, f) for f in DERIVED_CAP_FIELDS}
        if previous is not None:
            for field, value in current.items():
                assert value >= previous[field], (
                    f"{field} fell from {previous[field]} to {value} going to "
                    f"{per_device_n}/device"
                )
        previous = current


def test_the_capacity_ladder_is_powers_of_two():
    """Each distinct capacity is a distinct compiled shape.

    A ladder keeps recompiles across an N sweep to log2(N) rather than one per N
    -- the property ``_sub_million_minimum_memory_pair_queue``'s docstring exists
    to protect, and the reason this rule rounds rather than fits.
    """
    for per_device_n in _SIZES:
        resolved = DistributedFMMConfig().resolved_for(per_device_n, _NDEV)
        for field in ("max_pair_queue", "max_neighbors_per_leaf"):
            value = getattr(resolved, field)
            assert _is_power_of_two(value), (
                f"{field}={value} at {per_device_n}/device is not a power of two, "
                "so an N sweep recompiles once per N"
            )


def test_the_derived_queue_crosses_the_single_device_rule_at_its_calibration_point():
    """The ported rule is linear; the requirement is not. That is the whole fix.

    ``_sub_million_minimum_memory_pair_queue`` is ``max(32768, N/4)`` rounded to a
    power of two, measured on the single-GPU lane at leaf 256, and its docstring
    records where it was calibrated: *"N=131072 fits, N=262144 raises 'Pair queue
    capacity exceeded'"*. At leaf 256 those are 512 and 1024 leaves.

    The measured requirement here is ``2.83 * num_leaves ** 1.5``, and
    ``2.83 * 512 ** 1.5 == 32786`` -- that rule's 32768 floor, to four figures. So
    the linear rule is this curve evaluated at its own calibration point, held flat
    in both directions from there. Which means it is wrong in *both* directions,
    and the derived rule must cross it exactly there:

    * **below**, the flat floor over-sizes, and that is not free -- the walk's
      wavefront loop is linear in the capacity, so 32768 on a 15-node tree is a
      28.8x tax for an identical answer;
    * **above**, it under-sizes, which is the wall its own docstring reports one
      rung past the calibration point.
    """
    rule = _sub_million_minimum_memory_pair_queue
    # Below the calibration point the derivation does not inherit the flat floor.
    for per_device_n in (256, 1024, 8192, 32768):
        derived = DistributedFMMConfig(leaf_size=256).resolved_for(per_device_n, _NDEV)
        assert derived.max_pair_queue <= rule(num_particles=per_device_n), (
            f"at {per_device_n}/device and leaf 256 the derived queue "
            f"({derived.max_pair_queue}) exceeds the flat rule "
            f"({rule(num_particles=per_device_n)}), so it inherited an over-sizing "
            "that costs run time for no coverage"
        )

    # At and above it, the derived rule pulls away monotonically.
    ratios = []
    for per_device_n in (131072, 262144, 524288, 1048576):
        derived = DistributedFMMConfig(leaf_size=256).resolved_for(per_device_n, _NDEV)
        linear = rule(num_particles=per_device_n)
        assert derived.max_pair_queue > linear, (
            f"at {per_device_n}/device the derived queue "
            f"({derived.max_pair_queue}) did not exceed the linear rule ({linear}), "
            "which was measured to overflow in this range"
        )
        ratios.append(derived.max_pair_queue / linear)
    assert ratios == sorted(
        ratios
    ), f"the gap to the linear rule must not shrink as N grows; got {ratios}"

    # And at leaf 64 the same N has 4x the leaves. At 2048 leaves the eager ladder
    # converged on 262144, while ``N/4`` is 32768 -- measured to overflow, and to
    # collapse ``self_near_pairs`` from 351246 to 29594.
    at_64 = DistributedFMMConfig(leaf_size=64).resolved_for(131072, _NDEV)
    assert at_64.max_pair_queue >= 262144, (
        "the derived queue is below the capacity the eager ladder converged on at "
        f"2048 leaves (262144): got {at_64.max_pair_queue}"
    )


def test_the_derived_queue_covers_every_converged_ladder_capacity():
    """Grounded on the oracle, not on the curve fit.

    The eager retry ladder converges on a capacity that fits, so these are
    measurements rather than extrapolations (``--sweep occupancy`` reports them as
    ``converged_pair_queue``). Thin disc, leaf 64, dehnen MAC, theta=0.4.
    """
    for per_device_n, converged in ((8192, 4096), (32768, 32768), (131072, 262144)):
        derived = DistributedFMMConfig(leaf_size=64).resolved_for(per_device_n, _NDEV)
        assert derived.max_pair_queue >= converged, (
            f"at {per_device_n}/device the eager ladder needed {converged} and the "
            f"rule gives {derived.max_pair_queue}"
        )


def test_the_dense_buffer_floors_are_the_shipped_constants():
    """Small problems keep exactly the per-node buffers they had.

    The tier's own tests run 16 to 128 particles per device; a derivation that
    shrank these would be changing what those tests exercise while claiming to
    lift a ceiling. Both DID size real buffers on the traced path, so preserving
    them preserves behaviour.
    """
    resolved = DistributedFMMConfig().resolved_for(16, _NDEV)
    assert resolved.max_interactions_per_node >= 512
    assert resolved.max_neighbors_per_leaf >= 128
    assert resolved.cross_max_interactions_per_node >= 512
    assert resolved.cross_max_neighbors_per_leaf >= 128


def test_a_tiny_tree_does_not_inherit_the_old_pair_queue_constant():
    """The shipped 32768 was never what a traced walk used, so carrying it down
    to a 15-node tree would not preserve behaviour -- it would introduce a cost.

    The walk's wavefront loop evaluates the full capacity-length array every
    round and its round count is O(tree depth), so per-round work is linear in
    the capacity whether or not the pairs are live. Measured traced on a
    64-particle/leaf-8 tree, identical answer (56 near pairs) throughout: 3.64 ms
    at 1024, 32.1 ms at 8192, 105 ms at 32768 -- a 28.8x tax for nothing. The
    ladder's own first rung was 1024, and 1024 already exceeds the 120 distinct
    node pairs a 15-node tree has.

    This is the one place the derivation deliberately goes *below* a shipped
    constant, so it gets its own test rather than an exception inside another.
    """
    resolved = DistributedFMMConfig(leaf_size=8).resolved_for(64, _NDEV)
    assert resolved.max_pair_queue == 1024, (
        "a 15-node tree resolved to a wavefront of "
        f"{resolved.max_pair_queue}; the ladder's own floor is 1024 and nothing "
        "that small can need more"
    )
    assert resolved.cross_max_pair_queue == 1024


def test_the_wavefront_rule_dominates_its_floor_from_128_leaves_up():
    """So the floor never decides a capacity that matters.

    128 leaves is where the eager ladder first converges above 1024 (it converged
    on 4096 there), and it is also around where the single-GPU rule's own 32768
    floor was calibrated. Above that point the rule is what sets the number, and
    the floor is only there for trees too small to have a requirement.
    """
    for per_device_n in (8192, 32768, 131072, 1_048_576):
        resolved = DistributedFMMConfig(leaf_size=64).resolved_for(per_device_n, _NDEV)
        assert resolved.max_pair_queue > 1024
        assert resolved.cross_max_pair_queue > 1024


def test_the_cross_caps_are_derived_too():
    """They were the wall immediately behind the self ones.

    At 32768/device and leaf 64 on 2 A100s the self flags are clear and
    ``cross_near_overflow`` fires with a 49 % force error; clearing it moves the
    flag to ``cross_queue_overflow``. Deriving all three drops the error to 5.2e-4
    with no flag set, so leaving them constant would have made the lift stop one
    buffer short of doing anything.
    """
    resolved = DistributedFMMConfig(leaf_size=64).resolved_for(32768, _NDEV)
    num_leaves = 32768 // 64
    assert resolved.cross_max_neighbors_per_leaf >= num_leaves
    assert resolved.cross_max_pair_queue >= 131072


def test_an_explicit_capacity_is_never_overridden():
    """A number the caller wrote down is the number the walk gets.

    Including a deliberately tiny one: ``tests/distributed/test_distributed_fmm_driver.py``
    starves the caps on purpose to prove the overflow flags fire, and a resolver
    that "helpfully" grew them would silently delete that test's premise.
    """
    starved = DistributedFMMConfig(
        max_pair_queue=64,
        max_interactions_per_node=4,
        max_neighbors_per_leaf=4,
        process_block=8,
        cross_max_pair_queue=64,
        cross_max_interactions_per_node=4,
        cross_max_neighbors_per_leaf=4,
    )
    resolved = starved.resolved_for(131072, _NDEV)
    assert resolved.max_pair_queue == 64
    assert resolved.max_interactions_per_node == 4
    assert resolved.max_neighbors_per_leaf == 4
    assert resolved.process_block == 8
    assert resolved.cross_max_pair_queue == 64
    assert resolved.cross_max_interactions_per_node == 4
    assert resolved.cross_max_neighbors_per_leaf == 4


def test_resolution_is_idempotent():
    """Resolving a resolved config is a no-op, so the driver may resolve twice."""
    once = DistributedFMMConfig().resolved_for(131072, _NDEV)
    twice = once.resolved_for(131072, _NDEV)
    assert twice == once
    # And a different N cannot move an already-resolved value: after the first
    # pass there are no sentinels left to fill.
    assert once.resolved_for(16, _NDEV) == once


def test_the_process_block_clears_the_documented_single_device_floor():
    """256 is asserted for the single-device lane; the mesh had 64.

    ``process_block`` stopped being a capacity when the traced wavefront started
    coming from ``max_pair_queue`` (yggdrax ``fix/traced-wavefront-capacity``), so
    this is purely about vectorisation width -- but 64 was chosen when it *was* a
    capacity, and nothing has argued for it since.
    """
    for per_device_n in _SIZES:
        resolved = DistributedFMMConfig().resolved_for(per_device_n, _NDEV)
        assert resolved.process_block >= 256


def test_scaling_an_unresolved_config_leaves_the_sentinels_alone():
    """``with_scaled_caps`` must not turn "derive this" into a number times two.

    The retry loop runs on a resolved config, but the method is public and a
    caller can reach it first; ``None`` there means "still derived from N", which
    is not something a multiply can express.
    """
    scaled = DistributedFMMConfig().with_scaled_caps(2.0)
    for field in DERIVED_CAP_FIELDS:
        assert getattr(scaled, field) is None, f"{field} was scaled from a sentinel"


def test_scaling_a_resolved_config_still_grows_every_capacity():
    """And once resolved, the retry loop works exactly as it did."""
    resolved = DistributedFMMConfig().resolved_for(131072, _NDEV)
    scaled = resolved.with_scaled_caps(2.0)
    for field in DERIVED_CAP_FIELDS:
        if field == "process_block":  # not a capacity; with_scaled_caps leaves it
            continue
        assert getattr(scaled, field) == 2 * getattr(resolved, field), field


def test_the_cross_caps_scale_with_the_mesh_and_the_self_caps_do_not():
    """The coarse frontier is every device's leaves; the local tree is local.

    ``build_coarse_frontier`` is "all leaf nodes of ``tree``", so a local target
    leaf has ``(ndev - 1) * num_leaves`` remote coarse leaves that could be near it,
    and the same worst-leaf behaviour that sets the self near cap sets this one that
    many times larger. Sizing the cross caps as if ``ndev`` were 2 overflowed
    ``cross_near`` at every rung on 4 A100s -- 32768 through 1048576
    particles/device, leaf 256 -- with a 61 % force error and no self flag set at
    all, which is exactly the shape of failure this file exists to prevent.
    """
    base = DistributedFMMConfig(leaf_size=64)
    two = base.resolved_for(131072, 2)
    four = base.resolved_for(131072, 4)

    for field in (
        "max_pair_queue",
        "max_interactions_per_node",
        "max_neighbors_per_leaf",
    ):
        assert getattr(two, field) == getattr(four, field), (
            f"{field} is a self-walk cap over the LOCAL tree and must not move with "
            "the mesh size"
        )
    for field in (
        "cross_max_pair_queue",
        "cross_max_interactions_per_node",
        "cross_max_neighbors_per_leaf",
    ):
        assert getattr(four, field) > getattr(two, field), (
            f"{field} did not grow from ndev 2 to ndev 4, so a 4-device mesh gets "
            "the buffers of a 2-device one"
        )

    num_leaves = 131072 // 64
    assert four.cross_max_neighbors_per_leaf >= 3 * num_leaves, (
        "the cross near cap must cover the 3 remote frontiers a 4-device mesh has: "
        f"needs {3 * num_leaves}, got {four.cross_max_neighbors_per_leaf}"
    )


def test_the_default_leaf_size_is_a_production_leaf():
    """``leaf_size=8`` was the smallest leaf in the codebase.

    It put the shipped default below the documented operating point (64 to 1024 in
    production runs) and made both dense traversal buffers -- which are
    ``[nodes, cap]`` with the cap tracking ``num_leaves`` -- quadratically larger
    than they need to be for the same particle count. Measured at 131072
    particles/device on 2 A100s, leaf 256 is 3x faster than leaf 64 *and* more
    accurate, so 64 is the conservative end of the range rather than the fastest.
    """
    assert DistributedFMMConfig().leaf_size >= 64


def test_dataclass_replace_still_reaches_every_derived_field():
    """The sentinel must not have cost the fields their normal dataclass behaviour."""
    config = dataclasses.replace(DistributedFMMConfig(), max_pair_queue=1 << 20)
    assert config.max_pair_queue == 1 << 20
    assert config.resolved_for(131072, _NDEV).max_pair_queue == 1 << 20

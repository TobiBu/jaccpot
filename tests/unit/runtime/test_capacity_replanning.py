"""A capacity ceiling must be re-planned or explained, never just raised.

Two failures from the tranche-1 measurements were flat constants, not device
limits:

* ``large_n_gpu``/static_radix/leaf 256/order 4/theta 0.6 at N=262144 raised
  "Pair queue capacity exceeded; increase max_pair_queue and rebuild" -- with the
  queue resolved to 65536, the device holding 29.62 GiB free, and the largest
  statically-sized buffer at 0.31 GiB. Growing the queue then hit the *next* flat
  constant, ``JACCPOT_STATIC_STRICT_FUSED_COMPACT_FAR_PAIR_CAP`` at 131072.
* ``preset="accurate"``/leaf 32 at N=1048576 died on an 8.00 GiB single
  allocation on a 40 GB card, naming neither the buffer nor a preset that fits.

Device-independent: the ladder and the report are closed-form, so these run on a
CPU and cannot be flaky on a shared GPU.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaccpot.runtime.capacity_diagnostics import (
    capacity_report,
    is_capacity_failure,
    reraise_with_capacity_report,
)
from jaccpot.runtime.fmm_constants import (
    _GPU_MINIMUM_MEMORY_PAIR_QUEUE,
    _minimum_memory_streamed_gpu_traversal_seed,
    _sub_million_minimum_memory_pair_queue,
)


class TestPairQueueScalesWithN:
    def test_it_no_longer_flatlines_below_a_million(self) -> None:
        """The bug: one constant for every N below 1048576."""

        queues = [
            _sub_million_minimum_memory_pair_queue(num_particles=n)
            for n in (131072, 262144, 524288)
        ]
        assert queues == sorted(queues)
        assert len(set(queues)) > 1, f"still constant across N: {queues}"

    @pytest.mark.parametrize("n", (1024, 4096, 32768, 65536, 131072))
    def test_the_validated_small_end_is_untouched(self, n: int) -> None:
        """Nothing at or below N=131072 changes, so that ladder is not re-tuned."""

        assert _sub_million_minimum_memory_pair_queue(num_particles=n) == int(
            _GPU_MINIMUM_MEMORY_PAIR_QUEUE
        )

    def test_it_meets_the_large_n_branch_exactly(self) -> None:
        """Continuity at the 1048576 boundary, so there is no step or gap."""

        below = _sub_million_minimum_memory_pair_queue(num_particles=1048575)
        at = _minimum_memory_streamed_gpu_traversal_seed(
            num_particles=1048576
        ).max_pair_queue
        assert below == int(at) == 262144

    def test_the_whole_ladder_is_monotone(self) -> None:
        ns = [1 << k for k in range(10, 23)]
        queues = [
            int(
                _minimum_memory_streamed_gpu_traversal_seed(
                    num_particles=n
                ).max_pair_queue
            )
            for n in ns
        ]
        assert queues == sorted(queues), dict(zip(ns, queues))

    def test_every_capacity_is_a_power_of_two(self) -> None:
        """Each distinct capacity is a distinct compiled shape; a ladder bounds it."""

        for n in (1 << k for k in range(10, 21)):
            q = _sub_million_minimum_memory_pair_queue(num_particles=n)
            assert q & (q - 1) == 0, f"N={n} gave a non-power-of-two {q}"


class TestCapacityFailureClassification:
    @pytest.mark.parametrize(
        "message",
        (
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.00GiB",
            "Pair queue capacity exceeded; increase max_pair_queue and rebuild.",
            "strict fused compact far-pair cap exceeded: cap=131072.",
        ),
    )
    def test_capacity_failures_are_recognised(self, message: str) -> None:
        assert is_capacity_failure(RuntimeError(message))

    @pytest.mark.parametrize(
        "message",
        (
            "max_order must be >= 0",
            "basis must be one of 'cartesian', 'solidfmm', 'complex', or 'real'",
        ),
    )
    def test_other_failures_are_left_alone(self, message: str) -> None:
        """A misclassified error would be re-raised with an irrelevant report."""

        assert not is_capacity_failure(ValueError(message))


class TestCapacityReport:
    CONFIG = dict(
        num_particles=1048576,
        leaf_size=32,
        max_order=4,
        working_dtype=jnp.float32,
        preset="accurate",
    )

    def _traversal(self):
        from yggdrax.interactions import DualTreeTraversalConfig

        return DualTreeTraversalConfig(
            max_pair_queue=262144,
            process_block=256,
            max_interactions_per_node=8192,
            max_neighbors_per_leaf=4096,
        )

    def test_it_names_buffers_largest_first(self) -> None:
        entries = capacity_report(**self.CONFIG, traversal_config=self._traversal())
        sizes = [gib for _, gib, _ in entries]
        assert sizes == sorted(sizes, reverse=True)
        assert all(gib > 0 for gib in sizes)

    def test_the_reported_scale_matches_the_observed_failure(self) -> None:
        """The measured E9 failure was a single 8.00 GiB allocation.

        The largest buffer this configuration implies must be of that order --
        not 100x off -- or the report points somewhere useless.
        """

        entries = capacity_report(**self.CONFIG, traversal_config=self._traversal())
        assert 1.0 <= entries[0][1] <= 100.0, entries[:3]

    def test_every_entry_names_a_knob(self) -> None:
        for _, _, knob in capacity_report(
            **self.CONFIG, traversal_config=self._traversal()
        ):
            assert knob and knob.strip()

    def test_it_needs_no_traversal_config(self) -> None:
        """Callable from a handler where the config could not be resolved."""

        entries = capacity_report(**self.CONFIG, traversal_config=None)
        assert entries  # the coefficient and leaf-payload entries still apply

    def test_the_reraise_chains_the_original(self) -> None:
        original = RuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.00GiB"
        )
        with pytest.raises(RuntimeError) as excinfo:
            reraise_with_capacity_report(
                original, **self.CONFIG, traversal_config=self._traversal()
            )
        assert excinfo.value.__cause__ is original
        text = str(excinfo.value)
        assert "could not fit N=1048576" in text
        assert "GiB" in text
        assert "sized by:" in text
        # And a way forward, not just a diagnosis.
        assert "large_n_gpu" in text
        assert "TraversalOverrides" in text

    # The fused flat-walk lane at N = 8e6 (bench caps of 2026-10-03): its own
    # buffers, not the dual walk's per-node and per-leaf ones, which it never builds.
    FLAT_CAPS = dict(
        flat_walk=True,
        compact_far_pair_capacity=134217728,
        near_edge_capacity=268435456,
        queue_capacity=4194304,
        peak_wavefront=3241059,
    )

    def test_the_flat_walk_lane_lists_its_own_buffers(self) -> None:
        config = dict(self.CONFIG, num_particles=8_000_000, leaf_size=64, max_order=5)
        entries = capacity_report(
            **config,
            traversal_config=self._traversal(),
            walk_caps=self.FLAT_CAPS,
            num_leaves=466944,
        )
        names = [name for name, _, _ in entries]
        assert not any("max_interactions_per_node" in n for n in names)
        assert not any("fast-lane source payload" in n for n in names)
        assert any("far-pair list" in n for n in names)
        # sort scratch 16 B x 2^28 + 8 B x 2^27 = 5 GiB, near partials
        # 16 B x (2^28 + 466944 x 64) = 4.4 GiB: the two largest
        assert "sort scratch" in names[0] and "near-field partials" in names[1]
        assert 4.9 <= entries[0][1] <= 5.1 and 4.3 <= entries[1][1] <= 4.6

    def test_the_flat_walk_reraise_names_the_cap_rule(self) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            reraise_with_capacity_report(
                RuntimeError("RESOURCE_EXHAUSTED: Out of memory"),
                **self.CONFIG,
                walk_caps=self.FLAT_CAPS,
            )
        text = str(excinfo.value)
        assert "unnamed" in text and "JACCPOT_FLAT_WALK_CAP_HEADROOM" in text

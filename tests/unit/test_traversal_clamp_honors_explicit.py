"""Static fixed-sizing must honor an EXPLICIT large-N traversal config.

Regression guard for a bug where the production large-N GPU memory-safety clamp
(``_clamp_gpu_traversal_config_for_memory``) reduced even a caller-supplied
(explicit) ``DualTreeTraversalConfig`` down to the fixed streamed minimum-memory
ceiling (``max_pair_queue`` 262144 / 524288) *on the static fixed-sizing path*.
That ceiling is far below the traversal frontier a concentrated multi-million-
particle disk needs, so with ``fail_fast`` on it turned a deliberate,
memory-fitting user override into a hard "Pair queue capacity exceeded" overflow
at N >= ~2M on >=40 GB GPUs (e.g. a 4M-particle A100 disk that otherwise fits at
~36 GB).

Static fixed sizing means "use the sizes I gave you", so it now passes an explicit
config through unclamped (auto-sized *preset* seeds are still bounded). The
adaptive path, which still clamped explicit configs, went in the 2026-10 cleanup
(X4), so static sizing is the only sizing and this is its one contract.
"""

from __future__ import annotations

from yggdrax.interactions import DualTreeTraversalConfig

import jaccpot.runtime._fmm_impl as fmm_impl_private
from jaccpot import FMMPreset
from jaccpot.config import FarFieldConfig, RuntimePolicyConfig

# Streamed minimum-memory ceiling for the 1M <= N < 4.19M band (see fmm_constants).
_CEILING_PAIR_QUEUE = 262_144


def _impl(traversal: DualTreeTraversalConfig):
    return fmm_impl_private.FMMEngine(
        preset=FMMPreset.LARGE_N_GPU,
        expansion_basis="solidfmm",
        mac_type="engblom",
        farfield=FarFieldConfig(streamed_far_pairs=True, grouped_interactions=False),
        runtime_policy=RuntimePolicyConfig(
            memory_objective="minimum_memory",
            fail_fast=True,
            traversal_config=traversal,
        ),
    )


def _oversized() -> DualTreeTraversalConfig:
    # Caps sized for a ~4M concentrated disk on a 40 GB A100 (~2N pair queue),
    # well above the fixed streamed ceiling.
    return DualTreeTraversalConfig(
        max_pair_queue=8_000_000,
        process_block=256,
        max_interactions_per_node=16_384,
        max_neighbors_per_leaf=4_096,
    )


def test_static_fixed_sizing_honors_oversized_explicit_traversal():
    requested = _oversized()
    impl = _impl(requested)
    # Sizing is always static (2026-10 cleanup, X4); explicit caps are kept as
    # given.
    overrides = impl._resolve_runtime_execution_overrides(
        num_particles=4_000_000, backend="gpu"
    )
    tc = overrides.traversal_config
    assert tc is not None
    assert int(tc.max_pair_queue) == requested.max_pair_queue
    assert int(tc.max_interactions_per_node) == requested.max_interactions_per_node
    assert int(tc.max_neighbors_per_leaf) == requested.max_neighbors_per_leaf
    # ...definitely not pulled down to the streamed ceiling.
    assert int(tc.max_pair_queue) > _CEILING_PAIR_QUEUE

"""The ``large_n_gpu`` split-build default must survive a caller-supplied config.

``preset="large_n_gpu"`` exists to run the low-peak *split* dual-tree build. Whether
it does is decided by ``_streamed_minimum_memory_gpu_default_split_build``, derived
from ``memory_objective`` and ``streamed_far_pairs``. Those two fields are coerced by
``_apply_large_n_gpu_production_contract`` -- and the predicate used to be computed
*before* that coercion.

On the bare preset that was harmless: preset resolution had already set both fields.
It only bit when the caller passed an ``advanced=`` config too, which **replaces** the
preset's config, so the predicate read the dataclass defaults instead
(``memory_objective="balanced"``, ``streamed_far_pairs=None`` -> ``False``) and came
out ``False``. The preset then silently ran the monolithic build. Measured
consequence: an N=1e7 census OOMed on a 4.77 GiB allocation inside
``_dual_tree_build_raw``, on the one lane whose purpose is to avoid that allocation.

The failure was silent in both directions -- no warning fired, and
``_explicit_streamed_far_pairs`` correctly recorded the value as *not* requested, so
the predicate was derived from an option the caller never set.

Most of what follows tests :func:`derive_split_build_default` directly rather than a
constructed solver, because the predicate requires a GPU backend and would otherwise
be untestable in CPU CI -- the whole conjunct collapses to ``False`` there whether or
not the ordering bug is present, which is precisely the shape of test that passes
while covering nothing.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import pytest

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig
from jaccpot.runtime._fmm_impl import derive_split_build_default

#: The configuration the ``large_n_gpu`` preset resolves to, on a GPU.
LANE = dict(
    memory_objective="minimum_memory",
    backend="gpu",
    tree_type="radix",
    expansion_basis="solidfmm",
    streamed_far_pairs=True,
)


def test_the_lane_configuration_selects_the_split_build():
    """All five conjuncts satisfied is the only case that selects the split build."""

    assert derive_split_build_default(**LANE) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_objective", "balanced"),
        ("backend", "cpu"),
        ("tree_type", "octree"),
        ("expansion_basis", "real"),
        ("streamed_far_pairs", False),
    ],
)
def test_every_conjunct_is_load_bearing(field, value):
    """Breaking any one conjunct must turn the split build off.

    Parametrised over all five so a future edit cannot drop one silently -- a
    predicate that ignores a field it claims to read is the failure this whole
    module exists for.
    """

    assert derive_split_build_default(**{**LANE, field: value}) is False


def test_streamed_far_pairs_is_the_conjunct_the_advanced_config_breaks():
    """Setting only ``memory_objective`` does **not** recover the split build.

    Pinned because the original bug report named ``memory_objective`` as the cause.
    Measured on an A100: an advanced config carrying
    ``runtime.memory_objective="minimum_memory"`` but no ``streamed_far_pairs``
    still produced ``False``. The load-bearing conjunct is the other one, and a fix
    aimed at ``memory_objective`` alone would have looked right and changed nothing.
    """

    # What an `advanced=FMMAdvancedConfig()` supplies at `_resolve_derived_lane_flags`
    # time: both fields at their dataclass defaults.
    both_defaulted = {
        **LANE,
        "memory_objective": "balanced",
        "streamed_far_pairs": False,
    }
    assert derive_split_build_default(**both_defaulted) is False

    # Setting only the field the original report blamed.
    memory_objective_only = {**both_defaulted, "memory_objective": "minimum_memory"}
    assert derive_split_build_default(**memory_objective_only) is False

    # Setting the one that actually gates it.
    streamed_too = {**memory_objective_only, "streamed_far_pairs": True}
    assert derive_split_build_default(**streamed_too) is True


@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="the split-build predicate is a GPU-only default; on CPU it is always False",
)
def test_an_advanced_config_does_not_disable_the_preset_split_build():
    """The bug itself: preset alone and preset+advanced must agree.

    This is the integration half and it needs a GPU, so CPU CI skips it. The
    conjunct logic above is what CI actually covers; this asserts the *ordering* --
    that the predicate is derived after ``_apply_large_n_gpu_production_contract``
    rather than before it.
    """

    common = dict(preset="large_n_gpu", expansion_basis="solidfmm", theta=0.5)
    bare = FastMultipoleMethod(**common)._impl

    advanced = FMMAdvancedConfig()
    advanced = replace(advanced, tree=replace(advanced.tree, tree_type="radix"))
    with_config = FastMultipoleMethod(advanced=advanced, **common)._impl

    assert bare._streamed_minimum_memory_gpu_default_split_build is True
    assert with_config._streamed_minimum_memory_gpu_default_split_build is True, (
        "passing an advanced config alongside preset='large_n_gpu' disabled the "
        "split dual-tree build the preset exists to select"
    )

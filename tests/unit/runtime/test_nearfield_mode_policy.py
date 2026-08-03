"""The GPU near-field auto policy must never pick the per-pair reference scan.

``nearfield_mode="baseline"`` is a ``lax.scan`` over leaf pairs *one pair at a
time* with a ``lax.cond`` per pair. It is the readable reference traversal and it
is fine on a CPU; on a GPU it serialises one launch per leaf pair. Measured on an
idle A100, ``preset="accurate"``, leaf 64, p=4, real basis, evaluating a prebuilt
tree:

===========  ==========  ==========  =======
N            baseline    bucketed    factor
===========  ==========  ==========  =======
4096          5.225 s     0.0103 s     507x
16384        49.961 s     0.1177 s     424x
===========  ==========  ==========  =======

That is what made ``preset="accurate"`` 139x slower than ``large_n_gpu`` on the
flagship device, and ~30x slower on an A100 than the same preset on the CPU
beside it. The two traversals visit the same leaf pairs -- at fp64 on CPU,
N=8192/leaf 64, both sit at 4.4075521942e-05 rel-L2 against a direct O(N^2) sum
and differ from each other by 4.2e-16 -- so this is a scheduling choice, not an
accuracy one.

These tests run on whatever backend pytest is given. The CPU assertions are the
ones that guard against regressing the measured CPU crossovers; the GPU
assertion only runs on a GPU.
"""

from __future__ import annotations

import jax
import pytest

from jaccpot import FastMultipoleMethod

PRESETS = ("fast", "balanced", "accurate", "large_n_gpu")
PARTICLE_COUNTS = (512, 4096, 65536, 262144, 1 << 20)


def _solver(preset: str) -> FastMultipoleMethod:
    return FastMultipoleMethod(preset=preset, basis="real")


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="the GPU near-field policy needs a GPU"
)
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("n", PARTICLE_COUNTS)
def test_gpu_never_resolves_the_per_pair_baseline_scan(preset: str, n: int) -> None:
    """No preset, at any N, may land on ``baseline`` on a GPU."""

    resolved = _solver(preset)._impl._resolve_nearfield_mode(num_particles=n)
    assert resolved == "bucketed", (
        f"{preset} at N={n} resolved nearfield_mode={resolved!r} on a GPU. "
        "The per-pair baseline scan measured 424-507x slower than bucketed."
    )


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="guards the measured CPU crossovers"
)
def test_cpu_crossovers_are_unchanged() -> None:
    """The CPU answers are as they were measured; only the GPU answer moved.

    Spelled out per preset rather than derived, so a change to the policy has to
    change this table too. ``accurate`` staying on ``baseline`` throughout is why
    the CPU characterization goldens are untouched by the GPU fix.
    """

    expected = {
        "fast": ["baseline", "bucketed", "bucketed", "bucketed", "bucketed"],
        "balanced": ["bucketed"] * 5,
        "accurate": ["baseline"] * 5,
        "large_n_gpu": ["bucketed"] * 5,
    }
    actual = {
        preset: [
            _solver(preset)._impl._resolve_nearfield_mode(num_particles=n)
            for n in PARTICLE_COUNTS
        ]
        for preset in PRESETS
    }
    assert actual == expected


@pytest.mark.parametrize("mode", ("baseline", "bucketed"))
def test_an_explicit_mode_is_still_honoured(mode: str) -> None:
    """The policy is an ``auto`` policy; an explicit request still wins.

    Someone A/B-ing the two traversals -- which is how the 507x above was
    measured -- must be able to ask for the slow one.
    """

    from dataclasses import replace

    from jaccpot.config import FMMPreset
    from jaccpot.solver import _default_advanced_for_preset

    base = _default_advanced_for_preset(FMMPreset("accurate"))
    advanced = replace(base, nearfield=replace(base.nearfield, mode=mode))
    solver = FastMultipoleMethod(preset="accurate", basis="real", advanced=advanced)
    assert solver._impl._resolve_nearfield_mode(num_particles=4096) == mode


def test_accurate_no_longer_pins_the_reference_traversal() -> None:
    """``accurate``'s preset default is ``auto``, so the policy can act on it."""

    from jaccpot.config import FMMPreset
    from jaccpot.solver import _default_advanced_for_preset

    assert _default_advanced_for_preset(FMMPreset("accurate")).nearfield.mode == "auto"

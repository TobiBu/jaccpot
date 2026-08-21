"""The gradient path's configuration surface: resolution, auto policy, memoization.

``GradConfig`` replaces a set of ``JACCPOT_*`` environment variables as the way
to steer :meth:`FastMultipoleMethod.differentiable_accelerations`. Three
properties are worth pinning, and all three are regressions that actually
happened in this subsystem:

1. **Precedence.** An explicit field must beat the environment, and an unset
   field must fall back to it. Otherwise the migration silently breaks scripts.
2. **The auto lane.** ``nearfield_lane="auto"`` must select the leaf-major fast
   lane at galaxy N. The bucketed reverse OOMs there (30 GB peak at N=200000
   against the fast lane's 6.8 GB), so a default that picks bucketed hands the
   user a crash they cannot anticipate.
3. **Knobs are read per call, never at import.** Four of these were captured
   into module-level constants, which made setting the variable after
   ``import jaccpot`` silently do nothing.

These are host-only and take milliseconds; the numeric equivalence of the lanes
is covered in ``test_nearfield_fastlane_grad_path.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaccpot.config import GradConfig
from jaccpot.runtime.grad_options import (
    analytic_l2p_vjp_enabled,
    analytic_p2p_vjp_enabled,
    fused_m2l_pallas_enabled,
    grad_option_overrides,
    resolve_grad_options,
)


def _resolve(config=None, *, n=1024, supports=True):
    return resolve_grad_options(config, num_particles=n, supports_fast_lane=supports)


# --------------------------------------------------------------------------
# Auto near-field lane policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (256, "bucketed"),
        (99_999, "bucketed"),
        (100_000, "fast_lane"),
        (200_000, "fast_lane"),
        (1_000_000, "fast_lane"),
    ],
)
def test_auto_lane_crosses_over_at_the_measured_threshold(n, expected):
    assert _resolve(GradConfig(), n=n).nearfield_lane == expected


def test_auto_lane_never_selects_a_lane_the_state_cannot_run():
    """A LargeNPreparedState has no bucketed edge list and no selectable lane.

    ``supports_fast_lane=False`` must not make auto pick something unavailable;
    it falls back rather than producing an option the caller cannot honour.
    """
    assert _resolve(GradConfig(), n=1_000_000, supports=False).nearfield_lane == (
        "bucketed"
    )


def test_explicit_lane_overrides_the_auto_threshold():
    assert (
        _resolve(GradConfig(nearfield_lane="bucketed"), n=1_000_000).nearfield_lane
        == "bucketed"
    )
    assert (
        _resolve(GradConfig(nearfield_lane="fast_lane"), n=16).nearfield_lane
        == "fast_lane"
    )


def test_custom_crossover_threshold_is_honoured():
    cfg = GradConfig(nearfield_fast_lane_min_particles=500)
    assert _resolve(cfg, n=499).nearfield_lane == "bucketed"
    assert _resolve(cfg, n=500).nearfield_lane == "fast_lane"


def test_invalid_lane_is_rejected_at_construction():
    """An out-of-set lane must be refused at construction, not silently accepted.

    WHICH exception depends on whether the runtime typecheck is on, and both are
    correct refusals. `nearfield_lane` is annotated `GradNearFieldLane`, a
    `Literal`, so under `JACCPOT_RUNTIME_TYPECHECK=1` beartype rejects the value
    before `__post_init__`'s loud `ValueError` (STYLE_GUIDE §9) ever runs. Asserting
    only `ValueError` made this test fail under the hook while the behaviour it
    checks was working -- one of F40's 122. The property is "refused", so accept
    either refusal rather than weakening it to a bare `Exception`.
    """
    import jaxtyping

    # `jaxtyping.TypeCheckError`, not beartype's own class: the hook wraps
    # beartype's violation, and it is the wrapper that propagates. Both messages
    # name the parameter, so the `match` still pins WHICH field was refused.
    refusals = (ValueError, jaxtyping.TypeCheckError)
    with pytest.raises(refusals, match="nearfield_lane"):
        GradConfig(nearfield_lane="leaf_major")
    with pytest.raises(ValueError, match="min_particles"):
        GradConfig(nearfield_fast_lane_min_particles=-1)


# --------------------------------------------------------------------------
# Precedence: explicit field > environment > measured default
# --------------------------------------------------------------------------


def test_env_var_still_drives_the_auto_lane(monkeypatch):
    """The pre-GradConfig interface keeps working for callers who never pass one."""
    monkeypatch.setenv("JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE", "1")
    assert _resolve(None, n=64).nearfield_lane == "fast_lane"


def test_explicit_field_beats_the_environment(monkeypatch):
    monkeypatch.setenv("JACCPOT_DIFFERENTIABLE_NEARFIELD_FAST_LANE", "1")
    cfg = GradConfig(nearfield_lane="bucketed")
    assert _resolve(cfg, n=64).nearfield_lane == "bucketed"

    monkeypatch.setenv("JACCPOT_GRAD_REV_TIERS", "9")
    monkeypatch.setenv("JACCPOT_GRAD_REV_TIER_MIN_GAIN", "7.5")
    assert _resolve(None).reverse.max_tiers == 9
    assert _resolve(None).reverse.tier_min_gain == pytest.approx(7.5)
    explicit = _resolve(GradConfig(reverse_tiers=2, reverse_tier_min_gain=1.25)).reverse
    assert explicit.max_tiers == 2
    assert explicit.tier_min_gain == pytest.approx(1.25)


def test_reverse_defaults_match_the_measured_values():
    rev = _resolve(None).reverse
    assert (rev.tiered, rev.max_tiers, rev.tier_min_gain) == (True, 4, 3.0)
    assert (rev.skip_empty_tiles, rev.leaf_batch, rev.block_tile) == (True, 8, 8)


def test_malformed_env_values_fall_back_rather_than_raising(monkeypatch):
    """A typo in a tuning knob must not take down a run."""
    monkeypatch.setenv("JACCPOT_GRAD_REV_TIERS", "four")
    monkeypatch.setenv("JACCPOT_GRAD_REV_TIER_MIN_GAIN", "")
    rev = _resolve(None).reverse
    assert rev.max_tiers == 4
    assert rev.tier_min_gain == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Knobs are read per call, not captured at import
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,getter",
    [
        ("JACCPOT_ANALYTIC_P2P_VJP", analytic_p2p_vjp_enabled),
        ("JACCPOT_ANALYTIC_L2P_VJP", analytic_l2p_vjp_enabled),
        ("JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS", fused_m2l_pallas_enabled),
    ],
)
def test_gate_flags_respond_after_import(env, getter, monkeypatch):
    """Regression: these were module-level constants, so setting the variable
    after ``import jaccpot`` silently did nothing."""
    monkeypatch.setenv(env, "1")
    assert getter() is True
    monkeypatch.setenv(env, "0")
    assert getter() is False


def test_gate_helpers_are_bound_in_their_calling_modules():
    """The gates are consulted from modules that must import them by name.

    Regression: moving these from module-level constants to helper calls left one
    caller without the import. ``import jaccpot`` still succeeded -- the NameError
    only fires at call time, deep in the real-basis L2P reverse -- so this cheap
    host-side check is worth having next to the GPU tests that caught it.

    The module named for each gate is the one whose body **calls** it, which is
    what the guard is about. The L2P gate moved from ``real_harmonics`` to
    ``real_p2m_l2p`` when the former was split into its mathematical seams
    (Tier 1.3); ``real_harmonics`` is now a re-export aggregator and does not
    consult any gate itself, so asserting there would no longer test the thing
    this test exists to test.

    **The P2P gate moved the same way, and this test did not follow it.** Tier
    1.4/1.5 split ``near_field`` and the call went to ``nearfield/_kernels.py``
    (which imports the gate itself, at its own module scope); ``near_field`` kept
    only an unconsulted re-export. The hard-coded line below therefore asserted a
    re-export on an aggregator for two tiers -- the exact thing the paragraph above
    says is not worth asserting -- and it went unnoticed because a stale re-export
    keeps such an assertion green. It surfaced only when audit item 0.14 removed
    the dead import.

    Note the two halves are not redundant: the AST-derived scan below is the real
    invariant and it was already correct here, since it resolves the caller rather
    than trusting a name written years earlier. These three lines are a fast,
    readable smoke check, and they have to name the current callers to be worth
    anything.
    """
    from jaccpot.nearfield import _kernels as nf_kernels
    from jaccpot.operators import real_p2m_l2p
    from jaccpot.runtime.kernels import core

    assert callable(real_p2m_l2p.analytic_l2p_vjp_enabled)
    assert callable(nf_kernels.analytic_p2p_vjp_enabled)
    assert callable(core.fused_m2l_pallas_enabled)

    # And derive the same check rather than hard-coding three modules, so that a
    # future split cannot move a gate call into a module that forgot the import
    # and still leave this test green.
    import ast
    import importlib
    import pathlib

    gates = {
        "analytic_l2p_vjp_enabled",
        "analytic_p2p_vjp_enabled",
        "fused_m2l_pallas_enabled",
    }
    package_root = pathlib.Path(nf_kernels.__file__).resolve().parents[1]
    callers: dict[str, set[str]] = {}
    for path in sorted(package_root.rglob("*.py")):
        module_tree = ast.parse(path.read_text())
        called = {
            node.func.id
            for node in ast.walk(module_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in gates
        }
        if called:
            relative = path.relative_to(package_root.parent).with_suffix("")
            callers[relative.as_posix().replace("/", ".")] = called

    assert callers, "found no module calling a grad gate -- the scan is vacuous"
    for module_name, called in callers.items():
        module = importlib.import_module(module_name)
        for gate in sorted(called):
            assert callable(
                getattr(module, gate, None)
            ), f"{module_name} calls {gate}() but does not have it bound"


def test_tier_knobs_respond_after_import(monkeypatch):
    from jaccpot.nearfield import grad

    monkeypatch.setenv("JACCPOT_GRAD_REV_TIERS", "7")
    monkeypatch.setenv("JACCPOT_GRAD_REV_TIER_MIN_GAIN", "2.5")
    assert grad._grad_rev_tier_max() == 7
    assert grad._grad_rev_tier_min_gain() == pytest.approx(2.5)


# --------------------------------------------------------------------------
# Scoped overrides for the deep gates
# --------------------------------------------------------------------------


def test_overrides_apply_within_scope_and_restore_after():
    before = (
        fused_m2l_pallas_enabled(),
        analytic_p2p_vjp_enabled(),
        analytic_l2p_vjp_enabled(),
    )
    options = _resolve(
        GradConfig(
            fused_m2l_pallas=not before[0],
            analytic_p2p_vjp=not before[1],
            analytic_l2p_vjp=not before[2],
        )
    )
    with grad_option_overrides(options):
        assert fused_m2l_pallas_enabled() is (not before[0])
        assert analytic_p2p_vjp_enabled() is (not before[1])
        assert analytic_l2p_vjp_enabled() is (not before[2])
    assert (
        fused_m2l_pallas_enabled(),
        analytic_p2p_vjp_enabled(),
        analytic_l2p_vjp_enabled(),
    ) == before


def test_overrides_restore_even_when_the_body_raises():
    before = analytic_p2p_vjp_enabled()
    options = _resolve(GradConfig(analytic_p2p_vjp=not before))
    with pytest.raises(RuntimeError, match="boom"):
        with grad_option_overrides(options):
            raise RuntimeError("boom")
    assert analytic_p2p_vjp_enabled() is before


# --------------------------------------------------------------------------
# Memoization of frozen-topology host work
# --------------------------------------------------------------------------


def test_reverse_tiers_are_memoized_per_payload():
    """The tier plan is frozen topology, but building it pulls the padded
    validity mask to the host -- 32M elements at N=1000000. Uncached that ran on
    every forward call, i.e. once per optimisation step."""
    from jaccpot.nearfield import grad

    grad.clear_leafpair_reverse_tier_cache()
    # Occupancy spread wide enough that tiering is accepted at a low gain bar.
    mask = np.zeros((64, 8, 8), dtype=bool)
    for leaf in range(64):
        mask.reshape(64, 64)[leaf, : max(1, leaf)] = True

    calls = []
    real_builder = grad.build_leafpair_reverse_tiers

    def counting_builder(*args, **kwargs):
        calls.append(1)
        return real_builder(*args, **kwargs)

    grad.build_leafpair_reverse_tiers = counting_builder
    try:
        first = grad._leafpair_reverse_tiers_cached(
            mask, mask, slot_tile=8, max_tiers=4, min_gain=1.0
        )
        second = grad._leafpair_reverse_tiers_cached(
            mask, mask, slot_tile=8, max_tiers=4, min_gain=1.0
        )
    finally:
        grad.build_leafpair_reverse_tiers = real_builder

    assert len(calls) == 1, "second call should have hit the cache"
    assert first is second
    # Distinct tuning must not collide with the cached entry.
    grad.build_leafpair_reverse_tiers = counting_builder
    try:
        grad._leafpair_reverse_tiers_cached(
            mask, mask, slot_tile=8, max_tiers=2, min_gain=1.0
        )
    finally:
        grad.build_leafpair_reverse_tiers = real_builder
    assert len(calls) == 2


def test_tiering_covers_every_leaf_exactly_once():
    from jaccpot.nearfield import grad as nf

    mask = np.zeros((48, 4, 8), dtype=bool)
    for leaf in range(48):
        mask.reshape(48, 32)[leaf, : 1 + (leaf % 32)] = True
    tiers = nf.build_leafpair_reverse_tiers(
        mask, slot_tile=8, max_tiers=4, min_gain=1.0
    )
    if tiers is None:
        pytest.skip("this occupancy profile does not admit a paying split")
    seen = [leaf for members, _ in tiers for leaf in members]
    assert sorted(seen) == list(range(48))
    # Leaves stay in Morton order within a tier -- a global occupancy sort was
    # measured ~7x slower because it destroyed source-gather locality.
    for members, _ in tiers:
        assert list(members) == sorted(members)


def test_forward_permutation_is_memoized_on_the_engine():
    """Regression: this did a device_get plus an O(N log N) host argsort on every
    call, for a value fixed by the frozen topology."""
    from jaccpot import FastMultipoleMethod

    rng = np.random.default_rng(3)
    positions = jnp.asarray(rng.normal(size=(128, 3)))
    masses = jnp.asarray(rng.uniform(0.5, 1.5, size=128))
    fmm = FastMultipoleMethod(basis="solidfmm", use_pallas=False, theta=0.6)
    state = fmm.prepare_state(positions, masses, max_order=2, leaf_size=16)

    engine = fmm._impl
    first = engine._forward_permutation(state)
    second = engine._forward_permutation(state)
    assert first is second, "repeat call rebuilt the permutation"
    # It really is the inverse of the prepared permutation.
    inverse = np.asarray(state.inverse_permutation)
    assert np.array_equal(np.asarray(first)[inverse], np.arange(inverse.size))


def test_forward_permutation_memo_stays_correct_across_states():
    """The memo is a single slot on the engine, so two states used alternately
    thrash it. Thrashing is acceptable; returning the *other* state's permutation
    would silently scramble the force."""
    from jaccpot import FastMultipoleMethod

    fmm = FastMultipoleMethod(basis="solidfmm", use_pallas=False, theta=0.6)
    states = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        states.append(
            fmm.prepare_state(
                jnp.asarray(rng.normal(size=(96, 3))),
                jnp.asarray(rng.uniform(0.5, 1.5, size=96)),
                max_order=2,
                leaf_size=8,
            )
        )

    engine = fmm._impl
    for _ in range(3):
        for state in states:
            forward = np.asarray(engine._forward_permutation(state))
            inverse = np.asarray(state.inverse_permutation)
            assert np.array_equal(forward[inverse], np.arange(inverse.size))

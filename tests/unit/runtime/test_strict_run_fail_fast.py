"""The strict lane's fail-fast contract, and the counters that make it auditable.

This is the remainder of audit item **F33**. `fmm_strict_run.py` sat at 79% with
83 statements missed, most of them in two functions too large to unit-test
(`strict_run_v2`, 522 lines; `_refresh_large_n_same_topology`, 562). What *is*
reachable, and what this module covers, is everything those functions do **before
they touch a device**: the four rejection guards and the profile-key accounting.

The finding that made this possible is worth stating, because F33 has been
recorded as GPU-blocked since it was filed:
`_is_large_n_gpu_production_profile` checks **four config values** --
``preset="large_n_gpu"``, ``tree_type="radix"``, ``expansion_basis="solidfmm"``,
``execution_backend != "octree"`` -- and **no backend at all**. So the strict
lane's whole entry path opens on CPU with two constructor arguments. That is the
same shape as F27 (`preset`, not hardware) and the third time in this audit that
"needs a GPU" turned out to mean "needs the right profile".

Two behaviours here are contracts rather than implementation details.

**`refresh_every != 1` is a physics guard, not a performance knob.** The strict
lane's velocity-Verlet update is endpoint-correct only if the state is refreshed
every step; refreshing less often silently integrates against stale multipoles.
It raises.

**Fused mode refuses to degrade silently.** If fused mode is requested and the
particle count is not in ``JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET``, the lane
raises instead of quietly running the slower non-fused path -- and the message
names the variable and both ways to fix it. A silent fallback here would be a
performance cliff nobody would ever notice, which is exactly what the code
comment says. That predicate is `_strict_fused_profile_allows_n`, covered from
the other side in ``test_strict_cap_profile.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaccpot.runtime._fmm_impl import FMMEngine

_N = 8


def _plain():
    """An engine that does *not* satisfy the large-N production profile."""
    return FMMEngine(theta=0.6, working_dtype=jnp.float32)


def _strict():
    """An engine whose profile opens the strict lane -- on CPU, by config alone."""
    return FMMEngine(
        theta=0.6,
        working_dtype=jnp.float32,
        preset="large_n_gpu",
        expansion_basis="solidfmm",
    )


@pytest.fixture
def particles():
    """Positions and masses. Never integrated -- every guard fires before use.

    Returns
    -------
    tuple[Array, Array]
        ``(positions, masses)`` for ``_N`` particles.
    """
    return jnp.zeros((_N, 3)), jnp.ones((_N,))


def _run_kwargs(positions, masses, **overrides):
    kwargs = dict(
        state=positions,
        masses=masses,
        dt=0.1,
        num_steps=1,
        refresh_every=1,
        leaf_size=4,
        max_order=2,
    )
    kwargs.update(overrides)
    return kwargs


class TestTheProfileGateIsConfigNotHardware:
    """What opens the strict lane -- the premise the rest of this module rests on."""

    def test_a_default_engine_is_not_the_production_profile(self):
        assert not _plain()._is_large_n_gpu_production_profile()

    def test_two_constructor_arguments_open_it_on_cpu(self):
        """No GPU, no monkeypatching, no device query -- four config values."""
        assert _strict()._is_large_n_gpu_production_profile()

    def test_the_basis_is_part_of_the_contract(self):
        """`preset="large_n_gpu"` alone is not enough: its default basis is cartesian.

        Worth its own test because it is the trap -- setting the preset and
        assuming the profile is open leaves it closed, silently.
        """
        preset_only = FMMEngine(
            theta=0.6, working_dtype=jnp.float32, preset="large_n_gpu"
        )
        assert preset_only.expansion_basis == "cartesian"
        assert not preset_only._is_large_n_gpu_production_profile()


class TestFailFastGuards:
    """Each entry point rejects a wrong profile, and counts the rejection."""

    def test_strict_run_v2_rejects_a_non_production_profile(self, particles):
        engine = _plain()
        with pytest.raises(RuntimeError, match="large_n_gpu production profile"):
            engine.strict_run_v2(**_run_kwargs(*particles))
        assert engine._strict_v2_fail_fast_reject_count == 1

    def test_prepare_refresh_and_evaluate_rejects_a_non_production_profile(
        self, particles
    ):
        engine = _plain()
        with pytest.raises(RuntimeError, match="large_n_gpu production profile"):
            engine.strict_prepare_refresh_and_evaluate(None, *particles)
        assert engine._strict_runner_fail_fast_reject_count == 1

    def test_prepare_refresh_and_evaluate_rejects_a_foreign_prepared_state(
        self, particles
    ):
        """The lane's contract is a ``LargeNPreparedState``, not anything shaped like one."""
        engine = _strict()
        with pytest.raises(RuntimeError, match="LargeNPreparedState"):
            engine.strict_prepare_refresh_and_evaluate(object(), *particles)
        assert engine._strict_runner_fail_fast_reject_count == 1

    def test_the_two_lanes_count_their_rejections_separately(self, particles):
        """`strict_run_v2` and the prepare/refresh entry point are distinct lanes.

        A shared counter would make a rejection in one look like a rejection in
        the other, which is the sort of thing a profiling session reads as a
        different bug.
        """
        engine = _plain()
        with pytest.raises(RuntimeError):
            engine.strict_run_v2(**_run_kwargs(*particles))
        assert engine._strict_v2_fail_fast_reject_count == 1
        assert engine._strict_runner_fail_fast_reject_count == 0

    def test_refresh_prepared_state_raises_not_implemented_not_runtime(self, particles):
        """A different exception type from its siblings, deliberately.

        ``NotImplementedError`` says "this combination is not supported yet",
        where the sibling ``RuntimeError``s say "you called this wrongly". Pinned
        because a caller distinguishing the two would break if they converged.
        """
        engine = _plain()
        with pytest.raises(NotImplementedError, match="large_n_gpu"):
            engine.refresh_prepared_state(object(), *particles)


class TestArgumentValidation:
    """Guards that fire once the profile is open."""

    def test_num_steps_must_be_positive(self, particles):
        with pytest.raises(ValueError, match="num_steps must be positive"):
            _strict().strict_run_v2(**_run_kwargs(*particles, num_steps=0))

    @pytest.mark.parametrize("refresh_every", [2, 5])
    def test_refresh_every_must_be_one(self, particles, refresh_every):
        """A physics guard, not a performance knob.

        The velocity-Verlet update is endpoint-correct only when the state is
        refreshed every step. Refreshing less often would integrate against
        stale multipoles and still produce plausible-looking trajectories, which
        is the worst failure mode this library has.
        """
        engine = _strict()
        with pytest.raises(ValueError, match="endpoint-correct velocity-Verlet"):
            engine.strict_run_v2(
                **_run_kwargs(*particles, num_steps=2, refresh_every=refresh_every)
            )
        assert engine._strict_v2_fail_fast_reject_count == 1


class TestFusedModeRefusesToDegradeSilently:
    """Requested-but-not-allowed raises rather than quietly running the slow path."""

    @staticmethod
    def _engine_wanting_fused_at_another_n():
        engine = _strict()
        engine._strict_fused_mode_enabled = True
        engine._strict_fused_profile_set_raw = "999999"  # deliberately not _N
        return engine

    def test_it_raises_instead_of_falling_back(self, particles):
        engine = self._engine_wanting_fused_at_another_n()
        with pytest.raises(RuntimeError, match="refusing to silently fall back"):
            engine.strict_run_v2(**_run_kwargs(*particles))

    def test_the_message_names_the_variable_and_both_remedies(self, particles):
        """An operator has to be able to act on this without reading the source."""
        engine = self._engine_wanting_fused_at_another_n()
        with pytest.raises(RuntimeError) as excinfo:
            engine.strict_run_v2(**_run_kwargs(*particles))
        message = str(excinfo.value)
        assert "JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET" in message
        assert f"N={_N}" in message
        assert "Add this N to the profile set" in message
        assert "leave it empty to allow all N" in message

    def test_the_fallback_is_counted_and_its_reason_recorded(self, particles):
        engine = self._engine_wanting_fused_at_another_n()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                engine.strict_run_v2(**_run_kwargs(*particles))
        assert engine._strict_fused_fallback_count == 2
        assert (
            engine._strict_fused_last_fallback_reason
            == "particle_count_not_in_JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET"
        )

    def test_fused_mode_is_not_marked_active_when_it_was_refused(self, particles):
        engine = self._engine_wanting_fused_at_another_n()
        with pytest.raises(RuntimeError):
            engine.strict_run_v2(**_run_kwargs(*particles))
        assert not engine._strict_fused_mode_active


class TestProfileKeyAccounting:
    """The compile-count signal: a new profile key means a new compilation.

    Observed through the fused refusal above, which raises *after* the accounting
    and before any device work -- so these counters can be read on CPU without
    running a step.
    """

    @staticmethod
    def _engine():
        engine = _strict()
        engine._strict_fused_mode_enabled = True
        engine._strict_fused_profile_set_raw = "999999"
        return engine

    def _attempt(self, engine, particles, **overrides):
        with pytest.raises(RuntimeError):
            engine.strict_run_v2(**_run_kwargs(*particles, **overrides))

    def test_the_first_call_misses_and_the_second_hits(self, particles):
        engine = self._engine()
        self._attempt(engine, particles)
        assert (
            engine._strict_v2_profile_key_misses,
            engine._strict_v2_profile_key_hits,
        ) == (1, 0)
        self._attempt(engine, particles)
        assert (
            engine._strict_v2_profile_key_misses,
            engine._strict_v2_profile_key_hits,
        ) == (1, 1)

    def test_a_miss_is_what_counts_a_compile(self, particles):
        engine = self._engine()
        self._attempt(engine, particles)
        self._attempt(engine, particles)
        assert engine._strict_v2_compile_count == 1

    @pytest.mark.parametrize(
        "overrides",
        [
            {"leaf_size": 8},
            {"max_order": 3},
            {"dt": 0.2},
        ],
        ids=["leaf_size", "max_order", "dt"],
    )
    def test_each_shape_or_step_parameter_forms_a_new_key(self, particles, overrides):
        """These are baked into the compiled executable, so each must miss.

        ``dt`` is in the key for the same reason as the shapes: the strict lane
        bakes the step into the compiled step function.
        """
        engine = self._engine()
        self._attempt(engine, particles)
        self._attempt(engine, particles, **overrides)
        assert engine._strict_v2_profile_key_misses == 2
        assert engine._strict_v2_profile_key_hits == 0

    def test_every_attempt_counts_as_an_execution(self, particles):
        engine = self._engine()
        self._attempt(engine, particles)
        self._attempt(engine, particles)
        assert engine._strict_v2_execute_count == 2

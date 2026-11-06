"""The fused lane's CPU route (``RuntimePolicyConfig.cpu_fused_lane``), no solve.

What is pinned here:

* ``jaccpot._env.scoped_env_defaults``: a context-local default for an unset
  variable, which every ``_env`` reader (and ``env_snapshot``) sees; a value set
  in the process environment wins; ``os.environ`` is never written; the scope
  resets on exit.
* ``lane_policy_backend``: the real backend for every engine without the flag and
  on a GPU, ``"gpu"`` on a CPU with it -- so the engine's lane gates resolve the
  GPU policy there (the split dual-tree build default, the near-field mode), and
  on a card nothing changes.
* ``lane_platform_scope``: the interpret-mode CSR switches only on the route with
  ``use_pallas=True``.

The end-to-end check (the runner on the route, bitwise against the monkeypatched
backend jaccpot's other CPU tests use) is
``tests/integration/strict/test_cpu_fused_route.py``.
"""

from __future__ import annotations

import os

import jax
import pytest

from jaccpot import FastMultipoleMethod, FMMAdvancedConfig, RuntimePolicyConfig
from jaccpot._env import env_flag, env_float, env_snapshot, scoped_env_defaults
from jaccpot.runtime._lane_platform import (
    CPU_ROUTE_PALLAS_ENV,
    cpu_route_active,
    lane_platform_scope,
    lane_policy_backend,
)

_VAR = "JACCPOT_TEST_LANE_PLATFORM_SCOPED"


def _solver(*, route: bool, use_pallas: bool = False) -> FastMultipoleMethod:
    base = FastMultipoleMethod(preset="galaxy").advanced
    advanced = FMMAdvancedConfig(
        tree=base.tree,
        nearfield=base.nearfield,
        runtime=RuntimePolicyConfig(
            fixed_order=base.runtime.fixed_order, cpu_fused_lane=route
        ),
        mac_type=base.mac_type,
    )
    return FastMultipoleMethod(
        preset="galaxy", G=1.0, softening=1e-2, use_pallas=use_pallas, advanced=advanced
    )


class TestScopedEnvDefaults:
    def test_unset_variable_reads_the_scope(self, monkeypatch):
        monkeypatch.delenv(_VAR, raising=False)
        assert env_flag(_VAR, False) is False
        with scoped_env_defaults({_VAR: "1"}):
            assert env_flag(_VAR, False) is True
            assert _VAR not in os.environ
            assert env_snapshot(("JACCPOT_TEST_LANE",)) == {_VAR: "1"}
        assert env_flag(_VAR, False) is False
        assert env_snapshot(("JACCPOT_TEST_LANE",)) == {}

    def test_the_process_environment_wins(self, monkeypatch):
        monkeypatch.setenv(_VAR, "0")
        with scoped_env_defaults({_VAR: "1"}):
            assert env_flag(_VAR, True) is False
            assert env_snapshot(("JACCPOT_TEST_LANE",)) == {_VAR: "0"}

    def test_nested_scopes_merge_inner_wins(self, monkeypatch):
        monkeypatch.delenv(_VAR, raising=False)
        with scoped_env_defaults({_VAR: "2.5"}):
            with scoped_env_defaults({_VAR: "4.0"}):
                assert env_float(_VAR, 1.0) == 4.0
            assert env_float(_VAR, 1.0) == 2.5
        assert env_float(_VAR, 1.0) == 1.0


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the CPU route")
class TestLanePolicyBackend:
    def test_off_by_default(self):
        solver = _solver(route=False)
        assert solver._impl.cpu_fused_lane is False
        assert lane_policy_backend(solver) == "cpu"
        assert solver._impl._lane_backend() == "cpu"
        assert not cpu_route_active(solver)

    def test_route_reports_the_gpu_policy_on_a_cpu(self):
        solver = _solver(route=True)
        assert solver._impl.cpu_fused_lane is True
        assert lane_policy_backend(solver) == "gpu"
        assert solver._impl._lane_backend() == "gpu"
        assert cpu_route_active(solver)

    def test_route_engine_resolves_the_gpu_lane_policy(self, monkeypatch):
        """The route's engine = the engine built under a backend that says "gpu"."""
        routed = _solver(route=True)._impl
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        patched = _solver(route=False)._impl
        assert (
            routed._streamed_minimum_memory_gpu_default_split_build
            == patched._streamed_minimum_memory_gpu_default_split_build
        )
        assert routed._streamed_minimum_memory_gpu_default_split_build is True
        for n in (512, 1 << 20):
            assert routed._resolve_nearfield_mode(
                num_particles=n
            ) == patched._resolve_nearfield_mode(num_particles=n)

    def test_unrouted_engine_keeps_the_cpu_lane_policy(self):
        engine = _solver(route=False)._impl
        assert engine._streamed_minimum_memory_gpu_default_split_build is False

    def test_a_gpu_backend_ignores_the_flag(self, monkeypatch):
        """On a card the flag changes no lane gate (the GPU programs are unchanged)."""
        solver = _solver(route=True)
        monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
        assert lane_policy_backend(solver) == "gpu"
        assert not cpu_route_active(solver)
        with lane_platform_scope(solver):
            assert env_snapshot(("JACCPOT_NEARFIELD_PALLAS",)) == {
                k: v
                for k, v in os.environ.items()
                if k.startswith("JACCPOT_NEARFIELD_PALLAS")
            }


@pytest.mark.skipif(jax.default_backend() != "cpu", reason="the CPU route")
class TestLanePlatformScope:
    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for key in CPU_ROUTE_PALLAS_ENV:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.parametrize(
        ("route", "use_pallas"), [(False, False), (False, True), (True, False)]
    )
    def test_no_switches_off_the_route_or_without_pallas(self, route, use_pallas):
        solver = _solver(route=route, use_pallas=use_pallas)
        with lane_platform_scope(solver):
            for key in CPU_ROUTE_PALLAS_ENV:
                assert env_snapshot((key,)) == {}

    def test_route_with_pallas_scopes_the_interpret_csr_switches(self):
        solver = _solver(route=True, use_pallas=True)
        with lane_platform_scope(solver):
            assert env_flag("JACCPOT_NEARFIELD_PALLAS_INTERPRET", False) is True
            assert env_flag("JACCPOT_NEARFIELD_LEAFPAIR_CSR", False) is True
            assert env_float("JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB", 1024.0) == 0.0
            for key in CPU_ROUTE_PALLAS_ENV:
                assert key not in os.environ
        assert env_flag("JACCPOT_NEARFIELD_PALLAS_INTERPRET", False) is False

    def test_a_shell_value_wins(self, monkeypatch):
        monkeypatch.setenv("JACCPOT_NEARFIELD_LEAFPAIR_CSR", "0")
        solver = _solver(route=True, use_pallas=True)
        with lane_platform_scope(solver):
            assert env_flag("JACCPOT_NEARFIELD_LEAFPAIR_CSR", True) is False

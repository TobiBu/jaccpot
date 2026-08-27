"""The GPU gate's own checks must read the output format the gate produces.

``bench/gpu_gate.py`` exists to stop a vacuous GPU run from printing green. Its
two checks were parsing the wrong layout, so it could not do that:

* ``-v`` writes ``<node id> PASSED``, but ``-n`` writes
  ``[gw3] [ 42%] PASSED <node id>`` -- verdict first, behind a worker prefix. The
  gate always runs with ``-n 6``, so ``_check_not_vacuous`` matched nothing,
  reported all three GPU-gated files as "0 tests ran", and failed every run for a
  reason unrelated to the GPU.
* ``-rs`` replaced pytest's default ``-rfE``, removing every ``FAILED`` line from
  the short summary -- the only thing ``_classify_failures`` read. Real failures
  were invisible, and since the pytest exit code was printed but never consulted,
  a red suite could print ``GATE PASSED``.

Both were found by running the gate; neither could be caught by anything in CI,
because the gate is GPU-only and outside it. So the outputs below are *recorded*
from real pytest runs and replayed here on CPU, which is the only way these
checks get exercised at all.
"""

from __future__ import annotations

import pathlib

import pytest

from bench.gpu_gate import (
    _MUST_RUN_SM80,
    _check_not_vacuous,
    _classify_failures,
    _outcomes_by_node,
    _pytest_cmd,
)

# Recorded from `pytest -n 2 -rs -v` (pytest 9.1.1, xdist 3.8.0): verdict first,
# behind a `[gw<n>] [ <pct>%]` prefix, and the bare node id on its own line when
# the test starts.
XDIST_OUTPUT = """\
created: 2/2 workers
2 workers [4 items]

scheduling tests via LoadScheduling

tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a
tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a
[gw1] [ 25%] SKIPPED tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a
[gw0] [ 50%] PASSED tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a
tests/unit/test_large_n_config_thresholds.py::test_threshold_a
[gw0] [ 75%] PASSED tests/unit/test_large_n_config_thresholds.py::test_threshold_a
[gw1] [100%] FAILED tests/unit/runtime/test_split_build_default_predicate.py::test_pred_b
[gw0] [ 76%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_analytic_branch_holds_at_float32_on_the_fused_pallas_lanes[inside the band-2.5]
[gw0] [ 80%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[True]
[gw1] [ 84%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw1] [ 88%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw0] [ 92%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative
[gw1] [ 96%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative
=========================== short test summary info ============================
SKIPPED [1] tests/unit/runtime/test_split_build_default_predicate.py:4: needs gpu
==================== 1 failed, 2 passed, 1 skipped in 1.19s ====================
"""

# Recorded from `pytest -p no:xdist -v`: verdict last, no worker prefix.
PLAIN_OUTPUT = """\
tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a PASSED   [ 33%]
tests/unit/test_large_n_config_thresholds.py::test_threshold_a PASSED    [ 66%]
tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a FAILED [100%]
"""

# A CPU fallback: every GPU-gated test skips, and pytest still prints green.
VACUOUS_OUTPUT = """\
[gw0] [ 33%] SKIPPED tests/unit/test_large_n_config_thresholds.py::test_threshold_a
[gw1] [ 66%] SKIPPED tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a
[gw1] [100%] SKIPPED tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a
[gw0] [ 76%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_analytic_branch_holds_at_float32_on_the_fused_pallas_lanes[inside the band-2.5]
[gw0] [ 80%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[True]
[gw1] [ 84%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw1] [ 88%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw0] [ 92%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative
[gw1] [ 96%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative
==================== 3 skipped in 0.8s ====================
"""


# The blind spot this module gained a check for: a **real GPU**, but not Ampere+.
# Every backend-gated test runs and passes, so the original three-file assertion
# is satisfied and the gate printed green -- while all five fused-Pallas lanes
# self-skipped off sm_80 and asserted nothing. Note the
# `..._in_gradient[True]` line: the interpret half runs on any card, so a check
# keyed on the test *name* rather than the `[False]` parametrisation would find a
# passing test here and call the run non-vacuous.
NON_AMPERE_OUTPUT = """\
[gw0] [ 20%] PASSED tests/unit/test_large_n_config_thresholds.py::test_threshold_a
[gw1] [ 40%] PASSED tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a
[gw1] [ 50%] PASSED tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a
[gw0] [ 60%] PASSED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[True]
[gw0] [ 70%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_analytic_branch_holds_at_float32_on_the_fused_pallas_lanes[inside the band-2.5]
[gw1] [ 80%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw1] [ 85%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_fused_pallas_complex_m2l_matches_the_pure_jax_lane_in_gradient[False]
[gw0] [ 90%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_real_fused_m2l_kernel_carries_the_axis_derivative
[gw1] [100%] SKIPPED tests/unit/operators/test_transverse_degeneracy_jvp.py::test_the_production_complex_fused_m2l_kernel_carries_the_axis_derivative
=========================== short test summary info ============================
SKIPPED [5] the fused Pallas M2L lanes require an Ampere+ (sm_80) GPU
==================== 4 passed, 5 skipped in 61.2s ====================
"""


class TestOutcomeParsing:
    """``_outcomes_by_node`` must handle both layouts pytest emits."""

    def test_parses_xdist_prefix_layout(self) -> None:
        """Verdict-first lines behind a ``[gw<n>]`` prefix are parsed."""
        outcomes = _outcomes_by_node(XDIST_OUTPUT)
        assert outcomes[
            "tests/unit/runtime/test_nearfield_mode_policy.py::test_policy_a"
        ] == {"PASSED"}
        assert outcomes[
            "tests/unit/runtime/test_split_build_default_predicate.py::test_pred_b"
        ] == {"FAILED"}

    def test_parses_plain_suffix_layout(self) -> None:
        """Verdict-last lines from a non-xdist run are parsed."""
        outcomes = _outcomes_by_node(PLAIN_OUTPUT)
        assert outcomes[
            "tests/unit/test_large_n_config_thresholds.py::test_threshold_a"
        ] == {"PASSED"}
        assert outcomes[
            "tests/unit/runtime/test_split_build_default_predicate.py::test_pred_a"
        ] == {"FAILED"}

    def test_summary_and_progress_lines_collapse_to_one_node(self) -> None:
        """One test reported twice must not count as two."""
        doubled = (
            "[gw1] [100%] FAILED tests/unit/test_x.py::test_a\n"
            "FAILED tests/unit/test_x.py::test_a - AssertionError: nope\n"
        )
        assert list(_outcomes_by_node(doubled)) == ["tests/unit/test_x.py::test_a"]

    def test_node_ids_containing_spaces_are_not_truncated(self) -> None:
        """Parametrised ids have spaces in them, and truncation silently merges.

        Both ids below are real, from ``test_large_n_config_thresholds.py``.
        Splitting the line on whitespace cuts them at ``[delta0-exact-zero`` and
        the float32/float64 pair collapses to one key -- which undercounted the
        gate's own not-vacuous tally by 3 of 20 without any error.
        """
        stem = (
            "tests/unit/test_large_n_config_thresholds.py"
            "::test_translation_reverse_is_finite_at_degenerate_displacements"
        )
        f32 = f"{stem}[delta0-exact-zero displacement (single-child COM L2L)-float32]"
        f64 = f"{stem}[delta0-exact-zero displacement (single-child COM L2L)-float64]"
        outcomes = _outcomes_by_node(
            f"[gw3] [ 50%] PASSED {f32} \n[gw3] [ 55%] PASSED {f64} \n"
        )
        assert set(outcomes) == {f32, f64}

    def test_short_summary_reason_is_not_part_of_the_node_id(self) -> None:
        """``-rfEs`` appends ``- <reason>``, which must not join the id."""
        node = "tests/unit/test_x.py::test_a[a b]"
        outcomes = _outcomes_by_node(f"FAILED {node} - AssertionError: nope\n")
        assert set(outcomes) == {node}

    def test_plain_layout_with_a_progress_counter(self) -> None:
        """Non-xdist ``-v`` appends ``[ 33%]`` after the verdict."""
        node = "tests/unit/test_x.py::test_a[a b]"
        outcomes = _outcomes_by_node(f"{node} PASSED    [ 33%]\n")
        assert outcomes == {node: {"PASSED"}}

    def test_skip_summary_lines_are_not_mistaken_for_nodes(self) -> None:
        """``SKIPPED [1] path.py:4: reason`` carries no node id and is ignored."""
        line = "SKIPPED [1] tests/unit/test_x.py:4: needs gpu\n"
        assert _outcomes_by_node(line) == {}


class TestNotVacuous:
    """The check that makes a GPU run prove it was a GPU run."""

    def test_gated_tests_that_ran_are_accepted(self) -> None:
        """All three gated files report a run, so there is nothing to complain about."""
        assert _check_not_vacuous(XDIST_OUTPUT) == []

    def test_all_skipped_is_reported_as_vacuous(self) -> None:
        """A CPU fallback empties both gated sets and must fail the gate.

        Eight, not three: the three backend-gated files plus the five sm_80-gated
        node ids. The ``[True]`` interpret half is recorded as PASSED because on
        CPU it really does run -- and the ``[False]`` fragment still complains,
        which is the parametrisation trap the node-id form exists to avoid.
        """
        complaints = _check_not_vacuous(VACUOUS_OUTPUT)
        assert len(complaints) == 8
        assert all("0 tests ran" in c for c in complaints)

    def test_moving_a_gated_file_does_not_empty_the_check(self) -> None:
        """``_MUST_RUN`` holds bare filenames, so a directory move still matches.

        Two of the three moved into ``tests/unit/runtime/`` during the Tier 1
        refactor; the recorded output above already reflects that.
        """
        assert "runtime/test_nearfield_mode_policy.py" in XDIST_OUTPUT
        assert _check_not_vacuous(XDIST_OUTPUT) == []


class TestTheNonAmpereBlindSpot:
    """A real GPU that is not Ampere+ must not read as a validated run.

    This is the case the gate missed. ``_MUST_RUN``'s three files are all
    backend-gated, so they pass on any GPU; the fused Pallas M2L lanes are gated
    on ``sm_80`` instead and self-skip on, say, a V100. Every check the gate had
    was satisfied and it printed green, while the five tests
    ``docs/handoff_g10_gpu_validation.md`` calls "the headline item" asserted
    nothing.
    """

    def test_a_non_ampere_run_is_reported_as_vacuous(self) -> None:
        """Backend tests green, sm_80 lanes skipped -> five complaints."""
        complaints = _check_not_vacuous(NON_AMPERE_OUTPUT)
        assert len(complaints) == 5, complaints
        assert all("sm_80" in c for c in complaints)

    def test_the_interpret_half_passing_does_not_excuse_the_kernel_half(self) -> None:
        """The trap a name-keyed check would fall into.

        ``test_fused_pallas_m2l_matches_the_pure_jax_lane_in_gradient[True]``
        runs in interpret mode on any card and PASSES in the recorded output. A
        fragment keyed on the test name alone would match it, count a run, and
        declare the lane covered -- while ``[False]``, the half that reaches the
        real Triton kernel, skipped.
        """
        assert "_in_gradient[True]" in NON_AMPERE_OUTPUT
        complaints = _check_not_vacuous(NON_AMPERE_OUTPUT)
        assert any("_in_gradient[False]" in c for c in complaints)

    def test_a_full_ampere_run_leaves_nothing_to_complain_about(self) -> None:
        """The positive control: with the lanes actually run, no complaint."""
        assert _check_not_vacuous(XDIST_OUTPUT) == []


class TestSm80RegistryTracksTheSuite:
    """Every sm_80-gated test must be registered, or the blind spot reopens.

    The failure mode is not that a fragment goes stale -- a fragment matching
    nothing reports "0 tests ran", which fails loudly. It is that somebody adds a
    *new* sm_80-gated test and does not register it, which is exactly how the
    original five came to be unchecked. So this derives the gated set from the
    test module's source and compares.
    """

    GATED_MODULE = "tests/unit/operators/test_transverse_degeneracy_jvp.py"

    def _sm80_gated_functions(self) -> set[str]:
        """Every test function that can skip itself off sm_80.

        Returns
        -------
        set[str]
            Function names whose body contains a ``pytest.skip`` mentioning
            ``sm_80``. Read from the AST rather than by grepping, so a match
            inside a docstring or comment cannot inflate the set.
        """
        import ast

        tree = ast.parse(pathlib.Path(self.GATED_MODULE).read_text())
        gated = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                target = getattr(call.func, "attr", None)
                if target != "skip" or not call.args:
                    continue
                arg = call.args[0]
                if isinstance(arg, ast.Constant) and "sm_80" in str(arg.value):
                    gated.add(node.name)
        return gated

    def test_the_gated_module_still_exists(self) -> None:
        """A rename would make the guard below pass by finding nothing."""
        assert pathlib.Path(self.GATED_MODULE).is_file(), self.GATED_MODULE

    def test_every_sm80_gated_test_is_registered(self) -> None:
        """No sm_80-gated test may be missing from ``_MUST_RUN_SM80``."""
        gated = self._sm80_gated_functions()
        assert gated, "found no sm_80 gate at all -- the AST scan has drifted"
        unregistered = {
            name
            for name in gated
            if not any(name in fragment for fragment in _MUST_RUN_SM80)
        }
        assert not unregistered, (
            f"sm_80-gated but unchecked by the GPU gate: {sorted(unregistered)}. "
            "Add the node id (with its parametrisation) to _MUST_RUN_SM80 in "
            "bench/gpu_gate.py, or a non-Ampere run will report green."
        )

    def test_every_registered_fragment_names_a_real_test(self) -> None:
        """And the reverse: a fragment must correspond to a function that exists."""
        source = pathlib.Path(self.GATED_MODULE).read_text()
        for fragment in _MUST_RUN_SM80:
            name = fragment.split("[")[0]
            assert (
                f"def {name}(" in source
            ), f"{fragment} names no test in {self.GATED_MODULE}"


class TestClassifyFailures:
    """Known §9 failures are allowed through; anything else is not."""

    def test_unexpected_failure_is_reported(self) -> None:
        """A failure outside the documented list must surface as unexpected."""
        unexpected, known = _classify_failures(XDIST_OUTPUT)
        assert unexpected == [
            "tests/unit/runtime/test_split_build_default_predicate.py::test_pred_b"
        ]
        assert known == []

    def test_failure_is_found_without_a_short_summary_line(self) -> None:
        """The progress line alone is enough.

        This is the ``-rs`` bug: the short summary listed only skips, so a parser
        reading ``^FAILED`` saw a clean run.
        """
        no_summary = (
            "[gw1] [100%] FAILED tests/unit/test_x.py::test_a\n"
            "==================== 1 failed in 1.0s ====================\n"
        )
        unexpected, known = _classify_failures(no_summary)
        assert unexpected == ["tests/unit/test_x.py::test_a"]

    @pytest.mark.parametrize(
        "node",
        [
            "tests/characterization/test_constructor_state_golden.py"
            "::test_constructor_state_matches_the_committed_golden",
            "tests/unit/runtime/test_potential_stays_on_fast_lane.py"
            "::test_potential_does_not_delegate_to_the_generic_path",
            "tests/unit/runtime/test_fb_force_scale_estimator.py"
            "::test_the_far_field_term_is_load_bearing",
            "tests/unit/runtime/test_tree_geometry_compiled.py"
            "::test_compiled_dispatch_is_bit_identical[32]",
        ],
    )
    def test_documented_gpu_failures_are_allowed(self, node: str) -> None:
        """ARCHITECTURE §9's measured failures are known, not unexpected."""
        unexpected, known = _classify_failures(f"[gw0] [ 10%] FAILED {node}\n")
        assert known == [node]
        assert unexpected == []

    @pytest.mark.parametrize(
        "node",
        [
            "tests/characterization/test_fmm_grad_golden.py"
            "::test_fmm_grad_golden[clu_real_n128_p4]",
            "tests/integration/test_fmm.py::test_solidfmm_chunked_m2l_matches_fullbatch",
            "tests/integration/test_real_basis_runtime.py"
            "::test_real_basis_tracks_complex_basis[nearfield-only-f32]",
        ],
    )
    def test_failures_fixed_upstream_are_no_longer_tolerated(self, node: str) -> None:
        """A fixed failure must leave the tuple, or a regression reads as "known".

        These three were on §9's list and were fixed on 2026-08-14 by `aacd3cf`
        and `86163f1`. Leaving them behind is the dangerous direction: the entry
        stays, the test regresses later, and the gate classifies the regression as
        expected and passes. This is the assertion that makes removing them stick.
        """
        unexpected, known = _classify_failures(f"[gw0] [ 10%] FAILED {node}\n")
        assert known == []
        assert unexpected == [node]

    def test_errors_count_as_failures(self) -> None:
        """A collection or fixture ERROR must not be silently tolerated."""
        unexpected, _ = _classify_failures("[gw0] [ 10%] ERROR tests/u/test_x.py::t\n")
        assert unexpected == ["tests/u/test_x.py::t"]


class TestPytestCommand:
    """The flags the gate runs with are part of its contract."""

    def test_short_summary_includes_failures_and_errors(self) -> None:
        """``-r`` must name ``f`` and ``E``, not just ``s``.

        Passing ``-r`` at all replaces pytest's default ``fE``, so the original
        ``-rs`` switched off the very lines the failure classifier reads.
        """
        cmd = _pytest_cmd([])
        r_flags = [a for a in cmd if a.startswith("-r")]
        assert r_flags, "the gate must ask for a short test summary"
        assert all("f" in f and "E" in f and "s" in f for f in r_flags)

    def test_verbosity_is_absolute_not_a_counter(self) -> None:
        """`-v` is not enough: `addopts` carries `-q` and the two cancel.

        Measured on the 2026-08-20 gate run: with `-v`, pytest printed progress
        dots and not one per-test line, so the not-vacuous check had nothing to
        read whatever layout it expected. `--verbosity=N` is absolute and cannot
        be cancelled by a counter flag added to `addopts` later.
        """
        cmd = _pytest_cmd([])
        assert any(a.startswith("--verbosity=") for a in cmd), (
            "the gate must set an absolute verbosity; a bare -v is cancelled by "
            "the -q in pyproject.toml's addopts"
        )
        assert "-v" not in cmd

    def test_worker_count_is_capped(self) -> None:
        """``-n auto`` on this box means 64 workers on one card; see CLAUDE.md."""
        cmd = _pytest_cmd([])
        assert "-n" in cmd
        assert cmd[cmd.index("-n") + 1] == "6"

    def test_extra_args_come_last(self) -> None:
        """Caller arguments must be able to override the gate's own."""
        assert _pytest_cmd(["-x", "-k", "foo"])[-3:] == ["-x", "-k", "foo"]

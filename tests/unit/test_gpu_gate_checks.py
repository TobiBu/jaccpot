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

import pytest

from bench.gpu_gate import (
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
==================== 3 skipped in 0.8s ====================
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
        """A CPU fallback skips all three files and must fail the gate."""
        complaints = _check_not_vacuous(VACUOUS_OUTPUT)
        assert len(complaints) == 3
        assert all("0 tests ran" in c for c in complaints)

    def test_moving_a_gated_file_does_not_empty_the_check(self) -> None:
        """``_MUST_RUN`` holds bare filenames, so a directory move still matches.

        Two of the three moved into ``tests/unit/runtime/`` during the Tier 1
        refactor; the recorded output above already reflects that.
        """
        assert "runtime/test_nearfield_mode_policy.py" in XDIST_OUTPUT
        assert _check_not_vacuous(XDIST_OUTPUT) == []


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
            "tests/characterization/test_fmm_grad_golden.py::test_fmm_grad_golden[clu_real_n128_p4]",
            "tests/unit/test_solidfmm.py::test_solidfmm_chunked_m2l_matches_fullbatch",
            "tests/unit/test_dispatch.py::test_compiled_dispatch_is_bit_identical[32]",
        ],
    )
    def test_documented_gpu_failures_are_allowed(self, node: str) -> None:
        """ARCHITECTURE §9's measured failures are known, not unexpected."""
        unexpected, known = _classify_failures(f"[gw0] [ 10%] FAILED {node}\n")
        assert known == [node]
        assert unexpected == []

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

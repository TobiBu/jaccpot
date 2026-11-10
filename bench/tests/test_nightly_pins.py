"""`bench/nightly_pins.py`: the verdicts, the references and the reruns it chooses.

The driver itself needs cards, autocvd and the nightly worktrees; these are the
decisions inside it, on the shapes ``bench/dce_pins.py compare --json`` writes.
"""

from __future__ import annotations

import pytest

from bench.nightly_pins import (
    already_passed,
    expand_items,
    find_reference,
    is_compiling,
    judge_bitwise,
    judge_envelope,
    judge_gate,
    judge_timed,
    overall_status,
    parse_autocvd,
)

_KEY_A = {"jaccpot": "a" * 40, "jax": "0.11.2"}
_KEY_B = {"jaccpot": "b" * 40, "jax": "0.11.2"}


def _run(tag, started, key, **items):
    return {"tag": tag, "started": started, "key": key, "items": items}


def _bitwise(*names):
    return {n: {"bitwise": True, "max_abs": 0.0, "rel_l2": 0.0} for n in names}


# ------------------------------------------------------------ which reference


def test_the_reference_is_the_latest_record_on_another_stack():
    runs = [
        _run("n1", "2026-10-14T02:30", _KEY_A, S1={"recorded": True}),
        _run("n2", "2026-10-15T02:30", _KEY_B, S1={"recorded": True}),
        _run("n3", "2026-10-16T02:30", _KEY_A, S1={"recorded": True}),
    ]
    # tonight is on stack B: n3 (A) is newer than n2 (B, the same stack)
    assert find_reference(runs, "S1", _KEY_B)["tag"] == "n3"
    # tonight on A: the latest record on another stack is n2
    assert find_reference(runs, "S1", _KEY_A)["tag"] == "n2"


def test_a_run_that_did_not_record_the_pin_is_no_reference():
    runs = [
        _run("n1", "2026-10-14T02:30", _KEY_A, S1={"recorded": True}),
        _run("n2", "2026-10-15T02:30", _KEY_A, S1={"recorded": False}, S2={}),
    ]
    assert find_reference(runs, "S1", _KEY_B)["tag"] == "n1"
    assert find_reference(runs, "S2", _KEY_B) is None
    assert find_reference([], "S1", _KEY_B) is None


def test_m2_needs_a_reference_with_its_own_control():
    runs = [
        _run("n1", "1", _KEY_A, M2={"recorded": True, "control_recorded": True}),
        _run("n2", "2", _KEY_A, M2={"recorded": True, "control_recorded": False}),
    ]
    assert find_reference(runs, "M2", _KEY_B, need_control=True)["tag"] == "n1"
    assert find_reference(runs, "M2", _KEY_B)["tag"] == "n2"


def test_only_passes_on_the_identical_stack_are_not_rerun():
    runs = [
        _run(
            "n1",
            "1",
            _KEY_A,
            S1={"verdict": "PASS"},
            S2={"verdict": "NEW"},
            S3={"verdict": "FAIL"},
            L1={"verdict": "SKIP"},
            L2={"verdict": "INCONCLUSIVE"},
        ),
        _run("n2", "2", _KEY_B, S4={"verdict": "PASS"}),
    ]
    assert already_passed(runs, _KEY_A) == {"S1", "S2"}
    assert already_passed(runs, dict(_KEY_A, jax="0.11.3")) == set()


# ------------------------------------------------------------ the verdicts


def test_bitwise_pins_pass_only_on_bitwise_arrays_and_equal_metadata():
    meta = {"ref": {"rel_l2": 1e-3, "n": 200000}, "new": {"rel_l2": 1e-3, "n": 200000}}
    entry = {"arrays": _bitwise("force", "state"), "meta": meta}
    assert judge_bitwise(entry) == ("PASS", "2 arrays bitwise")

    moved = {
        "arrays": {
            **_bitwise("force"),
            "state": {"bitwise": False, "max_abs": 2.4e-7, "rel_l2": 5e-10},
        },
        "meta": meta,
    }
    verdict, detail = judge_bitwise(moved)
    assert verdict == "FAIL" and "state" in detail and "2.4e-07" in detail

    # bitwise arrays, but the record's scalars differ (a raise turned into "ok")
    meta_moved = {"ref": {"n30000": "ok"}, "new": {"n30000": "raises ValueError"}}
    assert judge_bitwise({"arrays": {}, "meta": meta_moved})[0] == "FAIL"


@pytest.mark.parametrize(
    "entry",
    [
        None,
        {"missing": True},
        {"missing_in_ref": True},
        {"arrays": {"acc": {"missing": True}}, "meta": {}},
    ],
)
def test_a_missing_record_fails(entry):
    assert judge_bitwise(entry)[0] == "FAIL"
    assert judge_envelope(entry)[0] == "FAIL"


def test_envelope_pins_follow_dce_pins_pass_flag():
    within = {
        "arrays": {
            "positions_step16": {"bitwise": False, "max_abs": 1e-7, "rel_l2": 1e-10}
        },
        "pass": True,
    }
    verdict, detail = judge_envelope(within)
    assert verdict == "PASS" and "within the control" in detail
    assert judge_envelope(dict(within, **{"pass": False}))[0] == "FAIL"
    assert judge_envelope({"arrays": _bitwise("g_pos"), "pass": True}) == (
        "PASS",
        "1 arrays bitwise",
    )


def _timed(passed: bool, contaminated: bool) -> dict:
    row = {
        "ms_ref": 78.3,
        "ms_new": 82.0,
        "time_change": 0.047,
        "time_limit": 0.03,
        "peak_change": 0.0,
        "peak_limit": 0.02,
        "contaminated": contaminated,
    }
    return {
        "rows": {"step": row},
        "pass": passed,
        "meta": {"new": {"step": {"bytes_per_particle": 317.2}}},
    }


def test_timing_pins_a_contaminated_row_is_inconclusive_not_a_failure():
    assert judge_timed(_timed(True, False))[0] == "PASS"
    assert judge_timed(_timed(False, False))[0] == "FAIL"
    verdict, detail = judge_timed(_timed(False, True))
    assert verdict == "INCONCLUSIVE" and "contaminated" in detail
    assert "82.0 ms" in detail and "317 B/p" in detail


def test_the_gate_passes_on_its_exit_code_and_reports_its_own_lines():
    log = "\n".join(
        [
            "========= 1234 passed, 56 skipped in 1800.00s =========",
            "  known GPU failures (ARCHITECTURE section 9), not yours: 2",
            "GATE PASSED (pytest exit 1; known-failure allowance applied)",
        ]
    )
    verdict, detail = judge_gate(0, log)
    assert verdict == "PASS" and "GATE PASSED" in detail and "1234 passed" in detail
    verdict, detail = judge_gate(1, log.replace("PASSED (", "FAILED ("))
    assert verdict == "FAIL" and detail.startswith("exit 1")


def test_overall_status():
    assert (
        overall_status({"S1": {"verdict": "PASS"}, "S2": {"verdict": "NEW"}}) == "PASS"
    )
    assert (
        overall_status({"S1": {"verdict": "PASS"}, "L1": {"verdict": "SKIP"}})
        == "PARTIAL"
    )
    assert (
        overall_status({"S1": {"verdict": "FAIL"}, "L1": {"verdict": "SKIP"}}) == "FAIL"
    )


# ------------------------------------------------------------ plumbing


def test_pins_and_groups_expand_in_run_order():
    assert expand_items("DET1,S1") == ["S1", "DET1"]
    assert expand_items("M,gate") == ["M1", "M2", "M3", "gate"]
    assert len(expand_items("all")) == 16
    with pytest.raises(SystemExit):
        expand_items("S7")


def test_autocvd_export_output_is_parsed_and_an_empty_one_refused():
    text = "export CUDA_DEVICE_ORDER=PCI_BUS_ID export CUDA_VISIBLE_DEVICES=3,5\n"
    assert parse_autocvd(text) == {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "3,5",
    }
    # an autocvd timeout prints nothing: the run must not fall through to the CPU
    with pytest.raises(ValueError):
        parse_autocvd("")


def test_a_compiling_stack_is_progress_and_a_collective_hang_is_not():
    compiling = "Thread 1 (active)\n    CompileModuleToLlvmIr (libjax.so)\n"
    pyspy = "    backend_compile (jax/_src/compiler.py:300)\n"
    hang = (
        "Thread 1 (idle)\n    BlockUntilReady (xla_extension.so)\n"
        "Thread 9 (idle)\n    tsl::thread::EigenEnvironment::WorkLoop\n"
    )
    assert is_compiling(compiling) and is_compiling(pyspy)
    assert not is_compiling(hang)

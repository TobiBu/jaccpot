"""The A4 result schema and its record / check comparison (``jaccpot.verification.baseline``)."""

from __future__ import annotations

import json
import math

import pytest

from jaccpot.verification import (
    SCHEMA_VERSION,
    Tolerance,
    check_result,
    load_result,
    make_result,
    save_result,
)


def _result(**metrics: float) -> dict:
    base = {"force.rel_l2": 1.0e-4, "energy.max_abs_dE": 2.0e-5, "mass.lagr_r50": 0.77}
    base.update(metrics)
    return make_result(
        "plummer",
        {"num_particles": 4096, "dt": 0.01, "mac": "dehnen_error", "note": "x"},
        base,
        comparable=("num_particles", "dt", "mac"),
        tolerances={
            "force.rel_l2": Tolerance("ratio", 1.5, "an error may grow by 1.5x"),
            "energy.max_abs_dE": Tolerance("ceiling", 1e-4, "plan U4"),
            "mass.lagr_r50": Tolerance("rel", 0.02),
        },
        series={"t": [0.0, 1.0]},
        provenance={"commit": "abc"},
    )


def test_round_trip_and_self_check(tmp_path) -> None:
    res = _result()
    path = save_result(res, tmp_path / "sub" / "b.json")
    back = load_result(path)
    assert back["schema"] == SCHEMA_VERSION and back == json.loads(json.dumps(res))
    out = check_result(back, res)
    assert out.passed and len(out.rows) == 3 and "CHECK PASS" in out.table()


def test_a_perturbed_run_fails() -> None:
    base = _result()
    worse = _result(**{"force.rel_l2": 1.6e-4})
    out = check_result(base, worse)
    assert not out.passed
    failed = [r["metric"] for r in out.rows if not r["ok"]]
    assert failed == ["force.rel_l2"] and "CHECK FAIL" in out.table()
    assert not check_result(base, _result(**{"energy.max_abs_dE": 2e-4})).passed
    assert not check_result(base, _result(**{"mass.lagr_r50": 0.80})).passed
    assert not check_result(base, _result(**{"force.rel_l2": math.nan})).passed
    # better is fine for ratio / ceiling
    assert check_result(base, _result(**{"force.rel_l2": 1e-6})).passed


def test_incomparable_configs_and_missing_metrics_fail() -> None:
    base = _result()
    other = _result()
    other["config"]["dt"] = 0.04
    out = check_result(base, other)
    assert not out.passed and out.mismatched_config == {"dt": (0.01, 0.04)}
    other = _result()
    other["config"]["note"] = "y"  # not a comparable key
    assert check_result(base, other).passed
    other = _result()
    del other["metrics"]["force.rel_l2"]
    out = check_result(base, other)
    assert not out.passed and out.missing == ["force.rel_l2"]
    other = _result()
    other["problem"] = "hernquist"
    assert not check_result(base, other).passed


def test_tolerance_kinds_and_schema_errors(tmp_path) -> None:
    assert Tolerance("abs", 0.1).holds(1.05, 1.0)
    assert not Tolerance("abs", 0.01).holds(1.05, 1.0)
    assert Tolerance("floor", 2.0).holds(2.5, None)
    assert not Tolerance("floor", 2.0).holds(1.5, None)
    assert not Tolerance("rel", 0.1).holds(1.0, None)
    with pytest.raises(ValueError):
        Tolerance("bogus", 1.0)
    with pytest.raises(KeyError):
        make_result("p", {}, {"a": 1.0}, comparable=("n",))
    with pytest.raises(KeyError):
        make_result("p", {}, {"a": 1.0}, tolerances={"b": Tolerance("abs", 1.0)})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "other/0"}))
    with pytest.raises(ValueError, match="schema"):
        load_result(bad)

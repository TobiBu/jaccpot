"""WP-G G5's bench tooling, checked without a card or jax.

``bench/multigpu_grad_matrix.py`` (the driver) and ``bench/multigpu_grad_gate.py``
(one row): the matrix's rows parse as the gate's arguments, the flags each row's
process gets, the resume rules, and the summary of finished, failed and
out-of-memory rows.
"""

from __future__ import annotations

import json

import pytest

from bench import multigpu_grad_gate as gate
from bench import multigpu_grad_matrix as drv
from bench._ragged_exchange_flags import RAGGED_EXCHANGE_XLA_FLAGS


def _args(*extra):
    return drv._args(["--out-dir", "unused", *extra])


def test_the_matrix_rows_and_their_tiers():
    rows, aliases = drv.matrix(_args(), ncards=4)
    tags = [r["tag"] for r in rows]
    assert len(tags) == len(set(tags)) == 51
    count = {t: sum(r["tier"] == t for r in rows) for t in drv.TIERS}
    assert count == {"weak": 16, "strong": 9, "memory": 4, "check": 22}
    # the strong rows a weak row already is run once
    assert aliases == {
        "strong_P1.6e7_solo": "weak_P1.6e7pc_solo",
        "strong_P1.6e7_1c": "weak_P1.6e7pc_1c",
        "strong_P1.6e7_4c": "weak_P4e6pc_4c",
    }
    # two cards: the four-card rows go, the memory rows run on both cards
    rows2, _ = drv.matrix(_args("--tiers", "memory,weak"), ncards=2)
    assert all(r["cards"] <= 2 for r in rows2)
    assert {r["cards"] for r in rows2 if r["tier"] == "memory"} == {2}


def test_every_row_parses_as_the_gates_arguments():
    for preset in ("galaxy", "geometric"):
        rows, _ = drv.matrix(_args("--preset", preset, "--disc-ic", "d.npz"), ncards=4)
        for row in rows:
            a = gate._args([*row["gate_args"], "--out", "x.json"])
            assert a.ndev == (1 if a.mode == "solo" else row["cards"])
            assert a.galaxy == (
                preset == "galaxy" and not row["tag"].startswith("check_P_geo")
            )
            if a.ic == "disc":
                assert a.disc_ic == "d.npz"
            if row["tier"] == "memory":
                assert a.grad_only and a.no_forward and a.reps == 3
                assert row["env"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
            if row["tier"] == "check":
                assert a.check and a.n == drv.CHECK_N


def test_each_rows_process_gets_the_flags(monkeypatch):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")
    rows, _ = drv.matrix(_args(), ncards=4)
    for row in rows:
        env = drv._row_env(row, ["3", "5", "6", "7"][: row["cards"]])
        flags = env["XLA_FLAGS"].split()
        assert flags[0] == "--xla_dump_to=/tmp/x" and drv.AUTOTUNER_OFF in flags
        ragged = set(RAGGED_EXCHANGE_XLA_FLAGS.split())
        assert ragged <= set(flags) if row["cards"] >= 2 else not ragged & set(flags)
        assert env["CUDA_VISIBLE_DEVICES"] == ",".join(
            ["3", "5", "6", "7"][: row["cards"]]
        )


def test_the_gate_adds_missing_flags_and_keeps_an_override():
    a = gate._args(["--n", "1e6", "--ndev", "2", "--out", "x"])
    env: dict = {}
    gate.prepare_env(a, env)
    assert gate.AUTOTUNER_OFF in env["XLA_FLAGS"]
    assert set(RAGGED_EXCHANGE_XLA_FLAGS.split()) <= set(env["XLA_FLAGS"].split())
    one_shot = RAGGED_EXCHANGE_XLA_FLAGS.split()[0].split("=")[0]
    env = {"XLA_FLAGS": f"{one_shot}=true"}
    gate.prepare_env(a, env)
    assert f"{one_shot}=true" in env["XLA_FLAGS"].split()
    # one card: no exchange, no ragged flags
    env = {}
    gate.prepare_env(gate._args(["--n", "1e6", "--out", "x"]), env)
    assert "ragged" not in env["XLA_FLAGS"]


def test_cpu_smoke_env_is_the_cpu_gates_set():
    a = gate._args(["--n", "2048", "--ndev", "2", "--cpu-smoke", "--out", "x"])
    env = {"JACCPOT_CROSS_WALK": "tiles", "CUDA_VISIBLE_DEVICES": "3"}
    gate.prepare_env(a, env)
    assert "JACCPOT_CROSS_WALK" not in env
    assert all(env[k] == v for k, v in gate.SMOKE_ENV.items())
    assert env["JAX_PLATFORMS"] == "cpu" and env["CUDA_VISIBLE_DEVICES"] == ""
    assert "--xla_force_host_platform_device_count=2" in env["XLA_FLAGS"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--n", "1e5", "--galaxy", "--params"],
        ["--n", "1e5", "--galaxy", "--order", "4"],
        ["--n", "1e5", "--mode", "solo", "--ndev", "2"],
        ["--n", "4e6", "--check"],
        ["--n", "1e5", "--mode", "solo", "--check"],
        ["--n", "1e5", "--leaf-capacity", "auto"],
        ["--n", "2e7", "--ic", "disc", "--seed", "1"],
    ],
)
def test_the_gate_refuses_rows_it_cannot_measure(argv):
    with pytest.raises(SystemExit):
        gate._args([*argv, "--out", "x"])


def _write(out, tag, record=None, smi=None):
    if record is not None:
        (out / f"{tag}.json").write_text(json.dumps(record))
    if smi is not None:
        (out / f"{tag}.smi.txt").write_text(json.dumps(smi))


def _done(**kw):
    record = dict(
        mode="mesh",
        ic="plummer",
        n=4_000_000,
        ndev=2,
        n_per_card=2e6,
        mac="dehnen_error",
        order=6,
        complete=True,
        phase="timed",
        MG1=dict(differing=0),
        force=dict(bits_ok=True),
        grad_bits=dict(grad_ok=True),
        grad_finite=True,
        cross_pairs=dict(far_pairs=[10, 12], near_pairs=[3, 4], nonempty=True),
        time_ms=dict(
            force=[10.0, 11.0],
            force_b=[9.0, 10.0],
            main=[40.0, 41.0],
            main_b=[45.0, 46.0],
            grad=[30.0, 31.0],
            main_is="value_and_grad",
        ),
        ratio_value_and_grad_over_force=40.0 / 9.0,
        ratio_grad_over_force=30.0 / 9.0,
        peak_B_per_p=[400.0, 420.0],
        memory_analysis=dict(grad=dict(total_B_per_p=300.0)),
        grad_AvsA_differing=[0, 0],
        compile_s=dict(force=20.0, grad=60.0),
    )
    record.update(kw)
    return record


def test_the_summary_of_done_failed_and_oom_rows(tmp_path):
    _write(
        tmp_path, "weak_P2e6pc_2c", _done(), dict(tier="weak", rc=0, contended=False)
    )
    _write(
        tmp_path,
        "check_P_s0_2c",
        _done(check=dict(x_mesh_vs_direct=1e-4, x_ratio=1.02, MG9=dict(rel=1e-7))),
        dict(tier="check", rc=0, contended=True),
    )
    _write(
        tmp_path,
        "memory_P5e7pc_4c",
        dict(mode="mesh", ndev=4, complete=False, phase="forward"),
        dict(tier="memory", rc=1, oom=True, contended=False),
    )
    _write(tmp_path, "memory_P6e7pc_4c", None, dict(tier="memory", rc="timeout"))
    (tmp_path / "matrix.json").write_text(json.dumps(dict(aliases={"a": "b"})))
    rows = {r["tag"]: r for r in drv.summary(tmp_path)}
    assert set(rows) == {
        "weak_P2e6pc_2c",
        "check_P_s0_2c",
        "memory_P5e7pc_4c",
        "memory_P6e7pc_4c",
    }
    done = rows["weak_P2e6pc_2c"]
    assert (
        done["force_ms"] == 9.0 and done["main_ms"] == 40.0 and done["grad_ms"] == 30.0
    )
    assert (
        done["cross_far"] == 12
        and done["peak_bpp"] == 420.0
        and done["compile_s"] == 80.0
    )
    assert (
        rows["check_P_s0_2c"]["x_ratio"] == 1.02
        and rows["check_P_s0_2c"]["mg9_rel"] == 1e-7
    )
    assert drv._status(rows["memory_P5e7pc_4c"]) == "OOM (forward)"
    assert drv._status(rows["memory_P6e7pc_4c"]) == "rc timeout at start"
    text = (tmp_path / "SUMMARY.md").read_text()
    assert "| weak_P2e6pc_2c | 2 |" in text and "no (contended)" in text
    assert "- a = b" in text
    lines = (tmp_path / "rows.jsonl").read_text().splitlines()
    assert len(lines) == 4


def test_resume_skips_done_and_tried_rows_and_stops_the_memory_tier(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    ran = []

    def fake_run(row, devices, args):
        ran.append(row["tag"])
        rc = 1 if row["tag"] == "memory_P3e7pc_4c" else 0
        return dict(tag=row["tag"], rc=rc)

    monkeypatch.setattr(drv, "_run_row", fake_run)
    monkeypatch.setattr(drv, "summary", lambda out: [])
    _write(tmp_path, "check_P_s0_1c", _done(), dict(rc=0))  # done
    _write(tmp_path, "check_P_s0_2c", dict(complete=False), dict(rc=1))  # tried, failed
    argv = ["--out-dir", str(tmp_path), "--tiers", "memory,check"]
    drv.main(argv)
    assert "check_P_s0_1c" not in ran and "check_P_s0_2c" not in ran
    # the memory tier stops after its first failure
    assert [t for t in ran if t.startswith("memory")] == [
        "memory_P2e7pc_4c",
        "memory_P3e7pc_4c",
    ]
    ran.clear()
    drv.main([*argv, "--retry-failed", "--tiers", "check"])
    assert "check_P_s0_2c" in ran and "check_P_s0_1c" not in ran

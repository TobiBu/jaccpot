"""Every paper benchmark script runs end to end, on CPU, at a tiny N.

This guards the *harness*, not the physics: that each script parses its
arguments, reaches the solver, and writes a well-formed artifact. The numbers
these runs produce are meaningless -- at N in the hundreds the far field is
usually empty and the error sits at round-off -- and nothing here asserts on
them. A figure's numbers come from the real sweeps in ``results/``.

Why that is worth a test anyway: the failure mode these scripts have is not a
wrong number, it is a script that has silently stopped running at all. Two
already happened while this branch was being written -- jaxFMM moved its public
API, so every jaxfmm row had been recorded as ``status=error`` for months, and a
figure-12 draft leaked a tracer out of an outer ``jax.jit``. Both would have been
caught here on the first CI run.

Each case runs in a subprocess, because the scripts choose a device and set
``JAX_ENABLE_X64`` before importing jax -- exactly the setup that cannot be
undone inside a process that has already imported it.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# (module, extra argv). Every case is pinned to CPU and to the smallest N that
# still exercises the script's full path.
CASES: list[tuple[str, list[str]]] = [
    (
        "bench.validation.force_error_vs_order",
        [
            "--n",
            "256",
            "--orders",
            "2,4",
            "--basis",
            "real",
            "--distribution",
            "plummer",
            "--leaf-size",
            "16",
        ],
    ),
    (
        "bench.validation.error_vs_theta",
        [
            "--n",
            "256",
            "--thetas",
            "0.5,0.7",
            "--orders",
            "4",
            "--basis",
            "real",
            "--distribution",
            "plummer",
            "--leaf-size",
            "16",
        ],
    ),
    # Sections 4-7 run preset="galaxy" on the fused lane (REBASE_PLAN D3); on CPU
    # bench/scaling/_galaxy.py takes the lane's CPU route (kernels interpreted),
    # so these cases run the programs a card runs, slowly.
    (
        "bench.validation.mac_comparison",
        [
            "--n",
            "2048",
            "--eps",
            "1e-3,1e-4",
            "--theta",
            "0.6,0.8",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--accuracy-targets",
            "256",
        ],
    ),
    (
        "bench.scaling.wallclock",
        [
            "--n-min-exp",
            "11",
            "--n-max-exp",
            "12",
            "--n-steps",
            "2",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--direct-max-n",
            "4096",
            "--accuracy-max-n",
            "4096",
            "--accuracy-targets",
            "256",
            "--fit-min-n",
            "0",
        ],
    ),
    (
        "bench.scaling.interaction_counts",
        [
            "--n-min-exp",
            "11",
            "--n-max-exp",
            "12",
            "--n-steps",
            "2",
            "--fit-min-n",
            "0",
        ],
    ),
    (
        "bench.scaling.gpu_vs_cpu_speedup",
        ["--n", "2048", "--repeats", "1", "--warmup", "0"],
    ),
    (
        "bench.scaling.stage_breakdown",
        ["--n", "2048", "--repeats", "1", "--warmup", "0", "--trace-calls", "1"],
    ),
    (
        "bench.scaling.capacity",
        # --no-grad: the gradient has no reverse on the fused lane's CPU route
        # (a dynamic-bound fori_loop in the L2L); the GPU run measures it.
        ["--n", "2048", "--fit-min-n", "0", "--no-grad"],
    ),
    # The gradient benches take the gates' CPU route (the S9 interpret set,
    # grad_bench_lib.prepare_env) at order 2: at p6 one value_and_grad compiles
    # for ~40 min on CPU. The paper runs stay at p6.
    (
        "bench.differentiability.grad_correctness",
        [
            "--n",
            "512",
            "--thetas",
            "0.6",
            "--eps",
            "",
            "--order",
            "2",
            "--orders",
            "2",
            "--wrt",
            "positions",
            "--fd-samples",
            "3",
            "--full-samples",
            "1",
        ],
    ),
    (
        "bench.differentiability.autodiff_overhead",
        [
            "--n",
            "512",
            "--order",
            "2",
            "--reps",
            "2",
            "--dtype",
            "float64",
            "--arms",
            "step",
        ],
    ),
    (
        # fig12's multi-GPU point: bench/multigpu_grad_gate.py --mode mesh as a
        # subprocess, two forced host devices
        "bench.differentiability.autodiff_overhead",
        [
            "--mode",
            "mesh",
            "--ndev",
            "2",
            "--n",
            "2048",
            "--distribution",
            "cube",
            "--softening",
            "1e-2",
            "--mac",
            "dehnen",
            "--theta",
            "0.8",
            "--order",
            "2",
            "--reps",
            "2",
            "--dtype",
            "float64",
        ],
    ),
    (
        # forces two host devices itself under JAX_PLATFORMS=cpu
        "bench.differentiability.distributed_grad_correctness",
        [
            "--ndev",
            "2",
            "--n",
            "2048",
            "--distribution",
            "cube",
            "--mac",
            "dehnen",
            "--theta",
            "0.8",
            "--order",
            "2",
            "--check-targets",
            "256",
        ],
    ),
    # The mesh lane on two forced host devices (each script's worker sets the
    # count). comm_overhead's CPU records are never `valid`: a CPU split of
    # collective vs compute time proves the trace plumbing, not a number.
    (
        "bench.multigpu.load_balance",
        [
            "--ndevs",
            "2",
            "--per-device-n",
            "512",
            "--distributions",
            "plummer",
            "--mac",
            "dehnen",
            "--segments",
            "0",
        ],
    ),
    (
        "bench.multigpu.comm_overhead",
        [
            "--ndevs",
            "2",
            "--per-device-n",
            "512",
            "--mac",
            "dehnen",
            "--steps",
            "1",
            "--warm-segments",
            "1",
            "--time-segments",
            "1",
            "--trace-segments",
            "1",
            "--trace-force",
            "0",
        ],
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    "module,extra",
    CASES,
    ids=[
        module.rsplit(".", 1)[-1] + ("_mesh" if "mesh" in extra else "")
        for module, extra in CASES
    ],
)
def test_bench_script_runs_and_writes_a_valid_artifact(
    module: str, extra: list[str], tmp_path: pathlib.Path
) -> None:
    out = tmp_path / "smoke.json"
    env = dict(os.environ)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            # The scripts call autocvd before importing jax; on a CPU-only CI
            # runner there is nothing to select, and asking would just warn.
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *extra,
            "--seed",
            "0",
            "--gpu-select",
            "none",
            "--json-out",
            str(out),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"{module} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n"
        f"--- stderr ---\n{proc.stderr[-4000:]}"
    )
    assert out.exists(), f"{module} exited 0 but wrote no artifact"

    payload = json.loads(out.read_text())
    assert set(payload) >= {
        "config",
        "meta",
        "data",
    }, f"{module} wrote a non-artifact JSON: keys {sorted(payload)}"

    # The traceability contract: `write_result` refuses to write without these,
    # so this also pins that the guard is still wired up rather than bypassed.
    from examples.jaccpot_paper.common.jsonio import REQUIRED_CONFIG_KEYS

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in payload["config"]]
    assert not missing, f"{module} artifact config is missing {missing}"

    # Seeded: the recorded seed must be the one we asked for, or a rerun of the
    # sweep would not reproduce the sample.
    assert int(payload["config"]["seed"]) == 0

    assert payload["data"], f"{module} wrote an empty data block"


def _gate_record(tag_ndev: int, *, bits: int = 0, gpu: bool = True) -> dict:
    """The fields of bench/multigpu_segment_gate.py's segment-mode record that
    from_matrix reads (25 needs entries: NEEDS_FIELDS + CROSS_NEEDS_FIELDS)."""
    rep = {
        "segment": 1,
        "steps": 2,
        "retries": 0,
        "bits": bits,
        "counts": [8] * tag_ndev,
        "needs": [0] * 25,
        "migrated": 0,
        "declined": False,
    }
    return {
        "mode": "segment",
        "mac": "dehnen",
        "ic": "plummer",
        "n": 16,
        "ndev": tag_ndev,
        "steps": 2,
        "segments": 5,
        "softening": 1e-7,
        "devices": [("cuda:%d" if gpu else "cpu:%d") % i for i in range(tag_ndev)],
        "start_report": dict(rep, segment=1, steps=0),
        "reports": [rep] * 5,
        "segment_s": [9.0, 3.0, 2.0, 1.0, 1.0],
        "force_only_ms": [400.0, 300.0],
        "rel_l2_start": 1e-3,
        "rel_l2_end": 1e-3,
        "counts0": [8] * tag_ndev,
    }


def test_from_matrix_validity_rules(tmp_path: pathlib.Path) -> None:
    m = tmp_path / "matrix"
    m.mkdir()
    rows = {
        "strong_P8e6_geo_1c": (_gate_record(1), False, 0),
        "strong_P8e6_geo_2c": (_gate_record(2, bits=0x8), False, 0),  # overflowed
        "strong_P8e6_geo_4c": (None, False, 1),  # died: no TAG.json
        "weak_P4e6pc_2c": (_gate_record(2), True, 0),  # contended
    }
    for tag, (rec, contended, rc) in rows.items():
        if rec is not None:
            (m / f"{tag}.json").write_text(json.dumps(rec))
        (m / f"{tag}.smi.txt").write_text(
            json.dumps({"tag": tag, "rc": rc, "contended": contended, "gate_args": []})
        )
    (m / f"strong_P8e6_geo_4c.log").write_text("Traceback: boom\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.multigpu.from_matrix",
            "--matrix-dir",
            str(m),
            "--strong-out",
            str(tmp_path / "s.json"),
            "--weak-out",
            str(tmp_path / "w.json"),
            "--device-kind",
            "test",
            "--interconnect",
            "test",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    strong = {
        r["tag"]: r
        for r in json.loads((tmp_path / "s.json").read_text())["data"]["records"]
    }
    weak = {
        r["tag"]: r
        for r in json.loads((tmp_path / "w.json").read_text())["data"]["records"]
    }
    one = strong["strong_P8e6_geo_1c"]
    assert (
        one["valid"]
        and one["step_s"] == pytest.approx(0.5)
        and one["force_s"] == pytest.approx(0.3)
    )
    assert not strong["strong_P8e6_geo_2c"]["valid"]
    assert strong["strong_P8e6_geo_2c"]["capacity_ok"] is False
    assert not strong["strong_P8e6_geo_4c"]["valid"]
    assert "Traceback" in strong["strong_P8e6_geo_4c"]["log_tail"]
    assert not weak["weak_P4e6pc_2c"]["valid"] and weak["weak_P4e6pc_2c"]["capacity_ok"]
    cfg = json.loads((tmp_path / "s.json").read_text())["config"]
    assert cfg["device_label_asserted_by_operator"] is True


@pytest.mark.slow
def test_every_paper_bench_script_is_covered() -> None:
    """A new bench script must be added to CASES or explicitly excused here.

    Without this, adding a script and forgetting to smoke it leaves it untested
    and free to rot exactly the way the jaxFMM import path did.
    """

    covered = {module.rsplit(".", 1)[-1] for module, _ in CASES}
    excused = {
        # Figures 08/09 are converted from bench/multigpu_scaling_matrix.py's
        # outputs: from_matrix is covered by test_from_matrix_validity_rules, and
        # run_matrix only launches the library matrix (its CPU run is the HoreKa
        # bundle's SMOKE job).
        "from_matrix",
        "run_matrix",
    }
    on_disk = set()
    for directory in ("validation", "scaling", "differentiability", "multigpu"):
        for path in (REPO_ROOT / "bench" / directory).glob("*.py"):
            name = path.stem
            if name.startswith("_") or name == "__init__":
                continue
            # The engineering benchmarks under these directories are driven by
            # their paper wrappers; only the wrappers need a smoke case.
            if name in {"mac_error_distribution", "force_scale_prepare_cost"}:
                continue
            # `main`'s MAC engineering benchmarks, which arrived with the merge
            # of the Dehnen mass-dependent MAC work. They answer development
            # questions -- how many far pairs a criterion accepts, what a leaf
            # sweep costs at a common matched median, whether the pair-policy
            # far tag fits in memory -- and none of them feeds a figure in the
            # manuscript, which is what this test is scoped to. If one later
            # becomes a figure's source it moves into CASES with that figure.
            if name in {
                "far_pair_census",
                "leaf_sweep_common_target",
                "mac_end_to_end_walltime",
                "pair_policy_far_tag_memory",
                "per_node_theta_fidelity",
                # Arrived from `main` with this merge -- it landed there 2026-08-28
                # (fa8e2ab), four days after this branch's last commit, so the
                # branch could not have listed it. Same category as the five above:
                # it serves tracks B2/B3 of `docs/plan_2026-08_B_nearfield.md` --
                # whether order 4->8 is free, whether leaf 128 beats leaf 1024 --
                # which are configuration questions, not manuscript figures.
                "order_leaf_accuracy_sweep",
            }:
                continue
            # Runnable scripts only. A helper module with no entry point (e.g.
            # grad_bench_lib) has nothing to run end to end; detect that
            # structurally rather than by maintaining a name list that would go
            # stale the moment someone adds another helper.
            if '__name__ == "__main__"' not in path.read_text():
                continue
            on_disk.add(name)

    uncovered = on_disk - covered - excused
    assert not uncovered, (
        f"paper bench scripts with no smoke case: {sorted(uncovered)}. "
        "Add them to CASES, or to `excused` with a reason."
    )

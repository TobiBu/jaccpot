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
    (
        "bench.validation.mac_comparison",
        [
            "--n",
            "512",
            "--order",
            "4",
            "--distribution",
            "plummer",
            "--theta",
            "0.5,0.7",
            "--eps",
            "1e-4,1e-5",
            # The eq-16b arm needs an O(N^2) force scale and a second
            # prepare_state per point; the two production arms are enough to
            # prove the wrapper drives the engine and re-envelopes its output.
            "--arm",
            "fixed,mass",
            "--leaf-size",
            "16",
        ],
    ),
    (
        "bench.scaling.wallclock",
        [
            "--n-min-exp",
            "9",
            "--n-max-exp",
            "10",
            "--n-steps",
            "2",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--leaf-size",
            "32",
            "--direct-max-n",
            "1024",
            "--accuracy-max-n",
            "1024",
            "--fit-min-n",
            "0",
            "--preset",
            "accurate",
        ],
    ),
    (
        "bench.scaling.interaction_counts",
        [
            "--n-min-exp",
            "9",
            "--n-max-exp",
            "10",
            "--n-steps",
            "2",
            "--leaf-size",
            "16",
            "--fit-min-n",
            "0",
        ],
    ),
    (
        "bench.scaling.gpu_vs_cpu_speedup",
        [
            "--n",
            "512",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--leaf-size",
            "16",
            "--cpu-max-n",
            "512",
            "--preset",
            "accurate",
        ],
    ),
    (
        "bench.differentiability.grad_correctness",
        [
            "--n",
            "128",
            "--thetas",
            "0.5",
            "--orders",
            "4",
            "--basis",
            "real",
            "--wrt",
            "positions",
            "--fd-samples",
            "3",
            "--leaf-size",
            "16",
        ],
    ),
    (
        "bench.differentiability.autodiff_overhead",
        [
            "--n",
            "256",
            "--basis",
            "real",
            "--repeats",
            "2",
            "--warmup",
            "1",
            "--leaf-size",
            "16",
        ],
    ),
]

# `stage_breakdown` is deliberately absent: it instruments the strict refresh
# path, which `refresh_prepared_state` implements only for
# preset='large_n_gpu'/radix/solidfmm and which raises on CPU. A CPU smoke test
# of it could only assert that it fails, which is not a guard. Its own script
# prints a warning when it is run on CPU.


@pytest.mark.slow
@pytest.mark.parametrize(
    "module,extra",
    CASES,
    ids=[module.rsplit(".", 1)[-1] for module, _ in CASES],
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


@pytest.mark.slow
def test_every_paper_bench_script_is_covered() -> None:
    """A new bench script must be added to CASES or explicitly excused here.

    Without this, adding a script and forgetting to smoke it leaves it untested
    and free to rot exactly the way the jaxFMM import path did.
    """

    covered = {module.rsplit(".", 1)[-1] for module, _ in CASES}
    excused = {
        # See the note above CASES.
        "stage_breakdown",
        # Needs >= 2 devices, so it cannot run on a CPU-only runner as itself.
        # It *can* be coerced onto two forced host devices
        # (XLA_FLAGS=--xla_force_host_platform_device_count=2), but only with
        # --halo-exchange buf: the production path uses jax.lax.ragged_all_to_all,
        # which XLA:CPU does not implement. That makes a CPU smoke case a 150 s
        # exercise of the deprecated fallback -- the buf path exists only for
        # JAX < 0.9.1 and goes away when the floor in pyproject.toml rises -- so
        # it would guard a route we intend to delete while leaving the real one
        # untested. Covered on real devices by
        # tests/distributed/test_distributed_grad_correctness.py instead.
        "distributed_grad_correctness",
    }
    on_disk = set()
    for directory in ("validation", "scaling", "differentiability"):
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
                # Landed on `main` 2026-08-28 (fa8e2ab), four days after this
                # branch's last commit, so the branch could not have listed it.
                # Same category as the five above: it serves tracks B2/B3 of
                # `docs/plan_2026-08_B_nearfield.md` -- whether order 4->8 is free
                # and whether leaf 128 beats leaf 1024 -- which are configuration
                # questions, not a manuscript figure.
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

"""Section 7 test 9: every bench script runs, at a tiny N, in CI.

These are smoke tests in the strict sense -- they assert that each script
parses its arguments, builds its objects, completes, and writes a results JSON
whose ``config`` satisfies ``jsonio``'s required-key contract. They assert
nothing about the *values*, which are measurements and belong in the artifacts.

They exist because the failure they catch is the expensive one: a bench script
that dies after forty minutes of GPU time on a typo in its last ten lines. Each
runs on the CPU backend at an N small enough to be a few seconds.


RUN THIS SUITE WITH ``JAX_PLATFORMS=cpu``. ``addopts`` carries ``-n auto``, so
even naming a single test here spawns one xdist worker per logical core, and
every worker that imports JAX takes a CUDA context of its own. Unpinned on a
GPU host that is ninety-odd contexts and tens of gigabytes of device memory,
held for as long as the run lasts, for tests that need no GPU at all -- 178 of
them across eight A100s, on the occasion that produced this note. Nothing in a
test file can prevent it; it is a property of how pytest is invoked.
"""

import json
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Every script under bench/payoff_static, with the smallest arguments that
#: still exercise its whole code path.
SCRIPTS = {
    "gradient_cost_vs_nparams": [
        "--n",
        "128",
        "--tracers",
        "32",
        "--order",
        "3",
        "--leaf-size",
        "16",
        "--repeats",
        "1",
        "--warmup",
        "1",
    ],
    "topology_switching": [
        "--n",
        "128",
        "--tracers",
        "32",
        "--iterations",
        "3",
        "--cadences",
        "1,2",
        "--learning-rates",
        "1e-3",
        "--order",
        "3",
        "--leaf-size",
        "16",
        "--fd-samples",
        "2",
    ],
    "reconstruction_runs": [
        "--n",
        "128",
        "--tracers",
        "32",
        "--iterations",
        "3",
        "--order",
        "3",
        "--leaf-size",
        "16",
        "--cases",
        "smoke",
        "--softenings",
        "1e-2",
        "--noise-fractions",
        "0.0",
        "--perturbers",
        "lmc_like",
        "--diagnostics-every",
        "1",
    ],
    "multigpu_scaling": [
        "--n",
        "128",
        "--tracers",
        "32",
        "--iterations",
        "3",
        "--device-counts",
        "1",
        "--order",
        "3",
        "--leaf-size",
        "16",
    ],
}

REQUIRED_CONFIG_KEYS = ("n", "theta", "order", "basis", "seed", "device", "precision")


@pytest.mark.slow
@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_bench_script_smoke(script, tmp_path):
    """Each bench script completes at tiny N and writes a conforming JSON."""
    out = tmp_path / f"{script}.json"
    command = [
        sys.executable,
        "-m",
        f"bench.payoff_static.{script}",
        *SCRIPTS[script],
        "--gpu-select",
        "none",
        "--json-out",
        str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        env={
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
    )
    assert completed.returncode == 0, (
        f"{script} exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )
    assert out.exists(), f"{script} wrote no JSON to {out}"

    record = json.loads(out.read_text())
    assert set(record) >= {"config", "meta", "data"}
    for key in REQUIRED_CONFIG_KEYS:
        assert key in record["config"], f"{script} config omits {key!r}"
    # Provenance the manuscript depends on: every artifact says what produced it.
    assert record["meta"].get("git_sha"), f"{script} recorded no git sha"
    assert record["meta"].get("jax_version"), f"{script} recorded no jax version"
    assert record["data"].get("records"), f"{script} produced no records"
    # And no record may be a silent failure.
    failures = [r for r in record["data"]["records"] if r.get("failed")]
    assert not failures, f"{script} recorded failed points at tiny N: {failures}"


@pytest.mark.slow
def test_distributed_caps_probe_smoke(tmp_path):
    """The caps probe runs too, on forced host devices rather than a GPU.

    It is not in ``SCRIPTS`` because it is shaped differently from the four
    sweeps: it needs a multi-device mesh, so it forces CPU devices into
    existence before JAX initialises, and its payload is a ladder of attempts
    rather than a record per measured point.
    """
    out = tmp_path / "caps.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.payoff_static.distributed_caps_probe",
            "--n",
            "256",
            "--tracers",
            "32",
            "--leaf-size",
            "16",
            "--device-count",
            "2",
            "--host-devices",
            "2",
            "--json-out",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
        env={
            # The probe forces host devices itself, but pin the platform here
            # as well: a CI runner with a visible GPU must not have this test
            # quietly claim it.
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
        },
    )
    assert completed.returncode == 0, (
        f"caps probe exited {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n"
        f"--- stderr ---\n{completed.stderr[-4000:]}"
    )
    payload = json.loads(out.read_text())
    for key in REQUIRED_CONFIG_KEYS:
        assert key in payload["config"], f"caps probe config lost {key}"
    assert payload["meta"]["git_sha"], "caps probe wrote no provenance"
    attempts = payload["data"]["attempts"]
    assert len(attempts) == len(
        payload["config"]["ladder"]
    ), "one attempt per rung of the caps ladder"
    # Every rung must report an outcome; at this size none should fail.
    for a in attempts:
        assert a["ran"] is True, f"caps probe rung failed at tiny N: {a}"

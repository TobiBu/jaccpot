"""Section 7 test 9: every bench script runs, at a tiny N, in CI.

These are smoke tests in the strict sense -- they assert that each script
parses its arguments, builds its objects, completes, and writes a results JSON
whose ``config`` satisfies ``jsonio``'s required-key contract. They assert
nothing about the *values*, which are measurements and belong in the artifacts.

They exist because the failure they catch is the expensive one: a bench script
that dies after forty minutes of GPU time on a typo in its last ten lines. Each
runs on the CPU backend at an N small enough to be a few seconds.
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

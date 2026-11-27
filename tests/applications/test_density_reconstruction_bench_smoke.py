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
    # The mesh lane at ndev=1 (a one-device mesh is now a measured point, not a
    # skip) and ndev=2, each in its own process as the script requires, on
    # forced host devices with the S9 interpret set (--cpu-interpret). One
    # jitted epoch, as in test_density_reconstruction_distributed.py: the eager
    # mesh gradient takes ~7 minutes per step in interpret mode.
    "multigpu_scaling": [
        "--cpu-interpret",
        "--jit-within-epoch",
        "--n",
        "128",
        "--tracers",
        "32",
        "--iterations",
        "2",
        "--rebuild-cadence",
        "2",
        "--device-counts",
        "1",
        "--order",
        "3",
        "--leaf-size",
        "16",
    ],
    "multigpu_scaling@ndev2": [
        "--cpu-interpret",
        "--jit-within-epoch",
        "--n",
        "128",
        "--tracers",
        "32",
        "--iterations",
        "2",
        "--rebuild-cadence",
        "2",
        "--device-counts",
        "2",
        "--order",
        "3",
        "--leaf-size",
        "16",
    ],
    # Parameter sharding of the single-card fused operator.
    "multigpu_scaling@parameter_sharding": [
        "--mode",
        "parameter_sharding",
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
    module = script.split("@")[0]
    out = tmp_path / f"{module}.json"
    command = [
        sys.executable,
        "-m",
        f"bench.payoff_static.{module}",
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
    # And no record may be a silent failure, or a skipped point.
    failures = [
        r for r in record["data"]["records"] if r.get("failed") or r.get("skipped")
    ]
    assert not failures, f"{script} recorded failed points at tiny N: {failures}"
    # Section 7 runs on the fused lane, chosen explicitly, not the library's
    # default; the mesh benches record the mesh lane.
    expected = "fused_mesh" if "--cpu-interpret" in SCRIPTS[script] else "fused"
    assert record["config"].get("lane") == expected, record["config"].get("lane")


@pytest.mark.slow
def test_distributed_caps_probe_smoke(tmp_path):
    """The cross-capacity probe runs on two forced host devices.

    It is not in ``SCRIPTS`` because it is shaped differently from the sweeps:
    it needs a two-device mesh (``--cpu-interpret`` forces the host devices and
    applies the S9 interpret set before JAX initialises), and its payload is a
    ladder of rungs times placements rather than a record per measured point.
    """
    out = tmp_path / "caps.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bench.payoff_static.distributed_caps_probe",
            "--cpu-interpret",
            "--device-count",
            "2",
            "--n",
            "256",
            "--tracers",
            "32",
            "--order",
            "3",
            "--leaf-size",
            "16",
            "--gpu-select",
            "none",
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
    assert payload["config"]["caps_kind"] == "CrossCapacities"
    sizing = payload["data"]["sizing"]
    assert sizing and not sizing[0].get("failed"), f"sizing failed: {sizing}"
    attempts = payload["data"]["attempts"]
    assert len(attempts) == len(payload["config"]["ladder"]) * len(
        payload["config"]["placements"]
    ), "one attempt per rung and placement"
    # At this size every rung, the library's unplanned default included, keeps
    # the verdict clean, and the gradient at the start placement is finite.
    for a in attempts:
        assert a["ran"] is True and a["verdict_ok"] is True, f"rung failed: {a}"
        if a["placement"] == payload["config"]["placements"][0]:
            assert a["gradient_finite"] is True, f"non-finite gradient: {a}"
    assert payload["data"]["suggested_caps"][0]["cli"], "no --caps string"

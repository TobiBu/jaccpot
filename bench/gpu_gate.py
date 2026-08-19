"""Pre-release GPU gate: run the suite on a real card and prove it did.

There is no GPU leg in CI (`.github/workflows/ci.yml` is five `ubuntu-latest`
jobs), and the decision recorded in the audit's item 2.6 is that there will not
be one: a self-hosted runner on a **public** repository lets a fork PR execute
arbitrary code on the box, and the box's GPUs are shared. This script is the
agreed substitute -- a scripted local gate, run deliberately before a release.

WHAT IT COVERS THAT CPU CI CANNOT
---------------------------------
* 17 tests gated on ``jax.default_backend() != "gpu"`` that simply skip on CPU
  (``test_large_n_config_thresholds`` 9, ``test_nearfield_mode_policy`` 4,
  ``test_split_build_default_predicate`` 4).
* Whole code paths that are never *entered* on CPU, which is the larger half and
  produces no skip to notice: ``can_use_large_n_prepare_path`` gates
  ``_large_n_grad.py`` and ``_large_n_farfield.py`` on the GPU backend
  (audit F27, both at 0% coverage), and the strict/refresh lane is only
  partially reachable (F33).
* Pallas kernels outside ``interpret=True``. On CPU the Triton lowering never
  runs at all, so a kernel can be wrong on hardware and green in CI.

WHAT IT DOES NOT COVER
----------------------
``tests/distributed/`` -- 24 tests across 10 files, all gated on
``device_count() < 2``. That needs two cards and is a separate, heavier run;
audit F34 records the distributed layer as out of scope for this pass.

WHY IT CHECKS THAT THE RUN WAS NOT VACUOUS
------------------------------------------
A GPU run that silently fell back to CPU **passes**: the gated tests skip, the
gated branches are not entered, and pytest prints a clean green line. That is the
failure mode this repository has been bitten by repeatedly -- a baseline that
suppressed everything, a scan of a test file that did not exist, a docstring
count read off stderr that was being discarded. So the gate asserts that the
GPU-only tests actually **ran**, and fails if they were skipped. Green here means
the GPU paths executed; it does not merely mean nothing complained.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Node-id fragments for tests that exist only to run on a GPU. If any of these is
# reported as skipped, the run was vacuous. Kept as fragments rather than full
# node ids so a parametrisation change does not silently empty the check -- the
# count assertion below is what catches that.
_MUST_RUN = (
    "test_large_n_config_thresholds.py",
    "test_nearfield_mode_policy.py",
    "test_split_build_default_predicate.py",
)

# Measured on an A100 sm_80 / jax 0.10.2 and documented in ARCHITECTURE.md §9.
# All five predate Tier 1 (they reproduce on `128a0e2`), so a hit here is not
# necessarily yours -- bisect against a pre-Tier-1 commit before attributing it.
# Four of the five go green under --xla_gpu_deterministic_ops=true, which this
# script sets by default; the survivor is a CPU-generated golden read on GPU.
_KNOWN_GPU_FAILURES = (
    "test_fmm_grad_golden[clu_real_n128_p4]",
    "test_real_basis_tracks_complex_basis[nearfield-only-f32]",
    "test_solidfmm_chunked_m2l_matches_fullbatch",
    "test_compiled_dispatch_is_bit_identical[None]",
    "test_compiled_dispatch_is_bit_identical[32]",
)

# CLAUDE.md, "Running the suite on a GPU": `-n auto` puts 64 xdist workers on one
# card and produced 331 and 288 spurious CUDA_ERROR_OUT_OF_MEMORY failures on two
# trees, with a 2x wall-time gap and differing failure sets. Six workers at a 12%
# per-worker share measured 0 OOM lines on a 40 GB A100.
_WORKERS = 6
_MEM_FRACTION = ".12"


def _build_env(*, deterministic: bool) -> dict[str, str]:
    """Environment for the pytest subprocess.

    Set in the parent and inherited, because every one of these is read by XLA or
    jax at import time -- setting them after ``import jax`` is a no-op.

    Parameters
    ----------
    deterministic : bool
        Add ``--xla_gpu_deterministic_ops=true``. On by default: ARCHITECTURE §10
        notes the GPU forward is nondeterministic by more than some of the deltas
        being asserted, so an exact-equality result means nothing without it.

    Returns
    -------
    dict[str, str]
        A copy of the caller's environment with the GPU-run settings applied.
    """
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _MEM_FRACTION
    if deterministic:
        existing = env.get("XLA_FLAGS", "")
        flag = "--xla_gpu_deterministic_ops=true"
        env["XLA_FLAGS"] = f"{existing} {flag}".strip() if existing else flag
    return env


def _preflight(env: dict[str, str]) -> str:
    """Confirm a GPU backend is actually visible before spending 20 minutes.

    Uses ``autocvd`` to claim the least-used card, which is the org rule and also
    the courteous thing on a shared box.

    Parameters
    ----------
    env : dict[str, str]
        Environment for the probe subprocess.

    Returns
    -------
    str
        The backend name jax reports.

    Raises
    ------
    RuntimeError
        If jax does not come up on a GPU. Deliberately fatal: continuing would
        produce a green CPU run wearing a GPU run's clothes, which is the whole
        thing this script exists to prevent.
    """
    probe = (
        "from autocvd import autocvd\n"
        "autocvd(num_gpus=1, least_used=True)\n"
        "import jax\n"
        "print(jax.default_backend(), len(jax.devices()), jax.devices()[0].device_kind)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "preflight failed -- could not bring jax up on a GPU.\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()[-2000:]}"
        )
    parts = result.stdout.strip().split()
    backend = parts[0] if parts else "<none>"
    if backend != "gpu":
        raise RuntimeError(
            f"jax reports backend {backend!r}, not 'gpu'. Refusing to run: a CPU "
            "fallback would skip every GPU-gated test and still print green."
        )
    print(
        f"  preflight: backend={backend} devices={parts[1]} kind={' '.join(parts[2:])}"
    )
    return backend


def _run_pytest(env: dict[str, str], extra: list[str]) -> tuple[int, str]:
    """Run the suite with the documented GPU caps, streaming output to the console.

    Parameters
    ----------
    env : dict[str, str]
        Environment from :func:`_build_env`.
    extra : list[str]
        Additional pytest arguments, appended after the gate's own.

    Returns
    -------
    tuple[int, str]
        ``(exit code, combined output)``. The output is captured *and* echoed, so
        a long run stays watchable while remaining parseable afterwards.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        str(_WORKERS),
        "-rs",  # report skip reasons -- the not-vacuous check reads them
        "-v",
        *extra,
    ]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        sys.stdout.write(line)
    proc.wait()
    return proc.returncode, "".join(chunks)


def _check_not_vacuous(output: str) -> list[str]:
    """Assert the GPU-only tests actually ran rather than skipping.

    Parameters
    ----------
    output : str
        Combined pytest output.

    Returns
    -------
    list[str]
        Human-readable complaints; empty means the run genuinely exercised the
        GPU paths.
    """
    complaints: list[str] = []
    for fragment in _MUST_RUN:
        ran = len(re.findall(rf"{re.escape(fragment)}::\S+ (?:PASSED|FAILED)", output))
        skipped = len(re.findall(rf"{re.escape(fragment)}::\S+ SKIPPED", output))
        if ran == 0:
            complaints.append(
                f"{fragment}: 0 tests ran ({skipped} skipped) -- these are the "
                "GPU-gated tests; if they skipped, this run proves nothing"
            )
        else:
            print(f"  not-vacuous: {fragment} ran {ran} ({skipped} skipped)")
    return complaints


def _classify_failures(output: str) -> tuple[list[str], list[str]]:
    """Split reported failures into known-GPU and unexpected.

    Parameters
    ----------
    output : str
        Combined pytest output.

    Returns
    -------
    tuple[list[str], list[str]]
        ``(unexpected, known)`` node ids.
    """
    failures = sorted(set(re.findall(r"^FAILED (\S+)", output, re.M)))
    known = [f for f in failures if any(k in f for k in _KNOWN_GPU_FAILURES)]
    unexpected = [f for f in failures if f not in known]
    return unexpected, known


def main() -> int:
    """Run the gate and report.

    Returns
    -------
    int
        ``0`` only when a real GPU ran the GPU-gated tests and produced no
        failure outside ARCHITECTURE §9's documented list.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "omit --xla_gpu_deterministic_ops=true. Only for triage: four of §9's "
            "five known failures are reduction nondeterminism and reappear."
        ),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        dest="extra",
        help="extra pytest argument (repeatable), e.g. --pytest-arg=-x",
    )
    args = parser.parse_args()

    env = _build_env(deterministic=not args.no_deterministic)
    print("jaccpot GPU gate")
    print(f"  XLA_FLAGS={env.get('XLA_FLAGS', '<unset>')}")
    try:
        _preflight(env)
    except RuntimeError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        return 2

    code, output = _run_pytest(env, args.extra)

    print("\n" + "=" * 72)
    complaints = _check_not_vacuous(output)
    unexpected, known = _classify_failures(output)

    if known:
        print(f"  known GPU failures (ARCHITECTURE section 9), not yours: {len(known)}")
        for node in known:
            print(f"    {node}")
    if unexpected:
        print(f"  UNEXPECTED failures: {len(unexpected)}")
        for node in unexpected:
            print(f"    {node}")
    if complaints:
        print("  VACUOUS RUN:")
        for line in complaints:
            print(f"    {line}")

    if complaints or unexpected:
        print("\nGATE FAILED")
        return 1
    print(f"\nGATE PASSED (pytest exit {code}; known-failure allowance applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

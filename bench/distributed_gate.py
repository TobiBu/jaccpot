"""Two-card gate for the distributed tier: run what CPU cannot, and prove it did.

`bench/gpu_gate.py` covers the single-GPU suite before a release and says in its
own docstring that it does NOT cover `tests/distributed/` -- it pins ONE card
deliberately, because with two visible that tier un-skips and the run stops
matching what that gate claims. This is the sibling it names: two cards, the
distributed tier, the same anti-vacuity discipline.

WHAT NEEDS CARDS AND WHAT DOES NOT
----------------------------------
Most of this tier does not. `--xla_force_host_platform_device_count=2` runs 44 of
its tests on ordinary CPU, and `docs/plan_2026-08_shared.md` records that
distributed correctness taken that way matches 2xA100 to six digits. CI already
does exactly this for the sibling file, in the `test-distributed-mutual` job,
with a step that fails the build if the suite skipped rather than ran. Extending
that to `tests/distributed/` is the cheap 90% and needs no hardware at all.

What CPU structurally cannot reach is smaller and sharper:

* **The native ragged halo exchange.** `jax.lax.ragged_all_to_all` has no XLA:CPU
  lowering, and `resolve_grad_halo_exchange("auto")` now resolves to `"buf"` on a
  CPU backend precisely because of that. So the native reverse pass -- the one
  whose corruption on jax < 0.9.1 the version gate exists to prevent -- is by
  construction never executed on CPU. `test_native_halo_exchange_is_fixed_upstream`
  is the tripwire for it and is opt-in behind an env var; this gate sets it.
* **The fused Pallas near field.** On CPU the Triton lowering never runs, so
  `nearfield_backend="pallas"` is untested there; the driver tests pin
  `"baseline"` for exactly this reason.
* **Real collectives.** Forced host devices exercise the collective *logic* but
  not NCCL.

MEASURED COST, A100 x2, jax 0.10.2, x64, this gate's target:

    tests/distributed/test_distributed_grad_correctness.py
    11 passed in 491.63s (8:11), of which
      113.0s  test_native_halo_exchange_is_fixed_upstream   <- CPU cannot run this
       96.6s  test_fd_vs_ad[positions]
       96.5s  test_grad_matches_direct_sum_oracle
       93.9s  test_fd_vs_ad[masses]

Eight minutes is affordable as a pre-merge step for a distributed change, which
is what makes a targeted gate the right shape here. The whole tier on two cards
is NOT: these are 16-to-128-particle problems where launch and compile dominate,
and it runs an order of magnitude slower per test than the same tests on forced
CPU devices. Run the cheap part on CPU in CI and spend the cards on what only
cards can answer.

USAGE
-----
    python -m bench.distributed_gate                 # the GPU-only-reachable set
    python -m bench.distributed_gate --full          # the whole tier, slow
    python -m bench.distributed_gate --no-deterministic
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from bench.gpu_gate import (
    _MEM_FRACTION,
    _REPO_ROOT,
    _build_env,
    _classify_failures,
    _outcomes_by_node,
)

#: Two, because that is the smallest device count at which every file in the tier
#: un-skips, and because the box's GPUs are shared -- a gate that claims four is a
#: gate people skip running.
_CARDS = 2

#: The default target: the file whose paths CPU cannot reach. `--full` widens it.
_TARGET = ("tests/distributed/test_distributed_grad_correctness.py",)
_FULL_TARGET = ("tests/distributed", "tests/integration/test_mutual_distributed.py")

#: Node-id fragments that MUST report a real verdict. A distributed run that fell
#: back to one device skips every one of them and prints green, which is the
#: failure this gate exists to prevent -- the same reasoning as `gpu_gate._MUST_RUN`,
#: with `device_count() < 2` as the trigger instead of the backend.
_MUST_RUN = (
    # The only test in the repository that exercises the native ragged reverse
    # pass. Opt-in behind JACCPOT_CHECK_UPSTREAM_RAGGED_FIX, which this gate sets:
    # without it the test skips, and a skipped tripwire is not a tripwire.
    "test_native_halo_exchange_is_fixed_upstream",
    "test_fd_vs_ad",
    "test_grad_matches_direct_sum_oracle",
)


def _check_not_vacuous(output: str) -> list[str]:
    """Assert the tests this gate exists for actually ran rather than skipping.

    Deliberately a local copy of ``gpu_gate``'s check rather than a shared
    parameterised one: that module is covered by its own tests, and widening a
    tested function's signature to save fifteen lines here is a worse trade than
    the duplication. The parser they both depend on IS shared.

    Parameters
    ----------
    output : str
        Combined pytest output.

    Returns
    -------
    list[str]
        Human-readable complaints; empty means the run genuinely exercised the
        multi-device paths.
    """
    complaints: list[str] = []
    outcomes = _outcomes_by_node(output)
    for fragment in _MUST_RUN:
        matching = {n: v for n, v in outcomes.items() if fragment in n}
        ran = sum(1 for v in matching.values() if v & {"PASSED", "FAILED"})
        skipped = sum(1 for v in matching.values() if v == {"SKIPPED"})
        if ran == 0:
            complaints.append(
                f"{fragment}: 0 tests ran ({skipped} skipped) -- this is one of "
                "the paths only two cards can reach; if it skipped, this run "
                "proves nothing"
            )
        else:
            print(f"  not-vacuous: {fragment} ran {ran} ({skipped} skipped)")
    return complaints


def _preflight_two_cards(env: dict[str, str]) -> str:
    """Claim two cards and confirm jax comes up on them, before spending minutes.

    Parameters
    ----------
    env : dict[str, str]
        Environment for the probe subprocess.

    Returns
    -------
    str
        The ``CUDA_VISIBLE_DEVICES`` value the probe ran under, so the pytest
        subprocess can be pinned to the SAME cards. ``autocvd`` sets it in the
        probe process, which dies with the probe.

    Raises
    ------
    RuntimeError
        If jax does not come up on at least two GPU devices. Fatal on purpose: a
        one-device run skips this whole tier and reports success, which is worse
        than a red run because it looks like coverage.
    """
    probe = (
        "import os\n"
        "if not os.environ.get('CUDA_VISIBLE_DEVICES'):\n"
        "    from autocvd import autocvd\n"
        f"    autocvd(num_gpus={_CARDS}, least_used=True)\n"
        "import jax\n"
        "print('|'.join((jax.default_backend(), str(len(jax.devices())),\n"
        "    os.environ.get('CUDA_VISIBLE_DEVICES', ''))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"device probe failed:\n{result.stderr}")
    backend, count, visible = result.stdout.strip().splitlines()[-1].split("|")
    if backend != "gpu":
        raise RuntimeError(
            f"jax came up on {backend!r}, not a GPU. Refusing to run: this gate "
            "would skip the entire tier and report success."
        )
    if int(count) < _CARDS:
        raise RuntimeError(
            f"only {count} GPU device(s) visible; this tier needs {_CARDS}. "
            "Every file in it is guarded on device_count() < 2 and would skip."
        )
    print(f"  preflight: {backend}, {count} devices, CUDA_VISIBLE_DEVICES={visible}")
    return visible


def main() -> int:
    """Run the distributed gate and report.

    Returns
    -------
    int
        ``0`` if the targeted tests ran and nothing unexpected failed, ``1``
        otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run the whole distributed tier rather than the GPU-only-reachable set",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help="drop --xla_gpu_deterministic_ops=true (see ARCHITECTURE.md section 10)",
    )
    args = parser.parse_args()

    env = _build_env(deterministic=not args.no_deterministic)
    # Without this the native tripwire skips, and a skipped tripwire is the exact
    # shape of vacuous green this gate exists to refuse.
    env["JACCPOT_CHECK_UPSTREAM_RAGGED_FIX"] = "1"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _MEM_FRACTION

    visible = _preflight_two_cards(env)
    env["CUDA_VISIBLE_DEVICES"] = visible

    target = _FULL_TARGET if args.full else _TARGET
    # `-n 0`: the devices live in one process and each test builds its own mesh,
    # so serial keeps the device set unambiguous -- the same reasoning the
    # test-distributed-mutual CI job gives for its own `-n 0`.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-n",
        "0",
        "-v",
        "-rfEs",
        *target,
    ]
    print(f"  running: {' '.join(target)}")
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, capture_output=True, text=True)
    output = proc.stdout + proc.stderr
    print(output[-4000:])

    complaints = _check_not_vacuous(output)
    unexpected, known = _classify_failures(output)
    if known:
        print(f"  known GPU failures (see ARCHITECTURE.md section 9): {known}")
    for complaint in complaints:
        print(f"  VACUOUS: {complaint}")
    for failure in unexpected:
        print(f"  UNEXPECTED FAILURE: {failure}")
    if complaints or unexpected:
        return 1
    print("  distributed gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

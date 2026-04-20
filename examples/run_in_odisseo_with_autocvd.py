"""Run a repo-local command with the standard odisseo GPU validation setup.

Example:
    micromamba run -n odisseo python examples/run_in_odisseo_with_autocvd.py \
        --use-autocvd -- \
        python -m pytest -q -o addopts='' tests/unit/test_solver_api.py -k specialized
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
YGGDRAX_ROOT = REPO_ROOT.parent / "yggdrax"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-autocvd", action="store_true")
    parser.add_argument("--autocvd-num-gpus", type=int, default=1)
    parser.add_argument("--autocvd-exclude", nargs="*", default=[])
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--allow-missing-autocvd", action="store_true")
    parser.add_argument("--no-x64", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'.",
    )
    return parser.parse_args()


def _configure_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        print("Set CUDA_VISIBLE_DEVICES =", env["CUDA_VISIBLE_DEVICES"])
    elif args.use_autocvd:
        try:
            from autocvd import autocvd

            autocvd(
                num_gpus=int(args.autocvd_num_gpus),
                least_used=True,
                exclude=list(args.autocvd_exclude),
            )
            env["CUDA_VISIBLE_DEVICES"] = os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                env.get("CUDA_VISIBLE_DEVICES", ""),
            )
            print(
                "autocvd selected CUDA_VISIBLE_DEVICES =",
                env.get("CUDA_VISIBLE_DEVICES", "<all visible>"),
            )
        except ImportError:
            if not args.allow_missing_autocvd:
                raise
            print("autocvd is not installed. Using existing CUDA visibility.")
    else:
        print(
            "Using existing CUDA visibility:",
            env.get("CUDA_VISIBLE_DEVICES", "<all visible>"),
        )

    if not args.no_x64:
        env["JAX_ENABLE_X64"] = "1"

    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    visible_physical_gpus = [
        part.strip()
        for part in env.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if part.strip()
    ]
    if visible_physical_gpus:
        env["JACCPOT_NVIDIA_SMI_GPU_INDEX"] = visible_physical_gpus[0]

    pythonpath_parts = [str(REPO_ROOT)]
    if YGGDRAX_ROOT.exists():
        pythonpath_parts.append(str(YGGDRAX_ROOT))
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    print("PYTHONPATH includes:", env["PYTHONPATH"])
    print("JAX_ENABLE_X64 =", env.get("JAX_ENABLE_X64", "0"))
    print(
        "JACCPOT_NVIDIA_SMI_GPU_INDEX =",
        env.get("JACCPOT_NVIDIA_SMI_GPU_INDEX", "<unset>"),
    )
    return env


def main() -> int:
    args = _parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("Provide a command after '--'.")

    env = _configure_environment(args)
    proc = subprocess.run(command, env=env, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

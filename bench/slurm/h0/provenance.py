"""H0 bundle: the provenance header of a job (git commits, packages, node, GPUs, environment).

Every job writes one ``provenance.json`` into its output directory before its first row
(``env.sh``'s ``h0_provenance``); ``summarize.py`` puts it in front of every row it
collects. It imports no jax backend (``jax.__version__`` only, no device), so it is cheap
and cannot hold a card.

    python bench/slurm/h0/provenance.py --out OUT/provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

#: the packages whose versions decide the numbers
PACKAGES = (
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "numpy",
    "scipy",
    "ml_dtypes",
    "optax",
    "equinox",
    "diffrax",
    "jaxtyping",
    "beartype",
    "nvidia-cuda-nvcc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
)
#: environment prefixes worth recording
ENV_PREFIXES = (
    "JAX_",
    "XLA_",
    "JACCPOT_",
    "YGGDRAX_",
    "SLURM_",
    "CUDA_",
    "H0_",
    "NCCL_",
)
ENV_KEYS = (
    "PYTHONPATH",
    "BENCH_DIR",
    "DISC_IC",
    "LD_LIBRARY_PATH",
    "PATH",
    "VIRTUAL_ENV",
)


def _run(cmd: list[str], timeout: float = 60.0) -> str | None:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc!r}"
    return (out.stdout + (out.stderr if out.returncode else "")).strip()


def _git(path: str) -> dict:
    if not path or not Path(path).exists():
        return {"path": path, "missing": True}
    head = _run(["git", "-C", path, "rev-parse", "HEAD"])
    return {
        "path": path,
        "head": head,
        "branch": _run(["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["git", "-C", path, "log", "-1", "--format=%h %cI %s"]),
        # the pre-#104 arm is a deliberate one-file modification: recorded, not hidden
        "status": _run(["git", "-C", path, "status", "--porcelain", "--untracked=no"]),
        "diff_stat": _run(["git", "-C", path, "diff", "--stat", "HEAD"]),
    }


def _sha256(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _gpus() -> dict:
    query = (
        "index,name,uuid,pci.bus_id,driver_version,memory.total,clocks.max.sm,"
        "clocks.max.mem,power.limit,persistence_mode,compute_mode"
    )
    return {
        "query": query,
        "rows": _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"]),
        "cuda_driver": _run(["nvidia-smi", "--version"]),
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "compute_apps": _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader",
            ]
        ),
    }


def _packages() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def _jax() -> dict:
    try:
        import jax
        import jaxlib
    except ImportError as exc:  # pragma: no cover - a broken env is what this records
        return {"import_error": repr(exc)}
    info = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "file": jax.__file__}
    try:
        import jax_plugins.xla_cuda12 as plugin

        # must be inside the venv (a --system-site-packages venv picked another's .so)
        info["xla_cuda12_plugin"] = plugin.__file__
    except ImportError as exc:
        info["xla_cuda12_plugin"] = f"not importable: {exc!r}"
    return info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--no-ic-hash", action="store_true", help="skip the IC's sha256 (573 MB read)"
    )
    args = p.parse_args()
    env = os.environ
    record = {
        "schema": "h0-provenance-1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "cpu": _run(
            ["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|NUMA node\\(s\\)'"]
        ),
        "memory": _run(["bash", "-c", "free -g | head -2"]),
        "modules": env.get("LOADEDMODULES"),
        "git": {
            "jaccpot": _git(env.get("H0_JACCPOT", "")),
            "yggdrax": _git(env.get("H0_YGGDRAX", "")),
            "nornax": _git(env.get("H0_NORNAX", "")),
            "mimirax": _git(env.get("H0_MIMIRAX", "")),
        },
        "pins": {
            k: env.get(k)
            for k in (
                "H0_YGGDRAX_COMMIT",
                "H0_NORNAX_COMMIT",
                "H0_MIMIRAX_COMMIT",
                "H0_YGGDRAX_PRE104_FILE_COMMIT",
            )
        },
        "packages": _packages(),
        "jax": _jax(),
        "gpus": _gpus(),
        "env": {
            k: v
            for k, v in sorted(env.items())
            if k.startswith(ENV_PREFIXES) or k in ENV_KEYS
        },
        "disc_ic": {
            "path": env.get("DISC_IC"),
            "sha256": None if args.no_ic_hash else _sha256(env.get("DISC_IC", "")),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"provenance: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

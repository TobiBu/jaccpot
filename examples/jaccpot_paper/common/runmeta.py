"""Seeding, device selection, and run provenance for the paper's bench scripts.

No JAX at module scope. :func:`select_gpu` exists precisely because ``autocvd``
has to run *before* the first ``import jax`` -- once JAX has initialised its
backend, ``CUDA_VISIBLE_DEVICES`` no longer has any effect, so a script that
imports jax at the top of the file cannot pick a free device however politely it
asks later.

Usage in a bench script::

    def main() -> int:
        args = _parse_args()
        select_gpu(args.gpu_select)          # must precede the jax import
        import jax                           # noqa: E402  -- deliberate
        ...
        jsonio.write_result(out, config=cfg, meta=run_meta(), data=payload)
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Any, Optional

__all__ = [
    "add_common_args",
    "enable_x64",
    "git_meta",
    "run_meta",
    "seed_sequence",
    "select_gpu",
]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def add_common_args(parser: Any) -> Any:
    """Register the flags every paper bench script shares."""

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; recorded in the output JSON (default: 0)",
    )
    parser.add_argument(
        "--gpu-select",
        choices=("least-used", "first", "none"),
        default="least-used",
        help=(
            "autocvd free-GPU selection, applied before importing JAX. 'none' "
            "leaves the device choice to the environment (default: least-used)"
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
        help="Working precision; recorded in the output JSON (default: float64)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help=(
            "Where to write the result JSON. Relative paths resolve under "
            "results/. Defaults to the script's canonical results/ path."
        ),
    )
    return parser


def select_gpu(mode: str = "least-used") -> Optional[str]:
    """Pin a free GPU via ``autocvd``. Call before the first ``import jax``.

    Returns the selected ``CUDA_VISIBLE_DEVICES`` value, or ``None`` if selection
    was skipped. Respects a ``CUDA_VISIBLE_DEVICES`` already set by the caller so
    an explicit outer pin (or a CI runner's) is never silently overridden.
    """

    if mode == "none":
        return os.environ.get("CUDA_VISIBLE_DEVICES")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    if "jax" in _imported_modules():
        raise RuntimeError(
            "select_gpu() was called after jax was already imported; the device "
            "choice would be silently ignored. Move select_gpu() above the jax "
            "import."
        )
    try:
        from autocvd import autocvd
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[bench] autocvd unavailable ({exc}); using the default device")
        return os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        autocvd(num_gpus=1, least_used=(mode == "least-used"))
    except Exception as exc:  # pragma: no cover - no free GPU, or no GPU at all
        print(f"[bench] autocvd could not select a GPU ({exc}); using the default")
    # Keep the allocator honest: preallocating the whole card makes concurrent
    # runs on this shared host fail with RESOURCE_EXHAUSTED rather than queue.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _imported_modules() -> set[str]:
    import sys

    return set(sys.modules)


def enable_x64(dtype: str) -> None:
    """Turn on JAX x64 when float64 was requested. Call before importing jax."""

    if str(dtype) == "float64":
        os.environ.setdefault("JAX_ENABLE_X64", "1")


def seed_sequence(seed: int, *labels: str) -> int:
    """Derive a stable per-case seed from a base seed and string labels.

    Every stochastic step in the paper is seeded, and a sweep needs a *different*
    draw per case without depending on iteration order (so that rerunning a
    single N reproduces exactly the sample the full sweep used). Hashing the
    labels rather than incrementing a counter gives that.
    """

    import hashlib

    digest = hashlib.sha256(
        "\x00".join((str(int(seed)),) + tuple(str(v) for v in labels)).encode()
    ).digest()
    # 63 bits: stays a positive Python int and inside jax.random's key range.
    return int.from_bytes(digest[:8], "big") >> 1


def git_meta() -> dict[str, Any]:
    """Return the commit the measurement was taken at, and whether it was dirty."""

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, cwd=_REPO_ROOT, check=False
            ).stdout.strip()
        except OSError:  # pragma: no cover
            return ""

    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the recorded sha does not fully describe the code
        # that ran. Figures for the manuscript should be regenerated clean.
        "git_dirty": bool(run("git", "status", "--porcelain")),
    }


def run_meta(extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return full run provenance. Safe to call only after JAX is imported."""

    meta: dict[str, Any] = dict(git_meta())
    try:
        import jax

        meta["jax_version"] = jax.__version__
        meta["jax_backend"] = jax.default_backend()
        meta["devices"] = [str(d) for d in jax.devices()]
        meta["device_kind"] = jax.devices()[0].device_kind
        meta["jax_enable_x64"] = bool(jax.config.jax_enable_x64)
    except Exception as exc:  # pragma: no cover
        meta["jax_version"] = f"unavailable: {exc}"

    meta["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if extra:
        meta.update(extra)
    return meta


def device_label() -> str:
    """Return a short device label (e.g. ``NVIDIA A100-PCIE-40GB``) for the JSON."""

    try:
        import jax

        return str(jax.devices()[0].device_kind)
    except Exception:  # pragma: no cover
        return "unknown"

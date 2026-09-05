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

import datetime
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
            "bench/results/. Defaults to the script's canonical results path."
        ),
    )
    return parser


def select_gpu(mode: str = "least-used", *, num_gpus: int = 1) -> Optional[str]:
    """Pin free GPU(s) via ``autocvd``. Call before the first ``import jax``.

    Returns the selected ``CUDA_VISIBLE_DEVICES`` value, or ``None`` if selection
    was skipped. Respects a ``CUDA_VISIBLE_DEVICES`` already set by the caller so
    an explicit outer pin (or a CI runner's) is never silently overridden.

    ``num_gpus`` is what the distributed benchmarks need: a multi-GPU run must
    claim its whole mesh in one selection, because selecting one card at a time
    races another process onto the same device and then reports a scaling number
    measured against a competitor. It is also why a partial selection is fatal
    below rather than a warning -- a strong-scaling point silently taken on
    fewer devices than requested is worse than no point at all.
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
        autocvd(num_gpus=num_gpus, least_used=(mode == "least-used"))
    except Exception as exc:  # pragma: no cover - no free GPU, or no GPU at all
        if num_gpus > 1:
            raise RuntimeError(
                f"autocvd could not claim {num_gpus} GPUs ({exc}). Refusing to "
                "fall back: a distributed measurement taken on a different "
                "device count than requested is not the measurement asked for."
            ) from exc
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
    """Return the commit the measurement was taken at, and whether it was dirty.

    Reports dirtiness twice, because the two answers mean different things and
    only one of them bears on whether a number is trustworthy.

    ``git_dirty`` is the whole working tree, as before. ``git_dirty_sources``
    excludes ``bench/results/``, and it is the one to judge an artifact by: what
    matters is whether the recorded sha describes the CODE that ran, and a
    rewritten results JSON says nothing about that.

    The distinction is not academic. The committed figure artifacts are tracked
    files under ``bench/results/``, and several benches now rewrite theirs after
    every point so that an interrupted multi-hour sweep still leaves something
    usable. The consequence is that any two benches running CONCURRENTLY see
    each other's output as a dirty tree, and both record ``git_dirty: true``
    from a perfectly clean checkout -- which is exactly what happened to the
    section-7 figures, and it made every one of them look unfit for the
    manuscript when the code behind them was pinned and clean. Serialising the
    runs to avoid it would cost about ten hours of GPU for no gain in accuracy.
    """

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, cwd=_REPO_ROOT, check=False
            ).stdout.strip()
        except OSError:  # pragma: no cover
            return ""

    porcelain = run("git", "status", "--porcelain")

    def paths_of(line: str) -> list[str]:
        """Return the path(s) one porcelain line refers to.

        Parameters
        ----------
        line : str
            One line of ``git status --porcelain`` output.

        Returns
        -------
        list[str]
            The path, or both paths of a rename. The two status characters are
            followed by a space, but staged and unstaged entries pad
            differently, so this strips rather than slicing at a fixed column
            -- slicing at 3 silently ate the first character of every staged
            path and produced "ench/results/...".
        """
        body = line[2:].strip()
        return [part.strip().strip('"') for part in body.split(" -> ") if part.strip()]

    source_changes = [
        line
        for line in porcelain.splitlines()
        if paths_of(line)
        and not all(path.startswith("bench/results/") for path in paths_of(line))
    ]
    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        # The whole tree, kept for continuity with artifacts already committed.
        "git_dirty": bool(porcelain),
        # The tree EXCLUDING bench/results/. This is the one that says whether
        # the recorded sha describes the code that produced the number.
        "git_dirty_sources": bool(source_changes),
        "git_dirty_source_paths": sorted(
            {path for line in source_changes for path in paths_of(line)}
        )[:20],
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
    # When the measurement happened, recorded by the run itself. The export's
    # provenance table used to take this from the artifact file's mtime, which
    # is the CHECKOUT time in a fresh worktree -- so exporting from a different
    # checkout silently rewrote every figure's date to the day the tree was
    # created. A date the run writes down cannot drift that way.
    meta["measured_at"] = (
        datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    if extra:
        meta.update(extra)
    return meta


def commit_date(sha: str) -> str:
    """Return the ISO author-committer date of ``sha``, or ``""`` if unknown.

    Used to order artifact provenance when the artifacts predate the
    ``measured_at`` field: a commit date is not when the number was measured,
    but it bounds it from below and it is the only ordering available.

    Parameters
    ----------
    sha : str
        A commit-ish resolvable in this repository.

    Returns
    -------
    str
        ISO-8601 committer date, or the empty string.
    """
    if not sha:
        return ""
    try:
        done = subprocess.run(
            ["git", "show", "-s", "--format=%cI", sha],
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
            check=False,
        )
    except OSError:  # pragma: no cover
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def merged_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Return provenance for an artifact merged from per-slice artifacts.

    A merge is bookkeeping, not a measurement, so the merged artifact must not
    advertise the sha it was merged AT -- that commit ran no benchmark, and
    stamping it over the slices' shas is how a provenance table came to name a
    commit that measured nothing. The merge-time state is kept under
    ``merged_at`` / ``merged_sha``, and the measurement fields are carried up
    from the slices.

    Where slices disagree the newest measurement wins, ordered by
    ``measured_at`` and falling back to the commit date of each slice's sha for
    artifacts written before ``measured_at`` existed. Dirtiness is OR-ed: one
    dirty slice makes the merged result dirty.

    Parameters
    ----------
    metas : list of dict
        The ``meta`` block of each source artifact, in any order.

    Returns
    -------
    dict
        Provenance for the merged artifact.
    """
    now = run_meta()
    out: dict[str, Any] = {
        **now,
        "merged_source_meta": metas,
        "merged_at": now.get("measured_at"),
        "merged_sha": now.get("git_sha"),
    }
    out.pop("measured_at", None)
    indexed = [(i, m) for i, m in enumerate(metas) if m.get("git_sha")]
    if not indexed:
        return out
    order = {
        i: (m.get("measured_at") or commit_date(str(m.get("git_sha"))))
        for i, m in indexed
    }
    newest = max(indexed, key=lambda pair: order[pair[0]])[1]
    shas = [str(m.get("git_sha")) for _, m in indexed]
    out["git_sha"] = str(newest.get("git_sha"))
    out["git_branch"] = newest.get("git_branch", out.get("git_branch"))
    out["git_dirty"] = any(m.get("git_dirty") for m in metas)
    out["git_dirty_sources"] = any(m.get("git_dirty_sources") for m in metas)
    out["git_dirty_source_paths"] = sorted(
        {p for m in metas for p in (m.get("git_dirty_source_paths") or [])}
    )[:20]
    if newest.get("measured_at"):
        out["measured_at"] = newest["measured_at"]
    if len(set(shas)) > 1:
        out["measured_shas"] = sorted(set(shas))
    return out


def device_label() -> str:
    """Return a short device label (e.g. ``NVIDIA A100-PCIE-40GB``) for the JSON."""

    try:
        import jax

        return str(jax.devices()[0].device_kind)
    except Exception:  # pragma: no cover
        return "unknown"

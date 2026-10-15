"""Nightly GPU pins on the main heads (galaxy production plan, S1 item I1).

A local cron job, deliberately not a CI leg: a self-hosted runner on a public
repository would let a fork PR run code on the shared box (``bench/gpu_gate.py``,
lines 3-8). The cron entry is ``bench/nightly_pins.sh``; this is the driver. It
needs only the standard library, so the system ``python3`` runs it, and every GPU
process runs in the jax 0.11.2 venv.

Each run:

1. fetches jaccpot, yggdrax and nornax ``main`` and Odisseo ``jaccpot-integration``
   (pins S4b and M2 import Odisseo's bench harness, ``benchmark_multigpu``), and
   moves DEDICATED detached worktrees (``--worktrees``) to those heads with a
   ``checkout --force`` and ``clean -ffdx``. The shared checkouts are never touched;
2. writes an import repoint (``<home>/site/sitecustomize.py``) for the venv and fails
   unless jaccpot, yggdrax, nornax, odisseo and the bench harness all resolve into
   those worktrees;
3. books cards with autocvd (the org rule) by its absolute path (``--autocvd``; cron
   has a minimal PATH): first, while the night is quietest, the L pins on a FREE card
   (``-n 1`` without ``-l``) -- with no card free within ``--free-wait-min`` they are
   SKIPPED, never timed on a busy card -- then the S and DET pins on the least-used
   card (``-l -n 1``), the M pins on the two least-used (``-l -n 2``) and
   ``bench/gpu_gate.py`` on the least-used card. After every booking a probe must see
   jax on that many GPUs, and the pins run with ``JAX_PLATFORMS=cuda,cpu``, so a
   process that cannot reach the card raises instead of falling back to the CPU;
4. records ``bench/dce_pins.py`` under ``nightly-<YYYYMMDD>-<jaccpot sha7>`` in
   ``--pins-root`` and judges each pin:

   * S1-S6, M1, M3: BITWISE (every array, and the metadata) against the latest
     earlier nightly that recorded the pin on a different stack (the four heads, jax,
     jaxlib, the driver); the commits in between are listed. With no such nightly
     the pin is NEW, the reference for the next change;
   * M2: its A-vs-A control is not bitwise (ulp-level, ``docs/cleanup_2026-10.md``),
     so every nightly records it twice (``<tag>-b``) and the next one compares
     with ``--envelope`` the reference's own control;
   * DET1, DET2: against a same-night re-record under ``<tag>-b``, bitwise;
   * L1-L3: against ``--l-ref`` with ``--envelope --l-envelope`` (default
     ``main-5bc4959`` / ``main-5bc4959-b``). A contaminated row (another process
     on the card) is INCONCLUSIVE, not FAIL;
   * ``gate``: ``bench/gpu_gate.py`` as its docstring says (its own preflight,
     six workers, deterministic ops), PASS on its exit code 0;

5. writes ``<home>/<tag>/SUMMARY.md`` (one screen) and ``run.json``, appends one
   line to ``<home>/history.log`` and exits 1 on any FAIL.

An item that passed (or was recorded NEW) on the identical stack is not rerun;
when nothing is left the run only appends an UNCHANGED line. ``--force`` reruns
everything.

Process control. A lock file keeps two runs from overlapping (exit 3 when busy).
``--total-cap-h`` bounds the whole run; each record has its own wall-clock limit.
Nothing is ever killed for being silent: at its limit a record's process tree is
stack-dumped (py-spy from ``<home>/tools``, else gdb; the repoint opts every venv
process in with ``PR_SET_PTRACER_ANY``) and, when a frame is inside XLA's compiler,
granted one ``--compile-grace-min`` extension first. Only then is its process group
killed and the item FAILs as a TIMEOUT, with the stacks in its log.

Usage::

    python3 bench/nightly_pins.py --dry-run          # print the plan, change nothing
    python3 bench/nightly_pins.py --pins S1,DET1     # a subset (groups: S M L DET gate)
    python3 bench/nightly_pins.py                    # everything not yet passed

Crontab (the lead installs it; the first line is the merged copy in the nightly
worktree, which the run itself keeps at ``main``)::

    30 2 * * * /export/home/tbuck/nightly/jaccpot/bench/nightly_pins.sh >> /export/scratch/tbuck/nightly/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HOME_DEFAULT = "/export/scratch/tbuck/nightly"
WORKTREES_DEFAULT = "/export/home/tbuck/nightly"
PINS_ROOT_DEFAULT = "/export/scratch/tbuck/dce_pins"
VENV_PYTHON_DEFAULT = "/export/scratch/tbuck/jax0112-venv/bin/python"
OVERLAY_DEFAULT = "/export/scratch/tbuck/odisseo-overlay"
AUTOCVD_DEFAULT = "/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd"
L_REF_DEFAULT = "main-5bc4959"
L_ENVELOPE_DEFAULT = "main-5bc4959-b"


@dataclass(frozen=True)
class Repo:
    name: str
    source: str
    branch: str
    package: str


REPOS = (
    Repo("jaccpot", "/export/home/tbuck/jaccpot", "main", "jaccpot"),
    Repo("yggdrax", "/export/home/tbuck/yggdrax", "main", "yggdrax"),
    Repo("nornax", "/export/home/tbuck/nornax", "main", "nornax"),
    Repo("Odisseo", "/export/home/tbuck/Odisseo", "jaccpot-integration", "odisseo"),
)

S_PINS = ("S1", "S2", "S3", "S4", "S4b", "S5", "S6")
DET_PINS = ("DET1", "DET2")
M_PINS = ("M1", "M2", "M3")
L_PINS = ("L1", "L2", "L3")
ITEMS = S_PINS + DET_PINS + M_PINS + L_PINS + ("gate",)
GROUPS = {"S": S_PINS, "DET": DET_PINS, "M": M_PINS, "L": L_PINS, "all": ITEMS}
#: recorded a second time under ``<tag>-b``: the DET pins' A-vs-A, and M2's envelope
CONTROLLED = DET_PINS + ("M2",)
#: wall-clock limit of one record, minutes: generous, since a busy card compiles
#: slowly (the 2026-10-15 trial on a shared A100: S1 8.4 min, DET1 and its re-record
#: 20.1 min together); the M, L and gate limits are not measured here yet
LIMIT_MIN = {
    **{pin: 45 for pin in S_PINS + DET_PINS},
    "M1": 40,
    "M2": 60,
    "M3": 40,
    "L1": 40,
    "L2": 75,
    "L3": 60,
    "gate": 240,
}


@dataclass(frozen=True)
class Stage:
    name: str
    items: tuple[str, ...]
    ncards: int
    free: bool


#: run in this order: the timings first, while the night is quietest
STAGES = (
    Stage("timing", L_PINS, 1, True),
    Stage("single", S_PINS + DET_PINS, 1, False),
    Stage("two-card", M_PINS, 2, False),
    Stage("gate", ("gate",), 1, False),
)

PASSED = ("PASS", "NEW")
#: frames that mean "still compiling" (py-spy's Python frames, gdb's native ones)
COMPILE_RE = re.compile(
    r"Compil|compile_or_get_cached|backend_compile|_cached_compilation|Autotun"
    r"|trace_to_jaxpr|llvm::|ptxas|NVPTX"
)
_COMPILER_COMMS = ("ptxas", "nvlink", "cicc")


class Aborted(Exception):
    """A signal (cron's backstop ``timeout``, Ctrl-C) ended the run."""


# ------------------------------------------------------------- pure helpers


def expand_items(spec: str) -> list[str]:
    """``"S1,M,gate"`` -> the items, in run order; unknown names raise."""
    wanted: set[str] = set()
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        if token in GROUPS:
            wanted.update(GROUPS[token])
        elif token in ITEMS:
            wanted.add(token)
        else:
            raise SystemExit(f"unknown pin {token!r}; known: {', '.join(ITEMS)}")
    return [item for item in ITEMS if item in wanted]


def parse_autocvd(text: str) -> dict[str, str]:
    """The ``CUDA_*`` assignments of ``autocvd -e``'s output."""
    pairs = re.findall(r"(CUDA_[A-Z_]+)=(\S*)", text)
    env = dict(pairs)
    if not env.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError(f"no CUDA_VISIBLE_DEVICES in autocvd output {text!r}")
    return env


def is_compiling(stacks: str) -> bool:
    return bool(COMPILE_RE.search(stacks))


def already_passed(runs: list[dict], key: dict) -> set[str]:
    """Items that passed, or were recorded NEW, on this exact stack."""
    return {
        name
        for run in runs
        if run.get("key") == key
        for name, item in run.get("items", {}).items()
        if item.get("verdict") in PASSED
    }


def find_reference(
    runs: list[dict], pin: str, key: dict, need_control: bool = False
) -> dict | None:
    """The latest earlier run that recorded ``pin`` on a different stack."""
    for run in sorted(runs, key=lambda r: r.get("started", ""), reverse=True):
        if run.get("key") == key:
            continue
        item = run.get("items", {}).get(pin, {})
        if not item.get("recorded"):
            continue
        if need_control and not item.get("control_recorded"):
            continue
        return run
    return None


def _array_rows(entry: dict) -> list[str]:
    out = []
    for name, row in sorted(entry.get("arrays", {}).items()):
        if row.get("bitwise"):
            continue
        if row.get("missing"):
            out.append(f"{name} missing")
        elif "shape" in row:
            out.append(f"{name} shape {row['shape']}")
        else:
            out.append(f"{name} max|d| {row['max_abs']:.3g} rel-L2 {row['rel_l2']:.3g}")
    return out


def _missing(entry: dict | None) -> str | None:
    if not entry:
        return "no comparison row"
    if entry.get("missing_in_ref"):
        return "missing in the reference"
    if entry.get("missing"):
        return "missing in this record"
    return None


def judge_bitwise(entry: dict | None) -> tuple[str, str]:
    """S/M pins: every array bitwise and the metadata identical."""
    gone = _missing(entry)
    if gone:
        return "FAIL", gone
    moved = _array_rows(entry)
    if moved:
        return "FAIL", "moved: " + "; ".join(moved)
    meta = entry.get("meta", {})
    if meta.get("ref") != meta.get("new"):
        keys = sorted(
            k
            for k in set(meta.get("ref", {})) | set(meta.get("new", {}))
            if meta.get("ref", {}).get(k) != meta.get("new", {}).get(k)
        )
        return "FAIL", f"arrays bitwise, metadata differs: {', '.join(keys)}"
    return "PASS", f"{len(entry.get('arrays', {}))} arrays bitwise"


def judge_envelope(entry: dict | None) -> tuple[str, str]:
    """M2 against the reference's own A-vs-A control; DET1/DET2 against tonight's."""
    gone = _missing(entry)
    if gone:
        return "FAIL", gone
    moved = _array_rows(entry)
    if entry.get("pass"):
        return "PASS", (
            ("within the control: " + "; ".join(moved))
            if moved
            else (f"{len(entry.get('arrays', {}))} arrays bitwise")
        )
    return "FAIL", "; ".join(moved) or "metadata differs"


def judge_timed(entry: dict | None) -> tuple[str, str]:
    """L pins: ``dce_pins compare --envelope``; a contaminated row is INCONCLUSIVE."""
    gone = _missing(entry)
    if gone:
        return "FAIL", gone
    parts, contaminated = [], False
    for name, row in sorted(entry.get("rows", {}).items()):
        if row.get("missing"):
            parts.append(f"{name} row missing")
            continue
        contaminated = contaminated or bool(row.get("contaminated"))
        text = (
            f"{name} {row['ms_new']:.1f} ms ({row['time_change']:+.1%} vs "
            f"{row['ms_ref']:.1f}, limit {row['time_limit']:.0%}), peak "
            f"{row['peak_change']:+.1%}"
        )
        if "rel_l2" in row:
            text += f", rel-L2 {row['rel_l2'][1]:.4g}"
        parts.append(text)
    new = entry.get("meta", {}).get("new", {})
    bpp = (new.get("step") or {}).get("bytes_per_particle")
    if bpp:
        parts.append(f"{bpp:.0f} B/p")
    detail = "; ".join(parts)
    if entry.get("pass"):
        return "PASS", detail
    if contaminated:
        return "INCONCLUSIVE", "contaminated (another process on the card): " + detail
    return "FAIL", detail


def judge_gate(rc: int | None, log_text: str) -> tuple[str, str]:
    """``bench/gpu_gate.py``: exit 0 passes; the detail is the gate's own verdict."""
    lines = log_text.splitlines()
    keep = [
        ln.strip()
        for ln in lines
        if re.search(
            r"GATE (PASSED|FAILED)|UNEXPECTED failures|known GPU failures|VACUOUS"
            r"|PARSE MISMATCH|^FAIL:",
            ln,
        )
    ]
    tally = [ln.strip("= ").strip() for ln in lines if re.search(r"\d+ passed", ln)]
    detail = "; ".join(keep[-4:] + tally[-1:]) or "no gate verdict in the log"
    if rc == 0:
        return "PASS", detail
    return "FAIL", f"exit {rc}: {detail}"


def overall_status(items: dict[str, dict]) -> str:
    verdicts = [item.get("verdict") for item in items.values()]
    if any(v == "FAIL" for v in verdicts):
        return "FAIL"
    if all(v in PASSED for v in verdicts):
        return "PASS"
    return "PARTIAL"


# ------------------------------------------------------------- the driver


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _git(repo: str | Path, *args: str, timeout: int = 600) -> str:
    env = dict(
        os.environ,
        GIT_TERMINAL_PROMPT="0",
        GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=30",
    )
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"git -C {repo} {' '.join(args)}: {out.stderr.strip()[-500:]}"
        )
    return out.stdout.strip()


def _descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text()
        except OSError:
            continue
        ppid = int(stat.rsplit(")", 1)[1].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(children.get(p, []))
    return out


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return "?"


class Nightly:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.home = Path(args.home)
        self.wt_root = Path(args.worktrees)
        self.pins_root = Path(args.pins_root)
        self.wt = {repo.name: self.wt_root / repo.name for repo in REPOS}
        self.start = time.time()
        self.deadline = self.start + args.total_cap_h * 3600.0
        self.child: subprocess.Popen | None = None
        self.items: dict[str, dict] = {}
        self.run: dict = {}
        self.tag = ""

    # ---- small utilities

    def remaining(self) -> float:
        return self.deadline - time.time()

    def say(self, msg: str) -> None:
        print(f"[{_now()}] {msg}", flush=True)

    def py(self, script: str) -> list[str]:
        return [self.args.python, str(self.wt["jaccpot"] / "bench" / script)]

    # ---- 1. worktrees

    def head(self, repo: Repo, fetch: bool) -> str:
        if fetch:
            spec = f"+refs/heads/{repo.branch}:refs/remotes/origin/{repo.branch}"
            for attempt in range(3):
                try:
                    _git(repo.source, "fetch", "--quiet", "origin", spec)
                    break
                except (RuntimeError, subprocess.TimeoutExpired):
                    if attempt == 2:
                        raise
                    time.sleep(60)
        return _git(repo.source, "rev-parse", f"origin/{repo.branch}")

    def update_worktree(self, repo: Repo, sha: str) -> None:
        path = self.wt[repo.name]
        if not path.exists():
            self.wt_root.mkdir(parents=True, exist_ok=True)
            _git(repo.source, "worktree", "add", "--detach", str(path), sha)
            return
        # only ever reset a worktree of the right repository at the right path
        top = Path(_git(path, "rev-parse", "--show-toplevel")).resolve()
        common = Path(
            _git(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        source_common = Path(
            _git(repo.source, "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        if top != path.resolve() or common.resolve() != source_common.resolve():
            raise RuntimeError(f"{path} is not a worktree of {repo.source}")
        _git(path, "checkout", "--quiet", "--force", "--detach", sha)
        _git(path, "clean", "-ffdxq")

    # ---- 2. environment

    def site_dir(self) -> Path:
        return self.home / "site"

    def write_site(self) -> None:
        site = self.site_dir()
        site.mkdir(parents=True, exist_ok=True)
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)

    def env(self, gpu: dict[str, str] | None, platforms: str | None) -> dict[str, str]:
        """The pins' environment: the caller's minus every knob that changes a run."""
        strip = ("JACCPOT_", "YGGDRAX_", "NORNAX_", "JAX_", "XLA_", "S1_", "NIGHTLY_")
        drop = {
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "CUDA_VISIBLE_DEVICES",
            "CUDA_DEVICE_ORDER",
            "BENCH_DIR",
            "VIRTUAL_ENV",
            "COVERAGE_PROCESS_START",
        }
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in drop and not k.startswith(strip)
        }
        wt = self.wt
        env.update(
            PATH=f"{Path(self.args.python).parent}:/usr/local/bin:/usr/bin:/bin",
            PYTHONPATH=":".join(
                [str(self.site_dir())]
                + [str(wt[r.name]) for r in REPOS]
                + [self.args.overlay]
            ),
            PYTHONUNBUFFERED="1",
            NIGHTLY_JACCPOT=str(wt["jaccpot"]),
            NIGHTLY_YGGDRAX=str(wt["yggdrax"]),
            NIGHTLY_PTRACER_ANY="1",
            YGGDRAX_WORKTREE=str(wt["yggdrax"]),
            NORNAX_WORKTREE=str(wt["nornax"]),
            BENCH_DIR=str(wt["Odisseo"] / "benchmark_multigpu"),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        if gpu is None:
            env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
        else:
            env.update(gpu)
        if platforms and gpu is not None:
            env["JAX_PLATFORMS"] = platforms
        return env

    def check_resolution(self) -> dict[str, str]:
        out = subprocess.run(
            [self.args.python, "-c", _RESOLVE],
            cwd=self.wt["jaccpot"],
            env=self.env(None, None),
            capture_output=True,
            text=True,
            timeout=600,
        )
        line = [
            ln for ln in out.stdout.splitlines() if ln.startswith("NIGHTLY_RESOLVE ")
        ]
        if out.returncode != 0 or not line:
            raise RuntimeError(f"import check failed: {out.stderr.strip()[-1500:]}")
        found = json.loads(line[-1].split(" ", 1)[1])
        expect = {r.package: self.wt[r.name] / r.package for r in REPOS}
        harness = self.wt["Odisseo"] / "benchmark_multigpu"
        expect.update({"codes.compare_force": harness, "common.ic": harness})
        bad = [
            f"{name} -> {found[name]} (want under {root})"
            for name, root in expect.items()
            if not found[name].startswith(str(root.resolve()) + os.sep)
        ]
        if bad:
            raise RuntimeError(
                "modules resolve outside the nightly worktrees: " + "; ".join(bad)
            )
        return found

    def stack_versions(self) -> dict[str, str]:
        out = subprocess.run(
            [
                self.args.python,
                "-c",
                "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                "PATH": "/usr/bin:/bin",
                "JAX_PLATFORMS": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
            },
            cwd="/",
        )
        jax_v, jaxlib_v = out.stdout.split()[-2:]
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        driver = smi.stdout.split()[0] if smi.stdout.split() else "?"
        return {"jax": jax_v, "jaxlib": jaxlib_v, "driver": driver}

    # ---- 3. cards

    def autocvd_cmd(self, stage: Stage, wait_s: int) -> list[str]:
        cmd = [
            self.args.autocvd,
            "-q",
            "-n",
            str(stage.ncards),
            "-e",
            "-t",
            str(wait_s),
        ]
        if not stage.free:
            cmd.insert(2, "-l")
        return cmd

    def book(self, stage: Stage, wait_s: int) -> tuple[dict[str, str] | None, str]:
        cmd = self.autocvd_cmd(stage, wait_s)
        self.say("booking: " + " ".join(cmd))
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=wait_s + 300)
        if out.returncode != 0:
            if stage.free and "TimeoutError" in out.stderr:
                return None, f"skipped: no free card within {wait_s // 60} min"
            return None, f"autocvd failed: {out.stderr.strip()[-300:]}"
        return parse_autocvd(out.stdout), ""

    def probe(self, gpu: dict[str, str], ncards: int) -> str:
        out = subprocess.run(
            [self.args.python, "-c", _PROBE],
            cwd=self.wt["jaccpot"],
            env=self.env(gpu, "cuda,cpu"),
            capture_output=True,
            text=True,
            timeout=600,
        )
        line = [ln for ln in out.stdout.splitlines() if ln.startswith("NIGHTLY_PROBE ")]
        if out.returncode != 0 or not line:
            raise RuntimeError(f"GPU probe failed: {out.stderr.strip()[-800:]}")
        _, backend, count, kind = line[-1].split()
        if backend != "gpu" or int(count) != ncards:
            raise RuntimeError(
                f"GPU probe: backend {backend}, {count} devices, want gpu x{ncards}"
            )
        return kind

    # ---- 4. run one process under the limits

    def stacks(self, pid: int) -> str:
        spy = self.home / "tools" / "bin" / "py-spy"
        parts = []
        for p in _descendants(pid):
            comm = _comm(p)
            if "python" not in comm:
                parts.append(f"--- pid {p} ({comm})")
                continue
            if spy.exists():
                cmd = [str(spy), "dump", "--native", "--pid", str(p)]
            else:
                cmd = ["gdb", "-p", str(p), "-batch", "-ex", "thread apply all bt"]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                text = out.stdout + out.stderr
            except subprocess.TimeoutExpired:
                text = "(stack dump timed out)"
            parts.append(f"--- pid {p} ({comm})\n{text}")
        return "\n".join(parts)

    def execute(
        self, cmd: list[str], env: dict, log: Path, limit_s: float
    ) -> tuple[int | None, str]:
        """Run ``cmd`` in its own session: ``(exit code or None on a timeout, note)``.

        The limit is wall-clock only; at it the process tree is stack-dumped and,
        while XLA is compiling, granted one ``--compile-grace-min`` extension.
        """
        log.parent.mkdir(parents=True, exist_ok=True)
        limit_s = min(limit_s, self.remaining())
        with open(log, "a") as fh:
            fh.write(f"[{_now()}] $ {shlex.join(cmd)}\n")
            fh.flush()
            self.child = subprocess.Popen(
                cmd,
                cwd=self.wt["jaccpot"],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        t0 = time.time()
        end, extended = t0 + limit_s, False
        try:
            while True:
                try:
                    return self.child.wait(timeout=max(1.0, end - time.time())), ""
                except subprocess.TimeoutExpired:
                    pass
                dump = self.stacks(self.child.pid)
                comms = {_comm(p) for p in _descendants(self.child.pid)}
                compiling = is_compiling(dump) or bool(comms & set(_COMPILER_COMMS))
                with open(log, "a") as fh:
                    fh.write(
                        f"\n[{_now()}] limit reached; compiling={compiling}\n{dump}\n"
                    )
                grace = self.args.compile_grace_min * 60.0
                if compiling and not extended and self.remaining() > grace:
                    self.say(
                        f"  limit reached while compiling: {grace / 60:g} min grace"
                    )
                    end, extended = time.time() + grace, True
                    continue
                self.kill()
                why = (
                    "still compiling after the grace" if compiling else "not compiling"
                )
                return None, f"TIMEOUT after {(time.time() - t0) / 60:.0f} min ({why})"
        except Aborted:
            self.kill()
            raise
        finally:
            self.child = None

    def kill(self) -> None:
        if self.child is None or self.child.poll() is not None:
            return
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 30)):
            try:
                os.killpg(self.child.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.child.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                continue

    # ---- 5. compare

    def compare(
        self, ref: str, new: str, pin: str, envelope: str | None
    ) -> dict | None:
        out_json = self.home / self.tag / "compare" / f"{pin}_vs_{ref}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.py("dce_pins.py") + [
            "compare",
            ref,
            new,
            "--pins",
            pin,
            "--root",
            str(self.pins_root),
            "--json",
            str(out_json),
        ]
        if envelope:
            cmd += ["--envelope", envelope]
        subprocess.run(
            cmd,
            cwd=self.wt["jaccpot"],
            env=self.env(None, None),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if not out_json.exists():
            return None
        return json.loads(out_json.read_text()).get(pin)

    def judge(self, pin: str, runs: list[dict], key: dict) -> None:
        item = self.items[pin]
        if pin in L_PINS:
            if not (self.pins_root / self.args.l_ref / f"{pin}.json").exists():
                item.update(verdict="FAIL", detail=f"no {pin} in {self.args.l_ref}")
                return
            entry = self.compare(self.args.l_ref, self.tag, pin, self.args.l_envelope)
            item["ref"] = self.args.l_ref
            item["verdict"], item["detail"] = judge_timed(entry)
        elif pin in DET_PINS:
            if not item.get("control_recorded"):
                item.update(verdict="FAIL", detail="the A-vs-A re-record failed")
                return
            entry = self.compare(self.tag, self.tag + "-b", pin, None)
            item["ref"] = self.tag + "-b"
            item["verdict"], item["detail"] = judge_envelope(entry)
        else:
            ref = find_reference(runs, pin, key, need_control=(pin == "M2"))
            if (
                ref is not None
                and not (self.pins_root / ref["tag"] / f"{pin}.npz").exists()
            ):
                ref = None
            if ref is None:
                item.update(
                    verdict="NEW",
                    detail="first record on a new stack: no earlier nightly",
                )
                return
            item["ref"] = ref["tag"]
            if pin == "M2":
                entry = self.compare(ref["tag"], self.tag, pin, ref["tag"] + "-b")
                item["verdict"], item["detail"] = judge_envelope(entry)
            else:
                entry = self.compare(ref["tag"], self.tag, pin, None)
                item["verdict"], item["detail"] = judge_bitwise(entry)

    # ---- the stages

    def run_stage(
        self, stage: Stage, todo: list[str], runs: list[dict], key: dict
    ) -> None:
        names = [n for n in stage.items if n in todo]
        if not names:
            return
        logs = self.home / self.tag / "logs"
        if stage.free:
            l_limit = sum(LIMIT_MIN[n] for n in names) * 60
            wait_s = int(min(self.args.free_wait_min * 60, self.remaining() - l_limit))
        else:
            wait_s = 600
        if wait_s < 60 or self.remaining() < 300:
            for n in names:
                self.items[n].update(verdict="SKIP", detail="skipped: total time cap")
            return
        gpu, why = self.book(stage, wait_s)
        if gpu is None:
            verdict = "SKIP" if why.startswith("skipped") else "FAIL"
            for n in names:
                self.items[n].update(verdict=verdict, detail=why)
            return
        cards = gpu["CUDA_VISIBLE_DEVICES"]
        try:
            kind = self.probe(gpu, stage.ncards)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            for n in names:
                self.items[n].update(verdict="FAIL", detail=str(exc)[:300], cards=cards)
            return
        self.say(f"stage {stage.name}: GPU {cards} ({kind}): {', '.join(names)}")
        for n in names:
            item = self.items[n]
            item["cards"] = cards
            if self.remaining() < 300:
                item.update(verdict="SKIP", detail="skipped: total time cap")
                continue
            t0 = time.time()
            if n == "gate":
                env = self.env(gpu, None)  # the gate's own preflight checks the backend
                rc, note = self.execute(
                    [
                        self.args.python,
                        str(self.wt["jaccpot"] / "bench" / "gpu_gate.py"),
                    ],
                    env,
                    logs / "gate.log",
                    LIMIT_MIN["gate"] * 60,
                )
                item["wall_s"] = round(time.time() - t0, 1)
                item["verdict"], item["detail"] = (
                    ("FAIL", note)
                    if rc is None
                    else judge_gate(rc, (logs / "gate.log").read_text())
                )
                self.save()
                continue
            env = self.env(gpu, "cuda,cpu")
            tags = [self.tag] + ([self.tag + "-b"] if n in CONTROLLED else [])
            for tag in tags:
                rc, note = self.execute(
                    self.py("dce_pins.py")
                    + [
                        "record",
                        "--tag",
                        tag,
                        "--pins",
                        n,
                        "--root",
                        str(self.pins_root),
                    ],
                    env,
                    logs / f"{n}.log",
                    LIMIT_MIN[n] * 60,
                )
                ok = rc == 0 and (self.pins_root / tag / f"{n}.json").exists()
                item["recorded" if tag == self.tag else "control_recorded"] = ok
                if not ok:
                    item.update(
                        verdict="FAIL",
                        detail=note
                        or f"record {tag} failed (exit {rc}); see logs/{n}.log",
                    )
                    break
            item["wall_s"] = round(time.time() - t0, 1)
            if "verdict" not in item:
                self.judge(n, runs, key)
            self.say(f"  {n}: {item['verdict']} -- {item.get('detail', '')}")
            self.save()

    # ---- bookkeeping

    def runs(self) -> list[dict]:
        out = []
        for path in sorted(self.home.glob("*/run.json")):
            if path.parent.name == self.tag:
                continue
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def unique_tag(self, sha7: str) -> str:
        base = f"nightly-{_dt.date.today():%Y%m%d}-{sha7}"
        tag, k = base, 1
        while any(
            p.exists()
            for p in (
                self.home / tag,
                self.pins_root / tag,
                self.pins_root / f"{tag}-b",
            )
        ):
            k += 1
            tag = f"{base}-{k}"
        return tag

    def save(self) -> None:
        self.run["items"] = self.items
        path = self.home / self.tag / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.run, indent=1, sort_keys=True))

    def changes(self, ref_run: dict) -> list[str]:
        """What moved between a reference run's stack and this one."""
        out = []
        for repo in REPOS:
            old = ref_run["key"].get(repo.name)
            new = self.run["key"][repo.name]
            if old == new:
                continue
            try:
                log = _git(
                    self.wt[repo.name],
                    "log",
                    "--oneline",
                    "--first-parent",
                    f"{old}..{new}",
                )
            except RuntimeError as exc:
                log = f"(git log failed: {exc})"
            lines = log.splitlines()
            out.append(f"{repo.name} {old[:7]}..{new[:7]}: {len(lines)} commit(s)")
            out += [f"    {ln}" for ln in lines[:6]] + (
                ["    ..."] if len(lines) > 6 else []
            )
        for name in ("jax", "jaxlib", "driver"):
            if ref_run["key"].get(name) != self.run["key"][name]:
                out.append(
                    f"{name} {ref_run['key'].get(name)} -> {self.run['key'][name]}"
                )
        return out

    def summary(self, runs: list[dict]) -> str:
        r = self.run
        mins = (time.time() - self.start) / 60
        lines = [
            f"# {self.tag}: {r['status']}",
            "",
            f"{r['started']} -> {r.get('finished', _now())} ({mins:.0f} min, cap "
            f"{self.args.total_cap_h:g} h) on {socket.gethostname()}; "
            f"jax {r['key']['jax']}, jaxlib {r['key']['jaxlib']}, "
            f"driver {r['key']['driver']}",
            "",
            "| repo | branch | sha | head |",
            "| --- | --- | --- | --- |",
        ]
        for repo in REPOS:
            info = r["repos"][repo.name]
            lines.append(
                f"| {repo.name} | {repo.branch} | {info['sha'][:7]} "
                f"| {info['subject'][:60]} |"
            )
        lines += [
            "",
            "| item | verdict | against | GPU | wall | detail |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, item in self.items.items():
            wall = f"{item['wall_s'] / 60:.1f} min" if item.get("wall_s") else ""
            lines.append(
                f"| {name} | {item.get('verdict', '?')} | {item.get('ref', '')} | "
                f"{item.get('cards', '')} | {wall} | {item.get('detail', '')[:160]} |"
            )
        if r.get("pruned"):
            lines += [
                "",
                f"Not rerun (passed on this exact stack): {', '.join(r['pruned'])}",
            ]
        if r.get("error"):
            lines += ["", f"**Setup error:** {r['error']}"]
        by_tag = {run["tag"]: run for run in runs}
        refs = sorted({i["ref"] for i in self.items.values() if i.get("ref") in by_tag})
        for ref in refs:
            moved = self.changes(by_tag[ref])
            lines += ["", f"Changes since {ref}:"] + [
                f"- {ln}" if not ln.startswith("    ") else ln for ln in moved
            ]
        lines += [
            "",
            f"Records: {self.pins_root / self.tag}; "
            f"logs: {self.home / self.tag / 'logs'}",
        ]
        return "\n".join(lines) + "\n"

    def history(self, tag: str, status: str, counts: str) -> None:
        heads = " ".join(
            f"{repo.name}={self.run['key'][repo.name][:7]}" for repo in REPOS
        )
        mins = (time.time() - self.start) / 60
        jax_v = self.run["key"].get("jax")
        with open(self.home / "history.log", "a") as fh:
            fh.write(
                f"{_now()} {tag} {status} {counts} {heads} jax={jax_v} {mins:.0f}min\n"
            )

    # ---- main

    def main(self) -> int:
        args = self.args
        todo = expand_items(args.pins)
        if args.dry_run:
            return self.dry_run(todo)
        # both the outputs and the worktrees: a second --home must not move the
        # worktrees under a running nightly either
        locks = []
        for path in (self.home / "nightly.lock", self.wt_root / ".nightly.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = open(path, "a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.seek(0)
                holder = lock.read().strip()
                print(f"another nightly holds {path}: {holder}", file=sys.stderr)
                with open(self.home / "history.log", "a") as fh:
                    fh.write(f"{_now()} - LOCKED ({path} held by {holder})\n")
                return 3
            lock.seek(0)
            lock.truncate()
            lock.write(f"pid {os.getpid()} since {_now()}\n")
            lock.flush()
            locks.append(lock)  # held until this process exits

        def _abort(signum, _frame):
            raise Aborted(signal.Signals(signum).name)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, _abort)

        self.run = {
            "started": _now(),
            "status": "RUNNING",
            "requested": todo,
            "key": {},
            "repos": {},
        }
        try:
            heads = {}
            for repo in REPOS:
                heads[repo.name] = self.head(repo, fetch=not args.no_fetch)
            self.tag = self.unique_tag(heads["jaccpot"][:7])
            self.run["tag"] = self.tag
            for repo in REPOS:
                self.update_worktree(repo, heads[repo.name])
                self.run["repos"][repo.name] = {
                    "branch": repo.branch,
                    "sha": heads[repo.name],
                    "subject": _git(self.wt[repo.name], "log", "-1", "--format=%s"),
                }
            self.write_site()
            self.run["resolved"] = self.check_resolution()
            self.run["key"] = {**heads, **self.stack_versions()}
        # any setup failure, a signal included, is the verdict
        except Exception as exc:
            return self.setup_failed(exc, todo)

        runs = self.runs()
        passed = set() if args.force else already_passed(runs, self.run["key"])
        self.run["pruned"] = [n for n in todo if n in passed]
        todo = [n for n in todo if n not in passed]
        if not todo:
            self.say(
                f"nothing to do: {', '.join(self.run['pruned'])} passed on this stack"
            )
            self.history(
                "-", "UNCHANGED", f"({len(self.run['pruned'])} items passed earlier)"
            )
            return 0
        self.items = {n: {} for n in todo}
        self.save()
        self.say(f"tag {self.tag}; items {', '.join(todo)}")
        try:
            for stage in STAGES:
                self.run_stage(stage, todo, runs, self.run["key"])
        except Aborted as exc:
            self.kill()
            for item in self.items.values():
                item.setdefault("verdict", "FAIL")
                item.setdefault("detail", f"aborted by {exc}")
            self.run["error"] = f"aborted by {exc}"
        return self.finish(runs)

    def finish(self, runs: list[dict]) -> int:
        self.run["status"] = overall_status(self.items)
        self.run["finished"] = _now()
        self.save()
        text = self.summary(runs)
        (self.home / self.tag / "SUMMARY.md").write_text(text)
        print(text)
        verdicts = [item.get("verdict") for item in self.items.values()]
        counts = " ".join(
            f"{v.lower()}={verdicts.count(v)}"
            for v in ("PASS", "NEW", "FAIL", "SKIP", "INCONCLUSIVE")
        )
        self.history(self.tag, self.run["status"], counts)
        return 1 if self.run["status"] == "FAIL" else 0

    def setup_failed(self, exc: Exception, todo: list[str]) -> int:
        self.run["error"] = f"{type(exc).__name__}: {exc}"[:1000]
        self.say("setup failed: " + self.run["error"])
        if not self.tag:
            self.tag = f"nightly-{_dt.date.today():%Y%m%d}-setup-{int(self.start)}"
            self.run["tag"] = self.tag
        for repo in REPOS:
            self.run["repos"].setdefault(repo.name, {"sha": "?" * 7, "subject": "?"})
            self.run["key"].setdefault(repo.name, self.run["repos"][repo.name]["sha"])
        for name in ("jax", "jaxlib", "driver"):
            self.run["key"].setdefault(name, "?")
        self.items = {n: {"verdict": "FAIL", "detail": "setup failed"} for n in todo}
        return self.finish([])

    def dry_run(self, todo: list[str]) -> int:
        print(
            "nightly plan -- dry run: nothing is fetched, booked, recorded or written\n"
        )
        heads = {}
        for repo in REPOS:
            sha = self.head(repo, fetch=False)
            heads[repo.name] = sha
            path = self.wt[repo.name]
            action = (
                f"checkout --force --detach {sha[:7]} && clean -ffdx"
                if path.exists()
                else f"git -C {repo.source} worktree add --detach {path} {sha[:7]}"
            )
            print(
                f"  {repo.name:8s} origin/{repo.branch} = {sha[:7]} "
                "(local ref; the run fetches first)"
            )
            print(f"           {path}: {action}")
        key = {**heads, **self.stack_versions()}
        self.tag = self.unique_tag(heads["jaccpot"][:7])
        runs = self.runs()
        passed = set() if self.args.force else already_passed(runs, key)
        print(f"\n  stack: jax {key['jax']}, jaxlib {key['jaxlib']}, ", end="")
        print(f"driver {key['driver']}")
        print(f"  tag: {self.tag} (records in {self.pins_root})")
        print(f"  python: {self.args.python}")
        print(f"  PYTHONPATH={self.env(None, None)['PYTHONPATH']}")
        if passed & set(todo):
            skip = ", ".join(n for n in todo if n in passed)
            print(f"  not rerun (passed on this stack): {skip}")
        todo = [n for n in todo if n not in passed]
        for stage in STAGES:
            names = [n for n in stage.items if n in todo]
            if not names:
                continue
            wait = self.args.free_wait_min * 60 if stage.free else 600
            print(
                f"\n  stage {stage.name}: "
                f"{' '.join(self.autocvd_cmd(stage, int(wait)))}"
            )
            for n in names:
                if n == "gate":
                    print(
                        f"    gate: {self.args.python} bench/gpu_gate.py "
                        f"(limit {LIMIT_MIN[n]} min)"
                    )
                    continue
                tags = [self.tag] + ([self.tag + "-b"] if n in CONTROLLED else [])
                if n in L_PINS:
                    against = f"{self.args.l_ref} --envelope {self.args.l_envelope}"
                elif n in DET_PINS:
                    against = f"{self.tag}-b (A-vs-A)"
                else:
                    ref = find_reference(runs, n, key, need_control=(n == "M2"))
                    against = (
                        "NEW (no earlier nightly on another stack)"
                        if ref is None
                        else ref["tag"]
                    )
                    if ref is not None and n == "M2":
                        against += f" --envelope {ref['tag']}-b"
                print(
                    f"    {n:4s}: record {' + '.join(tags)} "
                    f"(limit {LIMIT_MIN[n]} min); compare vs {against}"
                )
        print(f"\n  outputs: {self.home / self.tag}/{{SUMMARY.md,run.json,logs/}}")
        print(f"           {self.home / 'history.log'}")
        return 0


_SITECUSTOMIZE = """\
# Written by jaccpot bench/nightly_pins.py on every run -- edit that file, not this.
# The venv's editable finders point at other worktrees; aim them at the nightly's.
import os

for _name, _var in (("jaccpot", "NIGHTLY_JACCPOT"), ("yggdrax", "NIGHTLY_YGGDRAX")):
    _root = os.environ.get(_var)
    if not _root:
        continue
    try:
        _finder = __import__(f"__editable___{_name}_0_0_1_finder")
        _finder.MAPPING[_name] = os.path.join(_root, _name)
    except ImportError:
        pass

# Let the driver's watchdog stack-dump this process (ptrace_scope is 1 here):
# PR_SET_PTRACER, PR_SET_PTRACER_ANY.
if os.environ.get("NIGHTLY_PTRACER_ANY") == "1":
    try:
        import ctypes

        _u = ctypes.c_ulong
        ctypes.CDLL(None).prctl(0x59616D61, _u(2**64 - 1), _u(0), _u(0), _u(0))
    except Exception:
        pass
"""

_RESOLVE = """\
import importlib, json, os, sys
sys.path.insert(0, os.environ["BENCH_DIR"])
out = {}
for name in ("jaccpot", "yggdrax", "nornax", "odisseo", "codes.compare_force",
             "common.ic"):
    out[name] = os.path.realpath(importlib.import_module(name).__file__)
print("NIGHTLY_RESOLVE " + json.dumps(out))
"""

_PROBE = (
    "import jax; d = jax.devices(); "
    "print('NIGHTLY_PROBE', jax.default_backend(), len(d), "
    "d[0].device_kind.replace(' ', '_'))"
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--pins", default="all", help="items or groups (S, M, L, DET, gate, all)"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print the plan; change nothing"
    )
    ap.add_argument(
        "--force", action="store_true", help="rerun items that passed on this stack"
    )
    ap.add_argument("--no-fetch", action="store_true", help="use the local origin refs")
    ap.add_argument("--l-ref", default=L_REF_DEFAULT)
    ap.add_argument("--l-envelope", default=L_ENVELOPE_DEFAULT)
    ap.add_argument("--total-cap-h", type=float, default=10.0)
    ap.add_argument("--free-wait-min", type=float, default=90.0)
    ap.add_argument("--compile-grace-min", type=float, default=20.0)
    ap.add_argument("--home", default=HOME_DEFAULT)
    ap.add_argument("--worktrees", default=WORKTREES_DEFAULT)
    ap.add_argument("--pins-root", default=PINS_ROOT_DEFAULT)
    ap.add_argument("--python", default=VENV_PYTHON_DEFAULT)
    ap.add_argument("--overlay", default=OVERLAY_DEFAULT)
    ap.add_argument("--autocvd", default=AUTOCVD_DEFAULT)
    args = ap.parse_args(argv)
    if not 0 < args.total_cap_h <= 11.0:
        ap.error("--total-cap-h must be in (0, 11]: nightly_pins.sh stops at 12 h")
    return args


if __name__ == "__main__":
    raise SystemExit(Nightly(parse_args(sys.argv[1:])).main())

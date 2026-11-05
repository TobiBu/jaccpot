"""Localhost multi-process harness for the M2 gates (``jax.distributed``).

:func:`run_both` runs one scenario twice, each in fresh Python processes:

* ``sp``: ONE process with ``n`` devices, the single-process runner;
* ``mp``: ``n`` processes with one device each, joined by
  ``jaccpot.distributed.multiprocess.initialize`` on a free localhost port.

``platform="cpu"`` (the default, the CI gate): the devices are forced CPU devices
(``--xla_force_host_platform_device_count``), the collectives gloo's.
``platform="gpu"``: the first ``n`` cards of ``CUDA_VISIBLE_DEVICES`` (which the
caller selects, e.g. with autocvd: the harness never picks card numbers), NCCL
collectives. The single process drives all ``n``; process ``r`` of the
multi-process side owns the ``r``-th, by ``JAX_LOCAL_DEVICE_IDS=r`` (jax's
``local_device_ids``; every process still sees all ``n`` cards, as a SLURM task
without per-task GPU binding does) or, with ``gpu_binding="visible"``, by its own
``CUDA_VISIBLE_DEVICES`` (one card, as ``--gpus-per-task=1`` with binding gives).
Both sides use the same cards in the same order, so the comparison is on the same
hardware.

A scenario is a module-level function ``scenario(n) -> dict`` in an importable
module; process 0 of each side saves its arrays (``.npz``) and its JSON-able
values (``.json``) and the caller compares them. Every child runs in its own
session with a timeout; on a failure or a timeout every child's process group is
killed (no orphans) and the logs' tails are in the error.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

#: what the children run: initialise (mp), import the scenario, save
_CHILD = r"""
import importlib, json, os, sys
import numpy as np
try:  # let a stack dump (py-spy, gdb) attach on a timeout: ptrace_scope 1 hosts
    import ctypes
    ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)
except Exception:
    pass
mode, module, name, n, out = sys.argv[1:6]
n = int(n)
if mode == "mp":
    from jaccpot.distributed import multiprocess as _mp
    _mp.initialize(**(_mp.env_coordinator() or {}), initialization_timeout=120)
import jax
jax.config.update("jax_enable_x64", True)
result = getattr(importlib.import_module(module), name)(n)
if jax.process_index() == 0:
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez(os.path.join(out, mode + ".npz"), **arrays)
    rest = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    with open(os.path.join(out, mode + ".json"), "w") as fh:
        json.dump(rest, fh, default=str, sort_keys=True)
if mode == "mp":
    jax.distributed.shutdown()
"""


def free_port() -> int:
    """A free localhost TCP port (bound and released).

    Returns
    -------
    int
        The port.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env(extra: dict, platform: str = "cpu") -> dict:
    """The children's environment: this one, the platform, x64, the test's ``sys.path``.

    Parameters
    ----------
    extra : dict
        Variables to add.
    platform : str
        ``"cpu"`` or ``"gpu"``.

    Returns
    -------
    dict
        The environment.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("JACCPOT_MP_", "SLURM_")) or name in (
            "JAX_LOCAL_DEVICE_IDS",
            "JAX_COORDINATOR_PORT",
        ):
            del env[name]
    env.update(
        JAX_PLATFORMS="cpu" if platform == "cpu" else "cuda",
        JAX_ENABLE_X64="1",
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    if platform == "gpu":
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.update(extra)
    return env


def visible_cards(n: int) -> list:
    """The first ``n`` entries of ``CUDA_VISIBLE_DEVICES`` (the caller's selection).

    Parameters
    ----------
    n : int
        Cards wanted.

    Returns
    -------
    list
        Their ``CUDA_VISIBLE_DEVICES`` entries (strings).

    Raises
    ------
    RuntimeError
        If ``CUDA_VISIBLE_DEVICES`` lists fewer than ``n`` cards.
    """
    cards = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c]
    if len(cards) < n:
        raise RuntimeError(
            f"the GPU harness needs CUDA_VISIBLE_DEVICES to list {n} cards (select "
            f"them with autocvd); got {os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    return cards[:n]


def _dump_stacks(procs: list, logs: list) -> None:
    """Append each live child's stacks to its log (``py-spy dump --native``).

    Only when ``JACCPOT_MP_PYSPY`` names a py-spy binary (or ``py-spy`` is on
    ``PATH``); the children opt in to being traced (``PR_SET_PTRACER``). Read the
    dump before calling a timeout a hang: a fused-lane compile runs for minutes
    without output (frames ``Compile``, ``backend_compile``, ``Autotun``).

    Parameters
    ----------
    procs : list
        The children.
    logs : list
        Their log paths.
    """
    import shutil

    pyspy = os.environ.get("JACCPOT_MP_PYSPY") or shutil.which("py-spy")
    if not pyspy:
        return
    for proc, log in zip(procs, logs):
        if proc.poll() is not None:
            continue
        try:
            dump = subprocess.run(
                [pyspy, "dump", "--native", "--pid", str(proc.pid)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            text = dump.stdout + dump.stderr
        except Exception as exc:  # diagnostics only
            text = f"py-spy failed: {exc!r}"
        with open(log, "a") as fh:
            fh.write("\n===== py-spy dump at the timeout =====\n" + text)


def _kill(procs: list) -> None:
    """Kill every child's process group.

    Parameters
    ----------
    procs : list
        ``subprocess.Popen`` objects started with ``start_new_session=True``.
    """
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for proc in procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def _wait(procs: list, logs: list, timeout: float, what: str) -> None:
    """Wait for every child; on a failure or the deadline kill all and raise.

    Parameters
    ----------
    procs : list
        The children.
    logs : list
        Their log paths.
    timeout : float
        Seconds for all of them.
    what : str
        For the message.

    Raises
    ------
    RuntimeError
        If a child failed or the deadline passed.
    """
    deadline = time.monotonic() + timeout
    try:
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                failed = "a child failed"
                break
            if all(c == 0 for c in codes):
                return
            if time.monotonic() > deadline:
                failed = f"timeout after {timeout:.0f} s"
                _dump_stacks(procs, logs)
                break
            time.sleep(0.5)
    finally:
        if any(p.poll() is None for p in procs):
            _kill(procs)
    tails = []
    for path in logs:
        with open(path, errors="replace") as fh:
            tails.append(f"--- {os.path.basename(path)}\n{fh.read()[-4000:]}")
    raise RuntimeError(f"{what}: {failed}\n" + "\n".join(tails))


def run_side(
    mode: str,
    module: str,
    name: str,
    n: int,
    out: str,
    *,
    timeout: float = 3600.0,
    env: Any = None,
    platform: str = "cpu",
    gpu_binding: str = "local_device_ids",
    launcher: str = "coordinator",
) -> None:
    """Run ``module.name(n)`` as one side, ``"sp"`` or ``"mp"``.

    Parameters
    ----------
    mode : str
        ``"sp"`` (one process, ``n`` forced devices) or ``"mp"`` (``n`` processes).
    module : str
        The scenario's module (importable with the test's ``sys.path``).
    name : str
        The scenario function.
    n : int
        Devices.
    out : str
        Directory for the outputs and logs.
    timeout : float
        Seconds.
    env : Any
        Extra environment variables, or ``None``.
    platform : str
        ``"cpu"`` (forced CPU devices, gloo) or ``"gpu"`` (the first ``n`` cards of
        ``CUDA_VISIBLE_DEVICES``, NCCL; ``XLA_FLAGS`` passed through).
    gpu_binding : str
        GPU only: how process ``r`` owns card ``r`` -- ``"local_device_ids"``
        (``JAX_LOCAL_DEVICE_IDS=r``, every card visible) or ``"visible"`` (its own
        one-card ``CUDA_VISIBLE_DEVICES``).
    launcher : str
        mp only: ``"coordinator"`` (``JACCPOT_MP_*``: an explicit coordinator) or
        ``"slurm"`` (a SLURM step's variables -- ``SLURM_JOB_ID``,
        ``SLURM_STEP_NODELIST=localhost``, ``SLURM_NTASKS``, ``SLURM_PROCID``,
        ``SLURM_LOCALID``, ``JAX_COORDINATOR_PORT`` -- and ``initialize()`` with no
        arguments: jax's SLURM detection, as on a cluster; on a GPU no
        ``JAX_LOCAL_DEVICE_IDS``: jax or ``multiprocess.slurm_bound_device_ids``
        name the card).

    Raises
    ------
    ValueError
        On an unknown ``platform``, ``gpu_binding`` or ``launcher``.
    """
    if launcher not in ("coordinator", "slurm"):
        raise ValueError(f"unknown launcher {launcher!r}")
    if platform not in ("cpu", "gpu"):
        raise ValueError(f"platform must be 'cpu' or 'gpu'; got {platform!r}")
    if gpu_binding not in ("local_device_ids", "visible"):
        raise ValueError(f"unknown gpu_binding {gpu_binding!r}")
    os.makedirs(out, exist_ok=True)
    extra = dict(env or {})
    cards = visible_cards(n) if platform == "gpu" else []
    xla_flags = extra.pop("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
    argv = [sys.executable, "-c", _CHILD, mode, module, name, str(n), out]
    procs, logs = [], []
    try:
        if mode == "sp":
            if platform == "cpu":
                flags = f"--xla_force_host_platform_device_count={n}"
                side = dict(extra, XLA_FLAGS=flags)
            else:
                side = dict(
                    extra, XLA_FLAGS=xla_flags, CUDA_VISIBLE_DEVICES=",".join(cards)
                )
            log = os.path.join(out, "sp.log")
            logs.append(log)
            with open(log, "w") as fh:
                procs.append(
                    subprocess.Popen(
                        argv,
                        env=_env(side, platform),
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        else:
            port = free_port()
            for rank in range(n):
                log = os.path.join(out, f"mp{rank}.log")
                logs.append(log)
                if launcher == "slurm":
                    child = dict(
                        extra,
                        XLA_FLAGS="--xla_force_host_platform_device_count=1",
                        SLURM_JOB_ID="4242",
                        SLURM_STEP_NODELIST="localhost",
                        SLURM_NTASKS=str(n),
                        SLURM_PROCID=str(rank),
                        SLURM_LOCALID=str(rank),
                        JAX_COORDINATOR_PORT=str(port),
                    )
                else:
                    child = dict(
                        extra,
                        XLA_FLAGS="--xla_force_host_platform_device_count=1",
                        JACCPOT_MP_COORDINATOR=f"127.0.0.1:{port}",
                        JACCPOT_MP_NUM_PROCESSES=str(n),
                        JACCPOT_MP_PROCESS_ID=str(rank),
                    )
                if platform == "gpu":
                    child["XLA_FLAGS"] = xla_flags
                    if gpu_binding == "visible":
                        child["CUDA_VISIBLE_DEVICES"] = cards[rank]
                    else:
                        child["CUDA_VISIBLE_DEVICES"] = ",".join(cards)
                        if launcher != "slurm":
                            child["JAX_LOCAL_DEVICE_IDS"] = str(rank)
                with open(log, "w") as fh:
                    procs.append(
                        subprocess.Popen(
                            argv,
                            env=_env(child, platform),
                            stdout=fh,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    )
        _wait(procs, logs, timeout, f"{module}.{name} ({mode}, n={n})")
    except BaseException:
        _kill(procs)
        raise


def run_both(
    module: str,
    name: str,
    n: int,
    out: str,
    *,
    timeout: float = 3600.0,
    env: Any = None,
    platform: str = "cpu",
    gpu_binding: str = "local_device_ids",
    launcher: str = "coordinator",
    modes: tuple = ("sp", "mp"),
) -> tuple[dict, dict, dict, dict]:
    """``sp`` then ``mp`` for one scenario; their outputs.

    Parameters
    ----------
    module : str
        The scenario's module.
    name : str
        The scenario function.
    n : int
        Devices (= processes on the mp side).
    out : str
        Output directory.
    timeout : float
        Seconds per side.
    env : Any
        Extra environment variables.
    platform : str
        ``"cpu"`` or ``"gpu"`` (:func:`run_side`).
    gpu_binding : str
        GPU only (:func:`run_side`).
    launcher : str
        How the mp side starts (:func:`run_side`).
    modes : tuple
        The sides to run (an A-vs-A control repeats one side into another
        directory); outputs of a side not run are read from ``out``.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(sp arrays, mp arrays, sp json, mp json)``.
    """
    for mode in modes:
        run_side(
            mode,
            module,
            name,
            n,
            out,
            timeout=timeout,
            env=env,
            platform=platform,
            gpu_binding=gpu_binding,
            launcher=launcher,
        )
    (sp, sp_json), (mp, mp_json) = load_side(out, "sp"), load_side(out, "mp")
    return sp, mp, sp_json, mp_json


def load_side(out: str, mode: str) -> tuple[dict, dict]:
    """One side's saved outputs: ``(arrays, json values)``.

    Parameters
    ----------
    out : str
        The side's output directory.
    mode : str
        ``"sp"`` or ``"mp"``.

    Returns
    -------
    tuple[dict, dict]
        The arrays and the JSON-able values process 0 saved.
    """
    with np.load(os.path.join(out, f"{mode}.npz")) as data:
        arrays = {k: data[k] for k in data.files}
    with open(os.path.join(out, f"{mode}.json")) as fh:
        values = json.load(fh)
    return arrays, values


def differing(a: dict, b: dict) -> dict:
    """Per array name, how many elements differ bit for bit (``-1``: shape/key).

    Parameters
    ----------
    a : dict
        Arrays.
    b : dict
        Arrays.

    Returns
    -------
    dict
        ``name -> count`` for every name with a difference.
    """
    out = {}
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b or a[key].shape != b[key].shape:
            out[key] = -1
            continue
        x, y = np.ascontiguousarray(a[key]), np.ascontiguousarray(b[key])
        if x.dtype != y.dtype:
            out[key] = -1
            continue
        width = x.dtype.itemsize
        xb = x.reshape(-1).view(np.uint8).reshape(-1, width)
        yb = y.reshape(-1).view(np.uint8).reshape(-1, width)
        count = int(np.any(xb != yb, axis=1).sum()) if x.size else 0
        if count:
            out[key] = count
    return out

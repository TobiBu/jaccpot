"""Localhost multi-process harness for the M2 gates (``jax.distributed`` on CPU).

:func:`run_both` runs one scenario twice, each in fresh Python processes:

* ``sp``: ONE process with ``n`` forced CPU devices
  (``--xla_force_host_platform_device_count=n``), the single-process runner;
* ``mp``: ``n`` processes with one CPU device each, joined by
  ``jaccpot.distributed.multiprocess.initialize`` on a free localhost port (gloo
  collectives).

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
mode, module, name, n, out = sys.argv[1:6]
n = int(n)
if mode == "mp":
    from jaccpot.distributed import multiprocess as _mp
    _mp.initialize(**_mp.env_coordinator(), initialization_timeout=120)
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


def _env(extra: dict) -> dict:
    """The children's environment: this one, CPU, x64, the test's ``sys.path``.

    Parameters
    ----------
    extra : dict
        Variables to add.

    Returns
    -------
    dict
        The environment.
    """
    env = dict(os.environ)
    for name in list(env):
        if name.startswith("JACCPOT_MP_"):
            del env[name]
    env.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="1",
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    env.update(extra)
    return env


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
    """
    os.makedirs(out, exist_ok=True)
    extra = dict(env or {})
    argv = [sys.executable, "-c", _CHILD, mode, module, name, str(n), out]
    procs, logs = [], []
    try:
        if mode == "sp":
            flags = f"--xla_force_host_platform_device_count={n}"
            log = os.path.join(out, "sp.log")
            logs.append(log)
            with open(log, "w") as fh:
                procs.append(
                    subprocess.Popen(
                        argv,
                        env=_env(dict(extra, XLA_FLAGS=flags)),
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
                child = dict(
                    extra,
                    XLA_FLAGS="--xla_force_host_platform_device_count=1",
                    JACCPOT_MP_COORDINATOR=f"127.0.0.1:{port}",
                    JACCPOT_MP_NUM_PROCESSES=str(n),
                    JACCPOT_MP_PROCESS_ID=str(rank),
                )
                with open(log, "w") as fh:
                    procs.append(
                        subprocess.Popen(
                            argv,
                            env=_env(child),
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

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(sp arrays, mp arrays, sp json, mp json)``.
    """
    for mode in ("sp", "mp"):
        run_side(mode, module, name, n, out, timeout=timeout, env=env)
    arrays, values = [], []
    for mode in ("sp", "mp"):
        with np.load(os.path.join(out, f"{mode}.npz")) as data:
            arrays.append({k: data[k] for k in data.files})
        with open(os.path.join(out, f"{mode}.json")) as fh:
            values.append(json.load(fh))
    return arrays[0], arrays[1], values[0], values[1]


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

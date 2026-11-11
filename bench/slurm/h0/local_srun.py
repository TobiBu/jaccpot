"""A local stand-in for ``srun``: a job step's tasks as processes on THIS host.

The HoreKa bundle's multi-process jobs (``m1_*``, ``m2g_*``, ``mesc_*`` and the scaling
matrix's ``mp`` tier) launch their steps with ``srun``. On a machine without SLURM (the
CPU smoke, ``SMOKE=1``; the local GPU preflight, ``H0_LOCAL_SRUN=1``) ``env.sh`` puts
``smoke_bin/srun`` first on ``PATH``, which runs this script. It parses the ``srun``
options the jobs use, then starts ``--ntasks`` copies of the command, each with a SLURM
step's variables (``SLURM_JOB_ID``, ``SLURM_STEP_NODELIST=localhost``, ``SLURM_NTASKS``,
``SLURM_PROCID``, ``SLURM_LOCALID``, ``JAX_COORDINATOR_PORT``), as
``tests/distributed/_multiprocess_harness.py``'s ``launcher="slurm"`` does: jax's own
SLURM detection then joins the processes (gloo on the CPU, NCCL on GPUs).

* Every task runs on this host (``SLURM_LOCALID`` = the task's rank, so a two-"node"
  step's tasks take distinct cards).
* On the CPU (``JAX_PLATFORMS=cpu``) a one-task step gets as many forced host devices
  as its ``--gres=gpu:K`` asks for (the one-process-drives-K-cards arm), a multi-task
  step one device per task.
* On GPUs the tasks inherit ``CUDA_VISIBLE_DEVICES`` (every task sees every card and
  jax takes card ``SLURM_LOCALID``); with ``--gpus-per-task=1`` and ``--gpu-bind``
  each task sees only its own card (the per-task binding SLURM applies).
* When one task fails, the others are killed after ``LOCAL_SRUN_GRACE`` seconds
  (default 60: a peer blocked in a collective never returns); the step's exit code is
  the first non-zero one. ``LOCAL_SRUN_TIMEOUT`` (default 14400 s) bounds the step.

    local_srun.py --ntasks=2 --ntasks-per-node=2 --gres=gpu:2 python bench/x.py ...
"""

from __future__ import annotations

import os
import re
import signal
import socket
import subprocess
import sys
import time

#: srun options that take a value (``--opt=value`` or ``--opt value``); others are flags
_VALUED = {
    "-n",
    "--ntasks",
    "-N",
    "--nodes",
    "--ntasks-per-node",
    "-c",
    "--cpus-per-task",
    "--gres",
    "--gpus-per-task",
    "--gpu-bind",
    "--mem",
    "--cpu-bind",
    "-J",
    "--job-name",
    "-o",
    "--output",
    "-t",
    "--time",
}


def _parse(argv: list[str]) -> tuple[dict, list[str]]:
    """``(options, command)``: the srun options before the first non-option word."""
    opts: dict = {}
    i = 0
    while i < len(argv):
        word = argv[i]
        if not word.startswith("-"):
            break
        if word == "--":
            i += 1
            break
        if "=" in word:
            key, value = word.split("=", 1)
        elif word in _VALUED:
            key, value = word, argv[i + 1]
            i += 1
        else:
            key, value = word, True
        opts[key] = value
        i += 1
    if i >= len(argv):
        raise SystemExit("local_srun: no command")
    return opts, argv[i:]


def _int(opts: dict, *keys: str, default: int) -> int:
    for key in keys:
        if key in opts:
            return int(opts[key])
    return default


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _with_device_count(flags: str, count: int) -> str:
    flags = re.sub(r"--xla_force_host_platform_device_count=\d+", "", flags)
    return f"{flags} --xla_force_host_platform_device_count={count}".strip()


def main(argv: list[str]) -> int:
    opts, cmd = _parse(argv)
    ntasks = _int(opts, "-n", "--ntasks", default=1)
    gres = str(opts.get("--gres", ""))
    gpus = int(gres.rsplit(":", 1)[1]) if gres.startswith("gpu") and ":" in gres else 1
    bound = "--gpus-per-task" in opts and "--gpu-bind" in opts
    on_cpu = os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"
    visible = [c for c in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if c]
    port = os.environ.get("JAX_COORDINATOR_PORT") or str(_free_port())
    procs = []
    for rank in range(ntasks):
        env = dict(os.environ)
        if ntasks > 1:
            env.update(
                SLURM_JOB_ID=env.get("SLURM_JOB_ID", "4242"),
                SLURM_STEP_ID="0",
                SLURM_STEP_NODELIST="localhost",
                SLURM_NTASKS=str(ntasks),
                SLURM_PROCID=str(rank),
                SLURM_LOCALID=str(rank),
                SLURM_NODEID="0",
                JAX_COORDINATOR_PORT=port,
            )
        else:
            for key in ("SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID"):
                env.pop(key, None)
        if on_cpu:
            count = 1 if ntasks > 1 else gpus
            env["XLA_FLAGS"] = _with_device_count(env.get("XLA_FLAGS", ""), count)
        elif bound and ntasks > 1 and visible:
            env["CUDA_VISIBLE_DEVICES"] = visible[rank % len(visible)]
        procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))
    grace = float(os.environ.get("LOCAL_SRUN_GRACE", "60"))
    deadline = time.time() + float(os.environ.get("LOCAL_SRUN_TIMEOUT", "14400"))
    rc = 0
    failed_at = None
    while True:
        codes = [p.poll() for p in procs]
        bad = [c for c in codes if c not in (None, 0)]
        if bad and not rc:
            rc = bad[0]
            failed_at = time.time()
        if all(c is not None for c in codes):
            break
        now = time.time()
        if (failed_at is not None and now - failed_at > grace) or now > deadline:
            if not rc:
                rc = 124
            for p in procs:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
            print(
                f"local_srun: killed the step's remaining tasks (rc {rc})", flush=True
            )
            break
        time.sleep(0.5)
    for p in procs:
        p.wait()
    return int(rc)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Figures 08 and 09, measured: the library's scaling matrix with the paper's tiers.

There is no paper-specific multi-GPU harness. Strong and weak scaling are the
``strong`` and ``weak`` tiers of ``bench/multigpu_scaling_matrix.py`` (each row
``bench/multigpu_segment_gate.py`` in a fresh process on the fused mesh lane,
``preset="galaxy"``):

=========  ==================================================================
strong     the 25M disc + bulge and Plummer 8e6, each MAC of ``--macs``, on
           1 / 2 / 4 cards (one card: the one-device mesh ``_1c`` and the
           single-card lane ``_solo``); migrating
weak       Plummer 4e6 per card on 1 / 2 / 4 cards, the first MAC; migrating
=========  ==================================================================

This driver only adds what a run off the HoreKa bundle lacks, then calls the
matrix unchanged:

* autocvd claims ``--max-cards`` free cards in ONE selection
  (``runmeta.select_gpu(num_gpus=k)``) before any process imports JAX; the matrix
  then runs each row on the first ``k`` of them (skipped when
  ``CUDA_VISIBLE_DEVICES`` is already set, e.g. by SLURM, and on the CPU);
* ``provenance.json`` in the out-dir (``bench/slurm/h0/provenance.py``: card names,
  ``nvidia-smi topo -m``, packages, git), which ``from_matrix.py`` reads the device
  type and interconnect from;
* ``BENCH_DIR`` defaults to the vendored Odisseo harness
  (``bench/slurm/h0/harness/benchmark_multigpu``), where the Plummer IC lives.

The ragged-exchange XLA flags and the fusion autotuner switch are set per row by
the matrix itself. A row whose JSON exists is skipped (``--redo`` reruns it), so an
interrupted run continues. ``--convert`` runs ``bench/multigpu/from_matrix.py``
afterwards (``bench/results/multigpu/{strong,weak}_scaling.json``).

On HoreKa nothing here is needed: the bundle's job ``scaling_a`` already runs the
strong and weak tiers (``bench/slurm/h0/jobs/scaling_a.sh``, out-dir
``$H0_RUN/scaling_a/matrix``, provenance in ``$H0_RUN/scaling_a``); convert it with
``python -m bench.multigpu.from_matrix --matrix-dir $H0_RUN/scaling_a/matrix``.

Usage
-----
::

    python -m bench.multigpu.run_matrix --out-dir OUT --max-cards 4 \\
        --disc-ic /path/disk_bulge_25m.npz --convert

CPU smoke (forced host devices through the bundle's CPU route; every point
converts as invalid, by design)::

    JAX_PLATFORMS=cpu python -m bench.multigpu.run_matrix --out-dir /tmp/m \\
        --max-cards 2 --macs dehnen --smoke-n 2048 --steps 2 --segments 5 \\
        --rows '^(strong_P8e6_geo_(1c|solo|2c)|weak_P4e6pc_(1c|2c))$' --convert \\
        --strong-out /tmp/strong.json --weak-out /tmp/weak.json
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.multigpu import _mesh_common as mc  # noqa: E402
from examples.jaccpot_paper.common import runmeta  # noqa: E402

MATRIX = REPO_ROOT / "bench" / "multigpu_scaling_matrix.py"
PROVENANCE = REPO_ROOT / "bench" / "slurm" / "h0" / "provenance.py"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Unknown flags are an error; the matrix's own --help lists its flags.",
    )
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--tiers", default="strong,weak", help="matrix tiers (paper: these)")
    p.add_argument(
        "--macs",
        default=None,
        help="matrix --macs (default: the matrix's, dehnen_error,dehnen)",
    )
    p.add_argument("--max-cards", type=int, default=4)
    p.add_argument("--steps", type=int, default=None, help="matrix --steps")
    p.add_argument("--segments", type=int, default=None, help="matrix --segments")
    p.add_argument("--disc-ic", default=None, help="the 25M disc + bulge .npz")
    p.add_argument("--smoke-n", type=int, default=None, help="matrix --smoke-n")
    p.add_argument("--rows", default=None, help="matrix --rows (a tag regex)")
    p.add_argument("--timeout", type=float, default=None, help="seconds per row")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true")
    p.add_argument(
        "--gpu-select",
        choices=("least-used", "first", "none"),
        default="least-used",
        help="autocvd selection of --max-cards cards (skipped when "
        "CUDA_VISIBLE_DEVICES is set, and on the CPU)",
    )
    p.add_argument(
        "--convert",
        action="store_true",
        help="run bench/multigpu/from_matrix.py on the out-dir afterwards",
    )
    p.add_argument("--strong-out", default=None, help="--convert: from_matrix's")
    p.add_argument("--weak-out", default=None, help="--convert: from_matrix's")
    return p.parse_args()


def main() -> int:
    """Select cards, record provenance, run the matrix, optionally convert.

    Returns
    -------
    int
        The matrix's return code, else the converter's.
    """
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    disc = args.disc_ic or os.environ.get("DISC_IC")
    if (
        "strong" in args.tiers.split(",")
        and not args.rows
        and not (disc and os.path.exists(disc))
        and not args.dry_run
    ):
        raise SystemExit(
            "the strong tier includes the 25M disc + bulge rows: pass --disc-ic (or "
            "DISC_IC), or restrict the rows, e.g. --rows '^(strong_P|weak_)'"
        )
    if not mc.on_cpu() and args.gpu_select != "none" and not args.dry_run:
        # before any process of this run imports jax
        chosen = runmeta.select_gpu(args.gpu_select, num_gpus=int(args.max_cards))
        print(f"[run_matrix] cards: CUDA_VISIBLE_DEVICES={chosen}", flush=True)
    env = dict(os.environ)
    env.setdefault("BENCH_DIR", str(mc.VENDORED_BENCH_DIR))
    if disc:
        env.setdefault("DISC_IC", disc)
    if mc.on_cpu():
        # the bundle's CPU route (H0_SMOKE_CPU): the matrix forces the host devices
        env.setdefault("H0_SMOKE_CPU", "1")
        path = [p for p in env.get("PYTHONPATH", "").split(":") if p]
        for extra in (str(REPO_ROOT), str(mc.SMOKE_SITE)):
            if extra not in path:
                path.insert(0, extra)
        env["PYTHONPATH"] = ":".join(path)
    prov = args.out_dir / "provenance.json"
    if not prov.exists() and not args.dry_run:
        subprocess.run(
            [sys.executable, str(PROVENANCE), "--out", str(prov), "--no-ic-hash"],
            env=env,
            check=False,
        )
    cmd = [sys.executable, str(MATRIX), "--out-dir", str(args.out_dir)]
    cmd += ["--tiers", args.tiers, "--max-cards", str(args.max_cards)]
    for flag, val in (
        ("--macs", args.macs),
        ("--steps", args.steps),
        ("--segments", args.segments),
        ("--disc-ic", args.disc_ic),
        ("--smoke-n", args.smoke_n),
        ("--rows", args.rows),
        ("--timeout", args.timeout),
    ):
        if val is not None:
            cmd += [flag, str(val)]
    cmd += [f for f, on in (("--dry-run", args.dry_run), ("--redo", args.redo)) if on]
    print("[run_matrix]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT)).returncode
    if rc != 0 or not args.convert or args.dry_run:
        return rc
    conv = [sys.executable, "-m", "bench.multigpu.from_matrix"]
    conv += ["--matrix-dir", str(args.out_dir), "--tiers", args.tiers]
    if args.strong_out:
        conv += ["--strong-out", args.strong_out]
    if args.weak_out:
        conv += ["--weak-out", args.weak_out]
    return subprocess.run(conv, env=env, cwd=str(REPO_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())

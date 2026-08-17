"""Execute the figure notebooks in place, which is what writes the PDFs.

Executing rather than just rendering matters: a notebook whose stored output was
produced by an older artifact is a figure nobody can trust. This runs each one
against whatever is currently in ``results/`` and saves the executed copy, so the
committed notebook and its PDF always agree.

    python examples/jaccpot_paper/run_notebooks.py                # all
    python examples/jaccpot_paper/run_notebooks.py fig_01 fig_04  # a subset

A notebook whose artifact is missing is reported and skipped rather than failing
the whole run, so a partial set of benchmark results still regenerates the
figures it can.
"""

from __future__ import annotations

import pathlib
import sys
import traceback

import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def run_one(path: pathlib.Path) -> tuple[bool, str]:
    nb = nbf.read(str(path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        # Run with the repo root as cwd so `results/` resolves the same way it
        # does for the bench scripts.
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        message = str(exc)
        nbf.write(nb, str(path))
        if "FileNotFoundError" in message or "does not exist" in message:
            return False, "artifact missing - run the bench script first"
        return False, message.strip().splitlines()[-1][:200]
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"[:200]
    nbf.write(nb, str(path))
    return True, "ok"


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    notebooks = sorted(HERE.glob("fig_*.ipynb"))
    if wanted:
        notebooks = [p for p in notebooks if any(w in p.stem for w in wanted)]
    if not notebooks:
        print("no matching notebooks")
        return 1

    failures = []
    for path in notebooks:
        ok, message = run_one(path)
        status = "OK  " if ok else "FAIL"
        print(f"[{status}] {path.name}: {message}", flush=True)
        if not ok:
            failures.append((path.name, message))

    print()
    pdfs = sorted((REPO_ROOT / "bench" / "results" / "figures").glob("*.pdf"))
    print(f"{len(pdfs)} PDF(s) in bench/results/figures/:")
    for pdf in pdfs:
        print(f"  {pdf.name}  ({pdf.stat().st_size // 1024} KiB)")

    if failures:
        print(f"\n{len(failures)} notebook(s) failed:")
        for name, message in failures:
            print(f"  {name}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

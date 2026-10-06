"""Capture the fused lane's near-field and walk inputs, for timing the kernels alone.

usage: python bench/nearfield_capture.py OUT.npz -- <fused_memory_budget.py arguments>

Runs ``bench/fused_memory_budget.py`` with the given arguments (use ``--no-scan
--eval-repeats 1 --no-analysis``) and saves the arguments of the FIRST call of
``nearfield_leafpair_csr_sorted_direct_pallas`` -- sorted positions and masses, leaf
ranges, the neighbour CSR -- plus its static options (``OUT.json``). The call runs
inside the jitted eval, so the arrays come out through a ``jax.debug.callback``.
``bench/nearfield_kernel_tune.py`` times kernel variants on the file.

The first ``mutual_walk_pallas`` call's inputs (children, MAC centres and radii, node
activity, root) and its static sizes go to ``OUT_walk.npz`` / ``OUT_walk.json``, for
``bench/walk_tune.py``.
"""

from __future__ import annotations

import json
import os
import runpy
import sys

out, rest = sys.argv[1], sys.argv[2:]
if rest and rest[0] == "--":
    rest = rest[1:]

import jax  # noqa: E402
import numpy as np  # noqa: E402

import jaccpot.pallas.nearfield_leafpair_csr as _nf  # noqa: E402

_orig = _nf.nearfield_leafpair_csr_sorted_direct_pallas
_done = {"saved": False, "traced": False}
_NAMES = (
    "positions",
    "masses",
    "leaf_start",
    "leaf_count",
    "neighbors",
    "offsets",
    "counts",
)


def _save(*arrays) -> None:
    if _done["saved"]:
        return
    _done["saved"] = True
    data = {k: np.asarray(v) for k, v in zip(_NAMES, arrays)}
    np.savez(out, **data)
    print(
        f"[capture] saved {out}: "
        + ", ".join(f"{k} {v.shape} {v.dtype}" for k, v in data.items()),
        flush=True,
    )


def _wrapped(*args, **kw):
    if not _done["traced"]:
        _done["traced"] = True
        static = {
            k: (v if isinstance(v, (int, float, bool, str, type(None))) else None)
            for k, v in kw.items()
        }
        static["softening_sq"] = None
        static["G"] = None
        arrays = list(args[:7]) + [kw["softening_sq"], kw["G"]]
        with open(os.path.splitext(out)[0] + ".json", "w") as fh:
            json.dump(static, fh, indent=1)

        def _cb(*a):
            _save(*a[:7])
            meta = json.load(open(os.path.splitext(out)[0] + ".json"))
            meta["softening_sq"] = float(np.asarray(a[7]))
            meta["G"] = float(np.asarray(a[8]))
            with open(os.path.splitext(out)[0] + ".json", "w") as fh:
                json.dump(meta, fh, indent=1)

        jax.debug.callback(_cb, *arrays)
    return _orig(*args, **kw)


_nf.nearfield_leafpair_csr_sorted_direct_pallas = _wrapped

import jaccpot.pallas.mutual_walk_pallas as _mwp  # noqa: E402

_walk_orig = _mwp.mutual_walk_pallas
_walk_done = {"traced": False, "saved": False}
_WALK = ("left", "right", "centers", "radii", "root", "node_active")
_walk_out = os.path.splitext(out)[0] + "_walk"


def _walk_save(*arrays) -> None:
    if _walk_done["saved"]:
        return
    _walk_done["saved"] = True
    np.savez(_walk_out + ".npz", **{k: np.asarray(v) for k, v in zip(_WALK, arrays)})
    print(f"[capture] saved {_walk_out}.npz", flush=True)


def _walk_wrapped(left, right, centers, radii, theta, root, **kw):
    if not _walk_done["traced"]:
        _walk_done["traced"] = True
        meta = {k: v for k, v in kw.items() if isinstance(v, (int, float, bool, str))}
        meta["theta"] = float(theta)
        with open(_walk_out + ".json", "w") as fh:
            json.dump(meta, fh, indent=1)
        act = kw.get("node_active")
        jax.debug.callback(
            _walk_save,
            left,
            right,
            centers,
            radii,
            root,
            act if act is not None else jax.numpy.ones(left.shape, bool),
        )
    return _walk_orig(left, right, centers, radii, theta, root, **kw)


_mwp.mutual_walk_pallas = _walk_wrapped

import jaccpot.pallas.csr_place as _csr  # noqa: E402

_csr_orig = _csr.directed_csr_pallas
_csr_state = {"traced": 0}


def _csr_wrapped(nodes_a, nodes_b, count, **kw):
    k = _csr_state["traced"]
    if k < 2:
        _csr_state["traced"] = k + 1
        stem = os.path.splitext(out)[0] + f"_lists{k}"
        meta = {
            key: (v if isinstance(v, (int, float, bool, str)) else str(v))
            for key, v in kw.items()
        }
        with open(stem + ".json", "w") as fh:
            json.dump(meta, fh, indent=1)

        def _cb(a, b, c, stem=stem):
            if os.path.exists(stem + ".npz"):
                return
            np.savez(
                stem + ".npz", a=np.asarray(a), b=np.asarray(b), count=np.asarray(c)
            )
            print(
                f"[capture] saved {stem}.npz (count {int(np.asarray(c))})", flush=True
            )

        jax.debug.callback(_cb, nodes_a, nodes_b, count)
    return _csr_orig(nodes_a, nodes_b, count, **kw)


_csr.directed_csr_pallas = _csr_wrapped

import jaccpot.runtime._mac_geometry as _mg  # noqa: E402

_comr_orig = _mg._com_radii
_comr_state = {"traced": False}
_COMR = (
    "node_ranges",
    "left_child",
    "right_child",
    "parent",
    "positions_sorted",
    "centers",
)


def _comr_wrapped(*args, **kw):
    if not _comr_state["traced"]:
        _comr_state["traced"] = True
        stem = os.path.splitext(out)[0] + "_comr"
        with open(stem + ".json", "w") as fh:
            json.dump(
                {
                    k: v
                    for k, v in kw.items()
                    if isinstance(v, (int, float, bool, str, type(None)))
                },
                fh,
                indent=1,
            )

        def _cb(*a, stem=stem):
            if os.path.exists(stem + ".npz"):
                return
            np.savez(stem + ".npz", **{k: np.asarray(v) for k, v in zip(_COMR, a)})
            print(f"[capture] saved {stem}.npz", flush=True)

        jax.debug.callback(_cb, *args[:6])
    return _comr_orig(*args, **kw)


_mg._com_radii = _comr_wrapped
sys.argv = [os.path.join(os.path.dirname(__file__), "fused_memory_budget.py"), *rest]
runpy.run_path(sys.argv[0], run_name="__main__")

"""``jaccpot.checkpoint`` without orbax: optional, lazily imported, named when missing.

orbax is the ``io`` extra (multi-node plan M3). Importing jaccpot, the production
runner and the checkpoint module must not import it; using a sharded checkpoint
without it is an ``ImportError`` that names the extra. The manifest's set hash
(pure NumPy) is order- and split-independent and sees a changed row or id.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest


def test_importing_jaccpot_does_not_import_orbax():
    code = (
        "import sys, jaccpot, jaccpot.production, jaccpot.checkpoint, "
        "jaccpot.distributed.mesh_runner; "
        "print('orbax' in ' '.join(sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_a_missing_orbax_names_the_extra(monkeypatch, tmp_path):
    from jaccpot import checkpoint

    monkeypatch.setitem(sys.modules, "orbax", None)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", None)
    with pytest.raises(ImportError, match=r"jaccpot\[io\]"):
        checkpoint.write_checkpoint(tmp_path / "ck", {}, {})


def test_a_path_without_a_checkpoint_is_none(tmp_path):
    from jaccpot.checkpoint import is_checkpoint, latest_checkpoint, resolve

    assert not is_checkpoint(tmp_path / "nothing")
    (tmp_path / "file.npz").write_bytes(b"x")
    assert not is_checkpoint(tmp_path / "file.npz")
    assert latest_checkpoint(str(tmp_path / "ck_{step:08d}")) is None
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path / "nothing")


def _fake_checkpoint(path):
    (path / "arrays").mkdir(parents=True)
    (path / "meta").mkdir()
    (path / "meta" / "metadata").write_text("{}")
    (path / "_CHECKPOINT_METADATA").write_text("{}")


def test_latest_checkpoint_takes_the_highest_committed_step(tmp_path):
    from jaccpot.checkpoint import latest_checkpoint, resolve

    for step in (10, 200, 30):
        _fake_checkpoint(tmp_path / f"ck_{step:08d}")
    # a half-written one (no commit) is not a checkpoint
    (tmp_path / f"ck_{999:08d}").mkdir()
    assert latest_checkpoint(str(tmp_path / "ck_{step:08d}")).endswith("ck_00000200")
    # a crash between a replace's renames: the committed .next is read
    _fake_checkpoint(tmp_path / "fixed.next")
    assert resolve(tmp_path / "fixed").endswith("fixed.next")


def test_the_set_hash_is_order_and_split_independent():
    from jaccpot.checkpoint import set_hash

    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 2, 3)).astype(np.float32)
    ids = rng.permutation(50).astype(np.int32)
    ids[[3, 7]] = -1  # padding rows are left out
    whole = set_hash(ids, values)
    order = rng.permutation(50)
    assert set_hash(ids[order], values[order]) == whole
    parts = (set_hash(ids[:20], values[:20]) + set_hash(ids[20:], values[20:])) % (
        1 << 64
    )
    assert parts == whole
    changed = values.copy()
    changed[10, 1, 2] = np.nextafter(changed[10, 1, 2], np.float32(np.inf))
    assert set_hash(ids, changed) != whole
    swapped = ids.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert set_hash(swapped, values) != whole
    # a padding row's contents do not count
    padded = values.copy()
    padded[3] = 7.0
    assert set_hash(ids, padded) == whole

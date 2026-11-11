"""Sharded checkpoints for the multi-GPU mode (orbax; multi-node plan track M3).

A checkpoint is ONE orbax checkpoint directory with two items:

* ``arrays``: global ``jax.Array`` objects as they live on the mesh -- the
  segment runner's layout (``state``, ``masses``, ``ids``, ``count``,
  ``acceleration``, ``force_scale`` and, with escapers, the ``tail_*`` arrays).
  orbax writes each array's shards from the process that holds them (OCDBT: one
  database per process, merged at the commit), so nothing is gathered through
  one host and nothing is copied on the device for the write.
* ``meta``: one JSON document (written by process 0): what the runner records in
  its ``.npz`` restart's ``meta`` (:mod:`jaccpot.production`), ``format``
  (:data:`CHECKPOINT_FORMAT`) and the ``manifest`` (:func:`build_manifest`): every
  array's shape and dtype, the sha256 of every shard's bytes (by its first row) and,
  for arrays with particle ids, an order-independent 64-bit hash over the live rows
  keyed by id (:func:`set_hash`), which a restore onto ANOTHER device count -- whose
  shards are not the written ones -- still checks.

**Atomic commit.** orbax writes into a temporary directory and renames it once
every process finished: a crash mid-write leaves no checkpoint at ``path`` and
every earlier one intact. A ``path`` that already holds a checkpoint is replaced
through ``<path>.next`` (written and committed first), ``path -> <path>.prev``,
``<path>.next -> path`` and removing ``<path>.prev``; :func:`resolve` reads
``<path>.next`` or ``<path>.prev`` if a crash fell between the renames.

**orbax is optional** (the ``jaccpot[io]`` extra): it is imported only when a
checkpoint is written or read, and its absence is an ``ImportError`` that names
the extra. Importing jaccpot and every default path never touches it.

**Reading.** :func:`restore_arrays` restores named arrays onto a sharding (the
run's own: each process reads its own shards; another mesh's: orbax re-shards,
with ``strict=False`` padding at the end); :func:`verify_shards` and
:func:`verify_set_hashes` check what was read against the manifest;
:func:`load_checkpoint` reads a checkpoint on one host by id (analysis, a
single-card continuation).
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointCorruptError",
    "build_manifest",
    "is_checkpoint",
    "latest_checkpoint",
    "load_checkpoint",
    "read_meta",
    "resolve",
    "restore_arrays",
    "set_hash",
    "verify_set_hashes",
    "verify_shards",
    "write_checkpoint",
]

#: a checkpoint's ``meta["format"]``
CHECKPOINT_FORMAT = "jaccpot.production.checkpoint/1"

_MISSING_ORBAX = (
    "sharded checkpoints need orbax-checkpoint: install jaccpot's io extra "
    "(pip install 'jaccpot[io]')"
)


class CheckpointCorruptError(ValueError):
    """A checkpoint's data do not match its manifest (a corrupted or partial shard)."""


def _ocp() -> Any:
    """``orbax.checkpoint``, imported on first use.

    Returns
    -------
    Any
        The module.

    Raises
    ------
    ImportError
        Without orbax; the message names the extra.
    """
    try:
        import orbax.checkpoint as ocp
    except ImportError as exc:  # pragma: no cover - the CI shard installs it
        raise ImportError(_MISSING_ORBAX) from exc
    return ocp


def _checkpointer() -> Any:
    """A synchronous composite orbax checkpointer.

    Returns
    -------
    Any
        ``ocp.Checkpointer(CompositeCheckpointHandler())``.
    """
    ocp = _ocp()
    return ocp.Checkpointer(ocp.CompositeCheckpointHandler())


def _sync(name: str) -> None:
    """A barrier over every process (nothing in one process).

    Parameters
    ----------
    name : str
        The barrier's name (the same on every process).
    """
    import jax

    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices(name)


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------


def _is_committed(path: str) -> bool:
    """Whether ``path`` is a committed checkpoint of this format (a quick test).

    Parameters
    ----------
    path : str
        A directory.

    Returns
    -------
    bool
        It has the ``meta`` and ``arrays`` items and orbax's metadata.
    """
    return (
        os.path.isdir(os.path.join(path, "arrays"))
        and os.path.isfile(os.path.join(path, "meta", "metadata"))
        and os.path.isfile(os.path.join(path, "_CHECKPOINT_METADATA"))
    )


def resolve(path: Any) -> str:
    """The committed checkpoint ``path`` names.

    ``path`` itself; if a crash fell between the renames of a replace
    (:func:`write_checkpoint`), ``<path>.next`` (the newer, committed) or else
    ``<path>.prev``.

    Parameters
    ----------
    path : Any
        A checkpoint path.

    Returns
    -------
    str
        The directory to read.

    Raises
    ------
    FileNotFoundError
        If none of them is a committed checkpoint.
    """
    path = os.path.abspath(os.fspath(path))
    for candidate in (path, f"{path}.next", f"{path}.prev"):
        if _is_committed(candidate):
            return candidate
    raise FileNotFoundError(f"no committed checkpoint at {path}")


def is_checkpoint(path: Any) -> bool:
    """Whether ``path`` names a sharded checkpoint (a directory; a ``.npz`` is not).

    Parameters
    ----------
    path : Any
        A path.

    Returns
    -------
    bool
        ``True`` for a committed checkpoint directory (or its ``.next`` /
        ``.prev``).
    """
    try:
        resolve(path)
    except FileNotFoundError:
        return False
    return True


def latest_checkpoint(pattern: str) -> Optional[str]:
    """The committed checkpoint of the highest step a restart path pattern names.

    For SLURM job chaining (a job continues from the last checkpoint and
    resubmits itself): ``pattern`` is ``SegmentConfig.restart_path`` with its
    ``{step}`` / ``{segment}`` fields (format specs allowed); each field matches
    digits. A checkpoint left half-written by a killed job is not committed and
    not returned.

    Parameters
    ----------
    pattern : str
        E.g. ``"ckpt/step_{step:08d}"``.

    Returns
    -------
    Optional[str]
        The path, or ``None`` when there is none.
    """
    pattern = os.path.abspath(pattern)
    fields = re.compile(r"\{(step|segment)(:[^}]*)?\}")
    wildcard = fields.sub("*", pattern)
    regex = re.compile(
        "^"
        + "".join(
            (
                f"(?P<{m.group(1)}{i}>\\d+)"
                if isinstance(m, re.Match)
                else re.escape(m)
            )
            for i, m in enumerate(_split_fields(fields, pattern))
        )
        + "$"
    )
    best: Optional[tuple] = None
    for candidate in glob.glob(wildcard):
        match = regex.match(candidate)
        if match is None or not _is_committed(candidate):
            continue
        values = {k.rstrip("0123456789"): int(v) for k, v in match.groupdict().items()}
        key = (values.get("step", -1), values.get("segment", -1))
        if best is None or key > best[0]:
            best = (key, candidate)
    return None if best is None else best[1]


def _split_fields(fields: Any, text: str) -> list:
    """``text`` cut at ``fields``' matches: literal strings and the matches.

    Parameters
    ----------
    fields : Any
        A compiled pattern.
    text : str
        The text.

    Returns
    -------
    list
        Strings and ``re.Match`` objects in order.
    """
    out: list = []
    at = 0
    for match in fields.finditer(text):
        out.append(text[at : match.start()])
        out.append(match)
        at = match.end()
    out.append(text[at:])
    return out


# ----------------------------------------------------------------------
# The manifest.
# ----------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64's finaliser, elementwise on ``uint64`` (wrapping).

    Parameters
    ----------
    x : np.ndarray
        ``uint64``.

    Returns
    -------
    np.ndarray
        ``uint64``.
    """
    x = x ^ (x >> np.uint64(30))
    x = x * _MIX1
    x = x ^ (x >> np.uint64(27))
    x = x * _MIX2
    return x ^ (x >> np.uint64(31))


def _row_words(values: np.ndarray) -> np.ndarray:
    """Each row's bytes as ``uint64`` words (zero-padded to 8 bytes).

    Parameters
    ----------
    values : np.ndarray
        ``(n, ...)``.

    Returns
    -------
    np.ndarray
        ``(n, words)`` ``uint64``.
    """
    n = int(values.shape[0])
    raw = np.ascontiguousarray(values).reshape(n, -1).view(np.uint8)
    pad = (-raw.shape[1]) % 8
    if pad:
        raw = np.concatenate([raw, np.zeros((n, pad), np.uint8)], axis=1)
    return np.ascontiguousarray(raw).view(np.uint64)


def set_hash(ids: np.ndarray, values: np.ndarray) -> int:
    """An order-independent hash of the live rows of ``values`` keyed by ``ids``.

    ``sum_i mix(id_i, bytes_i) mod 2**64`` over the rows with ``ids >= 0``: the
    same for any order and any split of the rows over devices, so a restore onto
    another device count checks what it read against the writer's value by adding
    the processes' partial sums.

    Parameters
    ----------
    ids : np.ndarray
        ``(n,)`` integer ids (``-1``: padding, left out).
    values : np.ndarray
        ``(n, ...)``.

    Returns
    -------
    int
        In ``[0, 2**64)``.
    """
    live = np.asarray(ids) >= 0
    if not live.any():
        return 0
    with np.errstate(over="ignore"):
        h = _mix(np.asarray(ids)[live].astype(np.uint64) + _GOLDEN)
        for word in _row_words(np.asarray(values)[live]).T:
            h = _mix(h ^ word)
        return int(np.sum(h, dtype=np.uint64))


def _sha256(values: np.ndarray) -> str:
    """sha256 of an array's bytes (C order).

    Parameters
    ----------
    values : np.ndarray
        The array.

    Returns
    -------
    str
        Hex digest.
    """
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _local_shards(arr: Any) -> list:
    """``(first_row, host_data)`` of this process's distinct shards of ``arr``.

    Parameters
    ----------
    arr : Any
        A ``jax.Array`` sharded on its first axis (or replicated).

    Returns
    -------
    list
        One entry per shard this process writes (``replica_id == 0``).
    """
    import jax

    out = []
    for shard in arr.addressable_shards:
        if int(shard.replica_id) != 0:
            continue
        first = shard.index[0] if shard.index else slice(None)
        start = 0 if first.start is None else int(first.start)
        out.append((start, np.asarray(jax.device_get(shard.data))))
    return out


def _allgather(obj: Any) -> list:
    """Every process's ``obj`` (``[obj]`` in one process).

    Parameters
    ----------
    obj : Any
        A picklable object.

    Returns
    -------
    list
        In process order.
    """
    import jax

    if jax.process_count() == 1:
        return [obj]
    from jaccpot.distributed.multiprocess import allgather_object

    return allgather_object(obj)


def build_manifest(arrays: Mapping[str, Any], id_arrays: Mapping[str, str]) -> dict:
    """The manifest of ``arrays`` (every process calls it: a collective).

    Each process hashes its own shards (one shard on the host at a time); the
    small per-shard records are all-gathered.

    Parameters
    ----------
    arrays : Mapping[str, Any]
        ``name -> jax.Array`` (sharded on the first axis, or replicated).
    id_arrays : Mapping[str, str]
        ``name -> the name of its ids array``: those arrays also get a
        :func:`set_hash` over their live rows.

    Returns
    -------
    dict
        ``{"arrays": {name: {"shape", "dtype", "shards": [[first_row, rows,
        sha256], ...], "set_hash": hex or None}}, "bytes": total}``.
    """
    local: dict = {}
    for name, arr in arrays.items():
        ids_name = id_arrays.get(name)
        ids_by_start = (
            None if ids_name is None else dict(_local_shards(arrays[ids_name]))
        )
        records, partial = [], 0
        for start, data in _local_shards(arr):
            records.append([start, int(data.shape[0]) if data.ndim else 1, _sha256(data)])
            if ids_by_start is not None:
                partial = (partial + set_hash(ids_by_start[start], data)) % (1 << 64)
            del data
        local[name] = (records, partial)
    merged: dict = {}
    total = 0
    for part in _allgather(local):
        for name, (records, partial) in part.items():
            entry = merged.setdefault(name, {"shards": {}, "set_hash": 0})
            for start, rows, digest in records:
                entry["shards"][int(start)] = [int(start), int(rows), digest]
            entry["set_hash"] = (entry["set_hash"] + int(partial)) % (1 << 64)
    out: dict = {}
    for name, arr in arrays.items():
        entry = merged[name]
        itemsize = np.dtype(arr.dtype).itemsize
        per_row = int(np.prod(arr.shape[1:], dtype=np.int64)) * itemsize
        shards = [entry["shards"][k] for k in sorted(entry["shards"])]
        total += sum(rows for _, rows, _ in shards) * per_row
        out[name] = {
            "shape": [int(v) for v in arr.shape],
            "dtype": str(np.dtype(arr.dtype)),
            "shards": shards,
            "set_hash": (
                None if name not in id_arrays else f"{entry['set_hash']:016x}"
            ),
        }
    return {"arrays": out, "bytes": int(total)}


# ----------------------------------------------------------------------
# Writing.
# ----------------------------------------------------------------------


def _local_bytes(arrays: Mapping[str, Any]) -> int:
    """Bytes of the shards this process writes.

    Parameters
    ----------
    arrays : Mapping[str, Any]
        ``name -> jax.Array``.

    Returns
    -------
    int
        Bytes.
    """
    total = 0
    for arr in arrays.values():
        for shard in arr.addressable_shards:
            if int(shard.replica_id) == 0:
                total += int(np.prod(shard.data.shape, dtype=np.int64)) * int(
                    np.dtype(arr.dtype).itemsize
                )
    return total


def _swap(path: str, target: str) -> None:
    """Put the committed ``target`` (``<path>.next``) in place of ``path``.

    ``path -> <path>.prev``, ``target -> path``, ``<path>.prev`` removed: a crash
    between the renames leaves ``<path>.next`` or ``<path>.prev`` committed, which
    :func:`resolve` reads.

    Parameters
    ----------
    path : str
        The checkpoint path.
    target : str
        The new checkpoint, committed.
    """
    _rename(path, f"{path}.prev")
    _rename(target, path)
    shutil.rmtree(f"{path}.prev", ignore_errors=True)


#: the swap's rename (a seam for the crash tests)
_rename = os.rename


def write_checkpoint(
    path: Any,
    arrays: Mapping[str, Any],
    meta: Mapping[str, Any],
    *,
    id_arrays: Optional[Mapping[str, str]] = None,
) -> dict:
    """Write a checkpoint (every process calls it: a collective).

    Parameters
    ----------
    path : Any
        The checkpoint directory (absolute or relative; replaced atomically if it
        exists).
    arrays : Mapping[str, Any]
        ``name -> jax.Array``; ``None`` values are left out.
    meta : Mapping[str, Any]
        JSON-able; ``format`` and ``manifest`` are added.
    id_arrays : Optional[Mapping[str, str]]
        As :func:`build_manifest`.

    Returns
    -------
    dict
        ``path``, ``seconds`` (this process, wall), ``bytes_local`` (the bytes of
        the shards this process wrote, before orbax's compression) and ``bytes``
        (every process's).
    """
    ocp = _ocp()
    import jax

    started = time.perf_counter()
    path = os.path.abspath(os.fspath(path))
    arrays = {k: v for k, v in arrays.items() if v is not None}
    id_arrays = {k: v for k, v in (id_arrays or {}).items() if k in arrays}
    manifest = build_manifest(arrays, id_arrays)
    document = dict(meta, format=CHECKPOINT_FORMAT, manifest=manifest)
    primary = jax.process_index() == 0
    replace = os.path.exists(path)
    target = f"{path}.next" if replace else path
    if primary:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # leftovers of a crash during an earlier replace: older than what this
        # call writes
        for stale in (f"{path}.next", f"{path}.prev"):
            shutil.rmtree(stale, ignore_errors=True)
    _sync(f"jaccpot-ckpt-pre:{target}")
    _checkpointer().save(
        target,
        args=ocp.args.Composite(
            arrays=ocp.args.PyTreeSave(dict(arrays)),
            meta=ocp.args.JsonSave(document),
        ),
    )
    if replace and primary:
        _swap(path, target)
    _sync(f"jaccpot-ckpt-post:{path}")
    return {
        "path": path,
        "seconds": time.perf_counter() - started,
        "bytes_local": _local_bytes(arrays),
        "bytes": int(manifest["bytes"]),
    }


# ----------------------------------------------------------------------
# Reading.
# ----------------------------------------------------------------------


def read_meta(path: Any) -> dict:
    """A checkpoint's ``meta`` (every process reads it).

    Parameters
    ----------
    path : Any
        The checkpoint.

    Returns
    -------
    dict
        The JSON document.

    Raises
    ------
    ValueError
        If it is not a checkpoint of this format.
    """
    import json

    where = resolve(path)
    with open(os.path.join(where, "meta", "metadata")) as fh:
        meta = json.load(fh)
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint "
            f"(format {meta.get('format')!r})"
        )
    return meta


def restore_arrays(
    path: Any,
    specs: Mapping[str, tuple],
) -> dict:
    """Restore named arrays onto shardings (every process calls it).

    Parameters
    ----------
    path : Any
        The checkpoint.
    specs : Mapping[str, tuple]
        ``name -> (sharding, global_shape)``: ``global_shape`` ``None`` keeps the
        written shape (each process reads its shards); a longer first axis pads
        with zeros at the end (orbax ``strict=False``, a re-sharding restore onto
        another mesh). ``sharding`` ``None``: a host ``np.ndarray`` of the whole
        array (one host).

    Returns
    -------
    dict
        ``name -> jax.Array`` (or ``np.ndarray``).
    """
    ocp = _ocp()
    import jax

    where = resolve(path)
    manifest = read_meta(where)["manifest"]["arrays"]
    item, restore_args = {}, {}
    for name, (sharding, shape) in specs.items():
        written = tuple(manifest[name]["shape"])
        dtype = np.dtype(manifest[name]["dtype"])
        if sharding is None:
            item[name] = np.zeros(0, dtype)
            restore_args[name] = ocp.RestoreArgs(restore_type=np.ndarray)
            continue
        shape = written if shape is None else tuple(int(v) for v in shape)
        item[name] = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
        restore_args[name] = ocp.ArrayRestoreArgs(
            sharding=sharding, global_shape=shape, strict=shape == written
        )
    got = _checkpointer().restore(
        where,
        args=ocp.args.Composite(
            arrays=ocp.args.PyTreeRestore(
                item=item, restore_args=restore_args, partial_restore=True
            )
        ),
    )
    return dict(got["arrays"])


def verify_shards(arrays: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    """Check restored arrays' shards against the manifest's sha256 (same layout).

    Every process checks its own shards; the verdict is all-gathered, so every
    process raises together.

    Parameters
    ----------
    arrays : Mapping[str, Any]
        ``name -> jax.Array`` restored with the written shape and a sharding
        whose shards are the written ones (or ``np.ndarray``: every shard).
    manifest : Mapping[str, Any]
        The checkpoint's ``meta["manifest"]``.

    Raises
    ------
    CheckpointCorruptError
        Naming the arrays and first rows of the shards that differ.
    """
    bad: list = []
    for name, arr in arrays.items():
        records = {int(r[0]): r for r in manifest["arrays"][name]["shards"]}
        if isinstance(arr, np.ndarray):
            pieces = [(start, arr[start : start + rows]) for start, rows, _ in records.values()]
        else:
            pieces = _local_shards(arr)
        for start, data in pieces:
            record = records.get(int(start))
            if record is None or _sha256(data) != record[2]:
                bad.append(f"{name}[{start}:]")
    found = sorted({b for part in _allgather(bad) for b in part})
    if found:
        raise CheckpointCorruptError(
            f"checkpoint shards differ from the manifest's sha256: {found}"
        )


def verify_set_hashes(
    rows: Sequence[tuple], manifest: Mapping[str, Any], names: Sequence[str]
) -> None:
    """Check rows read in ANY split against the manifest's set hashes.

    Parameters
    ----------
    rows : Sequence[tuple]
        This process's ``(ids, {name: values})`` pieces (padding ids ``-1``).
    manifest : Mapping[str, Any]
        The checkpoint's ``meta["manifest"]``.
    names : Sequence[str]
        The arrays to check (each must have a set hash).

    Raises
    ------
    CheckpointCorruptError
        If a sum differs.
    """
    partial = {name: 0 for name in names}
    for ids, values in rows:
        for name in names:
            partial[name] = (partial[name] + set_hash(ids, values[name])) % (1 << 64)
    total = {name: 0 for name in names}
    for part in _allgather(partial):
        for name in names:
            total[name] = (total[name] + int(part[name])) % (1 << 64)
    bad = [
        name
        for name in names
        if f"{total[name]:016x}" != manifest["arrays"][name]["set_hash"]
    ]
    if bad:
        raise CheckpointCorruptError(
            f"checkpoint arrays {bad} differ from the manifest's set hash"
        )


def load_checkpoint(
    path: Any, names: Optional[Sequence[str]] = None, *, verify: bool = True
) -> tuple:
    """A checkpoint on one host, BY ID (caller row order), and its ``meta``.

    Reads every shard (one host holds the whole set: analysis, a single-card
    continuation). The layout's live rows (and the escaper tail's, ``tail_*``)
    are placed at their ids.

    Parameters
    ----------
    path : Any
        The checkpoint.
    names : Optional[Sequence[str]]
        Per-particle arrays (``state``, ``masses``, ``acceleration``,
        ``force_scale``); ``None``: every one written.
    verify : bool
        Check every shard's sha256 against the manifest.

    Returns
    -------
    tuple
        ``(arrays, meta)``: ``name -> (N, ...)`` in id order (a name the tail
        lacks is zero on its rows; with an escaper tail also ``escaper``, a
        boolean per id), and the meta.

    Raises
    ------
    CheckpointCorruptError
        With ``verify``, for a shard that differs from the manifest.
    ValueError
        If the ids are not exactly ``0..N-1`` once each.
    """
    meta = read_meta(path)
    written = meta["manifest"]["arrays"]
    particle = ("state", "masses", "acceleration", "force_scale")
    names = [n for n in (names or particle) if n in written]
    wanted = list(names) + ["ids"]
    tail = "tail_ids" in written
    if tail:
        wanted += ["tail_ids"] + [f"tail_{n}" for n in names if f"tail_{n}" in written]
    host = restore_arrays(path, {n: (None, None) for n in wanted})
    if verify:
        verify_shards(host, meta["manifest"])
    ids = [host["ids"]] + ([host["tail_ids"]] if tail else [])
    live_ids = np.concatenate([i[i >= 0] for i in ids]).astype(np.int64)
    n = int(live_ids.shape[0])
    seen = np.zeros(n, bool)
    if n and (live_ids.min() < 0 or live_ids.max() >= n):
        raise ValueError("particle identity broken: an id outside 0..N-1")
    seen[live_ids] = True
    if not seen.all():
        raise ValueError("particle identity broken: ids are not exactly 0..N-1 once")
    out = {}
    for name in names:
        values = host[name]
        parts = [values[host["ids"] >= 0]]
        if tail:
            tvals = host.get(f"tail_{name}")
            count = int((host["tail_ids"] >= 0).sum())
            parts.append(
                tvals[host["tail_ids"] >= 0]
                if tvals is not None
                else np.zeros((count,) + values.shape[1:], values.dtype)
            )
        full = np.empty((n,) + values.shape[1:], values.dtype)
        full[live_ids] = np.concatenate(parts)
        out[name] = full
    if tail:
        escaper = np.zeros(n, bool)
        escaper[host["tail_ids"][host["tail_ids"] >= 0]] = True
        out["escaper"] = escaper
    return out, meta

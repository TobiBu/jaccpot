# Session prompt — vectorise the distributed scatter-back

Paste this into a fresh session. Self-contained: it names the exact site, the
replacement, the two things that must **not** change, and the gates.

Scope is deliberately one function body. This is a **performance and clarity fix,
not a differentiability fix** — see "What this is not" before widening it.

---

## Situation

`jaccpot/distributed/fmm.py::distributed_fmm_accelerations` scatters the
per-device forces back into input order with an O(N) **Python loop over rows**:

```python
# jaccpot/distributed/fmm.py, currently lines 1830-1842
n = part["n"]
accel = np.zeros((n, 3), np.float64)
seen = np.zeros(n, bool)
for row in range(accel_o.shape[0]):
    g = gid_o[row]
    if g >= 0:
        accel[g] = accel_o[row]
        seen[g] = True
if not seen.all():
    raise RuntimeError(
        f"{int((~seen).sum())} particles missing from the distributed result "
        "(padding/capacity bug)"
    )
```

`gid_o` is the global-id array threaded through `partition_for_devices`; padding
rows carry `-1`. So the loop is a masked scatter plus a completeness check, both
of which vectorise directly.

Measured on this host (CPU, NumPy, best of 5, ndev=4, leaf=128, results verified
identical with `np.array_equal`):

| N | Python loop | vectorised | speedup |
| --- | --- | --- | --- |
| 10 000 | 8.1 ms | 0.4 ms | 21x |
| 100 000 | 92.3 ms | 8.2 ms | 11x |
| 1 000 000 | 996 ms | 109 ms | 9x |

**Honest impact.** This is *not* on the hot path for the multi-GPU scaling
figures, which drive `make_force_evaluator` build-once and read 42-64 ms per eval
in steady state. And `distributed_fmm_accelerations` itself rebuilds and
recompiles per call (~50-80 s), which dwarfs even the 1 s loop at N=1e6. So this
is a real defect that is currently not a bottleneck. Do it because a
million-iteration Python loop in a GPU library is wrong and cheap to remove, not
because it will show up in a figure.

## The change

Replace the loop with the masked scatter. Keep it **NumPy / host-side** (see
"What this is not"):

```python
n = part["n"]
accel = np.zeros((n, 3), np.float64)
seen = np.zeros(n, bool)
valid = gid_o >= 0
g = gid_o[valid]
accel[g] = accel_o[valid]
seen[g] = True
if not seen.all():
    raise RuntimeError(...)   # unchanged
```

Keep the error message and its `(~seen).sum()` count exactly as they are — it is
the padding/capacity tripwire and its wording is referenced by the capacity work.

## What must NOT change

Two things in this function look like the same class of problem and are not.
Leave both alone; if you think either is wrong, stop and say so rather than
changing it.

1. **`partition_for_devices` stays host-side NumPy.** It computes `cap` from the
   data (`counts.max()` rounded up to a `leaf_size` multiple), and `cap` is a
   *shape* fed to `shard_map`. A data-dependent shape cannot be traced. This is
   the same fixed-topology contract as the single-GPU path, where `prepare_state`
   also runs concretely on the host.
2. **The `while True` cap-retry loop stays Python.** It reads an overflow
   diagnostic and *recompiles* the evaluator with larger capacities via
   `with_selective_scaled_caps`. "Recompile with bigger buffers" is not a
   traceable operation.

## What this is not

This does **not** make `distributed_fmm_accelerations` differentiable, and must
not be sold as doing so. The differentiable entry point is and remains
`make_force_evaluator(..., differentiable=True)`, whose gradients are taken
w.r.t. the padded per-device layout — see
`docs/differentiable_fmm_distributed_audit.md`.

Making the driver differentiable **in input order** is a separate, larger change:
split it into a concrete `prepare_distributed_state(...)` (partition, shapes,
`gid`, `counts` — frozen, host-side, mirroring single-GPU `prepare_state`) plus a
traceable `evaluate(state, positions, masses)` that gathers into the frozen order
and scatters back with `.at[gid].set(...)`. That needs 2 GPUs to test and is its
own PR. Do not start it here.

Consequently: do **not** convert this scatter to `jnp`. The function returns
`DistributedFMMResult.accelerations` as a host `np.ndarray`, and moving the
scatter to device would change that contract for no benefit until the
prepare/evaluate split lands.

## Test first

`jaccpot/distributed/` is production library code, so per `CLAUDE.md` the test
comes first. There is currently no unit test that pins the scatter's contract.
Add one that runs **without GPUs** by calling the scatter logic directly (extract
it to a small module-level helper such as `_scatter_to_input_order(accel_o,
gid_o, n)` and test that, so the test needs no mesh, no devices and no JAX):

- round-trip: a known permutation with `-1` padding rows scatters back to input
  order exactly;
- padding rows are ignored (`-1` never writes);
- an incomplete `gid` (one particle missing) still raises `RuntimeError` with the
  count in the message;
- equivalence: the vectorised result is `np.array_equal` to the old loop on a
  randomised case with padding — keep a copy of the loop *inside the test* as the
  reference oracle, so the test states the equivalence rather than assuming it.

Extracting the helper is the point of the change as much as the speed is: it is
what makes the contract testable at all.

## Forward numerics

The scatter is a permutation, so the forward must be **bit-identical**, not
"within tolerance". The equivalence test above is the local proof; the
characterization goldens are the global one.

## Gates

```bash
black --check .
isort --check-only .
pre-commit run --all-files
JAX_ENABLE_X64=1 pytest -q
JAX_ENABLE_X64=1 pytest -q tests/characterization
```

The distributed suite skips below 2 devices, so a CPU run will collect and skip
`tests/distributed/`. If GPUs are free, also run:

```bash
export CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
  pytest tests/distributed/ -o addopts="" -q
```

At the time this prompt was written every GPU on the host was occupied (7 of 8
cards holding 34-39 GB of 40 GB), so the GPU leg was not run. It is not required
to land this — the change is host-side and CPU-testable — but say in the PR which
legs you actually ran.

## Commit

One commit, conventional format, e.g.

```
perf(distributed): vectorise the scatter back to input order

An O(N) Python loop over rows became a masked NumPy scatter: 9-21x at
N=1e4..1e6, results bit-identical (it is a permutation). Extracted to
_scatter_to_input_order so the contract is testable without a mesh.

The partition and the cap-retry loop stay host-side deliberately: the
former computes a shard_map shape, the latter recompiles on overflow.
```

Do not fold in the prepare/evaluate split, and do not touch `bench/` or the
paper branch from this session.

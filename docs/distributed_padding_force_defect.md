# RESOLVED: per-device padding does not corrupt the distributed force

_Reported 2026-08-22 on `paper/jaccpot-i` at `7cf2903`, 6xA100, jax 0.10.2, yggdrax
main `783ca6b`. Diagnosed and closed 2026-08-22. **The forces were always correct.**
The 40% error was in the benchmark harness's accuracy check, which read the
accelerations back in the wrong order whenever a device was padded._

## Verdict

`bench/multigpu/harness.py` discarded the `gid` the evaluator returns and indexed the
accelerations with the **input** `part["gid_flat"]` instead:

```python
accel, _gid, diag = evaluate(*args)          # <- returned gid thrown away
...
gid_np = np.asarray(args[2]).reshape(-1)     # <- the INPUT gid
pick = rng.choice(np.flatnonzero(gid_np >= 0), size=k, replace=False)
tgt_ids = gid_np[pick]
got = accel_np[pick]                         # <- output row `pick`, input id `pick`
```

That is valid only while the per-device pipeline hands its rows back in the order it
received them. It does -- right up until a device is padded, and never again after.

Nothing about the solver, the tree, the MAC, the LET or the halo import is wrong.
Every timing, throughput, work-count and load-balance number ever produced by this
harness stands, on padded and unpadded configurations alike: the accuracy block runs
*after* the timing loop and feeds nothing back into it. Only the
`accuracy.rel_l2_vs_direct` and `accuracy.max_abs_err` columns, and only on
configurations where `count % leaf_size != 0`, were wrong.

## The mechanism

`partition_for_devices` pads each device up to `cap` with rows placed at
`positions[chunk[0]]` -- that device's **first particle in Morton order**, hence its
**smallest Morton code**:

```python
pos_g[d, c:] = positions[chunk[0]]   # pad at a real point
```

The per-device body then rebuilds a local tree, which re-sorts the shard by Morton
code. The padding rows tie with row 0 for the minimum code, so a stable sort puts them
immediately after it and pushes every other real particle back by the padding count.
Measured on device 0 at `ndev=3`, `count=621`, `cap=640` (57 padding rows):

```
perm = [0, 621, 622, ..., 677, 1, 2, 3, ...]     639 of 640 rows displaced
```

With `cap == count` there is no padding, the permutation is the identity, and reading
the output by input row is correct **by accident**. That accident is the whole defect:
it made the wrong code work for as long as no per-device count failed to divide
`leaf_size`.

And it is a *robust* accident, which is why it survived so long. The obvious other way
to break it -- tied Morton codes, where the host's unstable `np.argsort` and the
device's `jnp.argsort(stable=True)` could disagree -- cannot: the device receives the
chunk already in the host's order, and a stable sort leaves tied keys in the order they
arrive, so it reproduces whatever the host chose. Measured on 64 particles with 55 tied
codes, and on 16 positions each repeated four times: zero rows displaced in both.
Padding is the only thing that breaks the identity, and it breaks it completely.

Because the displacement is by a handful of slots *in Morton order*, each particle's
force was compared against a spatial near-neighbour's -- smooth, plausible, and wrong
by tens of percent rather than obviously garbage. That is also why the original
report's `rel_L2 = sqrt(k/N)` inference ("16% of particles are badly wrong") was
misleading: the error is not a few catastrophic outliers, it is every particle off by
a fraction of the local force gradient.

## Evidence

All on CPU with forced host devices, which reproduces the distributed path faithfully.

**1. The permutation, and the two readouts of the same forces** (`ndev=3`, leaf 64,
order 6, Gaussian IC, softening 0.02). Same evaluator call, same accelerations; only
the map back to input order differs:

| n | count/dev | cap | pad | rows displaced | rel-L2 via input gid | rel-L2 via returned gid |
| --- | --- | --- | --- | --- | --- | --- |
| 1920 | 640 | 640 | 0 | 0 | 1.258e-15 | 1.258e-15 |
| 1863 | 621 | 640 | 57 | 639 | **8.441e-01** | **1.206e-15** |

The padded forces are exact to machine precision. The 0.84 is the readout.

**2. The local tree is not damaged by padding**, at the report's own sizes
(`ndev=3`, leaf 64, the exact `n=32832` / `n=32769` pair). Per device, padded vs
unpadded:

- 171 leaves either way, every leaf exactly 64 particles;
- every row covered by exactly one leaf;
- root mass equal to the true per-device mass to all printed digits;
- **zero** empty leaves -- the padding is coincident with row 0, so it lands in leaf 0
  rather than forming one of its own;
- frontier radius max 5.63 (padded) vs 5.83 (unpadded); node radius max identical.

This kills both hypotheses in the original report. There is no degenerate radix
subtree (the Karras build's `delta` already breaks code ties on index, and leaves are
fixed-size blocks of the sorted order, so duplicates cannot make an oversized or
malformed leaf), and there is no inflated boundary-leaf radius (the padding sits on a
real particle *of the leaf it lands in*, so it cannot extend that leaf's extent).

**3. End to end through the driver**, which has always used the returned `gid`
(`ndev=3`, leaf 16, order 6, cross far field engaged):

| n | count/dev | cap | pad | cross far pairs | rel-L2 vs direct |
| --- | --- | --- | --- | --- | --- |
| 3072 | 1024 | 1024 | 0 | 0 | 3.86e-07 |
| 3027 | 1009 | 1024 | 45 | 168 | 2.21e-06 |

**4. The regression test**, `tests/distributed/test_distributed_padded_partition.py`,
on the separated-cluster IC at defaults (order 3, leaf 8, `ndev=3`): padded aggL2
1.2e-05 against an unpadded control at 1.3e-05, and the same accelerations read
through the input gid at 1.322194.

**5. The harness itself**, before and after the one-line change, on 3 forced CPU
devices at leaf 16 with `--accuracy-targets 256`:

| n | count/dev | cap | pad | `rel_l2_vs_direct` before | after |
| --- | --- | --- | --- | --- | --- |
| 768 | 256 | 256 | 0 | 8.47210304036577e-16 | 8.47210304036577e-16 |
| 750 | 250 | 256 | 6 | **9.203e-01** | **9.09e-16** |

The unpadded point is bit-identical across the change, to every digit and in
`max_abs_err` too: the fix cannot move a number that was already right. The padded
point's pre-fix `max_abs_err / ref_rms` is 866.4 / 402.3 = **2.15**, which is the
"per-particle worst case ~2.0" the original report measured on a completely different
problem size -- the signature of a Morton-neighbour mismatch, not of a size-dependent
numerical defect.

## What changed

- **`jaccpot.distributed.scatter_to_input_order`** -- the reassembly, promoted out of
  `distributed_fmm_accelerations` into a public helper, so the four call sites that
  hand-rolled this loop have one correct thing to call. It raises on a missing
  particle rather than returning a zero row.
- **`tests/distributed/test_distributed_padded_partition.py`** -- a deliberately
  padded partition against a direct sum, in both the equal-count and ragged-count
  shapes, plus a test that pins the permutation itself: with padding, the input gid
  readout *must* be wrong, and the file says what to re-audit if that ever stops being
  true.
- **`tests/unit/test_distributed_scatter_to_input_order.py`** -- the helper's own
  contract, in CPU CI on one device. The distributed suite skips below two devices,
  which is part of why this went unnoticed for so long.
- **`tests/distributed/test_distributed_grad_correctness.py`** -- the same latent
  misuse, fixed. Its oracle scattered the accelerations with the input gid; its IC
  never pads, so it was right by the same accident. The *cotangents* really are in
  the input layout and keep the input map -- two arrays, two orders, now spelled out.
- **Docstrings** on `partition_for_devices` and `make_force_evaluator`, which
  documented the contract but not the trap.
- **`bench/multigpu/harness.py`** and
  `bench/differentiability/distributed_grad_correctness.py` -- the defect itself, and
  the same misuse beside it. They live on `paper/jaccpot-i`, so they are fixed there
  on their own branch, reading the returned gid inline rather than importing the
  helper: it is not on that branch yet. Route them through it when the two meet.

## What this unblocks

**Nothing published needs re-measuring.** Audited every record in
`bench/results/multigpu/` (7 files, 36 records): all of them were taken at
`n=30720`, whose per-device count is a leaf multiple at every `ndev` in the sweep
(15360, 10240, 7680, 6144, 5120 at leaf 128). So `cap == count` throughout, no point
was padded, and every accuracy entry was read back correctly. Their 8e-05 to 1.5e-04
rel-L2 is the real number.

What changes is that the constraint disappears. `n=30720` was picked *because* it
divides cleanly -- the original report's suggestion 3, applied as a workaround -- and
strong scaling no longer has to choose N that way. Any N at any device count now
reports its true accuracy.

The original report's "older results are suspect" was too broad in one direction and
too narrow in another: no *force* was ever wrong, on any configuration, so every
timing, throughput and work-count number ever taken stands; but any accuracy number
read off a padded configuration during the investigation was wrong, and those are the
ones in the report's own table.

## The lesson

The report closed with "the direct-sum accuracy check found this one, and it is the
only instrument that would have." It was the other way round: the direct-sum check
*was* the defect. An instrument that only fires on a configuration nothing else tests
is an instrument nothing tests, and the padded configuration was exactly that -- every
distributed IC in the suite is `ndev * per` with `per` a leaf multiple, so the padding
branch had never been executed by anything but the benchmark.

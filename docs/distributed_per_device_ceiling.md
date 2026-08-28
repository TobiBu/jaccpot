# The distributed per-device ceiling is a leaf-count limit, not a density limit

> **Status, 2026-08-28.** The ceiling this note measures is **gone**. The root cause
> was found on 2026-08-24 (see [Root cause](#root-cause-the-ladder-cannot-be-climbed-under-a-trace)
> below) and fixed in yggdrax (`#52`, `_walk_inputs_are_traced`) together with the
> capacity derivation in `jaccpot/distributed/fmm.py` (`_derive_walk_caps`,
> `DERIVED_CAP_FIELDS`). Per-device N now reaches **1 048 576**, 128x the 8000 this
> note set out to correct -- see [After the fix](#after-the-fix-measured-on-main).
>
> The note is kept, and kept in its original order, for two reasons. The measurements
> in it are the evidence the fix was derived from, and the failure mode it describes --
> a silent truncation that reads as a speedup -- is the reason `self_near_pairs` is
> still recorded and still checked before any wall clock is quoted.

_Measured 2026-08-21, 2xA100-PCIE-40GB (and confirmed at 5 devices), jax 0.10.2,
real basis + dehnen MAC, float64, `make_force_evaluator` build-once._

## Summary

`docs/phase5_multigpu_pallas_foldin_plan.md` records a "healthy per-GPU N ~ 8000"
and describes strong scaling as **density-limited**, with 400-720 ms readings
above that load attributed to padded pair-queue overhead. All three of those
statements need correcting.

The real limit is **64 leaves per device**, set by `process_block`, whose default
is 64. It is not a particle-count limit and not a density limit, and the readings
above the limit are not overhead: they are a **truncated dual-tree walk**.

| leaves/device | `process_block` | result |
| --- | --- | --- |
| 64 | 64 (default) | valid, 0 retries, at every leaf size tried |
| 128 | 64 | truncated |
| 128 | 128 | truncated |
| 128 | **256** | **valid** |

Per-device capacity is `(leaves per device) x leaf_size`, and both factors are
configuration. The 8000 figure is exactly `leaf_size=128 x 64 leaves`, i.e. an
artefact of two defaults meeting, and nothing more fundamental.

## The evidence, in the order it landed

**1. The default leaf size cannot even reach the documented load.** At the
distributed default `leaf_size=8`, 8000 particles/device overflows three buffers
and still fails after four capacity retries (16x). The phase-5 table was measured
at `leaf_size=128`; the log says so in passing ("ndev=2, per=8000/leaf=128") but
the 8000 was carried around without it.

**2. It is not buffer sizing.** At 32768 particles/device with `leaf_size=128`,
the self pair queue was grown to **16 777 216** entries -- 64x the single-device
large-N value -- and the walk still truncated. A few hundred nodes per device
cannot legitimately need 16M node-pairs in flight.

**3. The failure is truncation, and it is visible in the pair counts.**
`self_near_pairs` *falls* as particles rise, which is physically impossible:

| per-device N | `leaf_size` | self_near_pairs | valid |
| --- | --- | --- | --- |
| 8 000 | 128 | 3882 | yes |
| 12 000 | 128 | 940 | no |
| 16 000 | 128 | 394 | no |
| 24 000 | 128 | 12 | no |
| 32 000 | 128 | 2 | no |

**4. The cliff is at 64 leaves, not at a particle count.** 8192/device
(64 leaves) is valid with zero retries; 8500 (67 leaves) is not.

**5. The controlled test.** 16 384 particles/device is **valid at `leaf_size=256`
(64 leaves) and truncated at `leaf_size=128` (128 leaves)** -- same particle
count, same caps, only the leaf count differs. And in every valid run
`self_near_pairs` is ~4020 regardless of particle count, because 64 leaves x ~63
neighbours is a constant: the work tracks leaf count, not N.

**6. `process_block` moves the wall.** 128 leaves/device truncates at
`process_block` 64 and 128, and is valid at 256, where `self_near_pairs` reaches
15 796 -- ~4x the 64-leaf value, exactly as 4x the leaves should give.

## Why this matters more than a tuning note

**The failure mode reads as faster.** A truncated walk does less work, so the
wall clock *improves*. 32 000 particles/device came in at 157 ms against 98 ms at
a quarter of the load. Anything that plots time against device count without
checking validity will produce a beautiful superlinear speedup out of a force
that was never computed. This is why `bench/multigpu/harness.py` marks a point
`valid=false` on any overflow flag and why `self_near_pairs` is recorded: the
counter is the honest check, since it collapses on truncation.

The 400-720 ms strong-scaling readings in the phase-5 log are almost certainly
this, not padding overhead.

## What was reached by raising `process_block`

This is the workaround, not the fix: the numbers below were reached by handing the
walk a bigger `process_block`, before anyone knew why that was the knob that moved.
At `leaf_size=256`, `process_block=256`, pair queues 262144:

- **32 768 particles/device** valid at `leaf_size=512` (64 leaves).
- **16 384 particles/device** valid at `leaf_size=128` once `process_block=256`.
- Weak scaling ndev 2..5 at 16 384/device: all four points valid, 0-1 retries,
  throughput 5.79e5 -> 9.11e5 particles/s, `self_near_pairs` ~4000 throughout.
- 81 920 particles at 5 devices in 89.9 ms, against the log's 40 000 in 64.3 ms.

## Root cause: the ladder cannot be climbed under a trace

_Found 2026-08-24. This is what all of the above is a symptom of._

`yggdrax/_interactions_impl.py` sizes the traversal wavefront by trying a ladder of
capacities and retrying whenever the walk reports an overflow. Reading that report
needs a **concrete** flag. Under an outer trace -- `jax.jit`, and therefore
`shard_map` -- the flag is a tracer, so the retry site took this shape:

```python
overflow_queue = result.queue_overflow
if isinstance(overflow_queue, jax_core.Tracer):
    success = True          # declares success without ever checking
    break
if bool(overflow_queue):
    ...                     # the real retry, unreachable under trace
    continue
```

The ladder therefore ran its **first rung and left**, whatever happened. The first
rung is `max(1024, process_block * 16)`.

So on every traced path the wavefront was sized from `process_block` -- a
vectorisation width -- and `max_pair_queue` was never reached at all. That single
fact explains every observation in this note, and each of them is a signature of it
rather than an independent puzzle:

- `process_block=64`, the shipped distributed default, gives a **1024-pair
  wavefront**. 1024 is what caps a device at ~64 leaves.
- **Raising `max_pair_queue` to 16 777 216 did nothing** (observation 2 above)
  because it is only the ladder's *upper bound*, and the ladder was never climbed.
  That measurement reads as inexplicable above; it is the cleanest confirmation
  here.
- 128 leaves failing at `process_block` 64 **and** 128 but working at 256
  (observation 6) is exactly `1024 < 2048 < needed <= 4096`.
- **The single-device lane was immune** because `prepare_state` runs concretely on
  the host, so the flag is a real bool and the ladder works. That is the whole
  difference between the two lanes, and it is why this never showed up on one card.

Confirmed directly, traced against the eager ladder's answer for the same tree
(thin disc, dehnen MAC, theta=0.4). `process_block` 32 and 64 agreeing **exactly**
is the signature -- both floor to 1024, and nothing but a capacity derived from
`process_block` produces it:

| leaves | `process_block` | traced near pairs | eager | ratio |
| --- | --- | --- | --- | --- |
| 256 | 32 | 204 | 43 854 | 0.0047 |
| 256 | 64 | 204 | 43 854 | 0.0047 |
| 256 | 128 | 646 | 43 854 | 0.015 |
| 256 | 256 | 4026 | 43 854 | 0.092 |
| 256 | 512 | 43 854 | 43 854 | 1.0 |

and flat across four decades of `max_pair_queue` (4096 ... 16 777 216): 204 at
every one of them.

The fix is yggdrax `#52`: `_walk_inputs_are_traced` names the predicate, and with
one attempt available the traced path spends it on the largest capacity the caller
authorised, clamped to `n * (n + 1) / 2` -- the number of distinct node pairs the
tree can produce, so above the bound the caller's number stands and below it the
capacity removed is capacity the tree provably cannot fill. Honouring a capacity is
not free: the wavefront loop evaluates the full capacity-length array every round,
so per-round cost is linear in the capacity whether or not the pairs are live.
Eager behaviour is untouched.

On the jaccpot side the caps stopped being fixed constants and are now derived from
per-device N (`_derive_walk_caps`, `DERIVED_CAP_FIELDS`), with the wavefront scaling
as `2.83 * num_leaves ** 1.5` -- measured, and constant to three figures over a 16x
range in leaves. `process_block` is a vectorisation width again, and the mesh's
default moved 64 -> 256. The reasoning is kept where the constants are, in
`jaccpot/distributed/fmm.py`.

## After the fix: measured on main

`bench/results/distributed_ceiling/`, harness `bench/distributed_ceiling_sweep.py`
and `bench/distributed_ceiling_ladder.py`. A100-PCIE-40GB, jax 0.10.2, x64, leaf
256, theta=0.4, order 3, dehnen MAC. `rel_l2` is against a 512-target direct-sum
probe; validity is the overflow flags, checked **before** any wall clock is quoted.

| ndev | per-device N | total N | valid | rel_l2 | median s |
| --- | --- | --- | --- | --- | --- |
| 2 | 32 768 | 65 536 | yes | 1.6e-4 | 0.136 |
| 2 | 131 072 | 262 144 | yes | 4.1e-4 | 0.200 |
| 2 | 524 288 | 1 048 576 | yes | 1.3e-3 | 1.324 |
| 2 | **1 048 576** | **2 097 152** | yes | 1.8e-3 | 4.583 |
| 4 | 32 768 | 131 072 | yes | 9.7e-5 | 0.141 |
| 4 | 131 072 | 524 288 | yes | 7.6e-4 | 0.475 |
| 4 | 1 048 576 | 4 194 304 | **no** | -- | skipped |

Per-device N is **1 048 576**, against the 8000 this note set out to correct: 128x,
and 4096 leaves per device against the 64 that named the ceiling. The two largest
points need `m2l_chunk=65536` and `nearfield_chunk=512`, which bound the far-field
and near-field peak memory rather than the pair count.

The last row is the wall that replaced this one, and it is recorded as a failure on
purpose: at 4 devices and 1 048 576/device the **cross** near list overflows on
every device. The harness set `timing_skipped: "overflow: cross_near_overflow"` and
refused to report a time, which is the discipline this note argued for -- its
`rel_l2` of 0.61 is what a truncated force looks like when someone does read it.

## Follow-ups

- ~~**`process_block` should scale with leaves per device, or the walk should report
  a required size rather than truncating.**~~ **Done, and better than asked**:
  `process_block` is no longer a capacity at all, and the wavefront is derived from
  the leaf count (`2.83 * num_leaves ** 1.5`). Truncating silently, with only a
  diagnostic flag, is what made this cost a day to find.
- ~~`DistributedFMMConfig.leaf_size = 8` is a poor default.~~ **Done**: the default
  is now 64, matching the single-device presets.
- ~~The traversal caps are fixed constants with no particle-count dependence, unlike
  the single-device path.~~ **Done**: `DERIVED_CAP_FIELDS` sizes them from
  per-device N, and the two lanes' rules corroborate each other -- the single-GPU
  `_sub_million_minimum_memory_pair_queue` floor is this curve evaluated at its own
  calibration point.
- ~~`theta_cross` defaults to 0.1, strict enough that nearly all cross-domain work
  lands in the near field.~~ **Superseded**: there is no `theta_cross` any more, and
  the cross walk runs at one theta with accepted pairs routed through M2L.
- **Open: the cross near field is the next ceiling.** It is what fails at 4 devices
  and 1 048 576/device in the table above, and unlike the self caps it is not
  bounded by anything the local tree knows.

## Where this is used

- `docs/phase5_multigpu_pallas_foldin_plan.md` -- the claims this note corrects.
  Its strong-scaling paragraph is marked superseded and points here.
- `jaccpot/distributed/fmm.py` -- the constants, with their measurements.
- `bench/distributed_ceiling_sweep.py`, `bench/distributed_ceiling_ladder.py` --
  the harnesses, and `--sweep occupancy` reports the converged capacity directly.

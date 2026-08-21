# The distributed per-device ceiling is a leaf-count limit, not a density limit

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

## What was reached after the fix

At `leaf_size=256`, `process_block=256`, pair queues 262144:

- **32 768 particles/device** valid at `leaf_size=512` (64 leaves).
- **16 384 particles/device** valid at `leaf_size=128` once `process_block=256`.
- Weak scaling ndev 2..5 at 16 384/device: all four points valid, 0-1 retries,
  throughput 5.79e5 -> 9.11e5 particles/s, `self_near_pairs` ~4000 throughout.
- 81 920 particles at 5 devices in 89.9 ms, against the log's 40 000 in 64.3 ms.

## Follow-ups

- **`process_block` should scale with leaves per device, or the walk should
  report a required size rather than truncating.** Truncating silently, with only
  a diagnostic flag, is what made this cost a day to find.
- `DistributedFMMConfig.leaf_size = 8` is a poor default: it is the smallest leaf
  in the codebase, and the single-device presets use 64 with production large-N
  runs at 128-1024. It makes the near field quadratically more expensive in
  buffer terms and puts the default configuration below the documented operating
  point.
- The traversal caps are fixed constants with no particle-count dependence,
  unlike the single-device path, which sizes them from N
  (`_minimum_memory_streamed_gpu_traversal_ceiling`). Worth aligning.
- `theta_cross` defaults to 0.1, strict enough that nearly all cross-domain work
  lands in the near field. That is the likely reason cross-domain work dominates
  strong scaling (see `bench/results/multigpu/strong_scaling.json`: self work
  falls 6.3x from 2 to 5 devices while cross work falls only 1.5x).

# Where the distributed force evaluation actually spends its time

_Measured 2026-08-21, 4-5x A100-PCIE-40GB, quiet cards, jax 0.10.2, real basis +
dehnen MAC, float64, `make_force_evaluator` build-once, leaf 256,
`process_block` 256. Medians of 3 repeats; see the note on replicates below._

## Result: about two thirds of a force evaluation is fixed per-call cost

| per-device N | leaves | pairs of work per device | median |
| --- | --- | --- | --- |
| 1 024 | 4 | **60** | **66.4 ms** |
| 4 096 | 16 | 1 008 | 61.8 ms |
| 16 384 | 64 | 16 320 | 97.1 ms |

Sixty pairs of interaction work take 66 ms. There is a **fixed floor of roughly
62-66 ms per evaluation at 4 devices**, independent of problem size. At the
production configuration (16 384/device) that is about two thirds of the
runtime; below it, effectively all of it.

Fitting `time = fixed + k * work` over a wider per-device sweep
(16 384 / 32 768 / 65 536 at 97.3 / 170.5 / 327.5 ms) gives the same picture:
a fixed term near 80 ms and about 17 ms of work-proportional cost at
16 384/device.

## This single fact explains the scaling behaviour

- **Strong scaling is flat** (79-84 ms across 2-5 devices at N=65 536) because a
  fixed cost cannot be divided by adding devices.
- **Weak scaling tops out near 70% efficiency** because the fixed term itself
  grows with device count (59 ms at 2 devices to 86 ms at 5).
- **More work per device improves throughput** (6.7e5 -> 8.0e5 particles/s from
  16 384 to 65 536 per device) because it dilutes the floor. That is a workaround,
  not a fix.

## Three hypotheses that were wrong

Recorded because reasoning about where the cost *ought* to be has a poor track
record on this path, and each of these cost a measurement to kill.

**1. `theta_cross` is mis-tuned.** It is not. The default 0.1 is the fastest
setting measured, despite being 4x stricter than `theta` and leaving the
cross-domain far field *identically empty* (`cross_far = 0` at every device
count, so every cross-domain interaction is direct-summed).

| `theta_cross` | median | cross_near | cross_far |
| --- | --- | --- | --- |
| 0.1 | **98.3 ms** | 12 288 | 0 |
| 0.2 | 116.4 ms | 10 674 | 1 504 |
| 0.3 | 118.1 ms | 5 684 | 4 331 |
| 0.4 | 117.2 ms | 2 836 | 4 109 |

Relaxing it works *mechanically* -- the cross far field activates and near work
drops 4.3x -- and still loses time. Removing ~9 450 cross leaf-pairs (each 256x256,
so ~6.2e8 particle interactions) while adding ~4 100 M2L operations costs 19 ms
**more**. With a 60-pair problem already taking 66 ms, the reason is now obvious:
near-field work is a minor term, so reducing it cannot help, while any added path
costs real time.

**2. The fused M2L Pallas kernel simply was not enabled.** Toggling
`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` changes nothing: 100.1 vs 103.2 ms at
`theta_cross` 0.1, and 116.1 vs 122.0 ms at 0.4. Within noise, if anything worse.

**3. The near field is padding-bound.** It is not. Varying
`max_neighbors_per_leaf` over 128 / 512 / 2048 -- a 16x range -- at *identical*
pair counts gives 102.8 / 97.6 / 98.4 ms. Flat.

## What the floor most plausibly is, and why we cannot yet say

`docs/differentiable_fmm_distributed_audit.md` already names the suspect: the
forward is "dominated by fixed-size traversal buffers and a per-call topology
rebuild rather than by N", and the distributed body builds its tree *inside*
`shard_map` on every call, unlike the single-device path where `prepare_state` is
built once and reused.

That is consistent with everything above, but **it is not isolated**, and it
cannot be with the present instrumentation: the distributed path exposes no
per-stage timers at all (see `bench/multigpu/comm_overhead.py`). So the 64 ms is
measured; its attribution is inferred.

This raises the value of that instrumentation considerably. It was previously
wanted for a figure; it is actually needed to find the dominant cost in the
code.

## Ranked next steps

1. **Instrument the sharded region.** Without it, the largest single cost in the
   evaluation cannot be attributed. A `jax.profiler` trace attributed to stages
   is the least invasive route.
2. **Hoist the topology build out of the per-call path.** If the floor is the
   rebuild, this is the whole game: an N-body integration at fixed topology pays
   it every step for nothing. It is architectural -- the audit notes there is
   "nothing to hoist here" as the code stands -- so it needs a
   `prepare_distributed_state` / `evaluate` split analogous to single-device.
3. **Raise per-device N in the meantime.** It works, for the honest reason that
   it dilutes the floor. 32 768/device runs clean at leaf 512, and 65 536/device
   at leaf 256 with `process_block` 1024.
4. **Do not touch `theta_cross`, the fused-M2L flag, or the near-field caps** for
   performance. All three are measured no-ops or losses.

## Method notes

- **Timings need replicate invocations on this host.** Within-run repeats span
  0-34%; between-invocation spread reached 2.5x on one contended run. Three
  independent invocations of the scaling sweeps agreed to 1.1-12.5%. The numbers
  in this document are single invocations of 3 repeats and should be treated as
  order-of-magnitude for the *differences* being tested, which are large; the
  scaling figures themselves are replicated.
- **`bench/multigpu/harness.py`'s `uniform` distribution samples a fixed
  [-1,1] cube, so raising per-device N also raises density.** The per-device
  sweep above is therefore confounded: pair counts grew quadratically (4 026 ->
  15 347 -> 53 214) because the problem got denser as well as bigger. The floor
  measurement is unaffected -- it compares against a 60-pair problem -- but a
  clean "does more work per device help" answer needs the box to scale as
  N^(1/3). That is a small harness fix and is not yet done.

# H100 fast-lane investigation — launch-latency-bound (2026-07-10)

Context: on H100 the device-only fused lane runs ~1122 ms/step at 200k (leaf 256,
order 4, fp32, theta 0.6) vs the doc's ~119 ms A100. GPU verified healthy (408
TFLOP/s fp32 matmul, idle). `guard_large_n_radix_fast_lane.py` FAILS its own
speedup assertion here: steady_eval 1.05x (baseline 340 ms, fast 323 ms; wants >=2x),
full 0.68x (baseline 674 ms, fast 985 ms — fast is SLOWER). jax/jaxlib 0.9.0.

## Root cause: launch-latency-bound, not compute-bound
nsys (`--trace=cuda`) kernel summary of the fused step:
- `input_dynamic_slice_reduce_fusion`: 41%, ~852 instances/step, ~1.7 us each
- `loop_add_fusion`: 30%, ~950 instances/step, ~1.1 us each
- Total GPU-busy ~75 ms across a 20-step run => GPU ~99% IDLE. The ~1000 ms/step
  is the gaps between ~1,800 sequential tiny-kernel launches per step.

=> an unfused ~852-iteration loop lowered by XLA to one tiny kernel per iteration.
On the slow 2080 the per-kernel compute hid the launch gaps (88% SM, compute-bound);
on the fast H100 the gaps dominate (Amdahl) => the ~10x regression and the fast
lane showing no speedup.

## Ruled out (swept, no effect on per-step time)
- `nearfield_edge_chunk_size` 64 -> 512 -> 4096: flat (~1.55-1.63 s)
- `m2l_chunk_size` default -> 16384 -> 131072 (one-shot): flat (~1.55-1.75 s);
  recent_dual_m2l_chunk_size confirmed = requested, far_pairs pinned at cap 131072.
So the 852-loop is NOT the near-field edge scan nor the M2L accumulation scan.

## LOCALIZED: the ~852-loop is in EVAL + NEAR-FIELD (not M2L/downward)
nsys on the stage-truncated diag modes (`JACCPOT_STRICT_REFRESH_DIAG_MODE`):
- `upward_only` (tree+upward): NO `input_dynamic_slice_reduce_fusion`; loop_add only 141.
- `downward_only` (+M2L+L2L): still NO `input_dynamic_slice_reduce_fusion`; loop_add 1107,
  input_add_gather_reduce 250, input_gather_reduce_select 125 (M2L/L2L work -- modest).
- `full` (+eval+near): `input_dynamic_slice_reduce_fusion` jumps to ~852/step (41%).
=> the launch-bound tiny-kernel scan is the NEAR-FIELD P2P / L2P (eval stage).

`nearfield_edge_chunk_size` had no effect because the fused lane's near-field runs
through the radix fast-payload / static-target-block path
(`JACCPOT_LARGE_N_STATIC_TARGET_BLOCKS_MAX_PER_LEAF`), not that scan. And
`use_pallas=True` did NOT change timing (1122 vs 1134 ms) => the existing
register-tiled near-field Pallas kernel (`jaccpot/pallas/nearfield_fused_leaf.py`,
sm_80-gated) is NOT engaging in the fused lane. That is the gap to close.

## Phase 5 reframe (REDIRECT)
On Hopper the bottleneck is the NEAR-FIELD launch-bound tiny-kernel scan, not M2L.
=> Prioritize target #2 (near-field Pallas) over the M2L kernel: wire the existing
`nearfield_fused_leaf` kernel into the fused radix near-field path so the ~852 tiny
launches collapse to a few big register-tiled kernels. The M2L Pallas kernel (PR #28)
stays valuable for the FLOP-bound regime but is not the H100 launch-bound culprit;
keep it, pursue near-field first.

## Config knobs do NOT fix it (all swept, flat) => structural
- nearfield_edge_chunk_size 64->4096: flat
- m2l_chunk_size default->131072: flat
- target-block: TILE_SIZE 4->16, TILE_SCAN_UNROLL 1->4, BATCH_SCAN_UNROLL 1->4,
  TARGET_LEAF_BATCH_SIZE 16->64: 1639->1609 ms (flat)
- run-to-run variance is high on an "idle" GPU (1122-1639 ms on the same GPU 2),
  consistent with a host-scheduling / launch-latency-bound regime.
Conclusion: the ~852 tiny near-field kernels are structural to the target-block
near-field path (`compute_leaf_p2p_accelerations_target_block_pairs_only` in
`_large_n_nearfield.py`), which is a jnp scan and does NOT take `use_pallas`.

## Fix requires architectural change, not config
Route the fused-lane target-block near-field through the register-tiled Pallas
kernel (`nearfield_fused_leaf`) so the many tiny launches collapse to a few big
ones -- OR restructure the target-block accumulation to emit far fewer kernels.
This is production-runtime work in `_large_n_nearfield.py` / `near_field.py`.
External signal: a Bonsai (hand-tuned CUDA) comparison is "way faster", consistent
with this being a GPU-idle / launch-latency problem rather than a compute one.

## use_pallas=True is SLOWER (not a simple toggle fix)
Enabling use_pallas in the fused lane: ~2858 ms/step vs ~1122 ms baseline. The
radix_fast (path #1) pallas branch runs IN ADDITION to the target-block jnp scan
(path #2), not instead of it => extra work, slower. So the fix is NOT "flip
use_pallas"; it is to make ONE fused Pallas near-field the SOLE near-field path.

## Real fix (architectural): single-launch near-field
`nearfield_leafpair_pallas` already works on H100 (microbench: parity OK; ~180 ms
for L=782, W=256, S=2048 as a few big kernels). The fused near-field (eval+near)
is ~461 ms of the step. Plan: derive a per-target-leaf source-leaf layout
[L, S] (+valid mask) from the fused neighbor payload and run the whole near-field
as ONE `nearfield_leafpair_pallas` call, replacing the target-block scan. This
collapses the ~852 tiny launches to one kernel. Building block is proven; the work
is the payload->[L,S] mapping + making it the sole path + handling the source
overflow (leaves with > S neighbors) the target-block split currently handles.

## RELIABILITY NOTE (important): timings were on SHARED GPUs
All 4 H100s had resident processes during measurement (util 0/89/95/100%), so
WALL-CLOCK numbers above (1122/1638/2858 ms, config sweeps, "use_pallas slower")
are NOT reliable and should be treated as indicative only. Use nsys KERNEL COUNTS
(contention-independent) as the metric instead:
- baseline vs use_pallas=True: `input_dynamic_slice_reduce_fusion` = 19,600 in BOTH,
  `loop_add_fusion` = 21,938 in BOTH => use_pallas is a true structural NO-OP here
  (not slower; the 2858 ms was contention). Retract the "use_pallas slower" note.
- top-2 kernels total ~59 ms GPU-busy over a 23-step run (~2.6 ms/step) => the
  GPU-idle / launch-bound finding holds regardless of contention.
SUCCESS METRIC for the fix = REDUCE the ~852/step tiny-kernel count (collapse to a
few big kernels). Final wall-clock speedup must be validated on an EXCLUSIVE GPU
(none free right now; use a quiet GPU / MIG slice / off-hours).

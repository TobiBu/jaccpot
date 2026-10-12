# Ragged exchange on jax 0.11.2: which XLA implementation (2026-10-12)

`bench/multigpu_exchange_bench.py`, one process per mode, two interleaved rounds
(`<mode>_r1.json`, `<mode>_r2.json`). Each cell is the min over 15 calls after 3 warm-ups,
shown as the range over the two rounds, in ms. Every device sends `rows` rows to the other
device. The receive buffer is 2x (or 8x) the live rows. Every ragged row was checked
`correct` in every mode.

- **Cards.** GPUs 3+5 of compgpu9: A100-PCIE-40GB, across CPU sockets (SYS), peer access
  on. GPU 3 carried another user's ~3 GB job, so the absolute numbers are indicative; the
  ratios between modes, which match the 2026-10-02 measurement on jax 0.10.2 (4-6x), are
  the result.
- **Software.** jax 0.11.2, NCCL 2.32.3, jaccpot e25633d, yggdrax 1f0e36f.

| mode | `XLA_FLAGS` |
| --- | --- |
| ncclfix | `jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS` (NCCL send/recv, barrier off) |
| decomposer | `--xla_gpu_unsupported_enable_ragged_all_to_all_decomposer=true` |
| oneshot | none (XLA's default one-shot kernel; the barrier flag at its 0.11 default, on) |
| oneshot_nobar | `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` |
| devkernel | `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true` |

| rows x width (dtype, recv cap) | ncclfix | decomposer | oneshot | oneshot_nobar | devkernel |
| --- | --- | --- | --- | --- | --- |
| 262,144 x 2 (int32, x2) | 0.64-0.76 | 0.51-0.65 | 0.70-0.77 | 0.72-0.81 | 0.67-0.77 |
| 1,048,576 x 2 (int32, x2) | 1.03-1.14 | 1.02-1.14 | 4.20-4.23 | 2.61-2.69 | 4.31-4.36 |
| 1,048,576 x 2 (int32, x8) | 1.11-1.23 | 1.18-1.27 | 4.39-4.60 | 4.25-4.31 | 4.23-4.26 |
| 4,194,304 x 2 (int32, x2) | 2.98-3.04 | 3.63-3.72 | 15.68-15.70 | 15.41-15.81 | 15.59-15.63 |
| 16,384 x 56 (float32, x2) | 1.02-1.09 | 0.87-0.92 | 1.60 | 1.70-1.77 | 1.50-1.72 |
| 65,536 x 56 (float32, x2) | 1.56-1.69 | 1.88-1.95 | 10.35-10.37 | 9.38-10.26 | 8.34-10.38 |
| 65,536 x 56 (float32, x8) | 1.61-1.86 | 1.88-1.93 | 7.18-7.34 | 7.02-7.22 | 7.17-7.38 |
| 262,144 x 56 (float32, x2) | 5.25-5.44 | 6.59-6.68 | 27.14-27.42 | 27.05-27.08 | 26.86-27.23 |
| 4,096 x 312 (float32, x2) | 0.82-0.99 | 0.80-0.89 | 1.77-1.92 | 1.28-1.43 | 1.86-1.93 |
| 16,384 x 312 (float32, x2) | 2.09-2.18 | 2.50-2.60 | 9.70-9.72 | 9.60-9.63 | 9.65-9.67 |
| 65,536 x 312 (float32, x2) | 6.37-6.97 | 8.56-8.89 | 37.33-37.55 | 37.03-37.44 | 37.54-37.82 |

**Reading.**
- The NCCL path is the fastest at the cross field's large messages: 3.6-6.6x faster
  than the default from 1M int32 rows, 65,536 rows of 56 and 16,384 rows of 312 up
  (5.4-5.9x at 65,536 x 312).
- The decomposer comes second: 1.0-1.4x NCCL's time on those rows, and slightly faster
  (up to ~1.2x) on the smallest messages.
- The one-shot kernel (barrier on or off) and the device kernel cost about the same at
  large messages. The exception is 1M int32 rows, where the barrier-off one-shot takes
  2.6 ms against 4.2.
- The rows under 2 ms are latency-bound, so the mode barely matters there.

Not timed here: the NCCL path with the barrier at its 0.11 default. It returns no remote
rows and deadlocks (`bench/repro_jax_ragged_all_to_all_nccl_fallback.py`).

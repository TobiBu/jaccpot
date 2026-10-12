# Draft upstream bug report: NCCL ragged-all-to-all with `use_barrier_with_nccl`

> **Not filed yet.** Outbound draft for <https://github.com/openxla/xla/issues>. The
> reproducer is `bench/repro_jax_ragged_all_to_all_nccl_fallback.py` (pure JAX). How it
> was found and what jaccpot changed: `docs/cleanup_2026-10.md`, "The fused two-card
> lane on jax 0.11.2". No matching issue in openxla/xla or jax-ml/jax on 2026-10-12
> (searched: `use_barrier_with_nccl`, "RaggedAllToAll fallback to NCCL",
> `send_recv_fallback`, "ragged_all_to_all nccl fallback hang").

---

**Title:** [XLA:GPU] RaggedAllToAll NCCL send/recv fallback returns no remote rows,
then deadlocks, when `xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl`
is on (the default since jax 0.11)

### Description

With the NCCL send/recv fallback selected
(`--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false
--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true`), a
`jit(shard_map(jax.lax.ragged_all_to_all))` over two GPUs of one process:

1. returns, on its **first** call, only each device's block to itself; every row that
   should have come from the other device still holds the output operand's initial
   value;
2. **never returns** from its second call. Both GPUs sit at 100 % utilization, every
   XLA execute thread is idle and the host thread waits in `BlockUntilReady`.

It happens only while `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl`
is on, and only above a message size: correct at 64 rows of 56 float32 (14 KB per
message), broken from 512 rows (115 KB) up. jax 0.11's XLA turns that flag on by
default (`debug_options_flags.cc` at 91888df6); jax 0.10.2's (5a9e73cb) left it off, so
a setup that worked on 0.10 hangs after the upgrade. Setting the flag to `false` fixes
it at every size. XLA's default one-shot kernel is correct either way.

### Reproducer

`bench/repro_jax_ragged_all_to_all_nccl_fallback.py` (about 120 lines, jax only). Every
device sends `--rows` rows of `--width` float32 to every device, itself included; each
call runs in a thread with a timeout, and the process exits on the first call that
does not return.

```
CUDA_VISIBLE_DEVICES=0,1 python repro.py --mode nccl --rows 16384 --width 56
jax 0.11.2, devices [CudaDevice(id=0), CudaDevice(id=1)]
XLA_FLAGS='--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false --xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true'
rows 16384 x width 56 float32 per message (3.670 MB)
call 0: 32768/65536 rows right, 32768 still fill
call 1: HUNG (no result after 45 s); exiting
```

### Results (10 calls each, width 56)

| jax | flags | 1 and 64 rows | 512, 2048, 8192, 16384 rows |
| --- | --- | --- | --- |
| 0.11.2 | default (one-shot kernel) | | 10/10 (16384) |
| 0.11.2 | NCCL fallback (barrier on: the default) | 10/10 | call 0 wrong (remote half = fill), call 1 hangs |
| 0.11.2 | NCCL fallback + `..._use_barrier_with_nccl=false` | 10/10 | 10/10 |
| 0.10.2 | NCCL fallback (barrier off: the 0.10 default) | | 10/10 (16384) |
| 0.10.2 | NCCL fallback + `..._use_barrier_with_nccl=true` | | call 0 wrong, call 1 hangs (16384) |

A broader scratch harness (16,384 rows of 56 per peer, nothing sent to self) agrees.
With the barrier on, the first call of each process returned fill for the whole remote
block and the next call hung (sizes computed on device or passed in as inputs alike).
With it off, every pattern passed: repeated calls, donated inputs, two executables
alternating, four exchanges in one program, a gradient through the exchange.

What is not the cause:
- **NCCL version.** jax 0.10.2 with NCCL 2.32.3 (the 0.11.2 wheel's, preloaded) is
  correct.
- **Command buffers.** `--xla_gpu_enable_command_buffer=` changes nothing.
- **Topology.** These runs used a cross-socket pair (SYS). The hang was first seen in
  our application on a pair under one PCIe switch (PIX, GPUs 2+3).
- **The send/recv code.** `RunNcclFallbackRaggedAllToAll` is unchanged between the two
  XLA commits apart from macro renames.

### What we observed inside

- With the barrier flag on, the optimized HLO puts the ragged output, and the broadcast
  that initializes it, in memory space `S(1)`. The ROOT is a `copy` out of it:
  ```
  %loop_broadcast_fusion = f32[32768,56]{1,0:S(1)} fusion()
  %ragged-all-to-all.3 = f32[32768,56]{1,0:S(1)} ragged-all-to-all(...), ..."is_sync":true...
  ROOT %copy.4 = f32[32768,56]{1,0} copy(%ragged-all-to-all.3)
  ```
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,P2P` shows matching Send/Recv counts on both
  ranks (917,504 elements to the peer, 0 to self). The receive buffers are at
  `0x4028...`/`0x4030...`, inside NCCL's symmetric VA window (`Symmetric VA size=40GB`,
  `cuMemEnable 1`). The same log with the flag off shows ordinary allocator addresses.
- The thunk logs `lsa_size: 2` (`== num_devices`). By `RunCollective`'s dispatch, the
  one-shot kernel then runs `RunOneShotRaggedAllToAllWithNccl` on the symmetric
  output, and that is correct (inferred from the source, not logged). Only the
  send/recv fallback loses the data when it runs with the output in symmetric memory.

A guess, for whoever owns the thunk: the fallback's receives land in the `S(1)`
allocation in a way the following `copy` cannot see. Small messages go through
NCCL's internal buffers and work, consistent with that. The fallback should either
work with a symmetric output or be refused, as the
fallback-not-allowed check already refuses other unsupported cases.

### Environment

- 2x NVIDIA A100-PCIE-40GB in one process (`CUDA_VISIBLE_DEVICES=3,5`; peer access
  OK); driver 595.71.05 (CUDA 13.2).
- jax / jaxlib / jax-cuda12-plugin 0.11.2 (XLA 91888df6), NCCL 2.32.3; and
  jax 0.10.2 (XLA 5a9e73cb), NCCL 2.31.2.
- Python 3.12.3, Linux 6.8.

### Workaround

`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false` with the NCCL
fallback, or the default one-shot kernel. On these PCIe A100s, the one-shot kernel takes
4-6x the NCCL path's time at our message sizes (65,536 rows of 312 float32: 37.5 vs
6.4-7.0 ms). That is why we use the fallback.

# Small leaves on the fused single-GPU lane (2026-09-07 .. 2026-09-10)

Plan: `~/.claude/plans/ok-please-draft-a-effervescent-crown.md` (Odisseo box). Branch
`perf/small-leaves-p1`. Harness: `Odisseo-bench-multigpu/benchmark_multigpu/codes/{fit_smallleaf_caps,
smallleaf_baseline,smallleaf_dynamics}.py`, `FAST_LANE_ENV_BY_LEAF` in `compare_force.py`; results under
`artifacts/smallleaf/`.

## Why

At N=200k Plummer, p=4, theta 0.6 the fused lane at leaf 256 sums 58 % of N directly per target
(pkdgrav3: 0.24 %). The direct share falls as ~W^0.9 with the leaf size, so smaller leaves are the
only route to pkdgrav3's arithmetic -- and before this work leaf 64 was 3.1x SLOWER per step than
leaf 256 with 3x fewer pair evaluations.

## What the attributed baseline said

`strict_run_v2` per step, N=200k, theta 0.6, idle A100 (`baseline_base_*.json`):

| leaf | step ms | eval-only ms | leaf-pair kernel | downward (M2L+L2L) | launches/step |
|---|---|---|---|---|---|
| 256 | 131.8 | 70.0 | 79 | 41 | 7.8k |
| 128 | 187.5 | 43.2 | 48 | 130 | 15.8k |
| 64 | 409.6 | 26.5 | 22 | 372 | 32k |
| 32 | 1168 | 19.6 | 13 | ~1100 | -- |

The Pallas leaf-pair kernel has **no small-W cliff** (its time is exactly the direct-volume ratio),
the self-leaf scan was already negligible after #334, and the whole penalty was the far field:
one XLA scatter fusion launched once per 4096-pair M2L chunk at 686 us (`_chunk_segment_scatter_add`:
`segment_sum` with hundreds of duplicates per address plus ~3800 zero-adds onto node 0 per chunk,
both serialised by the atomic-add lowering) = 169 ms of the 410 ms leaf-64 step, and behind it a
21k-launch/step storm from the chunked rotation cascade.

## What was built

1. **Self-leaf block folded into the leaf-pair Pallas kernel** (`nearfield_fused_leaf.py`,
   `include_self`; flag `JACCPOT_NEARFIELD_LEAFPAIR_FOLD_SELF`, default on, forward prepacked lane
   only). Time-neutral (the scan was already batched), removes the per-leaf launch family.
2. **Contention-free chunk reduction** (`_m2l.py::_chunk_segment_scatter_add`): segmented
   `associative_scan` + out-of-bounds sink (`mode="drop"`, unique in-bounds indices). Leaf 64:
   410 -> 237 ms/step; leaf 128: 188 -> 145; leaf 32: 1168 -> 649.
3. **Target-tiled CSR M2L Pallas kernel** (`jaccpot/pallas/m2l_real_csr.py`, flag
   `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=1`): one program per target owns its local row; rotations
   assembled on chip from the two alignment angles in the centred (degree, m) layout, degree-only
   z-core, no per-pair operand in HBM. Microbench on an idle A100 (`m2l_csr_microbench.json`):

   | pairs | p | CSR ns/pair | pure-JAX | degree-batched |
   |---|---|---|---|---|
   | 1M | 4 | 19.8 | 69.4 | 55.6 |
   | 6M | 4 | 12.1 | 68.9 | 55.3 |
   | 1M | 6 | 12.4 | 126.1 | 86.2 |
   | 6M | 6 | 12.0 | 124.9 | 85.6 |

   Parity 1e-6 (fp32) against `m2l_rot_scale_real_batch`; interpret parity < 1e-10 (fp64) for orders
   2-6. In the step at leaf 64 the kernel costs 12.9 ms where the pure-JAX lane cost ~360.

## Where it landed (per step, theta 0.6, N=200k, idle A100, foreign-process-free rows only)

| leaf | baseline | fold + scatter | + CSR M2L | eval-only | aggL2 vs fp64 direct |
|---|---|---|---|---|---|
| 256 | 131.8 | 126.4 | **120.3** | 69 | 8.27e-4 |
| 128 | 187.5 | 145.2 | **122.3** | 42 | 1.15e-3 |
| 64 | 409.6 | 236.5 | **176.7** | 24 | 1.20e-3 |
| 32 | 1168 | 649.0 | **509.4** | 15 | 1.30e-3 |

Theta 0.8 with CSR: leaf 64 113.5 (baseline 192.0), leaf 32 364.1 (baseline OOM'd on a shared card).

**The far field is solved; the per-step leaf optimum did not move.** With M2L at 13 ms, the leaf-64
step is ~100 ms of traced dual-tree walk and list compaction (pair-queue-sized scatter fusions at
~800 us x ~55 launches, ~200 memcpys, sorts) that scale with the leaf count and the traced
`max_pair_queue` (1M at leaf 64, 2M at leaf 32). Eval-only (force at fixed lists) is 3-5x faster at
leaf 32-64 than at 256, so a workload that refreshes lists every k steps, or a walk that costs
O(leaves) rather than O(queue), would collect the gain. Gate G3 (<= 35 ms per step at 200k) is
missed; G1, G2 and G2a are met. Next lever: the traced walk's queue buffers (yggdrax side).

## Capacity rules for small leaves (fitted, `FAST_LANE_ENV_BY_LEAF`)

* compact far-pair cap: pow2 >= 1.5x the far-pair count (101k / 363k / 992k / 2.34M at 256/128/64/32).
* neighbour-EDGE cap: the traced refresh pads to `num_leaves x traced_neighbour_cap`, and the #333
  carry-over sets that cap to pow2(1.5 x longest eager row + 1) with the longest row =
  `num_leaves - 1` -> 2^21 already fails at leaf 128 (2^23), 2^25 at 64, 2^27 at 32.
* `max_neighbors_per_leaf` must be explicit above the 2048 clamp; `max_interactions_per_node`
  needs 16384 at leaf 32 (and at leaf 64 for theta 0.4).

## Traps recorded

* The perfetto trace of a leaf-32 step hung for 44 h after the timing (use `--no-trace` there).
* Back-to-back configs on one card get rejected by the guard on the previous process's stale
  utilisation reading; the queue scripts wait for 0 % and retry.
* `JACCPOT_M2L_DEGREE_BATCHED` is read at trace time; a microbench that flips it must
  `jax.clear_caches()` or the second variant reuses the first trace (bit-identical output).
* The `l2l_only` detail diag mode is inconsistent with the cumulative modes -- do not attribute from it.

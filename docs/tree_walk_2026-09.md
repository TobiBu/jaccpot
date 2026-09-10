# The per-step tree walk on the fused single-GPU lane (2026-09-10)

Plan `~/.claude/plans/ok-then-please-make-replicated-rain.md` (Odisseo box). Follows
`small_leaves_2026-09.md`, which ended with the far field solved (CSR M2L kernel) and the traced dual-tree
walk as the remaining wall: ~100 ms of a 177 ms step at leaf 64. yggdrax PR #74 carries the walk side
(`docs/traversal_walk_cost.md` there has the isolated-walk table); this note is the jaccpot side.

## What the walk cost and why

yggdrax's `_dual_tree_walk_impl`, one call per step at the full traced queue, spent its time on its OUTPUT
layout, not on the traversal: dense per-node rows (`total_nodes x max_interactions_per_node`, 820 MB at
leaf 64) carried through the `while_loop` and copied in and out by `lax.cond` identity branches every
round, a full-queue argsort per round to place each pair in its rows, four `segment_sum`s per round, and a
post-loop flatten over `nodes x K` slots. Isolated on one tree (A100, N=200k, theta 0.6) it took 503 ms per
walk at leaf 64 and 1291 ms at leaf 32. yggdrax's flat-emission `dual_tree_walk_mutual` -- same wavefront,
each unordered pair appended once to a flat list with a cumsum -- produces the same far and near pairs as
SETS in 13-40 ms. The scatter-lowering promise (`unique_indices`) is not a lever (0.10 ms per 1M scatter
either way); the dense rows, the conditionals and the sorts are.

## The lane

`JACCPOT_STATIC_STRICT_FUSED_FLAT_WALK=1` routes `_build_dual_tree_artifacts_split_strict_streamed` to
`_build_flat_walk_artifacts_strict_streamed` (`runtime/_interaction_cache.py`), which calls
`dual_tree_walk_mutual` with the dual walk's own `mac_extents` (`_build_mac_extents(...)[0]`) and
`mac_type`, then:

* un-mutualises the far pairs INTERLEAVED (`[b->a, a->b]` per canonical pair) so the live pairs stay a
  prefix -- every M2L consumer masks `idx < far_pair_count`, and a concatenation would silently drop the
  second direction; capacity is the compact far-pair cap;
* builds the leaf neighbour CSR from the directed near pairs with one stable argsort by target leaf and
  `searchsorted` offsets; width is the neighbour-edge cap, so eager and traced carries match without a pad;
* eager: a queue ladder on `queue_overflow`, far/near overflow raise naming the cap (never widened);
  traced: any overflow saturates `far_pair_count` to the capacity, which trips `strict_run_v2`'s existing
  saturation guard -- a truncated refresh is fatal, not silent;
* always emits the capacity report (the treecode graft's early return left the guard dark) with
  `peak_wavefront`; `_strict_fused_capacity_handoff` sizes the traced queue as pow2(1.5 x peak). In the real
  fused geometry the leaf-64 peak is 466,626 pairs over 24 rounds, so the traced queue is 2^20.

Supported: `mac_type` bh/dehnen, `pair_policy=None`. Both walk flags set -> refused. Indices: the harness sets
`YGGDRAX_INDEX_PRECISION=int32` and `JACCPOT_INDEX_PRECISION=int32` for the fused lane (read at import; set
both, yggdrax falls back to jaccpot's variable but not the reverse).

## Per step (`strict_run_v2`, N=200k Plummer, p=4, idle A100, no foreign process; flat walk + CSR M2L + int32)

| leaf | theta | small-leaves start | + CSR M2L | **+ flat walk** | eval-only | near kernel | M2L kernel | upward | downward (M2L + walk + L2L) | aggL2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 0.6 | 131.8 | 120.3 | **96.2** | 68.6 | 79.9 | 1.7 | 5.4 | 8.4 | 8.27e-4 |
| 128 | 0.6 | 187.5 | 122.3 | **71.3** | 42.7 | 48.5 | 5.1 | 8.5 | 12.1 | 1.15e-3 |
| 64 | 0.6 | 409.6 | 176.7 | **63.3** | 23.6 | 26.5 | 13.0 | 10.5 | 25.4 | 1.20e-3 |
| 32 | 0.6 | 1168 | 509.4 | **82.5** | 16.2 | -- | -- | 12.9 | 53.7 | 1.30e-3 |
| 64 | 0.8 | 192.0 | 113.5 | **41.6** | 21.3 | 13.0 | 7.1 | 10.8 | 16.9 | 5.33e-3 |
| 32 | 0.8 | -- | 364.1 | **53.6** | 12.8 | -- | -- | 11.0 | 33.5 | 1.44e-2 |

Launches per step 3.8-4.6k (from 7-32k). Forces identical to the dual-walk lane to 4 digits at every leaf
(the lists are the same sets; only the fp32 order changes); the #333 truncation check passes with the flag
(`tests/integration/test_strict_run_v2_refresh_capacity.py` is parametrised over it).

**The per-step optimum moved to leaf 64: 63.3 ms against 96.2 at leaf 256 with the same lanes and 120 before
them -- 1.9x per step at 200k, 6.5x against the leaf-64 step the small-leaves work started from.** At leaf 32
the walk is again the largest item (downward 53.7 with M2L ~25): the wavefront's long thin tail (46-61 rounds
at ~0.15 ms each) and a 2^21 queue; a narrow-width branch for the tail rounds (plan Tier 3) is the next lever
there, not at leaf 64.

## Traps recorded

* `tests/conftest.py` used to put the sibling `/export/home/tbuck/yggdrax` checkout (a paper branch) at the
  front of `sys.path` unconditionally: every local jaccpot pytest run exercised THAT yggdrax. It now honours
  `YGGDRAX_WORKTREE`; the bench harness's `sitecustomize` repoints the yggdrax editable finder the same way.
* `autocvd -l -o -q` can return an empty device list when every card is busy; the job then runs on CPU and
  GPU-only tests skip silently. Check the device line in the log.
* The strict fused prepared-eval seam exists only above the large-N threshold on a GPU, so the lane's wiring
  test runs in the GPU suite at N=70k.

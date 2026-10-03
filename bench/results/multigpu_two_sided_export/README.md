# Two-sided export walk: result rows (2026-10-03)

Record: `docs/multigpu_fused_2026-09.md`, "The two-sided export walk". Probe:
`bench/multigpu_c4_cross_force_probe.py`. Two A100s (cards 6+7, a PIX pair), Plummer, leaf 64, theta 0.8, p6,
fp32, min of 15 calls after 3 warm-ups. Each arm ran from its own frozen worktree.

| arm | jaccpot | yggdrax | settings |
| --- | --- | --- | --- |
| A | c4a548c | 678709e | one-sided export (the code before this work) |
| B | 6f8af5d | 16be300 | `JACCPOT_CROSS_TWO_SIDED=1`, 4-leaf cells |
| C | 6f8af5d | 16be300 | B + `PROBE_MAX_LEAVES_PER_CELL=1` |
| D | 17fecdc | 16be300 | the defaults (two-sided, one leaf per cell, receiver caps from live counts, plan fix) |

* `ab_one_vs_two_sided/`: A vs B, the probe's default (old) receiver caps, interleaved, 2 rounds.
  `n8000000_B_r2` timed 202.25 ms; its process was stopped before it wrote the JSON (the driver log has it).
* `ab_receiver_caps/`: A vs B with the receiver caps set from the live counts
  (`PROBE_RECV_NEAR_CSR_BITS`, `PROBE_WALK_QUEUE_BITS`, `PROBE_RECV_FAR_BITS`, `PROBE_RECV_NEAR_BITS`,
  `PROBE_RECV_NODE_BITS`, `PROBE_SEND_NODE_BITS` = 18/18/18/17/15/15 at 2e5 per card, 20/20/20/19/17/17 at 1e6,
  22/22/22/21/19/19 at 4e6). The first four are what the probe's new per-leaf factors give; the probe keeps the
  node caps at their old factor (one power of two above these).
* `ab_cells_vs_leaves/`: B vs C, both with those caps.
* `gate/`: arm D, one card at N (`s<seed>_n<N>_1card`) and two cards at 2N (`..._2card`, local and cross
  arms), seeds 0-2.
* `rollout_gate/`: Gate G2.2 (`bench/multigpu_rollout_gate.py`) on arm D, mesh (cards 6+7) and solo (card 6).

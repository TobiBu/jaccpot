# Fused-memory round 2: result rows (2026-10-04)

Record: `docs/fused_memory_2026-10.md`, section "Round 2". Bench `bench/fused_memory_budget.py`, one
configuration per process.

**Configuration.** Leaf 64 cell leaves, cell_min_level 8, leaf capacity 1.15x the live cells, theta 0.8, p5,
unnamed caps, command buffers on.
- `clip<N>`: clipped Plummer (`--ic plummer_clipped --rmax 20`).
- `p<N>`: seed-0 Plummer.
- `disc25`: the 25M disc+bulge IC.

**Hardware.**
- Cards: A100 40 GB. Card 3 carried an idle 4.2 GiB foreign process; cards 4-7 were empty when the late rows
  ran.
- Allocator: PREALLOCATE off unless the row says `prealloc 0.88` (the ceiling ladders).

**Worktrees** (frozen per arm):

| arm | jaccpot | yggdrax |
| --- | --- | --- |
| base | `jaccpot-r2-base` = 5dd4fa3 (#359 + bench IC/prealloc) | eee5c7a (COM fix) |
| near / com / lanes | 19eca99 / fd5c4f1 / 81d4725 | eee5c7a |
| p4 | 04f4f26 (lanes default, particle carry) | c1617b2 |
| c1b4 | c1b40b3 (COM jitted) | c1617b2 |
| head | c199998 (four-level COM passes: the regression) | 2e8f380 |
| head2 | 43bef3d (one level per pass, sqrt after max) | 2e8f380 |
| head4 | ccc630d (+ bench `--skip-eval`; library code as head2 + the int64 flat-offset guard) | 2e8f380 |

**Directories:**
- `ceiling_base/`: the one-card ladder on the base arm, preallocated 0.88.
- `ab_8m/<step>/`: 8e6 clipped A/B rows, same frozen worktree per comparison, interleaved:
  - `near_sorted`: table vs sorted near layout, bitwise;
  - `near_direct`: table / direct / direct + particle L2P;
  - `comradii`: COM level passes;
  - `lanes`: level vs lane cascades;
  - `p4run`: state vs particle carry;
  - `ab_head2`: c1b4 vs head2;
  - `newcheck`: c1b4 vs head, correctness on card 2.
- `ceiling/`: the one-card ladder on head2 with `JACCPOT_STRICT_CARRY=particles`, preallocated 0.88.
- `minlevel/`: cell_min_level 8 / 7 / 6 at 8e6 clipped (head2, particle carry); `minlevel_seed0/` the same on
  seed-0 Plummer (no effect there).
- `ceiling_prod/`: the production sequence (`--skip-eval`, head4 = ccc630d): `clip88p` in the preallocated arena,
  `clip96a` / `clip104a` with `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` (card 4, which also ran a small test suite).
- `jzfmm/`: jz-fmm (`Odisseo codes/jzfmm_force_eval.py --memory`), clipped Plummer, leaf 32.
  - `n<N>.json`: p4 theta 0.6, preallocated 0.88, card 6.
  - `n8000000_l32_*`: a p/theta front on card 2 under other users' load.
- `trace/`: per-stage kernel time of one warm 2-step `strict_run_v2` call at 8e6 (no command buffers;
  `bench/analyse_trace_by_stage.py`). `trace8m` is the COM-radii arm, `trace8m_lanes` the lane cascades,
  `trace8m_head` the head (four-level COM passes).

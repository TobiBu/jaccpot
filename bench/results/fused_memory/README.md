# The fused lane's memory: result rows (2026-10-03)

Record: `docs/fused_memory_2026-10.md`. Bench `bench/fused_memory_budget.py` (one configuration per process),
one A100 (card 3, which held an idle 4.2 GiB foreign process; host load ~20 from other users' jobs, so every row
carries the contention flag). Plummer seed 0 unless named, leaf 64 cell leaves, cell_min_level 8, leaf capacity
1.15x the live cells in steps of 1024, theta 0.8, p5, command buffers on. yggdrax main 2c3eef1 (#82) throughout.

| arm | jaccpot | content |
| --- | --- | --- |
| base | b296ec2 | #357's head (81a17ef) + the budget bench |
| new | 9c91797 | perf/fused-memory: caps from counts, sorts, retry, masses fix (the A/B, the ladder to 2.4e7, `twocard/`) |
| new + queue | a7767b8 | + the flat walk's queue ceiling 2^28 (the 3.2e7 rungs, `disc_25m`, the `acc_*` rows; bench file of the working tree) |

* `step0/`: `b_<N>_<bench|tight>_r<k>`, base code; `bench` = the harness's named caps, `tight` = caps named at 1.5x
  the counts the `bench` row measured. `r1` rows carry `memory_analysis()` and XLA's buffer assignment.
* `drift/`: list counts along a rollout (Plummer 2e6 with equilibrium velocities; the 25M disc+bulge IC
  subsampled to 2e6 with the NFW halo as external force).
* `ab/`: `<arm>_<N>_r<k>`, interleaved; `newnamed` = new code with the harness's caps, `new125` = headroom 1.25.
* `ladder/`: the one-card N ladder on the new code, caps unnamed, memory fraction 0.9; `acc_*` the fp64
  accuracy rows; `disc_25m` the production IC whole.
* `twocard/`: the c4 probe, two cards (2+1, a NODE pair) against one, caps unnamed.

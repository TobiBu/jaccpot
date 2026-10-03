# Beyond two cards: result rows (2026-10-03)

Record: `docs/multigpu_fused_2026-09.md`, "Beyond two cards". Probe `bench/multigpu_c4_cross_force_probe.py`,
cards 4-7 (two PIX pairs, one socket), Plummer, leaf 64, theta 0.8, p6, min of 15 calls, frozen worktrees:

| arm | jaccpot | content |
| --- | --- | --- |
| E0 | 8cf863c | #356 (two-sided over leaves) |
| E1 | 27c9994 | + near receiver walk skipped (pass-through) |
| E2 | 5e0ee37 | + probe floors 2^18 |
| E3 | f205b6e | + symmetric exchange, sort-based rows |
| E4 | 78302ac | + presence-map rows |
| F | fd945ae | + unrolled searchsorted, + #356's annotation fix (the weak-scaling sweep) |
| G | de8f8e2 | F + the probe's PROBE_PARTITION=rcb (the RCB rows) |

yggdrax 16be300 (#82) throughout.

* `ab/`: interleaved arm comparisons, `ab_<arm>_nd<ndev>_n<N>_r<round>`.
* `level_sweep/`: PROBE_CELL_MIN_LEVEL 8 / 10 / 11 on arm E4 (`lvl<L>_diag_nd4` are diagnostic runs).
* `weak/`: `w_s<seed>_p<per card>_nd<ndev>`, arm F.
* `rcb/`: Morton vs RCB partition on arm G (`rcbdiag_*` diagnostic, `rcb_*` timing).

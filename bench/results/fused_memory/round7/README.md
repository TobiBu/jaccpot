# Fused lane, round 7: result rows (2026-10-06)

Record: `docs/fused_memory_2026-10.md`, section "Round 7". The bench is `bench/fused_memory_budget.py`, one
configuration per process, with:
- `JACCPOT_STRICT_CARRY=particles` and unnamed caps;
- a preallocated arena of 0.88;
- p6, theta 0.8, cell_min_level 8, clipped Plummer;
- one A100 40 GB (card 0, picked by autocvd), held between runs by the round-6 guard (`scripts/`);
- jax 0.11.2 (`/export/scratch/tbuck/jax0112-venv`).

`*_s` / `ab_*` / `C_*_s*` rows are production-sequence steps (`--skip-eval`); `*_f` / `acc_*` rows are force rows
with `--accuracy-targets 4096` against the cached fp64 references; `bit_*` rows saved the state after 4 steps at 2e6
(`--save-forces`, compared in `logs/`).

Frozen trees (jaccpot; all `perf/fused-round7` commits before squashing): `base` = main `6cca378`; `a1` `0c2e7d0`
(optimization barrier before the gather); `a2` `67451e7` (leaf-major Pallas L2P, opt-in); `a3` `777293b` (list tune
tools); `a4` `8ba18f4` (scatter back to input order); `a5` `8b96e56` (particle-major L2P, opt-in); `a7` `191b74a`
(COM-radii dead-block skip + capture); `a8` `3636f11` (the round's defaults); `a9` `5a8bd97` (+ the L2P's leaf by a
window compare); `a10` `416ac19` (by bisection: the code as merged). Local tags `bench-arm/r7-*` keep them. yggdrax: `r6-main` = main `bad5445`, `y1` `a63adf4` (level offsets from the level sort, 8 unrolled
depth rounds).

| dir | what | code |
| --- | --- | --- |
| `step0/` | main profiled at 8e6 and 1e8 (`--no-command-buffers`, 2 steps): `stages_*.txt` per named scope, `kernels_*.txt` every kernel over 0.1 / 1 ms with its bytes, floor and source lines (`stage_kernels.py`) | base |
| `step_ab/` | 8e6 steps, interleaved: `B*` main; `G` barrier, `Y` yggdrax, `F` COM-radii fold (+barrier), `S` scatter, `L` leaf L2P (+barrier), `P` particle L2P (+scatter), `S2`/`S4` 2 / 4 placement passes; `acc_*` the force rows. `ab_B_8e6_2`, `ab_Y_8e6_2` and `ab_F_8e6_1` ran under a host load of 82/64 cores (a CPU test suite of ours) and are not counted | base, a1, a2, a4, a5, y1 |
| `bitwise/` | `bit_*`: main twice (`B1`, `B2`), `G`, `S`, `Y` (yggdrax), `F` (COM-radii fold), `X` (the round without the L2P), `C` / `C2` / `C3` (a8 / a9 / a10) | base, a1, a4, a7, a8, a9, a10 |
| `tunes/` | the capture rows of `bench/nearfield_capture.py` (`c8*`, `d8*`: near field, walk, lists, COM radii at 8e6); the list and COM-radii tunes are in `logs/chain2.out`, `chain4.out`, `chain7.out` | a1, a3, a7 |
| `final/` | the round's defaults (`C`) against main (`B`), the tree-order carry on them (`T`) and jz-fmm (`jz`, p5 theta 0.8 at 8e6, p5 theta 0.7 at 1e8), 8e6 (two rounds) and 1e8; `stages_*.txt` / `kernels_*.txt` their profile (`a8`); `C2_*` / `C3_*` the L2P's in-kernel leaf lookup by a window compare / by bisection, interleaved with `C` (`*_t1`-`t4`) | base, a8, a9, a10 |
| `logs/` | every chain's printed rows, including the atomics probe (`chain2.out`) | |
| `scripts/` | the chains, `common.sh` (helpers), `atomic_probe.py` (relaxed vs acq_rel atomics; needs the dropped `_relaxed_atomic` module from `a1`-`a7`) | |

The captured inputs (`*.npz`) and the traces are not kept.

# Fused lane, round 6: result rows (2026-10-06)

Record: `docs/fused_memory_2026-10.md`, section "Round 6". The bench is `bench/fused_memory_budget.py`, one configuration per process, with:
- `JACCPOT_STRICT_CARRY=particles` and unnamed caps;
- a preallocated arena of 0.88;
- p6, theta 0.8, cell_min_level 8;
- clipped Plummer unless the name says `unc` (the unclipped draw) or `disc`;
- one A100 40 GB (card 0, picked by autocvd), held between runs by `compare_jzfmm/guard.py`;
- jax 0.11.2 (`/export/scratch/tbuck/jax0112-venv`), yggdrax main `bad5445`.

`*_s` rows are production-sequence steps (`--skip-eval`); `*_f` rows are force rows with `--accuracy-targets 4096` against the cached fp64 references.

Frozen trees: `base` = main `a7da8a6`; `k1` `ab3e9b8`, `k2` `1309465`, `k3` `c7932ea` (options opt-in, env toggles); `k4` `998e59b` (carry with a barrier); `k5` `880eab9` (walk + near field by default); `k6` `0a6073c` (the carry reuses the tree's sorted arrays). The frozen worktrees were made at the same trees before the commits were re-authored (same content, older hashes).

| dir | what | code |
| --- | --- | --- |
| `compare_jzfmm/` | Step 0: jz-fmm at p6 theta 0.8, p5 theta 0.7, p6 theta 0.7 (1e8) and p6 / p5 theta 0.8 (8e6), interleaved with our main at p6 and p5 (`table.md`; round 5's jz-fmm rows are read from `../round5/compare_jzfmm`) | base |
| `tunes/` | near-field and walk kernels timed alone on inputs captured from a real step (`bench/nearfield_capture.py`, `nearfield_kernel_tune.py`, `walk_tune.py`): `t8e6`/`t1e8` (first sweep), `t2_*` (classes, lean), `t3_*` (2D operands), `w*` walk; `c*.json` the captures' static options and the bench rows they came from; `itest_carry.txt` the particle-carry GPU tests | k1, k2, k3 |
| `step_ab/` | full step, interleaved: `B` main's paths, `W` + walk options, `WT` + tree-order carry, `WTN` + tiled near field (two rounds each), `*_f` force rows; `*b_*` the carry with an optimization barrier | k3, k4 |
| `carry_bitwise/` | tree-order against input-order carry, the same 9-step sequence saved and compared (`bitwise.out`); `I6`/`T6` the same on k6 | k3, k6 |
| `ics/` | main (`B_`) against the new defaults (`K_`): unclipped Plummer at cell_min_level 6 (the long-row case) and 8, the disc; `stages_*.txt` / `trace_*` the profile of the new defaults with the tree-order carry | base, k5 |
| `final/` | the defaults as merged (`D`) and with the tree-order carry (`DT`) at 2e5-1e8, main (`B`) at 2e5 and 2e6 | base, k6 |

The captured inputs themselves (`*.npz`, 4.6 GiB at 1e8) are not kept.

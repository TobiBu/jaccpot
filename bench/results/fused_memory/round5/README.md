# Fused-memory round 5: result rows (2026-10-05/06)

Record: `docs/fused_memory_2026-10.md`, section "Round 5". The bench is `bench/fused_memory_budget.py`, one configuration per process, with:
- `JACCPOT_STRICT_CARRY=particles` and unnamed caps;
- a preallocated arena of 0.88;
- clipped Plummer unless the file name says `plummer` (the unclipped draw) or `disc`;
- one A100 40 GB (card 6), held between runs by a guard process (`compare_jzfmm/guard.py`).

`*_s` rows are production-sequence steps (`--skip-eval`); `*_f` rows are force rows with `--accuracy-targets 4096` against the cached fp64 references.

| dir | what | code |
| --- | --- | --- |
| `compare_jzfmm/` | jz-fmm against jaccpot, 8e6 / 3.2e7 / 1e8, interleaved, the same targets and reference (`table.md`, `stages_*.txt`) | jaccpot 041956c on jax 0.10.2; jz-fmm on jax 0.11.1 |
| `defaults/` | A = p5 cml8, B = p5 cml6, E = p6 cml6, P = p6 cml8 across clipped / unclipped Plummer and the disc, 2e5-8e6 | frozen round 4 (041956c) on jax 0.11.2 |
| `jax_ab/` | `old_*` jax 0.10.2 against `new_*` jax 0.11.2, the same frozen tree | 041956c |
| `tunes/` | COM radii (chain against table) and P2M (per leaf against blocked) timed alone on the 8e6 and 1e8 cell trees, plus their scripts | round-5 working tree |
| `kernel_ab/` | full step, old kernels (`JACCPOT_P2M_BLOCK=0 JACCPOT_COM_RADII_VARIANT=chain`) against new, 8e6 and 1e8; `slices*` = CSR placement passes at 1e8; `trace_*`, `stages_*.txt` = the new default's stage split | 596b987 |
| `long_rows/` | cell_min_level 8 against 6 after the near-field row limit; `trace_c6_stages.txt` puts the unclipped 2e6 step in the CSR rank kernel | 596b987 |
| `cml/` | cell_min_level 8 against 6 with all of round 5, every case including 1e8 | 6d97fe0 |
| `rank_scan_ab/` | 1e8 step: `f` = 596b987 with 4 passes, `g` = 6d97fe0 (long-row scan outside the cond), `h` = bdf0674 (inside) | as named |

`f_s1` in `rank_scan_ab/` died in jax 0.11.2's BFC allocator: `Check failed: central_gap_ == kInvalidChunkHandle ... spatial partitioning expects one central gap`, while freeing a buffer during the eager prepare. It is 1 in ~120 runs on 0.11.2; see the record.

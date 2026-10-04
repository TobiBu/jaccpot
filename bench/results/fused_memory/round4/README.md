# Fused-memory round 4: result rows (2026-10-04)

Record: `docs/fused_memory_2026-10.md`, section "Round 4". Bench `bench/fused_memory_budget.py`, one configuration
per process, `JACCPOT_STRICT_CARRY=particles`, the round-2/3 configuration (clipped Plummer, leaf 64 cell leaves,
cell_min_level 8, theta 0.8, p5, unnamed caps, command buffers on).

- `c<N>` / `i<N>` / `s<N>` / `f<N>` rows: the production sequence (`--skip-eval`: one prepare inside
  `strict_run_v2`, then the scan), preallocated arena 0.88 of a 40 GB A100; `ds` = `--donate-state`.
- 8e6 rows: PREALLOCATE off; `--save-forces` writes the eager force and the scan's final state, compared bitwise
  with the previous arm in `run.out`. NOTE: at 8e6 the process peak is set by the eager prepare, not by the scan,
  so the scan's own block is read from the runner dumps (`bench/analyse_step_liveness.py`, dumps not kept).

**Hardware.** A100 40 GB, card 3 (its idle 4.2 GiB foreign process throughout), picked by autocvd.

**Worktrees** (frozen per arm; yggdrax `yggdrax-r2-head` = 2e8f380 for all):

| dir | arm (jaccpot commit) | what |
| --- | --- | --- |
| `donation_freeze/` | R/RS: 52c4c6b (donate_state + freeze-and-resume + relocation), D2: 6690a15 | bitwise; 136M still fails (`c136_off/on/ds`) |
| `kick/` | K: 0431176 (+ P2M gathers, copy-free CSR placement, kick on the drifted state), R3: 52c4c6b | bitwise; `KD128`: 1.28e8 with `--donate-state` 27.46 GiB (230 B/p) |
| `whole_rows/` | WW: 871a18e (`JACCPOT_NEARFIELD_DIRECT_ROWS=whole`), WC: same with `chunked` | `--accuracy-targets 4096`: rel-L2 identical to 8 digits; `w136*` still fail (eager force) |
| `start_in_scan/` | I: d1baa8e (start force inside the scan's program) | bitwise vs WW; but the one-step program's block doubled (`i136*` ask 26-28 GiB) |
| `start_program/` | S: 29bd360 (start force as its own program) | bitwise vs WW; 1.36e8-1.52e8 fit with `--donate-state` (`s*ds`); `s136` without donation 31.40 GiB |
| `walk_fills/` | F: 02967be (distinct fills for the walk's buffers) | bitwise vs S1; `FD128`: 1.28e8 26.19 GiB (220 B/p); 1.52e8-1.68e8 fit, 1.76e8 fails in the eager prepare |

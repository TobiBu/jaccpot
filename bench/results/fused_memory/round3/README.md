# Fused-memory round 3: result rows (2026-10-04)

Record: `docs/fused_memory_2026-10.md`, section "Round 3". Bench `bench/fused_memory_budget.py`, one configuration
per process, `JACCPOT_STRICT_CARRY=particles`, the round-2 configuration (leaf 64 cell leaves, cell_min_level 8,
theta 0.8, p5, unnamed caps, command buffers on).

- `clip<N>`: clipped Plummer (`--ic plummer_clipped --rmax 20`), the production sequence (`--skip-eval`: one
  prepare inside `strict_run_v2`, then the scan), preallocated arena 0.88 of a 40 GB A100.
- 8e6 rows: PREALLOCATE off, eager prepare + force-only timing + the scan; `--save-forces` writes the force and
  the scan's final state, compared bitwise with the previous arm (`run.out`: `equal=True rows_diff=0`).

**Hardware.** A100 40 GB, card 3 (an idle 4.2 GiB foreign process on it throughout), picked by autocvd.

**Worktrees** (frozen per arm; yggdrax `yggdrax-r2-head` = 2e8f380 for all):

| dir | arms (jaccpot commit) | what |
| --- | --- | --- |
| `near_subtile16/` | `jaccpot-r2-head4` = ccc630d, default vs `JACCPOT_NEARFIELD_PALLAS_TARGET_SUBTILE=16` | 16-lane near sub-tiles: 111 -> 124-151 ms per step (negative) |
| `com_radii_kernel/` | X: ccc630d (round-2 head, XLA level passes), K: 7381938 | the fused COM radii kernel; bitwise |
| `far_csr/` | K: 7381938, F: c2298a0 | far list emitted in CSR order; bitwise; `K64`/`F64` at 6.4e7 |
| `box_geometry/` | F: c2298a0, G: 26a9de8 | box geometry deferred on the COM lane; `G64`, `G96` |
| `ceiling_tags/` | 09dff2c | zero-length far tags; ceiling 112M / 120M fit, 128M fails in the scan |
| `row_offsets/` | T: 09dff2c, O: 2911274 | far list carries row offsets; `P1`/`TP`: the `pair` M2L route (expansion) on both arms, bitwise; `OD` the runner dump's run |
| `p2m_in_place/` | Q: 8da2071 | leaf P2M into the table; 128M fits, 136M / 144M fail (fragmentation) |
| `pallas_csr/` | Q2: 8da2071, C: d35526a | the CSR lists without a sort; bitwise; interleaved A/B; 136M-152M fail; `CD128` the 128M runner dump |
| `donation/` | D: 6690a15 | the particle runner donates its initial acceleration; bitwise vs C1; 136M-152M still fail in the arena |
| `async_ceiling/` | 6690a15, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`, PREALLOCATE off | 136M fits (249 B/p); 152M / 168M fail on the scan's temporary block itself (22.6 / 24.7 GiB) |

Per-step liveness: `bench/analyse_step_liveness.py` on the runner dumps (the dumps themselves are not kept: ~0.5 GB).

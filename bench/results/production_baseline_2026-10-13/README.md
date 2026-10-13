# Production baseline, 2026-10-13 (plan session S0)

Record: `docs/production_2026-10.md`, section "S0".

- **Card and host:** one A100-PCIE-40GB (GPU 7, booked through autocvd, with no foreign process during any row).
  Host load was 27-41 on 64 cores.
- **Software:** jax 0.11.2 from `/export/scratch/tbuck/jax0112-venv`. Every row records its own contention flags.

**Code:**
- **main**: jaccpot `5bc4959` + yggdrax `2758fa9` (detached worktrees `jaccpot-review-5bc4959`,
  `yggdrax-review-2758fa9`); the pins and probes ran from this branch (`bench/production-baseline`, main + bench
  files only).
- **round 7**: the frozen `jaccpot-r7-a10` (`416ac19`) + `yggdrax-r7-y1` (`a63adf4`), the round-7 code as merged.
- **Odisseo**: `jaccpot-integration` `adca4f6`, with the overlay `/export/scratch/tbuck/odisseo-overlay` and nornax
  main.

| dir | what |
| --- | --- |
| `rows/` | `bench/fused_memory_budget.py` rows and their logs; `scripts/chainA.sh` and `scripts/chainB.sh` hold the command lines |
| `pins/` | L1-L3 / DET1-2 records: `main-5bc4959` and the A-vs-A control `-b`, plus the `compare` outputs |
| `probes/` | the three D1 probes: `activity_25m`, `frozen_1e6`, `ab_2e4` (k 3, 4) and `ab_2e4_k5` |
| `traces/` | per-stage (`stages_*`) and per-kernel (`kernels_*`) tables of the step traces |
| `scripts/` | the drivers, the Odisseo / block-step / gradient timing scripts, and `stage_kernels.py` (per-kernel attribution, from round 7) |

The traces and HLO dumps themselves are not kept.

**Rows in `rows/`:**
- **8e6 and 1e8 (clipped Plummer):**
  - `R7_*`: round 7;
  - `Mp_*`: main with `--softening-kernel plummer`;
  - `Mf_*`: main with the default ferrers3.

  `_s*` rows are production-sequence steps (`--skip-eval`, particle carry); `_f` rows are force rows scored on 4096
  targets.
- **`D_*`: the 25M disc+bulge** (`--ic disc --n 25000000 --softening 1.5e-3 --softening-kernel ferrers3`):
  - `geo`: geometric theta 0.8;
  - `de4` / `de5`: `dehnen_error` 1e-4 / 1e-5.
- **`O_*`: Odisseo `integrate()` at 8e6** (`scripts/odisseo_time.py`):
  - `def`: Odisseo's defaults;
  - `tun`: leaf 64, p6, theta 0.8, with an fp64 or fp32 state;
  - `hist`: the snapshot path.

  Each runs three calls. Per step = (C - B) / 8.
- **`G_*`: `bench/grad_step_profile.py`** at 2e5, 1e6 and 8e6 (`*_peak.json` is the process's device peak, from
  `scripts/grad_wrap.py`). `G_8e6` first failed in the bench's own direct-sum reference (a fixed 1024-target block,
  61 GiB). That block is fixed on this branch, and the row was re-run.
- **`BS_*`: the Odisseo block-step jitted lane** (`scripts/blockstep_time.py`), fp64, k_max 3.

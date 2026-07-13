# GPU benchmark plan: real (Dehnen) vs complex/solidfmm basis

Goal: measure on a GPU whether the real basis should become the default. It is
already **correct**, **feature-complete** (P2M/M2M/M2L/L2L/L2P, grouped +
class-major, acceleration-derivative towers), and uses **2x less coefficient
memory**. On CPU it is ~0.5-0.9x the complex/solidfmm wall-clock; this plan
confirms and quantifies the advantage on GPU, where it should be larger.

Everything is driven by one script: [`bench_real_vs_complex.py`](bench_real_vs_complex.py).

## Prerequisites on the GPU box
- `yggdrax` importable on a branch that has
  `rebuild_static_radix_tree_from_template` (installed, or a sibling
  `../yggdrax` checkout on `main` — the script adds `../yggdrax` to `sys.path`).
- `autocvd` available (the script grabs a free GPU before importing JAX; falls
  back gracefully if missing). Per the standard setup, GPU JAX work uses
  `autocvd`.

## Run
```bash
# default sweep (N = 3000, 8000, 20000; orders 4, 8), writes a markdown report
python bench/bench_real_vs_complex.py --output bench_real_vs_complex_gpu.md

# push to larger N once the defaults look sane
python bench/bench_real_vs_complex.py --n 8000,50000,200000 --orders 4,6,8 \
    --output bench_real_vs_complex_gpu_largeN.md
```
Useful flags: `--gpu-select {least-used,first,none}`, `--dtype {float64,float32}`,
`--theta`, `--leaf-size`, `--runs`, `--skip-grouped`, `--skip-pallas`.

## What the script reports
1. **Correctness (trust gate).** rel-L2 vs direct sum for real / complex /
   solidfmm across orders. **Real must converge with order** (toward ~1e-6). If
   it does not, stop — something is wrong in the build; do not trust timings.
2. **Compute wall-clock (flat far-field).** `real / solidfmm` ratio per (N,
   order). `< 1` = real faster.
3. **Real flat vs grouped vs class-major.** Grouped modes precompute one
   rotation per interaction class; on GPU that should beat flat. Grouped is an
   opt-in **approximation** — treat its speed as a speed/accuracy trade.
4. **Real z-M2L core: pure-JAX vs Pallas.** `use_pallas=True` should beat
   pure-JAX on GPU (identical on CPU — silent fallback). The Pallas kernel is
   now correctness-guarded (interpret-mode parity test).
5. **Coefficient memory.** real float vs complex128 (fixed 2x).

## Decision criteria (flip the default to real?)
Consider making `basis="real"` the default only if, on the target GPU:
- correctness matches complex/solidfmm across the tested orders, AND
- `real / solidfmm` compute ratio is consistently `< 1` at production
  (N, order, theta), AND
- the fastest real configuration (flat vs grouped vs Pallas) is identified so
  the default wires up the right path.

If real wins, the default flip is small (it is the `basis=` resolution in
`solver.py`) — but do it as a separate change with the GPU numbers attached, and
re-check the adaptive/large-N presets that currently assume the complex path.

## Notes / caveats
- All prior numbers in the PR are CPU/x64. `class_major` is slow on CPU (segment
  scheduling overhead) — that is a CPU artifact, not representative of GPU.
- Grouped accuracy at COM centers + theta>=0.5 is ~1% (pair_grouped) for **both**
  bases; the accurate default path is flat.

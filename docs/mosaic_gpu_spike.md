# Mosaic GPU on the A100: the S1b spike (2026-11-01)

Dev plan `galaxy-production-dev-plan.md`, "### S1b". Question: jaccpot's kernels are Triton-Pallas
(`pl.pallas_call` with `pallas.triton.CompilerParams`, `jaccpot/pallas/_compat.py`); Triton-Pallas is deprecated in
jax 0.11.2 and jax is pinned below 0.12. Can the kernels move to Mosaic GPU on our hardware (A100, sm_80), at what
cost, and should new kernels be written to a portable subset?

Branch `s1b/mosaic-l2p` (from main 1dbf7e13), jax/jaxlib 0.11.2, local A100-PCIE-40GB cards 6/7 (shared:
functional and memory checks only; the timings are owed by HoreKa, below). Probes and logs:
`/export/scratch/tbuck/s9-scratch/s1b/`.

## Answer in brief

- **It runs on sm_80, but only through `plgpu.kernel`, and only the scalar/slice subset of Pallas.** The ported L2P
  forward is correct at the galaxy preset: at 2e5 and 2e6 it equals Triton to fp32 rounding (not bitwise: 6-10 % of
  the components differ by 1-2 ulp), with the same error against fp64 (max relative 9.0e-7 for both at 2e6), and the
  full force step agrees to 2.2e-7 max relative. Its force-step program has 23.4 MB (11.7 B/p) LESS temp than
  Triton's at 2e6 (no aliased zero buffer).
- **The gather is the blocker.** Mosaic GPU cannot index a ref by an array; every jaccpot kernel's inner loop is a
  per-lane gather. The L2P got around it two ways: a select over the block's leaf window (pure Pallas, ~doubles the
  ALU work) or hand-written per-thread loads in `inline_mgpu` (MLIR on `jax._src` bindings, no interpret mode).
  Neither carries over to the M2L, near-field or walk kernels without most of each kernel becoming inline IR.
- **Timing is owed by HoreKa** (`bench/slurm/s1b_mosaic_l2p.sbatch`).
- **Recommendation:** stay on Triton for the existing kernels; write new kernels so their arithmetic is a shared
  backend-free function and their memory access goes through a small set of helpers (below); revisit when Mosaic
  GPU grows a gather or jax 0.12 forces the move.

## What Mosaic GPU expresses on sm_80 (jax 0.11.2), measured

Probes `probe_features.py` (F1-F9) and `probe_features2.py` (F10-F13), each a tiny `plgpu.kernel` over a grid of
128-thread warpgroups, under both lowering semantics; on the GPU each feature runs in its own process (an LLVM
abort kills the process, see F7/F8).

| feature | Lane | Warpgroup | CPU interpret |
|---|---|---|---|
| F1 `ref[idx_array, c]` gather from GMEM | FAIL `NotImplementedError: Unimplemented primitive ... broadcast_to` (a 1-D `ref[idx]` reaches `int_indexer_shape non-empty`) | FAIL (same) | FAIL `TypeError: Only scalar arrays can be converted` |
| F2 scalar dynamic load `ref[b, 2]` | OK | OK | OK |
| F3 `broadcasted_iota(layout=WG_STRIDED)`, int/float arith, `//`, `min`, `where` | OK | OK | OK |
| F4 per-thread gather in `inline_mgpu` (`foreach` + `memref.load`) | OK | OK | refused: no `inline_mgpu` |
| F5 masked per-thread scatter in `inline_mgpu` (`scf.if` + `memref.store`) | OK | OK | refused |
| F6 `fori_loop` with array carries | OK | OK | OK |
| F7 SMEM scratch staging (`run_scoped(SMEM)`, store, `commit_smem`, load) | **process abort**: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.fence.proxy.async.shared_cta` | same abort | not isolated (the probe stages a value read by the `inline_mgpu` gather, which interpret refuses) |
| F8 `copy_gmem_to_smem` + barrier | FAIL `ValueError: copy_gmem_to_smem with a barrier is only supported Hopper and newer GPUs` | same | OK (interpret ignores the hardware) |
| F9 contiguous slice `ref[pl.ds(s, 128)]` | OK | OK | OK |
| F10/F11 column slice get/store of `[N, 3]` (`ref[pl.ds(s,128), c]`) | OK | OK | OK |
| F12 select-gather: `fori_loop` over rows, scalar loads `tab[j, k]` (49 k) `where`d into 49 carries | OK | OK | OK |
| F13 `fori_loop` with a trip count loaded from GMEM | OK | OK | OK |
| int32 loop carry initialised from an iota-derived array, updated by `where(hit, int, acc)` | FAIL: carry type mismatch (`FragmentedArray ... False` vs `True`: iota-derived ints are tagged unsigned, `where` with an int scalar returns signed); workaround `jnp.where(lane >= 0, 0, lane)` as the initial value | n/a | n/a |

Entry point and programming model:
- `pl.pallas_call(..., compiler_params=plgpu.CompilerParams())` fails (bare `AssertionError` in
  `pallas_call.gpu_lowering`): in 0.11.2 only Triton and TPU register a backend rule for `pallas_call`. The entry
  point is `plgpu.kernel(body, out_type=..., grid=..., grid_names=..., compiler_params=...)` (no `name=`, no
  `input_output_aliases`): refs are whole GMEM arrays, the program index is `lax.axis_index(grid_name)`.
- Arrays are warpgroup-distributed `FragmentedArray`s. One particle per thread is a `(128,)` array in
  `plgpu.Layout.WG_STRIDED((128,), vec_size=1)` (via `plgpu.broadcasted_iota(..., layout=...)`; arrays derived from
  it keep the layout, which matters for loop carries). There is no per-lane scalar semantics: the "warp" semantics
  (`plgpu.warp_map`) are per-warp scalars, not per-thread.
- Element-wise integer/float arithmetic, comparisons, `where`, `min/max`, integer `//`, `fori_loop` with a dynamic
  (GMEM-scalar) trip count and many array carries, `jnp.sum` over a 1-D array: all lower in both semantics.
- NOT lowered: gather (`ref[idx_array]` raises `NotImplementedError: int_indexer_shape non-empty`; `gather_p` has no
  rule), scatter to per-lane addresses, `cumsum` (no rule), sorting. Atomics (`plgpu.atomic_add`) take only
  scalar or slice indices and return nothing (no fetch-and-add: the slot-claiming idiom of the walk and
  `csr_place` cannot be written). Masked loads/stores (`plgpu.load/store(mask=)` of the Triton dialect) do not exist;
  a block must stay in bounds.
- Hopper/Blackwell-only: TMA (`copy_gmem_to_smem`), wgmma, clusters, tcgen05/TMEM -- see F7/F8 for what happens on
  sm_80. SMEM scratch through `pl.run_scoped(..., plgpu.SMEM(...))` exists, but staging registers through SMEM needs
  `plgpu.commit_smem()` (a `fence.proxy.async`, Hopper).
- The escape hatch is `plgpu.inline_mgpu`: a Python function that receives `FragmentedArray`s and raw memrefs and
  emits Mosaic GPU / MLIR directly (`jax._src.lib.mlir` dialects). A per-thread gather is
  `idx.foreach(lambda v, _: memref.load(ref, [index_cast(v), col]), create_array=True)`; a masked per-thread store is
  an `scf.if` around `memref.store`. Lane semantics only, and NOT supported by the interpret mode.

## The port: `jaccpot/pallas/l2p_real_mosaic.py`

`l2p_real_particles_mosaic(...)` -- the same inputs and output as the Triton
`l2p_real_particles_pallas` -- and `l2p_real_particles_mosaic_cvjp` (forward Mosaic, reverse the existing Triton
`l2p_real_particles_reverse_pallas`). Opt-in through `JACCPOT_L2P_KERNEL`:
`mosaic` (inline gather), `mosaic_select` (pure Pallas), `mosaic_interpret` (the latter, interpret mode); `auto`
still resolves to `pallas`/`xla` as before.

Shape: one warpgroup (128 threads) per 128 consecutive tree-ordered particles, one particle per thread, the Triton
kernel's arithmetic (`l2p_real_grad_cartesian` is shared, the identical Python traced into the other backend).
What had to change:
- **The gather.** The Triton kernel bisects each lane's leaf and gathers the leaf's centre and `(p+1)^2 = 49`
  coefficients per lane. Mosaic cannot index a ref by an array, so:
  - `gather="select"` (pure Pallas): the block's particles lie in a window of consecutive leaves
    (`first`, `nwin` per block, precomputed by two `searchsorted`); a `fori_loop` over the window loads each leaf's
    52 values as warpgroup-uniform scalars and `where`s them into the lanes whose particle lies in that leaf. Values
    are moved, never combined, so the result is the stored value bit for bit. Extra work: `nwin x 52` selects per
    lane (galaxy cell leaves: 12.3 (2e5) to 16.4 (2e6) particles per live leaf on average, so ~9 leaves per 128-particle block).
  - `gather="inline"`: the window loop selects only the leaf's node id; 52 per-thread `memref.load`s in
    `inline_mgpu` do the gather, as Triton's loads do.
- **No masked stores.** The last block is shifted back to `[N - 128, N)` (overlap, identical values written twice)
  instead of masking lanes past `N`; `N < 128` is padded. Rows past the leaves' particles get explicit zeros (no
  aliased zero buffer: `plgpu.kernel` has no `input_output_aliases`).
- **Positions** are read as contiguous column slices `pos_ref[pl.ds(s, 128), c]` of the `[N, 3]` array, the output
  written the same way.
- **Semantics.** `select` lowers and is correct under both Lane and Warpgroup; `inline` needs Lane
  (`inline_mgpu` hands `FragmentedArray`s only there). fp32 only was exercised.

## Correctness

CPU interpret mode (`tests/unit/runtime/test_l2p_mosaic.py`, 6 tests, with and without
`JACCPOT_RUNTIME_TYPECHECK=1`): `select` equals the Triton kernel's interpret mode bit for bit (leaves of 1-8,
padding leaves, dead rows, `N < 128`, an overlapping last block), and through the `JACCPOT_L2P_KERNEL` dispatch
including `jax.grad` (the custom_vjp).

GPU, synthetic leaves (`check_gpu_l2p.py`; p4 and p6, up to 20,500 rows):

| case (rows, live, padding leaves, p) | Triton vs fp64, max rel | Mosaic (all three variants) vs Triton | Mosaic vs fp64, max rel |
|---|---|---|---|
| 200 / 200 / 0 / p4 | 8.07e-7 | 364 of 600 components differ, max abs 2.6e-6 (max \|g\| 15.0) | 5.69e-7 |
| 200 / 187 / 2 / p4 | 6.17e-7 | 376 of 600, 1.9e-6 | 1.09e-6 |
| 97 / 97 / 1 / p4 | 3.91e-7 | 175 of 291, 1.9e-6 | 4.84e-7 |
| 333 / 300 / 4 / p6 | 1.25e-6 | 630 of 999, 7.6e-6 | 5.78e-7 |
| 20,500 / 20,000 / 30 / p6 | 2.21e-6 | 40,946 of 61,500, 4.6e-5 (max \|g\| 115) | 1.86e-6 |

(fp64 reference here: the Triton kernel on float64 inputs.) `inline`/Lane, `select`/Lane and `select`/Warpgroup gave
the same numbers. Not bitwise: the arithmetic is the same Python, but the two compilers emit it differently (FMA
contraction/ordering; not isolated further), so ~60 % of the components of these random expansions differ in the last
bits -- with no accuracy cost.

GPU, the production operating point (`bench/s1b_mosaic_l2p.py --mode check`): `preset="galaxy"` (leaf 64, p6,
`dehnen_error`, cell leaves, fp32), the disc+bulge IC (`disk_bulge_25m.npz`, a prefix rescaled), the L2P on the
prepared state's own inputs, then one full force step `FusedForceProvider.step` (rebuild + accelerations) per
kernel:

| | 2e5 (19,456 leaves, 16,276 live, 12.3 per live leaf) | 2e6 (140,288 leaves, 121,883 live, 16.4 per live leaf) |
|---|---|---|
| L2P: Mosaic vs Triton | 57,827 of 600,000 components differ, max abs 1.9e-6 | 353,160 of 6,000,000, max abs 3.8e-6 |
| L2P vs fp64 (`xla` autodiff form on float64), Triton | max rel 7.97e-7, p99.9 5.31e-7, rel L2 1.570e-7 | 8.99e-7, 4.76e-7, 1.443e-7 |
| L2P vs fp64, Mosaic | max rel 7.97e-7, p99.9 5.29e-7, rel L2 1.570e-7 | 8.99e-7, 4.76e-7, 1.443e-7 |
| `mosaic` vs `mosaic_select` | bitwise (same sha256) | bitwise |
| force step: Mosaic vs Triton | 34,953 of 600,000 differ; max rel 2.06e-7, rel L2 1.4e-8 | 282,890 of 6,000,000; max rel 2.16e-7, rel L2 1.2e-8 |

Rows: `/export/scratch/tbuck/s9-scratch/s1b/check_rows.jsonl` (logs `check_2e5.log`, `check_2e6.log`), card 6, the
disc IC prefix with G and softening as `bench/multigpu_force_repeat.py` sets them.

## Memory

Compiled temp bytes (`compiled.memory_analysis().temp_size_in_bytes`) of the L2P program and of the force step
program, per kernel:

| program | 2e5 Triton | 2e5 Mosaic | 2e6 Triton | 2e6 Mosaic |
|---|---|---|---|---|
| L2P alone, temp | 98 KB | 208 KB | 0.75 MB | 1.63 MB |
| force step (`FusedForceProvider.step`), temp | 145.85 MB | 143.53 MB | 931.10 MB | 907.66 MB |

The L2P's own extra temp is the per-block `first`/`nwin` window arrays and their `searchsorted` (N/128 entries
each, ~0.4 B/p). The step is 23.4 MB (11.7 B/p) lighter at 2e6 and 2.3 MB at 2e5, consistent with the Triton call's
aliased `[N, 3]` zero buffer (12 B/p) that the Mosaic kernel does not need (it writes every row). Both
`mosaic` variants compile to the same temp.

## Interpret mode (CPU CI)

`plgpu.kernel(..., interpret=plgpu.InterpretGPUParams())` runs on CPU (`JAX_PLATFORMS=cpu`), both semantics: the
pure-Pallas features (scalar loads, slices, iota/arith/where, `fori_loop`, even the TMA copy F8) execute, and the
`select` L2P equals Triton's interpret mode bitwise -- so CPU CI can test a Mosaic kernel the way it tests the
Triton ones today. Two limits: `inline_mgpu` is refused (`ValueError: GPU interpret mode does not support
inline_mgpu`), so any kernel that needs the escape hatch (every gather/scatter-heavy kernel at full speed) is
untestable on CPU; and an array-indexed get fails in interpret mode too (`TypeError: Only scalar arrays can be
converted to Python scalars`), consistent with the GPU lowering.

## Port effort per kernel family (judged from reading them)

Every family's inner loop is a per-lane gather from GMEM, which Mosaic GPU can only do in `inline_mgpu` (or by the
select trick when the indices are block-local and few).

| family | files | what blocks a pure-Pallas port | effort |
|---|---|---|---|
| L2P (forward) | `l2p_real.py` | per-lane coefficient gather | done here (select: pure Pallas; inline: hand IR) |
| L2P / P2M reverse, P2M | `l2p_real.py` (rev), `p2m_real_leaf.py` | `(block, chunk)` 2-D tiles with `[:, None]` broadcasts and per-row `jnp.sum(axis=1)`; per-lane position gathers; masked tile stores into node rows | medium: 2-D layouts (`TILED`/`WGMMA`-family) and cross-lane reductions are Hopper-shaped; would be re-blocked as 1-D, positions are contiguous per leaf (slices) |
| M2L CSR lanes | `m2l_real_csr_lanes.py`, `m2l_real_csr.py`, `m2l_real_fused.py` | per-lane source-node id gather (`src_ref[i]`, then centre / multipole rows of that node), per-target reductions | high: the gather is the operand fetch of every pair (no window to select over); inline IR needed; the fused complex kernel needs fp64 |
| near field CSR | `nearfield_leafpair_csr*.py`, `nearfield_fused_leaf.py`, `nearfield_mutual.py` | per-lane gathers of source particles, in-kernel `cumsum`, per-lane scattered stores and atomics | high: same gather problem at the largest volume, plus scatter/atomics |
| walk | `mutual_walk_pallas.py`, `csr_place.py`, `com_radii_leaf.py` | per-lane node gathers, fetch-and-add slot claims (Mosaic's `atomic_add` returns nothing), in-block `cumsum`, scattered stores to claimed slots | highest: the emission pattern has no Mosaic equivalent outside inline IR |
| cascades (M2M/L2L) | `cascade_real_lanes.py`, `cascade_real_level.py` | per-lane node / child gathers by level lists; small dense `(Bp, Bp)` contractions | medium-high: gathers again; the dense part could map to plain broadcasts |

Each port is a rewrite, not a re-target: in practice "inline_mgpu for every gather and scatter", i.e. most of the
kernel in hand-written MLIR on `jax._src` bindings, which are themselves not a stable API.

## HoreKa timing still owed

`bench/slurm/s1b_mosaic_l2p.sbatch`: one card of an exclusive accelerated node, `bench/s1b_mosaic_l2p.py --mode
time` at 2e6 and 2.5e7 (disc) and 1e8 (Plummer): the L2P alone and one full force step, each compiled once per
kernel (`pallas`, `pallas#ctrl` = Triton compiled twice, the A-vs-A control, `mosaic`, `mosaic_select`), timed in
interleaved rounds (`REPS` rounds of `INNER` calls). The question it answers: is the select gather's extra work
(and Mosaic's codegen for scalar-heavy Lane code) within noise of Triton's at 1e8, where the Triton L2P forward is a small part of a ~0.9 s step
(1.15 ms at 8e6, p5, measured 2026-10; the XLA form it replaced took ~52 ms at 1e8). No timing is claimed here.

## Recommendation

1. **Do not migrate now.** On sm_80 Mosaic GPU is a different programming model (warpgroup arrays, no gather, no
   per-lane scatter, no fetch-and-add, no in-kernel scan, Hopper-only SMEM/TMA paths that abort or refuse on the
   A100), not a re-target of `pallas_call`. The L2P -- our simplest kernel -- ports correctly, but every other family
   needs its gathers, scatters and atomics in `inline_mgpu` IR. Keep Triton-Pallas and the `jax<0.12` pin; the
   decision point is the jax release that drops Triton-Pallas (or a Mosaic GPU gather), not now.
2. **New kernels (S5, S6, S8, S12) target a "portable core", not a portable kernel.** A kernel is then three parts:
   (a) its arithmetic as a backend-free function on broadcastable arrays (as `l2p_real_grad_cartesian` is: it
   traced unchanged into both backends here); (b) its memory access through a handful of helpers -- gather rows by a
   lane index, contiguous block load/store, masked scatter, block-claimed slots -- implemented per backend (Triton:
   `ref[idx]`/`plgpu.store(mask=)`/`atomic_add`; Mosaic: select-over-window or `inline_mgpu`); (c) a thin launcher.
   Rules that cost nothing in Triton and keep (b) small: 1-D lane arrays (one item per lane, no `(block, chunk)` 2-D
   tiles with cross-lane `sum(axis=1)`), blocks that stay in bounds (shift the last block back) instead of masked
   loads/stores, no in-kernel `cumsum`/sort, no reliance on fp64 in kernels, uniform (scalar) loop trip counts.
3. **Interpret-mode CI works for the pure-Pallas subset** (`select` L2P tested bitwise on CPU here), not for
   `inline_mgpu`: if a migration happens, the gather helpers need a select fallback so CPU CI still covers the
   arithmetic.
4. **HoreKa** decides whether the select gather is affordable (it adds ~9 x 52 selects per particle to ~500 flops
   at the galaxy occupancy): run `bench/slurm/s1b_mosaic_l2p.sbatch`.

# Fast gradients for the sub-10 ms FMM (2026-09-13)

Plan: `~/.claude/plans/fast-gradients-for-the-sub10ms-fmm.md`. Forward record: `docs/sub10ms_2026-09.md`.
Branch `perf/fast-fmm-gradients` (cut from `perf/sub10ms-lanes` at d06dee0). Operating point throughout:
N = 2x10^5 Plummer, Morton-cell leaves of 64, theta 0.8, order 5, fp32, `preset="large_n_gpu"`; timings from
`bench/grad_step_profile.py` on one A100 under `GpuMonitor` (host load 40-60 the whole session -- every row
below is CONTENDED and is comparable only with rows from the same run; see the trap list).

## What was wrong before this branch

1. **Phase 0 (d06dee0) was half a fix.** It routed the gradient path around the Pallas cascades and the leaf
   P2M, whose `pallas_call` has no AD rule (the generic JVP rule asserts on `program_id`). But the pair-per-lane
   M2L CSR kernel (`_m2l_csr_pallas_active`, default-on since Phase 5) had no guard at all, so on an Ampere card
   `jax.grad` still died in `m2l_real_csr_lanes.py::_m2l_lanes_kernel -> program_id` (Phase 1 log,
   `artifacts/grad/phase1_d06dee0.json`). CI never saw it: the CSR M2L does not lower on CPU.
2. **The Phase 0 test was vacuous in the far field.** At N = 512, leaf 32, theta 0.5 the frozen topology has
   zero far pairs, so the multipoles are dead code, JAX never calls their reverse, and the near field alone
   reproduces the direct sum. The test could only fail through JVP *tracing* of the dead forward -- which it did,
   so it "failed before the fix" and proved less than it seemed. Re-cut at N = 1024, leaf 8 with a far-pair
   assertion (`tests/unit/runtime/test_grad_path_pallas_cascades.py`) and reverse-kernel call counters.
3. **The differentiable forward was wrong at the operating point.** With Phase 0 in place the differentiable
   forward disagreed with the production eval by rel-L2 4.9e-2 and with the fp64 direct sum by aggL2 5.0e-2
   (the production eval: ~1e-3). The differentiable near field ran the prepacked RECTANGLE lane, whose rows are
   capped (`max_neighbors_per_leaf`), and on the cell tree one far-out cell neighbours 10950 of 10956 leaves.
   The CSR lane is the one that sees the real rows; it was guarded `not differentiable`.

## What the branch adds

Every fast kernel of the forward now carries a `custom_vjp` whose forward is the production launch and whose
reverse is a Pallas kernel of its own. Each reverse is the transpose of the SAME body the forward runs,
obtained by `jax.vjp` traced *inside* the kernel (one call gives both halves: the coefficient transpose and the
geometry contraction); the guarded (`safe=True`) geometry keeps the transpose finite at `rho == 0` / `r == 0`
without changing the primal (double-`where`, as `operators/real_rotations.py`).

| stage | forward kernel | reverse kernel | structure |
|---|---|---|---|
| leaf P2M | `p2m_real_leaf.py` | `_p2m_rev_leaf_kernel` | per leaf; L2P-shaped contraction of the leaf cotangent, lane-wise vjp for positions, block output scattered to particles (unique indices) |
| M2M cascade | `cascade_real_level.py` | `_m2m_rev_level_kernel` | top-down, one program per NODE (`g[c] += J_c^T g[parent]`, the L2L shape); edge cotangent per child slot, folded onto centres in XLA |
| L2L cascade | same | `_l2l_rev_level_kernel` | bottom-up, one program per PARENT gathering two children (the M2M shape) |
| M2L CSR lanes | `m2l_real_csr_lanes.py` | `_m2l_rev_lanes_kernel` | one program per SOURCE over the by-source CSR (`csr_by_target` with the roles swapped); per-pair target-centre cotangents by masked store, one segment sum |
| near field CSR | `nearfield_leafpair_csr.py` | `_nearfield_leafpair_csr_rev_kernel` | same chunk table and rows as the forward; the list is symmetric so a target-centric pass sees each pair from both ends; G and softening cotangents ride along |
| walk | -- | none | fixed topology |

No atomics anywhere: every reverse writes rows it alone owns. Integer topology crosses the `custom_vjp` as
ordinary arguments with `None` cotangents (verified on jax 0.10.2; the older float-cast trick is unnecessary).

Wiring: the call sites (`upward/real_tree_expansions.py`, `runtime/kernels/_l2l.py`,
`runtime/kernels/_downward_prep.py`) use the `*_cvjp` seams unconditionally (the forward is byte-identical; only
forward-mode `jax.jvp` is refused, which nothing here uses); the near-field fast lane takes the CSR `custom_vjp`
when `differentiable` and no potential is requested. `GradConfig.cascade_pallas` (env `JACCPOT_CASCADE_PALLAS`,
default on) is the A/B switch back to the pure-JAX loops; `on_grad_path()` forces the lanes M2L kernel on the
gradient path since only that CSR kernel is differentiable.

## Validation

Per kernel, both halves (coefficients AND geometry), fp64, `jax.vjp` of the Pallas seam vs `jax.vjp` of the
exact pure-JAX twin, interpret mode and native: `tests/unit/operators/test_{cascade_real_level,p2m_real_leaf,
m2l_real_csr_lanes,pallas_nearfield_leafpair_csr}_vjp.py` (rel-L2 < 1e-9 everywhere; on-axis pairs finite
where the unguarded twin is NaN). End to end: `test_grad_path_pallas_cascades.py` (radix path, direct-sum
oracle, reverse kernels counted, Pallas vs loop reverse agree to 1e-9) and
`test_grad_path_fast_kernels_large_n.py` (the production `large_n_gpu` path on CPU with every interpret flag,
all five reverse kernels counted, direct-sum oracle).

## Numbers (one A100, host load 40-80 throughout; `artifacts/grad/*.json`)

Every row is min of 7 warm calls after 2 warm-ups under `GpuMonitor`, and every row carries the `loadavg>=8`
flag: the step is launch-bound and host load moves it ~20 %, so these are ratios within one session, not the
record. "eval" is the production eval-only closure (no tree rebuild); "grad" is `jax.grad` of a quadratic loss
through `differentiable_accelerations` w.r.t. positions and masses. The eager grad retraces the whole pipeline
per call (6-120 s) and is host-bound; the jitted number is the one that means anything.

| lane | code | diff. forward vs eval | vs fp64 direct sum (aggL2) | grad, jitted |
|---|---|---|---|---|
| Phase 0 (d06dee0): loop cascades, pure-JAX M2L, rectangle near field | d06dee0 | 4.9e-2 | 5.0e-2 | 942 ms |
| same lanes, this branch (env: `CASCADE_PALLAS=0 FUSED_M2L_CSR=0 LEAFPAIR_CSR=0`) | ba9abfe | 4.2e-8 | 1.27e-3 | OOM on a 6 GB-free card (14.4 GB) |
| **this branch, defaults** (five Pallas reverses) | ba9abfe | **1.8e-11** | **1.27e-3** | **31.8 ms** |

The production eval was 2.5-4.6 ms in these runs (contended; the record's idle number is ~4.6 at cells64), so
the reverse costs about 7-10x one eval, or ~3x the ~11 ms fused step. **942 -> 31.8 ms: 29.6x.** Accuracy:
the differentiable forward is now the production force (1.8e-11 is fp32 reassociation), where d06dee0's was
5e-2 off -- and the same lanes on this branch are right to 4e-8, so d06dee0's error came from what
`prepare_state` bakes into the rectangle payload when the CSR lane is the production near field, not from
the lane arithmetic (mechanism not chased further: the gradient path no longer touches the rectangle).

Stage split on this branch (jitted grad of the stage's own loss; the stages overlap in the COM/upward chain):

| stage | grad jitted, Phase 0 lanes on this branch | grad jitted, this branch |
|---|---|---|
| upward (COM + P2M + M2M) | 205.6 ms | 13.5 ms (15x) |
| far field (upward + M2L + L2L + L2P) | 756 ms | 23.6 ms (32x) |
| near field (P2P) | 1031 ms (rectangle lane, full rows) | 13.9 ms (74x) |
| full step | 942 ms (d06dee0 code, empty card) | 31.8 ms (29.6x) |

The Phase 0 lanes' near field alone (1031 ms, complete rows) costs more than d06dee0's whole gradient (942 ms):
d06dee0's near field was cheaper because it was incomplete -- the same fact as its 5e-2 force error, seen from
the timing side. The lanes' full-step number on this branch is missing because its jitted reverse needed
14.4 GB on a card with 6 GB free; the stage numbers come from a quieter host (load 16-20) than the rest.

Kernel table of the jitted full grad (Perfetto, 3 calls): device busy 31.6 of 32.4 ms, 577 launches.

| kernel | ms / call | launches |
|---|---|---|
| `l2l_rev_real_level_p5` (47 levels) | 4.18 | 47 |
| `input_scatter_fusion_10/11/7/3/1/5/4` (XLA scatters, together) | 13.3 | 7 |
| `near_rev_leafpair_csr` | 3.34 | 1 |
| `m2m_rev_real_level_p5` | 1.73 | 47 |
| `nearfield_leafpair_csr` (forward) | 1.59 | 1 |
| `m2m_real_level_p5` (forward) | 1.34 | 47 |
| `l2l_real_level_p5` (forward) | 0.76 | 47 |
| `m2l_rev_real_csr_lanes_p5_k32` | 0.56 | 1 |
| `p2m_rev_real_leaf_p5_w64` | 0.24 | 1 |
| `m2l_real_csr_lanes_p5_k32` (forward) | 0.21 | 1 |
| `p2m_real_leaf_p5_w64` (forward) | 0.18 | 1 |

So the five reverse kernels together cost 10.1 ms (the plan's "~78 ms prize" for the two cascades alone became
5.9 ms), the forward kernels 4.1 ms, and the remaining ~13 ms is seven XLA scatter fusions of one launch each. The
candidates, by the sizes involved (the fusions are anonymous; not yet attributed one by one): the transposes of
the leaf-table gathers (positions/masses -> `(leaves, W)` layout, 1M rows), the P2M reverse's particle scatter
(1M rows), the M2L reverse's per-pair segment sum (690k rows), the near-field partial reduce onto the leaves
(22k chunks x 64 x 8) and the COM cotangent's per-node segment sums. **That is the next lever, and it is XLA,
not Pallas**; the P2M scatter and the leaf-table transposes have unique indices and could be gathers. The reverse cascades are 3.1x (M2M) and 5.5x (L2L) their
forwards, which is the expected 2-3x transpose cost plus the geometry half.

## Bookkeeping

* `artifacts/grad/phase1_d06dee0.json` is the crash evidence for item 1 above (the traceback names
  `_m2l_lanes_kernel -> program_id`); `phase0_baseline_m2lcsr0.json` is the 942 ms row (its stage split died
  in d06dee0's own code: outside the grad override the old Pallas cascades have no reverse);
  `phase2_fast_kernels.json` is this branch. The `git_head` field in the JSONs reads d06dee0 for every run
  because the profile started before commit ba9abfe existed; the code under test was the working tree.
* `tests/integration/test_grad_fmm_vs_directsum.py::test_grad_fmm_matches_grad_directsum_masses[*]` is red on
  `perf/sub10ms-lanes` (d06dee0) already: its own inertness guard trips ("got 0 M2L pairs" at N = 64, leaf 4,
  theta 0.6), i.e. the far field went empty there after the MAC-geometry change. Not touched here.
* The base branch is not black-clean under the repo's `line-length = 88`; the hooks reformat untouched
  regions. Those hunks were reverted so the diff rebases onto `perf/sub10ms-lanes`; only edited regions
  carry the new formatting.

## Traps met

* **`jax.vjp` inside a `pallas_call` body works** (interpret and Triton): the transposed jaxpr is plain
  arithmetic; multi-axis `reduce_sum` is peeled one axis at a time by the Triton lowering.
* **A Python-int index inside `ref.at[...]` breaks `plgpu.store`** on jax 0.10.2 (`TypedInt` has no `.shape`);
  use `jnp.asarray(i, jnp.int32)`.
* **Unguarded `sqrt`/`arctan2` transposes are NaN at `rho == 0`**, and every PADDING lane of a lane kernel sits
  at `rho == 0` -- so the reverse must use the guarded geometry even if real pairs never hit the axis.
* **Dead far fields.** A test whose topology has no far pairs never calls the multipole reverse. Assert the pair
  count.

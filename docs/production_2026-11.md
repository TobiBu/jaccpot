# High-accuracy, multi-node galactic dynamics, 2026-11

**[open]** The record of the multi-node accuracy plan (sessions H0, A1-A4, I1-I2, B1-B3, M1-M6). There is one
section per session. The plan lives outside the repo (`multinode-accuracy-dev-plan.md`); this file records what each
session measured and decided. It follows [`production_2026-10.md`](production_2026-10.md) (S0-S13).

**The user's decisions behind it (2026-11-03):** U1 nornax owns the integration schemes; U2 positions as an fp64
anchor per leaf plus fp32 offsets; **U3 fix `dehnen_error`** (#487), the geometric MAC stays only as a documented
fallback; U4 the targets (1e9 on <= 8 nodes, force rel-L2 <= 1e-4); U5 orbax restarts; U6 hybrid Hermite.

## A1 — the dehnen_error stall (#487)

**Result: the stall was a NaN in eq (16a)'s walk table, not a vanishing f_b.** Two changes, both on by
default: a non-finite `s_n` in the walk table becomes 1 (the fix), and CSR rows longer than 2048 are merge-sorted
(the safety net). The floor (`mac_force_scale_floor`) did not bind at the stall; it stays opt-in.

**Setup.** One A100-PCIE-40GB, booked through autocvd and shared with other users' jobs (card 6): every timing
below is contended and is a functional observation only. Software: jax 0.11.2, yggdrax main (`yggdrax-mn-ref`).
Code arms: jaccpot main 96122f63; A1 at 8ba2d2b8 (the safety net and the floor, no NaN guard); A1 at bfa876c8
(plus the NaN guard). State: the 1e8 bulge-free bar disc, `stall_a/restart_3.npz` (step 30), `preset="galaxy"`,
dt 1e-3, one step per segment. A wrapper around the flat-walk builder reported every traced build's near-list
volume and longest row, and dumped the longest row's sink leaf: its eq (16a) table row, radius, particle ids,
ancestors and a sample of 8192 row sources (scripts in the A1 scratch directory, `probe_stall.py`,
`fb_offline.py`).

### What made one near row 70 % of the tree

Steps 31-40 reproduce the issue's builds exactly: steps 38 and 39 have rows of 5,302,664 and 5,300,783 leaves at
nodes 11,881,815 and 11,878,135 (#487's numbers). The sink leaf of step 38's row:

| | value |
| --- | --- |
| particles | **one** (id 45,614,311, at (0.02, -14.10, 20.58), r = 24.9) |
| MAC radius | 1.86e-9 (the rounding residue of its float32 COM) |
| carried f_b | 0.1131, against \|a\| = 0.142: the right size at r = 25 |
| threshold `eps * f_b` | 1.13e-5 (the tree's smallest leaf thresholds are 1.06e-5) |
| walk-table `s_1..s_6` | **1.0000006, 0, 0, 0, NaN, NaN** |
| eq (16a), 8192 sampled row sources | convergent 95.4 %, B -> A passes 95.3 %, **A -> B passes 0 %** |
| its ancestors' rows | finite |

`dehnen_walk_table` forms `s_n = P_n / (M rho^n)` as `(P_n / M) / rho / ... / rho` precisely so that no `rho^n`
is formed. On a GPU, XLA folds the chain into one division by `rho^n`. With rho = 1.86e-9, `rho^5` and `rho^6` are
subnormal in float32 and flush to zero, while `P_5` and `P_6` underflow to zero: `s_5 = s_6 = 0 / 0`. The CPU
keeps subnormals, so no CPU run shows it.

A NaN in a source's row fails `G M_A E_{A->B} / r^2 < thr_B` for every `B`, so that leaf could never be a far
source and the walk paired it with every leaf it reached. The geometric MAC never reads `s_n`, which is why it was
unaffected. The ~3,000-entry rows seen in "normal" builds were the same: one-particle leaves at r ~ 25 with radii
3.7e-9 to 1.5e-8. The merger after first pericentre (`restart_final_136.npz`) shows the same pattern in 4 of 5
steps: rows of 4.75M leaves, each at a one-particle leaf with a radius of 3.7e-9 to 1.1e-8.

**The f_b hypothesis is refuted.** f_b is cancellation-free (a sum of positive terms), so it cannot vanish at a
saddle or a disc centre. The carried f_b of the leaf is right. With the floor on (`mac_force_scale_floor=0.5`)
every build's lists are the floor-off lists, row for row.

### The fix: a non-finite `s_n` becomes 1

`s_n` lies in `[0, 1]` whenever `rho` bounds the node, so a non-finite entry is set to 1, its largest value
(`jaccpot/runtime/_walk_criterion.py`). Every finite entry is untouched, so the table is bitwise wherever it was
finite. The mesh's cross walks build their rows with the same function.

At the stalling builds (bfa876c8, from `restart_10.npz`, steps 38-40):

| | longest near row | near entries | far pairs |
| --- | --- | --- | --- |
| A1 without the guard (8ba2d2b8), step 38 | 5,302,664 | 126,404,102 | 304,887,022 |
| A1 with the guard (bfa876c8), step 38 | 169 | 115,645,264 | 304,978,392 |
| A1 with the guard, steps 39, 40 | 166, 188 | 115.6M | |

### Accuracy at step 38's build

One production step from `restart_10.npz` per arm, then the self-gravity against an fp64 direct sum (ferrers3,
softening 1.5e-3) at the new positions: 4096 random targets (the same ids in every arm), the long row's particle,
and the 1024 particles with the smallest carried f_b.

| arm | rel-L2 | median | p99 | max | long-row particle | smallest-f_b 1024, rel-L2 | far pairs | near entries |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dehnen_error`, with the guard | 2.192e-4 | 7.48e-5 | 5.67e-4 | 2.08e-3 | 7.8e-5 | 3.87e-5 | 305.0M | 115.6M |
| `dehnen_error`, without | 2.192e-4 | 7.48e-5 | 5.67e-4 | 2.08e-3 | 6.8e-6 | 3.87e-5 | 304.9M | 126.4M |
| geometric, theta 0.7 | 2.762e-4 | 6.61e-5 | 7.30e-4 | 3.20e-3 | 6.3e-6 | 5.85e-5 | 604.0M | 174.5M |

**The merger after first pericentre** (`restart_final_136.npz`, the same probe and targets, one step):

| arm | rel-L2 | median | p99 | max | smallest-f_b 1024, rel-L2 | far pairs | near entries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dehnen_error`, with the guard | 1.941e-4 | 7.44e-5 | 5.68e-4 | 1.85e-3 | 5.54e-5 | 321.5M | 117.9M |
| `dehnen_error`, without | 1.941e-4 | 7.44e-5 | 5.68e-4 | 1.85e-3 | 5.54e-5 | 321.4M | 127.5M |
| geometric, theta 0.7 | 1.683e-4 | 6.57e-5 | 5.91e-4 | 1.95e-3 | 1.30e-4 | 642.2M | 185.0M |

Here theta 0.7 is 1.15x lower in rel-L2 and 1.13x in the median, while `dehnen_error` is lower in p99 (1.04x), max
(1.05x) and on the smallest-f_b particles (2.3x), from half the far pairs and 64 % of the near entries. Whether it is
"no worse at equal time" needs the HoreKa times; at equal list volume it is not established for the bulk rel-L2.

- The guard leaves the bulk unchanged. The one particle it moves goes from 6.8e-6 (summed directly against 70 %
  of the tree) to 7.8e-5, inside the 1e-4 target.
- **Against the geometric fallback at theta 0.7**, `dehnen_error` is 1.26x lower in rel-L2, 1.29x in p99 and 1.54x
  in max, from half the far pairs and two-thirds of the near entries. The median is 13 % higher.
- Timings at equal lists come from HoreKa (`bench/slurm/a1_dehnen_error_stall.sbatch`).

### Time on a shared card (contended, functional only)

| run | per step |
| --- | --- |
| reported, main, steps 31-40 (#487) | 205-231 s per 10 steps against 9.3 s before; one build's rank up to 124 s |
| main 96122f63, steps 31-40 in one 10-step segment (incl. compile) | 290.0 s |
| A1 safety net (8ba2d2b8), steps 32-40 | 1.19-1.59 s; the 5.3M-row steps 38/39: 1.41 / 1.48 s |
| A1 safety net, merger steps 1362-1365 (4.75M rows) | 1.32-1.62 s |
| A1 with the guard (bfa876c8), steps 39-40 | 1.16 / 1.49 s |

These rows ran with the build probe on and one step per segment, which add their own costs (0.93 s per step is the
1e8 disc's unprobed rate). They show that the stall is gone; they are not timings.

**The whole stalled segment and the merger, with the guard** (bfa876c8, the same probe, contended):

| run | longest near row per build | near entries | per step | peak |
| --- | --- | --- | --- | --- |
| bar disc, steps 31-40 from `restart_3.npz` | 166-190 in all 10 builds | 115.6-115.7M | 1.19-2.39 s | 20.27 GiB |
| merger, steps 1361-1365 from `restart_final_136.npz` | 545-591 in all 5 (without the guard: 4.75M in 4) | 117.9M (127.5M) | 1.27-3.65 s | 20.86 GiB |
| merger with the floor at 0.5, no guard | 4.75M in 4 of 5, as without the floor | 127.5M | | |

### Gates

| gate | result |
| --- | --- |
| small reproduction fixed by each fix | the NaN needs the GPU's flush-to-zero; on CPU: the guard test (inf/NaN injected, finite rows bitwise, eager and jitted), the floor's injected-f_b flood (878 -> 116 of 873 leaves), the hub-row merge sort |
| the stalling 1e8 restart at about the pre-stall wall time | yes, with either change; the stalled steps take what the others take (tables above; contended) |
| accuracy no worse than geometric theta 0.7 at equal time | bar disc: rel-L2 / p99 / max 1.26x / 1.29x / 1.54x lower from half the far pairs; merger: rel-L2 1.15x HIGHER, p99 / max 1.04x / 1.05x lower, from half the lists; equal-time verdict pending HoreKa |
| safety net: CSR arrays bitwise | it is a sort; interpret-mode tests against the NumPy directed sort; CPU identity 36/36 at 79e42475 (the CPU lane does not lower the Pallas placement) |
| the guard: bitwise where finite | CPU identity at bfa876c8: 32/36 identical; the 4 `dehnen_error` variants move their compiled runners' TEXT, every array bitwise (1302 arrays) |
| the floor: bitwise where it does not bind | CPU lane test bitwise; at #487 it never bound |
| S12 mesh `dehnen_error` gates | `tests/distributed/test_mesh_segment_dehnen_error.py` 4/4 (CPU, 2 devices) |
| GPU pins S1, S6, S1p/t/e/x/g, G1 | base main 96122f63 vs A1 bfa876c8, with a base A-vs-A control (bitwise in every pin): S1, S6, S1p, S1t, S1x, S1g, G1 **bitwise**; S1e: forces bitwise, the final state moved, **because the guard replaced real NaN entries** (below) |

Other tests: the numeric goldens (`tests/characterization`) 21/21; `tests/unit/runtime` and `tests/unit/operators`
(not slow) 997 passed, 139 skipped; the new and touched unit files also under `JACCPOT_RUNTIME_TYPECHECK=1`. The
constructor-state golden gained one line, `mac_force_scale_floor: None`.

### S1e attributed: the guard replaced real NaN entries at 2e5

S1e (`bench/dce_pins.py`: clipped Plummer 2e5, `dehnen_error` at 1e-4, 4 steps, both carries) kept its forces
bitwise and moved its final state: 11,675 / 13,595 of 1.2M elements (state / particle carry), rel-L2 3.2e-9 /
6.5e-9. To attribute it, the pin's bench was rerun with every walk table it builds dumped (`dehnen_walk_table`
wrapped). Four arms ran back to back on one card (deterministic ops): main 96122f63; main + the guard alone
(cherry-pick of bfa876c8); the A1 tip; and main again.
- **Main has real NaN entries.** Every traced build (2-14) has 1-5 LIVE one-particle leaves (mass 5.0e-6 = 1/N)
  with a non-finite `s_n`; the eager builds 0-1 (the first force) have none. The rest of the non-finite rows are
  dead nodes (mass 0, threshold +inf), which the walk masks. At build 2:

  | carry | node | radius | non-finite |
  | --- | --- | --- | --- |
  | state | 62855, 67655 | 7.45e-8 | `s_6` (`rho^6` = 1.7e-43 is subnormal, `rho^5` = 2.3e-36 is not) |
  | state | 74323 | 7.3e-12 | `s_4`-`s_6` (`rho^4` is subnormal) |
  | particles | 62854, 79283 | 7.45e-8 | `s_6` |

- **Finite entries stay bitwise.** At build 2, the first traced build, every arm has the same inputs. Every finite
  entry of the table is bitwise between main and the guard (and the tip), and every non-finite one became exactly
  1.0. Inputs first differ at build 3, because the guard changed build 2's lists.
- **The diff is the pin's.** The final states reproduce the recorded pins bitwise in both arms (the same 11,675 /
  13,595 elements); the guard alone and the A1 tip are bitwise to each other; main against main is bitwise.

So S1e moved because the guard bound, which is the intended change. No finite value was rewritten, and no code
change followed. Scripts and dumps: the A1 scratch directory, `s1e/` (`s1e_probe.py`, `run_s1e.sh`, `analyse.py`,
`compare.py`).

### Restoring at 1e8: a 14.9 GiB block the BFC allocator could not place

With one step per segment, main 96122f63 failed to restore `restart_3.npz` on an empty card: "Out of memory while
trying to allocate 14.68GiB" (the runner's temporaries, one block), with the allocator map mostly free but
fragmented. The frozen run's code (eb5b3f37) restored the same state, and so did A1 without the build probe
(peak 20.27 GiB). With the probe, both A1 and eb5b3f37 + A1 failed on 14.92 GiB. `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`
places it. Not investigated further; a 1e8 restore on BFC is placement-sensitive.

### The safety net: long CSR rows are merge-sorted (default)

`directed_csr_pallas` (`jaccpot/pallas/csr_place.py`) orders each row of a directed pair list by an in-row rank:
one program per row, every entry ranked against the row's entries tile by tile. That costs `(n / 32)^2` tile pairs
for a row of `n`. Rows past `RANK_LONG_ROW` (2048) used to have their tiles spread over 256 programs, but each tile
still ranked against the whole row, so a near row of 5.3M leaves cost up to 124 s per list build
(`directed_csr_rank_long_k32`, #487's nsys capture).

Those rows are now **merge-sorted** (`_merge_pass_kernel`):
- `ceil(log2 n)` passes; pass `j` merges sorted runs of `2^j` into runs of `2^(j+1)`.
- Each entry goes to its offset in its own run plus its rank in the partner run, found by a binary search
  (`log2(run) + 1` halvings). Left-run entries count partner entries `<` them, right-run entries `<=`: a stable
  merge, a permutation even if a value repeats.
- The passes ping-pong between the unsorted list and the output and touch only the long rows' slots, so no buffer
  is added. The cost is `O(n log^2 n)` per row.
- It is a sort of each row, so the CSR is the same array, bit for bit, whatever the MAC. The mesh's local walks
  build their CSRs through the same function; its cross lists are sorted by `_lex_sorted` (XLA) and never used the
  rank.

Tests (`tests/unit/runtime/test_near_csr_from_canonical.py`, interpret mode): the existing long-row cases, and a hub
row of 255 entries with rows of 16-33 (an even pass count, a power-of-two row) in batches of 1, 2 and 64 rows over
1, 3 and 5 programs, each against the NumPy directed sort.

### The floor: `mac_force_scale_floor` (opt-in)

Eq (16a) accepts a far pair when its error estimate is below `eps * min_b f_b` of the sink node. A node whose
carried `f_b` is far too small accepts almost nothing: its leaves pair with nearly every leaf directly.

**A certified bound.** For a particle `b` of node `P`, split the other sources into `P` itself and, for each `Q` on
`P`'s ancestor chain (`P` included, the root excluded), `Q`'s sibling `S`. With `rho` a radius bounding a node's
particles about its centre `c`:

    f_b >= G (M_P - m_b) / ((4 rho_P)^2 + eps^2)
         + sum_Q G M_S / ((|c_S - c_Q| + rho_S + 3 rho_Q)^2 + eps^2).

This holds for the fused lane's **estimate** of `f_b`, not only for its exact value:
- Each source `a` reaches `b` through a near pair (exact) or through a far pair `(X -> B)`, which adds
  `G M_X / ((|c_X - c_B| + rho_B)^2 + eps^2)` (eq 16b's far half).
- The walk pairs disjoint nodes only, so `X` and `B` lie in the same two subtrees: both inside `P` (reach
  `<= 4 rho_P`), or `X` inside `S` and `B` inside `Q` (reach `<= |c_S - c_Q| + rho_S + 3 rho_Q`).

`force_scale_floor_nodes` (`jaccpot/runtime/_force_scale_levels.py`) computes it with one
`ancestor_sum_by_level` pass over the sibling terms. `mac_force_scale_floor=f`, `f` in `(0, 1]`, raises each node's
`min_b f_b` to `f` times the bound. That happens in `_flat_walk_criterion`, the one place the threshold is formed;
the local walk and the mesh's cross walks both read it.

- **On a correct estimate the floor never binds**, so eq (16a) decides exactly as without it.
- Where the estimate falls below it, the estimate is wrong.
- The carried `f_b` itself is not changed, so restart files are unchanged. The fingerprint records the floor.

How tight the bound is: on a 1500-particle Plummer cell tree (leaf 16) it sits a median 31x below the exact `f_b`
at the leaves (0.011-0.10 of it). `f_b` is dominated by the nearest neighbours, and the bound places every source
at its node's farthest reach. A floored leaf at `f = 0.5` decides like `adaptive_eps` about 1e-4 / 60 there: a
longer row, not a flood.

Tests:
- `tests/unit/runtime/test_force_scale_floor.py`: a lower bound on the exact `f_b` of every particle under every
  node, brute force, on bucket and cell trees.
- `tests/integration/strict/test_dehnen_error_force_scale_floor.py` (CPU, 4096 Plummer, leaf 16, p4, eps 1e-2):
  - on the lane's own `f_b` the near CSR, the far count and the accelerations are bitwise with and without the
    floor, at the prepare and at a traced rebuild;
  - an `f_b` 1e-9 too small at a few central leaves floods a near row to 878 of 873 live leaves (normal 67);
    floored, the row is 116 and the force keeps its accuracy.

### Not done: the memory of `dehnen_error` off the fused lane (design note)

Two records, neither on the production path (`strict_run_v2` / the runner):
- `compute_accelerations` under `preset="galaxy"`: 4,553 B/particle at 2e5 against 736 with the geometric MAC
  (`docs/production.md`, "The MAC"; a shared A100);
- the pre-S12 distributed lane (`distributed/fmm.py`) at 21M on 6 cards, leaf 512: one 46-64 GiB buffer, which the
  M2L and near-field chunk sizes did not change (46.17 GiB either way).

**Attribution (code reading, not measured).** Both take the dual walk with `adaptive_pair_policy`, which carries
DENSE per-node rows: `total_nodes x max_interactions_per_node` far slots, `num_leaves x max_neighbors_per_leaf`
near slots (`_interaction_cache.py`: 820 MB at leaf 64 for the far rows alone). Under the criterion the far cap
is not set by theta: measured 2026-08-31 (`distributed/cap_presets.py`), it needs `0.5 * num_leaves` at eps 1e-5
and `1.0 * num_leaves` at eps 1e-3. A cap proportional to the leaf count makes the dense rows O(num_leaves^2) in
memory: the fused lane's flat walk, whose lists are CSRs sized by the pairs found, never had that shape.

**Proposed fix.** Route `compute_accelerations` under `dehnen_error` through the flat-walk builder
(`_build_flat_walk_artifacts_strict_streamed` with `walk_criterion=`), as the fused prepare already does, and
retire the dense-row criterion path. **First step:** dump the buffer assignment of the program that OOMs
(`XLA_FLAGS=--xla_dump_to=... --xla_dump_hlo_pass_re=buffer-assignment`) and confirm that the biggest buffer is the
dense far rows. This was not done in A1 (time spent on the stall).

## M2 (CPU part) — the multi-GPU mode as N processes (2026-11-03)

Plan M2, first session (CPU only): `ProductionRunner(mesh=MeshRunConfig(...))` runs as N processes with one device each
(`jax.distributed`) instead of one process driving N devices. Branch `m2/multiprocess-cpu` on main `96122f63`,
head `9da1fedc`.

### What jax 0.11.2 gives on CPU (the spike)
| item | result |
| --- | --- |
| `jax.distributed.initialize` on localhost, gloo collectives | works at 2 and 4 processes x 1 CPU device |
| `psum`/`pmax`/`pmin`, `all_gather`, `all_to_all`, `ppermute` inside `jit(shard_map)` | work; all but float `psum` bitwise the single process's |
| float `psum` (and an eager reduction over a sharded array) | bitwise at 2 processes, NOT at 4: XLA:CPU's in-process all-reduce folds the devices in mesh order (`((a0+a1)+a2)+a3`, checked at 3/4/8), gloo does not (12,656 of 49,152 bytes differ; `sum m x` 9.3e-10) |
| `ragged_all_to_all` | `UNIMPLEMENTED` on XLA:CPU (one process or several); yggdrax's `"buf"` exchange (all_gather based, the CPU tests' path already) is used, bitwise across processes |
| MPI collectives | need an MPItrampoline build of MPI (none here); gloo is the CI choice |
| `np.asarray` of a sharded array spanning other processes | raises; replicated (`P()`) arrays read locally |

### What was built
| piece | what |
| --- | --- |
| `jaccpot/distributed/multiprocess.py` | `initialize` (explicit coordinator or the SLURM environment; gloo on CPU), `process_info`, `global_mesh` (devices in process order), `host_value`, `allgather_object` / `allgather_by_device`, `host_copy` / `put_host_copy` (per-process shards), `mesh_sum` / `mesh_moments` (float sums folded in mesh order: bitwise the single process at any process count), `local_id_range` / `ProcessLocalParticles` (each process reads only its id range), `refuse_in_multiprocess`. Every helper is the old code path when `jax.process_count() == 1` |
| `rollout.decompose_local` | the decomposition from each process's id range: the same per-device layout and the same repartition program as `decompose` from one host |
| `template_force.prepare_mesh_force` | under several processes the caps-only prepares run only on a process's own devices; the box (exact live extents), auto leaf capacity, level plans, walk caps, live leaves and the upward depth bound are gathered and merged in mesh order; the template is device 0's (its leaf shapes gathered; static leaves checked) |
| segment runner and the runner's mesh mode | host reads through `host_value`, the retry's host copy per process, the energy's sums by `mesh_sum`, the moments by `mesh_moments`, `run(ProcessLocalParticles(...))`; restart files REFUSED under several processes until M3 (orbax, U5), `restart_from=<RunResult>` works |

Usage, one process per device:

```python
from jaccpot.distributed import multiprocess as mp
mp.initialize()                          # SLURM env, or (coordinator, num_processes, process_id)
mesh = mp.global_mesh("gpus")
lo, hi = mp.local_id_range(N, mesh)      # read only these ids of the IC
runner = ProductionRunner(solver, settings=..., mesh=MeshRunConfig(mesh, ...))
res = runner.run(mp.ProcessLocalParticles(state[lo:hi], masses[lo:hi], N), None, dt=dt, num_steps=k)
```

### Gates (CPU; N processes x 1 device vs 1 process x N forced devices, same host)
`tests/distributed/test_multiprocess_runtime.py` (marker `distributed_multiprocess`, job
`test-distributed-multiprocess`); every side in fresh child processes (`_multiprocess_harness.py`).

| case | 2 processes | 4 processes |
| --- | --- | --- |
| helpers: `mesh_sum`, `mesh_moments`, `host_value`, `decompose_local` == `decompose` | bitwise | bitwise |
| runner + `DirectMeshForce` (N = 256, f64): migrating and not, a repartition before every segment, energy and moments at every edge, continued in memory; restart files refused under several processes | bitwise (state, acceleration, owner, every edge's diagnostics and plan record, every segment report) | bitwise |
| runner + `FusedMeshForce` `dehnen`, migrating, cross capacity undersized (segment 1 re-planned and retried), energy at every edge | bitwise | bitwise (manual) |
| runner + `FusedMeshForce` `dehnen_error`, migrating, cross capacity undersized, f_b carried | bitwise | bitwise (manual) |
| runner + `FusedMeshForce` `dehnen_error` on the A3 collapse start: TREE re-plan (leaf capacity 44 -> 1024), the library's cross sizing at the start and the drift | bitwise | -- (the start does not fit 44 leaves on four devices, in one process either) |

The file end to end (shared 64-core host, `-n 4`, manual cases on): 8 passed in 35 min (fused cases 11-24 min
each, both sides); the CI subset under `JACCPOT_RUNTIME_TYPECHECK=1`: 6 passed, 2 skipped (manual). CI job
`test-distributed-multiprocess` (path-gated, cap 60 min, an estimate) runs the fast cases at 2 and 4 processes
and the two 2-process fused cases.

Mutation: `mesh_sum` back to a plain `psum` -> the 4-process helper and direct-runner gates fail (`mesh_sum`,
`moment_mass_moment`, `moment_momentum` differ; the runner's edge diagnostics differ); 2 processes still pass (why
the gate runs 4).

### Default paths
- Single-card CPU program identity: 36/36 identical against `m2-base-cpu` (main `96122f63`).
- The single-process MESH programs (every top-level jitted program the runner runs on two forced CPU devices,
  `lower().as_text(debug_info=False)`, captured with `provider_hlo_identity`'s jit wrapper) and their results,
  main `96122f63` vs the branch: direct force (migrating and not, energy at every edge) 49/49, fused `dehnen`
  migrating 54/54, non-migrating 47/47, fused `dehnen_error` with a tree re-plan 87/87: text-identical, arrays
  identical.
- Existing distributed tiers on the branch (two forced CPU devices): `distributed-tier` 96 passed (7 skipped:
  need 4 devices / opt-in), `distributed-criterion` 8, `distributed-segment` 10, `distributed-segment-criterion`
  4, `distributed-segment-migration` 2, `distributed-segment-runner` 5; no failure.

### Open (for M2 session 2 / M3)
- GPU: NCCL `ragged_all_to_all` across processes is untested; with jax 0.11.2 the cross-process fallback is the
  NCCL send/recv path that needs `use_barrier_with_nccl=false` (#398). First check on one HoreKa node.
- `RunResult.state` (and `handle.gather()`) gathers the whole state to every process (`process_allgather`): test
  scale only; per-rank output belongs to M3.
- Callbacks run on every process; a callback that writes must check `jax.process_index() == 0`.
- The stacked route (`setup_fused_force`) and `distributed/fmm.py` stay single-process.

## I1 — nornax owns the integration scheme (2026-11-05)

Decision U1: nornax owns the integration schemes, jaccpot the force, decomposition and orchestration, Odisseo the
science. The design note (`mn-scratch/i1/I1-DESIGN.md`) is summarised in `docs/production.md`, "Integration schemes".

### What was built
- **jaccpot.** `jaccpot/scheme.py` holds the structural protocol `SplitStepScheme`: `rowwise`, `key()`, `open_step`,
  `close_step`. Each step of the particle carry's scan is cut at its one force evaluation:
  `open_step` -> refresh + self + external force -> `close_step`. Refresh, verdict, re-plan/retry, restart and
  diagnostics stay the runner's. jaccpot does not import nornax.
  - `strict_run_v2(scheme=None)`: `None` takes the old inline expressions, so default programs are unchanged.
  - `ProductionRunner(scheme=None)` and `production.run(scheme=)`.
  - A scheme's `key()` joins the scan's compile key.
  - A non-rowwise scheme keeps the input order under the tree-order carry.
  - A restart file records `meta["scheme"]` only when a scheme is set; a restore under another scheme reports
    `"scheme"`.
  - Refused: `mesh=` (I2), the gradient mode, and `carry="state"`.
- **nornax.** `nornax.schemes.LeapfrogKDK`. Its expressions, `drift_positions` and `kick_velocities`, now live in
  one place, `leapfrog_kdk.py`, and `leapfrog_kdk_step` / `leapfrog_kdk_rollout` call them too.
- **Odisseo.** `SimulationConfig.fmm_scheme` takes `None`, `"nornax_kdk"` or a scheme object.
  - It selects lane `fmm_runner` for fixed-step `FMM_ACC` runs.
  - The block-step, differentiable and older mesh lanes refuse it; `fmm_mesh_runner` refuses it until I2.
  - Odisseo's FMM leapfrog loop (`integrate_leapfrog_jaccpot_active`, lane `fmm_forward` with fixed steps) and the
    older mesh lane's KDK are deprecated with a `DeprecationWarning` naming the replacement. Through `integrate()`
    the warning is issued once, on the caller's line.
- **Bench.** `fused_memory_budget.py --runner-scheme nornax_kdk`; pins `R1pN` / `R1xN` in `dce_pins.py`.
- **CI.** Shard `integration-strict-scheme`.

### Gates (CPU unless noted)
- Identity: `i1-tip-cpu` vs `w2-base-cpu`, 36/36 variants identical.
- `tests/integration/strict/test_runner_scheme.py`: 9 passed. The KDK scheme is the built-in runner bit for bit
  (literal and traced parameters, an NFW halo with traced parameters, the tree-order carry). A dt/2 scheme is the
  built-in at dt/2. Restart and refusals pass. The shard ran 12.4 min at `-n 2` with coverage on a loaded host.
- nornax: `ProductionRunner(scheme=LeapfrogKDK())` is the built-in runner bit for bit (static, traced, halo).
  - The scheme's scan over the direct sum is `leapfrog_kdk_rollout` bit for bit.
  - Suite: 360 passed, coverage 95.5 %; type-check smoke 61 passed.
- Odisseo: `fmm_scheme="nornax_kdk"` is the built-in runner lane bit for bit (N = 512, NFW halo, 6 steps).
  - Suite (`-m 'not slow'`): 291 passed, 11 skipped (2-device), 2 xfailed.
  - The 2-device mesh files on forced devices: 43 passed.
- Notebook `jaccpot_fmm_first_integration.ipynb`: cells 1-4, 7 and 9 were run on the base stack and the I1 stack.
  The 4 arrays are bitwise equal and the printed numbers are identical. Cells 7 and 9 fail on both stacks
  (Odisseo's `E_pot` has no `FMM_ACC` branch; this predates I1).
- Also passed: `test_shards.py check`; `tests/unit` top level 517 passed; bench-tooling 117; gated
  `distributed-segment-runner` 5 (2 forced devices).
- GPU: one shared A100, functional only (`dce_pins.py`, N = 2e5, deterministic ops). R1p and R1x A-vs-A are
  bitwise. `R1pN` and `R1xN` (the runner with nornax's `LeapfrogKDK`) are bitwise to the built-in's R1p and R1x:
  states and forces, both arms of R1x.

### Open
- I2: the mesh runner's seam.
- U6: the hybrid Hermite scheme (`rowwise=False`; it needs a per-step scheme carry and the `target_points=`
  exclusion).
- Retiring the deprecated loops.
- Whether the default fixed-step FMM lane should switch to the runner. That changes numerics and is a user decision.

## M2 (GPU part) — N processes x 1 GPU on NCCL (2026-11-05)

Plan M2, second session: the multi-process runtime of the CPU part on GPUs. Branch `m2/multiprocess-gpu` on main
`f868c393`. Local cards are shared A100-PCIE-40GB (selected by autocvd, other users' jobs on them): functional,
bitwise and memory results only, no timing. The timing and the 4-GPU / 2-node runs are prepared for HoreKa
(`bench/slurm/m2g/`, not run).

### What jax 0.11.2 gives on GPUs across processes (the spike)
`tests/distributed/_multiprocess_scenarios.collectives` inside `jit(shard_map)`, three calls each with new data,
2 processes x 1 GPU against 1 process x 2 GPUs on the same two cards, each side run twice:

| XLA flags | result |
| --- | --- |
| production (`RAGGED_EXCHANGE_XLA_FLAGS`: NCCL send/recv ragged fallback, `use_barrier_with_nccl=false`) | `psum` f32/f64/int32, `pmax`, `pmin`, `all_gather`, `all_to_all`, `ppermute`, `jax.lax.ragged_all_to_all` (600-900 rows of 56 f32 per message, above the #398 threshold) and yggdrax's `ragged_all_to_all_exchange` all equal the host references, both arms bitwise, each arm A-vs-A clean; the same with each process given its card by its own `CUDA_VISIBLE_DEVICES` (as SLURM's per-task binding does) |
| XLA's default (the one-shot kernel, no ragged flags) | works across processes too, the same bits |
| the NCCL fallback with `use_barrier_with_nccl` at its 0.11 default (on) | **hangs** across processes, as in one process (#398): 300 s, every main thread in `BlockUntilReady` (py-spy, no compile frames). yggdrax's exchange refuses this flag set before tracing (`RuntimeError`) |

So the flag is needed across processes exactly as in one process, and the production flag set already has it. The
mesh evaluator's warning (`fused._warn_if_slow_ragged_exchange`) covers both; `bench/multiprocess_gate.py` sets the
flags like every multi-GPU bench.

**Float all-reduce order.** At two devices every order is `a0 + a1`, so the 2-GPU gate cannot see an order
difference (the probe reports the `psum` against the mesh-order fold: 0 differing elements, both arms). The force
path has no float all-reduce (verdicts are integer `pmax`/`pmin`, the exchanges move rows). The diagnostics' float
sums are `mesh_sum`/`mesh_moments`: a psum (XLA's partitioned reduction, NCCL on a GPU) in one process, the
all-gathered mesh-order fold under several. From three GPUs on, NCCL's in-process order may differ from that fold
in the last bits of the moments and the energy (never of the state or forces); the HoreKa `node1` job's
collectives probe measures it at 4 (`psum_*_mesh_order_differs`) and its compares report the edges' diagnostics
separately from the arrays. If it differs, the fix is to fold in mesh order in one process too (a change to the
single-process mesh programs: the user's call).

### Gates (2 processes x 1 GPU vs 1 process x 2 GPUs, the same cards, production flags unless noted)
`tests/distributed/test_multiprocess_gpu.py` (manual: `JACCPOT_RUN_MANUAL_TESTS=1`, two cards in
`CUDA_VISIBLE_DEVICES`); `_multiprocess_harness.run_side(..., platform="gpu")`: process `r` owns card `r`
(`JAX_LOCAL_DEVICE_IDS`). Every case: each side twice (A-vs-A), then the sides.

| case | A-vs-A (sp / mp) | sp vs mp |
| --- | --- | --- |
| helpers (`mesh_sum`, `mesh_moments`, `host_value`, `decompose_local` == `decompose`) | clean / clean | bitwise |
| runner + `DirectMeshForce` (N = 256, f64), migrating and not, energy and moments at every edge | clean / clean | bitwise (only `restart_refused` differs, by design) |
| runner + fused `dehnen`, N = 2048, the real GPU lane, migrating, undersized cross capacity (segment 1 re-planned and retried) | clean / clean | bitwise |
| runner + fused `dehnen_error`, A3 collapse start: TREE re-plan (leaf capacity 44 -> 1024, retries [1, 1]), f_b carried, energy at every edge | clean / clean | bitwise (state, acceleration, owner, force_scale, every edge, every report) |
| `bench/multiprocess_gate.py` (the HoreKa entry point) under a simulated SLURM step, `initialize()` without arguments: galaxy preset `dehnen_error`, Plummer 1e5 per GPU, 2 segments x 2 steps, migration, energy at every edge, 2 repeats in the process; every task seeing both cards (jax's `[SLURM_LOCALID]`) and each task bound to one card (`slurm_bound_device_ids` -> `[0]`) | clean (sp) | bitwise, both bindings (state, acceleration, owner, every edge's diagnostics) |
| runner + fused `dehnen`, galaxy preset (leaf 64, p6), Plummer 2e5 per GPU (4e5), 3 segments x 2 steps, migration (4x the equilibrium speeds), cross capacity far too small at the start (segment 1: 3 re-plans), energy at every edge | clean / clean | bitwise |
| the same with `dehnen_error` (1e-4), f_b carried | NOT clean on the production flags: one float32 velocity component differs by 1 ulp from run to run (one process and two alike), so the edges' moments differ in the last bits; clean with `--xla_gpu_deterministic_ops=true` | bitwise with deterministic ops (both pairs); on the production flags each sp/mp pair happened to agree bitwise too |

Only `dehnen_error` at 2e5 per GPU needed `--xla_gpu_deterministic_ops=true` for its A-vs-A control: the lane's
known run-to-run variation on the production flags (`bench/multigpu_runner_gate.py` attributes it to the
`dehnen_error` seed's scatter-add and the cross field's), present in one process, not something the processes add.

**Discarded attempts.** A segment attempt that overflows a capacity is re-planned and rerun. The counts it reported
past the overflowing buffer (`SegmentReport.failures[*]["needs"]`: 2 of 25 cross-field counters, at 4e5) vary from
run to run in one process and two, also with deterministic ops (a kernel's slot counting past capacity, not an XLA
scatter). Their flag bits, the kept attempt's needs and the re-planned capacities are bitwise; the gate compares
those (`test_multiprocess_gpu._kept`).

**Memory** (peak bytes in use per card, `memory_stats`): fused 2e5 per GPU 0.136 GiB (`dehnen`) and 0.262 GiB
(`dehnen_error`, deterministic ops), the same in both arms (process 0's card in the mp arm); the HoreKa entry point at
1e5 per GPU (`dehnen_error`) 0.14 GiB. Contended cards: no timings.

### What was built
| piece | what |
| --- | --- |
| `multiprocess.initialize` | under SLURM with nothing naming the devices and ONE card in `CUDA_VISIBLE_DEVICES` (`--gpus-per-task=1` with binding), passes `local_device_ids=[0]` (`slurm_bound_device_ids`): jax's own rule, `[SLURM_LOCALID]`, names a missing device from local rank 1 on. Otherwise jax's rule (right when every task sees the node's cards) |
| `SnapshotWriter`, `DiagnosticsLog` | write their files on process 0 only (every process still takes part in the state gather, a collective, and keeps the rows); `all_processes=True` writes on each. One process: unchanged |
| `_multiprocess_harness` | `platform="gpu"` (the first n cards of `CUDA_VISIBLE_DEVICES`; `gpu_binding="local_device_ids"` or `"visible"`), `launcher="slurm"` (a simulated SLURM step, `initialize()` without arguments), a py-spy dump of every child before a timeout kills it |
| scenarios | `collectives`, `fused_runner_large` (the galaxy preset at 2e5 per device, Plummer), `writers`, `launch`; `fused_runner` runs the real lane on a GPU |
| `bench/multiprocess_gate.py`, `bench/slurm/m2g/` | the HoreKa jobs: `node1` (1 node: 4 processes x 1 GPU vs 1 x 4, both MACs at 2e6 per GPU, A-vs-A, per-task binding arm, deterministic-ops arms; bitwise + timing) and `node2` (2 nodes x 4: functional, A-vs-A across nodes, collectives at 8, the barrier probe across nodes); README documents jax's SLURM detection and the binding choice |

CPU tests (in the `distributed-multiprocess` CI shard): the writers write on process 0 only (and on each with
`all_processes=True`); `initialize()` from a simulated SLURM step; `slurm_bound_device_ids` (4 cases).

### Default paths
- Single-card CPU program identity: `w2-base-cpu` vs `m2g-tip-cpu` (`4300467e`): **36/36 identical**.
- `distributed-multiprocess` shard (CPU): 12 passed, 2 skipped (manual) in 45 min (shared host); with
  `JACCPOT_RUNTIME_TYPECHECK=1` the fast cases 10 passed. `tests/unit` top level: 517 passed, 32 skipped.
  `test_shards.py check` rc 0.

### Open
- Edge 0's energy is `None` when the start's cross capacity is too small (the fused `dehnen` 2048 case): the
  potential at step 0 reports not-ok and is not re-planned, so there is no `E0` and `energy_drift` stays `None` for
  the whole run. Also in one process; not changed here.
- The float reduction order at >= 3 GPUs (above): HoreKa `node1`.
- `RunResult.state` and the snapshots gather the whole state to every process (M3: per-rank output).

## H0 — the HoreKa single-node baseline bundle (2026-11-04)

`bench/slurm/h0/` runs every H0 measurement on dedicated HoreKa nodes:
- the multi-GPU scaling matrix (#480), on 1/2/4 cards;
- the S9 gradient timing and the S1b Mosaic L2P timing;
- the L1-L3 pins with an A-vs-A control;
- the cost of yggdrax #104's cummax at 1e8 (A-B-A-B-A);
- `cell_min_level="auto"` vs 8.

Start with `bench/slurm/h0/README.md`. In short: edit `site.env`, then run `setup_env.sh`, `submit_all.sh` and
`collect.sh`. The bundle pins yggdrax, nornax, mimirax and the exact runtime wheels, and stamps every row with its
provenance.

It was checked on CPU only: every job script ran locally at tiny N with the exact arguments it passes. That run found
one bench bug, now fixed: `multigpu_segment_gate.py` crashed on every one-device mesh row.

Expected cost: about 19 node-hours.

## A2 — diagnostics and domain robustness (2026-11-05)

Branches `a2/diagnostics-robustness` in jaccpot and yggdrax (from f868c393 / 235e1f9). Four items: #488, the
64-level tree bound, an IC guard, and a prototype escaper policy.

### #488: why the energy diagnostic went blind

The force and the energy disagreed about fitting because they ran on **different level-loop widths**. The level loops'
batch width lives in a process-level registry (`_level_shapes`). Programs read it when they TRACE, and no provider
key holds it. A tree re-plan that widens only that batch therefore clears the engine's compiled runners
(`_replan_tree_from_needs`), so the force scans retrace with the new width. `ProductionRunner._energy_programs`
was never cleared: the energy program compiled before the re-plan kept the old width and reported
`energy_ok=False` at every edge whose widest level outgrew it. The force segments ran on with `retries 0`.

- The 4.7e7 merger (`c_fp64_t06`, geometric 0.6, fp64 state) shows it: one retry at step 3040 with leaf, far and
  near caps unchanged (a width-only re-plan), then blind energy from step 6150 to the end. In a fresh process the
  energy fitted (the issue's workaround).
- CPU reproduction, N = 512: the cusp of `test_strict_runners_on_provider.py` drifting a tenth of the way into four
  blobs. The energy program was compiled on the cusp at rest (width 18); the first segment re-planned the width to 33
  with the leaf capacity at 128. Before the fix, steps 1 and 2 were blind (step 2 with retries 0). Step 3's leaf
  re-plan (128 -> 1024, a new key) made the energy fit again.

**The fix** (`jaccpot/production.py`):

- The energy key holds the level registry (`_level_registry_key`). The same reproduction then has no blind row, and
  its energies equal a fresh run's bitwise (1130.512663611531, 1130.0507250483863, 1129.9680966311746).
- A rebuild that still does not fit is re-planned for the energy alone, at most twice (`_segment_retry_mode`):
  - the rebuild's needs come from a small program;
  - `FusedForceProvider.replanned(caps_only=True)` re-plans at the edge's positions;
  - the energy program on the grown provider is compiled ahead of time;
  - all of it runs inside `_trace_state_kept`, so the force's plan, compiled runners, walk caps and level registry
    come back unchanged;
  - the grown program is kept for the plan and tried first at later edges.

  Checked with the registry forced to width 4: one re-plan, `W` bitwise the full plan's (-12.166413280530833), and
  the registry, cache object and validated caps all unchanged.
- A row that is still blind (retry off, a named cap) warns: `RuntimeWarning` with the time, the overflow bits and the
  re-plans tried. A failed `E0` warns too. Rows carry `energy_replans`.
- The mesh energy (`MeshSegmentRunner.potential_energy`) shares its program cache with the mesh force and gets the
  same warning, but has no energy-only re-plan (open).

### Tree depth > 64: now loud

yggdrax's level tables hold `MAX_TREE_LEVELS` = 64 levels and `num_levels` is clipped to them. A node deeper than
that is in no level, so every level-ordered sweep (M2M, L2L) skipped it **silently**. It happens when many leaves
share one level-21 cell, i.e. a box set by far outliers. Before this change:

| path | before | now |
|---|---|---|
| yggdrax eager build | clipped, silent | `TreeDepthError` naming the box, the level-21 cell, the farthest particle and the 99.9th-percentile radius (`domain_report`) |
| jaccpot eager prepare (jitted tree build) | level loops bounded at `min(64, depth + 8)`, silent | `_resolve_upward_num_levels` raises `TreeDepthError` (box and cell from the tree's bounds) |
| fused scan | flagged, then a wasted re-plan (an eager re-prepare at 1e8), then an error that said "raise leaf_capacity" | never re-planned past 64 (`_segment_retry_mode`); the error says it is yggdrax's bound, with the box and the outliers at the segment's start |

Traced builds are unchanged (no program changes): their callers fold the depth into their verdicts, as jaccpot's
rebuild already did.

### The IC guard

`jaccpot.domain.domain_check` gives the Morton box, the level-21 cell, the farthest particle, the 99.9th-percentile
radius about the median centre, the box's dominance over the bulk and a depth bound. The depth bound is
`64 + ceil(log2 leaves)`, where `leaves` is the densest level-21 cell divided by the leaf size: a tree of unique cell
codes has at most 64 levels, and the leaves of one wide cell add a balanced subtree below it. The densest cell is
estimated on the subsample, scaled by its stride, from three hits up. The densest cell need not be at the centre.

| IC (1e8 Sgr, seed 5) | box | level-21 cell | box / bulk diameter | densest level-21 cell | depth bound | observed depth |
|---|---|---|---|---|---|---|
| unclipped (this session, `--r-clip 1e12`) | 45.7 Mpc | 21.8 pc | 25 | ~665 (17 at the median centre) | 68 | 68 (`smoke_1e8_deep`) |
| production clip (800 / 100 kpc) | 1.60 Mpc | 0.76 pc | 1.11 | 1 | 64 | fits |

On the unclipped IC the guard warns, naming particle #27382923 at 29.7 Mpc and the 467 particles beyond 10x the
99.9 % radius (916 kpc).

Cost: device reductions in one jitted program plus a strided subsample of 2^20 rows. It allocates no
[N]-sized temporaries.

`ProductionRunner` runs it before the first segment as `SegmentConfig(domain_guard=...)`:

- `"warn"` (default) warns when the box is more than 20x the bulk's diameter;
- `"error"` refuses the run;
- `"off"` skips the guard.

It reads the state only. A `TreeDepthError` from the first prepare is re-raised with the guard's report. The depth
itself is always checked by the prepare: the guard's bound is advisory.

### The escaper policy (prototype, opt-in): `jaccpot.escapers`

**Option (a): escapers leave the tree.** Option (b), clamping the Morton box and treating outliers against the root,
was rejected. A clamped outlier still lands in a boundary leaf whose particle-range geometry spans the outlier's true
distance: the near list floods and the corner leaves mix with real particles.

- **Static shapes.** The N rows stay, in an internal order with the `n_b` tree rows first and the escapers last.
  - The tree is built with `FusedForceProvider.rebuild(num_valid=n_b)`, the multi-GPU lane's capacity-padded cut
    (rows past `n_b` are in no leaf), with `n_b` traced.
  - Escaper rows are mapped onto row 0 with mass 0, so the inferred box is the bound rows'.
  - A changed split is a row permutation between segments, and the provider is kept. Nothing compiles: one scan and
    one edge program over the whole test.
- **Hamiltonian** `H = K + W_tree + W_c`, with `W_c = -G M_b sum_e m_e / |x_e - X_b|`:
  - an escaper feels `-G M_b (x_e - X_b)/|..|^3`;
  - every tree row feels the FMM force plus the same uniform `G sum_e m_e (x_e - X_b)/|..|^3`;
  - the pair forces are equal and opposite, so momentum holds up to the FMM's own error;
  - it is evaluated every step, so the leapfrog conserves `E` as it does without the split.
  - Neglected: the escapers' tidal field across the system (relative `(r_b / r_e)^2`) and their mutual gravity
    (O(K^2); far tails are tracers).
- **The ledger.** A transfer changes the representation of the same phase-space state, and the edge records that
  jump. `E - sum(jumps)` is the conserved quantity. Rows also carry the escapers' kinetic energy and `W_c` (what they
  carried away). An escaper returns inside `r_return` (default `r_escape / 2`).

CPU gate: a Plummer sphere (N = 1024) plus 8 fliers at 3x escape speed, `r_escape` 5, dt 1e-3, 200 steps, CSR
interpret, leaf 16, p 4, theta 0.6. A control is the same run with no split.

| run | max \|ΔE/E0\| | max \|ΔP\| / Σ m\|v\| |
|---|---|---|
| with the split (`E - Σ jumps`) | 1.19e-5 | 3.3e-8 |
| control | 8.7e-6 | 2.3e-8 |

Both sit at the FMM/fp32 noise floor, which is the same at dt 5e-3.

Scope of the prototype:

- `EscaperRunner` is its own segment loop: a jitted KDK scan per segment, with `replanned` on an overflow.
- One card, the geometric MAC, no external field. `ProductionRunner`'s restarts, callbacks and mesh mode are not
  wired in.
- Production wiring (the `n_b` cut and the coupling inside `strict_run_v2`'s scan, as a gated program variant) is
  the follow-up. **Done in wave 3 (ESC, below):** `ProductionRunner(escapers=...)`; `EscaperRunner` is removed.

**Sgr gate (local A100, shared card: functional only).** The 1e8 light-Sgr IC was re-assembled with the same seed
and inputs but **no clip** (`assemble_sgr.py --r-clip 1e12 --r-clip-sgr 1e12`: 100,000,001 rows, a 45.7 Mpc box).
A downsampled copy keeps every original tail particle beyond 800 kpc of the origin (182,859) plus a stride-49 sample
of the rest, with the sample's masses scaled up: 2,219,944 rows.

It ran on the escaper path with `r_escape` 80 (800 kpc), the production settings (geometric MAC theta 0.7,
softening 1.5e-3, dt 1e-3) and 2 segments of 5 steps:

- 182,828 to 182,833 escapers, no clip;
- every row fitted;
- `E - Σ jumps` within 5.6e-7 of `E0`;
- total momentum moved by 4.6e-8, absolute.

The full 1e8 unclipped run needs a free 40 GB card. It was queued when this was written
(`mn-scratch/a2/sgr/gpu_gate.sh`).

### Gates

- Identity: `provider_hlo_identity compare w2-base-cpu a2-tip-cpu` is **36/36**.
- CI shards run locally with CI's arguments, all rc 0: `unit`, `integration-strict-a2` (new), `integration-strict`,
  `integration-strict-runner`, and `distributed-segment-runner` (2 forced devices).
- The partition check covers 2816 tests.

## ESC — escapers in `ProductionRunner` (wave 3, 2026-11-06)

`ProductionRunner(solver, ..., escapers=EscaperPolicy(r_escape, r_return=None, every=1))` runs a system with far
tails unclipped through the normal runner path (`jaccpot/escapers.py`, `jaccpot/production.py`,
`strict_run_v2(..., live_rows=)`). Opt-in; the default path is unchanged (identity 36/36, below). User decisions
(2026-11-05): the escapers are in the runner, opt-in, and do not attract each other.

### What was built

- **The split in the scan.** `StepParams` gained an optional traced `live` (the tree's row count; `None` leaves the
  pytree, and every program, as before). With it every refresh in `strict_run_v2`'s particle carry (the start
  program and each step) builds the tree over the first `n_b` rows: `rebuild(num_valid=n_b)` (the mesh lane's
  capacity-padded cut: the rows past it get the sentinel Morton code and are in no leaf), in the box of the tree rows
  alone (`live_bounds`: the escapers folded onto row 0 inside the min/max reductions, no second `[N, 3]` array),
  with the escapers' masses zero. The self-acceleration is the tree force on the tree rows plus the monopole
  coupling (`coupling`): escapers feel `-G M_b (x - X_b)/r^3`, every tree row the equal and opposite uniform pull.
  An `optimization_barrier` orders the coupling after the tree force (measured below: otherwise its rows sat in the
  arena through the force's peak). The eager caps-only prepare and the segment retry's re-prepare build the same
  tree (`eager_live_rows`, `replanned(bounds=)`, new kwarg, default `None`).
- **`dehnen_error`.** The f_b carry is threaded unchanged; an escaper row's f_b is its coupling's magnitude, so a
  returning particle starts its thresholds from its monopole force.
- **The runner.** The rows are kept in an internal order (tree rows first; the order on the host). At the start,
  and at every `every`-th edge, `membership` (one jitted pass) finds who crosses `r_escape` outward or `r_return`
  (default `r_escape / 2`) inward. A change permutes the state, masses and f_b, drops the handle's start force (the
  next segment's start program evaluates it in the new split: `strict_run_v2` now continues a handle whose
  `self_acceleration` is `None` that way) and, with the energy on, evaluates the potential on both sides: the
  **ledger**. While nobody escapes the segments run the default programs.
- **Diagnostics rows:** `n_escapers`, `escaper_transfers`, `escaper_jump`, `escaper_ledger`; with the energy
  `potential_couple` (`W_c`), `kinetic_escapers`, `energy_ledgered = E - ledger` and `energy_ledgered_drift`.
  `energy` includes `W_c`; `energy_drift` stays raw.
- **Row order.** Callbacks (`SegmentEdge.state/masses/self_acceleration`), `SnapshotWriter`, restart files and
  `RunResult.state/masses` are in the caller's order (one gather at the edge, only when a callback exists, and one
  at the end). The in-scan `step_callback` and the external field see the internal order (documented: a
  per-particle field works, one with per-row parameters does not).
- **Restarts** write every array in the caller's rows plus `escaper_order` and `meta["escapers"]` (policy,
  `n_bound`, ledger, transfers); `_restore` re-orders and prepares the split's tree. A run continues only in its own
  escaper mode (`ValueError` otherwise). The fingerprint gains `escapers` only when the policy is set.
- **Refused:** `mesh=` (ValueError naming what it would take: escaper rows outside the equal-count domain cuts, the
  coupling's centre of mass as one more all-reduce per step, a split-aware migration and restart), the gradient
  mode (`gradient_plan`/`value_and_grad`), a non-`EscaperPolicy`. `scheme=` works (the split changes only the
  force; not separately gated beyond the scheme seam's own tests).
- **The IC guard** (`domain_guard(escapers=)`) names the option and a radius to start from (ten times the 99.9 %
  radius); with escapers on it checks the tree rows and says to lower `r_escape` if they still dominate.
- **`EscaperRunner` removed** (one home for the logic: the runner). It was unreleased (only on `mn/wave2`).

### Gates

1. **CPU synthetic** (`tests/integration/strict/test_escapers.py`; N = 1024 Plummer clipped at 3, 8 fliers at 6-12x
   the escape speed just inside `r_escape` 5, one particle at 5.3 falling in past `r_return` 4.5; CSR interpret,
   leaf 16, p4, theta 0.6, dt 1e-3, 6 x 10 steps, energy every edge; numbers from `mn-scratch/esc/exp/gate1.py`):
   escapers 1, 1, 2, 4, 9, 9, 8 (transfers at four edges, the returner back at the last).

   | MAC | max \|`energy_ledgered_drift`\| | no-split control \|dE/E0\| | raw \|dE/E0\| | largest jump / \|E0\| | max momentum drift (control) |
   |---|---|---|---|---|---|
   | `dehnen` | 1.47e-5 | 1.56e-5 | 5.2e-5 | 5.3e-5 | 4.3e-9 (7.8e-9) |
   | `dehnen_error` | 1.7e-7 | 2.3e-7 | 5.9e-5 | 5.3e-5 | 7.6e-9 (7.3e-9) |

   The ledgered energy sits at the control's own noise; without the ledger the jumps dominate. One split scan
   program and one energy program across every split (the row count is traced). A restart written mid-run and a
   chained run (`restart_from=RunResult`) equal the run in one go bitwise, ledger included; the callbacks and the
   result see the caller's rows. Tests: 9 in shard `integration-strict-esc` (new; one worker per MAC).
2. **Nobody escaping == escapers off, bitwise** (state and every diagnostics field the off run has): by
   construction (the default programs run while `n_b == N`), and also for the split program itself:
   `strict_run_v2(live_rows=N)` against `live_rows=None`, 2 x 10 steps, state and self-acceleration bitwise under
   `dehnen` and `dehnen_error` (`exp/forced_split.py`).
3. **GPU, 2.2e6 unclipped Sgr** (A2's downsample: all 182,859 tail particles plus a stride-49 bulk; preset galaxy,
   softening 1.5e-3, dt 1e-3, `r_escape` 80; card 4, shared host, functional only):
   - geometric theta 0.7, 2 x 5 steps: escapers 182,832 -> 182,834, `energy_ledgered_drift` -5.0e-7, momentum
     drift 1.7e-10, retries 0 (A2's prototype: 182,828 -> 182,833, within 5.6e-7; the runner's start re-checks the
     membership from the tree rows' own centre, hence the few more at step 0);
   - `dehnen_error` (the preset's MAC), 4 x 5 steps: 182,832 -> 182,837, |ledgered drift| <= 5.8e-7, momentum
     drift 3.8e-10, retries 0.
4. **GPU, the FULL 1e8 unclipped Sgr** (100,000,001 rows, 45.7 Mpc box; the plain prepare raises `TreeDepthError`
   at 68 levels, A2) through `ProductionRunner(escapers=EscaperPolicy(80))`, geometric theta 0.7, 3 x 5 steps,
   `donate_input`, energy every edge, one A100 40 GB (card 4), `XLA_PYTHON_CLIENT_PREALLOCATE=true`, fraction 0.95:
   runs, no retry; escapers 182,878 -> 182,889; `energy_ledgered_drift` <= 2.7e-7; momentum drift 2.9e-10.

   | 1e8, same card and environment | peak | B / particle |
   |---|---|---|
   | escapers OFF, every particle beyond 80 pulled onto r = 80 (same N) | 20.09 GiB | 215.7 |
   | escapers ON, unclipped (first wiring, cacb7823) | 22.28 GiB | 239.2 |
   | + coupling after the tree force (barrier) and the order on the host (3c28bdec) | 21.44 GiB | 230.2 |
   | + the old split's buffers freed at a re-split (7d196bd8, the tip's code) | **21.07 GiB** | **226.2** |

   The remaining +10.5 B/p (+4.9 %) is not attributed (candidates: the tree masses' `where` and the live mask in
   the step, the split energy program). The trajectory and diagnostics are identical across the three escaper
   rows. With `PREALLOCATE=false` (fraction 0.75 or 0.95) BOTH arms failed to place one 16-17 GiB block (the
   escaper arm in the start program, the clipped baseline in the scan): A1's BFC placement sensitivity at 1e8,
   not the split.
5. **CI.** New shard `integration-strict-esc` (`test_shards.py`, `ci.yml`, 25 min cap); `test_escapers.py` left
   `integration-strict-a2`, whose run with it took 27.9 min at `-n 2` on a loaded host (now 4.7 min alone). Local
   runs with CI's arguments (loaded host): `integration-strict-esc` 11 passed (13.0 min; with
   `JACCPOT_RUNTIME_TYPECHECK=1` 12.0 min -- it found that a handle's `self_acceleration` must be `Optional`),
   `integration-strict-a2` 3, `integration-strict` 44, `integration-strict-runner` 44 (36 min),
   `integration-strict-scheme` 9, `distributed-segment-runner` 5 (2 forced devices), `unit` (typecheck 1) 1328 +
   the 2 pre-existing `test_domain_guard` failures fixed here (the type checker refuses an unknown `Literal` mode
   before the guard's `ValueError`), `integration-strict-lane` and `integration-strict-grad-runner`: see the
   report. Identity: `provider_hlo_identity compare w2-integ-cpu esc-tip-cpu` **PASS 36/36**.
6. **HoreKa** (prepared, not run): `bench/slurm/escaper_timing.sbatch` + `bench/escaper_timing.py`: step time,
   per-edge cost and peak memory at 1e8, escapers on / off (clipped) / on-but-nobody-escapes, both MACs, and the
   energy every 10th edge.

## MESC — escapers in the runner's mesh mode (wave 3, 2026-11-06)

`ProductionRunner(solver, ..., mesh=MeshRunConfig(...), escapers=EscaperPolicy(...))` runs in both mesh runtimes (one
process x N devices, N processes x 1 device) with ESC's semantics: the same policy, Hamiltonian, ledger, diagnostics
and row-order contract. Opt-in (user decision 2026-11-06: no preset turns it on; for merger ICs like Sgr); escapers
do not attract each other; the re-split edge cost is accepted. Code: `jaccpot/distributed/escapers.py` (new),
`jaccpot/distributed/segment.py` (the tail in the start/segment programs), `jaccpot/distributed/mesh_runner.py`
(the orchestration), `jaccpot/production.py` (the refusal removed; `MeshSegmentEdge.escapers`).

### The layout: a tail of their own (and why)

The bound rows are an ordinary `MeshState`: the decomposition, the cuts, the global and Morton boxes, the cross walk,
the migration, the repartition and every capacity see them only. The escapers live in an `EscaperTail` -- flat
`(ndev * cap_e, ...)` arrays sharded `P(axis)` like the layout, `count[d]` leading rows per device, a capacity of
their own (`tail_capacity`: 1.5x the largest device's count, >= 64 rows, a multiple of 64). They need no tree and no
neighbours, so the tail is never sorted, cut or migrated: a particle stays on the device it escaped from, a returner
re-enters that device's bound rows, and every move at an edge is device-local (no exchange).

The alternative the brief named -- escaper rows beyond each device's live cut in the same arrays -- would put a second
live block behind the padding of a layout whose invariant ("`count[d]` live rows lead, the rest is padding") the
force (`num_valid`), the migration, the repartition, the gathers and the restart files all read, and would spend the
bound rows' headroom on escapers. A separate tail touches none of them; the bound path's programs are unchanged.

### Per step, per edge

- **The step** (inside the segment's `jit(shard_map(scan))`): the tail drifts and kicks with the bound rows (the same
  velocity-Verlet association). After the tree force -- ordered by an `optimization_barrier`, as on one card -- the
  bound rows' `[sum m, sum m x]` (fp64 per device) are all-gathered and folded in mesh order (`M_b`, `X_b`); the
  escapers get `a_e = -G M_b (x - X_b)/r^3`, and their total pull (a 3-vector) is all-gathered and folded again
  (every bound row's `a_u`). All-gathers and a fold, not `psum`: the bits do not depend on the runtime's all-reduce
  order (one process or several, gloo or NCCL). The external field acts on both.
- **The edge** (every `every`-th): one small program counts, per device, bound rows beyond `r_escape` and escapers
  within `r_return` (the single card's test in the state's dtype). On a change: the potential before, the tail grown
  if a device's escapers outgrow `cap_e` (a new capacity compiles the tail programs again: `tail_grown`), one
  device-local program that compacts both (stable order; layout padding on the first live row; a returner's f_b =
  its coupling's magnitude), the potential after (the ledger's jump). The carried accelerations are dropped (the next
  segment's start force evaluates the new split).
- **The re-cut under `migrate=True`** (found by the 4-device gate): the bound box changes with the split (a far row
  leaves it, a returner widens it), and the migration's fixed cuts are Morton codes of the OLD box -- in the new box
  they cut anywhere; the next step's migration overflowed a shard (`OVERFLOW_SHARD`, 4 devices). After a changed
  split the runner re-cuts with a full repartition in the new box, as at the start (`RunResult.escapers["recuts"]`).
  The default path is untouched (no split, no re-cut).
- **While the tail is empty** the segments run the DEFAULT programs: escapers on but nobody escaping is the run
  without them, bitwise. Once someone escaped, ONE start and ONE segment program carry the tail for every split.
- **The initial split** (host): the membership of the initial state with the sums of each device's initial id
  chunk folded in mesh order (the same bits from one host or from each process's chunk); nobody beyond `r_escape`:
  the default decomposition and an empty tail; else each device's chunk WITHOUT its escapers is laid out as
  `decompose` lays a chunk and the same Morton repartition runs over the bound rows only; the chunk's escapers are
  its tail.
- **Output** by id over both (`RunResult.state`, `MeshSegmentEdge.state_host()`/`gather()`, `SnapshotWriter`,
  restart files: `escaper_ids` is the tail's layout, `meta["escapers"]` the policy, split, ledger, `tail_cap`,
  `tail_grown`, `recuts`). `edge.plan["escapers"]` has the tail's `cap` and per-device `counts`.
- **Not supported:** the gradient mode (the mesh refuses it anyway) and `scheme=` (the mesh's built-in KDK only, I2).

### Gates

1. **CPU, ESC's synthetic system** (N = 1024 Plummer + 8 fliers + a returner, CSR interpret, leaf 16, p4, theta 0.6,
   dt 1e-3, 6 x 10 steps, energy every edge; ESC's single-card solver and settings on the mesh; fused mesh force,
   migration on; `mn-scratch/mesc/exp/gate1.py`). Escapers per edge on the mesh: **1, 1, 2, 4, 9, 9, 8 -- the single
   card's** (transfers 1, 2, 5, 1 at four edges), both MACs, 2 and 4 devices.

   | MAC, devices | max \|ledgered drift\| | mesh no-split control \|dE/E0\| | raw \|dE/E0\| | max momentum drift (control) | restart / chained == once |
   |---|---|---|---|---|---|
   | `dehnen`, 2 | 1.4e-5 | 5.3e-5 | 1.0e-4 | 8.0e-9 (1.1e-8) | bitwise / bitwise |
   | `dehnen_error`, 2 | 1.2e-7 | 3.0e-7 | 5.9e-5 | 8.5e-9 (7.9e-9) | bitwise / bitwise |
   | `dehnen`, 4 | 2.5e-5 | 5.7e-5 | 1.1e-4 | 7.8e-9 (5.5e-9) | bitwise / bitwise |
   | `dehnen_error`, 4 | 1.4e-7 | 3.0e-7 | 5.9e-5 | 8.2e-9 (8.5e-9) | bitwise / bitwise (fresh solver) |

   Trajectories against the single card's escaper run (`dehnen`, 2 devices), with the mesh-vs-single-card difference
   WITHOUT escapers as the yardstick: bound rows max |dx| 9.2e-6 (yardstick 1.09e-5), rms 1.35e-6 (1.55e-6); fliers
   and the returner 2.2e-6 (2.7e-6). 4 devices: 9.2e-6 (1.08e-5), fliers 2.2e-6 (1.4e-6; both ~2 fp32 ulps at r ~ 8:
   the escapers' `X_b` is a mesh-order fold, one sum on the card). `dehnen_error`: bound rows 1.3e-7 (1.3e-7) on 2 and
   4 devices, fliers bitwise. The single-card references reproduce ESC's record (1.47e-5 vs 1.56e-5; 1.7e-7 vs
   2.3e-7). At 4 devices under `dehnen_error` the second segment was re-planned twice (capacities) with the tail on:
   the retry restores the tail from its host copy (also tested bitwise with a one-row migration buffer). A chained run
   on the SAME solver then starts from the re-planned capacities (they live on the engine) and is not bitwise; on a
   fresh solver, as a chained production run would be, it is.

   Compiles: ONE escaper start and ONE escaper segment program for every split (the tail's counts are traced), shared
   by the run, its restart and its chained continuation (the engine's program cache). Tests
   (`tests/distributed/test_production_runner_mesh_escapers.py`, 2 forced devices, direct force migrating and not,
   plus the fused force under both MACs): counts, ledger, one program, id order, restart and chained bitwise,
   nobody escaping == off bitwise (state, acceleration, diagnostics, plan), a tail grown past its capacity, a
   retried segment (a one-row migration buffer) bitwise the run whose buffer fitted -- the tail restored from its
   host copy --, an external field on the escapers (ledgered energy at the control's noise).
2. **CPU, N processes x 1 device == 1 process x N devices, bitwise, escapers crossing** (gloo;
   `tests/distributed/test_multiprocess_runtime.py::test_escaper_runner_is_the_single_process_run`, direct force,
   migration, 3 segments continued in memory to 6): state, acceleration, owner, the tail's layout, the split and
   ledger, every edge's diagnostics -- equal at 2 and 4 processes. (The fused-force cases are manual.)
3. **GPU, 2.2e6 unclipped Sgr on 2 A100s** (A2's downsample: the 182,859 tail particles plus a stride-49 bulk;
   preset galaxy, softening 1.5e-3, dt 1e-3, `r_escape` 80, energy every edge; cards 5,6 via autocvd, shared host,
   functional only; production flags incl. M2G's ragged-exchange set; `mn-scratch/mesc/sgr/`):

   | arm | escapers | \|ledgered drift\| | momentum drift | 1 proc x 2 GPUs vs 2 procs x 1 GPU | peak per card (sp / mp) |
   |---|---|---|---|---|---|
   | geometric theta 0.7, 2 x 5 steps | 182,832 -> 182,834 | 5.7e-7 | 1.7e-10 | bitwise (state, every diagnostic) | 0.728 / 0.737 GiB |
   | `dehnen_error`, 4 x 5 steps | 182,832 -> 182,837 | <= 6.5e-7 | 3.8e-10 | state bitwise; step 0's `W_self` 1.1e-13 rel (below) | 0.833 / 0.826 GiB |
   | control: escapers off, r > 80 pulled onto 80 (same N), geometric | -- | (raw 4.9e-7) | 1.6e-10 | bitwise | 0.807 / 0.807 GiB |

   ESC's single card on the same IC: 182,832 -> 182,834, -5.0e-7, 1.7e-10 (geometric); 182,832 -> 182,837, <= 5.8e-7,
   3.8e-10 (`dehnen_error`): the same counts and conservation. No retries; 2-3 re-cuts per run. With escapers the
   peak per card is LOWER than the clipped control's (the 182,859 far particles leave the tree; the tail is ~16 MB).
   The `dehnen_error` arms differ only in the step-0 potential's last bits (the bound rows' existing mesh potential
   program; M2G's run-to-run 1-ulp non-reproducibility of `dehnen_error` under the production flags, accepted
   2026-11-05), hence the energies after it.
   **The FULL 1e8 unclipped Sgr on the same two cards** (100,000,001 rows, a 45.7 Mpc box; one process x 2 GPUs,
   ~5e7 rows per card; geometric theta 0.7, 3 x 5 steps, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, fraction 0.95): runs,
   no retry, 3 re-cuts; escapers 182,878 -> 182,889 (ESC's single card: the same), ledgered drift <= 2.7e-7 (the
   clipped control's raw drift 2.8e-7), momentum drift 3.3e-10. Peak per card:

   | 1e8 on 2 A100s, same cards and environment | peak per card | B / particle per card |
   |---|---|---|
   | escapers OFF, every particle beyond 80 pulled onto r = 80 (same N) | 21.41 GiB | 459.9 |
   | escapers ON, unclipped (`EscaperPolicy(80)`; tail `cap_e` 191,616) | 21.45 GiB | 460.7 (+0.17 %) |

   The escapers add +0.8 B/p per card (+0.17 %, not attributed further; the tail itself is ~0.2 B/p). The ~460 B/p
   per card at 5e7 rows is the mesh mode's own at this size (both arms), not the escapers'.
4. **The default paths.** `provider_hlo_identity compare w2-integ-cpu mesc-tip-cpu` (98f6aa00; also at bc82f156):
   **PASS 36/36** (the tool has no mesh programs). The mesh programs with `escapers=None` -- the start, segment and potential
   programs of the direct force (`dehnen`) and the fused force (`dehnen`, `dehnen_error`), migrating and not, 18
   programs lowered at 2 forced CPU devices -- have the SAME lowered text on `esc/stacked` a4bfea96 and on this branch
   (`mn-scratch/mesc/exp/mesh_hlo_dump.py`; empty diff). ESC's tests (`integration-strict-esc`) pass unchanged.
5. **CI** (CI's arguments, each job's forced device count and typecheck; a loaded 64-core host): new shard
   `distributed-segment-escapers` (job `test-distributed-segment-escapers`, 2 forced devices, `-n 1`, 40 min cap,
   gated like the other segment jobs): 15 passed, 1 skipped, 12.3 min. `distributed-segment-runner` 5,
   `distributed-segment-migration` 2, `distributed-segment-criterion` 4, `distributed-segment` 10 (one stand-in test
   needed `coupling_G` read only for the tail programs, fixed; rerun at the tip with the escaper shard: 10 and 15), `distributed-tier` 96 (+7 skipped),
   `distributed-criterion` 8, `distributed-multiprocess` 14 (+11 skipped; the new escaper cases add ~1 min),
   `unit` (typecheck 1) 1330 (+158 skipped), `integration-strict-esc` 11, `integration-strict-runner` 44; the
   top-level `tests/unit` guards 23; `test_shards.py check` rc 0 (2863 covered).
6. **HoreKa** (prepared, not run): `bench/slurm/mesc/` + `bench/mesh_escaper_timing.py`: the FULL unclipped 1e8 Sgr
   as 4 processes x 1 GPU (one node) and 8 x 1 (two nodes) with escapers against the clipped control, both MACs, 6 x
   10 steps, energy every edge, plus the escaper row with the energy every 10th edge: step time, edge time (the
   ledger energies and the re-cut), peak per card. Stage the IC to `$H0_ROOT/data/` first (README).

### Open

- **The re-cut's cost.** At 1e8 Sgr a few particles cross at EVERY edge, so every edge pays a full repartition on top
  of the two ledger energies. If HoreKa's `edge_s` shows it matters: skip the re-cut when the bound box is bitwise
  unchanged (the old cuts are then still exact Morton ranges), or check membership every `k` edges
  (`EscaperPolicy(every=k)`).
- **Tail placement.** Escapers stay on the device they escaped from (the downsampled Sgr's 182,832 initial escapers
  all sat in the last id chunk, so on one device's tail). Harmless at these sizes; a rebalance of the tail would only
  matter at ~1e6+ escapers per device.
- Multi-process restart files remain refused (M3); the gradient mode and `scheme=` are refused on the mesh as before.

## I2 — the multi-GPU runner takes nornax's scheme; a sharded force-model protocol (wave 4, 2026-11-07)

The multi-node plan's I2 (decision U1: nornax owns the integration schemes, jaccpot the force, decomposition and
orchestration). Branch `i2/sharded-scheme` in jaccpot, nornax and Odisseo (from main 4ace6562, nornax main
9003651, Odisseo `jaccpot-integration` c43f4a0).

### What was built
1. **jaccpot: the scheme on the mesh.** `ProductionRunner(..., mesh=MeshRunConfig(...), scheme=...)` no longer
   refuses a scheme. `MeshSegmentRunner(scheme=)` (and `MeshSegmentRunner.fused(scheme=)`) calls the scheme's
   `open_step` / `close_step` on each device's `(cap, 2, 3)` rows inside the segment's `shard_map`, in the plain
   and the migrating segment (the opening half sees the rows owned at the step's start, the closing half the rows
   owned after the migration). `scheme=None` takes the old inline expressions, so the default programs are the
   same; `("scheme", key)` joins the segment programs' compile key only when a scheme is set. The mesh restart
   file records `meta["scheme"]` only with a scheme; a restore under another scheme lists `"scheme"` in the
   mismatches.
2. **jaccpot: schemes that read other rows.** `jaccpot.scheme.ShardedSplitStepScheme` is a scheme's MESH FORM:
   `open_step_shard` / `close_step_shard` with `live_rows` and `axis_name`, reducing through collectives whose
   result is independent of the transport's order (`nornax.sharded.mesh_sum`). `check_mesh_scheme`: rowwise
   passes, a mesh form passes (and is used, rowwise or not), anything else is a `ValueError` at construction.
3. **nornax: the protocol** (`nornax/sharded.py`). `ShardedForceModel.shard_accelerations(positions, masses,
   ids, live_rows, *, carry=None, args=None) -> (acc, carry)` inside `shard_map` (optional `context()`,
   `shard_potential`); `mesh_sum` / `mesh_moments` (all-gather folded in mesh order); `gather_by_id`,
   `decompose_rows`, `is_writer` (by-id I/O, writers on process 0); `ShardedDirectSumGravity` (the toy: sources
   all-gathered and summed in id order); `sharded_split_step_rollout` (the reference driver);
   `nornax.conformance.check_sharded_force_model` (protocol, shape, padding rows exactly zero, layout
   independence by id, momentum, an oracle, the `mesh_sum` order, the scheme's rollout on the mesh vs one device).
   Nothing in nornax imports jaccpot.
4. **jaccpot implements it:** `jaccpot.distributed.sharded_force.ShardedMeshForce` wraps a prepared
   `FusedMeshForce` / `DirectMeshForce` (per-device carry: the OR of the overflow bits, the max of the walk needs,
   `f_b` under `dehnen_error`, `initial_carry()` from the prepare's seed). The runner itself does not go through
   it (its segment programs call the force directly and stay main's); nornax's drivers and kit do.
5. **Odisseo:** the `fmm_mesh_runner` lane takes `config.fmm_scheme` as the single-card runner lane does
   (`runner_scheme`; part of the kept runner's key).

### Gates (CPU unless noted; forced host devices)
| gate | result |
|---|---|
| 1, direct force (`test_production_runner_mesh_scheme.py`, 4 forced devices) | 8 passed: KDK scheme == `scheme=None` bitwise by id (state, every edge's state, gathers, diagnostics) on 2 and 4 devices, repartition and migrate; chained run, restart file (arrays byte-equal to the built-in's file, meta records the scheme) continued bitwise; a retried segment; mesh-form scheme keeps the momentum to 1e-14 and a non-rowwise one without it is refused; `ShardedMeshForce` == the runner's self-gravity |
| 1, fused force (`test_production_runner_mesh_scheme_fused.py`, 2 forced devices) | 3 passed, bitwise by id incl. `f_b`: geometric MAC with a cross-capacity retry (retries [1,0,0]); geometric MAC, migration, `migration_cap=1` (a buffer retry); `dehnen_error`, migration, 2+2 chained |
| 2, N processes x 1 device == 1 process x N devices, with the scheme (`test_multiprocess_runtime.py`) | passed: direct runner at 2 and 4 processes (and without the scheme), fused `dehnen` migrating with a cross retry at 2; the KDK arrays also equal the no-scheme arrays (0 differing, both layouts, 2 and 4) |
| default path: `provider_hlo_identity compare w3-main-cpu i2-tip-cpu` | **PASS 36/36** |
| default path: mesh programs with `scheme=None`, lowered text main vs tip (2 forced devices; source locations stripped) | IDENTICAL: direct (12 modules), direct migrating (18), fused geometric migrating (597), fused `dehnen_error` migrating (1169) |
| I1 pins (`integration-strict-scheme`, CI args) | 9 passed (its refusal test now checks the non-rowwise refusal on the mesh: the mesh refusal is gone); the single-card runner code is untouched |
| 4, nornax `tests/unit/test_sharded.py` | 8 passed (JAX 0.11.2); with `test_conformance.py` 25 passed on JAX 0.10.2. Kit on 2 and 4 devices: layout 0.0, rollout 2 vs 1 device 1.2e-18, 4 vs 1 2.4e-18 |
| 4, nornax `tests/integration/test_jaccpot_sharded.py` (jaccpot's `ShardedMeshForce(DirectMeshForce)`) | 2 passed: every check (oracle 4.4e-16, rollout 0.0) at 2 and 4 devices; the runner with `LeapfrogKDK` == built-in, 0 differing |
| 5, Odisseo mesh-lane files at 2 forced devices | 38 passed (new: `fmm_scheme="nornax_kdk"` on the mesh lane == the built-in lane bitwise); suite `-m 'not slow'` 274 passed, 6 skipped, 2 xfailed |
| 3, GPU (functional; 2 A100s by autocvd, one process; shared host, no timings) | N = 262,144 Plummer (a = 1), galaxy-preset solver, fused mesh force, migration on, 10 x 10 steps at dt 1e-3, energy every edge, PRODUCTION flags (no `--xla_gpu_deterministic_ops`): `dehnen_error` and `dehnen` each: A-vs-A control 0 differing, `LeapfrogKDK` vs both built-in arms 0 differing (state, acceleration, `f_b`, diagnostics), 0 retries, energy drift -3.2e-7 / -2.5e-7 |
| 6, CI | new gated shard/job `distributed-segment-scheme`; `jaccpot/scheme.py` joins `GATED_PATHS`; `test_shards.py check` rc 0; top-level `tests/unit` guards with `JACCPOT_RUNTIME_TYPECHECK=1` rc 0; nornax solver-tests job lists `test_jaccpot_sharded.py` (skips until its jaccpot pin has I2) |

## M1 — symmetric cross-domain forces (#173): no gap, no new transport (wave 4, 2026-11-07)

**Question** (#173, plan M1): does the fused mesh lane lose momentum conservation at domain boundaries? Each cross pair
has one end on each device. If the two devices evaluate their halves independently, `sum m a` might only cancel to the
force error. The proposed fix was to evaluate each pair once on an owner device and send the reaction back.

**Answer: the cross terms already cancel to round-off, at the one-card level.** No reaction-return transport was built
(the brief's Step 0 stop rule). Library code is unchanged; this session adds a test, a CI shard, a HoreKa job and
docs.

### Why (read in `jaccpot/distributed/cross.py`)

- **The pair set is the same at both ends.** Every device walks the gathered two-sided summary trees from the same
  seed, `(min(me, d), max(me, d))`, reading only the gathered floats (`_symmetric_walk`). Both ends of every cross pair
  therefore derive the identical pair set (WP-G G0.1 confirmed it).
- **Each end evaluates its own half from bitwise-identical shipped data** (`_symmetric_exchange`):
  - far: the peer's multipole and centre go through M2L into my node's local expansion;
  - near: the peer's leaf particles go through the cross near term.
- **One card does exactly the same with its own pairs.** Its directed lists hold each canonical pair once per
  direction (`_directed_csr_from_canonical`), with the same M2L and near kernels. A cross pair therefore cancels in
  `sum m a` as well as a local pair does.
- **Measured: in fp64 the lane conserves momentum to round-off on one card and on the mesh alike** (the table below).
  The floor is set by summation, not by the expansion. This is presumably because the two directed M2Ls of a pair about
  the cells' expansion centres are antisymmetric in the total force they exert; that was not derived here.

### Step 0 (measured before building)

Harness: `mn-scratch/m1/exp/step0.py`. The `ProductionRunner` mesh mode (migration on), then at a probe the force with
and without the cross hook on the same state. The residual is `|sum m a| / sum m |a|` of the total, of the local part
and of the cross part alone (normalised by its own `sum m |a|`). Accuracy is against an fp64 direct sum.

**Configuration:** Plummer (half the mass, a = 0.3) + exponential disc (R_d 1, z_d 0.1), seed 0; leaf 64, p4,
theta 0.8, eps 1e-2 (ferrers3); `dehnen_error` at eps 1e-4; dt 1e-3; forced CPU devices with the interpret-mode lane.

**fp64, N = 8192, one force:**

| devices | total | local part | cross part alone | cross share of sum m\|a\| | rel-L2 / p99 / max vs direct |
|---|---|---|---|---|---|
| 1 | 3.7e-17 | 3.7e-17 | - | 0 | 4.02e-3 / 6.30e-2 / 0.266 |
| 2 | 1.9e-17 | 2.2e-17 | 2.6e-17 | 0.57 | 4.02e-3 / 6.30e-2 / 0.266 |
| 4 | 3.0e-17 | 3.2e-17 | 7.9e-18 | 0.77 | 4.02e-3 / 6.30e-2 / 0.266 |

The cross part is a large share, not a token one. At 4 devices each device has 6.5-7.3k cross far pairs against
1.8-3.0k local far pairs, and 2.2-2.7k cross near pairs against 1.6-2.7k local near pairs. That is a larger cross share
than production (1e8 on 4-8 cards: ~1e7 per card, cut surfaces a small fraction of the volume), so the toy runs are the
harder test. fp32 at 4 devices: total 1.0e-8, cross part alone 1.0e-8, local part 2.4e-9, which is fp32 round-off (one
A100 card: 1.1e-8).

**GPU, functional** (2.6e5, fp32 working dtype, fp64 state, 1000 steps, 10 x 100; cards via autocvd on the shared host;
M2G's production flags):

| MAC | cards | per-step total at 100 / 1000 | cross part alone (share) | momentum_drift at 1000 | rel-L2 vs direct (4096 probes) |
|---|---|---|---|---|---|
| `dehnen` | 1 (card 5) | 1.1e-8 / 7.6e-9 | - | 2.8e-10 | 2.50e-3 |
| `dehnen` | 2 (5,6) | 7.9e-9 / 5.9e-9 | 4.0e-9 / 2.0e-9 (part of total) | 6.2e-10 | 2.50e-3 |
| `dehnen` | 3 (2,5,6) | 2.7e-9 / 3.2e-9 | 2.2e-9 / 7.8e-9 (0.72 / 0.69) | 4.8e-10 | 2.53e-3 |
| `dehnen_error` | 1 (card 0) | 4.1e-9 / 1.3e-9 | - | 3.22e-10 | 3.34e-4 |
| `dehnen_error` | 2 (2,5) | 5.1e-9 / 2.5e-9 | 5.0e-9 / 4.5e-9 (0.59 / 0.57) | 3.19e-10 | 3.34e-4 |
| `dehnen_error` | 3 (0,2,5) | 1.8e-9 / 3.4e-9 | 2.5e-9 / 5.9e-9 (0.72 / 0.69) | 4.65e-10 | 3.35e-4 |

The drifts are a random walk of the per-step fp32 residual (~sqrt(1000) x dt x 1e-8 = 3e-10) on every card count. The
force accuracy (p99 and max too) is the one card's.

**1000-step drift matrix (CPU, N = 2048, the plan's gate wording).** 1000 steps (10 x 100), Plummer + disc, the
configuration above, forced CPU devices on a shared host (`mn-scratch/m1/exp/runs/`). Every device count sits in its
1-device class: fp64 drifts below 1e-16, fp32 at the fp32 floor. The accuracy is the one device's or better (4 devices:
other trees, slightly lower rel-L2).

| MAC | dtype | devices | max momentum_drift (1000 steps) | per-step total @100 / @1000 | cross part alone @1000 (share) | rel-L2 / p99 / max @1000 |
|---|---|---|---|---|---|---|
| `dehnen` | fp64 | 1 | 9.00e-17 | 6.3e-17 / 4.8e-17 | - | 3.27e-03 / 6.62e-02 / 2.39e-01 |
| `dehnen` | fp64 | 2 | 3.60e-17 | 3.1e-17 / 3.0e-17 | 2.7e-17 (0.52) | 3.31e-03 / 6.43e-02 / 2.39e-01 |
| `dehnen` | fp64 | 4 | 2.76e-17 | 3.1e-17 / 4.2e-17 | 8.4e-18 (0.70) | 2.89e-03 / 6.32e-02 / 2.40e-01 |
| `dehnen` | fp32 | 1 | 6.60e-09 | 1.2e-08 / 1.8e-08 | - | 3.27e-03 / 6.62e-02 / 2.39e-01 |
| `dehnen` | fp32 | 2 | 5.74e-09 | 9.7e-09 / 8.6e-09 | 6.8e-09 (0.52) | 3.31e-03 / 6.42e-02 / 2.39e-01 |
| `dehnen` | fp32 | 4 | 5.05e-09 | 4.3e-09 / 4.2e-09 | 1.9e-09 (0.70) | 2.89e-03 / 6.32e-02 / 2.40e-01 |
| `dehnen_error` | fp64 | 1 | 5.01e-17 | 1.2e-17 / 4.3e-17 | - | 1.27e-05 / 2.35e-04 / 1.20e-03 |
| `dehnen_error` | fp64 | 2 | 7.73e-17 | 1.7e-17 / 4.3e-17 | 8.8e-17 (0.52) | 1.27e-05 / 2.32e-04 / 1.20e-03 |
| `dehnen_error` | fp64 | 4 | 3.55e-17 | 4.1e-17 / 2.1e-17 | 3.1e-17 (0.70) | 9.93e-06 / 1.96e-04 / 9.31e-04 |
| `dehnen_error` | fp32 | 1 | 1.25e-10 | 3.3e-09 / 2.6e-09 | - | 1.26e-05 / 2.28e-04 / 1.20e-03 |
| `dehnen_error` | fp32 | 2 | 1.96e-10 | 2.1e-09 / 1.5e-09 | 2.1e-09 (0.52) | 1.26e-05 / 2.28e-04 / 1.21e-03 |
| `dehnen_error` | fp32 | 4 | 2.26e-10 | 2.3e-09 / 2.4e-09 | 1.5e-09 (0.70) | 9.94e-06 / 1.96e-04 / 9.31e-04 |


### What this session added

- `tests/distributed/test_mesh_cross_momentum.py`: fp64, 2 forced devices, N = 2048 cube, both MACs. It asserts:
  - total and cross-part residual < 1e-13 (measured: `dehnen` 1.7e-17 / 4.1e-17, `dehnen_error` 6.5e-17 / 1.2e-16);
  - cross share > 0.1 (measured 0.52);
  - each device's cross part non-zero;
  - control: dropping the cross far pairs on ONE device gives 0.15.

  It runs in a CI job of its own, `distributed-segment-momentum` (20 min cap, ~5 min of calls). With the test inside it,
  the `distributed-segment` job took 1789 s of its 40 min on this host.
- `bench/mesh_cross_momentum.py` + `bench/slurm/m1/` (HoreKa, prepared, not run): the same split and `momentum_drift`
  at 1e7-1e8, galaxy preset, 1 GPU (yardstick), 4 GPUs (one node) and 8 GPUs (two nodes) as N processes x 1 GPU, 200
  steps. M2G's jobs record `momentum_drift` but not the split.
- `docs/production.md`, mesh section: "Momentum across devices".

### Gates

1. Momentum: the matrix above. Every device count is in its 1-device class, fp64 and fp32, both MACs.
2. Force accuracy: identical to one card (fp64 8192 at 1/2/4 devices; GPU 2.6e5 at 1/2/3 cards).
3. Bitwise:
   - the 1-device mesh == one card, and N processes == 1 process x N devices: library code unchanged
     (`git diff 4ace6562 -- jaccpot/` is empty), so the existing pins hold (`test_a2_one_device_segment_is_the_single_card_particle_carry`
     in `distributed-segment`, run here: 12 passed; the multiprocess shard as on main);
   - identity: `provider_hlo_identity compare w3-main-cpu m1-tip-cpu` **PASS 36/36**. The mesh programs are unchanged
     by construction (no library diff).
4. GPU functional: the table above (2 and 3 cards, 1000 steps, 2.6e5).
5. CI:
   - `test_shards.py check` rc 0 (3064 covered);
   - `distributed-segment` 12 passed (CI args, 2 devices);
   - top-level `tests/unit/test_*.py` with `JACCPOT_RUNTIME_TYPECHECK=1`: 480 passed, 31 skipped;
   - pre-commit clean;
   - yggdrax untouched.
6. HoreKa: `bench/slurm/m1/` (above).

### What M6 (block steps on the mesh) needs from the cross layout

The canonical-owner design is NOT a prerequisite for M6.

- **Momentum is no reason for it.** Under individual timesteps, a pair whose two ends sit on different rungs is
  kicked at different times. Momentum then cancels to round-off only across pairs whose ends are kicked together, on
  one card as on the mesh. An owner-evaluated reaction changes that only if the scheme kicks both ends of a pair
  together (a mutual block kick, as `jaccpot/mutual/` writes both halves in one kernel). If B2/M6 adopt such a scheme,
  the owner orientation is already available (last bullet); if they keep the target-centric KDK of the fused lane, the
  present both-ends evaluation is equivalent.
- **What M6 does need:**
  1. **An activity column in the published summary rows** (one more float per row of `_summary_rows`): each device's
     active nodes at the substep. Because the gathered table is identical everywhere, both ends of a pair still agree,
     with no extra message, on which pairs are live (target active) and on which rows a sender must ship (the peer's
     node in the pair is active). This is the active-only exchange: `_rows_by_peer` filtered by the peer's activity bit.
  2. **Per-level (per-substep-class) cross capacities.** The pair and row counts follow the active fraction, so one
     capacity set per level, or one sized for the deepest active set plus a re-plan.
  3. **Source state at the substep.** Inactive particles' predicted positions, or B2's HOLD of the multipoles at the
     last synchronisation, must be what the summary and the shipped multipoles carry. Otherwise the two ends' pair sets
     would differ (both must read the same floats).
  4. The `(target, source)` order of the pair lists (`_in_pair_order`) stays, so an active subset's sums are a function
     of the pair set (bitwise restarts).
- **The canonical-pair data M6 can reuse is the walk result itself** (`res.far_a/far_b`, `near_a/near_b`, in the
  gathered index space). A one-owner orientation would follow from `a < b` on the global `[ndev x S]` index (device,
  then summary index), which the walk already emits. Nothing has to be stored now.
## FUP — the fused lane's CPU route, the geometric MAC's accuracy, two re-plans (wave 3, 2026-11-06)

Three follow-ups of ODY (Odisseo's default FMM lane is the runner). jaccpot base main 4ace6562.

### The CPU route: `RuntimePolicyConfig(cpu_fused_lane=True)` (opt-in)

The fused lane gated its prepare on `jax.default_backend() == "gpu"`; Odisseo and jaccpot's own CPU tests reached it by
making the whole process report `"gpu"`. Now an engine built with `cpu_fused_lane=True` takes, on a CPU backend, the
GPU's lane policy at every engine gate (`FMMEngine._lane_backend` -> `runtime/_lane_platform.lane_policy_backend`:
the large-N prepare, the split dual-tree build default, the near-field mode and chunk, the streamed traversal cap, the
precompute/schedule caps, the tree jit), while kernel availability (`pallas_*_supported`) keeps the real backend: the
pure-JAX kernels. With `use_pallas=True` the near field runs the CSR lane in Pallas interpret mode (what
`dehnen_error`'s force scale needs): the three switches (`JACCPOT_NEARFIELD_PALLAS_INTERPRET`, `..._LEAFPAIR_CSR`,
`JACCPOT_LARGE_N_RADIX_FAST_PAYLOAD_MAX_MB=0`) are context-local defaults (`jaccpot._env.scoped_env_defaults`, a
`ContextVar`; `os.environ` is never written, a shell value wins), entered by `ProductionRunner`'s and
`FusedForceProvider`'s entry points and `strict_run_v2`. On a GPU, and for every engine without the flag,
`_lane_backend()` is `jax.default_backend()`: the programs are unchanged. Without the flag the provider's CPU refusal
names it. **Verified:** the runner on the route = the monkeypatched lane bit for bit (N = 512 4 steps and N = 256 2
steps; the geometric MAC on the pure-JAX near field and `dehnen_error` on the interpret-mode CSR lane). jaccpot's own
CPU tests keep their monkeypatch (both routes are pinned equal by `tests/integration/strict/test_cpu_fused_route.py`).

### The geometric MAC's accuracy: cell leaves, not the MAC (no numerics changed)

ODY measured the fused lane's geometric Dehnen MAC ~36x less accurate than Odisseo's old coupler at the same theta
(leaf 16, p4, theta 0.6: rel-L2 4.3e-3 vs 1.2e-4, N = 512). Both test `r_A + r_B <= theta d` about COM centres with
exact radii; the difference is the TREE: the galaxy preset builds cell leaves (`leaf_size` a ceiling: 6.1 particles a
leaf on average at N = 512, down to one), the coupler bucket leaves of exactly 16. The fused lane sends 68 % of the
particle pairs to the far field there, the coupler 2.3 %. The general path on the same cell tree gives the same
4.28e-3, the galaxy solver with bucket leaves the coupler's 1.196e-4; every accepted pair is within theta, every
particle pair counted once, fp64 = fp32 (truncation). The gap closes with N (36x at 512, 5.7x at 2048, 3.8x at 8192).
The coupler's accuracy needs theta ~0.3 at N ~ 500, ~0.45 at N ~ 2000, or leaf 64 at theta 0.6
(`mn-scratch/fup/t2/FINDINGS.md`). Eq (16a) sets the accuracy directly. Not changed (an optional enhancement: never
accept a far pair whose node holds fewer than ~p particles).

### Two capacities the segment retry did not re-plan

ODY's case: the FMM notebook's N = 512 cube collapsing (masses 0.5-1.5, softening 1e-3 `ferrers3`, dt 5e-3, 200
steps, 4-step segments), the geometric MAC on the pure-JAX near field (CPU route), leaf 64 and 256: both raised
"a fixed capacity saturated" with every walk and tree flag 0.

1. **A far list filled EXACTLY.** The verdict reads a full far buffer as saturated (`far_count >= width`: an overflow
   saturates the count to the width), the walk raises no flag (every pair fitted), so `_replan_walk_caps_from_needs`
   re-planned nothing (leaf 256: 64 of 64 directed pairs). Now such a list grows (`_far_list_full`; at least one
   pair more); a NAMED far cap that filled still raises.
2. **The rectangle near-field layout** (the pure-JAX near field only; the CSR lane never reads it): its static
   target-block cap is sized at the eager prepare from the START's densest row and only ever grown; rows that outgrow
   it mid-segment (leaf 64: 71 against 64; leaf 256: 39 against 32) failed the verdict with no flag, and the re-prepare
   at the same start sized it the same. Now, when neither the walk nor the tree overflowed and the rectangle guard
   is active, `FusedForceProvider.replanned` doubles it (`_replan_rectangle_blocks`, up to the caps ladder's last
   rung); the progress retries continue while each re-run gets further.

**Result** (CPU route, default environment): leaf 64 runs all 200 steps with 2 retries (segments 3 and 19), leaf 256
with 7 (segments 2-4), each recorded as the edge's `retries` (`SegmentEdge`, the diagnostics' `retries`). Before: leaf
64 raised at segment 19, leaf 256 at segment 2. The default production path (CSR lane on a GPU, no exact fill) runs
the same programs: identity below.

## A4 — the accuracy verification suite (wave 5, 2026-11-08)

Plan A4 builds the reference force, the standard problems, the convergence ladders and the check mode that later
tracks must pass. The worktree is `~/jaccpot-a4` (+ `~/Odisseo-a4`), branch `a4/verification-suite`, based on main
`cc90bb6f`. How to run it, the schema and the tolerances are in `docs/accuracy_suite.md`. The HoreKa jobs are in
`bench/slurm/a4/README.md`.

### What was built
| piece | what |
| --- | --- |
| `jaccpot/verification/reference.py` (new) | `reference_force`: an fp64 direct sum at `K` target ids (`select_targets(N, K, seed)`). Each pair term is rounded onto a per-target grid (`2^-86` of the global max term, a `pmax`) and summed as three 30-bit `int64` limbs (an integer `psum`), so the result is the same bits in ANY order. `gather_by_id` (exact rows by id, `all_gather` of `[K, D]`), `score_force` (rel-L2, p50/p90/p99/max), `force_law_of`, and the runner adapters `score_single_result` / `score_mesh_result` (no gather of the particles) |
| `jaccpot/verification/baseline.py` (new) | the result schema `jaccpot.accuracy/1`; `check_result` with comparable config keys (different N/dt/MAC/precision/targets/seed: "not comparable") and per-metric tolerances (`abs`, `rel`, `ratio`, `ceiling`, `floor`) |
| `bench/accuracy/` (new) | `ics.py` (Eddington spheres: Plummer, Hernquist, truncated NFW); `analysis.py` (Lagrangian radii and sigma, disc sigma_z^2 / sigma_R^2, `run_bar.py`'s bar probe + `bar_summary.py`'s numbers + the buckling time, merger centres and pericentres); `problems.py` (the six problems, per-tier defaults, tolerances); `suite.py` (`run`, `record`, `check`, `analyse-existing`, `ladder`, `summarize`, `ic`); `ladders.py` (N, dt, MAC and precision ladders, the latter with A3's `anchored` slot); `local_tier.sh` |
| `bench/slurm/a4/` (new) | `card1.sbatch` / `node.sbatch` (one rung per array task: one A100, or one exclusive node with `--devices 4`), `ics.sbatch` (agama), `summarize.sbatch`, `submit.sh` (`--dry-run`, `--local-smoke`), `site.env`, `README.md` |
| Odisseo `tools/a4_galaxy_ics.py` (new) | the 2026-10-31 galaxy, bar-disc and merger ICs in the suite's format (agama seeded too) |
| CI | `tests/unit/verification/` (unit shard), `tests/distributed/test_reference_force_devices.py` (distributed tier, 2 forced devices), `tests/distributed/test_multiprocess_reference.py` (joins the `distributed-multiprocess` shard; `jaccpot/verification/` added to `GATED_PATHS`), `bench/tests/test_accuracy_suite.py` (bench-tooling) |

No library file changed: the default paths are untouched. A3's files were not touched either.

### Gates
| gate | result | configuration |
| --- | --- | --- |
| 1. the reference equals an fp64 direct sum | max relative difference from `jaccpot.runtime.reference.direct_sum` < 1e-14 for `plummer`, `ferrers3`, `wendland_c2` (eps 0.02) and Newton; potential < 1e-14; against an extended-precision sum of the same terms < 4e-16 (smoke: 4.9e-16 at N = 3000) | CPU, N = 400 (with a dense clump), K = 48 |
| 1. bitwise across devices | identical bits (acceleration, potential, positions) on 1 / 2 / 4 forced devices, each with its own row shuffle and padding; any row order, padding or split into two source groups on one device | CPU forced devices, N = 512, K = 64 |
| 1. bitwise across processes | 2 and 4 processes x 1 device == 1 process x 2 / 4 devices, bit for bit (two kernels), and the runner's multi-GPU mode (`DirectMeshForce`, migrating) scored at its end state identically on both sides, rel-L2 < 1e-13 | gloo on localhost (M2's harness) |
| 2. problems end to end | all six (`plummer`, `hernquist`, `nfw`, `disc_heating`, `bar`, `merger`) run through `ProductionRunner` and write `result.json` (CPU route, smoke sizes); the local GPU tier: see below | CPU, `dehnen_error`, fp32 |
| 2. reproduces the recorded runs | the bar ladder (`analyse-existing bar`): `t_bar`, peak time and amplitude, bar length and pattern speed equal `ladder_final.json`; buckling 2736 / 2673 / 1736 / 1472 Myr (1e8 / 4e7 / 1e7 / 2.5e6, theta 0.7) against the record's 2.74 / 2.68 / 1.73 / 1.47 Gyr, s33 / s34 2156 / 2169 (record 2186 / 2172). The merger: first pericentre 9.04 kpc at t 1.307 in all three dt runs (record: 9.1 kpc, t 1.30), apocentre 84.7 kpc at t 4.14, second pericentre 0.57 kpc at t 7.61 (record: t 7.6, 0.6 kpc) | the 2026-10-31 `probe.jsonl` files |
| 3. check passes / fails | baseline = Plummer N = 1024, 4 t_dyn. The re-run: CHECK PASS (bitwise on CPU). `--perturb dt=4` (same end time): FAIL, `energy.max_abs_dE` 2.67e-2 vs 5.07e-5 and `force.p99_end` 1.5x. `--perturb theta=0.95`: FAIL, `force.rel_l2_end` 1.87e-2 vs 8.73e-5 and energy 7x. N 1000 vs 1024: FAIL, "not comparable" | CPU route, `dehnen_error` 1e-4, fp32, K = 256 |
| 4. HoreKa suite | `bash -n` on every script; `submit.sh --dry-run`: 21 arrays (14 one-card, 7 node) + ics + summary; every rung's command line parses (`bench/tests`); `--local-smoke`: every rung at tiny N on the CPU (node tier on 4 forced devices): see below | |
| 5. CI | `test_shards.py check` rc 0 (3199 tests covered); new files under `JACCPOT_RUNTIME_TYPECHECK=1`: unit 13 passed (1.36 GB peak RSS, 35 s), forced 2 devices 3 passed (1.77 GB, 23 s), 2 processes 1 passed (1.58 GB per process, 40 s); top-level `tests/unit` guards pass | CPU |
| identity | `provider_hlo_identity compare w5-main-cpu a4-tip-cpu`: PASS 36/36 identical (recorded at 997fcd3d; later commits touch `bench/`, `docs/`, CI lists only) | CPU |

**The reference's cost.** It makes two passes over the `N * K` pairs, about 3x an fp64 direct sum on them. It is a
verification step, not a per-edge diagnostic. The runs score it at the first edge with a force, every
`--score-every` segments and the last edge.

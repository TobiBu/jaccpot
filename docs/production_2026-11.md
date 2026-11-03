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
| accuracy no worse than geometric theta 0.7 at equal time | rel-L2 / p99 / max 1.26x / 1.29x / 1.54x lower from half the far pairs; HoreKa times pending |
| safety net: CSR arrays bitwise | it is a sort; interpret-mode tests against the NumPy directed sort; CPU identity 36/36 at 79e42475 (the CPU lane does not lower the Pallas placement) |
| the guard: bitwise where finite | CPU identity at bfa876c8: 32/36 identical; the 4 `dehnen_error` variants move their compiled runners' TEXT, every array bitwise (1302 arrays) |
| the floor: bitwise where it does not bind | CPU lane test bitwise; at #487 it never bound |
| S12 mesh `dehnen_error` gates | `tests/distributed/test_mesh_segment_dehnen_error.py` 4/4 (CPU, 2 devices) |
| GPU pins S1, S6, S1p/t/e/x/g, G1 | <!-- PINS --> |

Other tests: the numeric goldens (`tests/characterization`) 21/21; `tests/unit/runtime` and `tests/unit/operators`
(not slow) 997 passed, 139 skipped; the new and touched unit files also under `JACCPOT_RUNTIME_TYPECHECK=1`. The
constructor-state golden gained one line, `mac_force_scale_floor: None`.

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

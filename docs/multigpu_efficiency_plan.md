# Plan: multi-GPU efficiency

_Drafted 2026-08-21 against the measurements in
`docs/distributed_performance_findings.md` and
`docs/distributed_per_device_ceiling.md`. 5x A100-PCIE-40GB available._

## The one lesson that sets this plan's order

**Four attribution attempts have failed this session.** `theta_cross` mis-tuning,
the fused-M2L flag, near-field padding, and the traversal pair queue were each
predicted to be the dominant cost; each was measured and each was wrong. Twice I
drew a conclusion from a single invocation and had to retract it after replicates
disagreed.

So: **instrumentation comes first**, and every stage below carries a measurement
that can falsify it. Nothing here is committed to on the strength of an argument.

## What is actually established

Robust across replicates:

- **A large fixed floor.** At 4 devices, 1024 particles/device generating **60
  cell-pair interactions** takes **~50 ms**. At the production 16 384/device it is
  ~97 ms, so roughly half to two thirds of an evaluation is independent of
  interaction work.
- **Strong scaling is flat** (79-84 ms, ndev 2-5, N=65 536): a fixed cost cannot
  be divided across devices.
- **Weak scaling ~70% efficiency** (59 -> 86 ms, ndev 2-5): the fixed term grows
  with device count.
- **The cross-domain far field is empty at the default.** `cross_far = 0` at
  `theta_cross=0.1` for every device count measured, so all cross-domain
  interaction is direct-summed and `cross_near = 4096 x (ndev-1)`.
- **Relaxing `theta_cross` currently loses.** 0.1 -> 99 ms; 0.2/0.3/0.4 -> ~117 ms,
  reproducible. It works mechanically (far field activates, cross near work falls
  4.3x) and still costs ~18 ms.

Measured *no* effect: `max_pair_queue` over a 256x range (non-monotonic,
48.9-58.9 ms), `max_neighbors_per_leaf` over 16x (flat), the
`JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS` flag.

Established from the **code**, independent of timing
(`jaccpot/distributed/fmm.py`, cross-far M2L):

```python
xfar_cap = max(1 << 14, KC * num_tgt_leaves)   # >= 16384
x_src = cross.interaction_sources[:xfar_cap]   # tail is -1
x_valid = x_tgt >= 0                           # MASKED, not skipped
```

The cross-far M2L is evaluated over a padded buffer of at least 16 384 entries
and the invalid tail is multiplied by zero. At the shipped default, where
`cross_far = 0`, **every call performs >= 16 384 M2L operations that contribute
nothing.** Note this does *not* by itself explain the `theta_cross` penalty: if
the cap were the whole cost, adding 4109 real far pairs inside the same cap would
be free, and it is not. Both facts are true and unreconciled, which is precisely
what instrumentation is for.

## Measurement protocol (applies to every stage)

Non-negotiable, because it is what the retractions above cost:

- **Three independent invocations minimum**, medians compared. Within-run repeats
  measure allocator jitter, not the variance that matters.
- **The noise band at these sizes is ~10 ms / ~20%.** A difference smaller than
  that is not a result. The `theta_cross` penalty (~18 ms, reproducible across
  separate invocations) is right at the edge and was only trusted because 0.1 was
  measured twice and agreed to 2 ms.
- **Never infer validity from a wall clock.** Truncation reads as *faster*. Check
  `self_near_pairs` against the leaf-count law.
- **Quote the configuration with every number**: leaf size, `process_block`, caps,
  `theta_cross`, device count, per-device N. A per-device load without its leaf
  size is what caused the 8000-particle myth.

## STAGE 1 RESULT (2026-08-21): the evaluation is launch-bound

**642 distinct device ops and 15 518 kernel launches per device per force
evaluation** (4 devices, 16 384 particles/device, leaf 256). That alone accounts
for the floor: at a few microseconds of launch overhead each, 15.5k launches is
the whole ~75 ms of device time.

| op | ms/dev/eval | launches/dev/eval |
| --- | --- | --- |
| `nearfield_leafpair_t32_s256_w256` | 16.94 | **1** |
| `ncclDevKernel_AllGather_RING_LL` | 4.19 | 4 |
| `memcpy32_post` | 2.41 | **1136** |
| `RaggedAllToAllKernelImpl` | 1.95 | 3 |
| `MultiGpuBarrierKernelImpl` | 1.33 | 6 |
| `loop_multiply_fusion` | 0.91 | 256 |
| `loop_add_fusion_1` | 0.81 | 447 |
| `input_reduce_fusion` | 0.78 | 256 |

The fused near-field kernel does its work in **one** launch and is not the
problem. The collectives are ~10 ms and also not the problem. The remaining
~15 517 launches are, and the top dozen ops account for only ~1 300 of them: the
rest is spread over 642 distinct kernel types. That distribution -- many op
*types*, each launched a few hundred times -- is the signature of unrolled loops
emitting separate ops rather than a `scan` reusing one.

This is inside a jitted `shard_map`, so it is not eager dispatch: XLA compiled it
*into* 15.5k launches.

**Why five configuration hypotheses failed.** `theta_cross`, the fused-M2L flag,
the near-field cap, the traversal pair queue and the L2L level bound were each
predicted to dominate and each measured wrong. They were all *configuration*
explanations for a *dispatch* problem. (The L2L one died most cleanly: the library
raises `l2l_num_levels only applies to differentiable=True (the forward path
resolves the L2L depth exactly, on device)` -- so the audit's "~2 orders of
magnitude loose" bound is real but only on the differentiable path, which the
scaling benchmarks do not use.)

**Scope attribution does not work here and the tool says so.** Only ~5% of device
time lands in a `named_scope`: collectives keep their identity because they do not
fuse, everything else folds into anonymous fusion kernels. The instrument that
works is the op listing with **call counts** -- a slow stage shows up as time, a
dispatch-bound path shows up as a count. `bench/multigpu/stage_profile.py` reports
both and does not spread the unattributed remainder over the stages.

### RETRACTED (2026-08-21, same day): the sort is NOT the floor

**Measured, under `shard_map` at the same device count and per-device sizes:**

| per-device N | `argsort` only | full `Tree.from_particles` |
| --- | --- | --- |
| 1 024 | 2.90 ms | 2.75 ms |
| 16 384 | 2.55 ms | **0.95 ms** |

The tree build costs **1-3 ms**, not the 40-90 ms a floor of ~50-97 ms would need,
and it does not grow with particle count. The section below is wrong and is kept
only as a record of how it went wrong.

**Two lessons, both expensive.**

1. **Op count is a bad proxy for time.** The sort lambda is ~11 600 HLO ops, 40% of
   the module, and costs 1-3 ms. Never rank optimisation targets by op count.
2. **The "quantitative confirmation" was a coincidence.** The log^2(N) prediction
   (1.96x) matching the measurement (1.94x) was presented as a passing check. It
   was two points agreeing by chance. Two points are not a test.

### What the measurements actually support

| | per device per evaluation |
| --- | --- |
| near-field Pallas kernel | 16.9 ms (**1** launch) |
| collectives (AllGather + ragged + barrier) | ~7.5 ms (~13 launches) |
| tree build incl. Morton `argsort` | **1-3 ms** |
| ~15 496 remaining small kernels @ ~2.5 us | ~39 ms |
| total device time | ~75 ms |
| total launches | 15 510 |

About a third of the evaluation is real work and two thirds is thousands of small
kernels. The HLO shows those coming from per-iteration launches inside the
pipeline's `while` loops -- `jc_geometry/while/body`, `jc_upward_local/while/body`
(M2M), `jc_coarse_m2m/while/body`, and the L2L cascade -- which run on **every**
call irrespective of topology.

### Consequence: the prepare/evaluate split is NOT a performance fix

This is a direct answer to the plan's own top item, and it is negative. Hoisting
topology removes the tree build, which is ~2 ms of ~75 ms. It cannot recover the
floor, because the floor is in the numeric pipeline's loop structure, not in the
topology.

The split may still be worth doing for other reasons -- a stable state object
avoids recompiles when shapes change, it matches the single-device API, and it is
the natural home for a right-sized cap set -- but it must not be sold as a speed
fix, and it should not be prioritised as one.

**What the evidence does point at:** reduce the *number* of kernels in the numeric
pipeline. The per-level loops each launch several kernels per iteration; batching
across levels, or fusing the small per-level updates, attacks the two thirds. That
is an XLA-structure problem in the upward/downward sweeps, shared with the
single-GPU path, and it should be measured on the cheaper single-GPU path first.

### Scoreboard: six hypotheses, six failures

`theta_cross`, the fused-M2L flag, near-field padding, the traversal pair queue,
the L2L level bound, the Morton sort. Every one predicted this floor; every one was
measured and wrong. What survived every attempt was the launch count -- which was
visible in the very first profile and which no configuration knob moves.

### LOCALISED (2026-08-21): the floor is the Morton `argsort` in the tree build

The compiled HLO names it. Ops grouped by scope path, from
`module_*.jit_evaluate.sm_8.0_gpu_after_optimizations.txt`:

| HLO ops | scope path |
| --- | --- |
| **7978** | `jit(evaluate)/shard_map/jit(<lambda>)` |
| **3642** | `jit(evaluate)/shard_map/jit(<lambda>)/jit(_where)` |
| 890 | `jc_cross_walk/while/body/cond/branch_1_fun` |
| 604 | `jc_l2p/.../evaluate_local_real_with_grad` |
| 523 | `jc_upward_local/while/body/.../m2m_real` |
| 523 | `jc_coarse_m2m/while/body/.../m2m_real` |
| 514 | `jc_l2p/_propagate_solidfmm_locals_by_level/.../l2l_real` |

**~11 600 ops -- 40% of the module -- come from one anonymous jitted lambda**, an
order of magnitude more than any physics stage. The ops inside it are
`shift_left`, `shift_right_logical`, `xor`, `eq`, `iota`, `select_n` on `s64`:
bit manipulation, not arithmetic. That is Morton encoding and sorting.

`yggdrax.sort_by_morton` is `jnp.argsort`, which XLA lowers on GPU to a **bitonic
sort network**: O(log^2 N) stages, each a separate kernel launch, none of which
fuse with each other.

**The quantitative check passes.** If the floor is a launch-bound sort its cost
tracks the *number of stages*, log^2(N), not N:

| particles/device | log^2 N | predicted ratio | measured |
| --- | --- | --- | --- |
| 1 024 | 100 | -- | ~50 ms |
| 16 384 | 196 | 1.96x | **1.94x** (97 ms) |

This also explains every negative result at once. The sort depends on the
particle count alone, so it is invariant to `leaf_size`, `theta_cross`, the
near-field cap, the traversal queue and the L2L bound -- which is exactly what
five measurements found.

### CORRECTION: static structures were wrongly demoted

The re-ordering below demoted static structures on the reasoning that "a per-call
tree rebuild cannot be the floor when a 4-leaf tree also takes ~50 ms". **That
reasoning was wrong.** The rebuild's cost is dominated by the sort over
*particles*, which is independent of leaf count -- a 4-leaf tree still sorts 1024
particles. "Small tree" is not "little to sort".

So the original plan's Stage 4, and the instinct behind it, target precisely the
dominant cost. Revised priority:

1. **Hoist the tree build out of the per-call path** (was Stage 4, now first).
   The Morton codes and the sort are *topology*: at fixed topology they are
   recomputed every evaluation for no reason. A
   `prepare_distributed_state` / `evaluate(state, pos, mass)` split removes
   ~40% of the module's ops from the per-call path. The seam already exists --
   the differentiable path builds topology from `stop_gradient`-ed inputs and
   re-evaluates numerics on live ones -- so this is hoisting, not inventing.
   Hazards unchanged: the coarse tree's COMs are expansion centres and must stay
   live, and the host-side driver is not differentiable.
2. **If the sort must stay traced, replace `argsort` on the hot path.** A radix
   sort over fixed-width Morton keys is O(bits) passes rather than O(log^2 N)
   stages, and each pass is one kernel. This is an upstream `yggdrax` change and
   should be raised there rather than worked around here.
3. Right-size the cross-far M2L cap and revisit `theta_cross` -- unchanged, and
   still much easier to judge once the floor is gone.
4. Pallas cross M2L -- still gated on the far field being non-empty.
5. Communication -- ~10 ms of ~75 ms; real, still not dominant.

### Consequence: the stages below are re-ordered

The original ordering put a Pallas cross-M2L kernel and the static-structure
rebuild ahead of dispatch. Both address minority terms if 60% of the time is
launch overhead. Revised priority:

1. **Cut the launch count** (new Stage 2). Find what emits ~15.5k launches and
   whether unrolled loops can become `lax.scan`/`fori_loop`, so XLA emits one
   kernel run many times instead of many kernels. Also worth checking what
   `memcpy32_post` x1136 is: 1136 device-to-device copies per evaluation suggests
   layout changes or slicing that could be avoided.
2. Right-size the cross-far M2L cap and revisit `theta_cross` (old Stage 2) --
   still worth doing, and cheaper to judge once the launch floor is down, because
   an 18 ms difference is currently measured against a 50 ms floor of noise-adjacent
   overhead.
3. Static structures (old Stage 4) -- **demoted**. A per-call tree rebuild cannot
   be the floor when a 4-leaf tree also takes ~50 ms, and the profile shows the
   cost is not in one stage.
4. Pallas cross M2L (old Stage 3) -- **demoted**, and gated on the far field being
   non-empty and on the M2L actually appearing in the profile, which at
   `theta_cross=0.1` it does not.
5. Communication (old Stage 5) -- unchanged position. Measured at ~10 ms of ~75 ms,
   so it is real but not dominant.

## Stage 1 -- Instrument the sharded region (prerequisite)

Without this, the largest single cost in the evaluation cannot be attributed, and
four attempts prove that guessing does not work.

Route: a `jax.profiler` trace of one built evaluator, with XLA ops attributed to
pipeline stages (local build, geometry, self walk, upward, coarse gather, coarse
M2M, cross walk, cross M2L, halo import, near-field P2P, L2P). Non-invasive; the
brittleness is name-matching against fusion, which is acceptable for a
development tool.

Rejected alternatives, with reasons: a host callback per stage serialises the
thing being measured; stage ablation changes what XLA fuses.

**Acceptance:** the stage times sum to within ~20% of the measured wall clock at
two configurations (1024 and 16 384 particles/device), and name which stage owns
the ~50 ms floor. This also finally unblocks figure 10.

**Falsifies:** any claim about where the floor is, including the tree-rebuild
hypothesis in `docs/differentiable_fmm_distributed_audit.md`, which is plausible
but was never isolated -- and is in tension with a 4-leaf tree taking 50 ms.

## Stage 2 -- Right-size the cross-far M2L, and make the far field non-empty

Two changes that only make sense together, because each is what makes the other
worth having.

**2a. Size `xfar_cap` to the actual far volume.** Currently floored at 16 384
regardless. Options: derive from the walk's reported far count with a retry on
overflow (the pattern `auto_scale_caps` already uses), or skip the M2L entirely
when the far count is zero. The skip is trivial and, at the shipped default,
removes >= 16 384 no-op M2L operations per call.

**2b. Then loosen `theta_cross`.** Only after 2a, because today's ~18 ms penalty
is measured against a cost structure where the M2L is padding-dominated. With the
cap proportional to real work, the trade changes.

**Acceptance for 2a:** at `theta_cross=0.1` (far field empty), wall clock drops by
more than the noise band, with forces bit-identical -- it is a no-op removal, so
anything else is a bug.

**Acceptance for 2b:** a `theta_cross` sweep where relaxing it *reduces* total
time, **with a force-error check against direct summation at every point.** The
harness measures no accuracy today; that must be added first (Stage 2c) or the
sweep can only report speed, which is worthless for a criterion that trades
accuracy. This is the same instrument the MAC work will need.

## Stage 3 -- A Pallas kernel for the cross-domain M2L

Only after Stage 1 says the cross M2L is worth kernel work, and after Stage 2
makes its cost proportional to real interactions rather than to a constant.

The single-GPU fused real M2L exists (`_apply_real_m2l`, and the chunked
`_chunked_real_m2l_accumulate` the cross path already uses when
`config.m2l_chunk` is set). So this is plausibly *reusing* the existing kernel
against the coarse tree rather than writing a new one -- which should be checked
before any kernel is written.

**Acceptance:** parity against the current path to the level the golden tests
require, and a speedup outside the noise band on a far field made non-empty by
Stage 2.

## Stage 4 -- Static structures, payload-only rebuild

The single-device path builds `prepare_state` once concretely and reuses it. The
distributed path builds its tree *inside* `shard_map` on every call.

Worth separating two things that get conflated: **recompilation is already
avoided** by `make_force_evaluator` build-once with static shapes -- the steady
state we measure has no compilation in it. What is paid per call is tree
*construction*. So the win here is not "avoid recompiling", it is "stop rebuilding
a topology that has not changed".

Shape: a `prepare_distributed_state(...)` producing the frozen topology
(partition, `gid`, counts, node structure) plus an `evaluate(state, pos, mass)`
that re-gathers only the float payload into that frozen order. This is the same
seam the differentiable path already relies on -- it builds topology from
`stop_gradient`-ed inputs and re-evaluates numerics on live ones -- so the
mechanism exists and would be hoisted rather than invented.

Two hazards, both already documented: the coarse tree's COMs are expansion
centres and must stay live (freezing them is a real missing gradient term), and
the host-side driver is not differentiable, so the state object must be usable
from the `shard_map` evaluator.

**Acceptance:** forces bit-identical to a full rebuild at fixed topology; a
measured drop in per-call time at fixed topology; and the distributed gradient
tests still green, since this touches the seam they depend on.

**Prerequisite:** Stage 1. If the floor is not the rebuild, this is a large change
for nothing, and a 4-leaf tree taking 50 ms is reason for real doubt.

## Stage 5 -- Communication

`cross_near = 4096 x (ndev-1)` exactly: each device interacts with every other,
so per-device cross work grows linearly and total work quadratically. The coarse
exchange is an `all_gather` -- every device receives every domain's frontier.

The standard remedy is a locally-essential-tree fetch: pull only the coarse
subtrees a device's own domain actually needs. Note the upward sweep already
halved this traffic by shipping real rather than complex multipoles, so the
remaining win is in *volume of tree*, not in element size.

**Acceptance:** per-device cross work grows sublinearly in device count, and weak
scaling efficiency at 5 devices improves beyond the noise band.

**Deliberately last.** It is the largest change, and Stage 1 may well show
communication is not where the time goes -- the current section 5 text claimed it
was, on no evidence, and that claim has already been corrected once.

## Sequencing and what to run while GPUs are free

1. Stage 1 instrumentation -- pure development, no long runs, unblocks everything.
2. Stage 2a -- small, and its acceptance test is bit-identity plus a timing drop.
3. Stage 2c accuracy harness, then 2b sweep.
4. Re-run the scaling ladder at whatever configuration 2b lands on, with
   replicates, and regenerate figures 08/09/11.
5. Stages 3-5 in whatever order Stage 1's attribution justifies.

Also worth doing cheaply and independently: fix
`bench/multigpu/harness.py::make_distribution` to scale the box as N^(1/3), so
"more work per device" can be answered without confounding scale with density.
The present per-device sweep cannot distinguish the two.

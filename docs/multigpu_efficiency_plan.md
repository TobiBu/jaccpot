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

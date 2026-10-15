# Differentiable FMM — user guide

Exact gradients of the Jaccpot FMM force with respect to particle **positions**
and **masses**, at fixed tree topology, on a single GPU. Verified from N=64 to
N=1,000,000.

This is the reference page. The engineering history — what was measured, what was
tried and reverted, and why — is in
[`differentiable_fmm_design.md`](differentiable_fmm_design.md) and
[`differentiable_fmm_audit.md`](differentiable_fmm_audit.md). Read those before
changing the reverse pass; read this one to use it.

---

## Quick start

```python
import jax, jax.numpy as jnp
from jaccpot import FastMultipoleMethod

fmm = FastMultipoleMethod(basis="real", theta=0.6, softening=1e-3)

# Build the topology ONCE, concretely, outside the differentiated function.
state = fmm.prepare_state(positions0, masses0, max_order=4, leaf_size=64)

def loss(positions, masses):
    accel = fmm.differentiable_accelerations(state, positions, masses)
    return jnp.sum(accel ** 2)

grad_positions, grad_masses = jax.grad(loss, argnums=(0, 1))(positions, masses)
```

That is the whole interface. `differentiable_accelerations` returns `(N, 3)`
accelerations in the original input order and is differentiable in both
arguments.

### Galaxy scale (N ≳ 10⁵)

```python
# required: keeps the frozen M2L pair list. A CONSTRUCTOR option (or
# FarFieldConfig(retain_far_pairs_for_grad=True), or
# JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD=1); prepare_state raises TypeError on it.
fmm = FastMultipoleMethod(
    preset="large_n_gpu", basis="real", retain_far_pairs_for_grad=True
)
state = fmm.prepare_state(positions0, masses0, max_order=4, leaf_size=256)

# Hoist the plan out of the optimisation loop.
from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan
plan = prepare_large_n_grad_plan(fmm, state)

grads = jax.grad(loss)(positions, masses)   # pass grad_plan=plan inside `loss`
```

`preset="large_n_gpu"` gives a `LargeNPreparedState` only on a GPU; on a CPU it
prepares an ordinary radix state.

Measured 2026-07 on an A100 40 GB, fp32, θ=0.7, order 4, clustered galaxy disc,
before the Pallas reverses of 2026-09 (so this is the prepacked-rectangle reverse
the performance notes below describe):

| N | prepare | forward (steady) | forward+backward (steady) | reverse peak |
|---|---|---|---|---|
| 200 k | 57 s | 0.86 s | 2.59 s | 2.62 GB |
| 1 M | 79 s | 2.50 s | 66.4 s | 11.07 GB |

`prepare_state` is dominated by **compilation**, not computation: at 1 M the
first call is 82 s and the second is 1.5 s. Set `jax_compilation_cache_dir` and a
cold process drops from 81 s to 17 s.

**The current numbers** (2026-10-13, [`production_2026-10.md`](production_2026-10.md);
`bench/grad_step_profile.py`: cells of 64, θ=0.8, order 5, fp32, jitted gradient
w.r.t. positions and masses):

| N | production force eval | jitted gradient | process peak |
|---|---|---|---|
| 2e5 | 4.63 ms | 31.9 ms | 3.22 GiB |
| 1e6 | 13.0 ms | 101.6 ms | 15.8 GiB |
| 8e6 | 74.5 ms | out of memory (a 27.37 GiB request) | 19.5 GiB |

---

## What the gradient lane covers (checked against main, 2026-10-15)

* **Positions and masses only.** `G` and the softening enter as Python floats
  (`-float(engine.G)` in `runtime/_large_n_grad.py`, `float(softening)` in the
  near field), so there is no d/dG and no d/dε on this lane.
* **Reverse mode only.** Every fast kernel is a `custom_vjp`, and JAX refuses
  `jax.jvp` through one (`TypeError: can't apply forward-mode autodiff (jvp) to a
  custom_vjp function`, on a CPU radix state too). Forward-mode adjoints, such as
  diffrax's `ForwardMode`, fail; reverse-mode ones work.
* **One force at a time.** `strict_run_v2` and the multi-GPU `FusedRollout` are
  forward-only: the strict refresh evaluates the forward-only
  `evaluate_large_n_state`, and the cross-domain M2L and near-field reverses raise.
  A rollout gradient differentiates its own integrator around
  `differentiable_accelerations`.
* **Slower layouts than the forward.** On an Ampere+ GPU the gradient runs the same
  Pallas P2M, cascades, CSR M2L and CSR near-field kernels as the fused forward, each
  with a Pallas reverse ([`fast_gradients_2026-09.md`](fast_gradients_2026-09.md)).
  Two forward kernels have no reverse, so the gradient takes another layout: the
  near field runs the CSR lane's table layout instead of the production direct
  layout (`nearfield/_fast_lane.py:1829`), and the L2P runs the leaf-major padded
  XLA L2P with its analytic VJP instead of the particle-major Pallas L2P.
* **Inert options on that path.** `GradConfig.fused_m2l_pallas` and the five
  `reverse_*` fields tune lanes the default Ampere path does not run (see the
  options table below), and `nearfield_lane` is ignored on a `LargeNPreparedState`.

---

## The contract: fixed topology

Gradients are taken at **fixed topology**. The reverse pass differentiates the
numeric pipeline — P2M, the centre-of-mass expansion centres, the M2M/M2L/L2L
translations, L2P, and the near-field P2P — while treating every integer index
array as a constant: the Morton permutation, node membership, the M2L interaction
list, the near-field neighbour CSR, and every MAC accept/reject decision.

What this drops is the *implicit* dependence of the force on position through
"which cell a particle lands in". That contribution is nonzero only on the
measure-zero set where a pair crosses a MAC boundary — the same piecewise
structure Dehnen (2014) App. B.3 describes for the force itself. For a small
enough finite-difference step no particle crosses a boundary, so finite
differences and autodiff agree to FMM force accuracy.

**Expansion centres are differentiated through, not held fixed.** They are
centres of mass, so they depend on positions and masses, and the cotangents flow
through them. Only the pair list and the near/far partition are frozen.

Two consequences worth internalising:

* **`prepare_state` is not traceable** (it does host-side tree construction), so
  the topology must be built before `jax.grad`. This is the contract, not a
  limitation to work around.
* **A finite-difference reference must perturb the same frozen function.** FD of
  a full `compute_accelerations`, which rebuilds the tree and the MAC, will
  disagree near a partition flip. That is expected.

### Accuracy

`jax.grad` of the FMM matches `jax.grad` of an exact direct O(N²) sum
(`jaccpot.autodiff.direct_sum_gravitational_acceleration`) to **the FMM's own
force accuracy** — not to machine precision. If you need a tighter gradient, you
need a more accurate force: raise the expansion order or lower θ.

Worth knowing independently of gradients: in 2026-07 the shipped `large_n_gpu`
preset had **~6 % force error** (rel-L2 6.1e-2, worst component 27 %) at order 4 on
a clustered galaxy disc. The lane has changed a lot since (the fused lane, COM
expansion centres, cell leaves), so measure the configuration you differentiate:
its force accuracy is the accuracy your gradients are gradients *of*.

---

## Configuration

Use [`GradConfig`](../jaccpot/config.py). Every field also has a `JACCPOT_*`
environment fallback for existing scripts, and an explicit field always wins:

```
explicit GradConfig field  >  JACCPOT_* environment variable  >  measured default
```

```python
from jaccpot import GradConfig

cfg = GradConfig(nearfield_lane="fast_lane")
accel = fmm.differentiable_accelerations(state, positions, masses, grad_config=cfg)
```

### The one setting that matters on a radix state: `nearfield_lane`

On a radix `FMMPreparedState` two traversals of the **same edge set**
(bit-identical force checksums) are available. (It mattered most when the near
field was ~83 % of the forward and ~91 % of the reverse: N=1024, order 4, 2026-07.)
A `LargeNPreparedState` has one differentiable near field: it ignores `"auto"` and
`"fast_lane"`, and raises on an explicit `"bucketed"`.

| lane | what it is | when |
|---|---|---|
| `"bucketed"` | edge-list kernel | best at small N |
| `"fast_lane"` | leaf-major; with `use_pallas` on Ampere+ it takes the **CSR row-chunk kernel and its Pallas reverse** (since 2026-09). The **analytic O(N) leaf-pair rule** of the prepacked rectangle is the fallback (`JACCPOT_NEARFIELD_LEAFPAIR_CSR=0`, or interpret mode) | the only lane that survives galaxy scale |
| `"auto"` (default) | bucketed below `nearfield_fast_lane_min_particles` (100 000), fast lane at or above | — |

**Why the default is `"auto"` and not `"bucketed"`:** at N=200 000 the bucketed
reverse peaks at 30 GB and OOMs; the fast lane completes in 6.8 GB. Requiring a
user to know that in advance is a trap, so the crossover is applied for you.

At N ≤ 4096 the two are within ±20 % with no consistent winner — the lane buys
*asymptotic reverse memory*, and by the time it matters it is not a speedup, it
is feasibility.

### Other options

| field | default | effect |
|---|---|---|
| `cascade_pallas` | on | The Pallas leaf P2M and M2M/L2L cascades with their reverse kernels (Ampere+, or interpret mode). `False` restores the pure-JAX level loops, for A/B measurement. |
| `fused_m2l_pallas` | off | Fused-Pallas M2L (Ampere+, falls back elsewhere). **Inert where the CSR M2L runs** (the default on Ampere+ since 2026-09-10): it acts only with `JACCPOT_STATIC_STRICT_FUSED_M2L_CSR=0`. Measured 2026-07, before the CSR M2L: ~2× the forward at small N; ~1.3–1.5× end-to-end since the reverse is near-field-bound. |
| `reverse_tiers` | 4 | Occupancy tiers for the analytic reverse; lets low-occupancy leaves read a narrower slot window. |
| `reverse_tier_min_gain` | 3.0 | Predicted slot-visit reduction below which tiering is declined. |
| `reverse_skip_empty_tiles` | on | Skip reverse tiles with no valid source slot. Semantics-preserving. |
| `reverse_leaf_batch` / `reverse_block_tile` | 8 / 8 | Reverse tiling, deliberately independent of the forward's. |
| `analytic_p2p_vjp` / `analytic_l2p_vjp` | on | Analytic reverse rules. Turn off only for A/B measurement. |

**The five `reverse_*` fields tune the prepacked rectangle's leaf-pair reverse
only.** On an Ampere+ GPU the near field takes the CSR lane, whose reverse they do
not reach, so they act only with `JACCPOT_NEARFIELD_LEAFPAIR_CSR=0` (or in interpret
mode), and only on a radix state: a `LargeNPreparedState` does not pass them on (it
reads the `JACCPOT_GRAD_REV_*` variables instead).

Tiering is calibrated on **two points on one A100** (declined at 200 k where it
would be 4.3× slower, accepted at 1 M where it is 1.9× faster). Re-measure before
trusting the 3.0 threshold on other hardware.

---

## Requirements and limits

**Supported states**

* Radix `FMMPreparedState` with the real basis (`basis="real"`, or its alias
  `"solidfmm"`; the complex basis was removed in the 2026-10 cleanup, C).
* `LargeNPreparedState` from `preset="large_n_gpu"`, with
  `retain_far_pairs_for_grad=True`. Without it the frozen M2L pair list is
  discarded at `prepare_state` and the call raises — it does not silently give
  you a far field with zero mass-sensitivity. The state `strict_run_v2` returns
  carries no far list either (its scan rebuilds the list every step; the one it
  started with is the start positions'), so it raises too: `prepare_state` at
  the returned positions and differentiate that.

**Not supported** (all raise, none degrade silently)

* `expansion_basis="cartesian"` -- since the 2026-10 cleanup (K) the basis is
  gone and naming it raises at construction, before any state exists.
* `target_indices` on the large-N path.
* Potentials on the fast lane (`return_potential=True`).
* The radix *overflow* and *target-block* near-field payloads, and the
  materialized per-particle "pairs" layout — each would differentiate a different
  force than the forward computes.
* `jax.jvp` / forward mode (the `custom_vjp` refuses it with a `TypeError`).
* `strict_run_v2` and the multi-GPU `FusedRollout`: forward-only (their
  cross-domain reverses raise; the strict refresh has no differentiable seam).

**Not differentiable on this lane:** `G` and the softening length. The solver
holds them as Python floats and the kernels read them as constants, so there is no
gradient w.r.t. either here.

**There is no grouped M2L any more.** The grouped/class-major M2L classified
pairs on the host, so it was not traceable, and it was an *approximation*: one
representative displacement per lattice class (the flat grad path was **6.6×
more accurate** against an exact direct sum on a deliberately deep tree). The
grad path used to force it off; it was removed in the 2026-10 cleanup
(`docs/cleanup_2026-10.md`, X3, with its `grouped_interactions` option in Z-b), so
the forward and the gradient always run the same far field.

**Outer `jax.jit`.** Bare `jax.grad`/`jax.vjp` is the recommended usage — the
inner kernels are already jit-compiled. Wrapping the entire call in `jax.jit`
works at moderate N but can hit host-side ops in the re-run sweeps at large N.
Reverse *compile* time is substantial at scale: ~10 min at N=16384, ~25 min at
N=1 M.

**`OdisseoFMMCoupler`.** Pass `differentiable=True`. The default forward path
evaluates prebaked expansions and reads no live input, so differentiating it
would return exactly zero; it now raises instead.

**Multi-GPU** gradients are a separate entry point, not this one:

```python
from jaccpot.distributed import DistributedFMMConfig
from jaccpot.distributed.fmm import make_force_evaluator, partition_for_devices

part = partition_for_devices(positions, masses, ndev, leaf_size=config.leaf_size)
evaluate = make_force_evaluator(
    config, ndev, part["cap"], mesh, jit=True, differentiable=True
)
# gradients are w.r.t. the PADDED, per-device layout `part` produces;
# `accel` rows come back in per-device Morton order -- map them with the gid output.
g = jax.grad(lambda p, m: jnp.sum(evaluate(p, m, gid, counts)[0] ** 2))(pos_f, mass_f)
```

Same fixed-topology contract, but the topology is rebuilt inside every call (the
tree build lives inside `shard_map`), so there is no state to hoist. Differentiate
the evaluator, **not** `distributed_fmm_accelerations` — that one partitions and
reassembles in NumPy and is not traceable. `nearfield_chunk` raises on this path.
Verified for correctness on 2 GPUs only. It is **correct, not fast**: the reverse
costs roughly 6-11x the forward at N=512-8192 and the reverse compile is 3-6 min.
There is no known quick win — the reverse hotspot has not been profiled, and
`l2l_num_levels` (the one obvious knob) measures ~6% at N=512 and nothing at
N=2048, so treat it as a correctness setting rather than a speed lever. If you do
set it, pass a value >= your per-device tree depth and check `l2l_level_overflow`
in the diagnostics, which fires if it was too small. Details and measurements:
`docs/differentiable_fmm_distributed_audit.md`.

That is the mesh lane (`distributed/fmm.py`). The fused multi-GPU lane
(`distributed/fused.py`, `FusedRollout`) has no gradient: its cross-domain M2L and
near-field reverses raise `NotImplementedError` ("the one-sided cross-domain lane
is forward-only").

---

## Performance notes

**Dated: these notes are from 2026-07**, when the galaxy-scale reverse was the
prepacked rectangle's leaf-pair rule. On an Ampere+ GPU the near field now runs the
CSR lane, whose rows are the real ones rather than a padded rectangle, with a Pallas
reverse ([`fast_gradients_2026-09.md`](fast_gradients_2026-09.md)); the rectangle
analysis below applies only where that lane is off. For where a current gradient's
time goes, see the stage split in [`production_2026-10.md`](production_2026-10.md)
(at 1e6: upward 14.5, far 56.8, near 47.1 ms of the gradient).

**Where the time goes.** At galaxy scale the reverse is 94–98 % near field; the
M2L/L2L/P2M cascade is 0.7 s of 66 s at 1 M. Do not optimise the far field.

**Why the 1 M reverse is 27× the forward.** The prepacked payload is a rectangle
padded to the *global maximum* neighbour count, and some leaf neighbours every
other leaf at every N and every geometry tried. Fill is 45 % at 200 k and 14.5 %
at 1 M, and the reverse cost tracks padded slots almost exactly (padding grows
40× for 5× N; the measured reverse grows 38×). Padded particle-pair work is
`(leaves × leaf)² = N²` in every configuration — the padded reverse is running a
direct sum.

**`leaf_size` and `theta` are the accuracy dial, not free performance.**
Near-field work and force error are monotonically related, because the near field
*is* the exact part of the calculation. An M2L pair costs ~1.9× a full
near-field leaf-pair block, so shrinking the near field by pushing work to the
far field is a net loss. The shipped defaults sit at the accurate end.

**Padding is geometry, not just N.** Fill (valid / padded slots) at leaf 256 is
21.8% for a uniform cube and 80.8% for a clustered disc at N=200 000 — same leaf
count, same MAC, different particle distribution. At 1 M it is 5.9% vs 39.9%. So a
uniform/cosmological volume wastes far more of the reverse on padding than a
galaxy disc does, and the tuning that pays differs accordingly. Check yours with
`python bench/audit_nearfield_padding.py`; it reads the structure off a prepared
state in seconds and needs no reverse pass. Decision record for the CSR-sources
rewrite this bears on: [`differentiable_fmm_csr_sources_plan.md`](differentiable_fmm_csr_sources_plan.md).

**Method note, from four attempted optimisations.** Three were predicted to win
and did not: the leaf-major traversal (±20 % at small N, though decisive for
memory at 200 k), the empty-tile skip (1.00× at 200 k), and a global occupancy
sort (~7× *worse* — it destroyed the Morton-order locality of the source gather).
Isolated micro-benchmarks mispredicted this subsystem by 30×, then by 6–20×.
Measure in context, at the target scale.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| OOM in the backward at N ≳ 10⁵ | bucketed reverse retains O(pairs) residuals | leave `nearfield_lane="auto"`, or set `"fast_lane"` |
| OOM at 8e6 on a `large_n_gpu` state (2026-10-13) | the full and the far-field gradients each ask for ~27.4 GiB beyond the prepared state; the far reverse runs the leaf-major padded L2P and leaf-table layouts | no option fixes it today; production-layout reverses are planned (`production_2026-10.md`) |
| `RuntimeError: compact_far_pairs is None` | large-N state built without the pair list | construct the solver with `retain_far_pairs_for_grad=True` |
| `TypeError: ... prepare_state() got an unexpected keyword argument 'retain_far_pairs_for_grad'` | it is a constructor option | pass it to `FastMultipoleMethod(...)` |
| `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` | forward mode (`jax.jvp`, a forward-mode adjoint) | use reverse mode |
| all-NaN gradients in fp32 | historical; the squared-radius floors underflowed in fp32 | fixed — ensure you are on this branch |
| `ConcretizationTypeError` under outer `jax.jit` | host-side ops in the re-run sweeps at large N | use bare `jax.grad` |
| gradients are exactly zero | you differentiated `evaluate_prepared_state` or the Odisseo forward path | use `differentiable_accelerations` / `differentiable=True` |
| `TracerArrayConversionError` on centres | historical: the grouped M2L reached the grad path | removed in the 2026-10 cleanup (X3); file a bug |
| FD and AD disagree | FD perturbed a *different* function (full rebuild) | FD the same frozen `state` |

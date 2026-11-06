# Differentiable FMM — user guide

Exact gradients of the Jaccpot FMM force with respect to particle **positions**
and **masses** (and, on a `large_n_gpu` state, `G` and the softening), at fixed tree
topology, on a single GPU. Verified from N=64 to N=1,000,000; on the production
layout, memory measured to N=10^8 (plan S8).

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
import jax, jax.numpy as jnp
from jaccpot import FastMultipoleMethod, ForceParams
from jaccpot.runtime._large_n_grad import prepare_large_n_grad_plan

# required: keeps the frozen M2L pair list. A CONSTRUCTOR option (or
# FarFieldConfig(retain_far_pairs_for_grad=True), or
# JACCPOT_RETAIN_FAR_PAIRS_FOR_GRAD=1); prepare_state raises TypeError on it.
fmm = FastMultipoleMethod(
    preset="large_n_gpu", basis="real", retain_far_pairs_for_grad=True
)
state = fmm.prepare_state(positions0, masses0, max_order=4, leaf_size=64)
# once per topology: the frozen lists, checked symmetric (the reverses need it)
plan = prepare_large_n_grad_plan(fmm, state)

def loss(state, plan, positions, masses, params):
    acc = fmm.differentiable_accelerations(
        state, positions, masses, grad_plan=plan, force_params=params
    )
    return jnp.sum(acc ** 2)

# the state and the plan are ARGUMENTS of the compiled gradient (both pytrees):
# closed over, jax.jit embeds the whole topology in the program as constants
grad = jax.jit(jax.grad(loss, argnums=(2, 3, 4)))
params = ForceParams(G=jnp.asarray(1.0), eps2=jnp.asarray(softening ** 2))
g_positions, g_masses, g_params = grad(state, plan, positions, masses, params)
```

`force_params` is optional (`None` is the solver's own `G` and softening, the
production program); with arrays, `g_params.G` and `g_params.eps2` are the
derivatives in `G` and the softening squared at the frozen lists.
`fmm.differentiable_step_fn(state)` builds the same binding for you (it returns
`f(positions, masses[, force_params])`).

`preset="large_n_gpu"` gives a `LargeNPreparedState` only on a GPU; on a CPU it
prepares an ordinary radix state.

**The gradient runs the production forward's kernels** (plan S8, 2026-10-22,
[`production_2026-10.md`](production_2026-10.md) "S8"): the near field's CSR direct
layout, the M2L over the walk's target-sorted far CSR, the M2M/L2L cascades, the
leaf P2M and the particle-major L2P, each under a `custom_vjp` with a reverse Pallas
kernel, so the differentiable forward is the production evaluation **bitwise**
(pin G1). Measured on A100s with `bench/grad_production_gate.py` (clipped Plummer,
cells of 64, θ=0.8, order 5, fp32; the gradient of `sum(w · acc)` in positions and
masses; [`production_2026-10.md`](production_2026-10.md) "S8" has the card states):

| N | force eval | value_and_grad / grad | temporaries (v&g / grad) | device peak |
|---|---|---|---|---|
| 8e6 | 30.0 ms | 161.6 / 131.5 ms (5.4x / 4.4x) | 1.98 / 1.71 GiB | 3.65 GiB |
| 2.5e7 | | | 5.18 / 4.33 GiB | 9.42 GiB |
| 1e8 (production leaf capacity) | | | 19.15 / 13.92 GiB | 28.29 GiB (grad) |

Two identical gradients are bitwise equal. `value_and_grad` keeps the forward's arrays
through the reverse (+5.2 GiB at 1e8): when memory binds, take the value from a
separate forward.

`prepare_state` is dominated by **compilation**, not computation (2026-07: at 1 M
the first call 82 s, the second 1.5 s). Set `jax_compilation_cache_dir` (or
`JACCPOT_COMPILATION_CACHE_DIR`, which the runner and `GradientPlan` apply: see
[Forward mode](#forward-mode-ad_modeforward)) and a cold process drops from 81 s to
17 s.

---

## What the gradient lane covers (plan S8, 2026-10-22)

* **Positions, masses, `G` and the softening squared.** `force_params=ForceParams(G,
  eps2)` with arrays differentiates the force in both at the frozen lists (the
  fused lane's traced parameters, S2; FD-checked in float64). For a compact
  softening kernel the derivative in `eps2` is the frozen lists', valid up to the
  softening the walk was built for (`FusedForceProvider.walk_eps2`).
  `mac_type="dehnen_error"` refuses any `force_params` but the solver's own
  (`ValueError`: its eq (16a) thresholds were built with them); its positions and
  masses differentiate at its frozen lists like any other.
* **Reverse mode by default; forward mode opt-in on the large-N lane.** Every fast
  kernel is a `custom_vjp`, and JAX refuses `jax.jvp` through one (`TypeError:
  can't apply forward-mode autodiff (jvp) to a custom_vjp function`).
  `ad_mode="forward"` routes the large-N lane's seams to `custom_jvp` siblings
  instead, for `jax.jvp`, `jax.jacfwd` and `jax.linearize` (see
  [Forward mode](#forward-mode-ad_modeforward) below). The radix lane is reverse
  mode only.
* **One force at a time.** `strict_run_v2` and the multi-GPU `FusedRollout` are
  forward-only: the strict refresh evaluates the forward-only
  `evaluate_large_n_state`, and the cross-domain M2L and near-field reverses raise.
  A rollout gradient differentiates its own integrator around
  `differentiable_accelerations` (plan S9 builds the differentiable step).
* **The production layout.** Since S8 the gradient's forward is the production
  evaluation, kernel for kernel (above). Before it, two forward kernels had no
  reverse and the gradient ran other layouts in their place -- the CSR near
  field's table layout, the leaf-major XLA L2P -- which also ran out of memory at
  8e6. The near field's reverse reads every pair from both ends and the M2L reverse
  reads the forward's CSR in both of its passes: both need the lists to be
  symmetric, which `prepare_large_n_grad_plan` checks (it refuses an asymmetric
  near list; an asymmetric far list takes the sorting M2L reverse).
* **Not on the lane.** The caller's external field (`strict_run_v2`'s
  `external_acceleration_fn`) is outside the self-gravity: differentiate it in the
  caller. The near field's overflow and target-block payloads of the pure-JAX
  rectangle route are refused by the plan (the CSR lane never builds them).
* **Inert options on that path.** `GradConfig.fused_m2l_pallas` and the five
  `reverse_*` fields tune lanes the default Ampere path does not run (see the
  options table below), and `nearfield_lane` is ignored on a `LargeNPreparedState`.

---

## Forward mode (`ad_mode="forward"`)

Forward mode differentiates the large-N lane (`preset="large_n_gpu"` /
`"galaxy"`: `FusedForceProvider.differentiable_step`, `differentiable_rollout`,
`GradientPlan`, `differentiable_accelerations` / `differentiable_step_fn` on a
`LargeNPreparedState`) with `jax.jvp`, `jax.jacfwd` and `jax.linearize`. It is a
second dispatch beside the reverse one: each production kernel's `custom_vjp` has a
`custom_jvp` sibling (`jaccpot/runtime/_forward_seams.py`) whose primal is the same
production kernel -- so the forward is the production forward **bitwise** -- and
whose tangent is a tangent kernel (Pallas where one exists, else a pure-`jnp`
reference). The reverse programs are unchanged.

```python
from jaccpot.runtime.grad_rollout import differentiable_rollout
from jaccpot.runtime.grad_options import ad_mode_override

def final_state(x0, v0, m, dt):
    out = differentiable_rollout(provider.differentiable_step, x0, v0, m, dt, T,
                                 ad_mode="forward", checkpoint=None)
    return out.positions, out.velocities

# one direction
(xT, vT), (dxT, dvT) = jax.jit(lambda p, t: jax.jvp(final_state, p, t))(
    (x0, v0, m, dt), (dx0, dv0, dm, ddt))
# a few parameters: one batched program (the tangent kernels carry up to 8 per launch)
J = jax.jit(jax.jacfwd(lambda theta: ...))(theta)

# a single step, outside a rollout: install the mode around the trace
with ad_mode_override("forward"):
    acc, t_acc = jax.jit(lambda x, t: jax.jvp(
        lambda y: provider.differentiable_step(y, m)[0], (x,), (t,)))(x, t)

# the production runner's gradient run (S9): forward twins of value_and_grad
plan = runner.gradient_plan(state, masses, dt=dt, num_steps=T)
res = plan.jvp(None, {"positions": dx0}, wrt=("positions",))   # value, tangent, result
jac = plan.jacfwd(lambda r, inp: jnp.sum(w * r.positions), wrt=("dt",))
lin = plan.linearize(wrt=("positions",)).tangent                  # tangents -> output

# more columns than fit a card: k directions per pass, in one jitted program
from jaccpot.runtime import forward_memory_estimate, jacfwd_batched
k = forward_memory_estimate(n=len(x0), k=8)["largest_k_that_fits"]   # 1 at 1e8 / A100-40
jac = plan.jacfwd(fn, wrt=("external_params",), directions_per_pass=k)
J = jax.jit(jacfwd_batched(lambda theta: ..., directions_per_pass=1))(theta)
cols = lin.apply({"positions": D}, directions_per_pass=2)   # D: [n_dirs, N, 3]
```

* **Selecting it.** `ad_mode="forward"` on `differentiable_rollout`; the
  `GradientPlan` methods above; `GradConfig(ad_mode="forward")` for
  `differentiable_accelerations` / `differentiable_step_fn`;
  `jaccpot.runtime.grad_options.ad_mode_override("forward")` around any trace; or
  `JACCPOT_GRAD_AD_MODE=forward`. `GradConfig.forward_tangent_kernels` (or
  `JACCPOT_GRAD_FORWARD_TANGENT_KERNELS`) picks the tangent kernels: `"auto"`
  (Pallas where it exists, else the jnp reference), `"jnp"`, `"pallas"`.
* **Memory is flat in the number of steps.** A jvp through a `lax.scan` stores no
  residuals, so `differentiable_rollout(ad_mode="forward")` is the plain scan
  (`checkpoint=None`; the default `"step"` means the same; `"nested"` /
  `"offload"`, reverse-pass storage schedules, are refused). The cost is one
  tangent pass per direction (`jax.jacfwd` batches them); the adjoint's
  `W + 16 B/p` per step (S9) does not apply. `jax.linearize` keeps the
  linearization point's intermediates, so it grows with the steps like a reverse
  pass.
* **Memory per direction, and `directions_per_pass`.** `jax.jacfwd` pushes every
  column through the program at once: the tangent kernels carry all `K` columns,
  and the device peak is `W(K) = a + b K` bytes per particle (round-1 F8, A100-40,
  galaxy preset at p = 6, float32, `T = 16`: `a` ~ 260-370 and `b` ~ 120-195 B/p
  per column, smaller at larger N). At **1e8 on an A100-40 one direction already
  peaks at 384 B/p = 35.8 GiB** (of 37.5 GiB usable at 0.95 preallocation), so a
  Jacobian there runs **one direction per pass**.
  `GradientPlan.jacfwd(..., directions_per_pass=k)` and
  `jaccpot.runtime.jacfwd_batched(f, directions_per_pass=k)` (any forward-mode
  function, `jax.jacfwd`'s result and structure) build ONE jitted program that
  `lax.map`s over chunks of `k` columns, each a `jax.jvp` vmapped over its `k`
  columns (the tangent kernels see `K = k`), the last chunk padded with zero
  directions and sliced off. Memory is `W(k)`; the price is the primal recomputed
  per chunk plus one primal after the loop for the returned value (ordered there by
  an `optimization_barrier`, so no `N`-sized final state is carried through the
  loop). `None` (default) is today's single-pass program, unchanged. The result
  equals the single pass to round-off (gate G-batch). `plan.linearize(...).tangent`
  is a `LinearMap`: call it on one direction, or `.apply(directions,
  directions_per_pass=k)` on a leading axis of directions.
  `jaccpot.runtime.forward_memory_estimate(n, k, steps)` is the F8 model: the
  estimated peak and the largest `k` that fits a card (`card_bytes`, default an
  A100-40 at 0.95). An estimate from one card and one configuration: measure with
  `bench/forward_mode_memory.py` before a production run.
* **Tangent groups: less memory per direction, primal once
  (`JACCPOT_FORWARD_TANGENT_GROUP=<g>`, opt-in, round 2).** Most of `b` is the
  force's own tangent intermediates, `K`-stacked: at the M2L the multipole-tangent
  table is read while the local-tangent table is written, two `f32[K, nodes, 49]`
  tables (p = 6) live at once. With the knob set (`g >= 1`; unset or `0`: the
  ungrouped program, unchanged), each force evaluation of the fused lane
  (`FusedForceProvider.differentiable_step`, so every forward-mode rollout,
  `jacfwd`, `GradientPlan` forward twin built on it) is traced once, linearized
  once -- the primal runs once, with the production kernels -- and its linear
  tangent map is applied `g` directions at a time in a loop inside ONE primitive
  (`jaccpot.runtime._forward_group`): every tangent kernel runs at `K = g` and only
  `g` directions' tables are live; a partial last group runs once more at its own
  size. Unlike `directions_per_pass` nothing is recomputed, and it composes with
  it. Only the tangents crossing the force's boundary (positions, masses, `G`,
  `eps2`, accelerations, the integrator's carries) stay `K`-stacked. The tangents
  equal the single pass to round-off (bitwise on CPU, float64; strict test
  `test_tangent_groups_are_the_single_pass`). F8 at `T = 16`, `g = 2` (A100,
  contended: memory and compile only), temporaries in B/p:

  | N | K = 1 | K = 8 (ungrouped) | slope per direction (ungrouped) | compile K = 8 |
  |---|---|---|---|---|
  | 2e6 | 494 (494) | 990 (1690) | 71 (171) | 392 s (417 s) |
  | 8e6 | 410 (410) | 877 (1397) | 67 (141) | 387 s (414 s) |

  `K = 1` is unchanged (one group). At 2e6, `K = 8`, by group size: `g = 1` 854
  B/p (compile 179 s: the tangent kernels at `K = 1` only), `g = 2` 990 (392 s),
  `g = 4` 1265 (398 s), `g = 8` = ungrouped 1690 (417 s): each direction a group
  holds costs ~137 B/p, each direction outside it ~51 (the boundary tangents).
  The run-time price (`K / g` launches of each
  tangent kernel instead of one) is measured on HoreKa
  (`t_2p5e7_jacfwd_k8_group{1,2,4}` in `bench/slurm/forward_mode_timing.sbatch`);
  whether it becomes the default is decided there.
* **Compile time, tangents per program, and the persistent compilation cache.**
  Round 1's tangent kernels unrolled all `K` columns in one Triton program body, and
  their compile grew superlinearly in `K` (A100, 2e6 `jacfwd`: `K = 1` ~5 min,
  `K = 4` 23 min, `K = 8` ~75 min). Since round 2 every Pallas tangent kernel (P2M,
  M2M, L2L, M2L, L2P, near field) has a grid axis over *tangent groups*: one program
  carries `tpp` columns (`jaccpot.pallas._tangent_grid`), a partial last group is
  masked, so any `K` runs without padding. Choose `tpp` per seam with
  **`JACCPOT_TANGENT_TPP_<SEAM>`** (`SEAM` in `P2M M2M L2L M2L L2P NEAR`; values
  1/2/4/8, default 2, clamped to the power of two at or above `K`; `tpp = K` is the
  old fused body). A smaller `tpp` compiles faster and spills less but recomputes
  the shared primal terms once per group. Compile of the 2e6 `jacfwd` at `K = 8`,
  tpp 2 everywhere: ~7 min (418 s, contended A100; per kernel at tpp 1 / 2 / 8:
  M2M 137 / 377 / 4243 s, L2L 37 / 131 / 1598 s, M2L 45 / 114 / 1565 s). The
  run-time defaults come from the HoreKa sweep (`bench/forward_mode_timing.py`).
  Set **`JACCPOT_COMPILATION_CACHE_DIR=<dir>`** (opt-in,
  off by default) and the `ProductionRunner`, `GradientPlan` and the forward-mode
  benches point JAX's persistent compilation cache there
  (`jaccpot.runtime.compile_cache.maybe_enable_compilation_cache`: once per
  process, idempotent, re-armed if something compiled before it was set), so a
  second process loads the executable instead of compiling it.
  `JACCPOT_COMPILATION_CACHE_MIN_COMPILE_S` (default 1 s) skips the small programs;
  `JACCPOT_COMPILATION_CACHE_MAX_BYTES` bounds the directory (LRU eviction). Entries
  match only the same jax / jaxlib / CUDA / driver and the same program (N, order,
  leaf capacity, `K`, `T` are all in it). Leave it unset when timing compiles.
* **The verdict.** The tangent of a step whose capacity verdict failed
  (`overflow_ok(bits)` false) is NaN; `GradientPlan.jvp` / `jacfwd` / `linearize`
  refuse it on the host (`RolloutVerdictError`). Forward mode needs no symmetric
  lists, so `OVERFLOW_GRAD_LISTS` is never set.
* **First order only.** `jax.grad` of a forward-mode program, and a jvp of a
  forward-mode tangent (`jacfwd(jacfwd)`, `jax.hessian`), refuse.
* **Not covered:** the radix lane, the multi-GPU lanes, the mutual and block-step
  lanes, the near field's table and prepacked layouts (they raise), and the
  `dehnen_error` force-scale twins' tangents (f_b is stopped anyway).
* **Gates** (CPU, float64, the S9 G2 system):
  `tests/integration/strict/test_forward_mode.py` (the jvp primal bitwise the
  production step and `strict_run_v2`; `<w, J d> == <J^T w, d>` against the
  reverse path for every input; central differences on identical lists;
  `jacfwd == jacrev`; NaN on overflow; `linearize == jvp`) and
  `tests/unit/runtime/test_forward_seams.py` (each seam's primal bitwise, its
  tangent the reverse kernel's transpose).

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
* `jax.jvp` / forward mode by default (the `custom_vjp` refuses it with a
  `TypeError`): opt in with `ad_mode="forward"` on the large-N lane
  ([Forward mode](#forward-mode-ad_modeforward)).
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
| OOM, or a compile of minutes, jitting a large-N gradient | the state (or the plan) is closed over: `jax.jit` embeds every array of the topology as a constant | pass the state and the plan as arguments (above), or use `differentiable_step_fn` |
| `NotImplementedError: ... needs a SYMMETRIC near list` | the near reverse reads each pair from both ends | the walk's lists are symmetric; a hand-built neighbour list must hold both directions |
| `RuntimeError: compact_far_pairs is None` | large-N state built without the pair list | construct the solver with `retain_far_pairs_for_grad=True` |
| `TypeError: ... prepare_state() got an unexpected keyword argument 'retain_far_pairs_for_grad'` | it is a constructor option | pass it to `FastMultipoleMethod(...)` |
| `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` | forward mode (`jax.jvp`, `jax.jacfwd`, a forward-mode adjoint) through the reverse-mode kernels | trace it under `ad_mode="forward"`: `differentiable_rollout(..., ad_mode="forward")`, `GradientPlan.jvp` / `jacfwd` / `linearize`, `GradConfig(ad_mode="forward")`, or `with ad_mode_override("forward"):` around `jax.jvp(provider.differentiable_step, ...)`; or use reverse mode |
| `NotImplementedError: jaccpot_tangent_...: has no transpose` | `jax.grad` of a forward-mode program | build reverse-mode programs with `ad_mode="reverse"` (the default) |
| all-NaN gradients in fp32 | historical; the squared-radius floors underflowed in fp32 | fixed — ensure you are on this branch |
| `ConcretizationTypeError` under outer `jax.jit` | a radix state: host-side ops in the re-run sweeps at large N | use bare `jax.grad` (a `LargeNPreparedState` compiles whole, as arguments) |
| gradients are exactly zero | you differentiated `evaluate_prepared_state` or the Odisseo forward path | use `differentiable_accelerations` / `differentiable=True` |
| `TracerArrayConversionError` on centres | historical: the grouped M2L reached the grad path | removed in the 2026-10 cleanup (X3); file a bug |
| FD and AD disagree | FD perturbed a *different* function (full rebuild) | FD the same frozen `state` |

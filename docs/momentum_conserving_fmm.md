# Momentum-conserving (mutual) FMM for block-step integration

`jaccpot.mutual` implements the Dehnen (2014) *mutual* restructure of the FMM, and
`jaccpot.nornax_adapter.BlockStepFMM` exposes it in the shape a
block-power-of-two individual-timestep KDK leapfrog consumes. The goal is a force
that a block-step integrator can legally use at a scale where direct summation is
too slow.

## The problem

A block-step KDK splits interactions by level `k = max(rung_i, rung_j)` and
requires each level's contribution to be applied **antisymmetrically**, so that an
inactive coarse partner of an active fine interaction still receives its
equal-and-opposite kick. `sum_i m_i Δv_i == 0` per level is therefore not a
diagnostic — it is the defining correctness property of the scheme (Dehnen 2014;
Farr & Bertschinger 2007).

Jaccpot's production force cannot supply it. The near field is a target-centric
*gather* (each target sums over its neighbours) and the far field is a
one-directional downward sweep, so every pair is evaluated **twice,
independently**. The two evaluations round differently, and the momentum residual
lands at the FMM's force accuracy (~1e-3 … 1e-5) rather than at round-off.
Restricting *sources* to a rung subset was not expressible at all.

## The three mechanisms

### 1. Symmetric topology (`mutual/topology.py`)

A host-side dual-tree traversal visits each unordered node pair once and emits it
canonically (`a < b`), using the **mutual MAC**

```
theta * |c_B - c_A| > R_A + R_B
```

with `c` the node centre of mass and `R` the node radius. This criterion is
symmetric in `A` and `B` by construction, so one acceptance decision serves both
directions — a target-centric MAC (`R_source / d < theta`) is not, and would
accept a pair in one direction only.

Building it on the host is deliberate. The discrete topology is exactly what
`differentiable_accelerations` freezes and severs from the gradient, and what
nornax `stop_gradient`s on its side; making it a host constant means there is no
traced control flow to accidentally differentiate through, and every device
kernel is static-shape.

### 2. Mutual near field (`mutual/nearfield.py`)

Each leaf pair is evaluated **once** and applied twice: `+F` to the target leaf,
`−F` to the source leaf. The antisymmetry is exact at the bit level because IEEE
guarantees `fl(x_j − x_i) == −fl(x_i − x_j)` and the prefactor
`c_ij = G m_i m_j / r_ij³` is symmetric to the last bit. Intra-leaf blocks use the
full block minus the diagonal, which is already exactly antisymmetric.

**Forces are accumulated, not accelerations**; the mass division happens once at
the end. That keeps `sum_i m_i a_i == sum_i F_i`, which is the quantity that
cancels structurally.

### 3. Dual M2L (`mutual/farfield.py`)

For a well-separated pair `(A, B)` a single M2L evaluation adds B's field to A's
local expansion **and** A's to B's. Both forces are gradients of the same
truncated mutual interaction energy `W_AB = <M_A, T(R) M_B>` — `F_A = −∂W/∂x_A`
and `F_B = −∂W/∂x_B = +∂W/∂x_A` — so `F_A + F_B` cancels *algebraically*.

This was checked before building on it, because it is the plan's load-bearing
assumption. Measured pair-level residual `|F_A + F_B| / |F|`:

| order `p` | force error | momentum residual |
|---|---|---|
| 2 | 4.0e-03 | 4.1e-16 |
| 4 | 2.8e-05 | 4.1e-16 |
| 6 | 1.6e-07 | 4.2e-16 |

The force error moves by five orders of magnitude; the momentum residual does
not move at all. Notably the *plain* (square) truncation already cancels — no
symmetrised `n + m <= p` truncation is required, and raising the local order
above the multipole order changes nothing.

The remaining cascades preserve the property because they are exact
re-expansions: M2M loses nothing mapping child degrees `<= n` into a parent
degree `n`, and L2L re-centres a degree-`p` polynomial into another degree-`p`
polynomial.

> This is why a COM correction (`a_i -= Σ_j m_j a_j / Σ_j m_j`) is not a
> substitute. It also zeroes the momentum sum, but by smearing a uniform nonlocal
> offset across every particle rather than delivering each pair's back-reaction to
> the partner that caused it — and it does not survive a per-level decomposition.

## Rung awareness and the fused boundary

Both kernels take a `level_weights` vector and multiply every pair by
`level_weights[max(rung_i, rung_j)]`. One traversal therefore evaluates *any*
linear combination of levels:

* one-hot at `k` → `level_accelerations(level=k)`;
* the boundary's kick weights → the whole sub-step boundary in one pass.

The second form is the **fused-boundary primitive**. A per-level interface costs
one full tree traversal per active level, i.e. `Σ_s (active levels at s)` per base
step — 19 for `k_max = 3`. Weighting instead of masking makes it `n_sub + 1` = 9,
a measured **2.1×** reduction: cost scales with *boundaries*, not
boundaries × levels. For a masked direct sum that distinction is cheap; for an
FMM, where each traversal is the dominant cost, it is the difference between
keeping and losing the individual-timestep advantage.

Because each weight is a single symmetric scalar per pair, it multiplies `+f` and
`−f` identically and cannot break the cancellation — momentum is exact for *any*
weighting.

### Driving the boundaries with a scan

`boundary_kick` takes `active_floor`/`half`, which an integrator naturally has as
*static* Python values — one per boundary. That makes a fused base step unroll:
the traced graph carries `2**k_max` boundary kicks even though the runtime cost is
only `n_sub + 1` traversals. For an FMM each of those is a whole tree traversal's
worth of graph.

`boundary_weight_table(k_max, dt_max)` returns the `(n_sub + 1, k_max + 1)` table
of every boundary's weights, so a caller can index it with a **traced** boundary
index and pass the row through `boundary_kick(..., level_weights=...)`:

```python
table = boundary_weight_table(k_max, dt_max, dtype=positions.dtype)

def body(carry, s):
    pos, vel = carry
    vel = fmm.boundary_kick(pos, vel, masses, rung=rung, level_weights=table[s])
    pos = pos + jnp.where(s < n_sub(k_max), dt_min, 0.0) * vel
    return (pos, vel), None
```

`level_weights_from_floor` also accepts tracers for `active_floor`/`half`/`dt_max`,
so either shape works. Both are bit-identical to the static weights — powers of two
are exact in binary floating point, so scaling by `1 / 2**k` is the same operation
as dividing by `2**k`.

`BlockStepFMM.advance_base_step` uses the table by default
(`scan_boundaries=True`): measured **10.35×** fewer top-level jaxpr equations at
`k_max = 3` than the unrolled form, for the same trajectory to round-off. Pass
`scan_boundaries=False` for the unrolled reference the tests compare against.

> Measure this with `len(jaxpr.jaxpr.eqns)`, not `len(str(jaxpr))`. The printed
> form is dominated by the embedded topology constants, which are identical on both
> paths, so it reads only ~2× and understates the difference five-fold.

### Exact vs. cell-level splitting

| | predicate | exactness |
|---|---|---|
| near field | per-particle `max(rung_i, rung_j)` | exact |
| far field | per-cell `max(rung_A, rung_B)`, cell rung = finest particle it holds | falcON activity-gating (strategy B2) |

Both are genuine **partitions** of the interaction set, so they sum to the total
force and each level conserves momentum exactly. The cell-level split
over-refines: a coarse particle sharing a cell with a fine one is treated at the
fine level. It therefore does *not* reproduce a direct-sum oracle's per-level
decomposition, so cross-checks are made on total force, momentum and energy.
Keeping one multipole set per cell is what avoids the `(k_max + 1)×` multipole
memory an exact per-rung-multipole scheme (strategy B1) would need.

`test_uniform_rung_reproduces_the_oracle_level_split` pins the boundary case: with
no rung-mixed cells the two splits coincide and the oracle is reproduced level by
level.

## Usage

```python
from jaccpot import BlockStepFMM  # or: from jaccpot.nornax_adapter import BlockStepFMM

fmm = BlockStepFMM(softening=1e-2, k_max=3, theta=0.7, max_order=4, leaf_size=32)

# Once per base step: build the frozen topology (host operation, not traceable).
fmm.prepare(positions, masses)

# Production path -- one mutual traversal per sub-step boundary.
velocities = fmm.boundary_kick(
    positions, velocities, masses,
    rung=rung, active_floor=1, dt_max=1e-3, half=1.0,
)

# Or a whole base step, still one traversal per boundary.
positions, velocities, acc = fmm.advance_base_step(
    positions, velocities, masses, rung=rung, dt_max=1e-3
)

# MutualForceModel contract -- what a stock nornax integrator drives.
a_k = fmm.level_accelerations(positions, masses, rung=rung, level=2)
```

`BlockStepFMM` structurally satisfies nornax's `MutualForceModel` protocol without
importing nornax, keeping the dependency graph acyclic (`Jaccpot → Yggdrax`,
`Nornax` standalone, `ODISSEO → Nornax + Jaccpot`). It is part of the frozen
top-level API surface (`jaccpot.__all__`), so ODISSEO can import it directly; the
adapter chain costs ~4 ms of import time.

## Differentiability

Everything is pure JAX at fixed topology, holding the reverse-mode-legality
discipline the differentiable-FMM work established: static (Python) loop bounds
over tree levels, double-`where` guards before every reciprocal so no masked pair
produces a NaN cotangent, and host-concrete permutations. `jax.grad` over a force
evaluation is an exact fixed-topology gradient, and `grad(FMM)` matches
`grad(direct sum)` to the FMM's own force accuracy.

Trajectory-level gradients work by composing fixed-topology evaluations across the
rollout with the tree rebuilt per base step and frozen — the same treatment nornax
gives its rung schedule. `test_rollout_gradient_finite_difference` checks
`d(summary)/d(IC)` through a multi-rung rollout against finite differences.

## Backends

`backend="jax"` (default) runs the pure-JAX kernels everywhere. `backend="pallas"`
routes each stage to whichever implementation *measured* fastest on an Ampere+
GPU, and falls back to pure JAX wherever the hardware cannot run the kernel:

| stage | what `backend="pallas"` dispatches | measured effect |
|---|---|---|
| near field | `pallas/nearfield_mutual.py` — Pallas forward + hand-written analytic reverse | **2.2–3.6× forward, 3.1–4.1× reverse** |
| far field | pure JAX | both Pallas M2L shapes measured *slower* (0.84×/0.61×) |

The far-field routing is a measured decision, not an oversight — see
[Phase 5 outcome](#phase-5-outcome). Both Pallas M2L lanes remain wired,
differentiable and covered by tests; `JACCPOT_MUTUAL_M2L=fused|zcore|jax` forces
one on hardware to reproduce the A/B.

Every Pallas lane goes through a `custom_vjp` wrapper, **not** the bare kernel.
`pallas_call` has no JVP/transpose rule, so a path reaching a bare kernel is
silently non-differentiable — and jaccpot's own `m2l_core_z_real` helper calls the
bare kernel, so `use_pallas=True` there cannot be differentiated. On CPU the
Pallas path is simply unsupported and falls back, which means the defect is
invisible off-GPU.

`pallas_interpret=True` runs the Pallas kernels in interpret mode, which works on
CPU. That is what makes the Pallas tests non-vacuous without a GPU: they execute
the real kernel logic and assert forward parity (< 1e-10), gradient parity against
pure JAX (< 1e-8) and the momentum residual (< 1e-13). Interpret mode is also why
`JACCPOT_MUTUAL_M2L=auto` still selects the *fused* M2L under `interpret` — CI has
no GPU, and routing interpret to pure JAX would make those assertions compare pure
JAX against itself.

> That failure mode is not hypothetical. A parity test on this branch was once
> vacuous for exactly this reason, and during Phase 5 a scratch check reported a
> flattering `0.0e+00` forward error purely because its configuration produced no
> far pairs at all. `test_pallas_near_field_kernel_actually_runs_in_interpret_mode`
> now counts kernel invocations instead of trusting the flag, and the accuracy
> tests assert `num_far_pairs > 0`.

### Phase 5 outcome

All measurements: 1× A100-PCIE-40GB (sm_80), jax 0.9.0.1, float64 unless stated,
θ = 0.7, order 4, leaf 32, best of 3 after a discarded compile call. Reproduce
with

```bash
python -m bench.bench_mutual_backends --sizes 10000 100000 --theta 0.7 --order 4
python -m bench.bench_mutual_farfield_stages --sizes 10000 100000
```

#### 1. Near-field ROI — re-measured, and the PR-2 verdict does **not** transfer

PR-2 descoped its fused near field after measuring ~1.1× at N≈1024 against the
pure-JAX bucketed lane. That kernel was a *gather*. The mutual kernel is a
double-sided scatter that evaluates each pair once, and it wins decisively:

| | N = 10⁴ | N = 10⁵ |
|---|---|---|
| near forward | 2.84 → 1.32 ms (**2.15×**) | 43.18 → 11.85 ms (**3.64×**) |
| near reverse | 12.79 → 3.15 ms (**4.06×**) | 154.60 → 49.62 ms (**3.12×**) |

Forward parity vs pure JAX 3.8e-16 / 6.0e-16, gradient parity 2.4e-16 / 6.4e-16 —
summation reordering only. The win grows with N and is largest on the *reverse*,
which is what the hand-written analytic rule buys: the pure-JAX reverse
re-materialises `(chunk, S, S, 3)` pair tensors per scan step, while the analytic
kernel keeps the `S × S` tile in registers and moves only `O(pairs × S)`.

#### 2. Fused real M2L — implemented, tested, and **not** the default

The plan expected this to be "the cheapest real win", on the reasoning that the
two pure-JAX rotation `vmap`s dominate the remaining far-field time. Measured on
the whole far field:

| lane | fwd (N=10⁴) | rev (N=10⁴) | fwd (N=10⁵) |
|---|---|---|---|
| pure JAX | 17.11 ms | 29.64 ms | 75.16 ms |
| Pallas fused | 20.03 ms (0.85×) | 48.70 ms (0.61×) | 116.33 ms (**0.65×**) |
| Pallas z-core sandwich | 20.28 ms (0.84×) | 34.81 ms (0.85×) | 129.66 ms (**0.58×**) |

Both shapes are *regressions at both sizes*, and at N=10⁴ the forward numbers are
within 1% of each other. They cannot be, if the rotations dominated — so the
premise was wrong. The regression deepens at N=10⁵ (0.65×/0.58×) precisely
because that is where the M2L becomes 80% of the far field, so the kernel's
overhead is no longer diluted by stages it does not touch.

Two independent reasons:

* **Operand traffic.** The fused kernel takes the world↔z rotations as explicit
  `(pairs, Bp, mdp, mdp)` arrays: at order 4 in float64 that is 32 KB per pair
  against the sandwich's 0.39 KB of coefficient vectors, an **82×**
  amplification (5× of it pure power-of-two padding, `5×9×9 → 8×16×16`). The
  sandwich builds the same blocks inside a fused XLA kernel and never spills them
  to HBM. The reverse is worse still (0.61×) because its VJP kernel writes
  `bto_bar`/`bfr_bar` — the same 32 KB per pair back *out*.
* **At N=10⁴, the M2L is not even the far-field hotspot** (20.5% — see the stage
  split below), so the lane swap could not have moved much either way there. That
  is *not* an excuse for the regression, though: at N=10⁵ the M2L is 80.5% of the
  far field, and the Pallas lanes lose by more, not less. The two sizes together
  rule out "it just needs a bigger problem to amortise".

A kernel that took `deltas` and built the rotations *on chip* would avoid the
traffic entirely, but that is a new kernel (Wigner-d recurrences in Triton), not
a wiring change.

So `_m2l_batch` gained the fused lane exactly as specified — it is wired,
differentiable through `m2l_real_fused_pallas_cvjp`, and exercised on every CI run
in interpret mode — but `JACCPOT_MUTUAL_M2L=auto` resolves to pure JAX on
hardware. Set `fused` or `zcore` to force it.

#### 3–4. Mutual P2P kernel and its analytic reverse — landed

`jaccpot/pallas/nearfield_mutual.py`. One program per leaf pair builds the
`S × S` tile **once** and emits `+F` by reducing over `j` and `−F` by reducing the
*same* tile over `i`; the sign is applied on the way out. The level weight
`level_weights[max(rung_i, rung_j)]` is applied inside the kernel before the
reductions, as one symmetric scalar per pair.

The reverse is hand-written, never `jax.vjp` of the twin, for the reason
`_radix_fast_lane_prepacked_accel_bwd` records: a hand-written `bwd` is never
itself differentiated, so its intermediates stay tile-bounded. Its one structural
difference from the existing gather-shaped rule is that each pair's cotangent is
`Fbar_a[i] − Fbar_b[j]`, fed from **both** endpoints, because the forward wrote to
both. `jax.vjp` of the pure-jnp twin is kept as the oracle in
`tests/unit/test_custom_vjp_parity.py`; agreement is ~6e-16 on all four
differentiable inputs, in both block modes.

Defects found and fixed while building it:

* **The near-field scan never clamped its chunk to the pair count**
  (pre-existing, `mutual/nearfield.py`). The chunk is a memory budget; when it
  exceeded the work, the scan padded out to a full chunk of dead slots that the
  kernels still evaluated. Mild on the JAX lane, pathological on the Pallas one,
  whose budget is much larger.
* **Two Pallas-GPU lowering failures that interpret mode cannot see** — see below.

##### Interpret mode validates *logic*, not *lowerability*

This is the operational lesson of Phase 5, and it cuts against how the existing
tests are structured. `interpret=True` runs the kernel's jaxpr under CPU
semantics, so it happily accepts primitives the Triton backend does not implement.
Two kernels passed every CPU interpret test and then failed on the first GPU run:

| written as | lowers to | Pallas GPU |
|---|---|---|
| `xa_ref[0][:, None, 0]` — slice a component out of a packed `(S, 3)` array | `gather` | **unimplemented** |
| `level_weights[k]` — read one scalar from the table with a *static* `k` | `slice` | **unimplemented** |

Both now use whole-array forms: components come straight off their refs
(`xa_ref[0, :, 0]`, a strided load), and the weight tile is a masked reduction
over a broadcast `(K, 1, 1)` table (`_pair_weight_tile`).

The second one is the more instructive failure. It only appears on the
*level-weighted* path, which the throughput benchmark never exercises — the
benchmark runs unweighted. It was caught solely because
`test_mutual_nearfield_pallas_custom_vjp_matches_twin` is parametrized over
`interpret ∈ {True, False}` and always passes level weights, so the
`interpret=False` variants run the real Triton lowering on a GPU. Keep that
parametrization: CI (no GPU) will skip those cases, which means **a GPU run is
the only thing standing between a lowering regression and a broken
`backend="pallas"`**.

#### End to end, and the result that matters more than the speedup

| whole force | `backend="jax"` | `backend="pallas"` | | parity |
|---|---|---|---|---|
| N=10⁴ forward | 19.35 ms | 16.62 ms | 1.16× | 3.5e-16 |
| N=10⁴ reverse | 37.13 ms | 35.58 ms | 1.04× | 8.5e-16 |
| N=10⁵ forward | 96.47 ms | 80.77 ms | **1.19×** | 2.9e-16 |
| N=10⁵ reverse | **out of memory** | **258.94 ms** | — | n/a |

| momentum residual | `backend="jax"` | `backend="pallas"` |
|---|---|---|
| N=10⁴ | 2.446e-17 | 3.039e-17 |
| N=10⁵ | 1.166e-16 | 1.188e-16 |

The forward factor is far below the near field's 3.64× because the near field is
only ~45% of the total at N=10⁵ — Amdahl, not a kernel problem. The N=10⁴ reverse
barely moves (1.04×) for the same reason from the other side: there the far field
is ~70% of the reverse, so a 4× near-field win is nearly invisible.

The reverse row is the real outcome. `backend="jax"` does not get a slower
gradient at N=10⁵; it does not get one **at all**. It requests a single 30.50 GiB
allocation and dies `RESOURCE_EXHAUSTED` on a 40 GB A100, because the pure-JAX
reverse stacks a `(chunk, S, S, 3)` pair tensor per scan step, so its memory grows
as `pairs × S²` — 512 326 leaf pairs × 32² × 3 × 8 B alone is ~12.6 GB before the
far field. The analytic reverse keeps the `S × S` tile in registers and moves only
`O(pairs × S)`, and completes. This is exactly the argument
`_radix_fast_lane_prepacked_accel_bwd`'s docstring makes for hand-writing the
reverse rather than linearizing a twin, now reproduced on the mutual kernel.

Because the baseline OOMs, that row is a **feasibility** result and not a parity
measurement — there is nothing to compare against at N=10⁵. Gradient parity is
established where both lanes fit: 2.4e-16 (near, N=10⁴), 6.4e-16 (near, N=10⁵),
and < 1e-8 end-to-end in the CPU interpret tests.

#### Where the far field actually spends its time

`python -m bench.bench_mutual_farfield_stages --sizes 10000 100000`:

| stage | N = 10⁴ | N = 10⁵ |
|---|---|---|
| upward (P2M + M2M) | 5.85 ms (38.6%) | 9.33 ms (12.1%) |
| M2L (dual) | 3.10 ms (20.5%) | **61.91 ms (80.5%)** |
| L2L | 6.00 ms (39.7%) | 5.37 ms (7.0%) |
| L2P | 0.18 ms (1.2%) | 0.27 ms (0.3%) |
| sum | 15.13 ms | 76.88 ms |


The composition inverts between the two sizes, and both halves matter:

* **At N=10⁴ the M2L is a fifth of the far field** and the two cascades are
  four fifths. L2L costs 6.00 ms at N=10⁴ and *5.37 ms* at N=10⁵ — flat across a
  10× problem, which is the signature of launch-bound work: both cascades are
  Python loops over tree *levels*, emitting a kernel per level whose cost is
  dominated by launch overhead, not by N. No M2L kernel can touch that, which is
  why swapping the M2L lane moved the N=10⁴ far field by only ±3 ms.
* **At N=10⁵ the M2L is 80% of the far field**, and therefore ~44% of the whole
  force. That makes it the single largest remaining target — and it is precisely
  the stage where the current Pallas kernels lose, for the operand-traffic reason
  above. A rewritten kernel that takes `deltas` and builds the Wigner-d rotations
  *on chip* would attack the largest block of remaining time; the existing
  block-operand kernels cannot.

#### 5. float32

`python -m bench.bench_mutual_backends --dtype float32 --lanes near,total --no-grad`:

| float32 | N = 10⁴ | N = 10⁵ |
|---|---|---|
| near forward speedup | 2.05× | 1.90× |
| total forward speedup | 0.99× | **1.71×** |
| near Pallas vs pure JAX (rel-L2) | 2.32e-07 | 3.95e-07 |
| momentum residual, `backend="jax"` | 1.882e-08 | 6.476e-08 |
| momentum residual, `backend="pallas"` | 2.079e-08 | 6.370e-08 |

(These are the numbers *after* the padding fix below. Before it, the whole N=10⁵
float32 column was NaN on both backends.)

This is exactly the predicted behaviour and worth stating precisely: the pair
antisymmetry is **not** what degrades in float32 — the negation `−fl(dr)` is exact
at any precision, and the prefactor is still one symmetric float. What degrades is
only the order in which the two sides are *reduced*. The residual duly lands at
float32 round-off — ~2e-8 at N=10⁴, ~6e-8 at N=10⁵, growing with the reduction
width as expected — which is nine orders coarser than the float64 ~3e-17 and still
~5 orders below the float32 force error (~1e-3).

The Pallas and pure-JAX residuals differ in the last digit and in *either*
direction (2.079e-08 vs 1.882e-08 at N=10⁴; 6.370e-08 vs 6.476e-08 at N=10⁵).
That is noise between two different summation orders, not evidence that either
lane conserves momentum better: both sit at machine epsilon for the reduction
width, which is the claim being made.

**The N=10⁵ float32 column was NaN on _both_ backends** — pure JAX included, so
not something the Pallas kernels introduced. Localising it turned up a real
latent defect, now fixed; the 1.09× total-forward number measured alongside it was
the cost of computing NaNs, not a speedup.

##### The float32 NaN: a padding-slot defect in `_dual_m2l`

Stage-by-stage, the NaNs first appear in the M2L locals — **exactly 25 entries**,
which is `sh_size(4)`, i.e. one single node's expansion — and the L2L cascade then
spreads them to 156 225 and L2P to all 300 000.

`_dual_m2l` pads its directed pair list out to a whole number of chunks with
`src == tgt == 0`. Those slots have `delta == 0`, so `_m2l_batch` floors the
radius to 1e-30 and the z-core evaluates `r**-(p+1)`:

| | order 4 | order 6 |
|---|---|---|
| `r**-(p+1)` at r = 1e-30 | 5.6e151 | 2.0e213 |
| float64 (max 1.8e308) | finite | finite |
| float32 (max 3.4e38) | **inf** | **inf** |

The trailing `contrib * where(live, w, 0)` then computes `inf * 0 = NaN` instead
of dropping the slot — the single-trailing-mask antipattern this codebase warns
about everywhere else. One poisoned node is all it takes; L2L does the rest.

Two independent conditions kept it hidden, which is why every existing test and
the whole N=10⁴ sweep missed it:

* it needs **float32** — float64 only overflows from order 10, where
  `30 × (p + 1)` passes 308;
* it needs the directed pair list **not to divide the chunk**, i.e. > 65536
  directed far pairs. At N=10⁴ the list is 17 438 and fits one chunk exactly; at
  N=10⁵ it is 562 392 and pads by 27 432.

The fix substitutes a unit axis *before* the reciprocal, so no `inf` is ever
formed; the zero weight still discards the slot, and float64 results are
bit-unchanged. `test_far_field_chunk_padding_cannot_poison_the_expansion` pins it
by shrinking `_M2L_BATCH_BUDGET` to force padding at test scale — verified to fail
(all-NaN accelerations, orders 4 and 6) with the guard removed.

With the guard in place every stage is finite at N=10⁵ in both dtypes. float64
remains the supported configuration; no test asserts a float64 tolerance on a
float32 path.

#### Status against the Phase 5 exit criteria

| criterion | outcome |
|---|---|
| `backend="pallas"` faster than `backend="jax"` on GPU at N ≥ 10⁵ | **yes** — 1.19× forward float64, 1.71× forward float32; and the reverse runs at all, where pure JAX OOMs |
| forward parity ~1e-6 (fp32) / ~1e-10 (fp64) | **yes** — 1.7e-07 fp32, 2.9e-16 fp64 |
| gradient parity vs pure-JAX autodiff | **yes** — 8.5e-16 end-to-end at N=10⁴, ~6e-16 at the kernel level, < 1e-8 in the CPU interpret tests |
| momentum residual characterised per dtype, far below force error | **yes** — float64 ~3e-17, float32 ~2e-8 (N=10⁴) / ~6e-8 (N=10⁵), against a force error of ~1e-3 |
| full mutual suite green on GPU | **yes** — 101 passed |
| CPU interpret tests green (the CI gate) | **yes** |

#### What is still open

* **The M2L at scale.** It is 80% of the far field at N=10⁵ and ~44% of the whole
  force, and no lane in the tree currently beats pure JAX on it. A kernel taking
  `deltas` and building the Wigner-d rotations on chip is the one change that
  would attack it; the block-operand kernels structurally cannot.
* **The cascades at small N.** `upward` + `L2L` are 78% of the far field at N=10⁴
  and flat in N — launch-bound, one kernel per tree level. A batched-across-levels
  formulation, not a kernel, is what that needs.
* **N=10⁶ was not measured.** The ROI question was settled at 10⁴/10⁵ and the
  trend is monotone in N for the near field, but the largest size named in the
  brief is unmeasured here.
* **float32 above N=10⁵** is unvalidated; the padding defect is fixed but no
  accuracy budget has been established for it.

## Scaling

The mutual restructure does not change the interaction-list asymptotics. Pairs per
particle, mutual traversal vs. jaccpot's own production traversal on the same tree
(θ = 0.7, leaf 32, CPU):

| N | leaves | mutual/N | production/N | momentum residual |
|---|---|---|---|---|
| 4 096 | 128 | 1.92 | 1.88 | 3.2e-17 |
| 16 384 | 512 | 4.67 | 4.30 | 1.1e-17 |
| 65 536 | 2 048 | 7.28 | 6.39 | 8.0e-18 |

The two traversals track each other within ~7–14% at every size, so whatever
scaling jaccpot's radix tree and MAC already deliver is preserved. The momentum
residual is flat in N. Reproduce with:

```bash
python -m bench.bench_mutual_fmm --sizes 4096 16384 65536 --theta 0.7 --order 4
```

## Tests

* `tests/integration/test_mutual_fmm.py` — topology coverage, near/far accuracy,
  momentum independence from θ and order, level partition, fused boundary,
  gradients (FD-vs-AD and vs. the direct-sum gradient), rollout momentum/energy,
  backend parity. `test_momentum_is_exact_independently_of_theta_and_order` is
  parametrized over **both** backends, so the Pallas P2P kernel is held to the
  same 1e-13 residual as pure JAX across the whole θ × order sweep — that test is
  the one that would catch a kernel recomputing `dr` instead of negating it.
  `test_pallas_near_field_kernel_actually_runs_in_interpret_mode` counts kernel
  invocations so the parity tests cannot quietly become vacuous.
* `tests/unit/test_custom_vjp_parity.py` — the mutual P2P `custom_vjp` against
  `jax.vjp` of its pure-jnp twin, in both block modes (cross-leaf and intra-leaf),
  plus a direct bitwise-antisymmetry assertion on the kernel's two outputs. The
  analytic reverse is hand-written, so this oracle comparison is the only thing
  standing between a sign slip in it and a silently wrong gradient.
* `tests/integration/test_mutual_fmm_nornax.py` — cross-repo. Protocol
  conformance, schedule identity against nornax's own, agreement with
  `MutualDirectSumGravity`, and nornax's real `leapfrog_kdk_rollout` /
  `block_kdk_rollout` driven by the FMM force. Skipped when nornax cannot be
  imported.

### Running the cross-repo tests

nornax is a test-only dependency; the library never imports it. `tests/conftest.py`
puts a sibling `nornax` checkout on `sys.path` (searching upward through
`REPO_ROOT`'s ancestors, so a git worktree at `<repo>/.claude/worktrees/<name>`
finds it too), and the module skips when it is absent — so no `PYTHONPATH` and no
version overlay are needed.

An earlier version of this doc carried a `pip install --target` overlay recipe,
because nornax could not be imported alongside `equinox < 0.13` / `diffrax < 0.7.2`
(`nornax.terms.NBodyTerm` was a non-frozen dataclass inheriting a frozen
`AbstractTerm`). That is fixed upstream; the tests now run in a stock environment.
The skip guard still catches `Exception` rather than `ImportError`, so a future
import-time incompatibility reports as *skipped* rather than *errored*.

## Resolved: the shared real-basis L2P azimuth gradient

Bringing this path up found a defect in the *production* real-basis L2P.
`evaluate_local_real` built the azimuth as
`cos_phi = where(rho2 > floor, x / rho, 1.0)`. The constant branch is harmless in
the forward pass — every `|m| >= 1` term carries a `sin^|m| θ = (rho/r)^|m|`
factor that annihilates the arbitrary azimuth — but a constant has no x/y
derivative, so under `jax.grad` the entire transverse gradient of the `m != 0`
terms was dropped.

The mutual path surfaced it as a two-body far-field force pointing along z only:
a leaf holding a single particle hits `delta == 0` every time, because the
particle *is* its leaf's centre of mass. It was worked around here by nudging
degenerate offsets off the centre.

Fixed at the source instead, and the fix is broader than the workaround was: the
operator now divides by the **floored** `rho` unconditionally, so the degree-1
limit falls out of the algebra (`sin θ cos φ = (rho/r)(x/rho) = x/r`) rather than
being branched away. That also covers `rho == 0` with `z != 0` — anywhere on the
expansion centre's z axis, reachable at *any* leaf size in axis-aligned or
symmetric configurations — which the offset nudge never triggered on, since such a
point has `r2 >> floor`. `p2m_real_direct` had the same branch and the same fix.

The workaround is gone from `mutual/farfield.py`; no guard is needed at the L2P
call site.

**Transferable lesson.** A `where(cond, expr, constant)` guard placed to keep a
*forward* value finite silently zeroes the corresponding reverse-mode component,
and no forward-only test can see it. Prefer making the degeneracy cancel
analytically (divide by a floored quantity) over branching it away; where a branch
is unavoidable, use the double-`where` form so both branches stay
differentiable — the same discipline the rest of this module follows before every
reciprocal.

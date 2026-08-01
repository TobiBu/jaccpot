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

`backend="jax"` (default) runs the pure-JAX kernels. `backend="pallas"` routes the
real-basis z-axis M2L translation — the far-field hotspot — through jaccpot's
Pallas kernel where the hardware supports it, and falls back to pure JAX
otherwise.

Crucially it goes through `m2l_core_z_real_pallas_cvjp`, **not** the bare kernel.
`pallas_call` has no JVP/transpose rule, so a path reaching the bare kernel is
silently non-differentiable — and jaccpot's own `m2l_core_z_real` helper calls the
bare kernel, so `use_pallas=True` there cannot be differentiated. On CPU the
Pallas path is simply unsupported and falls back, which means the defect is
invisible off-GPU. `mutual/farfield.py::_m2l_batch` therefore reimplements the
rotate → translate → rotate-back sandwich around the `custom_vjp` wrapper.

`pallas_interpret=True` runs the Pallas kernel in interpret mode, which works on
CPU. That is what makes the Pallas tests non-vacuous without a GPU: they execute
the real kernel logic and assert both forward parity (< 1e-10) and gradient parity
against pure JAX (< 1e-8).

**Not yet done — see "Completing Phase 5" below**: a Pallas port of the mutual
near-field P2P, use of the *fused* real M2L (rotate+translate+rotate in one
kernel) rather than only the z-core, and a hand-written analytic reverse for the
mutual kernels. The pure-JAX kernels are the correctness and AD oracle each such
kernel is checked against.

### Completing Phase 5

Ordered by value, and grounded in what the repo already learned in
`docs/differentiable_fmm_pallas_vjp_plan.md` (PR-2):

1. **Re-run the ROI check for the mutual near field.** PR-2 *descoped* the fused
   Pallas near-field after measuring only ~1.1× at N≈1024 and slower at moderate
   N against the pure-JAX bucketed lane. That verdict does not transfer
   unexamined: the mutual kernel evaluates each pair **once** instead of twice, so
   its arithmetic intensity and its bandwidth profile are different. Measure
   before building — on an Ampere+ GPU, selecting it with `autocvd` before
   `import jax` per org policy.
2. **Use the fused real M2L, not just the z-core.** `pallas/m2l_real_fused.py`
   already fuses rotate+translate+rotate and already ships
   `m2l_real_fused_pallas_cvjp`. Swapping `_m2l_batch`'s three-stage sandwich for
   that single call removes the two pure-JAX rotation `vmap`s, which dominate the
   remaining far-field time. This is the cheapest real win and needs no new
   kernel.
3. **Mutual near-field Pallas kernel.** No existing kernel fits: every near-field
   Pallas lane in the tree is a *gather*, and the mutual kernel needs a
   double-sided scatter (`+F` to the target leaf, `−F` to the source leaf).
   **The correctness trap**: the exactness of the momentum cancellation depends on
   the kernel computing `dr` once and *negating* it. A port that recomputes `dr`
   for the source-leaf pass reintroduces the residual at the force accuracy —
   which is exactly why the momentum tests assert `< 1e-13` rather than `< 1e-3`.
4. **Analytic reverse.** `near_field.py::_leafpair_accel_analytic_vjp` is the
   template, and its docstring records why a hand-written reverse rather than
   `jax.vjp` of the twin: linearizing the twin reinstates O(edges × W) scan
   residuals (~67 GB at N=2²⁰), while a hand-written `bwd` is never
   differentiated, so its intermediates stay tile-bounded and memory is O(N). The
   mutual version must accumulate the cotangent from **both** endpoints of each
   pair, which the existing gather-shaped rule does not do.
5. **Characterise fp32.** The Pallas near-field lanes run fp32 (measured rel-L2
   ~4e-6 against the fallback, i.e. summation reordering). Bit-level antisymmetry
   of the pair force survives fp32 — the negation is exact — so what degrades is
   only the *reduction* order. Expect a momentum residual around fp32 round-off
   rather than the fp64 ~1e-17, still far below the force error. Worth measuring
   and stating, not assuming.

Steps 1–3 need a GPU for throughput; all of them can be developed and checked for
*correctness* on CPU in interpret mode, as the current Pallas tests do.

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
  backend parity.
* `tests/integration/test_mutual_fmm_nornax.py` — cross-repo. Protocol
  conformance, schedule identity against nornax's own, agreement with
  `MutualDirectSumGravity`, and nornax's real `leapfrog_kdk_rollout` /
  `block_kdk_rollout` driven by the FMM force. Skipped when nornax cannot be
  imported.

### Running the cross-repo tests

nornax is a test-only dependency and is not installed in jaccpot's environment.
It also fails to *import* against `equinox < 0.13` / `diffrax < 0.7.2`
(`nornax.terms.NBodyTerm` is a non-frozen dataclass inheriting a frozen
`AbstractTerm`), so the module skips rather than errors on a version mismatch.
To run them without disturbing the environment, overlay the newer versions:

```bash
pip install --target /tmp/overlay --no-deps "diffrax==0.7.2" "equinox==0.13.6"
PYTHONPATH=/tmp/overlay:/path/to/nornax pytest tests/integration/test_mutual_fmm_nornax.py
```

## Known issue in the shared real-basis L2P

`evaluate_local_real_with_grad(coeffs, delta)` loses the x and y components of its
gradient when `delta` is **exactly** zero: `evaluate_local_real` builds the azimuth
from `x / rho` with `rho` clamped by `squared_radius_floor`, which keeps the
potential finite but collapses the azimuth. A leaf holding a single particle hits
this every time, because the particle *is* the leaf's centre of mass — it produced
a force with only a z-component in a two-body test.

This is a pre-existing defect in `jaccpot/operators/real_harmonics.py` affecting
the production L2P too, not only the mutual path. It is worked around here in
`mutual/farfield.py::_nondegenerate_offsets`, which nudges exactly-degenerate
offsets just above the floor (~1e-27 in float64, ~1e-16 in float32) where the
gradient is recovered to full precision. Fixing the operator itself is a separate,
wider-blast-radius change.

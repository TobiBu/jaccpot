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

`BlockStepFMM.advance_base_step` can use the table (`scan_boundaries=True`):
measured **10.35×** fewer top-level jaxpr equations at `k_max = 3` than the
unrolled form, for the same trajectory to round-off.

It is **off by default**, because that win costs peak memory rather than saving
it. Jaccpot's inner kernels are individually jitted, so the unrolled Python loop
reuses their cached executables, while the scan must inline the whole force into
one program and compile that. Over a 6 base-step rollout at `k_max = 2`, float64:

| | N = 512 | N = 256 |
|---|---|---|
| scan | 2.67 GB | 2.70 GB |
| unroll | 2.08 GB | 1.92 GB |

`N` barely moves either number — this is compile/executable memory, not data, so
shrinking the problem does not help. Turning the scan on by default regressed the
CI integration shard from passing to OOM-killing its workers.

Use the scan when *trace size* is what binds — an outer `jax.jit` over a rollout,
or a deep `k_max` where `2**k_max` unrolled kicks stop fitting. An integrator that
wants small traces *and* one traversal per boundary should drive `boundary_kick`
with rows of the table from its own scan, which is the seam nornax uses.

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

### Rungs across devices

`mutual/distributed.py` takes the same `rung` / `level_weights` pair, and
`jaccpot.DistributedBlockStepFMM` exposes them in the same shape nornax consumes.
The two halves of a cross-domain pair need different machinery, and the difference
is where the interesting part is.

**The near half is exact and per-particle, so the remote endpoint's rung has to
travel with the remote endpoint.** It rides round B of the demand-driven halo import
(`import_near_halo(payload_sorted=...)`, yggdrax), in the same buffer as the
positions and masses, so it is sized by the halo rather than by the system. It could
*not* go on the coarse frontier: that is `all_gather`-ed, so publishing `leaf_size`
rung columns there would ship every remote particle's rung to every device —
`O(N_total)`, which is exactly what the demand-driven import exists to avoid. The
weight then multiplies the one tensor both sides of the tile read, so a weighted
cross block is antisymmetric for the same reason an unweighted one is.

**The far half is cell-level, so the frontier is the right channel**, one scalar per
leaf appended to the multipole row it belongs to. The local endpoint may be an
internal node — `accept_only_leaf_pairs` constrains the *remote* side, which has to
be addressable — so node rungs are needed for the whole tree. `mutual/force.py`
gets them by propagating a leaf maximum up the level schedule; the distributed lane
has only the inclusive node ranges, so it takes the range maximum instead, as one
prefix count per level (`k_max` is small). The two agree by construction and
`tests/unit/mutual/test_distributed_node_rungs.py` checks that they do.

The weight on the far half is applied **once, on the evaluating device, to both
directions of the batched M2L, before either leaves**. That is a real decision and
not the obvious one, because a far pair's `−f` is not a force on a particle: it is a
local expansion returned to the owner's node. Applying the weight again on import
would square it; applying it on neither side would leave the far half unweighted
while the near half is weighted. Weighting the expansion is legal because everything
downstream of one is linear in its coefficients — L2L re-centres, L2P evaluates — so
scaling the coefficients scales the force by the same factor.

#### Momentum is blind to all of it

This is the part worth carrying away. Measured by injecting each fault into the cross
far half and reading three criteria off a 2-device run at N = 256, leaf 4,
θ = cross_θ = 0.5 (33 cross far pairs, 991 cross near pairs):

| fault | level partition | weight linearity | momentum |
|---|---|---|---|
| none | 2.1e-16 | 2.9e-16 | < 4e-17 |
| far half left **unweighted** | 3.9e-02 | 2.0e-02 | < 4e-17 |
| far weight applied **twice** | 2.1e-16 | 9.4e-03 | < 4e-17 |
| far ignores the **remote** rung | 2.1e-16 | 2.9e-16 | < 4e-17 |

A per-level momentum residual at 1e-17 is therefore *not* evidence that the levels
are right. Any single symmetric scalar per pair conserves momentum exactly, whatever
that scalar is — so three distinct criteria are needed:

* **the level partition** (one-hot weights must sum to the unweighted total) catches
  a half left unweighted, and *cannot* catch squaring, because `w² == w` for 0 and 1;
* **linearity** (`a(u) + a(1 − u) == a(1)` at fractional weights) catches squaring;
* **a two-clump run with one rung per device**, where every cross pair provably
  belongs to one level and clump A's level-0 force is an exact direct sum, catches an
  assignment that is wrong yet consistent. Nothing else does.

### Distributed backend

`DistributedMutualConfig.backend` routes the intra-domain near field to the mutual
Pallas kernel on sm_80+, mirroring the single-device lane — including its two
measured decisions: the far field stays pure JAX (both Pallas M2L shapes are slower,
see [Phase 5 outcome](#phase-5-outcome)), and every Pallas lane is reached through its
`custom_vjp` wrapper. The cross-domain near field is pure JAX either way; it is a
different kernel (`_tile_forces`, a local-leaf × halo-block tile) and giving it a
Pallas lane is a new kernel, not a wiring change.

### Compile once, not per force

`distributed_mutual_fmm` rebuilds its mapped program on every call — `shard_map`
wraps a fresh closure, so `jax.jit` sees a fresh cache key — which is fine for one
force and ruinous for the `n_sub + 1` evaluations a base step asks for.
`make_distributed_mutual_evaluator` splits the host side out: the partition, the
padding layout, the bounds and the capacities are frozen once and the compiled
program is held. Measured on 2 forced CPU devices at N = 128: build < 0.1 s, first
evaluation 20.5 s, each later one ~8 s.

Note what is *not* frozen. The per-device tree is rebuilt inside the mapped program
from the positions handed in, so successive boundaries within a base step see trees
built from their own positions — a *finer* rebuild cadence than the single-device
lane's, not a coarser one. Every evaluation stays internally self-consistent, so its
levels partition its own pairs and each level's momentum cancels exactly; what is
lost is only the bit-level identity of the topology across a base step. So
`prepare()` is called when the *partition* should change, not on a fixed cadence.

The readout is a `jnp` scatter rather than a NumPy assignment, and the
"every real particle appears exactly once" check moved to the partition, where it
belongs — it depends only on the frozen gid layout. That is what makes a whole
evaluation traceable, which is what lets nornax's `block_kdk_rollout` (a `lax.scan`
over base steps) drive this lane at all. The overflow read is *attempted* rather
than gated on `isinstance(..., Tracer)`, so it raises on every eager call and is
skipped under trace — a traced driver gets no overflow check and must evaluate once
eagerly first.

## Topology backends: `host` and `device`

`BlockStepFMM(topology_backend=...)` decides where the topology is *built*, and it
is the difference between this lane scaling and not. **The default is `"host"`**,
and the Usage block below shows the device path because a production run wants it.

`"host"` builds the topology with the NumPy dual-tree traversal in
`mutual/topology.py`: six device-to-host transfers, a scalar Python loop over every
node for the centre/radius pass, a host BFS for the depths, and a NumPy wavefront
walk. It is correct and **untraceable**, so a block-step run pays a host round trip
per base step — measured **22 s at N = 20 000**, against 0.5 s for the whole rest of
the base step once the force is jitted. At that point the topology build *is* the
run.

`"device"` builds the same thing in JAX with static shapes throughout
(`mutual/device_topology.py`), so `rebuild_state` can live inside a `jax.jit` or a
`lax.scan` and a whole rollout becomes one program. It implies `static_shapes`,
because every device output shape comes from the capacities. The traversal is
`yggdrax.interactions.dual_tree_walk_mutual`, verified to reproduce the host walk
pair-for-pair, and the node radii are the exact `max_i |x_i - c_n|` the host
computes rather than the looser bounding-sphere merge a bottom-up cascade would
give — a looser radius changes MAC outcomes, which changes the accepted pair set,
which changes the force.

### What `freeze_template` freezes, and what stays live

This is the distinction to get right, because "frozen topology" invites the wrong
reading. `freeze_template` is host-side and runs **once**; it is idempotent, and a
second call is deliberately a no-op, since re-freezing would change the capacities
out from under an already-compiled program.

**Frozen, once:**

* the **static-radix linkage** — parent/child links and leaf bucket boundaries.
  For a static-radix tree those buckets are `arange(0, N, leaf_size)`, so they do
  not depend on the particle distribution at all.
* the **capacity profile** (`MutualCapacities`), which is what makes every shape a
  compile-time constant and lets one program serve the run.

**Live, on every `rebuild_state`:**

* the **Morton re-sort** of the particles;
* the **centres of mass and radii**, recomputed from the current positions;
* the **MAC decisions** — which pairs are far is re-decided every call. Only the
  *number* of such pairs is bounded, by the capacities. Measured: far pairs 404 →
  542 on a displaced system with the jit cache still at one entry.

So the tree's *shape* is frozen and its *content* is not. A rollout stays physically
honest across a base step; what it gives up is the freedom to re-refine the tree.

**Cost, and the preset that removes it.** `freeze_template` is dominated by the
capacity trial: the wavefront `queue` cannot be derived from a finished topology,
only found by probing. Measured **31–65 s across a leaf/N sweep**, almost all of it
climbing `2^14 → 2^20/2^21/2^22`, i.e. 6–8 full device topology builds spent
discovering a capacity the leaf count already predicts. Seeding the ladder from the
leaf count brought N = 10⁶ at leaf 64 from **51.05 s to 16.77 s**, and a recorded
`mutual/cap_presets` profile removes even the remaining 2–3 probes. Pass a recorded
profile as `caps=` to make the *first* compile reusable too.

**The overflow contract.** On the device backend an overflowed capacity drops
interactions while leaving momentum exactly conserved — so momentum cannot detect
it. `prepare()` is the eager entry point and raises there. **A driver stepping
through `rebuild_state` under trace gets no such check and must do it itself.**

## Usage

```python
from jaccpot import BlockStepFMM  # or: from jaccpot.nornax_adapter import BlockStepFMM

# leaf_size=64 is the measured optimum for this lane (a U-curve; see the note on
# `BlockStepFMM`). topology_backend="device" is what a production run wants --
# the default is "host", which rebuilds an untraceable host dual-tree traversal
# every base step. See "Topology backends" above.
fmm = BlockStepFMM(
    softening=1e-2, k_max=3, theta=0.7, max_order=4, leaf_size=64,
    topology_backend="device",          # implies static_shapes=True
)

# Once per base step: build the topology. On the device backend the first call
# also freezes the static-radix template and the capacities (host-side, once);
# every later build is traceable and re-decides the MAC from live positions.
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

**Retired 2026-08-22 (`6b7cc1b`) — the operand-traffic diagnosis is right, the
prescription is not.** Building the rotations on chip saves *arithmetic*, and the
arithmetic is not what costs. Profiling the stage: time is ~linear in the
coefficient count (exponent 0.91 over orders 2–6) while the arithmetic is
quadratic, at **0.24 % of fp64 peak and 0.2 % of HBM** — neither roofline binds.
The compiled module says why: **~121 HLO ops per coefficient, exactly linear**,
fusions 53 → 229 over orders 2 → 6. Over that range fusions grow 4.32×, measured
time 4.68×, flops 29.6× — **time tracks the kernel count, not the flops**. So the
cost is rotation *construction* (p+1 unrolled degree blocks, twice, ~99 % of the
stage), and the way to attack it is to emit fewer kernels, not to do the same
arithmetic somewhere cheaper. See the replacement below.

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
  above. ~~A rewritten kernel that takes `deltas` and builds the Wigner-d rotations
  *on chip* would attack the largest block of remaining time; the existing
  block-operand kernels cannot.~~ **What actually attacked it was a batched
  rotation, not a kernel** — see [The M2L, as measured](#the-m2l-as-measured).

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

#### The M2L, as measured

_2026-08-22, `6b7cc1b`. This section replaces the on-chip-Wigner-d recommendation
that stood above it._

**What was built.** `_rotate_degree_batched` assembles the rotation block from its
factors instead of calling the per-degree builder p+1 times, which is possible
because the block is `B_U @ Dz(-ax) @ B_U @ Dz(az)`: `B_U` depends only on the
degree and is a compile-time constant, and `Dz` is block-diagonal in |m| and can be
built twice at full width instead of 2(p+1) times. Centred padding is load-bearing —
within a degree the layout is `m = -ell..ell` at index `ell+m`, so m=0 sits at the
block centre and the padded blocks become degree-independent. What remains is one
batched contraction over a p+1 <= 7 axis.

**Measured**, GPU, flag off -> on:

| config | off | on | speedup |
|---|---|---|---|
| order 4, N=2×10⁵ | 96.67 ms | 59.60 ms | **1.62×** |
| order 4, N=10⁶ | 547.72 ms | 334.52 ms | **1.64×** |
| order 6, N=10⁶ | 908.57 ms | 667.96 ms | 1.36× |

Fusions 108 -> 42 at order 4. Parity is exact at every order, both sides, forward
(<= 1.5e-16) and under `jax.grad` (<= 4.2e-17). It does **2.45× more arithmetic**
(dense padding) and still finishes 1.64× sooner, which is what un-binding a
launch-bound stage looks like: 23.43 -> 38.37 GFLOP/s, 3.47 -> 5.68 GB/s, both
still far from any roofline.

**It is off by default** (`JACCPOT_M2L_DEGREE_BATCHED=1`), because the order-6
falloff suggests order-dependent selection rather than a global flip, and that
decision was left separate. The flag is read at call time, so it can be set after
`import jaccpot` — but note it is *not* A/B-able within one process on the lanes
that read it at import.

**Two claims this retires, and one it does not reach.**

* The on-chip-Wigner-d premise is wrong about the term, as set out above: the
  arithmetic it saves runs at 0.24 % of peak.
* "M2L is gather-latency bound" is also refuted. Each source multipole *is*
  gathered 141.7 times, but as-built 97.09 ms, sorted-by-source 95.67 ms and
  deliberately **shuffled** 96.41 ms are all within 1.5 % and numerically
  identical. Shuffling the pair list costs nothing, so locality is not the lever.
* **The Pallas M2L rejection above is now stale.** Both shapes were measured at
  0.85× against a JAX baseline this makes 1.64× faster. That does not make them
  wins — it means the comparison would have to be redone before either shape could
  be argued for again.

**Where this reaches, and where it does not.** The flag lives in
`operators/m2l_real_rot_scale`, i.e. the **real** basis. The mutual lane's dual M2L
routes its rotations through exactly those helpers (`_rotate_multipole_to_z_single`
and `_rotate_local_from_z_single` in `mutual/farfield.py`), so it inherits the
change; the mutual stage split above has not been re-measured with the flag on. The
large-N single-GPU lane runs solidfmm and **does not reach it at all** — measured
end to end at N=10⁶, order 8, the flag moves nothing (full 637.08 ms off against
645.77 ms on, inside a 2.4–5.8 % spread).

**And the end-to-end context, which is the part worth carrying away.** On the
single-GPU large-N lane at N=10⁶, the far field is **under 1 % of the evaluation at
order 4** — 0.7–1.2 % across leaf 256/1024, against a near field at 97.5–99.5 %.
Three separate M2L projects have now been aimed at that 1 %. Before a fourth,
note the one place the premise breaks: at **order 8 the far field is 10.8 %**, not
under 1 %, and that is also where issue #248's accuracy gap lives. So "the far
field is negligible" is an order-4 statement, and the honest version of it is that
single-card performance lives in the near field at production order, not that the
M2L can never matter.

#### What is still open

* **The M2L at scale, on the mutual lane.** It is 80% of the far field at N=10⁵ and
  ~44% of the whole force there, and pure JAX is still the fastest lane for it. The
  batched rotation above is 1.64× on the stage and reaches this lane through the
  shared rotate helpers, but it is off by default and the mutual stage split has not
  been re-measured with it on. That measurement, not a new kernel, is the next step.
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

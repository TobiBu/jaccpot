# The fused force provider — user guide

`jaccpot.FusedForceProvider` is the fused single-GPU lane's per-step force as an object:
the two halves the `strict_run_v2` scan runs at every step, callable inside your own
jitted program. Since S2 (2026-10) `strict_run_v2` itself runs on it: every refresh,
evaluation, capacity verdict, compile key and segment re-plan of the fused scan goes
through the provider. A velocity Verlet of five lines over `prov.step` reproduces
`strict_run_v2(carry="particles")` (see [Quick start](#quick-start-velocity-verlet-in-your-own-scan)).

This is the reference page. What was measured while it was built, and why the API is
what it is, is in [`production_2026-10.md`](production_2026-10.md) (section S2). Code:
`jaccpot/runtime/fused_provider.py` (the provider) and
`jaccpot/runtime/fused_topology.py` (what it hands on: `FusedTopology`, `ForceParams`,
`StepParams`, the overflow bits).

---

## What it computes

One force at new positions, in two halves:

| half | what runs | traced? |
|---|---|---|
| `rebuild(positions, masses)` | the static-radix tree rebuilt from a shape template, P2M and M2M, the flat walk with fresh far and near lists, M2L and L2L | yes |
| `accelerations(topology, params)` | the near field (P2P), L2P and the scatter back to the input order | yes |
| `potential(topology, params)` | the potential per particle, `-G sum_{j != i} m_j psi(r_ij)`: the CSR direct kernel's fourth lane, the far field's L2P value, the scatter (S3 session 2) | yes |

`accelerations` takes no positions or masses: the rebuild already ran the far field at
the positions it was given. `step(positions, masses, params, aux)` is the two in a row,
with the capacity verdict. `potential` is O(N) and compiles a program of its own
(`evaluate_large_n_potential`; no acceleration program changes): the runner's energy
diagnostic. On an A100 (preset galaxy, clipped Plummer) its rel-L2 against an fp64
direct sum is 1.0e-5 at 2e5 (the force's 8.0e-4), and a rebuild plus the potential
costs 1.1x a rebuild plus the accelerations (25M: 247 against 220 ms; 1e8: 960 against
851 ms, on a shared card) with fewer temporaries (123-133 against 133-143 B/particle).
Traced parameters reach its near field on the fused Pallas lane; the generic near field
(`use_pallas=False`) takes Python floats.

The provider holds no device buffer. It holds the engine, the static configuration
(leaf size, order, opening angle, runtime overrides) and the **shape template** of the
lane's prepared state (`jaccpot/runtime/strict_carry.py`). A rebuild reads only the
template's shapes: no leaf value of a carried state reaches a step's outputs (the
dead-code test in `tests/integration/strict/test_strict_run_v2.py`). So a caller carries
particles and a template, never a stale far list.

**Requirements.** The `large_n_gpu` engine profile on a GPU (the lane needs a
`LargeNPreparedState`; on a CPU the profile declines the lane, and the CPU tests reach it
by making `jax.default_backend` report `"gpu"`, as
`tests/characterization/test_lane_goldens.py` does), a static-radix tree
(`TreeConfig(mode="static_radix")`) and, for a rebuild from the template, **cell
leaves** (`TreeConfig(leaf_partition="cells", leaf_capacity=...)`). `preset="galaxy"`
is all of that in one name (below). On bucket leaves
the rebuild keeps the eager tree's node ranges and children, values a shape template
does not hold, so `rebuild`, `step` and `strict_run_v2(carry="particles")` refuse them
(`require_cell_leaves`, `ValueError`). If `JACCPOT_STATIC_STRICT_FUSED_PROFILE_SET` is
set, N must be in it.

## Quick start: velocity Verlet in your own scan

`FastMultipoleMethod(preset="galaxy", softening=...)` is the production configuration
(S3, `jaccpot/production.py`): `large_n_gpu` on cell leaves of 64 (`cell_min_level` 8)
in a cubic Morton box (`cubic_bounds=True`, S3b; see `docs/production.md`, "The Morton
box"), p6, the geometric Dehnen MAC at theta 0.8, ferrers3, fp32, and
`leaf_capacity="auto"` -- sized at the first eager prepare from the live cell leaves of
the particles it is given (x1.15, in steps of 1024; on device,
`capacity_plan.live_cell_leaves`), then fixed. It compiles the same programs as the
explicit configuration below with that capacity
(`bench/provider_hlo_identity.py compare TAG TAG --pair galaxy_explicit:galaxy_preset`).
Leaf size and order are call arguments: `GALAXY_SETTINGS.leaf_size` / `.max_order`
from `jaccpot.production`. Spelled out:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaccpot import FastMultipoleMethod, FusedForceProvider
from jaccpot.config import FMMAdvancedConfig, NearFieldConfig, TreeConfig
from jaccpot.runtime.fused_topology import overflow_ok

solver = FastMultipoleMethod(
    preset="large_n_gpu",
    theta=0.8,
    softening=1.5e-3,
    working_dtype=jnp.float32,
    advanced=FMMAdvancedConfig(
        tree=TreeConfig(
            mode="static_radix",
            leaf_target=64,
            leaf_partition="cells",
            leaf_capacity=leaf_capacity,  # or "auto"; above the live count at every step
            cell_min_level=8,
            cubic_bounds=True,
        ),
        nearfield=NearFieldConfig(mode="auto"),
        mac_type="dehnen",
    ),
    fixed_order=6,
)
# eager: prepare at the start positions and freeze the template
prov, _ = FusedForceProvider.prepare(solver, x0, m, leaf_size=64, max_order=6)

dt = jnp.asarray(1e-3, x0.dtype)

def verlet(carry, m):
    x, v, a, aux = carry
    x = x + v * dt + 0.5 * a * dt**2
    a_new, aux = prov.step(x, m, aux=aux)
    return (x, v + 0.5 * (a + a_new) * dt, a_new, aux), None

@jax.jit
def run(carry, m):
    return jax.lax.scan(lambda c, _: verlet(c, m), carry, None, length=100)[0]

a0, aux0 = jax.jit(prov.step)(x0, m)          # the start force
x, v, a, (f_b, bits, needs) = run((x0, v0, a0, aux0), m)
assert bool(overflow_ok(bits))                # every step's force was right
```

This is `strict_run_v2(carry="particles")` bit for bit:
`tests/integration/strict/test_strict_runners_on_provider.py`
(`test_a_user_written_scan_over_the_provider_is_the_particle_carry`) compares the two
at N = 512 on CPU, 3 steps. **Pass the masses as an argument of your jit, not as a
closure constant**, as the lane's runner does. Closed over, they are a constant of the
program, which XLA compiles differently: in that test one float32 rounding of one force
component moved at the third step (1 of 3072 state elements, 1.5e-8). That is
round-off, not a different force, but it is not the lane's bits.

`aux = (f_b, bits, needs)` rides in the carry: the next rebuild's force scale (only
under `mac_type="dehnen_error"`), the overflow bits OR-ed over the steps, and the walk's
needs max-ed over them. Check `bits` once per segment, after the scan, as
`strict_run_v2` does.

## The eager / traced split

**Eager** (host work, outside any trace):

| call | what it does |
|---|---|
| `FusedForceProvider.prepare(solver, pos, m, *, leaf_size, max_order, theta=None, bounds=None, runtime_overrides=None, caps_only=False)` | the eager prepare `strict_run_v2` bootstraps with (fused device-resident layout; validated walk caps recorded on the engine, which size the traced walk; the eq (16b) prepass under `dehnen_error`). Returns `(provider, FusedTopology)`. `caps_only=True`: only what a particle-carry scan reads of it -- the tree, the upward pass, the eager walk's caps, the seed, no list, no downward pass, no near-field payload; the template is the shapes of a rebuild under `jax.eval_shape`, equal to the full prepare's, and the topology holds a `PreparedCaps` (the eager tree and the verdict) that nothing can be evaluated on. The particle carry's bootstrap, segment retry and the runner's restore use it (`JACCPOT_STRICT_CAPS_ONLY_PREPARE=0`: the full prepare). |
| `FusedForceProvider.freeze_template(solver, prepared, *, leaf_size, max_order, theta=None, runtime_overrides=None)` | a provider for an existing prepared state: its shapes, not its buffers |
| `prov.replanned(needs, pos_start, m_start, *, walk_only=False, caps_only=False)` | the segment retry: grows the unnamed flat-walk caps to cover `needs`, prepares again at the segment's start (caps-only on request) and returns `(provider, topology)`, or `None` when nothing can be re-planned (a named cap, no walk flag, or a cell-leaf tree overflow alone) |
| `prov.retry_mode(needs, retries, tree_first)` | the runners' retry policy, for a caller that drives its own segments (nornax): `"full"` for the first retry, `"walk"` (`replanned(..., walk_only=True)`) for a second after a first that re-planned the tree, else `None` (raise); `None` also under `JACCPOT_STRICT_SEGMENT_RETRY=0` |
| `prov.key(like=None)` | what a compile cache keyed on the provider must hold: leaf, order, theta, whether f_b is carried, the diagnostic modes, the level-loop bound, the validated walk caps, the template's shapes. `G`, the softening, the MAC and the kernels are fixed per engine and not in it. |

**Traced** (call inside your own `jax.jit`; each returns arrays or tracers):

| call | returns |
|---|---|
| `prov.rebuild(pos, m, *, force_scale=None, bounds=None, num_valid=None, cross_hook=None)` | a `FusedTopology` (`fresh=True`) |
| `prov.accelerations(topology, params=None, *, order="input", force_scale=False)` | `[N, 3]` accelerations in the state's input dtype, or `(acc, f_b)`. `order="tree"` keeps the topology's Morton order (row `j` is particle `tree.particle_indices[j]`, no gather back). |
| `prov.capacity_ok(topology)` | the step's boolean verdict, as the scan folds it; equal to `overflow_ok(topology.overflow_bits())` |
| `prov.step(pos, m, params=None, aux=None)` | `(acc, (f_b, bits, needs))` |

Under `mac_type="dehnen_error"` each rebuild takes eq (16a)'s thresholds from the
previous evaluation's f_b, so `rebuild` needs `force_scale` and the first `step` needs
an `aux` that carries one, `(f_b, None, None)`: for example the f_b of
`prov.accelerations(topology0, force_scale=True)` on the prepare's topology.

## `FusedTopology` and the overflow bits

`FusedTopology` (`jaccpot.runtime.fused_topology`) is what a rebuild returns: the
refreshed prepared state (tree, local expansions after L2L, CSR near list, near-field
payload; its far list is `None`: the rebuild built one, used it for M2L and kept none),
plus what the walk and the tree reported about their fixed capacities, as tracers moved
off the engine's side channels. `FusedTopology.from_prepared(prepared)` wraps an eager
state (no rebuild verdict).

`topology.overflow_bits(extra=0)` gives one `uint32` bit per fixed capacity:

| bit | name | set when | verdict? |
|---|---|---|---|
| 0 | `OVERFLOW_WALK_FAR` | the walk's far list overflowed | no: names what overflowed, for a re-plan |
| 1 | `OVERFLOW_WALK_NEAR` | the walk's near list overflowed | no |
| 2 | `OVERFLOW_WALK_QUEUE` | the walk's queue overflowed | no |
| 3 | `OVERFLOW_TREE` | the cell-leaf rebuild: a leaf over `leaf_capacity`, a tree deeper than the level-loop bound, or a level wider than the level loops' batch | yes |
| 4 | `OVERFLOW_SATURATED` | a far list this rebuild built is saturated (the flat walk saturates it on any of bits 0-2) | yes |
| 5 | `OVERFLOW_NEIGHBOR_ROW` | a near-field row reached the traced per-leaf cap | yes |
| 6 | `OVERFLOW_FAR_CAP` | the far count reached the traced compact far-pair cap | yes |
| 7 | `OVERFLOW_RECTANGLE` | the rectangle near-field layout's per-target capacity (CSR lane off) | yes |
| 8 | `OVERFLOW_LEVEL_PLAN` | the caller's (multi-GPU) level plan, passed in through `extra` | yes |
| 9 | `OVERFLOW_CROSS` | the caller's (multi-GPU) cross-domain capacities, through `extra` | yes |
| 10 | `OVERFLOW_PARAMS` | traced force parameters outside what the walk was built for (below) | yes |

The verdict is `overflow_ok(bits)`, i.e. `(bits & OVERFLOW_FAIL_MASK) == 0` with
`OVERFLOW_FAIL_MASK = 0x7F8` (bits 3-10). **Any verdict bit means that step's force is
wrong**, not merely slow. Before S2 a tree overflow inside the scan left the verdict
`True` (rel-L2 0.546 at N = 512); bit 3 now fails it.

`topology.needs()` is `int32[11]` (`NEEDS_FIELDS`): the walk's `far_needed`,
`near_needed`, `peak_wavefront`, `far_overflow`, `near_overflow`, `queue_overflow`,
`lower_bound`, then `tree_overflow`, then what the cell-leaf tree needed
(`TREE_NEEDS_FIELDS`): `leaf_needed` (the live cell leaves, counted before the cut to
`leaf_capacity`, so exact past an overflow), `depth_needed` (levels) and
`widest_needed` (the widest level's node count). A failed segment re-plans from their
running maximum with `prov.replanned`, once per segment, at the segment's start: the
walk's caps from the first seven, the tree's bounds from the last three (S3) -- the leaf
capacity grows to `auto_leaf_capacity(leaf_needed, headroom)` (`TreeConfig.leaf_headroom`
under `leaf_capacity="auto"`, x1.15 for an explicit capacity; rounded up to 1024; sticky
on the engine), the level-loop depth bound to `depth_needed + 8`, and the level loops'
batch width to `ceil(1.25 x widest_needed)`. The width is what a long galaxy run reaches
first (the 25M quarter-orbit probe: the widest radix level +10.9 % in 489 steps, the
live leaves +0.26 %). A grown capacity is a new template shape, so the re-run compiles.
A segment whose first retry re-planned the tree may retry once more, for the walk only
(`prov.replanned(..., walk_only=True)`): the walk's needs in a segment whose tree
overflowed were measured on that broken tree and can fall short, the re-run's are real.
At most two retries per segment; a further failure raises, names what overflowed against
which bound, and says how many retries ran.

## Force and step parameters

`ForceParams(G, eps2)` (exported as `jaccpot.ForceParams`) and
`StepParams(force, dt, external)` (`jaccpot.runtime.fused_topology`) are pytrees whose
**Python floats are static** (aux data: compile-time literals, exactly the closure
constants the lane always traced) and whose **arrays are traced**. The softening is
carried squared, as the kernels use it.

- **The default is static.** `params=None`, or the engine's own values as Python
  floats, gives the program the lane always compiled: no extra argument, the same
  constants and compile keys (CPU program identity 36/36 variants, GPU 18/18, the nine
  GPU pins bitwise; [`production_2026-10.md`](production_2026-10.md), S2).
- **Traced is the opt-in.** Arrays are cast at the boundary (G and eps2 to the working
  dtype, dt to the state's dtype) and become arguments of the compiled program: one
  compile serves every value. Three calls with three different (dt, G, eps2, external)
  values compile and trace once (`tests/integration/strict/test_traced_params.py`).

On `strict_run_v2`: `dt` may be an array, `force_params=ForceParams(...)`, and
`external_params=` (any pytree; given, the field is called as
`external_acceleration_fn(state, external_params)`, the one-argument form otherwise).
`strict_fused_prepared_eval_fn(force_params=...)` takes the same. The particle carry's
handle records the force parameters its self-gravity was computed with
(`StrictParticleCarry.force_key`); a call continuing with others evaluates its start
force again.

**Traced vs static, measured.** On an A100 (deterministic ops, A-vs-A control 0
elements) the traced programs are bitwise the static ones: S1 force 0 of 600,000
elements and final state 0 of 1,200,000, the same at G = 4.3e-6, and the 25M disc's
force (ferrers3, eps 1.5e-3) 0 of 75,000,000. On CPU they are bitwise under Plummer, and
under ferrers3 differ by one float32 rounding (753 of 1536 elements on the pure-JAX
near field, 732 on the CSR lane in interpret mode, at most 1.35e-7 of the largest
component): XLA's algebraic simplifier rewrites the kernel's `rsqrt(x)^3` into
`rsqrt(x) / x` only where it can prove `x = max(r^2, h^2) > 0`, which a literal `h^2`
gives it and a parameter does not. On the tree-order carry any one traced argument
moves 2 of 3072 velocity components on CPU (1 and 8 ulps of their own values; the
positions are bitwise): the extra argument changes XLA's compilation, not the arithmetic.

**Gradients in G and eps2.** `jax.grad` through `prov.accelerations` with traced
parameters equals central differences (G to 1e-10, eps2 to 1e-6) on the **pure-JAX
near field in float64** (CPU). It is not verified on the GPU's Pallas lane, whose
production forward has two kernels without a reverse rule (the direct near-field layout
and the particle-major L2P; `ARCHITECTURE.md` section 6).

## Limits

- **`mac_type="dehnen_error"` takes the solver's own G and softening only.** Eq (16a)'s
  thresholds and the far half of its force scale are built with them in the rebuild, so
  any other value, traced or not, raises `ValueError` (user decision, S2).
- **A traced softening beyond the walk's sets `OVERFLOW_PARAMS`.** The rebuild never reads
  the parameters: the walk is the engine's. A compact kernel (`ferrers3`, the default, or
  `wendland_c2`) is Newtonian beyond its support `h = k eps`, and the walk keeps pairs
  inside its separation floor out of the far field. It covers
  `eps2 <= prov.walk_eps2() = max((floor / k)^2, softening^2)`; a larger `eps2` would let
  the unsoftened far field take pairs inside the support. `prov.step` sets bit 10 (a
  tracer for a traced `eps2`); `strict_run_v2` raises before any work. Build the solver
  with the largest softening the run uses. Plummer has no support and no limit.
- **Bucket leaves are refused** for a rebuild from the template (above); the state carry
  (`strict_run_v2(carry="state")`, the library default) still runs on them.
- **A tree overflow is re-planned at the segment's start** (S3; S2 raised): see the needs
  above. A walk that overflowed a NAMED cap declines the retry altogether, tree or not.
- **Not yet on the provider:** the fused multi-GPU lane (`jaccpot/distributed/fused.py`)
  has its own force step; nornax's block-step (multi-rung) integrators do not call it
  yet. nornax's single-rung KDK leapfrog does (S4): `nornax.adapters.FusedProviderForce`
  carries `step`'s aux through `leapfrog_kdk_rollout`, and `nornax.retry_segment`
  re-plans a segment with `retry_mode` and `replanned`. Its forces are this lane's; nornax's
  `tests/integration/test_jaccpot_provider_rollout.py` compares its rollout with
  `strict_run_v2(carry="particles")` and the production runner.

## The fp64 validation lane

`preset="large_n_gpu"` computes the force in float32 (the production lane). An explicit
`precision="fp64"` (or `working_dtype=jnp.float64`) selects the **validation lane**: the
same fused lane in float64, correct and selectable, with its time and memory recorded
but no budget (user decision D3). Nothing a caller does today changes: the preset still
defaults to float32.

- **The state keeps its own dtype.** A float64 state under the float32 force works: the
  refresh casts positions and masses to float32, the force is cast up at the scatter, and
  the drift, kick, f_b and external field stay float64. Before S2 the tree-order carry
  (`JACCPOT_STRICT_CARRY_ORDER=tree`) rounded such a state to float32 at every step; it no
  longer does.
- **Every fused-lane Pallas kernel runs float64 natively** (A100, jax 0.11.2), so no kernel
  falls back to XLA: the walk (`soa` nodes), the COM radii table, the leaf P2M, the
  M2M/L2L and M2L CSR lanes, the particle-major L2P, the CSR near field's table and direct
  kernels. 23 native cases, the largest error 6.4e-15 of the reference
  (`tests/unit/operators/test_pallas_float64_sweep.py`).
- **Accuracy.** p = 12, theta 0.4, unsoftened, against a direct sum (CPU): 7.93e-10 in
  float64, 9.20e-6 in float32. Where the FMM is exact (theta 1e-3), float64 Verlet equals
  the float64 direct-sum Verlet to 1.11e-15 (float32 force: 1.43e-4). At production
  settings the truncation error dominates: 2e5, the S1 configuration on an A100, 7.7656e-4
  float64 against 7.7637e-4 float32; p8, theta 0.4: 6.434e-7 against 6.946e-7.
- **Cost.** At 2e6 (shared card, `bench/fused_memory_budget.py` defaults, no donation),
  state / force dtype: 32/32 508 B/particle, 64/32 632, 32/64 707, 64/64 820. The 8e6 and
  1e8 rows are in [`production_2026-10.md`](production_2026-10.md) (S2). The M2M/L2L
  cascade kernels' Triton compile grows steeply with the order in either dtype (140 s at
  p8; at p12 in float64 it had not finished after 12 minutes): validate at p12 with
  `JACCPOT_CASCADE_PALLAS=0`, which runs the cascades as XLA level loops.
- **Bench:** `bench/fused_memory_budget.py --state-dtype float64 --force-dtype float64`
  (both float32 by default).

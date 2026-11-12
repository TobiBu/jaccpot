# Production runs on the fused lane — `ProductionRunner`

`jaccpot.ProductionRunner` runs a long simulation on the fused single-GPU lane (with
`mesh=`, on several cards: [Multi-GPU mode](#multi-gpu-mode)): a host loop of K-step
segments, each ONE `strict_run_v2(carry="particles")` call continuing from the previous
call's `StrictParticleCarry` handle. Between segments it computes the
diagnostics, writes restart files and calls your callbacks. Code:
`jaccpot/production.py`; the configuration it runs is `preset="galaxy"`
(`GalaxySettings`, same module; see [`fused_provider.md`](fused_provider.md)). What was
measured while it was built is in [`production_2026-10.md`](production_2026-10.md)
(section S3).

## Quick start

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaccpot import FastMultipoleMethod, ProductionRunner, SegmentConfig
from jaccpot.production import DiagnosticsLog, SnapshotWriter

solver = FastMultipoleMethod(preset="galaxy", softening=1.5e-3)

def halo(state, p):          # module level: ONE function object for the whole run
    x = state[:, 0]          # the packed [N, 2, 3] state; read the positions only
    r = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    return -p["gm"] * x / (r * (r + p["rs"]) ** 2)   # a Hernquist halo

def halo_potential(state, p):   # its potential per unit mass, for the energy
    r = jnp.sqrt(jnp.sum(state[:, 0] ** 2, axis=1))
    return -p["gm"] / (r + p["rs"])

runner = ProductionRunner(
    solver,
    segment=SegmentConfig(steps=50, restart_every=4, restart_path="run/restart.npz"),
    external_acceleration_fn=halo,
    external_potential_fn=halo_potential,
    callbacks=(SnapshotWriter("run/snapshots", every=2), DiagnosticsLog("run/diag.jsonl")),
)
res = runner.run(state, masses, dt=1e-3, num_steps=500,
                 external_params={"gm": 1.0, "rs": 2.0})
# killed? continue in a fresh process, same script:
res = runner.run(dt=1e-3, num_steps=500, restart_from="run/restart.npz",
                 external_params={"gm": 1.0, "rs": 2.0})
```

`state` is the packed `[N, 2, 3]` positions and velocities, `masses` `[N]`. `num_steps` is
the run's total from `t0`: a continued run runs the steps still missing. The one-shot
form is `jaccpot.production.run(solver, state, masses, dt=..., num_steps=..., ...)`.
The solver runs Dehnen's error MAC at 1e-4 (next section): `G` and the softening are the
solver's, so a run with another value needs another solver.

## The timestep

The runner takes one shared `dt`. For the 25M disc + bulge in its NFW halo at softening
1.5e-3, **dt 1e-3** is the production step: plan session C1 ran this configuration
through `ProductionRunner` (`preset="galaxy"`) for one orbit at 2.5 R_d (t = 1.5) at dt
1e-3, 5e-4 and 2.5e-4. Against the 2.5e-4 reference, dt 1e-3 gave:

- energy: max |dE/E0| 1.9e-7, a bounded offset, not a drift;
- per-particle energy diffusion: within 0.35 % of the reference;
- Lagrangian radii (0.5-90 %, every component): within 0.25 %;
- Sigma(R) and sigma_z(R): worst bin within 6.3e-4 and 1.4e-4 of the reference; A2
  within 5.5e-6;
- |dL|/L: 3.5-9.0e-7 in all three runs, the float32 floor.

This step is the acceleration criterion `eta sqrt(eps / |a|)`'s shortest step at eta
about 0.33. C1 also found that block timesteps do not pay on this disc at equal accuracy,
with that criterion or an orbit-time one ([`production_2026-10.md`](production_2026-10.md),
C1; rows in `bench/results/c1_criterion_2026-10/`).

Two limits on carrying dt 1e-3 elsewhere:
- **It covers one orbit.** The energy error stayed bounded there; keep the energy
  diagnostic on in longer runs.
- **It holds for this N and softening.** With fewer particles or a smaller softening,
  close encounters at the softening scale set the step. C1's 1e5-particle N-body at this
  softening (its configuration A) is such a case.

## The MAC

`preset="galaxy"` accepts a far pair by Dehnen (2014) eq (16a) at a relative
force-accuracy target of 1e-4 (`GalaxySettings(mac_type="dehnen_error",
adaptive_eps=1e-4)`, user decision 2026-10-21); before, by the geometric Dehnen test at
theta 0.8. The criterion runs inside the flat walk, with eq (16b)'s force scale f_b
carried from each step's force to the next step's walk
([`dehnen_error_fused_2026-10.md`](dehnen_error_fused_2026-10.md)); the preset passes
what `bench/fused_memory_budget.py --mac-type dehnen_error --adaptive-eps 1e-4` passes
and compiles the same programs (`provider_hlo_identity.py`'s galaxy pair). Measured on
one A100 with no other job, the S3b branches merged, the second of two runs each
([`production_2026-10.md`](production_2026-10.md), S3b):

| | geometric, theta 0.8 | `dehnen_error`, 1e-4 |
|---|---|---|
| 25M disc + bulge, step | 233.6 ms | 260.5 ms (1.12x) |
| 25M disc + bulge, rel-L2 against the fp64 direct sum | 4.18e-4 | 1.49e-4 (2.8x lower) |
| 1e8 clipped Plummer, step | 855.3 ms | 831.3 ms |
| 1e8 clipped Plummer, peak | 257 B/particle | 279 B/particle |

What it costs:

- **Memory:** about 22 B/particle more at 1e8, so a lower one-card ceiling.
- **`G` and the softening are the solver's.** The rebuild builds eq (16a)'s thresholds
  and the far half of f_b with them, so `strict_run_v2` refuses any other value, traced
  or not, and the runner passes none (`SegmentConfig.traced_params` traces `dt` and the
  field's parameters only). Another `G` or softening is another solver, and compiles.
- **The tail:** the bulk of the error distribution narrows as in Dehnen (2014); its
  0.1-1 % tail does not narrow like it.
- **Restart files** carry f_b (`force_scale`, +4 B/particle).
- **Off the fused lane** -- `compute_accelerations` on this preset, the large-N lane's
  general route -- the criterion costs far more memory: at 2e5 (clipped Plummer, an
  A100 other jobs shared) one call peaked at 4,553 B/particle against 736 with the
  geometric MAC (not broken down further). The production path is `strict_run_v2` /
  the runner, on the fused lane.

**A long near row (#487), fixed in A1 (2026-11).** In the 1e8 bulge-free disc and after the 1.06e8 merger's first
pericentre, one sink leaf's near row held up to 5.3M leaves (70 % of the tree). The CSR build's in-row rank took up
to 124 s on that row in each list build, 20-90x the step. The sink was a one-particle leaf whose MAC radius was a
float32 rounding residue (~1e-9). On a GPU, its walk-table entries `s_5`, `s_6` came out NaN, and a NaN fails eq
(16a) against every node. Two changes, both default:
- a non-finite `s_n` is set to 1, its upper bound (finite entries untouched): the row is 169 leaves again;
- rows longer than 2048 are merge-sorted in `O(n log^2 n)`, the same arrays bit for bit, so no single row can stall
  a build again whatever the MAC.

At that state `dehnen_error` has rel-L2 2.19e-4 and p99 5.7e-4 against the geometric MAC's 2.76e-4 and 7.3e-4 at
theta 0.7, from half its far pairs; after the merger's first pericentre 1.94e-4 against theta 0.7's 1.68e-4 (p99
5.7e-4 against 5.9e-4), again from half the lists ([`production_2026-11.md`](production_2026-11.md), A1).
`FastMultipoleMethod(preset="galaxy", mac_force_scale_floor=0.5)` floors each node's `min_b f_b` at half a lower
bound that a correct f_b estimate always meets. It is opt-in: it never binds on a correct estimate, and it did not
bind at #487.

`theta` decides no pair under `dehnen_error`: the walk accepts by eq (16a) and its
convergence clause `rho_A + rho_B < mac_theta_max r` (`mac_theta_max` 1.0), and the first
step's f_b seed walks at theta 0.8 whatever the solver's (`mac_force_scale_prepass_theta`
sets it). Theta 0.3, 0.5 and 0.8 gave the same lists and the same states, bit for bit
(CPU, N = 2048, 2 steps); it is still part of the compile keys. A tighter target is
`FastMultipoleMethod(preset="galaxy", adaptive_eps=1e-5)`.

**The geometric MAC back** -- the documented fallback (user decision U3, 2026-11-03: `dehnen_error` stays the
default and is fixed, the geometric MAC is the fallback). It is the preset before 2026-10-21, the same engine
(see
`tests/unit/test_preset_galaxy.py`):

```python
import dataclasses

from jaccpot import FastMultipoleMethod, ProductionRunner
from jaccpot.production import GalaxySettings

base = FastMultipoleMethod(preset="galaxy").advanced
solver = FastMultipoleMethod(
    preset="galaxy", softening=1.5e-3,
    advanced=dataclasses.replace(base, mac_type="dehnen"),
)
runner = ProductionRunner(solver, settings=GalaxySettings(mac_type="dehnen", adaptive_eps=None))
```

The runner reads only the run-time half of `settings` (`leaf_size`, `max_order`,
`carry`) and records all of it in the restart fingerprint; it does not build the solver
from it, so pass both. The legacy keyword `FastMultipoleMethod(preset="galaxy",
mac_type="dehnen")` builds the same engine with a `DeprecationWarning`.

## Launch environment

Start every production run with XLA's fusion autotuner off, before jax starts:

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_enable_fusion_autotuner=false"
```

For some fusions XLA times the native emitter against the block-level (Triton) one on
the card and keeps the faster. On a card other jobs share those timings vary, so a fresh
process can compile other code from the same program: a resumed run, or a rerun, is then
correct but not bitwise the uninterrupted one. In S3 session 2 (A100s shared with other
jobs; the first 2e6 particles of the 25M disc with its NFW halo, 20 steps, resumed from
step 10), 8 of 12 fresh-process resumes differed in 436,234 of the 12M state elements (max
4.4e-6); in the pair whose optimised programs were compared, 7 fusions had switched
emitter, one of them the NFW field added to the self-gravity. With the flag 8 of 8 were
bitwise, and DET2 (`bench/dce_pins.py`, the gradient on the production flags) was
bitwise A-vs-A 5 times in 5, against 3 in 5 without it. On a free card L1-L3 kept their
step time, peak and compile time (`compare --envelope` passed), and the
deterministic-ops pins S1, S1x and S4b did not move.

The flag has to be in `XLA_FLAGS` when the backend starts (XLA reads it once), so the
library cannot set it: `ProductionRunner.run` warns once per process on a GPU when it is
missing. `bench/fused_memory_budget.py`, the production-flag pins of `bench/dce_pins.py`
(L1-L3, DET1-DET3) and `bench/provider_hlo_identity.py --device gpu` add it unless
`XLA_FLAGS` sets it.

**The flag is experimental: re-check it at every jax upgrade.** XLA refuses to start on a
flag it no longer knows, and what an experimental flag covers can change. After a bump,
record the DET pins twice on a shared card (`record --tag X --pins DET1,DET2,DET3`, then
`X-b`, `compare X X-b`) and L1-L3 on a free one. A per-fusion autotune cache
(`--xla_gpu_per_fusion_autotune_cache_dir`) also made resumes bitwise (4 of 4), but it is
state: it holds what the first process timed, it has to travel with the restart files,
and XLA does not invalidate it across versions or devices.

## What a segment is, and why

The runner does not scan over `FusedForceProvider.step` itself: its segments ARE
`strict_run_v2` calls. So "the runner equals one long call" reduces to "chained calls
equal one call", and the scan's memory work, its segment retry and its re-plan come
with it. Each call passes `carry_total_acceleration=True`: the handle carries the scan's
own total acceleration at the returned state and the next call starts from it. Without
that, a call continuing from a handle evaluates the external field again, eagerly,
outside the scan, and that differs from the scan's fused value in the last bit (CPU,
N = 512: 1 of 3072 state components, 1.9e-6, after 3 x 2 steps against 6).

| `SegmentConfig` | default | |
|---|---|---|
| `steps` | 50 | steps per segment. **Choose it to divide `num_steps`.** |
| `traced_params` | `True` | `dt`, `G`, the softening squared (not under `mac_type="dehnen_error"`, the preset's MAC: there they are the solver's) and the field's float parameters as arrays: arguments of the compiled programs, so other values compile nothing. `strict_run_v2` itself keeps literals by default. |
| `donate_input` | `False` | the first segment consumes the caller's state; later segments always consume their (the runner's) input |
| `diagnostics_every` | 1 | moments at step 0, every n-th edge and the last; 0 = off |
| `diagnostics` | all | of `"momentum"`, `"angular_momentum"`, `"com"`, `"kinetic"`, `"energy"` |
| `diagnostics_every_energy` | 10 | edges between energy diagnostics |
| `restart_every`, `restart_path` | 0, `None` | a restart file every n segments; `{step}` / `{segment}` are formatted into the path, otherwise each restart replaces the last |
| `domain_guard` | `"warn"` | the IC guard before a run's first segment (`jaccpot.domain.domain_guard`, A2): warns when a few far particles set the Morton box (more than 20x the diameter holding 99.9 % of the particles), with the box, the level-21 cell, the farthest particle and a tree-depth bound; `"error"` refuses the run; `"off"` measures nothing. It reads the state only: no program changes. |

**Segment lengths.** `num_steps` is part of a scan's compile key: every distinct segment
length compiles its own scan (a ragged 14-step last segment recompiled for 52.8 s at 25M
on an A100). The plan is `num_steps // steps` segments, then the remainder, last
(`segment_lengths`); the runner warns once when there is one. A remainder of one step joins
the last segment (5 steps of 2 run as 2 + 3): XLA inlines a scan of length one and fuses it
differently from a longer scan's loop body, so a 1-step segment is not bitwise the steps of
one long call (2 + 2 + 1 differed from 5 on a CI runner). For the same reason
`SegmentConfig(steps=1)` gives the trajectory to round-off, not bitwise one long call.

**Compiles.** A fresh run compiles the eager prepare's programs, the start force and one
scan. The first segment's prepare is caps-only (`FusedForceProvider.prepare(caps_only=True)`:
the tree, the upward pass and the walk's caps, no list and no downward pass -- the start
program evaluates the force), and so are a segment retry's and a restore's. With `donate_input=False` the second segment compiles the scan once more (with
the state donated) and every later segment reuses it; `donate_input=True` gives every
segment one program. Chained `strict_run_v2` calls without `carry_total_acceleration`
also recompiled at the second call (the handle's start force was not donated: 17.7 s
against 2.4 s at N = 512 on CPU).

**Retry.** A segment whose lists or tree overflow is re-planned and re-run by
`strict_run_v2` (`FusedForceProvider.replanned`) from what its steps up to the first
failure needed (the later ones ran on a truncated force, off the trajectory): once, or
twice when the first re-plan grew the tree (the second for the walk only), then again
while each re-run gets further (see `docs/fused_provider.md`). Each retry recompiles the
segment's scan (52.8 s at 25M on an A100, as a ragged segment's compile); a segment whose
needs stay within the re-plan's headroom (x1.5 for a list) after its first failure takes
one. A failure that learned nothing raises a `RuntimeError` naming the last restart
file. A donated segment keeps
a host copy of its start state for the re-runs (24 B per particle, one device-to-host
copy per segment). On a GPU it is an asynchronous copy into pinned host memory
(`strict_carry.host_copy`): 0.10 s at 1e8 on an A100 (the run's first 1.4 s), where the
synchronous copy into NumPy memory took 1.0-1.5 s per segment, the larger part of an
edge's cost at 1e8. The moments themselves cost 15 ms at 1e8 (their first call compiles,
0.6 s once per process).

**List limit.** A re-plan sizes an outgrown list at 1.5x what the segment needed
(`JACCPOT_FLAT_WALK_CAP_HEADROOM`), but never past the largest list the int32 index range
builds, `_interaction_cache.flat_walk_list_limit`: 1,431,306,240 directed entries on the
Pallas walk, whose CSR build addresses 3x the canonical width. A list that needs more
raises the `RuntimeError` before the re-prepare, naming the remedies. A far list that
keeps growing usually means the per-axis Morton box stretched. In the S3 gate's clipped
Plummer (softening 1e-7, dt 1e-2) close encounters fling a few particles out from step
~200, every cell elongates, and at 1e8 the far list needed 402M directed entries at the
start, 632M by step 301 and 1,005M by step 401. The third re-plan clamped the far list
at the limit instead of 1,508,900,864 (which raised `pair width 754450432 overflows
int32`), and the run went on to step 551 without another re-plan: the step took 874 ms
on the cubic box, 1,596 ms at the stretched one (steps 452-501) and 1,004 ms once
particles flung along z had made the box cubic again (502-551). On the same IC at 1e7 the
far count grows 44 % by step 451 with the per-axis box; with `JACCPOT_CUBIC_BOUNDS=1` the
same positions need 22 % fewer pairs than at step 0. `preset="galaxy"` builds the cubic
box (below); other configurations get it with `TreeConfig(cubic_bounds=True)`.

**The Morton box.** `morton_encode` normalises each axis by its own extent, so the
per-axis box the lane infers from the particles (the library default) gives every cell
the box's aspect, and a few outliers set it. `preset="galaxy"` builds a cube on the
longest extent instead (`GalaxySettings.cubic_bounds`, `TreeConfig(cubic_bounds=True)`,
S3b): every cell is a cube. `JACCPOT_CUBIC_BOUNDS` overrides the configuration when it is
set (`1`: cube, `0`: per axis; read per call, and a cube is part of the provider's compile
key); the multi-GPU lane reads only the variable. Measured with the eager prepare on one
A100 (lists in directed entries; rel-L2 against the fp64 direct sum on 4096 targets;
ferrers3, the geometric MAC at theta 0.8, cells of 64, `cell_min_level` 8, p6):

| particles | box aspect | far, per axis -> cube | near | rel-L2 | live leaves |
|---|---|---|---|---|---|
| 25M disc + bulge, step 0 (L3) | 1.004 | 94.80M -> 94.75M | 37.58M -> 37.58M | 4.18e-4 -> 4.21e-4 | 1,502,983 -> 1,504,705 |
| the same run, step 250 | 1.009 | 95.59M -> 94.75M | 37.89M -> 37.91M | 4.26e-4 -> 4.22e-4 | 1,495,590 -> 1,496,838 |
| the same run, step 500 | 1.012 | 96.80M -> 95.78M | 38.45M -> 38.46M | 3.85e-4 -> 4.19e-4 | 1,507,252 -> 1,508,319 |
| its disc alone (21M, thin in z) | 3.01 | 215.92M -> 78.87M | 56.47M -> 28.14M | 1.02e-3 -> 5.28e-4 | 1,494,684 -> 1,273,521 |
| 8e6 clipped Plummer (L1) | 1.004 | 39.09M -> 38.95M | 9.41M -> 9.42M | 4.33e-4 -> 4.30e-4 | 763,545 -> 762,927 |
| 2e6 unclipped Plummer | 2.77 | 11.10M -> 7.20M | 4.99M -> 2.35M | 8.29e-4 -> 5.23e-4 | 102,648 -> 105,438 |
| 4e6 unclipped Plummer | 3.20 | 58.95M -> 14.65M | 13.22M -> 4.80M | 2.83e-3 -> 4.89e-4 | 219,355 -> 217,961 |
| 8e6 unclipped Plummer | 2.14 | 22.34M -> 29.14M | 13.14M -> 9.53M | 4.87e-4 -> 4.35e-4 | 405,630 -> 434,822 |

Where the box is a cube anyway -- the production disc + bulge, whose bulge reaches r = 20
on every axis, and the clipped Plummer -- the lists agree to about 1 % and the rel-L2
within 9 %, either way; over the 25M production run (snapshots of the S3 gate) the
per-axis box stays within 1.2 % of a cube. Where it is not, the cube walks up to 75 % fewer far and 64 % fewer near pairs and its
error is up to 5.8x lower. The one row where it walks more far pairs (8e6 unclipped) it
walks 27 % fewer near ones, the costlier list, at a lower rel-L2 (its worst target is
1.08e-2 against 7.3e-3; on single targets either box can be worse). The step's peak
follows the lists (the 21M disc's step, `fused_memory_budget.py --skip-eval`: 6.75 ->
5.09 GiB, 346 -> 261 B/p; L1 and L3 unchanged); the leaf capacity, depth and level widths of the cube moved like the per-axis box's along the
production run, so the same headroom serves both.

## Edges, callbacks, diagnostics

A callback is `Callable[[SegmentEdge], Optional[bool]]`, called at step 0 and after every
segment; a truthy return stops the run after that edge (`RunResult.stopped`). Pass the
result as `restart_from` to continue in memory. `SegmentEdge` holds `step`, `time`,
`segment`, `retries`, `wall_s`, `state`, `masses`, `self_acceleration`, `diagnostics`,
`plan` and `state_host()`; its arrays are valid DURING the callback only (the next
segment consumes the state). `SnapshotWriter(directory, every=k)` writes
`snapshot_<step>.npz` (or keeps host copies with `directory=None`); `DiagnosticsLog(path)`
appends a JSON line per edge. `step_callback` is `strict_run_v2`'s in-scan hook, passed
through; its step index counts from 0 in each segment.

The diagnostics are fp64 reductions on device (one small jitted program per edge,
fetched to the host), against step 0's baselines (`RunResult.baselines`):

| field | definition |
|---|---|
| `momentum`, `momentum_drift` | `P = sum m v`; `|P - P0| / sum m |v|_0` |
| `angular_momentum`, `angular_momentum_drift` | `L = sum m x x v`; `|L - L0| / |L0|` |
| `com`, `com_drift` | `sum m x / M`; `|COM - COM0 - P0/M (t - t0)|` (against its free motion) |
| `kinetic`, `kinetic_change` | `K = 1/2 sum m v^2`; `(K - K0) / K0` |
| `potential_self`, `potential_external`, `potential` | `W_self = 1/2 sum m phi` (the fused lane's O(N) potential at the edge's positions), `W_ext = sum m Phi_ext` (`external_potential_fn`; 0 without), `W = W_self + W_ext` |
| `energy`, `energy_drift`, `energy_ok` | `E = K + W`; `(E - E0) / |E0|`; whether the potential's rebuild fitted its capacities (otherwise the energy fields are `None`, and the row WARNS) |
| `energy_replans` | the energy-only re-plans the potential needed at this edge (#488; 0 normally) |

An external field changes P and L physically; the drifts then measure that.

**The energy** is due at every `diagnostics_every_energy`-th edge (10; the user's
cadence: the moments are nearly free, the energy costs a force evaluation). Its program -- the topology rebuilt at the edge's positions from the handle's
provider, `FusedForceProvider.potential`, the external potential, the sums in fp64 -- is
compiled once per plan and costs about one force evaluation (1.1x at 25M and 1e8 on an
A100). Step 0's `E0` needs a provider, so the runner keeps step 0's state (the caller's
array, or a host copy when `donate_input` consumes it) and evaluates it at the first
diagnostics edge after a segment: the step-0 row's energy fields are `None`, and
`RunResult.baselines` holds `energy` and `potential`. `external_potential_fn(state[,
external_params]) -> [N]` is the field's potential per unit mass, with the field's
contract and parameters. The energy program's trace leaves the engine's trace-time state
as it was (`_trace_state_kept`), so the trajectory with it is the one without (R4,
bitwise).

**It is never silently blind (#488, A2).** The energy program's key holds the level
registry (the level loops' batch widths, read when a program traces and in no provider
key): before, a width-only tree re-plan left the energy program on the old width, and
the 4.7e7 merger lost every energy row from t 6.15 to the end while the force ran with
`retries 0`. A rebuild that still does not fit is re-planned for the energy alone --
the rebuild's needs, `FusedForceProvider.replanned` at the edge's positions, at most
twice, all inside `_trace_state_kept`, so the force's plan and programs are untouched --
and the grown program is kept for the plan. Only when that fails (the segment retry off,
a named cap) is the row blind, and then it warns (`RuntimeWarning` with the time, the
overflow bits and the re-plans tried); a blind `E0` warns too.

**Wide boxes (A2).** A tree deeper than yggdrax's `MAX_TREE_LEVELS` (64) raises --
`yggdrax.TreeDepthError` at the eager prepare, a `RuntimeError` from the scan (never
re-planned: no bound covers it) -- with the Morton box, the level-21 cell and the
farthest particle. The IC guard above warns first. Far particles can leave the tree on
the escaper path below.

## Escapers: far particles off the tree (`escapers=`, opt-in)

A few particles far from the system set the Morton box. The 1e8 Sgr IC's sampler tails
(out to Mpc) made a 45.7 Mpc box and a tree 68 levels deep (`TreeDepthError`), and the
production runs clipped the tail by hand. Run such a system unclipped with the escaper
path:

```python
from jaccpot.escapers import EscaperPolicy

runner = ProductionRunner(
    solver, settings=settings, segment=segment,
    escapers=EscaperPolicy(r_escape=80.0),   # code units; r_return defaults to r_escape / 2
)
res = runner.run(state, masses, dt=1e-3, num_steps=10_000)
```

- **Who.** At the start and at every `every`-th segment edge, a tree particle farther
  than `r_escape` from the tree particles' centre of mass becomes an escaper; an escaper
  nearer than `r_return` comes back (the hysteresis). The IC guard's warning names a
  radius to start from (ten times the radius holding 99.9 % of the particles).
- **The tree** is built over the tree rows only, in their own box, and the escapers are
  in no leaf. Their count is a traced argument of the programs, so a changed split
  compiles nothing; while nobody escapes, the segments run the default programs, bit for
  bit.
- **The force.** An escaper feels the tree rows' monopole `-G M_b (x - X_b)/|x - X_b|^3`
  (and the external field, if one is set). Every tree row feels the equal and opposite
  uniform pull, so the total momentum is conserved up to the FMM's own error. Under
  `dehnen_error` an escaper's `f_b` is its coupling's magnitude, so a returning particle
  starts its thresholds from its monopole force.
- **Neglected:** the escapers' tidal field across the bound system (relative
  `(r_b / r_e)^2`) and the escapers' gravity on each other (far tails are tracers).
- **The energy ledger.** A particle that changes side changes the representation of the
  same phase-space state, and the energy jumps by the far field's multipole error there.
  With the energy diagnostic on, the runner evaluates the potential on both sides of a
  changing edge (two energy programs at that edge) and keeps the sum of the jumps. The
  rows carry `n_escapers`, `escaper_transfers`, `escaper_jump`, `escaper_ledger` and, with
  the energy, `potential_couple` (`W_c`), `kinetic_escapers`, `energy_ledgered`
  (`E - ledger`, the conserved quantity) and `energy_ledgered_drift`. `energy_drift`
  stays the raw `(E - E0)/|E0|`.
- **Row order.** The runner keeps the rows in an internal order (tree rows first).
  Callbacks, `SnapshotWriter`, restart files and `RunResult.state` see the caller's order
  (gathered at the edge, only when something looks). The in-scan `step_callback` and the
  external field see the internal order: a per-particle field works unchanged, a field
  with per-row parameters does not. `RunResult.escapers` holds the order, `n_bound`, the
  ledger and the transfers.
- **Restarts** carry the split and the ledger (`meta["escapers"]`, the arrays in the
  caller's rows plus `escaper_order`); a restart and a chained run continue bitwise. A run
  continues only in its own escaper mode.
- **Not supported:** the gradient mode (it refuses).
- **Cost.** One fused reduction over the rows per step (the bound centre of mass and the
  coupling sum), a membership check per edge, and at a changing edge a row permutation
  and two energy evaluations. HoreKa rows: `bench/slurm/escaper_timing.sbatch`.

### Escapers on the mesh (`mesh=` with `escapers=`; wave 3, MESC)

The same policy, Hamiltonian, ledger and diagnostics in the multi-GPU mode, in both of its
runtimes (one process with N devices, and N processes with one device each):

```python
runner = ProductionRunner(
    solver, settings=settings, segment=segment,
    mesh=MeshRunConfig(mesh),
    escapers=EscaperPolicy(r_escape=80.0),
)
```

- **The layout.** The bound rows are the ordinary mesh layout (`MeshState`), so the
  decomposition, the cuts, the global and Morton boxes, the cross walk, the migration,
  the repartition and every capacity see the bound rows only: an escaper at Mpc stretches
  no device's box. The escapers live in a TAIL of their own
  (`jaccpot.distributed.escapers.EscaperTail`): a block of `cap_e` rows per device,
  sharded like the layout, never sorted, cut or migrated. A particle stays on the device
  it escaped from; a returner re-enters that device's bound rows (the next migration or
  repartition takes it to its domain). Every move at an edge is device-local.
- **Per step**, inside the segment program: the tail drifts and kicks with the bound
  rows; after the tree force, `M_b` and `X_b` (per-device fp64 sums) and then the
  escapers' total pull (a 3-vector) are all-gathered and folded in mesh order -- two
  small all-gathers, not `psum`, so one process and N processes give the same bits.
- **At an edge** (every `every`-th): a small program counts, per device, the bound rows
  beyond `r_escape` and the escapers within `r_return`; on a change the potential is
  evaluated before and after (the ledger's jump), the tail grows first if a device's
  escapers outgrow `cap_e` (`tail_capacity`: 1.5x the largest device's count, at least
  64 rows; a new `cap_e` compiles the tail programs again, counted in
  `RunResult.escapers["tail_grown"]`), and one device-local program moves the rows. The
  carried accelerations are dropped (the next segment's start force evaluates the new
  split). Under `migrate=True` the bound box changed with the split, and the migration's
  fixed cuts are Morton codes of the old box, so the runner re-cuts with a full
  repartition in the new box (`RunResult.escapers["recuts"]`).
- **While the tail is empty** the segments run the DEFAULT programs: escapers on but
  nobody escaping is the run without them, bitwise (state, diagnostics). Once someone
  has escaped, one start and one segment program carry the tail for every split.
- **Output.** `RunResult.state`, `MeshSegmentEdge.state_host()` / `gather()` (and so
  `SnapshotWriter`) and the restart files are by id over the bound rows and the tail.
  `edge.state` / `edge.layout` are the bound layout only; `edge.escapers` is the tail.
  `edge.plan["escapers"]` has the tail's `cap` and per-device `counts`.
- **Restart files** (one process; several processes continue from a `RunResult`, as
  without escapers): the by-id arrays cover every particle, `escaper_ids` is the tail's
  layout and `meta["escapers"]` the policy, split and ledger. On the same mesh a restart
  is bitwise; on another mesh the bound rows are decomposed there and the escapers placed
  in its tail. After a changed split the file holds no acceleration (the continuation's
  start force computes it), as on one card.
- **What it neglects:** as on one card (the escapers' tides across the bound system and
  their mutual gravity). The tail's cost per step is O(`cap_e`) per device plus the two
  all-gathers; its memory is ~60 B per tail row (Sgr 2.2e6 on two cards: see
  `docs/production_2026-11.md`, MESC).

## The drift order (A3)

The drift is `x + dt (v + 0.5 dt a)`: the increment is summed first, then added to
the positions in ONE rounding (`GalaxySettings.drift_order="summed"`, the default since
multi-node plan A3; `jaccpot.runtime.fmm_state.drift_positions`). Before A3 the drift
was evaluated left to right, `x + v dt + 0.5 a dt**2`: in a float32 state `x + v dt` is
rounded to `x`'s grid and then `0.5 a dt**2` -- below `x`'s last bit for most
particles at `dt` 5e-4 -- is rounded against `x` on its own, with the same sign along
an orbit. On the 2026-10-31 merger (4.7e7 particles, `dt` 5e-4, t 0.8, `dehnen_error`,
[`production_2026-11.md`](production_2026-11.md), A3):

| drift | dE/E0 | dL/L0 | force rel-L2 | B/p at 1e8 (`dehnen_error`) |
| --- | --- | --- | --- | --- |
| legacy (before A3), float32 state | 8.66e-6 | 7.60e-6 | 2.04e-4 | 254.5 |
| summed (the default), float32 state | -1.24e-6 | 3.17e-7 | 2.07e-4 | 254.3 |
| float64 state (float32 force) | -1.25e-6 | 3.17e-7 | 2.01e-4 | out of memory at 9e7 |

- **The drift and the kick run in kernels of their own** (`fmm_state._isolated`: an `optimization_barrier` on the
  operands and the result): a backend that contracts multiply-adds into FMAs decides per kernel, so the same
  expression fused with different neighbours rounds differently; isolated, every program -- the scan, the
  multi-GPU segments migrating or not, the gradient rollout, nornax's `LeapfrogKDK` -- compiles the same kernel and
  agrees bit for bit. No peak memory (1e8: 232.8 / 253.1 B/p, geometric / `dehnen_error`, as legacy).
- **`drift_order="legacy"`** restores the old expression: the bitwise path to a run, or
  a restart file, from before A3. The single-card scan, the multi-GPU mode (its escaper
  tail too) and the gradient mode follow the setting; a `scheme` brings its own drift
  (nornax's `LeapfrogKDK(drift_order=...)`, summed by default).
- **Restarts.** A legacy run's fingerprint records no `drift_order`, so a file written
  before A3 continues bitwise under `drift_order="legacy"`; under the default it
  continues correctly but not bitwise (the warning names the drift order).
- **What it does not fix:** float32 velocities at much smaller steps (block-step rungs):
  at `dt` 2.5e-5 they limit dL to ~2e-7 where a float64 state keeps 1e-13.
  `jaccpot.position_precision.CompensatedKDK(velocities=True)` (opt-in scheme, a
  double-float32 state `[N, 4, 3]`) follows the float64 state there, but costs 301 B/p
  at 1e8 under `dehnen_error`: **over the 277 B/p budget**, a prototype, not a
  production setting.

## Restart files

One `.npz`, written to a temporary file and renamed (a crash keeps the previous one):

| entry | |
|---|---|
| `state`, `masses`, `self_acceleration` | `[N, 2, 3]`, `[N]`, `[N, 3]` |
| `acceleration` | with an external field: the handle's total `[N, 3]` |
| `force_scale` | under `mac_type="dehnen_error"`: f_b `[N]` |
| `meta` (JSON, UTF-8 bytes) | counters (step, time, dt, t0, segment, num_steps), the baselines, the plan (leaf capacity, validated walk caps, the level registry, the upward depth bound, the force key, the shape template's leaf shapes), the sha256 of the template's and the provider's compile keys, the field's name and parameter digest, and a fingerprint (solver configuration, settings, `JACCPOT_*` / `YGGDRAX_*` environment, XLA flags, device, versions and git commits) |

Size: 40 B per particle in float32 (52 with an external field, +4 under
`dehnen_error`) plus a few kB of `meta`: about 1.0 GB at 25M (1.3 GB with the halo).

`run(..., restart_from=path)` restores: the file's leaf capacity and walk caps are
installed, the lane is prepared (caps-only) at the restart's positions, the caps are installed
again and the level registry and depth bound set exactly, then the digests are checked.
Equal (and an equal fingerprint, `dt`, force and field parameters), the next segments
trace the programs the uninterrupted run would have, and the continuation is bitwise
that run (`tests/integration/strict/test_production_runner.py`: 2 segments, a restart,
4 more in a fresh process against 6 straight — geometric MAC, `dehnen_error`, an
external field with traced parameters, traced force parameters). Not equal, the runner
warns, records the mismatches in `RunResult.restored`, and continues: correct, not
bitwise.

## Integration schemes (multi-node plan I1)

nornax owns the integration schemes, jaccpot the force and the run (decision U1).
`ProductionRunner(scheme=...)` takes a scheme through the structural protocol
`jaccpot.scheme.SplitStepScheme`; jaccpot does not import nornax.

```python
from nornax import LeapfrogKDK                 # nornax.schemes.LeapfrogKDK
runner = ProductionRunner(solver, settings=GALAXY_SETTINGS, scheme=LeapfrogKDK())
```

A step of the segment scan is cut at its one force evaluation:

```
state_open = scheme.open_step(state, acceleration, dt)          # nornax
acceleration_new = self-gravity + external field at state_open[:, 0]   # jaccpot
state_new = scheme.close_step(state_open, acceleration, acceleration_new, dt)  # nornax
```

`state` is the packed `[N, 2, 3]` carry, `acceleration` the total acceleration at its
positions. The tree refresh, the capacity verdict, the walk/tree re-plan and segment
retry, donation, restarts and diagnostics all stay the runner's; a scheme changes only
the two expressions around the force. The carry's acceleration is the field at the
carried positions (first same as last), so the protocol hosts every scheme with one
force evaluation per step at its end: the KDK leapfrog, and later the P3T-style hybrid
Hermite (U6), whose `open_step` integrates the direct-sum subsystems. Block steps and
multi-stage schemes are not split steps.

- `scheme=None` (the default) is the built-in velocity-Verlet KDK (its drift summed
  since A3, `GalaxySettings.drift_order`).
- `LeapfrogKDK()` writes the built-in's expressions operand for operand (nornax's
  `drift_positions` / `kick_velocities`, which its own rollouts call), so it is the
  built-in, bit for bit (`tests/integration/strict/test_runner_scheme.py`: literal and
  traced parameters, an external field, the tree-order carry; nornax's
  `tests/integration/test_jaccpot_provider_rollout.py` with the real one).
- The scheme's `key()` joins the scan's compile key: its programs are its own.
- `rowwise`: the tree-order carry (`JACCPOT_STRICT_CARRY_ORDER=tree`) hands the
  scheme permuted rows; a scheme that is not rowwise keeps the input order.
- A restart file records the scheme (`meta["scheme"]`, only when there is one);
  continuing it under another scheme warns `"scheme"` in the mismatches.
- Refused: the gradient mode (it differentiates the built-in; nornax's
  `leapfrog_kdk_rollout` over `FusedProviderForce` is the differentiable route), and
  `strict_run_v2(carry="state")`.

### On the mesh (multi-node plan I2)

`ProductionRunner(..., mesh=MeshRunConfig(...), scheme=LeapfrogKDK())` runs the
scheme in the multi-GPU mode too, in both runtimes (one process driving N devices; N
processes with one device each). Each segment step calls `open_step` / `close_step`
inside the segment's `shard_map`, on ONE device's `(cap, 2, 3)` rows, padding rows
included (zero mass, zero velocity, zero acceleration), with or without the
migration to fixed cuts (the opening half sees the rows the device owned at the
step's start, the closing half the rows it owns after the migration).

- A `rowwise` scheme (the KDK) runs per shard unchanged and is the built-in mesh
  runner BITWISE: state, edges, gathers, diagnostics and restart files by id, with
  repartitions, migration, segment retries, chained runs and restarts, on the direct
  and the fused force, both MACs (`tests/distributed/test_production_runner_mesh_scheme.py`,
  `..._scheme_fused.py`; `test_multiprocess_runtime.py`'s scheme cases: 2 and 4
  processes == one process).
- A scheme whose step reads other rows (a centre of mass, a global timestep) is not
  rowwise and must declare a MESH FORM: `jaccpot.scheme.ShardedSplitStepScheme`,
  i.e. `open_step_shard(state, acceleration, dt, *, live_rows, axis_name)` and
  `close_step_shard(state_open, acceleration, acceleration_new, dt, *, live_rows,
  axis_name)`, reducing with collectives whose result does not depend on the
  transport's reduction order (`nornax.sharded.mesh_sum`: an `all_gather` folded in
  mesh order). A scheme with a mesh form uses it whether or not it is rowwise. A
  non-rowwise scheme without one is refused when the runner is built
  (`jaccpot.scheme.check_mesh_scheme`); on one card it runs as before.
- `scheme=None` keeps the mesh programs: their lowered text is main's (the
  segment, start, potential and repartition programs, migrating or not, direct and
  fused force, both MACs; source locations aside).
- A scheme together with `escapers=` is refused on the mesh: the escaper tail's rows
  are stepped by the built-in KDK (routing them through the scheme is a follow-up).
  One card takes both.
- The mesh restart file records the scheme as the single card's does; a restore
  under another scheme lists `"scheme"` in the mismatches and is not bitwise.
- The force behind nornax's protocol: `jaccpot.distributed.sharded_force.ShardedMeshForce`
  wraps a prepared mesh force (`FusedMeshForce`, `DirectMeshForce`) as a
  `nornax.sharded.ShardedForceModel` (per-device carry: the overflow bits, the walk
  needs and, under `dehnen_error`, `f_b`). The runner does not use it (its programs
  call the force directly); nornax's drivers and its sharded conformance kit do.

## Gradient mode (S9 session 2)

A gradient of a production rollout, segments, re-plans and all, in two passes
(`jaccpot/production_grad.py`):

```python
plan = runner.gradient_plan(state, masses, dt=dt, num_steps=T,
                            external_params=halo)   # 1. the forward pre-pass
res = plan.value_and_grad(lambda out, inputs: loss(out.positions, out.velocities),
                          wrt=("positions", "velocities", "masses", "external_params"))
res.value, res.grads["positions"], res.result        # 2. the gradient run
# or both at once: runner.value_and_grad(loss, state, masses, dt=..., num_steps=..., wrt=...)
```

1. **The pre-pass is `run()` itself** -- its segments, segment retries and re-plans,
   callbacks and restart files -- with a recorder that keeps, per segment, the plan
   its scan traced with (leaf capacity, walk caps, level registry, depth bound), its
   provider, and how its start acceleration and first f_b were made (`strict_run_v2`
   records the successful attempt's start, host-side, only while the recorder is on).
   Each segment's provider gets the gradient plan's statics frozen on a rebuild at its
   start. Cost: one host copy of the state per segment, one rebuild program per plan.
   `gradient_plan(..., freeze_cache={})` (opt-in, S10) keeps that program across the
   pre-passes of one runner, by compile key: a caller that re-plans often (multiple
   shooting, a plan per segment) compiles it once (a warm 2-step pre-pass 32 -> 5 s,
   CPU interpret, N = 512, `dehnen_error`).
2. **The gradient run is one jitted program**: a chain of `differentiable_rollout`
   calls over each segment's `differentiable_step`, each segment traced under its own
   plan. Only statics change between segments, so the chain is one pure function of
   `(positions, velocities, masses, dt, force_params, external_params)`.
   `GradientPlan.value_and_grad` compiles it once per `(loss, wrt, checkpoint)` and
   refuses (`RolloutVerdictError`) a run whose verdict failed -- e.g. inputs moved so
   far from the pre-pass's that its caps no longer hold: make a new plan there.

**The boundaries are the runner's.** A segment that was not retried continues the
previous one's total acceleration and f_b; a retried one starts the way
`strict_run_v2`'s retry started it (the start force from the re-prepare's seed, the
same seed for the first step; with a field, the previous segment's last self-gravity
plus the field). So the gradient run's forward is the runner's trajectory, bitwise,
re-planned segments included (CPU, N = 512, float64). One exception: the runner adds
the field to a start force EAGERLY, op by op, and XLA contracts multiply-adds inside a
jitted field, so a field with more than one multiply (the NFW halo) moves a few
elements by an ulp at step 0 and after a retry with a field (9-26 of 3072, 5.6e-17).

**From a restart.** `gradient_plan(restart_from=path)` (or a `RunResult`) starts the
gradient run at the restart's state. The restored start acceleration is data, so the
chain keeps its VALUE (the forward stays the runner's) and differentiates the force
RECOMPUTED at the restart positions in its place: the gradient with respect to the
restart state is the one the uninterrupted run carries across that edge -- to
round-off with the geometric MAC (1.8e-16 of the largest component, CPU, N = 512,
float64); under `dehnen_error` to 3e-8 / 6e-8 (positions / masses), because a restart
file holds the f_b of the restart state's own force, not the one its thresholds came
from, so the recompute linearises the edge's force on neighbouring lists.

**Opt-ins for a caller with many plans (S10 follow-ups; defaults unchanged).**
- `plan.rollout(inputs, start_force="adjoint")` (also `value_and_grad(...,
  start_force=)`): a segment that starts from a seed (`"start"`, `"seed_kept"`,
  `"eager_self"` -- every multiple-shooting segment) hands its start force to the
  rollout, whose reverse runs it as unit 0 of its one reverse loop, instead of a
  `jax.checkpoint` beside it. W at the S9 2e6 Plummer calibration 883 -> 406 B/p; the
  value and the gradients are the default's (CPU: bitwise but the masses' gradient at
  1.2e-16). The start total's field is then added inside the program: with a field of
  more than one multiply step 0 may move by an ulp off the runner's eager add.
- `plan.rollout(inputs, arrays=plan.arrays(device))`: the plan's array data (the
  array leaves of `dt` / the force parameters, every segment's f_b seed, a restart's
  restored handle) as program ARGUMENTS instead of constants copied into the program.
- `gradient_plan(..., freeze_cache={})` also keeps the eager prepare's per-call
  programs on the engine (`jaccpot/runtime/eager_cache.py`): a warm pre-pass compiles
  nothing (it compiled 11 programs, ~1.5 s, at 1e5).
- `GalaxySettings(cell_min_level="auto")` / `TreeConfig(cell_min_level="auto")`: the
  level floor picked at the first eager prepare -- the level <= 8 whose occupied
  cells hold, on average, nearest a leaf's worth of particles; 8 at production N.

Single card only. Checkpointing per segment as `differentiable_rollout`'s
(`checkpoint="step"` by default; memory: `docs/production_2026-10.md`, "S9").

**Tests.** `tests/integration/strict/test_runner_gradient_mode.py`: CI runs its G1 (the
gradient run's forward against the runner). Its gradient gates -- finite and A-vs-A, FD,
restart, refusal -- are manual: one gradient compile of the three-segment chain takes
19-21 GB of RSS on CPU, more than a hosted runner holds. Run them on a large host after
changing `production_grad.py` or the rollout's reverse:
`JACCPOT_RUN_MANUAL_TESTS=1 pytest -n 2 --dist loadgroup tests/integration/strict/test_runner_gradient_mode.py`.

## Multi-GPU mode

`ProductionRunner(solver, ..., mesh=...)` runs the same loop -- segments, edges,
callbacks, the diagnostics with the energy, restart files and their continuation, the
segment retry -- on the fused multi-GPU lane: one process drives every card of a 1-D
`jax.sharding.Mesh`, and each segment is ONE `jit(shard_map(scan))` program of
`MeshSegmentRunner` (`jaccpot/distributed/segment.py`; the mode's backend is
`jaccpot/distributed/mesh_runner.py`). `mesh=None`, the default, is the single-card runner
above, unchanged (its programs are byte-identical to main's: `bench/provider_hlo_identity.py`,
36/36 on CPU).

```python
import jax
from yggdrax.distributed.sharding import make_mesh

jax.config.update("jax_enable_x64", True)

from jaccpot import FastMultipoleMethod, ProductionRunner, SegmentConfig
from jaccpot.production import DiagnosticsLog, MeshRunConfig, SnapshotWriter

solver = FastMultipoleMethod(preset="galaxy", softening=1.5e-3)
runner = ProductionRunner(
    solver,
    segment=SegmentConfig(steps=50, restart_every=4, restart_path="run/restart.npz"),
    external_acceleration_fn=halo,          # the Quick start's, unchanged
    external_potential_fn=halo_potential,
    callbacks=(SnapshotWriter("run/snapshots", every=2), DiagnosticsLog("run/diag.jsonl")),
    mesh=MeshRunConfig(make_mesh(4)),       # or mesh=make_mesh(4)
)
res = runner.run(state, masses, dt=1e-3, num_steps=500,
                 external_params={"gm": 1.0, "rs": 2.0})
res.state      # [N, 2, 3] on the host, in input (id) order
res = runner.run(dt=1e-3, num_steps=500, restart_from="run/restart.npz",
                 external_params={"gm": 1.0, "rs": 2.0})
```

**Launching.** One process, every card of the mesh visible to it. On this machine pick the
cards with autocvd (the org rule), then set the flags BEFORE jax starts -- XLA reads
`XLA_FLAGS` once:

```bash
eval $(/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd -q -l -n 4 -e -t 120)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_experimental_enable_fusion_autotuner=false \
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false \
--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true \
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
```

The first flag is the single card's (Launch environment: a resumed run is bitwise only
with it). The other three are `jaccpot.distributed.fused.RAGGED_EXCHANGE_XLA_FLAGS`, XLA's
NCCL send/recv path for the cross-domain exchange: XLA's default one-shot kernel takes
4-6x its time across PCIe, jax 0.11 refuses the path without the fallback flag, and with
jax 0.11's NCCL barrier on, the exchange delivers no cross-device rows and the next call
deadlocks (how the fused two-card lane hung on 0.11.2). The force warns once when they are
missing. Check after an `autocvd` timeout that `jax.devices()` are GPUs: the
timeout leaves `CUDA_VISIBLE_DEVICES` empty and the run lands on the CPU.

**`MeshRunConfig`** (`jaccpot.production`, frozen):

| field | default | |
|---|---|---|
| `mesh` | -- | the 1-D mesh |
| `cap` | `None` | rows per device; `None`: `mesh_shard_capacity(N, ndev, cap_headroom)` = `ceil(ceil(N / ndev) x 1.15)` (the S11/S12 gates' headroom; the repartition's balance overshoot is about `2 ndev / num_samples` of a shard) |
| `repartition_every` | 1 | a repartition before every segment after the first (the S12 gate's cadence); 0: only the initial decomposition. Its own program, between segments; a declined one (a device would overflow `cap`) moves nothing and the edge says so |
| `num_samples` | 256 | pivot samples per device of the decomposition and the repartitions, each weighted by its device's live count: a repartition cuts within about `N / ndev / num_samples` rows of even, and the imbalance does not accumulate |
| `cross_caps` | `"auto"` | the cross-domain capacities: sized in the library from the needs measured at the start state and at the first segment's drift (on small buffers, `CrossCapacities.for_measurement`; 1.5x the needs), or a `CrossCapacities` |
| `migrate` | `True` | after every drift, hand each particle that left its device's Morton range to the owner before the force, keys and cuts in the force's own (cubic) box (`MeshSegmentRunner(migrate=True)`, `jaccpot/distributed/migration.py`): the domains stay exact, so no device holds a sparse leaf inside another's core. At 8e6 on two A100s 4-5x fewer cross near pairs and ~11x fewer particles shipped per step, the force 219-262 -> 51-54 ms on contended cards; the repartitions then only rebalance (even cuts plus a migration). The default since S11 session 2's free-card A/B (`production_2026-10.md`: never slower beyond the repetitions' spread on two and four cards, both MACs, 1.2-3.8x faster on Plummer and large discs, +1-6 % peak); `False` lets the domains drift between repartitions |
| `migration_cap` | `"auto"` | `migrate=True`: the migration buffer's rows per device, sized from one step's drift, re-planned exactly when it overflows |
| `force` | `None` | the fused lane's `FusedMeshForce` on `solver` (`settings.leaf_size`, `settings.max_order`, the solver's theta); a force object of the segment runner's protocol (`DirectMeshForce` in tests) is used as given |

**What is the same and what is not, against one card:**

- **Input and output by id.** `run` decomposes the `[N, 2, 3]` state over the mesh
  (`rollout.decompose`: Morton domains, ids `0..N-1` in input order); the caller's arrays
  are copied, never consumed (`donate_input` has no effect here). `RunResult.state` and
  `.masses` are the final arrays gathered by id ON THE HOST (one device-to-host copy at
  the end; 24 B/particle in float32); `RunResult.handle` is the `MeshSegmentRunner`, so
  `restart_from=res` continues in memory; `RunResult.mode` is `"mesh"`, and each mode
  continues only its own results and files.
- **Edges** are `MeshSegmentEdge`s: `state` and `masses` are the layout's flat
  `(ndev * cap, ...)` arrays (rows by device, padding included), `self_acceleration` is
  `None` (the segments carry the TOTAL acceleration, `layout.acceleration`), and nothing is
  gathered unless asked: `edge.state_host()` (what `SnapshotWriter` calls) and
  `edge.gather(names)` copy to the host by id, during the callback. `edge.retries` is the
  segment runner's own retry count; `edge.plan` holds the leaf capacity, walk caps, the
  merged level plan and its floor, the depth bound, the cross capacities (and what their
  sizing measured), `mesh` (`ndev`, `cap`, live `counts`), `repartition` (whether one ran
  before the segment, declined, rows sent off device) and the kept run's overflow `bits`.
- **Parameters.** `dt` and the external field's parameters are program arguments (one
  compile for every value). `G` and the softening are the force's own: the fused mesh
  force is static in them, as the single card under `dehnen_error`, so `force_params` must
  be `None` or equal them (otherwise a `ValueError` says so). The external field and its
  potential see ONE device's `(cap, 2, 3)` rows: the single card's `field(state, params)`
  contract restricted to rows, so a per-particle field works unchanged. There is no
  `step_callback`.
- **Diagnostics.** The moments are the single card's fp64 reductions, over the flat sharded
  arrays: a padding row has zero mass and zero velocity, so each moment is exact without a
  mask (gated against the host fp64 moments of the gathered state). The energy is `K +
  W_self + W_ext` from `MeshSegmentRunner.potential_energy()`: the force's `potential_shard`
  per device (`phi_i = -G sum_{j != i} m_j psi(r_ij)` over every device's particles) and
  the external potential per row, summed in fp64 per device and over the mesh, in a program
  of its own compiled once per plan whose trace leaves the engine's trace-time state as it
  found it -- the trajectory with the energy is the one without, bitwise. The mesh force is
  prepared before the first segment, so `E0` is evaluated AT step 0 (the step-0 row has its
  energy, drift 0), not at the first segment as on one card. A potential that overflows its
  capacities reports `energy_ok = False` and no energy (at step 0 that leaves `E0` and every
  drift `None`: the energy program shares the force's capacities, which a first segment's
  retry may still have to grow). The fused mesh force's `potential_shard` is the
  template route's rebuild, `FusedForceProvider.potential` (local near, local far and the
  cross far pairs through the locals) and the cross near pairs' potential lane. A force
  without `potential_shard` reports every energy field as `None`, `energy_unavailable`
  says why, and the run warns once.
- **Momentum across devices (M1, #173).** The mesh conserves momentum as well as one card,
  with no reaction sent back. Each cross pair is evaluated on BOTH of its devices: every
  device walks the gathered summary trees from the same seed, so both ends get the same pair
  set (`cross._symmetric_walk`). Each device then applies its own half from what the other
  shipped: the peer's multipole through M2L into its locals, and the peer's particles
  through the cross near term. Both ends read bitwise-identical multipoles, centres and
  particles. One card does the same with its own pairs (each canonical pair once per
  direction, `_directed_csr_from_canonical`), so a cross pair cancels in `sum m a` as well
  as a local pair does. Measured per force, as `|sum m a| / sum m |a|`, with the cross part
  alone carrying about half of `sum m |a|`:
  - fp64, forced CPU devices: ~1e-17 total and ~1e-16 for the cross part, for both MACs
    (`tests/distributed/test_mesh_cross_momentum.py` pins it; dropping the cross far pairs
    on one device gives 0.15);
  - fp32 on two A100s: ~4e-9 for each part, against 1.1e-8 for one card.

  So `momentum_drift` on N devices is in the one-card class (`docs/production_2026-11.md`,
  `## M1`).
- **Retry.** The segment runner's: a fired flag on any device re-plans what the segment
  outgrew (the walk caps, the tree's leaf capacity, depth and level plan, the cross
  capacities), re-prepares every device at the segment's start from a host copy of the
  donated start state and re-runs it from a start force -- at most twice, then while each
  re-run fits more steps. A segment that cannot be re-planned raises a `RuntimeError`
  naming the segment, its steps and the last restart file.

**Restart files** (`SegmentConfig.restart_every`, `save_restart`): one `.npz` as on one
card, with the arrays BY ID -- `state`, `masses`, `acceleration` (the total) and under
`dehnen_error` `force_scale` -- plus the layout, `mesh_ids` (each row's id, `-1` on padding)
and `mesh_padding` (the padding rows' positions: they stay where the last repartition put
them; every other padding value is zero), and in `meta` the single card's counters,
baselines, field record and fingerprint, `mesh` (`ndev`, `cap`, counts, the force's class)
and the edge's plan record with the template's leaf shapes and the digest of the force's
compile key. Size: 40 B/particle in float32 (+4 under `dehnen_error`) plus 4 B per row and
12 B per padding row.

- **On the mesh that wrote it** (the same device count and `cap`): the restore rebuilds the
  identical layout, installs the run's leaf capacity, walk caps and depth bound, builds the
  force with the run's cross capacities and level floor, prepares it at the restart's
  positions, puts back the run's level plan, walk caps and template shapes, and checks the
  force key's digest. Equal, the next segments trace the run's programs and the
  continuation is the uninterrupted run, BITWISE (`restored["bitwise"]`; CPU gates for the
  direct and the fused force, the latter from a file written after a retried segment, in a
  fresh process). On GPUs the cross-domain path is bitwise run to run since S11 session 2
  (its pair lists in (target, source) order, its long rows summed in a fixed order), and a
  continuation's STATE is the uninterrupted run's, bitwise (two A100s, 24 steps); since
  the single card's near kernel sums the pieces of very long rows in a fixed order too
  (below), so is the carried acceleration.
- **On another mesh** (another device count or `cap`): the by-id state is decomposed there,
  its carried acceleration and f_b placed by id (no start force), and the force prepared
  and sized afresh: the same run to round-off, `restored["bitwise"] = False`,
  `"mesh_layout"` among the mismatches.

### Sharded checkpoints (multi-node plan M3)

Under several processes the `.npz` above would pass the whole state through one host, so
the multi-GPU mode writes a **sharded checkpoint** instead: `MeshRunConfig(checkpoint=...)`
is `"auto"` (the default: sharded under several processes, the `.npz` in one), `"orbax"`
(always sharded) or `"npz"` (always the file; refused under several processes). orbax is
an optional dependency: `pip install 'jaccpot[io]'` (orbax-checkpoint 0.12.9 is the version
gated). Importing jaccpot and every default path never import it; using a checkpoint
without it is an `ImportError` that names the extra.

**Writing.** `SegmentConfig(restart_every=k, restart_path="run/ck_{step:08d}")` (or
`runner.save_restart(path)` at an edge) writes one DIRECTORY per checkpoint, every process
calling it (the runner's cadence does). Inside:

| item | what |
|---|---|
| `arrays/` (orbax, OCDBT) | the layout's arrays as they live on the mesh: `state` `(ndev*cap, 2, 3)`, `masses`, `ids` (`-1` on padding), `count` `(ndev,)`, `acceleration` (the total), `force_scale` (under `dehnen_error`); with escapers the tail's `tail_state`, `tail_masses`, `tail_ids`, `tail_count`, `tail_acceleration`. Each process writes ITS OWN shards (`arrays/ocdbt.process_<p>/`): nothing passes through one host and nothing is copied on the device for the write; host memory per process is one shard at a time |
| `meta/metadata` (JSON, process 0) | what the `.npz`'s `meta` holds (counters, baselines, the plan record, the template's shapes, the force key's digest, the field record, the fingerprint, `mesh`, the escaper split and ledger, the scheme) plus `checkpoint` (processes, N) and the **manifest**: per array its shape and dtype, the sha256 of every shard's bytes (by first row), and for the per-particle arrays a set hash over the live rows keyed by id (`jaccpot.checkpoint.set_hash`: order- and split-independent) |

The commit is atomic (orbax writes a temporary directory and renames it after every
process finished): a job killed mid-write leaves no checkpoint at the new path and every
earlier one intact. Writing over an existing path goes through `<path>.next` (committed
first), `path -> <path>.prev`, `<path>.next -> path`, `<path>.prev` removed; a crash between
the renames leaves a committed `.next` or `.prev` that every reader finds
(`jaccpot.checkpoint.resolve`). Size: the layout's rows, not N: ~48 B per row in float32
under `dehnen_error` (state 24, mass 4, id 4, acceleration 12, f_b 4) at `cap_headroom`
1.15, before orbax's compression.

**Restoring** (`run(restart_from=<directory>)`, in a fresh process, one or many):

- **The same device count and `cap`** (any process count: N processes x 1 device or 1 x
  N): orbax reads each process's own shards back onto the run's sharding, every shard's
  sha256 is checked, and the plan is restored exactly as from a `.npz`: the continuation is
  the uninterrupted run, BITWISE (`restored["layout"] == "checkpoint"`). The launch details
  (`JACCPOT_MP_*`, XLA's forced CPU device count) are not part of the comparison, so a
  checkpoint written by 2 processes continues bitwise in 1 process with 2 devices.
- **Another device or process count (elastic)**: orbax restores the written rows onto THIS
  mesh in equal blocks (a re-sharding restore: each process reads only its blocks), each
  process compacts its live rows on the host (`O(N/P)`), the set hashes are checked (summed
  over the processes), and one repartition program (the decomposition's) makes the new
  Morton domains, the carried acceleration and f_b travelling with their particles (no
  start force); the escaper tail is read whole and split evenly. Correct, not bitwise:
  `restored["layout"] == "decomposed"`, `"mesh_layout"` among the mismatches. A block whose
  rows would overflow a device's `cap` is an error naming `MeshRunConfig(cap=...)`.
- **One card**: `ProductionRunner(solver)` (no `mesh=`) continues a mesh checkpoint by id
  (every shard read and checked on the one host); the carried total is the start force
  when the run had no external field (with one, the single card evaluates its start
  force); an escaper split continues with the same membership. The plan is the single
  card's own prepare: correct, not bitwise.
- **From one card**: the mesh mode continues a single card's `.npz` by id (every process
  reads the file -- a single card's N fits one host -- and decomposes its own id range),
  the carried total and f_b placed by id; with escapers, the file's membership.

A checkpoint written and read under a manifest mismatch (a shard that is not the one the
manifest hashed, a corrupted file) raises `jaccpot.checkpoint.CheckpointCorruptError` (or
tensorstore's decoder error first, for flipped compressed bytes) instead of continuing.

**Snapshots** that never pass through one host: `SnapshotWriter("run/snaps", every=2,
sharded=True)` writes `run/snaps/snapshot_<step:08d>/` in the same format (state, masses,
ids and counts, the escaper tail; `meta` holds `kind="snapshot"`, step and time). Read one
back on one host by id with `jaccpot.checkpoint.load_checkpoint(path, ["state"])`.

**Job chaining** (SLURM walltime): `jaccpot.checkpoint.latest_checkpoint("run/ck_{step:08d}")`
is the newest COMMITTED checkpoint of a restart-path pattern (a half-written one is
skipped). `bench/slurm/m3/chain.sbatch` is the pattern: each job first submits its
successor (`afterany` on itself), continues from the latest checkpoint, and stops itself
with one more checkpoint before its deadline (a decision every process takes together:
`bench/m3_checkpoint_io.py chain`); the successor resumes or, after `DONE`, exits.

**Diagnostics** stay `DiagnosticsLog`'s: one JSON line per edge, written by process 0
after the edge's collectives (every process keeps the rows); a run continued from a
checkpoint appends its edges to the same file, which then equals the run in one go's
(wall times aside).

**Gates** (`tests/distributed/test_checkpoint_*.py`, job `test-distributed-checkpoint`, the
CI shard that installs the `io` extra): the numbers are in `docs/production_2026-11.md`
(`## M3`).

**What is bitwise, and what is not:**

| | |
|---|---|
| the mode == `MeshSegmentRunner` driven by hand (same segments, cadence, force) | bitwise by id (CPU: direct force, and the fused force with a retried segment) |
| chained runs (`restart_from=RunResult`) == one run | bitwise |
| a restart file continued in a fresh process on the same mesh | bitwise, digests matching (CPU: direct and fused force); on two A100s the state bitwise, the carried acceleration in 1-9 of 6M components not (below) |
| the energy diagnostic on or off | bitwise |
| a restart on another device count or `cap` | the same run to round-off; `bitwise = False` |
| a retried segment against one that never failed | to round-off: the retry recomputes the start force the other carried |
| the mesh against one card on the same particles | not bitwise (other trees and lists); the accuracy is in class (`production_2026-10.md`, S11 and S12) |

**On two A100s, before the cross path's order fix** (`bench/multigpu_runner_gate.py`, 2026-10-23, cards 0 + 7, both shared with
other users' jobs at 90-100 % utilisation: memory and accuracy are results, times are not).
Plummer, N = 2e6 (1e6 per card), `preset="galaxy"` (`dehnen_error` 1e-4, leaf 64, p6), the
library's cross capacities and shard capacity (1,150,000 rows), 3 segments of 8 steps at dt
5e-3, a repartition before segments 2 and 3 (17,476 and 17,447 rows moved), every diagnostic
at every edge, a restart file after each segment:

| | |
|---|---|
| flags | segment 1 retried once (S11's first-segment retry: `OVERFLOW_CROSS` at its 8th step, 7 of 8 fitted), the kept runs clean; segments 2 and 3 no retry |
| rel-L2 of the carried acceleration against the fp64 direct sum (512 ids) | 1.793e-4 at step 8, 1.638e-4 at step 24 (S12's segment gate: 1.324e-4 / 1.636e-4; one card 1.767e-4 / 1.672e-4) |
| peak per device | 616-619 B/particle (616,157,696 and 619,280,128 B); 850-854 with `--xla_gpu_deterministic_ops=true` |
| energy | `None`, `energy_unavailable` (this run predates the fused mesh force's `potential_shard`) |
| restart after segment 2, continued in a fresh process | digest matched, `layout="file"`; the final state differs from the straight run's in 165,799 of 12,000,000 elements (max 4.8e-7) |
| control: a second straight run (A-vs-A) | 412,084 of 12,000,000 differ (max 4.2e-5) |
| control: a second continuation from the same file | 166,835 of 12,000,000 differ from the first (max 2.4e-7) |
| the same with `--xla_gpu_deterministic_ops=true` | continuation vs straight 169,580, continuation vs continuation 168,667, straight vs straight 412,218 (`restored["bitwise"]` True, digests equal) |

The continuation is as close to the uninterrupted run as a second continuation is to the
first: the restore reproduces the run's state and programs (the digests), and the
difference is the lane's own run-to-run spread. Where it comes from: in ONE process, the
start force evaluated three times on the same state with the same compiled program is
bitwise on a one-device mesh but differs in 612,657 of 1,380,000 components (max 9.5e-7) on
two devices (Plummer 4e5, production flags; geometric MAC, `dehnen_error` 513,742, max 3.4e-7):
the cross-domain path is
not deterministic at run time on GPUs, and deterministic ops do not fix it.

**Fixed for the cross path** (S11 session 2, track G; `bench/multigpu_force_repeat.py`): the
symmetric Pallas cross walk emits the same pairs in another order each run (atomic cursors)
and the merge kept that order inside each CSR row; the pairs are now sorted by (target,
source) and the cross near kernel's chunks and long-row pieces summed in a fixed order. The
start force is bitwise run to run and across processes (Plummer 4e5 and 2e6, both MACs), and
the restart above continues with the state bitwise (0 of 12,000,000) and the carried
acceleration in 1-9 of 6,000,000 components (max 3.0e-8). What was left -- and on the disc
+ bulge 2e6 some 8.5k of 6.9M start-force components -- came from outside the cross terms:
the single card's direct near kernel scatter-added the pieces of rows past 2 x 1024 entries
in atomic order. It now sums each row's pieces in one fixed order and adds that once, as the
cross term does (2026-10-30): the single-card start force on the disc + bulge 2e6
(`dehnen_error`, `bench/multigpu_force_repeat.py --ndev 1 --ic disc`) went from ~28k of 6.9M
acceleration and ~7k of 2.3M f_b components differing run to run to bitwise, within one
process and across processes. Rows up to 2 x 1024 entries keep their bits. The two-GPU start force on the same
disc (`--ndev 2`) is bitwise too (0 of 6.9M acceleration and 0 of 2.3M f_b components, in process and across
processes, 2026-11-01; it had ~8.5k differing before).

**Measuring the lane on a dedicated node.** The A/B and scaling rows in
`production_2026-10.md` (S11 session 2, track T) ran on four free cards of a shared
machine whose cards talk over PCIe only; there the mesh scales to two cards in time and
beyond that buys capacity. `bench/multigpu_scaling_matrix.py` repeats the matrix --
migration on / off on two and four cards, strong scaling on 1 / 2 / 4 cards (one card:
the single-card lane on the same particles), weak scaling at 4e6 per card, the memory
ramp to 4e7 per card -- one fresh process per row on subsets of the job's cards, with
the XLA flags set, `nvidia-smi` before and after every row and a `SUMMARY.md`;
`bench/slurm/multigpu_scaling.sbatch` runs it on a HoreKa accelerated node (four
A100-40GB on NVLink). It needs Odisseo's `benchmark_multigpu/` (`BENCH_DIR`) and the
25M disc + bulge IC (`--disc-ic`); a resubmitted job skips the rows it has.

## Gates (S3)

| gate | what | result |
|---|---|---|
| R1 | 3 x 2 segments vs one call of 6; 2 + 2 + 1 vs 5 (CPU, N = 512) | bitwise |
| R1p | `bench/dce_pins.py` S1p's calls through the runner (A100, 2e5, deterministic ops) | bitwise S1p (force and state) |
| R2 | an NFW field across edges, one-argument and traced parameters (CPU, pure-JAX near field) | bitwise with the carried total; without it 1 of 3072 components differs (1.9e-6) |
| R1x | S1x's disc + NFW through the runner, chained vs one segment of 13 (A100, 2e5) | bitwise (and bitwise the plain chained S1x: on this card the eager field matched the fused one) |
| R3 | runner (traced) vs `strict_run_v2` (traced), CPU | bitwise |
| R4 | diagnostics on vs off, CPU; with the energy at every edge (S3 session 2), and with an NFW field and its potential, traced parameters | bitwise, the same scan programs; `W_self` within 1e-3 of the fp64 direct sum, `W_ext` within 1e-6 |
| R5 | restart in a fresh process: geometric, `dehnen_error`, external (traced parameters), traced (CPU) | bitwise, digests matched |
| R1r | R1p with a restart at step 5, resumed in a fresh process without the eval prepare (A100, 2e5) | bitwise, digests matched (file 8,005,201 B) |
| R9 | `donate_input` on vs off, CPU | bitwise; one scan program with it, two without |
| DET3 | R1p on the production flags (no deterministic ops), recorded twice on one card | bitwise |

`bench/fused_memory_budget.py --runner` runs the bench's scan phase through the runner
(the same calls as continuing runs, one segment each), so its timings include the
edges; `--runner-once`, `--runner-restart`, `--runner-resume` are the pins' modes.

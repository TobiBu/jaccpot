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

`theta` decides no pair under `dehnen_error`: the walk accepts by eq (16a) and its
convergence clause `rho_A + rho_B < mac_theta_max r` (`mac_theta_max` 1.0), and the first
step's f_b seed walks at theta 0.8 whatever the solver's (`mac_force_scale_prepass_theta`
sets it). Theta 0.3, 0.5 and 0.8 gave the same lists and the same states, bit for bit
(CPU, N = 2048, 2 steps); it is still part of the compile keys. A tighter target is
`FastMultipoleMethod(preset="galaxy", adaptive_eps=1e-5)`.

**The geometric MAC back** (the preset before 2026-10-21, the same engine:
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
| `energy`, `energy_drift`, `energy_ok` | `E = K + W`; `(E - E0) / |E0|`; whether the potential's rebuild fitted its capacities (otherwise the energy fields are `None`) |

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
| `num_samples` | 256 | pivot samples per device of the decomposition and the repartitions |
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
  continuation's STATE is the uninterrupted run's, bitwise (two A100s, 24 steps); the
  carried acceleration can still differ in a few components where the single card's own
  near kernel scatter-adds the pieces of very long rows (below).
- **On another mesh** (another device count or `cap`): the by-id state is decomposed there,
  its carried acceleration and f_b placed by id (no start force), and the force prepared
  and sized afresh: the same run to round-off, `restored["bitwise"] = False`,
  `"mesh_layout"` among the mismatches.

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
acceleration in 1-9 of 6,000,000 components (max 3.0e-8). What is left -- and on the disc +
bulge 2e6 some 8.5k of 6.9M start-force components -- comes from outside the cross terms,
most likely the single card's direct near kernel, which scatter-adds the pieces of rows past
2 x 1024 entries in atomic order (`JACCPOT_NEARFIELD_DIRECT_ROW_LIMIT=1e8` made the repeats
bitwise): a change of the single-card default path, not made.

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

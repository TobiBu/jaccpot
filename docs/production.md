# Production runs on the fused lane — `ProductionRunner`

`jaccpot.ProductionRunner` runs a long simulation on the fused single-GPU lane: a host
loop of K-step segments, each ONE `strict_run_v2(carry="particles")` call continuing
from the previous call's `StrictParticleCarry` handle. Between segments it computes the
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

runner = ProductionRunner(
    solver,
    segment=SegmentConfig(steps=50, restart_every=4, restart_path="run/restart.npz"),
    external_acceleration_fn=halo,
    callbacks=(SnapshotWriter("run/snapshots", every=2), DiagnosticsLog("run/diag.jsonl")),
)
res = runner.run(state, masses, dt=5e-4, num_steps=500,
                 external_params={"gm": 1.0, "rs": 2.0})
# killed? continue in a fresh process, same script:
res = runner.run(dt=5e-4, num_steps=500, restart_from="run/restart.npz",
                 external_params={"gm": 1.0, "rs": 2.0})
```

`state` is the packed `[N, 2, 3]` positions and velocities, `masses` `[N]`. `num_steps` is
the run's total from `t0`: a continued run runs the steps still missing. The one-shot
form is `jaccpot.production.run(solver, state, masses, dt=..., num_steps=..., ...)`.

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
| `traced_params` | `True` | `dt`, `G`, the softening squared (not under `mac_type="dehnen_error"`) and the field's float parameters as arrays: arguments of the compiled programs, so other values compile nothing. `strict_run_v2` itself keeps literals by default. |
| `donate_input` | `False` | the first segment consumes the caller's state; later segments always consume their (the runner's) input |
| `diagnostics_every` | 1 | moments at step 0, every n-th edge and the last; 0 = off |
| `diagnostics` | all | of `"momentum"`, `"angular_momentum"`, `"com"`, `"kinetic"`, `"energy"` |
| `diagnostics_every_energy` | 10 | edges between energy diagnostics |
| `restart_every`, `restart_path` | 0, `None` | a restart file every n segments; `{step}` / `{segment}` are formatted into the path, otherwise each restart replaces the last |

**Segment lengths.** `num_steps` is part of a scan's compile key: every distinct segment
length compiles its own scan (a ragged 14-step last segment recompiled for 52.8 s at 25M
on an A100). The plan is `num_steps // steps` segments, then the remainder, last
(`segment_lengths`); the runner warns once when there is one.

**Compiles.** A fresh run compiles the eager prepare's programs, the start force and one
scan. With `donate_input=False` the second segment compiles the scan once more (with
the state donated) and every later segment reuses it; `donate_input=True` gives every
segment one program. Chained `strict_run_v2` calls without `carry_total_acceleration`
also recompiled at the second call (the handle's start force was not donated: 17.7 s
against 2.4 s at N = 512 on CPU).

**Retry.** A segment whose lists overflow is re-planned and re-run once by
`strict_run_v2` (`FusedForceProvider.replanned`); a second failure raises a
`RuntimeError` naming the last restart file. A donated segment keeps a host copy of its
start state for that re-run (24 B per particle, one device-to-host copy per segment).

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
| `potential`, `energy`, `energy_drift` | **`None` until S3 session 2**: the energy needs an O(N) potential the fused lane does not have yet (the generic potential path ran out of memory at 25M) |

An external field changes P and L physically; the drifts then measure that.

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
installed, the lane is prepared at the restart's positions, the caps are installed
again and the level registry and depth bound set exactly, then the digests are checked.
Equal (and an equal fingerprint, `dt`, force and field parameters), the next segments
trace the programs the uninterrupted run would have, and the continuation is bitwise
that run (`tests/integration/strict/test_production_runner.py`: 2 segments, a restart,
4 more in a fresh process against 6 straight — geometric MAC, `dehnen_error`, an
external field with traced parameters, traced force parameters). Not equal, the runner
warns, records the mismatches in `RunResult.restored`, and continues: correct, not
bitwise.

## Gates (S3)

| gate | what | result |
|---|---|---|
| R1 | 3 x 2 segments vs one call of 6; 2 + 2 + 1 vs 5 (CPU, N = 512) | bitwise |
| R1p | `bench/dce_pins.py` S1p's calls through the runner (A100, 2e5) | see the S3 record |
| R2 | an external field across edges (CPU; GPU pin R1x) | bitwise with the carried total |
| R3 | runner (traced) vs `strict_run_v2` (traced) | bitwise |
| R4 | diagnostics on vs off | bitwise |
| R5 | restart in a fresh process (CPU x 4 variants; GPU pin R1r) | bitwise |
| R9 | `donate_input` on vs off | bitwise |

`bench/fused_memory_budget.py --runner` runs the bench's scan phase through the runner
(the same calls as continuing runs, one segment each), so its timings include the
edges; `--runner-once`, `--runner-restart`, `--runner-resume` are the pins' modes.

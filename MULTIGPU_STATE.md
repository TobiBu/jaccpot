# Multi-GPU state: what exists, what is missing, and how distributed gradients work

_Step 0 of the §5 + distributed-differentiability tranche. Established by reading
the code and docs on `paper/jaccpot-i` at the post-merge commit, not from any
prior summary. 2026-08-17._

**Bottom line.** The two halves of this tranche are in opposite states.
Distributed differentiability is **already built, already verified, and already
documented** — the mechanism is not what the tranche prompt anticipated.
The paper-facing multi-GPU **scaling pipeline does not exist**: the bench scripts
are pre-Phase-5 scaffolds, there is no JSON, and there are no notebooks. So this
tranche is *wrap-and-write* for §2.6/§1, and *build-then-measure-then-write* for
§5 and figs 08–11.

**Update 2026-08-20.** Two things changed since this was written. The
distributed gradient result has now been *measured and recorded* (§3), and the
scaling pipeline of §2 has been built, though not yet run at scale. What remains
blocked is the scaling measurement itself, and figure 10, which turns out to have
no mechanism behind it at all.

---

## 1. Phase-5 status (`docs/phase5_multigpu_pallas_foldin_plan.md`)

Items **5a, 5b, 5c, 5d are all DONE**. Read from the STATUS block:

- **Distributed default basis/MAC is `real` + `dehnen`** — confirmed, and this is
  the expectation the prompt asked to check. It achieves **0.19% vs direct** at
  4 devices, beating the `bh` baseline's 0.24%. `solidfmm`/`bh` are kept behind
  explicit config plus a test.
- Native-real upward means the coarse-tree `all_gather` ships **real** multipoles
  (half the inter-GPU comm), and the complex→real M2L-boundary conversions are gone.
- 5c: fused real M2L Pallas is parity-neutral (4.4e-9 flag-on vs flag-off); fused
  leafpair Pallas near-field is 2.1e-7 vs baseline P2P. **The near field was the
  entire bottleneck**: 10.7 s → 43.5 ms (~245x) at ndev=2, per=8000, leaf=128.
- **Healthy per-GPU N ≈ 8000.**
- **Strong scaling is density-limited — confirmed, not assumed.** At per-GPU
  N > ~8000 the fixed-topology traversal caps overflow even at cap×64 (max
  retries), so forces are *truncated* and the 400–720 ms readings are padded
  pair-queue overhead, not physics. At total N=40 000 only ndev=5 (per=8000) is a
  valid point. The scaling story is therefore weak scaling; strong scaling
  saturates for a capacity reason, and §5 must say so in those terms.

**Numbers that exist only as a markdown table** (weak scaling, ndev 2→5,
N 16 000→40 000, 41.6→64.3 ms, throughput 3.8e5→6.2e5 part/s). Per the tranche's
own rule these are an engineering log, not paper figures: they need the
seeded-script → JSON → notebook treatment before §5 can quote them.

**Live caveat.** The `jit=True` illegal-address crash is **intermittent**
(nondeterministic OOB) — most runs succeed, one recurred at weak ndev=2. Run each
eval in its own process. Root-cause and a padded-pair-queue right-sizing (which
would lift the per-GPU-N ceiling and make strong scaling measurable) are the two
open follow-ups.

---

## 2. Paper-facing scaling pipeline — BUILT, not yet run at scale

Rewritten 2026-08-20 (`a6269a6`, `348ca65`). The scaffolds are gone.

| piece | state |
| --- | --- |
| `bench/multigpu/harness.py` | rebuilt: single-point worker + subprocess sweep |
| `bench/multigpu/_sweep_cli.py` | new: shared args, artifact writing, provenance |
| `bench/multigpu/{strong,weak}_scaling.py` | rewritten on the new harness |
| `bench/multigpu/load_balance.py` | rewritten; sweeps uniform **and** plummer |
| `bench/multigpu/comm_overhead.py` | **refuses to run** — see below |
| `bench/results/multigpu/*.json` | still only `.gitkeep` — nothing measured yet |
| `examples/jaccpot_paper/fig_{08,09,11}*.ipynb` | still absent |

Two structural facts drove the rewrite, both discovered rather than assumed:

1. **A device-count sweep cannot happen in one process.** JAX fixes its device
   count at backend initialisation, so the scaffold's
   `[run_once(n, g) for g in gpu_counts]` could never have worked. Every point is
   now its own process, which additionally isolates the intermittent `jit`
   illegal-address fault and makes the sweep an array job on a scheduler.
2. **There are no per-stage timers on the distributed path.** `DIAG_FIELDS`
   carries interaction counts and overflow flags and nothing else. Figure 06's
   single-device breakdown is not transferable: it reads `_refresh_timing_*`
   counters that live on the strict refresh path with fusion disabled, and the
   `shard_map` pipeline has no analogue.

**Figure 10 (comm/compute split) therefore has no mechanism.** The scaffold
declared eight stage names and a `COMM_STAGES` subset; nothing in the code emits
any of them, so anything built on them would have reported invented structure.
They are deleted, and `comm_overhead.py` exits 2 with an explanation rather than
producing a plausible artifact. Getting the figure needs one of: a host callback
per stage inside the sharded region (perturbs what it measures), attributing a
`jax.profiler` trace's XLA ops to stages (most promising, brittle against
fusion), or stage ablation (cheapest, least trustworthy). That is library work
and belongs in its own change.

What *is* measurable: total wall clock (figs 08, 09) and per-device pair counts
(fig 11). An overflowing point is recorded `valid=false` — its wall clock is
padding overhead over a **truncated force**, so it must never be plotted.

The whole pipeline was validated without GPUs on forced host devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=2`).

## 3. Distributed differentiability — ALREADY DONE, and the mechanism is not the expected one

`docs/differentiable_fmm_distributed_audit.md` (25 KB) already covers this, and
`tests/distributed/test_distributed_grad_correctness.py` already exists. So does
the entry point the audit names, verified in the code:

```python
make_force_evaluator(config, ndev, cap, mesh, *, jit=True,
                     differentiable=False, l2l_num_levels=None,
                     halo_exchange="auto")   # jaccpot/distributed/fmm.py:1591
```

### The mechanism (this is what §2.6 must describe)

**The ragged collectives' transpose was never the blocker.** The tranche prompt
expected to need custom transpose / `custom_vjp` rules for the ragged halo
all-to-all and `dual_tree_walk_cross`. The audit's verdict is explicit: *"The
anticipated hard part — the transpose of the ragged all-to-all — turned out not
to be a blocker at all. Its values are exact. Its side effects were not."*

What actually makes it differentiate is **the same fixed-topology seam as the
single-GPU path**: build the topology from `stop_gradient`-ed inputs, then
re-evaluate the numeric pipeline on the live inputs gathered into that frozen
order. The distributed body builds its tree *inside* `shard_map`, so the tree
build, the geometry pass and both dual-tree walks are all traced `while_loop`s —
five sites, in `yggdrax/_geometry_impl.py`, `_interactions_impl.py` (×2),
`yggdrax/distributed/cross_walk.py`, and `jaccpot/runtime/kernels/core.py`.
Every one produces **integer-only** output (permutations, node membership,
interaction lists, MAC decisions), so cutting the cotangent path there costs
nothing physical. That is exactly the fixed-topology contract — the walks never
needed a transpose.

Three supporting fixes, none of them a collective rule:

1. Near field routed through the existing `_radix_fast_lane_prepacked_accel_cvjp`
   (Pallas forward, analytic O(N) leaf-pair reverse) instead of raw `pallas_call`,
   which has no autodiff rule. Forward stays bit-identical.
2. The L2L cascade gets a **static** loop bound derived from a shape
   (`num_internal - 1`); `l2l_num_levels` tightens it. Too small a value is
   reported as `l2l_level_overflow` rather than silently truncating.
3. `jnp.asarray` normalisation of `G`/`softening_sq` in
   `_leafpair_accel_analytic_vjp` — a latent bug the distributed path exposed.

**Directly relevant to the §2 invariants:** the coarse (LET) tree's float payload
**stays live**. Its topology is built from a frozen frontier, but the coarse
leaves' COMs *are* the coarse expansion centres, and freezing them while the
seeded multipoles stay live would translate live multipoles by stale centre
deltas — *"a real missing gradient term, not a measure-zero topology effect."*
`_live_coarse_payload` re-gathers them by reproducing `build_remote_coarse_tree`'s
own integer-only selection. This is the distributed realisation of the paper's
existing "COM centres are differentiated through, not frozen" claim, and §2.6
should say so.

### The headline defect (worth a sentence in §5, not just the docs)

Executing a gradient **poisoned subsequent forward evaluations**: the ragged halo
exchange silently returned nothing for the rest of the process, so the forward
fell back to the local-only sum. Measured at 2×A100, N=64, `theta_cross=0.001`:
forward rel-L2 vs exact went 2.639e-16 → **4.199e-01** after one gradient, while
matching the *local-only* force to 2.249e-16. It hit every evaluator built before
the gradient, including forward-only ones.

An upstream JAX/XLA ragged-collective bug (draft report in
`docs/jax_ragged_all_to_all_bug_report.md`), worked around behind a **version
gate** rather than a fixed default. `halo_exchange=` defaults to `"auto"`, and
`resolve_grad_halo_exchange` returns `"native"` on JAX >=
`JAX_RAGGED_GRAD_FIXED_VERSION` = (0, 9, 1) and the bandwidth-hungry but safe
`"buf"` below it.

**This host is on jax 0.10.2, so `auto` already resolves to `"native"`** — the
workaround is not in force here, and the suite result above was obtained on the
native ragged exchange, `test_forward_survives_a_gradient` included. The audit's
own text still says the default is `"buf"`, which predates the gate; read the
code, not that sentence.

The remaining cleanup is therefore only the dead-code half of the audit's
follow-up: `_grad_halo_exchange` and the `"buf"` path still exist for users below
the floor. Keep them until the JAX floor in `pyproject.toml` rises above 0.9.1,
at which point both become unreachable and can go.

### What is already verified

2×A100, N=64 (32/device), `order=3`, `theta=0.4`, `leaf_size=8`, real basis,
dehnen MAC, float64, separated-cluster IC (so the cross-domain path is genuinely
engaged):

| check | positions | masses |
| --- | --- | --- |
| forward invariance (`differentiable=True` vs shipped) | bit-identical | bit-identical |
| FD vs AD, baseline near field | 2.7e-08 | 5.9e-09 |
| FD vs AD, fused-Pallas near field | 2.7e-08 | 6.2e-09 |
| **`grad(FMM)` vs `grad(direct sum)`** | **1.13e-03** | **4.49e-03** |
| forward force error, same configuration | 5.86e-03 | — |

The primary-oracle gate the tranche asks for is **already met**: the gradient
matches the direct-sum oracle to better than the forward force's own accuracy.

### What is NOT covered (this is the honest gap list for §5)

- **2 devices and N=64 only.** No 4/8-GPU run, no scale sweep. `ndev` is a
  parameter everywhere so nothing is device-count-specific by design, but it is
  untested above 2.
- **Performance only coarsely characterised.** Reverse/forward ≈ **6–11x** at
  N ≤ 8192 on a *contended* host, reverse compile 3–6 min. The audit warns
  explicitly to compare `forward+backward` times and not ratios, because the
  forward alone swings ~50% under load and the ratio inherits that through its
  denominator. No reliable peak-memory figure, no multi-GPU scaling of the
  gradient. **There is no "overhead vs #GPUs" data — fig 12's distributed variant
  does not exist yet.**
- **Configurations not exercised:** `treecode` local walk, complex
  (`basis="solidfmm"`) far field, `m2l_chunk`, `far_m2l_fp32`, `return_potential`.
  The seam is basis-agnostic by construction but **only the real basis was
  measured** — so the tranche's "both bases" ask is currently unmet.
- **`nearfield_chunk` raises** under `differentiable=True` (no rule for the
  decoupled kernel) — deliberately, rather than failing obscurely.
- **The host-side driver `distributed_fmm_accelerations` is NOT differentiable**
  and will not become so: it partitions and reassembles in NumPy with a Python
  loop. Gradients are taken of the **`shard_map` evaluator**, w.r.t. the padded
  per-device layout from `partition_for_devices`; mapping back to input order is
  the caller's job via the returned `gid`. **This directly contradicts the
  tranche prompt's Step 0 framing**, which asks how
  `distributed_fmm_accelerations` differentiates. It does not — `make_force_evaluator`
  is the differentiable entry point, and any harness or figure must use it.
- **Topology is rebuilt inside every call** from `stop_gradient`-ed inputs, so it
  tracks current positions; unlike the single-GPU path there is no `state` to
  hoist, and the build cost is paid per gradient (forward-only, no cotangent
  path).

---

## 4. Consequences for the tranche

**Wrap-and-write (no GPU needed, mechanism is pinned down):**
- §2.6 scope extension and the Introduction touch-up (Step 3) can be written
  now, from §3 above. The mechanism is established from code + audit, so the
  guardrail's "leave a precise flag rather than inventing" does not apply.
  §2.6 must say *fixed-topology seam + integer-only topology loops*, **not**
  "custom transpose rules for the ragged collectives", which would be false.

**Build-then-measure-then-write (blocked on GPUs):**
- Steps 1, 2, 4, 5, 6 — the harness rewrite, figs 08–11, the distributed-grad
  JSON and the fig-12 "vs #GPUs" variant, §5 prose, and the §8 MAC probe.

**A number that does not exist yet must not be written.** Per the tranche's own
rule: §5 cannot be drafted from the phase-5 markdown table. `bench/results/multigpu/`
is empty, so there is nothing to quote.

### Scope corrections to carry forward

1. **Paths moved.** The merge of `main` renamed `results/` → `bench/results/`
   (upstream `6839a6c`). The tranche prompt's `results/multigpu/*.json` and
   `results/differentiability/*.json` are now under `bench/results/`.
   `jsonio.RESULTS_ROOT` is the single authority; figures land in
   `bench/results/figures/`.
2. **`tests/unit/test_distributed_gradient_correctness.py` should not be created.**
   `tests/distributed/test_distributed_grad_correctness.py` already exists, and
   `tests/distributed/` is the correct home — those files skip below 2 devices,
   which is exactly the required behaviour. A copy under `tests/unit/` would run
   (and fail or vacuously skip) in the single-device CPU suite.
3. **`docs/distributed_differentiability.md` is redundant.** The audit already
   is that document. Extend it if anything new is measured; do not start a
   parallel one.
4. **`docs/multigpu_differentiability_model.md`** is a 34-line stub of unfilled
   checkboxes asking for exactly what §1 and §3 above now establish. It should be
   completed or superseded rather than left contradicting the audit.

---

## 5. Blocked on hardware

All 8 A100-PCIE-40GB cards were occupied at the time of writing (34–39 GB of
40 GB held on 7 of them). Nothing was run. When cards free up:

```bash
# Distributed gradient correctness — reproduces the audit's table (2 GPUs)
export CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
  pytest tests/distributed/test_distributed_grad_correctness.py -o addopts="" -q

# Whether the upstream ragged bug is fixed in the current JAX
JACCPOT_CHECK_UPSTREAM_RAGGED_FIX=1 \
  pytest tests/distributed/test_distributed_grad_correctness.py -o addopts="" -q
```

Scaling runs (figs 08–11) additionally need the harness rewritten first — the
scaffolds cannot produce the per-stage timers or per-GPU interaction counts.
Weak scaling wants ndev 2→5 at per-GPU N=8000; strong scaling is only valid at
per-GPU N≈8000, so a strong-scaling figure is capacity-limited until the
padded-pair-queue right-sizing lands. **Run each eval in its own process** because
of the intermittent `jit=True` OOB.

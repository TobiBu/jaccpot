# Distributed (multi-GPU) differentiability — audit and outcome

Phase 5 of the differentiable-FMM work: make the multi-GPU `shard_map` force
pipeline differentiable, and establish single-shot gradient correctness on 2
GPUs before claiming multi-GPU differentiability anywhere.

**Verdict.** The distributed FMM is now end-to-end differentiable w.r.t. particle
positions and masses at fixed topology, verified on 2×A100 against both finite
differences and an exact direct-sum gradient oracle. Getting there needed four
fixes, one of which is a **correctness bug in the ragged halo exchange's reverse
pass that silently corrupted subsequent forward evaluations** — the most
consequential finding here, and the reason the audit went past "does `jax.grad`
run".

The anticipated hard part — the transpose of the ragged all-to-all — turned out
not to be a blocker at all. Its *values* are exact. Its *side effects* were not.

Entry point: `make_force_evaluator(config, ndev, cap, mesh, differentiable=True)`.
Contract, scope and limits: see "What is verified" and "What is not covered".

---

## Method: trace first, then measure

Two cheap probes did nearly all the work, in this order.

1. **`jax.make_jaxpr(jax.grad(loss))`** rather than executing the gradient.
   Missing autodiff rules, `pallas_call`, and `while_loop` reverse-mode all raise
   at *trace* time, so blockers surface in seconds instead of after a multi-minute
   XLA compile. The first attempt at executing the gradient was killed at the
   10-minute mark having printed nothing; the trace probe answered the same
   question in under a minute.
2. **A liveness sweep over the forward jaxpr.** Propagate "reachable from the
   differentiand" through every equation, stopping at `stop_gradient`, and report
   every `while` equation with a live float input. That names the exact blocking
   loops with source locations, instead of a transpose-time error with no user
   frames in its traceback. Both probes recurse into the `shard_map` sub-jaxpr.

A liveness sweep that *doesn't* stop at `stop_gradient` is worth keeping around
as a control: run both ways and the difference is precisely what the
fixed-topology seam bought.

---

## Blockers, ranked, and what each cost to fix

### 1. Raw `pallas_call` on the near-field path

`fn` called `_radix_fast_lane_prepacked_pallas` directly
(`jaccpot/distributed/fmm.py`), and `pallas_call` has no autodiff rule:

```
Linearization failed to produce known values for all output primals.
  jaccpot/distributed/fmm.py:1141 in fn -> _radix_fast_lane_prepacked_pallas
  jaccpot/pallas/nearfield_fused_leaf.py:571 in nearfield_leafpair_pallas
```

**Fix:** route the differentiable path through the existing production wrapper
`_radix_fast_lane_prepacked_accel_cvjp` (Pallas forward + analytic O(N) leaf-pair
reverse). Same kernel, so the forward stays bit-identical. The reverse runs
untiered here: occupancy tiers need a *concrete* validity mask to read each
tier's static slot width from, and inside `shard_map` that mask is a tracer.

The chunked near field (`nearfield_chunk`) calls the *decoupled* Pallas kernel,
which has no `custom_vjp` — it now raises under `differentiable=True` rather than
failing obscurely deep in a linearization.

### 2. `lax.while_loop` / dynamic-bound `fori_loop` on the topology path

```
Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop
with dynamic start/stop values.
```

The distributed body builds its tree *inside* `shard_map` (unlike the single-GPU
path, where `prepare_state` runs concretely on the host), so the tree build, the
geometry pass and both dual-tree walks are all traced, and all of them are
`while_loop`s carrying float geometry. The liveness sweep found five sites:

| site | stage |
| --- | --- |
| `yggdrax/_geometry_impl.py:299` | `compute_tree_geometry` |
| `yggdrax/_interactions_impl.py:903` | `_propagate_extents` |
| `yggdrax/_interactions_impl.py:1764` | `_dual_tree_walk_impl` (self walk) |
| `yggdrax/distributed/cross_walk.py:395` | `dual_tree_walk_cross_impl` |
| `jaccpot/runtime/kernels/core.py:2129` | L2L level cascade (see #3) |

**Fix:** the fixed-topology seam. Build the topology from `stop_gradient`-ed
inputs, then re-evaluate the numeric pipeline on the live inputs gathered into
that frozen order. Every one of those four topology loops produces *integer*
output only (permutations, node membership, interaction lists, MAC decisions), so
cutting the cotangent path costs nothing physical — it is exactly the
fixed-topology contract, and the walks never needed a transpose in the first
place.

Two details that are easy to get wrong:

* **Geometry always takes the frozen positions.** It feeds the MAC only, and the
  values are identical to the live ones, so the forward does not move.
* **The coarse (LET) tree's float payload must stay live.** Its *topology* is
  built from a frozen frontier, but the coarse leaves' COMs *are* the coarse
  expansion centres. Freezing them while the seeded multipoles stay live would
  make the coarse M2M translate live multipoles by stale centre deltas — a real
  missing gradient term, not a measure-zero topology effect. `_live_coarse_payload`
  re-gathers them by reproducing `build_remote_coarse_tree`'s own selection, which
  is integer-only (it depends on the axis index, not on any float) and therefore
  reproduces `rct.positions_sorted` exactly.

### 3. The L2L cascade's dynamic loop bound

`_propagate_solidfmm_locals_by_level` defaults to a device-side
`jnp.max(node_levels)` bound, which makes the cascade a dynamic-bound
`fori_loop`. It already accepts a static `num_levels` for exactly this reason —
but no caller can supply the concrete depth here, because the tree is built
inside `shard_map`.

**Fix:** derive a *static* bound from a shape. `num_internal - 1` is the tightest
one available and is safe for any distribution (a binary tree with `num_internal`
internal nodes has no internal node deeper than `num_internal - 1`). Levels past
the real depth have no active parents, contribute exactly zero, and leave the
force bit-identical — they cost time, not accuracy. `l2l_num_levels` overrides it
when the depth is known (a balanced tree is ~`log2(num_leaves)` deep, so the
default can be ~2 orders of magnitude loose at scale). Too small a value
truncates the cascade, so it is reported as `l2l_level_overflow` in the
diagnostics rather than silently producing a wrong force and gradient.

This looked like the one blocker whose fix carried a real performance cost. It
does not: see "Cost" below — tightening the bound is worth ~6% at N=512 and
nothing at N=2048, so the safe default is cheap and this is a correctness knob.

### 4. A JAX host-side literal reaching the analytic near-field reverse

With the near field routed through its `custom_vjp`, the reverse died on:

```
TypeError: bad operand type for unary -: 'TypedNdArray'
  jaccpot/nearfield/grad.py:409 in _apply  ->  tgt_contrib = -G * jnp.sum(pair, ...)
```

`TypedNdArray` is JAX's host-side literal wrapper. A caller whose `G` /
`softening_sq` are Python constants *inside a jitted region* hands them over as
literals, which reach the rule through the `custom_vjp` residual and do not
implement `__neg__`. The single-GPU caller happens to pass concrete device
arrays, so only a jitted caller trips it — a latent bug, found by the distributed
path rather than caused by it.

**Fix:** normalise both scalars with `jnp.asarray` at the top of
`_leafpair_accel_analytic_vjp`.

---

## The headline finding: the native ragged all-to-all poisons later forwards

**Symptom.** With everything above fixed, FD and AD disagreed by 4–500 % while
the forward was exact. FD was *converged*, not noisy — a step-size sweep from
1e-2 down to 1e-7 showed a stable plateau, ruling out both round-off and
topology flips.

**What it actually was.** Executing the gradient changes the forward. Measured on
2×A100, N=64, `theta_cross=0.001` (all cross-domain pairs near, so the FMM force
is the exact direct sum):

| | rel-L2 vs exact force | rel-L2 vs *local-only* force |
| --- | --- | --- |
| forward, before any gradient | 2.639e-16 | 4.896e-01 |
| forward, after tracing a gradient | 2.639e-16 | — |
| forward, after **executing** a gradient | **4.199e-01** | **2.249e-16** |

After one gradient, the forward reproduces the **local-only** sum to machine
precision: the cross-domain near field is gone entirely. The halo exchange
returns nothing, silently, for the rest of the process. Further properties:

* it hits **every evaluator built before the gradient**, including a
  `differentiable=False` forward-only one, while a **freshly built** evaluator is
  clean;
* **tracing** the gradient is harmless; only executing it does damage;
* the inputs are untouched (positions and masses verified bit-intact);
* the **single-GPU** differentiable path is unaffected (forward drift 0.0, FD-vs-AD
  4.5e-08), so this is distributed-specific.

**Ruled out along the way.** Both suspected mechanisms were tested and cleared:

* *The ragged all-to-all's VJP values.* Exact. AD matches FD to 1.6e-10–3.3e-09
  for both the native and the `buf` implementation, and the two agree with each
  other to 0.0.
* *`all_gather`'s transpose under `check_vma=False`* (how the coarse multipoles
  cross devices). Exact in all four `tiled` × `check_vma` combinations: AD vs FD
  5.4e-09, AD vs an independent single-device reference 0.0.
* *jaccpot's own `donate_argnums`.* Re-jitting all five donating kernels
  (`_accumulate_m2l_fullbatch`, `_accumulate_m2l_chunked_scan`, the three
  `_propagate_*`) without donation reproduced the corruption **identically**.
* *jaccpot's analytic VJP gates.* `JACCPOT_ANALYTIC_P2P_VJP=0
  JACCPOT_ANALYTIC_L2P_VJP=0` reproduced it identically too; the rules are not on
  this path.

### It is a JAX bug, reproducible in 40 lines with none of our code

Attribution is not a judgement call — it is measured. The following uses only
`jax.lax.ragged_all_to_all` inside `shard_map`; no jaccpot, no yggdrax. It is
checked in as `bench/repro_jax_ragged_all_to_all_grad.py`:

```python
def body(x, s, i, o, r):
    out = jnp.full((CAP,), -1.0, x.dtype)          # the ragged output buffer
    return jax.lax.ragged_all_to_all(x, out, i[0], s[0], o[0], r[0], axis_name="gpus")

run = maybe_jit(lambda x: shard_map(body, mesh=mesh, in_specs=(P("gpus"),) * 5,
                                    out_specs=P("gpus"), check_vma=False)(x, so, io, oo, ro))
before = run(x)
jax.grad(lambda v: jnp.sum(run(v) ** 2))(x)        # execute a gradient
after = run(x)                                     # <-- differs from `before`
```

```
bare shard_map(...)        before=[1. 2. 5. 6. 3. 4. 7. 8.]  after=[1. 2. 5. 6. 3. 4. 7. 8.]  CLEAN
jax.jit(shard_map(...))    before=[1. 2. 5. 6. 3. 4. 7. 8.]  after=[ 1. 2. 5. 6. -1. -1. -1. -1.]  CORRUPT
```

`-1.0` is the *fill value*: after the gradient, the exchange no longer writes its
output — the forward returns the untouched buffer. jax/jaxlib 0.9.0.1, 2xA100,
CUDA. The trigger is **`jax.jit` around the `shard_map`**; the bare `shard_map` is
clean, which is also why the corruption never showed up in the isolated VJP-value
test (that one never re-ran a jitted forward afterwards).

Two things about the reproducer worth knowing before trusting a run of it. *Which*
device's rows are lost varies run to run (either half, sometimes both), though
whether a jitted run corrupts does not. And it must exercise **one configuration
per process**: a case that corrupts in a fresh process can report CLEAN if another
jit+grad sequence ran earlier in the same process — which is exactly how a first
version of this reproducer briefly appeared to exonerate the constant-buffer case.
In fresh processes the outcome is deterministic: 6/6 CORRUPT under `--jit` for both
the constant and the data-dependent output buffer.

**What this rules out, including one of our own earlier hypotheses.** An earlier
revision of this document blamed XLA input/output aliasing on the `output`
operand: JAX's transpose rule passes `ad.zeros_like_aval(...)` there, and an
in-place write into a *shared constant* would have explained the
already-compiled-executables-break pattern. **That is wrong.** Making the output
buffer data-dependent (`jnp.full(...) + jnp.zeros_like(x[:CAP]) * x[:CAP]`), so it
cannot be a shared constant, corrupts identically — and in that variant *every*
row comes back as fill, not just one device's. The fixed
`channel_id = mlir.COLLECTIVE_CHANNEL_ID` in `_ragged_all_to_all_lowering` is
likewise not a distinguishing suspect: every JAX collective uses that same id, and
`psum`/`all_gather` differentiate correctly. **The mechanism is now known, and it is not JAX.** It is XLA:GPU
`ragged_all_to_all_thunk.cc`: in the XLA that jaxlib 0.9.0.1 pins,
`RaggedAllToAllStartThunk::Initialize` early-returns once a stream state exists, so
the rendezvous exchanging peer output-buffer *device addresses* runs only once and
caches addresses from the first execution -- while `Thunk::Initialize` really does
run every execution with fresh `buffer_allocations`. After allocator churn (running
a gradient) the one-shot kernel P2P-writes to stale addresses and the true output
buffer keeps its fill value, which is exactly the observed symptom. Introduced in
XLA `bf4fd02e5a` (2026-01-15), fixed in `4e0cc7e356` (2026-02-06). It requires GPU
peer access, so it cannot reproduce on a box without P2P -- a CLEAN result there
proves nothing. **Do not file the draft in
`docs/jax_ragged_all_to_all_bug_report.md`**; JAX's autodiff rules for the
collective are correct.

**Upstream status: FIXED in JAX 0.9.1** (XLA-side; see the mechanism above).
Measured, not read off a changelog -- the fix appears in neither the JAX changelog
nor any tracked JAX issue, because the repair was in XLA. On 0.9.0.1 the
reproducer is 6/6 CORRUPT under `--jit`; on 0.9.1 it is 4/4 CLEAN, and
`test_native_halo_exchange_is_fixed_upstream` passes against the real pipeline
(it fails on 0.9.0.1 with a 0.42 rel-L2 forward drift). No issue was filed, so
`docs/jax_ragged_all_to_all_bug_report.md` is now only of historical interest --
though it is still the artifact to send if a regression shows up.

`halo_exchange="auto"` (the default) acts on this: it selects the cheap ragged
exchange on JAX >= `JAX_RAGGED_GRAD_FIXED_VERSION` = 0.9.1 and the safe
`all_gather` fallback below it, so a user on an affected JAX cannot silently
compute wrong forces and a user on a fixed one needs no action.

**The package is now on it.** That took removing a ceiling: `pyproject.toml` used
to cap JAX at `<0.9.1` because 0.9.1 *also* removed `pallas_call`'s `backend=`
kwarg, which the fused M2L kernels passed -- so the ragged fix and a Pallas break
arrived in the same release. `jaccpot/pallas/_compat.pallas_backend_kwargs` now
selects Triton on either API (`backend=` below 0.9.1, `triton.CompilerParams`
from 0.9.1 on), the floor is `>=0.9.1`, and `"auto"` therefore resolves to
`"native"` in every supported configuration. The `"buf"` path stays as the
fallback a regression would need.

**Fix on our side.** `_grad_halo_exchange` pins the halo import to the `all_gather`-based
`"buf"` exchange for the halo import on the differentiable path. It computes the
identical result out of an `all_gather` plus an index gather, whose reverse is an
ordinary psum/scatter-add. After the fix: forward bit-identical before and after a
gradient, and FD-vs-AD 1.9e-06.

The price is communication volume — `"buf"` gathers every device's whole send
buffer instead of exchanging ragged blocks point-to-point, i.e.
O(ndev × capacity) rather than O(actual halo). That is why it is scoped to the
gradient path and the forward keeps the native exchange.

**Do we even need the ragged primitive?** No. The halo exchange only has to move
each requested leaf's particles from owner to requester; `ragged_all_to_all` is a
*padding* optimisation for irregular sizes, not a requirement. Measured on 2xA100,
all three alternatives are exact under `jit` + `grad` **and** leave the forward
intact afterwards:

| exchange | VJP vs FD | forward after grad | communication |
| --- | --- | --- | --- |
| `jax.lax.ragged_all_to_all` | exact | **CORRUPT** | O(actual halo) |
| `jax.lax.all_to_all`, fixed per-peer block | 7.0e-09 | stable | O(ndev x block) |
| `jax.lax.ppermute` ring, ndev-1 shifts | 2.2e-09 | stable | O(ndev x block) |
| `jax.lax.all_gather` (the `"buf"` path we ship) | 7.0e-09 | stable | O(ndev^2 x block) |

So the workaround we ship is the *most* expensive of the three safe options, by a
factor `ndev`. **`all_to_all` is the better replacement**, and
`yggdrax.distributed.comm.all_to_all_dense` already implements exactly that shape
(`[ndev, capacity, *feat]` in, per-source blocks out) — `import_near_halo` simply
does not route through it. Upgrading means either teaching `import_near_halo` to
take an exchange backend or writing a jaccpot-side halo exchange over
`all_to_all_dense`.

Not done here, deliberately: at the only device count we have verified (ndev=2)
the win is 2x on one stage of a pipeline whose hotspot is unprofiled, so it is a
**scaling** improvement to make when >2 GPUs are actually in play — and the cost
model above says the gap widens as `ndev` grows (8x at ndev=8).

**Guard.** `tests/test_distributed_grad_correctness.py::test_forward_survives_a_gradient`
asserts a forward evaluated after a gradient is bit-identical, for both the
differentiable and the forward-only evaluator. Without that assertion this class
of bug is invisible: every individual gradient looks right, and the *forward*
quietly loses a term.

---

## What is verified

2×A100-PCIE-40GB, N=64 (32/device), `order=3`, `theta=0.4`, `leaf_size=8`, real
basis, `dehnen` MAC, dual-tree local walk, float64, separated-cluster IC (one
cluster per Morton domain, so the cross-domain path is genuinely engaged).

| check | positions | masses |
| --- | --- | --- |
| forward invariance (`differentiable=True` vs shipped) | bit-identical | bit-identical |
| FD vs AD, baseline near field | 2.7e-08 | 5.9e-09 |
| FD vs AD, fused-Pallas near field | 2.7e-08 | 6.2e-09 |
| `grad(FMM)` vs `grad(direct sum)` | 1.13e-03 | 4.49e-03 |
| forward force error, same configuration | 5.86e-03 | — |

## Cost: correct, not fast

2xA100 host, `jit=True`, steady state (median of 3 after warmup), forward = the
shipped `differentiable=False` evaluator. **Read this table with its noise in
mind** (see below); the ratios are order-of-magnitude, not benchmark-grade.

| N | `l2l_num_levels` | forward | forward+backward | ratio | reverse compile |
| --- | --- | --- | --- | --- | --- |
| 512 | default | 1.98 s | 22.3 s | 11.2x | 193 s |
| 512 | default *(repeat, fresh process)* | 2.35 s | 17.1 s | 7.3x | 182 s |
| 512 | 12 | 2.45 s | 16.13 s | 6.6x | 188 s |
| 512 | 12 *(repeat, fresh process)* | 1.63 s | 16.17 s | 9.9x | 188 s |
| 2048 | default | 10.6 s | 92.2 s | 8.7x | 269 s |
| 2048 | 12 | 9.06 s | 96.4 s | 10.6x | 263 s |
| 8192 | default | 27.3 s | 156.9 s | 5.7x | 377 s |

**Compare forward+backward times, not ratios.** The repeats localise the noise: on
a fresh process `forward+backward` reproduces to **0.2%** (16.13 s vs 16.17 s at
N=512, `l2l=12`), while the *forward* alone swings ~50% on the same config
(1.63-2.45 s) because the host is shared and was under load from other users. The
**ratio column inherits that noise through its denominator** -- which is how the
same tightened config reads 6.6x in one run and 9.9x in the next despite an
identical backward. Across differently-conditioned processes even
`forward+backward` moves (N=512 default: 22.3 s in a multi-config process, 17.1 s
fresh), so only same-shape comparisons are meaningful.

Do not read the absolute per-call times as throughput either: they are implausible
next to the single-GPU path (1 M particles forward in 2.5 s). The slow forward is
pre-existing and not from the gradient work; it is the shipped distributed forward,
dominated by fixed-size traversal buffers and a per-call topology rebuild rather
than by N.

What survives the noise:

* The reverse costs a **single-digit multiple of the forward**, roughly 6-11x at
  these sizes.
* **Reverse compile time (3-6 min) is a usability problem** in its own right,
  mirroring the single-GPU path's known ~10 min at N=16384.

What does **not** hold up — and it is the claim this document made first:

* **The loose static L2L bound is not the bottleneck.** An earlier revision
  reported that tightening it cut the N=512 reverse from 11.2x to 6.3x (~1.8x) and
  called it "the first thing to fix for speed". That compared two runs made under
  different conditions. Controlled, same-conditions comparisons show
  **~6% at N=512** (17.1 s -> 16.13/16.17 s, the tightened figure reproduced twice
  to 0.2%) and **nothing at N=2048** (92.2 s -> 96.4 s, i.e. inside the noise, if
  anything worse) -- despite the tightened bound running 13 L2L iterations instead
  of 127. So `l2l_num_levels` is a correctness knob with a marginal performance
  effect, not a speed lever, and the safe default costs little.
* Which means **the real hotspot is unidentified**. By analogy with the single-GPU
  path (reverse 94-98% near field) the near field is the place to look, but that is
  an inference; no profile of the distributed reverse has been taken. Anyone
  optimising this should start by profiling, not by tightening the L2L bound.

**The obvious escape route does not work.** Since the JAX bug needs `jax.jit` to
trigger, `jit=False` plus the *fast* native ragged exchange should have been
strictly better. Measured: it failed to complete one gradient at N=64 in 10
minutes (eager dispatch of the whole body), so it is not a viable alternative and
the `buf` exchange stays.

The oracle agreement is *better* than the forward force accuracy it
differentiates, which is the expected relationship and the strongest available
signal that the reverse pass is the exact reverse of what the forward computes.
The two near-field lanes produce gradients agreeing to every printed digit.

Regime sweeps, which is how the halo bug was localised:

| regime | outcome |
| --- | --- |
| `theta_cross=1e6` (all cross-domain **far**: coarse M2L, `all_gather` seeding, L2L, L2P) | FD vs AD 9.0e-08 (pos), 3.9e-10 (mass) |
| `theta_cross=0.001` (all cross-domain **near**: halo import + combined P2P) | forward == exact direct sum to 2.9e-16; AD-FMM vs AD-direct **3.7e-16 / 2.9e-16** |

That second row is the cleanest statement available: where the FMM *is* the exact
direct sum, its gradient *is* the direct sum's gradient, to round-off.

---

## What is not covered

Scoped honestly, so nothing here reads as a broader claim than was measured.

* **2 devices, N=64.** No 4/8-GPU run, and no scale sweep. Nothing in the design
  is device-count-specific (`ndev` is already a parameter everywhere), but it is
  untested above 2.
* **Performance is only coarsely characterised** (see "Cost" above): reverse/forward
  ratios on a contended host at N <= 8192, no reliable peak-memory figure, no
  4/8-GPU scaling. Distributed gradients are *correct*, not *fast* — treat the
  Phase-4 overhead numbers as single-GPU only.
* **Configurations not exercised:** the `treecode` local walk, the complex
  (`basis="solidfmm"`) far field, `m2l_chunk`, `far_m2l_fp32`, and potentials
  (`return_potential`). The seam is basis-agnostic by construction, but only the
  real basis was measured.
* **`nearfield_chunk` raises** rather than working (no autodiff rule for the
  decoupled kernel).
* **The host-side driver `distributed_fmm_accelerations` is not differentiable**
  and will not become so: it partitions and reassembles in NumPy, with a Python
  loop over rows. Gradients are taken of the `shard_map` evaluator, w.r.t. the
  padded per-device layout `partition_for_devices` produces. Mapping back to input
  order is the caller's job, via the returned `gid`.
* **Topology is rebuilt inside every call**, from `stop_gradient`-ed inputs, so it
  always tracks the current positions. That differs from the single-GPU entry
  point, where a `state` is built once and reused: there is nothing to hoist here,
  and the topology build cost is paid per gradient (forward-only — no cotangent
  path through it).
* **The upstream ragged bug is worked around, not fixed.** The workaround is one
  argument: `make_force_evaluator(..., halo_exchange=...)`, defaulting to the safe
  `"buf"`. After a JAX upgrade, run
  `JACCPOT_CHECK_UPSTREAM_RAGGED_FIX=1 pytest tests/test_distributed_grad_correctness.py`
  -- `test_native_halo_exchange_is_fixed_upstream` differentiates through
  `"native"` and checks the forward afterwards. When it passes, make `"native"` the
  default and delete `_grad_halo_exchange`. A draft of the upstream report is in
  `docs/jax_ragged_all_to_all_bug_report.md`.

---

## Reproducing

```bash
export CUDA_VISIBLE_DEVICES=$(autocvd -n 2 -l -o)
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
  pytest tests/test_distributed_grad_correctness.py -o addopts="" -q
```

The upstream ragged-collective bug reproduces on its own with
`RAGGED_METHOD` unset vs `RAGGED_METHOD=buf` around any two-device
`import_near_halo` call: forward, gradient, forward again, and compare the two
forwards.

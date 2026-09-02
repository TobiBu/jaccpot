# Figure 12 (autodiff overhead): DELIVERED on the real basis 2026-08-22

**Status 2026-08-22: delivered for the real basis over N=256..65536. The complex
basis is unusable above N=2048 for a reason not yet identified.** The original
2026-08-01 analysis below is kept verbatim, because its diagnosis was right and
its route 1 is what worked.

## What unblocked it

Route 1 ("time inside one jit rather than around it") now exists as the public
`differentiable_step_fn`, which returns a compiled `f(positions, masses)` and
takes `compile_now` so the compile is paid at setup. `autodiff_overhead.py` gained
`--mode step` for it. Measured against the wall this document records:

* the compile wall does **not** reappear: N=16384 and N=65536 both complete,
  where the outer-`jax.jit` route had not finished at N=16384 after 30 minutes;
* the forward stops measuring tracing: CPU at N=256 goes 1392 ms eager to
  **1.73 ms**.

## The result

Real basis, A100, fp64, p=4, theta 0.5, leaf 32, minimum over repeats, both M2L
lanes and all three differentiation targets:

| N | forward (loss) | ratio range |
|---|---|---|
| 256 | 0.51 ms | 1.95 - 2.33x |
| 1024 | 1.49 ms | 1.78 - 2.69x |
| 2048 | 3.44 ms | 2.02 - 3.09x |
| 4096 | 15.99 ms | 1.58 - 2.56x |
| 8192 | 30.49 ms | 2.29 - 3.17x |
| 16384 | 65.58 ms | 3.08 - 3.77x |
| 65536 | 546.63 ms | 2.66 - 3.46x |

So the reverse pass costs **1.6-3.8x** the forward across a 256x range of N. That
is the bounded small multiple the figure exists to show. The ratio rises with N
rather than staying flat, so it is bounded but not constant over this range.

## What is still open: the complex basis above N=2048

Complex reports ratios of 0.78-1.07 for N in 4096..16384 -- a forward-plus-backward
apparently cheaper than its own forward, which is impossible and therefore a
measurement artifact. Its forward at those N costs ~2.5x the real basis at the same
N (39.6 vs 16.0 ms at N=4096), and the ratio is depressed exactly where that
happens. It recovers to 1.38-1.74x by N=65536.

Three hypotheses are ruled out; do not retry them:

1. **The jit boundary.** Both arms already cross the same `jax.jit`; making it
   explicit moved nothing.
2. **Near-field lane auto-selection.** The crossover is
   `nearfield_fast_lane_min_particles=100000`, so every N in this sweep resolves
   to `bucketed`. Pinning `bucketed` reproduced `auto` to within 2%; pinning
   `fast_lane` kept the sub-1 ratios (0.95-1.02 at complex/N=4096).
3. **The loss-vs-force denominator.** Timing the bare `(N,3)` force against
   `value_and_grad` of a scalar is genuinely not a ratio, and that is fixed --
   but it was not the cause. After the fix the loss and force forwards agree to
   within noise at nearly every N, and the run-to-run spread at real/N=4096
   (16.0-24.3 ms) is larger than the gap involved.

The real basis is the production choice and is unaffected, so the figure is
delivered on it and the complex arm is reported as a limitation. Note that
`basis="complex"` selects the solidfmm path and is bit-identical to it, so this is
a performance question, not an accuracy one.

---

## Original analysis, 2026-08-01 (kept: the diagnosis was correct)

## What the figure needs

A ratio: (forward + backward) / forward for `differentiable_accelerations` at fixed
topology, versus N. The claim it supports is that the reverse pass costs a bounded
small multiple of the forward, so gradient-based inference over an FMM force is
affordable. The ratio is the whole point -- an absolute reverse time means nothing
without the forward it is measured against.

## Why neither dispatch mode gives that ratio

**Jitted: does not compile.** An outer `jax.jit` over the whole differentiable call
did not finish compiling after **18 minutes at N=256** on an A100, at 85% CPU and
0% GPU. At N=16384 XLA spent 3m21s on `jit__accumulate_m2l_fullbatch` alone and had
not finished the module after 30 minutes. Dropping N did not help, so the cost is
the unrolled pipeline -- order-4 rotation cascades -- and not the problem size.
`docs/differentiable_fmm_design.md` documents a jit limitation at *large* N; this
measurement says it is effectively unconditional.

Note the failure mode: it is slow, not raising. Any `try: jit / except: eager`
structure -- including the one in `autodiff_overhead.py` -- never fires, so a
harness can appear to hang rather than fall back.

**Eager: measures tracing, not compute.** With `--no-jit`, both arms run eagerly and
the ratio is at least self-consistent, but every call re-traces:

| device | N | forward | ratio (both) |
|---|---|---|---|
| CPU | 256 | 2543 ms | 1.88x |
| A100 | 256 | 6176 ms | 11.73x |

A forward pass of 6.2 s at N=256 is not compute; the direct O(N^2) sum at that size
is under a millisecond. The two devices disagree by 6x on a quantity that should be
device-insensitive, which is the tell: this is per-call trace and dispatch cost, and
the reverse graph is simply larger to rebuild. The GPU's 11.7x would read as "the
reverse pass costs 12x the forward" and that is not what it measures.

An upper bound is worth publishing when it is tight. 11.7x is not a bound on
anything useful.

## What would make it measurable

In rough order of cost:

1. **Time inside one jit rather than around it.** `prepare_state` once, then a
   single jitted step function that takes positions/masses as arguments, so the
   compile is paid once and amortised over repeats. This is the standard fix and
   probably the right one; it needs the seam to be traceable, which is exactly the
   host-op limitation the design doc describes.
2. **Let the compile finish once and cache it.** With a persistent compilation cache
   (`JAX_COMPILATION_CACHE_DIR`) a 20-40 minute compile is paid once per shape.
   Feasible as an overnight run: 3 N values x 2 bases.
3. **Report the ratio of the pieces.** The M2L/L2L reverse is a transpose of a linear
   operator, so a per-stage forward-vs-reverse comparison on the individual kernels
   would support the same claim without needing the whole call jitted.

## What is already committed

`bench/differentiability/autodiff_overhead.py` (records `mode`, `jit_skipped`,
`jit_compile_wall`, and per-target ratios; never joins a jitted point to an eager
one) and `examples/jaccpot_paper/fig_12_autodiff_overhead.ipynb`, which is generated
and dry-run against a smoke artifact. Both are ready for whichever route above wins.

Figure 13 is unaffected and delivered: it needs gradient *accuracy*, not timing, and
eager dispatch is fine for that.

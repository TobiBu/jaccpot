# Figure 12 (autodiff overhead) is blocked on a compile wall

**Status 2026-08-01: the bench script and notebook are done and tested; the
measurement is not deliverable on this hardware.** Recording why, with numbers, so
the next attempt starts from evidence rather than repeating it.

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

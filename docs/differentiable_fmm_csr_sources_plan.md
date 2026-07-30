# CSR sources for the near-field reverse — measured go/no-go

**Status: investigation, not implemented.** This is the decision record for the
last open performance lever named in
[`differentiable_fmm_design.md`](differentiable_fmm_design.md): replacing the
padded source *rectangle* with a CSR (ragged) source list in the analytic
leaf-pair reverse.

**Recommendation: conditional on your target geometry — and the condition is
met for one of them.**

| target | post-tiering residual | verdict |
|---|---|---|
| clustered disc (galaxy), N ≤ 200 000 | 1.16× | **decline** |
| clustered disc (galaxy), N = 1 000 000 | 1.78× | **decline, but close** |
| uniform cube (cosmological), N = 200 000 | 1.73× | decline |
| **uniform cube (cosmological), N = 1 000 000** | **2.50×** (raw ceiling **16.8×**) | **implement, if this is a target** |

The payoff is real but **dominated by geometry, not by N alone**, and it is
largest exactly where the already-shipped tiering heuristic also fires — so the
figure of merit is the *residual after tiering*, not the raw `1/fill`. On the
clustered disc that every galaxy-scale number in the design doc was measured on,
the residual stays below 2× even at 1 M. On a uniform distribution at 1 M it is
2.5× against a 16.8× raw ceiling, and the trend with N is steep.

So: **if Jaccpot's differentiable path is only ever pointed at galaxy discs, do
not build this.** If cosmological volumes are on the roadmap, build it — and note
that the two cheaper levers in "Why not implement it now" below help there too and
should be tried first regardless.

---

## The claim being tested

The prepacked leaf-major payload is a rectangle `(leaves, slots)` whose width is
the **global maximum** neighbour count. The design doc established:

* `max_nbrs == leaves - 1` at every N and every geometry tried — one leaf
  neighbours every other leaf and sets the width for all of them;
* fill was 45.3% at N=200 000 and 14.5% at N=1 000 000;
* the reverse tracks **padded** slots, not valid work — padding grew 40× for a 5×
  increase in N while valid pairs grew 12.8×, and the measured reverse grew 38.3×;
* therefore padded particle-pair work is `leaves × slots × leaf² ≈ (leaves ×
  leaf)² = N²`, i.e. *the padded reverse is running a direct sum*.

If all of that holds, CSR sources should buy `1/fill` — up to ~6.9× at 1 M.

## What the measurement adds

`bench/audit_nearfield_padding.py` reads the padding structure straight off a
prepared state, so the ceiling can be read without paying for a reverse pass.
A100, `basis="real"`, θ=0.7, order 4:

### leaf_size = 256 (the production preset)

| N | geometry | leaves | slots | fill | max nbrs | median nbrs | max == leaves−1 | CSR ceiling | tier gain |
|---|---|---|---|---|---|---|---|---|---|
| 50 000 | cube | 196 | 200 | **52.9%** | 195 | 100 | yes | 1.89× | 1.39× |
| 200 000 | cube | 782 | 784 | **21.8%** | 781 | 152 | yes | 4.58× | 2.64× |
| 50 000 | disc | 196 | 200 | **96.2%** | 195 | 195 | yes | 1.04× | 1.01× |
| 200 000 | disc | 782 | 784 | **80.8%** | 781 | 666 | yes | **1.24×** | 1.07× |
| 1 000 000 | cube | 3907 | 3912 | **5.9%** | 3906 | 192 | yes | **16.82×** | 6.74× |
| 1 000 000 | disc | 3907 | 3912 | **39.9%** | 3906 | 1569 | yes | **2.51×** | 1.41× |

### leaf_size = 64

| N | geometry | leaves | slots | fill | max nbrs | median nbrs | max == leaves−1 | CSR ceiling | tier gain |
|---|---|---|---|---|---|---|---|---|---|
| 20 000 | cube | 313 | 312 | **38.5%** | 312 | 113 | yes | 2.60× | 1.78× |
| 50 000 | cube | 782 | 784 | **20.2%** | 781 | 139 | yes | 4.96× | 2.74× |
| 20 000 | disc | 313 | 312 | **95.1%** | 312 | 304 | yes | 1.05× | 1.02× |
| 50 000 | disc | 782 | 784 | **74.1%** | 781 | 609 | yes | 1.35× | 1.10× |

`max == leaves − 1` reproduces in **every** row, confirming the earlier finding
that a single pathological leaf sets the width. But the conclusion usually drawn
from it does not follow.

### The finding that changes the decision

**Fill is dominated by geometry, not by the pathological leaf.** At N=200 000 and
leaf 256 — the same leaf count, the same MAC, the same everything but the particle
distribution — a uniform cube fills 21.8% of its rectangle and a clustered disc
fills 80.8%. That is a 3.7× spread in the CSR ceiling (4.58× vs 1.24×) from
geometry alone.

The reason is visible in the median column: the disc's *median* leaf neighbours
666 of 781 leaves, so its rectangle is nearly dense. There is barely any padding
to remove. The cube's median leaf neighbours 152, so most of its rectangle really
is padding.

The previously recorded 45.3% fill at N=200 000 sits inside this range, between
the two geometries measured here — consistent with it, and evidence that the
figure is a property of the particle distribution rather than of the payload
representation. **A single number for "the fill" does not exist**, and any CSR
decision taken from one is taken from a sample of one geometry.

**The spread widens with N, and that is what decides the case.** Fill falls on
both geometries as the leaf count grows, but far faster on the uniform one:

| geometry | 50 000 | 200 000 | 1 000 000 |
|---|---|---|---|
| cube | 52.9% | 21.8% | **5.9%** |
| disc | 96.2% | 80.8% | 39.9% |

At 1 M the uniform cube's rectangle is 94% padding. Its median leaf neighbours 192
of 3906 leaves while the widest neighbours all of them — the pathological-leaf
story, but now with three orders of magnitude between median and max. On the disc
at the same N the median is still 1569 of 3906, so the rectangle remains 40% full.

**Where CSR wins, tiering already wins.** The two columns track each other closely:
tiering captures 1.39× of an available 1.89×, and 2.64× of 4.58× — 58–74% of the
reduction, on the rows where there is any to capture. Tiering is implemented,
tested, on by default, and declines itself when it would not pay. CSR has to beat
*the residual after tiering*, not the raw ceiling:

| N | geometry | ceiling | tiering | **residual** |
|---|---|---|---|---|
| 200 000 | disc | 1.24× | 1.07× | **1.16×** |
| 200 000 | cube | 4.58× | 2.64× | **1.73×** |
| 1 000 000 | disc | 2.51× | 1.41× | **1.78×** |
| 1 000 000 | cube | 16.82× | 6.74× | **2.50×** |

Tiering's share *falls* as fill drops — it captures 86% of the available reduction
on the 200 000 disc but only 40% on the 1 000 000 cube — which is precisely why
the extreme-sparsity corner is the one where a real CSR structure earns its keep.
(Note tiering is evaluated here at `min_gain=1.0`; in production it declines below
3.0×, so on the two disc rows and the 200 000 cube row it would not engage at all
and the residual *is* the full ceiling. That does not change the verdict — those
ceilings are 1.24×, 1.73× and 2.51× — but it is the honest comparison.)

**And the ceiling is optimistic.** `1/fill` assumes ragged access costs the same
per valid pair as rectangular access. It does not: the shipped code has already
measured, twice, that changing the access pattern in this kernel costs more than
the arithmetic it saves — a global occupancy sort was **~7× slower** because it
destroyed the Morton-order locality of the source gather, and tiering itself
"costs throughput ... so it only pays once the saving is large". CSR gives up more
regularity than tiering does, not less.

---

## Why not implement it now

*(This section applies to the galaxy-disc target. For the uniform-1M case see the
recommendation table above — there the residual clears the bar.)*

1. **On the clustered disc, the residual is 1.16× at 200 000 and 1.78× at
   1 000 000** — *before* paying anything for ragged access.
2. **Tiering already captures 40–86% of the available reduction**, at a fraction
   of the implementation and maintenance cost.
3. **The ceiling is an upper bound that this kernel has twice failed to approach**
   when regularity was traded away.
4. **Two cheaper levers dominate it**, both already identified and neither tried:
   - **Leaf compactness / the MAC.** Morton-range leaves are not spatially
     compact — a measured leaf half-extent of `[1.0, 0.992, 0.25]` in a `[-1,1]`
     box spans the whole domain in x and y. This is *upstream* of padding, of
     valid work, and of the forward (which is 77% near field at 200 k). Fixing it
     shrinks the numerator instead of packing the denominator. Accuracy is safe
     either way: a compact node makes the multipole expansion *more* valid.
   - **Splitting the pathological leaf out of the width.** `max == leaves − 1`
     comes from ~1% of real pairs. A two-tier split that puts the outliers in
     their own full-width pass is most of CSR's benefit for a fraction of the
     work — and it is a parameter change to the existing tiering, not a new data
     structure.

## What would flip this

Implement CSR sources if **any** of these is measured:

The bar used here is a **post-tiering residual above ~2×** — the honest figure of
merit, since tiering already exists and is free.

* **MET: uniform-like geometry at N ≳ 10⁶.** 5.9% fill, 16.8× ceiling, 2.50×
  residual, and the trend with N is steep (52.9% → 21.8% → 5.9%). Cosmological
  volumes look like this. **This is the case that justifies building it.**
* **NOT met, but close: clustered disc at N = 10⁶.** 39.9% fill, 2.51× ceiling,
  1.78× residual. Worth re-measuring if the target N grows past 10⁶.
* **NOT met: everything at N ≤ 200 000**, either geometry.
* **Re-measure after any MAC or leaf-compactness change.** Compact leaves mean
  fewer near-field neighbours, lowering *both* valid and padded work — but if the
  variance across leaves grows, fill falls and CSR becomes more attractive, not
  less.

## If it is implemented

Notes for whoever does it, so the known traps are not re-discovered:

* **Keep Morton order in the source ids.** The one measured catastrophe in this
  kernel came from permuting them. CSR must be ragged *only* in extent, never in
  ordering.
* **Segment boundaries must be static.** They ride through the `custom_vjp` in
  `nondiff_argnums`, so they have to be built on the **host, in NumPy**, from the
  concrete frozen topology — exactly as `build_leafpair_reverse_tiers` and
  `build_leaf_major_nearfield_payload` do. Inside the `bwd` rule the validity mask
  is a residual, hence a tracer, and no static extent can be read from it. Both
  failure modes (`ConcretizationTypeError`, then `UnexpectedTracerError` from
  jnp arrays leaking across traces via a memo) are already pinned by tests.
* **Gate on the loss checksum, not on timing.** The grouped-M2L episode is the
  precedent: an assumed equivalence turned out to be 7.1e-2 vs 1.1e-2 against an
  exact sum. A CSR reverse must reproduce the rectangle reverse's gradient to
  round-off before any timing is read.
* **Measure in context and at the target scale.** Isolated micro-benchmarks have
  mispredicted this subsystem by 30×, then by 6–20×; three of four attempted
  optimisations were predicted to win and did not.

## Reproducing

```bash
python bench/audit_nearfield_padding.py --n 20000 50000 --geometry cube disc
python bench/audit_nearfield_padding.py --n 50000 200000 --geometry cube disc --leaf 256
python bench/audit_nearfield_padding.py --n 1000000 --geometry cube disc --leaf 256
```

The script derives fill, the occupancy histogram, and the tier plan the shipped
heuristic would pick, without running a reverse pass — so the ceiling can be
re-checked on new hardware or a new geometry in minutes rather than hours.

# The distributed cross-domain far field was wrong, and why

> **RESOLVED.** The defect is fixed upstream in **TobiBu/yggdrax#47**, and jaccpot
> dropped the `theta_cross` knob that existed to compensate for it (one `theta` now
> serves both walks). Measured through the driver on the configuration whose four
> assertions this note was written to explain: **0.018223 -> 0.000003** at the default,
> and **0.260355 -> 0.000006** at `theta=1.0` with the far path still engaged. The two
> strict xfails that pinned the bug are gone, because both cases pass. Everything below
> is kept as the diagnosis: it is the record of what was wrong, how it was localised,
> and what would say it had come back.

**Verdict: a real defect, in the cross-domain MAC's source extents.** Four of the five
`tests/distributed/` failures found when that tier was first executed on two cards
(2026-08-21, audit row F34) were one bug. Their assertions were correct and their
tolerances were not touched. The fifth failure
(`test_driver_auto_scale_caps`) was unrelated and a test-construction defect; it was
fixed in the same change as this note.

The root cause was in **yggdrax**, not jaccpot, so the change that first landed here
touched nothing in the force path. What it added is that the defect is reproducible in
seconds without a GPU, and pinned by tests that run in CPU CI.

```
tests/integration/test_distributed_cross_domain_far_extents.py   the pin
bench/diagnose_cross_domain_far.py                               the evidence
```

## In one paragraph

`build_coarse_frontier` reduces every remote leaf to a single point — its centre of
mass — and `build_remote_coarse_tree` builds the coarse tree over those points and
computes its geometry from them. The MAC extent the cross walk divides by therefore
bounds the **centres of mass**, not the particles behind them: a coarse *leaf* presents
an extent of ~0 however large the remote leaf it stands for actually is. The
cross-domain MAC consequently accepts pairs whose true separation is smaller than the
source's own radius, and the M2L is then evaluated inside the region it is expanding,
where the series does not converge. The error exceeds the term being approximated,
which is why **dropping the cross far field entirely is more accurate than computing
it**.

## Reproducing it (no GPU)

The failure is geometric, not hardware-dependent. Two forced CPU devices reproduce the
two-A100 aggL2 to six digits, so the whole diagnosis runs anywhere in about two
minutes:

| `theta_cross` | 2 × A100 | 2 forced CPU devices |
| --- | --- | --- |
| 1e6 | 10.090450 | 10.090412 |
| 1.0 | 0.260355 | 0.260355 |
| 0.1 (default) | 0.018223 | 0.018223 |
| 0.01 | 0.000003 | 0.000003 |
| 0.001 | 0.000003 | 0.000003 |

```bash
python bench/diagnose_cross_domain_far.py               # everything
python bench/diagnose_cross_domain_far.py --skip-driver # geometry only, seconds
python bench/diagnose_cross_domain_far.py --inflate     # with the correction applied
```

`ndev=2`, `per=64`, `N=128`, order 3, dehnen, leaf 8 — the failing default, with the
caps grown so an overflow can never be mistaken for a numerical result
(`overflow=False` on every row).

## The IC does not do what its docstring says

Both failing test files construct `ndev` clusters of radius 0.5 spaced 6 apart and
document them as "one cluster per Morton domain, so cross-domain interactions are
genuinely far-field". **That holds only at `ndev=4`:**

```
ndev=2: dev0=[38, 26]  dev1=[26, 38]                        <- INTERPENETRATING
ndev=3: dev0=[29, 26, 9]  dev1=[35, 8, 21]  dev2=[0, 30, 34] <- INTERPENETRATING
ndev=4: dev0=[64,0,0,0] ... dev3=[0,0,0,64]                  <- one cluster per domain
```

The global box is 7 × 1 × 1, so the leading bits of the Morton code are not the axis
that separates the clusters, and a contiguous split of the Morton order cuts across
them. At `ndev=2` the cross-domain field is **74.8%** of the local field, not the 1.0%
it is when the domains really are separated.

This is not the bug, and it is not grounds to change the IC. A domain decomposition
does not have to be convex, and the FMM owes correct forces for any partition;
interpenetrating SFC domains are exactly what happens at every real domain boundary.
The interleaved IC is simply the configuration that exposes the defect. The docstrings
were corrected, the ICs and tolerances were not.

## What is *not* wrong

Each of these was measured and eliminated before the extents were suspected.

- **Halo import, cross P2P, decomposition and reassembly.** At `theta_cross <= 0.01`
  every cross pair resolves near and the forward equals the exact direct sum to
  **3e-6** — the float32 round-off floor. Nothing about shipping and summing remote
  particles is wrong.
- **The M2L, the coarse seeding, the M2M, the L2L cascade and the L2P.** Replayed
  single-device against the exact cross-domain direct sum with *separated* domains:
  **1.4e-3** relative, an order 3 expansion's truncation error at the separations the
  MAC accepts. The coarse root's monopole matches the remote domain's total mass to
  2e-6 (79.78582 vs 79.78598, float32), and seeding the coarse tree reproduces the
  remote tree's own root multipole to the same 1.4e-3 — so the frontier → coarse tree
  → M2M chain conserves mass and reproduces the expansion it stands in for.
- **The near/far partition.** The cross walk emits `accept` or `near` for a pair and
  never both, and never refines an accepted pair further. With separated domains, far
  + near reproduces the exact cross field at **every** `theta_cross`: 1.4e-3 at 1e6,
  1.4e-3 at 0.1, exact at 0.01. Nothing is double-counted.
- **A sign error.** Both paths form the displacement as target-minus-source
  (`_m2l_chunk_contributions` for the self list, `centers[xt] - c_centers[xs]` for the
  cross list). `2 x 0.008814 = 0.017628` versus the measured `0.018223` is a
  coincidence of this IC, not a doubling: the cross field is 74.8% of the local field
  here, not 0.88%.
- **The basis and the rotation.** `basis="real"` and `basis="solidfmm"` agree to eight
  digits (0.03970400337 vs 0.03970400306) because at order 3 they are the same
  expansion in two representations. The error is structural, and identical numbers
  across the two bases is evidence of that, not of a shared bug.

## The mechanism, measured

With the real (interpenetrating) `ndev=2` Morton domains, per accepted far pair:

| `theta_cross` | far | near | relerr(cross) | true `(r_src+r_tgt)/d` | as the MAC computed it | overlapping pairs |
| --- | --- | --- | --- | --- | --- | --- |
| 1e6 | 1 | 0 | 21.27 | 7.263 | 6.441 | 1/1 |
| 1.0 | 19 | 28 | 0.6416 | 2.704 | 0.998 | 19/19 |
| 0.1 | 12 | 51 | 0.0522 | **1.193** | **0.104** | 2/12 |
| 0.01 | 0 | 64 | 0.000000 | — | — | 0/0 |

At the default `theta_cross=0.1` the walk accepts a pair whose true extents put the
target **inside** the source's own radius (ratio 1.193 > 1) while the MAC computes that
same pair at 0.104 — an 11× understatement, and just under the 0.1 threshold it is
being tested against. The reason the understatement is so large is visible directly:

```
true radius of a coarse "point" source:  min 0.4264   median 0.5327   max 5.0473
```

An interleaved domain has leaves holding particles from *both* clusters, so one "leaf"
spans 5 units of space. The coarse tree represents it as a single point of extent zero.

A detail whoever writes the fix needs: the MAC does not literally divide by that zero.
`_build_mac_extents` routes leaves through `_compute_leaf_effective_extents`, which
substitutes a **depth-based** padding (`root_extent / 2**(depth+1)`) for any leaf whose
extent is `<= 0`. That is what produces the 0.104 above rather than 0.0 — a heuristic
standing in for a bound, with no relation to the extent it is standing in for. It also
means the fix composes correctly rather than being overridden: once the coarse extents
are inflated they are strictly positive, so the `extents <= 0.0` branch stops firing and
the real bound is what the MAC tests.

The same replay with separated domains never overlaps (worst ratio 0.199 over 47
accepted pairs) and stays at 1.4e-3, which is the counterfactual: same code, same
walk, same order — only the domain geometry differs.

### Two isolation levels, two numbers — do not conflate them

The table above measures `far + near` against the **whole** cross field, because that
is what the driver's assertion sums. `tests/integration/test_distributed_cross_domain_far_extents.py`
measures something tighter: the far term alone, against the direct sum over **exactly
the pairs the walk accepted as far**. Dropping the near pairs from both sides removes
the dilution and the defect shows up at full size:

| | far + near vs whole cross field | far alone vs its own pairs |
| --- | --- | --- |
| separated | 0.001437 | 0.001535 |
| interpenetrating | 0.052164 | **1.042014** |

The right-hand column is the honest statement of severity: the far term is off by more
than 100% of its own reference, so it is not an inaccurate approximation of the cross
field — it is uncorrelated with it. Every step down the chain (whole field 1.8e-2 →
cross field 5.2e-2 → far term alone 1.04) is dilution being removed, not a different
defect.

The replay is faithful. Its pair counts reproduce the driver's own diagnostics exactly:
12 far and 51 near for device 0 at `theta_cross=0.1`, against the driver's
`cross_far_pairs: [12, 10]` and `cross_near_pairs: [51, 51]`.

## It accounts for the whole aggregate error, exactly

Running the replay from *both* domains' points of view and summing the absolute
cross-field errors in quadrature reproduces the driver's aggregate at **every**
`theta_cross`, to six digits. `||direct||` over all 128 particles is 2855.2100.

| `theta_cross` | abs err, domain 0 | abs err, domain 1 | predicted aggL2 | driver, measured |
| --- | --- | --- | --- | --- |
| 1e6 | 21069.3691 | 19649.7305 | 10.090412 | 10.090412 |
| 1.0 | 635.6215 | 385.4626 | 0.260355 | 0.260355 |
| 0.1 | 51.6783 | 6.0419 | **0.018223** | **0.018223** |
| 0.01 | 0.0001 | 0.0001 | 0.000000 | 0.000003 |

Nothing else contributes: the local field, the near field, the halo and the reassembly
leave no residual at any opening angle. The per-domain far-pair counts (12 and 10) match
the driver's `cross_far_pairs: [12, 10]` as well. The failing assertion is measuring
exactly one thing, and this is it.

`bench/diagnose_cross_domain_far.py` prints this table (both domains plus the
quadrature) next to the driver sweep it has to match.

One detail from the two-sided run worth keeping: domain 1's view of domain 0's frontier
has a *median* true coarse-source radius of 2.1050, against 0.5327 the other way round.
The interleaving is not symmetric, and the domain whose "leaves" are most spread out is
not the one with the larger error — the error tracks which pairs the MAC accepts, not
the spread alone.

## The workaround already in the tree

`docs/north_star_phase3_scale_plan.md:132` says:

> θ_cross ≤ 0.1 (under-separated far-field goes garbage).

That is this defect, found empirically and worked around by shrinking `theta_cross` to
a quarter of `theta` (0.1 against 0.4). There is no physical reason for the cross walk
to need a stricter opening angle than the local walk — both are the same MAC on the
same expansion order. It is compensation for extents that are understated, and the
compensation is incomplete: it holds where the domains are separated and fails where
they interpenetrate. The same doc's note that the rising aggL2 with `ndev`
(3.2e-3 / 7.2e-3 / 1.8e-2 / 2.6e-2 at `ndev` 2/3/4/5) "is the cross-domain far field"
is correct in attribution; those runs used `per=1000` and the treecode local walk, a
different configuration from the failing tests, so they are not directly comparable
with the numbers above.

## The fix, and what it is worth

`CoarseFrontier` should carry each leaf's radius about the centre of mass it is reduced
to, and `build_remote_coarse_tree` should inflate the coarse geometry's `radius` and
`max_extent` by the largest such radius among the coarse particles a node holds. It
costs one extra float per leaf in the frontier all-gather.

Both are in **yggdrax**, `yggdrax/distributed/let.py` — `CoarseFrontier` and
`build_coarse_frontier` at `:44`/`:61`, `build_remote_coarse_tree` at `:243`. Verified
present in yggdrax `main` at cb9cfe8 (the installed copy in jaccpot's venv is
byte-identical to that checkout), so this is a live upstream defect and not a stale
pin. Nothing in jaccpot can fix it at the right layer: jaccpot only receives
`rct.geometry` and passes it to the walk, so a jaccpot-side correction would be
patching a dependency's output from outside and would silently diverge the day yggdrax
fixes it too. The per-node bound is a plain bottom-up max over the coarse tree
(`maxr[internal] = max(maxr[left], maxr[right])`), so it costs one level pass and no
extra communication beyond the single float.

Measured by rerunning the replay with exactly that correction applied, interpenetrating
domains, cross-walk angle 1.0:

| | far pairs | relerr(cross) | true ratio | as the MAC computes it |
| --- | --- | --- | --- | --- |
| COM-only extents | 19 | 0.641596 | 2.704 | 0.998 |
| bounding the particles | 10 | **0.000008** | 0.176 | 0.179 |

Five orders of magnitude, with the far path still engaged, and the MAC's belief now
tracking the truth instead of understating it. Separated domains are unaffected
(1.4e-3 before and after).

## `theta_cross` is gone

The paragraph that used to sit here said the knob "should be revisited once the fix
lands". It was, and the answer was to delete it. Swept through the driver against the
fixed extents, on the same IC (`cross_near` is halo traffic — every near pair ships a
remote leaf's particles):

| `theta` applied to both walks | 0.1 | 0.2 | **0.4** | 0.6 | 1.0 | 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| aggL2 | 3e-6 | 3e-6 | **6e-6** | 6e-6 | 6e-6 | 0.227 |
| cross far pairs | **0** | 1 | **18** | 18 | 18 | 16 |
| cross near pairs | 128 | 127 | **106** | 106 | 106 | 47 |

At the old `theta_cross=0.1` the walk accepts **zero** far pairs: the entire cross field
goes through the halo import, for no accuracy gain over 0.4. Accuracy holds at the
float32 floor to 1.0 and breaks at 2.0 — correctly, since an opening angle of 2 is not
well separated however honest the extents are. So `theta` alone, at its existing 0.4,
is both the accurate and the cheap choice, and a second knob only invited the mistake
of "tuning" it. See `DistributedFMMConfig` for the note that now guards against
reintroducing it.

One cost note survives the change: honest extents reject pairs the MAC used to accept,
and those become halo imports. The table above is one N=128 IC and it saturates at 18
far pairs from 0.4 upward; the effect has **not** been measured at production N.

## What would change the answer

- A regression in the coarse extents upstream. The pin in
  `tests/integration/test_distributed_cross_domain_far_extents.py` reads
  `frontier.radius` and passes it to `compute_tree_geometry` exactly as the fixed
  builder does, so it tracks production rather than freezing either behaviour; if it
  goes red, read this note before touching the test.
- A change to `partition_for_devices` that makes Morton domains spatially compact would
  hide this class of defect on this IC. The pin's `interpenetrating` case is built
  *from* the partitioner, so it would have been fooled the same way — it therefore
  asserts the geometry it is named for (each domain holds particles on both sides of
  x=3) before it measures anything, and fails with "the interpenetrating case is no
  longer interpenetrating" rather than flipping to a false pass.
- Any reintroduction of a separate cross-domain opening angle. The table above is the
  reason it is one knob; a second one hides extent bugs instead of fixing them.

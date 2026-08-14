# G.9 diagnosis: why one gradient element drifts on GPU

Answers `docs/handoff_g9_grad_golden_gpu.md`. Diagnostic only — **no golden, tolerance or
library code was touched**, and no PR is opened, per that handoff.

**Verdict: H1, benign.** The drift is ordinary float64 reassociation, bounded at **55 ULP**
in the norm that matches how the error actually scales. The failure is an artefact of the
gate's *norm*: it applies an elementwise relative tolerance to a per-particle **vector**
whose round-off is proportional to the vector's magnitude, not to each component's own
magnitude. `clu_real_n128_p4` is the one case where a component small enough relative to its
vector coincides with the geometry that produces the largest round-off.

Environment: A100-PCIE-40GB (sm_80), jax/jaxlib 0.10.2, `JAX_ENABLE_X64=1`, `-n 0`,
`use_pallas=False` (the golden's own configuration). CPU rows are the same box's CPU backend.

---

## 1. The headline

`grad_positions`, `clu_real_n128_p4`, against the committed golden:

| | GPU | Linux CPU |
|---|---|---|
| max **relative** drift | **7.936e-12** at element **(57, 2)** | 7.783e-15 at (71, 1) |
| max **absolute** drift | 2.146e-09 at (94, 2) | 5.821e-11 at (4, 0) |
| elements rel > 1e-13 | **5** / 384 | **0** / 384 |
| elements rel > 1e-12 | **1** / 384 | 0 / 384 |
| elements with any drift | 368 / 384 | 190 / 384 |
| relative-drift tail | 7.94e-12, 3.82e-13, 1.43e-13, 1.02e-13, 1.01e-13, 7.85e-14 … | 7.78e-15, 4.72e-15, 4.39e-15 … |
| median nonzero relative drift | 1.390e-15 | 2.193e-16 |

`grad_masses` is **not** clean, contrary to the earlier report: its worst relative drift is
1.994e-13, also at **particle 57**. It stays under the gate, so the test never flagged it.

The handoff's discriminator was the count above 1e-13. The answer, **5**, is neither of the
two cases it anticipated: not "sixty" (H1 nearly proved) and not "one, with the next largest
at 1e-16" (H1 dead). It is a smooth tail with one element crossing — which required
identifying the mechanism rather than reading it off the scale.

## 2. Interaction lists are bit-identical across devices

Handoff item 3, which outranks H1 and H2 if it fails. It does not fail:

| | CPU | GPU |
|---|---|---|
| `num_far_pairs` | 72 | 72 |
| sorted pair array, sha256 | `e54087e3af64306fae0cfb5b9e2f87f6` | `e54087e3af64306fae0cfb5b9e2f87f6` |

Same accepted M2L set, same order. No MAC flip, no near/far reclassification.

## 3. H2 is ruled out — the particle is nowhere near a guard

Particle 57 at `(0.2169, 0.5395, 0.6705)`:

| quantity | value | rank of 128 |
|---|---|---|
| nearest-neighbour distance | 2.606e-02 (2.6× the softening) | 10 (0 = closest pair) |
| neighbours inside the softening | **0** | — |
| smallest `rho = sqrt(dx²+dy²)` over its deltas | 1.811e-02 | 45 (0 = most degenerate) |
| `rho/\|dz\|` at that delta | **0.966** | — |
| fp64 analytic-branch band | `rho/r <= sqrt(eps)` = **1.490e-08** | — |

`rho/|dz| = 0.966` against a band of 1.49e-08 puts it **six to seven orders of magnitude**
away from the G.10-style transverse-degeneracy crossover, and its degeneracy rank is a median
45/128. It is a geometrically unremarkable particle. D.6/G.10 are not implicated.

Direct-sum cancellation is also excluded. Decomposing the direct-sum gradient into per-pair
contributions analytically (verified against `jax.grad` to 1.05e-15) and measuring
`kappa = sum|terms| / |sum terms|`:

| | value |
|---|---|
| particle 57's kappa | **1.687** — 1× the median |
| its kappa rank | **126 of 128** (0 = worst-cancelling) |
| corr(log kappa, log drift) over 128 particles | **−0.016** |
| drift predicted from kappa × eps_f64 | 3.7e-16 vs 7.9e-12 observed — four orders short |

## 4. The mechanism

Particle 57's gradient vector and its drift, component by component:

| component | value | absolute drift | relative drift |
|---|---|---|---|
| x | −1.6888e+05 | 2.125e-09 | 1.258e-14 |
| y | +8.7472e+04 | 3.492e-10 | 3.993e-15 |
| **z** | **−2.6187e+02** | **2.078e-09** | **7.936e-12** |

The absolute drift is the same order on all three components, because it is set by the
magnitudes of the terms summed — which the three components share. The vector norm is
1.9019e+05, the **6th largest of 128**, while the z-component is **726× smaller than the
norm**. The elementwise relative test divides a common absolute drift by each component's own
value, so z inherits ~726× the relative error of x.

Restating the same data in the norm that matches how the error scales:

| statistic | GPU max | in ULP | CPU max | in ULP |
|---|---|---|---|---|
| `\|drift\| / \|g_element\|` — **what the gate uses** | 7.936e-12 | **35,741** | 7.783e-15 | 35 |
| `\|drift\| / \|g_particle\|` | **1.212e-14** | **55** | 1.365e-15 | 6 |
| per-particle `\|Δg\| / \|g\|` | 1.714e-14 | 77 | 1.436e-15 | 6 |

Every one of the top elementwise drifters is a small component of a large vector, and all of
them are flat in the norm-scaled statistic:

| particle | comp | \|g_element\| | \|g_particle\| | ratio | elementwise | norm-scaled |
|---|---|---|---|---|---|---|
| 57 | 2 | 2.619e+02 | 1.902e+05 | **726** | 7.936e-12 | 1.093e-14 |
| 40 | 0 | 2.184e+03 | 1.394e+05 | 64 | 3.824e-13 | 5.990e-15 |
| 115 | 1 | 1.123e+03 | 2.573e+04 | 23 | 1.432e-13 | 6.247e-15 |
| 97 | 1 | 3.154e+03 | 4.900e+04 | 16 | 1.015e-13 | 6.533e-15 |
| 94 | 2 | 3.118e+04 | 1.771e+05 | 5.7 | 6.884e-14 | 1.212e-14 |

## 5. Why only this case — and a prediction of mine that failed

The elementwise maximum is bounded by a product of two factors:

- **R** = max over elements of `|g_particle| / |g_element|` — a property of the committed
  golden, readable without a GPU;
- **D** = max over elements of `|drift| / |g_particle|` — the norm-scaled round-off, a
  property of the device and the geometry.

**I first predicted the pass/fail split from R alone, and it was wrong.** `uni_real_n256_p4`
has R = 2779, four times the clustered case's 726, and passes. Measuring D for all six cases
on the same card supplies the missing factor:

| case | far pairs | R | D | D in ULP | R·D | observed max rel | gate |
|---|---|---|---|---|---|---|---|
| uni_real_n128_p2 | 110 | 126 | 1.94e-15 | 8.8 | 2.45e-13 | 9.28e-14 | under |
| uni_real_n128_p4 | 110 | 174 | 1.94e-15 | 8.8 | 3.37e-13 | 1.48e-13 | under |
| uni_complex_n128_p2 | 110 | 126 | 1.94e-15 | 8.8 | 2.45e-13 | 8.32e-14 | under |
| uni_complex_n128_p4 | 110 | 174 | 1.94e-15 | 8.8 | 3.37e-13 | 1.32e-13 | under |
| **clu_real_n128_p4** | **72** | **726** | **1.21e-14** | **54.6** | **8.80e-12** | **7.94e-12** | **OVER** |
| uni_real_n256_p4 | 1012 | 2779 | 4.05e-15 | 18.2 | 1.12e-11 | 4.47e-13 | under |

R·D is an **upper bound**, tight only where the worst-R element and the worst-D element
coincide — 0.90× in the clustered case, 0.04× for `uni_real_n256_p4`, where the large-R
element is not the one with the largest norm-scaled drift.

The clustered case is the only one with **both** factors elevated: its D is 3× the worst of
the others and its R is 4× the uniform-N=128 cases. Clustering raises D (closer particles →
larger, more-cancelling near-field terms per unit of result), and the small-component element
then converts it into a large *elementwise* relative error.

## 6. One difference at a time

| variant | max rel drift | element | reading |
|---|---|---|---|
| GPU, default XLA | 7.936029e-12 | (57, 2) | baseline |
| GPU, `--xla_gpu_deterministic_ops=true` | **7.936246e-12** | (57, 2) | **unchanged** — not atomics; this is deterministic reassociation, so it is a different phenomenon from the four flaky failures in audit B.3 |
| GPU, `jax.default_matmul_precision("highest")` | 7.908244e-12 | (57, 2) | unchanged (0.4%) — as expected for a float64 case; **no finding about the fp32 pinning** |

### Does the drift follow the geometry or this one seed?

Two further clustered systems at N=128, compared **CPU against GPU** at the same seed — there
is no golden for another seed, so comparing one against the original seed's golden would be
comparing different physical systems. (My first attempt did exactly that and reported a
relative drift of 6.6e+02; the number is meaningless and is discarded.)

| seed | far pairs | R | D | D in ULP | max rel elem | worst element |
|---|---|---|---|---|---|---|
| `0xC0FFEE` (the golden's) | 72 | 726 | 1.22e-14 | 54.8 | **7.94e-12** | p57 c2 |
| `0xBEEF01` | 100 | 101 | 5.57e-15 | 25.1 | 9.66e-14 | p121 c2 |
| `0x5EED02` | 128 | 756 | 5.19e-15 | 23.4 | 6.28e-13 | p35 c0 |

**D is a property of the geometry class, not of one seed**: 23–55 ULP across all three, a
spread of 2.3×. R is what varies widely (101–756).

The decisive row is `0x5EED02`: R = 756, *larger* than the golden seed's 726, and it still
passes — because its D is 2.3× smaller. Crossing the gate needs both factors at their high
end simultaneously, and the golden's seed is that coincidence.

This also shows the elementwise gate has **no margin for this geometry class in general**, not
just at this seed: `0x5EED02` sits 1.6× under the gate and `0xC0FFEE` 0.13× (i.e. over). Under
a norm-scaled gate at 1e-12 all three seeds have 82–193× margin.

## 7. What this means for the gate

`D` is the well-behaved quantity: **≤55 ULP of fp64 across all six cases and both devices**
(CPU ≤6 ULP). The ~9× CPU→GPU increase is what a different reduction order on a different
device costs, and it is not concentrated anywhere — 368 of 384 elements move.

Margin available to each choice of statistic, worst case over the six:

| gate | worst observed | margin at a 1e-12 bound |
|---|---|---|
| elementwise relative (present) | 7.94e-12 | **0.13× — i.e. it fails** |
| norm-scaled `\|drift\|/\|g_particle\|` | 1.21e-14 | **82×** |

Across the six golden cases *and* the two extra clustered seeds, the worst D seen anywhere is
1.22e-14 (55 ULP), so 82× is the margin over every configuration measured here.

And a norm-scaled gate still bites. Tested by mutation, not argued:

| mutation | norm-scaled gate at 1e-12 | elementwise gate at 1e-12 |
|---|---|---|
| the L2P reverse rule × (1+1e-6) — the regression this file exists to catch | **rejects**, 6 orders of margin (D = 9.98e-07) on all six cases | rejects |
| `g[57,2]` × (1+1e-6) — one small component | **rejects** (D = 1.377e-09) | rejects |
| `g[57,2]` × (1+1e-9) | **rejects** (D = 1.377e-12) | rejects |
| `g[57,2]` × (1+1e-11) | misses | rejects (but 7.9e-12 of GPU noise sits on top of it) |

### Recommendation: G.9 option (2), not option (3)

Add the reduction-invariant summary and gate **that** at `1e-12` on every device, keeping the
existing elementwise gate where it demonstrably has margin (CPU: 7.8e-15 observed against
1e-12, i.e. 130×).

Option (3) — gate the elementwise assertion on device — is what I had built before doing this
diagnosis, and it is worse. To pass on GPU it needs an elementwise band around **1e-8**, four
orders looser than the mechanism requires; under it a genuine 1e-9 error in one small
component would go undetected, whereas the norm-scaled gate at 1e-12 rejects exactly that
(row 3 above). Option (1), regenerate on Linux, does not apply: the Linux CPU numbers are
fine.

The residual loss is real and worth stating: a norm-scaled gate cannot see an error confined
to one small component below ~1e-10 relative *on that component*. On GPU that sensitivity is
already unavailable — the device's own floor at that element is 7.9e-12 — so the loss is
confined to CPU, where keeping the elementwise gate preserves it.

## 8. Not concluded

- The physics anchor passing is **not** used as evidence: at `rel-L2 < 2.0e-3` over 384
  elements it is nine orders looser than the inertness gate and would dilute a single bad
  element by √384.
- "It fails on pre-Tier-1 too" establishes only that Tier 1 did not cause it.
- Nothing here is structural, so per the handoff's item 6 there is nothing to stop on.

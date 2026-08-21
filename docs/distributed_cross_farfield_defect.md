# DEFECT: the cross-domain far field's error grows with expansion order

_Measured 2026-08-21, 4xA100, jax 0.10.2, real basis + dehnen MAC, float64,
N=65 536 (16 384/device), leaf 256, `process_block` 256, order/`theta_cross` as
shown. Accuracy is relative L2 against a direct sum over all sources on 1024
sampled targets (`bench/multigpu/harness.py --accuracy-targets`)._

## The observation

| order | `theta_cross` | cross_far pairs | rel-L2 vs direct |
| --- | --- | --- | --- |
| 3 | 0.25 | 3497 | 2.90e-03 |
| 5 | 0.25 | 3497 | 4.54e-03 |
| 8 | 0.25 | 3497 | **1.29e-02** |
| **8** | **0.10** | **0** | **2.06e-07** |
| 3 | 0.10 | 0 | 6.40e-06 |

**With the cross far field engaged, accuracy gets monotonically worse as the
expansion order rises** -- 4.4x worse from p=3 to p=8. That is not physical:
multipole truncation error must fall as theta^(p+1).

It is a controlled comparison. `cross_far` is identical at 3497 across all three
orders, because `theta_cross` fixes the acceptance geometry and only the order
changed. Same interactions, same accept/reject decisions, higher order, worse
answer.

## The control isolates it to the cross path

With the cross far field **empty** (`theta_cross=0.1`, `cross_far=0`, so all
cross-domain coupling is direct-summed), raising the order behaves correctly:

- order 3 -> 6.40e-06
- order 8 -> 2.06e-07 (**31x better**, as theory requires)

So the local far field is fine. And at order 8 the cross-engaged error (1.29e-02)
is **~63 000x** the local-only error (2.06e-07), so with the cross far field on it
contributes essentially all of the error.

## What the signature suggests, and what it rules out

Error *growing* with order means the wrong contribution grows as coefficients are
added, rather than a fixed wrong term. That points at a coefficient
layout/normalisation problem in the coarse path, not at geometry:

- **Not acceptance geometry.** `cross_far` is unchanged across orders.
- **Not far-list truncation.** `xfar_cap` is >= 16 384 against 3497 actual far
  pairs, and an overflow would be reported in `cross_far_overflow` (it is not).
- **Not a plain order mismatch between sweeps.** Both the local
  (`prepare_real_upward_sweep(tree, ..., max_order=p)`) and the coarse
  (`prepare_real_upward_sweep(rct.tree, ..., max_order=p, max_leaf_size=1)`)
  sweeps take the same `p`.

Places worth examining, in order of suspicion:

1. **The coarse-leaf reseeding.** The coarse upward sweep computes point-mass
   multipoles for the frontier, and the coarse leaves are then overwritten with
   the remote leaves' *actual* order-p multipoles from the `all_gather`
   (`gpacked[dom, nod]`). If the remote leaf's expansion centre and the coarse
   leaf's node centre are not the same point, every coefficient above the
   monopole is expressed about the wrong origin and needs an M2M translation.
   The monopole would be unaffected -- which is exactly why the error would grow
   with order. The audit asserts the COMs *are* the coarse expansion centres,
   so this should be consistent; it needs verifying rather than assuming.
2. **The coarse M2M's input convention.** The reseeded leaves bypass the P2M
   that `aggregate_m2m_real_by_level` normally consumes. If P2M output and M2M
   input differ in normalisation or packing, the mismatch scales with order.
3. **Accumulation into the local expansion.** Self and cross contributions are
   summed into one array (`loc_self + segment_sum(x_contribs, xt, total_nodes)`).
   A convention difference between the two M2L call sites would show here.

## The decisive next test

Isolate the cross M2L numerically: for the accepted cross far pairs, compare the
M2L contribution against a direct sum over the source particles those coarse
nodes stand for. That separates "the M2L operator is wrong" from "the multipoles
it is fed are wrong", which the end-to-end force error cannot.

## Why this outranks the rest of the efficiency plan

`docs/multigpu_efficiency_plan.md` has items for relaxing `theta_cross`, adding a
Pallas cross-M2L kernel, and reducing cross work to fix scaling. **All three
depend on a cross far field that computes the right answer.** A far field whose
error grows with order cannot be tuned, cannot be accelerated, and must not be
written up.

It also explains why relaxing `theta_cross` looked like a pure loss earlier: it
was trading exact direct sums for a broken approximation. That measurement should
be repeated once this is fixed, because the trade it appeared to show is not the
trade the method offers.

## Scope note on the configuration measured

At leaf 256 with 64 leaves/device the **local** far field is also nearly empty
(`self_far` = 6-20 pairs against `self_near` = 4026). So the configuration the
scaling figures were measured at is a near-degenerate FMM: a direct sum with
tree-accelerated neighbour finding. That is worth revisiting independently --
jztree, the tree library this work takes inspiration from, defaults to
`max_leaf_size` 32-48, not 8 (our distributed default) or 256 (these benchmarks).

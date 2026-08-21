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

---

# COORDINATION (2026-08-21): same defect found independently; fix belongs upstream

A parallel session reached the same root cause on branch
`diag/cross-far-coarse-extent-mac` (worktree
`.claude/worktrees/reactive-forging-wall`), and its diagnosis is more complete
than this one. Read
**`docs/distributed_cross_domain_far_diagnosis.md` on that branch** as the
primary account; this file is retained for the independent corroboration below.

## Where that analysis is ahead of this one

- **It eliminates the alternatives by measurement**, rather than by reading: halo
  import / cross P2P / decomposition (exact to 3e-6 at `theta_cross<=0.01`), the
  M2L, the coarse seeding, the M2M, the L2L and the L2P (replayed single-device
  against the exact cross direct sum: 1.4e-3, which *is* order-3 truncation), the
  near/far partition (no double counting), a sign error, and the basis.
- **It quantifies the MAC's error directly.** At the default `theta_cross=0.1` the
  walk accepts a pair whose true `(r_src+r_tgt)/d` is **1.193** -- the target lies
  *inside* the source's own radius -- while the MAC computes **0.104**, an 11x
  understatement just under the threshold. True radius of a coarse "point" source:
  median 0.5327, max 5.0473, represented as zero.
- **It explains the real/solidfmm agreement to 8 digits** that this file flagged as
  suspicious: at order 3 the two bases are the same expansion in two
  representations, so identical numbers are evidence the error is structural, not
  evidence of a shared bug.
- **It found the pre-existing workaround.** `docs/north_star_phase3_scale_plan.md:132`
  already says "θ_cross ≤ 0.1 (under-separated far-field goes garbage)". So the
  strict default is compensation for this defect, not a physical choice -- which is
  the real answer to "why not just loosen `theta_cross`".
- **It reproduces without a GPU** (`bench/diagnose_cross_domain_far.py`, two forced
  CPU devices matching 2xA100 aggL2 to six digits) and pins the defect with strict
  xfails in `tests/integration/test_distributed_cross_domain_far_extents.py`.
- **It corrects the IC docstrings**: the "one cluster per Morton domain" claim holds
  only at `ndev=4`. At `ndev=2` the Morton split cuts across the clusters and the
  domains interpenetrate, which is why the four `tests/distributed/` driver
  failures appear at two cards. That supersedes the explanation offered earlier in
  this session, which put those failures down to a tolerance calibrated at four
  devices.

## The fix is in yggdrax, and a jaccpot-side correction was wrong

That branch locates the fix in `yggdrax/distributed/let.py` -- `CoarseFrontier`
must carry each leaf's radius and `build_remote_coarse_tree` must inflate the
coarse geometry -- verified live in yggdrax `main` at cb9cfe8, and argues:

> jaccpot only receives `rct.geometry` and passes it to the walk, so a
> jaccpot-side correction would be patching a dependency's output from outside and
> would silently diverge the day yggdrax fixes it too.

That is correct, and it refutes the correction implemented in this session. A
`_inflate_coarse_geometry` helper was added to `jaccpot/distributed/fmm.py` here
and has been **reverted**; the diff is kept out of the tree at
`scratchpad/jaccpot_side_inflation_REJECTED.patch` for reference only. Double
inflation once yggdrax lands its own bound is exactly the failure mode the
argument predicts.

## Independent corroboration this session adds

The two diagnoses used different configurations, which makes them a genuine
cross-check rather than a repetition. That one is `ndev=2`, `N=128`, leaf 8,
order 3, interpenetrating domains. This one is `ndev=4`, `N=65 536`
(16 384/device), leaf 256, uniform, with an order sweep:

| order | `theta_cross` | cross_far | rel-L2 vs direct |
| --- | --- | --- | --- |
| 3 | 0.25 | 3497 | 2.90e-03 |
| 5 | 0.25 | 3497 | 4.54e-03 |
| 8 | 0.25 | 3497 | **1.29e-02** |
| 8 | 0.10 | 0 | 2.06e-07 |

**Error rising with expansion order at a fixed accepted-pair count** is the
divergent-series signature, and it is a different observable from the extent-ratio
table -- both point at the same cause. The control (order 3 -> 8 improving 31x when
the cross far field is empty) isolates it to the cross path.

Also worth carrying over: the reverted jaccpot-side inflation measured
**5.6-8.2x** better accuracy at `theta_cross=0.25` (order 5: 4.54e-3 -> 8.15e-4;
order 8: 1.29e-2 -> 1.58e-3), with `cross_far` falling 3497 -> 227 as honest
extents reclassified pairs to near. Consistent with that branch's five-orders
result at `theta_cross=1.0`, and with its warning that the correction costs
near-field work. Note the error still *rose* with order after that partial
correction, which suggests a bottom-up bound of
`max(|c_child - c_node| + r_child)` is not equivalent to the simpler
`point_radius + max_represented_radius` that branch specifies -- their form should
be preferred.

## Tooling this session adds, which that branch can use

`bench/multigpu/harness.py --accuracy-targets K` measures relative L2 against a
direct sum over **all** sources on K sampled targets. It is what turns any
`theta_cross` or order sweep into a real measurement rather than a speed
comparison, and it will be needed for the "revisit `theta_cross` once the extents
are honest" follow-up that branch flags -- probably to `theta` itself, since 0.1
exists only to compensate.

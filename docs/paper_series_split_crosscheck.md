# Cross-check: how to split the jaccpot paper series

Hand this to the session that owns the paper-series narrative. It states a
proposed restructuring, the evidence for it, and **a conflict with a split that is
already planned** — the point of the exercise is to reconcile those two, not to
rubber-stamp either.

Everything factual below was read from the repo on 2026-08-24 at branch
`paper/jaccpot-i`. Verify rather than trust.

---

## The question

`jaccpot` is a **force and potential evaluator**. It is static: it computes
accelerations for a configuration, with no notion of time. Time integration lives
in `nornax` (leapfrog KDK, Hermite 4/6/8, adaptive) and above that in `ODISSEO`.

Section 7 of Paper I, the payoff application, currently reserves two figures:

- **fig14** — energy and angular-momentum conservation over a long integration.
- **fig15** — gradient-based parameter recovery from kinematics.

fig14 requires an integrator. So either Paper I restricts itself to the static
case and time integration becomes Paper II, or Paper I acquires a dependency on a
second package for its payoff figure.

## What Paper I currently is

Sections 1–6 and 8 are drafted and every number is traceable to a committed
artifact with `git_dirty=false` and a reachable commit. Section 7 is 2 lines and
empty.

| Section | Live lines | Content |
|---|---|---|
| 1 Introduction | 75 | drafted |
| 2 Method | 167 | formulation, MAC, kernels, decomposition, differentiability |
| 3 Validation | 112 | accuracy vs direct sum, basis cross-check, MAC comparison |
| 4 Complexity | 137 | single-device scaling and performance |
| 5 Multi-GPU | 167 | weak/strong scaling, load balance, distributed gradient |
| 6 Differentiability | 118 | gradient correctness, reverse-pass cost |
| 7 Payoff | **2** | **empty, blocked on code** |
| 8 Discussion | 111 | drafted |

Twelve figures exist with clean provenance (01–09, 11–13; figure 10 has no
mechanism behind it and is deliberately absent).

## The proposed split, and why the dividing line is principled

**Momentum conservation is algebraic; energy conservation is dynamical.** This is
not an arbitrary place to cut.

The mutual lane (`jaccpot/mutual/`) evaluates each pair once and applies $+f/-f$,
so $\sum_i m_i a_i = 0$ **structurally, on a single force evaluation, with no
integrator**. `tests/integration/test_mutual_fmm_static_device.py` asserts drift
below `1e-13` and notes momentum stays exactly conserved even when a canonical
pair is dropped, because dropping it removes both halves.

Energy conservation has no such static form. It is a property of a trajectory.

So the natural cut is:

- **Paper I (static):** force and potential evaluation, accuracy, scaling,
  differentiability of the *force*, momentum conservation as a structural
  property, and a payoff that is a static inference problem.
- **Paper II (dynamic):** jaccpot as the force evaluator inside a time
  integration — energy conservation and drift, timestep and integrator order,
  differentiation through a *trajectory*, and the Kessel Run story.

### Three further arguments for the split

1. **fig14 in Paper I would measure the wrong thing.** Energy drift over a long
   integration is dominated by integrator order and timestep, not by FMM force
   accuracy. In a jaccpot paper it invites the reading that it demonstrates
   jaccpot when it mostly demonstrates nornax's Hermite order. A reviewer asking
   "what would this look like with a worse force?" should get a sharp answer and
   would not.
2. **Paper II's central claim is harder and more interesting**, which is the
   strongest reason it deserves its own treatment. Differentiating through a
   trajectory is not differentiating through a force: gradients through chaotic
   N-body dynamics grow like $e^{\lambda t}$, and there are real questions about
   checkpointing and memory for long rollouts, and about whether adaptive and
   block timesteps are differentiable at all. Compressed into section 7 that is
   either superficial or it unbalances Paper I.
3. **Section 7 works statically anyway.** The agreed design is to express
   kinematics as Gauss–Hermite coefficients — what surveys such as GECKOS reduce
   their data to — and recover parameters from a phase-space snapshot. That is a
   static inference problem and it is exactly the payoff a differentiable force
   evaluator should demonstrate.

---

## THE CONFLICT: a different split is already planned

[`PROJECT_PLAN.md`](https://github.com/TobiBu/jaccpot-paper-i/blob/main/PROJECT_PLAN.md) (in the `jaccpot-paper-i` repo), Phase 5, has an
open checkpoint:

> assess whether the multi-GPU section has grown large enough that it, plus the
> differentiability section, could stand alone as a **systems paper** separate
> from the **accuracy/case-study material**. Decide then, once you see actual page
> budget, not now.

That is a **systems / science** split: {§4, §5, §6} versus {§3, §7}. The proposal
in this document is a **static / dynamic** split: {everything current} versus
{time integration}.

These are orthogonal axes and they produce different papers. They are also not
mutually exclusive — applied together they would give three. **This is the main
thing the series-coordinating session needs to decide.**

Two observations that bear on it:

- The checkpoint says to decide "once you see actual page budget, not now". The
  page budget now exists: 889 live section-lines and twelve figures, with section 7
  still to come. The checkpoint is ripe.
- The systems/science split would leave the science paper thin: §3 (validation)
  plus §7 (payoff) is not obviously a paper on its own, whereas the static/dynamic
  split leaves both halves substantial. But that judgement depends on the series
  plan, which this document does not own.

## A blocker for Paper II that should be checked now, not later

`docs/differentiable_fmm_design.md` records that
**`OdisseoFMMCoupler.accelerations` yields exactly zero gradients, silently** —
it routes to `evaluate_prepared_state`, which takes no live positions or masses.
`OdisseoFMMCoupler` is part of the frozen top-level API (`jaccpot.__all__`).

If Paper II is "jaccpot as a differentiable force evaluator inside ODISSEO", this
is squarely on the critical path, and a silently-zero gradient is the worst
failure mode this project has: it does not raise, and an optimiser fed zero
gradients simply does not move.

Also relevant to feasibility: the same document records reverse-pass **compile**
time of roughly 10 minutes at $N=16384$, growing to 25 minutes at $N=10^6$.
Differentiating through many timesteps compounds whatever that becomes.

The dependency graph is deliberate and acyclic: `Jaccpot → Yggdrax`, `Nornax`
standalone, `ODISSEO → Nornax + Jaccpot`. `BlockStepFMM` satisfies nornax's
`MutualForceModel` protocol *without importing nornax*, which is what keeps it
that way. Paper II should not casually invert this.

## What changes in Paper I if the static/dynamic split is adopted

Small and well-scoped:

1. **§7** becomes static: Gauss–Hermite parameter recovery from a snapshot,
   optionally with a momentum-conservation panel (cheap — the property is
   algebraic and already tested). Retitle from "Payoff application".
2. **fig14 moves to Paper II.** fig15 renumbers.
3. **§1 needs one sentence narrowed.** Its motivation paragraph currently says
   parameters "or of an initial condition, can be fitted to observations by
   descending a gradient through **the dynamics** rather than by sampling forward
   simulations." That promises Paper II's content. The abstract is already safe —
   it says "through an N-body force model", not through dynamics.
4. **§8's future-work paragraph** gains the dynamics companion paper.
5. `docs/section7_payoff_next_session_prompt.md` drops fig14, which also removes
   the open question of whether to install `nornax` into jaccpot's venv.

## Questions for the coordinating session

1. **Which split, or both?** Static/dynamic as proposed here, the planned
   systems/science one, or a three-way? This is the decision everything else
   waits on.
2. **What is the Kessel Run story?** It appears in [`PROJECT_PLAN.md`](https://github.com/TobiBu/jaccpot-paper-i/blob/main/PROJECT_PLAN.md) only as
   "standalone Kessel Run letter" with no description anywhere in the repo. It was
   omitted from §8's future work rather than guessed at. If it is dynamical it
   belongs to Paper II and strengthens the case for the split.
3. **Numbering.** The plan currently reads Jaccpot II = learned MAC, Jaccpot III =
   neural multipole terms, Jaccpot-Science I = real IFU data. Inserting a dynamics
   paper renumbers those. Suggestion: refer to companions by content in the text
   and assign numbers at submission, since referee timelines reorder these anyway.
4. **Does Jaccpot-Science I overlap section 7?** Both are parameter recovery from
   kinematics; one on synthetic data, one on real. If §7 is the synthetic
   proof-of-concept for the same pipeline, that should be stated deliberately
   rather than discovered at review.
5. **ODISSEO or nornax as Paper II's integrator?** nornax already has the
   `JaccpotForceModel` adapter and conservation tests, so it is the lower-friction
   path; ODISSEO sits above it. Whether Paper II is about the whole stack or just
   the jaccpot–nornax coupling is a series-level call.

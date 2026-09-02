# Multi-GPU basis/MAC status and differentiability model

_Fill in before drafting `sections/02_method.tex` — this doc should become
most of that section almost verbatim._

## Multi-GPU basis/MAC status (as of paper submission)

Per `docs/phase5_multigpu_pallas_foldin_plan.md` in the main repo:

- [ ] State which of items 5a (MAC `bh → dehnen`), 5b (basis
  `solidfmm(complex) → real`), 5c (Pallas M2L/P2P per device) have landed by
  the time this section is written.
- [ ] State the accuracy number the distributed path achieves at whatever
  config is shipped (0.19–0.24% vs. direct at 4×GPU as of the last
  engineering-log update — recheck before quoting).
- [ ] If 5a–5c have not fully landed, say explicitly that the paper reports
  the solidfmm/bh baseline and treats fast-lane convergence as future work —
  do not let this drift silently between the plan and the actual text.

## Differentiability model

Same framing as the yggdrax paper (kept consistent across both):

- Tree **topology** (Morton order, node membership, near/far MAC accept/reject
  decisions) is discrete and piecewise-constant — not differentiable, and the
  paper says so explicitly rather than oversell.
- Everything computed **given** a fixed topology (multipole moments, M2L
  evaluations, near-field P2P) is differentiable w.r.t. continuous inputs
  (positions, masses, potential-model parameters).
- There's a measure-zero discontinuity exactly at MAC accept/reject
  boundaries — quantified empirically in `sections/06_differentiability.tex`
  (finite-difference vs. autodiff agreement swept across theta) rather than
  just asserted.

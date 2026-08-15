# Handoff: why one gradient element drifts on GPU (audit G.9)

Paste the block below into a fresh session on the GPU box. It is self-contained.

Why it exists: the A100 validation of Tier 1 (audit B.3) found
`test_fmm_grad_golden[clu_real_n128_p4]` red on GPU — 1 of 384 elements of
`grad_positions` outside `rtol=atol=1e-12`, reproducible 3/3 under **both** default XLA
and `--xla_gpu_deterministic_ops=true`, and failing identically on pre-Tier-1
`128a0e2`. So it is neither nondeterminism nor a Tier 1 regression. G.9 predicted a
cross-platform bit-stability risk on the macOS→Linux CPU axis; that axis held, and this
broke cross-*device* instead.

**This is a characterization tripwire, and it is currently red. The job is to explain
it, not to silence it.**

---

Investigate a single-element gradient discrepancy between CPU and GPU in the `jaccpot`
FMM solver. This is diagnostic work: **I do not want a fix in this session, I want the
mechanism.**

## Hard constraints

Read `CLAUDE.md` and `agent_guides/NUMERICS_AND_JAX.md` first. In particular:

- **Do not touch the golden `.npz`, `INERT_RTOL`, `INERT_ATOL`, or any tolerance.** Not
  to "see if it passes", not temporarily. G.9 says so explicitly: relaxing the bound
  converts the tripwire into a formality on its first day.
- **Do not change what the code computes.** If you conclude something is wrong, say so
  and stop. That is its own PR with its own test.
- Report numbers, not reassurance. "Within noise" is not a finding; a magnitude is.
- Use `autocvd` to select a free GPU (`autocvd(num_gpus=1, least_used=True)`) and pin it
  for the session. The GPUs are shared.

## The failing case

```
tests/characterization/test_fmm_grad_golden.py::test_fmm_grad_golden[clu_real_n128_p4]
```

`("clu_real_n128_p4", "clustered", 128, "real", 4)` — clustered distribution, N=128, real
solid-harmonic basis, order 4, `LEAF_SIZE = 4`. 128 particles × 3 components = the 384
elements. `grad_masses` is **clean**; only `grad_positions` moved.

**The other five cases pass.** Four are `uniform`, one is `uniform` at N=256. This is the
only `clustered` case in the set, and it is the only one that fails. That asymmetry is
the most informative thing in the report and every hypothesis below is a way of
explaining it.

## Two traps, both of which already caught the previous GPU session

1. **`pytest -n auto` is 64 xdist workers against one GPU** on that box: 46,000+
   `CUDA_ERROR_OUT_OF_MEMORY` lines and hundreds of allocation-artifact "failures". Run
   `-n 0` for this work — you want one process and a debugger anyway.
2. **`pytest -q` double-quiets.** `addopts` in `pyproject.toml` already carries `-q`, so
   adding another hides the pass/fail counts.

Also: `JAX_ENABLE_X64=1` is required or the case is a different computation entirely.

## What to find out, in order

### 1. The magnitude. This is the missing number and it decides everything.

The previous report gave "`Mismatched elements: 1 / 384 (0.26%)`" but **not** the max
relative difference, which `np.testing.assert_allclose` prints on the same failure. Get
it, along with which element it is:

```python
# CPU and GPU, same script, jax.config.update("jax_platform_name", ...) or CUDA_VISIBLE_DEVICES=""
idx = np.unravel_index(np.argmax(np.abs(got - golden)), got.shape)   # (particle, component)
rel = np.abs(got - golden) / np.maximum(np.abs(golden), 1e-300)
```

Then read the answer off the scale:

| Relative drift | Reading |
|---|---|
| ~1e-12 to 1e-11 | **H1**, benign. Many elements sit just under the gate and one crossed it. |
| ~1e-8 to 1e-6 | **H2**. Something is resolving a branch or a near-cancellation differently. |
| ≳1e-4 | **H3**, structural. Stop and report; this is not a tolerance question. |

Also report **how many elements are within 10× of the gate**, i.e.
`np.sum(rel > 1e-13)`. If the answer is "sixty", H1 is nearly proved and the finding is
that this golden has almost no margin on GPU. If the answer is "one, and the next
largest is 1e-16", H1 is dead — a lone outlier in an otherwise bit-identical array is not
reassociation.

That single count is the cheapest discriminator available. Get it before anything else.

### 2. Which particle, and where is it?

Map the element index to a particle and characterise its geometry, because the
clustered-only signature suggests geometry:

- its distance to its own leaf-box centre, and to the nearest other particle;
- the smallest `rho = sqrt(x² + y²)` among the deltas it participates in, relative to
  the degeneracy guards in `jaccpot/operators/`;
- its tree depth, and whether it sits in the most populated or most degenerate leaf.

**Why this matters (H2).** Audit G.10 was a real bug of exactly this shape: a
`where(cond, expr, constant)` guard at `rho == 0` silently zeroed two transverse
gradient components, and it reached the force. It is fixed, but the fix works by
selecting between an expression and an analytic limit **near** a threshold — and a
clustered distribution is the one that puts particles into that neighbourhood. If the
drifting element belongs to a particle sitting near a guard crossover, then two devices
choosing different sides of it is the mechanism, and the finding is that the crossover is
sharper than the guard assumes. D.6 and G.10 are the context; read them before
concluding.

### 3. Is the interaction list the same?

The test asserts `num_far_pairs > 0` — **not** that it equals the CPU value. So a
different accepted interaction set is *not* excluded by the test passing that gate.
Print and compare across devices:

- `num_far_pairs` on both;
- the accepted pair list itself, sorted, not just its length;
- whether any pair's MAC quantity sits within ~1e-12 of the acceptance threshold. One
  pair flipping from far to near would change which code path computes an interaction,
  and in a clustered tree a near-threshold pair is far more likely than in a uniform one.

If the lists differ, that is the answer and it outranks H1 and H2.

### 4. Does it survive dropping to one device-visible difference at a time?

Cheap bisection of the cause, each independent of the others:

- CPU vs GPU with `--xla_gpu_deterministic_ops=true` on the GPU side (already known: still
  fails, but confirm in your own harness so you trust it);
- GPU with `jax.default_matmul_precision("highest")` — `operators/_precision.py` pins fp32
  matmul precision deliberately, and this case is float64, so this **should** change
  nothing. If it does, that is a finding about the pinning.
- the same case at order 2 and at `uniform`/N=128, to confirm the failure really is
  specific to `clustered` × order 4 rather than an accident of which cases exist;
- N=128 clustered with a different PRNG key, if `_make_inputs` allows it without editing
  the test — a drift that follows the geometry rather than this exact seed is a much
  stronger result.

## What NOT to conclude

**Do not argue that the physics anchor passing means the GPU number is fine.** I checked:
the order-4 anchor is `rel-L2 < 2.0e-3` on positions, nine orders looser than the
inertness gate, and it is an L2 norm over 384 elements — so a single element could be
badly wrong and be diluted by √384 below the bound. The anchor bounds gross error only.
It passing tells you the GPU result is not garbage; it does not tell you this element is
right.

Likewise, "it fails on pre-Tier-1 too" establishes only that Tier 1 did not cause it. It
says nothing about whether it is benign.

## Deliverable

A short report, numbers over prose:

1. The max relative drift, the element index, and the count of elements above 1e-13. ← the
   headline
2. Which of H1/H2/H3 the evidence supports, and what specifically rules the others out.
3. The particle's geometry, and whether it sits near a G.10-style guard crossover.
4. Whether the interaction lists match across devices.
5. A recommendation among G.9's options, re-read for *device* rather than platform:
   (2) snapshot a reduction-invariant summary alongside the raw arrays, or (3) gate the
   inertness assertion on device. Note that G.9's option (1) — regenerate on Linux — does
   **not** apply: there is nothing wrong with the Linux CPU numbers, CI is green.
6. If you found something structural, say so plainly and stop there.

Do not open a PR. The remediation is a separate change with its own test, and which
remediation is right depends on your answer to (2).

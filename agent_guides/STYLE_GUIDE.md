# jaccpot code style guide

This guide describes the house style already present in the codebase. Agents and
contributors may draft in their own style, but code is **converted back to this style before
it is committed or merged**. Treat this document as the reference for that conversion.

> Rule of thumb: optimize for a physicist reading the code top-to-bottom for the first time.
> Prefer clarity and narration over terseness. The compiler does not care about line count;
> the next reader does. Roughly +20% more code for materially better readability is a good
> trade.

---

## 1. Formatting is not a style question

`black` (line length 88) and `isort` (profile `black`) own all formatting and import
ordering. Both are pinned in `pyproject.toml` to exactly the pre-commit hook revisions, and
CI runs `black --check .` and `isort --check-only .`.

Consequences:

- **Do not hand-order or hand-group imports**, and do not add labelled import-group comments
  — `isort` will fight you and CI will fail.
- Do not argue with `black` about line breaks. Run it.
- Do not change the pins. The comments next to them record real incidents where a floating
  `>=` let CI resolve a different version whose style disagreed with the hook.

Every module starts with `from __future__ import annotations` (66 of 80 modules do; make it
80). Modules declare `__all__` (38 of 80 do; same direction of travel).

---

## 2. Module docstrings

Every `.py` file opens with a `"""..."""` docstring before any imports. For a small helper,
one sentence. For anything substantial, **explain the decisions**, because that is what this
codebase actually does well and what makes it maintainable.

`jaccpot/operators/_precision.py` is the reference example. It answers, in order: what this
is, **why it exists** (XLA lowers fp32 matmuls to TF32, capping M2L relative accuracy at
~6e-04 from order 4 up), **why it is built this way rather than the obvious way** (53 `@`
matmuls across 19 functions; `precision=` would obscure the algebra), and **what it costs**,
including what the honest fix would be if a profile ever puts it on the critical path.

The pattern worth copying:

- Lead with the decision, not the mechanism.
- **Quote the measurement.** "~6e-04 at p=4, 6, and 8 versus a float64 reference" is worth
  more than "less accurate". Numbers survive; adjectives do not.
- Say what the alternative was and why it lost.
- Say what would change the answer, so the next person knows when to revisit.
- Short ALL-CAPS rubric labels (`WHY THIS EXISTS.`, `COST.`) are used to break up long
  module docstrings and are welcome, but not mandatory.

No author/date/changelog lines — that is what git is for.

---

## 3. Function docstrings — NumPy style

**NumPy style, enforced by `pydoclint --style numpy` in pre-commit.** Summary line, blank
line, then `Parameters` / `Returns` / `Raises` / `Notes` / `References` with dashed
underlines. Sphinx cross-references (`:func:`, `:class:`, ``` ``literal`` ```) are used
throughout and should be.

```python
def _m2l_complex_batch_kernel_fused_pallas(src_mult, deltas, *, order):
    """Batched M2L through the single-launch fused Pallas kernel.

    Numerically equivalent to :func:`_m2l_complex_batch_kernel` (the solidfmm
    reference); the kernel is purely an execution accelerator.

    Parameters
    ----------
    src_mult : Array
        Complex multipole coefficients ``[N, (p+1)^2]`` for each pair.
    deltas : Array
        Target-minus-source center displacements ``[N, 3]``.
    order : int
        Expansion order ``p``. Static under ``jit``.

    Returns
    -------
    Array
        Complex local contributions ``[N, (p+1)^2]``.
    """
```

Public and substantial private functions get the full treatment. Tiny helpers may use a
one-line docstring.

For anything numerical, the docstring must carry what the signature cannot:

- **array shapes** in the `[N, (p+1)^2]` notation already used, and the units / conventions
  (G=1? which centre? periodic wrap? real or complex basis?)
- **which arguments must be static** under `jit`
- **whether the function is differentiable**, and in which arguments
- **the accuracy regime and known failure modes** — degenerate separations, empty cells,
  single-particle leaves, particles outside the root box, MAC violations the function does
  not itself check
- **equivalences**: if two paths are meant to produce the same numbers (reference vs. fused,
  real vs. complex basis, fast lane vs. general lane), say so explicitly in both.

Never write a docstring that restates the name. `"""Compute the multipole."""` on
`compute_multipole` is worse than nothing, because it looks like documentation.

---

## 4. Type annotations

- Full annotations on public functions and on internal functions whose types are not obvious.
- **`jaxtyping` for array arguments** (already in 49 modules) so shape and dtype live in the
  signature. Plain `Array` throws away exactly what the reader needs. Keep axis names
  consistent so `jaxtyping` can cross-check within a signature.
- `beartype` provides the runtime check; it is opt-in via `JACCPOT_RUNTIME_TYPECHECK=1`.
  Run it on the unit tests when you touch signatures.
- Use `TypeVar` for decorators that must preserve the wrapped signature (see
  `_precision.py`). A targeted `# type: ignore[return-value]` with an obvious reason is
  acceptable; a bare `# type: ignore` is not.

---

## 5. Comments narrate *why*

Full sentences explaining intent, trade-offs, and caveats — not a restatement of the code.
Multi-line rationale as prose paragraphs. `NOTE:`, `WARNING:`, `TODO:` (with a reason) for
emphasis.

Anywhere the code looks wrong but is right, say so at the site. That is the single
highest-value comment you can write in a numerics codebase, because the alternative is
someone "fixing" it in six months. Where a decision rests on a measurement, put the
measurement in the comment.

Constants that are part of a module's interface get a `#:` comment above them so they
document themselves.

---

## 6. Section dividers for long functions and modules

Long modules and functions are broken up with a plain dashed rule and a label:

```python
# --------------------------------------------------------------------------
# Scoped overrides for gates that are read deep inside the kernels.
# --------------------------------------------------------------------------
```

A unit needing more than about four such sections is usually asking to be split — but split
along conceptual seams (upward sweep / traversal / operator algebra / accumulation), never
by line count. `jaccpot/nearfield/near_field.py` (4182 lines) and
`jaccpot/runtime/kernels/core.py` (3827 lines) are the standing counter-examples; new code
should not grow that way.

---

## 7. Naming — readability first, literature second, brevity last

Names are the primary documentation. Spell things out, even at the cost of a longer line or
a temporary variable.

- `snake_case`, descriptive, no invented abbreviations: `expansion_order`, not `eo`;
  `num_particles`, not `np_`; `source_cell_index`, not `_sci`.
- **Match the literature for mathematical symbols.** The opening angle is `theta`; the
  expansion order is `p` (or `order` in a signature, with `p` in the docstring). Single
  letters are fine where they *are* the standard symbol and the docstring defines them; they
  are not fine for loop-carried state, configuration, or anything a reader would guess at.
- Established domain shorthand that appears in the papers and throughout this codebase —
  `p2m`, `m2m`, `m2l`, `l2l`, `l2p`, `p2p`, `mac`, `vjp` — is correct and should be used, not
  expanded.
- Prefer a longer self-explaining name over a short one plus a comment.
- Avoid leading-underscore *local* temporaries; the leading underscore marks module-private
  module-level names.
- Introducing an intermediate, well-named variable purely to make a dense expression legible
  is encouraged.

---

## 8. Module boundaries

Keep the seams in the layout clean and the import graph acyclic:

- `basis/` and `operators/` — the mathematical algebra. Pure, jittable. No tree logic, no
  config parsing, no I/O.
- `upward/`, `downward/`, `nearfield/` — the sweeps. Know about operators and tree
  artifacts; are not either.
- `pallas/` — accelerated kernels that must stay numerically equivalent to their reference
  counterparts. Equivalence is stated in both docstrings and tested.
- `runtime/` — orchestration, config resolution, lane selection, dispatch. The only place
  that reads environment variables or resolves `"auto"` policies.
- `distributed/` — decomposition, halo exchange, collectives.
- `experimental/` — prototypes. Not production, not held to production standards, not
  imported by production paths.

Physics must not live in a utility module; runtime policy must not leak into operators. If
you find yourself importing "up" that list, report it rather than adding the import.

Function-local imports are used deliberately in a few places to break cycles or defer heavy
Pallas imports (see `runtime/kernels/core.py`). Leave them; do not hoist them to module
scope as a cleanup.

---

## 9. Errors and validation

- Validate at the public API boundary and fail loudly, naming the offending value and the
  expectation.
- When a user explicitly requests a configuration that cannot run, **fail loudly rather than
  silently substituting something else** — this is the existing policy for lane selection in
  `runtime/grad_options.py`, and it generalises. `"auto"` may choose; an explicit request may
  not be quietly overridden.
- Silent NaN propagation into a production run is the worst outcome this library can produce.
  Where a kernel can produce NaN for degenerate input, handle it explicitly or document it
  loudly.
- Inside jitted kernels, prefer static shape/dtype checks to runtime data-dependent
  branching.

---

## 10. No development cruft in committed code

Removed during conversion: leftover `print` / `jax.debug.print` / `jax.debug.callback`;
environment-gated debug probes; commented-out code kept "just in case"; scratch variables and
dead branches; `# TEMPORARY` hacks with no tracked issue; ad-hoc timing that is not part of
`bench/`.

Genuine configurable features and the documented environment gates are **not** cruft.

---

## 11. Conversion checklist (draft style → house style)

1. [ ] `from __future__ import annotations` at the top; `__all__` declared.
2. [ ] Module docstring present, explaining the *decisions* with measurements where they
       exist.
3. [ ] Every substantial function has a NumPy-style docstring including shapes, units,
       static arguments, differentiability, accuracy regime, and stated equivalences.
4. [ ] `pydoclint --style numpy` clean (`pre-commit run --all-files`).
5. [ ] `jaxtyping` annotations on array arguments; no bare `# type: ignore`.
6. [ ] Comments explain *why*; anything that looks wrong but is right is flagged at the site.
7. [ ] Long units split with dashed section dividers; module seams in §8 respected.
8. [ ] Descriptive names; standard symbols and established `p2m`/`m2l`-style shorthand kept.
9. [ ] Loud validation at the public boundary; explicit requests never silently substituted.
10. [ ] No debug probes, stray prints, or commented-out code.
11. [ ] `black --check .` and `isort --check-only .` pass — and formatting was left to them.
12. [ ] `NUMERICS_AND_JAX.md` checklist also passed if operators, sweeps, Pallas, or
        distributed code was touched.

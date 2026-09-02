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

**A one-line docstring is not merely terser than a sectioned one — it is exempt.**
`pyproject.toml` leaves pydoclint's `--skip-checking-short-docstrings` at its default
`True`, so a docstring with no section headers is never checked against the signature at
all. Adding a `Parameters` or `Returns` section opts the function *in* to the full check,
and the signature must then carry type hints that match the docstring **textually**: a
`Returns` section saying `dict` does not satisfy a `-> dict[str, float]` annotation, or the
reverse.

Measured, same unannotated signature both times: prose-only docstring, 0 violations; with
`Parameters` and `Returns` added, DOC105 + DOC106 + DOC107 + DOC203. So *improving* a
docstring is what turns the check on, and it fails in CI rather than locally unless you run
the hook. This tripped three separate changes on one branch, all in `bench/` and `tests/`
where prose-only docstrings are the norm and neighbouring functions in the same file pass
unannotated. **When it fires, annotate the signature to match — do not delete the section
to silence it.**

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
- Use `TypeVar` for decorators that must preserve the wrapped signature (see
  `_precision.py`). A targeted `# type: ignore[return-value]` with an obvious reason is
  acceptable; a bare `# type: ignore` is not.

### 4.1 `jaxtyping` shapes: annotate what nothing else validates

Plain `Array` is an alias for `jax.Array`, so a bare annotation asserts "is an array" and
nothing more. **But do not read that as "shape-annotate everything."** Three measured pilots
(audit E.3; PRs #170, #171, #174) found the payoff varies by an order of magnitude, and the
predictor is not the module and not "the public surface":

| pilot | module | malformed inputs `main` already caught, so the annotation added nothing |
|---|---|---|
| 1 | `upward/tree_expansions.py` | **3 of 3** — yggdrax validated everything |
| 2 | `nearfield/near_field.py` | **0 of 4** — silently accepted, wrong answers returned |
| 3 | `runtime/fmm_evaluate.py` | **2 of 5** — split by parameter family |

**The rule that follows from that:**

> Shape-annotate an array parameter when nothing else validates it. Skip it when the value
> flows straight into a library that checks it.

In this package the unvalidated families are `precomputed_*` and `*_override`, plus internal
functions taking several arrays whose axes must agree. Pilot 3 is the sharpest case: in one
signature the `farfield_*` overrides were already rejected with a domain `ValueError`, while
every `precomputed_*` corruption — wrong dtype, wrong rank, mismatched lengths — reached the
kernels silently.

Annotating an already-validated parameter is not free: it replaces a domain error
(`"masses_sorted must match tree.num_particles"`) with a generic `TypeCheckError`. Do it only
for consistency within a signature you are already annotating.

### 4.2 Derive the shape by execution, not from the docstring

The docstrings are not a reliable source, and this is measured, not a slur:

- `NodeMultipoleData.packed` documented `(num_nodes, sh_size(order))`. It is
  `total_coefficients(order)` — 10 columns at p=2, not 9. Wrong for **every** order the
  package runs at, right only at p=1 (PR #170).
- 11 of 12 `Optional[Array]` parameters on `compute_leaf_p2p_accelerations` document no shape
  at all (PR #171).
- `farfield_leaf_nodes` and `farfield_node_ranges` sit on adjacent lines and share a prefix,
  and are **different axes**: `(3,)` with `(5, 2)`, `(5,)` with `(9, 2)`, `(9,)` with
  `(17, 2)` — leaves against `2*leaves-1` nodes (PR #174).

Instrument the function, run the suite, tally the shapes against the live `n`/`leaves` for each
call, and annotate what you observed. A wrong shape annotation is worse than none, because the
decorator enforces it.

### 4.3 The axis vocabulary

Lowercase, shared package-wide so a reader learns it once. Every new **single-identifier** name
must also be added to the flake8 hook's `--builtins` list — see 4.4.

| axis | meaning |
|---|---|
| `n` | particles |
| `t` | targets, when a call returns a subset of the particles |
| `nodes` | tree nodes |
| `internal` | internal nodes, i.e. those with children |
| `leaves` | leaf nodes |
| `leaves+1` | CSR-style offsets over leaves; symbolic expressions are legal |
| `w` | leaf width (`max_leaf_size`) |
| `sw` | the DECOUPLED lane's source-pool width, which is not the target block's `w` |
| `srcslots` | padded neighbour count per target leaf in the materialised source-particle layout |
| `edges` | entries of the flattened neighbour list |
| `pairs` | entries of a precomputed leaf-pair schedule |
| `chunks`, `chunkflat` | the 2-D chunked scatter schedule |
| `farleaves` | the **far-field** leaf view, which is not `leaves`: they differ on the octree backend |
| `blocks`, `blocksize` | target blocks and the block size (`JACCPOT_LARGE_N_TARGET_BLOCK_SIZE`) |
| `tiles` | source-block tiles in a fixed-shape tile sequence (`nearfield/_large_n_blocks.py`) |
| `tbatch` | target leaves per scan step, i.e. `target_leaf_batch_size` |
| `blockdim` | one solid-harmonic rotation block, square: `2*ell + 1` a side |
| `ct` | Cartesian packed coefficients, `(p+1)(p+2)(p+3)/6` |
| `levels` | block-step levels, `k_max + 1` of them |
| `2`, `3` | literals -- the `(start, end)` pair and the spatial dimension |
| `_` | anonymous: deliberately unnamed, see below |

**`srcslots` is not `w`, and every capture said it was.** `_fast_lane.py`'s materialised
source-particle layout is `(leaves, srcslots, w)`, and in all three recorded calls the middle
and trailing axes were equal -- 2 beside 2, then 256 beside 256. That is the `farleaves` trap
again: the equality held because of how the test payload is built, not because it is a
contract. It was settled by reading the builder rather than the capture. `_large_n_pipeline`
writes `source_particle_ids = target_particle_ids[safe_source_leaf_ids]`, so axis 1 is the
source LIST's length and axis 2 is a gather from the target table -- `w` by construction --
and a re-measurement at `srcslots` 2, 3 and 5 against `w` 16, 8 and 4 makes the equality
disappear. Both kernels read only `shape[1] * shape[2]` and flatten, so a table split the
other way was accepted and returned a force wrong by rel-L2 9.9e-01.

**`sw` exists because the obvious choice was measurably wrong.** `nearfield_mutual.py` shares
one `w` between its `a` and `b` sides, and gives a reason: the kernel pads both to
`_next_pow2` of the a-side count, so a wider `b` hands `jnp.pad` a negative width. Copying
that to `_fast_lane.py`'s decoupled lane would have been a lie -- measured through interpret
mode, a target width of 4 against a source pool padded to 8 with the extra slots masked off
is **bit-identical** to the equal-width result, so a wider source pool is correctly ignored
and asserting equality would reject a configuration that works (4.2: a wrong shape annotation
is worse than none). The two widths therefore get their own names; each still binds across
its own group of three arrays, which is measured, and nothing false is asserted about the
pair. The reverse direction is a defect neither name can express -- a source pool *narrower*
than the target block is accepted and returns NaN -- and is filed rather than papered over,
because jaxtyping cannot say `sw >= w`.

**`tbatch` is not `leaves`, and `tiles` is not `blocks`.** Both distinctions are measured, and
both looked interchangeable before the capture. `_accumulate_target_block_tile_sequence` takes
`target_pos` as `(tbatch, w, 3)` and `leaf_positions` as `(leaves, w, 3)` in the same signature,
observed at 16 against 5 -- `tbatch` is a *scan step's worth* of target leaves, set by
`target_leaf_batch_size`, and is unrelated to how many leaves exist. `tiles` is the sequence
axis over source-block tiles (observed 1 and 4) and sits *outside* `blocks blocksize`, which is
still the block/lane pair inside each tile: the full layout is
`tiles tbatch blocks blocksize`.

**`blockdim` asserts squareness, which is the whole point of naming it.** The rotation-block
tensors in `operators/complex_ops.py` were observed `(17, 3, 5, 5)`, `(17, 4, 7, 7)` and
`(17, 5, 9, 9)`: the trailing pair agrees at three distinct extents, and nothing else in the
package checks it -- a non-square block reaches a matmul and fails there with a message naming
neither parameter. Repeating the name on both the `to_z` and `from_z` tensors also ties them to
each other. The two leading axes stay anonymous, because `jax.vmap` already rejects a batch
mismatch against `multipoles` with a better message than an annotation would give.

None of the three is added to the flake8 `--builtins` list, because none is ever used as a
single-identifier axis -- see 4.4 for why that list exists and what it costs. The same goes
for `sw` and `srcslots`: both only ever appear beside another name.

**`ct` is not `C`.** Elsewhere in the package `C` means `sh_size(p) == (p+1)**2`, the
spherical-harmonic packing. `upward/tree_expansions.py` packs Cartesian moments, so its count is
`(p+1)(p+2)(p+3)/6`. The two agree only at p=1, which is how one symbol served both for so long.

**`farleaves` exists because of a mistake worth not repeating.** `leaf_nodes` in
`runtime/kernels/_evaluate.py` was annotated `leaves`, sharing the axis with
`nearfield_leaf_nodes`. That equality holds for the radix tree and fails on the octree
execution backend (5 against 3), and it broke 7 tests the moment a decorator made the
annotation enforced. The shapes had been derived from 64 captured calls -- through
`test_near_field.py` and `tests/integration/`, neither of which enters that backend. **Capture
coverage bounds annotation validity:** an axis equality observed in every call you recorded is
only as strong as the lanes you recorded.

**A named axis can be impossible even when the shape is known**, and this is the sharper case.
`runtime/kernels/_evaluate.py`'s `nearfield_leaf_particle_indices` is measurably `(leaves, w)`
when its lane is live -- pinned across three configurations in
`tests/unit/runtime/test_large_n_nearfield_shapes.py`. It still cannot be annotated that way.
A jitted signature cannot take `None` conditionally, so when the lane is off the array arrives
as a zero-sized sentinel, and the same parameter is `(leaves, w)` live and `(0, 0)` absent.
Naming the first axis asserts the sentinel has `leaves` rows; it has 0. Annotating it fails 87
tests, the characterization goldens among them.

The rule that falls out: **a name is unusable once something else in the same signature has
already bound it to a live extent.** In the same function `blocks blocksize` is fine, because
those names appear on nothing else, so `(0, 32)` and a live `(nblocks, 32)` both satisfy them --
and what gets asserted is that the two block arrays agree with each other, which is real.
Knowing the shape and being able to annotate it are different questions.

**Anonymous axes are a legitimate answer, not a cop-out.** Where a leading axis is
*caller-dependent*, naming it binds it to whichever caller ran first and breaks the other.
`near_field.py::node_ranges_override` is `(nodes, 2)` from the single-GPU path and
`(leaves+1, 2)` from `distributed/fmm.py`; the distributed lane cannot even be exercised
locally, since every test in `tests/distributed/` skips below 2 devices. So it is
`Int[Array, "_ 2"]` — which still rejects `(M, 3)`, the perturbation that matters.

### 4.4 Hard constraints, all measured

**`@jaxtyped(typechecker=beartype)` is UNCONDITIONAL.** The 45 decorated functions check on
every call, in production, not only under `JACCPOT_RUNTIME_TYPECHECK=1`. That env var installs
the package-wide import hook, which extends checking to *undecorated* functions. Adding a shape
to a decorated function therefore changes production behaviour;
`tests/unit/core/test_near_field.py` pins exactly this for `softening`.

**Never annotate the batch axis of a `vmap`ped function or a `scan` body.** It receives one
slice, so `Float[Array, "n 3"]` sees `(3,)` and fails. There are 122 `vmap`/`scan` sites, so
inside the kernels this is the common case, not an edge case.

**Return annotations are effectively unavailable.** Two independent reasons:

- pydoclint 0.9.1 **crashes** on a shaped return whose axis spec has more than one token:
  `ReturnAnnotation.decompose()` re-parses it as Python and `Float[Array, nodes ct]` is a
  `SyntaxError`. Shaped *parameters* are fine; single-token returns are fine.
- Where a call takes an optional `target_indices`, the result is `[n, 3]` without it and
  `[t, 3]` with it, and jaxtyping cannot express "this axis or that one".

Put the shape in the `Returns` section instead.

**Single-identifier axes trip flake8 F821.** `pyflakes` parses a string inside an annotation as
a forward reference, so `Float[Array, "n"]` reports `undefined name 'n'` while
`Float[Array, "n 3"]` is clean. The axis names are declared via `--builtins` on the flake8 hook
rather than suppressed per line; the cost — a bare `n` in code is no longer flagged — is
recorded there.

**Widths are wrong, families are right.** Use `Int`, never `Int32`/`Int64`: `INDEX_DTYPE` is
selectable via `JACCPOT_INDEX_PRECISION`, and pilot 3 observed `precomputed_target_leaf_ids` as
int32 alongside `precomputed_source_leaf_ids` as int64 *in the same call*. Likewise `Float`, not
a width — positions were observed as both float32 and float64. Scalars that accept a Python
number keep `Union[float, Array]`.

### 4.5 Mechanics

Docstring parameter types mirror the annotation verbatim — pydoclint enforces it — with single
quotes inside the docstring so the type does not terminate it:
`positions : Float[Array, 'n 3']`.

`tests/unit/test_type_annotation_guard.py` holds the modules already converted, so they cannot
regress to bare `Array`. Add a module to its list in the same PR that annotates it.

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
by line count. `jaccpot/nearfield/near_field.py` (4268 lines) and
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
- `mutual/` — the momentum-conserving evaluation path. Same tier as the sweeps, but a
  *second lane* rather than a variant of them: it evaluates each pair once and applies
  `+f`/`-f`, which is what makes `sum_i m_i a_i` cancel algebraically. It must not grow
  a dependency on `runtime/`, and `runtime/` must not reach into it — the two lanes
  compute different numbers on purpose.
- `pallas/` — accelerated kernels that must stay numerically equivalent to their reference
  counterparts. Equivalence is stated in both docstrings and tested.
- `runtime/` — orchestration, config resolution, lane selection, dispatch. The only place
  that **resolves `"auto"` policies**. Reading a `JACCPOT_*` variable is not restricted to
  `runtime/`: `jaccpot/_env.py` is the sanctioned reader for **any** layer, and seven modules
  outside `runtime/` use it. See the note at the end of this section.
- `distributed/` — decomposition, halo exchange, collectives.
- `experimental/` — prototypes. Not production, not held to production standards, and not
  **eagerly** imported by production paths. That last word is load-bearing and the claim is
  now tested: `tests/unit/test_experimental_is_not_on_an_import_path.py` asserts it in a
  clean subprocess per package. It was false until audit G.5's second edge was fixed —
  `import jaccpot` was clean while `import jaccpot.pallas` pulled in
  `experimental/treecode_walk`, a prose guarantee that held at the entry point and failed
  one package down. Deliberate **lazy** reach-ins remain: `runtime/_interaction_cache.py`
  imports `treecode_far_near` inside a function when `local_walk="treecode"` is selected,
  which G.5 accepted as a bounded exposure.

Physics must not live in a utility module; runtime policy must not leak into operators. If
you find yourself importing "up" that list, report it rather than adding the import.

**Existing upward imports, and which of them are deliberate.** The rule above is about not
adding new ones. **Seven** such relationships exist, across **eleven** import statements
(`nearfield/near_field.py` alone has six), and they are not equivalent. The four
substantive ones first:

- `operators/real_harmonics.py` and `nearfield/near_field.py` both import
  `runtime.grad_options` to read a trace-time gate (`analytic_l2p_vjp_enabled`,
  `analytic_p2p_vjp_enabled`). **This is deliberate and the alternative was considered and
  rejected** — see the comment block above the `ContextVar` overrides in
  `runtime/grad_options.py`, which explains that these gates are consulted several layers
  below the public entry point with no argument channel reaching them, and that threading a
  config object down every one of those paths would touch forward-only production code for
  no forward-only benefit. Do not "fix" this: replacing it with dependency injection is
  what the `GradConfig`/`ContextVar` mechanism exists to avoid, and the fields it backs go
  silently inert if the gate stops being readable from where it is read.
- `pallas/treecode_walk_pallas.py` imports the `TreecodeLeafLists` **NamedTuple** from
  `experimental/treecode_walk.py`. A data type, not logic, and the kernel's only consumer is
  `experimental/treecode_far_near.py` — so this is an accelerated twin of a prototype that
  happens to live one directory up, rather than production reaching into `experimental/`.
- `runtime/_interaction_cache.py` imports `experimental/treecode_far_near.py` (function-local)
  from `_build_treecode_artifacts_strict_streamed`, which `distributed/fmm.py` calls for
  `local_walk="treecode"`. This one **is** production depending on `experimental/`. It is
  reachable only with >= 2 devices, so CI never enters it; that makes it the weakest-covered
  production option in the tree, not merely a style wrinkle. Measured: that module sits at
  **0%** coverage.

And three that the earlier count missed. None is a new violation to fix; they are listed so a
reader auditing the layering does not have to rediscover them:

- `nearfield/near_field.py` imports `pallas/nearfield_fused_leaf.py` at **five** function-local
  sites. This is the production Pallas near-field path (`_radix_fast_lane_prepacked_accel_cvjp`,
  see ARCHITECTURE §7), and function-local is the deliberate "defer the heavy Pallas import"
  pattern named at the end of this section. Deliberate.
- `operators/m2l_real_rot_scale.py` imports `pallas/m2l_core_z_real.py` (function-local). This
  is the one that is genuinely arguable: this section defines `operators/` as pure algebra, and
  an accelerator import is not algebra. Left as-is pending a decision, not endorsed.
- `basis/complex_sh.py` imports `operators/{complex_ops,real_harmonics}`. Within the single
  "mathematical algebra" tier this section defines, so benign.

**Environment variables: `_env.py` is the reader, `runtime/` owns the policy.** This was audit
G.2, and it is now decided. The old wording — *"the only place that reads environment
variables"* — contradicted `jaccpot/_env.py`, whose docstring says any layer may use it. `_env.py`
wins, because it exists to prevent a real bug: four near-identical private readers had
accumulated and did **not** agree, so the same value meant different things depending on which
module read it. Item 2.2 consolidated them deliberately, and seven modules outside `runtime/`
now read through it. Reverting that would re-create the divergence.

So the rule is narrower: **`runtime/` is the only place that resolves an `"auto"` policy into a
concrete choice.** Reading a flag where it is used is fine.

Re-measured 2026-08-20, because the count this note used to carry was stale by an order of
magnitude — it said 16 raw reads outside `runtime/` (a pre-2.2 number):

| | count |
|---|---|
| modules outside `runtime/` reading via `_env.py` | 7 — sanctioned |
| raw `os.environ` / `os.getenv` outside `runtime/` | **2** |

**This count is now enforced, because it did not hold on its own.** By 2026-08-27 there were
three: `operators/m2l_real_rot_scale.py` had acquired a module-level `os.environ` read *after*
G.2 decided the rule, and nothing caught it — a prose count in a guide is not a check. It was
also the shape `_env.py` exists to prevent, a knob captured at import that silently does nothing
when set later. Removed, and `tests/unit/test_shared_env_switches.py` now pins the set to the two
below, so a fourth fails a test rather than aging into this table.

The two raw reads are not equivalent:

- `_typecheck.py:12` is **structural**. It reads its own import-hook flag before `jaccpot` is
  importable enough to use `_env`, so it cannot route through it.
- `mutual/farfield.py:135` is a **genuine violation of the narrowed rule**: it resolves
  `JACCPOT_MUTUAL_M2L="auto"` into `"fused"`/`"jax"` outside `runtime/`.

**Do not "fix" that second one by swapping in `env_choice`.** It raises `ValueError` on an
unrecognised value and documents that in its `Raises` section; `env_choice` warns once and falls
back to the default, which is 2.2's deliberate house semantics. A mechanical conversion would
silently turn a loud failure into a quiet default — the exact class of change 2.2 was careful
about. Moving the resolution into `runtime/` is the real fix, and it is the mutual lane's own
decision to make.

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

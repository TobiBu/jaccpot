# jaccpot — numerics, JAX rules, and verification

Read this before touching `jaccpot/operators/`, `jaccpot/upward/`, `jaccpot/downward/`,
`jaccpot/nearfield/`, `jaccpot/pallas/`, or `jaccpot/distributed/`.

Every rule below exists because breaking it produced a wrong number, a broken gradient, or a
silent performance cliff at least once — and in most cases the incident is already written up
in `docs/` or in a comment at the site. **The job is to reuse what is already right, not to
rediscover why it is that way.** Before reasoning from first principles about why something
is the way it is, grep `docs/` and the surrounding comments.

---

## 1. The invariants

### Numerics

A refactor does not change results.

- **Do not change the order or associativity of floating-point reductions.**
  `sum(a) + sum(b)` is not `sum(a + b)`, and the error budget here is set by expansion
  truncation, not by slack we can spend on sloppy accumulation.
- **Do not change accumulation dtypes**, and do not touch anything conditioned on
  `jax_enable_x64`. The default test invocation is `JAX_ENABLE_X64=1`.
- **Do not change matmul precision.** `jaccpot/operators/_precision.py` pins fp32 matmuls to
  `lax.Precision.HIGHEST` via the `highest_matmul_precision` decorator, because XLA otherwise
  lowers them to TF32 on Ampere+ and caps M2L relative accuracy at ~6e-04 from order 4
  upwards — measured, and independent of expansion order. Dropping back to TF32 reintroduces
  that floor and makes raising `p` buy nothing. If a profile ever puts this on the critical
  path, the honest fix is preset-dependent precision (ACCURATE → highest, FAST → default),
  not a quiet downgrade.
- **Do not restructure the rotation / translation algebra** in the M2L path. The
  rotate → z-translate → rotate-back decomposition and the involutory B-matrices are ordered
  for cancellation reasons, not aesthetics.
- **Do not change accumulation order in P2P or M2L.**
- **Equivalent paths must stay equivalent.** Reference vs. fused Pallas, real vs. complex
  basis, fast lane vs. general lane, single vs. multi-device — each pair has an asserted
  numerical equivalence. If you touch one side, prove the pair still agrees.

### Differentiability

The reason this library exists is that you can take gradients through it.

- Verify against **gradient** tests, not just the forward pass. A change that preserves
  potentials and accelerations exactly while breaking the VJP is a broken change, and
  forward-only tests will not catch it.
- Keep everything functionally pure: no hidden mutable state, no global config read inside a
  kernel, no in-place trick that is only correct under one trace.
- Watch for accidental `stop_gradient`, `jnp.where` guarding a NaN-producing branch (the
  classic silent-NaN-in-the-backward-pass trap), and non-differentiable integer paths
  creeping into traversal.
- `custom_vjp` on the Pallas kernels is load-bearing. Do not "simplify" a `custom_vjp` into
  autodiff-through-the-kernel, and do not change the residuals it saves without re-running
  `bench/audit_reverse_residuals.py`.

### Performance

- No **host sync** inside a jitted path: no `.item()`, `bool()`, `float()`, `print` on a
  traced value, or `.block_until_ready()` in library code.
- Do not wrap an extracted helper in its own `@jit`. Extracting a plain function out of a
  jitted region is normally free — it inlines. A new `@jit` adds a dispatch boundary.
- **Compile time is a first-class metric.** This suite is compilation-bound, which is why
  pytest runs under `-n auto`. A change that turns a static value into a traced one, or the
  reverse, can leave runtime untouched and triple compilation.

### Dependency floor and ceiling

The JAX `>=0.10.2, <0.11` window in `pyproject.toml` is **load-bearing on both ends** and the
full reasoning is written there:

- below the floor, the CPU backend takes a `SIGFPE` inside XLA while compiling the real-basis
  P2M `lax.scan` — it kills the worker and cannot be caught;
- `ragged_all_to_all` needs the XLA fix for its reverse pass, or one gradient silently breaks
  every later halo exchange;
- above the ceiling, JAX 0.11 requires Python ≥3.12 (dropping our 3.11 support at install
  time) and is ~2.6× slower on the CPU backend for this workload, which blew the CI caps.

**Do not touch either bound.** Any JAX bump must be validated on the **CPU** backend, not
just GPU — GPU-only validation does not see the crash.

---

## 2. JAX rules

- **Preserve `static_argnums` / `static_argnames` / `donate_argnums` exactly.** Changing what
  is static changes recompilation and buffer donation. Treat it as a design decision needing
  sign-off, not a cleanup.
- **No Python control flow on traced values.** Use `lax.cond` / `lax.switch` / `lax.select`,
  and say in a comment why the branch exists.
- **Do not alter PyTree registration** (`register_pytree_node`, dataclass flattening). Field
  ordering is load-bearing for correctness and donation; reordering silently permutes buffers.
- **Do not change `vmap` / `scan` / `shard_map` axis specs, `PartitionSpec`s, sharding
  annotations, or collectives.** Distributed behaviour is not refactorable without a
  multi-device test, and those do not run in the default suite.
- Prefer `lax.scan` to Python loops beyond a handful of iterations; Python loops unroll and
  blow up compile time.
- Static shapes only. Anything data-dependent in extent — leaf occupancy, interaction-list
  length, halo counts — is padded to a static bound. If you touch padding, verify the padded
  entries still contribute exactly zero and never `0 * inf`; see
  `bench/audit_nearfield_padding.py`.
- `"auto"` policies (lane selection, `halo_exchange`) may choose. An **explicit** user request
  must be honoured or fail loudly, never silently substituted.

---

## 3. Testing

Three kinds of test, not interchangeable.

**`tests/unit/`** — does the function do what its docstring claims? Fast, CPU, every commit.
Test the edge cases that matter *physically*: zero and one particle, coincident particles,
single-particle leaves, particles on a cell boundary, particles outside the root box, empty
cells, and the flattened centrally-concentrated distribution that stresses the tree.

**`tests/integration/`** and validation — does the result match something known independently?

- FMM potentials and accelerations against **direct O(N²) summation**, to within the
  expansion-order error bound. This is the test that says the library works.
- **Convergence**: error must fall at the expected rate as `p` rises and the MAC tightens. A
  test that passes at one `p` without demonstrating convergence tests nothing.
- Physics invariants that hold regardless of implementation: total force sums to ~zero,
  potential energy computed two ways agrees, known analytic cases.
- Gradients against finite differences and against `jax.grad` of the direct sum.
- Path equivalences: reference vs. fused, real vs. complex, fast lane vs. general.

**`tests/characterization/`** — the golden references (`test_fmm_golden.py`). These exist so a
refactor cannot silently move a number. If a golden moves:

> **Revert the change. Do not update the golden and do not relax the tolerance.** A
> refactoring change that alters numerics means the change was not what you thought it was.
> Goldens are updated only in a dedicated commit that states what physically changed and why.

If you are refactoring something whose golden coverage is thin, **extend the characterization
suite first, in its own commit**, before touching the code — including a gradient golden, not
only a forward one.

**Tolerances.** Never `==` on floats. Always an explicit `rtol`/`atol` **with a comment
justifying the value**, tied to the truncation error bound or to accumulated round-off, never
to "what made the test pass".

**Reproducibility.** Explicit PRNG keys everywhere. Never rely on a global seed.

**Markers.** Keep `slow` and `experimental` accurate — the smoke leg
(`-m "not slow and not experimental"`) is what most people actually run, and a mismarked test
either breaks it or hides from it.

---

## 4. Benchmarking

Any change to operators, sweeps, Pallas kernels, or traversal is benchmarked before and
after. CI has a guard (`bench/ci_benchmark_guard.py`); do not rely on it to catch what you
should have measured.

- Report **per stage** — upward sweep, traversal, M2L, downward sweep, near field — because
  an aggregate wall time hides a regression in one stage paid for by noise in another. The
  existing profilers already break this down (`bench/profile_downward_breakdown.py`,
  `bench/profile_fused_stage_ablation.py`, `bench/profile_refresh_stage_breakdown.py`).
- Report **compile time separately** from runtime.
- Benchmark the cases that are actually hard, not just the uniform cube: the centrally
  concentrated, flattened distribution is where near-field load imbalance shows up, and it is
  what the science runs use.
- "No noticeable change" is not a measurement. Put numbers in the PR.

---

## 5. Anti-goals

Things that look like helpfulness and are not:

- "Optimising" a numerical scheme while doing something else.
- Fusing, vectorising, or reassociating arithmetic "for speed" without a benchmark.
- Replacing an explicit recursion with a library call that changes accumulation order.
- Hoisting deliberate function-local imports to module scope.
- Unpinning `black` / `isort`, or moving the JAX floor or ceiling.
- Adding a dependency to avoid writing fifteen lines.
- Broad reformatting of untouched files, which buries the real diff.
- Deleting code that "looks unused" — grep the repo, `examples/`, `bench/`, and `docs/`
  first, and say in the commit message that you did.
- Batching unrelated fixes into one commit.

---

## 6. Checklist for numerics-touching changes

1. [ ] `JAX_ENABLE_X64=1 pytest -q` green on CPU.
2. [ ] `tests/characterization/` unmoved — **including gradient goldens**.
3. [ ] Validation tests still converge at the expected rate in `p` and the MAC.
4. [ ] Stated path equivalences (reference/fused, real/complex, lane/lane) still hold.
5. [ ] No new host syncs, no new `@jit` boundaries, no changed static or donated arguments.
6. [ ] PyTree registration, sharding annotations, and `custom_vjp` residuals untouched, or
       explicitly signed off.
7. [ ] Per-stage benchmark on the uniform *and* concentrated cases; deltas in the PR.
8. [ ] Compile time measured and reported.
9. [ ] Any deliberate numerics decision documented in a comment **at the site**, with the
       measurement — PRs get forgotten, comments get read.

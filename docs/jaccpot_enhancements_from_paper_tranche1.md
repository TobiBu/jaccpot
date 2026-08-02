# jaccpot enhancements surfaced while measuring paper tranche 1

Everything here was **measured** on this host (8x A100-PCIE-40GB, 72-core) between
2026-08-01 19:00 and 22:30 while producing figures 01-07 and 13. Each item states
the evidence, so none of it has to be re-derived. Where a cause is inferred rather
than measured, it says so.

Nothing in this list was fixed during the tranche: the paper work deliberately
changed no forward-force code. Items are ordered by what they cost a user, not by
effort.

---

## P1 -- these make the library measurably slower or unusable

### E1. `preset="accurate"` is ~139x slower than `large_n_gpu` on a GPU

Measured, idle A100 (0% util, 0 MiB used), N=16384, leaf=64, p=4, basis real,
`evaluate_prepared_state` on a prebuilt tree:

| preset | leaf | per evaluate |
|---|---|---|
| `accurate` | 64 | **27.3 s** |
| `accurate` | 256 | 11.0 s |
| `large_n_gpu` | 64 | **197 ms** |
| `large_n_gpu` | 256 | 240 ms |

Steady state, not compilation: calls 1-5 within 2% of each other. The same problem
on this box's CPU takes 861 ms, so `accurate` on an A100 is ~30x slower than
`accurate` on the CPU beside it. Cause (inferred): `accurate` does not take the radix
fast lane.

Impact beyond speed: the first GPU-vs-CPU measurement reported a *0.03x speedup* --
the A100 losing to the CPU by 30x -- and it was reproducible and not obviously
wrong-looking. A preset whose name implies "accurate, therefore slower" being 139x
slower on the flagship device is a trap for users and benchmarks alike.

**Wanted:** either route `accurate` through the fast lane on GPU, or make preset
resolution device-aware, or emit a one-time warning naming `large_n_gpu`. Silent is
the one option that should be off the table.

### E2. The acceleration path carries a large fixed overhead the potential path does not

Measured, A100, `large_n_gpu`, p=3, theta=0.77, leaf=128, evaluate on a prebuilt tree:

| N | potential | acceleration | ratio |
|---|---|---|---|
| 2048 | 4.5 ms | 211 ms | 47x |
| 4871 | 7.2 ms | 217 ms | 30x |
| 11585 | 12.3 ms | 223 ms | 18x |
| 65536 | 33.0 ms | 269 ms | 8.2x |
| 881744 | -- | ~290 ms | -- |

An acceleration is three components plus the expansion gradient, so 3-6x is
expected. A ~210 ms floor that barely moves while N grows 430x is not that. Fitted
over N=27554..881744 the acceleration path gives alpha=0.20 at R^2=0.89 -- it is
overhead-bound across the entire measured range, which is why figure 04 labels it
rather than quoting an exponent.

Consequence: every paper figure that reports acceleration wall-clock is reporting a
constant. Whatever this floor is, removing it would improve the headline number more
than any asymptotic work.

**Wanted:** profile the acceleration path against the potential path at fixed N and
find the constant. Candidates worth eliminating first: a fixed-size allocation or
scatter independent of N, or a host round-trip per call.

### E3. `differentiable_accelerations` cannot be `jax.jit`-wrapped end to end, at any N

An outer `jax.jit` over the whole call did not finish compiling after **18 minutes at
N=256** on an A100 (85% CPU, 0% GPU). At N=16384, XLA spent 3m21s on
`jit__accumulate_m2l_fullbatch` alone and had not finished the module after 30
minutes. Lowering N does not help, so the cost is the unrolled order-4 rotation
cascade rather than the problem size.

Two separate problems:

1. **It hangs rather than raising.** `docs/differentiable_fmm_design.md` documents a
   jit limitation at *large* N; in practice it is unconditional and manifests as an
   unbounded compile. Any `try: jit / except: fall back` -- including the one in
   `bench/differentiability/autodiff_overhead.py` -- never fires, because slowness is
   not an exception. A user putting this in a jitted training loop sees a hang.
2. **Eager dispatch is not a substitute.** It re-traces per call: 1.88x forward-vs-
   reverse on CPU against 11.73x on the A100, on a 6.2 s forward at N=256 where a
   direct O(N^2) sum is sub-millisecond. Two devices disagreeing 6x on a
   device-insensitive ratio shows both are measuring tracing.

This is what blocks figure 12. See `docs/fig12_autodiff_overhead_blocked.md`.

**Wanted:** a jittable step seam -- `prepare_state` once, then a jitted function
taking positions/masses as arguments so the compile is paid once and amortised. Plus
a fail-fast guard: if the call cannot be traced, raise with the reason instead of
handing XLA an unbounded compile.

---

## P2 -- these cost correctness of measurement, and mislead quietly

### E4. A partial `traversal_config` silently discards the preset's other traversal tuning

Passing
`FMMAdvancedConfig(runtime=RuntimePolicyConfig(traversal_config=DualTreeTraversalConfig(max_pair_queue=131072, ...)))`
on `large_n_gpu` -- with `max_pair_queue` set to **the value the preset already
uses** -- measured at N=65536:

| | per-step | attributed fraction |
|---|---|---|
| preset defaults (no override) | 1085 ms | 76% |
| explicit traversal_config | ~3200 ms | 27% |

Setting one field to the value it already had cost 3x, because supplying a
`traversal_config` replaces `process_block` and the interaction/neighbour caps too.
This produced a wrong conclusion ("raising the queue costs 3x") that reached a commit
message before a control run caught it.

**Wanted:** merge a user-supplied `traversal_config` onto the preset's resolved
values field-by-field rather than replacing the object, or require every field and
reject a partial one. Either is fine; silently substituting untuned defaults is not.

### E5. The downward pass is uninstrumented on the strict refresh route

`get_runtime_diagnostics()` exposes `refresh_dual_m2l_compute_seconds` and
`refresh_dual_l2l_compute_seconds`, but on the strict non-fused refresh path they stay
**zero** -- no `dual_*` counter is populated at all. Measured at N=65536, the counters
that do fire account for 70% of per-step time; the remaining ~30% contains the entire
far field plus the local-to-particle evaluation.

So the one figure that answers "where does per-step time go" cannot say anything about
M2L, the stage most characteristic of an FMM. Figure 06 reports the gap explicitly as
`unattributed (incl. downward)` rather than drawing an M2L band at zero.

Cause (inferred): `_record_timed_array` in `runtime/kernels/core.py` is a no-op when
`timing_recorder is None`, and the recorder is not threaded through this route.

**Wanted:** thread the recorder through the strict refresh downward path so M2L and
L2L are separately timed.

### E6. The stage-timing counters are a hierarchy, and nothing says which are aggregates

`refresh_nearfield_seconds` is the sum of the `refresh_nearfield_*` children;
`refresh_tree_upward_seconds` is the sum of the `refresh_upward_*` ones. Measured at
N=65536: nearfield 80.33 ms against its children summing to 79.4 ms. Any consumer that
sums "all the counters" double-counts, and the first version of figure 06 did exactly
that -- charging the near field twice and reading 7.4% in two different bands.

**Wanted:** expose the parent/child structure (a nested dict, or a documented naming
rule) so a consumer can sum a partition rather than guessing.

### E7. `upward_geometry` is 56-59% of a per-step refresh

Measured, A100, `large_n_gpu`/radix/solidfmm, order 4, theta 0.6, leaf 256:

| N | per-step | upward_geometry | P2M+M2M | near field | unattributed |
|---|---|---|---|---|---|
| 32768 | 1038 ms | **56.7%** | 2.1% | 4.9% | 36.2% |
| 65536 | 1085 ms | **59.2%** | 2.0% | 8.8% | 30.1% |
| 131072 | 1687 ms | **55.6%** | 2.2% | 10.4% | 31.8% |

Computing node geometry costs ~25x the P2M and M2M expansions it feeds. Since
topology is reused across a refresh, much of this may be recomputable-once rather
than per-step.

**Wanted:** check what `upward_geometry` recomputes that the frozen topology already
determines, and cache it across a refresh.

---

## P3 -- capacity and reproducibility

### E8. Default `max_pair_queue` overflows above N ~ 131072

`large_n_gpu`, leaf 256, order 4, theta 0.6: N=262144 and above raise
`Pair queue capacity exceeded; increase max_pair_queue and rebuild`. Raising it is not
a clean workaround, because of E4.

**Wanted:** scale the default with particle count, or catch the overflow and re-plan
with a larger queue instead of surfacing it to the caller.

### E9. OOM at N ~ 10^6 on the non-large-N paths

* `preset="accurate"`, leaf 32, order 4, theta 0.6: `prepare_state` at N=1048576 dies
  with `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 8.00GiB` on a
  40 GB card. (Cost figure 05 its top ladder point.)
* `preset="large_n_gpu"`, leaf 64: N=1048576 dies on a 4.00 GiB allocation. (Cost
  figure 07 its top point.)

The large-N path exists for this, so it is partly expected -- but a single 8 GiB
allocation on a 40 GB device suggests a capacity estimate rather than a hard limit.

**Wanted:** either chunk the offending allocation or raise a diagnostic naming the
buffer and the preset that would fit.

### E10. GPU results are not bit-reproducible run to run

Measured, A100, N=512/leaf=16/p=4/float64, 8 runs on identical inputs: **0 of 7**
later runs bit-identical to the first; worst deviation **3.8 eps** elementwise
(8.1e-17 relative to rms |a|). Cause: near-field and M2L accumulate via scatter-add,
which XLA lowers to atomics, and float addition is not associative.

This is normal for a GPU FMM and not a defect, but it is undocumented, and three
tests in `tests/unit/runtime` asserted bit-equality and failed on GPU while passing on
CPU (fixed in this tranche via `tests/unit/runtime/_reproducibility.py`).
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` does restore bit-equality but did not
finish three tests in 50 minutes.

**Wanted:** a line in the README stating that GPU results are reproducible to a few
ulps rather than to the bit, and pointing at the deterministic-ops flag.

### E11. `basis="complex"` and `basis="solidfmm"` are bit-identical

Measured at N=2048/p=4/theta=0.5: `complex` vs `solidfmm` **bit-identical**, max diff
0.0. (`real` vs `solidfmm` agree to 4.5e-13, which is the genuine independent-basis
cross-check and a good result.) Two names for one path is a documentation problem at
best and a silently ignored argument at worst.

**Wanted:** alias them explicitly, or document that `complex` selects the solidfmm
path.

### E12. `basis="cartesian"` is ~1.8e-1 rel-L2 independent of order

Pre-existing and already documented in `docs/dehnen_mass_mac_status_and_plan.md`;
listed here so it is not lost. Order-independent error is a divergent-series
signature, not truncation, and solidfmm is 8.1e-5 at the same configuration -- 2000x
better. The characterization goldens carry a 0.35 anchor for cartesian alone to
accommodate it.

**Wanted:** fix, or mark experimental so it cannot be selected by accident for
quantitative work.

---

## P4 -- process

### E13. The MAC criterion fixes are not on `main`

`fix/dehnen-mass-mac` carries four fixes to the Dehnen mass-dependent criterion (eq
12 power double-counting, sphere-merge ordering, COM-vs-SES geometry, eq 16a
asymmetry dropping leaf-leaf pairs from *both* M2L and P2P), the `mac_theta_max`
knob, the cached force-scale prepass, and 94 unit tests. It was merged into
`paper/jaccpot-i-tranche1` for this tranche but has never been pushed or PR'd.
`main` still has a `dehnen_error` criterion with D1, D3 and D4 unfixed.

**Wanted:** push the branch and open a PR to `main`.

### E14. Comparison benchmarks record an unavailable runner as a data point

`bench/bench_jaxfmm_paper_compare.py` imported `jaxfmm.fmm`, which jaxFMM removed
before 0.3.3. The import failed, `HAVE_JAXFMM` went False, and every jaxfmm row was
written to CSV as `status=error` -- so the comparison arm was dead rather than absent,
indefinitely, and the CSV still looked like output. Fixed in this tranche.

**Wanted:** when a runner is explicitly requested via `--runner` and is unavailable,
exit non-zero rather than emitting rows.

---

## Suggested order

E1 and E2 change every performance number in the paper. E3 unblocks figure 12 and
matters for anyone training through the force. E4/E5/E6 are cheap and stop future
measurements being wrong. E13 is a PR that is already written and tested.

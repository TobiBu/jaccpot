# How the distributed tier gets verified

**Decision, 2026-08-26.** Track D item 4 of `docs/plan_2026-08_D_signals.md`.

`tests/distributed/` has not run in CI since `e98340d`, and `bench/gpu_gate.py`
says in its own docstring that it does not cover it either. With multi-GPU
becoming the point of the project, and tracks A and C about to land substantial
distributed changes, "whatever the author remembered to run" is the current gate.
This records what replaces it.

## Why there is no GPU leg in CI, and why that stays

Every job in `.github/workflows/ci.yml` is `ubuntu-latest`. The audit's item 2.6
decided against a self-hosted runner and that reasoning is unchanged: the
repository is **public**, so a fork PR would execute arbitrary code on the box,
and the box's GPUs are shared with other users. Nothing below proposes changing
that.

## What was measured

Three things, all on this box, jax 0.10.2, x64.

**1. Most of the tier does not need cards at all.** With
`--xla_force_host_platform_device_count=2`, `tests/distributed` plus
`tests/integration/test_mutual_distributed.py` collects 44 tests and runs them:
37 passed, 1 skipped, and 6 failed only because `resolve_grad_halo_exchange`
gated on the JAX version and not the backend. With that fixed (D1) the file that
held all six goes to 12 passed, 1 skipped. `docs/plan_2026-08_shared.md` records
that distributed correctness taken this way matches 2xA100 to six digits.

**2. CI already does this for the sibling file, and that is where the pattern
ends.** The `test-distributed-mutual` job runs
`tests/integration/test_mutual_distributed.py` under
`--xla_force_host_platform_device_count=4`, serially, and then **fails the build if
the suite skipped rather than ran**. It works because that file is small -- 6:09
measured, 16 particles per device.

It does not generalise to `tests/distributed/`, and the same comment says why: it
budgets *"a 3-5x runner penalty"* for hosted runners against a workstation
measurement. The tier is 73-79 minutes here, so that factor puts it at 4 to 6.5
hours against a 6-hour job ceiling. What CI can afford is the mutual file, which it
already runs.

**3. Cards are slower here, not faster.** On the one file measured both ways,
`test_distributed_grad_correctness.py`:

| | two forced CPU devices | two A100s | two A100s, deterministic ops |
|---|---|---|---|
| wall | 220.9 s | 491.6 s | 1084.3 s |
| of which CPU cannot run | -- | 113.0 s | 113.0 s |

Like-for-like that is ~1.7x **slower** on the cards, and 2.2x again with
`--xla_gpu_deterministic_ops=true`, which this tier needs because
`test_differentiable_forward_is_bit_identical` is an exact-equality assertion
(ARCHITECTURE.md section 10). These are 16-to-128-particle
problems where kernel launch and compile dominate and there is no arithmetic to
speak of, so the hardware buys nothing. Running the whole tier on two cards is
therefore the worst of the options: it is slow, it occupies two shared GPUs, and
it re-runs 37 tests that CPU already answers identically.

## What CPU structurally cannot answer

Smaller and sharper than the tier as a whole:

- **The native ragged halo exchange.** `jax.lax.ragged_all_to_all` has no XLA:CPU
  lowering, and `resolve_grad_halo_exchange("auto")` resolves to `"buf"` on a CPU
  backend *because* of that -- a companion change on `fix/restore-distributed-signals`
  (track D item 1), which this document assumes has landed. So the native reverse pass -- the one whose
  corruption on jax < 0.9.1 the version gate exists to prevent -- is now, by
  construction, never executed on CPU. `test_native_halo_exchange_is_fixed_upstream`
  is its only tripwire and is opt-in behind `JACCPOT_CHECK_UPSTREAM_RAGGED_FIX`.
- **The fused Pallas near field.** On CPU the Triton lowering never runs, which is
  why the driver tests pin `nearfield_backend="baseline"`.
- **Real collectives.** Forced host devices exercise the collective *logic*, not
  NCCL.

## The decision

**Two tiers, split on that boundary.**

### Tier 1 -- CPU, run by hand, NOT in CI

`tests/distributed/` stays out of GitHub Actions. This reverses the first draft of
this document, which proposed a `test-distributed` job, and the reversal is an
arithmetic correction rather than a change of view.

The measurement was 73-79 minutes for the tier run serially on this box (4370 s and
4732 s, two runs, both under load). The `test-distributed-mutual` job -- which this
one was to be modelled on -- states the conversion in its own comment: *"Measured
6:09 for the four tests serially on a fast workstation; hosted runners are the
slower side of that... 35 leaves room for a 3-5x runner penalty"*. Applying that
factor to 79 minutes gives **4 to 6.5 hours**, against GitHub's 6-hour job ceiling.
The first draft quoted that job's structure and did not apply its arithmetic; the
90-minute timeout it proposed is only reachable at a runner penalty of ~1x.

Sharding it across jobs would fit the ceiling and is still the wrong trade: it
would spend hours of hosted-runner time per PR re-answering, on emulated devices,
questions that two forced CPU devices answer locally in 79 minutes, for a tier that
changes in bursts rather than continuously.

So the tier is a **documented manual step**, listed here so it is a checklist rather
than a memory:

```bash
cd <worktree> && export PYTHONPATH=$PWD
XLA_FLAGS="--xla_force_host_platform_device_count=2" JAX_PLATFORMS=cpu \
  JAX_ENABLE_X64=1 pytest -q -o addopts="" -n 0 -rs tests/distributed
```

Expect `43 passed, 1 skipped` in ~79 minutes; the skip is the opt-in native
tripwire, which Tier 2 runs. `-n 0` because the forced devices live in one process.
Run it before merging anything that touches `jaccpot/distributed/`, and quote the
result in the PR body.

**What CI does still cover, which is more than nothing.** The existing
`test-distributed-mutual` job runs
`tests/integration/test_mutual_distributed.py` on four forced host devices on every
run, with a step that fails the build if the suite skipped rather than ran. It has
no `-m "not slow"`, so the production-configuration grid added in this PR runs there
too -- the theta x cross_theta cells, the far-field engagement guard and the
momentum checks are all in CI, at a measured ~90 s added to that job's 6:09 against
its 35-minute timeout. It is `tests/distributed/` specifically, the expensive 33,
that stays manual.

### Tier 2 -- two cards, manually, before merging a distributed change

`python -m bench.distributed_gate`. It claims two cards with `autocvd`, sets
`JACCPOT_CHECK_UPSTREAM_RAGGED_FIX` so the native tripwire runs rather than
skipping, applies the determinism flag, and **asserts the tests it exists for
actually reported a verdict** -- a one-device run skips the whole tier and prints
green, which is the failure this gate is against.

Its default target is the GPU-only-reachable set. **Measured end to end at 18:04**
(11 passed, two A100s, determinism on), against 8:11 with the flag off -- the flag
is on by default because this tier contains an exact-equality assertion, and
`--no-deterministic` is there for when you only want the accuracy checks. Eighteen
minutes is affordable as a pre-merge step, which is the point: a gate people run
is worth more than a thorough one they skip. `--full` widens it to the whole tier
for a release, and is much slower for the reason in measurement 3 above.

The end-to-end run is quoted because the gate has been executed, not just written.
It reported:

```
  preflight: gpu, 2 devices, CUDA_VISIBLE_DEVICES=0,3
  not-vacuous: test_native_halo_exchange_is_fixed_upstream ran 1 (0 skipped)
  not-vacuous: test_fd_vs_ad ran 2 (0 skipped)
  not-vacuous: test_grad_matches_direct_sum_oracle ran 1 (0 skipped)
  distributed gate PASSED
```

The middle three lines are the whole point: they are the difference between "the
suite was green" and "the suite ran".

**Cadence.** Both tiers are manual and both run before merging any PR that touches
`jaccpot/distributed/` or `jaccpot/mutual/distributed.py`, with the results quoted
in the PR body the way `gpu_gate` results already are. Tier 1 answers "does it still
compute the right thing", Tier 2 answers "do the paths CPU cannot reach still work".
CI keeps the mutual suite on every run, as it already did.

**Why not automate Tier 1 anyway, at some cadence.** Because the thing that makes a
manual step work is that a person is attached to the result. A nightly or weekly job
whose 4-hour red goes to nobody is the same vacuous-green problem this document is
about, wearing a different colour. If the distributed lane starts changing faster
than it is reviewed, revisit -- but the fix then is a sharded job with an owner, not
an unattended one.

**Not a nightly, for Tier 2 either.** A nightly on this box would hold two shared
GPUs unattended and report to nobody; the failure it would catch is one a pre-merge
run catches earlier and with an owner attached.

## What is still not covered, stated plainly

Multi-node, more than two cards, and NCCL at scale. Weak-scaling numbers remain a
manual exercise recorded in `docs/`, not a gate. Tier 2 proves the paths execute
and agree with a direct sum on two cards; it does not prove they scale.

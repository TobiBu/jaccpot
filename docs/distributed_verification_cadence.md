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

**2. CI already does exactly this, for the sibling file.** The
`test-distributed-mutual` job runs `tests/integration/test_mutual_distributed.py`
under `--xla_force_host_platform_device_count=4`, serially, and then **fails the
build if the suite skipped rather than ran**. The pattern, the anti-vacuity step
and the reasoning are all already written down there. `tests/distributed/` is not
a new problem; it is the same problem with the job not yet written.

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

### Tier 1 -- CPU, in CI, every run

Add a `test-distributed` job mirroring `test-distributed-mutual`: forced host
devices, `-n 0`, and the same step that fails the build when the suite skips
instead of running. It needs no hardware and closes the 37-test gap.

Ready to apply, and deliberately NOT applied here -- track D owns `tests/` and the
halo resolver, and CI configuration was not its to change:

```yaml
  test-distributed:
    runs-on: ubuntu-latest
    timeout-minutes: 90        # 44 tests, each compiling its own shard_map program
    env:
      JAX_ENABLE_X64: "1"
      JAX_PLATFORMS: cpu
      XLA_FLAGS: --xla_force_host_platform_device_count=2
    steps:
      # ... checkout / setup / install, identical to test-distributed-mutual ...
      - name: Distributed FMM suite
        run: |
          set -o pipefail
          pytest tests/distributed -n 0 -rs -v | tee pytest-distributed.txt
      - name: Fail if the suite was skipped rather than run
        run: |
          if grep -qE "^SKIPPED|no tests ran" pytest-distributed.txt; then
            echo "::error::the distributed suite was skipped, not run"
            exit 1
          fi
```

Two devices rather than four: the tier's guards are all `device_count() < 2`, and
two is the cheaper way to satisfy them.

The 90-minute timeout is deliberately generous and the number behind it is soft:
the 1h12m49s measured locally was taken while a full GPU suite was compiling on
the same box, so it is an upper bound on a contended machine rather than a clean
figure. It is quoted that way on purpose -- a timeout wants the pessimistic
number. Take a clean measurement before tightening it, and if it proves too wide,
shard the job the way `test-full` already is rather than trimming the timeout.

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

**Cadence.** Tier 1 on every CI run. Tier 2 before merging any PR that touches
`jaccpot/distributed/` or `jaccpot/mutual/distributed.py`, and before a release,
with the result quoted in the PR body the way `gpu_gate` results already are.

**Not a nightly.** A nightly on this box would hold two shared GPUs unattended
and report to nobody; the failure it would catch is one a pre-merge run catches
earlier and with an owner attached. Revisit if the distributed lane starts
changing faster than it is reviewed.

## What is still not covered, stated plainly

Multi-node, more than two cards, and NCCL at scale. Weak-scaling numbers remain a
manual exercise recorded in `docs/`, not a gate. Tier 2 proves the paths execute
and agree with a direct sum on two cards; it does not prove they scale.

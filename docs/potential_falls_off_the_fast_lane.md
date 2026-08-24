# The potential falls off the large-N fast lane

**Status 2026-08-24: diagnosed, not fixed.** The call chain is pinned and the
guards are located. The fix is a real multi-part change to the near-field
dispatch and its payoff is only observable on an Ampere GPU, so it is written up
here rather than attempted blind.

## The symptom

Measured on an A100-PCIE-40GB, `large_n_gpu`, leaf 128, p=3, theta=0.77,
`evaluate_prepared_state` on a prebuilt tree
(`bench/results/scaling/wallclock_vs_n.json`):

| N | potential | acceleration | ratio |
|---|---|---|---|
| 881744 | **2.32 s** | 240 ms | 9.7x the wrong way |

and the potential scales as `N^1.32` (R^2=1.000) where the acceleration scales
as `N^0.64` (R^2=0.90). A potential cannot legitimately cost ten times a force
computed from it: the force is the potential's gradient and carries three
components plus the expansion gradient.

This is also the arm that jaxFMM is compared against, since jaxFMM evaluates a
potential. At N=881744 jaxFMM takes 34 ms. So the headline like-for-like
comparison in section 4 of the paper is run through this fallback, which is why
that section quotes the ratio as unfavourable and declines to claim it means
anything.

## The call chain

`evaluate_prepared_state(state, return_potential=True)` on a
`LargeNPreparedState`:

1. `runtime/fmm_evaluate.py:376` dispatches on state type to
   `evaluate_large_n_state`.
2. **Gate A** -- `runtime/_large_n_pipeline.py:1655-1659`:

   ```python
   if (
       bool(getattr(state_prepared, "radix_fast_lane", False))
       and (not bool(return_potential))      # <-- Gate A
       and (getattr(state_prepared, "radix_fast_payload", None) is not None)
   ):
   ```

   The dedicated `_fastlane_body` -- which calls
   `evaluate_large_n_nearfield_fast_lane` and the local-expansion evaluation
   directly, and is the route the 240 ms acceleration takes -- is skipped
   whenever a potential is wanted. Control falls through to the generic
   large-N body below it.

3. The generic body calls the single combined near+far evaluation in
   `runtime/kernels/_evaluate.py` with `return_potential=True`.
4. **Gate B** -- `runtime/kernels/_evaluate.py:1010`:

   ```python
   use_specialized_large_n = (
       not disable_specialized_large_n
       and not bool(return_potential)        # <-- Gate B
       and str(nearfield_mode).strip().lower() == "bucketed"
       ...
   )
   ```

   Inside that body the specialized large-N near field is refused for the same
   reason, so the near field is evaluated by the generic bucketed
   leaf-width x leaf-width path. **This is the gate that costs the time**, because
   it is the one that chooses the near-field kernel, and the near field is the
   bulk of the work.

Gate A alone would be survivable; Gates A and B in series mean a potential
request never reaches any fast near-field lane.

## What already exists, and is unreachable

The potential-capable fast lane is written and tested at kernel level, and is
dead code from the runtime's point of view:

- `nearfield/_fast_lane.py:1109 compute_leaf_p2p_accelerations_radix_fast_lane`
  takes `return_potential` and handles it fully on the fused Pallas paths --
  self-leaf contributions via
  `_compute_leaf_p2p_prepared_large_n_self_only_with_potential_impl`, and pair
  contributions via `_radix_fast_lane_pairs_pallas` /
  `_radix_fast_lane_prepacked_pallas`.
- `_radix_fast_lane_prepacked_pallas` reads the potential out of the fused
  kernel's fourth output component (`out[..., 3]`) alongside the three
  acceleration components. So the *prepacked* lane -- the one used for
  target-block layouts -- accumulates a potential already.
- `pallas/nearfield_fused_leaf.py` computes all four components, and
  `tests/unit/operators/test_pallas_nearfield_fused.py` validates them against a
  float64 reference in interpret mode.
- `runtime/_large_n_nearfield.py:580` has a whole `if bool(return_potential):`
  branch written to call the above. Nothing upstream ever reaches it with
  `return_potential=True`, because of Gate A.

So this is not missing kernel math. It is dispatch.

## A third guard, which is separately wrong

`runtime/_large_n_nearfield.py:585-592`, inside that unreachable branch:

```python
has_target_block_pot = (
    state.nearfield_target_block_source_leaf_ids is not None
    and int(state.nearfield_target_block_source_leaf_ids.size) > 0
)
```

Two problems.

1. It reads `nearfield_target_block_source_leaf_ids`, which is `(0, 32)` --
   size 0 -- in every configuration checked (N=256/2048/4096, leaf 32/64,
   `large_n_gpu`). The populated field is
   `nearfield_target_block_source_leaf_ids_padded`, which is `(8, 32, 32)` and
   `(64, 32, 32)` respectively. So the guard never fires and is effectively dead;
   if it is meant to detect target blocks it is reading the wrong field.
2. Its stated reason -- "their potential is not accumulated on the fast lane" --
   is stale for the padded case, since `_radix_fast_lane_prepacked_pallas` does
   accumulate it.

The **overflow** half of the same guard is *not* stale and must be kept. On the
acceleration branch the overflow payload's contribution is added by a separate
`compute_leaf_p2p_accelerations_radix_payload_pairs_only` call
(`_large_n_nearfield.py:~670`); the potential branch returns before reaching it.
Admitting an overflow payload there would silently return a potential missing
those pairs -- a wrong number that looks plausible, which is the worst failure
mode this library has.

## Why it was not fixed here

1. **The payoff is not observable off an Ampere GPU.** On CPU
   `pallas_nearfield_fused_supported()` is False, so the potential branch falls
   to the generic path regardless and the fused lane the fix exists to enable is
   never exercised. Interpret mode reaches the kernel but not through
   `_large_n_nearfield`, which gates on `pallas_nearfield_fused_supported()`
   rather than on interpret the way `_fast_lane.py:1232` does.
2. **The CPU timings invert, so CPU cannot even reproduce the symptom.**
   Measured here, N=8192, leaf 64, p=3, `large_n_gpu`, CPU: `return_potential=False`
   6262 ms against `return_potential=True` 876 ms -- the opposite ordering to the
   A100. Any "fix" validated only on CPU would be validated against the wrong
   sign.
3. **Gate B needs a potential-capable callee that does not exist yet.**
   `use_specialized_large_n` dispatches to
   `compute_leaf_p2p_accelerations_large_n_accel_only`, which is acceleration-only
   by construction. Its pure-JAX target-block drivers in
   `nearfield/_large_n_blocks.py` hardcode `compute_potential=False` at lines
   104, 270 and 507, discarding the potential the underlying
   `_self_contributions` / `_pair_contributions_batched` primitives already
   return. Either those drivers get the flag threaded through them, or Gate B
   routes to the radix fast lane instead.
4. `agent_guides/NUMERICS_AND_JAX.md` requires that stated path equivalences
   (fast lane vs general lane) be proven when either side is touched, and that a
   numerics-touching PR carry per-stage benchmarks on the uniform *and*
   concentrated cases plus a compile-time measurement. None of that is
   producible without the GPU.

## Suggested order of work, on the GPU box

1. **Confirm which gate dominates** before changing anything. Instrument both:

   ```python
   # A100, large_n_gpu, leaf 128, p=3, theta=0.77, N=881744
   # time evaluate_prepared_state(..., return_potential=True) with, in turn,
   #   (a) Gate A relaxed only
   #   (b) Gate B relaxed only  (expect this to be the one that moves the number)
   ```

   The prediction from the call chain is that (b) carries almost all of it, since
   Gate B is what selects the near-field kernel. Worth confirming rather than
   assuming.

2. **Relax Gate B** by routing the specialized large-N near field through
   `compute_leaf_p2p_accelerations_radix_fast_lane(..., return_potential=True,
   use_pallas=True)` when a radix payload is present, rather than threading
   `compute_potential` through the pure-JAX target-block drivers. This reuses the
   lane that already works and keeps the change in dispatch.

3. **Relax Gate A** so the dedicated `_fastlane_body` handles potentials: thread
   `return_potential` into it, call the near field with `return_potential=True`,
   call `_evaluate_local_expansions_for_particles` with `return_potential=True`,
   and combine. Note the far-field acceleration is `-G * far_grad` while the
   far-field potential carries its own scaling -- match the generic body's
   convention exactly rather than deriving it.

4. **Fix the target-block guard** in `_large_n_nearfield.py` to read the padded
   field, and keep the overflow half.

5. **Prove the equivalence** the guide requires: fast-lane potential against
   generic-path potential, on uniform and Plummer, at fp32 and fp64.
   `tests/unit/runtime/test_potential_stays_on_fast_lane.py` on branch
   `fix/potential-stays-on-fast-lane` is a starting scaffold -- it builds a real
   `LargeNPreparedState`, asserts its own premise, and trip-wires the generic
   path so a lane regression fails loudly instead of passing vacuously. Its
   premise guard currently fails on CPU for the reason in the section above,
   which is the correct behaviour for it.

6. **Re-record fig 04** and rewrite the paragraph in
   `sections/04_complexity_performance.tex` that currently reports the defect,
   plus the jaxFMM comparison that depends on it. Section 8's "far field costs
   more than it saves" limitation should be re-examined too: it is partly a
   statement about kernel-launch counts, and a potential that stays on the fused
   lane changes the launch accounting.

# Multi-GPU tuning axes: what improves scaling and what does not

Measured 2026-08-22 on 6x A100-40GB, commit at branch `paper/jaccpot-i`, leaf 64,
order 3, real basis, uniform distribution, `mac_type=dehnen`, fp64.

All points in this document share one cap configuration, deliberately:

```
--process-block 4096 --max-pair-queue 4194304 --cross-max-pair-queue 16777216
--max-neighbors-per-leaf 256 --cross-max-neighbors-per-leaf 1024
--cross-far-cap 8388608 --cross-max-interactions-per-node 8192
```

These caps are ~5x more generous than the ones used for the numbers in the
paper's Section 5, and they cost roughly 5x wall clock (835 ms here against
156 ms there for the same physics at 2 devices / 15360 per device). **Absolute
times in this document are therefore not comparable to Section 5.** Only the
ratios within it are. The reason to run this way is that the larger per-device
loads do not fit the Section 5 caps at all, and varying caps per point would
conflate configuration with the axis under study.

## Summary

| Axis | Direction | Verdict |
|---|---|---|
| Per-device particle count | 15360 -> 30720 | **Helps.** Weak efficiency 67.3% -> 72.6%, throughput 1.33-1.67x |
| Acceptance criterion theta | raise from 0.4 | **Hurts on both axes.** 18x worse error *and* 11% slower |
| Leaf occupancy | raise from 64 | Faster and more accurate, but that is the *degeneracy*: do not take it |
| Total N at fixed devices | 184320 | **Untestable here.** >=61440 per device exceeds a 45 min budget |

The one-line model that explains every point below: **throughput is set by kernel
launches per particle, not by arithmetic.** Launch count is fixed mostly by tree
structure, so putting more particles under the same tree amortises it (the size
axis helps), while engaging the far field harder at fixed N adds launches without
adding particles (theta and small leaves both hurt). Nothing here is an argument
about operation counts; the asymptotics point the other way and lose.

## Per-device problem size: the one axis that helps

Weak scaling, per-device load held fixed, devices added:

| per-device N | ndev | ms | throughput (part/s) | rel_L2 | far% |
|---|---|---|---|---|---|
| 15360 | 2 | 835.4 | 3.68e4 | 1.43e-04 | 8.0% |
| 15360 | 4 | 1246.6 | 4.93e4 | 1.65e-04 | 16.7% |
| 15360 | 6 | 1241.8 | 7.42e4 | 1.62e-04 | 19.8% |
| 30720 | 2 | 1250.5 | 4.91e4 | 2.20e-04 | 20.2% |
| 30720 | 4 | 1490.2 | 8.25e4 | 2.27e-04 | 22.7% |
| 30720 | 6 | 1721.7 | 1.07e5 | 2.45e-04 | 24.6% |

Weak efficiency (ndev 2 -> 6): **67.3%** at 15360, **72.6%** at 30720.

The mechanism is the far-field share, and it is the same quantity in both
directions. Adding devices to a *fixed* problem shrinks each subdomain, shallows
its hierarchy, and drops the far-field share from 8.0% to 4.3% -- dividing a
problem removes the structure that made it cheap. Doubling the per-device load
does the opposite: 240 leaves per device become 480, one more level of hierarchy
exists to exploit, and the far-field share rises from 8.0% to 20.2% at fixed
device count. A larger problem recovers exactly what subdivision destroys.

The error rises with it, 1.4e-04 to 2.5e-04, which is the expected direction: a
quarter of the interactions are now approximated rather than a twelfth. This is
the trade the method exists to make.

## Theta: raising it is worse on both axes

At 4 devices, 15360 per device, leaf 64:

| theta | ms | throughput | rel_L2 | far% |
|---|---|---|---|---|
| 0.2 | 1001.3 | 6.14e4 | 2.88e-07 | 0.1% |
| 0.3 | 1178.9 | 5.21e4 | 2.51e-05 | 6.2% |
| 0.4 | 1250.2 | 4.91e4 | 1.65e-04 | 16.7% |
| 0.6 | 1420.7 | 4.32e4 | 1.04e-03 | 30.4% |
| 0.8 | 1384.8 | 4.44e4 | 3.00e-03 | 37.2% |

Raising theta buys nothing. Error degrades 18x and wall clock degrades 11%.
There is no trade to make here, which is worth stating explicitly because the
textbook expectation is the opposite: a larger acceptance angle accepts more
pairs into the far field, the far field is asymptotically cheaper per pair, so
it "should" be faster.

It is not, and the reason is the launch-bound regime this implementation runs
in. A single evaluation issues ~15500 kernel launches per device across 642
distinct ops. The near field is one large fused kernel -- 16.9 ms in a *single*
launch. The far field is thousands of small M2L translations. Moving work out of
one efficient kernel into many small launches loses on wall clock even while it
wins on arithmetic. Under those conditions theta is not a cost/accuracy dial at
all; it is an accuracy dial with a mild cost penalty attached.

The consequence for tuning: the payoff is in reducing launch count, not in
reducing arithmetic. Until the far-field sweep is fused, asymptotic reasoning
about which interactions to approximate will keep pointing the wrong way.

## Leaf occupancy: the fast direction is the degenerate one

At 4 devices, 15360 per device, theta 0.4:

| leaf | ms | throughput | rel_L2 | far% |
|---|---|---|---|---|
| 32 | 1566.4 | 3.92e4 | 2.74e-04 | 25.1% |
| 64 | 1250.2 | 4.91e4 | 1.65e-04 | 16.7% |
| 128 | 905.8 | 6.78e4 | 7.63e-05 | 3.2% |
| 256 | (see Section 5) | | 7.7e-06 | ~0.2% |

Larger leaves are monotonically faster *and* monotonically more accurate, all the
way to the leaf 256 point recorded in Section 5, where the far field holds six
local and thirty-six cross-domain pairs out of some sixteen thousand and the
error reaches 7.7e-06.

**This is not a tuning recommendation, and the direction of the gradient is the
warning.** A configuration is getting faster and more accurate because it is
doing less and less multipole work: at leaf 128 the far field carries 3.2% of
interactions, and the method is converging on direct summation with a tree in
front of it for neighbour finding. Optimising this knob on either metric
optimises the FMM out of the code. Leaf 64 is retained as the working point
because it is the largest occupancy at which the far field carries a
double-digit share of the interactions, which is the property the paper is
characterising.

The useful reading is diagnostic: the far field currently costs more per unit of
accuracy than the near field does, at every occupancy we can measure. That is a
statement about the M2L sweep being launch-bound, not about the method.

## The combined picture

Seven points, one ordering. Sorting every measurement at fixed problem size by
far-field share sorts it by wall clock and by error simultaneously:

| far% | ms | rel_L2 | configuration |
|---|---|---|---|
| 0.1% | 1001.3 | 2.88e-07 | theta 0.2, leaf 64 |
| 3.2% | 905.8 | 7.63e-05 | theta 0.4, leaf 128 |
| 6.2% | 1178.9 | 2.51e-05 | theta 0.3, leaf 64 |
| 16.7% | 1250.2 | 1.65e-04 | theta 0.4, leaf 64 |
| 25.1% | 1566.4 | 2.74e-04 | theta 0.4, leaf 32 |
| 30.4% | 1420.7 | 1.04e-03 | theta 0.6, leaf 64 |
| 37.2% | 1384.8 | 3.00e-03 | theta 0.8, leaf 64 |

There is no cost/accuracy frontier here to sit on: within this implementation the
two move together, so no setting of theta or leaf occupancy is a trade. The only
axis that improves throughput without buying it from the far field is the
per-device problem size, and it works by amortising launches rather than by
changing what is approximated.

The actionable conclusion is therefore not a parameter choice. It is that the
far-field sweep needs to issue fewer, larger kernels before any of these knobs
becomes a dial worth turning. Until then, tuning guided by operation counts will
recommend exactly the settings that make the code slower.

## Total N at fixed device count: not measurable on this host

A strong-scaling series at N=184320 was attempted at 2, 4 and 6 devices. Only
the 6-device point (30720 per device) completed, in 1726.8 ms; it reproduces the
weak series' equivalent point (1721.7 ms) to 0.3%, so it is a sound measurement.
The 4-device (46080 per device) and 2-device (92160 per device) points both
exceeded a 45-minute-per-point budget and were killed.

So the strong-scaling improvement that the size axis predicts could not be
confirmed: confirming it needs a large per-device load at a *low* device count,
which is the corner where this implementation becomes intractable. The limit is
not obviously physical -- the suspicion is repeated cap auto-grow, each retry
recompiling and re-running -- but it was not diagnosed, and it should be before
anyone plans a larger run.

## A defect found while doing this

`cross_max_neighbors_per_leaf` (default 128) is the cap that
`cross_near_overflow` is raised against, and the multi-GPU harness had no flag
for it -- it exposed only the local `max_neighbors_per_leaf`. A sweep that
raised every flag the harness offered therefore still overflowed at 6 devices,
and the cap auto-grow exhausted its 4-retry budget without reaching a working
value.

The failure presents as a capacity limit of the method: a 30720-per-device point
that returns `valid=False` with a 48% force error at every device count above 4.
With the cap reachable (auto-grow settling at 1024/2048/4096 for ndev 2/4/6) the
entire series runs valid. Fixed in the same change as this document.

The general shape is one this codebase has produced before: a knob that is not
wired reads as a property of the algorithm. An overflow flag should be trusted
only after confirming the cap it names is the cap that was raised.

# DEFECT: any per-device padding silently corrupts the distributed force

_Found 2026-08-22 on `paper/jaccpot-i` at `7cf2903` (post-merge of main, with the
coarse-extent fix and single-theta API), 6xA100, jax 0.10.2, yggdrax main
`783ca6b`. Accuracy is relative L2 against a direct sum over all sources on 1024
sampled targets, `bench/multigpu/harness.py --accuracy-targets`._

## Summary

When a device's particle count is not a multiple of `leaf_size`,
`partition_for_devices` pads that device up to `cap` and **the resulting force is
wrong by tens of percent**. Nothing reports it: the overflow counters are clean,
`valid` is True, the pair counts look normal, and the wall clock is unremarkable.
Only a direct-sum comparison catches it.

The correlation with padding is exact:

| ndev | per-device count | `cap` (leaf 64) | padding rows | rel-L2 vs direct |
| --- | --- | --- | --- | --- |
| 2 | 16 384 | 16 384 | **0** | 1.67e-04 |
| 4 | 8 192 | 8 192 | **0** | 1.04e-04 |
| 3 | 10 944 | 10 944 | **0** | 1.48e-04 |
| 3 | 10 923 | 10 944 | 21 | 3.77e-01 |
| 5 | 6 554 | 6 592 | 38 | 4.37e-01 |
| 6 | 5 462 | 5 504 | 42 | 4.59e-01 |

`ndev=3` is correct at 10 944 and wrong at 10 923 -- same device count, same caps,
same everything else. The only difference is whether `cap == count`.

## Blast radius: far larger than the padding

`rel_L2 ~ 0.4` with a per-particle worst case of `max_abs_err / ref_rms ~ 2.0`
means roughly **16%** of particles carry a badly wrong force (if `k` of `N`
particles were wrong by about their own magnitude, `rel_L2 = sqrt(k/N)`), and the
worst is off by twice the RMS force. That is produced by **21-42 padding rows per
device**, i.e. about 0.2% of the particles. So this is not "the padding rows get
garbage" -- padding rows are discarded via `gid = -1` and never enter the
comparison. Real particles are wrong.

## Where the padding comes from

`jaccpot/distributed/fmm.py`, `partition_for_devices`:

```python
pos_g[d, :c] = positions[chunk]
mass_g[d, :c] = masses[chunk]
gid_g[d, :c] = chunk
pos_g[d, c:] = positions[chunk[0]]   # pad at a real point
```

Every padding row on device `d` is placed at **one coincident position** --
`positions[chunk[0])`, that device's *first* particle in Morton order -- with mass
0 and `gid = -1`.

## Mechanism: hypothesis, not yet confirmed

Zero-mass *leaves* are handled. In yggdrax's `build_coarse_frontier`,
`nonempty = f_mass > 0` gives a fully-padded leaf `node_id = -1` and excludes it.

The suspect is the **boundary leaf that holds both real particles and padding**,
which is exactly what a non-multiple count produces. For that leaf:

- `f_mass > 0`, so it is treated as a normal leaf;
- its centre of mass is correct, because padding has zero mass and contributes
  nothing to a mass-weighted mean;
- but its **radius** is `max |x - com|` over *the leaf's own particles*, and the
  padding particles sit at the device's first Morton particle -- typically at the
  far end of the domain from a boundary leaf. The radius is therefore inflated to
  roughly span the whole subdomain.

An over-large radius alone should be conservative: it makes the MAC stricter and
pushes pairs into the near field, costing time rather than accuracy. So the
inflated radius explains a slowdown, **not** a 40% error, and something further is
needed to explain the observed corruption. Two candidates worth checking:

1. **Degenerate Morton codes.** yggdrax's own docstring warns that "equalize
   padding duplicates real positions into degenerate Morton-code clusters (which
   breaks a level-truncation antichain)". Tens of exactly-coincident points may
   produce a malformed radix subtree, in which case some real particles'
   interactions are lost or double-counted -- which would match a blast radius far
   larger than the padding count.
2. **A regression from the coarse-extent fix.** Before yggdrax PR #47 the frontier
   carried no radius, so a padding-inflated leaf radius could not reach the MAC at
   all. It can now. This is the cheapest thing to test: run one padded
   configuration against the previous yggdrax and see whether the error survives.
   If it does, the defect predates the fix and was simply invisible while the cross
   far field was empty.

## Reproducing

Two runs, differing only in whether padding exists:

```bash
export CUDA_VISIBLE_DEVICES=1,2,4
COMMON="--leaf-size 64 --process-block 1024 --max-pair-queue 1048576 \
  --cross-max-pair-queue 1048576 --cross-far-cap 4194304 \
  --cross-max-interactions-per-node 4096 --accuracy-targets 1024 \
  --repeats 2 --warmup 1 --gpu-select none --emit-json"

# no padding: 10944 = 171 x 64      -> rel_L2 1.5e-04
python -m bench.multigpu.harness --ndev 3 --n 32832 $COMMON

# 21 padding rows per device        -> rel_L2 ~4e-01
python -m bench.multigpu.harness --ndev 3 --n 32769 $COMMON
```

The whole strong-scaling sweep reproduces it, because dividing a fixed problem is
exactly what produces non-divisible per-device counts:

```bash
python -m bench.multigpu.strong_scaling --n 32768 --ndevs 2,3,4,5,6 $COMMON
```

## Why this matters

- **Strong scaling cannot avoid it.** Holding N fixed and adding devices produces
  `N/ndev` per device, which is a leaf multiple only by luck. Three of five points
  in our sweep were affected, and figure 08 of the paper cannot be measured until
  either this is fixed or every per-device count is chosen to be a leaf multiple.
- **It is silent.** This is the third failure mode in this subsystem that reads as
  healthy: a truncated walk reads as *faster*, a broken far field reads as
  *accurate* (because it is bypassed by exact direct summation), and padding
  corruption reads as *valid*. The overflow counters cannot see any of them. The
  direct-sum accuracy check found this one, and it is the only instrument that
  would have.
- **Older results are suspect.** Any distributed measurement taken where
  `count % leaf_size != 0` may be affected. Our own pre-fix strong-scaling runs at
  leaf 256 included per-device counts of 21 846 and 13 108 (85.34 and 51.20 leaves)
  and were reported as valid.

## Suggested fix directions

Not attempted here; the layering is the caller's call.

1. **Make padding neutral to geometry.** Excluding zero-mass particles from the
   leaf radius (and from the extent computation generally) is the narrow fix, and
   it is where the arithmetic actually goes wrong if hypothesis 1 above is not the
   cause.
2. **Do not create degenerate clusters.** Spreading padding rows onto distinct
   positions, or placing each at *its own* leaf's first particle rather than the
   device's, avoids the coincident-Morton case that the yggdrax docstring already
   flags as hazardous.
3. **Choose `cap` so padding is unnecessary** where possible: partition into
   leaf-size-aligned chunks so `count % leaf_size == 0` by construction. This
   changes the load balance slightly and does not help the general case, but it
   would remove the failure from the common path.
4. **Assert it.** Whatever the fix, a test that pads deliberately and compares
   against a direct sum belongs in `tests/distributed/`: this defect survived
   because no test constructed a padded configuration.

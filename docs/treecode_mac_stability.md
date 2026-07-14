# Treecode MAC choice and multi-step stability (bh box vs dehnen sphere)

**TL;DR** — In the static-radix device-resident fast lane, the per-leaf treecode walk
(`JACCPOT_STATIC_STRICT_FUSED_TREECODE_WALK=1`) must use **bounding-sphere (`dehnen`)
MAC extents**, not the cheaper axis-aligned **box (`bh`) extents**, for multi-step
integration. The box extents are *statically* as accurate as the sphere (t=0 force
parity ~0.03 %) but are **dynamically unstable**: they inject a small non-conservative
force error every step that heats the system and blows the integration up. As of
2026-07-14 the treecode MAC defaults to **`dual`** (= reproduce the configured dual-tree
MAC extents, i.e. dehnen sphere radii for the large-N preset).

Env knob: `JACCPOT_STATIC_STRICT_FUSED_TREECODE_MAC` ∈ {`dual` (default), `bh`,
`dehnen`, `engblom`}. Code: `jaccpot/runtime/_interaction_cache.py`
(`_build_treecode_artifacts_strict_streamed`, `_treecode_mac_extents`) and
`jaccpot/experimental/treecode_walk.py` (`_mac_ok`).

## Background

The fast lane runs velocity-Verlet as a single device-resident `lax.scan` and refreshes
interactions under a **frozen tree topology** (`_refresh_large_n_same_topology`): the
radix tree structure / leaf–particle assignment is fixed for the whole run; only node
geometry, multipoles, and the far/near interaction lists are recomputed each step as
particles drift.

The treecode walk is a per-leaf single-tree Barnes-Hut descent: each leaf accepts
well-separated source *nodes* (→ M2L into that leaf's local expansion, no L2L) and marks
near source leaves (→ direct P2P). Acceptance uses the MAC
`(r_target + r_source)² ≤ θ² · d²`, where `r_*` are the per-node **MAC extents**.

Two extent recipes (both from `_treecode_mac_extents`, matching yggdrax's dual-tree):

| MAC       | extent used                          | relation to true source radius |
|-----------|--------------------------------------|--------------------------------|
| `bh`      | box `max_extent` (half-width)        | **under-bound** (sphere ⊇ box) |
| `dehnen`  | bounding-sphere `radius` (scaled)    | correct (over-)bound           |

## The bug

The multipole approximation error for an accepted far pair scales as
`O((r_source / d)^(p+1))`. The MAC is meant to keep `r_source/d ≲ θ` so this stays within
the accuracy budget. But `bh` feeds the **box half-width**, which *under-estimates* the
true mass extent (the bounding sphere always circumscribes the box). At t=0 the leaves
are compact, box ≈ sphere, and the error is negligible (parity vs the dual-tree/direct
N-body ~0.03 %). Under the **frozen topology**, particles drift and the fixed leaves
spread / elongate, so the box half-width increasingly under-bounds the real extent. The
MAC then admits far-field M2L pairs whose real `r_source/d` exceeds θ → the truncation
error grows → a small **non-conservative** force error is injected every step → the
system **heats and blows up**.

The dehnen bounding-sphere radius is the correct upper bound on the source extent, stays
valid as leaves spread, and reproduces the validated dual-tree acceptance → stable.

## Evidence (200k particles, order 4, θ=0.8, leaf 256, A100, real basis)

Reference: the production **dual-tree walk** is stable — `max|v|` ≈ 7 flat, `dKE/KE0` ≈
0.056, `|dLz/Lz0|` ≈ 2.2e-3 over 300 steps.

| config                      | 20 steps `max\|v\|` | 300 steps `max\|v\|` | 300 steps `dKE/KE0` | 300 steps `\|dLz/Lz0\|` |
|-----------------------------|--------------------:|---------------------:|--------------------:|------------------------:|
| dual-tree walk (baseline)   | ~6.9                | 7.33                 | 5.6e-2              | 2.2e-3                  |
| treecode **bh** (old default)| 24                 | 3.15e6 (exploded)    | 2.4e9               | 2.8e3                   |
| treecode **dual/dehnen** (new default) | 6.9      | 7.34–7.37            | 5.6e-2              | 1.9e-3                  |

`bh` blow-up trajectory of `max|v|`: 7 → 20 (step 5) → 142 (step 40) → >1000 (step 300).
Divergence is broad (≈80 % of particles >10 % velocity error by step 15), i.e. global
heating, not a few close encounters. It is **not** a capacity/overflow effect — raising
all treecode caps to 4 M only softens it (max|v| 1160 → 559), it does not fix it.

The step-1 (t=0) force parity between treecode-bh and the dual walk is ~0.03 % max, i.e.
the forces are correct initially; the failure is purely dynamic/accumulative.

## Cost

Dehnen sphere extents are larger → the MAC accepts deeper → more M2L pairs → some
slowdown vs bh. Measured per-step wall time (200k/order-4/real, A100, compile-subtracted
via `[runtime(300 steps) − runtime(20 steps)] / 560`):

| far/near builder                 | ms/step | stable? |
|----------------------------------|--------:|---------|
| yggdrax dual-tree walk (host)    | ~87     | yes     |
| treecode `bh` (box)              | ~50     | **NO**  |
| treecode `dual`/dehnen (default) | ~58     | yes     |

The fix (`bh` → `dual`) costs ~16 % vs the unstable `bh`, but the stable treecode walk is
still **~1.5× faster than the host dual-tree walk it replaces** (58 vs 87 ms/step) — the
device-resident walk still eliminates the dual walk's host launch-storm; the dehnen
extents only add M2L pairs, not host round-trips. So the correctness fix keeps the bulk
of the speed-up.

## Guidance

- **Multi-step integration** (the fast lane's purpose): use the default `dual` (dehnen).
- **Single-shot / static force evaluation** (one `evaluate_prepared_state`, no per-step
  accumulation): `bh` is safe and faster.

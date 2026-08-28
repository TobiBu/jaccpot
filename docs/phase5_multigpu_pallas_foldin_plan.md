# Folding the single-GPU fast-lane advances into the multi-GPU FMM

_Bookkeeping + plan, 2026-07-14._

> **Read `docs/distributed_per_device_ceiling.md` before configuring anything from this
> file.** Its strong-scaling bullet below is **refuted** and marked superseded in place;
> the per-GPU capacity numbers here were all taken while a traced retry ladder was
> silently truncating the walk, so any of them above 64 leaves per device is suspect.
> The rest of the file — the basis work, the near-field Pallas result, the LET wiring —
> is unaffected.

## STATUS (branch `feat/phase5-multigpu-foldin`, off `feat/phase5-pallas-m2l-prototype`)

- **DONE — 5a + 5b + native-real upward, CPU-validated (4 devices vs direct):**
  - Landed the stranded solidfmm driver as the baseline (0.24%).
  - `DistributedFMMConfig.basis` in {`real`,`solidfmm`}; real path routes self+cross M2L
    through `_apply_real_m2l`, L2L `basis_mode="real"`, L2P auto-selects real by dtype.
  - **Native-real upward** (`jaccpot/upward/real_tree_expansions.py`): real leaf P2M +
    `aggregate_m2m_real_by_level`, so the local upward, coarse upward, and coarse M2M are
    all real — the coarse-tree `all_gather` now ships REAL multipoles (half the inter-GPU
    comm), and the complex→real M2L-boundary conversions are gone. Unit-tested against the
    complex+`complex_to_dehnen_real_coeffs` oracle: 1e-16 match, p=1..4.
  - Flipped the DEFAULT to the converged fast-lane config **real + dehnen** → 0.19% vs
    direct (beats bh's 0.24%), no overflow; solidfmm/bh kept behind explicit config + test.
  - Tests: `tests/distributed/test_distributed_fmm_driver.py` (default real+dehnen, real+bh, legacy
    solidfmm, jit==eager) + `tests/unit/core/test_real_upward_sweep.py` — all green on CPU.
- **DONE — 5c (A100, 2026-07-19):**
  - **Fused real M2L Pallas** engaged via `JACCPOT_STATIC_STRICT_FUSED_M2L_PALLAS=1` — no code
    change; both self-M2L (`_accumulate_real_m2l_fullbatch`) and cross-M2L route through
    `_apply_real_m2l`. Parity-neutral: flag-on vs flag-off forces match to **4.4e-9**.
  - **Fused leafpair Pallas near-field**: new `DistributedFMMConfig.nearfield_backend`
    {`auto`,`pallas`,`baseline`} (auto→pallas on sm_80+). The pallas branch computes the
    intra-leaf self block via `_compute_leaf_p2p_prepared_large_n_self_only_impl` and the
    cross-leaf pairs via `_radix_fast_lane_prepacked_pallas` over the `_combined_neighbors`
    CSR densified to a padded `[u_leaves, S_near]` source-leaf table. Parity vs baseline P2P
    = **2.1e-7** (CPU `interpret=True` de-risk 2e-16), aggL2 vs direct **1.78e-4** @ per=8000.
  - **The near-field was the entire bottleneck.** Build-once steady-state whole-eval
    (ndev=2, per=8000/leaf=128): baseline near **10.7 s → 43.5 ms with Pallas (~245x)**.
    traversal + far-field are only ~40 ms — the ~10 s previously attributed to the
    overhead-bound traversal was actually the pure-JAX baseline near-field leaf-pair P2P.
    M2L-Pallas on/off is negligible at this near-field-dominated config.
  - `jit=True` is STABLE with both backends (the earlier illegal-address crash did not
    reproduce). NB: `distributed_fmm_accelerations` rebuilds+recompiles per call (~50-80 s) —
    build once via `make_force_evaluator(...,jit=True)` for steady-state timing.
- **DONE — 5d (A100, cap-calibrated build-once steady-state; 2026-07-20):**
  - `distributed_fmm_accelerations` gained `auto_scale_caps` (retry with `with_scaled_caps`
    on a traversal-buffer overflow) — the cross-domain LET grows with device count so the
    fixed default caps overflow at ndev≥4. Test `test_driver_auto_scale_caps`.
  - **Weak scaling** (per=8000/GPU, pallas near + fused M2L, overflow-free after calibration):

    | ndev | N | cap× | min ms | throughput (part/s) |
    |---|---|---|---|---|
    | 2 | 16 000 | 1 | 41.6 | 3.8e5 |
    | 3 | 24 000 | 1 | 46.6 | 5.2e5 |
    | 4 | 32 000 | 2 | 59.6 | 5.4e5 |
    | 5 | 40 000 | 2 | 64.3 | 6.2e5 |

    Throughput RISES with GPU count (positive weak scaling); per-eval grows 42→64 ms from
    LET comm + the ×2 padded-cap overhead at ndev≥4.
  - ~~**Strong scaling** (total N=40 000) is **density-limited**: at per-GPU N > ~8000 (ndev 2-4:
    per=20000/13333/10000) the fixed-topology traversal caps explode and STILL overflow at
    cap×64 (max retries) → forces truncated, 400-720 ms (padded pair-queue overhead dominates).
    Only ndev=5 (per=8000) is valid (64.7 ms). **Healthy regime = per-GPU N ≈ 8000**; scale by
    adding GPUs to keep per-GPU N there (= weak scaling).~~
    **SUPERSEDED 2026-08-21, and BOTH claims are refuted — do not configure from this
    bullet.** See `docs/distributed_per_device_ceiling.md`. The limit is not a density and
    not a particle count: it was **64 leaves per device**, set by `process_block`, and 8000
    is exactly `leaf_size=128 × 64 leaves` — two defaults meeting. The 400-720 ms readings
    are not padding overhead but a **truncated walk**, which does less work and so reads
    *faster*; `self_near_pairs` falls as N rises across them. Root-caused 2026-08-24 to a
    retry ladder that cannot be climbed under `shard_map` (the overflow flag is a tracer),
    so the traced wavefront came from `process_block`, not `max_pair_queue` — which is why
    raising `max_pair_queue` to 16.7 M changed nothing. Fixed in yggdrax `#52` plus the
    derived caps in `jaccpot/distributed/fmm.py`. On 2×A100 **no overflow flag fires
    anywhere in the ladder** up to 16 777 216 particles/device; the largest configuration
    clearing the accuracy gate too is **8 388 608/device = 16 777 216 total** (leaf 512,
    rel_l2 4.5e-3, 93.2 s), against the 8000/GPU above. Strong scaling inside the
    verified envelope is **flat**, not density-limited.
  - **CAVEAT: the jit=True illegal-address crash is INTERMITTENT** (nondeterministic OOB) — most
    runs succeed but one recurred at weak ndev=2; run each eval in its own process. Root-cause +
    a padded-pair-queue right-sizing (to lift the per-GPU-N ceiling for strong scaling) are the
    two remaining follow-ups.
    **The second of those is done** (2026-08-24): the queue was never the binding constraint
    — the traced walk ignored it entirely. See `docs/distributed_per_device_ceiling.md`.

---

_Original plan below._

## Two separate force paths (the crux)

| | Single-GPU fast lane | Multi-GPU distributed |
|---|---|---|
| entry | `_fmm_impl.strict_run_v2` (device-resident VV scan) | `jaccpot/distributed/fmm.py::distributed_fmm_accelerations` (shard_map, per-device) |
| basis | **real** (Dehnen) | **solidfmm** (complex) |
| far build | device-resident **treecode walk** (`_build_treecode_artifacts_strict_streamed`) | yggdrax **cross-walk** LET (`dual_tree_walk_cross`) + local `build_interactions` |
| M2L | fused **real M2L Pallas** kernel (`m2l_real_fused`, env-gated) or JAX rot-scale | `_accumulate_solidfmm_m2l_fullbatch` (JAX, complex) |
| near | fused Pallas P2P (fast lane) | `compute_leaf_p2p_accelerations(nearfield_mode="baseline")` |
| MAC | **dehnen** (sphere) — default since the stability fix | **bh** (box) — `DistributedFMMConfig.mac_type="bh"` |
| validated for | perf + energy/Lz over 300 steps; O(N) scaling | single-shot force accuracy (0.24 % vs direct, 4×GPU); **not** integrated, **not** perf/scaled |

They share low-level operators (solidfmm/real expansions, `compute_leaf_p2p_accelerations`) but **not** the assembled pipeline. So recent single-GPU advances do **not** propagate automatically.

## Do the three recent advances fold in? — No (per advance)

1. **Treecode dehnen-MAC stability fix.** The distributed path has no treecode walk, so the *code* change doesn't apply. But the *lesson* does: the distributed driver defaults to `mac_type="bh"` (box `max_extent`), which under-bounds the source multipole radius and effectively runs at a coarser opening angle than the requested θ. That is basis-independent. It is currently harmless only because the distributed FMM is run **single-shot** (accuracy check), never in a multi-step integrator — a bh-MAC integration would heat/blow up the same way. **Latent risk / prerequisite for any distributed integration.**
2. **Fused real M2L Pallas kernel.** Distributed uses the complex solidfmm M2L; the real Pallas kernel doesn't apply to it. No speedup carried over.
3. **Pallas near-field + O(N) fast-lane perf.** Distributed uses the baseline (non-fused) P2P and a full-batch (non-chunked) M2L, so the per-device force is *not* the optimized fast-lane path and does not inherit its O(N)/Pallas behaviour.

## Plan (feature branch, later) — recommended: converge the distributed per-device path onto the real-basis fast-lane kernels

- **5a — MAC first (prerequisite, cheap).** Switch `DistributedFMMConfig.mac_type` default `bh → dehnen`; re-validate the 4-GPU accuracy (driver was tuned at bh/θ_cross=0.1, so re-check the number and re-tune θ_cross if needed). Required before the distributed FMM is ever driven in an integrator.
- **5b — basis convergence.** Move the distributed far-field from solidfmm(complex) to the **real** basis so it can reuse the single-GPU real M2L path (memory-lighter, and matches the single-GPU default). Validate per-device parity vs direct before touching Pallas. (Alternative: write a solidfmm Pallas M2L — larger effort; the complex-fused prototype uses the incompatible *cached* convention, so real is the lower-risk route.)
- **5c — Pallas kernels per device.** Route both the local-self and cross-domain M2L through the fused real M2L Pallas kernel, and the combined [local;halo] P2P through the fused Pallas near-field. Ampere-gated (sm_80+), pure-JAX fallback elsewhere — same gating as the single-GPU lane.
- **5d — multi-GPU perf/scaling.** Weak + strong scaling with Pallas per device, via the `benchmark_multigpu/` harness. Compare per-device throughput to the single-GPU O(N) curve; check the LET/comm overhead doesn't dominate.

**North-star (bigger refactor, optional):** have the distributed per-device *local* force call the single-GPU fast-lane eval directly, so future single-GPU advances propagate for free. Keeps the LET (cross-walk far/near) distributed-specific but removes the duplicated per-device assembly. Higher risk (fast lane is a VV-scan runner, not a bare force eval) — defer unless the two paths keep diverging.

See [[treecode-mac-dynamic-stability]], [[phase5-fused-m2l-a100]], [[multi-gpu-fmm-effort]], jaccpot `docs/treecode_mac_stability.md`.

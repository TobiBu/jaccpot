# Folding the single-GPU fast-lane advances into the multi-GPU FMM

_Bookkeeping + plan, 2026-07-14. No code changes yet — this is the Phase-5 scope._

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

# `docs/` — what is in here, and which file is current

`CLAUDE.md` says: *"If you are about to reason from first principles about why something
is the way it is, check `docs/` first — the answer is usually already written down, with
measurements."* That instruction is load-bearing and it is only cheap to follow if you
can tell, without opening them, which of ~48 files answers your question and which of two
similarly-named ones is current. That is what this index is for.

**Two house rules it reflects.** A superseded note is **marked, never deleted** — this
repo keeps negative results reproducible, and `dehnen_mac_step3prime_prompt.md` is
retained specifically so its traps stay checkable. And a corrected measurement **keeps
the old number** alongside what replaced it, because the seam is usually the interesting
part.

Legend: **[current]** the one to read *among the alternatives here* ·
**[superseded]** kept, but read the successor first · **[closed]** the task is done;
kept for its traps or its record · **[open]** describes work not finished.

**What the labels are not.** `[current]` says which file to open when two or three
cover the same ground; it is **not** a warranty that every number inside is fresh.
Where a file declares its own status, this index repeats it; where it does not, the
label is a routing judgement. Trust a measurement because it names its harness and its
hardware, not because a row here called the file current.

---

## Start here, by question

| if you are asking… | read |
|---|---|
| how do I configure a **multi-GPU** run? | `distributed_per_device_ceiling.md`, then `distributed_verification_cadence.md` |
| what does the **mutual / block-step** lane do? | `momentum_conserving_fmm.md` |
| how do I take **gradients**? | `differentiable_fmm.md` |
| which **MAC**, and at which leaf size? | `dehnen_mass_mac_status_and_plan.md` |
| what is the **production single-GPU path**? | `large_n_runtime_contract_2026-04-21.md` |
| I am about to touch `operators/` | `operator_conventions.md` (and `agent_guides/NUMERICS_AND_JAX.md`) |
| what is still **open** in the codebase? | `refactor_audit_2026-08.md` |

---

## Distributed / multi-GPU

| file | |
|---|---|
| `distributed_per_device_ceiling.md` | **[current]** Per-device N is limited by **leaf count**, not density. Carries the root cause (a retry ladder that cannot be climbed under `shard_map`) and the post-fix envelope: 1 048 576 particles/device. Read before configuring a mesh. |
| `distributed_verification_cadence.md` | **[current]** How the distributed tier gets verified, and why `bench/gpu_gate.py` does not cover it. |
| `phase5_multigpu_pallas_foldin_plan.md` | **[superseded in part]** The basis work, LET wiring and the ~245× near-field Pallas result stand. Its **strong-scaling bullet is refuted** and marked in place; its per-GPU capacity numbers were taken while the walk was truncating. |
| `north_star_phase3_scale_plan.md` | **[open]** The earlier design note for lifting the distributed ceiling; items 3b–3d are still open. |
| `distributed_cross_domain_far_diagnosis.md` | **[closed]** The cross-domain far field was wrong; fixed upstream in yggdrax #47. |
| `distributed_padding_force_defect.md` | **[closed]** The per-device padding "force defect" was a **readout error**, not a defect. |
| `jax_ragged_all_to_all_bug_report.md` | **[closed]** Draft upstream report — **do not file**: not a JAX bug, and already fixed upstream. |

## The mutual (momentum-conserving) lane

| file | |
|---|---|
| `momentum_conserving_fmm.md` | **[current]** The lane's design, its measured momentum properties, the topology backends (`host` vs `device`, and what `freeze_template` freezes), rungs on the mesh, and the M2L as measured. |

## Differentiability

| file | |
|---|---|
| `differentiable_fmm.md` | **[current]** The user guide. Start here. |
| `differentiable_fmm_distributed_audit.md` | **[current]** What is and is not differentiable on the multi-GPU path, and how the grad halo exchange resolves. |
| `differentiable_fmm_audit.md` | **[closed]** Phase 0: can autodiff produce exact fixed-topology gradients at all, and where it breaks. The question that started the work. |
| `differentiable_fmm_design.md` | **[closed]** Phase 1 design. Points at the user guide for usage. |
| `differentiable_fmm_pallas_vjp_plan.md` | **[closed]** PR-2 spec — **implemented**. |
| `differentiable_fmm_nearfield_fastlane_plan.md` | **[open]** PR-3 spec, ready to implement; self-contained. |
| `differentiable_fmm_csr_sources_plan.md` | **[closed]** Investigation, **not implemented** — the measured go/no-go for CSR sources in the near-field reverse. |
| `handoff_g9_grad_golden_gpu.md` | **[superseded]** The handoff. Answered by → |
| `g9_grad_golden_gpu_diagnosis.md` | **[current]** …this: the drift is ordinary float64 reassociation, bounded at 55 ULP. Benign. |

## Near field, large-N runtime, and the fast lane

| file | |
|---|---|
| `large_n_runtime_contract_2026-04-21.md` | **[current]** The canonical production path for `preset="large_n_gpu"`. This is the contract; the status note below is the history. |
| `large_n_runtime_status_2026-03-23.md` | Dated handoff from the earlier `large_N_performance` branch. It does **not** point forward, so treat it as a snapshot: the contract above is the canonical statement of the path. |
| `potential_falls_off_the_fast_lane.md` | **[open]** Diagnosed, not fixed: why a potential request leaves the large-N fast lane. |
| `nearfield_structural_comparison_2026-04-15.md` | jaxFMM vs jaccpot near-field structure, with a 2026-04-20 status addendum. |
| `nearfield_tonb_runbook.md` | The copy/paste path for the cross-repo TONB A/B checks. |
| `radix_fast_lane_refactor_plan_2026-04-16.md` | **[closed]** The refactor plan… |
| `radix_fast_lane_implementation_checklist_2026-04-16.md` | **[closed]** …its checklist, with a 2026-04-20 status snapshot… |
| `radix_default_simplification_plan_2026-04-21.md` | **[closed]** …and the follow-up that removed the legacy branches and obsolete knobs. Read in that order. |
| `real_vs_complex_gpu_plan.md` | **[closed]** The plan for measuring whether the real (Dehnen) basis should be the default. It now is. |
| `fmm_pipeline_fusion_plan.md` | **[open]** Scoped, not started — drafted after the kernel-count work. |

## Profiling records (dated; read as snapshots, not as current state)

| file | |
|---|---|
| `fmm_fused_perstep_profiling_2026-07-08.md` | Fused-lane per-step profiling. Its raw data is `fmm_fused_perstep_profiling_h100.json`. |
| `h100_fastlane_launchbound_2026-07-10.md` | The H100 fast lane is launch-latency bound. |
| `h200_large_n_followup_plan_2026-03-23.md` | H200 large-N status and follow-up plan. |
| `small_gpu_large_n_followup_2026-03-25.md` | Small-GPU large-N follow-up. |
| `runtime_traversal_comparison_2026-03-20.md` | Traversal capacity/timing tables. **Caution:** predates the traced-wavefront fix. |

## Pallas kernels (phase 5)

| file | |
|---|---|
| `phase5_pallas_plan.md` | **[open]** The plan and first prototype. Note its hardware constraint: Ampere+ only. |
| `phase5_m2l_a100_findings_and_padding_plan.md` | **[open]** A100 fused-M2L findings, wiring, and the padding plan, with a 2026-07-14 update. |

## The MAC

| file | |
|---|---|
| `dehnen_mass_mac_status_and_plan.md` | **[current]** Status, the leaf-size question, and a long trap list worth reading **before** any MAC measurement. |
| `dehnen_mac_next_session_prompt.md` | **[current]** The fresh-session prompt. Supersedes → |
| `dehnen_mac_step3prime_prompt.md` | **[closed]** …this, whose goal is done. Kept for its traps, not its task. |
| `treecode_mac_stability.md` | **[current]** bh box vs dehnen sphere MAC over many steps. |

## Operators and numerics

| file | |
|---|---|
| `operator_conventions.md` | **[current]** Three places where the same shape means incompatible things. Read before touching `operators/`. |
| `rotation_degeneracy_derivative.md` | **[current]** The transverse derivative at `rho == 0` (G.10) — derived, validated, and wired into every M2L/M2M/L2L lane. |
| `derivatives_and_jerk.md` | **[current]** The higher-derivative and jerk-facing APIs, and their scope limits. |

## Trees and traversal

| file | |
|---|---|
| `adaptive_traversal_design.md` | Why the solver-specific adaptive-traversal concepts moved out of yggdrax. |
| `octree_fmm_task_list.md` | **[current]** Octree-native FMM status and the remaining work items. |
| `octree_status_2026-03-15.md` | **[superseded]** Archived handoff, moved here from the repository root. |

## Audits, handoffs, and process

| file | |
|---|---|
| `refactor_audit_2026-08.md` | **[current]** The 2026-08 refactor and code-hygiene audit, and its F-table of open items. |
| `handoff_g10_gpu_validation.md` | **[open]** GPU validation for the G.10 fix. Carries a 2026-08-26 status note: two things below it have moved. |
| `handoff_tier1_gpu_validation.md` | **[closed]** **Discharged** 2026-08-14 on A100. |
| `pytest_cleanup_handoff_2026-04-22.md` | Test-suite cleanup handoff and profiling rerun. |

---

## Keeping this file honest

It is an index, not a summary: one line per file, and the line says *which question the
file answers* and *whether to trust it*, nothing more. When you add a doc, add a row.
When you supersede one, mark it and point at the successor **in both files** — two files
disagreeing with no pointer between them is the failure this index exists to prevent.

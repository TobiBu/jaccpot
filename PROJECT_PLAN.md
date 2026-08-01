# jaccpot-i Paper — Project Plan

**Goal:** One paper — differentiable, GPU-native, multi-GPU-scalable FMM
(jaccpot), with accuracy/complexity validation against Dehnen (2014) and
other literature, a multi-GPU scaling section (the clearest novel
contribution over prior FMM codes, which are single-CPU/single-GPU), a
differentiability section, and a payoff case study (gradient-based
parametric potential recovery from synthetic kinematic data).

**Repos involved:**
- `jaccpot` (code) — work happens on branch `paper/jaccpot-i`
- `jaccpot-paper-i` (new, separate repo) — tex source, synced to Overleaf via
  git bridge, figures copied in from `jaccpot`'s benchmark outputs

**Conventions (mirrors the yggdrax paper setup):**
- Benchmarks/scaling/validation runs = Python **scripts** in `bench/`,
  seeded, dump results as JSON/NPZ to `results/`. No plotting logic in
  scripts.
- Figures = **notebooks** in `examples/jaccpot_paper/`, load from `results/`
  only, never recompute.
- All new application code (the payoff case study) lives under
  `jaccpot/applications/` (not the core FMM/runtime public API) so
  paper-specific code doesn't creep into the maintained library surface.
  Core library changes (e.g. distributed-path basis/MAC convergence) go in
  the normal `jaccpot/` package, same as any other feature work.

---

## Branch / repo setup (do first)

- [ ] `jaccpot`: create branch `paper/jaccpot-i` off `main`
- [ ] New directories on that branch:
  - `jaccpot/applications/potential_recovery/`
  - `bench/multigpu/`
  - `bench/differentiability/`
  - `bench/payoff/`
  - `results/` (gitignored except `.gitkeep`; large result files should not
    be committed — only summary JSON needed for figures)
  - `examples/jaccpot_paper/` (notebooks)
- [ ] New repo `jaccpot-paper-i`:
  - `main.tex`, `sections/`, `figures/`, `refs.bib`
  - Connect to Overleaf via git bridge
  - `figures/README.md` documenting which script/notebook produced each PDF,
    so figures are regenerable, not just copy-pasted

---

## Phase 0 — Multi-GPU basis/MAC convergence audit (blocking)

**Why first:** this determines whether the multi-GPU scaling numbers (Phase
2) describe the fast-lane path or a slower interim baseline, and any gaps
here are cheap to fix now and expensive to discover mid-writing.

### Code
- [ ] Audit: per `docs/phase5_multigpu_pallas_foldin_plan.md`, items 5a–5c
  (MAC default `bh → dehnen`, basis `solidfmm(complex) → real`, Pallas M2L/P2P
  kernels on the per-device path) are the prerequisites for the distributed
  driver to match the single-GPU fast lane. Confirm current STATUS block
  before proceeding.
- [ ] If gaps exist: either close them (5a–5c), or explicitly decide the
  paper reports the current solidfmm/bh distributed path as its own baseline
  and documents convergence onto the fast lane as future work. Scope this
  explicitly — don't silently absorb a multi-week detour.
- [ ] Write `docs/multigpu_differentiability_model.md`: precise statement of
  (a) what basis/MAC the distributed path runs as of paper submission, and
  (b) the differentiability model (per-topology continuous outputs
  differentiable; MAC accept/reject and tree topology are not). One page,
  becomes most of §2 (Method) almost verbatim.

### Tests
- [ ] `tests/unit/test_distributed_mac_convergence.py`: distributed-path
  accuracy vs. direct summation, at whatever basis/MAC config is shipped for
  the paper (conformance-style, extending `tests/test_distributed_fmm_driver.py`).
- [ ] `tests/unit/test_gradient_correctness.py`: finite-difference vs.
  `jax.grad` comparison for force outputs, swept over `theta` (MAC
  tightness), single-device and (if 5a–5c land) distributed.

### Text
- [ ] Draft `sections/02_method.tex` from the markdown doc above.

---

## Phase 1 — Validation and single-device performance benchmarks

### Code
- [ ] `bench/validation/` (reuse existing `bench/bench_fmm.py`,
  `bench/bench_real_vs_complex.py` — mostly adaptation, not new code):
  force error vs. direct summation across expansion orders; error vs. theta;
  fixed-theta vs. mass-dependent MAC interaction-count comparison. Results to
  `results/validation/*.json`.
- [ ] `bench/scaling/` (reuse `bench/bench_jaxfmm_paper_compare.py`,
  `bench/profile_refresh_stage_breakdown.py`, `profile_downward_breakdown.py`):
  wall-clock vs N; interaction counts vs N; per-stage time breakdown;
  single-GPU vs CPU speedup. Results to `results/scaling/*.json`.

### Tests
- [ ] Smoke test that each bench script runs end-to-end on tiny N in CI (not
  the full sweep — just correctness of the harness).

### Notebooks
- [ ] `examples/jaccpot_paper/fig_force_error_vs_order.ipynb`
- [ ] `examples/jaccpot_paper/fig_error_vs_theta.ipynb`
- [ ] `examples/jaccpot_paper/fig_mac_comparison.ipynb`
- [ ] `examples/jaccpot_paper/fig_scaling_wallclock.ipynb`
- [ ] `examples/jaccpot_paper/fig_interaction_counts.ipynb`
- [ ] `examples/jaccpot_paper/fig_stage_breakdown.ipynb`
- [ ] `examples/jaccpot_paper/fig_gpu_vs_cpu_speedup.ipynb`

### Text
- [ ] Draft `sections/03_validation.tex`
- [ ] Draft `sections/04_complexity_performance.tex`

---

## Phase 2 — Multi-GPU scaling (the clearest novel contribution)

Dehnen's falcON and most FMM literature comparisons are single-CPU/single-GPU
— this section is where jaccpot has no direct precedent to reproduce against,
only to contextualize.

### Code
- [ ] `bench/multigpu/harness.py`: shared driver wrapping
  `jaccpot/distributed/fmm.py::distributed_fmm_accelerations` with per-stage
  timers (local build, self M2L, all_gather, coarse M2M, cross-walk, halo
  import, remote M2L, P2P) and per-GPU interaction-count collection.
- [ ] `bench/multigpu/strong_scaling.py`: fixed N, sweep #GPUs →
  `results/multigpu/strong_scaling.json`
- [ ] `bench/multigpu/weak_scaling.py`: N scaled with #GPUs →
  `results/multigpu/weak_scaling.json`
- [ ] `bench/multigpu/comm_overhead.py`: reuses `harness.py`'s per-stage
  timers, splits into comm-bound (`all_gather_coarse`, `cross_walk`,
  `halo_import`) vs compute-bound stages →
  `results/multigpu/comm_overhead.json`
- [ ] `bench/multigpu/load_balance.py`: per-GPU interaction counts on a
  clustered (Plummer/NFW-like) distribution, not `uniform_cube` — this is
  specifically where naive space-filling-curve partitioning can go unbalanced
  → `results/multigpu/load_balance.json`

### Tests
- [ ] `tests/unit/test_multigpu_harness_smoke.py`: harness runs end-to-end on
  tiny N / 1 GPU in CI (correctness of the harness, not real scaling numbers)

### Notebooks
- [ ] `examples/jaccpot_paper/fig_strong_scaling.ipynb`
- [ ] `examples/jaccpot_paper/fig_weak_scaling.ipynb`
- [ ] `examples/jaccpot_paper/fig_comm_overhead.ipynb`
- [ ] `examples/jaccpot_paper/fig_load_balance.ipynb`

### Text
- [ ] Draft `sections/05_multigpu_scaling.tex`

---

## Phase 3 — Differentiability

### Code
- [ ] `bench/differentiability/autodiff_overhead.py`: forward-only vs.
  forward+backward wall-clock ratio vs N (single-device), and vs #GPUs at
  fixed N (using `bench/multigpu/harness.py`) →
  `results/differentiability/autodiff_overhead.json`
- [ ] `bench/differentiability/grad_correctness.py`: finite-difference vs.
  autodiff gradient agreement swept across theta, small N (FD is expensive —
  subsample particles/components, don't do the full array) →
  `results/differentiability/grad_correctness.json`

### Tests
- [ ] Already covered by `tests/unit/test_gradient_correctness.py` in
  Phase 0 — extend rather than duplicate if more coverage is needed here.

### Notebooks
- [ ] `examples/jaccpot_paper/fig_autodiff_overhead.ipynb`
- [ ] `examples/jaccpot_paper/fig_grad_correctness.ipynb`

### Text
- [ ] Draft `sections/06_differentiability.tex`

---

## Phase 4 — Payoff case study: gradient-based potential recovery

### Code
- [ ] `jaccpot/applications/potential_recovery/model.py`: parametric
  gravitational potential model (small number of parameters) + synthetic
  IFU-like kinematic data generator
- [ ] `jaccpot/applications/potential_recovery/recover.py`: gradient-based
  (and optionally HMC/VI) recovery of the potential parameters from synthetic
  kinematics, using jaccpot's FMM forward pass + autodiff end-to-end
- [ ] `jaccpot/applications/potential_recovery/energy_conservation.py`: check
  `nornax`'s integrator examples first for an existing long-integration
  energy/angular-momentum conservation test before writing a new one here

### Tests
- [ ] `tests/applications/test_potential_recovery_gradients.py`:
  finite-difference check through the recovery loss
- [ ] `tests/applications/test_potential_recovery_convergence.py`: recovery
  converges to the true parameters (within tolerance) on a noiseless
  synthetic dataset

### Experiments (`bench/payoff/`)
- [ ] `energy_conservation.py`: energy/angular-momentum vs. time, long
  integration → `results/payoff/energy_conservation.json`
- [ ] `parameter_recovery_demo.py`: loss/parameter-error convergence curve →
  `results/payoff/recovery.json`

### Notebooks
- [ ] `examples/jaccpot_paper/fig_energy_conservation.ipynb`
- [ ] `examples/jaccpot_paper/fig_payoff_convergence.ipynb`

### Text
- [ ] Draft `sections/07_case_study.tex`

---

## Phase 5 — Assembly and writing

### Text
- [ ] `sections/01_introduction.tex`
- [ ] `sections/08_discussion.tex` — position relative to falcON, Taichi
  (Zhu 2021), PKDGRAV3, Bonsai/ExaFMM; limitations (load imbalance at high
  clustering, MAC non-differentiability, distributed-basis convergence
  status); future work (Jaccpot II learned MAC, Jaccpot III neural multipole
  terms, Jaccpot-Science I real IFU data, standalone Kessel Run letter)
- [ ] Abstract, related-work pass across all sections for consistency
- [ ] Full read-through checking every figure is referenced and every claim
  in text matches a `results/*.json` value (no numbers typed from memory)

### Checkpoint
- [ ] After Phase 2: assess whether the multi-GPU section has grown large
  enough that it, plus the differentiability section, could stand alone as
  a systems paper separate from the accuracy/case-study material. Decide
  then, once you see actual page budget, not now.

---

## Summary of new files (for Claude Code session scoping)

```
jaccpot/ (branch: paper/jaccpot-i)
├── jaccpot/applications/
│   └── potential_recovery/{model,recover,energy_conservation}.py
├── bench/
│   ├── multigpu/{harness,strong_scaling,weak_scaling,comm_overhead,load_balance}.py
│   ├── differentiability/{autodiff_overhead,grad_correctness}.py
│   └── payoff/{energy_conservation,parameter_recovery_demo}.py
├── tests/
│   ├── unit/{test_distributed_mac_convergence,test_gradient_correctness,
│   │         test_multigpu_harness_smoke}.py
│   └── applications/
│       └── test_potential_recovery_{gradients,convergence}.py
├── docs/multigpu_differentiability_model.md
├── results/  (json outputs, gitignored bulk data)
└── examples/jaccpot_paper/*.ipynb

jaccpot-paper-i/ (new repo, Overleaf-synced)
├── main.tex
├── sections/01_introduction.tex ... 08_discussion.tex
├── figures/ (+ README mapping each figure to its source script/notebook)
└── refs.bib
```

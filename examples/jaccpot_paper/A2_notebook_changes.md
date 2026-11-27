# Notebook changes the A2 bench port needs (2026-11-27)

The A2 port changed what several benches write. The figure notebooks have not been
edited. This file lists what each one must change before step B's results can be
plotted. Schemas not listed here changed additively only.

## §4

- **fig03 `fig_mac_comparison`:** full rewrite.
  - Plot `records` with `arm in {"dehnen_error", "geometric"}`: `step.min_s` on x,
    `rel_l2` on y. Skip rows with `valid == False`.
  - Then show `comparisons` (`cost_ratio`, `error_ratio`) and `preset_pair` (eps 1e-4
    against theta 0.8, with the stored 1.12× / 2.8× reference).
  - Gone: `pair_work`, `*p90`, `order`, `matched_p90`, and the arm values `fixed` / `mass`.
- **fig04 `fig_scaling_wallclock`:**
  - Series: `jaccpot_acceleration` is gone. It is replaced by `jaccpot_step` (the headline:
    rebuild + evaluate) and `jaccpot_evaluate`.
  - Kept: `jaccpot_potential`, `jaxfmm_potential`, `direct_acceleration`. Optional:
    `jaxfmm_potential_full`.
  - `param_set` is `"galaxy"`. Filter rows on `r["valid"]`. The y label is "per step".
  - jaxFMM rows carry `matched` / `matched_at_n`. Plot only matched ones.
- **fig05 `fig_interaction_counts`:** schema-compatible. The failed-row note hard-codes
  "(out of memory)"; print `r["error"]` instead.
- **fig06 `fig_stage_breakdown`:**
  - New `stages_s` keys: `tree_build`, `upward`, `mac_geometry`, `walk`, `m2l`, `l2l`,
    `nearfield_p2p`, `l2p`, `evaluate_other`, `unscoped`, `unmapped`, `memcpy_memset`,
    `unattributed` (device idle: launch gaps, host syncs). In split mode the keys are
    `rebuild` and `evaluate`.
  - Grey out `unscoped` and `unmapped`. Show `trace.busy_fraction` and
    `trace.launches_per_step`.
  - Remove the obsolete "strict path is GPU-only" exit.
- **fig07:** reads `gpu_vs_cpu_speedup.py --pair GPU_JSON CPU_JSON` output (old schema).
  Say "per step". Per the plan this may become a table.
- **capacity (new):** `scaling/capacity.json` has no notebook yet.

## §5

- **fig08 `fig_strong_scaling`:** input is now `bench/multigpu/from_matrix.py` output.
  - Group by `series` and `lane`. `_min_over_invocations` keys on (series, lane, ndev).
  - `median_s` is a step time on mesh rows and a force time on solo rows (`median_s_is`).
    Label the axis "time per step" or plot `force_s` separately.
  - Take efficiency relative to the one-card mesh row: it is the same code path now. Plot
    the solo row as a reference marker.
  - Drop `cfg['healthy_per_device_n']`. Show `cfg['interconnect']`. Rewrite the caption:
    invalid now means a failed verdict, a contended card or a failed row.
  - Keep PCIe and NVLink as separate artifacts.
- **fig09 `fig_weak_scaling`:** throughput = N / step time ("per step").
  `per_device_n_fixed` is kept.
- **fig10 (new notebook):** reads `multigpu/comm_overhead.json`: `per_step_mesh`
  (`exposed_collective_us_max`, `busy_us_max`, ...), `per_force_call_mesh`,
  `collective_by_kind_us`. Add rows to `export_to_paper_repo.py` and
  `build_notebooks.py`. CPU records are `meaningless_on_cpu` and never valid.
- **fig11 `fig_load_balance`:** the old keys still resolve. Relabel "interaction pairs" as
  "pair-list entries". Consider plotting `count_imbalance` next to `work_imbalance`.

## §6

- **fig12 `fig_autodiff_overhead`:**
  - `mode` is now `"step"` (the figure) or `"fixed_topology"` (secondary series), not
    jit / eager / step.
  - `cfg.wrt` is `["both"]`. Drop `forward_accel_min_s`. Invalid rows carry `error`.
  - Merge the per-N files with `autodiff_overhead_mesh.json`. Plot the mesh point with
    `lane == "mesh"`.
  - Optional: `grad_only_ratio`, `jvp.ratio`.
- **fig13 `fig_grad_correctness`:**
  - Records carry `mac_type` and `mac_param`. The x axis is `mac_param`, split by MAC:
    theta for `dehnen`, `adaptive_eps` on a log axis for `dehnen_error`. Do not use
    `r["theta"]`.
  - Filter `full_pipeline_rel_l2` on not-None and show `full_pipeline_switched`.
  - Add a panel or table from `data.oracle`: `grad_x_rel_l2`, `grad_m_rel_l2` and
    `force_rel_l2` against order.

## §7

- **fig16:** extend the `fd_P` grid to 3e8. Move the headline annotation from the
  `"10000000"` key to 1e8 or 3e8 once measured. Mark annotations with
  `beyond_measured_P`. Optionally plot `data.ratio_summary`.
- **fig19:** optionally plot `rebuild_cost.prepare_over_gradient` next to the cadence
  panel.
- **fig20:**
  - The ceiling panel must use only `failure_kind == "out_of_memory"`. Show `verdict`
    failures as not-a-ceiling.
  - ndev=1 is now a mesh point, so the ideal line's baseline is on the same code path.
- **figs 17–18:** additive only. The `--merge-from` paths now pin `lane` and refuse to
  mix September slices with new ones.

Reuse-first directory. `wallclock.py` and `gpu_vs_cpu_speedup.py` should
call `bench/bench_jaxfmm_paper_compare.py` (repo root) rather than
reimplement its N-sweep. `interaction_counts.py` adapts
`jaccpot/runtime/fmm_diagnostics.py`. `stage_breakdown.py` aggregates
`bench/profile_refresh_stage_breakdown.py` /
`bench/profile_downward_breakdown.py`. All dump JSON to `results/scaling/`.

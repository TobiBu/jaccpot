Reuse-first directory. `force_error_vs_order.py`, `error_vs_theta.py`, and
`mac_comparison.py` here should mostly adapt the existing
`bench/bench_fmm.py` / `bench/bench_real_vs_complex.py` at repo root rather
than duplicate their oracle/harness code. Each dumps a JSON file to
`results/validation/`. No plotting here -- that's
`examples/jaccpot_paper/fig_*.ipynb`.

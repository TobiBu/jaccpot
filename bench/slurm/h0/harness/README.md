# The Odisseo bench harness, vendored for the H0 bundle

`benchmark_multigpu/common/` and `benchmark_multigpu/codes/compare_force.py` are copied from Odisseo's
`benchmark_multigpu/` (branch `bench/multi-gpu`, be3bbd9, clean; not on GitHub), the `BENCH_DIR` that
`bench/fused_memory_budget.py`, `bench/dce_pins.py` and `bench/multigpu_segment_gate.py` import from:
the ICs (`common.ic`), the fp64 direct sum (`common.reference`), the contention monitor
(`common.gpu_guard`), the fast-lane environment (`codes.compare_force`). Only the modules those imports
reach are here. Changes against the source: black / isort (jaccpot's pre-commit) and one
`# noqa: F821` (a pyflakes false positive on a closure in `compare_force.py`). `setup_env.sh` copies the
tree to `$H0_ROOT/bench_dir/benchmark_multigpu`, beside the staged fp64 reference cache
(`artifacts/reference/`), and `env.sh` points `BENCH_DIR` there.

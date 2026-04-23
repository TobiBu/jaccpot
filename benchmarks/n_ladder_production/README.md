# N-Ladder Production Sweep

This directory stores artifacts from the production-oriented large-`N` GPU sweep.

The sweep now lives in [`examples/benchmark_gpu_n_ladder_production.py`](/export/home/tbuck/jaccpot/examples/benchmark_gpu_n_ladder_production.py) instead of a notebook so it is easier to rerun and easier to diff.

Expected outputs:

- `n_ladder_production_<run_id>.csv`: raw sweep rows for a single run
- `n_ladder_production_latest.csv`: latest raw sweep rows
- `n_ladder_production_recommendations.json`: machine-readable recommended and fastest stable configurations
- `n_ladder_production_recommendations.md`: human-readable recommendation table

Run it from the `odisseo` environment, for example:

```bash
micromamba run -n odisseo python examples/benchmark_gpu_n_ladder_production.py
```

Traversal tuning is size-dependent. The production sweep now carries explicit
guidance for the lean `65536/64/4096/1024` seed at `N <= 65536` and the
larger-cap `262144/256/8192/4096` seed for larger `N`, based on the
prepare-state crossover measurements recorded in
[`docs/runtime_traversal_comparison_2026-03-20.md`](/export/home/tbuck/jaccpot/docs/runtime_traversal_comparison_2026-03-20.md).

The committed recommendation files are meant to be refreshed after meaningful benchmark updates so production defaults are not trapped inside ad hoc notebook outputs.

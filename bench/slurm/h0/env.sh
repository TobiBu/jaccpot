# HoreKa bundle: the run environment. SOURCED (not run) by setup_env.sh's checks, every job and collect.sh:
#   source bench/slurm/h0/env.sh
# It reads site.env and pins.env next to it and exports:
#   H0_JACCPOT   this jaccpot checkout          H0_ROOT    the workspace (site.env)
#   H0_YGGDRAX   the yggdrax checkout in use    H0_NORNAX  nornax      H0_MIMIRAX  mimirax
#   BENCH_DIR    the Odisseo bench harness copy (ICs, the fp64 reference cache)
#   DISC_IC      the 25M disc + bulge IC
#   H0_DATA      the staged data ($H0_ROOT/data; the big files under big/: SGR_IC, A1_BAR_RESTART,
#                A1_MERGER_RESTART), H0_ODISSEO (job a4's IC tool)
#   PYTHONPATH   jaccpot, yggdrax, nornax, mimirax (no editable installs: a PYTHONPATH swap is the
#                whole of an A-B arm, e.g. H0_YGGDRAX=$H0_ROOT/src/yggdrax-pre104)
#   XLA_FLAGS    + --xla_gpu_experimental_enable_fusion_autotuner=false (production; bitwise reruns)
#
# Overrides (the CPU smoke on the development machine sets them; see local_smoke.env):
#   H0_PY_OVERRIDE (a python executable instead of the venv), H0_YGGDRAX, H0_NORNAX, H0_MIMIRAX,
#   H0_BENCH_DIR, H0_DISC_IC, H0_EXTRA_PYTHONPATH (appended), H0_SKIP_MODULES=1.
# SMOKE=1 puts smoke_site/ first on PYTHONPATH and forces the CPU (JAX_PLATFORMS=cpu, no card).
# SMOKE=1 or H0_LOCAL_SRUN=1 (a machine without SLURM: the CPU smoke, the local GPU preflight) puts
# smoke_bin/ first on PATH: its `srun` runs a step's tasks as local processes with a SLURM step's variables
# (local_srun.py), as tests/distributed's harness does.
# shellcheck shell=bash

H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H0_JACCPOT=$(cd "$H0_DIR/../../.." && pwd)
export H0_DIR H0_JACCPOT
# shellcheck source=site.env
source "$H0_DIR/site.env"
# shellcheck source=pins.env
source "$H0_DIR/pins.env"
export H0_ROOT H0_ACCOUNT H0_PARTITION H0_GPUS_PER_NODE
# the pins, for provenance.py
export H0_YGGDRAX_COMMIT H0_NORNAX_COMMIT H0_MIMIRAX_COMMIT H0_ODISSEO_COMMIT H0_YGGDRAX_PRE104_FILE_COMMIT

if [ -z "${H0_SKIP_MODULES:-}" ] && [ -n "${H0_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $H0_MODULES
fi

if [ -n "${H0_PY_OVERRIDE:-}" ]; then
  H0_PY=$H0_PY_OVERRIDE
  # the sbatch files call a bare `python`: a wrapper first on PATH
  mkdir -p "$H0_ROOT/pybin"
  printf '#!/bin/bash\nexec %q "$@"\n' "$H0_PY" > "$H0_ROOT/pybin/python"
  chmod +x "$H0_ROOT/pybin/python"
  case ":$PATH:" in *":$H0_ROOT/pybin:"*) ;; *) export PATH=$H0_ROOT/pybin:$PATH ;; esac
else
  if [ ! -x "$H0_ROOT/venv/bin/python" ]; then
    echo "env.sh: no venv at $H0_ROOT/venv (run bench/slurm/h0/setup_env.sh first)" >&2
    return 1 2>/dev/null || exit 1
  fi
  # shellcheck disable=SC1091
  source "$H0_ROOT/venv/bin/activate"
  H0_PY=$H0_ROOT/venv/bin/python
fi
export H0_PY

export H0_YGGDRAX=${H0_YGGDRAX:-$H0_ROOT/src/yggdrax}
export H0_NORNAX=${H0_NORNAX:-$H0_ROOT/src/nornax}
export H0_MIMIRAX=${H0_MIMIRAX:-$H0_ROOT/src/mimirax}
export BENCH_DIR=${H0_BENCH_DIR:-$H0_ROOT/bench_dir/benchmark_multigpu}
export DISC_IC=${H0_DISC_IC:-$H0_ROOT/data/disk_bulge_25m.npz}
export H0_DATA=${H0_DATA:-$H0_ROOT/data}
export SGR_IC=${H0_SGR_IC:-$H0_DATA/big/sgr_light_1e8_noclip.npz}
export A1_BAR_RESTART=${H0_A1_BAR_RESTART:-$H0_DATA/big/bar_restart_3.npz}
export A1_MERGER_RESTART=${H0_A1_MERGER_RESTART:-$H0_DATA/big/merger_restart_136.npz}
export H0_ODISSEO=${H0_ODISSEO:-$H0_ROOT/src/Odisseo}
if [ "${SMOKE:-}" = 1 ] || [ "${H0_LOCAL_SRUN:-}" = 1 ]; then
  case ":$PATH:" in *":$H0_DIR/smoke_bin:"*) ;; *) export PATH=$H0_DIR/smoke_bin:$PATH ;; esac
fi
# h0_paths: PYTHONPATH (and the variables that name the checkouts) from H0_YGGDRAX & co.; a job
# calls it again after it points H0_YGGDRAX at another checkout (an A-B arm)
h0_paths() {
  PYTHONPATH=$H0_JACCPOT:$H0_YGGDRAX:$H0_NORNAX:$H0_MIMIRAX${H0_EXTRA_PYTHONPATH:+:$H0_EXTRA_PYTHONPATH}
  if [ "${SMOKE:-}" = 1 ]; then
    # the CPU smoke shim (smoke_site/sitecustomize.py): FIRST, so its sitecustomize is the one run
    PYTHONPATH=$H0_DIR/smoke_site:$PYTHONPATH
  fi
  export PYTHONPATH
  # the benches' own provenance fields and the local machine's editable-finder repoint read these
  export YGGDRAX_WORKTREE=$H0_YGGDRAX NORNAX_WORKTREE=$H0_NORNAX
  export WT_JACCPOT=$H0_JACCPOT WT_YGGDRAX=$H0_YGGDRAX
}
h0_paths
if [ "${SMOKE:-}" = 1 ]; then
  # SMOKE=1: every job's commands at a tiny N on the CPU (submit_all.sh); never a card
  export H0_SMOKE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
  # the multi-process collectives probe: yggdrax's exchange (XLA:CPU cannot lower the native ragged_all_to_all)
  export M2_RAGGED=${M2_RAGGED:-exchange}
fi

export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
case " ${XLA_FLAGS:-} " in
  *xla_gpu_experimental_enable_fusion_autotuner*) ;;
  *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_experimental_enable_fusion_autotuner=false" ;;
esac
# no persistent compilation cache: a warm cache changes a process's peak memory (the B/p rows)
unset JAX_COMPILATION_CACHE_DIR

# h0_provenance OUT_DIR: the git / environment / node header every job writes first
h0_provenance() {
  mkdir -p "$1"
  "$H0_PY" "$H0_DIR/provenance.py" --out "$1/provenance.json" || echo "h0_provenance: failed (rc $?)" >&2
}

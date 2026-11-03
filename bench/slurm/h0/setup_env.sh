#!/bin/bash
# H0 bundle: build the run environment ONCE, on a HoreKa login node (it needs the internet: pip and
# GitHub over ssh). Idempotent: a step whose result is already there is checked, not redone.
#
#   bash bench/slurm/h0/setup_env.sh
#
# Builds under $H0_ROOT (site.env):
#   venv/                  python -m venv (NO --system-site-packages) with requirements.txt, --no-deps
#   src/yggdrax            yggdrax at H0_YGGDRAX_COMMIT (pins.env)
#   src/yggdrax-pre104     the same commit with yggdrax/_cell_partition.py from just before #104
#                          (a git worktree; the one-file modification is the A-B arm of job "pins")
#   src/nornax, src/mimirax  at their pinned commits
#   bench_dir/benchmark_multigpu   the Odisseo bench harness (bench/slurm/h0/harness) + fp64 references
#   data/                  the 25M disc + bulge IC and the fp64 reference cache, STAGED FROM THE
#                          DEVELOPMENT MACHINE (bench/slurm/h0/stage_data.sh; checked against data.sha256)
set -euo pipefail

H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=site.env
source "$H0_DIR/site.env"
# shellcheck source=pins.env
source "$H0_DIR/pins.env"
if [ "$H0_ROOT" = CHANGE_ME ]; then
  echo "setup_env: set H0_ROOT in $H0_DIR/site.env first (a workspace: ws_allocate h0 60)" >&2
  exit 2
fi
if [ -n "${H0_MODULES:-}" ]; then
  # shellcheck disable=SC2086
  module load $H0_MODULES
fi
mkdir -p "$H0_ROOT"/{src,data,runs,bench_dir}
# the CUDA wheels are ~3 GB: keep pip's cache off a small $HOME quota
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$H0_ROOT/pip-cache}
log() { echo "[setup_env $(date +%H:%M:%S)] $*"; }

# --- 1. the venv -----------------------------------------------------------------------------
VENV=$H0_ROOT/venv
if [ ! -x "$VENV/bin/python" ]; then
  command -v "$H0_PYTHON" > /dev/null || {
    echo "setup_env: $H0_PYTHON not found: set H0_PYTHON or H0_MODULES in site.env" >&2
    exit 2
  }
  "$H0_PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), sys.version' || {
    echo "setup_env: $H0_PYTHON is older than 3.12 (jax 0.11.2 needs >= 3.12)" >&2
    exit 2
  }
  log "venv: $VENV from $("$H0_PYTHON" -V)"
  "$H0_PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
log "pip install --no-deps -r requirements.txt (exact local pins)"
"$VENV/bin/python" -m pip install --quiet --no-deps -r "$H0_DIR/requirements.txt"
# diagnostics only (a long compile is not a hang: py-spy dump --pid before killing anything)
"$VENV/bin/python" -m pip install --quiet py-spy || log "py-spy not installed (optional)"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - "$H0_DIR/requirements.txt" <<'PYCHK'
import importlib.metadata as md, sys
bad = []
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    if not line:
        continue
    name, want = line.split("==")
    have = md.version(name)
    if have != want:
        bad.append(f"{name} {have} != {want}")
if bad:
    sys.exit("version mismatch: " + "; ".join(bad))
import jax_plugins.xla_cuda12 as p, os
venv = os.environ.get("VIRTUAL_ENV") or sys.prefix
assert p.__file__.startswith(sys.prefix), f"the CUDA plugin is not the venv's: {p.__file__}"
print("pins exact; CUDA plugin", p.__file__)
PYCHK

# --- 2. the pinned sibling checkouts ---------------------------------------------------------
checkout() {  # url commit dir
  local url=$1 commit=$2 dir=$3
  if [ ! -d "$dir/.git" ] && [ ! -f "$dir/.git" ]; then
    log "clone $url -> $dir"
    git clone --quiet "$url" "$dir"
  fi
  if ! git -C "$dir" cat-file -e "$commit^{commit}" 2> /dev/null; then
    git -C "$dir" fetch --quiet origin
  fi
  git -C "$dir" checkout --quiet --detach "$commit"
  [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ] || { echo "setup_env: $dir not at $commit" >&2; exit 1; }
  log "$(basename "$dir") at $(git -C "$dir" log -1 --format='%h %cs %s')"
}
checkout "$H0_YGGDRAX_URL" "$H0_YGGDRAX_COMMIT" "$H0_ROOT/src/yggdrax"
checkout "$H0_NORNAX_URL" "$H0_NORNAX_COMMIT" "$H0_ROOT/src/nornax"
checkout "$H0_MIMIRAX_URL" "$H0_MIMIRAX_COMMIT" "$H0_ROOT/src/mimirax"

PRE=$H0_ROOT/src/yggdrax-pre104
if [ ! -e "$PRE/.git" ]; then
  git -C "$H0_ROOT/src/yggdrax" worktree add --quiet --detach "$PRE" "$H0_YGGDRAX_COMMIT"
fi
git -C "$PRE" checkout --quiet --force --detach "$H0_YGGDRAX_COMMIT"   # a rerun starts clean
git -C "$PRE" checkout --quiet "$H0_YGGDRAX_PRE104_FILE_COMMIT" -- yggdrax/_cell_partition.py
changed=$(git -C "$PRE" diff --name-only HEAD)
[ "$changed" = "yggdrax/_cell_partition.py" ] || { echo "setup_env: pre-104 arm changed: $changed" >&2; exit 1; }
log "yggdrax-pre104: $H0_YGGDRAX_COMMIT with yggdrax/_cell_partition.py at ${H0_YGGDRAX_PRE104_FILE_COMMIT:0:8}"

# --- 3. the bench harness and the staged data ------------------------------------------------
BD=$H0_ROOT/bench_dir/benchmark_multigpu
mkdir -p "$BD/artifacts/reference"
cp -r "$H0_DIR/harness/benchmark_multigpu/common" "$H0_DIR/harness/benchmark_multigpu/codes" "$BD/"
if [ -f "$H0_ROOT/data/disk_bulge_25m.npz" ]; then
  log "checking the staged data against data.sha256"
  (cd "$H0_ROOT/data" && sha256sum --quiet -c "$H0_DIR/data.sha256")
  cp -n "$H0_ROOT"/data/reference/*.npy "$BD/artifacts/reference/" 2> /dev/null || true
else
  log "WARNING: $H0_ROOT/data/disk_bulge_25m.npz missing: run bench/slurm/h0/stage_data.sh on the"
  log "         development machine (it copies the IC and the fp64 references here), then rerun this."
fi

# --- 4. the imports, from the jobs' own environment -------------------------------------------
H0_SKIP_MODULES=1 source "$H0_DIR/env.sh"
JAX_PLATFORMS=cpu "$H0_PY" - <<'PYIMP'
import jax, jaccpot, yggdrax, nornax, mimirax, optax, equinox, diffrax
import os
for m in (jaccpot, yggdrax, nornax, mimirax):
    print(f"{m.__name__:8s} {os.path.dirname(m.__file__)}")
print("jax", jax.__version__, "devices (login node, CPU):", jax.devices())
PYIMP
date -Is > "$H0_ROOT/setup.done"
log "done. Next: bash bench/slurm/h0/submit_all.sh --dry-run, then without --dry-run"

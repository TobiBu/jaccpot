#!/bin/bash
# GPU tests, serial (the repo's addopts say -n auto), no preallocation
source /tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad/r6/common.sh
W=${1:-/export/home/tbuck/jaccpot-r6-k3}; shift
own
(cd $W && PYTHONPATH=$SITE JACCPOT_WORKTREE=$W YGGDRAX_WORKTREE=$YGG CUDA_VISIBLE_DEVICES=$CARD \
  XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 3600 $PY -m pytest -n 0 -p no:cacheprovider "$@" 2>&1 | tail -25)
release

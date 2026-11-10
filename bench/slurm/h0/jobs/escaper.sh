#!/bin/bash
# HoreKa job "escaper" (~1.5-2 h): ESC's escaper cost at 1e8 on ONE card (bench/slurm/escaper_timing.sbatch):
# the full unclipped Sgr IC with escapers on / off / on-but-nobody-escapes / energy every 10th edge, both MACs;
# XLA_PYTHON_CLIENT_PREALLOCATE=true (ESC: 1e8 on one card needs it; ESC_PREALLOCATE=false overrides).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight escaper
h0_need_data "$SGR_IC"
IC=$SGR_IC
if [ "${SMOKE:-}" = 1 ]; then
  IC=$OUT/sgr_tiny.npz
  [ -s "$IC" ] || "$H0_PY" "$H0_DIR/tiny_ic.py" "$SGR_IC" "$IC" 4096
  export STEPS=2 SEGMENTS=2 ROWS=${ROWS:-'^(off_dehnen|esc_derr|escclip_dehnen|esc_derr_e10)$'}
fi
PY=$H0_PY JACCPOT=$H0_JACCPOT IC=$IC OUT=$OUT/timing bash bench/slurm/escaper_timing.sbatch
h0_done

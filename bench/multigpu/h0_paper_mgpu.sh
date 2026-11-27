#!/bin/bash
# H0 job "paper_mgpu" (~2 h, one exclusive 4-card node): Jaccpot I figures 10 and 11 on the fused mesh lane.
# Figures 08 / 09 need no job of their own: job "scaling_a" runs the matrix's strong and weak tiers; convert its
# output with bench/multigpu/from_matrix.py (below, after the run, on any machine).
#
# TO ADD IT TO THE BUNDLE (not done here; the bundle is shared):
#   cp bench/multigpu/h0_paper_mgpu.sh bench/slurm/h0/jobs/paper_mgpu.sh
#   and add to JOBS in bench/slurm/h0/submit_all.sh (nodes|tasks-per-node|time|exclusive|label, as scaling_a):
#     "paper_mgpu|1|1|02:00:00|yes|Jaccpot I figs 10-11: comm vs compute trace, per-device work"
#   plus a preflight entry "paper_mgpu" if the preflight should cover it (h0_need_preflight below reads it).
#
# shellcheck source=../slurm/h0/jobs/_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight scaling
PER=${PER:-4000000}            # the weak tier's particles per card
SEG=${SEG:-2} STEPS=${STEPS:-8}
LB_ARGS=() CO_ARGS=()
if [ "${SMOKE:-}" = 1 ]; then
  PER=1024 SEG=0 STEPS=2
  CO_ARGS=(--warm-segments 1 --time-segments 1 --trace-segments 1 --trace-force 1)
fi
# figure 11: per-device pair work, Plummer and the uniform contrast (and the 25M disc + bulge prefix), 2 and 4 cards
DISTS=plummer,uniform
if [ -s "${DISC_IC:-}" ] && [ "${SMOKE:-}" != 1 ]; then DISTS=$DISTS,disc; LB_ARGS+=(--disc-ic "$DISC_IC"); fi
"$H0_PY" -m bench.multigpu.load_balance --ndevs 2,4 --per-device-n "$PER" --distributions "$DISTS" \
  --segments "$SEG" --steps "$STEPS" --gpu-select none "${LB_ARGS[@]}" \
  --json-out "$OUT/load_balance.json" > "$OUT/load_balance.log" 2>&1 \
  || echo "paper_mgpu: load_balance rc $? (invalid points are recorded; see the log)"
# figure 10: collective vs compute device time per step (profiler trace + HLO dump), 2 and 4 cards
"$H0_PY" -m bench.multigpu.comm_overhead --ndevs 2,4 --per-device-n "$PER" --steps "$STEPS" \
  --gpu-select none "${CO_ARGS[@]}" --keep-trace-dir "$OUT/comm_traces" \
  --json-out "$OUT/comm_overhead.json" > "$OUT/comm_overhead.log" 2>&1 \
  || echo "paper_mgpu: comm_overhead rc $? (see the log)"
h0_done
# Afterwards, on the development machine (the collected tarball unpacked as RUN/):
#   python -m bench.multigpu.from_matrix --matrix-dir RUN/scaling_a/matrix
#   cp RUN/paper_mgpu/load_balance.json  bench/results/multigpu/load_balance.json
#   cp RUN/paper_mgpu/comm_overhead.json bench/results/multigpu/comm_overhead.json

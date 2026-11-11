#!/bin/bash
# HoreKa job "a4": the A4 accuracy suite. On HoreKa submit_all.sh does NOT sbatch this file: it delegates to
# bench/slurm/a4/submit.sh (an IC job on the CPU partition, one job array per ladder and tier, the summary), with
# the run directory inside the bundle's run and afterok on the preflight. This script is the CPU smoke's path:
# a4/submit.sh --local-smoke (every rung shrunk, the node tier on 4 forced host devices; the IC job only when
# A4_AGAMA_PY and A4_ODISSEO are set).
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
if [ "${SMOKE:-}" != 1 ]; then
  echo "job a4 is submitted by bench/slurm/h0/submit_all.sh --only a4 (bench/slurm/a4/submit.sh), not as one job" >&2
  exit 2
fi
# one ladder per IC family (sphere, galaxy disc, bar, merger) in both tiers; the other ladders differ in their
# rungs' parameters (A4's own gate ran every ladder's smoke); A4_SMOKE_ARGS="" runs all
# shellcheck disable=SC2086
A4_PY=$H0_PY A4_RUN=$OUT A4_IC=${A4_IC:-$OUT/ics} bash bench/slurm/a4/submit.sh --local-smoke \
  ${A4_SMOKE_ARGS---only plummer_n,disc_n,bar_n,merger_n}
h0_done

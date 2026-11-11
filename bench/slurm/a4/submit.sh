#!/bin/bash
# A4 accuracy suite: submit the HoreKa jobs (H0's account / partition / venv: bench/slurm/h0/site.env and
# setup_env.sh once; A4's agama python and Odisseo checkout: bench/slurm/a4/site.env).
#
#   bash bench/slurm/a4/submit.sh [--dry-run] [--tier 1card|node|both] [--only L1,L2,...] [--no-ics]
#   bash bench/slurm/a4/submit.sh --local-smoke     # gate 4: every job's exact commands at tiny N on this CPU
#
# The HoreKa bundle runs this as its job "a4" (bash bench/slurm/h0/submit_all.sh --only a4): A4_RUN inside the
# bundle's run, A4_DEPENDENCY (e.g. afterok:<preflight id>) on the IC job and every array, A4_PREFLIGHT (the
# preflight's status file for a4: a rung exits at once when the preflight found the entry broken).
#
# Jobs: ics (CPU, agama; skipped when every IC exists) -> one array per ladder and tier (afterok:ics) ->
# summarize (afterany: every array). Rerunning into the same A4_RUN resumes: rungs with a result are skipped.
set -euo pipefail
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JACCPOT=$(cd "$DIR/../../.." && pwd)
DRY=0; TIER=both; ONLY=""; ICS=1; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --tier) TIER=$2; shift ;;
    --only) ONLY=$2; shift ;;
    --no-ics) ICS=0 ;;
    --local-smoke) SMOKE=1 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

# per ladder: "<--time 1card> <--time node>" (the longest rung x ~1.5; from the local A100 rows and the
# 2026-10-31 runs: 1e8 ~1 s/step, linear in N above ~1e6; dehnen_error ~1.1x; fp64 force ~2.5x)
declare -A WALL=(
  [plummer_n]="02:00:00 02:00:00" [hernquist_n]="02:00:00 02:00:00" [nfw_n]="02:00:00 02:00:00"
  [plummer_dt]="02:00:00 02:00:00" [plummer_mac]="02:00:00 02:00:00" [plummer_precision]="03:00:00 04:00:00"
  [disc_n]="03:00:00 06:00:00" [disc_dt]="03:00:00 06:00:00" [disc_mac]="03:00:00 06:00:00"
  [disc_precision]="06:00:00 12:00:00"
  [bar_n]="10:00:00 16:00:00"
  [merger_n]="06:00:00 12:00:00" [merger_dt]="08:00:00 16:00:00" [merger_mac]="06:00:00 12:00:00"
)

PY=${A4_PY:-python3}
if [ "$SMOKE" = 1 ]; then
  # Gate 4's local smoke: every job's exact command line at tiny N (A4_SMOKE=1 shrinks every rung),
  # on this host's CPU (the node tier on 4 forced host devices), plus the IC job at 1e-4 scale.
  : "${A4_PY:?--local-smoke needs A4_PY (e.g. /export/scratch/tbuck/mn-scratch/bin/py-a4)}"
  cd "$JACCPOT"
  RUN=${A4_RUN:-$JACCPOT/.a4-smoke}
  export A4_LOCAL_SMOKE=1 A4_SMOKE=1 A4_RUN=$RUN A4_JACCPOT=$JACCPOT JAX_PLATFORMS=cpu
  export A4_IC=${A4_IC:-$RUN/ics} A4_IC_SCALE=${A4_IC_SCALE:-0.0001}
  if [ -n "${A4_AGAMA_PY:-}" ] && [ -n "${A4_ODISSEO:-}" ]; then
    bash "$DIR/ics.sbatch"
  else
    echo "local smoke: A4_AGAMA_PY / A4_ODISSEO unset: the galaxy ladders need ICs in $A4_IC"
  fi
  fails=0
  for tier in horeka-1card horeka-node; do
    case $TIER in 1card) [ "$tier" = horeka-1card ] || continue ;; node) [ "$tier" = horeka-node ] || continue ;; esac
    if [ "$tier" = horeka-node ]; then export XLA_FLAGS=--xla_force_host_platform_device_count=4; else unset XLA_FLAGS; fi
    for ladder in $("$A4_PY" -c "import sys; sys.path.insert(0, '$JACCPOT'); from bench.accuracy.ladders import ladder_names; print(' '.join(ladder_names('$tier')))"); do
      [ -z "$ONLY" ] || [[ ",$ONLY," == *",$ladder,"* ]] || continue
      n=$("$A4_PY" -m bench.accuracy.suite ladder "$ladder" --tier "$tier" | wc -l)
      for ((i = 0; i < n; i++)); do
        if (cd "$JACCPOT" && A4_TIER=$tier A4_LADDER=$ladder SLURM_ARRAY_TASK_ID=$i bash -c \
              "source bench/slurm/a4/_prologue.sh; a4_task $ladder $tier $i"); then :; else fails=$((fails + 1)); fi
      done
    done
  done
  export A4_TIER=summary
  bash "$DIR/summarize.sbatch" || fails=$((fails + 1))
  echo "local smoke: $fails failure(s); results in $RUN"
  exit $((fails > 0))
fi

# shellcheck source=../h0/site.env
source "$JACCPOT/bench/slurm/h0/site.env"
# shellcheck source=site.env
source "$DIR/site.env"
if [ "$DRY" = 0 ]; then
  for v in H0_ACCOUNT H0_ROOT A4_AGAMA_PY A4_ODISSEO; do
    [ "${!v}" != CHANGE_ME ] || { echo "set $v (bench/slurm/h0/site.env or bench/slurm/a4/site.env)" >&2; exit 2; }
  done
  [ -f "$H0_ROOT/setup.done" ] || { echo "run bench/slurm/h0/setup_env.sh first" >&2; exit 2; }
fi
RUN=${A4_RUN:-$H0_ROOT/a4/$(date +%Y%m%d-%H%M%S)}
COMMON=(-A "$H0_ACCOUNT" --chdir="$JACCPOT")
EXPORT=ALL,A4_JACCPOT="$JACCPOT",A4_RUN="$RUN",A4_IC="$A4_IC"${A4_PREFLIGHT:+,A4_PREFLIGHT="$A4_PREFLIGHT"}
# shellcheck disable=SC2206
EXTRA=($H0_SBATCH_EXTRA)
submit() {  # submit NAME SBATCH_ARGS...: prints the command; on a real run, the job id
  local name=$1; shift
  echo "sbatch ${COMMON[*]} ${EXTRA[*]} $*" >&2
  if [ "$DRY" = 1 ]; then echo "DRY_$name"; return 0; fi
  mkdir -p "$RUN/logs"
  sbatch --parsable "${COMMON[@]}" "${EXTRA[@]}" "$@"
}
dep=${A4_DEPENDENCY:+--dependency=$A4_DEPENDENCY}
if [ "$ICS" = 1 ]; then
  if [ "$DRY" = 1 ] || ! { [ -s "$A4_IC/galaxy.npz" ] && [ -s "$A4_IC/bar.npz" ] && [ -s "$A4_IC/merger.npz" ]; }; then
    ics=$(submit ics -p "$A4_CPU_PARTITION" --export="$EXPORT" --output="$RUN/logs/%x-%j.out" $dep "$DIR/ics.sbatch")
    dep="--dependency=afterok:$ics${A4_DEPENDENCY:+,$A4_DEPENDENCY}"
  fi
fi
arrays=()
for tier in horeka-1card horeka-node; do
  case $TIER in 1card) [ "$tier" = horeka-1card ] || continue ;; node) [ "$tier" = horeka-node ] || continue ;; both) ;; *) echo "--tier 1card|node|both" >&2; exit 2 ;; esac
  sb=$DIR/card1.sbatch; col=1; mem=(--mem=125G)
  if [ "$tier" = horeka-node ]; then sb=$DIR/node.sbatch; col=2; mem=(--mem="$H0_MEM"); fi
  for ladder in $(python3 -c "import sys; sys.path.insert(0, '$JACCPOT'); from bench.accuracy.ladders import ladder_names; print(' '.join(ladder_names('$tier')))"); do
    [ -z "$ONLY" ] || [[ ",$ONLY," == *",$ladder,"* ]] || continue
    n=$(cd "$JACCPOT" && python3 -c "import sys; sys.path.insert(0, '.'); from bench.accuracy.ladders import ladder_spec; print(len(ladder_spec('$ladder', '$tier')[2]))")
    wall=$(awk -v c="$col" '{print $c}' <<<"${WALL[$ladder]:-08:00:00 08:00:00}")
    id=$(submit "$ladder" -p "$H0_PARTITION" --array=0-$((n - 1)) --time="$wall" "${mem[@]}" $dep \
         --job-name="a4-$ladder" --output="$RUN/logs/%x-%A_%a.out" \
         --export="$EXPORT",A4_LADDER="$ladder" "$sb")
    arrays+=("$id")
  done
done
deps=$(IFS=:; echo "${arrays[*]}")
submit summary -p "$A4_CPU_PARTITION" --export="$EXPORT" --dependency=afterany:"$deps" --output="$RUN/logs/%x-%j.out" \
  "$DIR/summarize.sbatch" > /dev/null
echo "# run: $RUN  (results/<tier>/<ladder>/<rung>/result.json; SUMMARY-<tier>.md when all ended)"
echo "# send back: tar czf a4.tgz --exclude='*.npz' -C $RUN ."

#!/bin/bash
# H0 bundle: submit every job (or run them all locally on the CPU at a tiny N).
#
#   bash bench/slurm/h0/submit_all.sh --dry-run      # print the sbatch commands, submit nothing
#   bash bench/slurm/h0/submit_all.sh                # submit: preflight first, the rest afterok on it
#   bash bench/slurm/h0/submit_all.sh --only pins,auto   # a subset (the preflight is always included)
#   H0_SERIAL=1 bash bench/slurm/h0/submit_all.sh    # one node at a time: each job afterany the previous
#   H0_RUN_ID=20261104-0900 bash ...                 # resubmit into an existing run: rows already there
#                                                    # are skipped, so a job that ran out of time resumes
#   SMOKE=1 [H0_LOCAL_ENV=bench/slurm/h0/local_smoke.env] bash bench/slurm/h0/submit_all.sh  # run each job's commands HERE, on the CPU, at a tiny
#                                                    # N (no sbatch, no card): checks every entry point
#                                                    # and argument before the node-hours are spent
#
# Every job: one exclusive node (--exclusive, all four GPUs, --mem from site.env), its own --time, the
# production XLA flag (env.sh), a provenance.json header, outputs under $H0_ROOT/runs/<run id>/<job>/.
set -euo pipefail
H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
H0_JACCPOT=$(cd "$H0_DIR/../../.." && pwd)
if [ -n "${H0_LOCAL_ENV:-}" ]; then
  # a machine's own overrides (local_smoke.env: the development machine's CPU smoke), exported
  set -a
  # shellcheck disable=SC1090
  source "$H0_LOCAL_ENV"
  set +a
fi
# shellcheck source=site.env
source "$H0_DIR/site.env"

DRY=0
ONLY=
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --only) ONLY=$2; shift ;;
    -h | --help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "submit_all: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

# name | --time | what (README.md has the expected wall times; the limits leave ~1.5-2x headroom)
JOBS=(
  "preflight|00:45:00|the environment on a GPU node + one tiny row per entry point"
  "pins|03:00:00|L1/L2/L3 + A-vs-A control + #104 cummax A-B-A-B-A at 1e8"
  "auto|03:00:00|cell_min_level=auto vs 8 at 1e5/1e6/1e7 (mimirax S10 bench)"
  "s1b|02:00:00|S1b Mosaic vs Triton L2P at 2e6/2.5e7/1e8"
  "grad|09:00:00|S9 rollout-gradient + S8 force-gradient timings"
  "scaling_a|08:00:00|multi-GPU matrix: strong, weak, memory (4-card ceiling)"
  "scaling_b|12:00:00|multi-GPU matrix: migration on/off (ab tier)"
)

if [ "${SMOKE:-}" != 1 ]; then
  for v in H0_ACCOUNT H0_ROOT; do
    if [ "${!v}" = CHANGE_ME ]; then
      if [ "$DRY" = 1 ]; then
        echo "# NOTE: $v is still CHANGE_ME in site.env (a real submission refuses)"
      else
        echo "submit_all: set $v in $H0_DIR/site.env first" >&2
        exit 2
      fi
    fi
  done
  if [ "$DRY" = 0 ] && [ ! -f "$H0_ROOT/setup.done" ]; then
    echo "submit_all: $H0_ROOT/setup.done missing: run bench/slurm/h0/setup_env.sh first" >&2
    exit 2
  fi
fi

RUN_ID=${H0_RUN_ID:-$(date +%Y%m%d-%H%M%S)}
if [ "${SMOKE:-}" = 1 ]; then
  H0_RUN=$H0_ROOT/smoke/$RUN_ID
else
  H0_RUN=$H0_ROOT/runs/$RUN_ID
fi
[ "$DRY" = 1 ] || mkdir -p "$H0_RUN/logs"
echo "# H0 run $RUN_ID -> $H0_RUN (jaccpot $(git -C "$H0_JACCPOT" rev-parse --short HEAD))"

selected() {  # name
  [ -z "$ONLY" ] || [ "$1" = preflight ] || [[ ",$ONLY," == *",$1,"* ]]
}

if [ "${SMOKE:-}" = 1 ]; then
  # locally, one job after the other, on the CPU; a failed job does not stop the others
  failed=()
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name _limit what <<< "$spec"
    selected "$name" || continue
    if [ -f "$H0_RUN/$name/DONE" ]; then echo "=== SMOKE $name: done before, skipped"; continue; fi
    echo "=== SMOKE $name: $what  $(date -Is)"
    start=$SECONDS
    if SMOKE=1 H0_JACCPOT=$H0_JACCPOT H0_RUN=$H0_RUN H0_JOB=$name \
      bash "$H0_DIR/jobs/$name.sh" > "$H0_RUN/logs/$name.out" 2>&1; then
      echo "    ok in $((SECONDS - start)) s (log $H0_RUN/logs/$name.out)"
    else
      rc=$?
      echo "    FAILED rc=$rc in $((SECONDS - start)) s (log $H0_RUN/logs/$name.out)"
      failed+=("$name")
    fi
  done
  echo "# SMOKE done: failed: ${failed[*]:-none}"
  [ ${#failed[@]} = 0 ]
  exit
fi

prev=
pre=
n=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r name limit what <<< "$spec"
  selected "$name" || continue
  part=$H0_PARTITION
  [ "$name" = preflight ] && part=$H0_PREFLIGHT_PARTITION
  dep=()
  if [ "${H0_SERIAL:-}" = 1 ] && [ -n "$prev" ]; then
    # one node at a time; the preflight must still have passed
    if [ "$prev" = "$pre" ]; then
      dep=(--dependency="afterok:$pre")
    else
      dep=(--dependency="afterany:$prev${pre:+,afterok:$pre}")
    fi
  elif [ -n "$pre" ]; then
    dep=(--dependency="afterok:$pre")
  fi
  # shellcheck disable=SC2206
  cmd=(sbatch --parsable --job-name="h0-$name" --account="$H0_ACCOUNT" --partition="$part"
    --nodes=1 --ntasks=1 --gres="gpu:$H0_GPUS_PER_NODE" --exclusive --mem="$H0_MEM" --time="$limit"
    --output="$H0_RUN/logs/%x-%j.out" --chdir="$H0_JACCPOT"
    --export="ALL,H0_JACCPOT=$H0_JACCPOT,H0_RUN=$H0_RUN,H0_JOB=$name"
    "${dep[@]}" $H0_SBATCH_EXTRA "$H0_DIR/jobs/$name.sh")
  if [ "$DRY" = 1 ]; then
    n=$((n + 1))
    id=DRY$n
    echo "${cmd[*]}    # $what"
  else
    id=$("${cmd[@]}")
    id=${id%%;*}
    echo "$name: job $id ($limit; $what)"
    echo "$name $id $limit" >> "$H0_RUN/jobs.txt"
  fi
  [ "$name" = preflight ] && pre=$id
  prev=$id
done
[ "$DRY" = 1 ] || echo "# watch: squeue -u \$USER; when all are done: bash bench/slurm/h0/collect.sh $H0_RUN"

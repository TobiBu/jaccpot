#!/bin/bash
# HoreKa bundle: submit every timing job (or run them all locally on the CPU at a tiny N).
#
#   bash bench/slurm/h0/submit_all.sh --list         # the job table (nodes, tasks, --time, what)
#   bash bench/slurm/h0/submit_all.sh --dry-run      # print the sbatch commands, submit nothing
#   bash bench/slurm/h0/submit_all.sh                # submit the default set: preflight first, the rest afterok on it
#   bash bench/slurm/h0/submit_all.sh --only pins,auto   # a subset (the preflight is always included)
#   bash bench/slurm/h0/submit_all.sh --only a4      # the A4 accuracy suite (not in the default set: it needs
#                                                    # A4_AGAMA_PY in bench/slurm/a4/site.env; A4_ARGS passes
#                                                    # options to bench/slurm/a4/submit.sh, e.g. "--tier 1card")
#   bash bench/slurm/h0/submit_all.sh --all          # the default set and a4
#   H0_SERIAL=1 bash bench/slurm/h0/submit_all.sh    # one job at a time: each afterany the previous
#   H0_RUN_ID=20261110-0900 bash ...                 # resubmit into an existing run: rows already there
#                                                    # are skipped, so a job that ran out of time resumes
#   SMOKE=1 [H0_LOCAL_ENV=bench/slurm/h0/local_smoke.env] bash bench/slurm/h0/submit_all.sh  # run each job's
#                                                    # commands HERE, on the CPU, at a tiny N (no sbatch, no card;
#                                                    # the multi-process steps as local processes, gloo)
#
# Every job: exclusive nodes (--exclusive, all four GPUs, --mem from site.env), its own --time and node count,
# the production XLA flag (env.sh), a provenance.json header, outputs under $H0_ROOT/runs/<run id>/<job>/ and
# a DONE file last. README.md has the expected walls and the node-hour budget.
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
LIST=0
ONLY=
ALL=0
while [ $# -gt 0 ]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --list) LIST=1 ;;
    --only) ONLY=$2; shift ;;
    --all) ALL=1 ;;
    -h | --help) sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "submit_all: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

# name | nodes | tasks per node | --time | default set? | what
# (README.md has the expected wall times; the limits leave ~1.5-2x headroom). Jobs with several tasks per node
# start their multi-process steps with srun inside the allocation (the single-process steps run in the batch
# step). "a4" is delegated to bench/slurm/a4/submit.sh (its own IC job and job arrays, afterok the preflight).
G=$H0_GPUS_PER_NODE
JOBS=(
  "preflight|1|$G|02:00:00|yes|the environment on a GPU node + one tiny row per entry point"
  "pins|1|1|03:00:00|yes|L1/L2/L3 + A-vs-A control + #104 cummax A-B-A-B-A at 1e8"
  "auto|1|1|03:00:00|yes|cell_min_level=auto vs 8 at 1e5/1e6/1e7 (mimirax S10 bench)"
  "s1b|1|1|02:00:00|yes|S1b Mosaic vs Triton L2P at 2e6/2.5e7/1e8"
  "grad|1|1|09:00:00|yes|S9 rollout-gradient + S8 force-gradient timings"
  "scaling_a|1|1|08:00:00|yes|multi-GPU matrix: strong, weak, memory (4-card ceiling)"
  "scaling_mp|1|$G|06:00:00|yes|multi-GPU matrix mp tier: N processes x 1 GPU vs 1 process x N GPUs"
  "scaling_b|1|1|12:00:00|yes|multi-GPU matrix: migration on/off (ab tier)"
  "forward|1|1|16:00:00|yes|forward mode (jvp/jacfwd, groups, tpp) + the demonstrator"
  "mgrad|1|1|12:00:00|yes|WP-G multi-GPU reverse gradients (weak/strong/memory/check)"
  "escaper|1|1|04:00:00|yes|ESC: escapers on/off at 1e8, one card (the Sgr IC)"
  "a1|1|1|06:00:00|yes|A1: the dehnen_error stall from the 1e8 restarts (net/floor/geo07) + err rows"
  "a3|1|1|04:00:00|yes|A3: summed vs legacy drift at 1e8 and 2.6e5"
  "m1_node1|1|$G|02:00:00|yes|M1: momentum local vs cross, 1 GPU and 4 x 1 GPU"
  "m1_node2|2|$G|01:30:00|yes|M1: momentum local vs cross, 8 x 1 GPU over two nodes"
  "m2g_node1|1|$G|03:30:00|yes|M2G: 4 processes x 1 GPU vs 1 x 4, bitwise + timing"
  "m2g_node2|2|$G|02:00:00|yes|M2G: 8 x 1 GPU over two nodes + the barrier probe"
  "mesc_node1|1|$G|03:00:00|yes|MESC: escapers on the mesh at 1e8, 4 x 1 GPU (the Sgr IC)"
  "mesc_node2|2|$G|02:30:00|yes|MESC: escapers on the mesh at 1e8, 8 x 1 GPU (the Sgr IC)"
  "a4|-|-|per ladder|no|A4 accuracy suite: IC job (CPU) + one array per ladder and tier + summary"
)

if [ "$LIST" = 1 ]; then
  printf '%-11s %5s %6s %10s %7s  %s\n' job nodes tasks --time default what
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name nodes tasks limit def what <<< "$spec"
    printf '%-11s %5s %6s %10s %7s  %s\n' "$name" "$nodes" "$tasks" "$limit" "$def" "$what"
  done
  exit 0
fi

if [ -n "$ONLY" ]; then
  known=" $(for spec in "${JOBS[@]}"; do printf '%s ' "${spec%%|*}"; done)"
  for name in ${ONLY//,/ }; do
    [[ "$known" == *" $name "* ]] || { echo "submit_all: unknown job '$name' (--list)" >&2; exit 2; }
  done
fi

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
echo "# HoreKa bundle run $RUN_ID -> $H0_RUN (jaccpot $(git -C "$H0_JACCPOT" rev-parse --short HEAD))"

selected() {  # name default
  if [ "$1" = preflight ]; then return 0; fi
  if [ -n "$ONLY" ]; then [[ ",$ONLY," == *",$1,"* ]]; return; fi
  [ "$2" = yes ] || [ "$ALL" = 1 ]
}

if [ "${SMOKE:-}" = 1 ]; then
  # locally, one job after the other, on the CPU; a failed job does not stop the others. The multi-process
  # jobs run their steps at two processes: one node's jobs with 2 "GPUs" per node, two nodes' with 1.
  failed=()
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r name nodes _tasks _limit def what <<< "$spec"
    selected "$name" "$def" || continue
    if [ -f "$H0_RUN/$name/DONE" ]; then echo "=== SMOKE $name: done before, skipped"; continue; fi
    echo "=== SMOKE $name: $what  $(date -Is)"
    gpn=2
    [ "$nodes" = 2 ] && gpn=1
    start=$SECONDS
    if SMOKE=1 H0_JACCPOT=$H0_JACCPOT H0_RUN=$H0_RUN H0_JOB=$name H0_GPUS_PER_NODE=$gpn \
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
nodehours=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r name nodes tasks limit def what <<< "$spec"
  if ! selected "$name" "$def"; then
    [ "$name" = a4 ] && [ "$DRY" = 1 ] && [ -z "$ONLY" ] && echo "# a4 not selected (--only a4 or --all; README: its budget)"
    continue
  fi
  dep=()
  if [ "${H0_SERIAL:-}" = 1 ] && [ -n "$prev" ]; then
    # one job at a time; the preflight must still have passed
    if [ "$prev" = "$pre" ]; then
      dep=(--dependency="afterok:$pre")
    else
      dep=(--dependency="afterany:$prev${pre:+,afterok:$pre}")
    fi
  elif [ -n "$pre" ]; then
    dep=(--dependency="afterok:$pre")
  fi
  if [ "$name" = a4 ]; then
    # delegated: the A4 suite's own submit (IC job, one array per ladder and tier, the summary), into this run
    a4_dep=${dep[0]:-}
    a4_dep=${a4_dep#--dependency=}
    a4=(env A4_RUN="$H0_RUN/a4" A4_DEPENDENCY="$a4_dep" A4_PREFLIGHT="$H0_RUN/preflight/status/a4"
      bash "$H0_DIR/../a4/submit.sh")
    [ "$DRY" = 1 ] && a4+=(--dry-run)
    # shellcheck disable=SC2206
    a4+=(${A4_ARGS:-})
    echo "# a4: ${a4[*]}"
    "${a4[@]}"
    continue
  fi
  part=$H0_PARTITION
  [ "$name" = preflight ] && part=$H0_PREFLIGHT_PARTITION
  # shellcheck disable=SC2206
  cmd=(sbatch --parsable --job-name="h0-$name" --account="$H0_ACCOUNT" --partition="$part"
    --nodes="$nodes" --ntasks-per-node="$tasks" --gres="gpu:$H0_GPUS_PER_NODE" --exclusive --mem="$H0_MEM"
    --time="$limit" --output="$H0_RUN/logs/%x-%j.out" --chdir="$H0_JACCPOT"
    --export="ALL,H0_JACCPOT=$H0_JACCPOT,H0_RUN=$H0_RUN,H0_JOB=$name"
    "${dep[@]}" $H0_SBATCH_EXTRA "$H0_DIR/jobs/$name.sh")
  IFS=: read -r hh mm _ <<< "$limit"
  nodehours=$((nodehours + nodes * (10#$hh * 60 + 10#$mm)))
  if [ "$DRY" = 1 ]; then
    n=$((n + 1))
    id=DRY$n
    echo "${cmd[*]}    # $what"
  else
    id=$("${cmd[@]}")
    id=${id%%;*}
    echo "$name: job $id ($nodes node(s), $limit; $what)"
    echo "$name $id $limit $nodes" >> "$H0_RUN/jobs.txt"
  fi
  [ "$name" = preflight ] && pre=$id
  prev=$id
done
echo "# node-hours if every sbatch job above hit its --time: $((nodehours / 60)) (README.md: the expected budget)"
[ "$DRY" = 1 ] || echo "# watch: squeue -u \$USER; when all are done: bash bench/slurm/h0/collect.sh $H0_RUN"

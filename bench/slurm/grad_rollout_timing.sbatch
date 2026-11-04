#!/bin/bash
# S9 session 2's timings: the rollout gradient against its forward (bench/grad_rollout_timing.py) and S8's
# single-force gradient (bench/grad_production_gate.py), on ONE card of a dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths
# below. One process at a time, on the allocation's first card only: "offload" moves its residuals over PCIe,
# and a second process on the node would share the host link. Resumable: a row whose JSON exists is skipped,
# so a resubmission continues where the last one stopped (a row that failed left no JSON and runs again).
#
#   sbatch bench/slurm/grad_rollout_timing.sbatch                  # the whole matrix (~4-5 h, rough)
#   ROWS='^r_8e6' sbatch bench/slurm/grad_rollout_timing.sbatch    # the rows whose tag matches a regex
#   python bench/grad_rollout_timing.py --summarize $OUT/rows      # the table, with the ratios
#
# What the rows answer (preset="galaxy", clipped Plummer, float32, dehnen_error, dt 1e-3):
#   s8_*            S8's single-force gradient: grad / value_and_grad over the force eval (the gate's own
#                   unclipped Plummer). s8_8e6_galaxy: preset="galaxy", eps 1e-3 (S8's galaxy row), with a
#                   trace -- the per-kernel split of the gradient (the near reverse's share). s8_8e6, s8_2p5e7,
#                   s8_1e8: S8's G1 operating point (geometric MAC, p5), as recorded in production_2026-10.md;
#                   1e8: jax.grad only, the production leaf capacity, no parity check, preallocated (value_and_grad
#                   does not fit). The gate's --galaxy does not run at 2.5e7 (its eager prepare's walk runs out of
#                   memory) nor at 1e8 (the far list needs 1.51e9 entries, past the 1.43e9 index range); the
#                   rollout rows below are the galaxy rows at those N.
#   r_<N>_T<T>_fwd  (a) the production forward rollout over provider.step: the reference, ms per step.
#   r_..._dfwd      (b) the forward-only rollout over differentiable_step: b - a is the gradient's forward
#                   overhead (the grad lists, the duplicate far chain); bitwise to (a) via --reference (G1).
#   r_..._grad_<m>  (c) value_and_grad of the rollout in mode m: c / a per step, the reverse's cost.
#                   step vs nested: the price of one more forward per step; nested vs offload: the price of
#                   moving the carries and the step residuals over PCIe (offload's trace shows the copies).
#                   T=1 vs T=16: the start force's share, and whether the per-step cost is flat in T.
#   r_1e8_*         whether the 1e8 rows fit, and their time: T=1 step (BFC), T=4 offload inner 2 (BFC),
#                   T=16 offload inner 4 (cuda_async: BFC could not place its 25.9 GiB block beside the
#                   outputs; that row's time includes the async allocator's own cost).
#
#SBATCH --job-name=jaccpot-grad-rollout-timing
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), jaccpot (this checkout), yggdrax main
# source ~/envs/jaccpot-jax0112/bin/activate
JACCPOT=${JACCPOT:-$PWD}                                  # this jaccpot checkout
OUT=${OUT:-${WORKSPACE:-$HOME}/grad-rollout-timing-$(date +%Y%m%d)}
ROWS=${ROWS:-.}                                           # a regex over the row tags
PY=${PY:-python}
# SMOKE_N: every row at this N (bench/slurm/h0/submit_all.sh SMOKE=1, a CPU test of the commands)
nn() { echo "${SMOKE_N:-$1}"; }

# one card of the exclusive node; the others stay idle
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs" "$OUT/traces"
cd "$JACCPOT"
git rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true

T=bench/grad_rollout_timing.py
G=bench/grad_production_gate.py
BFC="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
ASYNC="XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async XLA_PYTHON_CLIENT_PREALLOCATE=false"
GATE="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
GATE_1E8="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"

# tag | allocator env | script and arguments (the row's JSON is $OUT/rows/<tag>.json)
ROW_LIST=(
  "s8_8e6_galaxy|$GATE|$G $(nn 8000000) --galaxy --softening 1e-3 --reps 7 --trace $OUT/traces/s8_8e6_galaxy"
  "s8_8e6|$GATE|$G $(nn 8000000) --reps 7"
  "s8_2p5e7|$GATE|$G $(nn 25000000) --reps 7"
  "s8_1e8|$GATE_1E8|$G $(nn 100000000) --leaf-capacity auto --grad-only --no-forward --reps 5"
)
for n in 8e6 2p5e7; do
  case $n in 8e6) N=$(nn 8000000); R=7 ;; 2p5e7) N=$(nn 25000000); R=5 ;; esac
  for steps in 1 16; do
    r=r_${n}_T${steps}
    ROW_LIST+=(
      "${r}_fwd|$BFC|$T --n $N --steps $steps --only forward --reps $R"
      "${r}_dfwd|$BFC|$T --n $N --steps $steps --only diffforward --reps $R --reference $OUT/rows/${r}_fwd.json"
    )
    for mode in step nested offload; do
      extra=""
      [ "$mode" = offload ] && extra="--inner 4"            # nested: round(sqrt(T)) = 4 at T = 16
      if [ "$n" = 8e6 ] && { [ "$steps" = 1 ] || [ "$mode" = offload ]; }; then
        extra="$extra --trace $OUT/traces/${r}_grad_${mode}"
      fi
      ROW_LIST+=("${r}_grad_${mode}|$BFC|$T --n $N --steps $steps --only grad --mode $mode --reps $R $extra")
    done
  done
done
N=$(nn 100000000)
ROW_LIST+=(
  "r_1e8_T1_fwd|$BFC|$T --n $N --steps 1 --only forward --reps 3"
  "r_1e8_T1_dfwd|$BFC|$T --n $N --steps 1 --only diffforward --reps 3 --reference $OUT/rows/r_1e8_T1_fwd.json"
  "r_1e8_T1_grad_step|$BFC|$T --n $N --steps 1 --only grad --mode step --reps 3"
  "r_1e8_T4_fwd|$BFC|$T --n $N --steps 4 --only forward --reps 3"
  "r_1e8_T4_dfwd|$BFC|$T --n $N --steps 4 --only diffforward --reps 3 --reference $OUT/rows/r_1e8_T4_fwd.json"
  "r_1e8_T4_grad_offload|$BFC|$T --n $N --steps 4 --only grad --mode offload --inner 2 --reps 3"
  "r_1e8_T16_fwd|$BFC|$T --n $N --steps 16 --only forward --reps 3"
  "r_1e8_T16_dfwd|$BFC|$T --n $N --steps 16 --only diffforward --reps 3 --reference $OUT/rows/r_1e8_T16_fwd.json"
  "r_1e8_T16_grad_offload|$ASYNC|$T --n $N --steps 16 --only grad --mode offload --inner 4 --reps 3"
)

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag envs cmd <<< "$row"
  [[ $tag =~ $ROWS ]] || continue
  json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; continue; fi
  case $cmd in
    $G*) cmd="$cmd --json $json" ;;
    *) cmd="$cmd --out $json" ;;
  esac
  echo "=== $tag  [$envs]  $(date -Is)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  if env $envs $PY $cmd > "$OUT/logs/$tag.log" 2>&1; then status=ok; else status="FAILED rc=$?"; fi
  echo "    $status in $(( $(date +%s) - start )) s (log $OUT/logs/$tag.log)"
done

$PY $T --summarize "$OUT/rows" | tee "$OUT/summary.txt" || true

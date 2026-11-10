#!/bin/bash
# Forward mode round 2's timings (bench/forward_mode_timing.py) and the demonstrator
# (bench/forward_mode_demonstrator.py), on ONE card of a dedicated node.
# Written for a HoreKa accelerated node (4x A100-40GB); adapt the partition, the environment and the paths
# below. One process at a time on the allocation's first card (the others idle: no contention, no shared
# host link). Resumable: a row whose JSON exists is skipped, so a resubmission continues where the last one
# stopped (a row that failed left no JSON and runs again).
#
#   sbatch bench/slurm/forward_mode_timing.sbatch                    # the whole matrix (~8-10 h, rough)
#   ROWS='^t_2p5e7_' sbatch bench/slurm/forward_mode_timing.sbatch   # the rows whose tag matches a regex
#   python bench/forward_mode_timing.py --summarize $OUT/rows        # the table, ratios over forward
#
# What the rows answer (F8's operating point: preset="galaxy", clipped Plummer, float32, dehnen_error,
# dt 1e-3, T = 16 scanned steps; the Sgr-like 8-parameter map of a 1 % satellite subset):
#   t_<N>_forward        the production forward rollout over provider.step: the reference, ms per step.
#   t_<N>_jvp            one direction (K = 1): the jvp's cost over the forward.
#   t_<N>_jacfwd_k<K>    K = 1, 2, 4, 8 columns in ONE pass: the cost per column as K grows (the tangent
#                        kernels share the primal across K) and compile_s(K) -- round 1: 2e6 K = 8 ~75 min.
#   t_<N>_jacfwd_k8_dpp<k>  the same 8 columns, k = 1 / 2 per pass (directions_per_pass): the price of
#                        recomputing the primal per chunk against the memory it saves (peak B/p in the row).
#   t_<N>_jacfwd_k8_group<g>  the same 8 columns in ONE pass, the force's tangent applied g = 1 / 2 / 4 directions
#                        at a time (JACCPOT_FORWARD_TANGENT_GROUP; primal once): its price against the single pass
#                        (the tangent kernels at K = g, K / g launches) and its memory (peak B/p). jvp_group1: the
#                        knob's cost at K = 1 (the linearized force keeps its residuals through the tangent map).
#   t_<N>_vgrad          value_and_grad (checkpoint="step", S9's reverse): where forward stops paying.
#   tpp_2p5e7            each seam's tangent op alone at 2.5e7 shapes, K = 8, tangents-per-program 1/2/4/8
#                        (JACCPOT_TANGENT_TPP_<SEAM>; skipped if the installed jaccpot has no such knob).
#   demo_*               the demonstrator: the 8 x 7 Jacobian of a satellite (6D offset + mass) through
#                        T = 200 steps at 2.5e7 (one pass) and at 1e8 (one direction per pass), and its
#                        check against value_and_grad at T = 4 (F2 identity).
# At 1e8 only what fits an A100-40 runs: forward, jvp (K = 1 peaks at 384 B/p = 35.8 GiB) and jacfwd K = 8 with
# directions_per_pass = 1.
#
# Compile times: leave JACCPOT_COMPILATION_CACHE_DIR UNSET for this job (a warm cache makes compile_s a cache
# load); set CACHE=<dir> only to re-run rows whose compile you do not need.
#
# On HoreKa this runs as job "forward" of the bundle (bench/slurm/h0/README.md: one checkout, one setup, every job).
#
#SBATCH --job-name=jaccpot-forward-mode-timing
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# --- adapt: a python with jax 0.11.2 (CUDA 12), jaccpot (this checkout), yggdrax main
# source ~/envs/jaccpot-jax0112/bin/activate
JACCPOT=${JACCPOT:-$PWD}                                  # this jaccpot checkout
OUT=${OUT:-${WORKSPACE:-$HOME}/forward-mode-timing-$(date +%Y%m%d)}
ROWS=${ROWS:-.}                                           # a regex over the row tags
PY=${PY:-python}
REPS=${REPS:-5}
# SMOKE_N: every row at this N (bench/slurm/h0/submit_all.sh SMOKE=1, a CPU test of the commands); FWD_EXTRA:
# arguments for every row (the smoke's --cpu-interpret)
nn() { echo "${SMOKE_N:-$1}"; }
if [ -n "${CACHE:-}" ]; then export JACCPOT_COMPILATION_CACHE_DIR=$CACHE; else unset JACCPOT_COMPILATION_CACHE_DIR; fi

# one card of the exclusive node; the others stay idle
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
echo "node $(hostname) card ${CUDA_VISIBLE_DEVICES}"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" || true
mkdir -p "$OUT/rows" "$OUT/logs"
cd "$JACCPOT"
git rev-parse HEAD > "$OUT/jaccpot_commit.txt" 2>/dev/null || true

T="bench/forward_mode_timing.py ${FWD_EXTRA:-}"
D="bench/forward_mode_demonstrator.py ${FWD_EXTRA:-}"
BFC="XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"

# tag | allocator env | script and arguments (the row's JSON is $OUT/rows/<tag>.json)
ROW_LIST=()
N=$(nn 25000000)
ROW_LIST+=(
  "t_2p5e7_forward|$BFC|$T --n $N --steps 16 --program forward --reps $REPS"
  "t_2p5e7_jvp|$BFC|$T --n $N --steps 16 --program jvp --reps $REPS"
)
for k in 1 2 4 8; do
  ROW_LIST+=("t_2p5e7_jacfwd_k${k}|$BFC|$T --n $N --steps 16 --program jacfwd --k $k --reps $REPS")
done
for dpp in 1 2; do
  ROW_LIST+=("t_2p5e7_jacfwd_k8_dpp${dpp}|$BFC|$T --n $N --steps 16 --program jacfwd --k 8 --directions-per-pass $dpp --reps $REPS")
done
# the force's tangent in groups of g directions (JACCPOT_FORWARD_TANGENT_GROUP, track M): no primal recompute
for g in 1 2 4; do
  ROW_LIST+=("t_2p5e7_jacfwd_k8_group${g}|$BFC JACCPOT_FORWARD_TANGENT_GROUP=$g|$T --n $N --steps 16 --program jacfwd --k 8 --reps $REPS")
done
ROW_LIST+=("t_2p5e7_jvp_group1|$BFC JACCPOT_FORWARD_TANGENT_GROUP=1|$T --n $N --steps 16 --program jvp --reps $REPS")
ROW_LIST+=(
  "t_2p5e7_vgrad|$BFC|$T --n $N --steps 16 --program vgrad --reps $REPS"
  "tpp_2p5e7|$BFC|$T --n $N --tpp-sweep --tpp-k 8 --reps $REPS"
)
N=$(nn 100000000)
ROW_LIST+=(
  "t_1e8_forward|$BFC|$T --n $N --steps 16 --program forward --reps 3"
  "t_1e8_jvp|$BFC|$T --n $N --steps 16 --program jvp --reps 3"
  "t_1e8_jacfwd_k8_dpp1|$BFC|$T --n $N --steps 16 --program jacfwd --k 8 --directions-per-pass 1 --reps 2"
  "demo_2p5e7_check_T4|$BFC|$D --n $(nn 25000000) --check --steps 4"
  "demo_2p5e7_T200|$BFC|$D --n $(nn 25000000) --steps ${DEMO_STEPS:-200}"
  "demo_1e8_T200_dpp1|$BFC|$D --n $(nn 100000000) --steps ${DEMO_STEPS:-200} --directions-per-pass 1"
)

for row in "${ROW_LIST[@]}"; do
  IFS='|' read -r tag envs cmd <<< "$row"
  [[ $tag =~ $ROWS ]] || continue
  json=$OUT/rows/$tag.json
  if [ -s "$json" ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $tag  [$envs]  $(date -Is)"
  start=$(date +%s)
  # shellcheck disable=SC2086
  if env $envs $PY $cmd --out "$json" > "$OUT/logs/$tag.log" 2>&1; then status=ok; else status="FAILED rc=$?"; fi
  echo "    $status in $(( $(date +%s) - start )) s (log $OUT/logs/$tag.log)"
done

$PY $T --summarize "$OUT/rows" | tee "$OUT/summary.txt" || true

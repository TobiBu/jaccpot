#!/bin/bash
#SBATCH --job-name=m2g-node2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=m2g-node2-%j.out
# M2G across TWO nodes: 8 processes x 1 GPU (the first multi-node run; functional). No single-process
# arm exists at 8 GPUs over 2 nodes: the checks are the A-vs-A of two runs, the retries, the energy and
# momentum at every edge, and the collectives probe at 8 (exact, and the float psum vs the mesh-order
# fold). The last step is the #398 barrier probe ACROSS nodes: the NCCL ragged fallback with
# use_barrier_with_nccl at its jax-0.11 default (on), under a 5-minute timeout (M4's re-check).
export M2G_JOB=node2
# shellcheck source=_prologue.sh
source "${M2G_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m2g/_prologue.sh"
N=${M2G_N_PER_GPU:-2e6}
NT=$((2 * GPN))
MP=(--ntasks="$NT" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
arm col_mp "${MP[@]}" -- collectives --arm mp
for mac in dehnen_error dehnen; do
  for rep in a b; do
    arm "${mac}_mp_$rep" "${MP[@]}" -- run --arm mp --mac "$mac" --n-per-gpu "$N"
  done
  cmp_arms "${mac}_mp_a" "${mac}_mp_b"
done
# the barrier probe: the flag set the production one minus barrier=false
BARRIER_OFF="--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false"
if [ ! -s "$OUT/col_barrier_on/collectives.json" ]; then
  next_port
  echo "--- col_barrier_on (timeout 300 s) $(date -Is)"
  XLA_FLAGS="${XLA_FLAGS/$BARRIER_OFF/} --xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false --xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true --xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=true" \
    timeout 300 srun "${MP[@]}" $GATE collectives --arm mp --out "$OUT/col_barrier_on" > "$OUT/col_barrier_on.log" 2>&1 \
    && echo "    finished" || echo "    rc $? (124 = the timeout: a hang)"
fi
m2g_done

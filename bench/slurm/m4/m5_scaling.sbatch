#!/bin/bash
# M5 TEMPLATE (optional, not part of the default bundle): weak and strong scaling of the production multi-GPU
# path on 1 / 2 / 4 / 8 nodes of 4 A100s, one process per GPU, with the M4 switches. Submit once per node count:
#   sbatch -A $H0_ACCOUNT -p $H0_PARTITION --nodes=N --ntasks-per-node=4 --gres=gpu:4 --exclusive \
#          --time=HH:MM:SS bench/slurm/m4/m5_scaling.sbatch          (MODE=weak|strong, ARMS=...)
# weak:   N_PER_GPU=3.125e7 (1e9 on 8 nodes = 32 GPUs), the per-step wall at 4 GPUs is the yardstick:
#         efficiency(n) = wall(4 GPUs) / wall(n GPUs); U4's floor is 70 % at 8 nodes, "the higher the better"
# strong: N_TOTAL=1e9 on 4 and 8 nodes (2.5e8 on 1-2 nodes: the per-card ceiling ~7e7 on 40 GB, see scaling_a's
#         memory rows), the speed-up per node count
# Budget (estimates, untested at this scale): weak 1/2/4/8 nodes x 2 arms x (setup ~15 min + 3 x 20 steps)
# ~ 1 + 2 + 4 + 8 = 15 node-hours x ~1 h = ~15-20 node-hours; strong 4 + 8 nodes ~ 12 node-hours.
#SBATCH --job-name=m5-scaling
#SBATCH --exclusive
#SBATCH --output=m5-scaling-%j.out
export M4_JOB=${M4_JOB:-m5_${MODE:-weak}_${SLURM_JOB_NUM_NODES:-1}n}
# shellcheck source=_prologue.sh
source "${M4_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m4/_prologue.sh"
NODES=${SLURM_JOB_NUM_NODES:-1}
NT=$((NODES * GPN))
MP=(--ntasks="$NT" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
if [ "${MODE:-weak}" = weak ]; then
  NPG=${N_PER_GPU:-3.125e7}
else
  NPG=$(python3 -c "print(float('${N_TOTAL:-1e9}') / $NT)")
fi
for spec in ${ARMS:-all_gather:count let:work}; do
  ex=${spec%%:*}; cuts=${spec##*:}
  arm "${MODE:-weak}_${NT}gpu_${ex}_${cuts}" "${MP[@]}" -- --arm mp --mac dehnen_error --exchange "$ex" \
    --cuts "$cuts" --n-per-gpu "$(nn "$NPG")" --segments "${M4_SEGMENTS:-3}"
done
m4_done

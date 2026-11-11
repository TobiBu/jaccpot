#!/bin/bash
#SBATCH --job-name=m2g-node1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=03:30:00
#SBATCH --output=m2g-node1-%j.out
# M2G on ONE node: 4 processes x 1 GPU (jax.distributed, initialize() from the SLURM step) against
# 1 process x 4 GPUs, on the same node's cards. Bitwise + timing (the dedicated node: timings count).
#   sbatch -A $H0_ACCOUNT -p accelerated bench/slurm/m2g/node1.sbatch     (bench/slurm/m2g/submit.sh does it)
# Arms (each twice for the A-vs-A control; a step whose outputs exist is skipped on a resubmit):
#   col_sp / col_mp     the collectives probe (exact results, float psum vs the mesh-order fold)
#   <mac>_{sp,mp}_{a,b} 2e6 particles per GPU, 3 segments x 4 steps, migration, energy at every edge,
#                       2 repeats in the process (repeat 0 compiles, repeat 1 is the warm timing)
#   <mac>_mpbind_a      the mp arm with SLURM's per-task GPU binding (--gpus-per-task=1 --gpu-bind=single:1:
#                       each task sees ONE card; initialize() passes local_device_ids=[0])
#   det_*               dehnen_error with --xla_gpu_deterministic_ops=true (the bitwise gate if the
#                       production flags' A-vs-A is not clean)
export M2G_JOB=${M2G_JOB:-node1}   # the HoreKa bundle's job sets m2g_node1
# shellcheck source=_prologue.sh
source "${M2G_JACCPOT:-$SLURM_SUBMIT_DIR}/bench/slurm/m2g/_prologue.sh"
N=${M2G_N_PER_GPU:-2e6}
SP=(--ntasks=1 --nodes=1 --cpus-per-task="$CPN" --gres=gpu:"$GPN")
MP=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
MPBIND=(--ntasks="$GPN" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gpus-per-task=1 --gpu-bind=single:1)

arm col_sp "${SP[@]}" -- collectives --arm sp
arm col_mp "${MP[@]}" -- collectives --arm mp
for mac in dehnen_error dehnen; do
  for rep in a b; do
    arm "${mac}_sp_$rep" "${SP[@]}" -- run --arm sp --mac "$mac" --n-per-gpu "$N"
    arm "${mac}_mp_$rep" "${MP[@]}" -- run --arm mp --mac "$mac" --n-per-gpu "$N"
  done
  arm "${mac}_mpbind_a" "${MPBIND[@]}" -- run --arm mp --mac "$mac" --n-per-gpu "$N"
  cmp_arms "${mac}_sp_a" "${mac}_sp_b"; cmp_arms "${mac}_mp_a" "${mac}_mp_b"
  cmp_arms "${mac}_sp_a" "${mac}_mp_a"; cmp_arms "${mac}_sp_b" "${mac}_mp_b"
  cmp_arms "${mac}_sp_a" "${mac}_mpbind_a"
done
for rep in a b; do
  arm "det_sp_$rep" "${SP[@]}" -- run --arm sp --mac dehnen_error --deterministic --n-per-gpu "$N" --repeats 1
  arm "det_mp_$rep" "${MP[@]}" -- run --arm mp --mac dehnen_error --deterministic --n-per-gpu "$N" --repeats 1
done
cmp_arms det_sp_a det_sp_b; cmp_arms det_mp_a det_mp_b; cmp_arms det_sp_a det_mp_a; cmp_arms det_sp_b det_mp_b
m2g_done

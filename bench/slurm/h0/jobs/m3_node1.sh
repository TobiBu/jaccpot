#!/bin/bash
# HoreKa job "m3_node1": M3's sharded checkpoints at 1e8 on ONE node (bench/m3_checkpoint_io.py; README
# "m3"). 4 processes x 1 GPU write (twice: the second replaces the first), then the restores: the same 4
# processes, 1 process x 4 GPUs (same count: bitwise), and elastic onto 2 processes and 1 GPU; then the
# walltime-chaining mechanics (a run stopped by its deadline, continued from its last checkpoint).
# Per rank: write/restore seconds, shard bytes, bytes on disk, peak host RSS, device peak.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight m3
N=${M3_N:-1e8}
STEPS=2
[ "${SMOKE:-}" = 1 ] && N=4096
GPN=${H0_GPUS_PER_NODE:-4}
CPN=${SLURM_CPUS_ON_NODE:-76}
# shellcheck source=../../m3/steps.sh
source "$H0_JACCPOT/bench/slurm/m3/steps.sh"
W=$OUT/ckpt
MP=(--nodes=1 --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN")
m3_step write_mp "${MP[@]}" --ntasks="$GPN" -- write --n "$N" --steps "$STEPS" --ckpt "$W/ck"
m3_step restore_mp "${MP[@]}" --ntasks="$GPN" -- restore --ckpt "$W/ck" --steps "$STEPS"
m3_step restore_sp --ntasks=1 --nodes=1 --cpus-per-task="$CPN" --gres=gpu:"$GPN" -- restore --arm sp --ckpt "$W/ck" --steps "$STEPS"
HALF=$((GPN / 2 > 0 ? GPN / 2 : 1))
m3_step restore_half --ntasks="$HALF" --nodes=1 --ntasks-per-node="$HALF" --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN" -- restore --ckpt "$W/ck" --steps "$STEPS"
m3_step restore_one --ntasks=1 --nodes=1 --cpus-per-task=$((CPN / GPN)) --gres=gpu:1 -- restore --ckpt "$W/ck" --steps "$STEPS"
# chaining: the first job's deadline falls inside its first segments, the second continues to the target
now=$(date +%s)
m3_step chain_a "${MP[@]}" --ntasks="$GPN" -- chain --n "$N" --steps "$STEPS" --pattern "$W/chain_{step:08d}" \
  --target-steps $((4 * STEPS)) --every 1 --deadline $((now + 1)) --margin 0
m3_step chain_b "${MP[@]}" --ntasks="$GPN" -- chain --n "$N" --steps "$STEPS" --pattern "$W/chain_{step:08d}" \
  --target-steps $((4 * STEPS)) --every 1
m3_summary
h0_done

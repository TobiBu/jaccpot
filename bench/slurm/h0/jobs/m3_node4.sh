#!/bin/bash
# HoreKa job "m3_node4": M3's sharded checkpoints at 1e9 on FOUR nodes, 16 processes x 1 GPU (6.25e7 per
# 40 GB A100; README "m3": 8 GPUs at 1.25e8 each is past the mesh's per-card ceiling), written twice, then
# restored on the same 16 and elastically on 12 (three nodes) and 8 (two nodes: expected to run out of
# device memory; recorded either way). Per rank: write/restore seconds, shard bytes, disk bytes, RSS.
# shellcheck source=_prologue.sh
source "$H0_JACCPOT/bench/slurm/h0/jobs/_prologue.sh"
h0_need_preflight m3
N=${M3_N:-1e9}
STEPS=2
[ "${SMOKE:-}" = 1 ] && N=8192
GPN=${H0_GPUS_PER_NODE:-4}
NODES=${SLURM_NNODES:-4}
CPN=${SLURM_CPUS_ON_NODE:-76}
# shellcheck source=../../m3/steps.sh
source "$H0_JACCPOT/bench/slurm/m3/steps.sh"
W=$OUT/ckpt
ALL=$((NODES * GPN))
m3_step write_mp --nodes="$NODES" --ntasks="$ALL" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) \
  --gres=gpu:"$GPN" -- write --n "$N" --steps "$STEPS" --ckpt "$W/ck"
m3_step restore_mp --nodes="$NODES" --ntasks="$ALL" --ntasks-per-node="$GPN" --cpus-per-task=$((CPN / GPN)) \
  --gres=gpu:"$GPN" -- restore --ckpt "$W/ck" --steps "$STEPS"
for k in 3 2; do
  nodes=$((NODES * k / 4 > 0 ? NODES * k / 4 : 1))
  m3_step "restore_${nodes}nodes" --nodes="$nodes" --ntasks=$((nodes * GPN)) --ntasks-per-node="$GPN" \
    --cpus-per-task=$((CPN / GPN)) --gres=gpu:"$GPN" -- restore --ckpt "$W/ck" --steps "$STEPS"
done
m3_summary
h0_done

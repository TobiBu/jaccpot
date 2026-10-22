#!/bin/bash
# usage: launch.sh NAME [run args...]  -- books the least-used card, runs one C1 run, logs to logs/NAME.log
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/12d2ae37-2f3d-4f0a-8604-a5693bc1a4c1/scratchpad
name=$1; shift
cd /export/home/tbuck/jaccpot-c1
eval $(/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd -q -l -n 1 -e -t 120)
export S1_JACCPOT=/export/home/tbuck/jaccpot-c1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_experimental_enable_fusion_autotuner=false"
{
echo "[launch] $name card $CUDA_VISIBLE_DEVICES $(date -Is)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
$SP/s58py bench/c1_criterion_calibration.py run $SP/c1/runs/$name.npz "$@"
echo "[launch] $name rc=$? $(date -Is)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
} > $SP/c1/logs/$name.log 2>&1

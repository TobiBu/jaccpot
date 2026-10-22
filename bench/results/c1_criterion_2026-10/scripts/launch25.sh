#!/bin/bash
# usage: launch25.sh NAME DT [EXCLUDE...] -- one 25M production run on a least-used card with >= 10 GiB free
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/12d2ae37-2f3d-4f0a-8604-a5693bc1a4c1/scratchpad
name=$1; dt=$2; shift 2
D=/export/scratch/tbuck/c1_criterion_2026-10/25m/$name
mkdir -p $D
cd /export/home/tbuck/jaccpot-c1-main
X=""; [ $# -gt 0 ] && X="-x $*"
eval $(/export/home/tbuck/micromamba/envs/odisseo/bin/autocvd -q -l -n 1 -e -t 120 $X)
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES)
{
echo "[launch25] $name dt $dt card $CUDA_VISIBLE_DEVICES free ${free} MiB $(date -Is)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
if [ "$free" -lt 10240 ]; then echo "[launch25] not enough free memory"; exit 3; fi
S1_JACCPOT=/export/home/tbuck/jaccpot-c1-main XLA_PYTHON_CLIENT_PREALLOCATE=false $SP/s58py $SP/c1_25m/c1_25m_run.py --dt $dt --scratch $D --out $SP/c1_25m/$name.json --jsonl $SP/c1_25m/$name.jsonl
echo "[launch25] $name rc=$? $(date -Is)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
} > $SP/c1_25m/$name.log 2>&1

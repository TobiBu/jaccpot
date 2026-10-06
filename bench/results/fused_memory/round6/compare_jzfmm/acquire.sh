#!/bin/bash
# Wait for a free card (autocvd, org rule), then hold it with the guard.
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/4e79cfff-6b30-497e-8630-4d6d2e67c890/scratchpad
C=$(/export/home/tbuck/jaccpot/.venv/bin/autocvd -n 1 -o -q -i 20)
echo "$C" > $SP/r6/card
echo hold > $SP/r6/gctl
CUDA_VISIBLE_DEVICES=$C setsid nohup /export/home/tbuck/jaccpot/.venv/bin/python $SP/r6/guard.py $SP/r6/gctl $SP/r6/gack 32 > $SP/r6/guard.log 2>&1 &
echo "guard pid $! on card $C at $(date +%H:%M:%S)"
until [ -s $SP/r6/gack ]; do sleep 0.5; done; cat $SP/r6/gack; echo

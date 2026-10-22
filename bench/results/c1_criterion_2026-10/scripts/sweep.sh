#!/bin/bash
# usage: sweep.sh LIST -- runs every line "NAME ARGS..." of LIST through launch.sh, in order
SP=/tmp/claude-2701/-export-home-tbuck-Odisseo/12d2ae37-2f3d-4f0a-8604-a5693bc1a4c1/scratchpad
while read -r name args; do
  [ -z "$name" ] && continue
  if [ -f "$SP/c1/runs/$name.json" ]; then echo "[sweep] skip $name (done)"; continue; fi
  echo "[sweep] start $name $(date -Is)"
  $SP/c1/launch.sh $name $args
  echo "[sweep] done $name $(tail -2 $SP/c1/logs/$name.log | head -1)"
done < "$1"
echo "[sweep] ALL DONE $(date -Is)"

#!/bin/bash
#SBATCH --job-name=a4-ics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=04:00:00
#SBATCH --output=a4-ics-%j.out
# A4 galaxy ICs at the ladders' largest N (agama; Odisseo tools/a4_galaxy_ics.py): every rung takes a
# per-component prefix, so the rungs are nested subsamples of one realisation. Skips an existing file.
#   galaxy.npz  disc + bulge + halo, 2e8   (disc_n / disc_dt / disc_mac / disc_precision)
#   bar.npz     the bar-unstable disc, 1e8 (bar_n)
#   merger.npz  two galaxies, 1e8 each    (merger_n / merger_dt / merger_mac)
# The sphere ladders sample their ICs in the run itself (numpy, seeded).
set -euo pipefail
A4_JACCPOT=${A4_JACCPOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
# shellcheck source=../h0/site.env
source "$A4_JACCPOT/bench/slurm/h0/site.env"   # H0_ROOT: A4_ODISSEO's default (setup_env.sh's clone)
# shellcheck source=site.env
source "$A4_JACCPOT/bench/slurm/a4/site.env"
: "${A4_IC_SCALE:=1}"   # the local smoke shrinks every IC (A4_IC_SCALE=0.0001)
mkdir -p "$A4_IC"
gen() {
  local kind=$1 n=$2 path=$A4_IC/$1.npz
  n=$(awk -v n="$n" -v s="$A4_IC_SCALE" 'BEGIN { printf "%d", n * s }')
  if [ -s "$path" ]; then echo "skip $path (exists)"; return 0; fi
  echo "--- $kind N=$n -> $path ($(date -Is))"
  "$A4_AGAMA_PY" "$A4_ODISSEO/tools/a4_galaxy_ics.py" "$kind" --output "$path.tmp.npz" \
    --n-particles "$n" --seed 41
  mv "$path.tmp.npz" "$path"
}
gen galaxy 200000000
gen bar 100000000
gen merger 100000000
sha256sum "$A4_IC"/*.npz > "$A4_IC/SHA256SUMS"
echo "=== done $(date -Is)"

#!/bin/bash
# HoreKa bundle: copy the data the jobs need from the DEVELOPMENT MACHINE to HoreKa (run it there, not on
# HoreKa). One staging step for every job; setup_env.sh checks the files against bench/slurm/h0/data.sha256.
#
#   bash bench/slurm/h0/stage_data.sh USER@horeka.scc.kit.edu:/path/to/H0_ROOT            # everything, ~12.4 GB
#   bash bench/slurm/h0/stage_data.sh --no-big USER@horeka.scc.kit.edu:/path/to/H0_ROOT   # the small part, 575 MB
#   bash bench/slurm/h0/stage_data.sh [--no-big] --tar h0-data.tar    # a tarball instead (unpack it in H0_ROOT)
#
# Small (always; jobs pins, scaling_*, s1b, grad, mgrad, a3 and the preflight read it):
#   data/disk_bulge_25m.npz        the 25M disc + bulge IC (Odisseo/agama, not in any repo), 573 MB
#   data/reference/*.npy           the fp64 reference cache of the Odisseo bench harness (direct sums)
# Big (data/big/, 11.9 GB; --no-big skips them and the jobs that read them skip with a message):
#   big/sgr_light_1e8_noclip.npz   the 1e8 light-Sgr merger, UNCLIPPED (A2), 2.8 GB: jobs escaper, mesc_node1/2
#   big/bar_restart_3.npz          the 1e8 bar-unstable disc's restart at step 30 (the #487 stall), 4.4 GB: job a1
#   big/merger_restart_136.npz     the 1.06e8 major merger's restart at t 1.36, 4.7 GB: job a1
# Jobs m1_*, m2g_*, auto, forward and a4 draw (or generate) their own ICs.
set -euo pipefail
H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IC=${H0_SRC_IC:-/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz}
REF=${H0_SRC_REF:-/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/artifacts/reference}
SGR=${H0_SRC_SGR:-/export/scratch/tbuck/mn-scratch/a2/sgr/sgr_light_1e8_noclip.npz}
BAR=${H0_SRC_BAR_RESTART:-/export/scratch/tbuck/bar-2026-10-31/stall_a/restart_3.npz}
MERGER=${H0_SRC_MERGER_RESTART:-/export/scratch/tbuck/merger-2026-10-31/s1_main/restart_final_136.npz}
BIG=1
if [ "${1:-}" = --no-big ]; then BIG=0; shift; fi
[ $# -ge 1 ] || { sed -n '2,19p' "$0"; exit 2; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/data/reference"
ln -s "$IC" "$stage/data/disk_bulge_25m.npz"
for f in "$REF"/*.npy; do ln -s "$f" "$stage/data/reference/"; done
sums=$stage/data.sha256
if [ "$BIG" = 1 ]; then
  mkdir -p "$stage/data/big"
  ln -s "$SGR" "$stage/data/big/sgr_light_1e8_noclip.npz"
  ln -s "$BAR" "$stage/data/big/bar_restart_3.npz"
  ln -s "$MERGER" "$stage/data/big/merger_restart_136.npz"
  cp "$H0_DIR/data.sha256" "$sums"
else
  grep -v ' big/' "$H0_DIR/data.sha256" > "$sums"
fi
echo "stage_data: checking $(wc -l < "$sums") files against data.sha256 (the big ones take a minute)"
(cd "$stage/data" && sha256sum --quiet -c "$sums") || {
  echo "stage_data: the local files do not match data.sha256" >&2
  exit 1
}
if [ "$1" = --tar ]; then
  tar -chf "${2:?a tar file name}" -C "$stage" data
  echo "wrote $2: copy it to HoreKa and run 'tar -xf $(basename "$2") -C \$H0_ROOT'"
else
  rsync -avL --progress "$stage/data/" "$1/data/"
  echo "staged into $1/data; now run bench/slurm/h0/setup_env.sh on HoreKa (it verifies the hashes)"
fi

#!/bin/bash
# H0 bundle: copy the data the jobs need from the DEVELOPMENT MACHINE to HoreKa (run it there, not on
# HoreKa). 575 MB: the 25M disc + bulge IC (Odisseo/agama, not in any repo) and the fp64 reference cache
# of the Odisseo bench harness (direct sums the L3 pin and the gates read instead of recomputing).
#
#   bash bench/slurm/h0/stage_data.sh USER@horeka.scc.kit.edu:/path/to/H0_ROOT
#   bash bench/slurm/h0/stage_data.sh --tar h0-data.tar      # a tarball instead (unpack it in H0_ROOT)
#
# setup_env.sh checks the files against bench/slurm/h0/data.sha256.
set -euo pipefail
H0_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IC=${H0_SRC_IC:-/export/scratch/tbuck/odisseo_ic/disk_bulge_25m.npz}
REF=${H0_SRC_REF:-/export/home/tbuck/Odisseo-bench-multigpu/benchmark_multigpu/artifacts/reference}
[ $# -ge 1 ] || { sed -n '2,10p' "$0"; exit 2; }

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/data/reference"
ln -s "$IC" "$stage/data/disk_bulge_25m.npz"
for f in "$REF"/*.npy; do ln -s "$f" "$stage/data/reference/"; done
(cd "$stage/data" && sha256sum --quiet -c "$H0_DIR/data.sha256") || {
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

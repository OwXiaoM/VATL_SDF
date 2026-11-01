#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=pert_manifest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Circle-of-Willis perturbation manifest (CPU; reads data, writes only the manifest JSON).
# Usage: sbatch slurm/perturbation_manifest.sh CHECKPOINT GRAPH_ATLAS_DIR SETS_JSON REFIT_DIR OUT_JSON [options]
set -euo pipefail
CKPT=${1:?checkpoint}; GRAPH=${2:?graph atlas}; SETS=${3:?sets}; REFIT=${4:?refit dir}; OUT=${5:?manifest}; shift 5
source /users/zhangx/VATL_SDF/slurm/environment.sh
echo "git $(git rev-parse --short HEAD)"
python -m analysis.deviation.perturbation_manifest --checkpoint "$CKPT" --graph-atlas "$GRAPH" --sets "$SETS" --refit "$REFIT" --output "$OUT" "$@"
echo MANIFEST_OK

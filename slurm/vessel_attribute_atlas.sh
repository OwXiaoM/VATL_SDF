#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=attr_atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
# Connectivity + radius atlases on the population vessel graph (CPU).
# Usage: sbatch slurm/vessel_attribute_atlas.sh CHECKPOINT FITTED_TEST_PT GRAPH_ATLAS_DIR OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; FITTED=${2:?fitted_test.pt}; GRAPH=${3:?graph atlas dir}; OUT=${4:?new output directory}; shift 4
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m vatl_sdf.vessel_attribute_atlas --checkpoint "$CKPT" --fitted "$FITTED" --graph-atlas "$GRAPH" --output "$OUT" --workers 16 "$@"
echo ATTRIBUTE_ATLAS_OK

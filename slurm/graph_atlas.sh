#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=graph_atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# Probabilistic vessel-graph atlas (CPU): centreline maps at 2/3 mm, population skeleton graph, held-out evaluation.
# Usage: sbatch slurm/graph_atlas.sh CHECKPOINT FITTED_TEST_PT OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; FITTED=${2:?fitted_test.pt}; OUT=${3:?new output directory}; shift 3
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m vatl_sdf.graph_atlas --checkpoint "$CKPT" --fitted "$FITTED" --output "$OUT" --workers 16 "$@"
echo GRAPH_ATLAS_OK

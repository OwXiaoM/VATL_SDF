#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=patlas_m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
# Probabilistic atlas (CPU): merge shards, centreline probability, connectivity, held-out calibration/coverage.
# Usage: sbatch slurm/prob_atlas_merge.sh CHECKPOINT SHARD_DIR OUT_DIR [FITTED_TEST_PT]
set -euo pipefail
CKPT=${1:?checkpoint}; SHARDS=${2:?shard directory}; OUT=${3:?new output directory}; FITTED=${4:-}
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m vatl_sdf.probabilistic_atlas merge --checkpoint "$CKPT" --shards "$SHARDS" --output "$OUT" --workers 16 ${FITTED:+--fitted "$FITTED"}
echo MERGE_OK

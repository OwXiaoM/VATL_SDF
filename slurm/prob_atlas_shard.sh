#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=patlas_s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Probabilistic atlas: template-space decodings of one shard of the training subjects (partial sums).
# Usage: sbatch slurm/prob_atlas_shard.sh CHECKPOINT SHARD_DIR SHARD NUM_SHARDS [options]
set -euo pipefail
CKPT=${1:?checkpoint}; OUT=${2:?shard directory}; SHARD=${3:?shard}; N=${4:?number of shards}; shift 4
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=4
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT; shard $SHARD/$N"
python -m vatl_sdf.probabilistic_atlas shard --checkpoint "$CKPT" --output "$OUT" --shard "$SHARD" --num-shards "$N" "$@"
echo SHARD_OK

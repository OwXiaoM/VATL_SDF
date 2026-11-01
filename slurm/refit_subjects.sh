#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=refit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Test-time refitting of one chunk of a deviation subject set (analysis/deviation/refit_subjects.py).
# Usage: sbatch slurm/refit_subjects.sh CHECKPOINT SETS_JSON SET CHUNK OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; SETS=${2:?sets json}; SET=${3:?set}; CHUNK=${4:?chunk}; OUT=${5:?output dir}; shift 5
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT; set $SET chunk $CHUNK"
python -m analysis.deviation.refit_subjects --checkpoint "$CKPT" --sets "$SETS" --set "$SET" --chunk "$CHUNK" --output "$OUT" "$@"
echo REFIT_OK

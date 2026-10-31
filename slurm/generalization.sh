#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=gen_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
# Generalization test by test-time fitting (analysis/generalization/evaluate_generalization.py).
# Usage: sbatch slurm/generalization.sh CHECKPOINT OUT_DIR [options, e.g. --align-epochs 10 --total-epochs 10]
set -euo pipefail
CKPT=${1:?checkpoint}; OUT=${2:?new output directory}; shift 2
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m analysis.generalization.evaluate_generalization --checkpoint "$CKPT" --output "$OUT" "$@"
echo GENERALIZATION_OK

#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=deviation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=12:00:00
# Deviation statistics with LOO null (CPU): vatl_sdf/deviation.py.
# Usage: sbatch slurm/deviation.sh CHECKPOINT SETS_JSON DECODINGS_DIR "REFIT_DIR [REFIT_DIR ...]" OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; SETS=${2:?sets}; DEC=${3:?decodings}; REFIT=${4:?refit dir}; OUT=${5:?new output dir}; shift 5
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m vatl_sdf.deviation --checkpoint "$CKPT" --sets "$SETS" --decodings "$DEC" --refit $REFIT --output "$OUT" --workers 16 "$@"
echo DEVIATION_OK

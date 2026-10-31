#!/bin/bash
#SBATCH -p l40s
#SBATCH --qos=dev
#SBATCH --job-name=align_s0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# Alignment Step 0 (CPU only): vessel-mask centroid/axis spread and large-vessel recurrence.
# Usage: sbatch slurm/alignment_step0.sh OUT_DIR
set -euo pipefail
OUT=${1:?new output directory}
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m analysis.alignment.step0_mask_geometry --output "$OUT" --workers 16
echo STEP0_OK

#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=atlas_cal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# Post-hoc recalibration of the occupancy atlas (CPU): Gaussian smoothing x {none, platt, isotonic}, fitted on the
# refitted training subjects, reported on the test subjects.
# Usage: sbatch slurm/atlas_calibration.sh CHECKPOINT ATLAS_DIR FITTED_TRAIN_REFIT_PT FITTED_TEST_PT OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; ATLAS=${2:?atlas dir}; CAL=${3:?fitted_train_refit.pt}; TEST=${4:?fitted_test.pt}; OUT=${5:?new output directory}; shift 5
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
python -m vatl_sdf.atlas_calibration --checkpoint "$CKPT" --atlas "$ATLAS" --calibration-fitted "$CAL" --test-fitted "$TEST" --output "$OUT" --workers 16 "$@"
echo ATLAS_CALIBRATION_OK

#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=sdf_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Screening pilot: single GPU, first N training subjects (YAML order), fixed step count and seed,
# then fitting diagnostics (native grid, reference-relative connectivity, centreline by radius, 1 mm atlas).
# Usage: sbatch slurm/pilot_sdf.sh CONFIG OUT_DIR SUBJECTS STEPS [--set extra overrides]
set -euo pipefail
CONFIG=${1:?config}; OUT=${2:?new output dir}; SUBJECTS=${3:?subjects}; STEPS=${4:?steps}
shift 4
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
echo "git $(git rev-parse --short HEAD); config $CONFIG; subjects $SUBJECTS; steps $STEPS; extra: $*"
python -m vatl_sdf.train --config "$CONFIG" --set data.max_subjects="$SUBJECTS" training.max_steps="$STEPS" \
  training.atlas_every=0 training.checkpoint_every=10 "output_root=$OUT/runs" data.num_workers=3 "$@"
RUN=$(ls -d "$OUT"/runs/*/ | head -1)
CKPT=$(ls "$RUN"/checkpoint_step_*.pt | sort | tail -1)
python -m analysis.phase0_graph.diagnose_checkpoint --checkpoint "$CKPT" --output "$OUT/diagnostics" \
  --factors 1 --atlas-spacings 1
echo PILOT_OK

#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=post_b00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# Analyses of a finished full run (latest checkpoint): Phase 0c diagnostics (native x1/x2, atlas 1/0.5 mm),
# Phase 3 population maps (1 mm) and Phase 5 centreline density/recurrence. Each step runs even if an
# earlier one fails; the log ends with one STEP_OK / STEP_FAILED line per step.
# Usage: sbatch slurm/post_baseline.sh RUN_ROOT OUT_DIR
set -uo pipefail
RUN_ROOT=${1:?run output root}; OUT=${2:?new output dir}
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$OUT"
RUN=$(ls -d "$RUN_ROOT"/*/ | head -1)
CKPT=$(ls "$RUN"/checkpoint_step_*.pt | sort | tail -1)
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT"
status=()
run() { local name=$1; shift; local t0=$SECONDS
  if "$@"; then status+=("STEP_OK $name $((SECONDS - t0)) s"); else status+=("STEP_FAILED $name $((SECONDS - t0)) s"); fi; }
run 0c python -m analysis.phase0_graph.diagnose_checkpoint --checkpoint "$CKPT" --output "$OUT/0c" --factors 1 2 --atlas-spacings 1 0.5
run population_maps python -m vatl_sdf.population_maps --checkpoint "$CKPT" --output "$OUT/population_maps_1mm"
run centerline_density python -m analysis.phase5.centerline_density --checkpoint "$CKPT" --output "$OUT/centerline_density"
printf '%s\n' "${status[@]}"

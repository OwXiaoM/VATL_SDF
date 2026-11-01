#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=post_abl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Measurements of a finished screening run (final checkpoint): Phase-0c diagnostics on fixed subjects (x1 grid,
# 1 mm atlas), alignment (transform stats + large-vessel recurrence, first 100 subjects), generalization by
# test-time fitting, divergence/MoE inspection. Each step runs even if an earlier one fails.
# Usage: sbatch slurm/post_ablation.sh RUN_ROOT OUT_DIR [generalization options]
# Env: POST_STEPS (default "0c alignment generalization inspect"), DIAG_SUBJECTS (space-separated IDs),
#      INSPECT_ARGS (default "--reference-steps 20000 30000").
set -uo pipefail
RUN_ROOT=${1:?run output root}; OUT=${2:?new output dir}; shift 2
POST_STEPS=${POST_STEPS:-0c alignment generalization inspect}
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$OUT"
RUN=$(ls -d "$RUN_ROOT"/*/ | head -1); RUN=${RUN%/}
CKPT=$(ls "$RUN"/checkpoint_step_*.pt | sort | tail -1)
echo "git $(git rev-parse --short HEAD); checkpoint $CKPT; steps $POST_STEPS"
status=()
run() { local name=$1; shift; local t0=$SECONDS
  if "$@"; then status+=("STEP_OK $name $((SECONDS - t0)) s"); else status+=("STEP_FAILED $name $((SECONDS - t0)) s"); fi; }
want() { [[ " $POST_STEPS " == *" $1 "* ]]; }
want 0c && run 0c python -m analysis.phase0_graph.diagnose_checkpoint --checkpoint "$CKPT" --output "$OUT/diagnostics" \
  --factors 1 --atlas-spacings 1 ${DIAG_SUBJECTS:+--subjects $DIAG_SUBJECTS}
want alignment && run alignment python -m analysis.alignment.evaluate_alignment --checkpoint "$CKPT" --output "$OUT/alignment" --max-subjects 100 --workers 8
want generalization && run generalization python -m analysis.generalization.evaluate_generalization --checkpoint "$CKPT" --output "$OUT/generalization" "$@"
want inspect && run inspect python -m analysis.divergence.inspect_run --run "$RUN" --output "$OUT/inspect" ${INSPECT_ARGS:---reference-steps 20000 30000}
printf '%s\n' "${status[@]}"

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
# then fitting diagnostics (native grid, reference-relative connectivity, centreline by radius, 1 mm atlas)
# and alignment diagnostics (transform statistics, large-vessel recurrence).
# Usage: sbatch [--gres=gpu:l40s:N] slurm/pilot_sdf.sh CONFIG OUT_DIR SUBJECTS STEPS [--set extra overrides]
# More than one allocated GPU -> DDP (torchrun, as train_sdf.sh). DIAG_SUBJECTS (env, space-separated IDs) fixes
# the diagnostic subjects (default: age/sex quantiles of the run's own subjects).
set -euo pipefail
CONFIG=${1:?config}; OUT=${2:?new output dir}; SUBJECTS=${3:?subjects}; STEPS=${4:?steps}
shift 4
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$OUT"
echo "git $(git rev-parse --short HEAD); config $CONFIG; subjects $SUBJECTS; steps $STEPS; extra: $*"
NPROC=${SLURM_GPUS_ON_NODE:-1}
if [ "$NPROC" -gt 1 ]; then
  LAUNCH=(srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" -m vatl_sdf.train_ddp)
else
  LAUNCH=(python -m vatl_sdf.train)
fi
"${LAUNCH[@]}" --config "$CONFIG" --set data.max_subjects="$SUBJECTS" training.max_steps="$STEPS" \
  training.atlas_every=0 training.checkpoint_every=10 "output_root=$OUT/runs" data.num_workers=3 "$@"
RUN=$(ls -d "$OUT"/runs/*/ | head -1)
CKPT=$(ls "$RUN"/checkpoint_step_*.pt | sort | tail -1)
python -m analysis.phase0_graph.diagnose_checkpoint --checkpoint "$CKPT" --output "$OUT/diagnostics" \
  --factors 1 --atlas-spacings 1 ${DIAG_SUBJECTS:+--subjects $DIAG_SUBJECTS}
python -m analysis.alignment.evaluate_alignment --checkpoint "$CKPT" --output "$OUT/alignment" --max-subjects "$SUBJECTS" --workers 8
echo PILOT_OK

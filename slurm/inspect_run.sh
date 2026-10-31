#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=inspect_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
# Divergence inspection of a run directory (checkpoints + losses.jsonl).
# Usage: sbatch slurm/inspect_run.sh RUN_DIR OUT_DIR [options, e.g. --reference-steps 20000 35000]
set -euo pipefail
RUN=${1:?run directory}; OUT=${2:?new output directory}; shift 2
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8
python -m analysis.divergence.inspect_run --run "$RUN" --output "$OUT" "$@"
echo INSPECT_OK

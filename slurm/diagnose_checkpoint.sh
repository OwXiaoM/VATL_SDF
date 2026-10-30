#!/bin/bash
#SBATCH -p l40s
#SBATCH --qos=dev
#SBATCH --job-name=diagnose_ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Phase 0: individual (native, 2x supersampled, centreline recall) and atlas 1/0.5 mm diagnostics.
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
CHECKPOINT=${1:?checkpoint}
OUT=${2:?new output directory}
shift 2
python -m analysis.phase0_graph.diagnose_checkpoint --checkpoint "$CHECKPOINT" --output "$OUT" "$@"

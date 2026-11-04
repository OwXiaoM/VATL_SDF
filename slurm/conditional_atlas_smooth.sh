#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=cond_smooth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Age-averaged mean-latent fields, smoothed occupancy and graph tubes for an existing conditional atlas, then figures.
# Usage: sbatch slurm/conditional_atlas_smooth.sh ATLAS_DIR MANIFEST_JSON [options]
set -euo pipefail
ATLAS=${1:?atlas directory}; MANIFEST=${2:?manifest}; shift 2
source /users/zhangx/VATL_SDF/slurm/environment.sh
python -m vatl_sdf.conditional_atlas_smooth --atlas "$ATLAS" "$@"
python -m analysis.atlas.plot_mean_latent --atlas "$ATLAS" --manifest "$MANIFEST"
echo CONDITIONAL_ATLAS_SMOOTH_OK

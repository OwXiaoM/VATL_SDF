#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=mlat_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
# Mean-latent baselines on unseen subjects. Usage: sbatch slurm/mean_latent_baseline.sh <run options>
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
python -m analysis.generalization.mean_latent_baseline run "$@"
echo MEAN_LATENT_BASELINE_OK

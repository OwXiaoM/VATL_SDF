#!/bin/bash
#SBATCH -A EUHPC_D32_008
#SBATCH -p boost_usr_prod
#SBATCH --job-name=sdf_individual_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=sdf_individual_eval_%j.log
#SBATCH --error=sdf_individual_eval_%j.err
set -euo pipefail
N=${1:?Usage: sbatch slurm/evaluate_sdf_only.sh 3-or-5 NEW_OUTPUT_DIRECTORY [evaluation options]}
OUTPUT=${2:?Provide a new output directory}
shift 2
case "$N" in 3|5) ;; *) echo 'N must be 3 or 5' >&2; exit 2 ;; esac
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Fixed seeds/settings; CUDA grid_sample backward is not bitwise deterministic.
# Strict determinism is available with --device cpu --determinism strict.
srun python -m mra_sdf_only.test_individual --num-test-subjects "$N" --output "$OUTPUT" \
  --device cuda --determinism seeded "$@"

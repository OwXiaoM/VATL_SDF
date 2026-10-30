#!/bin/bash
#SBATCH -p h200
#SBATCH --job-name=mra_sdf_only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=mra_sdf_only_%j.log
#SBATCH --error=mra_sdf_only_%j.err
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 PYTHONUNBUFFERED=1
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 -m vatl_sdf.train_ddp --config config.yaml "$@"

#sbatch slurm/train_ablation_4gpu.sh 00_baseline
#sbatch slurm/train_ablation_4gpu.sh 01_sdf_weight_1
#sbatch slurm/train_ablation_4gpu.sh 02_sdf_only
#sbatch slurm/train_ablation_4gpu.sh 03_all_experts
#sbatch slurm/train_ablation_4gpu.sh 04_surface_band_1mm
#sbatch slurm/train_ablation_4gpu.sh 05_latent_grid_5
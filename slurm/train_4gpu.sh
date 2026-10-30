#!/bin/bash
#SBATCH -A EUHPC_D32_008
#SBATCH -p boost_usr_prod
#SBATCH --job-name=mra_sdf_only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=mra_sdf_only_%j.log
#SBATCH --error=mra_sdf_only_%j.err
set -euo pipefail
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo_work/EUHPC_D32_008/xzhang01/envs/torch/bin/activate
cd /leonardo/home/userexternal/xzhang01/VATL_SDF
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 PYTHONUNBUFFERED=1
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 -m mra_sdf_only.train_ddp --config config.yaml "$@"

#sbatch slurm/train_ablation_4gpu.sh 00_baseline
#sbatch slurm/train_ablation_4gpu.sh 01_sdf_weight_1
#sbatch slurm/train_ablation_4gpu.sh 02_sdf_only
#sbatch slurm/train_ablation_4gpu.sh 03_all_experts
#sbatch slurm/train_ablation_4gpu.sh 04_surface_band_1mm
#sbatch slurm/train_ablation_4gpu.sh 05_latent_grid_5
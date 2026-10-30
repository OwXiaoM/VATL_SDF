#!/bin/bash
#SBATCH -p h200
#SBATCH --job-name=sdf_split_corrected
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=sdf_split_corrected_%j.log
#SBATCH --error=sdf_split_corrected_%j.err
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m vatl_sdf.train_ddp --config configs/ablations/02_sdf_only.yaml "$@"

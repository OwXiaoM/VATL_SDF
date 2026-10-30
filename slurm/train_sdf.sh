#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=vatl_sdf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
# Usage: sbatch [--gres=gpu:h200:4 -p h200] slurm/train_sdf.sh CONFIG [--set key=value ...]
# Four-GPU DDP training; all compared runs must use the same GPU count.
set -euo pipefail
CONFIG=${1:?Usage: sbatch slurm/train_sdf.sh CONFIG [--set key=value ...]}
shift
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Config: $CONFIG; overrides: $*; git: $(git rev-parse --short HEAD)"
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m vatl_sdf.train_ddp --config "$CONFIG" "$@"

#!/bin/bash
#SBATCH -A EUHPC_D32_008
#SBATCH -p boost_usr_prod
#SBATCH --job-name=vessel_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=vessel_ablation_%j.log
#SBATCH --error=vessel_ablation_%j.err
set -euo pipefail
ABLATION=${1:?Usage: sbatch slurm/train_ablation_4gpu.sh NAME [--set key=value ...]}
shift
case "$ABLATION" in
  00_baseline|01_sdf_weight_1|02_sdf_only|03_all_experts|04_surface_band_1mm|05_latent_grid_5) ;;
  *) echo "Unknown ablation: $ABLATION" >&2; exit 2 ;;
esac
module purge
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CONFIG="configs/ablations/${ABLATION}.yaml"
echo "Training ablation: $ABLATION; config: $CONFIG"
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
  -m mra_sdf_only.train_ddp --config "$CONFIG" "$@"

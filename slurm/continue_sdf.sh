#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=vatl_sdf_cont
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
# Continue a run from its own latest checkpoint (same config, same GPU count). Exits without
# training if the run already passed stop_after_epoch.
# Usage: sbatch [--gres=...] slurm/continue_sdf.sh CONFIG OUTPUT_ROOT [--set extra overrides as in the first job]
set -euo pipefail
CONFIG=${1:?config}; ROOT=${2:?output_root of the run}
shift 2
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUNS=( "$ROOT"/*/ ); [[ ${#RUNS[@]} -eq 1 ]] || { echo "Expected one run under $ROOT" >&2; exit 2; }
CKPT=$(ls "${RUNS[0]}"checkpoint_step_*.pt | sort | tail -1)
echo "Continuing $CKPT; git $(git rev-parse --short HEAD)"
NPROC=${SLURM_GPUS_ON_NODE:-4}
srun python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
  -m vatl_sdf.train_ddp --config "$CONFIG" --set "output_root=$ROOT" "resume=$CKPT" require_fresh_start=false "$@"

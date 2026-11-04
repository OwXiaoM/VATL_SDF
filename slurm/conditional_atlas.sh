#!/bin/bash
#SBATCH -p l40s
#SBATCH --job-name=cond_atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Age- and sex-conditional vessel and graph atlases of a trained checkpoint, then figures.
# Usage: sbatch slurm/conditional_atlas.sh CHECKPOINT DECODINGS_DIR POPULATION_GRAPH_PKL MANIFEST_JSON OUT_DIR [options]
set -euo pipefail
CKPT=${1:?checkpoint}; DECODINGS=${2:?decodings}; GRAPH=${3:?population graph}; MANIFEST=${4:?manifest}; OUT=${5:?new output directory}; shift 5
source /users/zhangx/VATL_SDF/slurm/environment.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m vatl_sdf.conditional_atlas --checkpoint "$CKPT" --decodings "$DECODINGS" --population-graph "$GRAPH" --output "$OUT" --workers 8 "$@"
python -m analysis.atlas.plot_conditional_atlas --atlas "$OUT" --manifest "$MANIFEST"
echo CONDITIONAL_ATLAS_OK

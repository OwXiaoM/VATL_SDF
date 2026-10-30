#!/bin/bash
#SBATCH -p h200
#SBATCH --qos=dev
#SBATCH --job-name=inspect_graphs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
# Read-only CPU job: graph pickles vs reference SDF/mask for all subjects.
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
OUT=${1:?output directory}
python -m analysis.phase0_graph.inspect_graphs --output "$OUT"

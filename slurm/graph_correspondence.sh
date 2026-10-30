#!/bin/bash
#SBATCH -p l40s
#SBATCH --qos=dev
#SBATCH --job-name=graph_corr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# Phase 0a/0b (CPU only): headers, convention scoring, radius units, QC PNGs, all-subject stats,
# and a search for the graph-generation script. Writes only under OUT.
set -euo pipefail
source /users/zhangx/VATL_SDF/slurm/environment.sh
OUT=${1:?new output directory}
cd /users/zhangx/VATL_SDF
mkdir -p "$OUT"
python -m unittest tests.test_graph_coords
python -m analysis.phase0_graph.correspondence headers --output "$OUT" --all
python -m analysis.phase0_graph.correspondence conventions --output "$OUT" --workers 5
python -m analysis.phase0_graph.correspondence qc --output "$OUT" --subjects IXI002-Guys-0828 IXI012-HH-1211 IXI363-HH-2050
python -m analysis.phase0_graph.correspondence stats --output "$OUT" --all --workers 16
# Generator search: Python/shell/notebook files mentioning graph-pickle keys.
timeout 1200 find /users/zhangx /medias/medical/data/Atlas_training -xdev \
    \( -name '*.py' -o -name '*.ipynb' -o -name '*.sh' \) -size -5M -print0 2>/dev/null \
  | xargs -0 grep -lE 'node_radius_map|_bin_masks|IXI_graphs|voxel_sizes.*input_file' 2>/dev/null \
  | grep -v '/VATL_SDF/analysis/' > "$OUT/generator_search.txt" || true
for repo in /users/zhangx/VATL_MM /users/zhangx/VATL_SDF; do
  git -C "$repo" log --all --oneline -S node_radius_map >> "$OUT/generator_search.txt" 2>/dev/null || true
done
echo done

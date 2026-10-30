#!/bin/bash
# Bounded numerical validation only, not production training.
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?new validation output directory}
python -m tests.smoke --real --device cuda --output "$ROOT/real_gpu" 2>&1 | tee "$ROOT/real_gpu.log"
RUN_DIR=$(cat "$ROOT/real_gpu/run_directory.txt")
python -m mra_sdf_only.build_atlas --checkpoint "$RUN_DIR/checkpoint_step_00000002.pt" --output "$ROOT/atlas_age20_M_spacing1_SMOKE_ONLY" --device cuda --chunk-size 20000 --age 20 --sex M --spacing 1 1 1
python -m mra_sdf_only.build_atlas --checkpoint "$RUN_DIR/checkpoint_step_00000002.pt" --output "$ROOT/subject_native_SMOKE_ONLY" --device cuda --chunk-size 20000 --subject-id IXI002-Guys-0828

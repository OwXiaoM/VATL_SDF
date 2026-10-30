#!/bin/bash
set -euo pipefail
source /leonardo/home/userexternal/xzhang01/VATL_SDF/slurm/environment.sh
ROOT=${1:?validation output root}
python -m mra_sdf_only.train --config config.yaml --set data.max_subjects=2 data.num_workers=0 sampling.mra.points=256 sampling.sdf.points=256 training.subjects_per_rank=1 training.task_points_per_step=256 training.max_steps=1 training.atlas_every=0 training.gradient_diagnostics_every=1 "output_root=$ROOT/default_arch_gpu" run_name=default_arch_smoke 2>&1 | tee "$ROOT/default_arch_gpu.log"
